from __future__ import annotations

import json
from time import monotonic
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from websockets.sync.client import connect as ws_connect

from .base import ModelRequest, ModelResponse


class CodexAppServerAdapter:
    name = "codex_app_server"

    def __init__(
        self,
        base_url: str,
        api_path: str = "/v1/responses",
        token: str = "",
        timeout_sec: float = 20.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_path = api_path
        self.token = token
        self.timeout_sec = timeout_sec

    def health(self) -> dict:
        return {"name": self.name, "status": "unknown"}

    def generate(self, req: ModelRequest) -> ModelResponse:
        if not self.base_url:
            return ModelResponse(final_text="Codex base URL is not configured.")

        scheme = urlparse(self.base_url).scheme.lower()
        if scheme in {"ws", "wss"}:
            return self._generate_ws(req)
        return self._generate_http(req)

    def _generate_http(self, req: ModelRequest) -> ModelResponse:
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        payload = {
            "input": req.messages,
            "instructions": req.system_prompt,
            "max_output_tokens": req.max_tokens,
            "temperature": req.temperature,
        }
        if req.model:
            payload["model"] = req.model

        with httpx.Client(timeout=self.timeout_sec) as client:
            resp = client.post(f"{self.base_url}{self.api_path}", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        text = self._extract_text(data)
        return ModelResponse(final_text=text, usage=data.get("usage", {}))

    def _generate_ws(self, req: ModelRequest) -> ModelResponse:
        user_text = self._compose_user_text(req.messages)
        if not user_text.strip():
            return ModelResponse(final_text="")

        with ws_connect(self.base_url, open_timeout=self.timeout_sec, close_timeout=1.0) as ws:
            self._rpc_call(
                ws,
                method="initialize",
                params={
                    "clientInfo": {"name": "akane-api", "version": "0.1.0"},
                    "capabilities": {},
                },
            )
            # For headless VPS deployment, allow explicit API-key login per connection.
            if self.token:
                self._rpc_call(
                    ws,
                    method="account/login/start",
                    params={"type": "apiKey", "apiKey": self.token},
                )
            thread_params: dict[str, object] = {"ephemeral": True}
            if req.model:
                thread_params["model"] = req.model
            if req.system_prompt:
                thread_params["baseInstructions"] = req.system_prompt
            thread_result = self._rpc_call(ws, method="thread/start", params=thread_params)

            thread_id = (
                thread_result.get("threadId")
                or (thread_result.get("thread") or {}).get("id")
                or ""
            )
            if not thread_id:
                raise RuntimeError("thread/start did not return thread id")

            turn_req_id = self._new_request_id("turn")
            ws.send(
                json.dumps(
                    {
                        "id": turn_req_id,
                        "method": "turn/start",
                        "params": {
                            "threadId": thread_id,
                            "input": [{"type": "text", "text": user_text}],
                        },
                    },
                    ensure_ascii=False,
                )
            )

            text_chunks: list[str] = []
            final_text = ""
            deadline = monotonic() + self.timeout_sec
            while True:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError("turn/start timed out")
                raw = ws.recv(timeout=remaining)
                data = json.loads(raw)

                if data.get("id") == turn_req_id and data.get("error"):
                    raise RuntimeError(self._format_rpc_error(data["error"]))

                method = data.get("method")
                params = data.get("params") or {}
                if method == "item/agentMessage/delta":
                    delta = params.get("delta")
                    if isinstance(delta, str):
                        text_chunks.append(delta)
                    continue

                if method == "item/completed":
                    item = params.get("item") or {}
                    if item.get("type") == "agentMessage" and isinstance(item.get("text"), str):
                        final_text = item["text"]
                    continue

                if method == "error":
                    err = params.get("error") or {}
                    if params.get("willRetry") is False:
                        raise RuntimeError(err.get("message") or "codex app-server error")
                    continue

                if method == "turn/completed":
                    turn = params.get("turn") or {}
                    if str(turn.get("status", "")).lower() == "failed":
                        err = turn.get("error") or {}
                        raise RuntimeError(err.get("message") or "turn failed")
                    break

            text = final_text.strip() or "".join(text_chunks).strip()
            return ModelResponse(final_text=text)

    def _rpc_call(self, ws, method: str, params: dict) -> dict:
        req_id = self._new_request_id(method.replace("/", "_"))
        ws.send(
            json.dumps(
                {
                    "id": req_id,
                    "method": method,
                    "params": params,
                },
                ensure_ascii=False,
            )
        )
        deadline = monotonic() + self.timeout_sec
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{method} timed out")
            raw = ws.recv(timeout=remaining)
            data = json.loads(raw)
            if data.get("id") != req_id:
                continue
            if data.get("error"):
                raise RuntimeError(self._format_rpc_error(data["error"]))
            result = data.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(f"{method} returned invalid result")
            return result

    @staticmethod
    def _new_request_id(prefix: str) -> str:
        return f"{prefix}-{uuid4().hex[:8]}"

    @staticmethod
    def _format_rpc_error(error_payload: dict) -> str:
        if not isinstance(error_payload, dict):
            return "unknown rpc error"
        message = error_payload.get("message")
        data = error_payload.get("data")
        if message and data:
            return f"{message}: {data}"
        return str(message or data or "unknown rpc error")

    @staticmethod
    def _compose_user_text(messages: list[dict]) -> str:
        chunks: list[str] = []
        for msg in messages:
            role = str(msg.get("role", "user"))
            content = msg.get("content")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts = [part.get("text", "") for part in content if isinstance(part, dict)]
                text = "\n".join([p for p in parts if p])
            else:
                text = str(content or "")
            chunks.append(f"{role}: {text}".strip())
        return "\n".join(chunks).strip()

    @staticmethod
    def _extract_text(data: dict) -> str:
        if data.get("output_text"):
            return str(data["output_text"])
        if data.get("text"):
            return str(data["text"])

        # responses-style payload compatibility
        outputs = data.get("output")
        if isinstance(outputs, list):
            chunks: list[str] = []
            for item in outputs:
                content = item.get("content") if isinstance(item, dict) else None
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("text"):
                            chunks.append(str(part["text"]))
            if chunks:
                return "\n".join(chunks)

        return ""
