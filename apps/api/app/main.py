from __future__ import annotations

import json
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from redis import Redis

from akane_providers.base import ModelRequest
from akane_providers.codex_app_server import CodexAppServerAdapter


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AKANE_", extra="ignore")

    internal_api_token: str = ""
    redis_url: str = "redis://redis:6379/0"
    codex_base_url: str = ""
    codex_api_path: str = "/v1/responses"
    codex_api_token: str = ""
    codex_auth_mode: str = "none"
    llm_total_timeout_sec: float = 20.0


settings = Settings()
if not settings.internal_api_token:
    raise RuntimeError("AKANE_INTERNAL_API_TOKEN is required")

app = FastAPI(title="akane-api", version="0.1.0")
redis_client = Redis.from_url(settings.redis_url, decode_responses=True)


class MessageRequest(BaseModel):
    session_key: str = Field(..., examples=["g123:c456:t789:u999"])
    text: str
    source: str = "discord"
    metadata: dict = Field(default_factory=dict)


class JobRequest(BaseModel):
    session_key: str = Field(..., examples=["g123:c456:t789:u999"])
    job_type: str
    payload: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)


@app.middleware("http")
async def verify_internal_token(request: Request, call_next):
    if request.url.path.startswith("/v1/"):
        expected = f"Bearer {settings.internal_api_token}"
        provided = request.headers.get("Authorization", "")
        if provided != expected:
            return JSONResponse(
                status_code=401,
                content={
                    "error_code": "UNAUTHORIZED",
                    "message": "missing or invalid internal token",
                    "trace_id": f"tr_{uuid4().hex[:12]}",
                },
            )
    return await call_next(request)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    return {"status": "ready"}


@app.post("/v1/messages")
def post_message(payload: MessageRequest):
    adapter = CodexAppServerAdapter(
        base_url=settings.codex_base_url,
        api_path=settings.codex_api_path,
        token=settings.codex_api_token,
        timeout_sec=settings.llm_total_timeout_sec,
    )
    model_req = ModelRequest(
        messages=[{"role": "user", "content": payload.text}],
        system_prompt="You are akane. Reply concisely and helpfully.",
    )
    try:
        model_res = adapter.generate(model_req)
        reply_text = model_res.final_text or "(empty response)"
    except Exception as exc:  # noqa: BLE001
        reply_text = f"provider error: {exc}"

    return {
        "status": "ok",
        "session_id": str(uuid4()),
        "reply_text": reply_text,
        "mode": "sync",
        "tool_events": [],
        "trace_id": f"tr_{uuid4().hex[:12]}",
    }


@app.post("/v1/jobs")
def post_job(payload: JobRequest):
    session_id = str(uuid4())
    job_id = f"job_{uuid4().hex[:8]}"
    job = {
        "job_id": job_id,
        "session_id": session_id,
        "session_key": payload.session_key,
        "job_type": payload.job_type,
        "payload": payload.payload,
        "metadata": payload.metadata,
    }
    redis_client.lpush("akane:jobs", json.dumps(job, ensure_ascii=False))
    return {
        "status": "queued",
        "session_id": session_id,
        "job_id": job_id,
        "trace_id": f"tr_{uuid4().hex[:12]}",
    }


def main():
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=False)


if __name__ == "__main__":
    main()
