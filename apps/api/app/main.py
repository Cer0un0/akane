from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from redis import Redis

from akane_providers.base import ModelRequest
from akane_providers.codex_app_server import CodexAppServerAdapter

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AKANE_", extra="ignore")

    internal_api_token: str = ""
    redis_url: str = "redis://redis:6379/0"
    codex_base_url: str = ""
    codex_api_path: str = "/v1/responses"
    codex_api_token: str = ""
    codex_auth_mode: str = "none"
    llm_model: str = ""
    llm_total_timeout_sec: float = 20.0
    conv_history_size: int = 20
    conv_history_ttl_sec: int = 86400
    memory_dir: str = "/workspace/memory"
    timezone: str = "Asia/Tokyo"


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


def _parse_session_key(session_key: str) -> dict[str, str]:
    """Parse 'g123:c456:t789:u999' into component dict."""
    parts = session_key.split(":")
    result: dict[str, str] = {}
    for part in parts:
        if part.startswith("g"):
            result["guild"] = part
        elif part.startswith("c"):
            result["channel"] = part
        elif part.startswith("t"):
            result["thread"] = part
        elif part.startswith("u"):
            result["user"] = part
    return result


def _conv_redis_key(guild: str) -> str:
    return f"akane:conv:{guild}"


def _load_history(guild: str) -> list[dict]:
    """Load recent conversation history from Redis."""
    key = _conv_redis_key(guild)
    raw_items = redis_client.lrange(key, 0, settings.conv_history_size - 1)
    # lrange returns newest-first (lpush), reverse to chronological order
    messages = []
    for raw in reversed(raw_items):
        try:
            entry = json.loads(raw)
            messages.append({"role": entry["role"], "content": entry["content"]})
        except (json.JSONDecodeError, KeyError):
            continue
    return messages


def _save_history(guild: str, role: str, content: str, user_id: str = "") -> None:
    """Push a message to the Redis conversation list."""
    key = _conv_redis_key(guild)
    entry = json.dumps(
        {"role": role, "content": content, "user_id": user_id, "ts": _now_iso()},
        ensure_ascii=False,
    )
    redis_client.lpush(key, entry)
    redis_client.ltrim(key, 0, settings.conv_history_size * 2 - 1)
    redis_client.expire(key, settings.conv_history_ttl_sec)


def _now_iso() -> str:
    try:
        import zoneinfo

        tz = zoneinfo.ZoneInfo(settings.timezone)
    except Exception:  # noqa: BLE001
        tz = timezone.utc
    return datetime.now(tz=tz).isoformat(timespec="seconds")


def _now_hhmm() -> str:
    try:
        import zoneinfo

        tz = zoneinfo.ZoneInfo(settings.timezone)
    except Exception:  # noqa: BLE001
        tz = timezone.utc
    return datetime.now(tz=tz).strftime("%H:%M")


def _now_date() -> str:
    try:
        import zoneinfo

        tz = zoneinfo.ZoneInfo(settings.timezone)
    except Exception:  # noqa: BLE001
        tz = timezone.utc
    return datetime.now(tz=tz).strftime("%Y-%m-%d")


def _append_markdown_log(
    session_parts: dict[str, str],
    user_text: str,
    reply_text: str,
) -> None:
    """Append conversation to daily Markdown file."""
    try:
        memory_dir = Path(settings.memory_dir)
        memory_dir.mkdir(parents=True, exist_ok=True)
        date_str = _now_date()
        filepath = memory_dir / f"{date_str}.md"

        conv_label = ":".join(
            filter(None, [
                session_parts.get("guild", ""),
                session_parts.get("channel", ""),
                session_parts.get("thread", ""),
            ])
        )
        user_id = session_parts.get("user", "")
        time_str = _now_hhmm()

        block = f"\n### {time_str}\n\n"
        block += f"**user ({user_id}):** {user_text}\n\n"
        block += f"**akane:** {reply_text}\n\n"

        # Check if we need a section header for this conv_label
        need_header = True
        if filepath.exists():
            existing = filepath.read_text(encoding="utf-8")
            if f"## {conv_label}" in existing:
                need_header = False
        else:
            existing = ""

        with filepath.open("a", encoding="utf-8") as f:
            if need_header:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write(f"## {conv_label}\n")
            f.write(block)
    except Exception:  # noqa: BLE001
        logger.warning("failed to write markdown log", exc_info=True)


@app.post("/v1/messages")
def post_message(payload: MessageRequest):
    session_parts = _parse_session_key(payload.session_key)
    guild = session_parts.get("guild", "global")
    user_id = session_parts.get("user", "")

    # 1. Load conversation history from Redis
    history = _load_history(guild)

    # 2. Build messages with history + new user message
    messages = history + [{"role": "user", "content": payload.text}]

    adapter = CodexAppServerAdapter(
        base_url=settings.codex_base_url,
        api_path=settings.codex_api_path,
        token=settings.codex_api_token,
        timeout_sec=settings.llm_total_timeout_sec,
    )
    model_req = ModelRequest(
        messages=messages,
        system_prompt="You are akane. Reply concisely and helpfully.",
        model=(settings.llm_model or None),
    )
    try:
        model_res = adapter.generate(model_req)
        reply_text = model_res.final_text or "(empty response)"
    except Exception as exc:  # noqa: BLE001
        reply_text = f"provider error: {exc}"

    # 3. Save user message and assistant reply to Redis (skip empty)
    _save_history(guild, "user", payload.text, user_id)
    if reply_text != "(empty response)":
        _save_history(guild, "assistant", reply_text)

    # 4. Append to Markdown log
    _append_markdown_log(session_parts, payload.text, reply_text)

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
