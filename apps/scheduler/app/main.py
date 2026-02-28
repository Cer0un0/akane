from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict
from redis import asyncio as aioredis


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AKANE_", extra="ignore")

    api_base_url: str = "http://akane-api:8080"
    internal_api_token: str = ""
    redis_url: str = "redis://redis:6379/0"
    heartbeat_enabled: bool = True
    heartbeat_interval_min: int = 30
    heartbeat_channel_id: str = ""
    scheduler_lock_ttl_sec: int = 60


settings = Settings()


async def run_heartbeat(redis: aioredis.Redis) -> None:
    """Call the heartbeat API and notify if not HEARTBEAT_OK."""
    if not settings.heartbeat_enabled:
        return

    # Distributed lock to avoid duplicate heartbeats
    lock_key = "akane:lock:heartbeat"
    acquired = await redis.set(lock_key, "1", ex=settings.scheduler_lock_ttl_sec, nx=True)
    if not acquired:
        return

    try:
        headers = {"Authorization": f"Bearer {settings.internal_api_token}"}
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{settings.api_base_url}/v1/heartbeat",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

        status = data.get("status", "")
        reply_text = data.get("reply_text", "")
        suppressed = data.get("suppressed", False)

        print(f"[scheduler] heartbeat status={status} suppressed={suppressed}")

        if status == "skipped":
            return

        if not suppressed and settings.heartbeat_channel_id:
            notification = {
                "session_key": "system:heartbeat",
                "channel_id": settings.heartbeat_channel_id,
                "message": f"[heartbeat] {reply_text}",
            }
            await redis.publish(
                "akane:notifications:system:heartbeat",
                json.dumps(notification, ensure_ascii=False),
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[scheduler] heartbeat failed: {exc}")


async def run_scheduler_loop():
    redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    tick_count = 0
    interval_ticks = settings.heartbeat_interval_min  # 1 tick = 60s

    while True:
        print(f"[scheduler] tick {datetime.now(tz=UTC).isoformat()}")

        if tick_count % interval_ticks == 0:
            await run_heartbeat(redis)

        tick_count += 1
        await asyncio.sleep(60)


def main():
    asyncio.run(run_scheduler_loop())


if __name__ == "__main__":
    main()
