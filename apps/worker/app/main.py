from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

from pydantic_settings import BaseSettings, SettingsConfigDict
from redis import asyncio as aioredis


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AKANE_", extra="ignore")

    redis_url: str = "redis://redis:6379/0"


settings = Settings()


async def process_job(redis: aioredis.Redis, raw_job: str):
    job = json.loads(raw_job)
    session_key = str(job.get("session_key", ""))
    metadata = job.get("metadata", {}) if isinstance(job.get("metadata"), dict) else {}
    channel_id = metadata.get("channel_id")

    progress_event = {
        "session_key": session_key,
        "channel_id": channel_id,
        "message": f"job started: {job.get('job_id')}",
    }
    await redis.publish(f"akane:notifications:{session_key}", json.dumps(progress_event, ensure_ascii=False))

    # TODO: replace with actual orchestrator/tool execution.
    await asyncio.sleep(2)

    done_event = {
        "session_key": session_key,
        "channel_id": channel_id,
        "message": f"job done: {job.get('job_id')} ({job.get('job_type')})",
    }
    await redis.publish(f"akane:notifications:{session_key}", json.dumps(done_event, ensure_ascii=False))


async def run_worker_loop():
    redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    while True:
        item = await redis.brpop("akane:jobs", timeout=15)
        if item is None:
            print(f"[worker] idle heartbeat {datetime.now(tz=UTC).isoformat()}")
            continue
        _, raw_job = item
        try:
            await process_job(redis, raw_job)
        except Exception as exc:  # noqa: BLE001
            print(f"[worker] failed job: {exc}")


def main():
    asyncio.run(run_worker_loop())


if __name__ == "__main__":
    main()
