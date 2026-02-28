from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import discord
import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict
from redis import asyncio as aioredis


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AKANE_", extra="ignore")

    bot_token: str = ""
    api_base_url: str = "http://akane-api:8080"
    bot_api_timeout_sec: float = 15.0
    internal_api_token: str = ""
    redis_url: str = "redis://redis:6379/0"


settings = Settings()

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
notification_task: asyncio.Task | None = None


@dataclass
class SessionRoute:
    channel_id: int
    guild_id: int


session_routes: dict[str, SessionRoute] = {}


def build_session_key(message: discord.Message) -> str:
    guild_id = message.guild.id if message.guild else 0
    if isinstance(message.channel, discord.Thread):
        channel_id = message.channel.parent_id or message.channel.id
        thread_id = message.channel.id
    else:
        channel_id = message.channel.id
        thread_id = message.channel.id
    return f"g{guild_id}:c{channel_id}:t{thread_id}:u{message.author.id}"


async def listen_notifications():
    redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    pubsub = redis.pubsub()
    await pubsub.psubscribe("akane:notifications:*")
    print("bot notification subscriber started")
    try:
        while True:
            msg = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if not msg:
                continue
            try:
                data = json.loads(msg["data"])
                await dispatch_notification(data)
            except Exception as exc:  # noqa: BLE001
                print(f"[bot] failed to handle notification: {exc}")
    finally:
        await pubsub.close()
        await redis.close()


async def dispatch_notification(data: dict):
    session_key = str(data.get("session_key", ""))
    message_text = str(data.get("message", "job update"))
    channel_id = data.get("channel_id")

    if channel_id is None and session_key in session_routes:
        channel_id = session_routes[session_key].channel_id
    if channel_id is None:
        print(f"[bot] dropped notification without channel: {data}")
        return

    channel = client.get_channel(int(channel_id)) or await client.fetch_channel(int(channel_id))
    await channel.send(message_text)


@client.event
async def on_ready():
    global notification_task
    print(f"akane-bot connected as {client.user}")
    if notification_task is None:
        notification_task = client.loop.create_task(listen_notifications())


@client.event
async def on_message(message: discord.Message):
    if message.author.bot or client.user is None:
        return
    if client.user not in message.mentions:
        return

    session_key = build_session_key(message)
    session_routes[session_key] = SessionRoute(
        channel_id=message.channel.id,
        guild_id=message.guild.id if message.guild else 0,
    )

    try:
        await message.add_reaction("⏳")
    except discord.HTTPException as exc:
        print(f"[bot] add_reaction failed: {exc}")

    payload = {
        "session_key": session_key,
        "text": message.content,
        "source": "discord",
        "metadata": {
            "message_id": str(message.id),
            "channel_id": str(message.channel.id),
            "guild_id": str(message.guild.id if message.guild else 0),
        },
    }

    try:
        async with httpx.AsyncClient(timeout=settings.bot_api_timeout_sec) as http:
            headers = {
                "Authorization": f"Bearer {settings.internal_api_token}",
            }
            if "#async" in message.content:
                job_resp = await http.post(
                    f"{settings.api_base_url}/v1/jobs",
                    json={
                        "session_key": session_key,
                        "job_type": "agent.long_run",
                        "payload": {"text": message.content},
                        "metadata": payload["metadata"],
                    },
                    headers=headers,
                )
                job_resp.raise_for_status()
                job = job_resp.json()
                await message.reply(f"queued: {job.get('job_id')}")
            else:
                resp = await http.post(
                    f"{settings.api_base_url}/v1/messages",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                reply = resp.json().get("reply_text", "No response text.")
                await message.reply(reply)
    except Exception as exc:  # noqa: BLE001
        await message.reply(f"bot error: {type(exc).__name__}: {exc}")
    finally:
        # Best-effort cleanup of thinking indicator.
        try:
            await message.remove_reaction("⏳", client.user)
        except discord.HTTPException as exc:
            print(f"[bot] remove_reaction failed: {exc}")


def main():
    if not settings.bot_token:
        raise RuntimeError("AKANE_BOT_TOKEN is required")
    if not settings.internal_api_token:
        raise RuntimeError("AKANE_INTERNAL_API_TOKEN is required")
    client.run(settings.bot_token)


if __name__ == "__main__":
    main()
