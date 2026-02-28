from __future__ import annotations

import re
from dataclasses import dataclass

SESSION_KEY_PATTERN = re.compile(r"^g(?P<guild>\d+):c(?P<channel>\d+):t(?P<thread>\d+):u(?P<user>\d+)$")


@dataclass(frozen=True)
class SessionKey:
    guild_id: int
    channel_id: int
    thread_id: int
    user_id: int

    def as_str(self) -> str:
        return f"g{self.guild_id}:c{self.channel_id}:t{self.thread_id}:u{self.user_id}"


def parse_session_key(value: str) -> SessionKey:
    match = SESSION_KEY_PATTERN.match(value)
    if not match:
        raise ValueError(f"invalid session key: {value}")
    return SessionKey(
        guild_id=int(match.group("guild")),
        channel_id=int(match.group("channel")),
        thread_id=int(match.group("thread")),
        user_id=int(match.group("user")),
    )

