from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ModelRequest:
    messages: list[dict]
    system_prompt: str
    tool_schemas: list[dict] = field(default_factory=list)
    max_tokens: int = 1024
    temperature: float = 0.2
    metadata: dict = field(default_factory=dict)


@dataclass
class ModelResponse:
    final_text: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class ProviderAdapter(Protocol):
    name: str

    def health(self) -> dict: ...

    def generate(self, req: ModelRequest) -> ModelResponse: ...

