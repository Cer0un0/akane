from __future__ import annotations

from .base import ModelRequest, ModelResponse


class KimiAdapter:
    name = "kimi"

    def health(self) -> dict:
        return {"name": self.name, "status": "unknown"}

    def generate(self, req: ModelRequest) -> ModelResponse:
        return ModelResponse(final_text="Kimi adapter skeleton")

