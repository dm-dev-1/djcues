"""Anthropic (Claude) provider adapter.

Imports the ``anthropic`` SDK lazily (inside methods, not at module import
time) since it's an optional dependency -- see the ``agentic`` extra in
pyproject.toml. SDK exceptions (AuthenticationError, RateLimitError,
APIStatusError, ...) are allowed to propagate naturally; retry/fallback
policy is agentic.py's orchestration concern, not this adapter's.
"""

from __future__ import annotations

import json
from typing import Any

from djcues.providers import GenerationResult, ModelInfo


class AnthropicProvider:
    """Adapter implementing the ModelProvider protocol for Anthropic."""

    name = "anthropic"

    def _client(self, api_key: str) -> Any:
        import anthropic

        return anthropic.Anthropic(api_key=api_key)

    def list_models(self, api_key: str) -> list[ModelInfo]:
        client = self._client(api_key)
        models = []
        for m in client.models.list():
            models.append(
                ModelInfo(
                    id=m.id,
                    display_name=getattr(m, "display_name", m.id),
                    provider=self.name,
                    context_window=getattr(m, "max_input_tokens", None),
                )
            )
        return models

    def generate_structured(
        self,
        api_key: str,
        model: str,
        system: str,
        user_content: str,
        schema: dict,
        max_tokens: int = 4096,
    ) -> GenerationResult:
        client = self._client(api_key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_content}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        text = next(b.text for b in response.content if b.type == "text")
        return GenerationResult(
            content=json.loads(text),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    def count_tokens(self, api_key: str, model: str, content: str) -> int:
        client = self._client(api_key)
        result = client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": content}],
        )
        return result.input_tokens
