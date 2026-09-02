"""Google Gemini provider adapter.

Imports the ``google-genai`` SDK lazily (inside methods, not at module
import time) since it's an optional dependency -- see the ``agentic``
extra in pyproject.toml. SDK exceptions are allowed to propagate
naturally; retry/fallback policy is agentic.py's orchestration concern.

Field names on the live model-list/count-tokens/usage-metadata responses
(context window field, token-count fields, generate_content's
usage_metadata.prompt_token_count/candidates_token_count) are taken from
Gemini's typical response shape but were not verified against a live API
call while writing this -- confirm against a real response the first
time this runs with a real key, and adjust the `getattr` fallbacks below
if the actual field names differ.
"""

from __future__ import annotations

import json
from typing import Any

from djcues.providers import GenerationResult, ModelInfo


class GeminiProvider:
    """Adapter implementing the ModelProvider protocol for Gemini."""

    name = "gemini"

    def _client(self, api_key: str) -> Any:
        from google import genai

        return genai.Client(api_key=api_key)

    def list_models(self, api_key: str) -> list[ModelInfo]:
        client = self._client(api_key)
        models = []
        for m in client.models.list():
            raw_name = getattr(m, "name", "")
            model_id = raw_name.split("/")[-1] if "/" in raw_name else raw_name
            models.append(
                ModelInfo(
                    id=model_id or raw_name,
                    display_name=getattr(m, "display_name", model_id or raw_name),
                    provider=self.name,
                    context_window=getattr(m, "input_token_limit", None),
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
        from google.genai import types

        client = self._client(api_key)
        response = client.models.generate_content(
            model=model,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_json_schema=schema,
                max_output_tokens=max_tokens,
            ),
        )
        usage = getattr(response, "usage_metadata", None)
        return GenerationResult(
            content=json.loads(response.text),
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
        )

    def count_tokens(self, api_key: str, model: str, content: str) -> int:
        client = self._client(api_key)
        result = client.models.count_tokens(model=model, contents=content)
        return getattr(result, "total_tokens", 0)
