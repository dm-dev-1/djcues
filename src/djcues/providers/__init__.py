"""Provider abstraction for the agentic analysis mode.

A small common interface so ``agentic.py``'s multi-agent orchestration is
written once and works against any supported LLM provider. Each provider's
SDK is an optional dependency (see the ``agentic`` extra in
``pyproject.toml``) -- adapter modules import their SDK lazily, at call
time, so a user with only one provider's SDK installed isn't broken by
the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class ModelInfo:
    """A model available from a provider, from a live metadata lookup."""

    id: str
    display_name: str
    provider: str
    context_window: int | None = None


@dataclass
class PriceInfo:
    """Per-1M-token pricing, in USD."""

    input_per_million: float
    output_per_million: float


@dataclass
class GenerationResult:
    """A structured-output call's parsed content plus real token usage,
    so callers can accumulate accurate post-flight cost telemetry instead
    of relying on the pre-flight estimate."""

    content: dict
    input_tokens: int
    output_tokens: int


# Locally-maintained pricing overlay, keyed by model id. Neither
# provider's model-metadata endpoint returns pricing, so this table is
# what powers cost estimates -- verify it against each provider's current
# pricing page before relying on it for real cost decisions, both
# landscapes change frequently and this is not fetched live. Verified
# against https://ai.google.dev/gemini-api/docs/pricing (2026-09-02).
#
# Deliberately excludes "-latest" alias ids (e.g. gemini-flash-lite-latest,
# gemini-flash-latest) even when a user picks one in `auth set`/`auth web`:
# Google repoints those aliases to a different underlying model over time,
# so any price recorded here for one would silently go stale the next time
# that happens, with no djcues code change to notice it. estimate_cost()
# correctly returns None for them (unknown, not free) until they're
# resolved to a concrete versioned id.
PRICING: dict[str, PriceInfo] = {
    "claude-haiku-4-5": PriceInfo(1.00, 5.00),
    "claude-sonnet-5": PriceInfo(2.00, 10.00),
    "gemini-2.5-flash-lite": PriceInfo(0.10, 0.40),
    "gemini-3.1-flash-lite": PriceInfo(0.25, 1.50),
    "gemini-3.5-flash-lite": PriceInfo(0.30, 2.50),
    "gemini-3.7-flash": PriceInfo(0.75, 3.75),
}

# The recommended lightweight default per provider, used when a user
# hasn't picked a specific model (e.g. "cheapest first" in `auth set`).
DEFAULT_MODEL: dict[str, str] = {
    "anthropic": "claude-haiku-4-5",
    "gemini": "gemini-2.5-flash-lite",
}


class ModelProvider(Protocol):
    """Common interface every provider adapter implements."""

    name: str  # "anthropic" | "gemini"

    def list_models(self, api_key: str) -> list[ModelInfo]:
        """Live-fetch available models from the provider's API."""
        ...

    def generate_structured(
        self,
        api_key: str,
        model: str,
        system: str,
        user_content: str,
        schema: dict,
    ) -> GenerationResult:
        """Single structured-output call. Returns the parsed JSON content
        plus real input/output token usage from the provider's response."""
        ...

    def count_tokens(self, api_key: str, model: str, content: str, system: str | None = None) -> int:
        """Pre-flight token count for cost estimation. When *system* is
        given, its tokens are included in the count -- exactly, on
        providers whose API supports it; approximated on ones that don't
        (see each adapter for specifics)."""
        ...


def get_provider(name: str) -> ModelProvider:
    """Factory: returns the adapter for the named provider."""
    if name == "anthropic":
        from djcues.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    if name == "gemini":
        from djcues.providers.gemini_provider import GeminiProvider

        return GeminiProvider()
    raise ValueError(f"Unknown provider: {name!r} (expected 'anthropic' or 'gemini')")


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimate USD cost for a request using the local pricing table.

    Returns None (not 0.0) when the model isn't in the table, so callers
    can distinguish "free" from "unknown" rather than silently
    understating cost.
    """
    price = PRICING.get(model)
    if price is None:
        return None
    return (
        input_tokens / 1_000_000 * price.input_per_million
        + output_tokens / 1_000_000 * price.output_per_million
    )
