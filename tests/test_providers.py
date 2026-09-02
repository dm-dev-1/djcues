import pytest

from djcues.providers import DEFAULT_MODEL, PRICING, estimate_cost, get_provider


def test_estimate_cost_known_model():
    cost = estimate_cost("claude-haiku-4-5", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == pytest.approx(1.00 + 5.00)


def test_estimate_cost_partial_tokens():
    cost = estimate_cost("claude-haiku-4-5", input_tokens=500_000, output_tokens=0)
    assert cost == pytest.approx(0.50)


def test_estimate_cost_unknown_model_returns_none():
    """None (not 0.0) for an unpriced model -- callers must be able to
    distinguish 'free' from 'unknown', per the security/honesty design."""
    assert estimate_cost("some-future-model", 1000, 1000) is None


def test_default_model_per_provider():
    assert DEFAULT_MODEL["anthropic"] in PRICING
    assert DEFAULT_MODEL["gemini"] in PRICING


def test_get_provider_anthropic():
    provider = get_provider("anthropic")
    assert provider.name == "anthropic"


def test_get_provider_gemini():
    provider = get_provider("gemini")
    assert provider.name == "gemini"


def test_get_provider_unknown_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("openai")
