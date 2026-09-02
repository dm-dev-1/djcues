import sys
from pathlib import Path

import pytest

from djcues import auth


class _FakeKeyringBackend:
    """In-memory stand-in for the OS credential store -- tests must never
    touch the real Windows Credential Manager / macOS Keychain / Linux
    Secret Service."""

    def __init__(self):
        self.store: dict[tuple[str, str], str] = {}

    def set_password(self, service, username, password):
        self.store[(service, username)] = password

    def get_password(self, service, username):
        return self.store.get((service, username))

    def delete_password(self, service, username):
        import keyring

        key = (service, username)
        if key not in self.store:
            raise keyring.errors.PasswordDeleteError("not found")
        del self.store[key]


@pytest.fixture
def fake_keyring(monkeypatch):
    import keyring

    backend = _FakeKeyringBackend()
    monkeypatch.setattr(keyring, "set_password", backend.set_password)
    monkeypatch.setattr(keyring, "get_password", backend.get_password)
    monkeypatch.setattr(keyring, "delete_password", backend.delete_password)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return backend


# --- config file (non-secret settings) ------------------------------------


def test_default_config_path_is_under_dot_djcues(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    path = auth.default_config_path()
    assert path == tmp_path / ".djcues" / "config.json"
    assert path.parent.is_dir()


def test_load_config_missing_file_returns_empty_dict(tmp_path):
    assert auth.load_config(tmp_path / "config.json") == {}


def test_save_then_load_config_roundtrip(tmp_path):
    config_path = tmp_path / "config.json"
    auth.save_config({"provider": "anthropic", "model": "claude-haiku-4-5"}, config_path)
    assert auth.load_config(config_path) == {"provider": "anthropic", "model": "claude-haiku-4-5"}


# --- API key storage (keyring, mocked) ------------------------------------


def test_set_then_resolve_api_key_from_keyring(fake_keyring):
    auth.set_api_key("anthropic", "sk-ant-realkey1234")
    key, source = auth.resolve_api_key("anthropic")
    assert key == "sk-ant-realkey1234"
    assert source == "keyring"


def test_resolve_api_key_falls_back_to_env_var(fake_keyring, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fromenv")
    key, source = auth.resolve_api_key("anthropic")
    assert key == "sk-ant-fromenv"
    assert source == "env"


def test_resolve_api_key_prefers_keyring_over_env_var(fake_keyring, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fromenv")
    auth.set_api_key("anthropic", "sk-ant-fromkeyring")
    key, source = auth.resolve_api_key("anthropic")
    assert key == "sk-ant-fromkeyring"
    assert source == "keyring"


def test_resolve_api_key_returns_none_when_nothing_found(fake_keyring):
    key, source = auth.resolve_api_key("anthropic")
    assert key is None
    assert source is None


def test_clear_api_key_removes_stored_key(fake_keyring):
    auth.set_api_key("gemini", "sk-gem-key")
    auth.clear_api_key("gemini")
    key, _source = auth.resolve_api_key("gemini")
    assert key is None


def test_clear_api_key_is_a_noop_when_nothing_stored(fake_keyring):
    auth.clear_api_key("anthropic")  # must not raise


def test_providers_use_independent_keyring_entries(fake_keyring):
    auth.set_api_key("anthropic", "sk-ant-one")
    auth.set_api_key("gemini", "sk-gem-two")
    assert auth.resolve_api_key("anthropic")[0] == "sk-ant-one"
    assert auth.resolve_api_key("gemini")[0] == "sk-gem-two"


# --- keyring unavailable (package not installed) --------------------------


def test_keyring_unavailable_raises_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "keyring", None)
    with pytest.raises(auth.KeyringUnavailableError) as exc_info:
        auth.set_api_key("anthropic", "sk-test")
    assert "djcues[agentic]" in str(exc_info.value)


def test_resolve_api_key_falls_through_to_env_when_keyring_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "keyring", None)
    monkeypatch.setenv("GEMINI_API_KEY", "sk-gem-envonly")
    key, source = auth.resolve_api_key("gemini")
    assert key == "sk-gem-envonly"
    assert source == "env"


# --- display safety ---------------------------------------------------


def test_mask_key_short_key_fully_masked():
    assert auth.mask_key("short") == "***"


def test_mask_key_normal_key_shows_prefix_and_suffix():
    assert auth.mask_key("sk-ant-api03-abcdefgh1234") == "sk-ant-...1234"
    # The middle of the real key must never appear.
    assert "api03-abcdefgh" not in auth.mask_key("sk-ant-api03-abcdefgh1234")
