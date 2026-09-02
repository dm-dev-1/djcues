"""BYOK credential and config management for the agentic analysis mode.

API keys live in the OS-native credential store via ``keyring`` --
Windows Credential Manager, macOS Keychain, or the Linux Secret Service --
never in a plaintext file, never logged, never written to
``~/.djcues/history.db`` or any session JSON. Non-secret settings
(provider, model, batch preference) live alongside in a plain JSON file,
following the exact directory convention ``history.py`` already
established.

``keyring`` is an optional dependency (see the ``agentic`` extra in
pyproject.toml) and is imported lazily so importing this module doesn't
break installs that don't have it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_ENV_VAR_BY_PROVIDER = {
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

_KEYRING_SERVICE = "djcues"


class KeyringUnavailableError(RuntimeError):
    """Raised when `keyring` isn't installed."""


def _require_keyring() -> Any:
    try:
        import keyring
    except ImportError as e:
        raise KeyringUnavailableError(
            "The agentic analysis mode needs the 'keyring' package. "
            "Install it with: pip install djcues[agentic]"
        ) from e
    return keyring


def default_config_path() -> Path:
    """Return ``~/.djcues/config.json``, creating the parent directory if needed."""
    config_dir = Path.home() / ".djcues"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "config.json"


def load_config(config_path: Path | None = None) -> dict:
    """Load non-secret settings (provider, model, batch preference, ...).

    Returns an empty dict if no config has been saved yet.
    """
    config_path = config_path or default_config_path()
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict, config_path: Path | None = None) -> None:
    """Save non-secret settings. Never pass an API key in `config` --
    this file is plaintext by design; keys belong in `set_api_key()`."""
    config_path = config_path or default_config_path()
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def set_api_key(provider: str, api_key: str) -> None:
    """Store an API key in the OS credential store."""
    keyring = _require_keyring()
    keyring.set_password(_KEYRING_SERVICE, f"{provider}_api_key", api_key)


def clear_api_key(provider: str) -> None:
    """Remove a stored API key from the OS credential store.

    No-ops (rather than raising) if nothing was stored, matching
    `keyring`'s own `delete_password` semantics on most backends.
    """
    keyring = _require_keyring()
    try:
        keyring.delete_password(_KEYRING_SERVICE, f"{provider}_api_key")
    except keyring.errors.PasswordDeleteError:
        pass


def resolve_api_key(provider: str) -> tuple[str | None, str | None]:
    """Resolve the API key for a provider.

    Returns (key, source) where source is "keyring", "env", or None if
    no key was found anywhere. Checked in that order: keyring first,
    then the provider's standard environment variable, so a locally
    configured key always wins over a stray env var left in a shell.
    """
    try:
        keyring = _require_keyring()
        key = keyring.get_password(_KEYRING_SERVICE, f"{provider}_api_key")
        if key:
            return key, "keyring"
    except KeyringUnavailableError:
        pass  # fall through to env var

    env_var = _ENV_VAR_BY_PROVIDER.get(provider)
    if env_var:
        key = os.environ.get(env_var)
        if key:
            return key, "env"

    return None, None


def mask_key(api_key: str) -> str:
    """Mask a key for display, e.g. 'sk-ant-...ab12'. Never log/print a raw key."""
    if len(api_key) <= 8:
        return "***"
    return f"{api_key[:7]}...{api_key[-4:]}"
