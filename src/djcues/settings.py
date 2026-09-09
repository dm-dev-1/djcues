"""Persistent, cross-session playback-preview settings for the review UI.

Unlike a review session (one ephemeral, gitignored JSON per playlist,
scoped to a single `djcues review` invocation on a random port -- see
`review.py`/`server.py`), these settings need to survive across every
future invocation too. Browser localStorage can't do that job here:
`start_server()` uses port=0 (an OS-assigned port), so the review page's
origin changes every session, and localStorage is origin-scoped -- a
value saved in one session is invisible in the next. This follows the
exact `~/.djcues/` directory convention `auth.py`'s `default_config_path()`
and `history.py`'s `default_db_path()` already established, as a sibling
file rather than folding into `auth.py`'s config.json (that file is
specifically BYOK provider/model config, a different concern).
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_PLAYBACK_SETTINGS: dict = {
    "preview_pre_roll_bars": 4,
    "preview_loop_bars": 8,
    "preview_loop_enabled": True,
}
# Deliberately "preview_*"-prefixed -- the review session JSON already has
# an unrelated "loop_length_bars" (the actual Rekordbox loop-cue length
# written to the DB, see server.py/writer.py) -- reusing a similar name
# here would be an easy real mix-up between two unrelated "loop length"
# concepts.


def default_settings_path() -> Path:
    """Return ``~/.djcues/playback_settings.json``, creating the parent
    directory if needed."""
    settings_dir = Path.home() / ".djcues"
    settings_dir.mkdir(parents=True, exist_ok=True)
    return settings_dir / "playback_settings.json"


def load_playback_settings(settings_path: Path | None = None) -> dict:
    """Defaults merged with whatever's saved. Never raises for a missing
    or unreadable file -- always returns a complete, usable dict."""
    settings_path = settings_path or default_settings_path()
    saved: dict = {}
    if settings_path.exists():
        try:
            with open(settings_path, encoding="utf-8") as f:
                saved = json.load(f)
        except (json.JSONDecodeError, OSError):
            saved = {}
    merged = dict(DEFAULT_PLAYBACK_SETTINGS)
    merged.update({k: v for k, v in saved.items() if k in DEFAULT_PLAYBACK_SETTINGS})
    return merged


def save_playback_settings(settings: dict, settings_path: Path | None = None) -> dict:
    """Merge `settings` onto the existing saved values (unknown keys
    ignored -- validating the *values* is the HTTP handler's job, same
    division of responsibility as auth.py's load_config/save_config) and
    persist. Returns the full merged, saved dict."""
    settings_path = settings_path or default_settings_path()
    merged = load_playback_settings(settings_path)
    merged.update({k: v for k, v in settings.items() if k in DEFAULT_PLAYBACK_SETTINGS})
    with open(settings_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
        f.write("\n")
    return merged
