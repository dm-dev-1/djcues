"""Tests for djcues.server -- the two local HTTP servers (the review
session bridge and the BYOK setup wizard).

Unlike viz.py/review.py, this module is genuinely about I/O -- request
routing, JSON bodies, session-file read/write, CORS -- so these tests
run real servers (start_server/start_auth_server, port=0 for an
OS-assigned ephemeral port, same as production) and make real HTTP
requests against them via the stdlib only (no requests/httpx --
neither is a declared djcues dependency, even though both happen to be
installed transitively in this venv right now). Only the two calls
that leave the process entirely (djcues.providers.get_provider,
djcues.auth.*) are mocked.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.client import HTTPResponse
from unittest.mock import patch

import pytest

from djcues.constants import CUE_SYSTEM_BY_PAD
from djcues.server import start_auth_server, start_dashboard_server, start_server
from djcues.server import _DbWorker
from tests.conftest import requires_rekordbox


# ---------------------------------------------------------------------------
# Minimal stdlib HTTP client helpers
# ---------------------------------------------------------------------------


def _request(method: str, url: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    # Retry once on a raw connection-level failure (ConnectionAbortedError/
    # ConnectionResetError, an OSError subclass -- not urllib.error.HTTPError,
    # so it isn't a real 4xx/5xx from the handler). Observed intermittently
    # in the full suite, never when this file runs alone or under light
    # load -- consistent with transient Windows TCP-stack jitter after many
    # rapid real socket open/close cycles in one process, not a logic bug:
    # the same request against the same running server succeeds on retry
    # every time this has been seen. Every route exercised here is either
    # a GET or a POST to a 404/error path with no side effect, so a retry
    # is safe -- this isn't papering over a real assertion failure, it's
    # tolerating the kind of transient hiccup any real HTTP client needs
    # to handle against a real server.
    last_conn_error: OSError | None = None
    for attempt in range(2):
        req = urllib.request.Request(url, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        try:
            resp: HTTPResponse = urllib.request.urlopen(req, timeout=5)
            status = resp.status
            raw = resp.read()
            break
        except urllib.error.HTTPError as e:
            status = e.code
            raw = e.read()
            break
        except (ConnectionAbortedError, ConnectionResetError) as e:
            last_conn_error = e
            time.sleep(0.2)
    else:
        raise last_conn_error
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        parsed = {"_raw": raw.decode("utf-8", errors="replace")}
    return status, parsed


def _get(url: str) -> tuple[int, dict]:
    return _request("GET", url)


def _post(url: str, body: dict | None = None) -> tuple[int, dict]:
    return _request("POST", url, body or {})


def _get_headers(url: str) -> dict[str, str]:
    resp = urllib.request.urlopen(url, timeout=5)
    return dict(resp.headers.items())


def _options_headers(url: str) -> tuple[int, dict[str, str]]:
    req = urllib.request.Request(url, method="OPTIONS")
    resp = urllib.request.urlopen(req, timeout=5)
    return resp.status, dict(resp.headers.items())


def _shutdown(server) -> None:
    """Stop a server's serving thread promptly instead of waiting out its
    10s handle_request() socket timeout -- set the flag, then send one
    harmless request to wake the currently-blocked handle_request() call
    so the loop re-checks the flag immediately. Then actually wait for
    the socket to close (not just for this call to return) before handing
    control back to the next test: this file starts 35+ real servers,
    and leaving each one's teardown to finish in the background let
    enough sockets/threads pile up across the file to cause intermittent
    ConnectionAbortedError on unrelated later tests -- reproducible in
    the full run, absent when this file runs alone. Waiting here for a
    real fileno()==-1 confirms full OS-level cleanup, not just that this
    function returned."""
    server._shutdown_flag = True
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/", timeout=2)
    except Exception:
        pass
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and server.socket.fileno() != -1:
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def offset_pads() -> tuple[str, str]:
    """(pad with memory_offset_bars == 0, pad with memory_offset_bars != 0)
    -- looked up from the real cue system rather than assumed, so this
    stays correct even if the cue system definition ever changes."""
    zero_pad = next(p for p, slot in CUE_SYSTEM_BY_PAD.items() if slot.memory_offset_bars == 0)
    nonzero_pad = next(p for p, slot in CUE_SYSTEM_BY_PAD.items() if slot.memory_offset_bars != 0)
    return zero_pad, nonzero_pad


def _memory_key(pad: str) -> str:
    """The handler derives a cue's memory-slot key from its 1-indexed
    alphabetical position (see _handle_cue_update's pads.index(pad)+1) --
    not from anything in the session data itself, so fixtures must key
    memory_cues the same way or the recalculation silently no-ops on a
    key that's never touched."""
    return str("ABCDEFGH".index(pad) + 1)


def _build_session(offset_pads: tuple[str, str], first_beat_ms: float = 100.0, bpm: float = 128.0) -> dict:
    zero_pad, nonzero_pad = offset_pads
    return {
        "playlist": "Test Playlist",
        "playlist_id": 1,
        "settings": {"memory_offset_bars": 16, "loop_length_bars": 4},
        "tracks": {
            "1": {
                "title": "Pending Track",
                "bpm": bpm,
                "first_beat_ms": first_beat_ms,
                "status": "pending",
                "cues": {
                    zero_pad: {"position_ms": 1000.0, "loop_end_ms": None, "status": "pending", "confidence": 0.9},
                    nonzero_pad: {"position_ms": 50_000.0, "loop_end_ms": None, "status": "pending", "confidence": 0.8},
                },
                "memory_cues": {
                    _memory_key(zero_pad): {"position_ms": 1000.0, "loop_end_ms": None, "status": "pending"},
                    _memory_key(nonzero_pad): {"position_ms": 20_000.0, "loop_end_ms": None, "status": "pending"},
                },
            },
            "2": {
                "title": "Already Accepted Track",
                "bpm": bpm,
                "first_beat_ms": first_beat_ms,
                "status": "accepted",
                "cues": {zero_pad: {"position_ms": 500.0, "loop_end_ms": None, "status": "accepted", "confidence": 1.0}},
                "memory_cues": {"1": {"position_ms": 500.0, "loop_end_ms": None, "status": "accepted"}},
            },
        },
    }


@pytest.fixture
def review_server(tmp_path, offset_pads):
    html_path = tmp_path / "review.html"
    html_path.write_text("<html><body>review page</body></html>", encoding="utf-8")
    session_path = tmp_path / "session.json"
    session_path.write_text(json.dumps(_build_session(offset_pads)), encoding="utf-8")

    server, port = start_server(html_path=html_path, session_path=session_path)
    base_url = f"http://127.0.0.1:{port}"
    yield base_url, session_path, offset_pads
    _shutdown(server)


@pytest.fixture
def auth_server():
    server, port = start_auth_server(html_body=b"<html><body>auth setup</body></html>")
    base_url = f"http://127.0.0.1:{port}"
    yield base_url, server
    _shutdown(server)


# ---------------------------------------------------------------------------
# ReviewHandler -- GET
# ---------------------------------------------------------------------------


class TestReviewHandlerGet:
    def test_root_serves_html(self, review_server):
        base_url, _session_path, _pads = review_server
        status, body = _request("GET", f"{base_url}/")
        assert status == 200

        req = urllib.request.Request(f"{base_url}/")
        resp = urllib.request.urlopen(req, timeout=5)
        assert resp.headers["Content-Type"] == "text/html; charset=utf-8"
        assert b"review page" in resp.read()

    def test_index_html_serves_same_page(self, review_server):
        base_url, _session_path, _pads = review_server
        resp = urllib.request.urlopen(f"{base_url}/index.html", timeout=5)
        assert resp.status == 200
        assert b"review page" in resp.read()

    def test_session_endpoint_returns_real_file_contents(self, review_server):
        base_url, session_path, _pads = review_server
        status, body = _get(f"{base_url}/session")
        assert status == 200
        on_disk = json.loads(session_path.read_text(encoding="utf-8"))
        assert body == on_disk

    def test_unknown_path_is_404(self, review_server):
        base_url, _session_path, _pads = review_server
        status, body = _get(f"{base_url}/nope")
        assert status == 404
        assert body == {"error": "not found"}

    def test_options_returns_cors_headers(self, review_server):
        base_url, _session_path, _pads = review_server
        status, headers = _options_headers(f"{base_url}/session")
        assert status == 200
        assert headers["Access-Control-Allow-Origin"] == "*"
        assert "POST" in headers["Access-Control-Allow-Methods"]


# ---------------------------------------------------------------------------
# ReviewHandler -- POST /session/accept-all
# ---------------------------------------------------------------------------


class TestReviewHandlerAcceptAll:
    def test_flips_only_pending_entries(self, review_server):
        base_url, session_path, _pads = review_server
        status, body = _post(f"{base_url}/session/accept-all")
        assert status == 200
        assert body == {"ok": True}

        session = json.loads(session_path.read_text(encoding="utf-8"))
        assert session["tracks"]["1"]["status"] == "accepted"
        for cue in session["tracks"]["1"]["cues"].values():
            assert cue["status"] == "accepted"
        for mc in session["tracks"]["1"]["memory_cues"].values():
            assert mc["status"] == "accepted"
        # Already-accepted track 2 is untouched, not re-processed
        assert session["tracks"]["2"]["status"] == "accepted"


# ---------------------------------------------------------------------------
# ReviewHandler -- POST /session/track/<id>[/status]
# ---------------------------------------------------------------------------


class TestReviewHandlerTrackStatus:
    def test_accept_cascades_to_cues_and_memory_cues(self, review_server):
        base_url, session_path, _pads = review_server
        status, body = _post(f"{base_url}/session/track/1/status", {"status": "accepted"})
        assert status == 200
        session = json.loads(session_path.read_text(encoding="utf-8"))
        t1 = session["tracks"]["1"]
        assert t1["status"] == "accepted"
        assert all(c["status"] == "accepted" for c in t1["cues"].values())
        assert all(m["status"] == "accepted" for m in t1["memory_cues"].values())

    def test_alias_path_without_status_suffix_also_works(self, review_server):
        base_url, session_path, _pads = review_server
        status, body = _post(f"{base_url}/session/track/1", {"status": "skipped"})
        assert status == 200
        session = json.loads(session_path.read_text(encoding="utf-8"))
        assert session["tracks"]["1"]["status"] == "skipped"

    def test_invalid_status_is_400(self, review_server):
        base_url, _session_path, _pads = review_server
        status, body = _post(f"{base_url}/session/track/1/status", {"status": "bogus"})
        assert status == 400
        assert body == {"error": "invalid status"}

    def test_unknown_track_id_is_404(self, review_server):
        base_url, _session_path, _pads = review_server
        status, body = _post(f"{base_url}/session/track/999/status", {"status": "accepted"})
        assert status == 404
        assert body == {"error": "track not found"}


# ---------------------------------------------------------------------------
# ReviewHandler -- POST /session/track/<id>/cue/<pad>
# ---------------------------------------------------------------------------


class TestReviewHandlerCueUpdate:
    def test_adjust_updates_position_and_marks_track_adjusted(self, review_server):
        base_url, session_path, (zero_pad, _nonzero) = review_server
        status, body = _post(
            f"{base_url}/session/track/1/cue/{zero_pad}", {"status": "adjusted", "position_ms": 1234.0}
        )
        assert status == 200
        session = json.loads(session_path.read_text(encoding="utf-8"))
        cue = session["tracks"]["1"]["cues"][zero_pad]
        assert cue["position_ms"] == 1234.0
        assert cue["status"] == "adjusted"
        assert cue["original_ms"] == 1000.0  # the pre-adjustment position, captured once
        assert session["tracks"]["1"]["status"] == "adjusted"

    def test_original_ms_is_captured_only_on_first_adjustment(self, review_server):
        base_url, session_path, (zero_pad, _nonzero) = review_server
        _post(f"{base_url}/session/track/1/cue/{zero_pad}", {"status": "adjusted", "position_ms": 1111.0})
        _post(f"{base_url}/session/track/1/cue/{zero_pad}", {"status": "adjusted", "position_ms": 2222.0})
        session = json.loads(session_path.read_text(encoding="utf-8"))
        cue = session["tracks"]["1"]["cues"][zero_pad]
        assert cue["position_ms"] == 2222.0
        assert cue["original_ms"] == 1000.0  # still the very first value, not 1111.0

    def test_zero_offset_pad_memory_cue_mirrors_hot_cue_exactly(self, review_server):
        base_url, session_path, (zero_pad, _nonzero) = review_server
        assert CUE_SYSTEM_BY_PAD[zero_pad].memory_offset_bars == 0
        status, _ = _post(
            f"{base_url}/session/track/1/cue/{zero_pad}", {"status": "adjusted", "position_ms": 5000.0, "loop_end_ms": 6000.0}
        )
        assert status == 200
        session = json.loads(session_path.read_text(encoding="utf-8"))
        mc = session["tracks"]["1"]["memory_cues"][_memory_key(zero_pad)]
        assert mc["position_ms"] == 5000.0
        assert mc["loop_end_ms"] == 6000.0
        assert mc["status"] == "auto"

    def test_nonzero_offset_pad_memory_cue_is_bars_before(self, review_server):
        base_url, session_path, (_zero, nonzero_pad) = review_server
        slot = CUE_SYSTEM_BY_PAD[nonzero_pad]
        assert slot.memory_offset_bars != 0
        new_position = 80_000.0
        status, _ = _post(
            f"{base_url}/session/track/1/cue/{nonzero_pad}", {"status": "adjusted", "position_ms": new_position}
        )
        assert status == 200
        session = json.loads(session_path.read_text(encoding="utf-8"))
        bar_ms = (60_000 / 128.0) * 4
        expected = max(new_position - 16 * bar_ms, 100.0)  # first_beat_ms=100.0 from the fixture
        mc = session["tracks"]["1"]["memory_cues"][_memory_key(nonzero_pad)]
        assert mc["position_ms"] == pytest.approx(expected)
        assert mc["loop_end_ms"] is None
        assert mc["status"] == "auto"

    def test_nonzero_offset_pad_memory_cue_clamped_to_first_beat(self, review_server):
        # A position early enough that offset_bars before it would go
        # negative -- the recalculated memory cue must not move earlier
        # than the track's first beat.
        base_url, session_path, (_zero, nonzero_pad) = review_server
        status, _ = _post(
            f"{base_url}/session/track/1/cue/{nonzero_pad}", {"status": "adjusted", "position_ms": 200.0}
        )
        assert status == 200
        session = json.loads(session_path.read_text(encoding="utf-8"))
        mc = session["tracks"]["1"]["memory_cues"][_memory_key(nonzero_pad)]
        assert mc["position_ms"] == 100.0  # clamped to first_beat_ms, not a negative number

    def test_skip_marks_cue_and_memory_cue_skipped_without_moving_them(self, review_server):
        base_url, session_path, (zero_pad, _nonzero) = review_server
        status, _ = _post(f"{base_url}/session/track/1/cue/{zero_pad}", {"status": "skipped"})
        assert status == 200
        session = json.loads(session_path.read_text(encoding="utf-8"))
        cue = session["tracks"]["1"]["cues"][zero_pad]
        assert cue["status"] == "skipped"
        assert cue["position_ms"] == 1000.0  # unchanged
        assert session["tracks"]["1"]["memory_cues"][_memory_key(zero_pad)]["status"] == "skipped"

    def test_invalid_status_is_400(self, review_server):
        base_url, _session_path, (zero_pad, _nonzero) = review_server
        status, body = _post(f"{base_url}/session/track/1/cue/{zero_pad}", {"status": "bogus"})
        assert status == 400
        assert body == {"error": "invalid status"}

    def test_unknown_track_id_is_404(self, review_server):
        base_url, _session_path, (zero_pad, _nonzero) = review_server
        status, body = _post(f"{base_url}/session/track/999/cue/{zero_pad}", {"status": "adjusted"})
        assert status == 404
        assert body == {"error": "track not found"}

    def test_pad_not_in_tracks_cues_is_404(self, review_server):
        base_url, _session_path, _pads = review_server
        # 'G' is deliberately not in the fixture's cues dict for track 1.
        status, body = _post(f"{base_url}/session/track/1/cue/G", {"status": "adjusted"})
        assert status == 404
        assert body == {"error": "cue not found"}

    def test_malformed_pad_key_present_in_cues_hits_invalid_pad_branch(self, tmp_path):
        # The "invalid pad" 400 branch is unreachable via any session a
        # real create_session() produces (cues is always keyed by real
        # pad letters) -- exercise it directly with a hand-built session
        # to confirm it's still correct, defensive code.
        html_path = tmp_path / "review.html"
        html_path.write_text("<html></html>", encoding="utf-8")
        session_path = tmp_path / "session.json"
        session_path.write_text(json.dumps({
            "settings": {"memory_offset_bars": 16},
            "tracks": {"1": {"bpm": 128.0, "first_beat_ms": 0.0, "status": "pending",
                              "cues": {"Z": {"position_ms": 0.0, "status": "pending"}},
                              "memory_cues": {}}},
        }), encoding="utf-8")
        server, port = start_server(html_path=html_path, session_path=session_path)
        try:
            status, body = _post(f"http://127.0.0.1:{port}/session/track/1/cue/Z", {"status": "adjusted"})
            assert status == 400
            assert body == {"error": "invalid pad"}
        finally:
            _shutdown(server)

    def test_missing_memory_cue_entry_does_not_error(self, tmp_path, offset_pads):
        # A cue whose memory_key has no corresponding memory_cues entry
        # (e.g. a memory cue slot the proposal skipped) -- the hot cue
        # update must still succeed even though there's nothing to
        # recalculate.
        zero_pad, _nonzero = offset_pads
        html_path = tmp_path / "review.html"
        html_path.write_text("<html></html>", encoding="utf-8")
        session_path = tmp_path / "session.json"
        session_path.write_text(json.dumps({
            "settings": {"memory_offset_bars": 16},
            "tracks": {"1": {"bpm": 128.0, "first_beat_ms": 0.0, "status": "pending",
                              "cues": {zero_pad: {"position_ms": 0.0, "status": "pending"}},
                              "memory_cues": {}}},
        }), encoding="utf-8")
        server, port = start_server(html_path=html_path, session_path=session_path)
        try:
            status, body = _post(
                f"http://127.0.0.1:{port}/session/track/1/cue/{zero_pad}",
                {"status": "adjusted", "position_ms": 500.0},
            )
            assert status == 200
            assert body == {"ok": True}
        finally:
            _shutdown(server)


# ---------------------------------------------------------------------------
# ReviewHandler -- POST routing edge cases
# ---------------------------------------------------------------------------


class TestReviewHandlerPostRouting:
    def test_unmatched_track_subpath_is_404(self, review_server):
        base_url, _session_path, _pads = review_server
        status, body = _post(f"{base_url}/session/track/1/cue/A/extra")
        assert status == 404

    def test_completely_unknown_post_path_is_404(self, review_server):
        base_url, _session_path, _pads = review_server
        status, body = _post(f"{base_url}/nonsense")
        assert status == 404


# ---------------------------------------------------------------------------
# AuthSetupHandler -- GET
# ---------------------------------------------------------------------------


class TestAuthSetupHandlerGet:
    def test_root_serves_html_body(self, auth_server):
        base_url, _server = auth_server
        resp = urllib.request.urlopen(f"{base_url}/", timeout=5)
        assert resp.status == 200
        assert resp.headers["Content-Type"] == "text/html; charset=utf-8"
        assert resp.read() == b"<html><body>auth setup</body></html>"

    def test_unknown_path_is_404(self, auth_server):
        base_url, _server = auth_server
        status, body = _get(f"{base_url}/nope")
        assert status == 404


# ---------------------------------------------------------------------------
# AuthSetupHandler -- POST /models
# ---------------------------------------------------------------------------


class TestAuthSetupHandlerModels:
    def test_missing_provider_or_key_is_400(self, auth_server):
        base_url, _server = auth_server
        status, body = _post(f"{base_url}/models", {"provider": "anthropic"})
        assert status == 400
        assert "required" in body["error"]

    def test_unknown_provider_is_400(self, auth_server):
        base_url, _server = auth_server
        status, body = _post(f"{base_url}/models", {"provider": "openai", "api_key": "sk-x"})
        assert status == 400

    def test_happy_path_returns_sorted_priced_models(self, auth_server, monkeypatch):
        from djcues.providers import ModelInfo
        from unittest.mock import MagicMock, patch

        base_url, _server = auth_server
        fake_provider = MagicMock()
        fake_provider.list_models.return_value = [
            ModelInfo(id="claude-cheap", display_name="Cheap", provider="anthropic"),
            ModelInfo(id="claude-good", display_name="Good", provider="anthropic", context_window=200_000),
        ]
        with patch("djcues.providers.get_provider", return_value=fake_provider), \
             patch("djcues.providers.DEFAULT_MODEL", {"anthropic": "claude-cheap"}), \
             patch("djcues.providers.RECOMMENDED_FOR_ACCURACY", {"anthropic": "claude-good"}):
            status, body = _post(f"{base_url}/models", {"provider": "anthropic", "api_key": "sk-secret-value"})

        assert status == 200
        assert body["default_model"] == "claude-cheap"
        assert body["recommended_accuracy_model"] == "claude-good"
        # Accuracy-recommended model sorts first, same real behavior
        # confirmed for the CLI's auth set in test_cli.py.
        assert [m["id"] for m in body["models"]] == ["claude-good", "claude-cheap"]
        assert body["models"][0]["recommended_accuracy"] is True
        assert body["models"][1]["recommended"] is True
        # The API key must never be echoed back in any response.
        assert "sk-secret-value" not in json.dumps(body)

    def test_no_models_returned_is_400(self, auth_server):
        from unittest.mock import MagicMock, patch

        base_url, _server = auth_server
        fake_provider = MagicMock()
        fake_provider.list_models.return_value = []
        with patch("djcues.providers.get_provider", return_value=fake_provider):
            status, body = _post(f"{base_url}/models", {"provider": "anthropic", "api_key": "sk-x"})
        assert status == 400
        assert "No models returned" in body["error"]

    def test_provider_error_is_400_and_never_leaks_key(self, auth_server):
        from unittest.mock import MagicMock, patch

        base_url, _server = auth_server
        fake_provider = MagicMock()
        fake_provider.list_models.side_effect = RuntimeError("invalid key")
        with patch("djcues.providers.get_provider", return_value=fake_provider):
            status, body = _post(f"{base_url}/models", {"provider": "anthropic", "api_key": "sk-should-not-leak"})
        assert status == 400
        assert "Could not fetch models" in body["error"]
        assert "sk-should-not-leak" not in json.dumps(body)


# ---------------------------------------------------------------------------
# AuthSetupHandler -- POST /save
# ---------------------------------------------------------------------------


class TestAuthSetupHandlerSave:
    def test_missing_fields_is_400(self, auth_server):
        base_url, _server = auth_server
        status, body = _post(f"{base_url}/save", {"provider": "anthropic", "api_key": "sk-x"})
        assert status == 400

    def test_happy_path_saves_and_flips_setup_complete(self, auth_server):
        from unittest.mock import patch

        base_url, server = auth_server
        assert server._setup_complete is False
        saved = {}
        with patch("djcues.auth.set_api_key") as mock_set_key, \
             patch("djcues.auth.load_config", return_value={}), \
             patch("djcues.auth.save_config", side_effect=saved.update):
            status, body = _post(
                f"{base_url}/save",
                {"provider": "anthropic", "api_key": "sk-real-key", "model": "claude-x"},
            )
        assert status == 200
        assert body == {"ok": True, "provider": "anthropic", "model": "claude-x"}
        mock_set_key.assert_called_once_with("anthropic", "sk-real-key")
        assert saved == {"provider": "anthropic", "model": "claude-x"}
        assert server._setup_complete is True
        assert "sk-real-key" not in json.dumps(body)

    def test_keyring_unavailable_is_500_and_does_not_complete_setup(self, auth_server):
        from unittest.mock import patch
        from djcues.auth import KeyringUnavailableError

        base_url, server = auth_server
        with patch("djcues.auth.set_api_key", side_effect=KeyringUnavailableError("no backend")):
            status, body = _post(
                f"{base_url}/save",
                {"provider": "anthropic", "api_key": "sk-x", "model": "claude-x"},
            )
        assert status == 500
        assert "no backend" in body["error"]
        assert server._setup_complete is False


# ---------------------------------------------------------------------------
# start_server / start_auth_server orchestration
# ---------------------------------------------------------------------------


class TestServerLifecycle:
    def test_start_server_returns_bound_ephemeral_port(self, tmp_path):
        html_path = tmp_path / "r.html"
        html_path.write_text("<html></html>", encoding="utf-8")
        session_path = tmp_path / "s.json"
        session_path.write_text("{}", encoding="utf-8")
        server, port = start_server(html_path=html_path, session_path=session_path, port=0)
        try:
            assert isinstance(port, int) and port > 0
            assert server.server_address[1] == port
            status, _ = _get(f"http://127.0.0.1:{port}/nonexistent")
            assert status == 404  # a real request actually reaches the server
        finally:
            _shutdown(server)

    def test_idle_timeout_closes_the_server(self, tmp_path):
        html_path = tmp_path / "r.html"
        html_path.write_text("<html></html>", encoding="utf-8")
        session_path = tmp_path / "s.json"
        session_path.write_text("{}", encoding="utf-8")
        # A fraction-of-a-minute timeout so this test doesn't take real
        # minutes to prove the mechanism works. Note this can't actually
        # close the server faster than ~10s regardless of how small
        # timeout_minutes is: _serve()'s own handle_request() socket
        # timeout is hardcoded to 10s, and the idle check only runs after
        # that call returns -- so the real floor here is
        # max(10s, timeout_seconds), not timeout_seconds alone.
        server, port = start_server(
            html_path=html_path, session_path=session_path, port=0, timeout_minutes=1 / 30
        )
        try:
            deadline = time.monotonic() + 15
            closed = False
            while time.monotonic() < deadline:
                if server.socket.fileno() == -1:
                    closed = True
                    break
                time.sleep(0.2)
            assert closed, "server did not close itself after its idle timeout elapsed"
        finally:
            server._shutdown_flag = True  # already closed; just don't leave the flag unset

    def test_audio_paths_param_is_optional_and_backward_compatible(self, tmp_path):
        html_path = tmp_path / "r.html"
        html_path.write_text("<html></html>", encoding="utf-8")
        session_path = tmp_path / "s.json"
        session_path.write_text("{}", encoding="utf-8")
        # No audio_paths passed at all -- every caller before this feature
        # existed still works exactly as before.
        server, port = start_server(html_path=html_path, session_path=session_path)
        try:
            status, body = _get(f"http://127.0.0.1:{port}/audio/1")
            assert status == 404
            assert "no audio available" in body["error"]
        finally:
            _shutdown(server)


# ---------------------------------------------------------------------------
# _parse_range_header -- pure unit tests, no server needed at all
# ---------------------------------------------------------------------------


class TestParseRangeHeader:
    def test_open_ended_range(self):
        from djcues.server import _parse_range_header
        assert _parse_range_header("bytes=10-", 100) == (10, 99)

    def test_bounded_range(self):
        from djcues.server import _parse_range_header
        assert _parse_range_header("bytes=10-19", 100) == (10, 19)

    def test_suffix_range(self):
        from djcues.server import _parse_range_header
        assert _parse_range_header("bytes=-10", 100) == (90, 99)

    def test_suffix_range_larger_than_file_clamps_to_start(self):
        from djcues.server import _parse_range_header
        assert _parse_range_header("bytes=-500", 100) == (0, 99)

    def test_end_beyond_file_size_clamps_to_last_byte(self):
        from djcues.server import _parse_range_header
        assert _parse_range_header("bytes=50-999", 100) == (50, 99)

    def test_start_beyond_file_size_is_unsatisfiable(self):
        from djcues.server import _parse_range_header
        assert _parse_range_header("bytes=100-200", 100) is None

    def test_missing_bytes_unit_is_rejected(self):
        from djcues.server import _parse_range_header
        assert _parse_range_header("items=0-10", 100) is None

    def test_multi_range_is_rejected(self):
        from djcues.server import _parse_range_header
        assert _parse_range_header("bytes=0-10,20-30", 100) is None

    def test_no_dash_is_malformed(self):
        from djcues.server import _parse_range_header
        assert _parse_range_header("bytes=10", 100) is None

    def test_non_numeric_is_malformed(self):
        from djcues.server import _parse_range_header
        assert _parse_range_header("bytes=abc-def", 100) is None

    def test_empty_suffix_length_is_malformed(self):
        from djcues.server import _parse_range_header
        assert _parse_range_header("bytes=-", 100) is None

    def test_zero_suffix_length_is_malformed(self):
        from djcues.server import _parse_range_header
        assert _parse_range_header("bytes=-0", 100) is None


# ---------------------------------------------------------------------------
# /settings and /audio/<id> -- real servers, real requests, same
# established pattern as the rest of this file.
# ---------------------------------------------------------------------------


def _get_raw(url: str, headers: dict | None = None) -> tuple[int, dict, bytes]:
    """Like _get, but returns raw response bytes and headers instead of
    JSON-parsing the body -- needed for the binary audio endpoint."""
    req = urllib.request.Request(url, headers=headers or {})
    try:
        resp = urllib.request.urlopen(req, timeout=5)
        return resp.status, dict(resp.headers.items()), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers.items()), e.read()


class TestSettingsEndpoint:
    def test_get_returns_defaults_when_no_file_exists(self, review_server, tmp_path, monkeypatch):
        from djcues import settings as settings_module

        monkeypatch.setattr(settings_module, "default_settings_path", lambda: tmp_path / "nonexistent" / "playback_settings.json")
        base_url, _session_path, _pads = review_server
        status, body = _get(f"{base_url}/settings")
        assert status == 200
        assert body == settings_module.DEFAULT_PLAYBACK_SETTINGS

    def test_post_partial_update_merges_and_persists(self, review_server, tmp_path, monkeypatch):
        from djcues import settings as settings_module

        settings_path = tmp_path / "playback_settings.json"
        monkeypatch.setattr(settings_module, "default_settings_path", lambda: settings_path)
        base_url, _session_path, _pads = review_server

        status, body = _post(f"{base_url}/settings", {"preview_pre_roll_bars": 8})
        assert status == 200
        assert body["preview_pre_roll_bars"] == 8
        assert body["preview_loop_bars"] == settings_module.DEFAULT_PLAYBACK_SETTINGS["preview_loop_bars"]

        # A second GET reflects the persisted value, not just the POST response.
        status, body2 = _get(f"{base_url}/settings")
        assert status == 200
        assert body2 == body

    def test_post_negative_number_is_400(self, review_server, tmp_path, monkeypatch):
        from djcues import settings as settings_module

        monkeypatch.setattr(settings_module, "default_settings_path", lambda: tmp_path / "playback_settings.json")
        base_url, _session_path, _pads = review_server
        status, body = _post(f"{base_url}/settings", {"preview_loop_bars": -1})
        assert status == 400
        assert "positive number" in body["error"]

    def test_post_wrong_type_for_boolean_is_400(self, review_server, tmp_path, monkeypatch):
        from djcues import settings as settings_module

        monkeypatch.setattr(settings_module, "default_settings_path", lambda: tmp_path / "playback_settings.json")
        base_url, _session_path, _pads = review_server
        status, body = _post(f"{base_url}/settings", {"preview_loop_enabled": "yes"})
        assert status == 400
        assert "boolean" in body["error"]

    def test_post_bool_for_numeric_field_is_400(self, review_server, tmp_path, monkeypatch):
        # bool is a subclass of int in Python -- confirm True/False are
        # explicitly rejected for a numeric field, not silently accepted as 1/0.
        from djcues import settings as settings_module

        monkeypatch.setattr(settings_module, "default_settings_path", lambda: tmp_path / "playback_settings.json")
        base_url, _session_path, _pads = review_server
        status, body = _post(f"{base_url}/settings", {"preview_pre_roll_bars": True})
        assert status == 400

    def test_post_unknown_key_is_ignored_not_an_error(self, review_server, tmp_path, monkeypatch):
        from djcues import settings as settings_module

        monkeypatch.setattr(settings_module, "default_settings_path", lambda: tmp_path / "playback_settings.json")
        base_url, _session_path, _pads = review_server
        status, body = _post(f"{base_url}/settings", {"bogus_setting": 123})
        assert status == 200
        assert "bogus_setting" not in body


class TestAudioEndpoint:
    def _start_with_audio(self, tmp_path, content: bytes = b"0123456789" * 10, filename: str = "test.mp3"):
        html_path = tmp_path / "r.html"
        html_path.write_text("<html></html>", encoding="utf-8")
        session_path = tmp_path / "s.json"
        session_path.write_text("{}", encoding="utf-8")
        audio_file = tmp_path / filename
        audio_file.write_bytes(content)
        server, port = start_server(
            html_path=html_path, session_path=session_path, audio_paths={"1": str(audio_file)}
        )
        return server, f"http://127.0.0.1:{port}", audio_file

    def test_whole_file_get(self, tmp_path):
        server, base_url, audio_file = self._start_with_audio(tmp_path)
        try:
            status, headers, body = _get_raw(f"{base_url}/audio/1")
            assert status == 200
            assert body == audio_file.read_bytes()
            assert headers["Content-Type"] == "audio/mpeg"
            assert headers["Accept-Ranges"] == "bytes"
            assert headers["Content-Length"] == str(len(body))
        finally:
            _shutdown(server)

    def test_range_request_returns_206_with_correct_slice(self, tmp_path):
        server, base_url, audio_file = self._start_with_audio(tmp_path)
        try:
            status, headers, body = _get_raw(f"{base_url}/audio/1", headers={"Range": "bytes=10-19"})
            assert status == 206
            assert body == audio_file.read_bytes()[10:20]
            assert headers["Content-Range"] == f"bytes 10-19/{len(audio_file.read_bytes())}"
            assert headers["Content-Length"] == "10"
        finally:
            _shutdown(server)

    def test_suffix_range_form(self, tmp_path):
        server, base_url, audio_file = self._start_with_audio(tmp_path)
        try:
            status, headers, body = _get_raw(f"{base_url}/audio/1", headers={"Range": "bytes=-5"})
            assert status == 206
            assert body == audio_file.read_bytes()[-5:]
        finally:
            _shutdown(server)

    def test_out_of_bounds_range_is_416(self, tmp_path):
        server, base_url, audio_file = self._start_with_audio(tmp_path)
        try:
            file_size = len(audio_file.read_bytes())
            status, headers, body = _get_raw(
                f"{base_url}/audio/1", headers={"Range": f"bytes={file_size + 10}-{file_size + 20}"}
            )
            assert status == 416
            assert headers["Content-Range"] == f"bytes */{file_size}"
        finally:
            _shutdown(server)

    def test_unknown_track_id_is_404(self, tmp_path):
        server, base_url, _audio_file = self._start_with_audio(tmp_path)
        try:
            status, body = _get(f"{base_url}/audio/999")
            assert status == 404
        finally:
            _shutdown(server)

    def test_file_moved_since_server_start_is_404(self, tmp_path):
        server, base_url, audio_file = self._start_with_audio(tmp_path)
        try:
            audio_file.unlink()
            status, body = _get(f"{base_url}/audio/1")
            assert status == 404
            assert "not found" in body["error"]
        finally:
            _shutdown(server)

    @pytest.mark.parametrize(
        "filename,expected_content_type",
        [
            ("t.mp3", "audio/mpeg"),
            ("t.flac", "audio/flac"),
            # .aiff/.aif deliberately excluded -- those now go through
            # _ensure_playable_audio's transcode path (see TestAudioTranscode),
            # so their real content-type is audio/wav on success, not
            # audio/aiff; this test's dummy (invalid-audio) fixture content
            # would only exercise the transcode-failure fallback, which
            # coincidentally also serves audio/aiff -- not a meaningful test
            # of either the pre- or post-transcode behavior.
            ("t.m4a", "audio/mp4"),
            ("t.mp4", "audio/mp4"),
            ("t.wav", "audio/wav"),
            ("t.ogg", "audio/ogg"),
            ("t.unknownext", "application/octet-stream"),
        ],
    )
    def test_content_type_per_extension(self, tmp_path, filename, expected_content_type):
        server, base_url, _audio_file = self._start_with_audio(tmp_path, filename=filename)
        try:
            status, headers, body = _get_raw(f"{base_url}/audio/1")
            assert status == 200
            assert headers["Content-Type"] == expected_content_type
        finally:
            _shutdown(server)


class TestAudioTranscode:
    """.aiff/.aif specifically -- confirmed live (real Chromium, via
    `new Audio().canPlayType('audio/aiff')` returning "") to be
    undecodable by a browser <audio> element, unlike every other format
    in _AUDIO_CONTENT_TYPES. _ensure_playable_audio transcodes those to a
    cached WAV; every other format is untouched (see
    test_non_transcoded_format_is_served_unchanged below and
    TestAudioEndpoint above, which already covers those)."""

    @staticmethod
    def _real_aiff_bytes(num_frames: int = 4410, samplerate: int = 44100) -> bytes:
        """A real, valid, decodable AIFF file's bytes -- silence is fine,
        transcoding only cares that soundfile can read the container/
        subtype, not the audio content."""
        import io

        import numpy as np
        import soundfile as sf

        buf = io.BytesIO()
        data = np.zeros((num_frames, 2), dtype="int16")
        sf.write(buf, data, samplerate, format="AIFF", subtype="PCM_16")
        return buf.getvalue()

    def _start_with_aiff(self, tmp_path, monkeypatch, content: bytes | None = None):
        from djcues import server as server_module

        cache_dir = tmp_path / "cache"

        def _fake_cache_dir():
            # Real _transcode_cache_dir() creates the directory as a side
            # effect -- a bare `lambda: cache_dir` stub skips that, so the
            # transcode's write silently fails (no parent dir) and falls
            # back to the original file, which looked exactly like an
            # actual transcode bug the first time this was caught.
            cache_dir.mkdir(parents=True, exist_ok=True)
            return cache_dir

        monkeypatch.setattr(server_module, "_transcode_cache_dir", _fake_cache_dir)

        html_path = tmp_path / "r.html"
        html_path.write_text("<html></html>", encoding="utf-8")
        session_path = tmp_path / "s.json"
        session_path.write_text("{}", encoding="utf-8")
        audio_file = tmp_path / "src" / "real.aiff"
        audio_file.parent.mkdir()
        audio_file.write_bytes(content if content is not None else self._real_aiff_bytes())
        server, port = start_server(
            html_path=html_path, session_path=session_path, audio_paths={"1": str(audio_file)}
        )
        return server, f"http://127.0.0.1:{port}", audio_file, cache_dir

    def test_real_aiff_is_transcoded_to_playable_wav(self, tmp_path, monkeypatch):
        server, base_url, _audio_file, cache_dir = self._start_with_aiff(tmp_path, monkeypatch)
        try:
            status, headers, body = _get_raw(f"{base_url}/audio/1")
            assert status == 200
            assert headers["Content-Type"] == "audio/wav"
            assert body[:4] == b"RIFF"  # real WAV container, not the AIFF's "FORM"
            assert (cache_dir / "1.wav").is_file()
        finally:
            _shutdown(server)

    def test_range_request_works_against_transcoded_wav(self, tmp_path, monkeypatch):
        """The whole point of transcoding through the cache rather than
        on-the-fly is that seeking still works -- Range support on the
        transcoded file, not just a whole-file fetch."""
        server, base_url, _audio_file, _cache_dir = self._start_with_aiff(tmp_path, monkeypatch)
        try:
            status, headers, body = _get_raw(f"{base_url}/audio/1", headers={"Range": "bytes=0-9"})
            assert status == 206
            assert len(body) == 10
            assert headers["Content-Range"].startswith("bytes 0-9/")
        finally:
            _shutdown(server)

    def test_second_request_hits_cache_not_a_fresh_transcode(self, tmp_path, monkeypatch):
        """soundfile is imported lazily inside _ensure_playable_audio, so
        there's no module-level name to mock and assert-not-called on --
        instead, a real re-transcode would rewrite (and thus re-stat a
        newer mtime for) the cached WAV; an unchanged mtime across two
        requests is direct proof the second one short-circuited on the
        cache check before ever touching soundfile."""
        server, base_url, _audio_file, cache_dir = self._start_with_aiff(tmp_path, monkeypatch)
        try:
            _get_raw(f"{base_url}/audio/1")
            cached_mtime = (cache_dir / "1.wav").stat().st_mtime
            time.sleep(0.05)

            status, headers, _body = _get_raw(f"{base_url}/audio/1")
            assert status == 200
            assert headers["Content-Type"] == "audio/wav"
            assert (cache_dir / "1.wav").stat().st_mtime == cached_mtime
        finally:
            _shutdown(server)

    def test_cache_invalidated_when_source_file_changes(self, tmp_path, monkeypatch):
        server, base_url, audio_file, cache_dir = self._start_with_aiff(tmp_path, monkeypatch)
        try:
            _get_raw(f"{base_url}/audio/1")
            first_size = (cache_dir / "1.wav").stat().st_size

            # A "changed" source file: different duration -> different
            # transcoded size, and a newer mtime than the cached WAV.
            time.sleep(0.05)
            audio_file.write_bytes(self._real_aiff_bytes(num_frames=8820))

            status, headers, _body = _get_raw(f"{base_url}/audio/1")
            assert status == 200
            assert (cache_dir / "1.wav").stat().st_size != first_size
        finally:
            _shutdown(server)

    def test_transcode_failure_falls_back_to_original_file(self, tmp_path, monkeypatch):
        """Not-actually-audio content (a corrupt/truncated download, say)
        makes soundfile raise -- must degrade to serving the original
        file/content-type, never a 500 or an empty response."""
        server, base_url, audio_file, cache_dir = self._start_with_aiff(
            tmp_path, monkeypatch, content=b"not a real aiff file"
        )
        try:
            status, headers, body = _get_raw(f"{base_url}/audio/1")
            assert status == 200
            assert headers["Content-Type"] == "audio/aiff"
            assert body == audio_file.read_bytes()
            assert not (cache_dir / "1.wav").is_file()
        finally:
            _shutdown(server)

    def test_non_transcoded_format_is_served_unchanged(self, tmp_path, monkeypatch):
        """A format that doesn't need transcoding (e.g. .wav) must never
        touch the cache dir or soundfile at all -- _ensure_playable_audio
        should short-circuit before either."""
        from djcues import server as server_module

        cache_dir = tmp_path / "cache"
        monkeypatch.setattr(server_module, "_transcode_cache_dir", lambda: cache_dir)
        html_path = tmp_path / "r.html"
        html_path.write_text("<html></html>", encoding="utf-8")
        session_path = tmp_path / "s.json"
        session_path.write_text("{}", encoding="utf-8")
        audio_file = tmp_path / "t.wav"
        audio_file.write_bytes(b"0123456789" * 10)
        server, port = start_server(
            html_path=html_path, session_path=session_path, audio_paths={"1": str(audio_file)}
        )
        base_url = f"http://127.0.0.1:{port}"
        try:
            status, headers, body = _get_raw(f"{base_url}/audio/1")
            assert status == 200
            assert headers["Content-Type"] == "audio/wav"
            assert body == audio_file.read_bytes()
            assert not cache_dir.exists()
        finally:
            _shutdown(server)


# ---------------------------------------------------------------------------
# DashboardHandler / _DbWorker / start_dashboard_server
#
# Unlike ReviewHandler/AuthSetupHandler, the dashboard fundamentally needs a
# real rekordbox connection for almost every route (playlist tree, track
# detail, job execution all go through _DbWorker or a job's own dedicated
# Rekordbox6Database -- neither is passed in, both construct their own).
# Gated on @requires_rekordbox, matching test_db.py's own established
# precedent for exactly this class of test -- a real DB is simpler and more
# trustworthy here than a large, fragile from-scratch mock of pyrekordbox's
# ORM. The expensive analysis internals (_get_proposer/_apply_refine_drops/
# verify_beat_grid) are still mocked for job tests, at their source in
# djcues.cli/djcues.beat_verify, matching test_cli.py's own established
# mocking boundary -- these tests exercise DashboardHandler's own routing/
# job-lifecycle/threading logic, not djcues's core analysis engines (which
# have their own tests elsewhere).
# ---------------------------------------------------------------------------

@pytest.fixture
def dashboard_server():
    server, port = start_dashboard_server(html_body=b"<html><body>dashboard shell</body></html>")
    base_url = f"http://127.0.0.1:{port}"
    yield base_url, server
    _shutdown(server)


@pytest.fixture(autouse=True)
def _no_real_cache_writes_from_dashboard_jobs():
    """Every dashboard job test in this section runs a real job thread
    through the real _apply_cache/analysis_cache path -- without this,
    a "fake proposer" test would still write a real (fake-content) row
    into the actual ~/.djcues/analysis_cache.db on whatever machine runs
    these tests. Matches test_cli.py's own no_analysis_cache autouse
    fixture exactly, for the same reason."""
    with patch("djcues.analysis_cache.get_cached", return_value=None), \
         patch("djcues.analysis_cache.store_result"):
        yield


@requires_rekordbox
class TestDbWorker:
    def test_run_returns_fn_result(self):
        worker = _DbWorker()
        assert worker.run(lambda db: 1 + 1) == 2

    def test_run_reraises_fn_exception_on_caller_thread(self):
        worker = _DbWorker()

        def raiser(db):
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            worker.run(raiser)

    def test_all_calls_observe_the_same_thread_identity(self):
        """The core safety property _DbWorker exists for -- every unit of
        DB work runs on the one thread the connection was created on,
        never the caller's own thread."""
        worker = _DbWorker()
        idents = [worker.run(lambda db: threading.get_ident()) for _ in range(5)]
        assert len(set(idents)) == 1
        assert idents[0] != threading.get_ident()

    def test_run_has_a_real_working_rekordbox_connection(self):
        worker = _DbWorker()
        playlists = worker.run(lambda db: list(db.get_playlist()))
        assert any(p.Name == "Tech House" for p in playlists)


@requires_rekordbox
class TestDashboardHandlerGet:
    def test_index_serves_html_body(self, dashboard_server):
        base_url, _server = dashboard_server
        resp = urllib.request.urlopen(base_url + "/", timeout=5)
        assert resp.status == 200
        assert b"dashboard shell" in resp.read()

    def test_playlists_returns_real_tree(self, dashboard_server):
        base_url, _server = dashboard_server
        status, data = _get(base_url + "/api/playlists")
        assert status == 200
        names = [n["name"] for n in data["tree"]]
        assert "Tech House" in names
        tech_house = next(n for n in data["tree"] if n["name"] == "Tech House")
        assert tech_house["kind"] == "playlist"
        assert tech_house["track_count"] == 10

    def test_playlist_tracks_returns_real_tracks(self, dashboard_server):
        base_url, _server = dashboard_server
        _, tree = _get(base_url + "/api/playlists")
        tech_house = next(n for n in tree["tree"] if n["name"] == "Tech House")
        status, data = _get(base_url + f"/api/playlists/{tech_house['id']}/tracks")
        assert status == 200
        assert len(data["tracks"]) == 10
        assert all(t["title"] for t in data["tracks"])

    def test_track_detail_returns_real_metadata(self, dashboard_server):
        base_url, _server = dashboard_server
        _, tree = _get(base_url + "/api/playlists")
        tech_house = next(n for n in tree["tree"] if n["name"] == "Tech House")
        _, listing = _get(base_url + f"/api/playlists/{tech_house['id']}/tracks")
        track = listing["tracks"][0]

        status, detail = _get(base_url + f"/api/tracks/{track['id']}")

        assert status == 200
        assert detail["title"] == track["title"]
        assert detail["has_phrases"] is True
        assert detail["existing_cue_count"] > 0
        # waveform/vocal_track are deliberately never included -- not
        # needed until a job actually renders a timeline.
        assert "waveform" not in detail
        assert "vocal_track" not in detail

    def test_track_detail_404_for_unknown_id(self, dashboard_server):
        base_url, _server = dashboard_server
        status, data = _get(base_url + "/api/tracks/not-a-real-track-id")
        assert status == 404

    def test_job_404_for_unknown_id(self, dashboard_server):
        base_url, _server = dashboard_server
        status, data = _get(base_url + "/api/jobs/not-a-real-job-id")
        assert status == 404

    def test_unknown_get_route_404(self, dashboard_server):
        base_url, _server = dashboard_server
        status, data = _get(base_url + "/api/totally/not/a/route")
        assert status == 404


@requires_rekordbox
class TestDashboardHandlerJobs:
    @staticmethod
    def _tech_house_track_id(base_url: str) -> str:
        _, tree = _get(base_url + "/api/playlists")
        tech_house = next(n for n in tree["tree"] if n["name"] == "Tech House")
        _, listing = _get(base_url + f"/api/playlists/{tech_house['id']}/tracks")
        return listing["tracks"][0]["id"]

    def test_rejects_unknown_kind(self, dashboard_server):
        base_url, _server = dashboard_server
        track_id = self._tech_house_track_id(base_url)
        status, data = _post(base_url + f"/api/tracks/{track_id}/jobs", {"kind": "not-a-real-kind"})
        assert status == 400

    def test_rejects_deep_without_refine_drops(self, dashboard_server):
        base_url, _server = dashboard_server
        track_id = self._tech_house_track_id(base_url)
        status, data = _post(
            base_url + f"/api/tracks/{track_id}/jobs",
            {"kind": "propose", "deep": True, "refine_drops": False},
        )
        assert status == 400
        assert "deep" in data["error"].lower()

    def _run_job_to_completion(self, base_url: str, track_id: str, body: dict, timeout: float = 20.0) -> dict:
        status, data = _post(base_url + f"/api/tracks/{track_id}/jobs", body)
        assert status == 202, data
        job_id = data["job_id"]

        deadline = time.monotonic() + timeout
        job: dict = {}
        while time.monotonic() < deadline:
            status, job = _get(base_url + f"/api/jobs/{job_id}")
            assert status == 200
            if job["status"] != "running":
                return job
            time.sleep(0.2)
        raise AssertionError(f"job {job_id} did not finish within {timeout}s: {job}")

    def test_propose_job_completes_with_mocked_proposer(self, dashboard_server):
        from djcues.models import CueProposal

        base_url, _server = dashboard_server
        track_id = self._tech_house_track_id(base_url)

        def fake_get_proposer(*args, **kwargs):
            def proposer(track):
                return CueProposal(
                    track=track, hot_cues=[], memory_cues=[], confidence={}, notes=["fake-propose-marker"]
                )
            return proposer, None, None, None

        with patch("djcues.cli._get_proposer", side_effect=fake_get_proposer):
            job = self._run_job_to_completion(base_url, track_id, {"kind": "propose"})

        assert job["status"] == "done"
        assert job["error"] is None
        assert "fake-propose-marker" in job["output_text"]
        assert job["html_fragment"]  # _render_track_body ran for real
        assert job["cache"] == {"hits": 0, "misses": 1}

    def test_compare_job_completes_with_mocked_proposer(self, dashboard_server):
        from djcues.models import CueProposal

        base_url, _server = dashboard_server
        track_id = self._tech_house_track_id(base_url)

        def fake_get_proposer(*args, **kwargs):
            def proposer(track):
                return CueProposal(track=track, hot_cues=[], memory_cues=[], confidence={}, notes=[])
            return proposer, None, None, None

        with patch("djcues.cli._get_proposer", side_effect=fake_get_proposer):
            job = self._run_job_to_completion(base_url, track_id, {"kind": "compare"})

        assert job["status"] == "done"
        assert "Precision" in job["output_text"] or "No existing hot cues" in job["output_text"]

    def test_beatgrid_job_completes_for_real(self, dashboard_server):
        # No mocking needed -- the free self-consistency tier is pure,
        # fast arithmetic over data the browsing routes already proved
        # readable; this is the one job kind cheap enough to run
        # genuinely end to end in a test.
        base_url, _server = dashboard_server
        track_id = self._tech_house_track_id(base_url)

        job = self._run_job_to_completion(base_url, track_id, {"kind": "beatgrid"})

        assert job["status"] == "done"
        assert job["error"] is None
        assert job["output_text"]
        assert job["html_fragment"] is None  # beatgrid has no (Track, CueProposal) pair to render

    def test_job_error_is_captured_not_a_crash(self, dashboard_server):
        base_url, _server = dashboard_server
        track_id = self._tech_house_track_id(base_url)

        def raising_get_proposer(*args, **kwargs):
            raise RuntimeError("synthetic failure for this test")

        with patch("djcues.cli._get_proposer", side_effect=raising_get_proposer):
            job = self._run_job_to_completion(base_url, track_id, {"kind": "propose"})

        assert job["status"] == "error"
        assert "synthetic failure" in job["error"]

    def test_browsing_stays_responsive_while_a_job_is_running(self, dashboard_server):
        """The actual proof of the whole thread-safety design (see
        _DbWorker's docstring) -- a slow job must not block a concurrent
        browsing request. An artificially slow mocked proposer stands in
        for a real multi-minute --deep run (already confirmed live,
        separately, against a real Demucs pass)."""
        base_url, _server = dashboard_server
        track_id = self._tech_house_track_id(base_url)

        def slow_get_proposer(*args, **kwargs):
            def proposer(track):
                time.sleep(1.5)
                from djcues.models import CueProposal
                return CueProposal(track=track, hot_cues=[], memory_cues=[], confidence={}, notes=[])
            return proposer, None, None, None

        with patch("djcues.cli._get_proposer", side_effect=slow_get_proposer):
            status, data = _post(base_url + f"/api/tracks/{track_id}/jobs", {"kind": "propose"})
            assert status == 202
            job_id = data["job_id"]

            # The job is now running (sleeping) on its own thread. A
            # browsing request must return promptly, not queue up behind it.
            start = time.monotonic()
            status, tree = _get(base_url + "/api/playlists")
            elapsed = time.monotonic() - start

            assert status == 200
            assert elapsed < 1.0, f"browsing blocked for {elapsed:.2f}s behind a running job"

            # Drain the job so the test doesn't leave a stray thread mid-sleep.
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                status, job = _get(base_url + f"/api/jobs/{job_id}")
                if job["status"] != "running":
                    break
                time.sleep(0.2)
            assert job["status"] == "done"

    def test_rejects_refine_drops_when_dependency_unavailable(self, dashboard_server):
        """The upfront synchronous check (mirrors cli.py's own propose/
        compare/review guard) -- a missing djcues[audio]/[ml] install is
        an immediate 400 before any thread spawns, not a job that fails
        after the fact. Mocked here rather than actually uninstalling
        librosa, matching how a real absent-dependency failure surfaces."""
        base_url, _server = dashboard_server
        track_id = self._tech_house_track_id(base_url)

        with patch("djcues.cli._check_refine_drops_available", return_value="Error: needs djcues[audio]"):
            status, data = _post(
                base_url + f"/api/tracks/{track_id}/jobs",
                {"kind": "propose", "refine_drops": True},
            )

        assert status == 400
        assert "djcues[audio]" in data["error"]


@requires_rekordbox
class TestDashboardHandlerLaunch:
    @staticmethod
    def _tech_house_playlist_and_track(base_url: str) -> tuple[str, str]:
        _, tree = _get(base_url + "/api/playlists")
        tech_house = next(n for n in tree["tree"] if n["name"] == "Tech House")
        _, listing = _get(base_url + f"/api/playlists/{tech_house['id']}/tracks")
        return tech_house["id"], listing["tracks"][0]["id"]

    def test_launch_viz_spawns_expected_argv_and_does_not_wait(self, dashboard_server):
        base_url, _server = dashboard_server
        playlist_id, track_id = self._tech_house_playlist_and_track(base_url)

        with patch("djcues.server.subprocess.Popen") as mock_popen:
            status, data = _post(
                base_url + f"/api/tracks/{track_id}/launch/viz", {"playlist_id": playlist_id}
            )

        assert status == 200
        assert data["playlist_name"] == "Tech House"
        mock_popen.assert_called_once()
        argv = mock_popen.call_args.args[0]
        assert argv[0] == sys.executable
        assert argv[1] == "-c"
        # Deliberately NOT "-m djcues.cli": cli.py has no
        # `if __name__ == "__main__"` guard, so -m silently does nothing
        # (confirmed live -- a real, shipped bug this test now guards
        # against regressing back to).
        assert "cli(sys.argv[1:])" in argv[2]
        assert argv[3] == "viz"
        assert argv[4] == "Tech House"
        assert argv[5] == data["track_title"]
        mock_popen.return_value.wait.assert_not_called()

    def test_launch_review_includes_its_own_flags_not_shared_with_propose_panel(self, dashboard_server):
        base_url, _server = dashboard_server
        playlist_id, track_id = self._tech_house_playlist_and_track(base_url)

        with patch("djcues.server.subprocess.Popen") as mock_popen:
            status, data = _post(
                base_url + f"/api/tracks/{track_id}/launch/review",
                {"playlist_id": playlist_id, "refine_drops": True, "deep": False, "agentic": False},
            )

        assert status == 200
        argv = mock_popen.call_args.args[0]
        assert argv[3] == "review"
        assert "--refine-drops" in argv
        assert "--deep" not in argv
        assert "--agentic" not in argv

    def test_launch_unknown_tool_404(self, dashboard_server):
        base_url, _server = dashboard_server
        playlist_id, track_id = self._tech_house_playlist_and_track(base_url)
        status, data = _post(
            base_url + f"/api/tracks/{track_id}/launch/not-a-real-tool",
            {"playlist_id": playlist_id},
        )
        assert status == 404

    def test_launch_missing_playlist_id_400(self, dashboard_server):
        base_url, _server = dashboard_server
        _playlist_id, track_id = self._tech_house_playlist_and_track(base_url)
        status, data = _post(base_url + f"/api/tracks/{track_id}/launch/viz", {})
        assert status == 400

    def test_launch_unknown_playlist_or_track_404(self, dashboard_server):
        base_url, _server = dashboard_server
        playlist_id, track_id = self._tech_house_playlist_and_track(base_url)

        with patch("djcues.server.subprocess.Popen") as mock_popen:
            status, data = _post(
                base_url + f"/api/tracks/{track_id}/launch/viz",
                {"playlist_id": "not-a-real-playlist-id"},
            )
        assert status == 404
        mock_popen.assert_not_called()

        with patch("djcues.server.subprocess.Popen") as mock_popen:
            status, data = _post(
                base_url + "/api/tracks/not-a-real-track-id/launch/viz",
                {"playlist_id": playlist_id},
            )
        assert status == 404
        mock_popen.assert_not_called()

    def test_launch_db_worker_failure_returns_500_not_a_crash(self, dashboard_server):
        base_url, _server = dashboard_server
        playlist_id, track_id = self._tech_house_playlist_and_track(base_url)

        with patch.object(_DbWorker, "run", side_effect=RuntimeError("worker exploded")):
            status, data = _post(
                base_url + f"/api/tracks/{track_id}/launch/viz", {"playlist_id": playlist_id}
            )

        assert status == 500
        assert "worker exploded" in data["error"]

    def test_launch_review_includes_agentic_and_deep_flags(self, dashboard_server):
        base_url, _server = dashboard_server
        playlist_id, track_id = self._tech_house_playlist_and_track(base_url)

        with patch("djcues.server.subprocess.Popen") as mock_popen:
            status, data = _post(
                base_url + f"/api/tracks/{track_id}/launch/review",
                {"playlist_id": playlist_id, "agentic": True, "deep": True, "refine_drops": False},
            )

        assert status == 200
        argv = mock_popen.call_args.args[0]
        assert "--agentic" in argv
        assert "--deep" in argv
        assert "--refine-drops" not in argv
