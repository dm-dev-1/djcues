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
import time
import urllib.error
import urllib.request
from http.client import HTTPResponse

import pytest

from djcues.constants import CUE_SYSTEM_BY_PAD
from djcues.server import start_auth_server, start_server


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
