"""Local HTTP server for djcues review sessions.

Bridges the browser review UI to the session JSON file, handling CORS,
session reads/writes, and cue adjustment logic including memory cue
recalculation.
"""

from __future__ import annotations

import json
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Hardcoded rather than stdlib `mimetypes` -- that reads the OS registry
# (inconsistent especially on Windows) -- covers exactly the extensions
# that appear in real Rekordbox libraries this project has been tested
# against.
_AUDIO_CONTENT_TYPES: dict[str, str] = {
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
}


def _parse_range_header(header: str, file_size: int) -> tuple[int, int] | None:
    """Parse a single-range HTTP Range header (RFC 7233 §2.1 forms:
    ``bytes=N-M``, ``bytes=N-``, ``bytes=-N``) against a known file size.

    Returns an inclusive ``(start, end)`` byte range, or ``None`` if the
    header is malformed, uses an unsupported unit, or the range is
    unsatisfiable for this file size (the caller should respond 416).
    Multi-range requests (comma-separated) are treated as unsupported --
    real browsers requesting a single seekable `<audio>` stream only ever
    send one range at a time.
    """
    if not header.startswith("bytes=") or "," in header:
        return None
    spec = header[len("bytes="):].strip()
    if "-" not in spec:
        return None
    start_str, _, end_str = spec.partition("-")

    if start_str == "":
        # Suffix form: "bytes=-N" -- the last N bytes of the file.
        if end_str == "":
            return None
        try:
            suffix_len = int(end_str)
        except ValueError:
            return None
        if suffix_len <= 0:
            return None
        start = max(0, file_size - suffix_len)
        end = file_size - 1
    else:
        try:
            start = int(start_str)
        except ValueError:
            return None
        if end_str == "":
            end = file_size - 1
        else:
            try:
                end = int(end_str)
            except ValueError:
                return None

    if start < 0 or end < start or start >= file_size:
        return None
    end = min(end, file_size - 1)
    return start, end


class ReviewServer(socketserver.ThreadingMixIn, HTTPServer):
    """Threaded HTTPServer with a larger listen backlog.

    Single-threaded handling meant serving the large initial page (all
    proposals + waveforms for a big playlist) blocked every other request
    — including the review UI's own accept/skip calls — for as long as
    that transfer took. Threading lets those proceed concurrently. The
    stdlib backlog default (5) also drops connections outright when the
    review page fires many concurrent fetch() calls at once (e.g.
    "Accept All"), so it's raised too.
    """

    request_queue_size = 128
    daemon_threads = True


class ReviewHandler(BaseHTTPRequestHandler):
    """Request handler for the review session server.

    Class-level attributes ``html_path`` and ``session_path`` must be set
    before the handler is used (set by ``start_server`` via dynamic subclass).
    Session file reads/writes are serialized via ``_session_lock`` since the
    server now handles requests concurrently across threads. ``audio_paths``
    (track ID string -> local file path string) is likewise set by
    ``start_server``; a missing/empty dict just means no track has a
    resolvable local audio file, not an error.
    """

    html_path: Path
    session_path: Path
    audio_paths: dict[str, str] = {}
    _session_lock = threading.Lock()
    _settings_lock = threading.Lock()

    # --- helpers --------------------------------------------------------

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Suppress default stderr logging."""

    def _read_session(self) -> dict:
        """Read and parse the session JSON file."""
        return json.loads(self.session_path.read_text(encoding="utf-8"))

    def _write_session(self, session: dict) -> None:
        """Write session dict back to JSON (pretty-printed)."""
        self.session_path.write_text(
            json.dumps(session, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _send_json(self, data: dict, status: int = 200) -> None:
        """Send a JSON response with CORS headers."""
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        """Read and parse the request body as JSON."""
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw) if raw else {}

    def _touch_activity(self) -> None:
        """Update the server's last-activity timestamp."""
        if hasattr(self.server, "_last_activity"):
            self.server._last_activity = time.monotonic()

    # --- CORS -----------------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802
        """Handle CORS preflight requests."""
        self._touch_activity()
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # --- GET ------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        """Handle GET requests."""
        self._touch_activity()
        path = self.path.split("?")[0]  # strip query string

        if path in ("/", "/index.html"):
            self._serve_html()
        elif path == "/session":
            session = self._read_session()
            self._send_json(session)
        elif path == "/settings":
            self._handle_settings_get()
        elif path.startswith("/audio/"):
            self._handle_audio_get(path[len("/audio/"):])
        else:
            self._send_json({"error": "not found"}, status=404)

    def _serve_html(self) -> None:
        """Serve the review HTML file."""
        body = self.html_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_settings_get(self) -> None:
        """Return the current playback-preview settings (persisted
        cross-session -- see settings.py)."""
        from djcues.settings import load_playback_settings

        with self._settings_lock:
            settings = load_playback_settings()
        self._send_json(settings)

    def _handle_audio_get(self, track_id: str) -> None:
        """Stream a track's real local audio file, with HTTP Range
        support so a browser <audio> element can seek without
        re-downloading from byte 0 every time. Not a nice-to-have --
        previewing a cue near the end of a long file would otherwise
        stall on downloading everything before it first."""
        path_str = self.audio_paths.get(track_id)
        if not path_str:
            self._send_json({"error": "no audio available for this track"}, status=404)
            return

        path = Path(path_str)
        if not path.is_file():
            self._send_json({"error": "audio file not found"}, status=404)
            return

        content_type = _AUDIO_CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
        file_size = path.stat().st_size
        range_header = self.headers.get("Range")

        if range_header:
            parsed = _parse_range_header(range_header, file_size)
            if parsed is None:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                self.end_headers()
                return
            start, end = parsed
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        else:
            start, end = 0, file_size - 1
            length = file_size
            self.send_response(200)

        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

        try:
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # A scrub to a new position cancels the in-flight request for
            # the old one -- routine and expected here, not a real error.
            return

    # --- POST -----------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802
        """Handle POST requests."""
        self._touch_activity()
        path = self.path.split("?")[0]

        if path == "/session/accept-all":
            self._handle_accept_all()
        elif path.startswith("/session/track/"):
            self._route_track_post(path)
        elif path == "/settings":
            self._handle_settings_post()
        else:
            self._send_json({"error": "not found"}, status=404)

    def _handle_settings_post(self) -> None:
        """Update playback-preview settings. Body may contain any subset
        of the known keys; unknown keys are silently ignored (matching
        auth.py's config save, which is similarly trusting), but a known
        key with the wrong type/an out-of-range value is a 400 -- unlike
        auth.py's config, these values feed directly into playback-timing
        math client-side, so a bad value here would misbehave silently
        rather than just being cosmetically wrong."""
        from djcues.settings import DEFAULT_PLAYBACK_SETTINGS, save_playback_settings

        body = self._read_body()
        for key, value in body.items():
            if key not in DEFAULT_PLAYBACK_SETTINGS:
                continue
            if key == "preview_loop_enabled":
                if not isinstance(value, bool):
                    self._send_json({"error": f"{key} must be a boolean"}, status=400)
                    return
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                    self._send_json({"error": f"{key} must be a positive number"}, status=400)
                    return

        with self._settings_lock:
            merged = save_playback_settings(body)
        self._send_json(merged)

    def _handle_accept_all(self) -> None:
        """Set all pending tracks and their cues to accepted."""
        with self._session_lock:
            session = self._read_session()
            for tdata in session.get("tracks", {}).values():
                if tdata.get("status") == "pending":
                    tdata["status"] = "accepted"
                    for cue in tdata.get("cues", {}).values():
                        if cue.get("status") == "pending":
                            cue["status"] = "accepted"
                    for mc in tdata.get("memory_cues", {}).values():
                        if mc.get("status") == "pending":
                            mc["status"] = "accepted"
            self._write_session(session)
        self._send_json({"ok": True})

    def _route_track_post(self, path: str) -> None:
        """Route track-level POST requests to the right handler."""
        # Strip the prefix to get the remainder
        remainder = path[len("/session/track/"):]
        parts = remainder.split("/")

        # POST /session/track/<track_id>
        # POST /session/track/<track_id>/status  (alias used by JS)
        if len(parts) == 1 or (len(parts) == 2 and parts[1] == "status"):
            track_id = parts[0]
            self._handle_track_status(track_id)
        # POST /session/track/<track_id>/cue/<pad>
        elif len(parts) == 3 and parts[1] == "cue":
            track_id = parts[0]
            pad = parts[2]
            self._handle_cue_update(track_id, pad)
        else:
            self._send_json({"error": "not found"}, status=404)

    def _handle_track_status(self, track_id: str) -> None:
        """Update a track's status, cascading to all cues and memory cues."""
        body = self._read_body()
        new_status = body.get("status")
        if new_status not in ("accepted", "skipped"):
            self._send_json({"error": "invalid status"}, status=400)
            return

        with self._session_lock:
            session = self._read_session()
            tracks = session.get("tracks", {})
            if track_id not in tracks:
                self._send_json({"error": "track not found"}, status=404)
                return

            tdata = tracks[track_id]
            tdata["status"] = new_status
            # Cascade to all cues and memory cues
            for cue in tdata.get("cues", {}).values():
                cue["status"] = new_status
            for mc in tdata.get("memory_cues", {}).values():
                mc["status"] = new_status

            self._write_session(session)
        self._send_json({"ok": True})

    def _handle_cue_update(self, track_id: str, pad: str) -> None:
        """Update an individual cue, recalculating the memory cue."""
        from djcues.constants import CUE_SYSTEM

        body = self._read_body()
        new_status = body.get("status")
        if new_status not in ("adjusted", "skipped"):
            self._send_json({"error": "invalid status"}, status=400)
            return

        with self._session_lock:
            session = self._read_session()
            tracks = session.get("tracks", {})
            if track_id not in tracks:
                self._send_json({"error": "track not found"}, status=404)
                return

            tdata = tracks[track_id]
            cues = tdata.get("cues", {})
            if pad not in cues:
                self._send_json({"error": "cue not found"}, status=404)
                return

            pads = list("ABCDEFGH")
            if pad not in pads:
                self._send_json({"error": "invalid pad"}, status=400)
                return
            slot_idx = pads.index(pad)
            slot = CUE_SYSTEM[slot_idx]
            memory_key = str(slot_idx + 1)

            memory_cues = tdata.get("memory_cues", {})
            cue_entry = cues[pad]

            if new_status == "adjusted":
                # Store original position if not already stored
                if "original_ms" not in cue_entry:
                    cue_entry["original_ms"] = cue_entry["position_ms"]

                # Update position
                new_position = body.get("position_ms", cue_entry["position_ms"])
                new_loop_end = body.get("loop_end_ms", cue_entry.get("loop_end_ms"))
                cue_entry["position_ms"] = new_position
                cue_entry["loop_end_ms"] = new_loop_end
                cue_entry["status"] = "adjusted"

                # Recalculate corresponding memory cue
                offset_bars = session.get("settings", {}).get(
                    "memory_offset_bars", 16
                )
                if memory_key in memory_cues:
                    mc = memory_cues[memory_key]
                    if slot.memory_offset_bars == 0:
                        mc["position_ms"] = new_position
                        mc["loop_end_ms"] = new_loop_end
                    else:
                        bpm = tdata.get("bpm", 128.0)
                        bar_ms = (60_000 / bpm) * 4
                        mem_pos = new_position - offset_bars * bar_ms
                        first_beat = tdata.get("first_beat_ms", 0)
                        mc["position_ms"] = max(mem_pos, first_beat)
                        mc["loop_end_ms"] = None
                    mc["status"] = "auto"

            elif new_status == "skipped":
                cue_entry["status"] = "skipped"
                # Also skip the corresponding memory cue
                if memory_key in memory_cues:
                    memory_cues[memory_key]["status"] = "skipped"

            # Set track status to adjusted
            tdata["status"] = "adjusted"

            self._write_session(session)
        self._send_json({"ok": True})


def start_server(
    html_path: Path,
    session_path: Path,
    port: int = 0,
    timeout_minutes: int = 30,
    audio_paths: dict[str, str] | None = None,
) -> tuple[HTTPServer, int]:
    """Start the review server in a daemon thread.

    Returns ``(server, actual_port)`` where *actual_port* is the
    OS-assigned port when *port* is 0. *audio_paths* (track ID string ->
    local file path string) enables the /audio/<id> preview endpoint for
    whichever tracks have a resolvable local file; omit it (or pass an
    empty dict) to serve a review page with no audio preview available --
    backward compatible with every existing caller.
    """
    # Dynamically create a handler subclass with paths baked in as class
    # attributes, so each request handler instance can access them via self.
    handler = type(
        "BoundReviewHandler",
        (ReviewHandler,),
        {
            "html_path": html_path,
            "session_path": session_path,
            "audio_paths": audio_paths or {},
        },
    )

    server = ReviewServer(("127.0.0.1", port), handler)
    actual_port = server.server_address[1]
    server._last_activity = time.monotonic()  # type: ignore[attr-defined]
    server._shutdown_flag = False  # type: ignore[attr-defined]

    timeout_seconds = timeout_minutes * 60

    def _serve() -> None:
        server.timeout = 10  # handle_request blocks at most 10 s
        while not server._shutdown_flag:  # type: ignore[attr-defined]
            server.handle_request()
            elapsed = time.monotonic() - server._last_activity  # type: ignore[attr-defined]
            if elapsed > timeout_seconds:
                break
        server.server_close()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    return server, actual_port


class AuthSetupServer(HTTPServer):
    """Local-only HTTP server for the one-time BYOK setup wizard
    (`djcues auth web`). Bound to 127.0.0.1 only, same as ReviewServer."""


class AuthSetupHandler(BaseHTTPRequestHandler):
    """Serves the setup-wizard page and its two endpoints.

    ``html_body`` is baked in as a class attribute (via a dynamic subclass,
    same pattern as ReviewHandler) — the page uses only same-origin
    relative fetch() calls, so unlike the review page it never needs a
    server URL embedded in it. The API key travels in POST bodies only,
    is never written to a response, a log line, or any file — only
    ``djcues.auth.set_api_key()`` (the OS credential store) ever sees it.
    """

    html_body: bytes

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Suppress default stderr logging (no request here ever carries
        the key in its path, but suppressed anyway for consistency)."""

    def _touch_activity(self) -> None:
        if hasattr(self.server, "_last_activity"):
            self.server._last_activity = time.monotonic()

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        return json.loads(raw) if raw else {}

    def do_GET(self) -> None:  # noqa: N802
        self._touch_activity()
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            body = self.html_body
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        self._touch_activity()
        path = self.path.split("?")[0]
        if path == "/models":
            self._handle_models()
        elif path == "/save":
            self._handle_save()
        else:
            self._send_json({"error": "not found"}, status=404)

    def _handle_models(self) -> None:
        from djcues.providers import DEFAULT_MODEL, PRICING, RECOMMENDED_FOR_ACCURACY, get_provider

        body = self._read_body()
        provider_name = body.get("provider")
        api_key = body.get("api_key", "")
        if provider_name not in ("anthropic", "gemini") or not api_key:
            self._send_json({"error": "provider and api_key are required"}, status=400)
            return

        try:
            provider = get_provider(provider_name)
            models = provider.list_models(api_key)
        except Exception as e:  # noqa: BLE001 -- surfaced to the user, not the key
            self._send_json({"error": f"Could not fetch models: {e}"}, status=400)
            return
        if not models:
            self._send_json({"error": "No models returned -- check your key."}, status=400)
            return

        default_id = DEFAULT_MODEL.get(provider_name)
        accuracy_id = RECOMMENDED_FOR_ACCURACY.get(provider_name)
        # Accuracy pick sorts first when this provider has one measured,
        # then the cheap default, then alphabetical.
        models_sorted = sorted(
            models, key=lambda m: (m.id != accuracy_id, m.id != default_id, m.id)
        )
        payload = []
        for m in models_sorted:
            price = PRICING.get(m.id)
            payload.append({
                "id": m.id,
                "display_name": m.display_name,
                "context_window": m.context_window,
                "recommended": m.id == default_id,
                "recommended_accuracy": m.id == accuracy_id,
                "price_input_per_million": price.input_per_million if price else None,
                "price_output_per_million": price.output_per_million if price else None,
            })
        self._send_json({
            "models": payload,
            "default_model": default_id,
            "recommended_accuracy_model": accuracy_id,
        })

    def _handle_save(self) -> None:
        from djcues.auth import KeyringUnavailableError, load_config, save_config, set_api_key

        body = self._read_body()
        provider_name = body.get("provider")
        api_key = body.get("api_key", "")
        model = body.get("model")
        if provider_name not in ("anthropic", "gemini") or not api_key or not model:
            self._send_json({"error": "provider, api_key, and model are required"}, status=400)
            return

        try:
            set_api_key(provider_name, api_key)
        except KeyringUnavailableError as e:
            self._send_json({"error": str(e)}, status=500)
            return

        config = load_config()
        config["provider"] = provider_name
        config["model"] = model
        save_config(config)

        self.server._setup_complete = True  # type: ignore[attr-defined]
        self._send_json({"ok": True, "provider": provider_name, "model": model})


def start_auth_server(
    html_body: bytes,
    port: int = 0,
    timeout_minutes: int = 30,
) -> tuple[HTTPServer, int]:
    """Start the BYOK setup-wizard server in a daemon thread.

    Returns ``(server, actual_port)``. The server stops itself as soon as
    a save succeeds (``server._setup_complete`` flips true) or after
    *timeout_minutes* of inactivity if the user abandons the tab --
    30 minutes by default (matches the review server's own default),
    since going to copy a freshly-created API key from a provider's
    console before pasting it back in can easily take a while. On an
    idle timeout, ``server._timed_out`` is set so callers polling for
    completion can tell "the server gave up waiting" apart from still
    being in progress, instead of polling forever against a closed port.
    """
    handler = type("BoundAuthSetupHandler", (AuthSetupHandler,), {"html_body": html_body})

    server = AuthSetupServer(("127.0.0.1", port), handler)
    actual_port = server.server_address[1]
    server._last_activity = time.monotonic()  # type: ignore[attr-defined]
    server._shutdown_flag = False  # type: ignore[attr-defined]
    server._setup_complete = False  # type: ignore[attr-defined]
    server._timed_out = False  # type: ignore[attr-defined]

    timeout_seconds = timeout_minutes * 60

    def _serve() -> None:
        server.timeout = 10
        while not server._shutdown_flag and not server._setup_complete:  # type: ignore[attr-defined]
            server.handle_request()
            elapsed = time.monotonic() - server._last_activity  # type: ignore[attr-defined]
            if elapsed > timeout_seconds:
                server._timed_out = True  # type: ignore[attr-defined]
                break
        server.server_close()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    return server, actual_port
