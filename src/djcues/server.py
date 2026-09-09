"""Local HTTP server for djcues review sessions, the BYOK setup wizard,
and the analysis dashboard.

Bridges the browser review UI to the session JSON file, handling CORS,
session reads/writes, and cue adjustment logic including memory cue
recalculation.
"""

from __future__ import annotations

import json
import queue
import socketserver
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
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


class _LocalJsonHandler(BaseHTTPRequestHandler):
    """Shared boilerplate for djcues's local-only (127.0.0.1) HTTP
    handlers -- request-logging suppression, JSON response/body helpers,
    and idle-timeout activity tracking. Extracted once a third handler
    (DashboardHandler) needed the exact same four methods ReviewHandler/
    AuthSetupHandler had each already duplicated independently. Route
    logic (do_GET/do_POST/_handle_*) stays on each subclass -- that part
    is genuinely different per handler, not boilerplate.
    """

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Suppress default stderr logging."""

    def _touch_activity(self) -> None:
        """Update the server's last-activity timestamp, for idle-timeout tracking."""
        if hasattr(self.server, "_last_activity"):
            self.server._last_activity = time.monotonic()

    def _send_json(self, data: dict, status: int = 200, cors: bool = False) -> None:
        """Send a JSON response. cors=True adds the Access-Control-Allow-*
        headers a file://-opened page needs (review.py's page, opened
        from a written .html file -- a different origin from this
        server); same-origin server-served pages (auth-setup, dashboard)
        don't need them and don't send them."""
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if cors:
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


class ReviewHandler(_LocalJsonHandler):
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
        """Always CORS-enabled -- review.py's page is opened via a
        written file:// URL, a different origin from this server, unlike
        the same-origin auth-setup/dashboard pages."""
        super()._send_json(data, status=status, cors=True)

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


class AuthSetupHandler(_LocalJsonHandler):
    """Serves the setup-wizard page and its two endpoints.

    ``html_body`` is baked in as a class attribute (via a dynamic subclass,
    same pattern as ReviewHandler) — the page uses only same-origin
    relative fetch() calls, so unlike the review page it never needs a
    server URL embedded in it. The API key travels in POST bodies only,
    is never written to a response, a log line, or any file — only
    ``djcues.auth.set_api_key()`` (the OS credential store) ever sees it.
    No CORS needed (inherited default is cors=False) -- same-origin, like
    the dashboard, unlike review.py's file://-opened page.
    """

    html_body: bytes

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


class _DbWorker:
    """Owns the ONE Rekordbox6Database connection the dashboard's
    browsing routes are allowed to touch, on the ONE thread it's created
    and used from exclusively.

    Required, not optional -- confirmed by reading pyrekordbox's and
    SQLAlchemy's own source, not assumed: ``Rekordbox6Database`` wraps a
    single, unlocked SQLAlchemy ``Session`` (not safe to query from more
    than one thread at once), and it connects through the
    ``sqlite+pysqlcipher`` dialect, which hardcodes
    ``pool.SingletonThreadPool`` -- whose own docstring warns it silently
    ``.close()``s connections once more than ``pool_size`` (default 5)
    distinct *thread identities* have ever touched it, a lifetime count
    a per-request-thread server (``ReviewServer``'s ``ThreadingMixIn``)
    blows past almost immediately just from ordinary browsing. Routing
    every browsing query through one fixed thread sidesteps both hazards
    by construction, not by reasoning about lock coverage.

    Background analysis jobs deliberately do NOT use this worker -- they
    open their own short-lived dedicated connection instead (see
    ``_run_analysis_job``), so a multi-minute job never blocks browsing,
    and browsing never waits behind a job either.
    """

    def __init__(self) -> None:
        self._queue: "queue.Queue[tuple]" = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        from pyrekordbox import Rekordbox6Database

        db = Rekordbox6Database()
        while True:
            fn, box, done = self._queue.get()
            try:
                box["value"] = fn(db)
            except Exception as e:  # noqa: BLE001 -- re-raised on the caller's thread in run()
                box["error"] = e
            done.set()

    def run(self, fn, timeout: float = 30.0):
        """Run ``fn(db)`` on the worker thread; block the calling thread
        (an HTTP handler thread) until it completes or *timeout* elapses.
        Re-raises whatever exception ``fn`` raised, on the caller's own
        thread, so a handler's normal try/except handles it naturally."""
        box: dict = {}
        done = threading.Event()
        self._queue.put((fn, box, done))
        if not done.wait(timeout):
            raise TimeoutError("dashboard database worker did not respond in time")
        if "error" in box:
            raise box["error"]
        return box.get("value")


def _run_analysis_job(
    job_id: str, track_id: str, params: dict, jobs: dict, jobs_lock: threading.Lock
) -> None:
    """Runs one propose/compare/beatgrid analysis on its own thread, with
    its own short-lived, dedicated Rekordbox6Database connection --
    deliberately never the shared djcues.db.get_db() singleton, and
    deliberately never routed through _DbWorker either (see its
    docstring for why sharing either across threads is unsafe). The
    database is touched exactly once, right at the start (loading the
    Track) -- the real work after that (the heuristic/agentic proposer,
    --refine-drops's audio analysis including --deep's Demucs pass, the
    analysis cache) runs entirely against the in-memory Track object,
    so this thread never blocks browsing or another job for however
    long it runs.

    Reuses cli.py's exact tested composition chain
    (_get_proposer -> _apply_refine_drops -> analysis cache -> proposer())
    and its already-tested _print_* functions for output -- captured
    verbatim via redirect_stdout/redirect_stderr (both, not just stdout:
    an agentic auth failure's real explanation is click.echo'd with
    err=True inside _get_proposer's own closure, and would otherwise be
    lost) rather than building any new rendering logic. For propose/
    compare specifically, also renders a real waveform/timeline via
    viz._render_track_body -- the same function review.py already
    reuses for its own cards -- for free.
    """
    import contextlib
    import io
    import time as time_module

    from pyrekordbox import Rekordbox6Database

    from djcues import analysis_cache
    from djcues import db as db_module
    from djcues.beat_verify import verify_beat_grid
    from djcues.cli import (
        _apply_cache,
        _apply_refine_drops,
        _get_proposer,
        _print_beatgrid_report,
        _print_cache_summary,
        _print_comparison,
        _print_cost_summary,
        _print_proposal,
        _print_refinement_summary,
        _rehydrate_beatgrid,
        _shape_beatgrid_for_cache,
    )
    from djcues.providers import estimate_cost
    from djcues.viz import _render_track_body

    started = time_module.monotonic()
    dedicated_db = Rekordbox6Database()
    output = io.StringIO()
    html_fragment = None
    cache_stats = {"hits": 0, "misses": 0}
    cost_usd = None
    result: dict

    try:
        content = dedicated_db.get_content(ID=track_id)
        if content is None:
            raise ValueError(f"track {track_id!r} not found")
        track = db_module.load_track(content, db=dedicated_db)

        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            if params["kind"] in ("propose", "compare"):
                proposer, telemetry_list, resolved_model, resolved_provider = _get_proposer(
                    params["agentic"], params["provider"], params["model"],
                    params["offset_bars"], params["loop_bars"], params["skip_critic"],
                )
                proposer, refinement_log = _apply_refine_drops(
                    proposer, params["refine_drops"], params["deep"],
                    params["offset_bars"], params["loop_bars"],
                )
                cache_key = analysis_cache.cue_proposal_key(
                    agentic=params["agentic"], provider=resolved_provider, model=resolved_model,
                    skip_critic=params["skip_critic"], refine_drops=params["refine_drops"],
                    deep=params["deep"], offset_bars=params["offset_bars"],
                    loop_bars=params["loop_bars"],
                )
                proposer, cache_stats = _apply_cache(
                    proposer, cache_key, params["offset_bars"], params["loop_bars"],
                    params["no_cache"], refinement_log=refinement_log,
                    telemetry_list=telemetry_list, resolved_model=resolved_model,
                )
                proposal = proposer(track)

                if params["kind"] == "propose":
                    _print_proposal(proposal, track)
                else:
                    _print_comparison(proposal, track)
                if telemetry_list is not None:
                    _print_cost_summary(telemetry_list, resolved_model, 1)
                    if telemetry_list and (telemetry_list[-1].input_tokens or telemetry_list[-1].output_tokens):
                        cost_usd = estimate_cost(
                            resolved_model, telemetry_list[-1].input_tokens,
                            telemetry_list[-1].output_tokens,
                        )
                if refinement_log is not None:
                    _print_refinement_summary(refinement_log)
                _print_cache_summary(cache_stats)

                html_fragment = _render_track_body(
                    track, proposal, compare=(params["kind"] == "compare")
                )

            else:  # beatgrid
                entries = db_module.extract_raw_beat_grid(content, db=dedicated_db)
                fp = analysis_cache.fingerprint_beat_grid(entries)
                cache_key = analysis_cache.beatgrid_key(
                    deep=params["deep"], tolerance_ms=params["tolerance_ms"]
                )

                report = None
                if not params["no_cache"]:
                    cached = analysis_cache.get_cached(track.id, cache_key, fp)
                    if cached is not None:
                        cache_stats["hits"] += 1
                        report = _rehydrate_beatgrid(cached.result, track)

                if report is None:
                    cache_stats["misses"] += 1
                    report = verify_beat_grid(
                        track, entries, force_deep=params["deep"],
                        audio_tolerance_ms=params["tolerance_ms"],
                    )
                    if not params["no_cache"]:
                        analysis_cache.store_result(
                            track.id, cache_key, fp, _shape_beatgrid_for_cache(report),
                            track_title=track.title, track_artist=track.artist,
                        )

                _print_beatgrid_report(report)
                _print_cache_summary(cache_stats)

        result = {
            "status": "done", "output_text": output.getvalue(), "html_fragment": html_fragment,
            "cache": cache_stats, "cost_usd": cost_usd, "error": None,
        }
    except SystemExit:
        # _get_proposer's agentic closure raises this on an unrecoverable
        # auth/permission failure -- the real explanation was already
        # click.echo(err=True)'d inside it, captured above into `output`
        # since both streams are redirected there.
        result = {
            "status": "error", "output_text": output.getvalue(), "html_fragment": None,
            "cache": cache_stats, "cost_usd": cost_usd,
            "error": "Request aborted -- see output for details (likely an auth/permission issue).",
        }
    except Exception as e:  # noqa: BLE001 -- surfaced to the dashboard UI, not swallowed
        result = {
            "status": "error", "output_text": output.getvalue(), "html_fragment": None,
            "cache": cache_stats, "cost_usd": cost_usd, "error": str(e),
        }
    finally:
        dedicated_db.close()

    result["finished_at"] = datetime.now().replace(microsecond=0).isoformat()
    result["elapsed_seconds"] = round(time_module.monotonic() - started, 1)
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(result)


class DashboardHandler(_LocalJsonHandler):
    """Serves the analysis dashboard: browse rekordbox playlists/tracks
    and run propose/compare/beatgrid against a selected track, or launch
    the existing viz/review commands for it.

    ``html_body`` is baked in as a class attribute (dynamic subclass,
    same pattern as AuthSetupHandler) -- same-origin page, no CORS needed
    (inherited default cors=False). ``_db_worker`` (one ``_DbWorker``
    per server instance) is likewise baked in by ``start_dashboard_server``
    and is the ONLY thing browsing routes are allowed to use to touch
    ``djcues.db`` -- see ``_DbWorker``'s docstring for why. Background
    analysis jobs run on their own threads with their own dedicated DB
    connection instead (``_run_analysis_job``), tracked in ``_jobs``
    (guarded by ``_jobs_lock`` -- same lock-per-shared-mutable-dict idiom
    ``ReviewHandler`` already uses for its own session lock).
    """

    html_body: bytes
    _db_worker: "_DbWorker"
    _jobs: dict = {}
    _jobs_lock = threading.Lock()

    # --- GET --------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        self._touch_activity()
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._serve_html()
        elif path == "/api/playlists":
            self._handle_playlists_get()
        else:
            parts = [p for p in path.split("/") if p]
            if len(parts) == 4 and parts[0:2] == ["api", "playlists"] and parts[3] == "tracks":
                self._handle_playlist_tracks_get(parts[2])
            elif len(parts) == 3 and parts[0:2] == ["api", "tracks"]:
                self._handle_track_detail_get(parts[2])
            elif len(parts) == 3 and parts[0:2] == ["api", "jobs"]:
                self._handle_job_get(parts[2])
            else:
                self._send_json({"error": "not found"}, status=404)

    def _serve_html(self) -> None:
        body = self.html_body
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_playlists_get(self) -> None:
        from djcues.db import build_playlist_tree

        def _node_to_dict(node) -> dict:
            return {
                "id": node.id, "name": node.name, "kind": node.kind,
                "track_count": node.track_count,
                "children": [_node_to_dict(c) for c in node.children],
            }

        try:
            tree = self._db_worker.run(lambda db: build_playlist_tree(db=db))
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)
            return
        self._send_json({"tree": [_node_to_dict(n) for n in tree]})

    def _handle_playlist_tracks_get(self, playlist_id: str) -> None:
        from djcues.db import list_playlist_tracks

        try:
            tracks = self._db_worker.run(lambda db: list_playlist_tracks(playlist_id, db=db))
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)
            return
        self._send_json({
            "playlist_id": playlist_id,
            "tracks": [
                {
                    "id": t.id, "track_no": t.track_no, "title": t.title,
                    "artist": t.artist, "bpm": t.bpm, "duration_ms": t.duration_ms,
                }
                for t in tracks
            ],
        })

    def _handle_track_detail_get(self, track_id: str) -> None:
        from djcues import db as db_module
        from djcues.review import _resolve_local_audio_path

        def work(db):
            content = db.get_content(ID=track_id)
            if content is None:
                return None
            return db_module.load_track(content, db=db)

        try:
            track = self._db_worker.run(work)
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)
            return
        if track is None:
            self._send_json({"error": "track not found"}, status=404)
            return

        self._send_json({
            "id": track.id, "title": track.title, "artist": track.artist,
            "bpm": track.bpm, "duration_ms": track.duration_ms,
            "phrase_count": len(track.phrases), "has_phrases": bool(track.phrases),
            "existing_cue_count": len(track.cues),
            "has_local_audio": _resolve_local_audio_path(track) is not None,
        })

    def _handle_job_get(self, job_id: str) -> None:
        with self._jobs_lock:
            job = self._jobs.get(job_id)
            job_copy = dict(job) if job is not None else None
        if job_copy is None:
            self._send_json({"error": "job not found"}, status=404)
            return
        self._send_json(job_copy)

    # --- POST -------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802
        self._touch_activity()
        path = self.path.split("?")[0]
        parts = [p for p in path.split("/") if p]
        if len(parts) == 4 and parts[0:2] == ["api", "tracks"] and parts[3] == "jobs":
            self._handle_job_post(parts[2])
        elif len(parts) == 5 and parts[0:2] == ["api", "tracks"] and parts[3] == "launch":
            self._handle_launch_post(parts[2], parts[4])
        else:
            self._send_json({"error": "not found"}, status=404)

    def _handle_job_post(self, track_id: str) -> None:
        from djcues.cli import _check_refine_drops_available

        body = self._read_body()
        kind = body.get("kind")
        if kind not in ("propose", "compare", "beatgrid"):
            self._send_json({"error": "kind must be propose, compare, or beatgrid"}, status=400)
            return

        deep = bool(body.get("deep", False))
        refine_drops = bool(body.get("refine_drops", False))
        if kind != "beatgrid":
            if deep and not refine_drops:
                self._send_json({"error": "deep only applies with refine_drops"}, status=400)
                return
            if refine_drops:
                err = _check_refine_drops_available(deep)
                if err:
                    self._send_json({"error": err}, status=400)
                    return

        params = {
            "kind": kind,
            "agentic": bool(body.get("agentic", False)),
            "provider": body.get("provider") or None,
            "model": body.get("model") or None,
            "skip_critic": bool(body.get("skip_critic", False)),
            "refine_drops": refine_drops,
            "deep": deep,
            "offset_bars": int(body.get("offset_bars", 16)),
            "loop_bars": int(body.get("loop_bars", 4)),
            "no_cache": bool(body.get("no_cache", False)),
            "tolerance_ms": float(body.get("tolerance_ms", 30.0)),
        }

        job_id = uuid.uuid4().hex
        job = {
            "job_id": job_id, "track_id": track_id, "kind": kind, "status": "running",
            "started_at": datetime.now().replace(microsecond=0).isoformat(), "finished_at": None,
            "elapsed_seconds": None, "output_text": "", "html_fragment": None,
            "cache": {"hits": 0, "misses": 0}, "cost_usd": None, "error": None,
        }
        with self._jobs_lock:
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=_run_analysis_job,
            args=(job_id, track_id, params, self._jobs, self._jobs_lock),
            daemon=True,
        )
        thread.start()
        self._send_json({"job_id": job_id, "status": "running"}, status=202)

    def _handle_launch_post(self, track_id: str, tool: str) -> None:
        if tool not in ("viz", "review"):
            self._send_json({"error": "not found"}, status=404)
            return

        body = self._read_body()
        playlist_id = body.get("playlist_id")
        if not playlist_id:
            self._send_json({"error": "playlist_id is required"}, status=400)
            return

        def work(db):
            playlist = db.get_playlist(ID=playlist_id)
            content = db.get_content(ID=track_id)
            return (
                playlist.Name if playlist is not None else None,
                content.Title if content is not None else None,
            )

        try:
            playlist_name, track_title = self._db_worker.run(work)
        except Exception as e:
            self._send_json({"error": str(e)}, status=500)
            return
        if playlist_name is None or track_title is None:
            self._send_json({"error": "playlist or track not found"}, status=404)
            return

        # sys.executable -m djcues.cli, never `uv run`/a bare `djcues` PATH
        # lookup -- this repo's own documented environment gotchas record
        # `uv run` corrupting this exact venv's editable install before;
        # sys.executable is the exact interpreter already running this
        # server, sidestepping that entirely.
        argv = [sys.executable, "-m", "djcues.cli", tool, playlist_name, track_title]
        if tool == "review":
            # Deliberately separate, default-off flags -- never inherited
            # from whatever the propose/compare panel currently has
            # toggled, so opening Review can't silently piggyback an
            # unwanted --agentic/--deep charge.
            if body.get("agentic"):
                argv.append("--agentic")
            if body.get("refine_drops"):
                argv.append("--refine-drops")
            if body.get("deep"):
                argv.append("--deep")

        popen_kwargs: dict = {}
        if sys.platform == "win32":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        subprocess.Popen(argv, **popen_kwargs)  # fire-and-forget, never .wait()ed on

        self._send_json({"ok": True, "playlist_name": playlist_name, "track_title": track_title})


def start_dashboard_server(
    html_body: bytes,
    port: int = 0,
    timeout_minutes: int = 30,
) -> tuple[HTTPServer, int]:
    """Start the analysis dashboard server in a daemon thread.

    Returns ``(server, actual_port)``. Reuses ``ReviewServer`` as-is for
    its ``ThreadingMixIn``/larger backlog (nothing in that class is
    review-specific despite the name) -- browsing requests benefit from
    concurrent handling too (e.g. a track-detail lookup shouldn't block
    behind an in-flight playlist-tree request), and every request
    handler thread only ever touches ``djcues.db`` via the one shared
    ``_DbWorker``, per its own docstring, never the request thread
    itself. A long-running analysis job naturally keeps this server
    alive past its idle timeout too, since the dashboard page polls
    ``GET /api/jobs/<id>`` every ~2s while one is in flight, and every
    request (including polls) touches activity via ``_touch_activity``.
    """
    db_worker = _DbWorker()
    handler = type(
        "BoundDashboardHandler",
        (DashboardHandler,),
        {
            "html_body": html_body,
            "_db_worker": db_worker,
            "_jobs": {},
            "_jobs_lock": threading.Lock(),
        },
    )

    server = ReviewServer(("127.0.0.1", port), handler)
    actual_port = server.server_address[1]
    server._last_activity = time.monotonic()  # type: ignore[attr-defined]
    server._shutdown_flag = False  # type: ignore[attr-defined]

    timeout_seconds = timeout_minutes * 60

    def _serve() -> None:
        server.timeout = 10
        while not server._shutdown_flag:  # type: ignore[attr-defined]
            server.handle_request()
            elapsed = time.monotonic() - server._last_activity  # type: ignore[attr-defined]
            if elapsed > timeout_seconds:
                break
        server.server_close()

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()

    return server, actual_port
