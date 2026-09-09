"""Review HTML generation — interactive review page for cue proposals."""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

from djcues.constants import CUE_SYSTEM_BY_PAD, KIND_TO_PAD
from djcues.models import CueProposal, Track
from djcues.viz import _PAGE_CSS, _render_track_body


def _resolve_local_audio_path(track: Track) -> Path | None:
    """Numpy-free equivalent of audio.py's resolve_audio_path().

    Not importing djcues.audio here: that module imports numpy at the
    top level (needed for LoadedAudio/load_audio), which is only part
    of the optional audio/ml extras -- confirmed live that
    `import djcues.audio` raises ModuleNotFoundError without numpy
    installed. review.py (and cli.py's `review` command) have to stay
    usable with only djcues's core dependencies (click, pyrekordbox) for
    every `djcues review` invocation, not just --refine-drops ones.

    A Track's audio_path isn't guaranteed to be a real filesystem path --
    Rekordbox stores a spotify:track:... URI in the same field for
    Spotify-streaming-linked tracks -- so this always checks is_file()
    rather than trusting a non-empty string.
    """
    if not track.audio_path:
        return None
    path = Path(track.audio_path)
    return path if path.is_file() else None


def create_session(
    playlist_name: str,
    playlist_id: int,
    tracks_and_proposals: list[tuple[Track, CueProposal]],
    memory_offset_bars: int = 16,
    loop_length_bars: int = 4,
) -> dict:
    """Create a session dict from tracks and their cue proposals.

    The returned dict is JSON-serializable and follows the session schema
    used for persisting review state.
    """
    tracks_dict: dict[str, dict] = {}

    for track, proposal in tracks_and_proposals:
        has_existing_cues = any(c.kind > 0 for c in track.cues)

        # Hot cues keyed by pad letter
        cues_dict: dict[str, dict] = {}
        for hc in proposal.hot_cues:
            pad = KIND_TO_PAD.get(hc.kind)
            if pad is None:
                continue
            cues_dict[pad] = {
                "position_ms": hc.position_ms,
                "loop_end_ms": hc.loop_end_ms,
                "status": "pending",
                "confidence": proposal.confidence.get(pad, 0.0),
            }

        # Memory cues keyed by 1-indexed string
        memory_cues_dict: dict[str, dict] = {}
        for i, mc in enumerate(proposal.memory_cues):
            memory_cues_dict[str(i + 1)] = {
                "position_ms": mc.position_ms,
                "loop_end_ms": mc.loop_end_ms,
                "status": "pending",
            }

        tracks_dict[str(track.id)] = {
            "title": track.title,
            "artist": track.artist,
            "bpm": track.bpm,
            "duration_ms": track.duration_ms,
            "phrase_count": len(track.phrases),
            "has_vocal_data": track.vocal_track is not None,
            "has_waveform_data": track.waveform is not None,
            "first_beat_ms": track.beat_grid.first_beat_ms,
            "status": "pending",
            "has_existing_cues": has_existing_cues,
            "cues": cues_dict,
            "memory_cues": memory_cues_dict,
        }

    return {
        "playlist": playlist_name,
        "playlist_id": playlist_id,
        "created": datetime.now().replace(microsecond=0).isoformat(),
        "settings": {
            "memory_offset_bars": memory_offset_bars,
            "loop_length_bars": loop_length_bars,
        },
        "tracks": tracks_dict,
    }


def render_review_html(
    playlist_name: str,
    tracks_and_proposals: list[tuple[Track, CueProposal]],
    session_path: str,
    server_url: str,
) -> str:
    """Generate an interactive HTML review page for cue proposals.

    The page extends the viz module's styling with review-specific controls
    for accepting, skipping, and adjusting individual cues. All actions
    communicate with the review server via fetch() POST requests.
    """
    escaped_name = html.escape(playlist_name)
    count = len(tracks_and_proposals)
    escaped_session_path = html.escape(session_path)

    # Build track cards with data attributes for JS interaction
    track_cards = []
    for track, proposal in tracks_and_proposals:
        has_existing = any(c.kind > 0 for c in track.cues)
        body = _render_track_body(track, proposal, compare=has_existing)

        # Build cue positions JSON for the data-cues attribute
        cues_data: dict[str, dict] = {}
        for hc in proposal.hot_cues:
            pad = KIND_TO_PAD.get(hc.kind)
            if pad is None:
                continue
            cues_data[pad] = {
                "position_ms": hc.position_ms,
                "loop_end_ms": hc.loop_end_ms,
                "confidence": proposal.confidence.get(pad, 0.0),
            }

        cues_json = html.escape(json.dumps(cues_data), quote=True)

        resolved_audio = _resolve_local_audio_path(track)
        has_audio = resolved_audio is not None
        audio_format = resolved_audio.suffix.lower().lstrip(".") if resolved_audio else ""

        warning_badge = ""
        if has_existing:
            warning_badge = (
                '<span class="overwrite-badge" '
                'title="This track has existing cues that will be overwritten">'
                "OVERWRITE</span>"
            )

        track_cards.append(
            f'<div class="track-card review-card" '
            f'data-track-id="{track.id}" '
            f'data-bpm="{track.bpm}" '
            f'data-first-beat-ms="{track.beat_grid.first_beat_ms}" '
            f'data-duration-ms="{track.duration_ms}" '
            f'data-has-audio="{str(has_audio).lower()}" '
            f'data-audio-format="{audio_format}" '
            f"data-cues='{cues_json}'>"
            f'<div class="review-controls">'
            f'<div class="review-controls-left">'
            f'<span class="status-indicator" data-status="pending">pending</span>'
            f"{warning_badge}"
            f"</div>"
            f'<div class="review-controls-right">'
            f'<button class="btn btn-accept" onclick="acceptTrack(\'{track.id}\')">Accept</button>'
            f'<button class="btn btn-skip" onclick="skipTrack(\'{track.id}\')">Skip</button>'
            f"</div>"
            f"</div>"
            f"{body}"
            f"</div>"
        )

    cards_html = "\n".join(track_cards)

    review_css = _REVIEW_CSS
    review_js = _render_review_js(server_url)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>djcues review &mdash; {escaped_name}</title>
<style>{_PAGE_CSS}</style>
<style>{review_css}</style>
</head>
<body>
<div class="sticky-header">
  <div class="sticky-header-content">
    <div class="sticky-header-left">
      <h1>{escaped_name}</h1>
      <p class="meta">{count} tracks &middot; Session: {escaped_session_path}</p>
    </div>
    <div class="sticky-header-center">
      <div class="summary-bar">
        <span class="summary-item" id="count-pending">0 pending</span>
        <span class="summary-item" id="count-accepted">0 accepted</span>
        <span class="summary-item" id="count-skipped">0 skipped</span>
        <span class="summary-item" id="count-adjusted">0 adjusted</span>
        <span class="summary-item" id="count-error">0 error</span>
      </div>
      <div class="transport-controls">
        <button class="btn btn-icon" id="transport-skip-back" onclick="transportSkip(-1)" title="Skip back 5s">&#9664;&#9664;</button>
        <button class="btn btn-icon" id="transport-play-pause" onclick="transportTogglePlayPause()" title="Play/Pause">&#9654;</button>
        <button class="btn btn-icon" id="transport-skip-forward" onclick="transportSkip(1)" title="Skip forward 5s">&#9654;&#9654;</button>
      </div>
      <span id="now-playing" class="now-playing hidden"></span>
    </div>
    <div class="sticky-header-right">
      <button class="btn btn-icon" id="shortcuts-toggle" onclick="toggleShortcuts()" title="Keyboard shortcuts">&#9000;</button>
      <button class="btn btn-icon" id="settings-toggle" onclick="toggleSettings()" title="Playback preview settings">&#9881;</button>
      <button class="btn" onclick="jumpToNextPending()">Next Pending</button>
      <button class="btn btn-accept-all" onclick="acceptAll()">Accept All</button>
      <div class="apply-command" title="Click to copy">
        <code id="apply-command-text">uv run djcues apply {escaped_session_path}</code>
        <button class="btn btn-copy" onclick="copyApplyCommand()">Copy</button>
      </div>
    </div>
  </div>
  <div id="settings-panel" class="dropdown-panel hidden">
    <label>Preview pre-roll (bars before the cue)
      <input type="number" id="setting-pre-roll" min="0" step="1">
    </label>
    <label>Loop length (bars)
      <input type="number" id="setting-loop-bars" min="1" step="1">
    </label>
    <label class="checkbox-label">
      <input type="checkbox" id="setting-loop-enabled"> Loop preview by default
    </label>
  </div>
  <div id="shortcuts-panel" class="dropdown-panel hidden">
    <div><kbd>&larr;</kbd> <kbd>&rarr;</kbd> nudge the selected cue by 1 bar</div>
    <div><kbd>Tab</kbd> / <kbd>Shift</kbd>+<kbd>Tab</kbd> select next/previous cue</div>
    <div><kbd>Space</kbd> play/pause preview of the selected cue</div>
    <div><kbd>Delete</kbd> skip the selected cue</div>
    <div><kbd>Esc</kbd> deselect</div>
    <div>Click a waveform to preview from that point</div>
  </div>
</div>
<div class="review-body">
{cards_html}
</div>
<div id="cue-editor-popover" class="cue-editor-popover hidden">
  <div class="cue-editor-handle" id="cue-editor-handle">Edit Cue</div>
  <div class="cue-editor-row">
    <label>Bar
      <input type="number" id="editor-bar" min="1" step="1" inputmode="numeric">
    </label>
    <label>Time
      <input type="text" id="editor-time" placeholder="M:SS.s" inputmode="decimal"
             pattern="^\\d+:\\d+(\\.\\d+)?$">
    </label>
  </div>
  <p class="cue-editor-hint">Bar jumps to that bar's exact downbeat. Time is exact, to the tenth of a second. Press Enter to apply.</p>
  <div class="cue-editor-row">
    <button class="btn btn-preview" id="editor-preview-btn" onclick="toggleSelectedCuePreview()">&#9654; Preview</button>
    <label class="checkbox-label cue-editor-loop-toggle">
      <input type="checkbox" id="editor-loop-toggle"> Loop
    </label>
    <label class="cue-editor-loop-bars">
      <input type="number" id="editor-loop-bars" min="1" step="1"> bars
    </label>
  </div>
</div>
<audio id="preview-audio" preload="none"></audio>
<script>
{review_js}
</script>
</body>
</html>"""


_REVIEW_CSS = """
  /* Utility: used across several independently-styled components
     (dropdown panels, the cue editor popover, the now-playing badge) to
     toggle visibility via JS. Needs !important -- otherwise a later,
     equal-specificity single-class rule (e.g. .dropdown-panel { display:
     flex }) wins the cascade by source order and this has no effect. */
  .hidden {
    display: none !important;
  }

  /* Sticky header */
  .sticky-header {
    position: sticky;
    top: 0;
    z-index: 100;
    background: #12122a;
    border-bottom: 1px solid #2a2a3e;
    padding: 12px 24px;
    margin: -24px -24px 24px -24px;
  }
  .sticky-header-content {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
  }
  .sticky-header-left h1 { font-size: 1.3rem; margin-bottom: 0; }
  .sticky-header-left .meta { font-size: 0.8rem; color: #888; }
  .sticky-header-center { flex: 1; text-align: center; }
  .sticky-header-right { display: flex; align-items: center; gap: 12px; }

  /* Summary bar */
  .summary-bar {
    display: inline-flex;
    gap: 16px;
    font-size: 0.85rem;
  }
  .summary-item { color: #888; }
  #count-pending { color: #ffc107; }
  #count-accepted { color: #28a745; }
  #count-skipped { color: #dc3545; }
  #count-adjusted { color: #17a2b8; }
  #count-error { color: #ff4136; }

  /* Apply command */
  .apply-command {
    display: flex;
    align-items: center;
    gap: 6px;
    background: #1e1e3a;
    border: 1px solid #2a2a3e;
    border-radius: 4px;
    padding: 4px 8px;
    font-size: 0.8rem;
    cursor: pointer;
  }
  .apply-command code {
    color: #aaa;
    font-family: monospace;
    font-size: 0.8rem;
  }

  /* Buttons */
  .btn {
    border: none;
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 0.85rem;
    cursor: pointer;
    font-weight: 600;
    transition: opacity 0.15s;
  }
  .btn:hover { opacity: 0.85; }
  .btn-accept { background: #28a745; color: #fff; }
  .btn-skip { background: #dc3545; color: #fff; }
  .btn-accept-all { background: #28a745; color: #fff; }
  .btn-copy { background: #444; color: #ddd; padding: 4px 8px; font-size: 0.75rem; }

  /* Review card controls */
  .review-card { position: relative; }
  .review-controls {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
    padding-bottom: 12px;
    border-bottom: 1px solid #2a2a3e;
  }
  .review-controls-left { display: flex; align-items: center; gap: 10px; }
  .review-controls-right { display: flex; gap: 8px; }

  /* Status indicator */
  .status-indicator {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  .status-indicator[data-status="pending"] { background: #ffc10733; color: #ffc107; }
  .status-indicator[data-status="accepted"] { background: #28a74533; color: #28a745; }
  .status-indicator[data-status="skipped"] { background: #dc354533; color: #dc3545; }
  .status-indicator[data-status="adjusted"] { background: #17a2b833; color: #17a2b8; }
  .status-indicator[data-status="error"] { background: #ff413633; color: #ff4136; }

  /* Overwrite warning badge */
  .overwrite-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 700;
    background: #e67e2233;
    color: #e67e22;
    letter-spacing: 0.5px;
  }

  /* Review body (push below sticky header) */
  .review-body { padding-top: 8px; }

  /* Proposed hot-cue markers are selectable/editable (existing ones are
     read-only), so they get a much wider invisible click target than
     their 2px visual line -- a real click reliably misses a 2px-wide
     target and falls through to the timeline's click-to-scrub handler
     instead of selecting the cue (reported live: "lost the ability to
     select the cue"). The visual line is redrawn via ::before, centered
     in the wider box; .existing markers are deliberately excluded so
     the Existing Cues timeline (still fully click-to-scrub-able) doesn't
     gain a larger dead zone for no benefit. */
  .hot-cue-marker.proposed {
    width: 14px;
    margin-left: -7px;
    border-left: none;
    cursor: pointer;
  }
  .hot-cue-marker.proposed::before {
    content: "";
    position: absolute;
    left: 6px;
    top: 0;
    height: 100%;
    border-left: 2px solid currentColor;
  }
  /* The letter itself is a small clickable chip, not just floating text --
     a clear "click here" affordance in addition to the wider invisible
     hit-box above, and a bigger target in its own right. */
  .hot-cue-marker.proposed .marker-label {
    padding: 1px 5px;
    border: 1.5px solid currentColor;
    border-radius: 3px;
    background: #111122;
  }

  /* Selected cue marker */
  .hot-cue-marker.selected::before {
    border-left-width: 5px;
    left: 4.5px;
  }
  .hot-cue-marker.selected {
    filter: brightness(1.4) drop-shadow(0 0 6px currentColor);
    z-index: 15;
  }
  .hot-cue-marker.selected .marker-label {
    /* Filled, glowing pill instead of plain text -- the border-width/
       brightness change alone was too subtle to register at a glance
       (real feedback: selection wasn't prominent enough). Dark text on
       a currentColor background reads reliably across every CUE_COLORS
       swatch, unlike white text, which would wash out on the lighter
       ones (e.g. pad C's yellow). */
    background: currentColor;
    color: #0d0d1a;
    padding: 1px 6px;
    border-radius: 3px;
    box-shadow: 0 0 8px currentColor;
    font-weight: 800;
  }

  /* Existing cue markers are non-interactive */
  .hot-cue-marker.existing,
  .mem-cue-marker.existing {
    cursor: default;
    opacity: 0.6;
  }

  /* Low-confidence proposed markers are visually de-emphasized, so the
     eye is drawn to the ones most worth double-checking without having
     to cross-reference the separate confidence-bars section below. */
  .hot-cue-marker.low-confidence {
    opacity: 0.55;
  }
  .hot-cue-marker.low-confidence::before {
    border-left-style: dotted;
  }

  /* Header icon buttons (settings / shortcuts) */
  .btn-icon {
    background: #1e1e3a;
    color: #ccc;
    padding: 5px 10px;
    font-size: 1rem;
    line-height: 1;
  }
  .btn-icon.active { background: #17a2b8; color: #fff; }

  /* Global transport controls -- lives in the sticky header, so it's
     visible at the top regardless of scroll position for free (no
     separate position:fixed treatment needed). */
  .transport-controls {
    display: inline-flex;
    gap: 4px;
    vertical-align: middle;
  }
  .transport-controls .btn-icon {
    padding: 5px 9px;
    font-size: 0.8rem;
  }
  #transport-play-pause.playing { background: #17a2b8; color: #fff; }

  /* Now-playing readout */
  .now-playing {
    display: block;
    margin-top: 4px;
    font-size: 0.78rem;
    color: #17a2b8;
  }

  /* Settings / shortcuts dropdown panels */
  .dropdown-panel {
    margin-top: 10px;
    padding: 12px 16px;
    background: #1e1e3a;
    border: 1px solid #2a2a3e;
    border-radius: 6px;
    font-size: 0.85rem;
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    align-items: center;
  }
  .dropdown-panel label {
    display: flex;
    align-items: center;
    gap: 6px;
    color: #ccc;
  }
  .dropdown-panel input[type="number"] {
    width: 60px;
    background: #12122a;
    border: 1px solid #2a2a3e;
    color: #eee;
    border-radius: 4px;
    padding: 3px 6px;
    font-size: 0.85rem;
  }
  .checkbox-label { cursor: pointer; }
  .shortcuts-panel div { color: #aaa; }
  kbd {
    display: inline-block;
    background: #2a2a3e;
    border: 1px solid #3a3a52;
    border-radius: 3px;
    padding: 1px 6px;
    font-family: monospace;
    font-size: 0.8rem;
  }

  /* Numeric cue-position editor popover */
  .cue-editor-popover {
    position: absolute;
    z-index: 200;
    background: #1e1e3a;
    border: 1px solid #17a2b8;
    border-radius: 6px;
    padding: 10px 14px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .cue-editor-handle {
    cursor: move;
    font-size: 0.75rem;
    font-weight: 700;
    color: #888;
    padding-bottom: 6px;
    margin-bottom: 2px;
    border-bottom: 1px solid #2a2a3e;
    user-select: none;
  }
  .cue-editor-row {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .cue-editor-row label {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 0.8rem;
    color: #ccc;
    white-space: nowrap;
  }
  .cue-editor-row input[type="number"] {
    width: 55px;
    background: #12122a;
    border: 1px solid #2a2a3e;
    color: #eee;
    border-radius: 4px;
    padding: 3px 6px;
  }
  .cue-editor-row input[type="text"] {
    width: 70px;
    background: #12122a;
    border: 1px solid #2a2a3e;
    color: #eee;
    border-radius: 4px;
    padding: 3px 6px;
  }
  .cue-editor-row input.invalid {
    border-color: #dc3545;
    background: #dc35451a;
  }
  .cue-editor-hint {
    font-size: 0.72rem;
    color: #888;
    max-width: 260px;
    line-height: 1.4;
  }
  .btn-preview { background: #17a2b8; color: #fff; padding: 5px 12px; }
  .btn-preview:disabled { background: #444; color: #888; cursor: not-allowed; opacity: 1; }

  /* Zoom controls, one row per timeline section */
  .zoom-controls {
    display: flex;
    gap: 4px;
    margin-bottom: 4px;
  }
  .btn-zoom {
    background: #1e1e3a;
    color: #aaa;
    padding: 2px 8px;
    font-size: 0.72rem;
  }
  .btn-zoom.active { background: #17a2b8; color: #fff; }

  /* Playhead -- inserted into whichever .timeline-container is currently
     previewing, moved between tracks/sections by JS rather than emitted
     once per track. */
  .playhead {
    position: absolute;
    top: 0;
    width: 2px;
    height: 100%;
    background: #ff9800;
    z-index: 20;
    pointer-events: none;
  }
"""


def _render_review_js(server_url: str) -> str:
    """Render the review page JavaScript with the given server URL."""
    escaped_url = server_url.replace("\\", "\\\\").replace("'", "\\'")
    return f"""
'use strict';

const SERVER = '{escaped_url}';
let selectedMarker = null;
let selectedTrackCard = null;

let previewSettings = {{ preview_pre_roll_bars: 4, preview_loop_bars: 8, preview_loop_enabled: true }};
let currentPreviewTrackId = null;
let playheadRAF = null;
const previewAudio = document.getElementById('preview-audio');

// --- Status updates ---

function updateSummaryCounts() {{
  const cards = document.querySelectorAll('.review-card');
  const counts = {{ pending: 0, accepted: 0, skipped: 0, adjusted: 0, error: 0 }};
  cards.forEach(card => {{
    const indicator = card.querySelector('.status-indicator');
    const status = indicator.getAttribute('data-status');
    if (counts.hasOwnProperty(status)) counts[status]++;
  }});
  document.getElementById('count-pending').textContent = counts.pending + ' pending';
  document.getElementById('count-accepted').textContent = counts.accepted + ' accepted';
  document.getElementById('count-skipped').textContent = counts.skipped + ' skipped';
  document.getElementById('count-adjusted').textContent = counts.adjusted + ' adjusted';
  const errEl = document.getElementById('count-error');
  if (errEl) errEl.textContent = counts.error + ' error';
}}

function setTrackStatus(trackId, status) {{
  const card = document.querySelector('[data-track-id="' + trackId + '"]');
  if (!card) return;
  const indicator = card.querySelector('.status-indicator');
  indicator.setAttribute('data-status', status);
  indicator.textContent = status;
  updateSummaryCounts();
}}

// --- Track actions ---

function postTrackStatus(trackId, status) {{
  return fetch(SERVER + '/session/track/' + trackId + '/status', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ status: status }})
  }}).then(res => {{
    if (!res.ok) throw new Error('server returned ' + res.status);
    setTrackStatus(trackId, status);
  }}).catch(err => {{
    console.error('Failed to set track ' + trackId + ' to ' + status + ':', err);
    setTrackStatus(trackId, 'error');
  }});
}}

function acceptTrack(trackId) {{
  return postTrackStatus(trackId, 'accepted');
}}

function skipTrack(trackId) {{
  return postTrackStatus(trackId, 'skipped');
}}

async function acceptAll() {{
  const btn = document.querySelector('.btn-accept-all');
  const cards = Array.from(document.querySelectorAll('.review-card'));
  if (btn) {{ btn.disabled = true; btn.textContent = 'Accepting…'; }}
  for (const card of cards) {{
    const trackId = card.getAttribute('data-track-id');
    const indicator = card.querySelector('.status-indicator');
    if (indicator.getAttribute('data-status') === 'pending') {{
      await acceptTrack(trackId);
    }}
  }}
  if (btn) {{ btn.disabled = false; btn.textContent = 'Accept All'; }}
}}

// --- Copy apply command ---

function copyApplyCommand() {{
  const text = document.getElementById('apply-command-text').textContent;
  navigator.clipboard.writeText(text).then(() => {{
    const btn = document.querySelector('.btn-copy');
    const orig = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => {{ btn.textContent = orig; }}, 1500);
  }});
}}

// --- Per-cue selection and adjustment ---

function deselectMarker() {{
  if (selectedMarker) {{
    selectedMarker.classList.remove('selected');
    selectedMarker = null;
    selectedTrackCard = null;
  }}
  hideEditor();
}}

// Shared lookup: everything the keyboard handler, the numeric editor,
// and cue-preview all need to know about the currently-selected marker.
// Returns null if nothing usable is selected, so every caller can just
// bail out the same way instead of re-deriving this by hand.
function getSelectedCueInfo() {{
  if (!selectedMarker || !selectedTrackCard) return null;
  const card = selectedTrackCard;
  const trackId = card.getAttribute('data-track-id');
  const bpm = parseFloat(card.getAttribute('data-bpm'));
  const firstBeatMs = parseFloat(card.getAttribute('data-first-beat-ms'));
  const durationMs = parseFloat(card.getAttribute('data-duration-ms'));
  const msPerBeat = 60000 / bpm;
  const msPerBar = msPerBeat * 4;

  const labelEl = selectedMarker.querySelector('.marker-label');
  if (!labelEl) return null;
  const pad = labelEl.textContent.trim();

  let cuesData = {{}};
  try {{ cuesData = JSON.parse(card.getAttribute('data-cues')); }} catch(ex) {{}}
  const cueInfo = cuesData[pad];
  if (!cueInfo) return null;

  return {{ card, trackId, bpm, firstBeatMs, durationMs, msPerBeat, msPerBar, pad, cuesData, cueInfo, marker: selectedMarker }};
}}

// The one place a cue's position actually gets moved: updates the
// marker's on-screen position AND its tooltip (previously only the
// position updated, leaving a stale tooltip after any move -- a real,
// separately-noticed gap), posts the change, and rolls both back if the
// write fails. Used by the arrow-key nudge and the numeric editor alike,
// so there's one correct implementation of "move a cue," not two.
function applyCueMove(card, trackId, pad, marker, newPosMs, cuesData, cueInfo) {{
  const durationMs = parseFloat(card.getAttribute('data-duration-ms'));
  newPosMs = Math.max(0, Math.min(newPosMs, durationMs));

  const previousPos = cueInfo.position_ms;
  const previousTitle = marker.getAttribute('title') || '';
  cueInfo.position_ms = newPosMs;
  card.setAttribute('data-cues', JSON.stringify(cuesData));
  const leftPct = (newPosMs / durationMs) * 100;
  marker.style.left = leftPct.toFixed(4) + '%';
  const newTimeStr = formatMsAsTime(newPosMs);
  const titlePrefix = previousTitle.split(' @ ')[0];
  marker.setAttribute('title', titlePrefix + ' @ ' + newTimeStr);
  setTrackStatus(trackId, 'adjusted');
  syncEditorFields(cueInfo, card);

  return fetch(SERVER + '/session/track/' + trackId + '/cue/' + pad, {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ position_ms: newPosMs, status: 'adjusted' }})
  }}).then(res => {{
    if (!res.ok) throw new Error('server returned ' + res.status);
  }}).catch(err => {{
    console.error('Failed to adjust cue ' + pad + ':', err);
    cueInfo.position_ms = previousPos;
    card.setAttribute('data-cues', JSON.stringify(cuesData));
    const revertPct = (previousPos / durationMs) * 100;
    marker.style.left = revertPct.toFixed(4) + '%';
    marker.setAttribute('title', previousTitle);
    setTrackStatus(trackId, 'error');
    syncEditorFields(cueInfo, card);
  }});
}}

document.addEventListener('click', function(e) {{
  const marker = e.target.closest('.hot-cue-marker');
  if (marker) {{
    // Only allow selecting proposed cue markers, not existing ones
    if (marker.classList.contains('existing')) return;
    e.stopPropagation();
    deselectMarker();
    marker.classList.add('selected');
    selectedMarker = marker;
    selectedTrackCard = marker.closest('.review-card');
    showEditorFor(marker, selectedTrackCard);
    return;
  }}

  // Clicks inside the popover itself (its inputs, Preview button, Loop
  // checkbox) must not fall through to scrub-preview or deselect below --
  // otherwise focusing a field to type in it immediately closes the editor.
  if (e.target.closest('#cue-editor-popover')) return;

  // A click inside a timeline that isn't on any marker previews from
  // that point -- "set anywhere on the track," not just relative to an
  // existing cue.
  const timeline = e.target.closest('.timeline-container');
  if (timeline && !e.target.closest('.mem-cue-marker')) {{
    const card = timeline.closest('.review-card');
    if (card && card.getAttribute('data-has-audio') === 'true') {{
      const rect = timeline.getBoundingClientRect();
      const fraction = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const durationMs = parseFloat(card.getAttribute('data-duration-ms'));
      startPreview(card.getAttribute('data-track-id'), fraction * durationMs, {{ loop: false }});
    }}
  }}

  // Click elsewhere deselects
  deselectMarker();
}});

document.addEventListener('keydown', function(e) {{
  // Once the numeric editor exists, arrow keys used to move a text
  // cursor inside it would otherwise also nudge the selected cue.
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  if (e.key === ' ') {{
    if (selectedMarker) {{
      e.preventDefault();
      togglePlayPauseOrStartSelected();
    }}
    return;
  }}

  if (e.key === 'Tab' && selectedTrackCard) {{
    e.preventDefault();
    cycleMarkerSelection(e.shiftKey ? -1 : 1);
    return;
  }}

  const info = getSelectedCueInfo();
  if (!info) return;
  const {{ card, trackId, firstBeatMs, msPerBeat, msPerBar, pad, cuesData, cueInfo, marker }} = info;

  if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {{
    e.preventDefault();
    const direction = e.key === 'ArrowRight' ? 1 : -1;
    let newPos = cueInfo.position_ms + direction * msPerBar;

    // Snap to beat grid
    const rawBeat = (newPos - firstBeatMs) / msPerBeat + 1;
    const snappedBeat = Math.max(1, Math.round(rawBeat));
    newPos = firstBeatMs + (snappedBeat - 1) * msPerBeat;

    applyCueMove(card, trackId, pad, marker, newPos, cuesData, cueInfo);
  }}

  if (e.key === 'Delete' || e.key === 'Backspace') {{
    e.preventDefault();
    const previousOpacity = marker.style.opacity;
    marker.style.opacity = '0.3';
    deselectMarker();
    setTrackStatus(trackId, 'adjusted');

    fetch(SERVER + '/session/track/' + trackId + '/cue/' + pad, {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ status: 'skipped' }})
    }}).then(res => {{
      if (!res.ok) throw new Error('server returned ' + res.status);
    }}).catch(err => {{
      console.error('Failed to skip cue ' + pad + ':', err);
      marker.style.opacity = previousOpacity;
      setTrackStatus(trackId, 'error');
    }});
  }}

  if (e.key === 'Escape') {{
    deselectMarker();
  }}
}});

function cycleMarkerSelection(direction) {{
  if (!selectedTrackCard) return;
  const markers = Array.from(selectedTrackCard.querySelectorAll('.hot-cue-marker:not(.existing)'));
  if (markers.length === 0) return;
  const currentIdx = selectedMarker ? markers.indexOf(selectedMarker) : -1;
  let nextIdx = currentIdx + direction;
  if (nextIdx < 0) nextIdx = markers.length - 1;
  if (nextIdx >= markers.length) nextIdx = 0;
  const next = markers[nextIdx];
  const card = selectedTrackCard;
  deselectMarker();
  next.classList.add('selected');
  selectedMarker = next;
  selectedTrackCard = card;
  showEditorFor(next, card);
}}

// --- Numeric cue-position editor (bar / time, synced) ---

function formatMsAsTime(ms) {{
  const totalSeconds = ms / 1000;
  const minutes = Math.floor(totalSeconds / 60);
  const secs = (totalSeconds - minutes * 60).toFixed(1).padStart(4, '0');
  return minutes + ':' + secs;
}}

function parseTimeInput(str) {{
  const m = /^(\\d+):(\\d+(?:\\.\\d+)?)$/.exec(str.trim());
  if (!m) return null;
  return (parseInt(m[1], 10) * 60 + parseFloat(m[2])) * 1000;
}}

function msToBar(ms, firstBeatMs, msPerBeat) {{
  const beat = Math.max(1, Math.round((ms - firstBeatMs) / msPerBeat + 1));
  return Math.floor((beat - 1) / 4) + 1;
}}

function barToMs(bar, firstBeatMs, msPerBeat) {{
  const beat = (bar - 1) * 4 + 1;
  return firstBeatMs + (beat - 1) * msPerBeat;
}}

function syncEditorFields(cueInfo, card) {{
  const popover = document.getElementById('cue-editor-popover');
  if (popover.classList.contains('hidden') || selectedTrackCard !== card) return;
  const firstBeatMs = parseFloat(card.getAttribute('data-first-beat-ms'));
  const bpm = parseFloat(card.getAttribute('data-bpm'));
  const msPerBeat = 60000 / bpm;
  const barEl = document.getElementById('editor-bar');
  const timeEl = document.getElementById('editor-time');
  barEl.value = msToBar(cueInfo.position_ms, firstBeatMs, msPerBeat);
  timeEl.value = formatMsAsTime(cueInfo.position_ms);
  // Setting .value programmatically doesn't fire 'input', so any
  // leftover invalid-state styling from a previous bad entry has to be
  // cleared explicitly -- these values were just derived from a real
  // position, so they're always valid.
  barEl.classList.remove('invalid');
  timeEl.classList.remove('invalid');
}}

function showEditorFor(marker, card) {{
  const info = getSelectedCueInfo();
  if (!info) return;

  // Unhide *before* syncing fields -- syncEditorFields deliberately
  // no-ops while the popover is hidden (so applyCueMove's calls to it
  // don't fight a popover that isn't showing), which meant the very
  // first population on selection was silently skipped: the popover
  // appeared with stale/blank fields until something else happened to
  // trigger a resync. This was the real bug behind "the input controls
  // don't seem to work for the specific cue."
  const popover = document.getElementById('cue-editor-popover');
  popover.classList.remove('hidden');
  syncEditorFields(info.cueInfo, card);

  // Anchor below the marker's full height (which spans the zoom controls
  // down through the waveform and phrase bar), not beside its top -- the
  // popover used to open right on top of the waveform/zoom-controls row,
  // so it had to be dragged aside before you could see either one.
  const markerRect = marker.getBoundingClientRect();
  popover.style.left = (markerRect.left + window.scrollX + 8) + 'px';
  popover.style.top = (markerRect.bottom + window.scrollY + 8) + 'px';

  // Positioning it below a marker that's near the bottom of the viewport
  // can push the popover's own bottom -- where Preview/Loop live -- past
  // the visible area, so those controls appear to just not respond
  // (they're actually rendering off-screen). "nearest" only scrolls if
  // something is actually cut off, so a fully-visible popover doesn't
  // cause a jump.
  popover.scrollIntoView({{ block: 'nearest', inline: 'nearest' }});

  updatePreviewButtonState(card);
  syncPreviewButtonLabel();
  document.getElementById('editor-loop-toggle').checked = previewSettings.preview_loop_enabled;
  document.getElementById('editor-loop-bars').value = previewSettings.preview_loop_bars;
}}

function hideEditor() {{
  document.getElementById('cue-editor-popover').classList.add('hidden');
}}

function parseBarInput(value) {{
  const bar = parseInt(value, 10);
  return (Number.isFinite(bar) && bar >= 1 && String(bar) === value.trim()) ? bar : null;
}}

// Real-time validation feedback (a red border, not a silent no-op) as
// soon as the field holds something that won't parse -- previously an
// invalid Time value just did nothing at all on Enter, with no
// indication of why, which read as "the input doesn't work."
document.getElementById('editor-bar').addEventListener('input', function() {{
  this.classList.toggle('invalid', this.value.trim() !== '' && parseBarInput(this.value) === null);
}});
document.getElementById('editor-time').addEventListener('input', function() {{
  this.classList.toggle('invalid', this.value.trim() !== '' && parseTimeInput(this.value) === null);
}});

document.getElementById('editor-bar').addEventListener('keydown', function(e) {{
  if (e.key !== 'Enter') return;
  const info = getSelectedCueInfo();
  if (!info) return;
  const bar = parseBarInput(this.value);
  if (bar === null) {{ this.classList.add('invalid'); return; }}
  this.classList.remove('invalid');
  const newMs = barToMs(bar, info.firstBeatMs, info.msPerBeat);
  applyCueMove(info.card, info.trackId, info.pad, info.marker, newMs, info.cuesData, info.cueInfo);
}});

document.getElementById('editor-time').addEventListener('keydown', function(e) {{
  if (e.key !== 'Enter') return;
  const info = getSelectedCueInfo();
  if (!info) return;
  const newMs = parseTimeInput(this.value);
  if (newMs === null) {{ this.classList.add('invalid'); return; }}
  this.classList.remove('invalid');
  applyCueMove(info.card, info.trackId, info.pad, info.marker, newMs, info.cuesData, info.cueInfo);
}});

// --- Audio preview ---

function canPlayFormat(format) {{
  const mimeByFormat = {{
    mp3: 'audio/mpeg', flac: 'audio/flac', aiff: 'audio/aiff', aif: 'audio/aiff',
    m4a: 'audio/mp4', mp4: 'audio/mp4', wav: 'audio/wav', ogg: 'audio/ogg'
  }};
  const mime = mimeByFormat[format];
  if (!mime) return false;
  return previewAudio.canPlayType(mime) !== '';
}}

function updatePreviewButtonState(card) {{
  const btn = document.getElementById('editor-preview-btn');
  const hasAudio = card.getAttribute('data-has-audio') === 'true';
  const format = card.getAttribute('data-audio-format');
  if (!hasAudio) {{
    btn.disabled = true;
    btn.title = 'No local audio file found for this track';
  }} else if (!canPlayFormat(format)) {{
    btn.disabled = true;
    btn.title = 'Preview unavailable: this browser can\\'t play .' + format + ' files';
  }} else {{
    btn.disabled = false;
    btn.title = '';
  }}
}}

// Reflects whether *this* cue is the thing actually playing right now --
// not just whether *something* is playing -- so switching the selection
// to a different cue while one previews doesn't leave a stale "Stop"
// label on a button that would actually start a new preview if pressed.
function syncPreviewButtonLabel() {{
  const btn = document.getElementById('editor-preview-btn');
  const info = getSelectedCueInfo();
  if (!info) return;
  const isPlayingThis = currentPreviewTrackId === info.trackId && !previewAudio.paused;
  btn.innerHTML = isPlayingThis ? '&#9632; Stop' : '&#9654; Preview';
}}

// The popover's Preview button is a start/stop toggle, not just a start
// button -- previously there was no way to stop a preview from inside
// the popover at all (real feedback: "there is no way to stop it
// there"). Stopping always fully stops (via stopPreview(), same as
// natural end-of-track) rather than pausing in place, so pressing
// Preview again starts fresh from the cue-relative position instead of
// resuming from an arbitrary paused point -- matching what "Preview"
// means here (audition this cue), not a generic transport control.
function toggleSelectedCuePreview() {{
  const info = getSelectedCueInfo();
  if (!info) return;
  if (currentPreviewTrackId === info.trackId && !previewAudio.paused) {{
    stopPreview();
  }} else {{
    previewSelectedCue();
  }}
  syncPreviewButtonLabel();
}}

function stopPreview() {{
  previewAudio.pause();
  currentPreviewTrackId = null;
  stopPlayheadLoop();
  const nowPlaying = document.getElementById('now-playing');
  nowPlaying.classList.add('hidden');
  nowPlaying.textContent = '';
  syncPreviewButtonLabel();
}}

function startPreview(trackId, startMs, opts) {{
  opts = opts || {{}};
  const loop = opts.loop !== undefined ? opts.loop : previewSettings.preview_loop_enabled;
  const card = document.querySelector('.review-card[data-track-id="' + trackId + '"]');
  if (!card || card.getAttribute('data-has-audio') !== 'true') return;

  const durationMs = parseFloat(card.getAttribute('data-duration-ms'));
  const bpm = parseFloat(card.getAttribute('data-bpm'));
  const loopBars = opts.loopBars !== undefined ? opts.loopBars : previewSettings.preview_loop_bars;
  const loopMs = (60000 / bpm) * 4 * loopBars;
  const loopStartMs = startMs;
  const loopEndMs = Math.min(durationMs, startMs + loopMs);

  const startNewSrc = currentPreviewTrackId !== trackId;
  currentPreviewTrackId = trackId;
  if (startNewSrc) {{
    previewAudio.src = SERVER + '/audio/' + trackId;
  }}
  previewAudio.currentTime = startMs / 1000;
  previewAudio.play().catch(err => console.error('Preview playback failed:', err));

  previewAudio.onended = stopPreview;
  previewAudio.ontimeupdate = function() {{
    if (loop && previewAudio.currentTime * 1000 >= loopEndMs) {{
      previewAudio.currentTime = loopStartMs / 1000;
    }}
  }};

  const nowPlaying = document.getElementById('now-playing');
  nowPlaying.textContent = 'Previewing: ' + (card.querySelector('h1') ? card.querySelector('h1').textContent : trackId) + (loop ? ' (looping)' : '');
  nowPlaying.classList.remove('hidden');

  startPlayheadLoop(card, durationMs);
}}

function togglePlayPauseOrStartSelected() {{
  const info = getSelectedCueInfo();
  if (!info) return;
  if (currentPreviewTrackId === info.trackId && !previewAudio.paused) {{
    previewAudio.pause();
  }} else if (currentPreviewTrackId === info.trackId && previewAudio.paused) {{
    previewAudio.play().catch(err => console.error('Preview playback failed:', err));
  }} else {{
    previewSelectedCue();
  }}
}}

function previewSelectedCue() {{
  const info = getSelectedCueInfo();
  if (!info) return;
  const loop = document.getElementById('editor-loop-toggle').checked;
  const loopBarsInput = parseInt(document.getElementById('editor-loop-bars').value, 10);
  const loopBars = (Number.isFinite(loopBarsInput) && loopBarsInput >= 1)
    ? loopBarsInput : previewSettings.preview_loop_bars;

  let startMs;
  if (loop) {{
    // Center the loop *on* the cue point rather than tying it to the
    // separate pre-roll setting -- for a 2-bar loop that's 1 bar before
    // the cue to 1 bar after it, so every pass actually crosses the
    // exact point being verified instead of stopping short of it (or,
    // if pre-roll happened to be shorter than the loop, overshooting
    // past it with no way to hear the moment itself repeat).
    startMs = Math.max(0, info.cueInfo.position_ms - (info.msPerBar * loopBars) / 2);
  }} else {{
    const preRollMs = info.msPerBar * previewSettings.preview_pre_roll_bars;
    startMs = Math.max(0, info.cueInfo.position_ms - preRollMs);
  }}
  startPreview(info.trackId, startMs, {{ loop: loop, loopBars: loopBars }});
}}

// --- Global transport bar (play/pause, skip back/forward) --
// Lives in the sticky header, so "persist at the top as you scroll" is
// free -- .sticky-header is already position:sticky. Controls whatever
// is currently loaded in the one shared previewAudio element; there's
// nothing to control until some preview has started at least once (a
// marker preview or a waveform click), same as a real player with
// nothing queued.

const TRANSPORT_SKIP_SECONDS = 5;

function transportTogglePlayPause() {{
  if (!currentPreviewTrackId) return;
  if (previewAudio.paused) {{
    previewAudio.play().catch(err => console.error('Preview playback failed:', err));
  }} else {{
    previewAudio.pause();
  }}
}}

function transportSkip(direction) {{
  if (!currentPreviewTrackId) return;
  const card = document.querySelector('.review-card[data-track-id="' + currentPreviewTrackId + '"]');
  const durationMs = card ? parseFloat(card.getAttribute('data-duration-ms')) : Infinity;
  const newTime = previewAudio.currentTime + direction * TRANSPORT_SKIP_SECONDS;
  previewAudio.currentTime = Math.max(0, Math.min(newTime, durationMs / 1000));
}}

function updateTransportPlayPauseButton() {{
  const btn = document.getElementById('transport-play-pause');
  const playing = currentPreviewTrackId && !previewAudio.paused;
  btn.innerHTML = playing ? '&#9208;' : '&#9654;';
  btn.classList.toggle('playing', !!playing);
}}

previewAudio.addEventListener('play', updateTransportPlayPauseButton);
previewAudio.addEventListener('pause', updateTransportPlayPauseButton);
previewAudio.addEventListener('ended', updateTransportPlayPauseButton);

function getOrCreatePlayhead() {{
  let el = document.getElementById('preview-playhead');
  if (!el) {{
    el = document.createElement('div');
    el.id = 'preview-playhead';
    el.className = 'playhead';
  }}
  return el;
}}

function startPlayheadLoop(card, durationMs) {{
  stopPlayheadLoop();
  const playhead = getOrCreatePlayhead();
  const containers = card.querySelectorAll('.timeline-container');
  containers.forEach(c => c.appendChild(playhead.cloneNode()));

  function tick() {{
    const pct = Math.max(0, Math.min(100, (previewAudio.currentTime * 1000 / durationMs) * 100));
    card.querySelectorAll('.playhead').forEach(p => {{ p.style.left = pct.toFixed(3) + '%'; }});
    // Catches pause/resume triggered from elsewhere (spacebar, the
    // global transport button) that don't go through stopPreview() --
    // this runs every frame while playing, including the last one
    // right after an external pause, so the popover's own button label
    // never drifts from what's actually happening.
    syncPreviewButtonLabel();
    if (!previewAudio.paused) {{
      playheadRAF = requestAnimationFrame(tick);
    }}
  }}
  playheadRAF = requestAnimationFrame(tick);
}}

function stopPlayheadLoop() {{
  if (playheadRAF) {{
    cancelAnimationFrame(playheadRAF);
    playheadRAF = null;
  }}
  document.querySelectorAll('.playhead').forEach(p => p.remove());
}}

// --- Global playback-preview settings (persisted server-side -- see
// settings.py; localStorage can't survive across sessions here since
// each `djcues review` invocation gets a fresh random port/origin) ---

function fetchSettings() {{
  return fetch(SERVER + '/settings').then(res => res.json()).then(data => {{
    previewSettings = data;
    document.getElementById('setting-pre-roll').value = data.preview_pre_roll_bars;
    document.getElementById('setting-loop-bars').value = data.preview_loop_bars;
    document.getElementById('setting-loop-enabled').checked = data.preview_loop_enabled;
  }}).catch(err => console.error('Failed to load playback settings:', err));
}}

function postSettings(partial) {{
  return fetch(SERVER + '/settings', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify(partial)
  }}).then(res => res.json()).then(data => {{
    previewSettings = data;
    // The loop-bars setting is editable from two places -- the settings
    // panel and the cue-editor popover -- so a change from either one
    // needs to update both, not just the field that triggered it.
    document.getElementById('setting-pre-roll').value = data.preview_pre_roll_bars;
    document.getElementById('setting-loop-bars').value = data.preview_loop_bars;
    document.getElementById('setting-loop-enabled').checked = data.preview_loop_enabled;
    document.getElementById('editor-loop-bars').value = data.preview_loop_bars;
  }}).catch(err => console.error('Failed to save playback settings:', err));
}}

document.getElementById('setting-pre-roll').addEventListener('change', function() {{
  const v = parseFloat(this.value);
  if (Number.isFinite(v) && v >= 0) postSettings({{ preview_pre_roll_bars: v }});
}});
document.getElementById('setting-loop-bars').addEventListener('change', function() {{
  const v = parseFloat(this.value);
  if (Number.isFinite(v) && v > 0) postSettings({{ preview_loop_bars: v }});
}});
document.getElementById('setting-loop-enabled').addEventListener('change', function() {{
  postSettings({{ preview_loop_enabled: this.checked }});
}});
// Same persistence as the settings panel's own loop-bars field above --
// editing it here updates the real saved default too (real feedback:
// "persist properly"), not just a value local to this popover.
document.getElementById('editor-loop-bars').addEventListener('change', function() {{
  const v = parseFloat(this.value);
  if (Number.isFinite(v) && v > 0) postSettings({{ preview_loop_bars: v }});
}});

function toggleSettings() {{
  const panel = document.getElementById('settings-panel');
  const shortcuts = document.getElementById('shortcuts-panel');
  shortcuts.classList.add('hidden');
  document.getElementById('shortcuts-toggle').classList.remove('active');
  panel.classList.toggle('hidden');
  document.getElementById('settings-toggle').classList.toggle('active');
}}

function toggleShortcuts() {{
  const panel = document.getElementById('shortcuts-panel');
  const settings = document.getElementById('settings-panel');
  settings.classList.add('hidden');
  document.getElementById('settings-toggle').classList.remove('active');
  panel.classList.toggle('hidden');
  document.getElementById('shortcuts-toggle').classList.toggle('active');
}}

// --- Misc UX helpers ---

function jumpToNextPending() {{
  const cards = Array.from(document.querySelectorAll('.review-card'));
  const next = cards.find(c => {{
    const indicator = c.querySelector('.status-indicator');
    return indicator && indicator.getAttribute('data-status') === 'pending';
  }});
  if (next) {{
    next.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
    next.style.outline = '2px solid #17a2b8';
    setTimeout(() => {{ next.style.outline = ''; }}, 1200);
  }}
}}

function applyConfidenceStyling() {{
  document.querySelectorAll('.review-card').forEach(card => {{
    let cuesData = {{}};
    try {{ cuesData = JSON.parse(card.getAttribute('data-cues')); }} catch(ex) {{ return; }}
    card.querySelectorAll('.hot-cue-marker:not(.existing)').forEach(marker => {{
      const labelEl = marker.querySelector('.marker-label');
      if (!labelEl) return;
      const pad = labelEl.textContent.trim();
      const info = cuesData[pad];
      if (info && typeof info.confidence === 'number' && info.confidence < 0.5) {{
        marker.classList.add('low-confidence');
      }}
    }});
  }});
}}

// --- Waveform zoom ---

function initZoomControls() {{
  document.querySelectorAll('.timeline-scroll-wrapper').forEach(wrapper => {{
    const container = wrapper.querySelector('.timeline-container');
    if (!container) return;
    const controls = document.createElement('div');
    controls.className = 'zoom-controls';
    [1, 2, 4, 8].forEach(level => {{
      const btn = document.createElement('button');
      btn.className = 'btn btn-zoom' + (level === 1 ? ' active' : '');
      btn.textContent = level + 'x';
      btn.type = 'button';
      btn.addEventListener('click', function() {{
        const oldWidth = wrapper.clientWidth;
        const centerFraction = (wrapper.scrollLeft + oldWidth / 2) / container.offsetWidth;
        container.style.width = (level * 100) + '%';
        controls.querySelectorAll('.btn-zoom').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        // Restore the centered time position instead of snapping back
        // to the start of the track.
        requestAnimationFrame(() => {{
          wrapper.scrollLeft = centerFraction * container.offsetWidth - oldWidth / 2;
        }});
      }});
      controls.appendChild(btn);
    }});
    wrapper.parentElement.insertBefore(controls, wrapper);
  }});
}}

// --- Cue editor popover: draggable by its handle ---
// showEditorFor repositions the popover next to the newly-selected marker
// on every selection; a drag only moves it for as long as the current
// selection stays open, matching the handle's job of getting the popover
// out of the way of whatever it's currently covering (a waveform detail,
// a nearby marker) rather than remembering a custom position long-term.
function initPopoverDrag() {{
  const popover = document.getElementById('cue-editor-popover');
  const handle = document.getElementById('cue-editor-handle');
  let dragging = false;
  let startX = 0, startY = 0, startLeft = 0, startTop = 0;

  handle.addEventListener('mousedown', function(e) {{
    dragging = true;
    startX = e.clientX;
    startY = e.clientY;
    startLeft = parseFloat(popover.style.left) || 0;
    startTop = parseFloat(popover.style.top) || 0;
    e.preventDefault();
  }});
  document.addEventListener('mousemove', function(e) {{
    if (!dragging) return;
    popover.style.left = (startLeft + (e.clientX - startX)) + 'px';
    popover.style.top = (startTop + (e.clientY - startY)) + 'px';
  }});
  document.addEventListener('mouseup', function() {{
    dragging = false;
  }});
}}

// Initialize on load
updateSummaryCounts();
fetchSettings();
initZoomControls();
applyConfidenceStyling();
initPopoverDrag();
"""
