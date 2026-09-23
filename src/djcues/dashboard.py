"""Analysis dashboard — browse rekordbox playlists/tracks in the browser
and run propose/compare/beatgrid against a selected track, or launch the
existing viz/review commands for it.

Served entirely from a local-only HTTP server (see server.py's
DashboardHandler/start_dashboard_server). Same-origin page, no server URL
ever embedded (matching auth_web.py's pattern, not review.py's -- this
page is never written to a file:// path). Vanilla JS, no framework,
matching every other page in this codebase.
"""

from __future__ import annotations

import json

from djcues.viz import _PAGE_CSS


def _js_value(value: object) -> str:
    """JSON-encode any JSON-serializable value for safe embedding inside
    an HTML <script> block. json.dumps() alone is NOT enough here: it
    escapes quotes and backslashes but, being pure JSON (not JS-in-HTML
    aware), leaves "/" untouched -- a value containing a literal
    "</script>" would close the real script tag early at the HTML-parsing
    stage, before the JS engine ever sees it as "just a string". Escaping
    every "/" as "\\/" (a no-op for JSON/JS string parsing, valid in both)
    neutralizes that regardless of where the slash falls."""
    return json.dumps(value).replace("/", "\\/")


def render_dashboard_html(
    initial_playlist: dict | None = None,
    default_offset: int = 16,
    default_loop_bars: int = 4,
) -> str:
    """Render the dashboard shell page.

    initial_playlist, if given, is {"id": ..., "name": ...} for a
    playlist the CLI already resolved from a `djcues dashboard
    "Playlist Name"` argument -- auto-selected on load so the user
    doesn't have to find it in the tree again. Always safe to omit
    (browsing manually is always the fallback); never a hard requirement
    for the page to work.
    """
    initial_playlist_js = f"const INITIAL_PLAYLIST = {_js_value(initial_playlist)};"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>djcues &mdash; Analysis dashboard</title>
<style>{_PAGE_CSS}{_DASHBOARD_CSS}</style>
</head>
<body>
<div class="dashboard">
  <div class="sidebar">
    <div class="sidebar-header">
      <h1 id="logo-home" title="Back to all playlists">djcues</h1>
      <input id="playlist-search" type="text" placeholder="Filter playlists&hellip;">
      <button id="audit-library-btn" class="btn btn-ghost audit-library-btn" type="button">Audit Library</button>
    </div>
    <div id="playlist-tree" class="playlist-tree">Loading&hellip;</div>
  </div>

  <div class="main">
    <div id="track-list-panel" class="panel">
      <div class="panel-header">
        <h2 id="current-playlist-name">Select a playlist</h2>
        <input id="track-search" type="text" placeholder="Filter tracks&hellip;" class="hidden">
        <button id="audit-playlist-btn" class="btn btn-ghost hidden" type="button">Audit this playlist</button>
        <button id="flow-playlist-btn" class="btn btn-ghost hidden" type="button">Suggest set order</button>
        <button id="clash-playlist-btn" class="btn btn-ghost hidden" type="button">Check vocal clashes</button>
        <button id="transition-playlist-btn" class="btn btn-ghost hidden" type="button">Suggest transitions</button>
      </div>
      <div id="track-list" class="track-list">
        <p class="meta">Pick a playlist on the left to see its tracks.</p>
      </div>
    </div>

    <div id="audit-panel" class="panel hidden">
      <div class="panel-header">
        <button id="audit-back" class="btn btn-ghost" type="button">&larr; Back</button>
        <h2 id="audit-scope-title"></h2>
      </div>
      <p id="audit-summary" class="meta"></p>
      <div class="flags-row">
        <label><input type="checkbox" id="audit-library"> Whole library instead</label>
        <label><input type="checkbox" id="audit-half-double" checked> Include half/double-time</label>
        <label>BPM tolerance % <input type="number" id="audit-bpm-tolerance" value="6" step="0.5" min="0" class="flag-num"></label>
      </div>
      <div id="audit-findings-list" class="suggestions-list"></div>
      <div id="audit-unusable-keys-list" class="suggestions-list"></div>
    </div>

    <div id="flow-panel" class="panel hidden">
      <div class="panel-header">
        <button id="flow-back" class="btn btn-ghost" type="button">&larr; Back</button>
        <h2 id="flow-scope-title"></h2>
      </div>
      <div class="flags-row">
        <label>Cooldown fraction <input type="number" id="flow-cooldown-fraction" value="0.15" step="0.05" min="0" max="1" class="flag-num"></label>
        <button id="flow-run-btn" class="btn btn-primary" type="button">Suggest order</button>
      </div>
      <div id="flow-status" class="job-status"></div>
      <div id="flow-results-header" class="suggestions-header-row hidden">
        <span class="suggestion-bpm">Was &rarr; Now</span>
        <span class="suggestion-title">Title</span>
        <span class="suggestion-artist">Artist</span>
        <span class="suggestion-key">Key</span>
        <span class="suggestion-relation">Energy (mean &middot; peak)</span>
      </div>
      <div id="flow-results-list" class="suggestions-list"></div>
      <p id="flow-unscored-summary" class="meta small"></p>
      <div class="flags-row">
        <button id="flow-apply-btn" class="btn btn-secondary hidden" type="button">Apply this order to Rekordbox</button>
      </div>
      <div id="flow-apply-status" class="job-status"></div>
    </div>

    <div id="clash-panel" class="panel hidden">
      <div class="panel-header">
        <button id="clash-back" class="btn btn-ghost" type="button">&larr; Back</button>
        <h2 id="clash-scope-title"></h2>
      </div>
      <div class="flags-row">
        <label>Min vocal region (ms) <input type="number" id="clash-min-vocal-region-ms" value="2000" step="100" min="0" class="flag-num"></label>
        <button id="clash-run-btn" class="btn btn-primary" type="button">Scan for clashes</button>
      </div>
      <div id="clash-status" class="job-status"></div>
      <div id="clash-findings-list" class="suggestions-list"></div>
      <p id="clash-unscorable-summary" class="meta small"></p>
    </div>

    <div id="transition-panel" class="panel hidden">
      <div class="panel-header">
        <button id="transition-back" class="btn btn-ghost" type="button">&larr; Back</button>
        <h2 id="transition-scope-title"></h2>
      </div>
      <div class="flags-row">
        <label>Min vocal region (ms) <input type="number" id="transition-min-vocal-region-ms" value="2000" step="100" min="0" class="flag-num"></label>
        <button id="transition-run-btn" class="btn btn-primary" type="button">Scan for transitions</button>
      </div>
      <div id="transition-status" class="job-status"></div>
      <div id="transition-results-list" class="suggestions-list"></div>
      <p id="transition-unscorable-summary" class="meta small"></p>
    </div>

    <div id="track-detail-panel" class="panel hidden">
      <div class="panel-header">
        <button id="back-to-list" class="btn btn-ghost" type="button">&larr; Back to tracks</button>
        <h2 id="track-title"></h2>
      </div>
      <p id="track-meta" class="meta"></p>

      <div class="action-section">
        <div class="action-tabs">
          <button class="action-tab active" data-action="propose" type="button">Propose</button>
          <button class="action-tab" data-action="compare" type="button">Compare</button>
          <button class="action-tab" data-action="beatgrid" type="button">Beatgrid</button>
        </div>

        <div id="action-flags" class="action-flags">
          <p class="meta small">Pick a preset, or fine-tune the flags below directly:</p>
          <div class="preset-row flags-propose-compare">
            <button class="preset-btn" type="button" data-agentic="false" data-refine-drops="false" data-deep="false"
              title="Local heuristic only -- no audio analysis, no API calls">Quick</button>
            <button class="preset-btn" type="button" data-agentic="false" data-refine-drops="true" data-deep="false"
              title="+ fast audio-based Drop/Breakdown/Special refinement">Refine Drops</button>
            <button class="preset-btn" type="button" data-agentic="false" data-refine-drops="true" data-deep="true"
              title="+ Demucs stem separation for a cleaner refinement signal -- the slowest local option">Deep Analysis</button>
            <button class="preset-btn" type="button" data-agentic="true" data-refine-drops="false" data-deep="false"
              title="LLM-based analysis instead of the heuristic -- costs API credits">Agentic</button>
            <button class="preset-btn" type="button" data-agentic="true" data-refine-drops="true" data-deep="false"
              title="LLM + fast audio refinement">Full Agentic</button>
            <button class="preset-btn" type="button" data-agentic="true" data-refine-drops="true" data-deep="true"
              title="LLM + Demucs refinement -- the slowest and most expensive option, but the most thorough">Full Agentic + Deep</button>
          </div>
          <div class="preset-row flags-beatgrid hidden">
            <button class="preset-btn" type="button" data-beatgrid-deep="false"
              title="Self-consistency check against rekordbox's own beat grid data only">Quick Check</button>
            <button class="preset-btn" type="button" data-beatgrid-deep="true"
              title="+ cross-check against the real audio file -- slower">Verify Audio</button>
          </div>
          <div id="estimate-display" class="meta small estimate-display"></div>

          <div class="flags-row device-row">
            <label>Device
              <select id="flag-device">
                <option value="auto">Auto</option>
                <option value="cpu">CPU</option>
                <option value="cuda">CUDA (NVIDIA)</option>
                <option value="directml">DirectML (AMD / Intel iGPU)</option>
              </select>
            </label>
            <span id="device-status" class="meta small device-status"></span>
          </div>

          <div class="flags-row flags-propose-compare">
            <label><input type="checkbox" id="flag-agentic"> Agentic</label>
            <label><input type="checkbox" id="flag-refine-drops"> Refine Drops</label>
            <label><input type="checkbox" id="flag-deep"> Deep</label>
            <label><input type="checkbox" id="flag-skip-critic"> Skip Critic</label>
            <label><input type="checkbox" id="flag-no-cache"> No Cache</label>
          </div>
          <div class="flags-row flags-propose-compare">
            <label>Provider <input type="text" id="flag-provider" placeholder="(configured default)" class="flag-text"></label>
            <label>Model <input type="text" id="flag-model" placeholder="(configured default)" class="flag-text"></label>
          </div>
          <div class="flags-row flags-propose-compare">
            <label>Offset (bars) <input type="number" id="flag-offset" value="{default_offset}" class="flag-num"></label>
            <label>Loop (bars) <input type="number" id="flag-loop-bars" value="{default_loop_bars}" class="flag-num"></label>
          </div>
          <div class="flags-row flags-beatgrid hidden">
            <label><input type="checkbox" id="flag-beatgrid-deep"> Force deep (real audio check)</label>
            <label>Tolerance (ms) <input type="number" id="flag-tolerance" value="30" class="flag-num"></label>
            <label><input type="checkbox" id="flag-beatgrid-no-cache"> No Cache</label>
          </div>
        </div>

        <button id="run-btn" class="btn btn-primary" type="button">Run</button>
      </div>

      <div class="launch-section">
        <p class="meta small">Open in the existing viz/review tools instead (their own flags, not shared with the panel above):</p>
        <div class="flags-row">
          <label><input type="checkbox" id="launch-agentic"> Agentic</label>
          <label><input type="checkbox" id="launch-refine-drops"> Refine Drops</label>
          <label><input type="checkbox" id="launch-deep"> Deep</label>
        </div>
        <div class="launch-buttons">
          <button id="open-viz-btn" class="btn btn-secondary" type="button">Open in Viz</button>
          <button id="open-review-btn" class="btn btn-secondary" type="button">Open in Review</button>
        </div>
      </div>

      <div class="playlist-actions-section">
        <p class="meta small">Move, add, or remove this track from a playlist (writes directly to rekordbox -- it must be closed):</p>
        <div class="flags-row">
          <label>Destination
            <select id="playlist-dest-select"></select>
          </label>
          <button id="playlist-move-btn" class="btn btn-secondary" type="button">Move here</button>
          <button id="playlist-add-btn" class="btn btn-secondary" type="button">Add (copy)</button>
        </div>
        <div class="playlist-remove-row">
          <button id="playlist-remove-btn" class="btn btn-danger" type="button">Remove from this playlist</button>
        </div>
        <div id="playlist-action-status" class="job-status"></div>
      </div>

      <div class="harmonic-suggestions-section">
        <p class="meta small">Harmonically- and tempo-compatible tracks (Camelot Wheel + BPM):</p>
        <div class="flags-row">
          <label><input type="checkbox" id="suggest-library"> Search whole library</label>
          <label><input type="checkbox" id="suggest-half-double" checked> Include half/double-time</label>
          <label>BPM tolerance % <input type="number" id="suggest-bpm-tolerance" value="6" step="0.5" min="0" class="flag-num"></label>
        </div>
        <div id="suggestions-list" class="suggestions-list"></div>
      </div>

      <div id="results-panel" class="results-panel hidden">
        <div id="job-status" class="job-status"></div>
        <div id="job-output-wrap" class="hidden">
          <pre id="job-output"></pre>
        </div>
        <div id="job-html-fragment"></div>
      </div>
    </div>
  </div>
</div>

<script>{_render_dashboard_js(default_offset, default_loop_bars)}
{initial_playlist_js}
initDashboard();
</script>
</body>
</html>"""


_DASHBOARD_CSS = """
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: #1a1a2e;
    color: #eee;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  .dashboard { display: flex; height: 100vh; }
  .sidebar {
    width: 280px;
    flex-shrink: 0;
    background: #12122a;
    border-right: 1px solid #2a2a3e;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .sidebar-header { padding: 16px; border-bottom: 1px solid #2a2a3e; }
  .sidebar-header h1 { font-size: 1.1rem; margin: 0 0 10px; color: #17a2b8; cursor: pointer; user-select: none; }
  .sidebar-header h1:hover { color: #1fc4dd; }
  .sidebar-header input, .flag-text, .flag-num {
    width: 100%;
    background: #1e1e3a;
    border: 1px solid #2a2a3e;
    color: #eee;
    border-radius: 4px;
    padding: 6px 8px;
    font-size: 0.85rem;
  }
  .audit-library-btn { margin-top: 8px; width: 100%; }
  .flag-num { width: 70px; }
  .playlist-tree { flex: 1; overflow-y: auto; padding: 8px 0; }
  .tree-row {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    cursor: pointer;
    font-size: 0.85rem;
    white-space: nowrap;
  }
  .tree-row:hover { background: #1e1e3a; }
  .tree-row.folder { color: #ccc; font-weight: 600; }
  .tree-row.smart_playlist { color: #777; cursor: default; }
  .tree-icon { display: inline-block; width: 1em; transition: transform 0.1s; }
  .tree-row.expanded .tree-icon { transform: rotate(90deg); }
  .tree-count { color: #666; font-size: 0.78rem; }
  .tree-children { border-left: 1px solid #2a2a3e; margin-left: 10px; }

  .main { flex: 1; overflow-y: auto; padding: 24px 32px; }
  .panel-header { display: flex; align-items: center; gap: 14px; margin-bottom: 14px; }
  .panel-header h2 { margin: 0; font-size: 1.15rem; }
  #track-search { max-width: 260px; }
  .meta { color: #888; font-size: 0.85rem; line-height: 1.5; }
  .meta.small { margin: 4px 0 8px; font-size: 0.78rem; }

  .track-list { display: flex; flex-direction: column; gap: 2px; }
  .track-row {
    display: grid;
    grid-template-columns: 1fr 1fr auto;
    gap: 12px;
    align-items: center;
    padding: 9px 12px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.88rem;
  }
  .track-row:hover { background: #1e1e3a; }
  .track-title { color: #eee; }
  .track-artist { color: #999; }
  .track-meta-small { color: #666; font-size: 0.78rem; white-space: nowrap; }

  .btn {
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-size: 0.88rem;
    cursor: pointer;
    font-weight: 600;
  }
  .btn:disabled { opacity: 0.5; cursor: default; }
  .btn-primary { background: #17a2b8; color: #fff; }
  .btn-secondary { background: #2a2a3e; color: #eee; }
  .btn-danger { background: #dc3545; color: #fff; }
  .btn-ghost { background: transparent; color: #aaa; padding: 4px 8px; }
  .hidden { display: none !important; }

  .action-section { margin-top: 20px; padding: 16px; background: #12122a; border: 1px solid #2a2a3e; border-radius: 6px; }
  .action-tabs { display: flex; gap: 4px; margin-bottom: 14px; }
  .action-tab {
    background: #1e1e3a;
    color: #999;
    border: 1px solid #2a2a3e;
    border-radius: 4px 4px 0 0;
    padding: 7px 16px;
    cursor: pointer;
    font-size: 0.85rem;
  }
  .action-tab.active { background: #17a2b8; color: #fff; border-color: #17a2b8; }
  .flags-row { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; margin-bottom: 10px; font-size: 0.82rem; color: #ccc; }
  .flags-row label { display: flex; align-items: center; gap: 4px; }

  .preset-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
  .preset-btn {
    background: #1e1e3a;
    color: #ccc;
    border: 1px solid #2a2a3e;
    border-radius: 4px;
    padding: 6px 12px;
    font-size: 0.8rem;
    cursor: pointer;
  }
  .preset-btn:hover { border-color: #17a2b8; color: #fff; }
  .preset-btn.active { background: #17a2b8; color: #fff; border-color: #17a2b8; }
  .estimate-display { min-height: 1.3em; margin-bottom: 12px; }

  .device-row { margin-bottom: 14px; padding-bottom: 14px; border-bottom: 1px solid #2a2a3e; }
  #flag-device {
    background: #1e1e3a;
    color: #eee;
    border: 1px solid #2a2a3e;
    border-radius: 4px;
    padding: 5px 8px;
    font-size: 0.82rem;
  }
  .device-status { margin-left: 4px; }
  .device-status.warn { color: #e0a030; }

  .launch-section { margin-top: 16px; padding: 16px; border-top: 1px solid #2a2a3e; }
  .launch-buttons { display: flex; gap: 8px; margin-top: 8px; }

  .playlist-actions-section { margin-top: 16px; padding: 16px; border-top: 1px solid #2a2a3e; }
  .playlist-remove-row { margin-top: 10px; }
  #playlist-dest-select {
    background: #1e1e3a;
    color: #eee;
    border: 1px solid #2a2a3e;
    border-radius: 4px;
    padding: 5px 8px;
    font-size: 0.82rem;
  }

  .harmonic-suggestions-section { margin-top: 16px; padding: 16px; border-top: 1px solid #2a2a3e; }
  .suggestions-list { margin-top: 10px; display: flex; flex-direction: column; gap: 4px; }
  .suggestion-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 8px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.82rem;
  }
  .suggestion-row:hover { background: #1e1e3a; }
  .suggestions-header-row {
    display: flex; align-items: center; gap: 10px; padding: 2px 8px 6px;
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.03em; color: #777;
    border-bottom: 1px solid #2a2a4a; margin-bottom: 2px;
  }
  .suggestion-title { flex: 2; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #eee; }
  .suggestion-artist { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #999; }
  .suggestion-key { color: #17a2b8; font-weight: 600; width: 2.5em; }
  .suggestion-bpm { color: #ccc; width: 5em; text-align: right; }
  .suggestion-relation { color: #999; }

  .results-panel { margin-top: 18px; padding: 16px; background: #12122a; border: 1px solid #2a2a3e; border-radius: 6px; }
  .job-status { font-size: 0.9rem; margin-bottom: 8px; }
  .job-status.info { color: #17a2b8; }
  .job-status.success { color: #28a745; }
  .job-status.error { color: #dc3545; }
  #job-output {
    background: #0d0d1a;
    border: 1px solid #2a2a3e;
    border-radius: 4px;
    padding: 12px;
    font-size: 0.78rem;
    line-height: 1.5;
    white-space: pre-wrap;
    max-height: 400px;
    overflow-y: auto;
  }
"""


def _render_dashboard_js(default_offset: int, default_loop_bars: int) -> str:
    return f"""
'use strict';

const playlistTreeEl = document.getElementById('playlist-tree');
const playlistSearchEl = document.getElementById('playlist-search');
const logoHomeEl = document.getElementById('logo-home');
const trackListPanelEl = document.getElementById('track-list-panel');
const trackDetailPanelEl = document.getElementById('track-detail-panel');
const currentPlaylistNameEl = document.getElementById('current-playlist-name');
const trackSearchEl = document.getElementById('track-search');
const trackListEl = document.getElementById('track-list');
const trackTitleEl = document.getElementById('track-title');
const trackMetaEl = document.getElementById('track-meta');
const backToListBtn = document.getElementById('back-to-list');
const actionTabs = document.querySelectorAll('.action-tab');
const runBtn = document.getElementById('run-btn');
const openVizBtn = document.getElementById('open-viz-btn');
const openReviewBtn = document.getElementById('open-review-btn');
const resultsPanelEl = document.getElementById('results-panel');
const jobStatusEl = document.getElementById('job-status');
const jobOutputWrapEl = document.getElementById('job-output-wrap');
const jobOutputEl = document.getElementById('job-output');
const jobHtmlFragmentEl = document.getElementById('job-html-fragment');
const estimateDisplayEl = document.getElementById('estimate-display');
const flagDeviceEl = document.getElementById('flag-device');
const deviceStatusEl = document.getElementById('device-status');
const playlistDestSelectEl = document.getElementById('playlist-dest-select');
const playlistMoveBtn = document.getElementById('playlist-move-btn');
const playlistAddBtn = document.getElementById('playlist-add-btn');
const playlistRemoveBtn = document.getElementById('playlist-remove-btn');
const playlistActionStatusEl = document.getElementById('playlist-action-status');
const suggestLibraryEl = document.getElementById('suggest-library');
const suggestHalfDoubleEl = document.getElementById('suggest-half-double');
const suggestBpmToleranceEl = document.getElementById('suggest-bpm-tolerance');
const suggestionsListEl = document.getElementById('suggestions-list');
const auditLibraryBtn = document.getElementById('audit-library-btn');
const auditPlaylistBtn = document.getElementById('audit-playlist-btn');
const auditPanelEl = document.getElementById('audit-panel');
const auditBackBtn = document.getElementById('audit-back');
const auditScopeTitleEl = document.getElementById('audit-scope-title');
const auditSummaryEl = document.getElementById('audit-summary');
const auditLibraryEl = document.getElementById('audit-library');
const auditHalfDoubleEl = document.getElementById('audit-half-double');
const auditBpmToleranceEl = document.getElementById('audit-bpm-tolerance');
const auditFindingsListEl = document.getElementById('audit-findings-list');
const auditUnusableKeysListEl = document.getElementById('audit-unusable-keys-list');
const flowPlaylistBtn = document.getElementById('flow-playlist-btn');
const flowPanelEl = document.getElementById('flow-panel');
const flowBackBtn = document.getElementById('flow-back');
const flowScopeTitleEl = document.getElementById('flow-scope-title');
const flowCooldownFractionEl = document.getElementById('flow-cooldown-fraction');
const flowRunBtn = document.getElementById('flow-run-btn');
const flowStatusEl = document.getElementById('flow-status');
const flowResultsHeaderEl = document.getElementById('flow-results-header');
const flowResultsListEl = document.getElementById('flow-results-list');
const flowUnscoredSummaryEl = document.getElementById('flow-unscored-summary');
const flowApplyBtn = document.getElementById('flow-apply-btn');
const flowApplyStatusEl = document.getElementById('flow-apply-status');
const clashPlaylistBtn = document.getElementById('clash-playlist-btn');
const clashPanelEl = document.getElementById('clash-panel');
const clashBackBtn = document.getElementById('clash-back');
const clashScopeTitleEl = document.getElementById('clash-scope-title');
const clashMinVocalRegionMsEl = document.getElementById('clash-min-vocal-region-ms');
const clashRunBtn = document.getElementById('clash-run-btn');
const clashStatusEl = document.getElementById('clash-status');
const clashFindingsListEl = document.getElementById('clash-findings-list');
const clashUnscorableSummaryEl = document.getElementById('clash-unscorable-summary');
const transitionPlaylistBtn = document.getElementById('transition-playlist-btn');
const transitionPanelEl = document.getElementById('transition-panel');
const transitionBackBtn = document.getElementById('transition-back');
const transitionScopeTitleEl = document.getElementById('transition-scope-title');
const transitionMinVocalRegionMsEl = document.getElementById('transition-min-vocal-region-ms');
const transitionRunBtn = document.getElementById('transition-run-btn');
const transitionStatusEl = document.getElementById('transition-status');
const transitionResultsListEl = document.getElementById('transition-results-list');
const transitionUnscorableSummaryEl = document.getElementById('transition-unscorable-summary');

let currentPlaylistId = null;
let currentTracks = [];
let currentTrackId = null;
let currentAction = 'propose';
// Flattened from /api/playlists' tree (real playlists only -- folders/
// smart-playlists excluded here, client-side, which also sidesteps the
// common case of pyrekordbox's stricter Attribute check rejecting a
// playlist djcues itself otherwise treats as ordinary). Populated once
// by loadPlaylists(), not re-fetched per track -- the destination
// dropdown doesn't need to be any fresher than the tree already is.
let allPlaylistsFlat = [];
// The audit view's own scope -- independent of currentPlaylistId/
// currentPlaylistNameEl (the track-list/track-detail views' state),
// since audit can be library-scoped with no playlist at all, or
// playlist-scoped while the user later toggles "whole library instead"
// without leaving the view.
let auditPlaylistId = null;
let auditPlaylistName = null;
let pollTimer = null;
let jobStartedAt = null;
// The flow view's own scope -- always playlist-scoped (unlike audit,
// there's no whole-library mode; see flow's own CLI/server docstrings
// for why), tracked separately so navigating audit<->flow<->track
// doesn't clobber each other's "which playlist" state.
let flowPlaylistId = null;
let flowPlaylistName = null;
let flowPollTimer = null;
let flowJobStartedAt = null;
// The last successfully completed flow job's result, cached in memory so
// re-entering this view for the SAME playlist (e.g. after clicking into
// a track and back) restores it instead of showing a blank panel the
// user has to re-run. Cleared implicitly by simply not matching
// lastFlowResultPlaylistId when a different playlist is opened -- lost
// on a real page reload, which is fine, only in-app navigation needs to
// preserve it.
let lastFlowResult = null;
let lastFlowResultPlaylistId = null;
let lastFlowElapsedSeconds = null;
// The clash view's own scope -- always playlist-scoped, same reasoning
// as flow's above (adjacency only means something within one playlist).
let clashPlaylistId = null;
let clashPlaylistName = null;
let clashPollTimer = null;
let clashJobStartedAt = null;
// Same in-memory result cache as flow's above, same reason.
let lastClashResult = null;
let lastClashResultPlaylistId = null;
let lastClashElapsedSeconds = null;
// The transition view's own scope -- always playlist-scoped, same
// reasoning as flow's/clash's above.
let transitionPlaylistId = null;
let transitionPlaylistName = null;
let transitionPollTimer = null;
let transitionJobStartedAt = null;
// Same in-memory result cache as flow's/clash's above, same reason.
let lastTransitionResult = null;
let lastTransitionResultPlaylistId = null;
let lastTransitionElapsedSeconds = null;
// Declared here, not inline near refreshEstimate() below, so it's
// already initialized by the time updateActionFlags()'s own initial
// call (which runs before this file's later `let` statements would
// otherwise execute) reaches into refreshEstimate() for the first time.
let estimateRequestId = 0;

// --- Routing (landing / playlist / track, all reachable via browser
// Back/Forward, a bookmark, or a page refresh -- not just forward-only
// clicks) -----------------------------------------------------------
//
// One rule keeps this from drifting out of sync with itself: the ONLY
// function allowed to touch history (pushState/replaceState) is
// navigateTo(). Every click handler below calls navigateTo() with a
// plain state object instead of mutating the DOM directly; a real
// Back/Forward (popstate) re-renders that exact same kind of state
// object through the exact same renderState() function, so "what a
// click shows" and "what Back shows" can never disagree.

function stateToUrl(state) {{
  const params = new URLSearchParams();
  // Explicit view=audit marker -- audit shares playlist_id with plain
  // playlist browsing (a playlist-scoped audit and just browsing that
  // playlist have the same playlist_id), so presence-based inference
  // alone can't tell the two views apart the way it can for track vs
  // playlist vs landing below.
  if (state.view === 'audit') {{
    params.set('view', 'audit');
    if (state.playlistId) {{
      params.set('playlist_id', state.playlistId);
      params.set('playlist_name', state.playlistName || '');
    }}
    if (state.library) params.set('library', 'true');
    const qs = params.toString();
    return qs ? ('?' + qs) : window.location.pathname;
  }}
  // Explicit view=flow marker for the same reason as audit above --
  // flow shares playlist_id with plain playlist browsing. Simpler than
  // audit's own branch: flow has no library-wide mode, so playlist_id
  // is always present here.
  if (state.view === 'flow') {{
    params.set('view', 'flow');
    params.set('playlist_id', state.playlistId);
    params.set('playlist_name', state.playlistName || '');
    return '?' + params.toString();
  }}
  // Same reasoning as flow's own branch above -- clash also has no
  // library-wide mode, playlist_id is always present here.
  if (state.view === 'clash') {{
    params.set('view', 'clash');
    params.set('playlist_id', state.playlistId);
    params.set('playlist_name', state.playlistName || '');
    return '?' + params.toString();
  }}
  // Same reasoning as flow's/clash's own branches above -- transition
  // also has no library-wide mode, playlist_id is always present here.
  if (state.view === 'transition') {{
    params.set('view', 'transition');
    params.set('playlist_id', state.playlistId);
    params.set('playlist_name', state.playlistName || '');
    return '?' + params.toString();
  }}
  if (state.playlistId) {{
    params.set('playlist_id', state.playlistId);
    params.set('playlist_name', state.playlistName || '');
  }}
  if (state.trackId) {{
    params.set('track_id', state.trackId);
    params.set('track_title', state.trackTitle || '');
  }}
  const qs = params.toString();
  return qs ? ('?' + qs) : window.location.pathname;
}}

function urlToState() {{
  const params = new URLSearchParams(window.location.search);
  if (params.get('view') === 'audit') {{
    return {{
      view: 'audit', playlistId: params.get('playlist_id'), playlistName: params.get('playlist_name'),
      library: params.get('library') === 'true',
    }};
  }}
  if (params.get('view') === 'flow') {{
    return {{ view: 'flow', playlistId: params.get('playlist_id'), playlistName: params.get('playlist_name') }};
  }}
  if (params.get('view') === 'clash') {{
    return {{ view: 'clash', playlistId: params.get('playlist_id'), playlistName: params.get('playlist_name') }};
  }}
  if (params.get('view') === 'transition') {{
    return {{ view: 'transition', playlistId: params.get('playlist_id'), playlistName: params.get('playlist_name') }};
  }}
  const playlistId = params.get('playlist_id');
  const trackId = params.get('track_id');
  if (trackId) {{
    return {{
      view: 'track', playlistId, playlistName: params.get('playlist_name'),
      trackId, trackTitle: params.get('track_title'),
    }};
  }}
  if (playlistId) {{
    return {{ view: 'playlist', playlistId, playlistName: params.get('playlist_name') }};
  }}
  return {{ view: 'landing' }};
}}

async function renderState(state) {{
  if (state.view === 'track') {{
    await renderTrackDetail(state.playlistId, state.playlistName, state.trackId, state.trackTitle);
  }} else if (state.view === 'playlist') {{
    await renderPlaylistTracks(state.playlistId, state.playlistName);
  }} else if (state.view === 'audit') {{
    renderAudit(state.playlistId, state.playlistName, state.library);
  }} else if (state.view === 'flow') {{
    renderFlow(state.playlistId, state.playlistName);
  }} else if (state.view === 'clash') {{
    renderClash(state.playlistId, state.playlistName);
  }} else if (state.view === 'transition') {{
    renderTransition(state.playlistId, state.playlistName);
  }} else {{
    renderLanding();
  }}
}}

function navigateTo(state, mode) {{
  // mode: 'push' for a real navigation (a click), 'replace' for
  // restoring/correcting the current entry (initial load) -- 'replace'
  // never adds a new Back stop.
  const url = stateToUrl(state);
  if (mode === 'push') {{
    history.pushState(state, '', url);
  }} else {{
    history.replaceState(state, '', url);
  }}
  renderState(state);
}}

window.addEventListener('popstate', (event) => {{
  renderState(event.state || urlToState());
}});

logoHomeEl.addEventListener('click', () => navigateTo({{ view: 'landing' }}, 'push'));

// A plain network-level fetch failure (server unreachable, connection
// refused, ...) throws a generic TypeError with an unhelpful browser
// message -- the most common real cause here is the local server's own
// idle timeout closing the port out from under an open tab, so say that
// instead of parroting the raw browser error (matches auth_web.py's own
// established pattern for this exact situation).
function friendlyErrorMessage(err) {{
  if (err instanceof TypeError) {{
    return 'Could not reach the local djcues server. It may have shut down ' +
      '(idle timeout) -- run "djcues dashboard" again in your terminal.';
  }}
  return err.message;
}}

async function fetchJson(url, opts) {{
  const res = await fetch(url, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || ('server returned ' + res.status));
  return data;
}}

function formatDuration(ms) {{
  const totalSeconds = Math.round(ms / 1000);
  const m = Math.floor(totalSeconds / 60);
  const s = totalSeconds % 60;
  return m + ':' + String(s).padStart(2, '0');
}}

// --- Playlist tree ----------------------------------------------------

function renderPlaylistNode(node, depth) {{
  const el = document.createElement('div');
  el.className = 'tree-node';

  const row = document.createElement('div');
  row.className = 'tree-row ' + node.kind;
  row.style.paddingLeft = (12 + depth * 16) + 'px';
  const icon = node.kind === 'folder' ? '\\u25b8' : (node.kind === 'smart_playlist' ? '\\u2699' : '\\u266a');
  const countLabel = node.track_count != null ? ' (' + node.track_count + ')' : '';
  const iconSpan = document.createElement('span');
  iconSpan.className = 'tree-icon';
  iconSpan.textContent = icon;
  const labelSpan = document.createElement('span');
  labelSpan.className = 'tree-label';
  labelSpan.textContent = node.name;
  const countSpan = document.createElement('span');
  countSpan.className = 'tree-count';
  countSpan.textContent = countLabel;
  row.appendChild(iconSpan);
  row.appendChild(labelSpan);
  row.appendChild(countSpan);
  el.appendChild(row);

  const childrenWrap = document.createElement('div');
  childrenWrap.className = 'tree-children hidden';
  node.children.forEach(child => childrenWrap.appendChild(renderPlaylistNode(child, depth + 1)));
  el.appendChild(childrenWrap);

  row.addEventListener('click', () => {{
    if (node.kind === 'folder') {{
      childrenWrap.classList.toggle('hidden');
      row.classList.toggle('expanded');
    }} else if (node.kind === 'playlist') {{
      navigateTo({{ view: 'playlist', playlistId: node.id, playlistName: node.name }}, 'push');
    }}
    // smart_playlist: not browsable here (rekordbox doesn't populate its
    // track membership the same way) -- click intentionally does nothing.
  }});

  return el;
}}

function flattenPlaylists(nodes, out) {{
  nodes.forEach(node => {{
    if (node.kind === 'playlist') out.push({{ id: node.id, name: node.name }});
    if (node.children && node.children.length) flattenPlaylists(node.children, out);
  }});
  return out;
}}

async function loadPlaylists() {{
  try {{
    const data = await fetchJson('/api/playlists');
    playlistTreeEl.innerHTML = '';
    data.tree.forEach(node => playlistTreeEl.appendChild(renderPlaylistNode(node, 0)));
    allPlaylistsFlat = flattenPlaylists(data.tree, []);
  }} catch (err) {{
    playlistTreeEl.textContent = 'Error: ' + friendlyErrorMessage(err);
  }}
}}

playlistSearchEl.addEventListener('input', () => {{
  const q = playlistSearchEl.value.trim().toLowerCase();
  playlistTreeEl.querySelectorAll('.tree-node').forEach(node => {{
    const label = node.querySelector(':scope > .tree-row .tree-label');
    if (!label) return;
    node.style.display = (!q || label.textContent.toLowerCase().includes(q)) ? '' : 'none';
  }});
}});

// --- Track list ---------------------------------------------------

function renderLanding() {{
  currentPlaylistId = null;
  currentTracks = [];
  currentTrackId = null;
  currentPlaylistNameEl.textContent = 'Select a playlist';
  trackSearchEl.value = '';
  trackSearchEl.classList.add('hidden');
  auditPlaylistBtn.classList.add('hidden');
  flowPlaylistBtn.classList.add('hidden');
  clashPlaylistBtn.classList.add('hidden');
  transitionPlaylistBtn.classList.add('hidden');
  trackListEl.innerHTML = '<p class="meta">Pick a playlist on the left to see its tracks.</p>';
  trackListPanelEl.classList.remove('hidden');
  trackDetailPanelEl.classList.add('hidden');
  auditPanelEl.classList.add('hidden');
  flowPanelEl.classList.add('hidden');
  clashPanelEl.classList.add('hidden');
  transitionPanelEl.classList.add('hidden');
  stopPolling();
  stopFlowPolling();
  stopClashPolling();
  stopTransitionPolling();
  document.title = 'djcues \\u2014 Analysis dashboard';
}}

async function renderPlaylistTracks(playlistId, playlistName) {{
  currentPlaylistId = playlistId;
  currentTrackId = null;
  currentPlaylistNameEl.textContent = playlistName;
  trackSearchEl.value = '';
  trackSearchEl.classList.remove('hidden');
  auditPlaylistBtn.classList.remove('hidden');
  flowPlaylistBtn.classList.remove('hidden');
  clashPlaylistBtn.classList.remove('hidden');
  transitionPlaylistBtn.classList.remove('hidden');
  trackListEl.innerHTML = 'Loading&hellip;';
  trackListPanelEl.classList.remove('hidden');
  trackDetailPanelEl.classList.add('hidden');
  auditPanelEl.classList.add('hidden');
  flowPanelEl.classList.add('hidden');
  clashPanelEl.classList.add('hidden');
  transitionPanelEl.classList.add('hidden');
  stopPolling();
  stopFlowPolling();
  stopClashPolling();
  stopTransitionPolling();
  document.title = 'djcues \\u2014 ' + playlistName;

  try {{
    const data = await fetchJson('/api/playlists/' + encodeURIComponent(playlistId) + '/tracks');
    currentTracks = data.tracks;
    renderTrackList(currentTracks);
  }} catch (err) {{
    trackListEl.textContent = 'Error: ' + friendlyErrorMessage(err);
  }}
}}

function renderTrackList(tracks) {{
  trackListEl.innerHTML = '';
  if (tracks.length === 0) {{
    trackListEl.innerHTML = '<p class="meta">No tracks in this playlist.</p>';
    return;
  }}
  tracks.forEach(t => {{
    const row = document.createElement('div');
    row.className = 'track-row';
    const titleSpan = document.createElement('span');
    titleSpan.className = 'track-title';
    titleSpan.textContent = t.title;
    const artistSpan = document.createElement('span');
    artistSpan.className = 'track-artist';
    artistSpan.textContent = t.artist;
    const metaSpan = document.createElement('span');
    metaSpan.className = 'track-meta-small';
    metaSpan.textContent = t.bpm.toFixed(1) + ' BPM \\u00b7 ' + formatDuration(t.duration_ms);
    row.appendChild(titleSpan);
    row.appendChild(artistSpan);
    row.appendChild(metaSpan);
    row.addEventListener('click', () => navigateTo({{
      view: 'track', playlistId: currentPlaylistId, playlistName: currentPlaylistNameEl.textContent,
      trackId: t.id, trackTitle: t.title,
    }}, 'push'));
    trackListEl.appendChild(row);
  }});
}}

trackSearchEl.addEventListener('input', () => {{
  const q = trackSearchEl.value.trim().toLowerCase();
  const filtered = !q ? currentTracks : currentTracks.filter(t =>
    t.title.toLowerCase().includes(q) || t.artist.toLowerCase().includes(q));
  renderTrackList(filtered);
}});

// "Back to tracks" deliberately just calls history.back() rather than
// re-implementing "go to the playlist view" directly -- that keeps this
// button and a real Back-button press doing the exact same thing (and
// means Forward correctly returns to this track afterward too), instead
// of two subtly different code paths that could drift apart.
backToListBtn.addEventListener('click', () => history.back());

// --- Track detail / actions --------------------------------------

async function renderTrackDetail(playlistId, playlistName, trackId, trackTitle) {{
  currentPlaylistId = playlistId;
  currentPlaylistNameEl.textContent = playlistName || currentPlaylistNameEl.textContent;
  currentTrackId = trackId;
  trackTitleEl.textContent = trackTitle;
  trackMetaEl.textContent = 'Loading&hellip;';
  trackListPanelEl.classList.add('hidden');
  trackDetailPanelEl.classList.remove('hidden');
  auditPanelEl.classList.add('hidden');
  flowPanelEl.classList.add('hidden');
  clashPanelEl.classList.add('hidden');
  transitionPanelEl.classList.add('hidden');
  resultsPanelEl.classList.add('hidden');
  jobOutputWrapEl.classList.add('hidden');
  jobHtmlFragmentEl.innerHTML = '';
  stopPolling();
  stopFlowPolling();
  stopClashPolling();
  stopTransitionPolling();
  document.title = 'djcues \\u2014 ' + trackTitle;

  // Reached directly (a bookmark, a refresh, or Back/Forward hopping
  // over the landing/list view) rather than via a track-row click --
  // currentTracks won't be populated yet, which "\\u2190 Back to tracks"
  // needs. Fetch it now so that still works; failure here is non-fatal,
  // the track detail below doesn't depend on it.
  if (playlistId && (currentTracks.length === 0 || currentTracks[0] === undefined)) {{
    try {{
      const data = await fetchJson('/api/playlists/' + encodeURIComponent(playlistId) + '/tracks');
      currentTracks = data.tracks;
    }} catch (err) {{ /* non-fatal */ }}
  }}

  // Move/Remove need a known source playlist (reached directly, e.g. via
  // a bookmark with no playlist_id, currentPlaylistId can be null --
  // same edge case handled above). Add only needs a destination, so it
  // stays enabled either way.
  populatePlaylistDestSelect();
  playlistMoveBtn.disabled = !currentPlaylistId;
  playlistRemoveBtn.disabled = !currentPlaylistId;
  playlistActionStatusEl.textContent = '';
  playlistActionStatusEl.className = 'job-status';
  loadSuggestions();

  try {{
    const track = await fetchJson('/api/tracks/' + encodeURIComponent(trackId));
    const items = [
      track.artist,
      track.bpm.toFixed(1) + ' BPM',
      formatDuration(track.duration_ms),
      track.phrase_count + ' phrases',
      track.existing_cue_count + ' existing cues',
      track.has_local_audio ? 'local audio available' : 'no local audio file found',
    ];
    trackMetaEl.textContent = items.join(' \\u00b7 ');
    runBtn.disabled = !track.has_phrases;
    if (!track.has_phrases) {{
      jobStatusEl.textContent = 'This track has no phrase data -- rekordbox hasn\\'t analyzed it yet.';
      jobStatusEl.className = 'job-status error';
      resultsPanelEl.classList.remove('hidden');
    }}
  }} catch (err) {{
    trackMetaEl.textContent = 'Error: ' + friendlyErrorMessage(err);
  }}
}}

actionTabs.forEach(tab => {{
  tab.addEventListener('click', () => {{
    actionTabs.forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    currentAction = tab.getAttribute('data-action');
    updateActionFlags();
  }});
}});

function updateActionFlags() {{
  const isBeatgrid = currentAction === 'beatgrid';
  document.querySelectorAll('.flags-propose-compare').forEach(el => el.classList.toggle('hidden', isBeatgrid));
  document.querySelectorAll('.flags-beatgrid').forEach(el => el.classList.toggle('hidden', !isBeatgrid));
  refreshEstimate();
}}
updateActionFlags();

// --- Presets + historical time/cost estimate -------------------------
//
// Presets are a fast path onto the same checkboxes below, not a
// separate mechanism -- clicking one just sets flag-agentic/-refine-
// drops/-deep (or flag-beatgrid-deep) and fires the run exactly like
// hand-checking those boxes would. The estimate is real history from
// this project's own analysis_cache (see analysis_cache.estimate() /
// server.py's /api/estimate), not a hardcoded guess -- 0 prior runs of
// a given combination shows honestly as "no data yet" rather than a
// fabricated number.

document.querySelectorAll('.preset-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    if (btn.hasAttribute('data-beatgrid-deep')) {{
      document.getElementById('flag-beatgrid-deep').checked = btn.getAttribute('data-beatgrid-deep') === 'true';
    }} else {{
      document.getElementById('flag-agentic').checked = btn.getAttribute('data-agentic') === 'true';
      document.getElementById('flag-refine-drops').checked = btn.getAttribute('data-refine-drops') === 'true';
      document.getElementById('flag-deep').checked = btn.getAttribute('data-deep') === 'true';
    }}
    btn.parentElement.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    refreshEstimate();
  }});
}});

// A manual checkbox edit stops matching any one preset exactly -- clear
// the highlight rather than leave it pointing at a now-stale combination.
['flag-agentic', 'flag-refine-drops', 'flag-deep', 'flag-beatgrid-deep'].forEach(id => {{
  document.getElementById(id).addEventListener('change', () => {{
    document.querySelectorAll('.preset-btn.active').forEach(b => b.classList.remove('active'));
    refreshEstimate();
  }});
}});

function formatEstimate(est) {{
  if (est.sample_count === 0) return 'No historical data yet for this combination.';
  const parts = [];
  if (est.avg_compute_seconds !== null) {{
    const s = est.avg_compute_seconds;
    parts.push(s < 90 ? Math.round(s) + 's' : (s / 60).toFixed(1) + 'min');
  }}
  if (est.avg_cost_usd) parts.push('$' + est.avg_cost_usd.toFixed(4));
  const n = est.sample_count;
  return '~' + parts.join(' \\u00b7 ') + ' (avg of ' + n + ' prior run' + (n === 1 ? '' : 's') + ')';
}}

// estimateRequestId (declared near the top, not here -- see its own
// comment) guards against a slow, now-superseded request overwriting a
// newer one's result -- rapid preset clicking fires overlapping fetches.
async function refreshEstimate() {{
  const isBeatgrid = currentAction === 'beatgrid';
  const params = new URLSearchParams({{ kind: currentAction }});
  if (isBeatgrid) {{
    params.set('deep', document.getElementById('flag-beatgrid-deep').checked ? 'true' : 'false');
  }} else {{
    params.set('agentic', document.getElementById('flag-agentic').checked ? 'true' : 'false');
    params.set('refine_drops', document.getElementById('flag-refine-drops').checked ? 'true' : 'false');
    params.set('deep', document.getElementById('flag-deep').checked ? 'true' : 'false');
  }}
  const requestId = ++estimateRequestId;
  estimateDisplayEl.textContent = 'Checking history\\u2026';
  try {{
    const est = await fetchJson('/api/estimate?' + params.toString());
    if (requestId !== estimateRequestId) return;
    estimateDisplayEl.textContent = formatEstimate(est);
  }} catch (err) {{
    if (requestId !== estimateRequestId) return;
    estimateDisplayEl.textContent = '';
  }}
}}

// --- Device selection -------------------------------------------------
//
// Reads/writes the same ~/.djcues/config.json the CLI's `djcues auth
// device` does (via GET/POST /api/devices), not a separate dashboard-
// only setting -- one source of truth. Device applies to both propose/
// compare and beatgrid, so this control lives outside the tab-specific
// flag rows (see updateActionFlags()) and is never hidden.

let deviceProbes = [];  // cached from the one /api/devices GET on load;
// availability is real-time hardware state, doesn't change based on
// what the user picks, so this isn't re-fetched on every selection.

function updateDeviceStatus(configured) {{
  const entry = deviceProbes.find(d => d.device === configured);
  if (!entry || configured === 'auto' || entry.ok) {{
    deviceStatusEl.textContent = '';
    deviceStatusEl.className = 'meta small device-status';
    return;
  }}
  deviceStatusEl.textContent = '⚠ ' + entry.error;
  deviceStatusEl.className = 'meta small device-status warn';
}}

async function loadDevices() {{
  try {{
    const data = await fetchJson('/api/devices');
    deviceProbes = data.available;
    flagDeviceEl.value = data.configured;
    updateDeviceStatus(data.configured);
  }} catch (err) {{
    deviceStatusEl.textContent = '';
  }}
}}

flagDeviceEl.addEventListener('change', async () => {{
  const chosen = flagDeviceEl.value;
  updateDeviceStatus(chosen);
  try {{
    await fetchJson('/api/devices', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ device: chosen }}),
    }});
  }} catch (err) {{
    deviceStatusEl.textContent = '⚠ ' + friendlyErrorMessage(err);
    deviceStatusEl.className = 'meta small device-status warn';
  }}
}});

runBtn.addEventListener('click', async () => {{
  if (!currentTrackId) return;

  let body;
  if (currentAction === 'beatgrid') {{
    body = {{
      kind: 'beatgrid',
      deep: document.getElementById('flag-beatgrid-deep').checked,
      device: document.getElementById('flag-device').value,
      no_cache: document.getElementById('flag-beatgrid-no-cache').checked,
      tolerance_ms: parseFloat(document.getElementById('flag-tolerance').value) || 30.0,
    }};
  }} else {{
    const deepChecked = document.getElementById('flag-deep').checked;
    const refineDropsChecked = document.getElementById('flag-refine-drops').checked;
    if (deepChecked && !refineDropsChecked) {{
      jobStatusEl.textContent = 'Deep only applies with Refine Drops.';
      jobStatusEl.className = 'job-status error';
      resultsPanelEl.classList.remove('hidden');
      return;
    }}
    body = {{
      kind: currentAction,
      agentic: document.getElementById('flag-agentic').checked,
      provider: document.getElementById('flag-provider').value.trim() || null,
      model: document.getElementById('flag-model').value.trim() || null,
      skip_critic: document.getElementById('flag-skip-critic').checked,
      refine_drops: refineDropsChecked,
      deep: deepChecked,
      device: document.getElementById('flag-device').value,
      offset_bars: parseInt(document.getElementById('flag-offset').value, 10) || {default_offset},
      loop_bars: parseInt(document.getElementById('flag-loop-bars').value, 10) || {default_loop_bars},
      no_cache: document.getElementById('flag-no-cache').checked,
    }};
  }}

  runBtn.disabled = true;
  resultsPanelEl.classList.remove('hidden');
  jobStatusEl.textContent = 'Starting&hellip;';
  jobStatusEl.className = 'job-status info';
  jobOutputWrapEl.classList.add('hidden');
  jobHtmlFragmentEl.innerHTML = '';

  try {{
    const data = await fetchJson('/api/tracks/' + encodeURIComponent(currentTrackId) + '/jobs', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(body),
    }});
    pollJob(data.job_id);
  }} catch (err) {{
    jobStatusEl.textContent = 'Error: ' + friendlyErrorMessage(err);
    jobStatusEl.className = 'job-status error';
    runBtn.disabled = false;
  }}
}});

function stopPolling() {{
  if (pollTimer) {{ clearTimeout(pollTimer); pollTimer = null; }}
}}

function pollJob(jobId) {{
  stopPolling();
  jobStartedAt = Date.now();
  const tick = async () => {{
    try {{
      const job = await fetchJson('/api/jobs/' + jobId);
      if (job.status === 'running') {{
        const elapsed = Math.round((Date.now() - jobStartedAt) / 1000);
        jobStatusEl.textContent = 'Running&hellip; (' + elapsed + 's)';
        jobStatusEl.className = 'job-status info';
        pollTimer = setTimeout(tick, 2000);
        return;
      }}
      runBtn.disabled = false;
      if (job.status === 'error') {{
        jobStatusEl.textContent = 'Error: ' + job.error;
        jobStatusEl.className = 'job-status error';
      }} else {{
        const cacheNote = (job.cache && job.cache.hits > 0)
          ? ' \\u00b7 cached (analyzed earlier, no recompute)'
          : ' \\u00b7 computed in ' + job.elapsed_seconds + 's';
        const costNote = job.cost_usd ? ' \\u00b7 $' + job.cost_usd.toFixed(4) : '';
        jobStatusEl.textContent = 'Done' + cacheNote + costNote;
        jobStatusEl.className = 'job-status success';
      }}
      if (job.output_text) {{
        jobOutputEl.textContent = job.output_text;
        jobOutputWrapEl.classList.remove('hidden');
      }}
      if (job.html_fragment) {{
        jobHtmlFragmentEl.innerHTML = job.html_fragment;
      }}
    }} catch (err) {{
      jobStatusEl.textContent = 'Error: ' + friendlyErrorMessage(err);
      jobStatusEl.className = 'job-status error';
      runBtn.disabled = false;
    }}
  }};
  tick();
}}

// --- Open in Viz / Review -------------------------------------------

openVizBtn.addEventListener('click', () => launchTool('viz'));
openReviewBtn.addEventListener('click', () => launchTool('review'));

async function launchTool(tool) {{
  if (!currentTrackId || !currentPlaylistId) return;
  const btn = tool === 'viz' ? openVizBtn : openReviewBtn;
  const orig = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Opening&hellip;';
  try {{
    await fetchJson('/api/tracks/' + encodeURIComponent(currentTrackId) + '/launch/' + tool, {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{
        playlist_id: currentPlaylistId,
        agentic: document.getElementById('launch-agentic').checked,
        refine_drops: document.getElementById('launch-refine-drops').checked,
        deep: document.getElementById('launch-deep').checked,
      }}),
    }});
    btn.textContent = 'Opened!';
  }} catch (err) {{
    btn.textContent = 'Error';
    jobStatusEl.textContent = 'Error: ' + friendlyErrorMessage(err);
    jobStatusEl.className = 'job-status error';
    resultsPanelEl.classList.remove('hidden');
  }} finally {{
    setTimeout(() => {{ btn.textContent = orig; btn.disabled = false; }}, 1500);
  }}
}}

// --- Playlist membership (move/add/remove) ---------------------------
//
// djcues's first dashboard controls that write to the database rather
// than just reading it. All three routes go through the same
// rekordbox-must-be-closed / auto-backup safety rules as the CLI's
// `djcues playlist` commands -- see writer.py. position is threaded
// through from the track's own track_no (already present in
// /api/playlists/<id>/tracks responses) rather than adding a new field
// -- the same disambiguation mechanism the CLI's --position uses, not
// a second one, for the rare case of the same track appearing more
// than once in one playlist.

function populatePlaylistDestSelect() {{
  playlistDestSelectEl.innerHTML = '';
  allPlaylistsFlat
    .filter(p => p.id !== currentPlaylistId)
    .forEach(p => {{
      const opt = document.createElement('option');
      opt.value = p.id;
      opt.textContent = p.name;
      playlistDestSelectEl.appendChild(opt);
    }});
}}

function currentTrackNo() {{
  const t = currentTracks.find(t => t.id === currentTrackId);
  return t ? t.track_no : null;
}}

async function runPlaylistAction(url, body, confirmMessage, successMessage) {{
  if (confirmMessage && !window.confirm(confirmMessage)) return false;
  playlistActionStatusEl.textContent = 'Working\\u2026';
  playlistActionStatusEl.className = 'job-status info';
  playlistMoveBtn.disabled = true;
  playlistAddBtn.disabled = true;
  playlistRemoveBtn.disabled = true;
  try {{
    await fetchJson(url, {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(body),
    }});
    playlistActionStatusEl.textContent = successMessage;
    playlistActionStatusEl.className = 'job-status success';
    return true;
  }} catch (err) {{
    playlistActionStatusEl.textContent = 'Error: ' + friendlyErrorMessage(err);
    playlistActionStatusEl.className = 'job-status error';
    return false;
  }} finally {{
    playlistAddBtn.disabled = false;
    playlistMoveBtn.disabled = !currentPlaylistId;
    playlistRemoveBtn.disabled = !currentPlaylistId;
  }}
}}

playlistMoveBtn.addEventListener('click', async () => {{
  if (!currentTrackId || !currentPlaylistId || !playlistDestSelectEl.value) return;
  const destId = playlistDestSelectEl.value;
  const destName = playlistDestSelectEl.options[playlistDestSelectEl.selectedIndex].textContent;
  const ok = await runPlaylistAction(
    '/api/playlists/' + encodeURIComponent(currentPlaylistId) + '/tracks/' + encodeURIComponent(currentTrackId) + '/move',
    {{ dest_playlist_id: destId, position: currentTrackNo() }},
    'Move this track to "' + destName + '"?',
    'Moved.',
  );
  if (ok) history.back();
}});

playlistAddBtn.addEventListener('click', async () => {{
  if (!currentTrackId || !playlistDestSelectEl.value) return;
  const destId = playlistDestSelectEl.value;
  const destName = playlistDestSelectEl.options[playlistDestSelectEl.selectedIndex].textContent;
  await runPlaylistAction(
    '/api/playlists/' + encodeURIComponent(destId) + '/tracks',
    {{ content_id: currentTrackId }},
    null,
    'Added to "' + destName + '".',
  );
}});

playlistRemoveBtn.addEventListener('click', async () => {{
  if (!currentTrackId || !currentPlaylistId) return;
  const ok = await runPlaylistAction(
    '/api/playlists/' + encodeURIComponent(currentPlaylistId) + '/tracks/' + encodeURIComponent(currentTrackId) + '/remove',
    {{ position: currentTrackNo() }},
    'Remove this track from "' + currentPlaylistNameEl.textContent + '"?',
    'Removed.',
  );
  if (ok) history.back();
}});

// --- Harmonic mixing suggestions -------------------------------------
//
// Unlike Propose/Compare/Beatgrid, this needs no async job-queue/polling
// machinery -- it's a fast, synchronous, in-memory computation over
// already-cheap TrackSummary lists (no ANLZ/audio/LLM I/O), so a plain
// fetchJson() call, auto-triggered on track load and re-triggered on
// control change, mirrors flagDeviceEl's own change-triggered refresh.

async function loadSuggestions() {{
  if (!currentTrackId || !currentPlaylistId) {{
    suggestionsListEl.innerHTML = '';
    return;
  }}
  const isLibrary = suggestLibraryEl.checked;
  const params = new URLSearchParams({{
    playlist_id: currentPlaylistId,
    library: isLibrary ? 'true' : 'false',
    half_double: suggestHalfDoubleEl.checked ? 'true' : 'false',
    bpm_tolerance: suggestBpmToleranceEl.value || '6',
  }});
  suggestionsListEl.innerHTML = '<p class="meta small">Loading&hellip;</p>';
  try {{
    const data = await fetchJson(
      '/api/tracks/' + encodeURIComponent(currentTrackId) + '/suggestions?' + params.toString()
    );
    renderSuggestions(data, isLibrary);
  }} catch (err) {{
    suggestionsListEl.innerHTML = '';
    const p = document.createElement('p');
    p.className = 'meta small';
    p.textContent = 'Error: ' + friendlyErrorMessage(err);
    suggestionsListEl.appendChild(p);
  }}
}}

function renderSuggestions(data, isLibrary) {{
  suggestionsListEl.innerHTML = '';
  if (data.suggestions.length === 0) {{
    suggestionsListEl.innerHTML = '<p class="meta small">No compatible tracks found.</p>';
  }}
  data.suggestions.forEach(s => {{
    const row = document.createElement('div');
    row.className = 'suggestion-row';

    const titleSpan = document.createElement('span');
    titleSpan.className = 'suggestion-title';
    titleSpan.textContent = s.track.title;

    const artistSpan = document.createElement('span');
    artistSpan.className = 'suggestion-artist';
    artistSpan.textContent = s.track.artist;

    const keySpan = document.createElement('span');
    keySpan.className = 'suggestion-key';
    keySpan.textContent = s.track.key || '';

    const bpmSpan = document.createElement('span');
    bpmSpan.className = 'suggestion-bpm';
    bpmSpan.textContent = s.track.bpm.toFixed(1) + ' BPM';

    const relationSpan = document.createElement('span');
    relationSpan.className = 'suggestion-relation';
    relationSpan.textContent = s.key_relation_label + ' \\u00b7 ' + s.bpm_relation.label;

    row.appendChild(titleSpan);
    row.appendChild(artistSpan);
    row.appendChild(keySpan);
    row.appendChild(bpmSpan);
    row.appendChild(relationSpan);

    // A --library match may not belong to currentPlaylistId -- navigate
    // with playlistId: null in that case; renderTrackDetail() already
    // handles that gracefully (disables Move/Remove, skips the stale-
    // list refetch), no special-casing needed here beyond passing it.
    row.addEventListener('click', () => navigateTo({{
      view: 'track',
      playlistId: isLibrary ? null : currentPlaylistId,
      playlistName: isLibrary ? null : currentPlaylistNameEl.textContent,
      trackId: s.track.id,
      trackTitle: s.track.title,
    }}, 'push'));

    suggestionsListEl.appendChild(row);
  }});

  const excluded = data.excluded_summary;
  const excludedTotal = excluded.no_key + excluded.non_camelot_key + excluded.encrypted_metadata;
  if (excludedTotal > 0) {{
    const note = document.createElement('p');
    note.className = 'meta small';
    const parts = [excluded.no_key + ' no-key', excluded.non_camelot_key + ' non-Camelot key'];
    if (excluded.encrypted_metadata) parts.push(excluded.encrypted_metadata + ' streaming-linked');
    note.textContent = '(excluded: ' + parts.join(', ') + ')';
    suggestionsListEl.appendChild(note);
  }}
}}

suggestLibraryEl.addEventListener('change', loadSuggestions);
suggestHalfDoubleEl.addEventListener('change', loadSuggestions);
suggestBpmToleranceEl.addEventListener('change', loadSuggestions);

// --- BPM/Key data-quality audit ---------------------------------------
//
// A library/playlist-wide report, not a per-track view -- a fourth
// client-side view (sibling to landing/playlist/track), not part of the
// track-detail panel. Same fast, synchronous, change-triggered-refresh
// pattern as harmonic suggestions above (no job-queue/polling needed).

function renderAudit(playlistId, playlistName, library) {{
  auditPlaylistId = playlistId;
  auditPlaylistName = playlistName;
  trackListPanelEl.classList.add('hidden');
  trackDetailPanelEl.classList.add('hidden');
  auditPanelEl.classList.remove('hidden');
  flowPanelEl.classList.add('hidden');
  clashPanelEl.classList.add('hidden');
  transitionPanelEl.classList.add('hidden');
  if (!playlistId) {{
    // Reached via the library-only entry point -- nothing else to
    // scope to, so force it and don't let the user uncheck it.
    auditLibraryEl.checked = true;
    auditLibraryEl.disabled = true;
  }} else {{
    auditLibraryEl.checked = !!library;
    auditLibraryEl.disabled = false;
  }}
  updateAuditScopeTitle();
  stopPolling();
  stopFlowPolling();
  stopClashPolling();
  stopTransitionPolling();
  document.title = 'djcues \\u2014 Audit';
  loadAuditFindings();
}}

function updateAuditScopeTitle() {{
  auditScopeTitleEl.textContent = auditLibraryEl.checked
    ? 'Audit: whole library'
    : 'Audit: ' + (auditPlaylistName || auditPlaylistId);
}}

async function loadAuditFindings() {{
  const isLibrary = auditLibraryEl.checked;
  const params = new URLSearchParams({{
    library: isLibrary ? 'true' : 'false',
    half_double: auditHalfDoubleEl.checked ? 'true' : 'false',
    bpm_tolerance: auditBpmToleranceEl.value || '6',
  }});
  if (!isLibrary && auditPlaylistId) params.set('playlist_id', auditPlaylistId);
  auditSummaryEl.textContent = '';
  auditFindingsListEl.innerHTML = '<p class="meta small">Loading&hellip;</p>';
  auditUnusableKeysListEl.innerHTML = '';
  try {{
    const data = await fetchJson('/api/audit?' + params.toString());
    renderAuditFindings(data, isLibrary);
  }} catch (err) {{
    auditFindingsListEl.innerHTML = '';
    const p = document.createElement('p');
    p.className = 'meta small';
    p.textContent = 'Error: ' + friendlyErrorMessage(err);
    auditFindingsListEl.appendChild(p);
  }}
}}

function renderAuditFindings(data, isLibrary) {{
  auditSummaryEl.textContent = data.scanned + ' scanned, ' + data.comment_hint_count + ' had a comment hint';
  auditFindingsListEl.innerHTML = '';
  auditUnusableKeysListEl.innerHTML = '';

  if (data.findings.length === 0) {{
    auditFindingsListEl.innerHTML = '<p class="meta small">No comment-hint disagreements found.</p>';
  }}
  data.findings.forEach(f => {{
    const row = document.createElement('div');
    row.className = 'suggestion-row';

    const titleSpan = document.createElement('span');
    titleSpan.className = 'suggestion-title';
    titleSpan.textContent = f.track.title;

    const artistSpan = document.createElement('span');
    artistSpan.className = 'suggestion-artist';
    artistSpan.textContent = f.track.artist;

    const detailSpan = document.createElement('span');
    detailSpan.className = 'suggestion-relation';
    const parts = [];
    if (f.bpm_mismatch) {{
      parts.push('BPM: tag ' + f.actual_bpm.toFixed(1) + ' vs comment ' + f.comment_bpm.toFixed(1));
    }}
    if (f.key_mismatch) {{
      parts.push('Key: tag ' + (f.actual_key || '(none)') + ' vs comment ' + f.comment_key);
    }}
    detailSpan.textContent = parts.join(' \\u00b7 ');

    row.appendChild(titleSpan);
    row.appendChild(artistSpan);
    row.appendChild(detailSpan);

    // A --library finding may not belong to auditPlaylistId -- navigate
    // with playlistId: null in that case, same handling harmonic
    // suggestions' own library-scoped rows already use.
    row.addEventListener('click', () => navigateTo({{
      view: 'track',
      playlistId: isLibrary ? null : auditPlaylistId,
      playlistName: isLibrary ? null : auditPlaylistName,
      trackId: f.track.id,
      trackTitle: f.track.title,
    }}, 'push'));

    auditFindingsListEl.appendChild(row);
  }});

  const u = data.unusable_keys_summary;
  const uTotal = u.no_key + u.non_camelot_key + u.encrypted_metadata;
  if (uTotal > 0) {{
    const note = document.createElement('p');
    note.className = 'meta small';
    const parts = [u.no_key + ' no-key', u.non_camelot_key + ' non-Camelot key'];
    if (u.encrypted_metadata) parts.push(u.encrypted_metadata + ' streaming-linked');
    note.textContent = '(unusable keys: ' + parts.join(', ') + ')';
    auditUnusableKeysListEl.appendChild(note);
  }}
}}

auditLibraryEl.addEventListener('change', () => {{ updateAuditScopeTitle(); loadAuditFindings(); }});
auditHalfDoubleEl.addEventListener('change', loadAuditFindings);
auditBpmToleranceEl.addEventListener('change', loadAuditFindings);

// Entry points into the audit view.
auditLibraryBtn.addEventListener('click', () => navigateTo({{ view: 'audit', playlistId: null, library: true }}, 'push'));
auditPlaylistBtn.addEventListener('click', () => navigateTo(
  {{ view: 'audit', playlistId: currentPlaylistId, playlistName: currentPlaylistNameEl.textContent, library: false }}, 'push'
));
// Same idiom as #back-to-list -- history.back() so Back/Forward and this
// button can never disagree about where "back" goes.
auditBackBtn.addEventListener('click', () => history.back());

// --- Energy flow (set ordering) ------------------------------------
//
// Its own top-level view, like audit -- covers a whole playlist, not
// one reference track. Unlike suggest/audit's synchronous GETs, this
// runs as a background job (POST to start, poll GET /api/jobs/<id>):
// ordering needs real per-track phrase/waveform data, too slow to
// load synchronously through the dashboard's shared browsing
// connection (see _run_flow_job's docstring in server.py). Can't
// reuse pollJob/stopPolling directly -- those manipulate track-detail-
// panel-specific elements -- so this is a sibling with the same
// semantics (2s poll interval, running/error/done) targeting
// #flow-panel's own elements instead.

function renderFlow(playlistId, playlistName) {{
  flowPlaylistId = playlistId;
  flowPlaylistName = playlistName;
  trackListPanelEl.classList.add('hidden');
  trackDetailPanelEl.classList.add('hidden');
  auditPanelEl.classList.add('hidden');
  flowPanelEl.classList.remove('hidden');
  clashPanelEl.classList.add('hidden');
  transitionPanelEl.classList.add('hidden');
  flowScopeTitleEl.textContent = 'Energy Flow: ' + (playlistName || playlistId);
  flowUnscoredSummaryEl.textContent = '';
  stopPolling();
  stopFlowPolling();
  stopClashPolling();
  stopTransitionPolling();
  document.title = 'djcues \\u2014 Energy Flow';
  // Restore the last completed result for this SAME playlist instead of
  // showing a blank panel -- e.g. after clicking into a track and back.
  // A different playlist (or no cached result yet) starts blank as before.
  if (lastFlowResultPlaylistId === playlistId && lastFlowResult) {{
    flowStatusEl.textContent = 'Done \\u00b7 computed in ' + lastFlowElapsedSeconds + 's';
    flowStatusEl.className = 'job-status success';
    renderFlowResult(lastFlowResult);
  }} else {{
    flowStatusEl.textContent = '';
    flowStatusEl.className = 'job-status';
    flowResultsListEl.innerHTML = '';
    flowResultsHeaderEl.classList.add('hidden');
    flowApplyBtn.classList.add('hidden');
    flowApplyStatusEl.textContent = '';
    flowApplyStatusEl.className = 'job-status';
  }}
}}

flowRunBtn.addEventListener('click', async () => {{
  if (!flowPlaylistId) return;
  const body = {{ cooldown_fraction: parseFloat(flowCooldownFractionEl.value) || 0.15 }};
  flowRunBtn.disabled = true;
  flowStatusEl.textContent = 'Starting&hellip;';
  flowStatusEl.className = 'job-status info';
  flowResultsListEl.innerHTML = '';
  flowUnscoredSummaryEl.textContent = '';
  flowApplyBtn.classList.add('hidden');
  flowApplyStatusEl.textContent = '';
  flowApplyStatusEl.className = 'job-status';
  try {{
    const data = await fetchJson('/api/playlists/' + encodeURIComponent(flowPlaylistId) + '/flow-jobs', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(body),
    }});
    pollFlowJob(data.job_id);
  }} catch (err) {{
    flowStatusEl.textContent = 'Error: ' + friendlyErrorMessage(err);
    flowStatusEl.className = 'job-status error';
    flowRunBtn.disabled = false;
  }}
}});

function stopFlowPolling() {{
  if (flowPollTimer) {{ clearTimeout(flowPollTimer); flowPollTimer = null; }}
}}

function pollFlowJob(jobId) {{
  stopFlowPolling();
  flowJobStartedAt = Date.now();
  const tick = async () => {{
    try {{
      const job = await fetchJson('/api/jobs/' + jobId);
      if (job.status === 'running') {{
        const elapsed = Math.round((Date.now() - flowJobStartedAt) / 1000);
        flowStatusEl.textContent = 'Running&hellip; (' + elapsed + 's)';
        flowStatusEl.className = 'job-status info';
        flowPollTimer = setTimeout(tick, 2000);
        return;
      }}
      flowRunBtn.disabled = false;
      if (job.status === 'error') {{
        flowStatusEl.textContent = 'Error: ' + job.error;
        flowStatusEl.className = 'job-status error';
      }} else {{
        flowStatusEl.textContent = 'Done \\u00b7 computed in ' + job.elapsed_seconds + 's';
        flowStatusEl.className = 'job-status success';
        renderFlowResult(job.result);
        lastFlowResult = job.result;
        lastFlowResultPlaylistId = flowPlaylistId;
        lastFlowElapsedSeconds = job.elapsed_seconds;
      }}
    }} catch (err) {{
      flowStatusEl.textContent = 'Error: ' + friendlyErrorMessage(err);
      flowStatusEl.className = 'job-status error';
      flowRunBtn.disabled = false;
    }}
  }};
  tick();
}}

function renderFlowResult(result) {{
  flowResultsListEl.innerHTML = '';
  flowResultsHeaderEl.classList.toggle('hidden', result.ordered_tracks.length === 0);
  flowApplyBtn.classList.toggle('hidden', result.ordered_tracks.length === 0);
  flowApplyBtn.disabled = false;
  result.ordered_tracks.forEach((t, i) => {{
    if (i === result.cooldown_start_index) {{
      const divider = document.createElement('p');
      divider.className = 'meta small';
      divider.textContent = '\\u2193 cooldown begins \\u2193';
      flowResultsListEl.appendChild(divider);
    }}
    const row = document.createElement('div');
    row.className = 'suggestion-row';

    const posSpan = document.createElement('span');
    posSpan.className = 'suggestion-bpm';
    posSpan.textContent = (t.current_position != null ? t.current_position : '?') + ' \\u2192 ' + (i + 1);

    const titleSpan = document.createElement('span');
    titleSpan.className = 'suggestion-title';
    titleSpan.textContent = t.title;

    const artistSpan = document.createElement('span');
    artistSpan.className = 'suggestion-artist';
    artistSpan.textContent = t.artist;

    const keySpan = document.createElement('span');
    keySpan.className = 'suggestion-key';
    keySpan.textContent = t.key || '';

    const energySpan = document.createElement('span');
    energySpan.className = 'suggestion-relation';
    energySpan.textContent = 'mean ' + t.mean_energy.toFixed(2) + ' \\u00b7 peak ' + t.peak_energy.toFixed(2);

    row.appendChild(posSpan);
    row.appendChild(titleSpan);
    row.appendChild(artistSpan);
    row.appendChild(keySpan);
    row.appendChild(energySpan);

    row.addEventListener('click', () => navigateTo({{
      view: 'track', playlistId: flowPlaylistId, playlistName: flowPlaylistName,
      trackId: t.id, trackTitle: t.title,
    }}, 'push'));

    flowResultsListEl.appendChild(row);
  }});

  const u = result.unscored_summary;
  if (u.no_phrase_data || u.no_waveform_data) {{
    flowUnscoredSummaryEl.textContent =
      '(skipped: ' + u.no_phrase_data + ' no phrase data, ' + u.no_waveform_data + ' no waveform data)';
  }}
}}

// Writes the just-computed order into the real playlist. Uses the
// already-rendered, in-memory lastFlowResult -- no server-side flow
// re-computation, since the user already ran a scan to see this order.
// A sibling of pollFlowJob, not a reuse of runPlaylistAction()'s own
// machinery -- that hardcodes track-detail-panel elements/status, the
// same reason renderFlow/pollFlowJob are already hand-written siblings
// of pollJob rather than reusing it. This write is fast (no per-track
// ANLZ I/O, just DjmdSongPlaylist.TrackNo updates) so it's a plain
// synchronous POST, not a job+poll -- same tier as the existing
// playlist add/remove/move controls, unlike flow/clash's own scans.
flowApplyBtn.addEventListener('click', async () => {{
  if (!flowPlaylistId || !lastFlowResult) return;
  const orderedIds = lastFlowResult.ordered_tracks.map(t => t.id)
    .concat(lastFlowResult.unscored.map(t => t.id));
  const trackCount = orderedIds.length;
  const confirmMsg = 'Apply this order to ' + trackCount + ' track(s) in "' +
    (flowPlaylistName || flowPlaylistId) + '"? This rewrites the real playlist order in Rekordbox.';
  if (!window.confirm(confirmMsg)) return;
  flowApplyBtn.disabled = true;
  flowApplyStatusEl.textContent = 'Applying\\u2026';
  flowApplyStatusEl.className = 'job-status info';
  try {{
    await fetchJson('/api/playlists/' + encodeURIComponent(flowPlaylistId) + '/reorder', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ ordered_content_ids: orderedIds }}),
    }});
    flowApplyStatusEl.textContent = 'Applied ' + trackCount + ' track(s) to Rekordbox.';
    flowApplyStatusEl.className = 'job-status success';
  }} catch (err) {{
    flowApplyStatusEl.textContent = 'Error: ' + friendlyErrorMessage(err);
    flowApplyStatusEl.className = 'job-status error';
    flowApplyBtn.disabled = false;
  }}
}});

// Entry point into the flow view -- always from a selected playlist
// (unlike audit, there's no library-wide entry point for this feature).
flowPlaylistBtn.addEventListener('click', () => navigateTo(
  {{ view: 'flow', playlistId: currentPlaylistId, playlistName: currentPlaylistNameEl.textContent }}, 'push'
));
flowBackBtn.addEventListener('click', () => history.back());

// --- Vocal clash detection --------------------------------------------
//
// Its own top-level view, like flow -- playlist-scoped only, no library
// mode (adjacency only means something within one ordered playlist).
// Same background-job reasoning as flow (see _run_clash_job's docstring
// in server.py): can't reuse pollJob/pollFlowJob -- those target
// different DOM elements -- so this is a third sibling with the same
// polling semantics (2s interval, running/error/done) targeting
// #clash-panel's own elements.

function renderClash(playlistId, playlistName) {{
  clashPlaylistId = playlistId;
  clashPlaylistName = playlistName;
  trackListPanelEl.classList.add('hidden');
  trackDetailPanelEl.classList.add('hidden');
  auditPanelEl.classList.add('hidden');
  flowPanelEl.classList.add('hidden');
  clashPanelEl.classList.remove('hidden');
  transitionPanelEl.classList.add('hidden');
  clashScopeTitleEl.textContent = 'Vocal Clash: ' + (playlistName || playlistId);
  clashUnscorableSummaryEl.textContent = '';
  stopPolling();
  stopFlowPolling();
  stopClashPolling();
  stopTransitionPolling();
  document.title = 'djcues \\u2014 Vocal Clash';
  // Same restore-on-return behavior as renderFlow's own cache, same reason.
  if (lastClashResultPlaylistId === playlistId && lastClashResult) {{
    clashStatusEl.textContent = 'Done \\u00b7 computed in ' + lastClashElapsedSeconds + 's';
    clashStatusEl.className = 'job-status success';
    renderClashResult(lastClashResult);
  }} else {{
    clashStatusEl.textContent = '';
    clashStatusEl.className = 'job-status';
    clashFindingsListEl.innerHTML = '';
  }}
}}

clashRunBtn.addEventListener('click', async () => {{
  if (!clashPlaylistId) return;
  const body = {{ min_vocal_region_ms: parseFloat(clashMinVocalRegionMsEl.value) || 2000 }};
  clashRunBtn.disabled = true;
  clashStatusEl.textContent = 'Starting&hellip;';
  clashStatusEl.className = 'job-status info';
  clashFindingsListEl.innerHTML = '';
  clashUnscorableSummaryEl.textContent = '';
  try {{
    const data = await fetchJson('/api/playlists/' + encodeURIComponent(clashPlaylistId) + '/clash-jobs', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(body),
    }});
    pollClashJob(data.job_id);
  }} catch (err) {{
    clashStatusEl.textContent = 'Error: ' + friendlyErrorMessage(err);
    clashStatusEl.className = 'job-status error';
    clashRunBtn.disabled = false;
  }}
}});

function stopClashPolling() {{
  if (clashPollTimer) {{ clearTimeout(clashPollTimer); clashPollTimer = null; }}
}}

function pollClashJob(jobId) {{
  stopClashPolling();
  clashJobStartedAt = Date.now();
  const tick = async () => {{
    try {{
      const job = await fetchJson('/api/jobs/' + jobId);
      if (job.status === 'running') {{
        const elapsed = Math.round((Date.now() - clashJobStartedAt) / 1000);
        clashStatusEl.textContent = 'Running&hellip; (' + elapsed + 's)';
        clashStatusEl.className = 'job-status info';
        clashPollTimer = setTimeout(tick, 2000);
        return;
      }}
      clashRunBtn.disabled = false;
      if (job.status === 'error') {{
        clashStatusEl.textContent = 'Error: ' + job.error;
        clashStatusEl.className = 'job-status error';
      }} else {{
        clashStatusEl.textContent = 'Done \\u00b7 computed in ' + job.elapsed_seconds + 's';
        clashStatusEl.className = 'job-status success';
        renderClashResult(job.result);
        lastClashResult = job.result;
        lastClashResultPlaylistId = clashPlaylistId;
        lastClashElapsedSeconds = job.elapsed_seconds;
      }}
    }} catch (err) {{
      clashStatusEl.textContent = 'Error: ' + friendlyErrorMessage(err);
      clashStatusEl.className = 'job-status error';
      clashRunBtn.disabled = false;
    }}
  }};
  tick();
}}

function renderClashResult(result) {{
  clashFindingsListEl.innerHTML = '';
  if (result.findings.length === 0) {{
    clashFindingsListEl.innerHTML = '<p class="meta small">No vocal clashes found.</p>';
  }}
  result.findings.forEach(f => {{
    const row = document.createElement('div');
    row.className = 'suggestion-row';

    const posSpan = document.createElement('span');
    posSpan.className = 'suggestion-bpm';
    posSpan.textContent = f.position_a + ' \\u2192 ' + f.position_b;

    const titleSpan = document.createElement('span');
    titleSpan.className = 'suggestion-title';
    titleSpan.textContent = f.track_a.title + ' \\u2192 ' + f.track_b.title;

    const artistSpan = document.createElement('span');
    artistSpan.className = 'suggestion-artist';
    artistSpan.textContent = f.track_a.artist + ' / ' + f.track_b.artist;

    const detailSpan = document.createElement('span');
    detailSpan.className = 'suggestion-relation';
    const aRegions = f.regions_a.map(r => formatDuration(r.start_ms) + '\\u2013' + formatDuration(r.end_ms)).join(', ');
    const bRegions = f.regions_b.map(r => formatDuration(r.start_ms) + '\\u2013' + formatDuration(r.end_ms)).join(', ');
    detailSpan.textContent = 'A tail ' + aRegions + ' \\u00b7 B head ' + bRegions;

    row.appendChild(posSpan);
    row.appendChild(titleSpan);
    row.appendChild(artistSpan);
    row.appendChild(detailSpan);

    row.addEventListener('click', () => navigateTo({{
      view: 'track', playlistId: clashPlaylistId, playlistName: clashPlaylistName,
      trackId: f.track_a.id, trackTitle: f.track_a.title,
    }}, 'push'));

    clashFindingsListEl.appendChild(row);
  }});

  const u = result.unscorable_summary;
  const total = u.no_phrase_data + u.no_vocal_data + u.no_intro_phrase + u.no_outro_phrase;
  if (total > 0) {{
    const parts = [];
    if (u.no_phrase_data) parts.push(u.no_phrase_data + ' no phrase data');
    if (u.no_vocal_data) parts.push(u.no_vocal_data + ' no vocal data');
    if (u.no_intro_phrase) parts.push(u.no_intro_phrase + ' no Intro phrase');
    if (u.no_outro_phrase) parts.push(u.no_outro_phrase + ' no Outro phrase');
    clashUnscorableSummaryEl.textContent = '(unscorable: ' + parts.join(', ') + ')';
  }}
}}

// Entry point into the clash view -- always from a selected playlist,
// same reasoning as flow's own entry point above.
clashPlaylistBtn.addEventListener('click', () => navigateTo(
  {{ view: 'clash', playlistId: currentPlaylistId, playlistName: currentPlaylistNameEl.textContent }}, 'push'
));
clashBackBtn.addEventListener('click', () => history.back());

// --- Transition point suggestions --------------------------------------
//
// Its own top-level view, like flow/clash -- playlist-scoped only, no
// library mode (adjacency only means something within one ordered
// playlist). Same background-job reasoning as flow/clash (see
// _run_transition_job's docstring in server.py): a fourth sibling of
// pollJob/pollFlowJob/pollClashJob with the same polling semantics (2s
// interval, running/error/done) targeting #transition-panel's own
// elements. Read-only, suggestion-only -- unlike flow, there is no
// "apply to Rekordbox" button here (see transition.py's own module
// docstring for why).

function renderTransition(playlistId, playlistName) {{
  transitionPlaylistId = playlistId;
  transitionPlaylistName = playlistName;
  trackListPanelEl.classList.add('hidden');
  trackDetailPanelEl.classList.add('hidden');
  auditPanelEl.classList.add('hidden');
  flowPanelEl.classList.add('hidden');
  clashPanelEl.classList.add('hidden');
  transitionPanelEl.classList.remove('hidden');
  transitionScopeTitleEl.textContent = 'Transitions: ' + (playlistName || playlistId);
  transitionUnscorableSummaryEl.textContent = '';
  stopPolling();
  stopFlowPolling();
  stopClashPolling();
  stopTransitionPolling();
  document.title = 'djcues \\u2014 Transitions';
  // Same restore-on-return behavior as renderFlow's/renderClash's own cache, same reason.
  if (lastTransitionResultPlaylistId === playlistId && lastTransitionResult) {{
    transitionStatusEl.textContent = 'Done \\u00b7 computed in ' + lastTransitionElapsedSeconds + 's';
    transitionStatusEl.className = 'job-status success';
    renderTransitionResult(lastTransitionResult);
  }} else {{
    transitionStatusEl.textContent = '';
    transitionStatusEl.className = 'job-status';
    transitionResultsListEl.innerHTML = '';
  }}
}}

transitionRunBtn.addEventListener('click', async () => {{
  if (!transitionPlaylistId) return;
  const body = {{ min_vocal_region_ms: parseFloat(transitionMinVocalRegionMsEl.value) || 2000 }};
  transitionRunBtn.disabled = true;
  transitionStatusEl.textContent = 'Starting&hellip;';
  transitionStatusEl.className = 'job-status info';
  transitionResultsListEl.innerHTML = '';
  transitionUnscorableSummaryEl.textContent = '';
  try {{
    const data = await fetchJson('/api/playlists/' + encodeURIComponent(transitionPlaylistId) + '/transition-jobs', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(body),
    }});
    pollTransitionJob(data.job_id);
  }} catch (err) {{
    transitionStatusEl.textContent = 'Error: ' + friendlyErrorMessage(err);
    transitionStatusEl.className = 'job-status error';
    transitionRunBtn.disabled = false;
  }}
}});

function stopTransitionPolling() {{
  if (transitionPollTimer) {{ clearTimeout(transitionPollTimer); transitionPollTimer = null; }}
}}

function pollTransitionJob(jobId) {{
  stopTransitionPolling();
  transitionJobStartedAt = Date.now();
  const tick = async () => {{
    try {{
      const job = await fetchJson('/api/jobs/' + jobId);
      if (job.status === 'running') {{
        const elapsed = Math.round((Date.now() - transitionJobStartedAt) / 1000);
        transitionStatusEl.textContent = 'Running&hellip; (' + elapsed + 's)';
        transitionStatusEl.className = 'job-status info';
        transitionPollTimer = setTimeout(tick, 2000);
        return;
      }}
      transitionRunBtn.disabled = false;
      if (job.status === 'error') {{
        transitionStatusEl.textContent = 'Error: ' + job.error;
        transitionStatusEl.className = 'job-status error';
      }} else {{
        transitionStatusEl.textContent = 'Done \\u00b7 computed in ' + job.elapsed_seconds + 's';
        transitionStatusEl.className = 'job-status success';
        renderTransitionResult(job.result);
        lastTransitionResult = job.result;
        lastTransitionResultPlaylistId = transitionPlaylistId;
        lastTransitionElapsedSeconds = job.elapsed_seconds;
      }}
    }} catch (err) {{
      transitionStatusEl.textContent = 'Error: ' + friendlyErrorMessage(err);
      transitionStatusEl.className = 'job-status error';
      transitionRunBtn.disabled = false;
    }}
  }};
  tick();
}}

function renderTransitionResult(result) {{
  transitionResultsListEl.innerHTML = '';
  if (result.suggestions.length === 0) {{
    transitionResultsListEl.innerHTML = '<p class="meta small">No transition suggestions.</p>';
  }}
  result.suggestions.forEach(s => {{
    const row = document.createElement('div');
    row.className = 'suggestion-row';

    const posSpan = document.createElement('span');
    posSpan.className = 'suggestion-bpm';
    posSpan.textContent = s.position_a + ' \\u2192 ' + s.position_b;

    const titleSpan = document.createElement('span');
    titleSpan.className = 'suggestion-title';
    titleSpan.textContent = s.track_a.title + ' \\u2192 ' + s.track_b.title;

    const artistSpan = document.createElement('span');
    artistSpan.className = 'suggestion-artist';
    artistSpan.textContent = s.track_a.artist + ' / ' + s.track_b.artist;

    const detailSpan = document.createElement('span');
    detailSpan.className = 'suggestion-relation';
    const detailParts = [
      formatDuration(s.mix_out_ms) + '\\u2192' + formatDuration(s.mix_in_ms),
      (s.overlap_ms / 1000).toFixed(1) + 's blend',
    ];
    if (s.key_relation_label) detailParts.push(s.key_relation_label);
    if (s.bpm_relation) detailParts.push(s.bpm_relation.label + ' ' + s.bpm_relation.pitch_shift_pct.toFixed(1) + '%');
    const hasVocalRisk = (s.vocal_regions_a && s.vocal_regions_a.length > 0) || (s.vocal_regions_b && s.vocal_regions_b.length > 0);
    if (hasVocalRisk) detailParts.push('vocal risk');
    detailSpan.textContent = detailParts.join(' \\u00b7 ');

    row.appendChild(posSpan);
    row.appendChild(titleSpan);
    row.appendChild(artistSpan);
    row.appendChild(detailSpan);

    row.addEventListener('click', () => navigateTo({{
      view: 'track', playlistId: transitionPlaylistId, playlistName: transitionPlaylistName,
      trackId: s.track_a.id, trackTitle: s.track_a.title,
    }}, 'push'));

    transitionResultsListEl.appendChild(row);
  }});

  const u = result.unscorable_summary;
  const total = u.no_phrase_data + u.no_intro_phrase + u.no_outro_phrase;
  if (total > 0) {{
    const parts = [];
    if (u.no_phrase_data) parts.push(u.no_phrase_data + ' no phrase data');
    if (u.no_intro_phrase) parts.push(u.no_intro_phrase + ' no Intro phrase');
    if (u.no_outro_phrase) parts.push(u.no_outro_phrase + ' no Outro phrase');
    transitionUnscorableSummaryEl.textContent = '(unscorable: ' + parts.join(', ') + ')';
  }}
}}

// Entry point into the transition view -- always from a selected
// playlist, same reasoning as flow's/clash's own entry points above.
transitionPlaylistBtn.addEventListener('click', () => navigateTo(
  {{ view: 'transition', playlistId: currentPlaylistId, playlistName: currentPlaylistNameEl.textContent }}, 'push'
));
transitionBackBtn.addEventListener('click', () => history.back());

// Called after INITIAL_PLAYLIST (server-embedded, from a
// `djcues dashboard "Playlist Name"` argument) is defined -- see the
// bottom of render_dashboard_html()'s <script> block.
async function initDashboard() {{
  await loadDevices();
  await loadPlaylists();

  const urlState = urlToState();
  if (urlState.view !== 'landing') {{
    // A real navigation already encoded in the URL (a bookmark, a
    // refresh, or the address bar edited by hand) -- trust it over
    // whatever the CLI's own playlist argument said.
    navigateTo(urlState, 'replace');
  }} else if (INITIAL_PLAYLIST) {{
    navigateTo({{ view: 'playlist', playlistId: INITIAL_PLAYLIST.id, playlistName: INITIAL_PLAYLIST.name }}, 'replace');
  }} else {{
    navigateTo({{ view: 'landing' }}, 'replace');
  }}
}}
"""
