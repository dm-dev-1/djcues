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
    </div>
    <div id="playlist-tree" class="playlist-tree">Loading&hellip;</div>
  </div>

  <div class="main">
    <div id="track-list-panel" class="panel">
      <div class="panel-header">
        <h2 id="current-playlist-name">Select a playlist</h2>
        <input id="track-search" type="text" placeholder="Filter tracks&hellip;" class="hidden">
      </div>
      <div id="track-list" class="track-list">
        <p class="meta">Pick a playlist on the left to see its tracks.</p>
      </div>
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

  .launch-section { margin-top: 16px; padding: 16px; border-top: 1px solid #2a2a3e; }
  .launch-buttons { display: flex; gap: 8px; margin-top: 8px; }

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

let currentPlaylistId = null;
let currentTracks = [];
let currentTrackId = null;
let currentAction = 'propose';
let pollTimer = null;
let jobStartedAt = null;
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

async function loadPlaylists() {{
  try {{
    const data = await fetchJson('/api/playlists');
    playlistTreeEl.innerHTML = '';
    data.tree.forEach(node => playlistTreeEl.appendChild(renderPlaylistNode(node, 0)));
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
  trackListEl.innerHTML = '<p class="meta">Pick a playlist on the left to see its tracks.</p>';
  trackListPanelEl.classList.remove('hidden');
  trackDetailPanelEl.classList.add('hidden');
  stopPolling();
  document.title = 'djcues \\u2014 Analysis dashboard';
}}

async function renderPlaylistTracks(playlistId, playlistName) {{
  currentPlaylistId = playlistId;
  currentTrackId = null;
  currentPlaylistNameEl.textContent = playlistName;
  trackSearchEl.value = '';
  trackSearchEl.classList.remove('hidden');
  trackListEl.innerHTML = 'Loading&hellip;';
  trackListPanelEl.classList.remove('hidden');
  trackDetailPanelEl.classList.add('hidden');
  stopPolling();
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
  resultsPanelEl.classList.add('hidden');
  jobOutputWrapEl.classList.add('hidden');
  jobHtmlFragmentEl.innerHTML = '';
  stopPolling();
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

runBtn.addEventListener('click', async () => {{
  if (!currentTrackId) return;

  let body;
  if (currentAction === 'beatgrid') {{
    body = {{
      kind: 'beatgrid',
      deep: document.getElementById('flag-beatgrid-deep').checked,
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

// Called after INITIAL_PLAYLIST (server-embedded, from a
// `djcues dashboard "Playlist Name"` argument) is defined -- see the
// bottom of render_dashboard_html()'s <script> block.
async function initDashboard() {{
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
