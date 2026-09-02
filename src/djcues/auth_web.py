"""Browser-based BYOK setup wizard — an alternative to `djcues auth set`'s
terminal prompt for entering an API key and picking a model.

Served entirely from a local-only HTTP server (see server.py's
AuthSetupServer). The page never embeds a server URL: every request it
makes is a same-origin relative fetch, since the page is only ever
loaded from that same local server. The API key is sent once, in a POST
body (never a query string), directly to 127.0.0.1 and nowhere else --
the server stores it via keyring exactly like the CLI flow does, and it
is never written to this HTML, logged, or echoed back in any response.
"""

from __future__ import annotations


def render_auth_setup_html() -> str:
    """Render the standalone setup-wizard page."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>djcues &mdash; Configure agentic analysis</title>
<style>{_CSS}</style>
</head>
<body>
<div class="card">
  <h1>Configure agentic analysis</h1>
  <p class="meta">
    Your API key is sent once to this local server (127.0.0.1) and stored in
    your OS credential store &mdash; Windows Credential Manager, macOS
    Keychain, or the Linux Secret Service. It is never written to a file,
    never logged, and never leaves this machine.
  </p>

  <div class="field">
    <label for="provider">Provider</label>
    <select id="provider">
      <option value="anthropic">Anthropic (Claude)</option>
      <option value="gemini">Google Gemini</option>
    </select>
  </div>

  <div class="field">
    <label for="api-key">API key</label>
    <div class="key-row">
      <input id="api-key" type="password" autocomplete="off" spellcheck="false"
             placeholder="Paste your API key">
      <button id="toggle-key" class="btn btn-ghost" type="button">Show</button>
    </div>
  </div>

  <button id="fetch-models" class="btn btn-primary" type="button">Fetch models</button>

  <div id="models-section" class="field hidden">
    <label for="model">Model</label>
    <select id="model"></select>
    <p id="model-price" class="meta small"></p>
  </div>

  <div id="status" class="status"></div>

  <button id="save" class="btn btn-primary hidden" type="button">Save API key &amp; model</button>

  <div id="next-steps" class="next-steps hidden">
    <p class="meta small">Next, try it on a playlist &mdash; swap in your own playlist name:</p>
    <div class="cmd-row">
      <code id="cmd-estimate">djcues propose "your playlist" --all --agentic --estimate-only</code>
      <button class="btn btn-copy" data-target="cmd-estimate" type="button">Copy</button>
    </div>
    <div class="cmd-row">
      <code id="cmd-review">djcues review "your playlist" --all --agentic</code>
      <button class="btn btn-copy" data-target="cmd-review" type="button">Copy</button>
    </div>
  </div>
</div>

<script>{_JS}</script>
</body>
</html>"""


_CSS = """
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: #1a1a2e;
    color: #eee;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    display: flex;
    justify-content: center;
    padding: 48px 16px;
  }
  .card {
    background: #111122;
    border: 1px solid #2a2a3e;
    border-radius: 8px;
    padding: 28px 32px;
    max-width: 520px;
    width: 100%;
  }
  h1 { font-size: 1.3rem; margin: 0 0 12px; }
  .meta { color: #888; font-size: 0.85rem; line-height: 1.5; margin: 0 0 20px; }
  .meta.small { margin: 6px 0 0; font-size: 0.8rem; }
  .field { margin-bottom: 18px; }
  label { display: block; font-size: 0.85rem; color: #aaa; margin-bottom: 6px; }
  select, input {
    width: 100%;
    background: #1e1e3a;
    border: 1px solid #2a2a3e;
    color: #eee;
    border-radius: 4px;
    padding: 8px 10px;
    font-size: 0.9rem;
  }
  .key-row { display: flex; gap: 8px; }
  .key-row input { flex: 1; }
  .btn {
    border: none;
    border-radius: 4px;
    padding: 9px 16px;
    font-size: 0.9rem;
    cursor: pointer;
    font-weight: 600;
  }
  .btn:disabled { opacity: 0.5; cursor: default; }
  .btn-primary { background: #17a2b8; color: #fff; width: 100%; margin-top: 4px; }
  .btn-ghost { background: #2a2a3e; color: #ccc; padding: 8px 12px; }
  .hidden { display: none; }
  .status { font-size: 0.85rem; margin: 14px 0; min-height: 1.2em; }
  .status.error { color: #dc3545; }
  .status.success { color: #28a745; }
  .status.info { color: #888; }
  .next-steps { margin-top: 8px; padding-top: 18px; border-top: 1px solid #2a2a3e; }
  .cmd-row {
    display: flex;
    align-items: center;
    gap: 6px;
    background: #1e1e3a;
    border: 1px solid #2a2a3e;
    border-radius: 4px;
    padding: 6px 8px;
    margin-bottom: 8px;
  }
  .cmd-row code {
    flex: 1;
    color: #aaa;
    font-family: monospace;
    font-size: 0.78rem;
    overflow-x: auto;
    white-space: nowrap;
  }
  .btn-copy { background: #444; color: #ddd; padding: 4px 10px; font-size: 0.75rem; flex-shrink: 0; }
"""

_JS = """
'use strict';

const providerEl = document.getElementById('provider');
const apiKeyEl = document.getElementById('api-key');
const toggleKeyBtn = document.getElementById('toggle-key');
const fetchModelsBtn = document.getElementById('fetch-models');
const modelsSection = document.getElementById('models-section');
const modelEl = document.getElementById('model');
const modelPriceEl = document.getElementById('model-price');
const statusEl = document.getElementById('status');
const saveBtn = document.getElementById('save');
const nextStepsEl = document.getElementById('next-steps');

document.querySelectorAll('.btn-copy').forEach(function(btn) {
  btn.addEventListener('click', function() {
    const target = document.getElementById(btn.getAttribute('data-target'));
    navigator.clipboard.writeText(target.textContent).then(function() {
      const orig = btn.textContent;
      btn.textContent = 'Copied!';
      setTimeout(function() { btn.textContent = orig; }, 1500);
    });
  });
});

let lastModels = [];

function setStatus(text, kind) {
  statusEl.textContent = text;
  statusEl.className = 'status' + (kind ? ' ' + kind : '');
}

toggleKeyBtn.addEventListener('click', function() {
  const showing = apiKeyEl.type === 'text';
  apiKeyEl.type = showing ? 'password' : 'text';
  toggleKeyBtn.textContent = showing ? 'Show' : 'Hide';
});

function updatePriceLabel() {
  const chosen = lastModels.find(m => m.id === modelEl.value);
  if (!chosen) { modelPriceEl.textContent = ''; return; }
  if (chosen.price_input_per_million == null) {
    modelPriceEl.textContent = 'Pricing not in the local table -- check the provider\\'s site.';
  } else {
    modelPriceEl.textContent = '$' + chosen.price_input_per_million.toFixed(2) + ' / $' +
      chosen.price_output_per_million.toFixed(2) + ' per 1M tokens (in/out)';
  }
}
modelEl.addEventListener('change', updatePriceLabel);

// A plain network-level fetch failure (server unreachable, connection
// refused, ...) throws a generic TypeError with an unhelpful browser
// message ("Failed to fetch" in Chrome, "NetworkError..." in Firefox,
// "Load failed" in Safari) that gives no hint why -- the most common
// real cause here is the local server's own idle timeout closing the
// port out from under an open tab, so say that instead of parroting
// the raw browser error.
function friendlyErrorMessage(err) {
  if (err instanceof TypeError) {
    return 'Could not reach the local djcues server. It may have shut down ' +
      '(idle timeout) -- run "djcues auth web" again in your terminal.';
  }
  return err.message;
}

fetchModelsBtn.addEventListener('click', async function() {
  const provider = providerEl.value;
  const apiKey = apiKeyEl.value.trim();
  if (!apiKey) { setStatus('Enter an API key first.', 'error'); return; }

  fetchModelsBtn.disabled = true;
  saveBtn.classList.add('hidden');
  modelsSection.classList.add('hidden');
  setStatus('Fetching available models...', 'info');

  try {
    const res = await fetch('/models', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider: provider, api_key: apiKey })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || ('server returned ' + res.status));

    lastModels = data.models;
    modelEl.innerHTML = '';
    data.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.display_name + (m.recommended ? ' (recommended, lightweight)' : '');
      modelEl.appendChild(opt);
    });
    if (data.default_model) modelEl.value = data.default_model;
    updatePriceLabel();

    modelsSection.classList.remove('hidden');
    saveBtn.classList.remove('hidden');
    setStatus('Models loaded. Pick one and save.', 'success');
  } catch (err) {
    setStatus('Error: ' + friendlyErrorMessage(err), 'error');
  } finally {
    fetchModelsBtn.disabled = false;
  }
});

saveBtn.addEventListener('click', async function() {
  const provider = providerEl.value;
  const apiKey = apiKeyEl.value.trim();
  const model = modelEl.value;
  if (!apiKey || !model) { setStatus('Fetch and choose a model first.', 'error'); return; }

  saveBtn.disabled = true;
  setStatus('Saving...', 'info');

  try {
    const res = await fetch('/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ provider: provider, api_key: apiKey, model: model })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || ('server returned ' + res.status));

    apiKeyEl.value = '';
    fetchModelsBtn.disabled = true;
    saveBtn.classList.add('hidden');
    providerEl.disabled = true;
    apiKeyEl.disabled = true;
    modelEl.disabled = true;
    setStatus('Saved. Provider: ' + data.provider + ', model: ' + data.model + '.', 'success');
    nextStepsEl.classList.remove('hidden');
  } catch (err) {
    setStatus('Error: ' + friendlyErrorMessage(err), 'error');
    saveBtn.disabled = false;
  }
});
"""
