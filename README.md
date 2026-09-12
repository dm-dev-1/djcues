# djcues

Automated hot cue and memory cue placement for rekordbox, based on phrase analysis (PSSI), vocal detection (PVDI), and a standardized cue strategy.

djcues reads your rekordbox database, analyzes each track's phrase structure and vocal content, proposes cue placements following your cue system, and lets you review, adjust, and apply them back to the database.

![Screenshot of djcues visualization](djcues.png)

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo-url> && cd dj
uv sync
```

Run commands with `uv run djcues <command>`, or activate the venv first:

```bash
source .venv/bin/activate
djcues <command>
```

## Cue System

djcues uses a standardized 8 hot cue + 8 memory cue system defined in `cue-system.csv`:

| Pad | Hot Cue         | Color  | Memory Cue       |
|-----|-----------------|--------|-------------------|
| A   | First Beat      | Green  | First Beat        |
| B   | Loop In         | Green  | Loop In           |
| C   | Vocal / Buildup | Yellow | Before Buildup    |
| D   | Drop            | Red    | Before Drop       |
| E   | Breakdown       | Blue   | Before Breakdown  |
| F   | Special         | Purple | Before Special    |
| G   | Outro           | Cyan   | Before Outro      |
| H   | Loop Out        | Orange | Loop Out          |

Memory cues 3-7 are placed a configurable number of bars before their corresponding hot cue (default: 16 bars). Memory cues 1, 2, and 8 share the same position as their hot cue. Colors are standardized per slot.

To customize the cue system, edit `cue-system.csv` and update the `CUE_SYSTEM` definition in `src/djcues/constants.py`.

## Usage

### Preview cues for a track

```bash
djcues propose "Playlist Name" "Track Name"
```

### Visualize with waveform, phrases, and vocal detection

```bash
# Single track
djcues viz "Playlist Name" "Track Name"

# Entire playlist
djcues viz "Playlist Name" --all

# Compare proposals against existing cues
djcues viz "Playlist Name" "Track Name" --compare
```

### Review and apply cues

```bash
# Launch interactive review in browser
djcues review "Playlist Name" --all

# Apply accepted cues to rekordbox (requires rekordbox to be closed)
djcues apply <session-file.json>

# Preview what would be written
djcues apply <session-file.json> --dry-run
```

In the review UI:
- **Accept** / **Skip** per track or use **Accept All**
- Click a cue marker to select it, then use arrow keys to nudge by 1 bar
- Press Delete to skip an individual cue
- Memory cues auto-recalculate when you adjust their hot cue

### Analysis dashboard

```bash
# Browse your library and run analysis from the browser
djcues dashboard

# Auto-select a playlist on load
djcues dashboard "Playlist Name"
```

Browse your real rekordbox playlists (including folders) and tracks without already knowing exact names to type as CLI arguments. Select a track to see its metadata, then run `propose`, `compare`, or `beatgrid` against it with the same flags the CLI exposes (`--agentic`, `--refine-drops`, `--deep`, offset/loop-bars, etc.) — results, including a real waveform/timeline for propose/compare, render inline as the job completes. A `--deep` or `--agentic` run genuinely takes as long as it does from the CLI (minutes for Demucs, real API latency for agentic) without blocking the rest of the page — browse a different track while one runs. Every dashboard-triggered run shares the same [analysis cache](#analysis-cache) as the CLI, so a track already analyzed either way shows up instantly the second time. "Open in Viz" / "Open in Review" launch the existing, unmodified commands for the selected track in their own tab. Fully read-only, same guarantee as `propose`/`compare`/`viz`/`review`/`beatgrid`.

### Compare accuracy against curated tracks

```bash
djcues compare "Processed" --all
```

### LLM-based analysis (optional, BYOK)

`--agentic` replaces the local heuristic with a multi-agent LLM analysis on `propose`, `compare`, and `review`: three specialists (Structure, Vocal, Energy) propose positions in parallel from the same phrase/vocal/energy data the heuristic uses, then a critic pass reviews their confidence and notes. No raw audio ever leaves your machine — only that same compact summary. Needs an API key first: run `djcues auth set` (see below).

```bash
# See the cost before spending anything
djcues propose "Playlist Name" "Track Name" --agentic --estimate-only

# Run it for real, using your configured provider/model
djcues propose "Playlist Name" "Track Name" --agentic
djcues compare "Playlist Name" --all --agentic
djcues review "Playlist Name" --all --agentic

# Override the configured provider/model, or skip the critic pass (3 calls/track instead of 4)
djcues propose "Playlist Name" "Track Name" --agentic --provider gemini --model gemini-3.7-flash
djcues propose "Playlist Name" "Track Name" --agentic --skip-critic
```

Every `--agentic` run prints its real cost when it finishes — actual tokens used, priced from djcues' local pricing table — not just an upfront guess. `--estimate-only` (propose only) does a genuine pre-flight token count against the selected tracks and exits without calling the model.

Install with `pip install djcues[agentic]` (adds the `anthropic` and `google-genai` SDKs and `keyring`).

### Configure an API key for --agentic

```bash
# Prompts for provider, API key, and a live-fetched model list
djcues auth set

# Show the configured provider/model and where the key came from (never the key itself)
djcues auth status

# Remove a stored key
djcues auth clear

# Same flow as `auth set`, in a local browser page instead of the terminal
djcues auth web
```

Keys are stored in your OS credential store (Windows Credential Manager, macOS Keychain, or the Linux Secret Service) via `keyring` — never in a plaintext file. If you'd rather not run `auth set`, `--agentic` also falls back to the `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` environment variables when no key is in the keyring. Requires `pip install djcues[agentic]`.

### View correction history

```bash
djcues history
```

Read-only. `djcues apply` automatically logs every accepted/adjusted/skipped hot cue to a local SQLite database at `~/.djcues/history.db` (independent of rekordbox's own database) as it writes; `history` just prints the per-pad totals and correction counts accumulated there so far. Nothing to configure — it fills in as you use `apply` normally.

### Analysis cache

```bash
# See what's cached
djcues cache status

# Force a fresh recompute for one run (still refreshes the cache)
djcues propose "Playlist Name" "Track Name" --refine-drops --deep --no-cache

# Wipe the cache entirely
djcues cache clear
```

`propose`, `compare`, `review`, and `beatgrid` all cache their results locally at `~/.djcues/analysis_cache.db`, keyed on the track plus every option that can change the result (`--agentic`/provider/model, `--refine-drops`, `--deep`, `--offset`, `--loop-bars`, etc.) — so re-running the same analysis on a track you've already checked reuses the cached result instead of re-paying `--deep`'s multi-minute Demucs pass or `--agentic`'s real API cost. A cached result is only ever served while rekordbox's own phrase/vocal/waveform/beat-grid data for that track hasn't changed since — re-analyze the track in rekordbox and the next run computes fresh automatically. Pass `--no-cache` on any of the four commands to force a fresh run without disabling the cache going forward.

### Audio-ML analysis (optional, local-only)

Two opt-in features analyze the real audio file instead of only rekordbox's pre-computed analysis. Both run entirely locally — no audio ever leaves your machine.

```bash
# Verify rekordbox's stored beat grid is still trustworthy
djcues beatgrid "Playlist Name" --all

# Force real audio-based verification even when the free check looks fine
djcues beatgrid "Playlist Name" "Track Name" --deep

# Refine Drop/Breakdown/Special cue positions against the real audio (composes with propose/compare/review)
djcues propose "Playlist Name" "Track Name" --refine-drops
djcues compare "Playlist Name" --all --refine-drops
djcues review "Playlist Name" --all --refine-drops

# --deep additionally runs Demucs source separation for a cleaner bass/drums
# signal before refining -- meaningfully slower (~7-12 minutes per track on
# CPU; see --device below to use a GPU instead), so use it sparingly,
# on 1-2 tracks at a time
djcues propose "Playlist Name" "Track Name" --refine-drops --deep

# Configure which hardware device runs --deep/beatgrid analysis, with a live smoke test
djcues auth device --device cuda
```

`beatgrid` always runs a free, audio-independent self-consistency check first (no extra install needed) using rekordbox's own full per-beat grid data, and only escalates to real audio when that check looks suspicious or `--deep` forces it. `--refine-drops` never does an independent redetection — it only looks for a dominant energy transition (a rise for Drop/Special, a dip for Breakdown) in a bounded window around the position the heuristic (or `--agentic`) already proposed, and only moves the cue when the audio evidence is clearly dominant; otherwise it leaves the existing position untouched and says so in the notes. Neither feature ever writes to rekordbox — same read-only guarantee as every other command until `apply`.

Every cue `--refine-drops` actually moves gets an extra diagnostic hint in the output — `energy pattern: oscillating` (the energy swings back within a beat or two, more likely a false positive) or `stable` (the new position holds steady, more likely a genuine improvement). This is an unverified heuristic, not a verdict — checked against real tracks but not proven reliable — meant to help you prioritize which moved cues are most worth a quick listen, not to replace listening.

Install with `pip install djcues[audio]` for `--refine-drops` (needs `librosa`/`soundfile`), or `djcues[ml]` for `--deep` and `beatgrid --deep` (adds `beat-this` and `demucs`, which pull in `torch` — a large, platform-specific download, hundreds of MB+).

#### Hardware acceleration (`--device`)

`--deep` and `beatgrid` default to CPU, but accept `--device auto|cpu|cuda|directml` (or a persistent default via `djcues auth device`, same precedence as `--provider`/`--model`: flag > configured preference > `auto`). `auto` tries CUDA then DirectML and falls back to CPU automatically; an explicit choice that turns out unavailable also falls back to CPU with a one-time warning rather than failing the run — CPU always works.

- **NVIDIA (CUDA)**: needs a CUDA-enabled torch build, which `pip install djcues[ml]` alone does not provide (CUDA wheels live on a separate index, not PyPI). If `torch.cuda.is_available()` is `False` after installing djcues, reinstall torch by following [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for your CUDA version, in the same environment.
- **AMD and Intel iGPU (DirectML, Windows only)**: detected but never used for real analysis, permanently -- `--device directml` always falls back to CPU with a clear message. This isn't a "not implemented yet"; it's confirmed live (a separate Python 3.12 environment, real Intel Iris Xe Graphics hardware) that DirectML itself works fine for ordinary tensor math, but both of djcues's actual models (Demucs, beat_this) depend on STFT/complex-number operations that DirectML cannot run -- they crash the whole process outright rather than raising a normal error, so no fallback logic could safely use this backend even if djcues tried. `cpu`/`cuda`/`auto` are fully working. If you install the `directml` extra anyway (needs Python 3.10-3.12 -- `torch-directml` has no build for 3.13+), `djcues auth device`/the dashboard will show real detected hardware info alongside this same explanation, rather than a generic "not installed" message.

Run `djcues auth device --device <choice>` any time to see what's actually detected on your machine and confirm your preference passes a real smoke test before saving it.

## How It Works

1. **Phrase analysis (PSSI)**: rekordbox analyzes tracks into phrases (Intro, Up, Down, Chorus, Outro). djcues maps these to cue slots using heuristics — e.g., Drop aligns with the first Chorus after 20% of the track.

2. **Vocal detection (PVDI)**: rekordbox's vocal detection data (stored in `.2EX` ANLZ files) provides per-frame vocal confidence. djcues uses the first strong vocal onset to place the Vocal/Buildup cue.

3. **Waveform (PWV5)**: The color waveform detail data is extracted and rendered in the HTML visualizer for visual reference.

4. **Beat grid**: All cue positions are snapped to the beat grid for precise alignment.

## Safety

- **Auto-backup**: `djcues apply` automatically backs up `master.db` before writing
- **Overwrite protection**: Tracks with existing cues require explicit confirmation
- **Rekordbox must be closed**: The apply command will not write while rekordbox is running
- **Read-only by default**: `propose`, `compare`, `viz`, `review`, `beatgrid`, and `dashboard` never modify the database
- **DB-only writes**: Cues are written to `master.db` only (not ANLZ files). rekordbox handles ANLZ sync on USB export.
- **Analysis cache**: `propose`/`compare`/`review`/`beatgrid` cache results locally at `~/.djcues/analysis_cache.db` to avoid redundant recomputation — never touches the rekordbox database; use `--no-cache` to force a fresh run.

## Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `--offset` | 16 | Memory cue offset in bars before hot cue |
| `--loop-bars` | 4 | Loop length in bars for Loop In / Loop Out |

These are sane defaults for dance tracks, EDM, house, etc. with 8 bar phrases. If you're mixing hip hop, rock, r&b, or pop, or mixing more aggressively, try 8 bar offsets and 2 bar loops. If you're mixing deeper forms of music with more overlap and longer phrasing, 32 bar offsets and 8 bar loops might be best. Use your ear and go with what feels right.

## Project Structure

```
src/djcues/
    models.py       # Data model (Track, CuePoint, Phrase, BeatGrid)
    constants.py    # PSSI mood tables, cue system definition, color maps
    db.py           # Rekordbox database reader
    strategy.py     # Cue placement heuristics
    agentic.py      # LLM-based multi-agent analysis (--agentic, BYOK)
    providers/      # Anthropic/Gemini provider adapters for --agentic
    auth.py         # BYOK API key storage (OS keyring)
    audio.py        # Real audio file loading (djcues[audio])
    beat_verify.py  # Beat-grid self-consistency + real audio verification
    drop_enhance.py # Drop/Breakdown/Special cue refinement against real audio (--refine-drops)
    history.py      # Correction-history logging
    analysis_cache.py # Persistent cache of completed analysis runs (propose/compare/review/beatgrid)
    metrics.py      # Precision/recall/F1 for compare
    viz.py          # HTML timeline visualizer
    review.py       # Interactive review HTML + session management
    dashboard.py    # Analysis dashboard HTML/CSS/JS (browse playlists/tracks, run propose/compare/beatgrid)
    server.py       # Local HTTP server for review sessions, the BYOK setup wizard, and the dashboard
    writer.py       # DB backup and cue writes
    cli.py          # Click CLI (propose, compare, viz, review, dashboard, apply, beatgrid, auth, history, cache)
```

## License

BSD 3-Clause. See [LICENSE](LICENSE).

## Colophon

Under the hood, djcues leverages [pyrekordbox](https://pyrekordbox.readthedocs.io) to interact with the [Rekordbox](https://rekordbox.com) database. It also makes extensive use of [analysis files](https://pyrekordbox.readthedocs.io/en/latest/formats/anlz.html#) from Rekordbox. Other dependencies include [click](https://click.palletsprojects.com/en/stable/) and [pytest](https://docs.pytest.org/en/stable/).

This project was created in fleeting free moments with the help of [Claude Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6) and [Superpowers](https://github.com/obra/superpowers).