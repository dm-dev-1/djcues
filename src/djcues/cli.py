"""CLI for djcues — propose and compare cue placements."""

from __future__ import annotations

import json
import logging
import sys
import warnings
import click

# Suppress noisy pyrekordbox warnings (PVDI tag, rekordbox running, etc.)
for _name in ("pyrekordbox", "pyrekordbox.db6", "pyrekordbox.anlz"):
    logging.getLogger(_name).setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", module="pyrekordbox")

from djcues.constants import CUE_SYSTEM_BY_PAD, KIND_TO_PAD
from djcues.db import find_playlist, load_playlist_tracks
from djcues.metrics import compare_cues, merge_pad_stats, overall_stats
from djcues.strategy import CueStrategy


def _format_time(ms: float) -> str:
    """Format milliseconds as M:SS.s"""
    total_seconds = ms / 1000
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:04.1f}"


def _print_proposal(proposal, track):
    """Print a full cue proposal for a single track."""
    click.echo(f"\n{'=' * 60}")
    click.echo(f"  {track.title} — {track.artist}")
    click.echo(f"  BPM: {track.bpm:.1f}")
    click.echo(f"{'=' * 60}")

    # Phrases
    click.echo(f"\n  Phrases ({len(track.phrases)}):")
    for p in track.phrases:
        bars = p.beat_length // 4
        click.echo(
            f"    {p.label:<12s}  beat {p.beat_start:>4d}–{p.beat_end:<4d}"
            f"  ({bars:>2d} bars)  {_format_time(p.position_ms)}"
        )

    # Hot cues (sorted by kind)
    click.echo(f"\n  Hot Cues ({len(proposal.hot_cues)}):")
    for hc in sorted(proposal.hot_cues, key=lambda c: c.kind):
        pad = KIND_TO_PAD.get(hc.kind, "?")
        conf = proposal.confidence.get(pad, 0.0)
        beat = track.beat_grid.ms_to_beat(hc.position_ms)
        loop_info = ""
        if hc.is_loop and hc.loop_end_ms is not None:
            loop_info = f"  loop→{_format_time(hc.loop_end_ms)}"
        click.echo(
            f"    [{pad}] {hc.comment:<20s}  {_format_time(hc.position_ms)}"
            f"  beat {beat:>4d}  conf={conf:.0%}{loop_info}"
        )

    # Memory cues (sorted by position)
    click.echo(f"\n  Memory Cues ({len(proposal.memory_cues)}):")
    for i, mc in enumerate(sorted(proposal.memory_cues, key=lambda c: c.position_ms)):
        slot_info = ""
        # Find which slot this memory cue belongs to by matching comment
        for pad, slot in CUE_SYSTEM_BY_PAD.items():
            if slot.memory_cue_label == mc.comment:
                if slot.memory_offset_bars > 0:
                    slot_info = f"  ({slot.memory_offset_bars} bars before {slot.hot_cue_label})"
                else:
                    slot_info = f"  (same as {slot.hot_cue_label})"
                break
        loop_info = ""
        if mc.is_loop and mc.loop_end_ms is not None:
            loop_info = f"  loop→{_format_time(mc.loop_end_ms)}"
        click.echo(
            f"    [{i + 1}] {mc.comment:<20s}  {_format_time(mc.position_ms)}"
            f"{slot_info}{loop_info}"
        )

    # Notes
    if proposal.notes:
        click.echo(f"\n  Notes:")
        for note in proposal.notes:
            click.echo(f"    • {note}")


def _print_comparison(proposal, track):
    """Print side-by-side comparison of existing vs proposed hot cues."""
    click.echo(f"\n{'=' * 60}")
    click.echo(f"  {track.title} — {track.artist}")
    click.echo(f"  BPM: {track.bpm:.1f}")
    click.echo(f"{'=' * 60}")

    existing_by_kind = {c.kind: c for c in track.cues if c.kind > 0}
    proposed_by_kind = {c.kind: c for c in proposal.hot_cues}
    all_kinds = sorted(set(existing_by_kind.keys()) | set(proposed_by_kind.keys()))

    pad_stats = compare_cues(track.cues, proposal.hot_cues)

    click.echo(f"\n  {'Pad':<5s} {'Label':<20s} {'Existing':<12s} {'Proposed':<12s} {'Delta':>8s}")
    click.echo(f"  {'-' * 5} {'-' * 20} {'-' * 12} {'-' * 12} {'-' * 8}")

    for kind in all_kinds:
        pad = KIND_TO_PAD.get(kind, "?")
        slot = CUE_SYSTEM_BY_PAD.get(pad)
        label = slot.hot_cue_label if slot else f"Kind {kind}"

        existing = existing_by_kind.get(kind)
        proposed = proposed_by_kind.get(kind)

        existing_str = _format_time(existing.position_ms) if existing else "—"
        proposed_str = _format_time(proposed.position_ms) if proposed else "—"

        delta_str = ""
        if existing and proposed:
            delta_ms = proposed.position_ms - existing.position_ms
            delta_str = f"{delta_ms:+.0f}ms"
            if abs(delta_ms) <= 1000:
                delta_str += " ✓"
        elif existing:
            delta_str = "missing"
        elif proposed:
            delta_str = "new (unmatched)"

        click.echo(
            f"  [{pad}]   {label:<20s} {existing_str:<12s} {proposed_str:<12s} {delta_str:>8s}"
        )

    total = overall_stats(pad_stats)
    denom = total.matches + total.misses + total.false_positives
    if denom > 0:
        click.echo(
            f"\n  Precision: {total.precision:.0%}  Recall: {total.recall:.0%}  "
            f"F1: {total.f1:.0%}  ({total.matches} matched, {total.misses} missed, "
            f"{total.false_positives} unmatched proposals)"
        )
    else:
        click.echo(f"\n  No existing hot cues to compare.")

    return pad_stats


def _resolve_agentic_provider(provider_name, model):
    """Resolve (provider_name, api_key, model) for --agentic, or exit
    with a clear error if no key is configured. Shared by propose/
    compare/review so the "run `djcues auth set`" message and the
    config/env-var resolution logic live in exactly one place."""
    from djcues.auth import load_config, resolve_api_key
    from djcues.providers import DEFAULT_MODEL

    config = load_config()
    provider_name = provider_name or config.get("provider", "anthropic")
    api_key, _source = resolve_api_key(provider_name)
    if not api_key:
        click.echo(
            f"Error: no API key configured for '{provider_name}'. "
            f"Run: djcues auth set",
            err=True,
        )
        raise SystemExit(1)
    resolved_model = model or config.get("model") or DEFAULT_MODEL.get(provider_name)
    return provider_name, api_key, resolved_model


def _get_proposer(agentic, provider_name, model, offset, loop_bars, skip_critic):
    """Returns (proposer, telemetry_list, resolved_model). `proposer(track)`
    returns a CueProposal, either from the local heuristic or agentic
    analysis. telemetry_list/resolved_model are None for the heuristic
    path; telemetry_list accumulates one AgenticTelemetry per call for
    the agentic path's cost summary."""
    if not agentic:
        strategy = CueStrategy(memory_offset_bars=offset, loop_length_bars=loop_bars)
        return strategy.propose, None, None

    from djcues.providers import get_provider
    from djcues.agentic import propose_with_telemetry

    provider_name, api_key, resolved_model = _resolve_agentic_provider(provider_name, model)
    provider = get_provider(provider_name)
    telemetry_list = []

    def proposer(track):
        proposal, telemetry = propose_with_telemetry(
            track,
            provider,
            api_key,
            resolved_model,
            memory_offset_bars=offset,
            loop_length_bars=loop_bars,
            skip_critic=skip_critic,
        )
        telemetry_list.append(telemetry)
        return proposal

    return proposer, telemetry_list, resolved_model


def _print_cost_summary(telemetry_list, model, track_count):
    """Print the post-run cost summary for an --agentic run, in the
    same voice as apply's 'Done: N tracks, M cues written.' line."""
    total_calls = sum(t.calls_made for t in telemetry_list)
    total_input = sum(t.input_tokens for t in telemetry_list)
    total_output = sum(t.output_tokens for t in telemetry_list)
    total_errors = sum(len(t.errors) for t in telemetry_list)

    from djcues.providers import estimate_cost

    cost = estimate_cost(model, total_input, total_output)
    cost_str = f"${cost:.4f}" if cost is not None else "unknown (model not in local price table)"
    click.echo(
        f"\nDone: {track_count} tracks analyzed. "
        f"Actual cost: {cost_str} ({total_calls} calls, {model})."
    )
    if total_errors:
        click.echo(
            f"  {total_errors} specialist/critic call(s) fell back to the heuristic "
            f"value for their pad — see notes above for which."
        )


@click.group()
def cli():
    """djcues — automated rekordbox cue placement based on phrase analysis."""
    # Windows consoles default stdout/stderr to the system codepage (e.g. cp1252),
    # which can't encode characters like → used in cue output.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


@cli.command()
@click.argument("playlist_name")
@click.argument("track_name", required=False)
@click.option("--all", "all_tracks", is_flag=True, help="Process all tracks in the playlist.")
@click.option("--offset", default=16, show_default=True, help="Memory cue offset in bars.")
@click.option("--loop-bars", default=4, show_default=True, help="Loop length in bars.")
@click.option("--agentic", is_flag=True, help="Use LLM-based multi-agent analysis instead of the local heuristic (BYOK — run 'djcues auth set' first).")
@click.option("--provider", default=None, help="Provider for --agentic: 'anthropic' or 'gemini'. Defaults to your configured provider.")
@click.option("--model", default=None, help="Model for --agentic. Defaults to your configured model.")
@click.option("--skip-critic", is_flag=True, help="Skip the --agentic critic pass (3 calls/track instead of 4).")
@click.option("--estimate-only", is_flag=True, help="Print an estimated --agentic cost and exit without calling the model.")
def propose(playlist_name, track_name, all_tracks, offset, loop_bars, agentic, provider, model, skip_critic, estimate_only):
    """Propose cue placements for tracks in a playlist."""
    if estimate_only and not agentic:
        click.echo("Error: --estimate-only only applies with --agentic.", err=True)
        raise SystemExit(1)

    if estimate_only:
        from djcues.agentic import estimate_cost as estimate_agentic_cost

        _provider_name, _api_key, resolved_model = _resolve_agentic_provider(provider, model)
        in_tok, out_tok, cost = estimate_agentic_cost(resolved_model, skip_critic)
        cost_str = f"${cost:.4f}" if cost is not None else "unknown (model not in local price table)"
        click.echo(
            f"Estimated cost per track: {cost_str} "
            f"(~{in_tok} input / ~{out_tok} output tokens, {resolved_model})"
        )
        return

    playlist = find_playlist(playlist_name)
    if playlist is None:
        click.echo(f"Error: playlist '{playlist_name}' not found.", err=True)
        raise SystemExit(1)

    tracks = load_playlist_tracks(playlist.ID)
    if not tracks:
        click.echo(f"Error: no tracks found in playlist '{playlist_name}'.", err=True)
        raise SystemExit(1)

    proposer, telemetry_list, resolved_model = _get_proposer(
        agentic, provider, model, offset, loop_bars, skip_critic
    )

    if all_tracks:
        selected = tracks
    elif track_name:
        selected = [t for t in tracks if track_name.lower() in t.title.lower()]
        if not selected:
            click.echo(f"Error: no track matching '{track_name}' in playlist.", err=True)
            raise SystemExit(1)
    else:
        click.echo("Error: provide a track name or use --all.", err=True)
        raise SystemExit(1)

    for t in selected:
        proposal = proposer(t)
        _print_proposal(proposal, t)

    if telemetry_list is not None:
        _print_cost_summary(telemetry_list, resolved_model, len(selected))


@cli.command()
@click.argument("playlist_name")
@click.argument("track_name", required=False)
@click.option("--all", "all_tracks", is_flag=True, help="Compare all tracks in the playlist.")
@click.option("--offset", default=16, show_default=True, help="Memory cue offset in bars.")
@click.option("--loop-bars", default=4, show_default=True, help="Loop length in bars.")
@click.option("--agentic", is_flag=True, help="Use LLM-based multi-agent analysis instead of the local heuristic (BYOK — run 'djcues auth set' first).")
@click.option("--provider", default=None, help="Provider for --agentic: 'anthropic' or 'gemini'. Defaults to your configured provider.")
@click.option("--model", default=None, help="Model for --agentic. Defaults to your configured model.")
@click.option("--skip-critic", is_flag=True, help="Skip the --agentic critic pass (3 calls/track instead of 4).")
def compare(playlist_name, track_name, all_tracks, offset, loop_bars, agentic, provider, model, skip_critic):
    """Compare existing cues with proposed placements."""
    playlist = find_playlist(playlist_name)
    if playlist is None:
        click.echo(f"Error: playlist '{playlist_name}' not found.", err=True)
        raise SystemExit(1)

    tracks = load_playlist_tracks(playlist.ID)
    if not tracks:
        click.echo(f"Error: no tracks found in playlist '{playlist_name}'.", err=True)
        raise SystemExit(1)

    proposer, telemetry_list, resolved_model = _get_proposer(
        agentic, provider, model, offset, loop_bars, skip_critic
    )

    if all_tracks:
        per_track_stats = []
        for t in tracks:
            proposal = proposer(t)
            per_track_stats.append(_print_comparison(proposal, t))

        if telemetry_list is not None:
            _print_cost_summary(telemetry_list, resolved_model, len(tracks))

        merged = merge_pad_stats(per_track_stats)
        total = overall_stats(merged)
        denom = total.matches + total.misses + total.false_positives
        if denom > 0:
            click.echo(f"\n{'=' * 60}")
            click.echo(
                f"  Overall — Precision: {total.precision:.0%}  "
                f"Recall: {total.recall:.0%}  F1: {total.f1:.0%}"
            )
            click.echo(
                f"  ({total.matches} matched, {total.misses} missed, "
                f"{total.false_positives} unmatched proposals)"
            )
            click.echo(f"\n  {'Pad':<5s} {'Matched':<9s} {'Missed':<8s} {'Unmatched':<11s} {'Precision':<11s} {'Recall':<8s}")
            click.echo(f"  {'-' * 5} {'-' * 9} {'-' * 8} {'-' * 11} {'-' * 11} {'-' * 8}")
            for pad in sorted(merged):
                s = merged[pad]
                click.echo(
                    f"  {pad:<5s} {s.matches:<9d} {s.misses:<8d} {s.false_positives:<11d} "
                    f"{s.precision:<11.0%} {s.recall:<8.0%}"
                )
            click.echo(f"{'=' * 60}")
    elif track_name:
        matched = [t for t in tracks if track_name.lower() in t.title.lower()]
        if not matched:
            click.echo(f"Error: no track matching '{track_name}' in playlist.", err=True)
            raise SystemExit(1)
        for t in matched:
            proposal = proposer(t)
            _print_comparison(proposal, t)
        if telemetry_list is not None:
            _print_cost_summary(telemetry_list, resolved_model, len(matched))
    else:
        click.echo("Error: provide a track name or use --all.", err=True)
        raise SystemExit(1)


@cli.command()
@click.argument("playlist")
@click.argument("track_name", required=False)
@click.option("--all", "all_tracks", is_flag=True, help="Visualize all tracks in playlist")
@click.option("--compare", "compare_mode", is_flag=True, help="Show existing vs proposed")
@click.option("--offset", default=16, help="Memory cue offset in bars (default: 16)")
@click.option("--loop-bars", default=4, help="Loop length in bars (default: 4)")
@click.option("--output", "-o", default=None, help="Output file path (default: auto-generated)")
def viz(playlist, track_name, all_tracks, compare_mode, offset, loop_bars, output):
    """Generate an HTML timeline visualization."""
    import pathlib
    import webbrowser
    from djcues.viz import render_timeline, render_playlist

    pl = find_playlist(playlist)
    if pl is None:
        click.echo(f"Playlist '{playlist}' not found.", err=True)
        raise SystemExit(1)

    tracks = load_playlist_tracks(pl.ID)
    strategy = CueStrategy(memory_offset_bars=offset, loop_length_bars=loop_bars)

    if all_tracks:
        pairs = []
        for t in tracks:
            if t.phrases:
                pairs.append((t, strategy.propose(t)))
            else:
                click.echo(f"  Skipping {t.title} (no phrase data)", err=True)
        click.echo(f"Rendering {len(pairs)} tracks...")
        page_html = render_playlist(playlist, pairs, compare=compare_mode)
        if output:
            out_path = pathlib.Path(output)
        else:
            safe_name = "".join(c if c.isalnum() or c in " -_" else "" for c in playlist).strip().replace(" ", "-").lower()
            out_path = pathlib.Path(f"{safe_name}-cues.html")
    else:
        if not track_name:
            click.echo("Provide a track name or use --all.", err=True)
            raise SystemExit(1)

        matches = [t for t in tracks if track_name.lower() in t.title.lower()]
        if not matches:
            click.echo(f"No track matching '{track_name}' in playlist '{playlist}'.", err=True)
            raise SystemExit(1)

        track = matches[0]
        if not track.phrases:
            click.echo(f"{track.title} has no phrase data.", err=True)
            raise SystemExit(1)

        proposal = strategy.propose(track)
        page_html = render_timeline(track, proposal, compare=compare_mode)
        if output:
            out_path = pathlib.Path(output)
        else:
            safe_name = "".join(c if c.isalnum() or c in " -_" else "" for c in track.title).strip().replace(" ", "-").lower()
            out_path = pathlib.Path(f"{safe_name}-cues.html")

    out_path.write_text(page_html, encoding="utf-8")
    click.echo(f"Written to {out_path}")
    webbrowser.open(f"file://{out_path.resolve()}")


@cli.command()
@click.argument("playlist")
@click.argument("track_name", required=False)
@click.option("--all", "all_tracks", is_flag=True, help="Review all tracks in playlist")
@click.option("--offset", default=16, help="Memory cue offset in bars (default: 16)")
@click.option("--loop-bars", default=4, help="Loop length in bars (default: 4)")
@click.option("--output", "-o", default=None, help="Output directory (default: current dir)")
@click.option("--agentic", is_flag=True, help="Use LLM-based multi-agent analysis instead of the local heuristic (BYOK — run 'djcues auth set' first).")
@click.option("--provider", default=None, help="Provider for --agentic: 'anthropic' or 'gemini'. Defaults to your configured provider.")
@click.option("--model", default=None, help="Model for --agentic. Defaults to your configured model.")
@click.option("--skip-critic", is_flag=True, help="Skip the --agentic critic pass (3 calls/track instead of 4).")
def review(playlist, track_name, all_tracks, offset, loop_bars, output, agentic, provider, model, skip_critic):
    """Launch interactive review session in browser."""
    import pathlib
    import time
    import webbrowser
    from djcues.review import create_session, render_review_html
    from djcues.server import start_server

    pl = find_playlist(playlist)
    if pl is None:
        click.echo(f"Playlist '{playlist}' not found.", err=True)
        raise SystemExit(1)

    tracks = load_playlist_tracks(pl.ID)
    if not tracks:
        click.echo(f"No tracks found in playlist '{playlist}'.", err=True)
        raise SystemExit(1)

    proposer, telemetry_list, resolved_model = _get_proposer(
        agentic, provider, model, offset, loop_bars, skip_critic
    )

    if all_tracks:
        selected = tracks
    elif track_name:
        selected = [t for t in tracks if track_name.lower() in t.title.lower()]
        if not selected:
            click.echo(f"No track matching '{track_name}' in playlist '{playlist}'.", err=True)
            raise SystemExit(1)
    else:
        click.echo("Provide a track name or use --all.", err=True)
        raise SystemExit(1)

    pairs = []
    for t in selected:
        if t.phrases:
            pairs.append((t, proposer(t)))
        else:
            click.echo(f"  Skipping {t.title} (no phrase data)", err=True)

    if telemetry_list is not None:
        _print_cost_summary(telemetry_list, resolved_model, len(pairs))

    if not pairs:
        click.echo("No tracks with phrase data to review.", err=True)
        raise SystemExit(1)

    # Generate session
    session = create_session(
        playlist_name=playlist,
        playlist_id=pl.ID,
        tracks_and_proposals=pairs,
        memory_offset_bars=offset,
        loop_length_bars=loop_bars,
    )

    # Determine output directory and safe name
    safe_name = "".join(
        c if c.isalnum() or c in " -_" else "" for c in playlist
    ).strip().replace(" ", "-").lower()

    if output:
        out_dir = pathlib.Path(output)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = pathlib.Path(".")

    session_path = out_dir / f"{safe_name}-session.json"
    session_path.write_text(
        json.dumps(session, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Start local server
    html_path = out_dir / f"{safe_name}-review.html"
    server, port = start_server(
        html_path=html_path,
        session_path=session_path,
    )
    server_url = f"http://127.0.0.1:{port}"

    # Generate review HTML with server URL embedded
    review_html = render_review_html(
        playlist_name=playlist,
        tracks_and_proposals=pairs,
        session_path=str(session_path),
        server_url=server_url,
    )
    html_path.write_text(review_html, encoding="utf-8")

    # Open browser
    webbrowser.open(server_url)

    # Print info
    click.echo(f"Session: {session_path}")
    click.echo(f"Server:  {server_url}")
    click.echo(f"Apply:   uv run djcues apply {session_path}")

    # Block until interrupted
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        click.echo("\nServer stopped.")


@cli.command()
@click.argument("session_file", type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, help="Show what would be written without making changes")
@click.option("--force", is_flag=True, help="Skip overwrite confirmation")
def apply(session_file, dry_run, force):
    """Apply a review session to the rekordbox database."""
    import pathlib
    from djcues.writer import apply_session

    apply_session(
        session_path=pathlib.Path(session_file),
        dry_run=dry_run,
        force=force,
    )


@cli.command()
def history():
    """Show correction-history stats logged from applied review sessions."""
    from djcues.history import default_db_path, summary

    stats = summary()
    db_path = default_db_path()
    if not stats:
        click.echo(f"No correction history yet at {db_path}.")
        click.echo("This fills in as you review and apply sessions — nothing to show yet.")
        return

    click.echo(f"Correction history: {db_path}\n")
    click.echo(f"  {'Pad':<5s} {'Total':<8s} {'Corrected':<11s} {'First seen':<20s} {'Last seen':<20s}")
    click.echo(f"  {'-' * 5} {'-' * 8} {'-' * 11} {'-' * 20} {'-' * 20}")
    for row in stats:
        click.echo(
            f"  {row['pad']:<5s} {row['total']:<8d} {row['corrected']:<11d} "
            f"{row['first_seen']:<20s} {row['last_seen']:<20s}"
        )


@cli.group()
def auth():
    """Manage BYOK API keys and settings for --agentic analysis.

    Keys are stored in your OS credential store (Windows Credential
    Manager / macOS Keychain / Linux Secret Service) via the `keyring`
    package — never in a plaintext file. Requires the 'agentic' extra:
    pip install djcues[agentic]
    """


@auth.command("set")
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "gemini"]),
    prompt=True,
    help="Which provider to configure.",
)
def auth_set(provider):
    """Configure an API key and default model, with a live model list."""
    from djcues.auth import KeyringUnavailableError, load_config, save_config, set_api_key
    from djcues.providers import DEFAULT_MODEL, get_provider

    api_key = click.prompt(f"Enter your {provider} API key", hide_input=True)

    provider_adapter = get_provider(provider)
    click.echo("Fetching available models...")
    try:
        models = provider_adapter.list_models(api_key)
    except Exception as e:
        click.echo(f"Error: could not validate key / fetch models: {e}", err=True)
        raise SystemExit(1)

    if not models:
        click.echo("Error: no models returned — check your key and try again.", err=True)
        raise SystemExit(1)

    default_model_id = DEFAULT_MODEL.get(provider)
    # Recommended lightweight default first, then alphabetical.
    models_sorted = sorted(models, key=lambda m: (m.id != default_model_id, m.id))

    click.echo("\nAvailable models (pricing from djcues' local table, not live):")
    default_choice = 1
    for i, m in enumerate(models_sorted, 1):
        marker = " (recommended, lightweight)" if m.id == default_model_id else ""
        if m.id == default_model_id:
            default_choice = i
        ctx = f", {m.context_window} context" if m.context_window else ""
        click.echo(f"  {i}. {m.display_name} [{m.id}{ctx}]{marker}")

    choice = click.prompt(
        "Choose a model number", type=click.IntRange(1, len(models_sorted)), default=default_choice
    )
    chosen_model = models_sorted[choice - 1].id

    try:
        set_api_key(provider, api_key)
    except KeyringUnavailableError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    config = load_config()
    config["provider"] = provider
    config["model"] = chosen_model
    save_config(config)

    click.echo(
        f"\nSaved. Provider: {provider}, model: {chosen_model}. "
        f"Key stored in your OS credential store, not in any djcues file."
    )


@auth.command("status")
def auth_status():
    """Show the configured provider/model and where the key came from (never the key itself)."""
    from djcues.auth import load_config, resolve_api_key

    config = load_config()
    provider = config.get("provider")
    model = config.get("model")
    if not provider:
        click.echo("No agentic provider configured. Run: djcues auth set")
        return

    api_key, source = resolve_api_key(provider)
    click.echo(f"Provider: {provider}")
    click.echo(f"Model: {model}")
    if api_key:
        click.echo(f"API key: configured (source: {source})")
    else:
        click.echo("API key: not found (checked OS credential store and environment variable)")


@auth.command("clear")
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "gemini"]),
    prompt=True,
    help="Which provider's key to remove.",
)
def auth_clear(provider):
    """Remove a stored API key from the OS credential store."""
    from djcues.auth import KeyringUnavailableError, clear_api_key

    if not click.confirm(f"Remove the stored {provider} API key?"):
        return
    try:
        clear_api_key(provider)
    except KeyringUnavailableError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
    click.echo(f"Removed the {provider} API key from your OS credential store.")


@auth.command("models")
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "gemini"]),
    default=None,
    help="Provider to list models for (defaults to your configured provider).",
)
def auth_models(provider):
    """Re-list live models for a provider without changing your saved config."""
    from djcues.auth import load_config, resolve_api_key
    from djcues.providers import get_provider

    config = load_config()
    provider = provider or config.get("provider")
    if not provider:
        click.echo("Error: no provider configured or specified. Use --provider.", err=True)
        raise SystemExit(1)

    api_key, _source = resolve_api_key(provider)
    if not api_key:
        click.echo(f"Error: no API key configured for '{provider}'. Run: djcues auth set", err=True)
        raise SystemExit(1)

    provider_adapter = get_provider(provider)
    models = provider_adapter.list_models(api_key)
    click.echo(f"Available {provider} models:")
    for m in models:
        ctx = f", {m.context_window} context" if m.context_window else ""
        click.echo(f"  {m.id} ({m.display_name}{ctx})")


@auth.command("web")
def auth_web():
    """Configure a provider/key/model from a local browser page instead
    of the terminal prompt.

    Opens a page served only on 127.0.0.1 — the key is sent once, in a
    POST body, straight to that local server and stored via the same
    OS-credential-store path as `djcues auth set`. Nothing is written to
    disk in plaintext and the key never leaves this machine.
    """
    import time
    import webbrowser

    from djcues.auth_web import render_auth_setup_html
    from djcues.server import start_auth_server

    html = render_auth_setup_html()
    server, port = start_auth_server(html_body=html.encode("utf-8"))
    server_url = f"http://127.0.0.1:{port}"

    webbrowser.open(server_url)
    click.echo(f"Opened {server_url} in your browser.")
    click.echo("Enter your API key there, fetch models, and save. Ctrl+C to cancel.")

    try:
        while not server._setup_complete and not server._shutdown_flag and not server._timed_out:
            time.sleep(0.5)
    except KeyboardInterrupt:
        server._shutdown_flag = True
        click.echo("\nCancelled — nothing was saved.", err=True)
        return

    if server._timed_out:
        click.echo(
            "\nTimed out waiting (30 min of inactivity) — the local server has "
            "shut down, so the page will now show 'Failed to fetch' if you try "
            "it. Run 'djcues auth web' again.",
            err=True,
        )
        raise SystemExit(1)

    if server._setup_complete:
        from djcues.auth import load_config

        config = load_config()
        click.echo(f"\nSaved. Provider: {config.get('provider')}, model: {config.get('model')}.")
