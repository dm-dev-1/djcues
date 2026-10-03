"""`djcues tags ...` -- smart tagging commands.

Registered onto the main CLI from cli.py (cli.add_command(tags)); kept in
its own module so the already-large cli.py doesn't grow by another
command group. Phase 1 (this file): everything here is read-only with
respect to Rekordbox -- it only writes to the local ~/.djcues/tags.db
store (and a session JSON when --out is given). Writing tags into
Rekordbox is Phase 2 and deliberately not implemented yet.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import click

from djcues import tag_analysis
from djcues.db import find_playlist, get_db
from djcues.tag_store import TagStore
from djcues.tagging import (
    CATEGORY_NAMES,
    DEFAULT_THRESHOLDS,
    ENERGY_BANDS,
    TAG_CATALOG,
    energy_band,
    is_calibration_eligible,
)

# Per-tag prevalence the plan treats as healthy for a non-Energy tag
# (Energy bands are ~20% each by construction). Outside this range a
# threshold is probably wrong -- see the plan's acceptance gate.
_HEALTHY_MIN_PCT = 5.0
_HEALTHY_MAX_PCT = 35.0
# Only descriptive tags are judged against that range: Energy bands are
# equal-count by construction, and Check tags are warnings that should be
# rare (measured on the real library: Tempo Varies 0 and Grid Suspect 2 of
# 607 analysable tracks -- Rekordbox's own grids there really are
# constant-tempo, not a data-coverage gap).
_RANGE_CHECKED_CATEGORIES = ("Mix In", "Mix Out", "Vocals")

_SKIP_LABELS = {
    "no_calibration": "not calibrated",
    "no_energy_data": "no waveform/phrase data",
    "sample_or_loop": "loop/sample",
    "no_intro_phrase": "no Intro phrase",
    "no_outro_phrase": "no Outro phrase",
    "no_vocal_data": "no vocal data",
}


@click.group()
def tags():
    """Smart tags for set-list building (read-only so far).

    Derives a small vocabulary of per-track tags -- energy band, intro/
    outro length, vocals in the mix-in/mix-out, key/grid warnings -- from
    analysis djcues already computes. Nothing here writes to Rekordbox
    yet; `propose` only prints (and optionally saves a session file).
    """


@contextmanager
def _quiet_db_warnings():
    """db.py logs a warning for every track whose analysis file can't be
    read -- streaming-linked tracks (encrypted "$A7:v1:..." titles) and
    unanalysed files hit this constantly in a whole-library run, burying
    the progress bar. Those tracks are counted and summarised by the
    commands instead (they simply get no energy/structure tags)."""
    logger = logging.getLogger("djcues.db")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous)


def _progress(label: str, total: int, fn):
    """Run fn(progress_callback) under a click progress bar."""
    with _quiet_db_warnings(), click.progressbar(length=total, label=label, show_pos=True) as bar:
        return fn(lambda _done, _total, _cached: bar.update(1))


def _no_analysis_count(features_by_id) -> int:
    """Tracks with neither waveform nor phrase data -- nothing to tag from."""
    return sum(1 for f in features_by_id.values() if f.mean_energy is None and f.intro_bars is None)


def _band_counts(features_by_id, cutpoints) -> list[int]:
    counts = [0] * ENERGY_BANDS
    for f in features_by_id.values():
        if is_calibration_eligible(f):
            counts[energy_band(f.mean_energy, cutpoints) - 1] += 1
    return counts


def _print_cutpoints(calibration) -> None:
    cp = calibration.cutpoints
    click.echo(f"  Calibrated {calibration.created_at} over {calibration.n_tracks} tracks "
               f"({calibration.n_excluded} excluded as loops/samples/no data).")
    edges = ["min"] + [f"{c:.3f}" for c in cp] + ["max"]
    for band in range(1, ENERGY_BANDS + 1):
        click.echo(f"    Energy {band}: {edges[band - 1]} - {edges[band]}")


@tags.command("rules")
def tags_rules():
    """Show the tag vocabulary, thresholds, and the current energy calibration."""
    t = DEFAULT_THRESHOLDS
    click.echo("\nTag vocabulary (v1):\n")
    for cat in TAG_CATALOG:
        click.echo(f"  {cat.name} ({cat.select}-select): {', '.join(cat.tags)}")
    click.echo("\nRules (untuned starting points -- see `tags stats` before trusting them):\n")
    click.echo(f"  Energy      library quintile of mean waveform energy (loops/samples under "
               f"{t.min_calibration_duration_ms / 1000:.0f}s or BPM 0 excluded)")
    click.echo(f"  Intro       Long >= {t.long_intro_bars:g} bars, Short <= {t.short_intro_bars:g} bars; "
               "Vocal Intro = a vocal region overlaps the Intro phrase")
    click.echo(f"  Outro       Long >= {t.long_outro_bars:g} bars, Short <= {t.short_outro_bars:g} bars; "
               "Vocal Outro = a vocal region overlaps the Outro phrase")
    click.echo(f"  Vocals      Instrumental < {t.instrumental_max_pct:g}% coverage, "
               f"Vocal-Led >= {t.vocal_led_min_pct:g}% (between: untagged)")
    click.echo("  Check       No Key (missing/non-Camelot/unreadable), Tempo Varies, Grid Suspect "
               "(Rekordbox's own beat grid, free tier-1 check)")

    with TagStore() as store:
        calibration = store.latest_calibration()
    click.echo("\nEnergy calibration:\n")
    if calibration is None:
        click.echo("  Not calibrated yet -- run `djcues tags calibrate`.")
    else:
        _print_cutpoints(calibration)


@tags.command("calibrate")
@click.option("--no-cache", is_flag=True, help="Re-read every track's analysis instead of using cached features.")
def tags_calibrate(no_cache):
    """Compute library-wide energy bands from every track in the collection.

    Reads every track's analysis (~0.3-0.6s/track on first run -- minutes
    for a large library; cached afterwards, so re-runs are fast) and
    stores the energy quintile cutpoints locally. Read-only with respect
    to Rekordbox.
    """
    db = get_db()
    n = len(tag_analysis.library_contents(db))
    if n == 0:
        click.echo("Error: the library has no tracks.", err=True)
        raise SystemExit(1)
    click.echo(f"Analysing {n} track(s) (cached tracks are instant)...")

    with TagStore() as store:
        try:
            calibration, run = _progress(
                "Calibrating", n,
                lambda cb: tag_analysis.calibrate_library(db, store, use_cache=not no_cache, progress=cb),
            )
        except ValueError as e:
            click.echo(f"Error: {e}", err=True)
            raise SystemExit(1)

    click.echo(f"\n  {run.cache_hits} from cache, {len(run.features) - run.cache_hits} freshly analysed, "
               f"{len(run.failures)} unreadable.")
    no_analysis = _no_analysis_count(run.features)
    if no_analysis:
        click.echo(f"  {no_analysis} track(s) have no usable waveform/phrase analysis "
                   "(streaming-linked or never analysed in Rekordbox) and will get no energy/structure tags.")
    for title, err in run.failures[:10]:
        click.echo(f"    skipped: {title} ({err})", err=True)
    _print_cutpoints(calibration)
    counts = _band_counts(run.features, calibration.cutpoints)
    click.echo("    Tracks per band: " + "  ".join(f"E{b}={c}" for b, c in enumerate(counts, start=1)))
    click.echo("\nNext: `djcues tags stats` to check how common each tag is before trusting the thresholds.")


def _scope_contents(db, playlist_name: str | None):
    if playlist_name is None:
        return tag_analysis.library_contents(db), "whole library"
    playlist = find_playlist(playlist_name, db)
    if playlist is None:
        click.echo(f"Error: playlist '{playlist_name}' not found.", err=True)
        raise SystemExit(1)
    contents = tag_analysis.playlist_contents(db, playlist.ID)
    if not contents:
        click.echo(f"Error: no tracks found in playlist '{playlist_name}'.", err=True)
        raise SystemExit(1)
    return contents, playlist_name


def _cached_proposals(db, store, playlist_name):
    """Proposals from the feature cache only -- never parses ANLZ."""
    contents, scope = _scope_contents(db, playlist_name)
    run = tag_analysis.collect_features(contents, db, store, only_cached=True)
    if not run.features:
        click.echo("Error: no cached analysis for this scope -- run `djcues tags calibrate` first.", err=True)
        raise SystemExit(1)
    proposals = tag_analysis.derive_proposals(
        contents, run.features, calibration=store.latest_calibration(),
    )
    return proposals, run, scope


@tags.command("stats")
@click.option("--playlist", "playlist_name", default=None, help="Limit to one playlist (default: whole library).")
def tags_stats(playlist_name):
    """How common is each tag? (cached analysis only -- no ANLZ parsing)

    The check to run before trusting the thresholds: a descriptive tag
    (Mix In / Mix Out / Vocals) on nearly every track, or almost none,
    carries no information for set building and its rule needs
    retuning. Percentages are of the tracks where that category could
    actually be evaluated -- tracks with no analysis at all (e.g.
    streaming-linked) are reported separately, not counted as "untagged".
    Energy bands are ~equal by construction; Check tags are warnings and
    are expected to be rare, so neither is range-checked.
    """
    db = get_db()
    with TagStore() as store:
        calibration = store.latest_calibration()
        proposals, run, scope = _cached_proposals(db, store, playlist_name)

    total = len(proposals)
    click.echo(f"\n{'=' * 60}\n  Tag prevalence: {scope} ({total} tracks analysed)\n{'=' * 60}")
    if run.uncached:
        click.echo(f"  ({run.uncached} track(s) have no cached analysis and are not counted -- "
                   "run `djcues tags calibrate`)")
    if calibration is None:
        click.echo("  (not calibrated: Energy tags unavailable -- run `djcues tags calibrate`)")

    counts: Counter[str] = Counter()
    skips: Counter[tuple[str, str]] = Counter()
    skipped_per_cat: Counter[str] = Counter()
    for p in proposals:
        for d in p.tags.decisions:
            counts[d.tag] += 1
        for sk in p.tags.skipped:
            skips[(sk.category, sk.reason)] += 1
            skipped_per_cat[sk.category] += 1

    for cat in TAG_CATALOG:
        evaluated = total - skipped_per_cat[cat.name]
        click.echo(f"\n  {cat.name}  (evaluated on {evaluated} track(s))")
        range_checked = cat.name in _RANGE_CHECKED_CATEGORIES
        for tag in cat.tags:
            n = counts[tag]
            pct = 100.0 * n / evaluated if evaluated else 0.0
            flag = ""
            if range_checked and evaluated and not (_HEALTHY_MIN_PCT <= pct <= _HEALTHY_MAX_PCT):
                flag = f"   <- outside {_HEALTHY_MIN_PCT:g}-{_HEALTHY_MAX_PCT:g}%"
            click.echo(f"    {tag:<14s} {n:>5d}  {pct:5.1f}%{flag}")
        if cat.name == "Check":
            click.echo("    (warnings -- expected to be rare; not range-checked)")
    if skips:
        click.echo("\n  Categories skipped (data missing):")
        for (cat, reason), n in sorted(skips.items()):
            click.echo(f"    {cat:<8s} {_SKIP_LABELS.get(reason, reason):<26s} {n}")


def _match_track(contents, query: str):
    q = query.lower()
    return [c for c in contents if q in (c.Title or "").lower()]


def _grouped_line(track_tags) -> str:
    parts = []
    for cat in CATEGORY_NAMES:
        chosen = track_tags.tags_in(cat)
        if chosen:
            parts.append(" + ".join(chosen))
    return "  ·  ".join(parts) if parts else "(no tags)"


def _proposal_to_session(proposals, playlist_name, calibration) -> dict:
    tracks = {}
    for p in proposals:
        by_cat: dict[str, list[dict]] = {}
        for d in p.tags.decisions:
            by_cat.setdefault(d.category, []).append(
                {"name": d.tag, "evidence": d.evidence, "status": "pending"}
            )
        tracks[p.track_id] = {
            "title": p.title,
            "artist": p.artist,
            "status": "pending",
            "tags": by_cat,
            "skipped": {s.category: s.reason for s in p.tags.skipped},
        }
    return {
        "version": 1,
        "kind": "tags",
        "created_at": datetime.now().replace(microsecond=0).isoformat(),
        "playlist": playlist_name,
        "calibration_id": calibration.id if calibration else None,
        "thresholds": asdict(DEFAULT_THRESHOLDS),
        "tracks": tracks,
    }


@tags.command("propose")
@click.argument("playlist_name")
@click.argument("track_name", required=False)
@click.option("--all", "all_tracks", is_flag=True, help="Propose tags for every track in the playlist.")
@click.option(
    "--categories", default=None,
    help=f"Comma-separated categories to evaluate (default: all). One of: {', '.join(CATEGORY_NAMES)}.",
)
@click.option("--evidence", is_flag=True, help="Show why each tag was chosen (always on for a single track).")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default=None,
              help="Also save the proposal as a session JSON file.")
@click.option("--no-cache", is_flag=True, help="Re-read analysis instead of using cached features.")
def tags_propose(playlist_name, track_name, all_tracks, categories, evidence, out_path, no_cache):
    """Propose tags for tracks in a playlist. Read-only -- writes nothing to Rekordbox."""
    wanted = None
    if categories:
        wanted = [c.strip() for c in categories.split(",") if c.strip()]
        unknown = [c for c in wanted if c not in CATEGORY_NAMES]
        if unknown:
            click.echo(f"Error: unknown category {', '.join(unknown)}. Choose from: {', '.join(CATEGORY_NAMES)}.", err=True)
            raise SystemExit(1)
    if not track_name and not all_tracks:
        click.echo("Error: provide a track name or use --all.", err=True)
        raise SystemExit(1)

    db = get_db()
    contents, scope = _scope_contents(db, playlist_name)
    if track_name:
        matches = _match_track(contents, track_name)
        if not matches:
            click.echo(f"Error: no track matching '{track_name}' in '{playlist_name}'.", err=True)
            raise SystemExit(1)
        if len(matches) > 1:
            click.echo(f"Error: '{track_name}' matches {len(matches)} tracks -- be more specific:", err=True)
            for c in matches[:10]:
                click.echo(f"  {c.Title}", err=True)
            raise SystemExit(1)
        contents = matches
        evidence = True

    with TagStore() as store:
        calibration = store.latest_calibration()
        if calibration is None:
            click.echo("Note: energy isn't calibrated yet (run `djcues tags calibrate`) -- "
                       "no Energy tags will be proposed.", err=True)
        proposals, run = _progress(
            "Analysing", len(contents),
            lambda cb: tag_analysis.propose_tags(
                contents, db, store, calibration=calibration, categories=wanted,
                use_cache=not no_cache, progress=cb,
            ),
        )

    for title, err in run.failures:
        click.echo(f"  Skipping {title} ({err})", err=True)

    click.echo(f"\n{'=' * 60}\n  Proposed tags: {scope}\n{'=' * 60}")
    totals: Counter[str] = Counter()
    for i, p in enumerate(proposals, start=1):
        click.echo(f"\n  {i:>3d}. {p.title} — {p.artist}")
        if evidence:
            for d in p.tags.decisions:
                click.echo(f"        [{d.category}] {d.tag}  —  {d.evidence}")
            if not p.tags.decisions:
                click.echo("        (no tags)")
            for s in p.tags.skipped:
                click.echo(f"        [{s.category}] skipped: {_SKIP_LABELS.get(s.reason, s.reason)}")
        else:
            click.echo(f"        {_grouped_line(p.tags)}")
        for d in p.tags.decisions:
            totals[d.tag] += 1

    click.echo(f"\n  {len(proposals)} track(s); " + (
        "  ".join(f"{tag}={n}" for tag, n in sorted(totals.items())) if totals else "no tags proposed"
    ))

    if out_path:
        session = _proposal_to_session(proposals, playlist_name, calibration)
        Path(out_path).write_text(json.dumps(session, indent=2), encoding="utf-8")
        click.echo(f"  Saved session: {out_path}  (nothing has been written to Rekordbox)")


# Which underlying number defines a tag, so `sample` can pick examples
# spread from one end of the tag's range to the other.
_TAG_METRIC = {
    **{f"Energy {i}": "mean_energy" for i in range(1, ENERGY_BANDS + 1)},
    "Long Intro": "intro_bars", "Short Intro": "intro_bars",
    "Long Outro": "outro_bars", "Short Outro": "outro_bars",
    "Instrumental": "vocal_pct", "Vocal-Led": "vocal_pct",
}


def _spread_indices(m: int, n: int) -> list[int]:
    """n indices spread evenly over range(m) (first, last and between)."""
    if m <= n:
        return list(range(m))
    if n == 1:
        return [m // 2]
    return sorted({round(i * (m - 1) / (n - 1)) for i in range(n)})


@tags.command("sample")
@click.argument("tag")
@click.option("-n", "count", default=3, show_default=True, help="How many example tracks to show.")
@click.option("--playlist", "playlist_name", default=None, help="Limit to one playlist (default: whole library).")
def tags_sample(tag, count, playlist_name):
    """Pick example tracks carrying TAG, spread across its range -- to check by ear.

    There is no automated ground truth for these tags: whether "Energy 5"
    really sounds like peak-time material is a judgement only listening
    can make. This lists tracks from the low end, middle and high end of
    the tag's underlying measurement so the check is representative.
    """
    from djcues.tagging import category_of

    if category_of(tag) is None:
        all_tags = [t for c in TAG_CATALOG for t in c.tags]
        click.echo(f"Error: unknown tag '{tag}'. Choose from: {', '.join(all_tags)}.", err=True)
        raise SystemExit(1)

    db = get_db()
    with TagStore() as store:
        proposals, run, scope = _cached_proposals(db, store, playlist_name)
        feats = run.features

    matching = [(p, d) for p in proposals for d in p.tags.decisions if d.tag == tag]
    if not matching:
        click.echo(f"No tracks in {scope} carry '{tag}'.")
        return

    metric = _TAG_METRIC.get(tag)
    if metric:
        matching.sort(key=lambda pd: getattr(feats[pd[0].track_id], metric))
    else:
        matching.sort(key=lambda pd: pd[0].title.lower())

    click.echo(f"\n{len(matching)} track(s) in {scope} carry '{tag}'. Examples to listen to"
               + (f" (spread by {metric.replace('_', ' ')}):" if metric else ":"))
    for idx in _spread_indices(len(matching), count):
        p, d = matching[idx]
        click.echo(f"\n  {p.title} — {p.artist}\n      {d.evidence}")
