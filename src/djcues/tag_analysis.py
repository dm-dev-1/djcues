"""I/O orchestration for smart tagging: reads tracks from Rekordbox,
extracts + caches their features, runs library-wide calibration and
produces tag proposals.

The pure rules live in tagging.py and the sqlite store in tag_store.py;
this module is the thin layer that connects them to a live
Rekordbox6Database -- shared by the CLI (`djcues tags ...`) and, later,
the dashboard's job runner, so neither re-implements the load/cache loop.

Read-only with respect to Rekordbox: nothing here writes to master.db.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from djcues import db as db_module
from djcues.tag_store import Calibration, TagStore
from djcues.tagging import (
    DEFAULT_THRESHOLDS,
    AnalysisFeatures,
    TagThresholds,
    TrackTags,
    compute_energy_cutpoints,
    derive_tags,
    extract_features,
    is_calibration_eligible,
    key_status,
)

logger = logging.getLogger(__name__)

# (done, total, from_cache) -- called after every track.
ProgressFn = Callable[[int, int, bool], None]

_COMMIT_EVERY = 50


@dataclass
class FeatureRun:
    """collect_features()'s result. failures is (title, error) for any
    track whose analysis couldn't be read -- never fatal to the run."""

    features: dict[str, AnalysisFeatures] = field(default_factory=dict)
    failures: list[tuple[str, str]] = field(default_factory=list)
    cache_hits: int = 0
    # Only ever non-zero with only_cached=True: tracks with no valid
    # cache entry that were left alone instead of parsed.
    uncached: int = 0


def playlist_contents(db: Any, playlist_id: Any) -> list[Any]:
    """DjmdContent rows for one playlist, in real TrackNo order."""
    songs = list(db.get_playlist_songs(PlaylistID=playlist_id))
    songs.sort(key=lambda s: (s.TrackNo or 0))
    return [s.Content for s in songs if s.Content is not None]


def library_contents(db: Any) -> list[Any]:
    return list(db.get_content())


def collect_features(
    contents: list[Any],
    db: Any,
    store: TagStore,
    *,
    use_cache: bool = True,
    only_cached: bool = False,
    progress: ProgressFn | None = None,
) -> FeatureRun:
    """Features for every content row. A cache hit (same ANLZ
    fingerprint) skips the load_track() parse entirely; a miss loads the
    track, extracts, and stores. One bad track is recorded in .failures
    and the run continues -- same per-track-failure philosophy as
    db.load_playlist_tracks().

    only_cached=True never parses: a miss is counted in .uncached and
    skipped, for read-only reports (stats/sample) that should be instant
    and never trigger a multi-minute library scan by accident."""
    run = FeatureRun()
    total = len(contents)
    for i, content in enumerate(contents, start=1):
        track_id = str(content.ID)
        from_cache = False
        try:
            fingerprint = db_module.fingerprint_anlz(content, db=db)
            features = store.get_features(track_id, fingerprint) if use_cache else None
            if features is not None:
                from_cache = True
                run.cache_hits += 1
            elif only_cached:
                run.uncached += 1
                features = None
            else:
                track = db_module.load_track(content, db=db)
                raw = db_module.extract_raw_beat_grid(content, db=db)
                features = extract_features(track, raw)
                store.put_features(features, fingerprint, commit=False)
            if features is not None:
                run.features[track_id] = features
        except Exception as e:  # noqa: BLE001 -- recorded, not swallowed
            logger.warning("Skipping track %s: %s", getattr(content, "Title", track_id), e)
            run.failures.append((getattr(content, "Title", None) or track_id, str(e)))
        if i % _COMMIT_EVERY == 0:
            store.commit()
        if progress is not None:
            progress(i, total, from_cache)
    store.commit()
    return run


def calibrate_library(
    db: Any,
    store: TagStore,
    *,
    thresholds: TagThresholds = DEFAULT_THRESHOLDS,
    use_cache: bool = True,
    progress: ProgressFn | None = None,
) -> tuple[Calibration, FeatureRun]:
    """Compute and store library-wide energy cutpoints over every
    eligible track (not a loop/sample, has energy data). Raises
    ValueError if too few tracks are eligible (tagging.MIN_CALIBRATION_TRACKS)."""
    run = collect_features(library_contents(db), db, store, use_cache=use_cache, progress=progress)
    eligible = [f for f in run.features.values() if is_calibration_eligible(f, thresholds)]
    cutpoints = compute_energy_cutpoints(f.mean_energy for f in eligible)
    n_excluded = len(run.features) - len(eligible)
    store.save_calibration(cutpoints, len(eligible), n_excluded, asdict(thresholds))
    calibration = store.latest_calibration()
    assert calibration is not None
    return calibration, run


@dataclass
class TrackProposal:
    """One track's proposed tags plus the identity needed to show them."""

    track_id: str
    title: str
    artist: str
    tags: TrackTags


def propose_tags(
    contents: list[Any],
    db: Any,
    store: TagStore,
    *,
    calibration: Calibration | None,
    thresholds: TagThresholds = DEFAULT_THRESHOLDS,
    categories: list[str] | None = None,
    use_cache: bool = True,
    progress: ProgressFn | None = None,
) -> tuple[list[TrackProposal], FeatureRun]:
    """Tag proposals for the given content rows, in input order. Key
    status comes from the cheap TrackSummary fields (Track carries no
    key), so it's always fresh even when ANLZ features are cached."""
    run = collect_features(contents, db, store, use_cache=use_cache, progress=progress)
    proposals = derive_proposals(
        contents, run.features, calibration=calibration, thresholds=thresholds, categories=categories,
    )
    return proposals, run


def derive_proposals(
    contents: list[Any],
    features_by_id: dict[str, AnalysisFeatures],
    *,
    calibration: Calibration | None,
    thresholds: TagThresholds = DEFAULT_THRESHOLDS,
    categories: list[str] | None = None,
) -> list[TrackProposal]:
    """Rules-only step: tag proposals for every content row that has
    features, in input order. Pure given its inputs (no ANLZ reads) --
    key status comes from the cheap TrackSummary fields, so it's always
    fresh even when the features themselves came from the cache."""
    cutpoints = calibration.cutpoints if calibration else None
    proposals: list[TrackProposal] = []
    for content in contents:
        features = features_by_id.get(str(content.ID))
        if features is None:
            continue
        summary = db_module.summarize_content(content)
        tags = derive_tags(
            features,
            key_reason=key_status(summary),
            key_raw=summary.key,
            cutpoints=cutpoints,
            thresholds=thresholds,
            categories=categories,
        )
        proposals.append(TrackProposal(
            track_id=str(content.ID), title=summary.title, artist=summary.artist, tags=tags,
        ))
    return proposals
