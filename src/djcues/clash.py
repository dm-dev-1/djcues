"""Vocal-clash detection -- flags adjacent track pairs in one playlist
where track A still has vocals near its end and track B already has
vocals near its start, a real risk of two vocals clashing at once when
mixed together. Reuses strategy.find_vocal_regions() (itself a shared
generalization of the vocal-onset-detection logic CueStrategy.propose()'s
inline C-slot detection and agentic._summarize_vocal_onsets() already
use -- see that function's own docstring) rather than inventing a new
vocal-presence signal.

Pure functions -- no I/O, no ``click.echo`` -- so they're usable both
from the CLI and from tests without a live Rekordbox connection.
Mirrors flow.py's/audit.py's own shape for the same reason.

Read-only diagnostic report, same category as audit.py: flags problems,
never suggests a new order, never writes to the database.

Zones are anchored on real PSSI phrase labels, not a fixed time window
-- a track's "tail" is from its (first) Outro-labeled phrase's start to
the track's end; a track's "head" is from 0ms to the end of its (first)
Intro-labeled phrase. Rekordbox's own phrase structure already marks a
meaningful, per-track-appropriate boundary; a fixed "last/first 30
seconds" window was considered and rejected as an arbitrary magic
number when a better, already-computed anchor exists (live-verified:
115/115 tracks in one real playlist had both an Intro- and an Outro-
labeled phrase -- the near-universal case, but still handled when
missing, see UnscorableTrack below).

NOTE on ordering: find_vocal_clashes() trusts `tracks`' own list order
as the playlist's real current order -- it does not re-sort. Getting
that order right is entirely the CALLER's responsibility:
db.load_playlist_tracks() alone does NOT guarantee TrackNo order
(confirmed: db.get_playlist_songs() has no ORDER BY, unlike
db.list_playlist_tracks()'s/db.find_playlist_song_entries()'s own
explicit TrackNo sort) -- callers (cli.py's `clash` command, server.py's
_run_clash_job) reorder load_playlist_tracks()'s result to match
list_playlist_tracks()'s TrackNo-sorted id sequence before calling this
module. See either call site's own comment for the exact reordering
step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from djcues.strategy import DEFAULT_MIN_VOCAL_REGION_MS, find_vocal_regions

if TYPE_CHECKING:
    from djcues.models import Track, VocalRegion


@dataclass(frozen=True)
class ClashFinding:
    """One adjacent pair flagged as a vocal-clash risk. position_a/
    position_b are 1-based positions in the input tracks sequence (the
    playlist's real current order) -- position_b is always position_a + 1
    (adjacency is the whole premise), kept as two explicit fields rather
    than one "pair index" so a report can label both sides without the
    reader re-deriving the second. regions_a/regions_b are the specific
    find_vocal_regions() regions that overlap track_a's tail zone /
    track_b's head zone respectively (never empty -- that's what makes
    this a finding)."""

    track_a: "Track"
    track_b: "Track"
    position_a: int
    position_b: int
    regions_a: list["VocalRegion"]
    regions_b: list["VocalRegion"]


@dataclass(frozen=True)
class UnscorableTrack:
    """A track that can't be checked at all, on either side of a pair --
    missing phrases, missing vocal_track, or missing an Intro- or Outro-
    labeled phrase. Checked uniformly regardless of which side (A or B)
    the track would appear on -- a small, deliberate over-exclusion in
    the rare case only one of the two needed phrases is missing, in
    exchange for much simpler code/tests."""

    track: "Track"
    reason: str  # "no_phrase_data" | "no_vocal_data" | "no_intro_phrase" | "no_outro_phrase"


@dataclass(frozen=True)
class ClashResult:
    """find_vocal_clashes()'s full result. scanned_pairs is every
    adjacent pair in the input (len(tracks) - 1, 0 if fewer than 2
    tracks) -- unconditional, mirroring audit.AuditResult.scanned's own
    convention of counting the input regardless of usability; a pair
    touching an unscorable track is included in this count but can
    never produce a finding."""

    scanned_pairs: int
    findings: list[ClashFinding] = field(default_factory=list)
    unscorable: list[UnscorableTrack] = field(default_factory=list)


def _is_unscorable(track: "Track") -> str | None:
    """Reason string if track can't be checked at all, else None. Checks
    all four conditions uniformly (not side-dependent) -- see
    UnscorableTrack's own docstring for why."""
    if not track.phrases:
        return "no_phrase_data"
    if not track.vocal_track:
        return "no_vocal_data"
    if not any(p.label == "Intro" for p in track.phrases):
        return "no_intro_phrase"
    if not any(p.label == "Outro" for p in track.phrases):
        return "no_outro_phrase"
    return None


def _tail_zone(track: "Track") -> tuple[float, float]:
    """track's tail zone: from its (first) Outro-labeled phrase's start
    to the track's end. Caller must have already confirmed an Outro
    phrase exists (_is_unscorable)."""
    outro = next(p for p in track.phrases if p.label == "Outro")
    return outro.position_ms, track.duration_ms


def _head_zone(track: "Track") -> tuple[float, float]:
    """track's head zone: from 0ms to the end of its (first) Intro-
    labeled phrase. Caller must have already confirmed an Intro phrase
    exists (_is_unscorable)."""
    intro = next(p for p in track.phrases if p.label == "Intro")
    return 0.0, intro.position_ms + intro.duration_ms


def _regions_overlapping(
    regions: list["VocalRegion"], zone_start: float, zone_end: float
) -> list["VocalRegion"]:
    """Regions from `regions` that overlap [zone_start, zone_end) --
    simple interval-overlap check."""
    return [r for r in regions if r.start_ms < zone_end and r.end_ms > zone_start]


def find_vocal_clashes(
    tracks: list["Track"], *, min_vocal_region_ms: float = DEFAULT_MIN_VOCAL_REGION_MS
) -> ClashResult:
    """Scan every adjacent pair in tracks (the playlist's real current
    order -- see this module's own docstring for the ordering contract
    callers must uphold) and flag pairs where track A still has vocals
    overlapping its tail zone AND track B already has vocals overlapping
    its head zone.

    Algorithm:
    1. A track is unscorable (excluded from both sides of any pair) if
       it lacks phrases, lacks vocal_track, lacks an Intro-labeled
       phrase, or lacks an Outro-labeled phrase -- checked uniformly
       regardless of which side it would appear on.
    2. For each adjacent pair (tracks[i], tracks[i+1]) where NEITHER is
       unscorable: find_vocal_regions(a) filtered to those overlapping
       a's tail zone, find_vocal_regions(b) filtered to those overlapping
       b's head zone. Both non-empty -> a ClashFinding.
    3. scanned_pairs counts every adjacent pair in the input, including
       ones touching an unscorable track (see ClashResult's docstring).

    Positions are 1-based indices into `tracks` as given.
    """
    unscorable: list[UnscorableTrack] = []
    reasons: dict[int, str] = {}
    for i, track in enumerate(tracks):
        reason = _is_unscorable(track)
        if reason is not None:
            reasons[i] = reason
            unscorable.append(UnscorableTrack(track=track, reason=reason))

    findings: list[ClashFinding] = []
    scanned_pairs = max(0, len(tracks) - 1)
    for i in range(len(tracks) - 1):
        if i in reasons or (i + 1) in reasons:
            continue
        track_a, track_b = tracks[i], tracks[i + 1]
        tail_start, tail_end = _tail_zone(track_a)
        head_start, head_end = _head_zone(track_b)
        regions_a = _regions_overlapping(
            find_vocal_regions(track_a, min_vocal_region_ms), tail_start, tail_end
        )
        regions_b = _regions_overlapping(
            find_vocal_regions(track_b, min_vocal_region_ms), head_start, head_end
        )
        if regions_a and regions_b:
            findings.append(ClashFinding(
                track_a=track_a, track_b=track_b, position_a=i + 1, position_b=i + 2,
                regions_a=regions_a, regions_b=regions_b,
            ))

    return ClashResult(scanned_pairs=scanned_pairs, findings=findings, unscorable=unscorable)
