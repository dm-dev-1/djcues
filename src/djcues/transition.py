"""Transition point suggestions -- for every adjacent track pair in one
playlist, suggests where in track A to mix out and where in track B to
mix in, using the same phrase-boundary data clash.py's vocal-clash
zones and strategy.py's pad G/A anchors already trust (strategy.
tail_zone()/head_zone(), promoted out of clash.py so both modules share
one copy of the same zone concept rather than each inventing their own).

Pure functions -- no I/O, no ``click.echo`` -- so they're usable both
from the CLI and from tests without a live Rekordbox connection.
Mirrors clash.py's/flow.py's own shape for the same reason.

Read-only suggestion, same category as audit.py/clash.py: never writes
to the database, never places a real cue.

NOTE on the two anchors:
- Mix-out point in A is strategy.tail_zone(track_a)[0] -- the same
  Outro-phrase-start anchor pad G and clash.py's tail zone already use.
- Mix-in point in B is track_b.beat_grid.beat_to_ms(1) -- pad A's own
  "First Beat" formula, NOT the literal 0.0 that clash.py's head_zone()
  starts from. head_zone()'s 0.0 start is correct for clash's purpose
  (a risk *window* that should include any lead-in silence); it is the
  wrong anchor for "the point a DJ actually cues B in from," which is
  beat 1 and can sit meaningfully after real lead-in silence.

NOTE on ordering: suggest_transitions() trusts `tracks`' own list order
as the playlist's real current order -- it does not re-sort. Getting
that order right is entirely the CALLER's responsibility, exactly like
clash.find_vocal_clashes() (see that module's own docstring for the
full db.load_playlist_tracks()-does-not-sort explanation and the
reorder-by-id pattern every call site must use).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from djcues.strategy import (
    DEFAULT_MIN_VOCAL_REGION_MS,
    find_vocal_regions,
    head_zone,
    regions_overlapping,
    tail_zone,
)

if TYPE_CHECKING:
    from djcues.models import Track, VocalRegion


@dataclass(frozen=True)
class TransitionSuggestion:
    """One adjacent pair's suggested transition. position_a/position_b
    are 1-based positions in the input tracks sequence (the playlist's
    real current order), position_b always position_a + 1 -- same
    convention as clash.ClashFinding.

    a_tail_available_ms/b_head_available_ms are the two raw inputs to
    overlap_ms's min() -- kept as separate fields (not collapsed into
    overlap_ms alone) so a consumer can explain *why* a window is short,
    not just that it is. limiting_side names which one was the
    bottleneck ("a_tail" | "b_head").

    vocal_regions_a/vocal_regions_b are find_vocal_regions() output
    filtered to the actual suggested overlap window (not the full tail/
    head zone -- the real blend window can be shorter than either zone
    when the other side is the limiting factor). None means "couldn't
    check" (the track has no vocal_track data at all) -- deliberately
    distinct from [] ("checked, genuinely clear"), same honest-missing-
    data convention as DropRefinement.sustain_signature.
    """

    track_a: "Track"
    track_b: "Track"
    position_a: int
    position_b: int
    mix_out_ms: float
    mix_in_ms: float
    a_tail_available_ms: float
    b_head_available_ms: float
    overlap_ms: float
    limiting_side: str  # "a_tail" | "b_head"
    vocal_regions_a: list["VocalRegion"] | None
    vocal_regions_b: list["VocalRegion"] | None


@dataclass(frozen=True)
class UnscorableTrack:
    """A track that can't anchor a transition suggestion on either side
    of a pair -- missing phrases, or missing an Intro- or Outro-labeled
    phrase. Checked uniformly regardless of which side (A or B) the
    track would appear on, same simplifying convention clash.
    UnscorableTrack already uses and justifies.

    Deliberately its own class here, not imported from clash.py or
    shared via strategy.py, despite the identical (track, reason)
    shape -- flow.UnscoredTrack's own docstring gives the reason this
    codebase already established: a caller importing multiple sibling
    modules together (exactly what cli.py/server.py do) makes a shared
    name a real collision hazard, not a hypothetical one.

    Narrower reason set than clash.UnscorableTrack's 4: this feature's
    base suggestion doesn't need vocal_track data at all (only the
    optional vocal_regions_a/b annotation does, which degrades to None
    instead of blocking the suggestion -- see TransitionSuggestion's
    own docstring), so "no_vocal_data" isn't a gate here.
    """

    track: "Track"
    reason: str  # "no_phrase_data" | "no_intro_phrase" | "no_outro_phrase"


@dataclass(frozen=True)
class TransitionResult:
    """suggest_transitions()'s full result. scanned_pairs is every
    adjacent pair in the input (len(tracks) - 1, 0 if fewer than 2
    tracks) -- unconditional, same convention as clash.ClashResult.
    scanned_pairs/audit.AuditResult.scanned. Unlike clash's findings
    (conditional -- only populated when a bad condition fires),
    suggestions is unconditional: every scorable adjacent pair gets
    exactly one suggestion, since every pair has *some* mix-point worth
    proposing. This gives a clean, testable invariant: len(suggestions)
    plus the number of pairs touching an unscorable track always equals
    scanned_pairs.
    """

    scanned_pairs: int
    suggestions: list[TransitionSuggestion] = field(default_factory=list)
    unscorable: list[UnscorableTrack] = field(default_factory=list)


def _is_unscorable(track: "Track") -> str | None:
    """Reason string if track can't anchor a transition suggestion at
    all, else None. See UnscorableTrack's own docstring for why this
    reason set is narrower than clash._is_unscorable's."""
    if not track.phrases:
        return "no_phrase_data"
    if not any(p.label == "Intro" for p in track.phrases):
        return "no_intro_phrase"
    if not any(p.label == "Outro" for p in track.phrases):
        return "no_outro_phrase"
    return None


def suggest_transitions(
    tracks: list["Track"], *, min_vocal_region_ms: float = DEFAULT_MIN_VOCAL_REGION_MS
) -> TransitionResult:
    """Suggest a mix-out/mix-in point for every adjacent pair in tracks
    (the playlist's real current order -- see this module's own
    docstring for the ordering contract callers must uphold).

    Algorithm:
    1. A track is unscorable (excluded from both sides of any pair) if
       it lacks phrases, lacks an Intro-labeled phrase, or lacks an
       Outro-labeled phrase -- checked uniformly regardless of which
       side it would appear on (see UnscorableTrack's own docstring).
    2. For each adjacent pair (tracks[i], tracks[i+1]) where NEITHER is
       unscorable:
       - mix_out_ms = tail_zone(track_a)[0] (the Outro-phrase start).
       - mix_in_ms = track_b.beat_grid.beat_to_ms(1) (pad A's own
         "First Beat" formula -- NOT head_zone(track_b)'s 0.0 start,
         see this module's own docstring for why).
       - a_tail_available_ms = tail_zone(track_a)[1] - mix_out_ms.
       - b_head_available_ms = head_zone(track_b)[1] - mix_in_ms,
         clamped to >= 0.0 -- measured from the REAL mix-in point, not
         from head_zone's raw 0.0 start, so a track whose first beat
         already lands at or after its own Intro-phrase end correctly
         yields 0 available headroom rather than a negative number.
       - overlap_ms = max(0.0, min(a_tail_available_ms,
         b_head_available_ms)); limiting_side records which side was
         the bottleneck. No arbitrary duration cap -- see this module's
         docstring / clash.py's own docstring for why a fixed window
         was rejected in favor of already-computed phrase anchors.
       - vocal_regions_a/b: find_vocal_regions() filtered (via
         regions_overlapping()) to the actual [mix_point, mix_point +
         overlap_ms) window on each side, or None if that track has no
         vocal_track data at all.
    3. scanned_pairs counts every adjacent pair in the input, including
       ones touching an unscorable track (see TransitionResult's
       docstring).

    Positions are 1-based indices into `tracks` as given.
    """
    unscorable: list[UnscorableTrack] = []
    reasons: dict[int, str] = {}
    for i, track in enumerate(tracks):
        reason = _is_unscorable(track)
        if reason is not None:
            reasons[i] = reason
            unscorable.append(UnscorableTrack(track=track, reason=reason))

    suggestions: list[TransitionSuggestion] = []
    scanned_pairs = max(0, len(tracks) - 1)
    for i in range(len(tracks) - 1):
        if i in reasons or (i + 1) in reasons:
            continue
        track_a, track_b = tracks[i], tracks[i + 1]

        tail_start, tail_end = tail_zone(track_a)
        mix_out_ms = tail_start
        a_tail_available_ms = tail_end - mix_out_ms

        _head_start, head_end = head_zone(track_b)
        mix_in_ms = track_b.beat_grid.beat_to_ms(1)
        b_head_available_ms = max(0.0, head_end - mix_in_ms)

        overlap_ms = max(0.0, min(a_tail_available_ms, b_head_available_ms))
        limiting_side = "a_tail" if a_tail_available_ms <= b_head_available_ms else "b_head"

        vocal_regions_a = (
            regions_overlapping(
                find_vocal_regions(track_a, min_vocal_region_ms),
                mix_out_ms, mix_out_ms + overlap_ms,
            )
            if track_a.vocal_track else None
        )
        vocal_regions_b = (
            regions_overlapping(
                find_vocal_regions(track_b, min_vocal_region_ms),
                mix_in_ms, mix_in_ms + overlap_ms,
            )
            if track_b.vocal_track else None
        )

        suggestions.append(TransitionSuggestion(
            track_a=track_a, track_b=track_b, position_a=i + 1, position_b=i + 2,
            mix_out_ms=mix_out_ms, mix_in_ms=mix_in_ms,
            a_tail_available_ms=a_tail_available_ms, b_head_available_ms=b_head_available_ms,
            overlap_ms=overlap_ms, limiting_side=limiting_side,
            vocal_regions_a=vocal_regions_a, vocal_regions_b=vocal_regions_b,
        ))

    return TransitionResult(scanned_pairs=scanned_pairs, suggestions=suggestions, unscorable=unscorable)
