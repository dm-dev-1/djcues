"""Energy-flow set ordering -- suggests a track play order for one
playlist that builds energy toward a peak and then cools down for the
finale (a "peak-then-cooldown" arc), reusing strategy.py's own
per-phrase waveform energy measurement rather than inventing a second
one.

Pure functions -- no I/O, no ``click.echo`` -- so they're usable both
from the CLI and from tests without a live Rekordbox connection.
Mirrors harmony.py's/audit.py's own shape for the same reason.

NOTE on naming: "energy" here means strategy.compute_phrase_energy()'s
mean waveform height (0.0-1.0) -- a track's real loudness/intensity
profile, read straight from Rekordbox's own PWV5 color waveform data.
This is unrelated to harmony.py's "energy_boost"/"energy_drop", which
name a Camelot Wheel key *relationship* (+-1 around the wheel), not a
waveform measurement -- same English word, two different concepts in
this codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from djcues.strategy import compute_phrase_energy

if TYPE_CHECKING:
    from djcues.models import Track

# Fraction of scorable tracks (excluding the opener) reserved for the
# post-peak cooldown close -- tunable, mirrors harmony.py's
# DEFAULT_BPM_TOLERANCE_PCT convention (a named default reused as the
# keyword parameter's own default, not a bare literal).
DEFAULT_COOLDOWN_FRACTION = 0.15


@dataclass(frozen=True)
class TrackEnergy:
    """One scorable track's energy summary. mean_energy is the
    duration-weighted mean of compute_phrase_energy(track)'s per-phrase
    values -- sum(energy_i * duration_i) / sum(duration_i), not a naive
    per-phrase average: real phrase durations range from ~1.5s to ~80s,
    so an unweighted mean would let a few short phrases outweigh the
    long ones that actually dominate the track. suggest_energy_flow()
    sorts on mean_energy; peak_energy (the single highest per-phrase
    value) is informational only, never used for ordering."""

    track: "Track"
    mean_energy: float
    peak_energy: float


@dataclass(frozen=True)
class UnscoredTrack:
    """A track that couldn't be scored at all -- no phrase data, no
    waveform data, or compute_phrase_energy() returned []. Named
    distinctly from harmony.ExcludedTrack/audit.UnusableKeyTrack (not
    reused) since a caller could plausibly import flow alongside
    harmony/audit and a name collision would be a real hazard."""

    track: "Track"
    reason: str  # "no_phrase_data" | "no_waveform_data"


@dataclass(frozen=True)
class FlowResult:
    """suggest_energy_flow()'s full result. ordered_tracks is the final
    suggested play order (build phase, then cooldown phase).
    cooldown_start_index is the index into ordered_tracks where the
    cooldown phase begins (== len(ordered_tracks) when there is no
    cooldown phase, e.g. fewer than 3 scorable tracks) -- callers don't
    need to re-derive k. unscored holds every track that couldn't be
    scored, in input order, mirroring harmony.SuggestionResult.excluded/
    audit.AuditResult.unusable_keys."""

    ordered_tracks: list[TrackEnergy]
    cooldown_start_index: int
    unscored: list[UnscoredTrack] = field(default_factory=list)


def compute_track_energy(track: "Track") -> TrackEnergy | None:
    """Duration-weighted mean/peak energy for one track, or None if it
    can't be scored (compute_phrase_energy(track) returned [] -- no
    phrase data, no waveform data, or duration_ms <= 0)."""
    phrase_energy = compute_phrase_energy(track)
    if not phrase_energy:
        return None
    total_duration_ms = sum(p.duration_ms for p, _e in phrase_energy)
    if total_duration_ms <= 0:
        # Defensive only -- every phrase would need zero duration
        # despite track.duration_ms > 0 for this to trigger. Not
        # observed in any real track checked this session, but this
        # codebase has a real precedent for a ZeroDivisionError-shaped
        # bug from an unguarded duration/BPM (BeatGrid.ms_per_beat on a
        # 0 BPM track), so it's guarded here too.
        return None
    mean_energy = sum(e * p.duration_ms for p, e in phrase_energy) / total_duration_ms
    peak_energy = max(e for _p, e in phrase_energy)
    return TrackEnergy(track=track, mean_energy=mean_energy, peak_energy=peak_energy)


def suggest_energy_flow(
    tracks: list["Track"],
    *,
    cooldown_fraction: float = DEFAULT_COOLDOWN_FRACTION,
) -> FlowResult:
    """Suggest a "peak-then-cooldown" play order for one playlist's
    already-loaded Track objects (not TrackSummary -- this needs real
    phrase/waveform data).

    Algorithm:
    1. Score every track via compute_track_energy(); unscorable tracks
       go to .unscored with a reason, excluded from ordering entirely.
    2. Sort scorable tracks ascending by mean_energy -> ranks r_1
       (lowest) .. r_N (highest).
    3. k = max(1, round(N * cooldown_fraction)); if N < 3, k = 0 and
       the whole ranked list is returned ascending, no cooldown phase
       (can't meaningfully build+peak+cooldown with 0-2 tracks).
    4. Cooldown pool = ranks r_2..r_(k+1) -- the k lowest-energy tracks
       EXCLUDING r_1, which always stays the true opener.
    5. Build phase = r_1 + r_(k+2)..r_N, kept ascending -- this
       naturally puts the single highest-energy track (the "peak") last
       in the build phase.
    6. Final order = build phase + (cooldown pool sorted DESCENDING by
       energy) -- the very last track is the calmest of the cooldown
       group, a genuine wind-down close.

    Ties in mean_energy fall back to input order (Python's sort is
    stable) -- mean_energy is a continuous float derived from real
    waveform data, so an exact tie isn't realistically expected and no
    extra tiebreaker field is used, unlike harmony.py's title
    tiebreaker.

    k is never explicitly clamped to N-1: Python's list-slice semantics
    (ranked[1:1+k], ranked[1+k:]) already clip safely when k overshoots,
    so r_1 can never end up inside the cooldown pool regardless of how
    large cooldown_fraction is.
    """
    scored: list[TrackEnergy] = []
    unscored: list[UnscoredTrack] = []
    for track in tracks:
        energy = compute_track_energy(track)
        if energy is not None:
            scored.append(energy)
        else:
            reason = "no_phrase_data" if not track.phrases else "no_waveform_data"
            unscored.append(UnscoredTrack(track=track, reason=reason))

    ranked = sorted(scored, key=lambda te: te.mean_energy)
    n = len(ranked)
    if n < 3:
        return FlowResult(ordered_tracks=ranked, cooldown_start_index=n, unscored=unscored)

    k = max(1, round(n * cooldown_fraction))
    cooldown_pool = ranked[1 : 1 + k]
    build_phase = [ranked[0]] + ranked[1 + k :]
    cooldown_sorted_desc = sorted(cooldown_pool, key=lambda te: te.mean_energy, reverse=True)

    ordered = build_phase + cooldown_sorted_desc
    return FlowResult(ordered_tracks=ordered, cooldown_start_index=len(build_phase), unscored=unscored)
