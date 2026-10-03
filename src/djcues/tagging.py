"""Smart tagging -- turns analysis djcues already computes (phrase
structure, vocal regions, waveform energy, beat-grid health, key) into
a small, fixed vocabulary of per-track tags a DJ can filter and sort by
when building a set.

Pure functions -- no I/O, no ``click.echo`` -- so they're usable from the
CLI, the dashboard and tests without a live Rekordbox connection.
Mirrors flow.py's/clash.py's own shape for the same reason.

Two deliberately separate steps:

1. extract_features(track, raw_entries) -> AnalysisFeatures: the
   expensive part (needs a loaded Track, i.e. real ANLZ parsing). The
   result is plain, JSON-serializable data so tag_store.py can cache it.
2. derive_tags(features, ...) -> TrackTags: cheap rules over those
   features. Retuning a threshold only re-runs this step -- never
   re-reads a single ANLZ file.

Everything here is *descriptive*: a tag never moves a cue or changes
any analysis result. Every decision carries an evidence string so a
reviewer can see why it was made (same role as cue-proposal notes).

NOTE on "energy": means flow.compute_track_energy()'s duration-weighted
mean waveform height -- Rekordbox's PWV5 color waveform, which is
effectively peak-normalized per track. It measures loudness *density*
(how much of the track sits near its own peak), not absolute loudness,
and has not been validated by ear. Bands are library-wide percentiles
(compute_energy_cutpoints), so they inherit genre skew -- see the plan's
Risks section.
"""

from __future__ import annotations

import statistics
from bisect import bisect_right
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Iterable

from djcues.beat_verify import check_grid_self_consistency
from djcues.flow import compute_track_energy
from djcues.harmony import parse_camelot_key
from djcues.strategy import (
    DEFAULT_MIN_VOCAL_REGION_MS,
    find_vocal_regions,
    head_zone,
    regions_overlapping,
    tail_zone,
)

if TYPE_CHECKING:
    from djcues.models import RawBeatGridEntry, Track, TrackSummary

ENERGY_BANDS = 5
# statistics.quantiles needs >= 2 points, and quintiles of fewer than
# this many tracks are meaningless -- calibrate() refuses rather than
# invent a scale from a handful of tracks.
MIN_CALIBRATION_TRACKS = 5


@dataclass(frozen=True)
class CategorySpec:
    """One My Tag category. select is "single" (at most one tag per
    track, e.g. an energy level) or "multi"."""

    name: str
    select: str  # "single" | "multi"
    tags: tuple[str, ...]


# The whole vocabulary lives here as data (not scattered through rule
# code) so collapsing categories later -- if Rekordbox turns out to cap
# the number of My Tag categories -- is a change to this table only.
TAG_CATALOG: tuple[CategorySpec, ...] = (
    CategorySpec("Energy", "single", tuple(f"Energy {i}" for i in range(1, ENERGY_BANDS + 1))),
    CategorySpec("Mix In", "multi", ("Long Intro", "Short Intro", "Vocal Intro")),
    CategorySpec("Mix Out", "multi", ("Long Outro", "Short Outro", "Vocal Outro")),
    CategorySpec("Vocals", "single", ("Instrumental", "Vocal-Led")),
    CategorySpec("Check", "multi", ("No Key", "Tempo Varies", "Grid Suspect")),
)
CATEGORY_NAMES: tuple[str, ...] = tuple(c.name for c in TAG_CATALOG)


def category_of(tag: str) -> str | None:
    """The catalog category a tag name belongs to, or None."""
    for cat in TAG_CATALOG:
        if tag in cat.tags:
            return cat.name
    return None


@dataclass(frozen=True)
class TagThresholds:
    """Every tunable rule value in one place. All are *untuned starting
    points* -- see the plan's acceptance gate: after a real full-library
    run, each emitted tag should land on roughly 5-35% of tracks, else
    retune before anything is written to Rekordbox."""

    long_intro_bars: float = 16.0
    short_intro_bars: float = 2.0
    long_outro_bars: float = 16.0
    short_outro_bars: float = 4.0
    instrumental_max_pct: float = 3.0
    vocal_led_min_pct: float = 45.0
    # Loops/samples (very short, or BPM 0) would drag the quintiles
    # around and make no sense as "energy 1-5" set material.
    min_calibration_duration_ms: float = 60_000.0
    min_vocal_region_ms: float = DEFAULT_MIN_VOCAL_REGION_MS


DEFAULT_THRESHOLDS = TagThresholds()


@dataclass(frozen=True)
class AnalysisFeatures:
    """The ANLZ-derived facts the rules need, extracted once per track.
    Plain data (see to_dict/from_dict) so it can be cached.

    None means "couldn't be determined" and is deliberately distinct
    from a real zero/False -- same honest-missing-data convention as
    transition.TransitionSuggestion.vocal_regions_a. vocal_intro /
    vocal_outro hold the overlapping (start_ms, end_ms) vocal regions
    ([] = checked and clear, None = couldn't check).
    """

    track_id: str
    duration_ms: float
    bpm: float
    mean_energy: float | None
    peak_energy: float | None
    intro_bars: float | None
    outro_bars: float | None
    vocal_pct: float | None
    vocal_intro: list[tuple[float, float]] | None
    vocal_outro: list[tuple[float, float]] | None
    tempo_varies: bool | None
    grid_consistent: bool | None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AnalysisFeatures":
        def _regions(v):
            return None if v is None else [(float(a), float(b)) for a, b in v]

        return cls(
            track_id=str(data["track_id"]),
            duration_ms=float(data["duration_ms"]),
            bpm=float(data["bpm"]),
            mean_energy=data.get("mean_energy"),
            peak_energy=data.get("peak_energy"),
            intro_bars=data.get("intro_bars"),
            outro_bars=data.get("outro_bars"),
            vocal_pct=data.get("vocal_pct"),
            vocal_intro=_regions(data.get("vocal_intro")),
            vocal_outro=_regions(data.get("vocal_outro")),
            tempo_varies=data.get("tempo_varies"),
            grid_consistent=data.get("grid_consistent"),
        )


@dataclass(frozen=True)
class TagDecision:
    """One tag djcues proposes for one track, with the evidence for it."""

    category: str
    tag: str
    evidence: str


@dataclass(frozen=True)
class SkippedCategory:
    """A category that couldn't be evaluated for this track at all.
    reason is one of: "no_calibration", "no_energy_data", "sample_or_loop",
    "no_intro_phrase", "no_outro_phrase", "no_vocal_data"."""

    category: str
    reason: str


@dataclass(frozen=True)
class TrackTags:
    track_id: str
    decisions: list[TagDecision] = field(default_factory=list)
    skipped: list[SkippedCategory] = field(default_factory=list)

    def tags_in(self, category: str) -> list[str]:
        return [d.tag for d in self.decisions if d.category == category]


def _bars(beat_length: int | float) -> float:
    return beat_length / 4.0


def extract_features(
    track: "Track",
    raw_entries: "list[RawBeatGridEntry] | None" = None,
    *,
    min_vocal_region_ms: float = DEFAULT_MIN_VOCAL_REGION_MS,
) -> AnalysisFeatures:
    """Pull every ANLZ-derived fact the tag rules need out of one loaded
    Track (plus, optionally, its raw per-beat grid from
    db.extract_raw_beat_grid()). Reuses existing analysis functions
    rather than re-deriving any signal:

    - flow.compute_track_energy for mean/peak energy,
    - strategy.find_vocal_regions / head_zone / tail_zone /
      regions_overlapping (clash.py's exact zone semantics) for vocals,
    - beat_verify.check_grid_self_consistency (the free tier-1 check,
      pure arithmetic) for grid health.

    Intro/outro length comes from the phrase's own beat_length (exact,
    tempo-independent) rather than duration / bar length, so a BPM-0
    track still gets structure tags.
    """
    energy = compute_track_energy(track)

    intro = next((p for p in track.phrases if p.label == "Intro"), None)
    outro = next((p for p in track.phrases if p.label == "Outro"), None)

    vocal_pct: float | None = None
    vocal_intro: list[tuple[float, float]] | None = None
    vocal_outro: list[tuple[float, float]] | None = None
    if track.vocal_track:
        regions = find_vocal_regions(track, min_vocal_region_ms)
        if track.duration_ms > 0:
            covered = sum(r.end_ms - r.start_ms for r in regions)
            vocal_pct = min(100.0, 100.0 * covered / track.duration_ms)
        if intro is not None:
            zs, ze = head_zone(track)
            vocal_intro = [(r.start_ms, r.end_ms) for r in regions_overlapping(regions, zs, ze)]
        if outro is not None:
            zs, ze = tail_zone(track)
            vocal_outro = [(r.start_ms, r.end_ms) for r in regions_overlapping(regions, zs, ze)]

    tempo_varies: bool | None = None
    grid_consistent: bool | None = None
    if raw_entries:
        sc = check_grid_self_consistency(raw_entries)
        tempo_varies = sc.tempo_varies
        grid_consistent = sc.is_consistent

    return AnalysisFeatures(
        track_id=str(track.id),
        duration_ms=track.duration_ms,
        bpm=track.bpm,
        mean_energy=energy.mean_energy if energy else None,
        peak_energy=energy.peak_energy if energy else None,
        intro_bars=_bars(intro.beat_length) if intro else None,
        outro_bars=_bars(outro.beat_length) if outro else None,
        vocal_pct=vocal_pct,
        vocal_intro=vocal_intro,
        vocal_outro=vocal_outro,
        tempo_varies=tempo_varies,
        grid_consistent=grid_consistent,
    )


def is_calibration_eligible(
    features: AnalysisFeatures, thresholds: TagThresholds = DEFAULT_THRESHOLDS
) -> bool:
    """Whether this track's energy should feed the library-wide
    quintiles (and may receive an Energy tag at all)."""
    return (
        features.mean_energy is not None
        and features.bpm > 0
        and features.duration_ms >= thresholds.min_calibration_duration_ms
    )


def compute_energy_cutpoints(energies: Iterable[float], bands: int = ENERGY_BANDS) -> list[float]:
    """bands-1 ascending cutpoints splitting energies into equal-count
    bands (quintiles for the default 5). Raises ValueError with fewer
    than MIN_CALIBRATION_TRACKS values."""
    values = sorted(energies)
    if len(values) < MIN_CALIBRATION_TRACKS:
        raise ValueError(
            f"need at least {MIN_CALIBRATION_TRACKS} scorable tracks to calibrate energy bands, "
            f"got {len(values)}"
        )
    return [float(c) for c in statistics.quantiles(values, n=bands, method="inclusive")]


def energy_band(mean_energy: float, cutpoints: list[float]) -> int:
    """1-based band for a mean energy. A value exactly on a cutpoint
    belongs to the higher band."""
    return bisect_right(cutpoints, mean_energy) + 1


def _band_range_text(band: int, cutpoints: list[float]) -> str:
    lo = f"{cutpoints[band - 2]:.2f}" if band >= 2 else "min"
    hi = f"{cutpoints[band - 1]:.2f}" if band - 1 < len(cutpoints) else "max"
    return f"{lo}-{hi}"


def _fmt_ms(ms: float) -> str:
    total = int(round(ms / 1000.0))
    return f"{total // 60}:{total % 60:02d}"


def _regions_text(regions: list[tuple[float, float]]) -> str:
    return ", ".join(f"{_fmt_ms(a)}-{_fmt_ms(b)}" for a, b in regions)


def key_status(summary: "TrackSummary") -> str | None:
    """None if the track's key is usable for harmonic mixing, else
    "no_key" | "non_camelot_key".

    Deliberately checks the key itself and NOT audit.find_unusable_keys'
    precedence, which reports any track with encrypted (streaming-linked)
    Title/Artist as "encrypted_metadata" before looking at its key at
    all. That's right for audit/suggest, which can't show such a track's
    title -- but wrong for a "No Key" tag: measured on the real library,
    290 of 460 encrypted-title tracks carry a perfectly valid Camelot
    key. Key parsing itself is still shared (harmony.parse_camelot_key).
    """
    if not summary.key:
        return "no_key"
    if parse_camelot_key(summary.key) is None:
        return "non_camelot_key"
    return None


def derive_tags(
    features: AnalysisFeatures,
    *,
    key_reason: str | None = None,
    key_raw: str | None = None,
    cutpoints: list[float] | None = None,
    thresholds: TagThresholds = DEFAULT_THRESHOLDS,
    categories: Iterable[str] | None = None,
) -> TrackTags:
    """Apply the tag rules to one track's features.

    key_reason/key_raw come from key_status()/the TrackSummary (Track
    carries no key). cutpoints are the stored library-wide energy
    cutpoints; None skips the Energy category with reason
    "no_calibration". categories restricts evaluation (default: all).

    Each category is evaluated independently -- missing data skips only
    the categories that need it, recorded in .skipped with a reason.
    """
    wanted = set(categories) if categories is not None else set(CATEGORY_NAMES)
    decisions: list[TagDecision] = []
    skipped: list[SkippedCategory] = []
    f = features

    # --- Energy (single-select, library-wide percentile) -----------------
    if "Energy" in wanted:
        if f.mean_energy is None:
            skipped.append(SkippedCategory("Energy", "no_energy_data"))
        elif not is_calibration_eligible(f, thresholds):
            skipped.append(SkippedCategory("Energy", "sample_or_loop"))
        elif not cutpoints:
            skipped.append(SkippedCategory("Energy", "no_calibration"))
        else:
            band = energy_band(f.mean_energy, cutpoints)
            decisions.append(TagDecision(
                "Energy", f"Energy {band}",
                f"mean energy {f.mean_energy:.2f} (band {band} = {_band_range_text(band, cutpoints)})",
            ))

    # --- Mix In / Mix Out (multi-select) ---------------------------------
    if "Mix In" in wanted:
        if f.intro_bars is None:
            skipped.append(SkippedCategory("Mix In", "no_intro_phrase"))
        else:
            if f.intro_bars >= thresholds.long_intro_bars:
                decisions.append(TagDecision("Mix In", "Long Intro", f"intro {f.intro_bars:g} bars"))
            elif f.intro_bars <= thresholds.short_intro_bars:
                decisions.append(TagDecision("Mix In", "Short Intro", f"intro {f.intro_bars:g} bars"))
            if f.vocal_intro:
                decisions.append(TagDecision(
                    "Mix In", "Vocal Intro", f"vocals {_regions_text(f.vocal_intro)} overlap the intro"
                ))

    if "Mix Out" in wanted:
        if f.outro_bars is None:
            skipped.append(SkippedCategory("Mix Out", "no_outro_phrase"))
        else:
            if f.outro_bars >= thresholds.long_outro_bars:
                decisions.append(TagDecision("Mix Out", "Long Outro", f"outro {f.outro_bars:g} bars"))
            elif f.outro_bars <= thresholds.short_outro_bars:
                decisions.append(TagDecision("Mix Out", "Short Outro", f"outro {f.outro_bars:g} bars"))
            if f.vocal_outro:
                decisions.append(TagDecision(
                    "Mix Out", "Vocal Outro", f"vocals {_regions_text(f.vocal_outro)} overlap the outro"
                ))

    # --- Vocals (single-select; the middle band is deliberately untagged)
    if "Vocals" in wanted:
        if f.vocal_pct is None:
            skipped.append(SkippedCategory("Vocals", "no_vocal_data"))
        elif f.vocal_pct < thresholds.instrumental_max_pct:
            decisions.append(TagDecision(
                "Vocals", "Instrumental", f"vocals cover {f.vocal_pct:.0f}% of the track"
            ))
        elif f.vocal_pct >= thresholds.vocal_led_min_pct:
            decisions.append(TagDecision(
                "Vocals", "Vocal-Led", f"vocals cover {f.vocal_pct:.0f}% of the track"
            ))

    # --- Check (multi-select; data-quality warnings for set building) ----
    if "Check" in wanted:
        if key_reason is not None:
            evidence = {
                "no_key": "no key tag",
                "non_camelot_key": f"key {key_raw!r} isn't Camelot notation",
            }.get(key_reason, key_reason)
            decisions.append(TagDecision("Check", "No Key", evidence))
        if f.tempo_varies:
            decisions.append(TagDecision(
                "Check", "Tempo Varies", "Rekordbox's own beat grid records tempo changes"
            ))
        if f.grid_consistent is False:
            decisions.append(TagDecision(
                "Check", "Grid Suspect", "beat grid is internally inconsistent with constant tempo"
            ))

    return TrackTags(track_id=f.track_id, decisions=decisions, skipped=skipped)
