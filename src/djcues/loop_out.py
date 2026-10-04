"""Loop Out (pad H) placement: the best stretch of the outro to loop.

The old rule put the loop at the Outro phrase's first bar for a fixed
length, whatever was playing there -- often a breakdown tail or a fade
with the drums already gone, which is the opposite of what a DJ wants to
loop while the next track comes in. This searches the outro for the
bar-aligned window that has steady drums and no vocals instead.

Pure functions over a Track (no I/O). "Drums" is a proxy read from
Rekordbox's colour waveform, whose bands are bass (red), mids (green) and
treble (blue): kick and hats show up as bass and treble together, while
melody, pads and vocals live in the mids -- so a bar scores well when bass
and treble are both present and the mids are not dominating. It is a proxy,
not a drum detector; the waveform is also downsampled (a few points per
bar), so only bar-level shape is visible, never individual hits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from djcues.models import Track
from djcues.strategy import _spectral_similarity, find_vocal_regions, regions_overlapping


@dataclass(frozen=True)
class LoopOutThresholds:
    """Untuned starting points -- tune from listening, not from guesses."""

    preferred_bars: int = 8
    fallback_bars: int = 4
    max_vocal_overlap: float = 0.05  # fraction of the window; above this the window is rejected
    min_drum_score: float = 0.25  # mean bar score below this isn't "drums" at all
    early_bias: float = 0.05  # score discount for the latest start vs the earliest


DEFAULT_THRESHOLDS = LoopOutThresholds()


@dataclass(frozen=True)
class LoopOutChoice:
    start_ms: float
    bars: int
    score: float  # 0-1, the window's drum/steadiness/repeat score
    drum_score: float  # mean per-bar drum score
    vocal_overlap: float | None  # fraction of the window with vocals; None = no vocal data
    clean: bool  # passed the vocal and drum filters
    note: str


def _bar_score(points: list) -> float:
    """Drum score for the waveform points covering one bar, 0-1."""
    n = len(points)
    if n == 0:
        return 0.0
    bass = sum(p.red for p in points) / n / 7
    mid = sum(p.green for p in points) / n / 7
    treble = sum(p.blue for p in points) / n / 7
    height = sum(p.height for p in points) / n
    total = bass + mid + treble
    if total <= 0:
        return 0.0
    purity = (bass + treble) / total  # 1 when the mids are empty
    return math.sqrt(bass * treble) * purity * height


def _window_features(track: Track, bar_edges_ms: list[float]):
    """(per-bar scores, waveform index range) for a window given its bar
    boundaries (len = bars + 1)."""
    wf = track.waveform
    n = len(wf)
    dur = track.duration_ms
    idx = [min(n, max(0, int(n * t / dur))) for t in bar_edges_ms]
    scores = []
    for a, b in zip(idx, idx[1:]):
        b = max(b, a + 1)
        scores.append(_bar_score(wf[a:b]))
    return scores, idx


def _score_window(track: Track, edges: list[float], th: LoopOutThresholds):
    scores, idx = _window_features(track, edges)
    mean = sum(scores) / len(scores)
    if mean <= 0:
        return 0.0, 0.0
    std = math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores))
    steadiness = max(0.0, 1.0 - min(1.0, std / mean))
    mid = (idx[0] + idx[-1]) // 2
    repeat = _spectral_similarity(track.waveform, idx[0], mid, idx[-1]) if idx[-1] - idx[0] >= 4 else 1.0
    return mean * (0.5 + 0.5 * steadiness) * (0.5 + 0.5 * repeat), mean


def _bar_starts(track: Track, from_ms: float) -> list[float]:
    """Bar starts (counted from Rekordbox's 1.1) from the first bar at or
    after from_ms to the end of the track."""
    bg = track.beat_grid
    beat = bg.bar_start_beat(bg.ms_to_beat(from_ms))
    if bg.beat_to_ms(beat) < from_ms - bg.ms_per_beat / 2:
        beat += 4
    starts = []
    while bg.beat_to_ms(beat) < track.duration_ms:
        starts.append(bg.beat_to_ms(beat))
        beat += 4
    return starts


def find_loop_out(
    track: Track, outro_start_ms: float, thresholds: LoopOutThresholds = DEFAULT_THRESHOLDS
) -> LoopOutChoice | None:
    """The best loop window at or after outro_start_ms, or None when it
    can't be judged (no waveform, no usable grid, or nothing fits).

    Prefers preferred_bars; falls back to fallback_bars when no
    preferred-length window both fits and passes the filters (vocals
    overlapping more than max_vocal_overlap, or no real drums). If nothing
    passes at either length, returns the best-scoring window anyway with
    clean=False so the caller can mark it low-confidence.
    """
    bg = track.beat_grid
    if not track.waveform or track.duration_ms <= 0 or bg.bpm <= 0:
        return None
    th = thresholds
    starts = _bar_starts(track, outro_start_ms)
    if not starts:
        return None
    regions = find_vocal_regions(track)
    have_vocals = bool(track.vocal_track)
    bar_ms = bg.bars_to_ms(1)

    found: dict[int, list[LoopOutChoice]] = {th.preferred_bars: [], th.fallback_bars: []}
    for bars in found:
        last = len(starts) - 1
        for k, start in enumerate(starts):
            end = start + bars * bar_ms
            if end > track.duration_ms + bar_ms / 2:
                break
            edges = [start + i * bar_ms for i in range(bars + 1)]
            score, drums = _score_window(track, edges, th)
            overlap = None
            if have_vocals:
                hit = regions_overlapping(regions, start, end)
                covered = sum(min(r.end_ms, end) - max(r.start_ms, start) for r in hit)
                overlap = covered / (end - start)
            score *= 1.0 - th.early_bias * (k / last if last else 0.0)
            clean = drums >= th.min_drum_score and (overlap is None or overlap <= th.max_vocal_overlap)
            vocal_txt = "vocals not measured" if overlap is None else (
                "no vocals" if overlap == 0 else f"{overlap:.0%} vocals"
            )
            found[bars].append(LoopOutChoice(
                start_ms=start, bars=bars, score=score, drum_score=drums,
                vocal_overlap=overlap, clean=clean,
                note=f"{bars} bars at {start / 1000:.1f}s, drums {drums:.2f}, {vocal_txt}, score {score:.2f}",
            ))

    for bars in (th.preferred_bars, th.fallback_bars):
        clean = [c for c in found[bars] if c.clean]
        if clean:
            return max(clean, key=lambda c: c.score)
    everything = found[th.preferred_bars] + found[th.fallback_bars]
    return max(everything, key=lambda c: c.score) if everything else None
