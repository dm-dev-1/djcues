"""Accuracy metrics for comparing proposed cues against ground truth.

Pure functions — no I/O, no ``click.echo`` — so they're usable both from
the CLI and from tests without a live Rekordbox connection.
"""

from __future__ import annotations

from dataclasses import dataclass

from djcues.constants import KIND_TO_PAD
from djcues.models import CuePoint


@dataclass
class PadStats:
    """Match/miss/false-positive counts for one cue pad (or an aggregate)."""

    matches: int = 0
    misses: int = 0
    false_positives: int = 0

    @property
    def precision(self) -> float:
        denom = self.matches + self.false_positives
        return self.matches / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.matches + self.misses
        return self.matches / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def compare_cues(
    existing: list[CuePoint],
    proposed: list[CuePoint],
    tolerance_ms: float = 1000.0,
) -> dict[str, PadStats]:
    """Per-pad match/miss/false-positive stats for one track.

    A cue kind present in both, within ``tolerance_ms`` of each other, is a
    match. Present only in ``existing`` (ground truth) is a miss — it
    penalizes recall. Present only in ``proposed`` is a false positive —
    it penalizes precision. This last case used to be silently free: a
    proposed cue with no ground-truth counterpart at all didn't affect the
    score either way.
    """
    existing_by_kind = {c.kind: c for c in existing if c.kind > 0}
    proposed_by_kind = {c.kind: c for c in proposed if c.kind > 0}
    all_kinds = set(existing_by_kind) | set(proposed_by_kind)

    stats: dict[str, PadStats] = {}
    for kind in all_kinds:
        pad = KIND_TO_PAD.get(kind, "?")
        e = existing_by_kind.get(kind)
        p = proposed_by_kind.get(kind)
        s = stats.setdefault(pad, PadStats())
        if e and p:
            if abs(p.position_ms - e.position_ms) <= tolerance_ms:
                s.matches += 1
            else:
                s.misses += 1
        elif e:
            s.misses += 1
        elif p:
            s.false_positives += 1
    return stats


def merge_pad_stats(pad_stats_list: list[dict[str, PadStats]]) -> dict[str, PadStats]:
    """Sum per-pad stats across multiple tracks (for ``--all`` runs)."""
    merged: dict[str, PadStats] = {}
    for stats in pad_stats_list:
        for pad, s in stats.items():
            m = merged.setdefault(pad, PadStats())
            m.matches += s.matches
            m.misses += s.misses
            m.false_positives += s.false_positives
    return merged


def overall_stats(pad_stats: dict[str, PadStats]) -> PadStats:
    """Flatten per-pad stats into one summary row."""
    total = PadStats()
    for s in pad_stats.values():
        total.matches += s.matches
        total.misses += s.misses
        total.false_positives += s.false_positives
    return total
