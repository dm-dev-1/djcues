import sys

import numpy as np
import pytest

from djcues import drop_enhance
from djcues.audio import AudioExtraUnavailableError
from djcues.drop_enhance import (
    DropRefinement,
    _sustain_signature,
    enhance_proposal_drops,
    refine_breakdown_position,
    refine_drop_position,
)
from djcues.models import BeatGrid, CuePoint, CueProposal, Track
from tests.conftest import requires_audio, requires_ml

_SR = 22050


def _track(audio_path: str | None) -> Track:
    return Track(
        id=1, title="Test", artist="Test", bpm=128.0, duration_ms=20_000.0,
        analysis_path="", cues=[], phrases=[],
        beat_grid=BeatGrid(first_beat_ms=0.0, bpm=128.0),
        audio_path=audio_path,
    )


def _proposal(track: Track, positions: dict[str, float]) -> CueProposal:
    """D=kind 5, E=kind 6, F=kind 7 -- matches constants.CUE_SYSTEM."""
    kind_by_pad = {"D": 5, "E": 6, "F": 7}
    hot_cues = [
        CuePoint(kind=kind_by_pad[pad], position_ms=ms, loop_end_ms=None,
                  color_table_index=0, color=0, comment=pad)
        for pad, ms in positions.items()
    ]
    confidence = {pad: 0.85 for pad in positions}
    return CueProposal(
        track=track, hot_cues=hot_cues, memory_cues=[],
        confidence=confidence, notes=["original note"],
    )


def _silence_with_burst(
    duration_s: float, burst_at_s: float, burst_len_s: float = 0.3, amp: float = 0.8,
    sr: int = _SR,
) -> np.ndarray:
    """A near-instant-attack, exponentially-decaying low-frequency burst
    in otherwise-silent audio -- a stand-in for a real drop/kick
    transient's energy envelope."""
    n = int(duration_s * sr)
    y = np.zeros(n, dtype=np.float32)
    b0 = int(burst_at_s * sr)
    b1 = min(n, b0 + int(burst_len_s * sr))
    t = np.arange(b1 - b0)
    env = np.exp(-t / (0.05 * sr))
    y[b0:b1] = amp * env * np.sin(2 * np.pi * 60 * t / sr)
    return y


def _loud_with_dip(
    duration_s: float, dip_at_s: float, dip_len_s: float = 1.5,
    loud_amp: float = 0.6, quiet_amp: float = 0.02, sr: int = _SR,
) -> np.ndarray:
    """Sustained loud tone that drops to near-silence for dip_len_s
    starting at dip_at_s, then resumes -- a stand-in for a real
    breakdown's energy envelope (the opposite shape from a burst)."""
    n = int(duration_s * sr)
    t = np.arange(n)
    y = (loud_amp * np.sin(2 * np.pi * 100 * t / sr)).astype(np.float32)
    d0 = int(dip_at_s * sr)
    d1 = min(n, d0 + int(dip_len_s * sr))
    y[d0:d1] = (quiet_amp * np.sin(2 * np.pi * 100 * t[d0:d1] / sr)).astype(np.float32)
    return y


# --- refine_drop_position: real synthetic-audio scoring, requires_audio ---


@requires_audio
def test_refine_drop_position_confirms_when_transient_matches_candidate():
    y = _silence_with_burst(duration_s=10.0, burst_at_s=5.0)
    result = refine_drop_position(y, _SR, candidate_ms=5000.0, pad="D")

    assert result.outcome == "confirmed"
    assert result.refined_ms == 5000.0
    assert result.offset_ms == 0.0


@requires_audio
def test_refine_drop_position_refines_when_dominant_transient_is_far_away():
    y = _silence_with_burst(duration_s=10.0, burst_at_s=6.5)  # 1500ms from candidate
    result = refine_drop_position(y, _SR, candidate_ms=5000.0, pad="D")

    assert result.outcome == "refined"
    assert result.refined_ms == pytest.approx(6500.0, abs=50.0)
    assert result.offset_ms > 0


@requires_audio
def test_refine_drop_position_inconclusive_on_silence():
    y = np.zeros(int(10.0 * _SR), dtype=np.float32)
    result = refine_drop_position(y, _SR, candidate_ms=5000.0, pad="D")

    assert result.outcome == "inconclusive"
    assert result.refined_ms == 5000.0


@requires_audio
def test_refine_drop_position_inconclusive_near_track_boundary():
    # A track shorter than the minimum analyzable window everywhere in it.
    y = _silence_with_burst(duration_s=0.5, burst_at_s=0.25, burst_len_s=0.1)
    result = refine_drop_position(y, _SR, candidate_ms=250.0, pad="D")

    assert result.outcome == "inconclusive"
    assert "boundary" in result.note


@requires_audio
def test_refine_drop_position_does_not_refine_a_close_alternative():
    # A second transient only 200ms away isn't a genuinely different event.
    y = _silence_with_burst(duration_s=10.0, burst_at_s=5.2)
    result = refine_drop_position(y, _SR, candidate_ms=5000.0, pad="D")

    assert result.outcome == "confirmed"


@requires_audio
def test_refine_drop_position_does_not_refine_a_weaker_distant_alternative():
    # Strong transient right at the candidate; a much weaker one further
    # away must not win just for being far away.
    y = _silence_with_burst(duration_s=10.0, burst_at_s=5.0, amp=0.8)
    n = len(y)
    b0, b1 = int(6.5 * _SR), min(n, int(6.5 * _SR) + int(0.3 * _SR))
    t = np.arange(b1 - b0)
    y[b0:b1] += 0.2 * np.exp(-t / (0.05 * _SR)) * np.sin(2 * np.pi * 60 * t / _SR)

    result = refine_drop_position(y, _SR, candidate_ms=5000.0, pad="D")
    assert result.outcome == "confirmed"


@requires_audio
def test_refine_drop_position_works_for_f_pad_too():
    y = _silence_with_burst(duration_s=10.0, burst_at_s=6.3)
    result = refine_drop_position(y, _SR, candidate_ms=5000.0, pad="F")

    assert result.pad == "F"
    assert result.outcome == "refined"


# --- refine_breakdown_position: mirror of the above for the dip direction ---


@requires_audio
def test_refine_breakdown_position_confirms_when_dip_matches_candidate():
    y = _loud_with_dip(duration_s=10.0, dip_at_s=5.0)
    result = refine_breakdown_position(y, _SR, candidate_ms=5000.0)

    assert result.pad == "E"
    assert result.outcome == "confirmed"
    assert result.refined_ms == 5000.0
    assert result.offset_ms == 0.0


@requires_audio
def test_refine_breakdown_position_refines_when_dominant_dip_is_far_away():
    y = _loud_with_dip(duration_s=10.0, dip_at_s=6.5)  # 1500ms from candidate
    result = refine_breakdown_position(y, _SR, candidate_ms=5000.0)

    assert result.outcome == "refined"
    assert result.refined_ms == pytest.approx(6500.0, abs=50.0)
    assert result.offset_ms > 0


@requires_audio
def test_refine_breakdown_position_inconclusive_on_constant_energy():
    # A perfectly sustained tone has no dip anywhere -- this is also the
    # regression case for a real edge-padding bug this session: RMS
    # computed directly on a tightly-clipped window slice picked up a
    # spurious "dip" right at the window boundary (reflection padding
    # of non-silent audio), fixed by scoring on a margin-extended
    # buffer and restricting selection back to the true window.
    n = int(10.0 * _SR)
    t = np.arange(n)
    y = (0.6 * np.sin(2 * np.pi * 100 * t / _SR)).astype(np.float32)
    result = refine_breakdown_position(y, _SR, candidate_ms=5000.0)

    assert result.outcome == "inconclusive"
    assert result.refined_ms == 5000.0


@requires_audio
def test_refine_breakdown_position_inconclusive_near_track_boundary():
    y = _loud_with_dip(duration_s=0.5, dip_at_s=0.25, dip_len_s=0.1)
    result = refine_breakdown_position(y, _SR, candidate_ms=250.0)

    assert result.outcome == "inconclusive"
    assert "boundary" in result.note


@requires_audio
def test_refine_breakdown_position_does_not_refine_a_close_alternative():
    y = _loud_with_dip(duration_s=10.0, dip_at_s=5.2)  # only 200ms away
    result = refine_breakdown_position(y, _SR, candidate_ms=5000.0)

    assert result.outcome == "confirmed"


@requires_audio
def test_refine_breakdown_position_defaults_pad_to_e():
    y = _loud_with_dip(duration_s=10.0, dip_at_s=5.0)
    result = refine_breakdown_position(y, _SR, candidate_ms=5000.0)

    assert result.pad == "E"


# --- _sustain_signature: unverified diagnostic hint on "refined" outcomes ---


def _flat_tone(duration_s: float, amp: float = 0.3, sr: int = _SR) -> np.ndarray:
    """Constant-amplitude tone -- a stand-in for a genuinely settled,
    sustained energy state (what a real drop/breakdown should look like
    after it lands)."""
    n = int(duration_s * sr)
    t = np.arange(n)
    return (amp * np.sin(2 * np.pi * 100 * t / sr)).astype(np.float32)


def _pulsing_region(
    duration_s: float, sr: int = _SR, period_s: float = 0.4,
    loud_amp: float = 0.3, quiet_amp: float = 0.02,
) -> np.ndarray:
    """Alternating loud/quiet blocks -- a stand-in for normal per-beat
    rhythmic pulsing that swings back within a beat or two, the shape
    that fooled the fast-path dip-detector on a real confirmed case."""
    n = int(duration_s * sr)
    y = np.zeros(n, dtype=np.float32)
    period = int(period_s * sr)
    half = period // 2
    for start in range(0, n, period):
        y[start:start + half] = loud_amp
    t = np.arange(n)
    return (y * np.sin(2 * np.pi * 100 * t / sr)).astype(np.float32) + (
        quiet_amp * np.sin(2 * np.pi * 100 * t / sr)
    ).astype(np.float32)


@requires_audio
def test_sustain_signature_oscillating_when_refined_is_choppier():
    # original: settles into a flat, stable state (a real transition's shape).
    # refined: keeps pulsing (a normal rhythmic dip/rise's shape).
    y = np.concatenate([_flat_tone(2.0), _pulsing_region(2.0)])
    result = _sustain_signature(y, _SR, original_ms=0.0, refined_ms=2000.0)

    assert result == "oscillating"


@requires_audio
def test_sustain_signature_stable_match_when_refined_is_steadier():
    # Mirror of the above -- original pulses, refined settles.
    y = np.concatenate([_pulsing_region(2.0), _flat_tone(2.0)])
    result = _sustain_signature(y, _SR, original_ms=0.0, refined_ms=2000.0)

    assert result == "stable_match"


@requires_audio
def test_sustain_signature_ambiguous_when_both_similar():
    y = np.concatenate([_flat_tone(2.0), _flat_tone(2.0)])
    result = _sustain_signature(y, _SR, original_ms=0.0, refined_ms=2000.0)

    assert result == "ambiguous"


@requires_audio
def test_sustain_signature_none_near_track_end():
    # _window_range degrades gracefully down to a 0.3s floor, so refined_ms
    # needs to leave LESS than that -- not just less than the full 1.5s --
    # to actually hit the "not enough audio" None path.
    y = _flat_tone(2.0)
    result = _sustain_signature(y, _SR, original_ms=0.0, refined_ms=1900.0)

    assert result is None


def _hum_with_burst(
    duration_s: float, burst_at_s: float, burst_len_s: float = 0.3,
    hum_amp: float = 0.05, burst_amp: float = 0.8, sr: int = _SR,
) -> np.ndarray:
    """A quiet-but-nonzero sustained baseline (so _sustain_signature has a
    genuine energy level to compare, unlike near-total silence) plus a
    real transient burst for the rise-detector itself to find."""
    n = int(duration_s * sr)
    t = np.arange(n)
    y = (hum_amp * np.sin(2 * np.pi * 100 * t / sr)).astype(np.float32)
    b0 = int(burst_at_s * sr)
    b1 = min(n, b0 + int(burst_len_s * sr))
    tb = np.arange(b1 - b0)
    env = np.exp(-tb / (0.05 * sr))
    y[b0:b1] += burst_amp * env * np.sin(2 * np.pi * 60 * tb / sr)
    return y


@requires_audio
def test_refine_drop_position_populates_sustain_signature_only_when_refined():
    confirmed = refine_drop_position(
        _hum_with_burst(duration_s=10.0, burst_at_s=5.0), _SR, candidate_ms=5000.0, pad="D"
    )
    assert confirmed.outcome == "confirmed"
    assert confirmed.sustain_signature is None

    refined = refine_drop_position(
        _hum_with_burst(duration_s=10.0, burst_at_s=6.5), _SR, candidate_ms=5000.0, pad="D"
    )
    assert refined.outcome == "refined"
    assert refined.sustain_signature in ("oscillating", "stable_match", "ambiguous")


# --- separate_stems: real Demucs inference, requires_ml smoke test only ---


@requires_ml
def test_separate_stems_runs_real_demucs_and_returns_four_stems():
    """Smoke test only, per the plan's testing strategy -- confirms the
    real Demucs call completes and returns sensible shapes, not exact
    separation quality from a real model."""
    sr = 22050
    duration_s = 2.0
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n, endpoint=False)
    mono = (0.3 * np.sin(2 * np.pi * 110 * t)).astype(np.float32)

    stems = drop_enhance.separate_stems(mono, sr)

    assert set(stems.keys()) == {"drums", "bass", "other", "vocals"}
    for arr in stems.values():
        assert arr.ndim == 1
        assert abs(len(arr) - n) < sr * 0.05  # resample round-trip, small tolerance
        assert not np.isnan(arr).any()


# --- enhance_proposal_drops: orchestration, monkeypatched, no real dependency ---


def test_enhance_proposal_drops_noop_when_no_audio_path(monkeypatch):
    track = _track(audio_path=None)
    proposal = _proposal(track, {"D": 10_000.0, "E": 12_000.0, "F": 14_000.0})

    result, refinements = enhance_proposal_drops(proposal, track, 16, 4)

    assert result is proposal
    assert refinements == []


def test_enhance_proposal_drops_noop_when_audio_extra_missing(monkeypatch, tmp_path):
    audio_path = tmp_path / "track.wav"
    audio_path.write_bytes(b"x")
    track = _track(audio_path=str(audio_path))
    proposal = _proposal(track, {"D": 10_000.0})

    def _raise(*args, **kwargs):
        raise AudioExtraUnavailableError("djcues[audio] not installed")

    monkeypatch.setattr(drop_enhance, "resolve_audio_path", lambda t: audio_path)
    monkeypatch.setattr(drop_enhance, "load_audio", _raise)

    result, refinements = enhance_proposal_drops(proposal, track, 16, 4)

    assert result is proposal
    assert refinements == []


def test_enhance_proposal_drops_noop_when_decode_fails(monkeypatch, tmp_path):
    audio_path = tmp_path / "track.wav"
    audio_path.write_bytes(b"x")
    track = _track(audio_path=str(audio_path))
    proposal = _proposal(track, {"D": 10_000.0})

    def _raise(*args, **kwargs):
        raise RuntimeError("corrupt file")

    monkeypatch.setattr(drop_enhance, "resolve_audio_path", lambda t: audio_path)
    monkeypatch.setattr(drop_enhance, "load_audio", _raise)

    result, refinements = enhance_proposal_drops(proposal, track, 16, 4)

    assert result is proposal
    assert refinements == []


class _FakeLoaded:
    def __init__(self, samples, sr):
        self.samples = samples
        self.sr = sr


def _patch_audio_loading(monkeypatch, audio_path, samples=None, sr=22050):
    samples = samples if samples is not None else np.zeros(1000, dtype=np.float32)
    monkeypatch.setattr(drop_enhance, "resolve_audio_path", lambda t: audio_path)
    monkeypatch.setattr(drop_enhance, "load_audio", lambda p: _FakeLoaded(samples, sr))


def test_enhance_proposal_drops_refines_d_e_and_f(monkeypatch, tmp_path):
    audio_path = tmp_path / "track.wav"
    audio_path.write_bytes(b"x")
    track = _track(audio_path=str(audio_path))
    proposal = _proposal(track, {"D": 10_000.0, "E": 12_000.0, "F": 14_000.0})
    _patch_audio_loading(monkeypatch, audio_path)

    rise_calls = []
    dip_calls = []

    def fake_refine_rise(samples, sr, candidate_ms, pad, source="full_mix"):
        rise_calls.append(pad)
        if pad == "D":
            return DropRefinement(
                pad="D", outcome="refined", original_ms=candidate_ms,
                refined_ms=candidate_ms + 800.0, offset_ms=800.0, strength=2.0,
                source=source, note="fake refine",
            )
        return DropRefinement(
            pad="F", outcome="confirmed", original_ms=candidate_ms,
            refined_ms=candidate_ms, offset_ms=0.0, strength=1.0,
            source=source, note="fake confirm",
        )

    def fake_refine_dip(samples, sr, candidate_ms, pad="E", source="full_mix"):
        dip_calls.append(pad)
        return DropRefinement(
            pad="E", outcome="refined", original_ms=candidate_ms,
            refined_ms=candidate_ms - 500.0, offset_ms=-500.0, strength=2.0,
            source=source, note="fake dip refine",
        )

    monkeypatch.setattr(drop_enhance, "refine_drop_position", fake_refine_rise)
    monkeypatch.setattr(drop_enhance, "refine_breakdown_position", fake_refine_dip)

    result, refinements = enhance_proposal_drops(proposal, track, 16, 4)

    assert sorted(rise_calls) == ["D", "F"]
    assert dip_calls == ["E"]
    assert len(refinements) == 3

    by_kind = {c.kind: c for c in result.hot_cues}
    assert by_kind[5].position_ms == 10_800.0  # D moved (rise)
    assert by_kind[6].position_ms == 11_500.0  # E moved (dip)
    assert by_kind[7].position_ms == 14_000.0  # F confirmed, unchanged

    assert any("D (drop-enhance" in n for n in result.notes)
    assert any("E (drop-enhance" in n for n in result.notes)
    assert any("F (drop-enhance" in n for n in result.notes)
    assert "original note" in result.notes


def test_enhance_proposal_drops_handles_e_only_proposal(monkeypatch, tmp_path):
    audio_path = tmp_path / "track.wav"
    audio_path.write_bytes(b"x")
    track = _track(audio_path=str(audio_path))
    proposal = _proposal(track, {"E": 12_000.0})
    _patch_audio_loading(monkeypatch, audio_path)

    calls = []

    def fake_refine_dip(samples, sr, candidate_ms, pad="E", source="full_mix"):
        calls.append(pad)
        return DropRefinement(
            pad="E", outcome="confirmed", original_ms=candidate_ms,
            refined_ms=candidate_ms, offset_ms=0.0, strength=1.0,
            source=source, note="fake",
        )

    def _boom(*args, **kwargs):
        raise AssertionError("refine_drop_position must not be called when only E is present")

    monkeypatch.setattr(drop_enhance, "refine_breakdown_position", fake_refine_dip)
    monkeypatch.setattr(drop_enhance, "refine_drop_position", _boom)

    result, refinements = enhance_proposal_drops(proposal, track, 16, 4)

    assert calls == ["E"]
    assert len(refinements) == 1


def test_enhance_proposal_drops_handles_d_only_proposal(monkeypatch, tmp_path):
    audio_path = tmp_path / "track.wav"
    audio_path.write_bytes(b"x")
    track = _track(audio_path=str(audio_path))
    proposal = _proposal(track, {"D": 10_000.0})
    _patch_audio_loading(monkeypatch, audio_path)

    calls = []

    def fake_refine(samples, sr, candidate_ms, pad, source="full_mix"):
        calls.append(pad)
        return DropRefinement(
            pad=pad, outcome="confirmed", original_ms=candidate_ms,
            refined_ms=candidate_ms, offset_ms=0.0, strength=1.0,
            source=source, note="fake",
        )

    monkeypatch.setattr(drop_enhance, "refine_drop_position", fake_refine)

    result, refinements = enhance_proposal_drops(proposal, track, 16, 4)

    assert calls == ["D"]
    assert len(refinements) == 1


def test_enhance_proposal_drops_does_not_mutate_original_proposal(monkeypatch, tmp_path):
    audio_path = tmp_path / "track.wav"
    audio_path.write_bytes(b"x")
    track = _track(audio_path=str(audio_path))
    proposal = _proposal(track, {"D": 10_000.0})
    original_positions = [c.position_ms for c in proposal.hot_cues]
    original_confidence = dict(proposal.confidence)
    original_notes = list(proposal.notes)
    _patch_audio_loading(monkeypatch, audio_path)

    def fake_refine(samples, sr, candidate_ms, pad, source="full_mix"):
        return DropRefinement(
            pad=pad, outcome="refined", original_ms=candidate_ms,
            refined_ms=candidate_ms + 1234.0, offset_ms=1234.0, strength=2.0,
            source=source, note="fake",
        )

    monkeypatch.setattr(drop_enhance, "refine_drop_position", fake_refine)

    enhance_proposal_drops(proposal, track, 16, 4)

    assert [c.position_ms for c in proposal.hot_cues] == original_positions
    assert proposal.confidence == original_confidence
    assert proposal.notes == original_notes


def test_enhance_proposal_drops_deep_uses_stems_and_falls_back_on_failure(monkeypatch, tmp_path):
    audio_path = tmp_path / "track.wav"
    audio_path.write_bytes(b"x")
    track = _track(audio_path=str(audio_path))
    proposal = _proposal(track, {"D": 10_000.0})
    mix_samples = np.ones(1000, dtype=np.float32)
    _patch_audio_loading(monkeypatch, audio_path, samples=mix_samples)

    seen_sources = []

    def fake_refine(samples, sr, candidate_ms, pad, source="full_mix"):
        seen_sources.append(source)
        return DropRefinement(
            pad=pad, outcome="confirmed", original_ms=candidate_ms,
            refined_ms=candidate_ms, offset_ms=0.0, strength=1.0,
            source=source, note="fake",
        )

    monkeypatch.setattr(drop_enhance, "refine_drop_position", fake_refine)

    # Deep path succeeds -> source should be "stems"
    monkeypatch.setattr(
        drop_enhance, "separate_stems",
        lambda samples, sr: {"bass": np.full(1000, 2.0, dtype=np.float32),
                               "drums": np.full(1000, 3.0, dtype=np.float32)},
    )
    enhance_proposal_drops(proposal, track, 16, 4, deep=True)
    assert seen_sources == ["stems"]

    # Deep path raises -> must fall back to full_mix, not crash the run
    seen_sources.clear()

    def _raise(samples, sr):
        raise RuntimeError("demucs blew up")

    monkeypatch.setattr(drop_enhance, "separate_stems", _raise)
    enhance_proposal_drops(proposal, track, 16, 4, deep=True)
    assert seen_sources == ["full_mix"]


def test_enhance_proposal_drops_deep_false_never_calls_separate_stems(monkeypatch, tmp_path):
    audio_path = tmp_path / "track.wav"
    audio_path.write_bytes(b"x")
    track = _track(audio_path=str(audio_path))
    proposal = _proposal(track, {"D": 10_000.0})
    _patch_audio_loading(monkeypatch, audio_path)

    def _boom(*args, **kwargs):
        raise AssertionError("separate_stems must not be called when deep=False")

    monkeypatch.setattr(drop_enhance, "separate_stems", _boom)
    monkeypatch.setattr(
        drop_enhance, "refine_drop_position",
        lambda samples, sr, candidate_ms, pad, source="full_mix": DropRefinement(
            pad=pad, outcome="confirmed", original_ms=candidate_ms,
            refined_ms=candidate_ms, offset_ms=0.0, strength=1.0,
            source=source, note="fake",
        ),
    )

    enhance_proposal_drops(proposal, track, 16, 4, deep=False)
