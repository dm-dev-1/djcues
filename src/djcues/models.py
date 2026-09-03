"""Data model for djcues — decoupled from pyrekordbox ORM types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BeatGrid:
    """Beat timing derived from BPM and first beat position."""

    first_beat_ms: float
    bpm: float

    @property
    def ms_per_beat(self) -> float:
        return 60_000 / self.bpm

    def beat_to_ms(self, beat: int) -> float:
        """Convert a 1-indexed beat number to milliseconds."""
        return self.first_beat_ms + (beat - 1) * self.ms_per_beat

    def ms_to_beat(self, ms: float) -> int:
        """Convert milliseconds to the nearest 1-indexed beat number."""
        raw = (ms - self.first_beat_ms) / self.ms_per_beat + 1
        return max(1, round(raw))

    def bars_to_ms(self, bars: int) -> float:
        """Convert a number of bars (4 beats each) to milliseconds."""
        return bars * 4 * self.ms_per_beat


@dataclass
class CuePoint:
    """A single cue point (hot cue or memory cue)."""

    kind: int  # 0=memory, 1-9=hot cue slot
    position_ms: float
    loop_end_ms: float | None  # None if not a loop
    color_table_index: int | None
    color: int
    comment: str

    @property
    def is_loop(self) -> bool:
        return self.loop_end_ms is not None


@dataclass
class Phrase:
    """A phrase segment from PSSI analysis."""

    beat_start: int
    beat_end: int  # start beat of next phrase (exclusive)
    kind: int  # raw PSSI kind value
    label: str  # resolved label (Intro, Up, Down, Chorus, Outro)
    position_ms: float
    duration_ms: float

    @property
    def beat_length(self) -> int:
        return self.beat_end - self.beat_start


@dataclass
class WaveformPoint:
    """A single point in the color waveform."""

    height: float  # 0.0–1.0 normalized amplitude
    red: int  # 0–7 (bass)
    green: int  # 0–7 (mid)
    blue: int  # 0–7 (treble)

    @property
    def rgb_hex(self) -> str:
        """Scale 3-bit color to hex, clamping to 0-7."""
        r = min(self.red, 7) * 255 // 7
        g = min(self.green, 7) * 255 // 7
        b = min(self.blue, 7) * 255 // 7
        return f"#{r:02x}{g:02x}{b:02x}"


@dataclass
class Track:
    """A rekordbox track with analysis data."""

    id: int
    title: str
    artist: str
    bpm: float  # actual BPM (128.0, not 12800)
    duration_ms: float
    analysis_path: str
    cues: list[CuePoint]
    phrases: list[Phrase]
    beat_grid: BeatGrid
    waveform: list[WaveformPoint] | None = None  # color waveform data
    vocal_track: list[int] | None = None  # per-frame vocal confidence (0-4), ~46ms per frame
    audio_path: str | None = None  # real audio file path (DjmdContent.FolderPath), for audio.py


@dataclass
class CueSlot:
    """One row from the cue system definition."""

    pad: str  # A-H
    kind: int  # DB Kind value (1,2,3,5,6,7,8,9)
    hot_cue_label: str
    memory_cue_label: str
    hot_cue_color_table_index: int
    hot_cue_color: int
    memory_cue_color_table_index: int | None
    memory_cue_color: int
    is_loop: bool
    memory_offset_bars: int  # 0 for same-position slots, 16 for others


@dataclass
class CueProposal:
    """The result of running the cue strategy on a track."""

    track: Track
    hot_cues: list[CuePoint]
    memory_cues: list[CuePoint]
    confidence: dict[str, float]  # pad letter -> 0.0-1.0
    notes: list[str]  # human-readable explanations


@dataclass
class RawBeatGridEntry:
    """One entry from Rekordbox's own full per-beat PQTZ array -- the
    beat-in-bar position (1-4), the tempo Rekordbox recorded at this
    specific beat, and its timestamp. djcues's BeatGrid collapses all of
    this into one constant (first_beat_ms, bpm); this preserves what
    Rekordbox's own analysis actually stored, entry by entry."""

    beat_in_bar: int
    bpm: float
    time_ms: float


@dataclass
class SelfConsistencyResult:
    """Whether Rekordbox's own full beat grid is internally consistent
    with the constant-tempo model djcues currently assumes -- computed
    purely from RawBeatGridEntry data already extracted from Rekordbox.
    No audio file needed."""

    is_consistent: bool
    tempo_varies: bool
    max_pairwise_gap_error_ms: float
    cumulative_drift_at_end_ms: float
    entry_count: int
    notes: list[str]


@dataclass
class AudioBeatVerification:
    """Result of comparing real, audio-detected beat times against a
    track's stored BeatGrid -- tier 2 of beat-grid verification, the
    real-model half. verdict is one of "consistent", "drift_detected",
    or "no_beats_detected"."""

    matched_beats: int
    mean_abs_drift_ms: float
    max_abs_drift_ms: float
    pct_within_tolerance: float
    tracker_name: str
    verdict: str


@dataclass
class BeatGridReport:
    """Top-level beat-grid verification result for one track --
    combines the free self-consistency check with the real audio-based
    one, if it ran. audio is None whenever tier 2 wasn't reached (the
    free check passed and wasn't forced, or real audio wasn't
    available). status is one of "ok", "flagged", "no_grid_data",
    "audio_unavailable", "audio_extra_missing", or "decode_failed"."""

    track_id: int
    title: str
    self_consistency: SelfConsistencyResult
    audio: AudioBeatVerification | None
    status: str


@dataclass
class DropRefinement:
    """Result of checking one already-placed cue (D, E, or F) against
    real audio for a dominant nearby energy transition -- a rise for
    D/F, a dip for E -- drop_enhance.py's core output. Deliberately
    never an independent redetection: refined_ms can only ever be a
    position found strictly inside the bounded search window around
    original_ms.

    outcome is one of:
    - "confirmed": audio agrees with the existing position, unchanged.
    - "refined": a clearly dominant transient was found elsewhere in
      the window, refined_ms replaces original_ms.
    - "inconclusive": nothing clear enough to act on either way, the
      original position is left untouched (same as "confirmed" in
      effect, but says so honestly rather than implying agreement).

    strength is the ratio of the winning frame's energy-rise to the
    rise measured right at original_ms (informational, not itself a
    threshold consumers should re-check). source is "full_mix" or
    "stems" (bass+drums from Demucs, --deep only) depending on which
    signal was analyzed."""

    pad: str
    outcome: str
    original_ms: float
    refined_ms: float
    offset_ms: float
    strength: float
    source: str
    note: str
