"""Audio file loading -- mirrors db.py's role, for the real audio file
instead of Rekordbox's pre-computed ANLZ analysis data.

Only ever touched when a caller explicitly asks for audio-based
analysis (beat_verify.py's tier 2, drop_enhance.py) -- normal djcues
commands (propose/compare/review/viz) never import this module, so they
pay zero extra cost or dependency requirement for it existing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from djcues.models import Track


class AudioExtraUnavailableError(RuntimeError):
    """Raised when `librosa`/`soundfile` aren't installed."""


def _require_librosa() -> Any:
    try:
        import librosa
    except ImportError as e:
        raise AudioExtraUnavailableError(
            "Audio-based analysis needs the 'librosa' and 'soundfile' packages. "
            "Install them with: pip install djcues[audio]"
        ) from e
    return librosa


def resolve_audio_path(track: Track) -> Path | None:
    """Resolve and validate the real audio file path for a track.

    Returns None if the track has no recorded audio_path, or if the
    path doesn't actually exist on disk right now (the file may have
    been moved, renamed, or its drive isn't currently mounted --
    Rekordbox itself tolerates this happening, so djcues must too).

    Pure pathlib -- no dependency on the `audio` extra, always safe and
    cheap to call regardless of whether librosa/soundfile are installed.
    """
    if not track.audio_path:
        return None
    path = Path(track.audio_path)
    if not path.is_file():
        return None
    return path


@dataclass
class LoadedAudio:
    """Decoded audio, ready for analysis. Mono (channels averaged),
    float32 samples in [-1.0, 1.0]."""

    samples: np.ndarray
    sr: int
    duration_ms: float


def load_audio(path: Path, sr: int | None = None) -> LoadedAudio:
    """Decode an audio file to mono float32 samples.

    Lazy-imports librosa (the `audio` extra) -- raises
    AudioExtraUnavailableError with a clear install hint if it's
    missing, rather than an unrelated-looking ImportError surfacing
    deep in some other call stack.

    sr=None (the default) keeps the file's native sample rate; pass an
    explicit rate to resample to whatever a specific model expects
    (e.g. Demucs and beat_this each have their own expected rate --
    resampling is their callers' job, not this function's, since the
    right rate depends on which model is about to consume the audio).
    """
    librosa = _require_librosa()
    samples, actual_sr = librosa.load(str(path), sr=sr, mono=True)
    duration_ms = (len(samples) / actual_sr) * 1000.0
    return LoadedAudio(samples=samples, sr=actual_sr, duration_ms=duration_ms)
