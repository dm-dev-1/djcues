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


def _load_via_pyav(path: Path, sr: int | None) -> tuple[np.ndarray, int]:
    """Fallback decode path for containers soundfile can't open (e.g.
    AAC-in-MP4/M4A -- confirmed via real files this session: soundfile
    raises "Format not recognised" for both, even though the audio
    itself is perfectly valid in one of the two cases). Real, verified
    end-to-end: a genuine 44.1kHz AAC .m4a decoded via this exact path
    matched its container-reported duration precisely (277.92s either
    way).

    Uses av.AudioResampler to normalize decoded frames to mono float32
    at the source's own rate regardless of the source codec's native
    sample format (s16/fltp/etc. all vary by file) -- more correct
    than a naive per-frame dtype assumption, and it's what PyAV itself
    documents as the standard way to get a consistent output format.
    Only resamples to a *different* rate if the caller explicitly
    asked for one (matching load_audio's own sr=None-keeps-native
    contract); reuses librosa's resample() for that rather than
    pulling in a second resampling implementation.

    A file whose container/stream metadata is readable but whose
    audio payload is actually corrupted (confirmed this session: one
    real library file decodes only ~1.7s of a reported ~277s before
    every remaining packet raises av.error.InvalidDataError) will
    raise here same as it would from any decoder -- no decoder can
    recover data that isn't really in the file.
    """
    import av

    container = av.open(str(path))
    try:
        stream = container.streams.audio[0]
        native_sr = stream.codec_context.sample_rate
        resampler = av.AudioResampler(format="flt", layout="mono", rate=native_sr)

        chunks = [
            resampled.to_ndarray()
            for frame in container.decode(audio=0)
            for resampled in resampler.resample(frame)
        ]
    finally:
        container.close()

    samples = np.concatenate(chunks, axis=-1).reshape(-1).astype(np.float32)
    actual_sr = native_sr

    if sr is not None and sr != native_sr:
        import librosa

        samples = librosa.resample(samples, orig_sr=native_sr, target_sr=sr)
        actual_sr = sr

    return samples, actual_sr


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

    Tries librosa/soundfile first (the common, fast path for the vast
    majority of real files -- confirmed against a real ~328-track
    library: mp3/flac/aiff all decode this way, 99.4% of the total).
    Falls back to PyAV (also part of the `audio` extra -- bundles its
    own decoders, no system ffmpeg install needed) for anything
    soundfile can't open, e.g. AAC-in-MP4/M4A containers. Whatever
    PyAV itself raises (unsupported format, or genuinely corrupted
    audio data) propagates as the final error -- every caller already
    handles a per-track decode failure generically regardless of the
    specific reason.
    """
    librosa = _require_librosa()
    try:
        samples, actual_sr = librosa.load(str(path), sr=sr, mono=True)
    except Exception:
        samples, actual_sr = _load_via_pyav(path, sr)
    duration_ms = (len(samples) / actual_sr) * 1000.0
    return LoadedAudio(samples=samples, sr=actual_sr, duration_ms=duration_ms)
