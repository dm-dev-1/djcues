import pytest

from djcues.audio import AudioExtraUnavailableError, load_audio, resolve_audio_path
from djcues.models import BeatGrid, Track
from tests.conftest import requires_audio


def _track_with_audio_path(audio_path: str | None) -> Track:
    return Track(
        id=1, title="Test", artist="Test", bpm=128.0, duration_ms=1000.0,
        analysis_path="", cues=[], phrases=[],
        beat_grid=BeatGrid(first_beat_ms=0.0, bpm=128.0),
        audio_path=audio_path,
    )


def test_resolve_audio_path_none_when_track_has_no_path():
    assert resolve_audio_path(_track_with_audio_path(None)) is None


def test_resolve_audio_path_none_when_file_does_not_exist(tmp_path):
    missing = tmp_path / "does-not-exist.wav"
    assert resolve_audio_path(_track_with_audio_path(str(missing))) is None


def test_resolve_audio_path_returns_path_when_file_exists(tmp_path):
    real_file = tmp_path / "track.wav"
    real_file.write_bytes(b"not really a wav, just needs to exist")
    result = resolve_audio_path(_track_with_audio_path(str(real_file)))
    assert result == real_file


def test_resolve_audio_path_none_when_path_is_a_directory(tmp_path):
    # A moved/renamed file could leave a stale path pointing at a
    # directory in a pathological case -- must not be treated as valid.
    assert resolve_audio_path(_track_with_audio_path(str(tmp_path))) is None


@requires_audio
def test_load_audio_decodes_a_real_wav(tmp_path):
    import numpy as np
    import soundfile as sf

    sr = 22050
    samples = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)  # 1s, 440Hz
    path = tmp_path / "tone.wav"
    sf.write(path, samples, sr)

    loaded = load_audio(path)

    assert loaded.sr == sr
    assert loaded.duration_ms == pytest.approx(1000.0, abs=5.0)
    assert len(loaded.samples) > 0
    assert loaded.samples.ndim == 1  # mono


@requires_audio
def test_load_audio_resamples_when_sr_given(tmp_path):
    import numpy as np
    import soundfile as sf

    orig_sr = 44100
    samples = np.zeros(orig_sr, dtype=np.float32)  # 1s of silence
    path = tmp_path / "silence.wav"
    sf.write(path, samples, orig_sr)

    loaded = load_audio(path, sr=16000)

    assert loaded.sr == 16000
    assert loaded.duration_ms == pytest.approx(1000.0, abs=5.0)


def test_load_audio_raises_typed_error_without_librosa(tmp_path, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "librosa", None)
    fake_path = tmp_path / "whatever.wav"
    fake_path.write_bytes(b"x")
    with pytest.raises(AudioExtraUnavailableError, match=r"djcues\[audio\]"):
        load_audio(fake_path)


def _write_synthetic_m4a(path, sr: int = 44100, duration_s: float = 1.0, freq: float = 440.0):
    """A real, if tiny, AAC-in-M4A file -- encoded with PyAV itself so
    this test needs no real music file and works the same in CI as on
    a real machine. soundfile can't open AAC/M4A at all (confirmed
    directly against real library files this session), so decoding
    this back through load_audio() exercises the real PyAV fallback
    path, not a mock of it.
    """
    import av
    import numpy as np

    n = int(sr * duration_s)
    t = np.arange(n)
    mono = (0.3 * np.sin(2 * np.pi * freq * t / sr)).astype(np.float32)

    container = av.open(str(path), mode="w")
    stream = container.add_stream("aac", rate=sr)
    stream.layout = "mono"
    frame = av.AudioFrame.from_ndarray(mono.reshape(1, -1), format="fltp", layout="mono")
    frame.sample_rate = sr
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()


@requires_audio
def test_load_audio_decodes_via_pyav_fallback_for_unsupported_container(tmp_path):
    import numpy as np

    path = tmp_path / "tone.m4a"
    _write_synthetic_m4a(path, sr=44100, duration_s=1.0)

    loaded = load_audio(path)

    assert loaded.sr == 44100
    assert loaded.duration_ms == pytest.approx(1000.0, abs=100.0)  # AAC encoder priming adds a little
    assert loaded.samples.ndim == 1
    assert loaded.samples.dtype == np.float32


@requires_audio
def test_load_audio_pyav_fallback_resamples_when_sr_given(tmp_path):
    path = tmp_path / "tone.m4a"
    _write_synthetic_m4a(path, sr=44100, duration_s=1.0)

    loaded = load_audio(path, sr=16000)

    assert loaded.sr == 16000
    assert loaded.duration_ms == pytest.approx(1000.0, abs=100.0)


@requires_audio
def test_load_audio_does_not_touch_pyav_for_a_supported_format(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    from djcues import audio as audio_module

    sr = 22050
    samples = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32)
    path = tmp_path / "tone.wav"
    sf.write(path, samples, sr)

    def _boom(*args, **kwargs):
        raise AssertionError("PyAV fallback must not run for a format soundfile already supports")

    monkeypatch.setattr(audio_module, "_load_via_pyav", _boom)

    loaded = load_audio(path)  # would raise via _boom if the fast path regressed
    assert loaded.sr == sr


@requires_audio
def test_load_audio_propagates_error_when_both_decoders_fail(tmp_path):
    garbage = tmp_path / "not-really-audio.m4a"
    garbage.write_bytes(b"this is not a real audio file, just garbage bytes" * 10)

    with pytest.raises(Exception):
        load_audio(garbage)
