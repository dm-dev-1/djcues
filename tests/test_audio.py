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
