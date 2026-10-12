from pathlib import Path
import wave

import numpy as np
import pytest

from src.seekaudio.features.audio import load_audio


def test_load_audio_pads_short_track(tmp_path: Path) -> None:
    audio_path = tmp_path / "short.wav"

    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(8_000)
        audio_file.writeframes(b"\x01\x00" * 8_000)

    waveform = load_audio(audio_path, sample_rate=16_000, duration=2.0)

    assert waveform.shape == (32_000,)
    assert waveform.dtype == np.float32
    assert np.all(waveform[:8_000] == 0)
    assert np.all(waveform[-8_000:] == 0)


def test_load_audio_rejects_invalid_settings(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Sample rate"):
        load_audio(tmp_path / "audio.wav", sample_rate=0)

    with pytest.raises(ValueError, match="Duration"):
        load_audio(tmp_path / "audio.wav", duration=0)
