import numpy as np
import pytest

from src.seekaudio.features.energy import extract_energy_features


def test_extract_energy_features_for_constant_signal() -> None:
    waveform = np.ones(4_096, dtype=np.float32)

    features = extract_energy_features(waveform, n_fft=1_024, hop_length=512)

    assert features["energy_rms_mean"] == pytest.approx(1.0)
    assert features["energy_rms_std"] == pytest.approx(0.0)


def test_extract_energy_features_for_silence() -> None:
    waveform = np.zeros(4_096, dtype=np.float32)

    features = extract_energy_features(waveform, n_fft=1_024, hop_length=512)

    assert features == {
        "energy_rms_mean": 0.0,
        "energy_rms_std": 0.0,
    }


def test_extract_energy_features_rejects_stereo() -> None:
    waveform = np.zeros((2, 4_096), dtype=np.float32)

    with pytest.raises(ValueError, match="mono"):
        extract_energy_features(waveform)
