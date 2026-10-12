import numpy as np
import pytest

from src.seekaudio.features.spectral import extract_spectral_features


def test_spectral_centroid_of_sine_wave() -> None:
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform = np.sin(2 * np.pi * 1_000 * time).astype(np.float32)

    features = extract_spectral_features(
        waveform,
        sample_rate=sample_rate,
        n_fft=1_024,
        hop_length=512,
    )

    assert features["spectral_centroid_mean"] == pytest.approx(1_000, abs=2)
    assert features["spectral_centroid_std"] == pytest.approx(0, abs=1)
    assert features["spectral_bandwidth_mean"] < 100
    assert features["spectral_bandwidth_std"] < 20
    assert features["spectral_rolloff_mean"] == pytest.approx(1_000, abs=10)
    assert features["spectral_rolloff_std"] == pytest.approx(0, abs=10)
    assert features["spectral_zero_crossing_mean"] == pytest.approx(0.25, abs=0.01)
    assert features["spectral_zero_crossing_std"] == pytest.approx(0, abs=0.01)


def test_spectral_centroid_of_silence() -> None:
    waveform = np.zeros(4_096, dtype=np.float32)

    features = extract_spectral_features(waveform, n_fft=1_024)

    assert features == {
        "spectral_centroid_mean": 0.0,
        "spectral_centroid_std": 0.0,
        "spectral_bandwidth_mean": 0.0,
        "spectral_bandwidth_std": 0.0,
        "spectral_rolloff_mean": 0.0,
        "spectral_rolloff_std": 0.0,
        "spectral_zero_crossing_mean": 0.0,
        "spectral_zero_crossing_std": 0.0,
    }


def test_spectral_features_reject_stereo() -> None:
    waveform = np.zeros((2, 4_096), dtype=np.float32)

    with pytest.raises(ValueError, match="mono"):
        extract_spectral_features(waveform)


def test_spectral_features_reject_invalid_roll_percent() -> None:
    waveform = np.zeros(4_096, dtype=np.float32)

    with pytest.raises(ValueError, match="Roll percent"):
        extract_spectral_features(waveform, roll_percent=1.0)
