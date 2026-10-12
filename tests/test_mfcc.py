import numpy as np
import pytest

from src.seekaudio.features.mfcc import extract_mfcc_features


def test_extract_mfcc_features() -> None:
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform = np.sin(2 * np.pi * 440 * time).astype(np.float32)

    features = extract_mfcc_features(
        waveform,
        sample_rate=sample_rate,
        n_mfcc=3,
        n_fft=512,
        hop_length=128,
        n_mels=16,
    )

    assert list(features) == [
        "mfcc_01_mean",
        "mfcc_01_std",
        "mfcc_02_mean",
        "mfcc_02_std",
        "mfcc_03_mean",
        "mfcc_03_std",
    ]
    assert all(np.isfinite(value) for value in features.values())


def test_extract_mfcc_features_rejects_invalid_count() -> None:
    waveform = np.zeros(4_096, dtype=np.float32)

    with pytest.raises(ValueError, match="n_mfcc"):
        extract_mfcc_features(waveform, n_mfcc=20, n_mels=10)
