import numpy as np
import pytest

from src.seekaudio.features.librosa_analysis import extract_librosa_characteristics


def test_librosa_flatness_distinguishes_tone_from_noise() -> None:
    sample_rate = 8_000
    time = np.arange(sample_rate * 3) / sample_rate
    tone = np.sin(2 * np.pi * 440 * time).astype(np.float32)
    noise = np.random.default_rng(42).normal(size=len(time)).astype(np.float32)
    tone_features = extract_librosa_characteristics(tone, sample_rate, 1_024, 256)
    noise_features = extract_librosa_characteristics(noise, sample_rate, 1_024, 256)

    assert tone_features["spectral_flatness"] < noise_features["spectral_flatness"]
    assert all(np.isfinite(value) for value in tone_features.values())
    assert all(np.isfinite(value) for value in noise_features.values())


def test_librosa_tempo_estimates_regular_pulses() -> None:
    sample_rate = 8_000
    waveform = np.zeros(sample_rate * 8, dtype=np.float32)
    for start in range(4_000, len(waveform), 4_000):
        waveform[start : start + 80] = 1.0

    features = extract_librosa_characteristics(waveform, sample_rate, 256, 80)
    assert features["librosa_tempo_bpm"] == pytest.approx(120, abs=3)


def test_librosa_handles_silence() -> None:
    features = extract_librosa_characteristics(np.zeros(8_000, dtype=np.float32), 8_000)
    assert all(value == 0 for value in features.values())
