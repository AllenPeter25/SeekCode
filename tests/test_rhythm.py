import numpy as np
import pytest

from src.seekaudio.features.rhythm import extract_rhythm_features


@pytest.mark.parametrize("tempo", [90, 120, 180])
def test_rhythm_estimates_regular_pulses(tempo: int) -> None:
    sample_rate = 8_000
    waveform = np.zeros(sample_rate * 8, dtype=np.float32)
    beat_samples = round(sample_rate * 60 / tempo)
    for start in range(beat_samples, len(waveform), beat_samples):
        waveform[start : start + 80] = 1.0

    features = extract_rhythm_features(
        waveform, sample_rate=sample_rate, n_fft=256, hop_length=80,
    )

    assert features["rhythm_tempo_bpm"] == pytest.approx(tempo, abs=3)
    assert features["rhythm_onset_mean"] > 0
    assert features["rhythm_onset_std"] > 0


def test_rhythm_handles_silence() -> None:
    features = extract_rhythm_features(np.zeros(22_050, dtype=np.float32))

    assert features == {
        "rhythm_tempo_bpm": 0.0,
        "rhythm_onset_mean": 0.0,
        "rhythm_onset_std": 0.0,
    }
