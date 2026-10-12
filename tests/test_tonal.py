import numpy as np

from src.seekaudio.features.tonal import extract_tonal_features


def test_tonal_features_detect_a_pitch_class_and_handle_silence() -> None:
    sample_rate = 8_000
    time = np.arange(sample_rate) / sample_rate
    waveform = np.sin(2 * np.pi * 440 * time).astype(np.float32)
    features = extract_tonal_features(waveform, sample_rate, n_fft=2_048)
    chroma = [features[f"tonal_chroma_{index:02d}_mean"] for index in range(1, 13)]
    assert np.argmax(chroma) == 9
    assert len(features) == 36
    silence = extract_tonal_features(np.zeros(sample_rate, dtype=np.float32), sample_rate)
    assert all(value == 0 for value in silence.values())
