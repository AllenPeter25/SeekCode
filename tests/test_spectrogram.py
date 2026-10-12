import numpy as np

from src.seekaudio.features.spectrogram import compute_mel_spectrogram
from src.seekaudio.viz import render_mel_spectrogram


def test_mel_spectrogram_highlights_a_known_tone() -> None:
    sample_rate = 8_000
    waveform = np.sin(2 * np.pi * 1_000 * np.arange(sample_rate) / sample_rate).astype(np.float32)
    decibels = compute_mel_spectrogram(waveform, sample_rate, n_fft=1_024, hop_length=256, n_mels=32)
    highest_mel = 2_595 * np.log10(1 + (sample_rate / 2) / 700)
    centers = np.linspace(0, highest_mel, 34)[1:-1]
    frequencies = 700 * (10 ** (centers / 2_595) - 1)
    peak_band = np.argmax(decibels.mean(axis=1))

    assert abs(frequencies[peak_band] - 1_000) < 100
    assert np.isfinite(decibels).all()
    assert decibels.min() >= -80
    assert decibels.max() <= 0


def test_silent_spectrogram_is_dark_and_exports_png() -> None:
    waveform = np.zeros(8_000, dtype=np.float32)
    decibels = compute_mel_spectrogram(waveform, sample_rate=8_000, n_mels=32)
    assert np.all(decibels == -80)
    image = render_mel_spectrogram(waveform, 8_000, 2_048, 512, 32)
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
