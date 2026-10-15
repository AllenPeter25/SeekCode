"""Create the display's log-Mel matrix using the same filter bank as custom MFCCs."""

import numpy as np

from .mfcc import calculate_mel_power


def compute_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = 22_050,
    n_fft: int = 2_048,
    hop_length: int = 512,
    n_mels: int = 128,
) -> np.ndarray:
    """Return a (Mel bands, frames) matrix in dB relative to the excerpt's peak.

    Values are clipped to [-80, 0]. This is a relative visualization, not dBFS or
    perceptual loudness. Silence is explicitly dark (-80), rather than a false
    bright zero-dB image caused by subtracting the log of an all-zero peak.
    """
    if sample_rate <= 0 or n_fft <= 0 or hop_length <= 0 or n_mels <= 0:
        raise ValueError("Spectrogram settings must be greater than zero")
    if waveform.ndim != 1 or waveform.size < n_fft:
        raise ValueError("Spectrogram requires mono audio of at least n_fft samples")
    if not np.isfinite(waveform).all():
        raise ValueError("Waveform must contain only finite numbers")

    mel_power = calculate_mel_power(waveform, sample_rate, n_fft, hop_length, n_mels)
    peak = float(mel_power.max())
    if peak == 0:
        return np.full((n_mels, len(mel_power)), -80.0)

    # A tiny positive floor makes log10 safe. Subtracting peak dB sets the maximum to 0.
    decibels = 10 * np.log10(np.maximum(mel_power, np.finfo(float).tiny)) - 10 * np.log10(peak)
    return np.clip(decibels.T, -80, 0)
