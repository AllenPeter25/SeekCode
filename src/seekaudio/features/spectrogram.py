import numpy as np

from .mfcc import _create_mel_filter_bank


def compute_mel_spectrogram(
    waveform: np.ndarray,
    sample_rate: int = 22_050,
    n_fft: int = 2_048,
    hop_length: int = 512,
    n_mels: int = 128,
) -> np.ndarray:
    if sample_rate <= 0 or n_fft <= 0 or hop_length <= 0 or n_mels <= 0:
        raise ValueError("Spectrogram settings must be greater than zero")
    if waveform.ndim != 1 or waveform.size < n_fft:
        raise ValueError("Spectrogram requires mono audio of at least n_fft samples")
    if not np.isfinite(waveform).all():
        raise ValueError("Waveform must contain only finite numbers")

    frames = np.lib.stride_tricks.sliding_window_view(waveform, n_fft)[::hop_length]
    power = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)) ** 2
    mel_power = power @ _create_mel_filter_bank(sample_rate, n_fft, n_mels).T
    peak = float(mel_power.max())
    if peak == 0:
        return np.full((n_mels, len(frames)), -80.0)

    decibels = 10 * np.log10(np.maximum(mel_power, np.finfo(float).tiny)) - 10 * np.log10(peak)
    return np.clip(decibels.T, -80, 0)
