import numpy as np
from numpy.typing import NDArray
from scipy.fft import dct


def _create_mel_filter_bank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
) -> NDArray[np.float64]:
    highest_mel = 2_595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_points = np.linspace(0, highest_mel, n_mels + 2)
    frequencies = 700 * (10 ** (mel_points / 2_595) - 1)
    bins = np.floor((n_fft + 1) * frequencies / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    filters = np.zeros((n_mels, n_fft // 2 + 1))

    for index in range(1, n_mels + 1):
        left, center, right = bins[index - 1 : index + 2]

        if center > left:
            filters[index - 1, left:center] = (
                np.arange(left, center) - left
            ) / (center - left)

        if right > center:
            filters[index - 1, center:right] = (
                right - np.arange(center, right)
            ) / (right - center)

    return filters


def extract_mfcc_features(
    waveform: NDArray[np.float32],
    sample_rate: int = 22_050,
    n_mfcc: int = 20,
    n_fft: int = 2_048,
    hop_length: int = 512,
    n_mels: int = 128,
) -> dict[str, float]:
    if waveform.ndim != 1:
        raise ValueError("Waveform must be mono")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be greater than zero")
    if n_mfcc <= 0 or n_mels <= 0:
        raise ValueError("MFCC settings must be greater than zero")
    if n_mfcc > n_mels:
        raise ValueError("n_mfcc cannot be greater than n_mels")
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("Window settings must be greater than zero")
    if waveform.size < n_fft:
        raise ValueError("Waveform is shorter than n_fft")

    frames = np.lib.stride_tricks.sliding_window_view(waveform, n_fft)[::hop_length]
    windowed_frames = frames * np.hanning(n_fft)
    power_spectrum = np.abs(np.fft.rfft(windowed_frames, axis=1)) ** 2
    mel_filters = _create_mel_filter_bank(sample_rate, n_fft, n_mels)
    mel_spectrum = power_spectrum @ mel_filters.T
    log_mel_spectrum = 10 * np.log10(
        np.maximum(mel_spectrum, np.finfo(np.float64).tiny)
    )
    log_mel_spectrum = np.maximum(
        log_mel_spectrum,
        np.max(log_mel_spectrum) - 80,
    )
    mfccs = dct(log_mel_spectrum, type=2, axis=1, norm="ortho")[:, :n_mfcc].T

    features = {}

    for index, coefficient in enumerate(mfccs, start=1):
        features[f"mfcc_{index:02d}_mean"] = float(np.mean(coefficient))
        features[f"mfcc_{index:02d}_std"] = float(np.std(coefficient))

    return features
