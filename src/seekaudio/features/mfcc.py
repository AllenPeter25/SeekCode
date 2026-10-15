"""Summarize timbre using Mel-frequency cepstral coefficients (MFCCs).

Waveform -> Hann-windowed power spectra -> triangular HTK Mel filters
-> log-power values -> discrete cosine transform -> coefficient means/stds.
These custom settings are not numerically equal to Librosa's default MFCCs.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dct

from .audio import calculate_power_spectrum, split_into_frames


def _create_mel_filter_bank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
) -> NDArray[np.float64]:
    """Build n_mels triangular filters over the FFT's non-negative frequency bins.

    HTK Mel conversion compresses high frequencies relative to low frequencies.
    Two extra boundary points define the left/right edges of the first/last
    triangle. Filters are peak-normalized triangles, without Slaney area scaling.
    Returns shape (n_mels, n_fft // 2 + 1).
    """
    highest_mel = 2_595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_points = np.linspace(0, highest_mel, n_mels + 2)
    frequencies = 700 * (10 ** (mel_points / 2_595) - 1)
    bins = np.floor((n_fft + 1) * frequencies / sample_rate).astype(int)
    # Map continuous Hz values to available FFT bins and keep indices in range.
    bins = np.clip(bins, 0, n_fft // 2)
    filters = np.zeros((n_mels, n_fft // 2 + 1))

    for index in range(1, n_mels + 1):
        left, center, right = bins[index - 1 : index + 2]

        if center > left:
            # Rising half: weights grow from zero at the left edge toward one.
            filters[index - 1, left:center] = (
                np.arange(left, center) - left
            ) / (center - left)

        if right > center:
            # Falling half: weights decrease from the center toward the right edge.
            filters[index - 1, center:right] = (
                right - np.arange(center, right)
            ) / (right - center)

    return filters


def calculate_mel_power(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    n_mels: int,
) -> np.ndarray:
    """Return Mel-band power shaped (frames, n_mels), shared with the spectrogram."""
    frames = split_into_frames(waveform, n_fft, hop_length)
    power_spectrum = calculate_power_spectrum(frames)
    mel_filters = _create_mel_filter_bank(sample_rate, n_fft, n_mels)
    # Matrix multiplication adds FFT-bin power according to each triangle's weights.
    return power_spectrum @ mel_filters.T


def extract_mfcc_features(
    waveform: NDArray[np.float32],
    sample_rate: int = 22_050,
    n_mfcc: int = 20,
    n_fft: int = 2_048,
    hop_length: int = 512,
    n_mels: int = 128,
) -> dict[str, float]:
    """Return two pooled values per MFCC coefficient: mean and standard deviation.

    n_mfcc cannot exceed n_mels. Coefficient names start at mfcc_01 for display,
    though that first column contains DCT coefficient zero (overall log energy).
    Negative MFCC values are valid; coefficients are not loudness percentages.
    """
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

    mel_spectrum = calculate_mel_power(waveform, sample_rate, n_fft, hop_length, n_mels)
    log_mel_spectrum = 10 * np.log10(
        np.maximum(mel_spectrum, np.finfo(np.float64).tiny)
    )
    # Avoid extreme low-power values by keeping only an 80 dB span below the peak.
    log_mel_spectrum = np.maximum(
        log_mel_spectrum,
        np.max(log_mel_spectrum) - 80,
    )
    # DCT combines correlated Mel bands. Transpose to iterate over coefficients.
    mfccs = dct(log_mel_spectrum, type=2, axis=1, norm="ortho")[:, :n_mfcc].T

    features = {}

    for index, coefficient in enumerate(mfccs, start=1):
        # Pool all frames into fixed-size track statistics, regardless of duration.
        features[f"mfcc_{index:02d}_mean"] = float(np.mean(coefficient))
        features[f"mfcc_{index:02d}_std"] = float(np.std(coefficient))

    return features
