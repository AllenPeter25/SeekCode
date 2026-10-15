"""Measure frequency distribution and sign changes over short audio windows."""

import numpy as np
from numpy.typing import NDArray

from .audio import split_into_frames


def extract_spectral_features(
    waveform: NDArray[np.float32],
    sample_rate: int = 22_050,
    n_fft: int = 2_048,
    hop_length: int = 512,
    roll_percent: float = 0.85,
) -> dict[str, float]:
    """Return mean/std for centroid, bandwidth, rolloff, and zero-crossing rate.

    Centroid, bandwidth, and rolloff are in Hz and use spectral magnitude weights.
    roll_percent defaults to 0.85, meaning 85% of cumulative magnitude, not power.
    Zero-crossing rate is a fraction of adjacent sample pairs changing sign.
    All-silent frames receive zero frequency statistics.
    """
    if waveform.ndim != 1:
        raise ValueError("Waveform must be mono")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be greater than zero")
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("Window settings must be greater than zero")
    if not 0 < roll_percent < 1:
        raise ValueError("Roll percent must be between zero and one")
    if waveform.size < n_fft:
        raise ValueError("Waveform is shorter than n_fft")

    frames = split_into_frames(waveform, n_fft, hop_length)
    windowed_frames = frames * np.hanning(n_fft)
    magnitudes = np.abs(np.fft.rfft(windowed_frames, axis=1))
    frequencies = np.fft.rfftfreq(n_fft, d=1 / sample_rate)
    magnitude_sums = magnitudes.sum(axis=1)

    # Centroid is the magnitude-weighted average frequency of each frame.
    # out/where avoid division by zero when a frame contains only silence.
    centroids = np.divide(
        magnitudes @ frequencies,
        magnitude_sums,
        out=np.zeros_like(magnitude_sums),
        where=magnitude_sums > 0,
    )
    squared_distances = np.square(frequencies - centroids[:, np.newaxis])
    # Bandwidth is the weighted RMS distance of frequencies from their centroid.
    # Adding a new axis broadcasts each frame's centroid across its frequency bins.
    bandwidths = np.sqrt(
        np.divide(
            np.sum(magnitudes * squared_distances, axis=1),
            magnitude_sums,
            out=np.zeros_like(magnitude_sums),
            where=magnitude_sums > 0,
        )
    )
    # Find the first frequency bin crossing the chosen cumulative-magnitude threshold.
    cumulative_magnitudes = np.cumsum(magnitudes, axis=1)
    rolloff_indices = np.argmax(
        cumulative_magnitudes >= roll_percent * magnitude_sums[:, np.newaxis],
        axis=1,
    )
    rolloffs = frequencies[rolloff_indices]
    rolloffs[magnitude_sums == 0] = 0
    # Compare neighboring signs in the original frames, not the windowed spectrum.
    # signbit treats exact zero as non-negative; no amplitude threshold is applied.
    zero_crossings = np.not_equal(
        np.signbit(frames[:, 1:]),
        np.signbit(frames[:, :-1]),
    )
    zero_crossing_rates = np.mean(zero_crossings, axis=1)

    # Each frame-level sequence becomes two stable columns for a track.
    return {
        "spectral_centroid_mean": float(np.mean(centroids)),
        "spectral_centroid_std": float(np.std(centroids)),
        "spectral_bandwidth_mean": float(np.mean(bandwidths)),
        "spectral_bandwidth_std": float(np.std(bandwidths)),
        "spectral_rolloff_mean": float(np.mean(rolloffs)),
        "spectral_rolloff_std": float(np.std(rolloffs)),
        "spectral_zero_crossing_mean": float(np.mean(zero_crossing_rates)),
        "spectral_zero_crossing_std": float(np.std(zero_crossing_rates)),
    }
