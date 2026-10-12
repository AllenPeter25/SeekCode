import numpy as np
from numpy.typing import NDArray


def extract_spectral_features(
    waveform: NDArray[np.float32],
    sample_rate: int = 22_050,
    n_fft: int = 2_048,
    hop_length: int = 512,
    roll_percent: float = 0.85,
) -> dict[str, float]:
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

    frames = np.lib.stride_tricks.sliding_window_view(waveform, n_fft)[::hop_length]
    windowed_frames = frames * np.hanning(n_fft)
    magnitudes = np.abs(np.fft.rfft(windowed_frames, axis=1))
    frequencies = np.fft.rfftfreq(n_fft, d=1 / sample_rate)
    magnitude_sums = magnitudes.sum(axis=1)

    centroids = np.divide(
        magnitudes @ frequencies,
        magnitude_sums,
        out=np.zeros_like(magnitude_sums),
        where=magnitude_sums > 0,
    )
    squared_distances = np.square(frequencies - centroids[:, np.newaxis])
    bandwidths = np.sqrt(
        np.divide(
            np.sum(magnitudes * squared_distances, axis=1),
            magnitude_sums,
            out=np.zeros_like(magnitude_sums),
            where=magnitude_sums > 0,
        )
    )
    cumulative_magnitudes = np.cumsum(magnitudes, axis=1)
    rolloff_indices = np.argmax(
        cumulative_magnitudes >= roll_percent * magnitude_sums[:, np.newaxis],
        axis=1,
    )
    rolloffs = frequencies[rolloff_indices]
    rolloffs[magnitude_sums == 0] = 0
    zero_crossings = np.not_equal(
        np.signbit(frames[:, 1:]),
        np.signbit(frames[:, :-1]),
    )
    zero_crossing_rates = np.mean(zero_crossings, axis=1)

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
