"""Describe signal strength with RMS energy, not perceived loudness in LUFS."""

import numpy as np
from numpy.typing import NDArray

from .audio import split_into_frames


def extract_energy_features(
    waveform: NDArray[np.float32],
    n_fft: int = 2_048,
    hop_length: int = 512,
) -> dict[str, float]:
    """Return the mean and population standard deviation of window RMS values.

    RMS = square root of the mean squared amplitude. Squaring prevents positive
    and negative samples from cancelling. The mean describes average strength;
    the standard deviation describes changes in strength across the excerpt.
    """
    frames = split_into_frames(waveform, n_fft, hop_length)
    # axis=1 computes one RMS value per frame, using float64 for the arithmetic.
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))

    return {
        "energy_rms_mean": float(np.mean(rms)),
        "energy_rms_std": float(np.std(rms)),
    }
