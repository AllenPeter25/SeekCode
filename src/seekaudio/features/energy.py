import numpy as np
from numpy.typing import NDArray

from .audio import split_into_frames


def extract_energy_features(
    waveform: NDArray[np.float32],
    n_fft: int = 2_048,
    hop_length: int = 512,
) -> dict[str, float]:
    frames = split_into_frames(waveform, n_fft, hop_length)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))

    return {
        "energy_rms_mean": float(np.mean(rms)),
        "energy_rms_std": float(np.std(rms)),
    }
