import numpy as np
from numpy.typing import NDArray


def extract_energy_features(
    waveform: NDArray[np.float32],
    n_fft: int = 2_048,
    hop_length: int = 512,
) -> dict[str, float]:
    if waveform.ndim != 1:
        raise ValueError("Waveform must be mono")
    if n_fft <= 0 or hop_length <= 0:
        raise ValueError("Window settings must be greater than zero")
    if waveform.size < n_fft:
        raise ValueError("Waveform is shorter than n_fft")

    frames = np.lib.stride_tricks.sliding_window_view(waveform, n_fft)[::hop_length]
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))

    return {
        "energy_rms_mean": float(np.mean(rms)),
        "energy_rms_std": float(np.std(rms)),
    }
