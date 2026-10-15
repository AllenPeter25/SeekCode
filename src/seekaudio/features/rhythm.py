"""The simple RMS-based rhythm features stored in version 2 retrieval tables.

The UI's separate Librosa tempo lives in librosa_analysis.py. Keeping the stored
estimator unchanged preserves compatibility with the already fitted catalog.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate

from .audio import split_into_frames


def extract_rhythm_features(
    waveform: NDArray[np.float32],
    sample_rate: int = 22_050,
    n_fft: int = 2_048,
    hop_length: int = 512,
) -> dict[str, float]:
    """Estimate 40-200 BPM repetition and summarize positive RMS changes.

    An onset here is a rise in window RMS, not a detected musical beat. Tempo is
    chosen from repeating onset patterns using autocorrelation. Zero means no
    supported repetition was found. Half/double-time estimates remain possible.
    """
    if waveform.ndim != 1:
        raise ValueError("Waveform must be mono")
    if sample_rate <= 0 or n_fft <= 0 or hop_length <= 0:
        raise ValueError("Sample rate and window settings must be greater than zero")
    if waveform.size < n_fft:
        raise ValueError("Waveform is shorter than n_fft")
    if not np.isfinite(waveform).all():
        raise ValueError("Waveform must contain only finite numbers")

    frames = split_into_frames(waveform, n_fft, hop_length)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    # Keep increases only. Falling energy is not an attack in this simple estimator.
    onsets = np.maximum(np.diff(rms, prepend=rms[0]), 0)
    frame_rate = sample_rate / hop_length
    # Convert the BPM search interval into frame delays: faster beats have shorter lags.
    minimum_lag = max(1, int(np.ceil(60 * frame_rate / 200)))
    maximum_lag = min(len(onsets) - 1, int(np.floor(60 * frame_rate / 40)))
    tempo = 0.0

    if np.any(onsets > 0) and maximum_lag >= minimum_lag:
        # Smoothing reduces sensitivity to a beat falling between two window starts.
        smoothed_onsets = gaussian_filter1d(onsets, sigma=1.0)
        autocorrelation = correlate(smoothed_onsets, smoothed_onsets, mode="full", method="fft")
        autocorrelation = autocorrelation[len(onsets) - 1 :]
        # Correlating the envelope with delayed copies measures repeated attack timing.
        # Retain positive lags and exclude lag zero (the envelope matched with itself).
        candidates = autocorrelation[minimum_lag : maximum_lag + 1]
        if candidates.max() > autocorrelation[0] * 1e-6:
            best_lag = minimum_lag + int(np.argmax(candidates))
            tempo = 60 * frame_rate / best_lag

    return {
        "rhythm_tempo_bpm": float(tempo),
        "rhythm_onset_mean": float(np.mean(onsets)),
        "rhythm_onset_std": float(np.std(onsets)),
    }
