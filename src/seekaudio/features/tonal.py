"""Optional pitch-class and tonal-centroid summaries derived directly from the FFT."""

import numpy as np
from numpy.typing import NDArray

from .audio import calculate_power_spectrum, split_into_frames


def extract_tonal_features(
    waveform: NDArray[np.float32],
    sample_rate: int = 22_050,
    n_fft: int = 2_048,
    hop_length: int = 512,
) -> dict[str, float]:
    """Return 36 values: mean/std for 12 chroma bins and 6 tonal coordinates.

    Chroma folds frequencies into pitch classes C through B, ignoring octave.
    This uses rounded FFT pitch classes, without tuning correction or harmonic
    source separation, so it is an approximate tonal representation.
    """
    if waveform.ndim != 1 or waveform.size < n_fft:
        raise ValueError("Tonal features require mono audio of at least n_fft samples")
    frames = split_into_frames(waveform, n_fft, hop_length)
    power = calculate_power_spectrum(frames)
    frequencies = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    audible = frequencies >= 27.5
    # Convert Hz to MIDI-like pitches (A4 = 69 at 440 Hz), then fold modulo 12.
    pitch_classes = np.rint(69 + 12 * np.log2(frequencies[audible] / 440)).astype(int) % 12
    chroma = np.zeros((len(frames), 12))
    for pitch_class in range(12):
        chroma[:, pitch_class] = power[:, audible][:, pitch_classes == pitch_class].sum(axis=1)
    totals = chroma.sum(axis=1, keepdims=True)
    # Normalize each chroma frame to remove overall power; keep silent frames zero.
    chroma = np.divide(chroma, totals, out=np.zeros_like(chroma), where=totals > 0)
    pitch = np.arange(12)
    # Project pitch classes onto three sine/cosine circle pairs: fifths and thirds.
    angles = np.array([7 / 6, 7 / 6, 3 / 2, 3 / 2, 2 / 3, 2 / 3])[:, None] * np.pi * pitch
    projection = np.empty((6, 12))
    projection[::2] = np.sin(angles[::2])
    projection[1::2] = np.cos(angles[1::2])
    projection[4:] *= 0.5
    tonnetz = chroma @ projection.T
    features = {}
    for name, values in [("chroma", chroma), ("tonnetz", tonnetz)]:
        for index in range(values.shape[1]):
            features[f"tonal_{name}_{index + 1:02d}_mean"] = float(values[:, index].mean())
            features[f"tonal_{name}_{index + 1:02d}_std"] = float(values[:, index].std())
    return features
