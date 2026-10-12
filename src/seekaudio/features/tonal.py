import numpy as np
from numpy.typing import NDArray


def extract_tonal_features(
    waveform: NDArray[np.float32],
    sample_rate: int = 22_050,
    n_fft: int = 2_048,
    hop_length: int = 512,
) -> dict[str, float]:
    if waveform.ndim != 1 or waveform.size < n_fft:
        raise ValueError("Tonal features require mono audio of at least n_fft samples")
    frames = np.lib.stride_tricks.sliding_window_view(waveform, n_fft)[::hop_length]
    power = np.abs(np.fft.rfft(frames * np.hanning(n_fft), axis=1)) ** 2
    frequencies = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    audible = frequencies >= 27.5
    pitch_classes = np.rint(69 + 12 * np.log2(frequencies[audible] / 440)).astype(int) % 12
    chroma = np.zeros((len(frames), 12))
    for pitch_class in range(12):
        chroma[:, pitch_class] = power[:, audible][:, pitch_classes == pitch_class].sum(axis=1)
    totals = chroma.sum(axis=1, keepdims=True)
    chroma = np.divide(chroma, totals, out=np.zeros_like(chroma), where=totals > 0)
    pitch = np.arange(12)
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
