from math import gcd
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(
    audio_path: str | Path,
    sample_rate: int = 22_050,
    duration: float = 30.0,
    mono: bool = True,
) -> NDArray[np.float32]:
    if sample_rate <= 0:
        raise ValueError("Sample rate must be greater than zero")
    if duration <= 0:
        raise ValueError("Duration must be greater than zero")

    waveform, original_sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True,
    )
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("Audio must contain finite, nonempty samples")

    if mono:
        waveform = waveform.mean(axis=1)
    else:
        waveform = waveform.T

    if original_sample_rate != sample_rate:
        common_factor = gcd(original_sample_rate, sample_rate)
        waveform = resample_poly(
            waveform,
            sample_rate // common_factor,
            original_sample_rate // common_factor,
            axis=-1,
        )

    target_samples = round(sample_rate * duration)
    current_samples = waveform.shape[-1]

    if current_samples > target_samples:
        start = (current_samples - target_samples) // 2
        waveform = waveform[..., start : start + target_samples]
    elif current_samples < target_samples:
        missing_samples = target_samples - current_samples
        left_padding = missing_samples // 2
        right_padding = missing_samples - left_padding
        padding = [(0, 0)] * waveform.ndim
        padding[-1] = (left_padding, right_padding)
        waveform = np.pad(waveform, padding)

    return waveform.astype(np.float32)


def split_into_frames(waveform: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if waveform.ndim != 1:
        raise ValueError("Waveform must be mono")
    if frame_length <= 0 or hop_length <= 0:
        raise ValueError("Window settings must be greater than zero")
    if waveform.size < frame_length:
        raise ValueError("Waveform is shorter than n_fft")
    if not np.isfinite(waveform).all():
        raise ValueError("Waveform must contain only finite numbers")

    all_windows = np.lib.stride_tricks.sliding_window_view(waveform, frame_length)
    return all_windows[::hop_length]


def calculate_power_spectrum(frames: np.ndarray) -> np.ndarray:
    window = np.hanning(frames.shape[1])
    windowed_frames = frames * window
    frequency_values = np.fft.rfft(windowed_frames, axis=1)
    return np.abs(frequency_values) ** 2
