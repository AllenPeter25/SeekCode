"""Librosa measurements for display only, separate from saved retrieval features."""

import librosa
import numpy as np

from .audio import calculate_power_spectrum, split_into_frames


def extract_librosa_characteristics(
    waveform: np.ndarray,
    sample_rate: int = 22_050,
    n_fft: int = 2_048,
    hop_length: int = 512,
) -> dict[str, float]:
    """Return spectral-flux tempo, spectral flatness, and mean spectral contrast.

    The prepared waveform uses the catalog's audio settings. Librosa expects a
    spectrum shaped (frequency bins, frames), so shared frame-first power is
    transposed. These values do not enter the scaler/PCA or change genre votes.
    """
    if sample_rate <= 400:
        raise ValueError("Librosa analysis requires a sample rate above 400 Hz")
    frames = split_into_frames(waveform, n_fft, hop_length)
    power_spectrum = calculate_power_spectrum(frames).T
    if not np.any(power_spectrum > 0):
        return {
            "librosa_tempo_bpm": 0.0,
            "spectral_flatness": 0.0,
            "spectral_contrast_db": 0.0,
        }

    magnitudes = np.sqrt(power_spectrum)
    # Onset strength measures positive spectral changes on a log-power spectrum.
    # center=False matches our complete-window framing without added edge padding.
    log_power = librosa.power_to_db(power_spectrum, ref=np.max)
    onsets = librosa.onset.onset_strength(
        S=log_power, sr=sample_rate, n_fft=n_fft,
        hop_length=hop_length, center=False,
    )
    tempo = 0.0
    if np.any(onsets > 0):
        # Librosa's default prior favors tempos near 120 BPM; this is still an estimate.
        tempo_values = librosa.feature.tempo(
            onset_envelope=onsets, sr=sample_rate,
            hop_length=hop_length, max_tempo=200,
        )
        tempo = float(tempo_values[0])

    # Flatness compares geometric/arithmetic spectral means: tonal vs noise-like power.
    flatness = librosa.feature.spectral_flatness(S=magnitudes)
    octave_bands = int(np.floor(np.log2((sample_rate / 2) / 200)))
    octave_bands = max(1, min(6, octave_bands))
    # Contrast compares peaks and valleys in octave bands. Limit bands by Nyquist.
    contrast = librosa.feature.spectral_contrast(
        S=magnitudes, sr=sample_rate, n_fft=n_fft, n_bands=octave_bands,
    )
    return {
        "librosa_tempo_bpm": tempo,
        "spectral_flatness": float(flatness.mean()),
        "spectral_contrast_db": float(contrast.mean()),
    }
