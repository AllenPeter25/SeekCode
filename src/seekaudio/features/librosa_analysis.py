import librosa
import numpy as np

from .audio import calculate_power_spectrum, split_into_frames


def extract_librosa_characteristics(
    waveform: np.ndarray,
    sample_rate: int = 22_050,
    n_fft: int = 2_048,
    hop_length: int = 512,
) -> dict[str, float]:
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
    log_power = librosa.power_to_db(power_spectrum, ref=np.max)
    onsets = librosa.onset.onset_strength(
        S=log_power, sr=sample_rate, n_fft=n_fft,
        hop_length=hop_length, center=False,
    )
    tempo = 0.0
    if np.any(onsets > 0):
        tempo_values = librosa.feature.tempo(
            onset_envelope=onsets, sr=sample_rate,
            hop_length=hop_length, max_tempo=200,
        )
        tempo = float(tempo_values[0])

    flatness = librosa.feature.spectral_flatness(S=magnitudes)
    octave_bands = int(np.floor(np.log2((sample_rate / 2) / 200)))
    octave_bands = max(1, min(6, octave_bands))
    contrast = librosa.feature.spectral_contrast(
        S=magnitudes, sr=sample_rate, n_fft=n_fft, n_bands=octave_bands,
    )
    return {
        "librosa_tempo_bpm": tempo,
        "spectral_flatness": float(flatness.mean()),
        "spectral_contrast_db": float(contrast.mean()),
    }
