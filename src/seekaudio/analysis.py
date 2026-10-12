import numpy as np
import pandas as pd

from .features.audio import load_audio
from .features.extract import extract_track_features
from .ingest import read_audio_metadata
from .retrieval import add_track_details, search_feature_record
from .viz import render_mel_spectrogram


def estimate_genre(matches: pd.DataFrame) -> str:
    if matches.empty or matches["genre"].isna().any():
        raise ValueError("Genre estimation requires labeled catalog matches")
    votes = matches.groupby("genre").agg(
        count=("genre", "size"),
        similarity=("similarity", "sum"),
    )
    votes = votes.sort_values(["count", "similarity"], ascending=False, kind="stable")
    return str(votes.index[0])


def analyze_audio(audio_path, vectors: pd.DataFrame, models: dict, manifest: pd.DataFrame) -> dict:
    if "extraction_settings" not in models:
        raise ValueError("Rebuild the saved models using the run command before analyzing audio")
    metadata = read_audio_metadata(audio_path)
    features = extract_track_features(audio_path, **models["extraction_settings"])
    results = search_feature_record(features, vectors, models, k=10)
    matches = add_track_details(results, manifest)
    settings = models["extraction_settings"]
    waveform = load_audio(audio_path, sample_rate=settings["sample_rate"], duration=settings["duration"])
    peak = float(np.max(np.abs(waveform)))
    return {
        "genre": estimate_genre(matches),
        "duration": metadata["duration"],
        "sample_rate": metadata["sample_rate"],
        "channels": metadata["channels"],
        "tempo_bpm": features.get("rhythm_tempo_bpm"),
        "spectral_centroid_hz": features["spectral_centroid_mean"],
        "rms_energy": features["energy_rms_mean"],
        "peak_dbfs": float(20 * np.log10(peak)) if peak > 0 else None,
        "energy_variation": features["energy_rms_std"],
        "spectral_bandwidth_hz": features["spectral_bandwidth_mean"],
        "spectral_rolloff_hz": features["spectral_rolloff_mean"],
        "zero_crossing_rate": features["spectral_zero_crossing_mean"],
        "onset_strength": features.get("rhythm_onset_mean"),
        "analysis_duration": models["extraction_settings"]["duration"],
        "mel_spectrogram_png": render_mel_spectrogram(
            waveform, settings["sample_rate"], settings["n_fft"],
            settings["hop_length"], settings["n_mels"],
        ),
    }
