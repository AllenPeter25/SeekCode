"""Live upload/sample analysis, separate from offline catalog preparation.

File -> metadata + one prepared waveform -> retrieval features -> saved transforms
-> nearest catalog labels -> genre vote. Reuse that waveform for display-only
Librosa measurements, peak level, and the Mel spectrogram.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .features.audio import load_audio
from .features.extract import extract_waveform_features
from .features.librosa_analysis import extract_librosa_characteristics
from .ingest import read_audio_metadata
from .retrieval import add_track_details, search_feature_record
from .viz import render_mel_spectrogram


def estimate_genre(matches: pd.DataFrame) -> str:
    """Choose the most frequent neighbor genre, breaking ties by summed similarity.

    Remaining exact ties follow the grouped genre order. This is a nearest-neighbor
    label estimate, not a trained classifier or calibrated confidence score.
    """
    if matches.empty or matches["genre"].isna().any():
        raise ValueError("Genre estimation requires labeled catalog matches")
    votes = matches.groupby("genre").agg(
        count=("genre", "size"),
        similarity=("similarity", "sum"),
    )
    votes = votes.sort_values(["count", "similarity"], ascending=False, kind="stable")
    return str(votes.index[0])


def analyze_audio(audio_path, vectors: pd.DataFrame, models: dict, manifest: pd.DataFrame) -> dict:
    """Return a genre estimate, original-file metadata, excerpt metrics, and PNG bytes.

    vectors/models/manifest are prebuilt catalog artifacts. The waveform is loaded
    once; feature extraction, display measurements, and plotting use the same
    mono/resampled/centered excerpt. File duration/rate/channels describe the
    original recording, not that prepared excerpt.
    """
    if "extraction_settings" not in models:
        raise ValueError("Rebuild the saved models using the run command before analyzing audio")
    audio_path = Path(audio_path)
    settings = models["extraction_settings"]
    metadata = read_audio_metadata(audio_path)
    waveform = load_audio(
        audio_path,
        sample_rate=settings["sample_rate"],
        duration=settings["duration"],
    )
    feature_settings = settings.copy()
    # Duration belongs to audio loading, not to the already-loaded waveform extractor.
    feature_settings.pop("duration")
    features = extract_waveform_features(waveform, track_id=audio_path.stem, **feature_settings)
    results = search_feature_record(features, vectors, models, k=10)
    matches = add_track_details(results, manifest)
    # This estimates sample peak in the prepared waveform; it is not true-peak dBTP.
    peak = float(np.max(np.abs(waveform)))
    librosa_features = extract_librosa_characteristics(
        # Display-only values must not replace the version 2 features used by saved models.
        waveform,
        sample_rate=settings["sample_rate"],
        n_fft=settings["n_fft"],
        hop_length=settings["hop_length"],
    )
    return {
        "genre": estimate_genre(matches),
        "duration": metadata["duration"],
        "sample_rate": metadata["sample_rate"],
        "channels": metadata["channels"],
        "tempo_bpm": librosa_features["librosa_tempo_bpm"],
        "spectral_flatness": librosa_features["spectral_flatness"],
        "spectral_contrast_db": librosa_features["spectral_contrast_db"],
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

