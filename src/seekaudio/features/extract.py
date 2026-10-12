from pathlib import Path

import pandas as pd

from .audio import load_audio
from .energy import extract_energy_features
from .mfcc import extract_mfcc_features
from .spectral import extract_spectral_features
from .rhythm import extract_rhythm_features
from .tonal import extract_tonal_features


def extract_track_features(
    audio_path: str | Path,
    sample_rate: int = 22_050,
    duration: float = 30.0,
    n_mfcc: int = 20,
    n_fft: int = 2_048,
    hop_length: int = 512,
    n_mels: int = 128,
    include_rhythm: bool = True,
    include_tonal: bool = False,
) -> dict[str, str | float]:
    audio_path = Path(audio_path)
    waveform = load_audio(
        audio_path,
        sample_rate=sample_rate,
        duration=duration,
        mono=True,
    )
    mfcc_features = extract_mfcc_features(
        waveform,
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    spectral_features = extract_spectral_features(
        waveform,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    energy_features = extract_energy_features(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    rhythm_features = extract_rhythm_features(waveform, sample_rate, n_fft, hop_length) if include_rhythm else {}
    tonal_features = extract_tonal_features(waveform, sample_rate, n_fft, hop_length) if include_tonal else {}

    return {
        "track_id": audio_path.stem,
        **mfcc_features,
        **spectral_features,
        **energy_features,
        **rhythm_features,
        **tonal_features,
    }


def extract_feature_table(
    manifest: pd.DataFrame,
    sample_rate: int = 22_050,
    duration: float = 30.0,
    n_mfcc: int = 20,
    n_fft: int = 2_048,
    hop_length: int = 512,
    n_mels: int = 128,
    include_rhythm: bool = True,
    include_tonal: bool = False,
    progress: bool = False,
) -> pd.DataFrame:
    required_columns = {"track_id", "path"}
    missing_columns = required_columns - set(manifest.columns)

    if missing_columns:
        missing_names = ", ".join(sorted(missing_columns))
        raise ValueError(f"Manifest is missing columns: {missing_names}")

    records = []

    for index, row in enumerate(manifest.itertuples(index=False), start=1):
        features = extract_track_features(
            row.path,
            sample_rate=sample_rate,
            duration=duration,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            include_rhythm=include_rhythm,
            include_tonal=include_tonal,
        )
        features["track_id"] = row.track_id
        records.append(features)
        if progress and (index % 100 == 0 or index == len(manifest)):
            print(f"Extracted {index}/{len(manifest)} tracks", flush=True)

    return pd.DataFrame(records)


def save_feature_table(features: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)

    return output_path
