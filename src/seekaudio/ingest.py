from pathlib import Path
import warnings

import pandas as pd
import soundfile as sf


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}
MANIFEST_COLUMNS = [
    "track_id",
    "path",
    "genre",
    "duration",
    "sample_rate",
    "channels",
]


def find_audio_files(audio_dir: str | Path) -> list[Path]:
    audio_dir = Path(audio_dir)

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory does not exist: {audio_dir}")
    if not audio_dir.is_dir():
        raise NotADirectoryError(f"Audio path is not a directory: {audio_dir}")

    audio_files = [
        path
        for path in audio_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]

    return sorted(audio_files)


def read_audio_metadata(audio_path: str | Path) -> dict[str, int | float]:
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    if not audio_path.is_file():
        raise IsADirectoryError(f"Audio path is not a file: {audio_path}")

    file_info = sf.info(audio_path)

    return {
        "duration": float(file_info.duration),
        "sample_rate": int(file_info.samplerate),
        "channels": int(file_info.channels),
    }


def create_manifest_record(audio_path: str | Path) -> dict[str, str | int | float]:
    audio_path = Path(audio_path)
    metadata = read_audio_metadata(audio_path)

    return {
        "track_id": audio_path.stem,
        "path": str(audio_path),
        "genre": audio_path.parent.name,
        **metadata,
    }


def build_manifest(audio_dir: str | Path) -> pd.DataFrame:
    audio_files = find_audio_files(audio_dir)
    records = []

    for path in audio_files:
        try:
            records.append(create_manifest_record(path))
        except sf.LibsndfileError:
            warnings.warn(f"Skipping unreadable audio file: {path}", stacklevel=2)

    return pd.DataFrame(records, columns=MANIFEST_COLUMNS)


def save_manifest(manifest: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output_path, index=False)

    return output_path


def ingest_dataset(
    audio_dir: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    manifest = build_manifest(audio_dir)
    save_manifest(manifest, output_path)

    return manifest
