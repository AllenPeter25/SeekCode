"""Inventory audio files before any musical features are extracted.

Folder -> file paths -> technical metadata records -> pandas manifest -> Parquet.
Genre comes from the parent folder; it is used for display and evaluation only.
"""

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
    """Return sorted WAV/MP3/FLAC paths from a directory and all its subfolders.

    A missing folder raises FileNotFoundError; a file used as a folder raises
    NotADirectoryError. This discovers files, but does not prove they are readable.
    """
    audio_dir = Path(audio_dir)

    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory does not exist: {audio_dir}")
    if not audio_dir.is_dir():
        raise NotADirectoryError(f"Audio path is not a directory: {audio_dir}")

    # rglob walks genre folders; lowercase suffixes also accept files like SONG.WAV.
    audio_files = [
        path
        for path in audio_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]

    # Repeatable ordering helps keep manifests and feature tables easy to compare.
    return sorted(audio_files)


def read_audio_metadata(audio_path: str | Path) -> dict[str, int | float]:
    """Read duration (seconds), sample rate (Hz), and channel count from a header.

    No complete waveform is loaded. Decoder errors are allowed to reach the
    caller, which can decide whether to skip an unreadable recording.
    """
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
    """Combine one file's ID, stored path, folder genre, and technical metadata.

    Path.stem removes the extension. IDs must later be unique for search-space
    building; the assumed dataset layout is genre/track_name.wav.
    """
    audio_path = Path(audio_path)
    metadata = read_audio_metadata(audio_path)

    return {
        "track_id": audio_path.stem,
        "path": str(audio_path),
        "genre": audio_path.parent.name,
        **metadata,  # Unpack duration, sample_rate, and channels into this record.
    }


def build_manifest(audio_dir: str | Path) -> pd.DataFrame:
    """Create one manifest row per readable file, warning about decoder failures."""
    audio_files = find_audio_files(audio_dir)
    records = []

    for path in audio_files:
        try:
            records.append(create_manifest_record(path))
        except sf.LibsndfileError:
            # A malformed file must not prevent the rest of the catalog being read.
            warnings.warn(f"Skipping unreadable audio file: {path}", stacklevel=2)

    return pd.DataFrame(records, columns=MANIFEST_COLUMNS)


def save_manifest(manifest: pd.DataFrame, output_path: str | Path) -> Path:
    """Create the output folder and save the table without pandas' row index."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output_path, index=False)

    return output_path


def ingest_dataset(
    audio_dir: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """Build and save the dataset manifest, returning the same table to callers."""
    manifest = build_manifest(audio_dir)
    save_manifest(manifest, output_path)

    return manifest
