from pathlib import Path
import wave

import pandas as pd
import pytest

from src.seekaudio.ingest import (
    build_manifest,
    create_manifest_record,
    find_audio_files,
    ingest_dataset,
    read_audio_metadata,
    save_manifest,
)


def test_find_audio_files_searches_subdirectories(tmp_path: Path) -> None:
    genre_dir = tmp_path / "blues"
    genre_dir.mkdir()

    (genre_dir / "track.wav").touch()
    (genre_dir / "notes.txt").touch()
    (tmp_path / "second.FLAC").touch()

    result = find_audio_files(tmp_path)

    assert result == [genre_dir / "track.wav", tmp_path / "second.FLAC"]


def test_find_audio_files_rejects_missing_directory(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Audio directory does not exist"):
        find_audio_files(missing_dir)


def test_read_audio_metadata(tmp_path: Path) -> None:
    audio_path = tmp_path / "example.wav"

    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(8_000)
        audio_file.writeframes(b"\x00\x00" * 8_000)

    metadata = read_audio_metadata(audio_path)

    assert metadata == {
        "duration": 1.0,
        "sample_rate": 8_000,
        "channels": 1,
    }


def test_create_manifest_record(tmp_path: Path) -> None:
    genre_dir = tmp_path / "jazz"
    genre_dir.mkdir()
    audio_path = genre_dir / "jazz.00001.wav"

    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(2)
        audio_file.setsampwidth(2)
        audio_file.setframerate(16_000)
        audio_file.writeframes(b"\x00\x00" * 2 * 16_000)

    record = create_manifest_record(audio_path)

    assert record == {
        "track_id": "jazz.00001",
        "path": str(audio_path),
        "genre": "jazz",
        "duration": 1.0,
        "sample_rate": 16_000,
        "channels": 2,
    }


def test_build_manifest(tmp_path: Path) -> None:
    for genre, sample_rate in [("blues", 8_000), ("jazz", 16_000)]:
        genre_dir = tmp_path / genre
        genre_dir.mkdir()
        audio_path = genre_dir / f"{genre}.00000.wav"

        with wave.open(str(audio_path), "wb") as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2)
            audio_file.setframerate(sample_rate)
            audio_file.writeframes(b"\x00\x00" * sample_rate)

    manifest = build_manifest(tmp_path)

    assert manifest.shape == (2, 6)
    assert manifest["track_id"].tolist() == ["blues.00000", "jazz.00000"]
    assert manifest["genre"].tolist() == ["blues", "jazz"]
    assert manifest["duration"].tolist() == [1.0, 1.0]


def test_build_manifest_skips_unreadable_audio(tmp_path: Path) -> None:
    broken_audio = tmp_path / "broken.wav"
    broken_audio.touch()

    with pytest.warns(UserWarning, match="Skipping unreadable audio file"):
        manifest = build_manifest(tmp_path)

    assert manifest.empty
    assert manifest.columns.tolist() == [
        "track_id",
        "path",
        "genre",
        "duration",
        "sample_rate",
        "channels",
    ]


def test_save_manifest(tmp_path: Path) -> None:
    manifest = pd.DataFrame(
        [
            {
                "track_id": "blues.00000",
                "path": "blues/blues.00000.wav",
                "genre": "blues",
                "duration": 30.0,
                "sample_rate": 22_050,
                "channels": 1,
            }
        ]
    )
    output_path = tmp_path / "nested" / "manifest.parquet"

    saved_path = save_manifest(manifest, output_path)
    loaded_manifest = pd.read_parquet(saved_path)

    assert saved_path == output_path
    assert loaded_manifest.equals(manifest)


def test_ingest_dataset(tmp_path: Path) -> None:
    genre_dir = tmp_path / "audio" / "rock"
    genre_dir.mkdir(parents=True)
    audio_path = genre_dir / "rock.00000.wav"
    output_path = tmp_path / "output" / "manifest.parquet"

    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(8_000)
        audio_file.writeframes(b"\x00\x00" * 8_000)

    manifest = ingest_dataset(tmp_path / "audio", output_path)
    saved_manifest = pd.read_parquet(output_path)

    assert manifest.shape == (1, 6)
    assert manifest.iloc[0]["track_id"] == "rock.00000"
    assert saved_manifest.equals(manifest)
