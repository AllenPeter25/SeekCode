from pathlib import Path
import wave

import joblib
import pandas as pd
import numpy as np

from src.seekaudio.__main__ import main
from src.seekaudio.space import apply_group_weights, normalize_feature_groups, normalize_vectors


def test_search_command(tmp_path: Path, capsys) -> None:
    vectors_path = tmp_path / "vectors.parquet"
    manifest_path = tmp_path / "manifest.parquet"
    pd.DataFrame({
        "track_id": ["query", "similar", "different"],
        "pca_01": [1.0, 1.0, 0.0],
        "pca_02": [0.0, 0.0, 1.0],
    }).to_parquet(vectors_path, index=False)

    pd.DataFrame({
        "track_id": ["different", "similar", "query"],
        "genre": ["jazz", "blues", "blues"],
        "path": ["different.wav", "similar.wav", "query.wav"],
    }).to_parquet(manifest_path, index=False)

    main([
        "search", "query", "--vectors", str(vectors_path),
        "--manifest", str(manifest_path), "--k", "1",
    ])

    output = capsys.readouterr().out
    assert "similar" in output
    assert "different" not in output
    assert "1.0000" in output
    assert "blues" in output
    assert "similar.wav" in output


def test_ingest_command(tmp_path: Path, capsys) -> None:
    genre_dir = tmp_path / "audio" / "classical"
    genre_dir.mkdir(parents=True)
    audio_path = genre_dir / "classical.00000.wav"
    output_path = tmp_path / "manifest.parquet"

    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(8_000)
        audio_file.writeframes(b"\x00\x00" * 8_000)

    main(
        [
            "ingest",
            "--audio-dir",
            str(tmp_path / "audio"),
            "--output",
            str(output_path),
        ]
    )

    manifest = pd.read_parquet(output_path)
    output = capsys.readouterr().out

    assert len(manifest) == 1
    assert "Saved 1 tracks" in output


def test_extract_command(tmp_path: Path, capsys) -> None:
    audio_path = tmp_path / "tone.wav"
    manifest_path = tmp_path / "manifest.parquet"
    output_path = tmp_path / "features" / "features_v1.parquet"

    with wave.open(str(audio_path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(22_050)
        audio_file.writeframes(b"\x01\x00" * 22_050)

    pd.DataFrame(
        [{"track_id": "custom-id", "path": str(audio_path)}]
    ).to_parquet(manifest_path, index=False)

    main(["extract", "--manifest", str(manifest_path), "--output", str(output_path)])

    features = pd.read_parquet(output_path)

    assert features.shape == (1, 54)
    assert features["track_id"].tolist() == ["custom-id"]
    assert np.isfinite(features.drop(columns="track_id").to_numpy()).all()
    assert "with 53 features" in capsys.readouterr().out


def test_build_space_command_saves_reusable_models(tmp_path: Path, capsys) -> None:
    features_path = tmp_path / "features.parquet"
    output_dir = tmp_path / "space"
    values = np.random.default_rng(42).normal(size=(10, 5))
    features = pd.DataFrame(
        values,
        columns=["mfcc_01_mean", "mfcc_02_mean", "spectral_centroid_mean", "energy_rms_mean", "rhythm_tempo_bpm"],
    )
    features.insert(0, "track_id", [f"track_{index}" for index in range(10)])
    features.to_parquet(features_path, index=False)

    main([
        "build-space", "--features", str(features_path),
        "--output-dir", str(output_dir), "--components", "2",
    ])

    vectors = pd.read_parquet(output_dir / "vectors.parquet")
    models = joblib.load(output_dir / "models.joblib")

    numeric = features[models["feature_columns"]]
    scaled = pd.DataFrame(models["scaler"].transform(numeric), columns=numeric.columns)
    scaled.insert(0, "track_id", features["track_id"])
    grouped = normalize_feature_groups(scaled)
    weighted = apply_group_weights(grouped, models["weights"])
    reduced = pd.DataFrame(
        models["pca"].transform(weighted.drop(columns="track_id")),
        columns=models["vector_columns"],
    )
    reduced.insert(0, "track_id", features["track_id"])
    reproduced = normalize_vectors(reduced)

    assert vectors["track_id"].equals(features["track_id"])
    assert np.allclose(vectors.drop(columns="track_id"), reproduced.drop(columns="track_id"))
    assert "with 2 dimensions" in capsys.readouterr().out
