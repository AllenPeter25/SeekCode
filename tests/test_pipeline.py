import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from src.seekaudio.config import load_config
from src.seekaudio.features.extract import extract_feature_table
from src.seekaudio.pipeline import build_configured_space, run_pipeline
from src.seekaudio.retrieval import find_similar_tracks, search_audio_file
from src.seekaudio.space import select_features, transform_features


def make_dataset(tmp_path: Path) -> pd.DataFrame:
    random = np.random.default_rng(7)
    rows = []
    for index in range(6):
        genre = "blues" if index < 3 else "jazz"
        path = tmp_path / "audio" / genre / f"track_{index}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        waveform = (random.normal(size=8_000) * (0.1 + index * 0.02)).astype(np.float32)
        sf.write(path, waveform, 8_000)
        rows.append({"track_id": path.stem, "genre": genre, "path": str(path)})
    return pd.DataFrame(rows)


def test_saved_query_transform_matches_dataset_search(tmp_path: Path) -> None:
    manifest = make_dataset(tmp_path)
    settings = {"sample_rate": 8_000, "duration": 1.0, "n_mfcc": 3,
                "n_fft": 256, "hop_length": 80, "n_mels": 16}
    features = extract_feature_table(manifest, **settings)
    config = load_config(overrides={"space": {"pca_components": 2}})
    vectors, models = build_configured_space(features, config)
    models["extraction_settings"] = settings
    model_path = tmp_path / "models.joblib"
    joblib.dump(models, model_path)
    loaded = joblib.load(model_path)

    transformed = transform_features(features.iloc[:1], loaded)
    assert np.allclose(transformed.drop(columns="track_id"), vectors.iloc[:1].drop(columns="track_id"))
    expected = find_similar_tracks(vectors, "track_0", k=3)
    actual = search_audio_file(manifest.iloc[0]["path"], vectors, loaded, k=3, exclude_track_id="track_0")
    assert actual.track_id.tolist() == expected.track_id.tolist()
    assert np.allclose(actual.similarity, expected.similarity)


def test_complete_pipeline_creates_reports_and_plots(tmp_path: Path) -> None:
    make_dataset(tmp_path)
    config_path = tmp_path / "config.yaml"
    ablation_path = tmp_path / "ablation.yaml"
    config = load_config(overrides={
        "audio": {"sample_rate": 8_000, "duration": 1},
        "features": {"n_mfcc": 3, "n_fft": 256, "hop_length": 80, "n_mels": 16},
        "space": {"pca_components": 2},
        "retrieval": {"k": 1},
        "paths": {"audio_dir": str(tmp_path / "audio"), "manifest": str(tmp_path / "manifest.parquet"),
                  "features_dir": str(tmp_path / "features"), "space_dir": str(tmp_path / "space"),
                  "evaluation_dir": str(tmp_path / "evaluation")},
    })
    config_path.write_text(yaml.safe_dump(asdict(config)), encoding="utf-8")
    ablation_path.write_text(yaml.safe_dump({"configs": {
        "mean_only": {"pooling": {"use_std": False}, "space": {"pca_components": None}},
    }}), encoding="utf-8")

    results = run_pipeline(config, config_path, ablation_path)

    assert len(results) == 6
    summary = json.loads((tmp_path / "evaluation/current/summary.json").read_text())
    assert summary["track_count"] == 6
    assert (tmp_path / "features/features_v2.json").exists()
    assert (tmp_path / "space/models.joblib").exists()
    assert (tmp_path / "evaluation/plots/ablations.png").stat().st_size > 0
    assert (tmp_path / "evaluation/ablations/comparison.csv").exists()


def test_mean_only_feature_selection_excludes_standard_deviation() -> None:
    features = pd.DataFrame({"track_id": ["a"], "mfcc_01_mean": [1.0], "mfcc_01_std": [2.0]})
    selected = select_features(features, {"mfcc": True}, use_std=False)
    assert selected.columns.tolist() == ["track_id", "mfcc_01_mean"]
