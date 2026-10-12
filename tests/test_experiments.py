from pathlib import Path

import numpy as np
import pandas as pd

from src.seekaudio.experiments import compare_feature_groups


def test_compare_feature_groups_uses_same_tracks_and_saves_reports(tmp_path: Path) -> None:
    features = pd.DataFrame(
        np.random.default_rng(42).normal(size=(12, 6)),
        columns=["mfcc_01_mean", "mfcc_02_mean", "spectral_centroid_mean",
                 "spectral_bandwidth_mean", "energy_rms_mean", "energy_rms_std"],
    )
    features.insert(0, "track_id", [f"track_{index}" for index in range(12)])
    manifest = pd.DataFrame({"track_id": features["track_id"], "genre": ["a", "b"] * 6})

    comparison = compare_feature_groups(features, manifest, tmp_path, k=2, pca_components=3)

    assert len(comparison) == 6
    assert comparison["track_count"].tolist() == [12] * 6
    assert comparison["k"].tolist() == [2] * 6
    assert comparison["precision_at_k"].between(0, 1).all()
    assert comparison["random_precision_at_k"].nunique() == 1
    saved = pd.read_csv(tmp_path / "comparison.csv").set_index("experiment")
    assert saved.loc["all_no_pca", "dimensions"] == 6
    assert saved.loc["all_pca", "dimensions"] == 3
    assert (tmp_path / "mfcc_only" / "per_track.csv").exists()
