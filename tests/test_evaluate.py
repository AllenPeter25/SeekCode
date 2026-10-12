import json
from pathlib import Path

import pandas as pd
import pytest

from src.seekaudio.evaluate import add_random_baseline, evaluate_precision_at_k, save_evaluation


def test_precision_at_k_excludes_self_and_matches_genres_by_id() -> None:
    vectors = pd.DataFrame({
        "track_id": ["a", "b", "c", "d"],
        "pca_01": [1.0, 0.9, 0.0, 0.1],
        "pca_02": [0.0, 0.1, 1.0, 0.9],
    })
    manifest = pd.DataFrame({
        "track_id": ["d", "c", "b", "a"],
        "genre": ["jazz", "jazz", "blues", "blues"],
    })

    top_one = evaluate_precision_at_k(vectors, manifest, k=1)
    top_three = evaluate_precision_at_k(vectors, manifest, k=3)

    assert top_one["precision_at_k"].tolist() == [1.0] * 4
    assert top_three["precision_at_k"].tolist() == pytest.approx([1 / 3] * 4)
    assert top_one["genre"].tolist() == ["blues", "blues", "jazz", "jazz"]
    assert top_three["recall_at_k"].tolist() == pytest.approx([1.0] * 4)
    assert top_three["average_precision_at_k"].tolist() == pytest.approx([1.0] * 4)
    assert top_three["ndcg_at_k"].tolist() == pytest.approx([1.0] * 4)


def test_precision_at_k_rejects_insufficient_candidates() -> None:
    vectors = pd.DataFrame({"track_id": ["a", "b"], "pca_01": [1.0, 2.0]})
    manifest = pd.DataFrame({"track_id": ["a", "b"], "genre": ["blues", "blues"]})

    with pytest.raises(ValueError, match="k must be"):
        evaluate_precision_at_k(vectors, manifest, k=2)


def test_rank_metrics_penalize_relevant_tracks_appearing_later() -> None:
    vectors = pd.DataFrame({
        "track_id": ["query", "wrong", "relevant", "other"],
        "pca_01": [1.0, 1.0, 1.0, -1.0],
        "pca_02": [0.0, 0.1, 0.2, 0.0],
    })
    manifest = pd.DataFrame({
        "track_id": vectors.track_id,
        "genre": ["a", "b", "a", "b"],
    })
    results = evaluate_precision_at_k(vectors, manifest, k=2).set_index("track_id")

    assert results.loc["query", "precision_at_k"] == pytest.approx(0.5)
    assert results.loc["query", "recall_at_k"] == pytest.approx(1.0)
    assert results.loc["query", "average_precision_at_k"] == pytest.approx(0.5)
    assert results.loc["query", "ndcg_at_k"] == pytest.approx(0.6309297536)

def test_save_evaluation_records_k_and_averages(tmp_path: Path) -> None:
    results = pd.DataFrame({
        "track_id": ["a", "b", "c"],
        "genre": ["blues", "blues", "jazz"],
        "precision_at_k": [1.0, 0.5, 0.0],
    })
    output_dir = save_evaluation(results, k=2, output_dir=tmp_path / "evaluation")

    per_track = pd.read_csv(output_dir / "per_track.csv")
    by_genre = pd.read_csv(output_dir / "by_genre.csv").set_index("genre")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))

    assert per_track["track_id"].tolist() == ["a", "b", "c"]
    assert per_track["k"].tolist() == [2, 2, 2]
    assert by_genre.loc["blues", "mean_precision_at_k"] == pytest.approx(0.75)
    assert by_genre.loc["blues", "track_count"] == 2
    assert summary["mean_precision_at_k"] == pytest.approx(0.5)
    assert summary["track_count"] == 3
    assert summary["k"] == 2


def test_random_baseline_accounts_for_genre_imbalance_and_excludes_self(tmp_path: Path) -> None:
    results = pd.DataFrame({
        "track_id": ["a", "b", "c", "d"],
        "genre": ["blues", "blues", "blues", "jazz"],
        "precision_at_k": [1.0, 1.0, 1.0, 0.0],
    })

    comparison = add_random_baseline(results)
    output_dir = save_evaluation(comparison, k=1, output_dir=tmp_path)
    summary = json.loads((output_dir / "summary.json").read_text())

    assert comparison["random_precision_at_k"].tolist() == pytest.approx([2 / 3] * 3 + [0])
    assert summary["random_precision_at_k"] == pytest.approx(0.5)
    assert summary["lift_over_random"] == pytest.approx(1.5)
    assert "random_precision_at_k" not in results.columns


def test_random_baseline_with_no_same_genre_candidates(tmp_path: Path) -> None:
    results = pd.DataFrame({
        "track_id": ["a", "b"],
        "genre": ["blues", "jazz"],
        "precision_at_k": [0.0, 0.0],
    })
    comparison = add_random_baseline(results)
    save_evaluation(comparison, k=1, output_dir=tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text())

    assert summary["random_precision_at_k"] == 0
    assert summary["lift_over_random"] is None
