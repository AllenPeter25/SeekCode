import json
from pathlib import Path

import numpy as np
import pandas as pd

from .retrieval import find_similar_tracks


def evaluate_precision_at_k(
    vectors: pd.DataFrame,
    manifest: pd.DataFrame,
    k: int = 10,
) -> pd.DataFrame:
    if "track_id" not in vectors.columns:
        raise ValueError("Vector table must contain track_id")
    if vectors["track_id"].isna().any() or not vectors["track_id"].is_unique:
        raise ValueError("Vector track IDs must be non-missing and unique")
    if not {"track_id", "genre"}.issubset(manifest.columns):
        raise ValueError("Manifest must contain track_id and genre")
    if manifest["track_id"].isna().any() or not manifest["track_id"].is_unique:
        raise ValueError("Manifest track IDs must be non-missing and unique")
    if not 1 <= k < len(vectors):
        raise ValueError("k must be between 1 and the number of tracks minus one")
    if not vectors["track_id"].isin(manifest["track_id"]).all():
        raise ValueError("Some vector track IDs are missing from the manifest")

    genres = manifest.set_index("track_id")["genre"]
    if genres.loc[vectors["track_id"]].isna().any():
        raise ValueError("Every evaluated track must have a genre")

    values = vectors.drop(columns="track_id").to_numpy(dtype=float)
    if np.any(np.linalg.norm(values, axis=1) == 0):
        raise ValueError("Evaluation requires nonzero vectors for every track")

    records = []
    candidate_genres = genres.loc[vectors["track_id"]]
    genre_counts = candidate_genres.value_counts()

    for track_id in vectors["track_id"]:
        results = find_similar_tracks(vectors, track_id, k)
        query_genre = genres.loc[track_id]
        matching_genres = genres.loc[results["track_id"]] == query_genre
        relevance = matching_genres.to_numpy(dtype=float)
        total_relevant = int(genre_counts[query_genre] - 1)
        ranks = np.arange(1, k + 1)
        precision_by_rank = np.cumsum(relevance) / ranks
        ideal_count = min(total_relevant, k)
        discounts = 1 / np.log2(ranks + 1)
        ideal_dcg = discounts[:ideal_count].sum()
        records.append({
            "track_id": track_id,
            "genre": query_genre,
            "precision_at_k": float(matching_genres.sum() / k),
            "recall_at_k": float(relevance.sum() / total_relevant) if total_relevant else 0.0,
            "average_precision_at_k": float((precision_by_rank * relevance).sum() / ideal_count) if ideal_count else 0.0,
            "ndcg_at_k": float((relevance * discounts).sum() / ideal_dcg) if ideal_dcg else 0.0,
        })

    return pd.DataFrame(records)


def add_random_baseline(results: pd.DataFrame) -> pd.DataFrame:
    if len(results) < 2:
        raise ValueError("Random baseline requires at least two tracks")
    if results["genre"].isna().any():
        raise ValueError("Every track must have a genre")

    comparison = results.copy()
    genre_counts = results.groupby("genre")["track_id"].transform("count")
    comparison["random_precision_at_k"] = (genre_counts - 1) / (len(results) - 1)

    return comparison


def save_evaluation(
    results: pd.DataFrame,
    k: int,
    output_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_track = results.copy()
    per_track["k"] = k
    per_track.to_csv(output_dir / "per_track.csv", index=False)

    by_genre = results.groupby("genre")["precision_at_k"].agg(
        track_count="count",
        mean_precision_at_k="mean",
    ).reset_index()
    by_genre["k"] = k
    for metric in ["recall_at_k", "average_precision_at_k", "ndcg_at_k"]:
        if metric in results.columns:
            by_genre[f"mean_{metric}"] = by_genre.genre.map(results.groupby("genre")[metric].mean())
    if "random_precision_at_k" in results.columns:
        random_by_genre = results.groupby("genre")["random_precision_at_k"].mean()
        by_genre["random_precision_at_k"] = by_genre["genre"].map(random_by_genre)
    by_genre.to_csv(output_dir / "by_genre.csv", index=False)

    summary = {
        "metric": "precision_at_k",
        "k": k,
        "track_count": len(results),
        "mean_precision_at_k": float(results["precision_at_k"].mean()),
        "relevance": "same_genre",
        "query_excluded": True,
        "evaluation_scope": "catalog_evaluation_with_catalog_fitted_transforms",
    }
    for metric in ["recall_at_k", "average_precision_at_k", "ndcg_at_k"]:
        if metric in results.columns:
            summary[f"mean_{metric}"] = float(results[metric].mean())
    if "random_precision_at_k" in results.columns:
        random_score = float(results["random_precision_at_k"].mean())
        summary["random_baseline_method"] = "expected_uniform_retrieval_without_replacement"
        summary["random_precision_at_k"] = random_score
        summary["lift_over_random"] = (
            summary["mean_precision_at_k"] / random_score if random_score > 0 else None
        )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, allow_nan=False)

    return output_dir
