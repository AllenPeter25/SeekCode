import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def find_similar_tracks(
    vectors: pd.DataFrame,
    track_id: str,
    k: int = 10,
) -> pd.DataFrame:
    if k < 1:
        raise ValueError("k must be greater than zero")
    if "track_id" not in vectors.columns:
        raise ValueError("Vector table must contain track_id")
    if vectors["track_id"].isna().any() or not vectors["track_id"].is_unique:
        raise ValueError("Track IDs must be non-missing and unique")

    matches = np.flatnonzero(vectors["track_id"].to_numpy() == track_id)
    if matches.size == 0:
        raise ValueError(f"Track not found: {track_id}")

    values = vectors.drop(columns="track_id").to_numpy(dtype=float)
    if values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("Vectors must contain finite numeric features")

    query_index = matches[0]
    lengths = np.linalg.norm(values, axis=1)
    if lengths[query_index] == 0:
        raise ValueError("Query track has a zero vector and cannot be compared")

    scores = cosine_similarity(values[query_index : query_index + 1], values)[0]
    results = pd.DataFrame({
        "track_id": vectors["track_id"].to_numpy(),
        "similarity": np.clip(scores, -1.0, 1.0),
    })
    candidates = (results["track_id"] != track_id) & (lengths > 0)

    return (
        results.loc[candidates]
        .sort_values("similarity", ascending=False, kind="stable")
        .head(k)
        .reset_index(drop=True)
    )


def add_track_details(results: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"track_id", "genre", "path"}
    if not required_columns.issubset(manifest.columns):
        raise ValueError("Manifest must contain track_id, genre, and path")
    if manifest["track_id"].isna().any() or not manifest["track_id"].is_unique:
        raise ValueError("Manifest track IDs must be non-missing and unique")
    if not results["track_id"].isin(manifest["track_id"]).all():
        raise ValueError("Some search results are missing from the manifest")

    return results.merge(
        manifest[["track_id", "genre", "path"]],
        on="track_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )


def search_audio_file(
    audio_path,
    vectors: pd.DataFrame,
    models: dict,
    k: int = 10,
    exclude_track_id: str | None = None,
) -> pd.DataFrame:
    from .features.extract import extract_track_features

    if "extraction_settings" not in models:
        raise ValueError("Models have no extraction settings; rebuild with build-space or run")
    record = extract_track_features(audio_path, **models["extraction_settings"])
    return search_feature_record(record, vectors, models, k, exclude_track_id)


def search_feature_record(
    record: dict,
    vectors: pd.DataFrame,
    models: dict,
    k: int = 10,
    exclude_track_id: str | None = None,
) -> pd.DataFrame:
    from .space import transform_features

    query = transform_features(pd.DataFrame([record]), models)
    if list(vectors.drop(columns="track_id").columns) != models["vector_columns"]:
        raise ValueError("Vectors and models have different columns; use artifacts from the same run")
    query_id = "__external_query__"
    while query_id in set(vectors["track_id"]):
        query_id += "_"
    query["track_id"] = query_id
    candidates = vectors.loc[vectors["track_id"] != exclude_track_id]
    combined = pd.concat([candidates, query], ignore_index=True)
    return find_similar_tracks(combined, query_id, k)
