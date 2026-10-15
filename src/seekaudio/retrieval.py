"""Rank acoustic vectors; genre labels do not participate in the ranking.

Dataset query -> find_similar_tracks. External file -> feature extraction
-> saved transform -> temporary query row -> the same ranking function.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def find_similar_tracks(
    vectors: pd.DataFrame,
    track_id: str,
    k: int = 10,
) -> pd.DataFrame:
    """Return up to k closest nonzero catalog vectors, excluding the query ID.

    Output columns are track_id and similarity, sorted highest first. Cosine
    ranges from -1 to 1 and is not a probability or a percent musical match.
    Stable sorting resolves equal scores using catalog row order. Unknown IDs,
    duplicate IDs, non-finite values, and zero query vectors are rejected.
    """
    if k < 1:
        raise ValueError("k must be greater than zero")
    if "track_id" not in vectors.columns:
        raise ValueError("Vector table must contain track_id")
    if vectors["track_id"].isna().any() or not vectors["track_id"].is_unique:
        raise ValueError("Track IDs must be non-missing and unique")

    # Use positional indices rather than pandas row labels to address the NumPy matrix.
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

    # Keeping the query slice 2-D satisfies sklearn's (rows, dimensions) interface.
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
    """Join genre/path by ID while preserving ranking and preventing duplicate joins.

    Details are added after retrieval for display or genre voting. Manifest order
    need not match vector order. Every result must have exactly one manifest row.
    """
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
    """Extract an external file using the saved recipe and rank catalog matches.

    exclude_track_id optionally removes the source track when a dataset file is
    used as the upload/query. Missing extraction metadata requires rebuilding
    saved models; guessing feature settings could invalidate comparisons.
    """
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
    """Transform an already extracted record and search without decoding it again.

    The query gets an unused temporary ID so it cannot replace a catalog row.
    Appending it allows reuse of find_similar_tracks, which excludes that query
    row. The stored vector DataFrame is not modified.
    """
    from .space import transform_features

    query = transform_features(pd.DataFrame([record]), models)
    if list(vectors.drop(columns="track_id").columns) != models["vector_columns"]:
        # Matching names catches different schemas; always pair files from the same run.
        raise ValueError("Vectors and models have different columns; use artifacts from the same run")
    query_id = "__external_query__"
    while query_id in set(vectors["track_id"]):
        query_id += "_"
    query["track_id"] = query_id
    candidates = vectors.loc[vectors["track_id"] != exclude_track_id]
    combined = pd.concat([candidates, query], ignore_index=True)
    return find_similar_tracks(combined, query_id, k)
