from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


FEATURE_GROUPS = ("mfcc", "spectral", "energy", "rhythm", "tonal")


def get_numeric_features(features: pd.DataFrame) -> pd.DataFrame:
    if "track_id" not in features.columns:
        raise ValueError("Feature table must contain track_id")
    numeric_features = features.drop(columns="track_id")
    if not np.isfinite(numeric_features.to_numpy(dtype=float)).all():
        raise ValueError("Features must contain only finite numbers")
    return numeric_features


def create_feature_table(values, columns, source: pd.DataFrame) -> pd.DataFrame:
    table = pd.DataFrame(values, columns=columns, index=source.index)
    table.insert(0, "track_id", source["track_id"])
    return table


def normalize_rows(values: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    lengths[lengths == 0] = 1
    return values / lengths


def get_group_columns(features: pd.DataFrame, group: str) -> list[str]:
    columns = []
    for column in features.columns:
        if column.startswith(f"{group}_"):
            columns.append(column)
    return columns


def standardize_features(
    features: pd.DataFrame,
) -> tuple[pd.DataFrame, StandardScaler]:
    numeric_features = get_numeric_features(features)
    if features.empty:
        raise ValueError("Feature table contains no tracks")

    if numeric_features.shape[1] == 0:
        raise ValueError("Feature table contains no numeric features")
    scaler = StandardScaler()
    values = scaler.fit_transform(numeric_features)
    standardized = create_feature_table(values, numeric_features.columns, features)

    return standardized, scaler


def normalize_feature_groups(features: pd.DataFrame) -> pd.DataFrame:
    numeric_features = get_numeric_features(features)
    group_prefixes = tuple(f"{group}_" for group in FEATURE_GROUPS)

    unknown_columns = [
        column
        for column in numeric_features.columns
        if not column.startswith(group_prefixes)
    ]
    if unknown_columns:
        raise ValueError(f"Unknown feature groups for columns: {unknown_columns}")

    normalized = features.copy()

    for group in FEATURE_GROUPS:
        columns = get_group_columns(numeric_features, group)
        if not columns:
            continue

        values = features[columns].to_numpy(dtype=float)
        normalized[columns] = normalize_rows(values)

    return normalized


def apply_group_weights(
    features: pd.DataFrame,
    weights: dict[str, float],
) -> pd.DataFrame:
    if "track_id" not in features.columns:
        raise ValueError("Feature table must contain track_id")

    for group, weight in weights.items():
        if group not in FEATURE_GROUPS:
            raise ValueError(f"Unknown feature group: {group}")
        if not np.isfinite(weight) or weight < 0:
            raise ValueError(f"Weight for {group} must be finite and non-negative")

    weighted = features.copy()

    for group, weight in weights.items():
        columns = get_group_columns(features, group)
        weighted[columns] = features[columns] * weight

    return weighted


def reduce_features_pca(
    features: pd.DataFrame,
    n_components: int = 40,
    whiten: bool = True,
) -> tuple[pd.DataFrame, PCA]:
    numeric_features = get_numeric_features(features)
    max_components = min(len(features) - 1, numeric_features.shape[1])
    if not 1 <= n_components <= max_components:
        raise ValueError(f"n_components must be between 1 and {max_components}")

    pca = PCA(n_components=n_components, whiten=whiten, svd_solver="full")
    pca.fit(numeric_features)

    if whiten:
        minimum_variance = np.finfo(float).eps * pca.explained_variance_.max()
        if np.any(pca.explained_variance_ <= minimum_variance):
            raise ValueError("Cannot whiten zero-variance components; use fewer components or whiten=False")

    values = pca.transform(numeric_features)
    columns = [f"pca_{index:02d}" for index in range(1, n_components + 1)]
    reduced = create_feature_table(values, columns, features)

    return reduced, pca


def normalize_vectors(features: pd.DataFrame) -> pd.DataFrame:
    numeric_features = get_numeric_features(features)
    if numeric_features.shape[1] == 0:
        raise ValueError("Feature table contains no numeric features")

    values = numeric_features.to_numpy(dtype=float)
    normalized_values = normalize_rows(values)
    normalized = create_feature_table(normalized_values, numeric_features.columns, features)

    return normalized


def build_feature_space(
    features: pd.DataFrame,
    n_components: int | None = 40,
    whiten: bool = True,
    weights: dict[str, float] | None = None,
    l2_normalize: bool = True,
) -> tuple[pd.DataFrame, dict]:
    if "track_id" not in features.columns:
        raise ValueError("Feature table must contain track_id")
    if features["track_id"].isna().any() or not features["track_id"].is_unique:
        raise ValueError("Track IDs must be non-missing and unique")

    if weights is None:
        weights = {}
    else:
        weights = weights.copy()
    standardized, scaler = standardize_features(features)
    grouped = normalize_feature_groups(standardized)
    weighted = apply_group_weights(grouped, weights)
    if n_components is None:
        reduced = weighted
        pca = None
    else:
        reduced, pca = reduce_features_pca(weighted, n_components, whiten)
    vectors = reduced
    if l2_normalize:
        vectors = normalize_vectors(reduced)
    models = {
        "scaler": scaler,
        "pca": pca,
        "weights": weights,
        "feature_columns": features.drop(columns="track_id").columns.tolist(),
        "vector_columns": vectors.drop(columns="track_id").columns.tolist(),
        "l2_normalize": l2_normalize,
    }

    return vectors, models


def save_feature_space(
    vectors: pd.DataFrame,
    models: dict,
    output_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vectors.to_parquet(output_dir / "vectors.parquet", index=False)
    joblib.dump(models, output_dir / "models.joblib")

    return output_dir


def select_features(
    features: pd.DataFrame,
    groups: dict[str, bool],
    use_std: bool = True,
) -> pd.DataFrame:
    columns = ["track_id"]
    for group, enabled in groups.items():
        if not enabled:
            continue
        group_columns = get_group_columns(features, group)
        if not use_std:
            group_columns = [column for column in group_columns if not column.endswith("_std")]
        if not group_columns:
            raise ValueError(f"No features found for enabled group: {group}")
        columns.extend(group_columns)
    if len(columns) == 1:
        raise ValueError("Enable at least one feature group")
    return features[columns].copy()


def transform_features(features: pd.DataFrame, models: dict) -> pd.DataFrame:
    columns = models["feature_columns"]
    missing = set(columns) - set(features.columns)
    if missing:
        raise ValueError(f"Query is missing feature columns: {sorted(missing)}")
    scaled_values = models["scaler"].transform(features[columns])
    scaled = create_feature_table(scaled_values, columns, features)
    grouped = normalize_feature_groups(scaled)
    weighted = apply_group_weights(grouped, models["weights"])
    if models["pca"] is None:
        reduced = weighted
    else:
        values = models["pca"].transform(weighted.drop(columns="track_id"))
        reduced = create_feature_table(values, models["vector_columns"], features)
    if models.get("l2_normalize", True):
        return normalize_vectors(reduced)
    return reduced
