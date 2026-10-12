import numpy as np
import pandas as pd
import pytest

from src.seekaudio.space import (
    apply_group_weights,
    normalize_feature_groups,
    normalize_vectors,
    reduce_features_pca,
    standardize_features,
)


def test_standardize_features_and_reuse_scaler() -> None:
    features = pd.DataFrame(
        {
            "track_id": ["a", "b", "c"],
            "spectral_centroid_mean": [1_000.0, 2_000.0, 3_000.0],
            "energy_rms_mean": [0.1, 0.2, 0.3],
            "constant": [1.0, 1.0, 1.0],
        },
        index=[4, 5, 6],
    )

    standardized, scaler = standardize_features(features)
    numeric = standardized.drop(columns="track_id")

    assert standardized["track_id"].equals(features["track_id"])
    assert np.allclose(numeric.mean(), 0)
    assert np.allclose(numeric.iloc[:, :2].std(ddof=0), 1)
    assert np.allclose(numeric["constant"], 0)

    query = pd.DataFrame(
        {"spectral_centroid_mean": [2_000.0], "energy_rms_mean": [0.2], "constant": [1.0]}
    )
    assert np.allclose(scaler.transform(query), 0)


def test_standardize_features_rejects_missing_values() -> None:
    features = pd.DataFrame({"track_id": ["a"], "energy_rms_mean": [np.nan]})

    with pytest.raises(ValueError, match="finite"):
        standardize_features(features)


def test_normalize_feature_groups_balances_groups_and_handles_zero() -> None:
    features = pd.DataFrame(
        {
            "track_id": ["a", "silent"],
            "mfcc_01_mean": [3.0, 0.0],
            "mfcc_02_mean": [4.0, 0.0],
            "energy_rms_mean": [10.0, 0.0],
        },
        index=[4, 5],
    )
    original = features.copy()

    normalized = normalize_feature_groups(features)

    assert normalized["track_id"].equals(features["track_id"])
    assert np.allclose(normalized.iloc[0, 1:].to_numpy(dtype=float), [0.6, 0.8, 1.0])
    assert np.allclose(normalized.iloc[1, 1:].to_numpy(dtype=float), 0)
    pd.testing.assert_frame_equal(features, original)


def test_normalize_feature_groups_rejects_unknown_columns() -> None:
    features = pd.DataFrame({"track_id": ["a"], "unknown_mean": [1.0]})

    with pytest.raises(ValueError, match="Unknown feature groups"):
        normalize_feature_groups(features)


def test_apply_group_weights_preserves_ids_and_unspecified_groups() -> None:
    features = pd.DataFrame(
        {
            "track_id": ["a"],
            "mfcc_01_mean": [0.6],
            "mfcc_02_mean": [0.8],
            "spectral_centroid_mean": [1.0],
            "energy_rms_mean": [1.0],
        },
        index=[4],
    )
    original = features.copy()

    weighted = apply_group_weights(features, {"mfcc": 2.0, "energy": 0.0})

    assert weighted["track_id"].equals(features["track_id"])
    assert np.allclose(weighted.iloc[0, 1:].to_numpy(dtype=float), [1.2, 1.6, 1.0, 0.0])
    pd.testing.assert_frame_equal(features, original)


@pytest.mark.parametrize("weight", [-1.0, np.nan, np.inf])
def test_apply_group_weights_rejects_invalid_weights(weight: float) -> None:
    features = pd.DataFrame({"track_id": ["a"], "energy_rms_mean": [1.0]})

    with pytest.raises(ValueError, match="finite and non-negative"):
        apply_group_weights(features, {"energy": weight})


def test_reduce_features_pca_preserves_ids_and_reuses_model() -> None:
    values = np.random.default_rng(42).normal(size=(10, 3))
    features = pd.DataFrame(values, columns=["a", "b", "c"], index=range(10, 20))
    features.insert(0, "track_id", [f"track_{index}" for index in range(10)])

    reduced, pca = reduce_features_pca(features, n_components=2)

    assert reduced.shape == (10, 3)
    assert reduced["track_id"].equals(features["track_id"])
    assert np.allclose(reduced.drop(columns="track_id").var(ddof=1), 1)
    assert np.allclose(
        pca.transform(features.drop(columns="track_id").iloc[:1]),
        reduced.drop(columns="track_id").iloc[:1],
    )


def test_reduce_features_pca_rejects_too_many_components() -> None:
    features = pd.DataFrame({"track_id": ["a", "b"], "energy_rms_mean": [0.1, 0.2]})

    with pytest.raises(ValueError, match="n_components"):
        reduce_features_pca(features, n_components=40)


def test_reduce_features_pca_rejects_whitening_constant_data() -> None:
    features = pd.DataFrame({"track_id": ["a", "b"], "energy_rms_mean": [1.0, 1.0]})

    with pytest.warns(RuntimeWarning):
        with pytest.raises(ValueError, match="Cannot whiten"):
            reduce_features_pca(features, n_components=1)


def test_normalize_vectors_preserves_direction_and_handles_zero() -> None:
    features = pd.DataFrame(
        {
            "track_id": ["a", "b", "zero"],
            "pca_01": [3.0, -3.0, 0.0],
            "pca_02": [4.0, 4.0, 0.0],
        },
        index=[5, 6, 7],
    )
    original = features.copy()

    normalized = normalize_vectors(features)
    values = normalized.drop(columns="track_id").to_numpy()

    assert np.allclose(values, [[0.6, 0.8], [-0.6, 0.8], [0.0, 0.0]])
    assert np.allclose(np.linalg.norm(values, axis=1), [1.0, 1.0, 0.0])
    assert values[0] @ values[1] == pytest.approx(0.28)
    assert normalized["track_id"].equals(features["track_id"])
    pd.testing.assert_frame_equal(features, original)


def test_normalize_vectors_rejects_nonfinite_values() -> None:
    features = pd.DataFrame({"track_id": ["a"], "pca_01": [np.inf]})

    with pytest.raises(ValueError, match="finite"):
        normalize_vectors(features)
