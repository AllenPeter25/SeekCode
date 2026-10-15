"""Compare basic feature groups using the same raw table and catalog labels.

This earlier comparison covers MFCC/spectral/energy. pipeline.run_ablations is
the configuration-driven comparison that also includes rhythm and mean pooling.
"""

from pathlib import Path

import pandas as pd

from .evaluate import add_random_baseline, evaluate_precision_at_k, save_evaluation
from .space import build_feature_space


def compare_feature_groups(
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    output_dir: str | Path,
    k: int = 10,
    pca_components: int = 40,
) -> pd.DataFrame:
    """Fit/evaluate six basic representations and save detailed comparison reports.

    Each tuple contains enabled prefixes and an optional PCA component count.
    No-PCA comparisons isolate feature-group effects; all_pca separately measures
    reduction plus whitening. Equal group weights are used throughout.
    """
    experiments = {
        "mfcc_only": (("mfcc",), None),
        "spectral_only": (("spectral",), None),
        "energy_only": (("energy",), None),
        "mfcc_spectral": (("mfcc", "spectral"), None),
        "all_no_pca": (("mfcc", "spectral", "energy"), None),
        "all_pca": (("mfcc", "spectral", "energy"), pca_components),
    }
    output_dir = Path(output_dir)
    records = []

    for name, (groups, components) in experiments.items():
        # Select the same ID column and just the groups named by this experiment.
        columns = ["track_id"]
        for group in groups:
            group_columns = [column for column in features.columns if column.startswith(f"{group}_")]
            if not group_columns:
                raise ValueError(f"Missing features for group: {group}")
            columns.extend(group_columns)

        vectors, _ = build_feature_space(features[columns], n_components=components)
        # The fitted catalog is also evaluated here; these are descriptive ablations.
        results = evaluate_precision_at_k(vectors, manifest, k)
        results = add_random_baseline(results)
        save_evaluation(results, k, output_dir / name)
        records.append({
            "experiment": name,
            "groups": ",".join(groups),
            "pca_components": components,
            "whiten": components is not None,
            "dimensions": len(vectors.columns) - 1,
            "track_count": len(results),
            "k": k,
            "precision_at_k": results["precision_at_k"].mean(),
            "random_precision_at_k": results["random_precision_at_k"].mean(),
        })

    comparison = pd.DataFrame(records).sort_values("precision_at_k", ascending=False)
    comparison.to_csv(output_dir / "comparison.csv", index=False)

    return comparison.reset_index(drop=True)
