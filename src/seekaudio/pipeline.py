"""Connect the project's offline preparation stages.

Run flow:
    audio folders -> manifest -> feature table -> fitted search space
    -> catalog evaluation -> ablation reports and plots.

This module builds the catalog. The web app loads its saved output instead of
rebuilding the dataset for every visitor. See analysis.py for the live query flow.
"""

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import SeekAudioConfig, load_ablation_configs
from .evaluate import add_random_baseline, evaluate_precision_at_k, save_evaluation
from .features.extract import extract_feature_table, save_feature_table
from .ingest import ingest_dataset
from .space import build_feature_space, save_feature_space, select_features


def write_json(path: str | Path, data: dict) -> None:
    """Save a readable settings/metadata dictionary at an existing parent folder."""
    path = Path(path)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def feature_path(config: SeekAudioConfig) -> Path:
    """Choose the feature file using its version, keeping older versions separate."""
    return Path(config.paths.features_dir) / f"features_v{config.paths.feature_spec_version}.parquet"


def extraction_settings(config: SeekAudioConfig) -> dict:
    """Translate configuration into arguments accepted by the audio extractors.

    Rhythm is always extracted so ablations can select it later. Tonal features
    are optional. Group selection and mean-only pooling happen after extraction.
    """
    if not config.audio.mono:
        raise ValueError("Feature extraction currently requires audio.mono: true")
    if config.features.n_chroma != 12:
        raise ValueError("Tonal extraction currently supports n_chroma: 12")
    return {
        "sample_rate": config.audio.sample_rate,
        "duration": config.audio.duration,
        "n_mfcc": config.features.n_mfcc,
        "n_fft": config.features.n_fft,
        "hop_length": config.features.hop_length,
        "n_mels": config.features.n_mels,
        "include_rhythm": True,
        "include_tonal": config.features.groups.tonal,
    }


def extract_dataset(manifest: pd.DataFrame, config: SeekAudioConfig, output_path: str | Path) -> pd.DataFrame:
    """Extract one row per manifest track and save both features and their recipe.

    Returns a DataFrame containing track_id and raw numeric feature columns.
    The adjacent JSON file records how audio was prepared and measured, allowing
    a future uploaded song to use the same sample rate, windows, and duration.
    """
    if manifest.empty:
        raise ValueError("Manifest contains no tracks to extract")
    settings = extraction_settings(config)
    features = extract_feature_table(manifest, **settings, progress=True)
    output_path = save_feature_table(features, output_path)
    metadata = {
        "feature_spec_version": config.paths.feature_spec_version,
        "extraction_settings": settings,
    }
    write_json(output_path.with_suffix(".json"), metadata)
    return features


def build_configured_space(features: pd.DataFrame, config: SeekAudioConfig) -> tuple[pd.DataFrame, dict]:
    """Select enabled feature columns, fit transforms, and return vectors/models.

    Configured weights, PCA, whitening, and final normalization are applied here.
    Group settings and pooling determine which raw columns reach the scaler.
    """
    selected = select_features(features, asdict(config.features.groups), config.pooling.use_std)
    vectors, models = build_feature_space(
        selected,
        n_components=config.space.pca_components,
        whiten=config.space.whiten,
        weights=asdict(config.features.weights),
        l2_normalize=config.space.l2_normalize,
    )
    models["config"] = asdict(config)
    return vectors, models


def attach_extraction_settings(models: dict, features_path: str | Path) -> None:
    """Add the feature file's extraction recipe to the model dictionary in place.

    Legacy feature files without JSON metadata still support dataset-ID search,
    but audio-file analysis requires the recipe to be present in saved models.
    """
    metadata_path = Path(features_path).with_suffix(".json")
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        models["extraction_settings"] = metadata["extraction_settings"]
        models["feature_spec_version"] = metadata["feature_spec_version"]


def run_ablations(
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    config_path: str | Path,
    ablation_path: str | Path,
    output_dir: str | Path,
    k: int,
) -> pd.DataFrame:
    """Evaluate each YAML ablation on the same catalog and save a comparison.

    Each experiment selects/weights existing features and fits its own transforms;
    audio is not decoded again. K and the evaluated tracks stay the same.
    Returns one summary row per experiment, ordered by Precision@K.
    """
    output_dir = Path(output_dir)
    records = []
    for name, config in load_ablation_configs(config_path, ablation_path).items():
        print(f"Evaluating {name}...", flush=True)
        vectors, _ = build_configured_space(features, config)
        results = add_random_baseline(evaluate_precision_at_k(vectors, manifest, k))
        save_evaluation(results, k, output_dir / name)
        write_json(output_dir / name / "config.json", asdict(config))
        records.append({
            "experiment": name,
            "dimensions": len(vectors.columns) - 1,
            "k": k,
            "precision_at_k": results["precision_at_k"].mean(),
            "random_precision_at_k": results["random_precision_at_k"].mean(),
        })
    comparison = pd.DataFrame(records).sort_values("precision_at_k", ascending=False)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    return comparison.reset_index(drop=True)


def run_pipeline(config: SeekAudioConfig, config_path: str | Path, ablation_path: str | Path) -> pd.DataFrame:
    """Execute all five preparation stages and return per-track evaluation scores.

    Existing output files at the configured paths are replaced. Raw recordings
    are only read. The results describe this catalog; they are not held-out
    classification accuracy because transforms are fitted on the same catalog.
    """
    # Step 1: inventory readable files and persist their IDs, genres, and metadata.
    print("1/5 Ingesting audio...", flush=True)
    manifest = ingest_dataset(config.paths.audio_dir, config.paths.manifest)
    # Step 2: turn each prepared waveform into a fixed set of acoustic numbers.
    print("2/5 Extracting features...", flush=True)
    features = extract_dataset(manifest, config, feature_path(config))
    # Step 3: fit scaling/PCA once, then save vectors and reusable query transforms.
    print("3/5 Building feature space...", flush=True)
    vectors, models = build_configured_space(features, config)
    attach_extraction_settings(models, feature_path(config))
    save_feature_space(vectors, models, config.paths.space_dir)
    # Step 4: search for every catalog track, excluding itself, and score genre matches.
    print("4/5 Evaluating retrieval...", flush=True)
    results = add_random_baseline(evaluate_precision_at_k(vectors, manifest, config.retrieval.k))
    report_dir = Path(config.paths.evaluation_dir) / "current"
    save_evaluation(results, config.retrieval.k, report_dir)
    write_json(report_dir / "config.json", asdict(config))
    # Step 5: compare representations and produce figures for the research report.
    print("5/5 Comparing ablations and creating plots...", flush=True)
    ablation_dir = Path(config.paths.evaluation_dir) / "ablations"
    comparison = run_ablations(
        features, manifest, config_path, ablation_path,
        ablation_dir, config.retrieval.k,
    )
    from .viz import plot_evaluation
    plot_evaluation(results, comparison, Path(config.paths.evaluation_dir) / "plots", config.retrieval.k)
    print(f"Finished: {len(vectors)} tracks; Precision@{config.retrieval.k} = {results.precision_at_k.mean():.4f}")
    print(comparison[["experiment", "precision_at_k"]].to_string(index=False))
    return results
