import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import SeekAudioConfig, load_ablation_configs
from .evaluate import add_random_baseline, evaluate_precision_at_k, save_evaluation
from .features.extract import extract_feature_table, save_feature_table
from .ingest import ingest_dataset
from .space import build_feature_space, save_feature_space, select_features


def feature_path(config: SeekAudioConfig) -> Path:
    return Path(config.paths.features_dir) / f"features_v{config.paths.feature_spec_version}.parquet"


def extraction_settings(config: SeekAudioConfig) -> dict:
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
    if manifest.empty:
        raise ValueError("Manifest contains no tracks to extract")
    settings = extraction_settings(config)
    features = extract_feature_table(manifest, **settings, progress=True)
    output_path = save_feature_table(features, output_path)
    output_path.with_suffix(".json").write_text(
        json.dumps({"feature_spec_version": config.paths.feature_spec_version, "extraction_settings": settings}, indent=2),
        encoding="utf-8",
    )
    return features


def build_configured_space(features: pd.DataFrame, config: SeekAudioConfig) -> tuple[pd.DataFrame, dict]:
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
    output_dir = Path(output_dir)
    records = []
    for name, config in load_ablation_configs(config_path, ablation_path).items():
        print(f"Evaluating {name}...", flush=True)
        vectors, _ = build_configured_space(features, config)
        results = add_random_baseline(evaluate_precision_at_k(vectors, manifest, k))
        save_evaluation(results, k, output_dir / name)
        (output_dir / name / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
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
    print("1/5 Ingesting audio...", flush=True)
    manifest = ingest_dataset(config.paths.audio_dir, config.paths.manifest)
    print("2/5 Extracting features...", flush=True)
    features = extract_dataset(manifest, config, feature_path(config))
    print("3/5 Building feature space...", flush=True)
    vectors, models = build_configured_space(features, config)
    attach_extraction_settings(models, feature_path(config))
    save_feature_space(vectors, models, config.paths.space_dir)
    print("4/5 Evaluating retrieval...", flush=True)
    results = add_random_baseline(evaluate_precision_at_k(vectors, manifest, config.retrieval.k))
    report_dir = Path(config.paths.evaluation_dir) / "current"
    save_evaluation(results, config.retrieval.k, report_dir)
    (report_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    print("5/5 Comparing ablations and creating plots...", flush=True)
    comparison = run_ablations(features, manifest, config_path, ablation_path,
                               Path(config.paths.evaluation_dir) / "ablations", config.retrieval.k)
    from .viz import plot_evaluation
    plot_evaluation(results, comparison, Path(config.paths.evaluation_dir) / "plots", config.retrieval.k)
    print(f"Finished: {len(vectors)} tracks; Precision@{config.retrieval.k} = {results.precision_at_k.mean():.4f}")
    print(comparison[["experiment", "precision_at_k"]].to_string(index=False))
    return results
