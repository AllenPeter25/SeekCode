"""One readable handler per CLI action; arguments override YAML paths/settings."""

from pathlib import Path

import joblib
import pandas as pd

from .evaluate import add_random_baseline, evaluate_precision_at_k, save_evaluation
from .experiments import compare_feature_groups
from .ingest import ingest_dataset
from .pipeline import (
    attach_extraction_settings,
    build_configured_space,
    extract_dataset,
    feature_path,
    run_ablations,
    run_pipeline,
)
from .retrieval import add_track_details, find_similar_tracks, search_audio_file
from .space import save_feature_space


def get_match_count(args, config) -> int:
    """Prefer explicit --k, otherwise use retrieval.k from configuration."""
    if args.k is None:
        return config.retrieval.k
    return args.k


def print_search_results(results: pd.DataFrame, manifest: pd.DataFrame) -> None:
    """Attach display metadata and print ranked scores to four decimal places."""
    details = add_track_details(results, manifest)
    display = details.to_string(index=False, float_format="{:.4f}".format)
    print(display)


def run_ingest(args, config) -> None:
    """Handle ingest: scan the selected audio directory and save its manifest."""
    audio_dir = args.audio_dir or config.paths.audio_dir
    output_path = args.output or config.paths.manifest
    manifest = ingest_dataset(audio_dir, output_path)
    print(f"Saved {len(manifest)} tracks to {output_path}")


def run_extract(args, config) -> None:
    """Handle extract: read the manifest, extract features, and save their recipe."""
    manifest_path = args.manifest or config.paths.manifest
    output_path = args.output or feature_path(config)
    manifest = pd.read_parquet(manifest_path)
    features = extract_dataset(manifest, config, output_path)
    feature_count = len(features.columns) - 1
    print(f"Saved {len(features)} tracks with {feature_count} features to {output_path}")


def run_build_space(args, config) -> None:
    """Handle build-space: fit and save the selected catalog representation."""
    if args.components is not None:
        # CLI zero maps to None, the internal representation of 'skip PCA'.
        config.space.pca_components = args.components or None
    features_path = args.features or feature_path(config)
    output_dir = args.output_dir or config.paths.space_dir
    features = pd.read_parquet(features_path)
    vectors, models = build_configured_space(features, config)
    attach_extraction_settings(models, features_path)
    save_feature_space(vectors, models, output_dir)
    dimensions = len(vectors.columns) - 1
    print(f"Saved {len(vectors)} tracks with {dimensions} dimensions to {output_dir}")


def run_search(args, config) -> None:
    """Handle dataset-ID search using saved vectors, without decoding audio."""
    manifest_path = args.manifest or config.paths.manifest
    vectors_path = args.vectors or Path(config.paths.space_dir) / "vectors.parquet"
    manifest = pd.read_parquet(manifest_path)
    vectors = pd.read_parquet(vectors_path)
    results = find_similar_tracks(vectors, args.track_id, get_match_count(args, config))
    print_search_results(results, manifest)


def find_existing_track_id(audio_path, manifest: pd.DataFrame) -> str | None:
    """Match an exact resolved file path to its catalog ID, or return None.

    This excludes self-retrieval for an existing dataset file. It does not detect
    copied files, renamed duplicates, or another recording of the same song.
    """
    query_path = Path(audio_path).resolve()
    for row in manifest.itertuples(index=False):
        if Path(row.path).resolve() == query_path:
            return row.track_id
    return None


def run_audio_search(args, config) -> None:
    """Handle search-file: load saved models and transform one audio-file query."""
    manifest_path = args.manifest or config.paths.manifest
    space_dir = Path(args.space_dir or config.paths.space_dir)
    manifest = pd.read_parquet(manifest_path)
    vectors = pd.read_parquet(space_dir / "vectors.parquet")
    models = joblib.load(space_dir / "models.joblib")
    exclude_id = find_existing_track_id(args.audio_path, manifest)
    results = search_audio_file(
        args.audio_path, vectors, models,
        k=get_match_count(args, config), exclude_track_id=exclude_id,
    )
    print_search_results(results, manifest)


def run_evaluate(args, config) -> None:
    """Handle evaluate: score the catalog, print aggregate results, and save reports."""
    manifest_path = args.manifest or config.paths.manifest
    vectors_path = args.vectors or Path(config.paths.space_dir) / "vectors.parquet"
    output_dir = args.output_dir or Path(config.paths.evaluation_dir) / "current"
    manifest = pd.read_parquet(manifest_path)
    vectors = pd.read_parquet(vectors_path)
    k = get_match_count(args, config)
    results = evaluate_precision_at_k(vectors, manifest, k)
    results = add_random_baseline(results)
    overall_score = results["precision_at_k"].mean()
    random_score = results["random_precision_at_k"].mean()
    print(f"Evaluated {len(results)} tracks")
    print(f"Mean Precision@{k}: {overall_score:.4f}")
    print(f"Expected random Precision@{k}: {random_score:.4f}")
    if random_score > 0:
        print(f"Lift over random: {overall_score / random_score:.2f}x")
    genre_scores = results.groupby("genre")["precision_at_k"].mean()
    print(genre_scores.to_string())
    save_evaluation(results, k, output_dir)
    print(f"Saved evaluation results to {output_dir}")


def run_comparison(args, config) -> None:
    """Handle either the basic group comparison or the YAML-defined ablations."""
    features_path = args.features or feature_path(config)
    manifest_path = args.manifest or config.paths.manifest
    output_dir = args.output_dir or Path(config.paths.evaluation_dir) / args.command
    features = pd.read_parquet(features_path)
    manifest = pd.read_parquet(manifest_path)
    k = get_match_count(args, config)
    if args.command == "ablate":
        comparison = run_ablations(features, manifest, args.config, args.ablations, output_dir, k)
    else:
        comparison = compare_feature_groups(features, manifest, output_dir, k, args.components)
    print(comparison[["experiment", "dimensions", "precision_at_k"]].to_string(index=False))
    print(f"Saved experiment reports to {output_dir}")


def run_all(args, config) -> None:
    """Handle run: optionally override the audio folder, then execute all stages."""
    if args.audio_dir:
        config.paths.audio_dir = args.audio_dir
    run_pipeline(config, args.config, args.ablations)
