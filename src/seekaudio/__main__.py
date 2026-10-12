import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import load_config
from .evaluate import add_random_baseline, evaluate_precision_at_k, save_evaluation
from .experiments import compare_feature_groups
from .ingest import ingest_dataset
from .pipeline import attach_extraction_settings, build_configured_space, extract_dataset, feature_path, run_ablations, run_pipeline
from .retrieval import add_track_details, find_similar_tracks, search_audio_file
from .space import save_feature_space


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="seekaudio")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Scan audio and save its manifest")
    ingest.add_argument("--audio-dir")
    ingest.add_argument("--output")
    extract = commands.add_parser("extract", help="Extract and save audio features")
    extract.add_argument("--manifest")
    extract.add_argument("--output")
    space = commands.add_parser("build-space", help="Save vectors and fitted models")
    space.add_argument("--features")
    space.add_argument("--output-dir")
    space.add_argument("--components", type=int, help="PCA component count; 0 skips PCA")
    search = commands.add_parser("search", help="Find matches for a dataset track ID")
    search.add_argument("track_id")
    search.add_argument("--vectors")
    search.add_argument("--manifest")
    search.add_argument("--k", type=int)
    audio_search = commands.add_parser("search-file", help="Find matches for an audio file")
    audio_search.add_argument("audio_path")
    audio_search.add_argument("--space-dir")
    audio_search.add_argument("--manifest")
    audio_search.add_argument("--k", type=int)
    evaluate = commands.add_parser("evaluate", help="Evaluate genre agreement and save reports")
    evaluate.add_argument("--k", type=int)
    evaluate.add_argument("--vectors")
    evaluate.add_argument("--manifest")
    evaluate.add_argument("--output-dir")
    compare = commands.add_parser("compare-features", help="Compare basic feature groups")
    compare.add_argument("--features")
    compare.add_argument("--manifest")
    compare.add_argument("--output-dir")
    compare.add_argument("--k", type=int)
    compare.add_argument("--components", type=int, default=40)
    ablate = commands.add_parser("ablate", help="Run experiments from ablation.yaml")
    ablate.add_argument("--features")
    ablate.add_argument("--manifest")
    ablate.add_argument("--output-dir")
    ablate.add_argument("--k", type=int)
    ablate.add_argument("--ablations", default="config/ablation.yaml")
    run = commands.add_parser("run", help="Build, evaluate, compare, and plot the full pipeline")
    run.add_argument("--audio-dir")
    run.add_argument("--ablations", default="config/ablation.yaml")
    for command in [ingest, extract, space, search, audio_search, evaluate, compare, ablate, run]:
        command.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args(arguments)
    try:
        config = load_config(args.config)
        requested_k = getattr(args, "k", None)
        k = config.retrieval.k if requested_k is None else requested_k
        manifest_path = getattr(args, "manifest", None) or config.paths.manifest
        features_path = getattr(args, "features", None) or feature_path(config)
        if args.command == "run":
            if args.audio_dir:
                config.paths.audio_dir = args.audio_dir
            run_pipeline(config, args.config, args.ablations)
        elif args.command == "ingest":
            output = args.output or config.paths.manifest
            manifest = ingest_dataset(args.audio_dir or config.paths.audio_dir, output)
            print(f"Saved {len(manifest)} tracks to {output}")
        elif args.command == "extract":
            output = args.output or feature_path(config)
            features = extract_dataset(pd.read_parquet(manifest_path), config, output)
            print(f"Saved {len(features)} tracks with {len(features.columns) - 1} features to {output}")
        elif args.command == "build-space":
            if args.components is not None:
                config.space.pca_components = args.components or None
            vectors, models = build_configured_space(pd.read_parquet(features_path), config)
            attach_extraction_settings(models, features_path)
            output = args.output_dir or config.paths.space_dir
            save_feature_space(vectors, models, output)
            print(f"Saved {len(vectors)} tracks with {len(vectors.columns) - 1} dimensions to {output}")
        elif args.command in {"search", "search-file"}:
            manifest = pd.read_parquet(manifest_path)
            if args.command == "search":
                vectors = pd.read_parquet(args.vectors or Path(config.paths.space_dir) / "vectors.parquet")
                results = find_similar_tracks(vectors, args.track_id, k)
            else:
                space_dir = Path(args.space_dir or config.paths.space_dir)
                vectors = pd.read_parquet(space_dir / "vectors.parquet")
                models = joblib.load(space_dir / "models.joblib")
                query_path = Path(args.audio_path).resolve()
                same_file = manifest["path"].map(lambda path: Path(path).resolve() == query_path)
                exclude_id = manifest.loc[same_file, "track_id"].iloc[0] if same_file.any() else None
                results = search_audio_file(args.audio_path, vectors, models, k, exclude_id)
            print(add_track_details(results, manifest).to_string(index=False, float_format=lambda value: f"{value:.4f}"))
        elif args.command == "evaluate":
            vectors = pd.read_parquet(args.vectors or Path(config.paths.space_dir) / "vectors.parquet")
            results = add_random_baseline(evaluate_precision_at_k(vectors, pd.read_parquet(manifest_path), k))
            overall = results.precision_at_k.mean()
            random_score = results.random_precision_at_k.mean()
            print(f"Evaluated {len(results)} tracks")
            print(f"Mean Precision@{k}: {overall:.4f}")
            print(f"Expected random Precision@{k}: {random_score:.4f}")
            if random_score > 0:
                print(f"Lift over random: {overall / random_score:.2f}x")
            print(results.groupby("genre").precision_at_k.mean().to_string())
            output = args.output_dir or Path(config.paths.evaluation_dir) / "current"
            save_evaluation(results, k, output)
            print(f"Saved evaluation results to {output}")
        elif args.command in {"compare-features", "ablate"}:
            features = pd.read_parquet(features_path)
            manifest = pd.read_parquet(manifest_path)
            output = args.output_dir or Path(config.paths.evaluation_dir) / args.command
            if args.command == "ablate":
                comparison = run_ablations(features, manifest, args.config, args.ablations, output, k)
            else:
                comparison = compare_feature_groups(features, manifest, output, k, args.components)
            print(comparison[["experiment", "dimensions", "precision_at_k"]].to_string(index=False))
            print(f"Saved experiment reports to {output}")
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
