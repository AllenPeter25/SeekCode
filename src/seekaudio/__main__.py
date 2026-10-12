import argparse

from .config import load_config
from .cli_commands import (
    run_all, run_audio_search, run_build_space, run_comparison,
    run_evaluate, run_extract, run_ingest, run_search,
)


def create_parser() -> argparse.ArgumentParser:
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

    return parser


def execute_command(args, config) -> None:
    if args.command == "ingest":
        run_ingest(args, config)
    elif args.command == "extract":
        run_extract(args, config)
    elif args.command == "build-space":
        run_build_space(args, config)
    elif args.command == "search":
        run_search(args, config)
    elif args.command == "search-file":
        run_audio_search(args, config)
    elif args.command == "evaluate":
        run_evaluate(args, config)
    elif args.command in {"compare-features", "ablate"}:
        run_comparison(args, config)
    elif args.command == "run":
        run_all(args, config)


def main(arguments: list[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(arguments)
    try:
        config = load_config(args.config)
        execute_command(args, config)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
