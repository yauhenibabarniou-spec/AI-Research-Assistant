import argparse
import logging
import sys
from pathlib import Path
from typing import Any

from app.harness.eval import ab_testing, generation, retrieval
from app.harness.eval.schema import EVAL_CONFIG_SCHEMA
from app.harness.utils.config import ConfigLoader
from app.harness.utils.logging import setup_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AI Research Assistant Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="eval_config.yaml",
        help="Path to evaluation config YAML file (default: eval_config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Evaluation command")

    # retrieval subcommand
    retrieval_parser = subparsers.add_parser("retrieval", help="Run retrieval evaluation")
    retrieval_parser.add_argument("--config", default=None, help="Path to evaluation config YAML file (overrides global)")
    retrieval_parser.add_argument("--k", type=int, default=None, help="Number of documents to retrieve")
    retrieval_parser.add_argument("--score-threshold", type=float, default=None, help="Minimum relevance score")
    retrieval_parser.add_argument("--search-type", type=str, default=None, choices=["weighted", "rrf", "two_stage"])
    retrieval_parser.add_argument("--alpha", type=float, default=None, help="Weight for BM25 in weighted search")
    retrieval_parser.add_argument("--output", type=str, default=None, help="Output CSV path (overrides config)")

    # generation subcommand
    generation_parser = subparsers.add_parser("generation", help="Run generation evaluation")
    generation_parser.add_argument("--config", default=None, help="Path to evaluation config YAML file (overrides global)")
    generation_parser.add_argument("--k", type=int, default=None, help="Number of documents to retrieve")
    generation_parser.add_argument("--score-threshold", type=float, default=None, help="Minimum relevance score")
    generation_parser.add_argument("--limit", type=int, default=None, help="Limit number of queries evaluated")
    generation_parser.add_argument("--output", type=str, default=None, help="Output JSON path (overrides config)")

    # ab subcommand
    ab_parser = subparsers.add_parser("ab", help="Run A/B testing")
    ab_parser.add_argument("--config", default=None, help="Path to evaluation config YAML file (overrides global)")
    ab_parser.add_argument("--output-dir", type=str, default=None, help="Output directory (overrides config)")

    return parser


def _merge_cli_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "retrieval":
        section = config.setdefault("retrieval", {})
        if args.k is not None:
            section["k"] = args.k
        if args.score_threshold is not None:
            section["score_threshold"] = args.score_threshold
        if args.search_type is not None:
            section["search_type"] = args.search_type
        if args.alpha is not None:
            section["alpha"] = args.alpha
        if args.output is not None:
            config.setdefault("global", {})["_retrieval_output"] = args.output
    elif args.command == "generation":
        section = config.setdefault("generation", {})
        if args.k is not None:
            section["k"] = args.k
        if args.score_threshold is not None:
            section["score_threshold"] = args.score_threshold
        if args.limit is not None:
            section["limit"] = args.limit
        if args.output is not None:
            config.setdefault("global", {})["_generation_output"] = args.output
    elif args.command == "ab":
        if args.output_dir is not None:
            config.setdefault("global", {})["_ab_output_dir"] = args.output_dir
    return config


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        subparser_config = getattr(args, "config", None)
        config_path = subparser_config if subparser_config is not None else parser.get_default("config")
        loader = ConfigLoader(config_path)
        config = loader.load_and_validate(EVAL_CONFIG_SCHEMA)
        config = _merge_cli_overrides(config, args)

        log_level = config.get("global", {}).get("log_level", "INFO")
        setup_logging(log_level)
        logger = logging.getLogger(__name__)
        logger.info("Starting harness with command: %s", args.command)

        project_root = Path(__file__).resolve().parents[3]

        if args.command == "retrieval":
            retrieval.run(config, project_root)
        elif args.command == "generation":
            generation.run(config, project_root)
        elif args.command == "ab":
            ab_testing.run(config, project_root)

        logger.info("Harness completed successfully")
        return 0
    except Exception as exc:
        logging.getLogger(__name__).exception("Unhandled exception in harness")
        return 1


if __name__ == "__main__":
    sys.exit(main())
