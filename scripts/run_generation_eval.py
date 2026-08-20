#!/usr/bin/env python3
"""Run generation evaluation using LLM-as-Judge (subset for demo)."""

import logging
import sys
from pathlib import Path

from app.eval.generation_metrics import run_generation_eval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    try:
        summary = run_generation_eval(
            k=3,
            score_threshold=0.0,
            output_path="eval/reports/generation_report.json",
        )
        print(summary)
    except Exception as exc:
        logger.error("Generation eval failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
