#!/usr/bin/env python3
"""Run A/B testing across RAG configurations."""

import logging
from pathlib import Path

from app.eval.ab_testing import ABConfig, ABTester

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    tester = ABTester(
        golden_dataset_path=project_root / "eval" / "golden_dataset.json",
        persist_directory=project_root / "chroma_db",
        knowledge_base_dir=project_root / "knowledge_base",
    )

    configs = [
        ABConfig(name="base", chunk_size=800, chunk_overlap=120, k=3),
        ABConfig(name="small_chunks", chunk_size=400, chunk_overlap=80, k=3),
        ABConfig(name="large_chunks", chunk_size=1200, chunk_overlap=200, k=3),
        ABConfig(name="k_5", chunk_size=800, chunk_overlap=120, k=5),
        ABConfig(name="k_7", chunk_size=800, chunk_overlap=120, k=7),
        ABConfig(name="strict_threshold", chunk_size=800, chunk_overlap=120, k=3, score_threshold=0.3),
    ]

    results = tester.run_suite(configs)
    md_path = tester.save_results(results, project_root / "eval" / "ab_results")
    print(f"Results saved to {md_path}")


if __name__ == "__main__":
    main()
