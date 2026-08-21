import csv
import json
import logging
from pathlib import Path
from typing import Any

from app.eval.ab_testing import ABConfig, ABTester
from app.harness.utils.storage import generate_run_id, ensure_output_dir, write_csv, write_json

logger = logging.getLogger(__name__)


def run(config: dict[str, Any], project_root: Path) -> None:
    global_config = config.get("global", {})
    ab_config = config.get("ab_testing", {})

    configs = [
        ABConfig(
            name=c["name"],
            chunk_size=c["chunk_size"],
            chunk_overlap=c["chunk_overlap"],
            embedding_model=c.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            k=c["k"],
            score_threshold=c.get("score_threshold", 0.0),
        )
        for c in ab_config.get("configs", [])
    ]

    output_dir = global_config.get("output_dir", "eval/reports")
    output_dir_override = global_config.get("_ab_output_dir")
    output_base = Path(output_dir_override) if output_dir_override else project_root / output_dir

    tester = ABTester(
        golden_dataset_path=project_root / "eval" / "golden_dataset.json",
        persist_directory=project_root / "chroma_db",
        knowledge_base_dir=project_root / "knowledge_base",
    )

    results = tester.run_suite(configs)

    run_id = generate_run_id()
    output_path = ensure_output_dir(output_base, run_id)

    md_path = tester.save_results(results, output_path)
    logger.info("A/B testing results saved to %s", md_path)

    payload = [
        {
            **r.config.to_dict(),
            "metrics": r.metrics,
            "error": r.error,
        }
        for r in results
    ]
    print(json.dumps({"run_id": run_id, "results": payload}, ensure_ascii=False, indent=2))
