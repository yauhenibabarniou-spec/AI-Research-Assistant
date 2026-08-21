import json
import logging
import sys
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.eval.generation_metrics import run_generation_eval
from app.harness.utils.storage import generate_run_id, ensure_output_dir, write_json

logger = logging.getLogger(__name__)


def run(config: dict[str, Any], project_root: Path) -> None:
    generation_config = config.get("generation", {})
    global_config = config.get("global", {})

    k = generation_config.get("k", 3)
    score_threshold = generation_config.get("score_threshold", 0.0)
    limit = generation_config.get("limit", None)
    output_dir = global_config.get("output_dir", "eval/reports")
    output_path = global_config.get("_generation_output")

    run_id = generate_run_id()
    output_dir_path = ensure_output_dir(project_root / output_dir, run_id)

    if output_path:
        output_path_obj = Path(output_path)
        if not output_path_obj.is_absolute():
            output_path_obj = output_dir_path / output_path_obj
    else:
        output_path_obj = output_dir_path / "generation_report.json"

    try:
        summary = run_generation_eval(
            k=k,
            score_threshold=score_threshold,
            output_path=str(output_path_obj),
            limit=limit,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    except Exception as exc:
        logger.error("Generation eval failed: %s", exc)
        raise

    result = {"run_id": run_id, "summary": summary}
    print(json.dumps(result, ensure_ascii=False, indent=2))
