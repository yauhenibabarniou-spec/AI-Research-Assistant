import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def generate_run_id() -> str:
    return hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]


def ensure_output_dir(base_dir: str | Path, run_id: str | None = None) -> Path:
    base = Path(base_dir)
    if run_id:
        output_dir = base / run_id
    else:
        output_dir = base
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_csv(rows: list[dict[str, Any]], fieldnames: list[str], path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
