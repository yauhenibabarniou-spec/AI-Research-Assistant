import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class GoldenItem(BaseModel):
    id: str
    question: str
    expected_answer: str
    expected_chunk_ids: list[str]
    source_doc: str
    difficulty: str
    category: str


class GoldenDataset:
    """Загрузчик и валидатор golden dataset."""

    def __init__(self, path: str | Path = "eval/golden_dataset.json"):
        self.path = Path(path)
        self.items: list[GoldenItem] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Golden dataset not found at {self.path}")
        with open(self.path, encoding="utf-8") as f:
            raw = json.load(f)
        try:
            self.items = [GoldenItem(**item) for item in raw]
        except ValidationError as exc:
            raise ValueError(f"Invalid golden dataset schema: {exc}") from exc
        logger.info("Loaded %d golden items from %s", len(self.items), self.path)

    def get_by_source(self, source_doc: str) -> list[GoldenItem]:
        return [item for item in self.items if item.source_doc == source_doc]

    def get_by_category(self, category: str) -> list[GoldenItem]:
        return [item for item in self.items if item.category == category]

    def get_by_difficulty(self, difficulty: str) -> list[GoldenItem]:
        return [item for item in self.items if item.difficulty == difficulty]

    def get_multidoc_items(self) -> list[GoldenItem]:
        return [item for item in self.items if item.source_doc == "multiple"]

    def all(self) -> list[GoldenItem]:
        return list(self.items)

    def summary(self) -> dict[str, Any]:
        return {
            "total": len(self.items),
            "by_source": {
                source: len(self.get_by_source(source)) for source in sorted({item.source_doc for item in self.items})
            },
            "by_difficulty": {
                difficulty: len(self.get_by_difficulty(difficulty))
                for difficulty in sorted({item.difficulty for item in self.items})
            },
            "by_category": {
                category: len(self.get_by_category(category))
                for category in sorted({item.category for item in self.items})
            },
            "multi_doc": len(self.get_multidoc_items()),
        }
