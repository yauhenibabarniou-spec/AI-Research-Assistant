import logging
import os
from pathlib import Path
from typing import Any

import yaml

from app.harness.eval.schema import EVAL_CONFIG_SCHEMA

try:
    import jsonschema
except ImportError:
    jsonschema = None


logger = logging.getLogger(__name__)


DEFAULT_CONFIG: dict[str, Any] = {
    "global": {
        "output_dir": "eval/reports",
        "log_level": "INFO",
        "seed": 42,
    },
    "retrieval": {
        "k": 3,
        "score_threshold": 0.3,
        "alpha": 0.6,
        "search_type": "weighted",
    },
    "generation": {
        "k": 3,
        "score_threshold": 0.0,
        "limit": None,
    },
    "ab_testing": {
        "configs": [
            {
                "name": "base",
                "chunk_size": 800,
                "chunk_overlap": 120,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "k": 3,
                "score_threshold": 0.0,
            }
        ],
    },
}


class ConfigLoader:
    def __init__(self, config_path: str = "eval_config.yaml") -> None:
        self.config_path = Path(config_path)
        self.config: dict[str, Any] = {}

    def load(self) -> dict[str, Any]:
        self.config = dict(DEFAULT_CONFIG)
        if self.config_path.exists():
            with open(self.config_path, encoding="utf-8") as f:
                user_config = yaml.safe_load(f) or {}
            self._deep_merge(self.config, user_config)
        else:
            logger.warning("Config file %s not found, using defaults", self.config_path)
        return self.config

    def load_and_validate(self, schema: dict[str, Any]) -> dict[str, Any]:
        config = self.load()
        if jsonschema is not None:
            try:
                jsonschema.validate(instance=config, schema=schema)
            except jsonschema.ValidationError as exc:
                raise ValueError(f"Config validation failed: {exc.message}") from exc
        else:
            logger.warning("jsonschema not installed, skipping schema validation")
        return config

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> None:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                ConfigLoader._deep_merge(base[key], value)
            else:
                base[key] = value
