import logging
from enum import Enum
from pathlib import Path
from typing import Any

from app.harness.utils.logging import setup_logging

logger = logging.getLogger(__name__)


class AppMode(str, Enum):
    DEV = "dev"
    TEST = "test"
    PROD = "prod"


class AppHarness:
    """Application harness for lifecycle management."""

    def __init__(self, mode: AppMode = AppMode.DEV, log_level: str = "INFO") -> None:
        self.mode = mode
        setup_logging(log_level)
        logger.info("AppHarness initialized in %s mode", mode)

    def run(self) -> None:
        if self.mode == AppMode.DEV:
            from app.harness.app import dev
            dev.run()
        elif self.mode == AppMode.TEST:
            from app.harness.app import test
            test.run()
        elif self.mode == AppMode.PROD:
            from app.harness.app import prod
            prod.run()

    def health_check(self) -> dict[str, Any]:
        from app.harness.app.health import check_chromadb, check_ollama, check_vector_store
        return {
            "chromadb": check_chromadb(),
            "ollama": check_ollama(),
            "vector_store": check_vector_store(),
        }
