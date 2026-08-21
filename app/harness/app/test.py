import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


def run() -> None:
    logger.info("Running test suite...")
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
