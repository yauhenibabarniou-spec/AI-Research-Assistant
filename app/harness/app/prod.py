import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


def run() -> None:
    logger.info("Starting production server...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--workers",
            "4",
        ]
    )
