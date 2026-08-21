import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


def run() -> None:
    logger.info("Starting development server with hot reload...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--reload",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
    )
