import logging
import sys
from pathlib import Path

from src.config import LOG_DIRECTORY

LOG_DIRECTORY.mkdir(exist_ok=True)
LOG_FILE = LOG_DIRECTORY / "simulation.log"

def get_logger(name: str, log_level = logging.INFO, log_file: str = LOG_FILE):
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(log_level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger