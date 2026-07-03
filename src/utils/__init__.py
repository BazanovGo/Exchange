"""Shared utilities: project paths, logging."""

from src.utils.logging import get_logger
from src.utils.paths import (
    DATA_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    ensure_dirs,
)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "REPORTS_DIR",
    "RESULTS_DIR",
    "ensure_dirs",
    "get_logger",
]
