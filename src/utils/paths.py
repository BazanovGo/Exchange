"""Canonical project paths.

Every module resolves data/report/result locations through these constants so
the layout can be changed in one place.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = PROJECT_ROOT / "results"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def ensure_dirs() -> None:
    """Create all project data directories if they do not exist."""
    for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
