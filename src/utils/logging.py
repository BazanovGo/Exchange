"""Project-wide logging configuration."""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT))
    root = logging.getLogger("src")
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under the project root logger."""
    _configure_root()
    if not name.startswith("src"):
        name = f"src.{name}"
    return logging.getLogger(name)
