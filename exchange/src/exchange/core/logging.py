"""Logging setup.

One helper so every module logs through the same handler and the CLI can
raise or lower verbosity in one place.
"""

from __future__ import annotations

import logging
import os
import sys

_CONFIGURED = False
_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def configure_logging(level: int | str = "INFO", *, stream=None) -> None:
    """Install a single stderr handler on the ``exchange`` logger."""
    global _CONFIGURED
    root = logging.getLogger("exchange")
    for handler in list(root.handlers):
        root.removeHandler(handler)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger, configuring the package logger once."""
    if not _CONFIGURED:
        configure_logging(os.environ.get("EXCHANGE_LOG_LEVEL", "INFO"))
    if not name.startswith("exchange"):
        name = f"exchange.{name}"
    return logging.getLogger(name)
