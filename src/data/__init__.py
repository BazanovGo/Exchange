"""Unified market data loading.

Usage::

    from src.data import load_data

    md = load_data("yahoo", "AAPL", start="2020-01-01")
    md = load_data("csv", "SAMPLE", base_dir="data/raw")
    close = md.close  # wide DataFrame, one column per symbol

New sources (MOEX, Polygon, ...) are added by subclassing
:class:`~src.data.base.BaseDataLoader` and decorating with
:func:`~src.data.registry.register_loader` — no call sites change.
"""

# Importing the modules registers the built-in loaders.
from src.data import csv_loader as _csv_loader  # noqa: F401
from src.data import parquet_loader as _parquet_loader  # noqa: F401
from src.data import yahoo_loader as _yahoo_loader  # noqa: F401
from src.data.base import OHLCV_COLUMNS, BaseDataLoader, MarketData
from src.data.registry import available_sources, get_loader, load_data, register_loader

__all__ = [
    "BaseDataLoader",
    "MarketData",
    "OHLCV_COLUMNS",
    "load_data",
    "get_loader",
    "register_loader",
    "available_sources",
]
