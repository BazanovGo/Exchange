"""Registry mapping source names to loader classes.

Adding a new source (MOEX, Polygon, ...)::

    @register_loader
    class MoexLoader(BaseDataLoader):
        source_name = "moex"

        def _load_symbol(self, symbol, *, start=None, end=None, timeframe="1d", **kwargs):
            ...  # fetch and return a raw DataFrame

After that ``load_data("moex", "SBER")`` works everywhere.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from src.data.base import BaseDataLoader, MarketData

_LOADERS: dict[str, type[BaseDataLoader]] = {}


def register_loader(cls: type[BaseDataLoader]) -> type[BaseDataLoader]:
    """Class decorator: register a loader under its ``source_name``."""
    if not cls.source_name:
        raise ValueError(f"{cls.__name__} must define a non-empty source_name")
    key = cls.source_name.lower()
    _LOADERS[key] = cls
    return cls


def available_sources() -> list[str]:
    return sorted(_LOADERS)


def get_loader(source: str, **loader_kwargs: Any) -> BaseDataLoader:
    """Instantiate the loader registered under ``source``."""
    key = source.lower()
    if key not in _LOADERS:
        raise KeyError(f"Unknown data source {source!r}; available: {available_sources()}")
    return _LOADERS[key](**loader_kwargs)


def load_data(
    source: str,
    symbols: str | Iterable[str],
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    timeframe: str = "1d",
    loader_kwargs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> MarketData:
    """Single entry point for loading market data from any registered source."""
    loader = get_loader(source, **(loader_kwargs or {}))
    return loader.load(symbols, start=start, end=end, timeframe=timeframe, **kwargs)
