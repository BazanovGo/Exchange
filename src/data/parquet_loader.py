"""Parquet data source: one file per symbol inside a base directory."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pandas as pd

from src.data.base import BaseDataLoader
from src.data.registry import register_loader
from src.utils.paths import PROCESSED_DATA_DIR


@register_loader
class ParquetLoader(BaseDataLoader):
    """Loads ``{base_dir}/{symbol}.parquet`` (or an explicit ``path=`` per call)."""

    source_name: ClassVar[str] = "parquet"

    def __init__(self, base_dir: str | Path = PROCESSED_DATA_DIR, filename_template: str = "{symbol}.parquet") -> None:
        self.base_dir = Path(base_dir)
        self.filename_template = filename_template

    def _resolve_path(self, symbol: str, path: str | Path | None) -> Path:
        resolved = Path(path) if path is not None else self.base_dir / self.filename_template.format(symbol=symbol)
        if not resolved.exists():
            raise FileNotFoundError(f"Parquet file for {symbol!r} not found: {resolved}")
        return resolved

    def _load_symbol(self, symbol: str, *, start=None, end=None, timeframe: str = "1d", path: str | Path | None = None, **kwargs) -> pd.DataFrame:
        return pd.read_parquet(self._resolve_path(symbol, path), **kwargs)
