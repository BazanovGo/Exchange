"""CSV data source: one file per symbol inside a base directory."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pandas as pd

from src.data.base import BaseDataLoader
from src.data.registry import register_loader
from src.utils.paths import RAW_DATA_DIR


@register_loader
class CSVLoader(BaseDataLoader):
    """Loads ``{base_dir}/{symbol}.csv`` (or an explicit ``path=`` per call).

    Files must contain a datetime column/index and OHLCV columns in any
    casing; normalization is handled by the base class.
    """

    source_name: ClassVar[str] = "csv"

    def __init__(self, base_dir: str | Path = RAW_DATA_DIR, filename_template: str = "{symbol}.csv") -> None:
        self.base_dir = Path(base_dir)
        self.filename_template = filename_template

    def _resolve_path(self, symbol: str, path: str | Path | None) -> Path:
        resolved = Path(path) if path is not None else self.base_dir / self.filename_template.format(symbol=symbol)
        if not resolved.exists():
            raise FileNotFoundError(f"CSV file for {symbol!r} not found: {resolved}")
        return resolved

    def _load_symbol(self, symbol: str, *, start=None, end=None, timeframe: str = "1d", path: str | Path | None = None, **kwargs) -> pd.DataFrame:
        return pd.read_csv(self._resolve_path(symbol, path), **kwargs)
