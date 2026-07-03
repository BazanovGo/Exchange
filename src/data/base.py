"""Base abstractions for market data loading.

All loaders return :class:`MarketData` — a container of per-symbol OHLCV
frames with a canonical schema:

* ``pd.DatetimeIndex`` (sorted, unique, tz-naive by default)
* columns ``Open, High, Low, Close, Volume`` (extra columns are preserved)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd

OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")

# Case-insensitive aliases accepted from raw files / APIs.
_COLUMN_ALIASES: Mapping[str, str] = {
    "open": "Open",
    "o": "Open",
    "high": "High",
    "h": "High",
    "low": "Low",
    "l": "Low",
    "close": "Close",
    "c": "Close",
    "adj close": "Adj Close",
    "adj_close": "Adj Close",
    "adjclose": "Adj Close",
    "volume": "Volume",
    "v": "Volume",
    "vol": "Volume",
}

_DATETIME_CANDIDATES = ("date", "datetime", "timestamp", "time", "begin", "tradedate")


def normalize_ohlcv(df: pd.DataFrame, *, symbol: str = "") -> pd.DataFrame:
    """Coerce a raw frame to the canonical OHLCV schema.

    Handles: a datetime column instead of an index, arbitrary column casing,
    duplicated/unsorted timestamps, string-typed numeric columns.

    Raises ``ValueError`` if a ``Close`` column cannot be identified.
    """
    out = df.copy()

    # Flatten MultiIndex columns (e.g. yfinance with a single ticker).
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]

    # Promote a datetime-like column to the index if needed.
    if not isinstance(out.index, pd.DatetimeIndex):
        for col in out.columns:
            if str(col).strip().lower() in _DATETIME_CANDIDATES:
                out[col] = pd.to_datetime(out[col])
                out = out.set_index(col)
                break
        else:
            out.index = pd.to_datetime(out.index)
    out.index.name = "Date"

    # Normalize column names.
    renamed = {}
    for col in out.columns:
        key = str(col).strip().lower()
        renamed[col] = _COLUMN_ALIASES.get(key, str(col).strip().title() if key in {c.lower() for c in OHLCV_COLUMNS} else col)
    out = out.rename(columns=renamed)

    if "Close" not in out.columns:
        raise ValueError(
            f"Cannot identify a 'Close' column for {symbol or 'data'}; got columns {list(df.columns)}"
        )

    # Numeric coercion for known price/volume columns.
    for col in (*OHLCV_COLUMNS, "Adj Close"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out[~out.index.duplicated(keep="last")].sort_index()
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)

    # Keep canonical columns first, preserve any extras after them.
    ordered = [c for c in (*OHLCV_COLUMNS, "Adj Close") if c in out.columns]
    extras = [c for c in out.columns if c not in ordered]
    return out[ordered + extras]


@dataclass
class MarketData:
    """Container for one or more symbols of OHLCV data."""

    frames: dict[str, pd.DataFrame]
    source: str = ""
    timeframe: str = "1d"
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("MarketData requires at least one symbol")

    @property
    def symbols(self) -> list[str]:
        return list(self.frames)

    def get(self, symbol: str | None = None) -> pd.DataFrame:
        """Return the OHLCV frame for ``symbol`` (or the only symbol)."""
        if symbol is None:
            if len(self.frames) != 1:
                raise ValueError(f"symbol is required, data holds {self.symbols}")
            return next(iter(self.frames.values()))
        return self.frames[symbol]

    def _field(self, column: str) -> pd.DataFrame:
        wide = pd.concat({s: f[column] for s, f in self.frames.items()}, axis=1)
        wide.columns.name = "symbol"
        return wide

    @property
    def open(self) -> pd.DataFrame:
        return self._field("Open")

    @property
    def high(self) -> pd.DataFrame:
        return self._field("High")

    @property
    def low(self) -> pd.DataFrame:
        return self._field("Low")

    @property
    def close(self) -> pd.DataFrame:
        return self._field("Close")

    @property
    def volume(self) -> pd.DataFrame:
        return self._field("Volume")

    def slice(self, start=None, end=None) -> MarketData:
        """Return a copy restricted to ``[start, end]``."""
        frames = {s: f.loc[start:end] for s, f in self.frames.items()}
        return MarketData(frames=frames, source=self.source, timeframe=self.timeframe, meta=dict(self.meta))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        spans = {s: f"{f.index[0].date()}..{f.index[-1].date()} ({len(f)})" for s, f in self.frames.items()}
        return f"MarketData(source={self.source!r}, timeframe={self.timeframe!r}, {spans})"


class BaseDataLoader(ABC):
    """Interface every data source implements.

    Subclasses set ``source_name`` and implement :meth:`_load_symbol`.
    The public :meth:`load` handles symbol iteration, normalization and
    date slicing uniformly for all sources.
    """

    source_name: ClassVar[str] = ""

    @abstractmethod
    def _load_symbol(self, symbol: str, *, start=None, end=None, timeframe: str = "1d", **kwargs) -> pd.DataFrame:
        """Fetch a raw frame for a single symbol."""

    def load(
        self,
        symbols: str | Iterable[str],
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
        timeframe: str = "1d",
        **kwargs: Any,
    ) -> MarketData:
        """Load one or more symbols into a :class:`MarketData` container."""
        if isinstance(symbols, str):
            symbols = [symbols]
        frames: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            raw = self._load_symbol(symbol, start=start, end=end, timeframe=timeframe, **kwargs)
            df = normalize_ohlcv(raw, symbol=symbol).loc[start:end]
            if df.empty:
                raise ValueError(f"{self.source_name}: no data for {symbol} in [{start}, {end}]")
            frames[symbol] = df
        return MarketData(frames=frames, source=self.source_name, timeframe=timeframe)
