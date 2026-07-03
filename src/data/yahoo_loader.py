"""Yahoo Finance data source (via yfinance)."""

from __future__ import annotations

from typing import ClassVar

import pandas as pd

from src.data.base import BaseDataLoader
from src.data.registry import register_loader
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Yahoo interval codes for common timeframe spellings.
_TIMEFRAME_MAP = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "1d": "1d",
    "d": "1d",
    "day": "1d",
    "1w": "1wk",
    "1wk": "1wk",
    "1mo": "1mo",
}


@register_loader
class YahooLoader(BaseDataLoader):
    """Downloads OHLCV history from Yahoo Finance.

    ``auto_adjust=True`` (default) returns split/dividend-adjusted prices,
    which is what backtests should normally consume.
    """

    source_name: ClassVar[str] = "yahoo"

    def __init__(self, auto_adjust: bool = True) -> None:
        self.auto_adjust = auto_adjust

    def _load_symbol(self, symbol: str, *, start=None, end=None, timeframe: str = "1d", **kwargs) -> pd.DataFrame:
        import yfinance as yf  # deferred: keeps offline usage dependency-free

        interval = _TIMEFRAME_MAP.get(timeframe.lower())
        if interval is None:
            raise ValueError(f"Unsupported Yahoo timeframe {timeframe!r}; supported: {sorted(set(_TIMEFRAME_MAP))}")

        logger.info("Downloading %s from Yahoo Finance (%s, %s..%s)", symbol, interval, start, end)
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=self.auto_adjust,
            progress=False,
            **kwargs,
        )
        if df is None or df.empty:
            raise ValueError(f"Yahoo Finance returned no data for {symbol!r}")
        return df
