"""Volatility / band / volume-weighted indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt


def bollinger_bands(
    close: pd.Series | pd.DataFrame,
    window: int = 20,
    alpha: float = 2.0,
) -> tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
    """Bollinger Bands (middle, upper, lower) via ``vbt.BBANDS``."""
    ind = vbt.BBANDS.run(close, window=window, alpha=alpha)
    return ind.middle, ind.upper, ind.lower


def realized_volatility(
    close: pd.Series | pd.DataFrame,
    window: int = 21,
    *,
    periods_per_year: int = 252,
) -> pd.Series | pd.DataFrame:
    """Annualized rolling realized volatility of log returns."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(periods_per_year)


def keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    atr_window: int = 10,
    multiplier: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner channels (middle EMA, upper, lower)."""
    middle = vbt.MA.run(close, window=window, ewm=True).ma
    atr_ = vbt.ATR.run(high, low, close, window=atr_window, ewm=True).atr
    return middle, middle + multiplier * atr_, middle - multiplier * atr_


def rolling_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Rolling volume-weighted average price over typical price.

    A rolling window stands in for session VWAP on timeframes (e.g. daily
    bars) that have no natural intraday session anchor.
    """
    tp = (high + low + close) / 3.0
    pv = (tp * volume).rolling(window).sum()
    v = volume.rolling(window).sum()
    return pv / v


def zscore(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling z-score of price against its own moving average."""
    mean = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    return (close - mean) / std
