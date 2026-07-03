"""Momentum / oscillator indicators."""

from __future__ import annotations

import pandas as pd
import vectorbt as vbt


def rsi(close: pd.Series | pd.DataFrame, window: int = 14, *, ewm: bool = True) -> pd.Series | pd.DataFrame:
    """Relative Strength Index via ``vbt.RSI``."""
    return vbt.RSI.run(close, window=window, ewm=ewm).rsi


def macd(
    close: pd.Series | pd.DataFrame,
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
) -> tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
    """MACD line, signal line and histogram via ``vbt.MACD``."""
    ind = vbt.MACD.run(close, fast_window=fast_window, slow_window=slow_window, signal_window=signal_window)
    return ind.macd, ind.signal, ind.hist


def atr(
    high: pd.Series | pd.DataFrame,
    low: pd.Series | pd.DataFrame,
    close: pd.Series | pd.DataFrame,
    window: int = 14,
    *,
    ewm: bool = True,
) -> pd.Series | pd.DataFrame:
    """Average True Range via ``vbt.ATR``."""
    return vbt.ATR.run(high, low, close, window=window, ewm=ewm).atr


def roc(close: pd.Series, window: int = 12) -> pd.Series:
    """Rate of Change: percent change over ``window`` bars."""
    return close.pct_change(window) * 100.0


def momentum(close: pd.Series, window: int = 10) -> pd.Series:
    """Raw momentum: price difference over ``window`` bars."""
    return close.diff(window)


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_window: int = 14,
    d_window: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Stochastic oscillator ``(%K, %D)`` via ``vbt.STOCH``."""
    ind = vbt.STOCH.run(high, low, close, k_window=k_window, d_window=d_window)
    return ind.percent_k, ind.percent_d


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Williams %R in [-100, 0]."""
    hh = high.rolling(window).max()
    ll = low.rolling(window).min()
    return -100.0 * (hh - close) / (hh - ll)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3.0
    sma_tp = tp.rolling(window).mean()
    mad = tp.rolling(window).apply(lambda x: (abs(x - x.mean())).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad)
