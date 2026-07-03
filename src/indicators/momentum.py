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
