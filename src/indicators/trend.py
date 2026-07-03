"""Trend indicators."""

from __future__ import annotations

import pandas as pd
import vectorbt as vbt


def ma(close: pd.Series | pd.DataFrame, window: int, *, ewm: bool = False) -> pd.Series | pd.DataFrame:
    """Moving average (simple or exponential) via ``vbt.MA``."""
    return vbt.MA.run(close, window=window, ewm=ewm).ma


def sma(close: pd.Series | pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    """Simple moving average."""
    return ma(close, window, ewm=False)


def ema(close: pd.Series | pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    """Exponential moving average."""
    return ma(close, window, ewm=True)
