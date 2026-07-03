"""Volatility indicators."""

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
