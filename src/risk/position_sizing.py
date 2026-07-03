"""Position sizing rules.

Each function returns a value (or series) compatible with the ``size``
argument of ``vbt.Portfolio.from_signals``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fixed_fraction_size(equity: float, price: float, fraction: float = 0.1) -> float:
    """Number of units so the position consumes ``fraction`` of equity."""
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if price <= 0:
        raise ValueError("price must be positive")
    return equity * fraction / price


def volatility_target_size(
    close: pd.Series,
    *,
    target_annual_vol: float = 0.15,
    vol_window: int = 21,
    periods_per_year: int = 252,
    max_leverage: float = 1.0,
) -> pd.Series:
    """Position size as a fraction of equity targeting a constant volatility.

    Returns a series in [0, max_leverage] suitable for
    ``size_type='percent'`` sizing: allocation shrinks when realized
    volatility rises above the target and grows when it falls below.
    """
    log_ret = np.log(close / close.shift(1))
    realized = log_ret.rolling(vol_window).std() * np.sqrt(periods_per_year)
    weight = (target_annual_vol / realized).clip(upper=max_leverage)
    return weight.fillna(0.0)
