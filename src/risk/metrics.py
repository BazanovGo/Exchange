"""Risk metrics computed from a return series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualized standard deviation of periodic returns."""
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value-at-Risk (positive number = loss magnitude)."""
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    return float(-np.quantile(returns.dropna(), 1 - confidence))


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR / expected shortfall beyond the VaR threshold."""
    clean = returns.dropna()
    threshold = -var_historical(clean, confidence)
    tail = clean[clean <= threshold]
    if tail.empty:
        return var_historical(clean, confidence)
    return float(-tail.mean())


def max_drawdown(equity: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of an equity curve (negative fraction)."""
    running_max = equity.cummax()
    return float((equity / running_max - 1.0).min())
