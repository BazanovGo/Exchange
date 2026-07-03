"""Indicator layer: thin, typed wrappers over vectorbt indicators.

Strategies should consume indicators through this package rather than
calling vectorbt directly, so the underlying implementation can be swapped
(e.g. for custom numba kernels) without touching strategy code.
"""

from src.indicators.momentum import atr, macd, rsi
from src.indicators.trend import ema, ma, sma
from src.indicators.volatility import bollinger_bands, realized_volatility

__all__ = [
    "ma",
    "sma",
    "ema",
    "rsi",
    "macd",
    "atr",
    "bollinger_bands",
    "realized_volatility",
]
