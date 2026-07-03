"""Indicator layer: thin, typed wrappers over vectorbt indicators.

Strategies should consume indicators through this package rather than
calling vectorbt directly, so the underlying implementation can be swapped
(e.g. for custom numba kernels) without touching strategy code.
"""

from src.indicators.momentum import atr, cci, macd, momentum, roc, rsi, stochastic, williams_r
from src.indicators.trend import adx, donchian, ema, ichimoku, ma, parabolic_sar, sma, supertrend
from src.indicators.volatility import (
    bollinger_bands,
    keltner_channels,
    realized_volatility,
    rolling_vwap,
    zscore,
)

__all__ = [
    # trend
    "ma",
    "sma",
    "ema",
    "donchian",
    "adx",
    "supertrend",
    "parabolic_sar",
    "ichimoku",
    # momentum / oscillators
    "rsi",
    "macd",
    "atr",
    "roc",
    "momentum",
    "stochastic",
    "williams_r",
    "cci",
    # volatility / bands / volume
    "bollinger_bands",
    "keltner_channels",
    "realized_volatility",
    "rolling_vwap",
    "zscore",
]
