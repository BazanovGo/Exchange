"""Strategy interface, registry and reference library."""

from __future__ import annotations

from exchange.strategies.base import (
    Strategy,
    StrategyMeta,
    atr,
    available,
    build,
    catalog,
    ema,
    grid_size,
    iter_grid,
    realised_volatility,
    register,
    rolling_zscore,
    rsi,
    sma,
    strategy_class,
    true_range,
    valid_combinations,
    volatility_target_size,
)
from exchange.strategies.library import FAMILIES

__all__ = [
    "FAMILIES",
    "Strategy",
    "StrategyMeta",
    "atr",
    "available",
    "build",
    "catalog",
    "ema",
    "grid_size",
    "iter_grid",
    "realised_volatility",
    "register",
    "rolling_zscore",
    "rsi",
    "sma",
    "strategy_class",
    "true_range",
    "valid_combinations",
    "volatility_target_size",
]
