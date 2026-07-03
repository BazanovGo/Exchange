"""Strategy layer.

Every strategy implements :class:`~src.strategies.base.BaseStrategy` and
returns :class:`~src.strategies.base.StrategySignals` (entries, exits,
short entries, short exits, params, description). Strategies register
themselves by name so they can be instantiated generically::

    from src.strategies import get_strategy

    strat = get_strategy("ma_crossover", fast_window=20, slow_window=100)
"""

from src.strategies.base import (
    BaseStrategy,
    StrategySignals,
    available_strategies,
    get_strategy,
    register_strategy,
)

# Importing the modules registers the built-in strategies.
from src.strategies.ma_crossover import MACrossoverStrategy
from src.strategies.rsi_reversion import RSIMeanReversionStrategy

__all__ = [
    "BaseStrategy",
    "StrategySignals",
    "register_strategy",
    "get_strategy",
    "available_strategies",
    "MACrossoverStrategy",
    "RSIMeanReversionStrategy",
]
