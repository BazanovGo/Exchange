"""Strategy layer.

Every strategy implements :class:`~src.strategies.base.BaseStrategy` and
returns :class:`~src.strategies.base.StrategySignals` (entries, exits,
short entries, short exits, params, description). Strategies register
themselves by name so they can be instantiated generically::

    from src.strategies import get_strategy

    strat = get_strategy("ema_cross", fast_window=20, slow_window=100)

Each strategy also carries a ``category`` (trend / mean_reversion /
momentum / hybrid) and a default optimization grid (``opt_grid``) so the
whole universe can be swept by :mod:`src.optimization` without
per-strategy code.
"""

# Importing the modules registers the built-in strategies.
from src.strategies import hybrid as _hybrid  # noqa: F401
from src.strategies import mean_reversion as _mean_reversion  # noqa: F401
from src.strategies import momentum as _momentum  # noqa: F401
from src.strategies import trend as _trend  # noqa: F401
from src.strategies.base import (
    BaseStrategy,
    StrategySignals,
    available_strategies,
    get_strategy,
    register_strategy,
    strategy_catalog,
    strategy_class,
)
from src.strategies.ma_crossover import MACrossoverStrategy
from src.strategies.rsi_reversion import RSIMeanReversionStrategy

__all__ = [
    "BaseStrategy",
    "StrategySignals",
    "register_strategy",
    "get_strategy",
    "strategy_class",
    "strategy_catalog",
    "available_strategies",
    "MACrossoverStrategy",
    "RSIMeanReversionStrategy",
]
