"""Backtesting layer: single entry point for running any strategy.

::

    from src.backtesting import run_backtest

    result = run_backtest("ma_crossover", data, params={"fast_window": 20})
    result.stats()
"""

from src.backtesting.config import BacktestConfig
from src.backtesting.engine import BacktestResult, run_backtest
from src.backtesting.position_management import (
    EXIT_REASONS,
    ExitRules,
    exit_reason_counts,
    managed_portfolio,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "run_backtest",
    "ExitRules",
    "EXIT_REASONS",
    "managed_portfolio",
    "exit_reason_counts",
]
