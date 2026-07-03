"""Backtesting layer: single entry point for running any strategy.

::

    from src.backtesting import run_backtest

    result = run_backtest("ma_crossover", data, params={"fast_window": 20})
    result.stats()
"""

from src.backtesting.config import BacktestConfig
from src.backtesting.engine import BacktestResult, run_backtest

__all__ = ["BacktestConfig", "BacktestResult", "run_backtest"]
