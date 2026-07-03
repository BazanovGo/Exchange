"""Backtest engine: one function runs any strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import vectorbt as vbt

from src.backtesting.config import BacktestConfig
from src.data.base import MarketData
from src.strategies.base import BaseStrategy, StrategySignals, get_strategy
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Everything produced by a single backtest run."""

    portfolio: vbt.Portfolio
    strategy_name: str
    params: dict[str, Any]
    signals: StrategySignals
    config: BacktestConfig
    symbol: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def stats(self) -> pd.Series:
        """Full vectorbt stats table."""
        return self.portfolio.stats()

    def equity(self) -> pd.Series | pd.DataFrame:
        """Portfolio value over time."""
        return self.portfolio.value()

    def returns(self) -> pd.Series | pd.DataFrame:
        return self.portfolio.returns()

    def drawdown(self) -> pd.Series | pd.DataFrame:
        """Drawdown series (negative fractions of the running peak)."""
        return self.portfolio.drawdown()

    def trades(self) -> pd.DataFrame:
        """Human-readable trade list."""
        return self.portfolio.trades.records_readable

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"BacktestResult(strategy={self.strategy_name!r}, symbol={self.symbol!r}, "
            f"total_return={float(self.portfolio.total_return()) * 100:.2f}%)"
        )


def run_backtest(
    strategy: BaseStrategy | str,
    data: MarketData | pd.DataFrame,
    *,
    params: dict[str, Any] | None = None,
    config: BacktestConfig | None = None,
    symbol: str | None = None,
    price_col: str = "Close",
) -> BacktestResult:
    """Run any strategy against any data with one call.

    Parameters
    ----------
    strategy:
        A :class:`BaseStrategy` instance or a registered strategy name
        (in which case ``params`` are used to instantiate it).
    data:
        A :class:`MarketData` container or a canonical OHLCV ``DataFrame``.
    config:
        Execution assumptions; defaults to :class:`BacktestConfig` defaults.
    symbol:
        Which symbol of a multi-symbol ``MarketData`` to test.
    price_col:
        Column used as the execution price.
    """
    if isinstance(strategy, str):
        strategy = get_strategy(strategy, **(params or {}))
    elif params:
        raise ValueError("params are only accepted when strategy is given by name")
    config = config or BacktestConfig()

    ohlcv = data.get(symbol) if isinstance(data, MarketData) else data
    if symbol is None and isinstance(data, MarketData):
        symbol = data.symbols[0]

    sig = strategy.signals(ohlcv)
    logger.info(
        "Backtesting %s on %s: %s bars, %s",
        strategy.name, symbol or "data", len(ohlcv), dict(sig.stats()),
    )

    portfolio = vbt.Portfolio.from_signals(
        close=ohlcv[price_col],
        entries=sig.entries,
        exits=sig.exits,
        short_entries=sig.short_entries,
        short_exits=sig.short_exits,
        **config.to_portfolio_kwargs(),
    )

    return BacktestResult(
        portfolio=portfolio,
        strategy_name=strategy.name,
        params=dict(strategy.params),
        signals=sig,
        config=config,
        symbol=symbol,
    )
