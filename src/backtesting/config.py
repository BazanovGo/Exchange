"""Backtest configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


def effective_freq(index: pd.Index) -> pd.Timedelta:
    """Average calendar time per bar: ``(last - first) / (bars - 1)``.

    vectorbt annualizes by ``year_freq / freq`` (365 days by default). With
    this ``freq`` a year holds as many bars as the market actually trades —
    about 252 daily bars of a stock or 3 600-5 000 hourly MOEX candles — rather than
    a 24/7 clock (8 760 hours), and drawdown/trade durations come out close
    to calendar time.
    """
    if len(index) < 2:
        raise ValueError("need at least two bars to infer the bar frequency")
    return (index[-1] - index[0]) / (len(index) - 1)


@dataclass(frozen=True)
class BacktestConfig:
    """Execution assumptions passed to ``vbt.Portfolio.from_signals``.

    Frozen so a config instance can be shared safely between runs
    (e.g. across a parameter grid).
    """

    init_cash: float = 100_000.0
    fees: float = 0.001          # proportional commission per trade
    slippage: float = 0.0005     # proportional slippage per trade
    size: float | None = None    # None -> invest all available cash
    size_type: str = "amount"    # used only when size is not None
    freq: str = "auto"           # bar frequency for annualization; "auto" -> effective_freq of the data
    sl_stop: float | None = None  # stop-loss, fraction of entry price
    tp_stop: float | None = None  # take-profit, fraction of entry price
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def resolve_freq(self, index: pd.Index | None = None) -> str | pd.Timedelta:
        """``freq`` as given, or measured from ``index`` when it is ``"auto"``."""
        if self.freq != "auto":
            return self.freq
        if index is None:
            raise ValueError('freq="auto" needs the price index to measure the bar frequency')
        return effective_freq(index)

    def to_portfolio_kwargs(self, index: pd.Index | None = None) -> dict[str, Any]:
        """Materialize kwargs for ``vbt.Portfolio.from_signals`` on prices with ``index``."""
        kwargs: dict[str, Any] = {
            "init_cash": self.init_cash,
            "fees": self.fees,
            "slippage": self.slippage,
            "freq": self.resolve_freq(index),
        }
        if self.size is not None:
            kwargs["size"] = self.size
            kwargs["size_type"] = self.size_type
        if self.sl_stop is not None:
            kwargs["sl_stop"] = self.sl_stop
        if self.tp_stop is not None:
            kwargs["tp_stop"] = self.tp_stop
        kwargs.update(self.extra_kwargs)
        return kwargs
