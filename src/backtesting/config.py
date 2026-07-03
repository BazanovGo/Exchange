"""Backtest configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
    freq: str = "1D"             # bar frequency for annualization
    sl_stop: float | None = None  # stop-loss, fraction of entry price
    tp_stop: float | None = None  # take-profit, fraction of entry price
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def to_portfolio_kwargs(self) -> dict[str, Any]:
        """Materialize kwargs for ``vbt.Portfolio.from_signals``."""
        kwargs: dict[str, Any] = {
            "init_cash": self.init_cash,
            "fees": self.fees,
            "slippage": self.slippage,
            "freq": self.freq,
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
