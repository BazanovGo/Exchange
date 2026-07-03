"""Standard result-analysis template.

For any :class:`~src.backtesting.engine.BacktestResult` this module
produces the same set of artifacts:

* equity curve
* drawdown curve
* trade list
* key performance metrics

so every strategy is evaluated identically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.engine import BacktestResult
from src.utils.paths import RESULTS_DIR


def _f(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value


def portfolio_metrics(pf, *, name: str = "") -> pd.Series:
    """Compact, uniformly named metric set for a vectorbt Portfolio.

    Computed via direct portfolio/trade methods rather than ``pf.stats()``
    — an order of magnitude faster, which matters in parameter sweeps.
    ``max_drawdown_pct`` is a positive magnitude (lower is better);
    ``cagr_pct`` duplicates ``annualized_return_pct`` under the
    conventional name.
    """
    trades = pf.trades
    metrics: dict[str, float] = {
        "total_return_pct": _f(pf.total_return()) * 100,
        "max_drawdown_pct": abs(_f(pf.max_drawdown())) * 100,
        "sharpe_ratio": _f(pf.sharpe_ratio()),
        "sortino_ratio": _f(pf.sortino_ratio()),
        "calmar_ratio": _f(pf.calmar_ratio()),
        "annualized_return_pct": _f(pf.annualized_return()) * 100,
        "annualized_volatility_pct": _f(pf.annualized_volatility()) * 100,
        "final_value": _f(pf.final_value()),
        "total_trades": _f(trades.count()),
        "win_rate_pct": _f(trades.win_rate()) * 100,
        "profit_factor": _f(trades.profit_factor()),
        "expectancy": _f(trades.expectancy()),
    }
    metrics["cagr_pct"] = metrics["annualized_return_pct"]
    dd = metrics["max_drawdown_pct"]
    metrics["recovery_factor"] = metrics["total_return_pct"] / dd if dd > 0 else np.nan

    return pd.Series(metrics, name=name)


def key_metrics(result: BacktestResult) -> pd.Series:
    """Compact, uniformly named metric set for a backtest result."""
    return portfolio_metrics(result.portfolio, name=f"{result.strategy_name}:{result.symbol or ''}")


@dataclass
class PerformanceReport:
    """Bundled analysis artifacts for one backtest run."""

    equity: pd.Series | pd.DataFrame
    drawdown: pd.Series | pd.DataFrame
    trades: pd.DataFrame
    metrics: pd.Series
    strategy_name: str
    params: dict
    symbol: str | None

    @classmethod
    def from_result(cls, result: BacktestResult) -> PerformanceReport:
        return cls(
            equity=result.equity(),
            drawdown=result.drawdown(),
            trades=result.trades(),
            metrics=key_metrics(result),
            strategy_name=result.strategy_name,
            params=result.params,
            symbol=result.symbol,
        )

    def summary(self) -> pd.Series:
        return self.metrics

    def save(self, name: str | None = None, directory: str | Path = RESULTS_DIR) -> Path:
        """Persist the report (metrics, trades, equity/drawdown) to disk."""
        directory = Path(directory)
        name = name or f"{self.strategy_name}_{self.symbol or 'data'}"
        target = directory / name
        target.mkdir(parents=True, exist_ok=True)

        self.metrics.to_frame("value").to_csv(target / "metrics.csv")
        self.trades.to_csv(target / "trades.csv", index=False)
        pd.DataFrame({"equity": self.equity, "drawdown": self.drawdown}).to_csv(target / "curves.csv")
        (target / "params.json").write_text(json.dumps(self.params, indent=2, default=str))
        return target
