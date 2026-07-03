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

# (metric name in vbt stats index, short name) — resolved defensively because
# stats labels can differ slightly between vectorbt versions.
_METRIC_MAP = [
    ("Total Return [%]", "total_return_pct"),
    ("Benchmark Return [%]", "benchmark_return_pct"),
    ("Max Drawdown [%]", "max_drawdown_pct"),
    ("Sharpe Ratio", "sharpe_ratio"),
    ("Sortino Ratio", "sortino_ratio"),
    ("Calmar Ratio", "calmar_ratio"),
    ("Win Rate [%]", "win_rate_pct"),
    ("Profit Factor", "profit_factor"),
    ("Total Trades", "total_trades"),
    ("Expectancy", "expectancy"),
]


def key_metrics(result: BacktestResult) -> pd.Series:
    """Compact, uniformly named metric set for a backtest result."""
    pf = result.portfolio
    stats = pf.stats()

    metrics: dict[str, float] = {}
    for stats_key, name in _METRIC_MAP:
        if stats_key in stats.index:
            value = stats.loc[stats_key]
            metrics[name] = float(value) if pd.notna(value) else np.nan

    metrics.setdefault("total_return_pct", float(pf.total_return()) * 100)
    metrics.setdefault("max_drawdown_pct", float(pf.max_drawdown()) * 100)
    metrics["annualized_return_pct"] = float(pf.annualized_return()) * 100
    metrics["annualized_volatility_pct"] = float(pf.annualized_volatility()) * 100
    metrics["final_value"] = float(pf.final_value())

    return pd.Series(metrics, name=f"{result.strategy_name}:{result.symbol or ''}")


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
