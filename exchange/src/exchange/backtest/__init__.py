"""Backtest engine and performance metrics."""

from __future__ import annotations

from exchange.backtest.engine import (
    BacktestConfig,
    BacktestResult,
    InstrumentSpec,
    Trade,
    run_backtest,
)
from exchange.backtest.metrics import (
    PerformanceMetrics,
    annualised_volatility,
    cagr,
    calmar_ratio,
    drawdown_series,
    expected_shortfall,
    kurtosis,
    max_drawdown,
    max_drawdown_duration,
    metrics_frame,
    performance_metrics,
    sharpe_ratio,
    skewness,
    sortino_ratio,
    tail_ratio,
    total_return,
    trade_stats,
    ulcer_index,
    value_at_risk,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "InstrumentSpec",
    "PerformanceMetrics",
    "Trade",
    "annualised_volatility",
    "cagr",
    "calmar_ratio",
    "drawdown_series",
    "expected_shortfall",
    "kurtosis",
    "max_drawdown",
    "max_drawdown_duration",
    "metrics_frame",
    "performance_metrics",
    "run_backtest",
    "sharpe_ratio",
    "skewness",
    "sortino_ratio",
    "tail_ratio",
    "total_return",
    "trade_stats",
    "ulcer_index",
    "value_at_risk",
]
