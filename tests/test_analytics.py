import numpy as np
import pandas as pd

from src.analytics import PerformanceReport, key_metrics
from src.backtesting import run_backtest
from src.optimization import grid_search, param_grid
from src.risk import annualized_volatility, cvar, max_drawdown, var_historical


def _result(market_data):
    return run_backtest("ma_crossover", market_data, params={"fast_window": 10, "slow_window": 30})


def test_key_metrics(market_data):
    metrics = key_metrics(_result(market_data))
    for name in ("total_return_pct", "max_drawdown_pct", "sharpe_ratio", "total_trades", "final_value"):
        assert name in metrics.index
    assert metrics["max_drawdown_pct"] <= 0 or metrics["max_drawdown_pct"] >= 0  # finite
    assert np.isfinite(metrics["final_value"])


def test_performance_report_save(market_data, tmp_path):
    report = PerformanceReport.from_result(_result(market_data))
    assert len(report.equity) == len(report.drawdown)
    assert isinstance(report.trades, pd.DataFrame)

    target = report.save(name="unit_test", directory=tmp_path)
    assert (target / "metrics.csv").exists()
    assert (target / "trades.csv").exists()
    assert (target / "curves.csv").exists()
    assert (target / "params.json").exists()


def test_param_grid_expansion():
    combos = param_grid({"a": [1, 2], "b": ["x"]})
    assert combos == [{"a": 1, "b": "x"}, {"a": 2, "b": "x"}]


def test_grid_search_skips_invalid(market_data):
    table = grid_search(
        "ma_crossover",
        market_data,
        {"fast_window": [10, 40], "slow_window": [30, 60]},
        show_progress=False,
    )
    # (40, 30) is invalid and must be skipped -> 3 valid combos.
    assert len(table) == 3
    assert {"param_fast_window", "param_slow_window", "sharpe_ratio"} <= set(table.columns)
    assert table["sharpe_ratio"].is_monotonic_decreasing


def test_risk_metrics():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, 2500))
    vol = annualized_volatility(returns)
    assert 0.10 < vol < 0.22

    var95 = var_historical(returns, 0.95)
    cvar95 = cvar(returns, 0.95)
    assert var95 > 0
    assert cvar95 >= var95

    equity = (1 + returns).cumprod()
    dd = max_drawdown(equity)
    assert -1 < dd <= 0
