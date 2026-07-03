import numpy as np
import pandas as pd
import pytest

from src.optimization import (
    ResearchConfig,
    composite_score,
    evaluate_strategy,
    research_universe,
    robust_score,
    score_combined,
    summarize_strategies,
    top_configurations,
)
from src.strategies import available_strategies, strategy_class


def test_all_strategies_have_research_contract():
    """Every registered strategy must expose description, params and an opt grid."""
    for name in available_strategies():
        cls = strategy_class(name)
        assert cls.description, f"{name}: empty description"
        assert cls.default_params, f"{name}: no default params"
        if name == "ma_crossover":  # legacy alias of sma_cross, excluded from sweeps
            continue
        assert cls.category in {"trend", "mean_reversion", "momentum", "hybrid"}, f"{name}: bad category"
        assert cls.opt_grid, f"{name}: no optimization grid"
        unknown = set(cls.opt_grid) - set(cls.default_params)
        assert not unknown, f"{name}: grid params {unknown} not in default_params"


def test_required_universe_registered():
    """The full required strategy universe is present."""
    required = {
        # trend
        "sma_cross", "ema_cross", "triple_ema", "supertrend", "donchian", "adx_trend", "psar", "ichimoku",
        # mean reversion
        "rsi_reversion", "bollinger_reversion", "cci_reversion", "williams_r", "stochastic",
        "vwap_reversion", "zscore_reversion",
        # momentum
        "roc_momentum", "momentum", "macd_momentum", "atr_breakout", "keltner_breakout", "volatility_expansion",
        # hybrid
        "ema_rsi", "ema_adx", "donchian_atr", "vwap_volume", "macd_trend_filter",
    }
    missing = required - set(available_strategies())
    assert not missing, f"missing strategies: {sorted(missing)}"


def test_composite_score_direction():
    table = pd.DataFrame(
        {
            "sharpe_ratio": [2.0, 0.5, -0.5],
            "sortino_ratio": [3.0, 0.7, -0.4],
            "calmar_ratio": [1.5, 0.3, -0.2],
            "cagr_pct": [30.0, 5.0, -10.0],
            "total_return_pct": [80.0, 10.0, -20.0],
            "max_drawdown_pct": [10.0, 25.0, 50.0],
            "profit_factor": [2.2, 1.1, 0.6],
            "win_rate_pct": [60.0, 45.0, 30.0],
            "expectancy": [50.0, 5.0, -20.0],
            "recovery_factor": [8.0, 0.4, -0.4],
            "total_trades": [50, 50, 50],
        }
    )
    score = composite_score(table)
    assert score.iloc[0] > score.iloc[1] > score.iloc[2]
    assert ((score >= 0) & (score <= 1)).all()


def test_composite_score_penalizes_few_trades():
    base = {
        "sharpe_ratio": [2.0, 2.0],
        "max_drawdown_pct": [10.0, 10.0],
        "profit_factor": [2.0, 2.0],
    }
    table = pd.DataFrame({**base, "total_trades": [50, 2]})
    score = composite_score(table)
    assert score.iloc[1] < score.iloc[0]


def test_robust_score_penalizes_inconsistency():
    is_score = pd.Series([0.9, 0.6])
    oos_score = pd.Series([0.1, 0.6])
    rs = robust_score(is_score, oos_score)
    assert rs.iloc[1] > rs.iloc[0]  # even 0.6/0.6 beats flashy-but-collapsing 0.9/0.1


def test_evaluate_strategy_is_oos(ohlcv):
    table = evaluate_strategy(
        "sma_cross", ohlcv,
        grid={"fast_window": [10, 20], "slow_window": [50, 100]},
    )
    assert len(table) == 4
    assert {"is_sharpe_ratio", "oos_sharpe_ratio", "is_total_trades", "oos_total_trades"} <= set(table.columns)
    assert table["is_total_trades"].sum() > 0
    assert table["oos_total_trades"].sum() > 0


def test_evaluate_strategy_skips_invalid_combos(ohlcv):
    table = evaluate_strategy(
        "sma_cross", ohlcv,
        grid={"fast_window": [10, 100], "slow_window": [50, 100]},
    )
    assert len(table) == 2  # (100,50) and (100,100) invalid


def test_research_universe_end_to_end(ohlcv):
    grids = {
        "sma_cross": {"fast_window": [10, 20], "slow_window": [50]},
        "rsi_reversion": {"window": [7, 14], "lower": [30.0]},
    }
    combined = research_universe(["sma_cross", "rsi_reversion"], ohlcv, grids=grids, show_progress=False)
    assert {"is_composite", "oos_composite", "robust_score"} <= set(combined.columns)
    assert combined["strategy"].nunique() == 2

    summary = summarize_strategies(combined)
    assert len(summary) == 2
    assert {"robust_score", "overfit_flags", "best_params"} <= set(summary.columns)
    assert summary["overfit_flags"].between(0, 3).all()

    top = top_configurations(combined, n=3, per_strategy=1)
    assert len(top) <= 3
    assert top["strategy"].is_unique
    assert top["robust_score"].is_monotonic_decreasing


def test_research_config_split_validation(ohlcv):
    with pytest.raises(ValueError, match="split"):
        evaluate_strategy(
            "sma_cross", ohlcv.iloc[:80],
            config=ResearchConfig(split=0.5),
            grid={"fast_window": [10], "slow_window": [50]},
        )


def test_score_combined_no_lookahead_columns(ohlcv):
    table = evaluate_strategy("sma_cross", ohlcv, grid={"fast_window": [10], "slow_window": [50]})
    scored = score_combined(table)
    row = scored.iloc[0]
    assert np.isfinite(row["robust_score"])
