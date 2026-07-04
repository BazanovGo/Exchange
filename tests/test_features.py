import numpy as np
import pandas as pd

from src.analytics import (
    REGIME_ORDER,
    efficiency_ratio,
    higher_tf_trend,
    market_features,
    redundant_pairs,
    trade_feature_dataset,
)
from src.optimization import (
    default_filters,
    filter_research,
    filter_scoreboard,
    regime_performance,
    summarize_filters,
)

STRATS = {"ema_cross": {"fast_window": 10, "slow_window": 50}, "supertrend": {}}


def test_market_features_schema(ohlcv):
    f = market_features(ohlcv)
    expected = {
        "volatility", "atr_pct", "adx", "rel_volume", "hour", "day_of_week",
        "dist_ma", "bb_width", "ema_slope", "efficiency_ratio", "htf_trend", "regime",
    }
    assert expected <= set(f.columns)
    assert len(f) == len(ohlcv)
    # After warm-up every numeric feature must be populated.
    tail = f.iloc[300:]
    for col in expected - {"regime"}:
        assert tail[col].notna().all(), col
    assert (tail["day_of_week"] == tail.index.dayofweek).all()


def test_features_have_no_lookahead(ohlcv):
    """Features at bar t must not change when the future is truncated."""
    full = market_features(ohlcv)
    cut = market_features(ohlcv.iloc[:1500])
    check = full.iloc[300:1500].drop(columns=["regime"])
    pd.testing.assert_frame_equal(check, cut.iloc[300:].drop(columns=["regime"]), rtol=1e-10)


def test_efficiency_ratio_bounds(ohlcv):
    er = efficiency_ratio(ohlcv["Close"], 20).dropna()
    assert ((er >= 0) & (er <= 1)).all()
    trending = pd.Series(np.arange(100, 200, dtype=float))
    assert efficiency_ratio(trending, 20).dropna().min() > 0.99


def test_higher_tf_trend_values(ohlcv):
    trend = higher_tf_trend(ohlcv["Close"])
    assert set(np.unique(trend.dropna())) <= {-1.0, 0.0, 1.0}


def test_regime_labels(ohlcv):
    f = market_features(ohlcv)
    seen = set(f["regime"].dropna().unique())
    assert seen <= set(REGIME_ORDER)
    assert len(seen) >= 2  # synthetic data should visit several regimes


def test_trade_feature_dataset(ohlcv):
    trades = trade_feature_dataset(STRATS, ohlcv)
    assert {"strategy", "ret_pct", "win", "duration_days", "adx", "regime"} <= set(trades.columns)
    assert trades["strategy"].nunique() == 2
    assert trades["win"].dtype == bool
    assert (trades["duration_days"] >= 0).all()
    # Features joined at entry must match the source frame.
    f = market_features(ohlcv)
    row = trades.iloc[0]
    assert row["adx"] == f.loc[row["entry_date"], "adx"]


def test_redundant_pairs():
    x = pd.Series(np.random.default_rng(0).normal(size=500))
    frame = pd.DataFrame({"a": x, "b": x * 2 + 0.01, "c": np.random.default_rng(1).normal(size=500)})
    pairs = redundant_pairs(frame, threshold=0.8)
    assert len(pairs) == 1
    assert {pairs.iloc[0]["feature_a"], pairs.iloc[0]["feature_b"]} == {"a", "b"}


def test_filter_research_end_to_end(ohlcv):
    f = market_features(ohlcv)
    filters = {"htf_up": default_filters()["htf_up"], "adx_trending": default_filters()["adx_trending"]}
    sweep = filter_research(STRATS, ohlcv, f, filters, show_progress=False)
    assert len(sweep) == 6  # (baseline + 2 filters) x 2 strategies
    assert (sweep["entries_kept"] <= sweep["entries_total"]).all()

    summary = summarize_filters(sweep)
    assert len(summary) == 4
    assert summary["verdict"].isin(
        ["real improvement", "false improvement", "oos-only (lucky)", "no effect / worse", "too few trades"]
    ).all()

    board = filter_scoreboard(summary)
    assert board["strategies"].sum() == len(summary)


def test_regime_performance(ohlcv):
    trades = trade_feature_dataset(STRATS, ohlcv)
    table = regime_performance(trades)
    assert {"strategy", "regime", "trades", "mean_ret", "win_rate"} <= set(table.columns)
    assert (table["win_rate"].dropna().between(0, 1)).all()
