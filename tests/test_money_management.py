import numpy as np
import pytest

from src.optimization import WalkForwardConfig, walk_forward, walk_forward_summary, window_bounds
from src.risk import (
    TradeStats,
    kelly_continuous,
    kelly_discrete,
    kelly_estimate_stability,
    kelly_robustness,
    position_fraction,
    simulate_trade_sequences,
    sizing_menu,
)


def coin_flip_returns(p_win=0.6, win=0.10, loss=0.05, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return np.where(rng.random(n) < p_win, win, -loss)


def test_kelly_discrete_textbook():
    # p=0.6, b=2 -> f* = 0.6 - 0.4/2 = 0.4
    r = coin_flip_returns(p_win=0.6, win=0.10, loss=0.05, n=200_000)
    stats = TradeStats.from_returns(r)
    assert stats.win_prob == pytest.approx(0.6, abs=0.01)
    assert stats.payoff_ratio == pytest.approx(2.0, abs=0.01)
    assert kelly_discrete(stats) == pytest.approx(0.4, abs=0.02)


def test_kelly_continuous_moments():
    r = coin_flip_returns(n=200_000)
    stats = TradeStats.from_returns(r)
    # mean = 0.6*0.1 - 0.4*0.05 = 0.04; var ~= E[x^2]-mean^2 = 0.007 - 0.0016
    assert stats.expectancy == pytest.approx(0.04, abs=0.002)
    expected = stats.expectancy / stats.variance
    assert kelly_continuous(stats) == pytest.approx(expected)
    assert kelly_continuous(stats) > 0


def test_negative_edge_kelly_is_zero():
    r = coin_flip_returns(p_win=0.4, win=0.05, loss=0.05, n=50_000)
    stats = TradeStats.from_returns(r)
    assert kelly_discrete(stats) == 0.0
    assert kelly_continuous(stats) == 0.0


def test_position_fraction_caps():
    r = coin_flip_returns(n=100_000)
    stats = TradeStats.from_returns(r)

    full = position_fraction(stats, variant="classical", max_fraction=10.0, risk_per_trade=10.0)
    half = position_fraction(stats, variant="half", max_fraction=10.0, risk_per_trade=10.0)
    quarter = position_fraction(stats, variant="quarter", max_fraction=10.0, risk_per_trade=10.0)
    assert half.fraction == pytest.approx(full.fraction / 2)
    assert quarter.fraction == pytest.approx(full.fraction / 4)
    assert full.binding == "kelly"

    capped = position_fraction(stats, variant="classical", max_fraction=0.10, risk_per_trade=10.0)
    assert capped.fraction == pytest.approx(0.10)
    assert capped.binding == "max_fraction"

    # tail loss is 0.05 -> risk 1% allows at most 0.2 of capital
    risk_bound = position_fraction(stats, variant="classical", max_fraction=10.0, risk_per_trade=0.01)
    assert risk_bound.fraction == pytest.approx(0.01 / stats.tail_loss)
    assert risk_bound.binding == "risk_limit"


def test_sizing_menu_variants():
    menu = sizing_menu(TradeStats.from_returns(coin_flip_returns(n=50_000)))
    assert list(menu.index) == ["classical", "half", "quarter"]
    assert menu["fraction"].is_monotonic_decreasing or menu["binding"].nunique() > 1


def test_custom_variant_and_validation():
    stats = TradeStats.from_returns(coin_flip_returns(n=10_000))
    custom = position_fraction(stats, variant=0.1, max_fraction=10.0, risk_per_trade=10.0)
    assert custom.fraction == pytest.approx(custom.kelly_full * 0.1)
    with pytest.raises(ValueError, match="variant"):
        position_fraction(stats, variant="third")
    with pytest.raises(ValueError, match="multiplier"):
        position_fraction(stats, variant=1.5)


def test_monte_carlo_no_losses_no_ruin():
    result = simulate_trade_sequences([0.05, 0.03, 0.08, 0.02], fraction=1.0, n_sims=500)
    assert result.ruin_prob == 0.0
    assert (result.final_returns > 0).all()
    assert (result.max_drawdowns == 0).all()


def test_monte_carlo_overbetting_ruins():
    r = coin_flip_returns(p_win=0.55, win=0.5, loss=0.5, n=500)
    conservative = simulate_trade_sequences(r, fraction=0.1, n_sims=2000, n_trades=100)
    reckless = simulate_trade_sequences(r, fraction=1.5, n_sims=2000, n_trades=100)
    assert reckless.ruin_prob > 0.9
    assert conservative.ruin_prob < reckless.ruin_prob
    assert reckless.max_drawdowns.mean() > conservative.max_drawdowns.mean()


def test_monte_carlo_reproducible_and_bounded():
    r = coin_flip_returns(n=300)
    a = simulate_trade_sequences(r, fraction=0.5, n_sims=100, seed=7)
    b = simulate_trade_sequences(r, fraction=0.5, n_sims=100, seed=7)
    np.testing.assert_array_equal(a.final_returns, b.final_returns)
    assert ((a.max_drawdowns >= 0) & (a.max_drawdowns <= 1)).all()
    assert (a.final_returns >= -1.0).all()  # cannot lose more than everything


def test_kelly_robustness_shape():
    r = coin_flip_returns(n=500)
    table = kelly_robustness(r, fractions=[0.25, 0.5, 1.0, 2.0], n_sims=500, n_trades=100)
    assert list(table.index) == [0.25, 0.5, 1.0, 2.0]
    assert table["ruin_prob"].iloc[-1] >= table["ruin_prob"].iloc[0]
    assert {"final_ret_median", "maxdd_median", "prob_loss"} <= set(table.columns)


def test_kelly_estimate_stability_interval():
    r = coin_flip_returns(n=200)
    ci = kelly_estimate_stability(r, n_boot=300)
    assert ci["kelly_p5"] <= ci["kelly_median"] <= ci["kelly_p95"]


def test_trade_stats_validation():
    with pytest.raises(ValueError, match="at least 2"):
        TradeStats.from_returns([0.1])


def test_window_bounds():
    cfg = WalkForwardConfig(train_bars=500, test_bars=100, step_bars=100)
    bounds = window_bounds(1000, cfg)
    assert bounds[0] == (0, 500, 600)
    assert bounds[-1][2] <= 1000
    assert len(bounds) == 5
    with pytest.raises(ValueError, match="too short"):
        window_bounds(400, cfg)


def test_walk_forward_end_to_end(ohlcv):
    cfg = WalkForwardConfig(train_bars=400, test_bars=150, step_bars=150)
    windows = walk_forward(
        "ema_cross", ohlcv,
        grid={"fast_window": [10, 20], "slow_window": [50]},
        cfg=cfg,
    )
    assert len(windows) == len(window_bounds(len(ohlcv), cfg))
    assert {"is_sharpe_ratio", "oos_sharpe_ratio", "params"} <= set(windows.columns)
    # test segments must be sequential and non-overlapping
    assert windows["test_start"].is_monotonic_increasing

    summary = walk_forward_summary(windows)
    assert 0 <= summary["pct_profitable_windows"] <= 1
    assert summary["n_windows"] == len(windows)
    assert 0 <= summary["param_stability"] <= 1
