import numpy as np
import pandas as pd
import pytest

from src.backtesting import BacktestConfig, ExitRules, managed_portfolio
from src.backtesting.position_management import EXIT_REASONS, managed_orders
from src.optimization import (
    exit_research,
    mechanic_scoreboard,
    mechanic_universe,
    summarize_mechanics,
    sweep_exit_rules,
)
from src.strategies import get_strategy

NO_FRICTION = BacktestConfig(fees=0.0, slippage=0.0)


def make_ohlcv(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build a tiny OHLCV frame from (open, high, low, close) tuples."""
    idx = pd.date_range("2024-01-01", periods=len(bars), freq="D")
    df = pd.DataFrame(bars, columns=["Open", "High", "Low", "Close"], index=idx)
    df["Volume"] = 1_000.0
    return df


def entry_at(ohlcv: pd.DataFrame, *positions: int) -> pd.Series:
    s = pd.Series(False, index=ohlcv.index)
    s.iloc[list(positions)] = True
    return s


def no_signals(ohlcv: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=ohlcv.index)


def reasons_of(reason: pd.Series) -> list[str]:
    return reason[reason > 0].map(EXIT_REASONS).tolist()


def test_stop_loss_intrabar_price():
    # Entry at close=100; two bars later low touches 94 -> SL 5% fills at 95.
    ohlcv = make_ohlcv([
        (100, 101, 99, 100),
        (100, 102, 98, 101),
        (100, 100, 94, 96),
        (96, 97, 95, 96),
    ])
    target, price, reason = managed_orders(
        ohlcv, entry_at(ohlcv, 0), no_signals(ohlcv), ExitRules(sl_stop=0.05)
    )
    assert reasons_of(reason) == ["stop_loss"]
    assert price.iloc[2] == pytest.approx(95.0)
    assert target.iloc[2] == 0.0


def test_stop_loss_gap_fills_at_open():
    ohlcv = make_ohlcv([
        (100, 101, 99, 100),
        (90, 92, 88, 91),  # gaps through the 5% stop
    ])
    _, price, reason = managed_orders(
        ohlcv, entry_at(ohlcv, 0), no_signals(ohlcv), ExitRules(sl_stop=0.05)
    )
    assert reasons_of(reason) == ["stop_loss"]
    assert price.iloc[1] == pytest.approx(90.0)  # open, not the 95 level


def test_take_profit_price():
    ohlcv = make_ohlcv([
        (100, 101, 99, 100),
        (101, 112, 100, 108),
    ])
    _, price, reason = managed_orders(
        ohlcv, entry_at(ohlcv, 0), no_signals(ohlcv), ExitRules(tp_stop=0.10)
    )
    assert reasons_of(reason) == ["take_profit"]
    assert price.iloc[1] == pytest.approx(110.0)


def test_stop_beats_take_profit_when_both_touched():
    # Wide bar hits both the 5% stop and the 10% target -> conservative: stop.
    ohlcv = make_ohlcv([
        (100, 101, 99, 100),
        (100, 112, 94, 100),
    ])
    _, price, reason = managed_orders(
        ohlcv, entry_at(ohlcv, 0), no_signals(ohlcv), ExitRules(sl_stop=0.05, tp_stop=0.10)
    )
    assert reasons_of(reason) == ["stop_loss"]
    assert price.iloc[1] == pytest.approx(95.0)


def test_trailing_stop_uses_prior_peak():
    ohlcv = make_ohlcv([
        (100, 101, 99, 100),
        (101, 120, 100, 118),   # peak 120 (recorded after this bar)
        (118, 119, 107, 108),   # trail 10% from 120 -> 108
    ])
    _, price, reason = managed_orders(
        ohlcv, entry_at(ohlcv, 0), no_signals(ohlcv), ExitRules(trail_stop=0.10)
    )
    assert reasons_of(reason) == ["trailing_stop"]
    assert price.iloc[2] == pytest.approx(108.0)


def test_break_even_protects_entry():
    ohlcv = make_ohlcv([
        (100, 101, 99, 100),
        (101, 106, 100, 105),   # +5% -> BE armed
        (105, 106, 98, 99),     # falls back through entry -> exit at 100
    ])
    _, price, reason = managed_orders(
        ohlcv, entry_at(ohlcv, 0), no_signals(ohlcv),
        ExitRules(break_even_trigger=0.05, break_even_offset=0.0),
    )
    assert reasons_of(reason) == ["break_even"]
    assert price.iloc[2] == pytest.approx(100.0)


def test_break_even_not_armed_without_trigger():
    ohlcv = make_ohlcv([
        (100, 101, 99, 100),
        (101, 103, 100, 102),   # only +3%, trigger is 5%
        (102, 103, 98, 99),
    ])
    _, _, reason = managed_orders(
        ohlcv, entry_at(ohlcv, 0), no_signals(ohlcv), ExitRules(break_even_trigger=0.05)
    )
    assert reasons_of(reason) == []


def test_time_stop_exits_at_close():
    ohlcv = make_ohlcv([(100, 101, 99, 100)] * 6)
    _, price, reason = managed_orders(
        ohlcv, entry_at(ohlcv, 0), no_signals(ohlcv), ExitRules(time_stop=3)
    )
    assert reasons_of(reason) == ["time_stop"]
    assert reason.iloc[3] > 0  # exactly 3 bars after entry
    assert price.iloc[3] == pytest.approx(100.0)


def test_partial_then_remainder_managed():
    ohlcv = make_ohlcv([
        (100, 101, 99, 100),
        (101, 109, 100, 107),   # +8% -> partial at 105
        (107, 122, 106, 121),   # +20% -> full TP at 120
    ])
    target, price, reason = managed_orders(
        ohlcv, entry_at(ohlcv, 0), no_signals(ohlcv),
        ExitRules(partial_tp=0.05, partial_fraction=0.5, tp_stop=0.20),
    )
    assert reasons_of(reason) == ["partial_tp", "take_profit"]
    assert target.iloc[1] == pytest.approx(0.5)
    assert price.iloc[1] == pytest.approx(105.0)
    assert target.iloc[2] == 0.0
    assert price.iloc[2] == pytest.approx(120.0)


def test_signal_exit_and_reentry():
    ohlcv = make_ohlcv([(100, 101, 99, 100)] * 5)
    entries = entry_at(ohlcv, 0, 3)
    exits = pd.Series([False, True, False, False, True], index=ohlcv.index)
    target, _, reason = managed_orders(ohlcv, entries, exits, ExitRules())
    assert reasons_of(reason) == ["signal", "signal"]
    assert list(target.dropna()) == [1.0, 0.0, 1.0, 0.0]


def test_baseline_matches_run_backtest(ohlcv):
    """Signal-exit-only managed portfolio ~= plain from_signals backtest."""
    from src.backtesting import run_backtest

    strategy = get_strategy("ema_cross", fast_window=10, slow_window=50)
    sig = strategy.signals(ohlcv)
    plain = run_backtest(strategy, ohlcv, config=NO_FRICTION)
    managed, _ = managed_portfolio(ohlcv, sig.entries, sig.exits, ExitRules(), NO_FRICTION)
    assert managed.total_return() == pytest.approx(float(plain.portfolio.total_return()), rel=0.02)


def test_exit_rules_validation():
    with pytest.raises(ValueError, match="sl_stop"):
        ExitRules(sl_stop=-0.05)
    with pytest.raises(ValueError, match="time_stop"):
        ExitRules(time_stop=0)
    with pytest.raises(ValueError, match="partial_tp"):
        ExitRules(partial_tp=0.2, tp_stop=0.1)


def test_sweep_and_summary(ohlcv):
    strategy = get_strategy("ema_cross", fast_window=10, slow_window=50)
    sig = strategy.signals(ohlcv)
    universe = {
        "baseline": [ExitRules()],
        "stop_loss": [ExitRules(label="sl=0.05", sl_stop=0.05), ExitRules(label="sl=0.1", sl_stop=0.1)],
        "time_stop": [ExitRules(label="time=10", time_stop=10)],
    }
    sweep = sweep_exit_rules(ohlcv, sig.entries, sig.exits, universe, strategy_name="ema_cross")
    assert len(sweep) == 4
    assert {"is_sharpe_ratio", "oos_sharpe_ratio", "mechanic"} <= set(sweep.columns)

    summary = summarize_mechanics(sweep)
    assert set(summary["mechanic"]) == {"stop_loss", "time_stop"}
    assert summary["verdict"].isin(
        ["real improvement", "false improvement", "oos-only (lucky)", "no effect / worse"]
    ).all()
    assert {"d_oos_cagr_pct", "d_oos_max_drawdown_pct", "d_is_sharpe_ratio"} <= set(summary.columns)

    board = mechanic_scoreboard(summary)
    assert board["strategies"].sum() == len(summary)


def test_exit_research_multi_strategy(ohlcv):
    universe = {"baseline": [ExitRules()], "stop_loss": [ExitRules(label="sl=0.08", sl_stop=0.08)]}
    table = exit_research(
        {"ema_cross": {"fast_window": 10, "slow_window": 50}, "supertrend": {}},
        ohlcv, universe, show_progress=False,
    )
    assert table["strategy"].nunique() == 2


def test_mechanic_universe_covers_required():
    universe = mechanic_universe()
    required = {"baseline", "stop_loss", "take_profit", "trailing_stop", "break_even",
                "time_stop", "partial_tp", "combo_sl_tp", "combo_sl_trail_be"}
    assert required <= set(universe)
    assert all(len(v) >= 1 for v in universe.values())


def test_partial_tp_reduces_position_value():
    """After a partial exit the invested value must drop by ~fraction."""
    ohlcv = make_ohlcv([
        (100, 101, 99, 100),
        (101, 109, 100, 106),   # partial at 105
        (106, 107, 105, 106),
        (106, 107, 105, 106),
    ])
    pf, reason = managed_portfolio(
        ohlcv, entry_at(ohlcv, 0), no_signals(ohlcv),
        ExitRules(partial_tp=0.05, partial_fraction=0.5), NO_FRICTION,
    )
    assets = pf.asset_value()
    assert reasons_of(reason) == ["partial_tp"]
    assert assets.iloc[3] < assets.iloc[0] * 0.7  # roughly half remains
    assert np.isfinite(pf.total_return())
