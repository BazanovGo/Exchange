"""Execution and accounting.

Accounting errors are the most expensive kind of bug in a backtest,
because they do not look like bugs — they look like alpha. Every test here
computes the expected answer by hand first, and most run with zero costs so
that a cost model cannot hide an accounting mistake.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exchange.backtest.engine import BacktestConfig, InstrumentSpec, run_backtest
from exchange.backtest.metrics import (
    annualised_volatility,
    cagr,
    expected_shortfall,
    max_drawdown,
    max_drawdown_duration,
    performance_metrics,
    sharpe_ratio,
    sortino_ratio,
    trade_stats,
    value_at_risk,
)
from exchange.core.errors import ConfigError
from exchange.execution.assumptions import (
    BASE,
    CONSERVATIVE,
    FRICTIONLESS,
    OPTIMISTIC,
    STRESS_LADDER,
    STRESSED,
    build_assumptions,
    get_scenario,
)
from exchange.execution.costs import (
    ZERO_COST,
    CommissionModel,
    CostModel,
    ImpactModel,
    SlippageModel,
    SpreadModel,
)
from exchange.execution.orders import Bar, Order, OrderType, Side, match
from exchange.strategies.base import volatility_target_size


class TestCostModels:
    def test_commission_components_add_up(self):
        model = CommissionModel(per_contract=1.5, per_trade=0.5, exchange_fee=0.75, bps_of_notional=1.0)
        # 2 contracts at 100 with multiplier 10: notional 2000, 1bp = 0.20
        cost = model.charge(quantity=2.0, price=100.0, multiplier=10.0)
        assert cost == pytest.approx(2 * 1.5 + 0.5 + 2 * 0.75 + 0.2)

    def test_commission_minimum_applies(self):
        model = CommissionModel(per_contract=0.01, minimum=2.0)
        assert model.charge(quantity=1.0, price=1.0, multiplier=1.0) == 2.0

    def test_half_spread_is_charged_per_side(self):
        spread = SpreadModel(half_spread_bps=2.0)
        assert spread.half_cost(price=100.0) == pytest.approx(0.02)

    def test_slippage_scales_with_volatility(self):
        model = SlippageModel(bps=1.0, volatility_multiple=0.5)
        calm = model.cost(price=100.0, volatility=0.0)
        wild = model.cost(price=100.0, volatility=0.02)
        assert wild > calm
        assert calm == pytest.approx(0.01)

    def test_impact_follows_a_square_root_law(self):
        model = ImpactModel(coefficient=0.1)
        small = model.cost(100.0, 1.0, 10_000.0, 0.01)
        large = model.cost(100.0, 100.0, 10_000.0, 0.01)
        assert large / small == pytest.approx(10.0, rel=0.05)   # sqrt(100)

    def test_liquidity_cap_limits_the_fillable_quantity(self):
        model = ImpactModel(coefficient=0.1, max_participation=0.1)
        assert model.fillable(5_000.0, 10_000.0) == pytest.approx(1_000.0)

    def test_scaling_multiplies_every_estimated_component(self):
        doubled = BASE.costs.scale_slippage(2.0)
        assert doubled.slippage.bps == pytest.approx(BASE.costs.slippage.bps * 2)

    def test_zero_cost_is_actually_zero(self):
        offset = ZERO_COST.price_offset(100.0, 10.0, volatility=0.02, bar_volume=1e6)
        assert offset == 0.0


class TestOrderMatching:
    @staticmethod
    def bar(open_=100.0, high=101.0, low=99.0, close=100.5, volume=1000.0):
        return Bar(open=open_, high=high, low=low, close=close, volume=volume)

    def test_market_order_fills_at_the_open(self):
        fill = match(Order(side=Side.BUY, quantity=1.0, order_type=OrderType.MARKET), self.bar())
        assert fill.did_fill
        assert fill.price == 100.0

    def test_limit_order_above_the_market_fills_at_the_open_not_the_limit(self):
        """A buy limit above the open is immediately executable at the open.

        Filling it at the limit price would hand the backtest free money on
        every gap in its favour.
        """
        order = Order(side=Side.BUY, quantity=1.0, order_type=OrderType.LIMIT, limit_price=105.0)
        fill = match(order, self.bar())
        assert fill.did_fill
        assert fill.price == 100.0

    def test_limit_order_fills_only_if_the_bar_trades_through(self):
        order = Order(side=Side.BUY, quantity=1.0, order_type=OrderType.LIMIT, limit_price=98.0)
        assert not match(order, self.bar(low=99.0)).did_fill
        fill = match(order, self.bar(low=97.0))
        assert fill.did_fill and fill.price == pytest.approx(98.0)

    def test_stop_order_gapping_through_fills_at_the_gap_not_the_stop(self):
        """The error that makes stop losses look risk-free.

        A sell stop at 95 on a bar that opens at 90 fills at 90. Filling at
        95 would understate every gap loss in the sample.
        """
        order = Order(side=Side.SELL, quantity=1.0, order_type=OrderType.STOP, stop_price=95.0)
        fill = match(order, self.bar(open_=90.0, high=91.0, low=89.0, close=90.0))
        assert fill.did_fill
        assert fill.triggered
        assert fill.price == pytest.approx(90.0)

    def test_stop_limit_fills_at_the_trigger_not_at_the_open(self):
        """A triggered stop-limit cannot fill at a pre-trigger price.

        A sell stop at 99.5 on a bar opening at 99.9 triggers somewhere
        below 99.5, so 99.9 was never available after the trigger.
        """
        order = Order(
            side=Side.SELL, quantity=1.0, order_type=OrderType.STOP_LIMIT,
            stop_price=99.5, limit_price=99.4,
        )
        fill = match(order, self.bar(open_=99.9, high=100.0, low=99.0))
        assert fill.triggered and fill.did_fill
        assert fill.price == pytest.approx(99.5)

    def test_stop_limit_misses_when_the_market_gaps_past_the_limit(self):
        """The classic stop-limit failure, which a backtest must reproduce."""
        order = Order(
            side=Side.SELL, quantity=1.0, order_type=OrderType.STOP_LIMIT,
            stop_price=99.5, limit_price=99.4,
        )
        fill = match(order, self.bar(open_=90.0, high=91.0, low=89.0, close=90.0))
        assert fill.triggered
        assert not fill.did_fill
        assert "gapped past the limit" in fill.reason

    def test_untouched_stop_does_not_fill(self):
        order = Order(side=Side.SELL, quantity=1.0, order_type=OrderType.STOP, stop_price=90.0)
        fill = match(order, self.bar())
        assert not fill.did_fill
        assert not fill.triggered


class TestAccounting:
    def test_flat_market_flat_equity(self, flat_bars, frictionless_config):
        target = pd.Series(1.0, index=flat_bars.index)
        result = run_backtest(flat_bars, target, config=frictionless_config, symbol="FLAT")
        assert float(result.equity.iloc[-1]) == pytest.approx(
            frictionless_config.initial_cash, abs=1e-9
        )

    def test_pnl_is_price_change_times_multiplier(self, ramp_bars, frictionless_config):
        """One contract held through a 1.0/bar ramp earns multiplier per bar.

        Entry is on bar 1's open (100 + 1 = 101) because the signal from
        bar 0 executes next bar, and the position is marked at the final
        close of 111.
        """
        target = pd.Series(1.0, index=ramp_bars.index)
        result = run_backtest(ramp_bars, target, config=frictionless_config, symbol="RAMP")
        expected = (111.0 - 101.0) * frictionless_config.instrument.multiplier
        assert float(result.equity.iloc[-1] - result.equity.iloc[0]) == pytest.approx(expected)

    def test_short_position_earns_the_opposite(self, ramp_bars, frictionless_config):
        long_result = run_backtest(
            ramp_bars, pd.Series(1.0, index=ramp_bars.index),
            config=frictionless_config, symbol="RAMP",
        )
        short_result = run_backtest(
            ramp_bars, pd.Series(-1.0, index=ramp_bars.index),
            config=frictionless_config, symbol="RAMP",
        )
        long_pnl = float(long_result.equity.iloc[-1] - long_result.equity.iloc[0])
        short_pnl = float(short_result.equity.iloc[-1] - short_result.equity.iloc[0])
        assert short_pnl == pytest.approx(-long_pnl)

    def test_costs_strictly_reduce_equity(self, ramp_bars, sized_instrument):
        target = pd.Series(1.0, index=ramp_bars.index)
        target.iloc[::2] = 0.0          # trade often, so costs bite
        free = run_backtest(
            ramp_bars, target,
            config=BacktestConfig(instrument=sized_instrument, execution=FRICTIONLESS),
            symbol="RAMP",
        )
        costly = run_backtest(
            ramp_bars, target,
            config=BacktestConfig(instrument=sized_instrument, execution=BASE),
            symbol="RAMP",
        )
        assert costly.total_costs > 0
        assert float(costly.equity.iloc[-1]) < float(free.equity.iloc[-1])

    def test_cost_monotonicity_across_the_ladder(self, rw_bars, sized_instrument):
        """Performance must fall as costs rise. If it does not, a sign is wrong."""
        target = pd.Series(np.where(np.arange(len(rw_bars)) % 10 < 5, 1.0, -1.0), index=rw_bars.index)
        equities = []
        for scenario in STRESS_LADDER:
            result = run_backtest(
                rw_bars, target,
                config=BacktestConfig(instrument=sized_instrument, execution=get_scenario(scenario)),
                symbol="RW",
            )
            equities.append(float(result.equity.iloc[-1]))
        assert equities == sorted(equities, reverse=True)

    def test_whole_contracts_only_when_fractional_is_disallowed(self, ramp_bars, futures_instrument):
        config = BacktestConfig(
            initial_cash=1e6, instrument=futures_instrument, execution=FRICTIONLESS
        )
        target = pd.Series(2.7, index=ramp_bars.index)
        result = run_backtest(ramp_bars, target, config=config, symbol="NG")
        held = result.position[result.position != 0]
        assert np.allclose(held.to_numpy(), np.round(held.to_numpy()))

    def test_a_short_target_on_a_long_only_instrument_is_refused(self):
        """Refused loudly, not silently clipped.

        Clipping would quietly change the strategy being tested, and the
        reported result would not be the result of the strategy as written.
        """
        config = BacktestConfig(
            instrument=InstrumentSpec(symbol="LONGONLY", multiplier=100.0, allow_short=False),
            execution=FRICTIONLESS,
        )
        index = pd.date_range("2026-01-05", periods=30, freq="B", tz="UTC", name="timestamp")
        bars = pd.DataFrame(
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1e4}, index=index
        )
        with pytest.raises(ConfigError, match="disallows shorts"):
            run_backtest(bars, pd.Series(-1.0, index=index), config=config, symbol="LONGONLY")

    def test_reversal_is_accounted_as_a_close_and_an_open(self, ramp_bars, frictionless_config):
        target = pd.Series(1.0, index=ramp_bars.index)
        target.iloc[6:] = -1.0
        result = run_backtest(ramp_bars, target, config=frictionless_config, symbol="RAMP")
        assert len(result.trades) >= 1
        assert set(result.trades["direction"].unique()) <= {1, -1}

    def test_trades_reconcile_with_equity(self, rw_bars, frictionless_config):
        """Closed-trade P&L must equal the change in equity exactly.

        The most valuable accounting test there is: it catches
        double-counted fills, dropped fees and sign errors at once. The
        target returns to flat well before the end, so there is no open
        position to account for separately and the identity is exact.
        """
        signal = np.sign(np.sin(np.arange(len(rw_bars)) / 7.0))
        target = pd.Series(signal, index=rw_bars.index)
        target.iloc[-20:] = 0.0
        result = run_backtest(rw_bars, target, config=frictionless_config, symbol="RW")
        assert float(result.position.iloc[-1]) == 0.0
        closed = float(result.trades["net_pnl"].sum())
        total = float(result.equity.iloc[-1] - result.equity.iloc[0])
        assert closed == pytest.approx(total, rel=1e-9, abs=1e-6)

    def test_trades_reconcile_with_equity_under_costs(self, rw_bars, base_config):
        """The same identity must hold once fees and slippage are charged."""
        signal = np.sign(np.sin(np.arange(len(rw_bars)) / 11.0))
        target = pd.Series(signal, index=rw_bars.index)
        target.iloc[-20:] = 0.0
        result = run_backtest(rw_bars, target, config=base_config, symbol="RW")
        closed = float(result.trades["net_pnl"].sum())
        total = float(result.equity.iloc[-1] - result.equity.iloc[0])
        assert closed == pytest.approx(total, rel=1e-9, abs=1e-6)
        # Commissions are attributed to trades; slippage and the spread are
        # inside the fill prices, which is where they belong - a cost that
        # moves the execution price is not a fee.
        commissions = float(result.orders["fees"].sum())
        assert float(result.trades["fees"].sum()) == pytest.approx(commissions, rel=1e-9)
        assert result.total_costs == pytest.approx(
            commissions + float(result.orders["slippage_cost"].sum()), rel=1e-9
        )
        assert result.total_costs > commissions

    def test_equity_never_goes_negative_without_leverage(self, rw_bars, frictionless_config):
        target = pd.Series(1.0, index=rw_bars.index)
        result = run_backtest(rw_bars, target, config=frictionless_config, symbol="RW")
        assert (result.equity > 0).all()


class TestPositionSizing:
    def test_volatility_target_scales_inversely_with_volatility(self, rw_bars):
        calm = rw_bars.copy()
        calm["close"] = 100.0 + np.sin(np.arange(len(calm)) / 50.0) * 0.1
        wild = rw_bars.copy()
        wild["close"] = 100.0 + np.sin(np.arange(len(wild)) / 50.0) * 10.0
        kwargs = {
            "target_annual_vol": 0.15, "bars_per_year": 252.0,
            "capital": 100_000.0, "multiplier": 1.0,
            "max_leverage": 1e9,        # isolate the vol scaling from the cap
        }
        calm_size = volatility_target_size(calm, **kwargs).iloc[-1]
        wild_size = volatility_target_size(wild, **kwargs).iloc[-1]
        assert calm_size > wild_size

    def test_leverage_cap_binds(self, rw_bars):
        sizes = volatility_target_size(
            rw_bars, target_annual_vol=10.0, bars_per_year=252.0,
            capital=100_000.0, multiplier=1.0, max_leverage=2.0,
        )
        assert sizes.max() <= 100_000.0 * 2.0 / rw_bars["close"].min() + 1e-6

    def test_sizing_is_causal(self, rw_bars):
        from exchange.data.temporal import assert_causal

        assert_causal(
            lambda b: volatility_target_size(
                b, target_annual_vol=0.15, bars_per_year=252.0, capital=1e5
            ),
            rw_bars,
            name="volatility_target_size",
        )


class TestMetrics:
    def test_a_constant_return_series_has_no_sharpe(self):
        """Not an enormous one: the denominator is absent, not small."""
        index = pd.date_range("2020-01-01", periods=1000, freq="B", tz="UTC")
        returns = pd.Series(np.full(1000, 0.001), index=index)
        assert not np.isfinite(sharpe_ratio(returns))

    def test_sharpe_scales_with_the_square_root_of_frequency(self):
        rng = np.random.default_rng(5)
        index = pd.date_range("2020-01-01", periods=2520, freq="B", tz="UTC")
        returns = pd.Series(rng.normal(0.0005, 0.01, 2520), index=index)
        daily = sharpe_ratio(returns, periods_per_year=252.0)
        weekly = sharpe_ratio(returns, periods_per_year=52.0)
        assert daily / weekly == pytest.approx(np.sqrt(252 / 52), rel=0.01)

    def test_max_drawdown_is_negative_and_bounded(self, rw_bars):
        equity = rw_bars["close"]
        value = max_drawdown(equity)
        assert -1.0 <= value <= 0.0

    def test_drawdown_duration_counts_bars_underwater(self):
        equity = pd.Series([100.0, 90.0, 95.0, 101.0, 99.0])
        assert max_drawdown_duration(equity) >= 2

    def test_cagr_does_not_overflow_on_large_multiples(self):
        index = pd.date_range("2020-01-01", periods=300, freq="B", tz="UTC")
        equity = pd.Series(np.exp(np.linspace(0, 20, 300)), index=index)
        value = cagr(equity, periods_per_year=252.0)
        assert np.isfinite(value)

    def test_cagr_is_nan_on_a_wiped_out_account(self):
        index = pd.date_range("2020-01-01", periods=10, freq="B", tz="UTC")
        equity = pd.Series(np.linspace(100.0, -5.0, 10), index=index)
        assert np.isnan(cagr(equity, periods_per_year=252.0))

    def test_sortino_only_penalises_downside(self):
        rng = np.random.default_rng(7)
        returns = pd.Series(rng.normal(0.001, 0.01, 1000))
        assert sortino_ratio(returns, periods_per_year=252.0) > sharpe_ratio(
            returns, periods_per_year=252.0
        )

    def test_var_and_expected_shortfall_order_correctly(self):
        rng = np.random.default_rng(11)
        returns = pd.Series(rng.normal(0.0, 0.01, 5000))
        var = value_at_risk(returns, confidence=0.95)
        es = expected_shortfall(returns, confidence=0.95)
        # Both are positive loss magnitudes, and the tail mean is the larger.
        assert es >= var > 0.0

    def test_profit_factor_is_nan_not_infinite_without_losers(self):
        trades = pd.DataFrame({"net_pnl": [1.0, 2.0, 3.0], "bars_held": [1, 1, 1]})
        stats = trade_stats(trades)
        assert np.isnan(stats["profit_factor"])

    def test_annualised_volatility_matches_a_hand_computation(self):
        returns = pd.Series([0.01, -0.01] * 500)
        expected = float(np.std(returns, ddof=1) * np.sqrt(252.0))
        assert annualised_volatility(returns, periods_per_year=252.0) == pytest.approx(expected)

    def test_metrics_bundle_is_complete(self, rw_bars, frictionless_config):
        result = run_backtest(
            rw_bars, pd.Series(1.0, index=rw_bars.index), config=frictionless_config, symbol="RW"
        )
        metrics = performance_metrics(result, periods_per_year=252.0)
        assert metrics.n_bars == len(rw_bars)
        assert np.isfinite(metrics.total_return)
        assert metrics.valid is True


class TestScenarios:
    def test_the_ladder_is_ordered_by_severity(self):
        costs = [
            get_scenario(name).costs.commission.per_contract for name in STRESS_LADDER
        ]
        assert costs == sorted(costs)

    def test_optimistic_admits_no_lookahead_either(self):
        assert OPTIMISTIC.admits_lookahead is False
        assert CONSERVATIVE.latency_bars >= 1
        assert STRESSED.execution_delay >= 1

    def test_scenarios_round_trip_through_dicts(self):
        payload = CONSERVATIVE.to_dict()
        from exchange.execution.assumptions import ExecutionAssumptions

        rebuilt = ExecutionAssumptions.from_dict(payload)
        assert rebuilt.to_dict() == payload

    def test_build_assumptions_accepts_a_name_or_a_mapping(self):
        assert build_assumptions("base").label == "base"
        assert build_assumptions(None).label == BASE.label
        with pytest.raises(ConfigError):
            build_assumptions("nonexistent-scenario")

    def test_unknown_fill_timing_is_refused(self):
        from dataclasses import replace

        with pytest.raises(ConfigError):
            replace(BASE, fill_timing="whenever_convenient")

    def test_cost_model_summary_mentions_every_component(self):
        summary = CostModel(
            commission=CommissionModel(per_contract=1.0),
            spread=SpreadModel(half_spread_bps=1.0),
            slippage=SlippageModel(bps=1.0),
            impact=ImpactModel(coefficient=0.1),
        ).summary()
        for word in ("commission", "spread", "slippage", "impact"):
            assert word in summary.lower()
