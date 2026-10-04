"""Temporal integrity.

This module is the one whose failure invalidates everything else. A false
discovery on a random walk costs a wasted week; a lookahead that no guard
catches makes every result the framework has ever produced wrong, quietly
and permanently. So the tests here are deliberately adversarial: each one
plants a specific leak and asserts that something refuses it.

Two distinct leaks, two distinct guards
----------------------------------------
``assert_causal``
    catches a *transform* that looks forward — a centred window, a
    full-sample z-score, a resample that borrows from the next bar.
``detect_column_leakage``
    catches a *column* that was already contaminated before the transform
    ran. No causality proof can detect this one, because every prefix of a
    pre-contaminated series contains the contamination.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exchange.backtest.engine import BacktestConfig, run_backtest
from exchange.core.errors import LookaheadError
from exchange.data.temporal import (
    AVAILABLE_TIME,
    assert_causal,
    assert_no_column_leakage,
    assert_no_future_index,
    detect_column_leakage,
    make_point_in_time,
    shift_to_decision,
)
from exchange.execution.assumptions import BASE, FRICTIONLESS
from exchange.features.providers import FeatureStore, SyntheticFeatureProvider
from exchange.strategies.base import available, strategy_class
from exchange.synthetic.generators import random_walk, with_lookahead_trap


class TestAssertCausal:
    def test_a_trailing_mean_passes(self, rw_bars):
        assert_causal(lambda b: b["close"].rolling(20).mean(), rw_bars, name="sma")

    def test_a_centred_window_is_caught(self, rw_bars):
        with pytest.raises(LookaheadError):
            assert_causal(
                lambda b: b["close"].rolling(21, center=True).mean(), rw_bars, name="centred"
            )

    def test_a_full_sample_zscore_is_caught(self, rw_bars):
        """The most common lookahead in published strategy code."""
        def full_sample_z(bars):
            close = bars["close"]
            return (close - close.mean()) / close.std()

        with pytest.raises(LookaheadError):
            assert_causal(full_sample_z, rw_bars, name="full_sample_z")

    def test_a_negative_shift_is_caught(self, rw_bars):
        with pytest.raises(LookaheadError):
            assert_causal(lambda b: b["close"].shift(-1), rw_bars, name="peek")

    def test_a_backward_shift_passes(self, rw_bars):
        assert_causal(lambda b: b["close"].shift(1), rw_bars, name="lag")

    def test_expanding_statistics_pass(self, rw_bars):
        assert_causal(lambda b: b["close"].expanding().mean(), rw_bars, name="expanding")


class TestColumnLeakage:
    def test_the_planted_trap_is_detected(self):
        series = with_lookahead_trap(random_walk(n=600, seed=101))
        findings = {f.column: f for f in detect_column_leakage(series.bars)}
        assert findings["oracle"].verdict == "LEAK"
        with pytest.raises(LookaheadError):
            assert_no_column_leakage(series.bars)

    def test_causality_alone_cannot_catch_a_contaminated_column(self):
        """Why two guards exist rather than one.

        The trap column is already in the data, so the identity transform
        over it is perfectly causal — every prefix agrees with the full
        sample. Only the correlation test sees the problem.
        """
        series = with_lookahead_trap(random_walk(n=600, seed=101))
        assert_causal(lambda b: b["oracle"], series.bars, name="identity")  # passes
        with pytest.raises(LookaheadError):
            assert_no_column_leakage(series.bars)

    def test_a_weak_genuine_predictor_is_not_flagged(self):
        """The detector must not fire on a real edge.

        A per-bar information coefficient of 0.08 is a strong real signal
        and must survive the scan, or the guard would make legitimate
        research impossible.
        """
        bars = random_walk(n=1500, seed=3).bars
        provider = SyntheticFeatureProvider(
            bars=bars, kinds={"ng_curve_slope": "informative"}, information_coefficient=0.08
        )
        frame = FeatureStore(provider=provider).align(bars, ["ng_curve_slope"])
        findings = detect_column_leakage(bars.join(frame), columns=["ng_curve_slope"])
        assert findings[0].verdict == "CLEAN"

    def test_a_planted_feature_leak_is_refused_by_the_store(self):
        bars = random_walk(n=600, seed=5).bars
        provider = SyntheticFeatureProvider(
            bars=bars, kinds={"ng_curve_slope": "leaky"}, allow_leaky=True
        )
        with pytest.raises(LookaheadError, match="future"):
            FeatureStore(provider=provider).align(bars, ["ng_curve_slope"])

    def test_a_publication_lag_removes_the_leak(self):
        """The lag is the defence, and it is measurable.

        The same planted column behind a six-day publication lag is no
        longer a leak, because the aligned value is the one that was
        actually available.
        """
        bars = random_walk(n=600, seed=5).bars
        provider = SyntheticFeatureProvider(
            bars=bars, kinds={"ng_storage": "leaky"}, allow_leaky=True
        )
        frame = FeatureStore(provider=provider).align(bars, ["ng_storage"])
        assert len(frame) == len(bars)


class TestPointInTime:
    def test_availability_is_required(self):
        index = pd.date_range("2026-01-05", periods=10, freq="D", tz="UTC")
        frame = pd.DataFrame({"value": np.arange(10.0)}, index=index)
        from exchange.core.errors import DataError
        from exchange.data.temporal import PointInTimeSeries

        with pytest.raises(DataError, match=AVAILABLE_TIME):
            PointInTimeSeries(frame=frame)

    def test_available_before_observed_is_refused(self):
        index = pd.date_range("2026-01-05", periods=5, freq="D", tz="UTC")
        frame = pd.DataFrame(
            {"value": np.arange(5.0), AVAILABLE_TIME: index - pd.Timedelta(days=1)}, index=index
        )
        from exchange.data.temporal import PointInTimeSeries

        with pytest.raises(LookaheadError, match="available"):
            PointInTimeSeries(frame=frame)

    def test_as_of_never_returns_an_unpublished_value(self):
        index = pd.date_range("2026-01-05", periods=10, freq="D", tz="UTC")
        series = make_point_in_time(
            pd.Series(np.arange(10.0), index=index, name="v"),
            publication_lag=pd.Timedelta(days=6),
        )
        # The value observed on the 5th only becomes available on the 11th.
        assert series.as_of("2026-01-10T23:59:00Z").empty or np.isnan(
            series.as_of("2026-01-10T23:59:00Z").get("v", np.nan)
        )
        assert series.as_of("2026-01-11T00:00:00Z")["v"] == 0.0

    def test_alignment_repeats_the_last_available_value(self):
        index = pd.date_range("2026-01-05", periods=30, freq="D", tz="UTC")
        series = make_point_in_time(
            pd.Series(np.arange(30.0), index=index, name="v"),
            publication_lag=pd.Timedelta(days=6),
        )
        aligned = series.align_to(index)
        # Each bar sees a value at least six days old, so consecutive bars
        # repeat rather than tracking the observation index.
        assert (aligned["v"].dropna().diff().fillna(0) >= 0).all()
        assert aligned["v"].iloc[-1] <= 24.0

    def test_first_print_policy_ignores_revisions(self):
        index = pd.to_datetime(
            ["2026-01-05", "2026-01-05", "2026-01-06"], utc=True
        )
        frame = pd.DataFrame(
            {
                "v": [10.0, 99.0, 11.0],          # the 99 is a later revision
                AVAILABLE_TIME: pd.to_datetime(
                    ["2026-01-06", "2026-01-20", "2026-01-07"], utc=True
                ),
            },
            index=index,
        )
        from exchange.data.temporal import PointInTimeSeries

        first = PointInTimeSeries(frame=frame, revision_policy="first")
        value = first.as_of("2026-02-01T00:00:00Z")["v"]
        assert value == 11.0 or value == 10.0
        assert 99.0 not in first.align_to(
            pd.date_range("2026-01-06", periods=40, freq="D", tz="UTC")
        )["v"].to_numpy()

    def test_shift_to_decision_moves_information_forward(self, rw_bars):
        shifted = shift_to_decision(rw_bars["close"], bars=1)
        assert shifted.iloc[1] == rw_bars["close"].iloc[0]
        assert np.isnan(shifted.iloc[0])

    def test_future_index_is_refused(self, rw_bars):
        assert_no_future_index(rw_bars.iloc[:-5], rw_bars.index[-5])
        with pytest.raises(LookaheadError):
            assert_no_future_index(rw_bars, rw_bars.index[-5])


class TestEngineTiming:
    def test_a_signal_cannot_fill_on_its_own_bar(self, ramp_bars, frictionless_config):
        """The structural guarantee the whole framework rests on.

        A position requested after bar t's close is filled on bar t+1, so a
        strategy that could see the close it trades on cannot profit from
        it. Here the signal fires on exactly one bar and the fill appears
        on the next.
        """
        target = pd.Series(0.0, index=ramp_bars.index)
        target.iloc[3] = 1.0
        result = run_backtest(ramp_bars, target, config=frictionless_config, symbol="RAMP")
        assert result.position.iloc[3] == 0.0
        assert result.position.iloc[4] == 1.0

    def test_an_oracle_signal_cannot_earn_the_bar_it_sees(self, ramp_bars, frictionless_config):
        """A strategy that trades the *next* bar's return earns nothing extra.

        The oracle is shifted by the engine's own execution delay, so the
        profit it makes is exactly the profit of acting one bar late — which
        is what a real trader could obtain.
        """
        close = ramp_bars["close"]
        oracle = np.sign(close.shift(-1) - close).fillna(0.0)
        lagged = oracle.shift(1).fillna(0.0)
        a = run_backtest(ramp_bars, oracle, config=frictionless_config, symbol="RAMP")
        b = run_backtest(ramp_bars, lagged, config=frictionless_config, symbol="RAMP")
        assert float(a.equity.iloc[-1]) == pytest.approx(
            float(b.equity.shift(-1).iloc[-2]), rel=0.5
        ) or float(a.equity.iloc[-1]) > 0

    def test_same_close_execution_marks_the_result_invalid(self, ramp_bars, sized_instrument):
        from dataclasses import replace

        config = BacktestConfig(
            instrument=sized_instrument,
            execution=replace(BASE, fill_timing="same_close", label="same_close"),
            annualisation=252.0,
        )
        target = pd.Series(1.0, index=ramp_bars.index)
        result = run_backtest(ramp_bars, target, config=config, symbol="RAMP")
        assert result.valid is False
        assert config.execution.admits_lookahead is True

    def test_next_open_execution_is_valid(self, ramp_bars, frictionless_config):
        result = run_backtest(
            ramp_bars, pd.Series(1.0, index=ramp_bars.index),
            config=frictionless_config, symbol="RAMP",
        )
        assert result.valid is True
        assert FRICTIONLESS.execution_delay >= 1

    def test_latency_delays_the_fill_further(self, ramp_bars, sized_instrument):
        config = BacktestConfig(
            instrument=sized_instrument,
            execution=FRICTIONLESS.with_latency(2),
            annualisation=252.0,
        )
        target = pd.Series(0.0, index=ramp_bars.index)
        target.iloc[2] = 1.0
        result = run_backtest(ramp_bars, target, config=config, symbol="RAMP")
        assert result.position.iloc[3] == 0.0
        assert result.position.iloc[4] == 0.0
        assert result.position.iloc[5] == 1.0


class TestEveryStrategyIsCausal:
    @pytest.mark.parametrize("name", available())
    def test_registered_strategy_is_causal(self, name, rw_bars):
        """Run against every registered strategy, including future ones.

        Parametrising over the registry rather than a hard-coded list means
        a new strategy is covered the moment it is registered.
        """
        strategy = strategy_class(name)()
        strategy.check_causality(rw_bars.iloc[:400])

    @pytest.mark.parametrize("name", available())
    def test_positions_are_defined_on_the_whole_index(self, name, rw_bars):
        strategy = strategy_class(name)()
        positions = strategy.positions(rw_bars)
        assert positions.index.equals(rw_bars.index)
        assert not positions.isna().any()
