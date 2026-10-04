"""Strategies: the interface contract and each registered family."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from exchange.core.errors import ConfigError
from exchange.strategies.base import (
    Strategy,
    available,
    build,
    catalog,
    grid_size,
    iter_grid,
    register,
    strategy_class,
    valid_combinations,
)
from exchange.strategies.library import FAMILIES


class TestRegistry:
    def test_eleven_families_are_registered(self):
        assert len(available()) >= 11
        assert "buy_and_hold" in available()

    def test_every_strategy_declares_a_family(self):
        for name in available():
            assert strategy_class(name).family in FAMILIES

    def test_a_strategy_without_a_rationale_is_refused(self):
        """A strategy with no mechanism is a pattern, not a hypothesis."""

        class Nameless(Strategy):
            name = "no_rationale"
            family = "trend"
            rationale = "it works"
            defaults: ClassVar[dict] = {}

            def target_position(self, bars):
                return pd.Series(0.0, index=bars.index)

        with pytest.raises(ConfigError, match="rationale"):
            register(Nameless)

    def test_an_unnamed_strategy_is_refused(self):
        class Unnamed(Strategy):
            rationale = "a" * 50
            defaults: ClassVar[dict] = {}

            def target_position(self, bars):
                return pd.Series(0.0, index=bars.index)

        with pytest.raises(ConfigError, match="name"):
            register(Unnamed)

    def test_unknown_parameters_are_refused(self):
        with pytest.raises(ConfigError, match="unknown parameter"):
            build("ma_crossover", fsat=5)

    def test_unknown_strategy_is_refused(self):
        with pytest.raises(ConfigError, match="unknown strategy"):
            build("perpetual_money_machine")

    def test_catalog_reports_the_search_size(self):
        table = catalog()
        assert len(table) == len(available())
        assert (table["grid_size"] >= 1).all()
        assert table["rationale"].str.len().min() >= 30


class TestParameterGrids:
    def test_grid_size_is_the_product_of_the_axes(self):
        assert grid_size({"a": [1, 2, 3], "b": [1, 2]}) == 6
        assert grid_size({}) == 1

    def test_iteration_is_deterministic(self):
        space = {"b": [1, 2], "a": [3, 4]}
        assert list(iter_grid(space)) == list(iter_grid(space))

    def test_iteration_covers_the_grid(self):
        space = {"a": [1, 2, 3], "b": [10, 20]}
        assert len(list(iter_grid(space))) == grid_size(space)

    def test_invalid_combinations_are_skipped(self):
        """ma_crossover rejects fast >= slow, so overlapping axes shrink."""
        cls = strategy_class("ma_crossover")
        space = {"fast": [10, 20, 50], "slow": [20, 50, 100]}
        assert len(valid_combinations(cls, space)) < grid_size(space)
        assert valid_combinations(cls, space)

    def test_the_declared_grid_never_undercounts_the_trials(self):
        """The trial count charged to the deflation must not be too small.

        A declared space may contain a cell the strategy rejects (breakout
        refuses an exit window as long as its entry window), so the
        declared size can exceed the usable one. That direction is safe:
        over-counting trials makes the statistical bar higher. The reverse
        would smuggle unpriced searches past the deflation.
        """
        for name in available():
            cls = strategy_class(name)
            usable = len(valid_combinations(cls))
            assert 1 <= usable <= grid_size(cls.param_space)


class TestStrategyBehaviour:
    def test_buy_and_hold_is_always_long(self, rw_bars):
        positions = build("buy_and_hold").positions(rw_bars)
        assert (positions > 0).all()

    def test_ma_crossover_requires_fast_below_slow(self):
        with pytest.raises(ConfigError):
            build("ma_crossover", fast=50, slow=10)

    def test_momentum_follows_the_trailing_sign(self, ramp_bars):
        positions = build("ts_momentum", lookback=3).positions(ramp_bars)
        assert positions.iloc[-1] > 0

    def test_momentum_is_short_in_a_downtrend(self, ramp_bars):
        falling = ramp_bars.iloc[::-1].copy()
        falling.index = ramp_bars.index
        positions = build("ts_momentum", lookback=3).positions(falling)
        assert positions.iloc[-1] < 0

    def test_long_only_variant_never_shorts(self, rw_bars):
        positions = build("ts_momentum", lookback=20, allow_short=False).positions(rw_bars)
        assert (positions >= 0).all()

    def test_zscore_reversion_fades_extremes(self):
        index = pd.date_range("2026-01-05", periods=200, freq="B", tz="UTC", name="timestamp")
        close = np.concatenate([np.full(190, 100.0), np.linspace(100.0, 130.0, 10)])
        bars = pd.DataFrame(
            {"open": close, "high": close + 0.1, "low": close - 0.1, "close": close,
             "volume": 1000.0},
            index=index,
        )
        positions = build("zscore_reversion", window=20, entry_z=1.5).positions(bars)
        assert positions.iloc[-1] < 0        # price spiked up, so the strategy is short

    def test_breakout_uses_lagged_channels(self, rw_bars):
        """A channel that includes the current bar is a lookahead."""
        strategy = build("breakout", window=20, exit_window=10)
        strategy.check_causality(rw_bars.iloc[:300])

    def test_seasonal_month_only_trades_its_months(self, rw_bars):
        positions = build("seasonal_month", long_months=(1,)).positions(rw_bars)
        held = positions[positions != 0]
        assert set(held.index.month) == {1}

    def test_vol_target_momentum_takes_bars_per_year_as_a_parameter(self, rw_bars):
        """Inferring it from the data makes the strategy non-causal.

        Deriving bars-per-year from the series means the position at bar t
        depends on how much data follows, which the causality proof
        catches. A trading calendar is known in advance, so it is a
        parameter.
        """
        strategy = build("vol_target_momentum", bars_per_year=252.0)
        assert "bars_per_year" in strategy.params
        strategy.check_causality(rw_bars.iloc[:300])

    def test_positions_are_finite_and_float(self, rw_bars):
        for name in available():
            positions = strategy_class(name)().positions(rw_bars)
            assert positions.dtype.kind == "f"
            assert np.isfinite(positions).all()

    def test_a_strategy_cannot_return_a_mismatched_index(self, rw_bars):
        class Broken(Strategy):
            name = "broken_index"
            family = "trend"
            rationale = "deliberately broken for the test" + " " * 20
            defaults: ClassVar[dict] = {}

            def target_position(self, bars):
                return pd.Series(0.0, index=bars.index[:-1])

        with pytest.raises(ConfigError, match="index"):
            Broken().positions(rw_bars)

    def test_describe_carries_the_rationale(self):
        description = build("ts_momentum").describe()
        assert description["family"]
        assert len(description["rationale"]) >= 30
        assert "lookback" in description["params"]
