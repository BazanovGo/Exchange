"""Shared fixtures.

Every fixture is synthetic and seeded. The test suite never reads a vendor
file or touches the network: a test that depends on a data subscription
cannot run in CI, and a statistical test whose expected answer is unknown
proves nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exchange.backtest.engine import BacktestConfig, InstrumentSpec
from exchange.core.config import build_config
from exchange.data.dataset import make_dataset
from exchange.execution.assumptions import BASE, FRICTIONLESS
from exchange.funnel.hypothesis import make_hypothesis
from exchange.synthetic.generators import (
    mean_reverting,
    momentum,
    random_walk,
    signal_plus_noise,
)

SEED = 20260902


@pytest.fixture(scope="session")
def rw_series():
    return random_walk(n=1200, seed=11, s0=50.0, vol=0.015)


@pytest.fixture(scope="session")
def rw_bars(rw_series):
    return rw_series.bars


@pytest.fixture(scope="session")
def mr_series():
    return mean_reverting(n=1500, seed=13, half_life=8.0)


@pytest.fixture(scope="session")
def mom_series():
    return momentum(n=1500, seed=17)


@pytest.fixture(scope="session")
def edge_series():
    """The calibrated positive control: a known, analytically stated edge."""
    return signal_plus_noise(n=2000, seed=7, edge_per_bar=0.003, lookback=5, vol=0.012)


@pytest.fixture
def rw_dataset(rw_bars):
    return make_dataset(rw_bars, symbol="RW", timeframe="1d")


@pytest.fixture
def edge_dataset(edge_series):
    return make_dataset(edge_series.bars, symbol="EDGE", timeframe="1d")


@pytest.fixture
def futures_instrument():
    """A natural-gas-like contract: multiplier 10,000, tick 0.001."""
    return InstrumentSpec(
        symbol="NG", multiplier=10_000.0, tick_size=0.001, allow_fractional=False
    )


@pytest.fixture
def sized_instrument():
    """A contract whose notional is comparable with the test capital."""
    return InstrumentSpec(symbol="TEST", multiplier=1000.0)


@pytest.fixture
def frictionless_config(sized_instrument):
    """Zero costs: use when testing *accounting*, so costs cannot hide an error."""
    return BacktestConfig(
        initial_cash=100_000.0,
        instrument=sized_instrument,
        execution=FRICTIONLESS,
        annualisation=252.0,
    )


@pytest.fixture
def base_config(sized_instrument):
    return BacktestConfig(
        initial_cash=100_000.0,
        instrument=sized_instrument,
        execution=BASE,
        annualisation=252.0,
    )


@pytest.fixture
def research_config(tmp_path):
    """A fast configuration pointed at a temporary registry."""
    return build_config(
        {
            "registry_root": str(tmp_path / "experiments"),
            "reports_root": str(tmp_path / "reports"),
            "funnel": {"n_bootstrap": 150, "n_monte_carlo": 150},
            "falsification": {"n_resamples": 150},
        }
    )


@pytest.fixture
def momentum_hypothesis():
    return make_hypothesis(
        "HYP-TEST-MOM",
        statement="The sign of the trailing k-bar return predicts the next bar positively.",
        mechanism=(
            "Information reaches slower participants over several bars, so their order flow "
            "arrives after the move has begun and extends it."
        ),
        expected_regime="Trending regimes; should fail in a random walk by construction.",
        strategy="ts_momentum",
        param_space={"lookback": [3, 5, 10, 20]},
    )


@pytest.fixture
def flat_bars():
    """Twelve perfectly flat bars — the clearest possible accounting fixture."""
    index = pd.date_range("2026-01-05", periods=12, freq="B", tz="UTC", name="timestamp")
    price = 100.0
    return pd.DataFrame(
        {
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": 1_000.0,
        },
        index=index,
    )


@pytest.fixture
def ramp_bars():
    """Bars rising by exactly 1.0 per step, so P&L is calculable by hand."""
    index = pd.date_range("2026-01-05", periods=12, freq="B", tz="UTC", name="timestamp")
    close = np.arange(100.0, 112.0)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1_000.0,
        },
        index=index,
    )
