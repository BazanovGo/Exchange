import pandas as pd
import pytest

from src.strategies import (
    MACrossoverStrategy,
    RSIMeanReversionStrategy,
    available_strategies,
    get_strategy,
)
from src.strategies.base import StrategySignals


def test_builtin_strategies_registered():
    assert {"ma_crossover", "rsi_reversion"} <= set(available_strategies())


def test_signals_contract(ohlcv):
    strat = MACrossoverStrategy(fast_window=10, slow_window=30)
    sig = strat.signals(ohlcv)

    assert isinstance(sig, StrategySignals)
    for leg in (sig.entries, sig.exits, sig.short_entries, sig.short_exits):
        assert leg.dtype == bool
        assert leg.index.equals(ohlcv.index)
    assert sig.params == {"fast_window": 10, "slow_window": 30, "ewm": False, "allow_short": False}
    assert sig.description
    assert sig.entries.sum() > 0
    assert not sig.has_shorts  # long-only by default


def test_short_legs_enabled(ohlcv):
    sig = MACrossoverStrategy(fast_window=10, slow_window=30, allow_short=True).signals(ohlcv)
    assert sig.has_shorts
    # Short entries mirror long exits in this strategy.
    pd.testing.assert_series_equal(sig.short_entries, sig.exits, check_names=False)


def test_rsi_strategy_signals(ohlcv):
    sig = RSIMeanReversionStrategy(window=14, lower=35, upper=65, allow_short=True).signals(ohlcv)
    assert sig.entries.sum() > 0
    assert sig.short_entries.sum() > 0


def test_registry_instantiation(ohlcv):
    strat = get_strategy("ma_crossover", fast_window=5, slow_window=20)
    assert isinstance(strat, MACrossoverStrategy)
    assert strat.params["fast_window"] == 5


def test_unknown_param_rejected():
    with pytest.raises(ValueError, match="unknown params"):
        MACrossoverStrategy(bogus=1)


def test_invalid_params_rejected():
    with pytest.raises(ValueError, match="fast_window"):
        MACrossoverStrategy(fast_window=50, slow_window=20)
    with pytest.raises(ValueError, match="lower"):
        RSIMeanReversionStrategy(lower=80, upper=70)


def test_unknown_strategy_raises():
    with pytest.raises(KeyError, match="Unknown strategy"):
        get_strategy("nope")


def test_market_data_input(market_data):
    sig = MACrossoverStrategy().signals(market_data)
    assert sig.entries.index.equals(market_data.get().index)
