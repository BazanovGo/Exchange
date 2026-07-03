import pandas as pd
import pytest

from src.backtesting import BacktestConfig, BacktestResult, run_backtest
from src.strategies import MACrossoverStrategy


def test_run_by_name(market_data):
    result = run_backtest("ma_crossover", market_data, params={"fast_window": 10, "slow_window": 30})
    assert isinstance(result, BacktestResult)
    assert result.strategy_name == "ma_crossover"
    assert result.symbol == "TEST"
    assert result.params["fast_window"] == 10


def test_run_with_instance_and_frame(ohlcv):
    strat = MACrossoverStrategy(fast_window=10, slow_window=30)
    result = run_backtest(strat, ohlcv)
    assert len(result.equity()) == len(ohlcv)


def test_params_rejected_with_instance(ohlcv):
    with pytest.raises(ValueError, match="params"):
        run_backtest(MACrossoverStrategy(), ohlcv, params={"fast_window": 5})


def test_config_applied(market_data):
    config = BacktestConfig(init_cash=50_000, fees=0.0, slippage=0.0)
    result = run_backtest("ma_crossover", market_data, params={"fast_window": 10, "slow_window": 30}, config=config)
    equity = result.equity()
    assert equity.iloc[0] == pytest.approx(50_000, rel=1e-6)
    assert result.config.init_cash == 50_000


def test_result_artifacts(market_data):
    result = run_backtest("ma_crossover", market_data, params={"fast_window": 10, "slow_window": 30})

    stats = result.stats()
    assert isinstance(stats, pd.Series)
    assert "Total Return [%]" in stats.index

    equity = result.equity()
    drawdown = result.drawdown()
    assert (equity > 0).all()
    assert (drawdown <= 1e-12).all()

    trades = result.trades()
    assert isinstance(trades, pd.DataFrame)
    assert len(trades) > 0


def test_short_strategy_runs(market_data):
    result = run_backtest(
        "ma_crossover",
        market_data,
        params={"fast_window": 10, "slow_window": 30, "allow_short": True},
    )
    directions = set(result.trades()["Direction"].unique())
    assert "Short" in directions and "Long" in directions
