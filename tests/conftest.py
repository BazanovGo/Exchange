import pandas as pd
import pytest

from src.data.base import MarketData
from src.data.sample import generate_ohlcv


@pytest.fixture(scope="session")
def ohlcv() -> pd.DataFrame:
    return generate_ohlcv("TEST", periods=750, seed=7)


@pytest.fixture(scope="session")
def market_data(ohlcv: pd.DataFrame) -> MarketData:
    return MarketData(frames={"TEST": ohlcv}, source="synthetic", timeframe="1d")
