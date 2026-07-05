import pandas as pd
import pytest

from src.data import available_sources, load_data
from src.data.base import OHLCV_COLUMNS, MarketData, normalize_ohlcv
from src.data.sample import generate_ohlcv


def test_builtin_sources_registered():
    assert {"csv", "parquet", "yahoo"} <= set(available_sources())


def test_normalize_ohlcv_handles_messy_input():
    raw = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-01", "2024-01-02", "2024-01-02"],
            "OPEN": ["10", "11", "12", "12.5"],
            "high": [11, 12, 13, 13],
            "Low": [9, 10, 11, 11],
            "CLOSE": [10.5, 11.5, 12.5, 12.6],
            "vol": [100, 200, 300, 400],
        }
    )
    df = normalize_ohlcv(raw, symbol="X")
    assert list(df.columns[:5]) == list(OHLCV_COLUMNS)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing
    assert not df.index.duplicated().any()
    assert len(df) == 3  # duplicate timestamp collapsed, keep last
    assert df.loc["2024-01-02", "Close"] == 12.6
    assert df["Open"].dtype.kind == "f"


def test_normalize_ohlcv_requires_close():
    raw = pd.DataFrame({"date": ["2024-01-01"], "open": [1.0]})
    with pytest.raises(ValueError, match="Close"):
        normalize_ohlcv(raw)


def test_csv_roundtrip(tmp_path):
    df = generate_ohlcv("AAA", periods=50)
    df.to_csv(tmp_path / "AAA.csv")
    md = load_data("csv", "AAA", loader_kwargs={"base_dir": tmp_path})
    assert md.symbols == ["AAA"]
    pd.testing.assert_index_equal(md.get().index, df.index)
    assert md.get()["Close"].round(6).equals(df["Close"].round(6))


def test_parquet_roundtrip(tmp_path):
    df = generate_ohlcv("BBB", periods=50)
    df.to_parquet(tmp_path / "BBB.parquet")
    md = load_data("parquet", "BBB", loader_kwargs={"base_dir": tmp_path})
    assert md.get()["Close"].equals(df["Close"])


def test_date_slicing(tmp_path):
    df = generate_ohlcv("CCC", periods=100, start="2020-01-01")
    df.to_csv(tmp_path / "CCC.csv")
    md = load_data("csv", "CCC", start="2020-02-01", end="2020-03-01", loader_kwargs={"base_dir": tmp_path})
    assert md.get().index.min() >= pd.Timestamp("2020-02-01")
    assert md.get().index.max() <= pd.Timestamp("2020-03-01")


def test_multi_symbol_wide_frames(tmp_path):
    for sym, seed in (("S1", 1), ("S2", 2)):
        generate_ohlcv(sym, periods=30, seed=seed).to_csv(tmp_path / f"{sym}.csv")
    md = load_data("csv", ["S1", "S2"], loader_kwargs={"base_dir": tmp_path})
    close = md.close
    assert list(close.columns) == ["S1", "S2"]
    assert isinstance(md, MarketData)


def test_unknown_source_raises():
    with pytest.raises(KeyError, match="Unknown data source"):
        load_data("nope", "AAA")


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data("csv", "GHOST", loader_kwargs={"base_dir": tmp_path})


def test_structural_generators():
    import numpy as np

    from src.data.sample import generate_ou_ohlcv, generate_trending_ohlcv

    ou = generate_ou_ohlcv(periods=1500, seed=1)
    tr = generate_trending_ohlcv(periods=1500, seed=1)
    for df in (ou, tr):
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert (df["High"] >= df[["Open", "Close"]].max(axis=1)).all()
        assert (df["Low"] <= df[["Open", "Close"]].min(axis=1)).all()
        assert (df["Close"] > 0).all()

    # Structure checks: OU has negative return autocorrelation, the Markov
    # trend series positive.
    ou_ret = np.log(ou["Close"]).diff().dropna()
    tr_ret = np.log(tr["Close"]).diff().dropna()
    assert ou_ret.autocorr(1) < -0.005
    assert tr_ret.autocorr(1) > -0.02  # weak but not negative

    # Trend series must spend long stretches in one direction.
    sma50 = tr["Close"].rolling(50).mean()
    above = (tr["Close"] > sma50).dropna()
    runs = (above != above.shift()).cumsum()
    assert runs.value_counts().max() > 100
