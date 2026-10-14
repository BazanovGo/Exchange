import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt

from src.backtesting import BacktestConfig
from src.backtesting.config import effective_freq
from src.data import available_sources, load_data
from src.utils.paths import RAW_DATA_DIR

COLUMNS = "ticker,secid,begin,end,open,high,low,close,volume,value"
ROLL_HEADER = "from_secid,to_secid,from_time,from_price,to_time,to_price,ratio,method"


def _write(base, ticker, year, rows):
    path = base / ticker / "60m" / f"{year}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [COLUMNS] + [f"{ticker},{secid},{begin},{begin},{p},{p + 1},{p - 1},{p},10,0" for secid, begin, p in rows]
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def algopack_dir(tmp_path):
    """Two contracts: AAH6 trades in 2025 and on 2026-01-05, AAM6 from 2026-01-06 at +10%."""
    _write(tmp_path, "aa", 2025, [("AAH6", "2025-12-30 10:00:00", 100.0), ("AAH6", "2025-12-30 11:00:00", 102.0)])
    _write(tmp_path, "aa", 2026, [("AAH6", "2026-01-05 10:00:00", 104.0), ("AAH6", "2026-01-05 13:00:00", 100.0),
                                  ("AAM6", "2026-01-06 10:00:00", 111.0), ("AAM6", "2026-01-06 11:00:00", 121.0)])
    (tmp_path / "aa" / "rolls.csv").write_text(
        f"{ROLL_HEADER}\nAAH6,AAM6,2026-01-05 13:52:00,100.0,2026-01-05 13:52:00,110.0,1.1,overlap\n")
    return tmp_path


def _load(base, **kwargs):
    return load_data("moex_algopack", "aa", loader_kwargs={"base_dir": base}, **kwargs).get()


def test_moex_algopack_registered():
    assert "moex_algopack" in available_sources()


def test_ratio_adjustment_rescales_history_before_roll(algopack_dir):
    df = _load(algopack_dir, timeframe="60m")
    raw = _load(algopack_dir, timeframe="60m", adjust=None)
    assert list(df.columns[:5]) == ["Open", "High", "Low", "Close", "Volume"]
    old = df["secid"] == "AAH6"
    np.testing.assert_allclose(df.loc[old, "Close"], raw.loc[old, "Close"] * 1.1)
    np.testing.assert_allclose(df.loc[~old, "Close"], raw.loc[~old, "Close"])
    np.testing.assert_allclose(df["Volume"], raw["Volume"])
    np.testing.assert_allclose(df["Close"] / df["adj_factor"], raw["Close"])
    # The 10% contango gap disappears: 110 -> 111 is the real move of the new contract.
    returns = df["Close"].pct_change()
    assert returns.iloc[4] == pytest.approx(111 / 110 - 1)
    assert raw["Close"].pct_change().iloc[4] == pytest.approx(0.11)
    # Returns inside a contract are unchanged.
    assert returns.iloc[1] == pytest.approx(102 / 100 - 1)


def test_start_skips_old_years_but_keeps_later_rolls(algopack_dir):
    df = _load(algopack_dir, timeframe="1h", start="2026-01-01")
    assert df.index.min() == pd.Timestamp("2026-01-05 10:00")
    assert df["Close"].iloc[0] == pytest.approx(104.0 * 1.1)


def test_unknown_timeframe_and_adjust_rejected(algopack_dir):
    with pytest.raises(ValueError, match="1m, 5m and 60m"):
        _load(algopack_dir, timeframe="1d")
    with pytest.raises(ValueError, match="adjust"):
        _load(algopack_dir, timeframe="60m", adjust="panama")


def test_rolls_out_of_sync_rejected(algopack_dir):
    (algopack_dir / "aa" / "rolls.csv").write_text(
        f"{ROLL_HEADER}\nAAH6,AAU6,2026-01-05 13:52:00,100.0,2026-01-05 13:52:00,110.0,1.1,overlap\n")
    with pytest.raises(ValueError, match="out of sync"):
        _load(algopack_dir, timeframe="60m")


def test_effective_freq_counts_trading_bars_per_year():
    days = pd.bdate_range("2020-01-01", "2023-12-31")
    assert 258 < pd.Timedelta("365 days") / effective_freq(days) < 263
    # 10 hourly bars per weekday: ~2 610 bars a year, not 8 760.
    hours = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in days for h in range(10, 20)])
    close = pd.Series(np.linspace(100, 120, len(hours)), index=hours)
    pf = vbt.Portfolio.from_holding(close, **BacktestConfig(fees=0, slippage=0).to_portfolio_kwargs(close.index))
    assert 2580 < pf.returns_acc.ann_factor < 2640


def test_backtest_config_freq():
    idx = pd.date_range("2024-01-01", periods=5, freq="h")
    assert BacktestConfig().to_portfolio_kwargs(idx)["freq"] == pd.Timedelta("1h")
    assert BacktestConfig(freq="1D").to_portfolio_kwargs()["freq"] == "1D"
    with pytest.raises(ValueError, match="price index"):
        BacktestConfig().to_portfolio_kwargs()


@pytest.mark.skipif(not (RAW_DATA_DIR / "moex_algopack" / "ng" / "rolls.csv").exists(),
                    reason="MOEX AlgoPack data not downloaded")
def test_real_ng_hourly_is_adjusted():
    df = load_data("moex_algopack", "ng", timeframe="60m").get()
    raw = load_data("moex_algopack", "ng", timeframe="60m", adjust=None).get()
    last = df["secid"] == df["secid"].iloc[-1]
    np.testing.assert_allclose(df.loc[last, "Close"], raw.loc[last, "Close"])
    assert df[["Open", "High", "Low", "Close"]].gt(0).all().all()
    roll = df["secid"].ne(df["secid"].shift()).to_numpy()[1:]

    def gap(f):
        return (f["Open"].to_numpy()[1:] / f["Close"].to_numpy()[:-1] - 1)[roll]

    assert np.median(np.abs(gap(df))) < np.median(np.abs(gap(raw))) / 2
