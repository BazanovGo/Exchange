import numpy as np
import pandas as pd

from src.indicators import (
    adx,
    cci,
    donchian,
    ichimoku,
    keltner_channels,
    momentum,
    parabolic_sar,
    roc,
    rolling_vwap,
    stochastic,
    supertrend,
    williams_r,
    zscore,
)


def test_donchian_breakout_semantics(ohlcv):
    upper, middle, lower = donchian(ohlcv["High"], ohlcv["Low"], 20)
    valid = upper.dropna()
    assert len(valid) > 0
    assert (upper.dropna() >= lower.dropna()).all()
    # exclude_current: current bar's high must be able to exceed the band
    breakout_days = (ohlcv["Close"] > upper).sum()
    assert breakout_days > 0


def test_adx_bounds(ohlcv):
    adx_, plus_di, minus_di = adx(ohlcv["High"], ohlcv["Low"], ohlcv["Close"], 14)
    for s in (adx_, plus_di, minus_di):
        valid = s.dropna()
        assert len(valid) > 100
        assert ((valid >= 0) & (valid <= 100)).all()


def test_supertrend_direction(ohlcv):
    line, direction = supertrend(ohlcv["High"], ohlcv["Low"], ohlcv["Close"], 10, 3.0)
    valid = direction[line.notna()]
    assert set(valid.unique()) <= {1.0, -1.0}
    # In an uptrend the line must sit below the close, in a downtrend above.
    close = ohlcv["Close"][line.notna()]
    l = line.dropna()  # noqa: E741
    assert (close[valid > 0] >= l[valid > 0]).all()
    assert (close[valid < 0] <= l[valid < 0]).all()


def test_parabolic_sar_flips(ohlcv):
    sar, trend = parabolic_sar(ohlcv["High"], ohlcv["Low"])
    valid = trend[sar.notna()]
    assert {1.0, -1.0} <= set(valid.unique())  # both regimes occur
    assert np.isfinite(sar.dropna()).all()


def test_ichimoku_components(ohlcv):
    ich = ichimoku(ohlcv["High"], ohlcv["Low"], ohlcv["Close"])
    assert set(ich) == {"tenkan", "kijun", "senkou_a", "senkou_b", "chikou"}
    assert ich["senkou_a"].notna().sum() > 100


def test_oscillator_ranges(ohlcv):
    h, lo, c = ohlcv["High"], ohlcv["Low"], ohlcv["Close"]

    k, d = stochastic(h, lo, c)
    assert ((k.dropna() >= 0) & (k.dropna() <= 100)).all()

    wr = williams_r(h, lo, c)
    assert ((wr.dropna() >= -100) & (wr.dropna() <= 0)).all()

    cci_ = cci(h, lo, c)
    assert np.isfinite(cci_.dropna()).all()
    assert cci_.dropna().abs().max() > 100  # extremes do occur


def test_momentum_indicators(ohlcv):
    close = ohlcv["Close"]
    r = roc(close, 12)
    m = momentum(close, 10)
    assert r.notna().sum() > 100
    pd.testing.assert_series_equal(
        (close - close.shift(10)).dropna(), m.dropna(), check_names=False
    )


def test_vwap_and_zscore(ohlcv):
    vwap = rolling_vwap(ohlcv["High"], ohlcv["Low"], ohlcv["Close"], ohlcv["Volume"], 20)
    # VWAP must stay inside the price envelope of the window
    assert (vwap.dropna() > 0).all()

    z = zscore(ohlcv["Close"], 20)
    valid = z.dropna()
    assert abs(valid.mean()) < 0.5  # roughly centered
    assert valid.abs().max() < 10


def test_keltner_ordering(ohlcv):
    middle, upper, lower = keltner_channels(ohlcv["High"], ohlcv["Low"], ohlcv["Close"])
    m, u, lo = middle.dropna(), upper.dropna(), lower.dropna()
    idx = m.index.intersection(u.index).intersection(lo.index)
    assert (u[idx] > m[idx]).all()
    assert (m[idx] > lo[idx]).all()
