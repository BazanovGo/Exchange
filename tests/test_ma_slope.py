import numpy as np
import pandas as pd
import pytest

from src.ma_slope.averages import MA_TYPES, atr, moving_average
from src.ma_slope.data import held_contract_path
from src.ma_slope.kernel import S_N, S_PNL, run_slow_state, trade_list
from src.ma_slope.signals import DOWN, INVALID, NEUTRAL, UP, cross_events, next_state, slope_state

N = 6000  # longer than the 4096-bar resync of the running sums


@pytest.fixture(scope="module")
def market():
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, N)))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.0005, N))
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.001, N))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.001, N))
    volume = rng.integers(1, 500, N).astype(float)
    return open_, high, low, close, volume


def _reference_ma(kind, x, v, n):
    s = pd.Series(x)
    if kind == "SMA":
        return s.rolling(n).mean().to_numpy()
    if kind in ("EMA", "SSMA"):
        a = 2 / (n + 1) if kind == "EMA" else 1 / n
        out = np.full(x.size, np.nan)
        out[n - 1] = x[:n].mean()
        for t in range(n, x.size):
            out[t] = a * x[t] + (1 - a) * out[t - 1]
        return out
    if kind == "WMA":
        w = np.arange(1, n + 1, dtype=float)
        return s.rolling(n).apply(lambda win: (win * w).sum() / w.sum(), raw=True).to_numpy()
    if kind == "VWMA":
        return ((s * v).rolling(n).sum() / pd.Series(v).rolling(n).sum()).to_numpy()
    if kind == "TMA":
        return s.rolling(-(-n // 2)).mean().rolling(n // 2 + 1).mean().to_numpy()
    raise AssertionError(kind)


@pytest.mark.parametrize("kind", MA_TYPES)
@pytest.mark.parametrize("n", [2, 3, 10, 57])
def test_moving_averages_match_reference(market, kind, n):
    _, _, _, close, volume = market
    got = moving_average(kind, close, volume, n)
    want = _reference_ma(kind, close, volume, n)
    assert np.isnan(got[: n - 1]).all() and not np.isnan(got[n - 1:]).any()
    np.testing.assert_allclose(got[n - 1:], want[n - 1:], rtol=1e-10)


def test_tma_has_triangular_weights():
    x = np.zeros(20)
    x[10] = 1.0  # impulse: TMA(5) responds with weights 1,2,3,2,1 / 9
    got = moving_average("TMA", x, np.ones(20), 5)
    np.testing.assert_allclose(got[10:15], np.array([1, 2, 3, 2, 1]) / 9)


def test_atr_is_wilder(market):
    _, high, low, close, _ = market
    tr = np.maximum.reduce([high[1:] - low[1:], abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])])
    tr = np.r_[high[0] - low[0], tr]
    np.testing.assert_allclose(atr(high, low, close, 14)[13:], _reference_ma("SSMA", tr, None, 14)[13:])


def test_cross_events_alternate_and_ignore_ties():
    close = np.array([1.0, 3.0, 2.0, 2.0, 3.0, 1.0, 2.0, 2.5])
    fast = np.array([np.nan, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    idx, d = cross_events(close, fast)
    # side: 1:+, 2:tie(+), 3:tie(+), 4:+, 5:-, 6:tie(-), 7:+
    assert idx.tolist() == [5, 7] and d.tolist() == [-1, 1]


def test_slope_state_methods():
    ma = np.array([np.nan, 100.0, 100.0, 100.5, 100.0, 99.0])
    a = np.full(6, 0.5)
    pct = slope_state(ma, 1, 0.1, a, 0)        # change in % over 1 bar
    assert pct.tolist() == [INVALID, INVALID, NEUTRAL, UP, DOWN, DOWN]
    atr_ = slope_state(ma, 1, 1.5, a, 1)       # change in ATRs: 0, +1, -1, -2
    assert atr_.tolist() == [INVALID, INVALID, NEUTRAL, NEUTRAL, NEUTRAL, DOWN]


def _naive(close, fast_ma, state, path, mult, model):
    """Literal bar-by-bar reading of the rules; orders fill at the next bar."""
    n = close.size
    side = np.zeros(n, int)
    for t in range(n):
        if np.isnan(fast_ma[t]):
            continue
        diff = close[t] - fast_ma[t]
        side[t] = 1 if diff > 0 else (-1 if diff < 0 else side[t - 1] if t else 0)
    up = np.r_[False, (side[:-1] == -1) & (side[1:] == 1)]
    down = np.r_[False, (side[:-1] == 1) & (side[1:] == -1)]
    trades, pos, entry, last_exit = [], 0, 0, -1
    for t in range(n - 1):
        if pos == 0:
            if t == last_exit:
                continue  # no reversal on the exit bar
            if up[t] and state[t] in (UP, NEUTRAL):
                pos, entry = 1, t + 1
            elif down[t] and state[t] in (DOWN, NEUTRAL):
                pos, entry = -1, t + 1
        else:
            if model == 0:
                leave = down[t] if pos == 1 else up[t]
            else:
                leave = state[t] == (DOWN if pos == 1 else UP)
            if leave:
                trades.append((entry, t + 1, pos, pos * (path[t + 1] - path[entry]) * mult))
                pos, last_exit = 0, t
    return np.array(trades).reshape(-1, 4)


@pytest.mark.parametrize("model", [0, 1])
@pytest.mark.parametrize("fast_kind,fast_n,slow_kind,slow_n,method,k,eps",
                         [("SMA", 2, "EMA", 50, 0, 5, 0.02), ("WMA", 15, "TMA", 120, 1, 10, 0.5),
                          ("VWMA", 40, "SSMA", 30, 0, 1, 0.0), ("EMA", 99, "SMA", 5, 1, 3, 0.1)])
def test_kernel_matches_naive_simulation(market, model, fast_kind, fast_n, slow_kind, slow_n, method, k, eps):
    open_, high, low, close, volume = market
    path = np.cumsum(np.r_[0, np.diff(open_)])
    fast = moving_average(fast_kind, close, volume, fast_n)
    slow = moving_average(slow_kind, close, volume, slow_n)
    state = slope_state(slow, k, eps, atr(high, low, close, 14), method)
    idx, d = cross_events(close, fast)
    minutes = np.arange(N, dtype=float) * 5
    year = (np.arange(N) // 1000).astype(np.int64)
    split = int(N * 0.7)
    rec, stats, years = trade_list(idx, d, model, state, next_state(state, DOWN), next_state(state, UP),
                                   path, 2.0, split, minutes, year, 6)
    want = _naive(close, fast, state, path, 2.0, model)
    assert len(want) > 20
    np.testing.assert_allclose(rec, want)
    assert stats[2, S_N] == len(want)
    assert stats[0, S_N] == (want[:, 1] < split).sum()
    np.testing.assert_allclose(stats[2, S_PNL], want[:, 3].sum())
    np.testing.assert_allclose(years[:, 0].sum(), want[:, 3].sum())
    # The batched kernel gives the same numbers.
    batch, _ = run_slow_state(idx, d, np.array([0, idx.size]), state, next_state(state, DOWN),
                              next_state(state, UP), path, 2.0, split, minutes, year, 6)
    np.testing.assert_allclose(batch[0, model], stats)


def test_held_contract_path_rolls_at_overlap_price():
    # Old contract trades 100 -> 104 (close), new one is 110 at the roll and opens at 111.
    raw_open = np.array([100.0, 102.0, 111.0, 115.0])
    raw_close = np.array([101.0, 104.0, 113.0, 116.0])
    secid = np.array(["A", "A", "B", "B"])
    ratio = 110.0 / 104.0
    factor = np.array([ratio, ratio, 1.0, 1.0])
    path = held_contract_path(raw_open * factor, raw_close * factor, secid, factor)
    # Long from the first open to the last: +4 on the old contract, +5 on the new one.
    assert path[3] - path[0] == pytest.approx((104 - 100) + (115 - 110))
    assert path[1] - path[0] == pytest.approx(2.0)


def test_t_stat_matches_scipy():
    from scipy import stats as sps

    from src.ma_slope.analysis import t_stat

    x = np.random.default_rng(3).normal(0.2, 1.0, 400)
    assert t_stat(x.size, x.sum(), (x * x).sum()) == pytest.approx(sps.ttest_1samp(x, 0).statistic)


def test_plateau_averages_valid_neighbours_only():
    from src.ma_slope.analysis import plateau

    t = np.zeros((1, 20, 1, 1, 9, 1))
    t[0, 10, 0, 0, 4, 0] = 55.0  # one spike among zeros: the 11 x 5 window mean is 1
    valid = np.ones(t.shape, bool)
    mean, share = plateau(t, valid)
    assert mean[0, 10, 0, 0, 4, 0] == pytest.approx(1.0)
    assert share[0, 10, 0, 0, 4, 0] == pytest.approx(1 / 55)
    valid[0, 5:16, 0, 0, 2:7, 0] = False
    valid[0, 10, 0, 0, 4, 0] = True  # an isolated valid cell has too few valid neighbours
    assert np.isnan(plateau(t, valid)[0][0, 10, 0, 0, 4, 0])


def test_multiple_testing_corrections():
    from src.ma_slope.analysis import benjamini_hochberg, holm

    p = pd.Series([0.01, 0.04, 0.03, 0.5])
    np.testing.assert_allclose(holm(p), [0.04, 0.09, 0.09, 0.5])
    np.testing.assert_allclose(benjamini_hochberg(p), [0.04, 0.0533333, 0.0533333, 0.5], rtol=1e-5)


def test_walk_forward_selects_on_train_years_only():
    from src.ma_slope.analysis import walk_forward

    rng = np.random.default_rng(0)
    years = np.zeros((1, 30, 1, 1, 12, 1, 5, 3), np.float32)
    pnl = rng.normal(0, 1, (30, 12, 5, 200))
    pnl[2:10, 1:5, :3] += 0.3     # region A: good in the training years 2020-2022 only
    pnl[18:28, 6:11, 3:] += 1.0   # region B: much better, but only in the test years
    years[0, :, 0, 0, :, 0, :, 0] = pnl.sum(-1)
    years[..., 1] = 200
    years[0, :, 0, 0, :, 0, :, 2] = (pnl ** 2).sum(-1)
    meta = {"n_years": 5, "first_year": 2020, "slow_types": ["SMA"], "slow_periods": list(range(30)),
            "fast_types": ["SMA"], "fast_periods": list(range(12)), "models": ["reverse_cross"]}
    wf = walk_forward(years, meta, train=3)
    assert list(wf["test_year"]) == [2023, 2024]
    first = wf.iloc[0]
    assert 2 <= first["slow_period"] < 10 and 1 <= first["fast_period"] < 5
    assert first["test_t"] < 2  # region A has no edge in the test year
