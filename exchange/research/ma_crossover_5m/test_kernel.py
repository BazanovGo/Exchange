"""Ядро против независимой побаровой реализации тех же правил."""
import numpy as np
import pytest

import kernel as K
import slope as S
import ma as M


def reference(close, ma_fast, state, is_roll, p_open, p_close, model, mult):
    """Наивный побаровый автомат: другая структура кода — те же правила."""
    n = len(close)
    side = 0
    pos = 0
    entry_bar = entry_px = None
    blocked_until = -1          # бар решения о выходе; входы только после него
    trades = []
    for t in range(n):
        m = ma_fast[t]
        cross = 0
        if np.isnan(m):
            side = 0
        else:
            new = 1 if close[t] > m else (-1 if close[t] < m else side)
            if side != 0 and new != 0 and new != side and not is_roll[t]:
                cross = new
            side = new
        exit_now = False
        if pos != 0:
            if model == 0 and cross == -pos:
                exit_now = True
            if model == 1 and ((pos > 0 and state[t] == -1) or (pos < 0 and state[t] == 1)):
                exit_now = True
            if exit_now:
                if t + 1 < n:
                    px, xb = p_open[t + 1], t + 1
                else:
                    px, xb = p_close[n - 1], n - 1
                trades.append((entry_bar, xb, pos, entry_px, px, pos * (px - entry_px) * mult))
                pos = 0
                blocked_until = t
                continue
        if pos == 0 and cross != 0 and t > blocked_until and t + 1 < n:
            s = state[t]
            if (cross > 0 and s in (1, 0)) or (cross < 0 and s in (-1, 0)):
                pos, entry_bar, entry_px = cross, t + 1, p_open[t + 1]
    if pos != 0:
        px = p_close[n - 1]
        trades.append((entry_bar, n - 1, pos, entry_px, px, pos * (px - entry_px) * mult))
    return trades


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("model", [0, 1])
def test_kernel_matches_reference(seed, model):
    rng = np.random.default_rng(seed)
    n = 4000
    close = 50 + np.cumsum(rng.normal(0, 0.1, n))
    close = np.round(close, 2)                       # тиковая сетка → равенства бывают
    p_open = np.r_[close[0], close[:-1]] + np.round(rng.normal(0, 0.02, n), 2)
    p_close = close.copy()
    is_roll = np.zeros(n, dtype=bool)
    is_roll[rng.choice(n, 8, replace=False)] = True
    fast = M.sma(close, int(rng.integers(2, 30)))
    slow = M.ema(close, int(rng.integers(5, 120)))
    sl = S.slope_pct(slow, int(rng.integers(1, 10)))
    eps = S.threshold(sl, n // 2, float(rng.choice([0.0, 0.2, 0.5])))
    state = S.classify(sl, eps)
    month = (np.arange(n) // 500).astype(np.int32)

    ev_bar, ev_dir = K.cross_events(close, fast, is_roll)
    nd, nu = K.next_state_index(state, -1), K.next_state_index(state, 1)
    stats = np.zeros((3, K.N_STATS))
    monthly = np.zeros(month.max() + 1)
    log = np.zeros((n, 6))
    ntr = K.simulate(ev_bar, ev_dir, state, nd, nu, p_open, p_close, month,
                     n // 2, 10.0, model, stats, monthly, log)
    ref = reference(close, fast, state, is_roll, p_open, p_close, model, 10.0)
    assert ntr == len(ref)
    np.testing.assert_allclose(log[:ntr], np.array(ref, dtype=float).reshape(-1, 6), atol=1e-9)
    assert stats[0, K.PNL] == pytest.approx(sum(r[5] for r in ref), abs=1e-6)
    assert stats[1, K.NTR] + stats[2, K.NTR] == ntr
    assert monthly.sum() == pytest.approx(stats[0, K.PNL], abs=1e-6)
