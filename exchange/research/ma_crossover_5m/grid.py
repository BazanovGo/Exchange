"""Сетка: быстрые события, состояния медленной средней, пакетный прогон."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import kernel as K
import ma as M
import slope as S
from data import Market

FAST_PERIODS = np.arange(2, 100)        # 98
SLOW_PERIODS = np.arange(5, 251)        # 246
TYPES = M.TYPES                         # 6
METHODS = ("A_pct", "B_atr")
EXITS = ("A_fast_cross", "B_slow_flip")


def build_fast_events(mk: Market):
    """Пересечения Close со всеми 588 быстрыми средними, одним плоским массивом."""
    bars, dirs, off = [], [], [0]
    for t in TYPES:
        for p in FAST_PERIODS:
            m = M.moving_average(t, mk.close, mk.volume, int(p))
            b, d = K.cross_events(mk.close, m, mk.is_roll)
            bars.append(b)
            dirs.append(d)
            off.append(off[-1] + len(b))
    return (np.concatenate(bars), np.concatenate(dirs), np.asarray(off, dtype=np.int64))


def slow_state(mk: Market, slow_ma: np.ndarray, method: str, k: int, coverage: float,
               atr_arr: np.ndarray | None):
    if method == "A_pct":
        sl = S.slope_pct(slow_ma, k)
    else:
        sl = S.slope_atr(slow_ma, atr_arr, k)
    eps = S.threshold(sl, mk.oos_bar, coverage)
    st = S.classify(sl, eps)
    info = (eps, S.neutral_share(st, 0, mk.oos_bar), S.neutral_share(st, mk.oos_bar, mk.n))
    return st, info


class GridWriter:
    """Memmap-хранилище: stats[config, bucket, stat] и monthly[config, month]."""

    def __init__(self, out: Path, n_configs: int, n_months: int, mode="w+"):
        out.mkdir(parents=True, exist_ok=True)
        self.stats = np.lib.format.open_memmap(
            out / "stats.npy", mode=mode, dtype=np.float32, shape=(n_configs, 3, K.N_STATS))
        self.monthly = np.lib.format.open_memmap(
            out / "monthly.npy", mode=mode, dtype=np.float32, shape=(n_configs, n_months))


def run_base_grid(mk: Market, out: Path, *, k: int = 5, coverage: float = 0.25,
                  atr_period: int = 14, per_method: dict | None = None, log_every: int = 50):
    """Полная базовая сетка при заданных параметрах наклона.

    Порядок осей результата: (method 2, slow_type 6, slow_period 246,
    exit 2, fast_type 6, fast_period 98) → 3 471 552 конфигурации.
    ``per_method`` — свои (k, coverage, atr_period) для каждого метода (этап 3).
    """
    t0 = time.time()
    if per_method is None:
        per_method = {m: {"k": k, "coverage": coverage, "atr_period": atr_period} for m in METHODS}
    ev_bar, ev_dir, ev_off = build_fast_events(mk)
    n_fast = len(ev_off) - 1
    atrs = {m: M.atr(mk.high, mk.low, mk.close, int(per_method[m]["atr_period"])) for m in METHODS}
    n_slow = len(METHODS) * len(TYPES) * len(SLOW_PERIODS)
    per_slow = 2 * n_fast
    W = GridWriter(out, n_slow * per_slow, len(mk.months))
    slow_info = np.zeros((n_slow, 3), dtype=np.float64)
    buf_s = np.zeros((2, n_fast, 3, K.N_STATS))
    buf_m = np.zeros((2, n_fast, len(mk.months)))
    job = 0
    for mi, method in enumerate(METHODS):
        for ti, st_type in enumerate(TYPES):
            for p in SLOW_PERIODS:
                slow_ma = M.moving_average(st_type, mk.close, mk.volume, int(p))
                pm = per_method[method]
                state, info = slow_state(mk, slow_ma, method, int(pm["k"]), float(pm["coverage"]),
                                         atrs[method])
                nd = K.next_state_index(state, -1)
                nu = K.next_state_index(state, 1)
                buf_s[:] = 0.0
                buf_m[:] = 0.0
                K.run_fast_block(ev_bar, ev_dir, ev_off, state, nd, nu,
                                 mk.pnl_open, mk.pnl_close, mk.month, mk.oos_bar,
                                 mk.multiplier, buf_s, buf_m)
                lo = job * per_slow
                W.stats[lo:lo + per_slow] = buf_s.reshape(per_slow, 3, K.N_STATS)
                W.monthly[lo:lo + per_slow] = buf_m.reshape(per_slow, -1)
                slow_info[job] = info
                job += 1
                if job % log_every == 0:
                    el = time.time() - t0
                    print(f"  {mk.ticker} {job}/{n_slow} slow-состояний, {el:,.0f} с, "
                          f"осталось ~{el / job * (n_slow - job):,.0f} с", flush=True)
    W.stats.flush()
    W.monthly.flush()
    np.save(out / "slow_info.npy", slow_info)
    meta = {
        "ticker": mk.ticker, "k": k, "coverage": coverage, "atr_period": atr_period,
        "per_method": per_method,
        "axes": ["method", "slow_type", "slow_period", "exit", "fast_type", "fast_period"],
        "shape": [2, 6, 246, 2, 6, 98], "methods": METHODS, "types": TYPES,
        "exits": EXITS, "fast_periods": [int(FAST_PERIODS[0]), int(FAST_PERIODS[-1])],
        "slow_periods": [int(SLOW_PERIODS[0]), int(SLOW_PERIODS[-1])],
        "months": [str(m) for m in mk.months], "oos_month": mk.oos_month,
        "n_events_total": int(len(ev_bar)), "seconds": time.time() - t0,
        "multiplier": mk.multiplier, "tick": mk.tick,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta
