"""Этап 2: оптимизация параметров наклона (k, ε через долю нейтрали, период ATR).

Полное произведение сетки наклона на полную базовую сетку — 200 настроек ×
3 471 552 ≈ 694 млн конфигураций на инструмент (~4 суток расчёта и ~0.5 ТБ
помесячных результатов на этой машине). Поэтому этап поэтапный и это
задокументировано в отчёте:

* решётка периодов — логарифмическая (числа Фибоначчи), задана ДО расчётов и
  не зависит от результатов: быстрые {2,3,5,8,13,21,34,55,89}, медленные
  {5,8,13,21,34,55,89,144,233};
* все 36 сочетаний типов, оба выхода, оба метода, полная сетка наклона:
  k ∈ {1,2,3,5,8,13,21,34}; доля нейтрали ∈ {0, 0.10, 0.25, 0.40, 0.60};
  ATR ∈ {7,14,28,56} для метода B.
  Итого 40 настроек A + 160 настроек B = 200; 54 × 54 × 200 × 2 = 1 166 400
  конфигураций на инструмент.

Лучшая настройка наклона для каждого метода выбирается по in-sample
(медиана месячного Шарпа по всей решётке) и затем используется на этапе 3 —
повторном прогоне ПОЛНОЙ базовой сетки.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import kernel as K
import ma as M
from data import load_market
from grid import GridWriter, slow_state

LAT_FAST = np.array([2, 3, 5, 8, 13, 21, 34, 55, 89])
LAT_SLOW = np.array([5, 8, 13, 21, 34, 55, 89, 144, 233])
K_GRID = (1, 2, 3, 5, 8, 13, 21, 34)
COV_GRID = (0.0, 0.10, 0.25, 0.40, 0.60)
ATR_GRID = (7, 14, 28, 56)


def settings():
    out = [("A_pct", k, c, 0) for k in K_GRID for c in COV_GRID]
    out += [("B_atr", k, c, a) for k in K_GRID for c in COV_GRID for a in ATR_GRID]
    return out


def lattice_events(mk):
    bars, dirs, off = [], [], [0]
    for t in M.TYPES:
        for p in LAT_FAST:
            m = M.moving_average(t, mk.close, mk.volume, int(p))
            b, d = K.cross_events(mk.close, m, mk.is_roll)
            bars.append(b); dirs.append(d); off.append(off[-1] + len(b))
    return np.concatenate(bars), np.concatenate(dirs), np.asarray(off, dtype=np.int64)


def run(mk, out: Path):
    t0 = time.time()
    S = settings()
    ev_bar, ev_dir, ev_off = lattice_events(mk)
    n_fast = len(ev_off) - 1
    atrs = {a: M.atr(mk.high, mk.low, mk.close, a) for a in ATR_GRID}
    per = 2 * n_fast
    n_slow_jobs = len(M.TYPES) * len(LAT_SLOW) * len(S)
    W = GridWriter(out, n_slow_jobs * per, len(mk.months))
    info = np.zeros((n_slow_jobs, 3))
    bs = np.zeros((2, n_fast, 3, K.N_STATS)); bm = np.zeros((2, n_fast, len(mk.months)))
    job = 0
    for st in M.TYPES:
        for p in LAT_SLOW:
            slow = M.moving_average(st, mk.close, mk.volume, int(p))
            for method, k, cov, a in S:
                state, inf = slow_state(mk, slow, method, k, cov, atrs.get(a))
                nd, nu = K.next_state_index(state, -1), K.next_state_index(state, 1)
                bs[:] = 0.0; bm[:] = 0.0
                K.run_fast_block(ev_bar, ev_dir, ev_off, state, nd, nu, mk.pnl_open,
                                 mk.pnl_close, mk.month, mk.oos_bar, mk.multiplier, bs, bm)
                lo = job * per
                W.stats[lo:lo + per] = bs.reshape(per, 3, K.N_STATS)
                W.monthly[lo:lo + per] = bm.reshape(per, -1)
                info[job] = inf
                job += 1
            el = time.time() - t0
            print(f"  {mk.ticker} slow {st}{p}: {job}/{n_slow_jobs}, {el:,.0f} с, "
                  f"осталось ~{el / job * (n_slow_jobs - job):,.0f} с", flush=True)
    W.stats.flush(); W.monthly.flush()
    np.save(out / "slow_info.npy", info)
    meta = {"ticker": mk.ticker, "axes": ["slow_type", "slow_period", "setting", "exit",
                                          "fast_type", "fast_period"],
            "shape": [6, len(LAT_SLOW), len(S), 2, 6, len(LAT_FAST)],
            "types": list(M.TYPES), "lat_fast": LAT_FAST.tolist(), "lat_slow": LAT_SLOW.tolist(),
            "settings": [list(s) for s in S], "months": [str(m) for m in mk.months],
            "oos_month": mk.oos_month, "multiplier": mk.multiplier, "tick": mk.tick,
            "exits": ["A_fast_cross", "B_slow_flip"], "seconds": time.time() - t0}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


if __name__ == "__main__":
    work = Path(os.environ.get("MA5M_WORK", "work"))
    for t in sys.argv[1:] or ("NG", "BR"):
        mk = load_market(t)
        m = run(mk, work / "stage2" / t)
        print(f"== {t} этап 2 готов за {m['seconds'] / 60:.1f} мин", flush=True)
