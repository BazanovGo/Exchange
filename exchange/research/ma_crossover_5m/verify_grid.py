"""Проверка раскладки сетки: случайные строки memmap против прямого пересчёта.

Ошибка в порядке осей молча переименовала бы конфигурации — все выводы были
бы о чужих стратегиях. Здесь каждая проверяемая строка декодируется в
параметры и пересчитывается с нуля одиночным вызовом ядра.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

import kernel as K
import ma as M
from data import load_market
from grid import slow_state
from results import load_grid


def main(stage, ticker, n=25, seed=7):
    work = Path(os.environ.get("MA5M_WORK", "work"))
    g = load_grid(work / stage / ticker)
    mk = load_market(ticker)
    pm = g.meta.get("per_method") or {m: {"k": g.meta["k"], "coverage": g.meta["coverage"],
                                          "atr_period": g.meta["atr_period"]} for m in g.meta["methods"]}
    rng = np.random.default_rng(seed)
    idx = rng.choice(g.n, n, replace=False)
    bad = 0
    for flat, row in zip(idx, g.decode(idx).itertuples()):
        fast = M.moving_average(row.fast_type, mk.close, mk.volume, int(row.fast_period))
        eb, ed = K.cross_events(mk.close, fast, mk.is_roll)
        slow = M.moving_average(row.slow_type, mk.close, mk.volume, int(row.slow_period))
        p = pm[row.method]
        atr = M.atr(mk.high, mk.low, mk.close, int(p["atr_period"]))
        st, _ = slow_state(mk, slow, row.method, int(p["k"]), float(p["coverage"]), atr)
        s = np.zeros((3, K.N_STATS)); mo = np.zeros(len(mk.months))
        K.simulate(eb, ed, st, K.next_state_index(st, -1), K.next_state_index(st, 1), mk.pnl_open,
                   mk.pnl_close, mk.month, mk.oos_bar, mk.multiplier, g.meta["exits"].index(row.exit),
                   s, mo, np.zeros((0, 6)))
        ok = np.allclose(s, g.stats[flat], rtol=1e-5, atol=1e-2) and np.allclose(mo, g.monthly[flat], atol=1e-2)
        bad += not ok
        print(f"{flat:>8} {row.fast_type}{row.fast_period}/{row.slow_type}{row.slow_period} {row.method} "
              f"{row.exit}: PnL {s[0, K.PNL]:>10,.1f} vs {g.stats[flat][0, K.PNL]:>10,.1f} -> {'OK' if ok else 'MISMATCH'}")
    print(f"несовпадений: {bad} из {n}")
    return bad


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
