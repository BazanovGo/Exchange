"""Сверка событийного ядра с VectorBT на реальных данных.

Сигналы строятся одинаково (пересечения и состояние наклона), а исполнение
и учёт сделок — независимо: ядро против ``vbt.Portfolio.from_signals``.
Совпасть должны число сделок, бары входа/выхода, цены и PnL каждой сделки.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

import kernel as K
import ma as M
from data import load_market
from grid import FAST_PERIODS, METHODS, SLOW_PERIODS, TYPES, slow_state


def vbt_trades(mk, ev_bar, ev_dir, state, model):
    n = mk.n
    cu = np.zeros(n, bool); cd = np.zeros(n, bool)
    cu[ev_bar[ev_dir > 0]] = True
    cd[ev_bar[ev_dir < 0]] = True
    le = cu & np.isin(state, (1, 0))
    se = cd & np.isin(state, (-1, 0))
    if model == 0:
        lx, sx = cd, cu
    else:
        lx, sx = state == -1, state == 1
    shift = lambda a: np.r_[False, a[:-1]]          # решение на t → исполнение на t+1
    idx = pd.RangeIndex(n)
    # VectorBT требует цены > 0, а аддитивный ряд NG уходит в минус. Сдвиг на
    # константу не меняет ни одной разности, то есть ни одного PnL.
    off = max(0.0, 1.0 - float(min(mk.pnl_open.min(), mk.pnl_close.min())))
    pf = vbt.Portfolio.from_signals(
        close=pd.Series(mk.pnl_close + off, idx), price=pd.Series(mk.pnl_open + off, idx),
        entries=pd.Series(shift(le), idx), exits=pd.Series(shift(lx), idx),
        short_entries=pd.Series(shift(se), idx), short_exits=pd.Series(shift(sx), idx),
        size=1.0, size_type="amount", init_cash=1e12, fees=0.0, slippage=0.0,
        upon_opposite_entry="ignore", accumulate=False, freq="5min",
    )
    r = pf.trades.records
    out = np.column_stack([
        r["entry_idx"], r["exit_idx"], np.where(r["direction"] == 0, 1, -1),
        r["entry_price"] - off, r["exit_price"] - off, r["pnl"] * mk.multiplier,
    ]).astype(float)
    return out[np.argsort(out[:, 0], kind="stable")]


def main(n_configs=40, seed=20261014):
    rng = np.random.default_rng(seed)
    report = []
    for ticker in ("NG", "BR"):
        mk = load_market(ticker)
        atr14 = M.atr(mk.high, mk.low, mk.close, 14)
        for _ in range(n_configs):
            ft, st = rng.choice(TYPES), rng.choice(TYPES)
            fp, sp = int(rng.choice(FAST_PERIODS)), int(rng.choice(SLOW_PERIODS))
            method = str(rng.choice(METHODS)); model = int(rng.integers(0, 2))
            k = int(rng.choice([1, 5, 20])); cov = float(rng.choice([0.0, 0.25, 0.5]))
            fast = M.moving_average(ft, mk.close, mk.volume, fp)
            ev_bar, ev_dir = K.cross_events(mk.close, fast, mk.is_roll)
            slow = M.moving_average(st, mk.close, mk.volume, sp)
            state, _ = slow_state(mk, slow, method, k, cov, atr14)
            nd, nu = K.next_state_index(state, -1), K.next_state_index(state, 1)
            stats = np.zeros((3, K.N_STATS)); monthly = np.zeros(len(mk.months))
            log = np.zeros((len(ev_bar) + 1, 6))
            ntr = K.simulate(ev_bar, ev_dir, state, nd, nu, mk.pnl_open, mk.pnl_close,
                             mk.month, mk.oos_bar, mk.multiplier, model, stats, monthly, log)
            ours = log[:ntr]
            ref = vbt_trades(mk, ev_bar, ev_dir, state, model)
            same_n = len(ref) == ntr
            max_diff = float(np.abs(ours - ref).max()) if same_n and ntr else (0.0 if same_n else np.inf)
            ok = same_n and max_diff < 1e-6
            report.append({"ticker": ticker, "fast": f"{ft}{fp}", "slow": f"{st}{sp}",
                           "method": method, "exit": "AB"[model], "k": k, "coverage": cov,
                           "trades_kernel": int(ntr), "trades_vbt": int(len(ref)),
                           "max_abs_diff": max_diff, "match": bool(ok),
                           "pnl_kernel": float(stats[0, K.PNL]),
                           "pnl_vbt": float(ref[:, 5].sum()) if len(ref) else 0.0})
            print(f"{ticker} {ft}{fp:>3} / {st}{sp:>3} {method} exit {'AB'[model]} k={k} cov={cov}: "
                  f"сделок {ntr} vs {len(ref)}, max diff {max_diff:.2e} -> {'OK' if ok else 'MISMATCH'}",
                  flush=True)
    df = pd.DataFrame(report)
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("validation_vbt.csv")
    df.to_csv(out, index=False)
    print(f"\nсовпало: {df['match'].sum()} из {len(df)}; сделок сверено: {df['trades_vbt'].sum():,}")
    return df


if __name__ == "__main__":
    main()
