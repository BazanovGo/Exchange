"""Финалисты через VectorBT: полная статистика, месяцы/годы, Monte Carlo.

На вход — конфигурации, отобранные по in-sample (picks.csv анализа этапа).
Каждая прогоняется через ``vbt.Portfolio.from_signals`` с размером позиции,
равным множителю контракта, — так PnL VectorBT сразу в USD за 1 контракт.

Капитал. Шарп и Сортино считаются по дневному PnL с переоценкой открытой
позиции (mark-to-market) и от капитала не зависят: это отношение денег к
деньгам. Для процентных величин капитал определён явно — номинал одного
контракта по максимальной цене периода (позиция без плеча); это указано в
каждой таблице, где он используется.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt

import kernel as K
import ma as M
from data import load_market
from grid import slow_state

WORK = Path(os.environ.get("MA5M_WORK", "work"))
TRADING_DAYS = 252


def signals(mk, cfg, slope):
    fast = M.moving_average(cfg["fast_type"], mk.close, mk.volume, int(cfg["fast_period"]))
    ev_bar, ev_dir = K.cross_events(mk.close, fast, mk.is_roll)
    slow = M.moving_average(cfg["slow_type"], mk.close, mk.volume, int(cfg["slow_period"]))
    atr = M.atr(mk.high, mk.low, mk.close, int(slope["atr_period"])) if cfg["method"] == "B_atr" else None
    state, info = slow_state(mk, slow, cfg["method"], int(slope["k"]), float(slope["coverage"]), atr)
    return ev_bar, ev_dir, state, info


def run_vbt(mk, cfg, slope):
    ev_bar, ev_dir, state, info = signals(mk, cfg, slope)
    n = mk.n
    cu = np.zeros(n, bool); cd = np.zeros(n, bool)
    cu[ev_bar[ev_dir > 0]] = True; cd[ev_bar[ev_dir < 0]] = True
    le = cu & np.isin(state, (1, 0)); se = cd & np.isin(state, (-1, 0))
    if cfg["exit"] == "A_fast_cross":
        lx, sx = cd, cu
    else:
        lx, sx = state == -1, state == 1
    sh = lambda a: pd.Series(np.r_[False, a[:-1]], index=mk.index)
    off = max(0.0, 1.0 - float(min(mk.pnl_open.min(), mk.pnl_close.min())))
    capital = float(np.max(mk.raw_close) * mk.multiplier)
    pf = vbt.Portfolio.from_signals(
        close=pd.Series(mk.pnl_close + off, mk.index), price=pd.Series(mk.pnl_open + off, mk.index),
        entries=sh(le), exits=sh(lx), short_entries=sh(se), short_exits=sh(sx),
        size=mk.multiplier, size_type="amount", init_cash=capital * 1e3, fees=0.0,
        slippage=0.0, upon_opposite_entry="ignore", accumulate=False, freq="5min")
    return pf, capital, info


def monte_carlo(trade_pnl: np.ndarray, n_sims=5000, seed=0):
    """Бутстрэп сделок (с возвращением) и перестановка порядка."""
    rng = np.random.default_rng(seed)
    n = len(trade_pnl)
    tot = np.empty(n_sims); dd_b = np.empty(n_sims); dd_p = np.empty(n_sims)
    for i in range(n_sims):
        x = trade_pnl[rng.integers(0, n, n)]
        eq = np.cumsum(x); tot[i] = eq[-1]
        dd_b[i] = np.max(np.maximum.accumulate(np.r_[0, eq])[1:] - eq)
        y = rng.permutation(trade_pnl); eq2 = np.cumsum(y)
        dd_p[i] = np.max(np.maximum.accumulate(np.r_[0, eq2])[1:] - eq2)
    q = lambda a: {f"p{p}": float(np.percentile(a, p)) for p in (5, 50, 95)}
    return {"total_pnl": q(tot), "p_total_negative": float((tot < 0).mean()),
            "max_dd_bootstrap": q(dd_b), "max_dd_permutation": q(dd_p)}


def describe(mk, pf, capital, cfg, slope, info):
    val = pf.value() - pf.init_cash
    # Дневной PnL с переоценкой: первый день — от нуля, торговые дни выходных
    # (у MOEX они бывают) не выбрасываются, иначе сумма не сходится с итогом.
    eod = val.resample("1D").last().dropna()
    daily = eod.diff()
    daily.iloc[0] = eod.iloc[0]
    days_per_year = len(daily) / ((daily.index[-1] - daily.index[0]).days / 365.25)
    oos_start = mk.index[mk.oos_bar]
    tr = pf.trades.records_readable
    pnl = tr["PnL"].to_numpy(float)
    exit_t = pd.to_datetime(tr["Exit Timestamp"])
    is_tr = exit_t < oos_start
    def block(d, t):
        if len(d) < 2 or d.std() == 0:
            return {}
        neg = d[d < 0]
        eq = d.cumsum()
        return {
            "pnl": float(d.sum()), "trades": int(len(t)), "sharpe_daily": float(d.mean() / d.std() * np.sqrt(days_per_year)),
            "sortino_daily": float(d.mean() / np.sqrt((d.clip(upper=0) ** 2).mean()) * np.sqrt(days_per_year)) if len(neg) else None,
            "max_dd_mtm": float((eq.cummax() - eq).max()),
            "win_rate": float((t > 0).mean()) if len(t) else None,
            "profit_factor": float(t[t > 0].sum() / -t[t < 0].sum()) if (t < 0).any() else None,
        }
    out = {
        "config": cfg, "slope": slope, "eps": info[0], "neutral_is": info[1], "neutral_oos": info[2],
        "capital_usd": capital, "trading_days_per_year": days_per_year,
        "full": block(daily, pnl),
        "is": block(daily[daily.index < oos_start], pnl[is_tr.to_numpy()]),
        "oos": block(daily[daily.index >= oos_start], pnl[~is_tr.to_numpy()]),
        "exposure": float((pf.assets() != 0).mean()),
        "avg_duration_bars": float((tr["Exit Timestamp"].map(mk.index.get_loc) -
                                    tr["Entry Timestamp"].map(mk.index.get_loc)).mean()) if len(tr) else None,
        "monte_carlo": monte_carlo(pnl) if len(pnl) > 10 else None,
        "monte_carlo_oos": monte_carlo(pnl[~is_tr.to_numpy()]) if (~is_tr).sum() > 10 else None,
    }
    monthly = val.resample("ME").last().diff()
    monthly.iloc[0] = val.resample("ME").last().iloc[0]
    yearly = val.resample("YE").last().diff()
    yearly.iloc[0] = val.resample("YE").last().iloc[0]
    return out, monthly, yearly, tr, val


def main(stage: str, tickers):
    slope_path = WORK / stage / "slope_params.json"
    for t in tickers:
        mk = load_market(t)
        base = WORK / "analysis" / stage / t
        picks = pd.read_csv(base / "picks.csv")
        picks = picks[~picks["scope"].str.startswith("family:")].drop_duplicates("config")
        meta = json.loads((WORK / stage / t / "meta.json").read_text())
        slopes = json.loads(slope_path.read_text())[t] if slope_path.exists() else None
        rows, curves = [], {}
        fdir = base / "finalists"; fdir.mkdir(exist_ok=True)
        for _, p in picks.iterrows():
            cfg = {k: p[k] for k in ("method", "slow_type", "slow_period", "exit", "fast_type", "fast_period")}
            slope = slopes[cfg["method"]] if slopes else {"k": meta["k"], "coverage": meta["coverage"],
                                                          "atr_period": meta["atr_period"]}
            pf, cap, info = run_vbt(mk, cfg, slope)
            d, monthly, yearly, tr, val = describe(mk, pf, cap, cfg, slope, info)
            d["scope"], d["rule"], d["grid_config"] = p["scope"], p["rule"], int(p["config"])
            d["grid_pnl_full"] = float(p["pnl_full"])
            name = f"cfg{int(p['config'])}"
            monthly.to_frame("pnl_usd").to_csv(fdir / f"{name}_monthly.csv")
            yearly.to_frame("pnl_usd").to_csv(fdir / f"{name}_yearly.csv")
            tr.to_parquet(fdir / f"{name}_trades.parquet", index=False)
            val.resample("1D").last().dropna().to_frame("cum_pnl_usd").to_parquet(fdir / f"{name}_equity.parquet")
            rows.append(d)
            print(f"  {t} {p['scope']:>26} {p['rule']:<14} {cfg['fast_type']}{cfg['fast_period']}/"
                  f"{cfg['slow_type']}{cfg['slow_period']} {cfg['method']} {cfg['exit']}: "
                  f"vbt PnL {d['full'].get('pnl', 0):,.0f} vs сетка {p['pnl_full']:,.0f} | "
                  f"OOS {d['oos'].get('pnl', float('nan')):,.0f} Sharpe {d['oos'].get('sharpe_daily', float('nan')):.2f}",
                  flush=True)
        (base / "finalists.json").write_text(json.dumps(rows, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
