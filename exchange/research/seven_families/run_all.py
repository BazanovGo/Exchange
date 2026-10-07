"""Бэктест и оптимизация семи семейств. Отбор только на in-sample."""
from __future__ import annotations

import itertools, json, logging, pickle, warnings
logging.disable(logging.INFO); warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import families as F
import harness as H
from exchange.backtest.metrics import performance_metrics, sharpe_ratio
from exchange.stats.cv import CombinatorialPurgedCV, WalkForward

IS_FRACTION = 0.60
EMBARGO = 10
MIN_TRADES = 10

GRIDS = {
    "1_tsmom":          [{"lookback": k} for k in (20, 40, 60, 90, 120, 180, 250)],
    "2_dual_ma":        [{"fast": f, "slow": s}
                         for f, s in itertools.product((10, 20, 40, 60), (60, 100, 150, 200, 250))
                         if f < s],
    "3_breakout":       [{"entry": e, "exit_": x}
                         for e, x in itertools.product((20, 40, 55, 80, 120), (10, 20, 40))
                         if x < e],
    "4_cross_sectional":[{"lookback": k} for k in (20, 60, 120, 250)],
    "5_carry":          [{"smooth": s, "threshold": t}
                         for s, t in itertools.product((1, 3, 6), (0.0, 0.05, 0.15))],
    "6_portfolio_ma":   [{"fast": f, "slow": s}
                         for f, s in itertools.product((20, 40, 60), (100, 150, 250))],
    "7_mom_reversal":   [{"momentum": m, "reversal": r}
                         for m, r in itertools.product((60, 120, 250), (375, 500, 750))],
}


def positions(bars, ppy, signal, band=H.REBALANCE_BAND, legs=1):
    units = H.vol_target_units(bars, ppy) / legs
    return H.apply_band(signal * units, band)


def run_single(family, market, params, bars, ppy, *, scenario="base", latency=0):
    signal = F.SINGLE[family](bars, **params)
    target = positions(bars, ppy, signal)
    result, metrics = H.backtest(bars, target, symbol=market, ppy=ppy,
                                 scenario=scenario, latency=latency)
    return result, metrics


def run_panel(family, params, panel, ppys, *, scenario="base", latency=0):
    """Портфельная стратегия: каждая нога через движок, P&L складываются."""
    signals = F.PANEL[family](panel, **params)
    legs = len(panel)
    index, equity, costs, trades = None, None, 0.0, 0
    for root, bars in panel.items():
        target = positions(bars, ppys[root], signals[root], legs=legs)
        result, metrics = H.backtest(bars, target, symbol=root, ppy=ppys[root],
                                     scenario=scenario, latency=latency)
        pnl = result.equity - H.CAPITAL
        equity = pnl if equity is None else equity.add(pnl, fill_value=0.0)
        index = result.equity.index if index is None else index.union(result.equity.index)
        costs += result.total_costs
        trades += int(metrics.n_trades)
    equity = (H.CAPITAL + equity.reindex(index).ffill().fillna(0.0)).rename("equity")
    returns = equity.pct_change().fillna(0.0)
    ppy = float(np.mean(list(ppys.values())))
    return equity, returns, ppy, costs, trades


def panel_metrics(equity, returns, ppy, costs, trades):
    from exchange.backtest.metrics import (
        annualised_volatility, cagr, max_drawdown, sortino_ratio)
    return {
        "sharpe_ratio": sharpe_ratio(returns, periods_per_year=ppy),
        "sortino_ratio": sortino_ratio(returns, periods_per_year=ppy),
        "cagr": cagr(equity, periods_per_year=ppy),
        "annualised_volatility": annualised_volatility(returns, periods_per_year=ppy),
        "max_drawdown": max_drawdown(equity),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "n_trades": trades,
        "total_costs": costs,
    }


def split(index):
    n = len(index)
    cut = int(n * IS_FRACTION)
    return index[:cut], index[cut + EMBARGO:]


def evaluate_cell(family, market, panel_all, universe):
    """Полный цикл по одной ячейке «семейство x рынок»."""
    is_panel = family in F.PANEL
    grid = GRIDS[family]

    if is_panel:
        common = panel_all["NG"].index.intersection(panel_all["BR"].index)
        panel = {r: b.loc[common] for r, b in panel_all.items()}
        ppys = {r: universe[r]["ppy"] for r in panel}
        index = common
    else:
        bars = panel_all[market]
        ppy = universe[market]["ppy"]
        index = bars.index

    is_idx, oos_idx = split(index)
    rows, full_returns = [], {}

    for params in grid:
        label = ",".join(f"{k}={v}" for k, v in params.items())
        try:
            if is_panel:
                eq_f, ret_f, ppy_f, c_f, t_f = run_panel(family, params, panel, ppys)
                eq_i, ret_i, ppy_i, c_i, t_i = run_panel(
                    family, params, {r: b.loc[is_idx] for r, b in panel.items()}, ppys)
                m_is = panel_metrics(eq_i, ret_i, ppy_i, c_i, t_i)
                full_returns[label] = ret_f
            else:
                res_f, _ = run_single(family, market, params, bars, ppy)
                res_i, m = run_single(family, market, params, bars.loc[is_idx], ppy)
                m_is = {"sharpe_ratio": m.sharpe_ratio, "n_trades": m.n_trades,
                        "cagr": m.cagr, "max_drawdown": m.max_drawdown,
                        "total_costs": m.total_costs}
                full_returns[label] = res_f.returns
        except Exception as exc:
            rows.append({"params": label, "error": str(exc)}); continue
        rows.append({"params": label, "is_sharpe": m_is["sharpe_ratio"],
                     "is_trades": m_is["n_trades"], "is_cagr": m_is["cagr"],
                     "is_max_dd": m_is["max_drawdown"]})

    sweep = pd.DataFrame(rows)
    eligible = sweep[(sweep.get("is_trades", 0) >= MIN_TRADES) & sweep["is_sharpe"].notna()]
    if eligible.empty:
        eligible = sweep.dropna(subset=["is_sharpe"])
    best_row = eligible.loc[eligible["is_sharpe"].idxmax()]
    best = next(p for p in grid if ",".join(f"{k}={v}" for k, v in p.items()) == best_row["params"])

    # --- единственная оценка out-of-sample -----------------------------
    out = {"family": family, "market": market, "n_configs": len(grid),
           "best_params": best_row["params"], "is_sharpe": float(best_row["is_sharpe"]),
           "n_bars": len(index), "start": str(index[0].date()), "end": str(index[-1].date()),
           "is_end": str(is_idx[-1].date()), "oos_start": str(oos_idx[0].date())}

    if is_panel:
        eq_o, ret_o, ppy_o, c_o, t_o = run_panel(
            family, best, {r: b.loc[oos_idx] for r, b in panel.items()}, ppys)
        m = panel_metrics(eq_o, ret_o, ppy_o, c_o, t_o)
        oos_returns, oos_equity, ppy_use = ret_o, eq_o, ppy_o
        out.update({k: m[k] for k in m})
        out["profit_factor"] = np.nan; out["win_rate"] = np.nan
    else:
        res_o, m = run_single(family, market, best, bars.loc[oos_idx], ppy)
        oos_returns, oos_equity, ppy_use = res_o.returns, res_o.equity, ppy
        out.update({"sharpe_ratio": m.sharpe_ratio, "sortino_ratio": m.sortino_ratio,
                    "cagr": m.cagr, "annualised_volatility": m.annualised_volatility,
                    "max_drawdown": m.max_drawdown, "total_return": m.total_return,
                    "n_trades": m.n_trades, "total_costs": m.total_costs,
                    "profit_factor": m.profit_factor, "win_rate": m.win_rate})

    out["calmar"] = out["cagr"] / abs(out["max_drawdown"]) if out["max_drawdown"] else np.nan
    out["cost_share"] = (out["total_costs"] /
                         max(out["total_costs"] + (oos_equity.iloc[-1] - oos_equity.iloc[0]), 1e-9))
    out["ppy"] = ppy_use

    # --- стресс-лестница и латентность (на OOS) -------------------------
    for scenario in ("optimistic", "conservative", "stressed"):
        if is_panel:
            _, r_s, p_s, _, _ = run_panel(family, best,
                                          {r: b.loc[oos_idx] for r, b in panel.items()},
                                          ppys, scenario=scenario)
            out[f"sharpe_{scenario}"] = sharpe_ratio(r_s, periods_per_year=p_s)
        else:
            _, m_s = run_single(family, market, best, bars.loc[oos_idx], ppy, scenario=scenario)
            out[f"sharpe_{scenario}"] = m_s.sharpe_ratio
    if is_panel:
        _, r_l, p_l, _, _ = run_panel(family, best, {r: b.loc[oos_idx] for r, b in panel.items()},
                                      ppys, latency=1)
        out["sharpe_latency1"] = sharpe_ratio(r_l, periods_per_year=p_l)
    else:
        _, m_l = run_single(family, market, best, bars.loc[oos_idx], ppy, latency=1)
        out["sharpe_latency1"] = m_l.sharpe_ratio

    return out, sweep, pd.DataFrame(full_returns), oos_returns, best, (panel if is_panel else bars)


if __name__ == "__main__":
    universe = H.load_universe()
    panel_all = {r: p["bars"] for r, p in universe.items()}
    cells, sweeps, matrices, oos_returns, winners = [], {}, {}, {}, {}

    jobs = [(f, m) for f in F.SINGLE for m in ("NG", "BR")] + [(f, "NG+BR") for f in F.PANEL]
    for family, market in sorted(jobs):
        out, sweep, matrix, ret, best, _ = evaluate_cell(family, market, panel_all, universe)
        key = f"{family}|{market}"
        cells.append(out); sweeps[key] = sweep; matrices[key] = matrix
        oos_returns[key] = ret; winners[key] = best
        print(f"{key:28} конфигураций {out['n_configs']:3d} | IS {out['is_sharpe']:+.3f} "
              f"| OOS {out['sharpe_ratio']:+.3f} | сделок {out['n_trades']:5d}", flush=True)

    with open("results.pkl", "wb") as fh:
        pickle.dump({"cells": cells, "sweeps": sweeps, "matrices": matrices,
                     "oos_returns": oos_returns, "winners": winners}, fh)
    print("\nвсего конфигураций:", sum(c["n_configs"] for c in cells))
