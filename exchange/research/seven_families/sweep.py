"""Полный перебор сеток семи семейств: метрики каждой конфигурации в выборке и вне её.

Отбор в отчёте шёл только по in-sample; здесь для каждой из 125 конфигураций
считаются и IS-, и OOS-метрики — чтобы показать всю карту перебора, а не одного
победителя. Прогоны те же, что в run_all.evaluate_cell: отдельный IS-прогон и
отдельный OOS-прогон, начинающийся без позиции. Победители сверяются с
docs/RESULTS_SEVEN_FAMILIES.md; при расхождении скрипт падает.

    EXCHANGE_MOEX_DATA=/path/to/moex python sweep.py
    -> results/sweeps.csv, results/winner_equity.csv
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import families as F
import harness as H
import numpy as np
import pandas as pd
import run_all as RA

from exchange.backtest.metrics import cagr, max_drawdown, sharpe_ratio

logging.disable(logging.INFO)
warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
REPORT = HERE.parents[1] / "docs" / "RESULTS_SEVEN_FAMILIES.md"


def label(params: dict) -> str:
    return ",".join(f"{k}={v}" for k, v in params.items())


def run(family, market, params, panel_all, ppys, common, idx):
    """Один прогон на отрезке idx: (доходности, кривая капитала, число сделок)."""
    if family in F.PANEL:
        panel = {r: b.loc[common].loc[idx] for r, b in panel_all.items()}
        signals = F.PANEL[family](panel, **params)
        pnl, trades = None, 0
        for r, bars in panel.items():
            target = RA.positions(bars, ppys[r], signals[r], legs=len(panel))
            result, metrics = H.backtest(bars, target, symbol=r, ppy=ppys[r])
            leg = result.equity - H.CAPITAL
            pnl = leg if pnl is None else pnl.add(leg, fill_value=0.0)
            trades += int(metrics.n_trades)
        equity = (H.CAPITAL + pnl.ffill().fillna(0.0)).rename("equity")
        ppy = float(np.mean(list(ppys.values())))
    else:
        bars = panel_all[market].loc[idx]
        result, metrics = RA.run_single(family, market, params, bars, ppys[market])
        equity, trades, ppy = result.equity, int(metrics.n_trades), ppys[market]
    returns = equity.pct_change().fillna(0.0)
    return returns, equity, trades, ppy


def metrics(returns, equity, ppy, trades) -> dict:
    return {"sharpe": sharpe_ratio(returns, periods_per_year=ppy), "cagr": cagr(equity, periods_per_year=ppy),
            "max_dd": max_drawdown(equity), "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
            "trades": trades}


def report_winners() -> pd.DataFrame:
    lines = REPORT.read_text().splitlines()
    def table(col):
        i = next(k for k, ln in enumerate(lines) if ln.startswith("| ячейка |") and f"| {col} |" in ln)
        rows = []
        for ln in lines[i:]:
            if not ln.startswith("|"):
                break
            rows.append([c.strip() for c in ln.strip("|").split("|")])
        return pd.DataFrame(rows[2:], columns=rows[0])
    a = table("лучшая")[["ячейка", "лучшая"]]
    b = table("IS Sharpe")[["ячейка", "IS Sharpe", "OOS Sharpe"]]
    df = a.merge(b, on="ячейка")
    df["family"] = df["ячейка"].str.split(" ").str[0].map(
        {"1": "1_tsmom", "2": "2_dual_ma", "3": "3_breakout", "4": "4_cross_sectional",
         "5": "5_carry", "6": "6_portfolio_ma", "7": "7_mom_reversal"})
    df["market"] = df["ячейка"].str.split(" · ").str[-1]
    return df


def main():
    OUT.mkdir(exist_ok=True)
    universe = H.load_universe()
    panel_all = {r: p["bars"] for r, p in universe.items()}
    ppys = {r: universe[r]["ppy"] for r in panel_all}
    common = panel_all["NG"].index.intersection(panel_all["BR"].index)
    jobs = [(f, m) for f in F.SINGLE for m in ("NG", "BR")] + [(f, "NG+BR") for f in F.PANEL]
    rows, curves = [], []
    winners = report_winners().set_index(["family", "market"])
    for family, market in sorted(jobs):
        index = common if family in F.PANEL else panel_all[market].index
        is_idx, oos_idx = RA.split(index)
        for params in RA.GRIDS[family]:
            r_is = run(family, market, params, panel_all, ppys, common, is_idx)
            r_oos = run(family, market, params, panel_all, ppys, common, oos_idx)
            m_is = metrics(r_is[0], r_is[1], r_is[3], r_is[2])
            m_oos = metrics(r_oos[0], r_oos[1], r_oos[3], r_oos[2])
            rows.append({"family": family, "market": market, "params": label(params), **params,
                         **{f"is_{k}": v for k, v in m_is.items()}, **{f"oos_{k}": v for k, v in m_oos.items()}})
            w = winners.loc[(family, market)]
            if label(params) == w["лучшая"]:
                # Сверка с отчётом: Шарп победителя в выборке и вне её.
                for got, ref in ((m_is["sharpe"], w["IS Sharpe"]), (m_oos["sharpe"], w["OOS Sharpe"])):
                    if abs(got - float(ref)) > 0.0015:
                        raise SystemExit(f"{family}|{market} {label(params)}: Шарп {got:.4f} против отчёта {ref}")
                eq = pd.concat([(1 + r_is[0]).cumprod(), (1 + r_oos[0]).cumprod() * (1 + r_is[0]).prod()])
                curves.append(pd.DataFrame({"family": family, "market": market, "date": eq.index,
                                            "equity": eq.to_numpy() * H.CAPITAL,
                                            "period": ["IS"] * len(r_is[0]) + ["OOS"] * len(r_oos[0])}))
        sub = pd.DataFrame([r for r in rows if r["family"] == family and r["market"] == market])
        eligible = sub[sub["is_trades"] >= RA.MIN_TRADES]
        best = (eligible if len(eligible) else sub).sort_values("is_sharpe").iloc[-1]["params"]
        if best != winners.loc[(family, market)]["лучшая"]:
            raise SystemExit(f"{family}|{market}: лучший по IS {best}, в отчёте {winners.loc[(family, market)]['лучшая']}")
        print(f"{family:18} {market:6} конфигураций {len(sub):3d} | победитель {best:28} совпал с отчётом")
    df = pd.DataFrame(rows)
    df["is_winner"] = [label({k: r[k] for k in RA.GRIDS[r["family"]][0]}) == winners.loc[(r["family"], r["market"])]["лучшая"]
                       for r in rows]
    df.to_csv(OUT / "sweeps.csv", index=False)
    eq = pd.concat(curves, ignore_index=True)
    eq["date"] = pd.to_datetime(eq["date"]).dt.tz_localize(None).dt.date
    eq.to_csv(OUT / "winner_equity.csv", index=False)
    print(f"{len(df)} конфигураций -> {OUT / 'sweeps.csv'}")


if __name__ == "__main__":
    main()
