"""Дополнения к анализу: чувствительность к издержкам, эталон для walk-forward,
годовое распределение отобранных, автокорреляция месячного PnL.

Издержки. Задание требует считать без комиссий и проскальзывания, и основные
результаты так и посчитаны. Но средняя сделка в этой сетке — доли тика или
единицы тиков, то есть того же порядка, что минимальная реальная цена сделки
(спред в 1 тик). Поэтому рядом с нулём показано, что остаётся при 0.5, 1 и 2
тиках на полную сделку: это вычитание ``сделки × c × стоимость тика`` из
итогов IS и OOS, без пересчёта сетки.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from results import load_grid

WORK = Path(os.environ.get("MA5M_WORK", "work"))
COSTS = (0.0, 0.5, 1.0, 2.0)


def run(stage, ticker):
    g = load_grid(WORK / stage / ticker)
    base = WORK / "analysis" / stage / ticker
    df = pd.read_parquet(base / "summary.parquet", columns=[
        "config", "pnl_is", "pnl_oos", "trades_is", "trades_oos", "exit", "method",
        "fast_type", "slow_type", "fast_period", "slow_period", "plateau_sharpe_is", "sharpe_m_is"])
    tick_usd = g.meta["tick"] * g.meta["multiplier"]
    out = {"tick_usd": tick_usd, "costs": {}}
    for c in COSTS:
        is_net = df["pnl_is"] - df["trades_is"] * c * tick_usd
        oos_net = df["pnl_oos"] - df["trades_oos"] * c * tick_usd
        ok = df["trades_is"] >= 100
        best = is_net.where(ok).idxmax()
        # отбор по чистому IS внутри каждого выхода
        by_exit = {}
        for e in df["exit"].cat.categories:
            m = ok & (df["exit"] == e)
            b = is_net.where(m).idxmax()
            by_exit[e] = {"config": int(df.loc[b, "config"]), "is_net": float(is_net[b]), "oos_net": float(oos_net[b]),
                          "share_pos_is": float((is_net[df["exit"] == e] > 0).mean()),
                          "share_pos_oos": float((oos_net[df["exit"] == e] > 0).mean())}
        top = is_net.where(ok).nlargest(max(1, int(ok.sum() * 0.01))).index
        out["costs"][str(c)] = {
            "share_pos_is": float((is_net > 0).mean()), "share_pos_oos": float((oos_net > 0).mean()),
            "share_pos_both": float(((is_net > 0) & (oos_net > 0)).mean()),
            "median_is": float(is_net.median()), "median_oos": float(oos_net.median()),
            "best_is_config": int(df.loc[best, "config"]), "best_is_net": float(is_net[best]),
            "best_is_oos_net": float(oos_net[best]),
            "top1pct_mean_oos_net": float(oos_net[top].mean()),
            "top1pct_share_pos_oos": float((oos_net[top] > 0).mean()),
            "by_exit": by_exit,
        }
    # эталон для walk-forward: средняя по всем конфигурациям месячная прибыль
    N, T = g.monthly.shape
    acc = np.zeros(T)
    for a in range(0, N, 400_000):
        acc += np.asarray(g.monthly[a:a + 400_000], dtype=np.float64).sum(0)
    avg_month = acc / N
    np.save(base / "avg_config_monthly.npy", avg_month)
    wf = {}
    for crit in ("pnl", "sharpe", "plateau"):
        m = np.load(base / f"wf_{crit}_months.npy")
        bench = avg_month[-len(m):]
        diff = m - bench
        wf[crit] = {"wf_total": float(m.sum()), "avg_config_total": float(bench.sum()),
                    "excess_total": float(diff.sum()),
                    "excess_t": float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))) if diff.std() > 0 else None}
    out["walk_forward_vs_average_config"] = wf
    # автокорреляция месячного PnL (для калибровки SPA): выборка конфигураций
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(N, 20_000, replace=False))
    x = np.asarray(g.monthly[idx], dtype=np.float64)
    x = x - x.mean(1, keepdims=True)
    ac1 = (x[:, 1:] * x[:, :-1]).sum(1) / np.maximum((x * x).sum(1), 1e-12)
    out["monthly_ac1_median"] = float(np.median(ac1))
    out["monthly_ac1_p90"] = float(np.quantile(ac1, 0.9))
    # годы для отобранных
    picks = pd.read_csv(base / "picks.csv")
    months = pd.PeriodIndex(g.meta["months"], freq="M")
    years = months.year
    rows = []
    for _, p in picks[~picks["scope"].str.startswith("family:")].iterrows():
        mo = pd.Series(np.asarray(g.monthly[int(p["config"])], dtype=np.float64), index=months)
        yr = mo.groupby(years).sum()
        rows.append({"scope": p["scope"], "rule": p["rule"], "config": int(p["config"]),
                     **{str(y): float(v) for y, v in yr.items()}})
    pd.DataFrame(rows).to_csv(base / "picks_yearly.csv", index=False)
    (base / "extras.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "costs"}, indent=1))
    for c, v in out["costs"].items():
        print(f"  издержки {c} тика: прибыльных IS {v['share_pos_is']:.3f} OOS {v['share_pos_oos']:.3f} | "
              f"лучший по IS: IS {v['best_is_net']:,.0f} -> OOS {v['best_is_oos_net']:,.0f} | "
              f"верхний 1%: OOS {v['top1pct_mean_oos_net']:,.0f}, доля OOS>0 {v['top1pct_share_pos_oos']:.2f}")
    return out


if __name__ == "__main__":
    for t in sys.argv[2:]:
        run(sys.argv[1], t)
