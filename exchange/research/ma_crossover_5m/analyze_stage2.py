"""Анализ этапа 2: какие параметры наклона лучше и насколько это устойчиво.

Выбор настройки для этапа 3 — только по in-sample: медиана месячного Шарпа
по всей решётке (все типы, периоды решётки, оба выхода). Медиана, а не
максимум: настройка должна улучшать семейство, а не одну удачную точку.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

import kernel as K
from results import monthly_moments, sharpe_monthly

WORK = Path(os.environ.get("MA5M_WORK", "work"))


def analyze(ticker: str, stage: str = "stage2"):
    base = WORK / stage / ticker
    meta = json.loads((base / "meta.json").read_text())
    stats = np.load(base / "stats.npy", mmap_mode="r")
    monthly = np.load(base / "monthly.npy", mmap_mode="r")
    info = np.load(base / "slow_info.npy")              # (slow_jobs, 3): eps, neutral IS, neutral OOS
    shape = tuple(meta["shape"])                         # (6, 9, 200, 2, 6, 9)
    oos = meta["oos_month"]; T = monthly.shape[1]
    mu_is, sd_is, _ = monthly_moments(type("G", (), {"n": stats.shape[0], "monthly": monthly}), 0, oos)
    mu_oos, sd_oos, _ = monthly_moments(type("G", (), {"n": stats.shape[0], "monthly": monthly}), oos, T)
    sh_is = sharpe_monthly(mu_is, sd_is).reshape(shape)
    sh_oos = sharpe_monthly(mu_oos, sd_oos).reshape(shape)
    pnl_is = np.asarray(stats[:, 1, K.PNL]).reshape(shape)
    pnl_oos = np.asarray(stats[:, 2, K.PNL]).reshape(shape)
    trades = np.asarray(stats[:, 0, K.NTR]).reshape(shape)
    n_set = len(meta["settings"])
    inf = info.reshape(6, 9, n_set, 3)
    rows = []
    for si, (method, k, cov, atr) in enumerate(meta["settings"]):
        a = sh_is[:, :, si]; b = sh_oos[:, :, si]
        rows.append({
            "setting": si, "method": method, "k": k, "coverage": cov, "atr_period": atr,
            "median_sharpe_is": float(np.nanmedian(a)), "share_pos_is": float((pnl_is[:, :, si] > 0).mean()),
            "median_sharpe_oos": float(np.nanmedian(b)), "share_pos_oos": float((pnl_oos[:, :, si] > 0).mean()),
            "median_trades": float(np.median(trades[:, :, si])),
            "neutral_is": float(np.nanmean(inf[:, :, si, 1])), "neutral_oos": float(np.nanmean(inf[:, :, si, 2])),
            "eps_median": float(np.nanmedian(inf[:, :, si, 0])),
            # по выходам отдельно
            "median_sharpe_is_exitA": float(np.nanmedian(a[:, :, 0])),
            "median_sharpe_is_exitB": float(np.nanmedian(a[:, :, 1])),
            "median_sharpe_oos_exitA": float(np.nanmedian(b[:, :, 0])),
            "median_sharpe_oos_exitB": float(np.nanmedian(b[:, :, 1])),
        })
    df = pd.DataFrame(rows)
    out = WORK / "analysis" / stage / ticker
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "settings.csv", index=False)
    res = {"ticker": ticker}
    best = {}
    for m in ("A_pct", "B_atr"):
        d = df[df["method"] == m]
        r = d.loc[d["median_sharpe_is"].idxmax()]
        best[m] = {"k": int(r["k"]), "coverage": float(r["coverage"]), "atr_period": int(r["atr_period"]) or 14,
                   "median_sharpe_is": float(r["median_sharpe_is"]),
                   "median_sharpe_oos": float(r["median_sharpe_oos"]), "setting": int(r["setting"])}
        res[f"{m}_spearman_settings_is_oos"] = float(sps.spearmanr(d["median_sharpe_is"], d["median_sharpe_oos"]).statistic)
        res[f"{m}_neutral_drift_abs_mean"] = float((d["neutral_oos"] - d["neutral_is"]).abs().mean())
        ref = d[(d["k"] == 5) & (d["coverage"] == 0.25) & ((d["atr_period"] == 14) | (d["atr_period"] == 0))]
        if len(ref):
            res[f"{m}_reference_row"] = ref.iloc[0][["median_sharpe_is", "median_sharpe_oos", "share_pos_oos"]].to_dict()
    res["best"] = best
    # A против B при одинаковых k и доле нейтрали (B — лучшая по IS ATR для пары)
    pair = []
    for (k, cov), d in df.groupby(["k", "coverage"]):
        a = d[d["method"] == "A_pct"].iloc[0]
        bb = d[d["method"] == "B_atr"]
        b = bb.loc[bb["median_sharpe_is"].idxmax()]
        pair.append({"k": k, "coverage": cov, "A_is": a["median_sharpe_is"], "B_is": b["median_sharpe_is"],
                     "A_oos": a["median_sharpe_oos"], "B_oos": b["median_sharpe_oos"],
                     "A_neutral_drift": a["neutral_oos"] - a["neutral_is"],
                     "B_neutral_drift": b["neutral_oos"] - b["neutral_is"], "B_atr": b["atr_period"]})
    pd.DataFrame(pair).to_csv(out / "method_pairs.csv", index=False)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    return best


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0].startswith("stage"):
        for t in args[1:]:
            analyze(t, args[0])
        sys.exit(0)
    allbest = {}
    for t in args or ("NG", "BR"):
        allbest[t] = analyze(t)
    (WORK / "stage3").mkdir(parents=True, exist_ok=True)
    (WORK / "stage3" / "slope_params.json").write_text(json.dumps(allbest, indent=1))
