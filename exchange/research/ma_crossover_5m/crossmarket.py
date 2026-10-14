"""Кросс-рыночная проверка: какие закономерности общие для NG и BR.

Сетки инструментов имеют одинаковую индексацию конфигураций, поэтому одна и
та же стратегия сравнивается сама с собой на двух рынках. Отбор кандидатов —
только по in-sample обоих рынков; out-of-sample обоих — проверка.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

WORK = Path(os.environ.get("MA5M_WORK", "work"))
COLS = ["config", "method", "slow_type", "slow_period", "exit", "fast_type", "fast_period",
        "pnl_is", "pnl_oos", "sharpe_m_is", "sharpe_m_oos", "plateau_sharpe_is", "trades_is",
        "trades_oos", "avg_trade_ticks_full", "stability_is"]


def run(stage: str, top: int = 25):
    base = WORK / "analysis" / stage
    ng = pd.read_parquet(base / "NG" / "summary.parquet", columns=COLS)
    br = pd.read_parquet(base / "BR" / "summary.parquet", columns=COLS)
    assert (ng["config"].to_numpy() == br["config"].to_numpy()).all()
    res = {}
    for col in ("sharpe_m_is", "sharpe_m_oos", "plateau_sharpe_is"):
        res[f"spearman_{col}_NG_vs_BR"] = float(sps.spearmanr(ng[col], br[col], nan_policy="omit").statistic)
    pos = lambda d, c: (d[c] > 0).to_numpy()
    for per in ("is", "oos"):
        a, b = pos(ng, f"pnl_{per}"), pos(br, f"pnl_{per}")
        res[f"share_pos_{per}_NG"] = float(a.mean()); res[f"share_pos_{per}_BR"] = float(b.mean())
        res[f"share_pos_{per}_both"] = float((a & b).mean())
        res[f"share_pos_{per}_both_if_independent"] = float(a.mean() * b.mean())
    all4 = pos(ng, "pnl_is") & pos(br, "pnl_is") & pos(ng, "pnl_oos") & pos(br, "pnl_oos")
    res["n_pos_all_four"] = int(all4.sum())
    is_both = pos(ng, "pnl_is") & pos(br, "pnl_is")
    res["n_pos_is_both"] = int(is_both.sum())
    res["share_oos_both_given_is_both"] = float(all4.sum() / max(is_both.sum(), 1))
    res["share_oos_both_unconditional"] = res["share_pos_oos_both"]

    # семейства: согласуются ли рейтинги сочетаний типов между рынками
    keys = ["fast_type", "slow_type", "exit", "method"]
    fam = []
    for name, d in (("NG", ng), ("BR", br)):
        f = d.groupby(keys, observed=True).agg(med_is=("sharpe_m_is", "median"),
                                               med_oos=("sharpe_m_oos", "median")).add_suffix(f"_{name}")
        fam.append(f)
    fam = fam[0].join(fam[1]).reset_index()
    fam.to_csv(base / "crossmarket_families.csv", index=False)
    res["families_spearman_is"] = float(sps.spearmanr(fam["med_is_NG"], fam["med_is_BR"]).statistic)
    res["families_spearman_oos"] = float(sps.spearmanr(fam["med_oos_NG"], fam["med_oos_BR"]).statistic)

    # отбор по IS обоих рынков: максимум худшего из двух плато-Шарпов
    ok = (ng["trades_is"] >= 100) & (br["trades_is"] >= 100)
    score = np.minimum(ng["plateau_sharpe_is"], br["plateau_sharpe_is"]).where(ok)
    idx = score.nlargest(top).index
    cand = ng.loc[idx, COLS[:7]].copy()
    for name, d in (("NG", ng), ("BR", br)):
        for c in ("pnl_is", "pnl_oos", "sharpe_m_is", "sharpe_m_oos", "plateau_sharpe_is",
                  "trades_is", "trades_oos", "avg_trade_ticks_full"):
            cand[f"{c}_{name}"] = d.loc[idx, c].to_numpy()
    cand["score_min_plateau_is"] = score.loc[idx].to_numpy()
    cand["oos_positive_both"] = (cand["pnl_oos_NG"] > 0) & (cand["pnl_oos_BR"] > 0)
    cand.to_csv(base / "crossmarket_candidates.csv", index=False)
    res["candidates"] = int(len(cand))
    res["candidates_oos_positive_both"] = int(cand["oos_positive_both"].sum())
    (base / "crossmarket.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "stage1")
