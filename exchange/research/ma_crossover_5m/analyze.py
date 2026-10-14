"""Анализ полной сетки одного этапа: метрики, семейства, отбор, устойчивость.

Запуск: python analyze.py stage1 NG BR
Пишет в work/analysis/<stage>/<ticker>/ компактные таблицы (parquet/csv/json)
и полную таблицу по конфигурациям (summary.parquet, ~0.5 ГБ — в git не идёт).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

import robust as R
from results import load_grid, summary_frame

MIN_TRADES_IS = 100          # минимум сделок на IS для отбора по Шарпу (задан заранее)
WORK = Path(os.environ.get("MA5M_WORK", "work"))


def family_table(df: pd.DataFrame, keys) -> pd.DataFrame:
    g = df.groupby(keys, observed=True)
    out = g.agg(
        configs=("config", "size"),
        median_sharpe_is=("sharpe_m_is", "median"),
        share_pos_is=("pnl_is", lambda x: float((x > 0).mean())),
        median_sharpe_oos=("sharpe_m_oos", "median"),
        share_pos_oos=("pnl_oos", lambda x: float((x > 0).mean())),
        median_pnl_is=("pnl_is", "median"),
        median_pnl_oos=("pnl_oos", "median"),
        median_trades_is=("trades_is", "median"),
        median_ticks_per_trade=("avg_trade_ticks_full", "median"),
    )
    # «Выбор лучших»: средний OOS-Шарп верхнего 1% по IS внутри семейства
    def top_oos(x):
        k = max(1, int(len(x) * 0.01))
        top = x.nlargest(k, "sharpe_m_is")
        return pd.Series({"top1pct_is_sharpe": top["sharpe_m_is"].mean(),
                          "top1pct_oos_sharpe": top["sharpe_m_oos"].mean(),
                          "top1pct_share_pos_oos": float((top["pnl_oos"] > 0).mean())})
    tops = g[["sharpe_m_is", "sharpe_m_oos", "pnl_oos"]].apply(top_oos)
    spear = g.apply(lambda x: sps.spearmanr(x["sharpe_m_is"], x["sharpe_m_oos"],
                                            nan_policy="omit").statistic, include_groups=False)
    out = out.join(tops)
    out["spearman_is_oos"] = spear
    return out.reset_index()


def deciles(df: pd.DataFrame, col_is="sharpe_m_is", col_oos="sharpe_m_oos") -> pd.DataFrame:
    d = df[[col_is, col_oos, "pnl_oos"]].dropna()
    d["decile"] = pd.qcut(d[col_is].rank(method="first"), 10, labels=range(1, 11))
    return d.groupby("decile", observed=True).agg(
        is_sharpe=(col_is, "mean"), oos_sharpe=(col_oos, "mean"),
        oos_pnl=("pnl_oos", "mean"), share_pos_oos=("pnl_oos", lambda x: float((x > 0).mean())),
        n=(col_is, "size")).reset_index()


def select(df: pd.DataFrame, plateau_is: np.ndarray, mask=None) -> dict:
    """Три правила отбора — ВСЕ только по in-sample."""
    d = df if mask is None else df[mask]
    out = {}
    out["R1_max_pnl"] = int(d["pnl_is"].idxmax())
    ok = d["trades_is"] >= MIN_TRADES_IS
    out["R2_max_sharpe"] = int(d.loc[ok, "sharpe_m_is"].idxmax())
    pl = pd.Series(plateau_is, index=df.index).loc[d.index]
    out["R3_plateau"] = int(pl[ok].idxmax())
    return out


def analyze(stage: str, ticker: str):
    t0 = time.time()
    g = load_grid(WORK / stage / ticker)
    out = WORK / "analysis" / stage / ticker
    out.mkdir(parents=True, exist_ok=True)
    print(f"== {stage}/{ticker}: {g.n:,} конфигураций", flush=True)
    df = summary_frame(g)
    df["plateau_sharpe_is"] = R.plateau(df["sharpe_m_is"].to_numpy(), g.shape)
    df["stability_is"] = R.sign_stability(df["sharpe_m_is"].to_numpy(), g.shape)
    df.to_parquet(out / "summary.parquet", index=False)
    print(f"   summary {time.time() - t0:.0f} с", flush=True)
    res = {"ticker": ticker, "stage": stage, "n_configs": int(g.n),
           "is_months": int(g.oos_month), "oos_months": int(g.monthly.shape[1] - g.oos_month)}

    # --- общие доли и IS→OOS -----------------------------------------------
    res["share_pos_is"] = float((df["pnl_is"] > 0).mean())
    res["share_pos_oos"] = float((df["pnl_oos"] > 0).mean())
    res["share_pos_full"] = float((df["pnl_full"] > 0).mean())
    res["share_pos_both"] = float(((df["pnl_is"] > 0) & (df["pnl_oos"] > 0)).mean())
    res["spearman_is_oos_sharpe"] = float(sps.spearmanr(df["sharpe_m_is"], df["sharpe_m_oos"],
                                                        nan_policy="omit").statistic)
    res["spearman_is_oos_pnl"] = float(sps.spearmanr(df["pnl_is"], df["pnl_oos"]).statistic)
    res["spearman_plateau_oos"] = float(sps.spearmanr(df["plateau_sharpe_is"], df["sharpe_m_oos"],
                                                      nan_policy="omit").statistic)
    deciles(df).to_csv(out / "deciles_is_oos.csv", index=False)
    deciles(df, "plateau_sharpe_is").to_csv(out / "deciles_plateau_oos.csv", index=False)

    # --- семейства ----------------------------------------------------------
    family_table(df, ["fast_type", "slow_type"]).to_csv(out / "by_type_pair.csv", index=False)
    family_table(df, ["fast_type", "slow_type", "exit", "method"]).to_csv(
        out / "by_type_pair_exit_method.csv", index=False)
    family_table(df, ["exit"]).to_csv(out / "by_exit.csv", index=False)
    family_table(df, ["method"]).to_csv(out / "by_method.csv", index=False)
    family_table(df, ["exit", "method"]).to_csv(out / "by_exit_method.csv", index=False)
    family_table(df, ["fast_ge_slow"]).to_csv(out / "by_fast_ge_slow.csv", index=False)
    family_table(df, ["fast_type"]).to_csv(out / "by_fast_type.csv", index=False)
    family_table(df, ["slow_type"]).to_csv(out / "by_slow_type.csv", index=False)

    # --- парные сравнения: та же конфигурация, другой метод / выход ----------
    shp = g.shape
    for axis, name in ((0, "method_B_minus_A"), (3, "exit_B_minus_A")):
        for col in ("sharpe_m_is", "sharpe_m_oos", "pnl_is", "pnl_oos"):
            v = df[col].to_numpy().reshape(shp)
            a = np.take(v, 0, axis=axis).ravel(); b = np.take(v, 1, axis=axis).ravel()
            ok = np.isfinite(a) & np.isfinite(b)
            res[f"{name}_{col}_median"] = float(np.median(b[ok] - a[ok]))
            res[f"{name}_{col}_share_B_better"] = float((b[ok] > a[ok]).mean())

    # --- матрицы для тепловых карт (fast × slow) по семействам ---------------
    for col in ("sharpe_m_is", "sharpe_m_oos", "plateau_sharpe_is", "stability_is",
                "pnl_is", "pnl_oos", "trades_full"):
        np.save(out / f"cube_{col}.npy", df[col].to_numpy(np.float32).reshape(shp))

    # --- отбор: глобально и в каждой (exit, method) ---------------------------
    plateau_is = df["plateau_sharpe_is"].to_numpy()
    picks = []
    for scope, mask in [("global", None)] + [
            (f"{e}|{m}", (df["exit"] == e) & (df["method"] == m))
            for e in g.meta["exits"] for m in g.meta["methods"]]:
        for rule, idx in select(df, plateau_is, mask).items():
            picks.append({"scope": scope, "rule": rule, "config": idx})
    # лучший по плато в каждом из 36 сочетаний типов × выход × метод
    for (ft, st, e, m), sub in df.groupby(["fast_type", "slow_type", "exit", "method"], observed=True):
        ok = sub["trades_is"] >= MIN_TRADES_IS
        if ok.any():
            picks.append({"scope": f"family:{ft}+{st}|{e}|{m}", "rule": "R3_plateau",
                          "config": int(sub.loc[ok, "plateau_sharpe_is"].idxmax())})
    picks = pd.DataFrame(picks)
    cols = ["config", "method", "slow_type", "slow_period", "exit", "fast_type", "fast_period",
            "pnl_is", "pnl_oos", "pnl_full", "trades_is", "trades_oos", "sharpe_m_is",
            "sharpe_m_oos", "plateau_sharpe_is", "stability_is", "win_rate_full",
            "profit_factor_full", "avg_trade_full", "avg_trade_ticks_full", "max_dd_full",
            "avg_bars_full", "max_loss_streak_full", "pos_months_is", "pos_months_oos",
            "long_share_full", "pnl_long_full"]
    picks = picks.merge(df[cols], on="config", how="left")
    picks.to_csv(out / "picks.csv", index=False)

    # --- walk-forward с переоптимизацией по всей сетке -----------------------
    wf = {}
    for crit in ("pnl", "sharpe", "plateau"):
        steps = R.walk_forward(g.monthly, g.shape, train=24, test=3, criterion=crit)
        months = np.concatenate([s["test_months"] for s in steps])
        wf[crit] = {
            "steps": len(steps), "test_months": int(len(months)),
            "total_pnl": float(months.sum()),
            "sharpe_annual": float(months.mean() / months.std(ddof=1) * np.sqrt(12)) if months.std() > 0 else None,
            "share_pos_steps": float(np.mean([s["test_pnl"] > 0 for s in steps])),
            "distinct_configs": int(len({s["config"] for s in steps})),
            "t_stat": float(months.mean() / (months.std(ddof=1) / np.sqrt(len(months)))) if months.std() > 0 else None,
        }
        pd.DataFrame([{k: v for k, v in s.items() if k != "test_months"} for s in steps]).to_csv(
            out / f"wf_{crit}.csv", index=False)
        np.save(out / f"wf_{crit}_months.npy", months)
        print(f"   WF {crit}: {wf[crit]}", flush=True)
    res["walk_forward"] = wf

    # --- PBO ----------------------------------------------------------------
    nb_is = 16 if g.oos_month >= 96 else 8
    p_is = R.pbo_cscv(g.monthly, n_blocks=nb_is, n_splits=1000, seed=1, months=slice(0, g.oos_month))
    p_full = R.pbo_cscv(g.monthly, n_blocks=16 if g.monthly.shape[1] >= 128 else 10,
                        n_splits=1000, seed=1)
    res["pbo_is"] = {k: v for k, v in p_is.items() if k in ("pbo", "n_splits", "n_blocks")}
    res["pbo_full"] = {k: v for k, v in p_full.items() if k in ("pbo", "n_splits", "n_blocks")}
    np.save(out / "pbo_full_logits.npy", p_full["logits"])
    np.save(out / "pbo_full_pairs.npy", p_full["is_oos_pairs"])
    print(f"   PBO IS {p_is['pbo']:.3f} | full {p_full['pbo']:.3f}", flush=True)

    # --- Reality Check / SPA по всей сетке ------------------------------------
    for tag, sl in (("is", slice(0, g.oos_month)), ("full", slice(0, g.monthly.shape[1]))):
        r = R.reality_check_spa(g.monthly, months=sl, B=1000, seed=2)
        r["rc_best_config"] = g.decode([r["rc_best"]]).iloc[0].to_dict()
        r["spa_best_config"] = g.decode([r["spa_best"]]).iloc[0].to_dict()
        res[f"rc_spa_{tag}"] = r
        print(f"   RC/SPA {tag}: RC p={r['rc_p']:.3f} SPA p={r['spa_p']:.3f}", flush=True)

    # --- DSR для отобранных ---------------------------------------------------
    sr_is_period = df["sharpe_m_is"].to_numpy() / np.sqrt(12)
    var_sr = float(np.nanvar(sr_is_period))
    n_eff = R.effective_trials(g.monthly, slice(0, g.oos_month))
    res["var_sr_is_monthly"] = var_sr
    res["n_eff_trials_lower"] = n_eff
    dsr_rows = []
    for _, p in picks[~picks["scope"].str.startswith("family:")].iterrows():
        x = np.asarray(g.monthly[int(p["config"]), :g.oos_month], dtype=np.float64)
        for label, n in (("N_full", g.n), ("N_eff", n_eff)):
            d = R.deflated_sharpe(x, n, var_sr)
            dsr_rows.append({"scope": p["scope"], "rule": p["rule"], "config": int(p["config"]),
                             "trials": label, **d})
    pd.DataFrame(dsr_rows).to_csv(out / "dsr_picks.csv", index=False)

    res["seconds"] = time.time() - t0
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print(f"== {stage}/{ticker} анализ за {res['seconds'] / 60:.1f} мин", flush=True)
    return res


if __name__ == "__main__":
    stage = sys.argv[1]
    for t in sys.argv[2:]:
        analyze(stage, t)
