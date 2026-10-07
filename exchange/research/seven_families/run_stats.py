"""Статистическая часть: дефляция, PBO, CPCV, walk-forward, SPA, серии убытков."""
from __future__ import annotations

import itertools, pickle, logging, warnings
logging.disable(logging.INFO); warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats as sps

import families as F
import harness as H
from run_all import EMBARGO, GRIDS, IS_FRACTION, MIN_TRADES, evaluate_cell, run_panel, run_single, split
from exchange.backtest.metrics import sharpe_ratio
from exchange.stats.bootstrap import block_bootstrap
from exchange.stats.cv import CombinatorialPurgedCV, WalkForward
from exchange.stats.multiple_testing import benjamini_hochberg, expected_false_discoveries, holm
from exchange.stats.pbo import probability_of_backtest_overfitting
from exchange.stats.reality_check import reality_check, superior_predictive_ability
from exchange.stats.sharpe import deflated_sharpe_ratio, probabilistic_sharpe_ratio

with open("results.pkl", "rb") as fh:
    R = pickle.load(fh)

universe = H.load_universe()
panel_all = {r: p["bars"] for r, p in universe.items()}
TOTAL_TRIALS = sum(c["n_configs"] for c in R["cells"])


def cell_panel(family, market):
    if family in F.PANEL:
        common = panel_all["NG"].index.intersection(panel_all["BR"].index)
        return {r: b.loc[common] for r, b in panel_all.items()}, \
               {r: universe[r]["ppy"] for r in panel_all}, common
    bars = panel_all[market]
    return bars, universe[market]["ppy"], bars.index


def returns_for(family, market, params, index):
    data, ppy, _ = cell_panel(family, market)
    if family in F.PANEL:
        sub = {r: b.loc[index] for r, b in data.items()}
        _, ret, ppy_p, _, _ = run_panel(family, params, sub, ppy)
        return ret, ppy_p
    res, _ = run_single(family, market, params, data.loc[index], ppy)
    return res.returns, ppy


rows = []
for cell in R["cells"]:
    family, market = cell["family"], cell["market"]
    key = f"{family}|{market}"
    data, ppy, index = cell_panel(family, market)
    grid = GRIDS[family]
    best = R["winners"][key]
    is_idx, oos_idx = split(index)
    oos_ret = R["oos_returns"][key]
    ppy_cell = cell["ppy"]

    # ---- дефляция -----------------------------------------------------
    trial_sharpes = (R["matrices"][key].apply(
        lambda c: sharpe_ratio(c, periods_per_year=ppy_cell)) / np.sqrt(ppy_cell)).dropna().to_numpy()
    psr = probabilistic_sharpe_ratio(oos_ret, benchmark_sr=0.0)
    dsr_family = deflated_sharpe_ratio(oos_ret, n_trials=cell["n_configs"],
                                       trial_sharpes=trial_sharpes, periods_per_year=ppy_cell)
    dsr_all = deflated_sharpe_ratio(oos_ret, n_trials=TOTAL_TRIALS,
                                    trial_sharpes=trial_sharpes, periods_per_year=ppy_cell)

    # ---- PBO по сетке семейства ---------------------------------------
    matrix = R["matrices"][key].dropna(how="any")
    try:
        pbo = probability_of_backtest_overfitting(matrix.to_numpy(), n_partitions=10)
        pbo_value, pbo_slope = pbo.pbo, pbo.performance_degradation
    except Exception as exc:
        pbo_value, pbo_slope = np.nan, np.nan

    # ---- CPCV победителя ----------------------------------------------
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, horizon=5, embargo=EMBARGO)
    cpcv = []
    for sp in cv.split(len(index)):
        sub = index[sp.test]
        if len(sub) < 60: continue
        try:
            ret, p = returns_for(family, market, best, sub)
            cpcv.append(sharpe_ratio(ret, periods_per_year=p))
        except Exception:
            continue
    cpcv = np.array([c for c in cpcv if np.isfinite(c)])

    # ---- walk-forward с переоптимизацией -------------------------------
    wf_train = max(int(len(index) * 0.35), 250)
    wf = WalkForward(train_size=wf_train, test_size=max(int(len(index) * 0.08), 120),
                     horizon=5, embargo=EMBARGO)
    wf_is, wf_oos, wf_params = [], [], []
    for sp in wf.split(len(index)):
        train, test = index[sp.train], index[sp.test]
        scored = []
        for params in grid:
            try:
                ret, p = returns_for(family, market, params, train)
                scored.append((sharpe_ratio(ret, periods_per_year=p), params))
            except Exception:
                continue
        scored = [s for s in scored if np.isfinite(s[0])]
        if not scored: continue
        best_is, chosen = max(scored, key=lambda t: t[0])
        try:
            ret, p = returns_for(family, market, chosen, test)
        except Exception:
            continue
        wf_is.append(best_is); wf_oos.append(sharpe_ratio(ret, periods_per_year=p))
        wf_params.append(tuple(sorted(chosen.items())))
    wf_is, wf_oos = np.array(wf_is), np.array(wf_oos)
    wf_eff = (np.nanmean(wf_oos) / np.nanmean(wf_is)
              if len(wf_is) and np.nanmean(wf_is) > 0 else np.nan)

    # ---- бутстрэп-интервал на OOS --------------------------------------
    try:
        boot = block_bootstrap(oos_ret.to_numpy(float),
                               lambda x: float(np.mean(x) / np.std(x, ddof=1)),
                               n_resamples=800, seed=20260902)
        lo, hi = boot.confidence_interval(0.95)
        boot_lo, boot_p = lo * np.sqrt(ppy_cell), boot.p_value_greater_than(0.0)
    except Exception:
        boot_lo, boot_p = np.nan, np.nan

    # ---- серии убыточных месяцев ---------------------------------------
    monthly = (1.0 + oos_ret).resample("ME").prod() - 1.0
    losses = (monthly < 0).to_numpy()
    streaks, run = [], 0
    for flag in losses:
        run = run + 1 if flag else 0
        streaks.append(run)
    max_streak = int(max(streaks)) if streaks else 0

    rows.append({**cell,
        "psr": psr, "dsr_family": dsr_family.deflated_sharpe_ratio,
        "dsr_programme": dsr_all.deflated_sharpe_ratio,
        "expected_max_sr_null": dsr_all.expected_max_sharpe * np.sqrt(ppy_cell),
        "pbo": pbo_value, "pbo_slope": pbo_slope,
        "cpcv_median": float(np.median(cpcv)) if len(cpcv) else np.nan,
        "cpcv_positive": float((cpcv > 0).mean()) if len(cpcv) else np.nan,
        "cpcv_dispersion": float(np.std(cpcv, ddof=1)) if len(cpcv) > 1 else np.nan,
        "cpcv_n": len(cpcv),
        "wf_efficiency": wf_eff, "wf_windows": len(wf_oos),
        "wf_profitable": float((wf_oos > 0).mean()) if len(wf_oos) else np.nan,
        "wf_param_stability": (float(np.mean([a == b for a, b in zip(wf_params, wf_params[1:])]))
                               if len(wf_params) > 1 else np.nan),
        "bootstrap_lo": boot_lo, "bootstrap_p": boot_p,
        "n_months_oos": int(len(monthly)), "negative_months": int(losses.sum()),
        "max_losing_streak": max_streak,
        "monthly": monthly,
    })
    print(f"{key:28} PBO {pbo_value:5.2f} | DSR_prog {dsr_all.deflated_sharpe_ratio:5.3f} "
          f"| CPCV {np.median(cpcv) if len(cpcv) else float('nan'):+.2f} "
          f"| WF {wf_eff if np.isfinite(wf_eff) else float('nan'):+.2f} "
          f"| серия {max_streak} мес", flush=True)

table = pd.DataFrame(rows)
with open("stats.pkl", "wb") as fh:
    pickle.dump({"table": table, "total_trials": TOTAL_TRIALS}, fh)
print("\nготово, ячеек:", len(table), "| всего конфигураций:", TOTAL_TRIALS)
