"""Analysis pipeline: grids -> tables, figures and summary for the report.

Everything here reads the raw grids written by ``scripts/ma_slope_research.py``
(stages 1-3) and writes ``research/ma_cross_slope_5m/{tables,figures}``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats as sps

from src.ma_slope import analysis as A
from src.ma_slope import finalist as F
from src.ma_slope import plots as P
from src.ma_slope.averages import MA_TYPES
from src.ma_slope.data import TICK_SIZES, TICKERS, load_instrument
from src.ma_slope.grid import (
    ATR_GRID,
    BASELINE,
    EPS_ATR,
    EPS_PCT,
    EXIT_MODELS,
    FAST_LATTICE,
    K_GRID,
    METHODS,
    SLOPE_GRID,
    SLOW_LATTICE,
    Slope,
)
from src.ma_slope.kernel import S_N, S_PNL, S_SQ
from src.utils.paths import PROJECT_ROOT, RESULTS_DIR

RAW = RESULTS_DIR / "ma_slope"
OUT = PROJECT_ROOT / "research" / "ma_cross_slope_5m"
TABLES, FIGS = OUT / "tables", OUT / "figures"
# WTI has 2 181 five-minute bars (three months): reported, never used for inference.
INFERENCE = tuple(t for t in TICKERS if t != "wti")
WF_MIN_YEARS = 5     # 3 training years + at least 2 test years
PBO_MIN_YEARS = 4
MIN_TRADES_OOS = 30
SAMPLE = 200_000     # configs sampled for rank correlations and scatter plots
NAMES = {"ng": "NG", "br": "BR (Brent)", "wti": "WTI (CL)", "gold": "GOLD (GC)", "silv": "SILV (SI)",
         "copper": "COPPER (HG)", "plt": "PLT (PL)", "pld": "PLD (PA)"}


def slope_of(meta: dict, method: int, model: int) -> Slope:
    if "chosen" in meta:
        return Slope(**meta["chosen"][f"{METHODS[method]}/{EXIT_MODELS[model]}"])
    return BASELINE[method]


def config_of(meta: dict, idx: tuple) -> F.Config:
    lab = A.labels(meta, idx)
    return F.Config(lab["fast_type"], lab["fast_period"], lab["slow_type"], lab["slow_period"],
                    slope_of(meta, idx[2], idx[5]), idx[5])


def _q(x, qs=(10, 50, 90)):
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, q)) for q in qs] if x.size else [np.nan] * len(qs)


def _spearman(a, b, rng):
    ok = np.flatnonzero(np.isfinite(a) & np.isfinite(b))
    if ok.size < 10:
        return np.nan
    pick = rng.choice(ok, min(SAMPLE, ok.size), replace=False)
    return float(sps.spearmanr(a[pick], b[pick]).statistic)


class GridResult:
    """Scores of one instrument's grid kept for cross-instrument work."""

    def __init__(self, ticker: str, stage: str):
        self.ticker, self.stage = ticker, stage
        self.stats, self.years, self.meta = A.load_grid(RAW / stage / ticker)
        self.t_is, self.score, self.share, self.valid = A.is_scores(self.stats)
        oos = np.asarray(self.stats[..., A.GROUP["oos"], :], np.float64)
        self.n_oos = oos[..., S_N]
        self.t_oos = A.t_stat(self.n_oos, oos[..., S_PNL], oos[..., S_SQ])
        self.t_oos = np.where(self.n_oos >= MIN_TRADES_OOS, self.t_oos, np.nan)
        self.cat = A.category_mask(self.meta, self.t_is.shape)
        self.shape = self.t_is.shape

    def row(self, idx) -> dict:
        r = A.config_row(self.stats, self.meta, idx, self.score, self.share)
        r["ticker"] = self.ticker
        r["category"] = "fast<slow" if self.cat[idx] else "fast>=slow"
        r["slope"] = slope_of(self.meta, idx[2], idx[5]).label
        tick_value = TICK_SIZES[self.ticker] * self.meta["multiplier"]
        for g_ in ("is", "oos", "all"):
            r[f"{g_}_avg_trade_ticks"] = r[f"{g_}_avg_trade"] / tick_value
        r["is_p"] = A.p_value(r["is_t"], r["is_trades"])
        r["oos_p"] = A.p_value(r["oos_t"], r["oos_trades"])
        return r


# ---------------------------------------------------------------- per instrument

def grid_summary(g: GridResult, rng) -> list[dict]:
    rows = []
    for m in range(2):
        for x in range(2):
            for cname, cval in (("fast<slow", True), ("fast>=slow", False)):
                sl = (slice(None), slice(None), m, slice(None), slice(None), x)
                ok = g.valid[sl] & (g.cat[sl] == cval)
                t_is, t_oos = np.where(ok, g.t_is[sl], np.nan).ravel(), np.where(ok, g.t_oos[sl], np.nan).ravel()
                i10, i50, i90 = _q(t_is)
                o10, o50, o90 = _q(t_oos)
                rows.append({"ticker": g.ticker, "stage": g.stage, "method": METHODS[m], "model": EXIT_MODELS[x],
                             "category": cname, "slope": slope_of(g.meta, m, x).label, "configs": int(ok.sum()),
                             "is_t_p10": i10, "is_t_median": i50, "is_t_p90": i90,
                             "is_share_pos": float(np.nanmean(t_is[np.isfinite(t_is)] > 0)),
                             "oos_t_p10": o10, "oos_t_median": o50, "oos_t_p90": o90,
                             "oos_share_pos": float(np.mean(t_oos[np.isfinite(t_oos)] > 0)),
                             "spearman_is_oos": _spearman(t_is, t_oos, rng)})
    return rows


def type_pairs(g: GridResult) -> list[dict]:
    rows = []
    for ft, fname in enumerate(MA_TYPES):
        for st, sname in enumerate(MA_TYPES):
            sl = (st, slice(None), slice(None), ft, slice(None), slice(None))
            ok = g.valid[sl]
            t_is, t_oos = g.t_is[sl][ok], g.t_oos[sl][ok]
            sub = np.full(g.shape, False)
            sub[sl] = True
            idx = A.best(g.score, sub)
            best = g.row(idx) if idx else {}
            rows.append({"ticker": g.ticker, "fast_type": fname, "slow_type": sname, "configs": int(ok.sum()),
                         "is_t_median": float(np.nanmedian(t_is)), "oos_t_median": float(np.nanmedian(t_oos)),
                         "is_share_pos": float(np.mean(t_is > 0)),
                         "oos_share_pos": float(np.nanmean(t_oos[np.isfinite(t_oos)] > 0)),
                         "best_plateau_t": best.get("is_plateau_t"), "best_oos_t": best.get("oos_t"),
                         "best_oos_pnl": best.get("oos_pnl"),
                         "best": f"{best.get('fast_type')}{best.get('fast_period')}/{best.get('slow_type')}"
                                 f"{best.get('slow_period')} {best.get('method')} {best.get('model')}" if best else ""})
    return rows


def group_bests(g: GridResult) -> list[dict]:
    """Plateau-best config of every (fast type, slow type, method, exit model) slice."""
    rows = []
    for ft in range(len(MA_TYPES)):
        for st in range(len(MA_TYPES)):
            for m in range(2):
                for x in range(2):
                    sub = np.full(g.shape, False)
                    sub[st, :, m, ft, :, x] = True
                    idx = A.best(g.score, sub)
                    if idx is not None:
                        rows.append(g.row(idx))
    return rows


def finalist_detail(ins, g: GridResult, idx, tag: str):
    """Row with vectorbt risk metrics and Monte Carlo, trades, monthly/yearly PnL, MTM equity."""
    cfg = config_of(g.meta, idx)
    tr = F.trades(ins, cfg)
    pf = F.portfolio(ins, cfg)
    closed = pf.trades.records_readable
    closed = closed[closed["Status"] == "Closed"]
    mismatch = float(np.abs(closed["PnL"].to_numpy() - tr["pnl"].to_numpy()).max()) if len(tr) else 0.0
    daily = F.daily_pnl(pf)
    split_day = g.meta["split_time"][:10]
    d_index = pd.to_datetime(daily.index)
    row = g.row(idx)
    row.update({"tag": tag, "label": cfg.label, "vbt_trade_pnl_max_abs_diff": mismatch})
    for name, part in (("is", daily[d_index < split_day]), ("oos", daily[d_index >= split_day]), ("all", daily)):
        for k, v in F.risk_metrics(part).items():
            row[f"{name}_{k}"] = v
    for name in ("is", "oos"):
        pnl = tr.loc[tr["sample"] == name, "pnl"].to_numpy()
        if pnl.size >= 10:
            for k, v in A.monte_carlo(pnl, seed=1).items():
                row[f"mc_{name}_{k}"] = v
    monthly, yearly = F.calendar(tr)
    equity = pd.Series(np.cumsum(daily.to_numpy()), index=d_index)
    return row, tr, monthly, yearly, equity


def analyze_instrument(ticker: str, stage: str, rng, figures: bool = True) -> dict:
    g = GridResult(ticker, stage)
    out = {"grid": g, "summary": grid_summary(g, rng)}
    out["type_pairs"] = type_pairs(g)
    out["group_bests"] = group_bests(g)
    picks = {"overall": A.best(g.score), "fast<slow": A.best(g.score, g.cat),
             "fast>=slow": A.best(g.score, ~g.cat)}
    out["picks"] = picks
    out["n_trials"] = int(g.valid.sum())
    if g.years is not None and g.meta["n_years"] >= WF_MIN_YEARS:
        wf = [A.walk_forward(g.years, g.meta).assign(scope="all")]
        for x in range(2):
            mask = np.zeros(g.shape, bool)
            mask[..., x] = True
            wf.append(A.walk_forward(g.years, g.meta, mask=mask).assign(scope=EXIT_MODELS[x]))
        out["walk_forward"] = pd.concat(wf).assign(ticker=ticker, stage=stage)
    if g.years is not None and g.meta["n_years"] >= PBO_MIN_YEARS:
        r = A.pbo_cscv(g.years, g.meta)
        out["pbo"] = {"ticker": ticker, "stage": stage, **{k: v for k, v in r.items() if k != "logits"}}
        out["pbo_logits"] = r["logits"]
    out["pick_rows"] = {tag: dict(g.row(idx), tag=tag) for tag, idx in picks.items() if idx is not None}
    out["split_time"] = pd.Timestamp(g.meta["split_time"])
    if stage == "stage3":
        ins = load_instrument(ticker)
        out["finalists"] = {tag: finalist_detail(ins, g, idx, tag) for tag, idx in picks.items() if idx is not None}
    if figures and picks["overall"] is not None:
        idx = picks["overall"]
        st, _, m, ft, _, x = idx
        sl = (st, slice(None), m, ft, slice(None), x)
        lab = A.labels(g.meta, idx)
        P.period_heatmaps(
            [("IS t", np.where(g.valid[sl], g.t_is[sl], np.nan).T), ("IS plateau t", g.score[sl].T),
             ("share of positive IS neighbours", g.share[sl].T), ("OOS t", g.t_oos[sl].T)],
            g.meta["fast_periods"], g.meta["slow_periods"], (lab["fast_period"], lab["slow_period"]),
            f"{NAMES[ticker]}, {stage}: {lab['fast_type']} fast x {lab['slow_type']} slow, "
            f"{slope_of(g.meta, m, x).label}, exit={lab['model']}",
            FIGS / f"heatmap_{stage}_{ticker}.png")
    return out


# ---------------------------------------------------------------- cross-market

def cross_market(grids: dict[str, GridResult]) -> dict:
    """Same rules on every instrument (stage 1): how often does an edge repeat?"""
    tick = list(grids)
    ok = np.stack([g.valid & np.isfinite(g.t_oos) for g in grids.values()])
    common = ok.all(axis=0)
    pos = np.stack([np.where(common, g.t_oos > 0, False) for g in grids.values()])
    both = np.stack([np.where(common, (g.t_is > 0) & (g.t_oos > 0), False) for g in grids.values()])
    counts = pos.sum(axis=0)[common]
    p = pos.reshape(len(tick), -1)[:, common.ravel()].mean(axis=1)
    # Poisson-binomial null: instruments independent with their own positive rates.
    null = np.array([1.0])
    for pi in p:
        null = np.convolve(null, [1 - pi, pi])
    observed = np.bincount(counts, minlength=len(tick) + 1) / counts.size
    hist = pd.DataFrame({"instruments_positive_oos": range(len(tick) + 1), "observed_share": observed,
                         "independence_null_share": null})
    # Configs positive IS and OOS everywhere, ranked by the weakest instrument.
    n_both = both.sum(axis=0)
    t_oos = np.stack([g.t_oos for g in grids.values()])
    weakest = np.where(n_both == len(tick), np.nanmin(t_oos, axis=0), np.nan)
    first = next(iter(grids.values()))
    rows = []
    for flat in np.argsort(np.nan_to_num(weakest, nan=-np.inf).ravel())[::-1][:30]:
        idx = np.unravel_index(flat, first.shape)
        if not np.isfinite(weakest[idx]):
            break
        r = A.labels(first.meta, idx)
        r["min_oos_t"] = float(weakest[idx])
        for t, g in grids.items():
            r[f"{t}_is_t"], r[f"{t}_oos_t"] = float(g.t_is[idx]), float(g.t_oos[idx])
        rows.append(r)
    # Transfer: IS plateau-best of one instrument, OOS t on every instrument.
    transfer = pd.DataFrame(index=tick, columns=tick, dtype=float)
    for a, ga in grids.items():
        idx = A.best(ga.score)
        for b, gb in grids.items():
            transfer.loc[a, b] = gb.t_oos[idx]
    agree = {"configs_common": int(common.sum()), "positive_rate": dict(zip(tick, map(float, p), strict=True)),
             "all_positive_oos": int((counts == len(tick)).sum()),
             "all_positive_oos_expected": float(null[-1] * counts.size),
             "all_positive_is_and_oos": int((n_both == len(tick)).sum())}
    return {"hist": hist, "robust": pd.DataFrame(rows), "transfer": transfer, "agree": agree}


def slope_tables() -> pd.DataFrame:
    frames = [pd.read_csv(RAW / "stage2" / t / "slope_table.csv") for t in TICKERS
              if (RAW / "stage2" / t / "slope_table.csv").exists()]
    return pd.concat(frames, ignore_index=True)


def analyze_stage(stage: str, write=print) -> dict:
    """Analyse every instrument of one stage and pickle the compact parts."""
    import pickle
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)  # empty slices of thin instruments
    FIGS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42 if stage == "stage1" else 43)
    part = {"summary": [], "pairs": [], "bests": [], "wf": [], "pbo": [], "finals": [], "monthly": [],
            "yearly": [], "trades": {}, "equity": {}, "hist": {}, "pbo_logits": {}, "scatter": {},
            "n_trials": {}, "split_times": {}}
    grids = {}
    for t in TICKERS:
        if not (RAW / stage / t / "meta.json").exists():
            continue
        write(f"analyze {stage} {t}")
        r = analyze_instrument(t, stage, rng)
        g = r["grid"]
        part["summary"] += r["summary"]
        part["pairs"] += [dict(p, stage=stage) for p in r["type_pairs"]]
        part["bests"] += [dict(b, stage=stage) for b in r["group_bests"]]
        part["n_trials"][t] = r["n_trials"]
        part["split_times"][t] = r["split_time"]
        if "walk_forward" in r:
            part["wf"].append(r["walk_forward"])
        if "pbo" in r:
            part["pbo"].append(r["pbo"])
            part["pbo_logits"][f"{NAMES[t]} {stage}"] = r["pbo_logits"]
        if stage == "stage3":
            ok = np.flatnonzero((g.valid & np.isfinite(g.t_oos)).ravel())
            pick = rng.choice(ok, min(SAMPLE, ok.size), replace=False) if ok.size else ok
            rho = _spearman(g.t_is.ravel(), g.t_oos.ravel(), rng)
            part["scatter"][NAMES[t]] = (g.t_is.ravel()[pick], g.t_oos.ravel()[pick], f"rho={rho:.2f}")
        else:
            part["finals"] += [dict(row, stage=stage) for row in r["pick_rows"].values()]
            if t in INFERENCE:
                g.stats = g.years = None
                grids[t] = g
        for tag, (row, tr, monthly, yearly, eq) in r.get("finalists", {}).items():
            part["finals"].append(dict(row, stage=stage))
            if tag == "overall":
                part["trades"][t] = tr
                part["equity"][NAMES[t]] = [("MTM PnL, USD/contract", eq.index, eq.to_numpy())]
                part["hist"][NAMES[t]] = {"IS": tr.loc[tr["sample"] == "is", "pnl"].to_numpy(),
                                          "OOS": tr.loc[tr["sample"] == "oos", "pnl"].to_numpy()}
                m = monthly.copy()
                m.insert(0, "ticker", t)
                part["monthly"].append(m.reset_index())
                part["yearly"] += [{"ticker": t, "year": int(y), "pnl": float(v)} for y, v in yearly.items()]
    if grids:
        write("cross-market")
        part["cross_market"] = cross_market(grids)
    with open(RAW / f"analysis_{stage}.pkl", "wb") as f:
        pickle.dump(part, f)
    return part


def finalize(write=print) -> dict:
    """Combine the stage analyses: multiple testing, tables, figures, summary.json."""
    import pickle

    parts = {}
    for stage in ("stage1", "stage3"):
        with open(RAW / f"analysis_{stage}.pkl", "rb") as f:
            parts[stage] = pickle.load(f)
    p1, p3 = parts["stage1"], parts["stage3"]
    TABLES.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(p1["summary"] + p3["summary"])
    pairs = pd.DataFrame(p1["pairs"] + p3["pairs"])
    bests = pd.DataFrame(p1["bests"] + p3["bests"])
    finals = pd.DataFrame(p1["finals"] + p3["finals"])
    wf = pd.concat(p1["wf"] + p3["wf"]) if p1["wf"] + p3["wf"] else pd.DataFrame()
    cm = p1["cross_market"]
    # Multiple testing: Holm over the instrument finalists, BH over the group bests.
    for stage in ("stage1", "stage3"):
        sel = (bests["stage"] == stage) & bests["ticker"].isin(INFERENCE)
        bests.loc[sel, "oos_q_bh"] = A.benjamini_hochberg(bests.loc[sel, "oos_p"])
        fsel = (finals["stage"] == stage) & (finals["tag"] == "overall") & finals["ticker"].isin(INFERENCE)
        finals.loc[fsel, "oos_p_holm"] = A.holm(finals.loc[fsel, "oos_p"])
    stage2_configs = len(MA_TYPES) * len(SLOW_LATTICE) * len(SLOPE_GRID) * len(MA_TYPES) * len(FAST_LATTICE) * 2
    trials = {t: p1["n_trials"][t] + p3["n_trials"][t] + stage2_configs for t in p3["n_trials"]}
    write("tables")
    summary.to_csv(TABLES / "grid_summary.csv", index=False)
    pairs.to_csv(TABLES / "type_pairs.csv", index=False)
    bests.sort_values(["stage", "ticker", "is_plateau_t"], ascending=[True, True, False]).to_csv(
        TABLES / "ranking_group_bests.csv", index=False)
    finals.to_csv(TABLES / "finalists.csv", index=False)
    wf.to_csv(TABLES / "walk_forward.csv", index=False)
    pd.DataFrame(p1["pbo"] + p3["pbo"]).to_csv(TABLES / "pbo.csv", index=False)
    cm["hist"].to_csv(TABLES / "cross_market_counts.csv", index=False)
    cm["robust"].to_csv(TABLES / "cross_market_robust_configs.csv", index=False)
    cm["transfer"].to_csv(TABLES / "cross_market_transfer.csv")
    if p3["monthly"]:
        pd.concat(p3["monthly"]).to_csv(TABLES / "finalist_monthly_pnl.csv", index=False)
        pd.DataFrame(p3["yearly"]).to_csv(TABLES / "finalist_yearly_pnl.csv", index=False)
    (TABLES / "finalist_trades").mkdir(exist_ok=True)
    for t, tr in p3["trades"].items():
        tr.to_csv(TABLES / "finalist_trades" / f"{t}.csv", index=False)
    slopes = slope_tables()
    slopes.to_csv(TABLES / "slope_sweep.csv", index=False)
    write("figures")
    figures(p3["split_times"], summary, pairs, bests, finals, wf, cm, slopes, p3["equity"], p3["hist"],
            {**p1["pbo_logits"], **p3["pbo_logits"]}, p3["scatter"])
    info = {"trials_per_instrument": trials,
            "expected_max_t_null": {t: A.expected_max_t(n) for t, n in trials.items()},
            "cross_market": cm["agree"], "split_times": {t: str(v) for t, v in p3["split_times"].items()}}
    (TABLES / "summary.json").write_text(json.dumps(info, indent=2, default=float))
    return info


def run(write=print) -> dict:
    for stage in ("stage1", "stage3"):
        analyze_stage(stage, write)
    return finalize(write)


def figures(split_times, summary, pairs, bests, finals, wf, cm, slopes, equity, hist_data, pbo_logits, scatter):
    inf = [t for t in INFERENCE if t in split_times]
    cats = [NAMES[t] for t in inf]
    # 2. 36 type pairs: median OOS t per instrument (stage 3).
    mats = {}
    for t in inf:
        p = pairs[(pairs["stage"] == "stage3") & (pairs["ticker"] == t)]
        mats[NAMES[t]] = p.pivot(index="fast_type", columns="slow_type", values="oos_t_median").reindex(
            index=MA_TYPES, columns=MA_TYPES).to_numpy()
    P.matrix_grid(mats, [f"fast {m}" for m in MA_TYPES], [f"slow {m}" for m in MA_TYPES],
                  "Median out-of-sample t of all configs per Fast x Slow MA type (stage 3)",
                  FIGS / "type_pairs_oos_median_t.png")
    mats = {}
    for t in inf:
        p = pairs[(pairs["stage"] == "stage3") & (pairs["ticker"] == t)]
        mats[NAMES[t]] = p.pivot(index="fast_type", columns="slow_type", values="best_oos_t").reindex(
            index=MA_TYPES, columns=MA_TYPES).to_numpy()
    P.matrix_grid(mats, [f"fast {m}" for m in MA_TYPES], [f"slow {m}" for m in MA_TYPES],
                  "Out-of-sample t of the in-sample plateau-best config per type pair (stage 3)",
                  FIGS / "type_pairs_best_oos_t.png")
    # 3. Exit models and 4. slope methods (stage 3, fast<slow).
    s3 = summary[(summary["stage"] == "stage3") & (summary["category"] == "fast<slow")]
    for key, values, fname, title in (("model", EXIT_MODELS, "exit_models", "Exit model"),
                                      ("method", METHODS, "slope_methods", "Slope method")):
        for sample in ("is", "oos"):
            series = {}
            for v in values:
                agg = s3[s3[key] == v].groupby("ticker")[[f"{sample}_t_p10", f"{sample}_t_median",
                                                          f"{sample}_t_p90"]].median().reindex(inf)
                series[v] = tuple(agg[c].to_numpy() for c in agg.columns)
            P.ranges(cats, series, f"{title}: {sample.upper()} t of all configs (median, p10-p90), stage 3",
                     "t of mean trade PnL", FIGS / f"{fname}_{sample}.png")
    # Stage-2 slope sweep: best in-sample median t per method and exit, and OOS of that setting.
    rows = []
    for t in inf:
        st = slopes[slopes["ticker"] == t]
        for (method, model), gg in st.groupby(["method", "model"]):
            r = gg.loc[gg["is_median_t"].idxmax()]
            rows.append({"ticker": t, "slot": f"{method}/{EXIT_MODELS[int(model)]}", "is": r["is_median_t"],
                         "oos": r["oos_median_t"]})
    sw = pd.DataFrame(rows)
    for model in (0, 1):
        for method, cols, colnames in (
                ("pct", list(EPS_PCT), [f"{e:g}%" for e in EPS_PCT]),
                ("atr", [(a, e) for a in ATR_GRID for e in EPS_ATR], [f"{e:g}xATR{a}" for a in ATR_GRID for e in EPS_ATR])):
            mats = {}
            for t in inf:
                st = slopes[(slopes["ticker"] == t) & (slopes["method"] == method) & (slopes["model"] == model)]
                key = st["eps"] if method == "pct" else list(zip(st["atr_n"].astype(int), st["eps"], strict=True))
                st = st.assign(col=key)
                mats[NAMES[t]] = st.pivot_table(index="k", columns="col", values="is_median_t").reindex(
                    index=list(K_GRID), columns=cols).to_numpy()
            P.matrix_grid(mats, [f"k={k}" for k in K_GRID], colnames,
                          f"Stage 2, method {method}, exit {EXIT_MODELS[model]}: median IS t over the period lattice",
                          FIGS / f"slope_sweep_{method}_{EXIT_MODELS[model]}.png", fmt="{:.2f}")
    P.grouped_bars(cats, {slot: sw[sw["slot"] == slot].set_index("ticker").reindex(inf)["oos"].to_numpy()
                          for slot in sw["slot"].unique()},
                   "Stage 2: OOS median t of the slope setting chosen in-sample", "median OOS t",
                   FIGS / "slope_choice_oos.png")
    # 6. Equity curves, 7. trade distributions.
    if equity:
        P.small_multiples_lines(equity, "Selected config per instrument: mark-to-market PnL, one contract "
                                "(line = IS/OOS split)", "USD", FIGS / "equity_finalists.png",
                                vlines={NAMES[t]: v for t, v in split_times.items()})
        P.small_multiples_hist(hist_data, "Trade PnL of the selected configs", "USD per trade",
                               FIGS / "trade_distributions.png")
    # 8. Cross-market.
    tr = cm["transfer"]
    P.matrix_grid({"IS-best of row instrument -> OOS t on column instrument": tr.to_numpy(float)},
                  [NAMES[t] for t in tr.index], [NAMES[t] for t in tr.columns],
                  "Cross-market transfer (stage 1, identical rules)", FIGS / "cross_market_transfer.png", ncols=1)
    h = cm["hist"]
    P.grouped_bars([str(v) for v in h["instruments_positive_oos"]],
                   {"observed": h["observed_share"].to_numpy(), "independent instruments": h[
                       "independence_null_share"].to_numpy()},
                   "Configs by number of instruments with positive OOS t (stage 1)", "share of configs",
                   FIGS / "cross_market_counts.png")
    f3 = finals[(finals["stage"] == "stage3") & (finals["tag"] == "overall") & finals["ticker"].isin(inf)]
    P.grouped_bars(cats, {"IS t": f3.set_index("ticker").reindex(inf)["is_t"].to_numpy(),
                          "OOS t": f3.set_index("ticker").reindex(inf)["oos_t"].to_numpy()},
                   "Selected config per instrument: t of mean trade PnL, IS vs OOS (stage 3)", "t",
                   FIGS / "finalists_is_vs_oos.png")
    # 9. IS vs OOS scatter.
    if scatter:
        P.small_multiples_hexbin(scatter, "All configs: in-sample vs out-of-sample t (stage 3, 200k sample)",
                                 "IS t", "OOS t", FIGS / "is_vs_oos.png")
    # Walk-forward and PBO.
    w = wf[(wf["scope"] == "all") & (wf["stage"] == "stage3")] if len(wf) else wf
    if len(w):
        P.signed_bars({NAMES[t]: (list(gg["test_year"]), list(gg["test_pnl"])) for t, gg in w.groupby("ticker")},
                      "Walk-forward (3 years train -> next year test): test-year PnL of the selected config",
                      "USD per contract", FIGS / "walk_forward.png")
    if pbo_logits:
        P.small_multiples_hist({k: {"logit of OOS rank": v} for k, v in pbo_logits.items()},
                               "CSCV: logit of the OOS rank of the IS-best config (< 0 = below median)",
                               "logit", FIGS / "pbo_logits.png", bins=30)
