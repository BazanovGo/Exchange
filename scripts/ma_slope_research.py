"""MA-cross + Slow-MA slope study on 5m MOEX futures: run the stages.

    python scripts/ma_slope_research.py stage1 [--tickers br gold ...]

Raw grids go to ``results/ma_slope/<stage>/<ticker>/`` (git-ignored, several
GB); the report and the compact tables go to ``research/ma_cross_slope_5m``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.ma_slope.analysis import GROUP, MIN_TRADES_IS, t_stat  # noqa: E402
from src.ma_slope.data import TICKERS, load_instrument  # noqa: E402
from src.ma_slope.grid import BASELINE, FAST_LATTICE, SLOPE_GRID, SLOW_LATTICE, build_fast, run_grid, run_selected  # noqa: E402
from src.ma_slope.kernel import S_N, S_PNL, S_SQ  # noqa: E402
from src.utils.paths import RESULTS_DIR  # noqa: E402

RAW_DIR = RESULTS_DIR / "ma_slope"


def stage1(tickers) -> None:
    """Full grid at the pre-registered baseline slope settings."""
    for t in tickers:
        if done("stage1", t):
            continue
        t0 = time.time()
        ins = load_instrument(t)
        fast = build_fast(ins)
        print(f"{t}: {ins.n} bars, {fast.cross_idx.size} crossings", flush=True)
        run_grid(ins, fast, RAW_DIR / "stage1" / t, slopes=BASELINE,
                 log=lambda m: print(m, flush=True))
        print(f"{t}: stage1 done in {time.time() - t0:.0f}s", flush=True)


def stage2(tickers) -> None:
    """All 96 slope settings on the log-spaced period lattice (no year buckets)."""
    for t in tickers:
        if done("stage2", t):
            continue
        t0 = time.time()
        ins = load_instrument(t)
        fast = build_fast(ins, periods=FAST_LATTICE)
        run_grid(ins, fast, RAW_DIR / "stage2" / t, slopes=SLOPE_GRID, slow_periods=SLOW_LATTICE,
                 with_years=False, log=lambda m: print(m, flush=True))
        print(f"{t}: stage2 done in {time.time() - t0:.0f}s", flush=True)


def slope_table(ticker: str) -> pd.DataFrame:
    """In-sample summary of every slope setting per exit model (stage 2)."""
    d = RAW_DIR / "stage2" / ticker
    s = np.load(d / "stats.npy", mmap_mode="r")  # (slow_t, slow_p, slope, fast_t, fast_p, model, group, stat)
    rows = []
    for k, sl in enumerate(SLOPE_GRID):
        for model in (0, 1):
            g = {name: np.asarray(s[:, :, k, :, :, model, GROUP[name], :], np.float64) for name in ("is", "oos")}
            t_is = t_stat(g["is"][..., S_N], g["is"][..., S_PNL], g["is"][..., S_SQ])
            t_oos = t_stat(g["oos"][..., S_N], g["oos"][..., S_PNL], g["oos"][..., S_SQ])
            ok = (g["is"][..., S_N] >= MIN_TRADES_IS) & np.isfinite(t_is)
            rows.append({"ticker": ticker, "method": ("pct", "atr")[sl.method], "k": sl.k, "eps": sl.eps,
                         "atr_n": sl.atr_n if sl.method else None, "slope": k, "model": model,
                         "configs": int(ok.sum()), "is_median_t": float(np.median(t_is[ok])) if ok.any() else np.nan,
                         "is_share_pos": float(np.mean(t_is[ok] > 0)) if ok.any() else np.nan,
                         "is_top1pct_t": float(np.percentile(t_is[ok], 99)) if ok.any() else np.nan,
                         "oos_median_t": float(np.nanmedian(t_oos[ok])) if ok.any() else np.nan,
                         "oos_share_pos": float(np.nanmean(t_oos[ok] > 0)) if ok.any() else np.nan})
    return pd.DataFrame(rows)


def choose_slopes(table: pd.DataFrame) -> dict:
    """Per (method, model): the setting with the best in-sample median t (OOS columns unused)."""
    chosen = {}
    for (method, model), g in table.groupby(["method", "model"]):
        r = g.loc[g["is_median_t"].idxmax()]
        chosen[(0 if method == "pct" else 1, int(model))] = SLOPE_GRID[int(r["slope"])]
    return chosen


def stage3(tickers) -> None:
    """Full grid with the stage-2 in-sample choice of slope setting per (method, model)."""
    for t in tickers:
        if done("stage3", t):
            continue
        t0 = time.time()
        table = slope_table(t)
        table.to_csv(RAW_DIR / "stage2" / t / "slope_table.csv", index=False)
        chosen = choose_slopes(table)
        print(t, {f"{k}": v.label for k, v in chosen.items()}, flush=True)
        ins = load_instrument(t)
        run_selected(ins, build_fast(ins), RAW_DIR / "stage3" / t, chosen,
                     log=lambda m: print(m, flush=True))
        print(f"{t}: stage3 done in {time.time() - t0:.0f}s", flush=True)


def done(stage: str, ticker: str) -> bool:
    """A stage is finished for a ticker once its meta.json is written (runs are resumable)."""
    if (RAW_DIR / stage / ticker / "meta.json").exists():
        print(f"{ticker}: {stage} already done, skipped", flush=True)
        return True
    return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["stage1", "stage2", "stage3"])
    p.add_argument("--tickers", nargs="+", default=list(TICKERS))
    args = p.parse_args()
    {"stage1": stage1, "stage2": stage2, "stage3": stage3}[args.stage](args.tickers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
