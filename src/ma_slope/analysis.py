"""Metrics, selection and robustness checks on the full grids.

Canonical config axes (stage 1 and stage 3 grids):
``(slow_type, slow_period, method, fast_type, fast_period, model)``.

Selection never uses the raw best: a config is scored by its *plateau* — the
mean in-sample t-statistic of the trade PnL over the neighbouring periods
(fast +-2, slow +-5) with the same MA types, slope method and exit model —
and must have enough in-sample trades. Out-of-sample data (the last 30% of
bars) is touched only to evaluate the already selected configs.
"""

from __future__ import annotations

import json
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy import stats as sps

from src.ma_slope.kernel import S_BARS, S_DD, S_GL, S_GP, S_MIN, S_N, S_NLONG, S_PNL, S_PNLLONG, S_SQ, S_STREAK, S_WIN

AXES = ("slow_type", "slow_period", "method", "fast_type", "fast_period", "model")
PERIOD_AXES = (1, 4)                 # slow_period, fast_period
WINDOW = {1: 11, 4: 5}               # slow +-5, fast +-2
MIN_TRADES_IS = 100
MIN_TRADES_TRAIN = 30
MIN_VALID_NEIGHBOURS = 0.8
GROUP = {"is": 0, "oos": 1, "all": 2}


def load_grid(path: Path):
    meta = json.loads((path / "meta.json").read_text())
    stats = np.load(path / "stats.npy", mmap_mode="r")
    years = np.load(path / "years.npy", mmap_mode="r") if (path / "years.npy").exists() else None
    return stats, years, meta


def t_stat(n, s, q):
    """t-statistic of the mean trade PnL from count, sum and sum of squares."""
    n = np.asarray(n, np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / n
        var = (q - n * mean * mean) / (n - 1)
        t = mean / np.sqrt(var / n)
    return np.where((n > 1) & (var > 0), t, np.nan)


def metrics(g: np.ndarray) -> dict[str, np.ndarray]:
    """Per-config metrics from a ``[..., NSTAT]`` slice of one group."""
    g = np.asarray(g, np.float64)
    n, wins = g[..., S_N], g[..., S_WIN]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = {
            "trades": n,
            "pnl": g[..., S_PNL],
            "win_rate": wins / n,
            "profit_factor": g[..., S_GP] / g[..., S_GL],
            "avg_trade": g[..., S_PNL] / n,
            "avg_win": g[..., S_GP] / wins,
            "avg_loss": -g[..., S_GL] / (n - wins),
            "t": t_stat(n, g[..., S_PNL], g[..., S_SQ]),
            "max_dd": g[..., S_DD],
            "max_loss_streak": g[..., S_STREAK],
            "avg_bars": g[..., S_BARS] / n,
            "avg_hours": g[..., S_MIN] / n / 60,
            "long_share": g[..., S_NLONG] / n,
            "long_pnl": g[..., S_PNLLONG],
        }
    out["recovery"] = np.where(out["max_dd"] > 0, out["pnl"] / out["max_dd"], np.nan)
    return out


def plateau(t: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Neighbourhood mean of ``t`` and share of positive neighbours (period axes only)."""
    size = [1] * t.ndim
    for ax, w in WINDOW.items():
        size[ax] = w
    vals = np.where(valid, t, 0.0)
    pos = np.where(valid, (t > 0).astype(float), 0.0)
    cnt = ndimage.uniform_filter(valid.astype(float), size, mode="constant")
    inb = ndimage.uniform_filter(np.ones(t.shape), size, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = ndimage.uniform_filter(vals, size, mode="constant") / cnt
        share = ndimage.uniform_filter(pos, size, mode="constant") / cnt
    ok = valid & (cnt >= MIN_VALID_NEIGHBOURS * inb)
    return np.where(ok, mean, np.nan), np.where(ok, share, np.nan)


def is_scores(stats: np.ndarray, min_trades: int = MIN_TRADES_IS):
    """In-sample t, its plateau, positive-neighbour share and the validity mask."""
    g = np.asarray(stats[..., GROUP["is"], :], np.float64)
    t = t_stat(g[..., S_N], g[..., S_PNL], g[..., S_SQ])
    valid = (g[..., S_N] >= min_trades) & np.isfinite(t)
    score, share = plateau(t, valid)
    return t, score, share, valid


def labels(meta: dict, idx: tuple) -> dict:
    st, sp, m, ft, fp, x = idx
    return {
        "slow_type": meta["slow_types"][st], "slow_period": meta["slow_periods"][sp],
        "method": ("pct", "atr")[m], "fast_type": meta["fast_types"][ft],
        "fast_period": meta["fast_periods"][fp], "model": meta["models"][x],
    }


def config_row(stats, meta, idx, score=None, share=None) -> dict:
    row = labels(meta, idx)
    for gname, g in GROUP.items():
        for k, v in metrics(stats[idx][g]).items():
            row[f"{gname}_{k}"] = float(v)
    if score is not None:
        row["is_plateau_t"] = float(score[idx])
        row["is_plateau_pos_share"] = float(share[idx])
    return row


def best(score: np.ndarray, mask: np.ndarray | None = None):
    s = np.where(mask, score, np.nan) if mask is not None else score
    if not np.isfinite(s).any():
        return None
    return np.unravel_index(int(np.nanargmax(s)), s.shape)


def category_mask(meta: dict, shape) -> np.ndarray:
    """True where fast period < slow period (the standard category)."""
    sp = np.array(meta["slow_periods"]).reshape(1, -1, 1, 1, 1, 1)
    fp = np.array(meta["fast_periods"]).reshape(1, 1, 1, 1, -1, 1)
    return np.broadcast_to(fp < sp, shape)


# ---------------------------------------------------------------- testing

def p_value(t, n):
    """One-sided p-value of a positive mean (Student t with n-1 dof)."""
    return float(sps.t.sf(t, max(n - 1, 1))) if np.isfinite(t) and n > 1 else np.nan


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m = p.notna().sum()
    adj, running = pd.Series(np.nan, index=p.index), 0.0
    for i, k in enumerate([k for k in order if pd.notna(p[k])]):
        running = max(running, min(1.0, (m - i) * p[k]))
        adj[k] = running
    return adj


def benjamini_hochberg(p: pd.Series) -> pd.Series:
    vals = p.dropna().sort_values()
    m = len(vals)
    q = (vals * m / np.arange(1, m + 1))[::-1].cummin()[::-1].clip(upper=1.0)
    return q.reindex(p.index)


def expected_max_t(n_trials: float) -> float:
    """Expected maximum of ``n_trials`` independent N(0,1) t-statistics (Bailey & Lopez de Prado)."""
    g = 0.5772156649
    return float((1 - g) * sps.norm.ppf(1 - 1 / n_trials) + g * sps.norm.ppf(1 - 1 / (n_trials * np.e)))


# ---------------------------------------------------------------- walk-forward

def year_t(years_slice: np.ndarray) -> np.ndarray:
    """t from ``[..., 3]`` year sums (pnl, trades, sum_sq)."""
    return t_stat(years_slice[..., 1], years_slice[..., 0], years_slice[..., 2])


def walk_forward(years: np.ndarray, meta: dict, train: int = 3, mask: np.ndarray | None = None) -> pd.DataFrame:
    """Rolling ``train``-year selection by plateau t, tested on the next calendar year."""
    y = np.asarray(years, np.float32)
    rows = []
    for test in range(train, meta["n_years"]):
        tr = y[..., test - train:test, :].sum(axis=-2, dtype=np.float64)
        t = year_t(tr)
        valid = (tr[..., 1] >= MIN_TRADES_TRAIN) & np.isfinite(t)
        if mask is not None:
            valid &= mask
        score, _ = plateau(t, valid)
        idx = best(score)
        if idx is None:
            continue
        te = y[idx][test].astype(np.float64)
        rows.append({"test_year": meta["first_year"] + test, **labels(meta, idx),
                     "train_plateau_t": float(score[idx]), "train_t": float(t[idx]),
                     "train_pnl": float(tr[idx][0]), "train_trades": float(tr[idx][1]),
                     "test_pnl": float(te[0]), "test_trades": float(te[1]),
                     "test_t": float(year_t(te))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- PBO (CSCV)

def pbo_cscv(years: np.ndarray, meta: dict, n_blocks: int = 10, mask: np.ndarray | None = None,
             max_configs: int | None = None, seed: int = 0, chunk: int = 12) -> dict:
    """Probability of backtest overfitting by combinatorially symmetric cross-validation.

    Calendar years are grouped into ``n_blocks`` contiguous blocks (fewer if
    the history is short; always even); for every half/half split the config
    with the best t on one half is ranked by t on the other half.
    """
    y = np.asarray(years, np.float32).reshape(-1, meta["n_years"], 3)
    flat_mask = np.ones(y.shape[0], bool) if mask is None else np.asarray(mask).reshape(-1)
    flat_mask &= y[:, :, 1].sum(axis=1) >= MIN_TRADES_IS
    ids = np.flatnonzero(flat_mask)
    if max_configs and ids.size > max_configs:
        ids = np.sort(np.random.default_rng(seed).choice(ids, max_configs, replace=False))
    y = y[ids].astype(np.float64)
    n_years = meta["n_years"]
    s = min(n_blocks, n_years - n_years % 2)
    edges = np.linspace(0, n_years, s + 1).round().astype(int)
    blocks = np.stack([y[:, a:b].sum(axis=1) for a, b in zip(edges[:-1], edges[1:], strict=True)], 1)
    del y
    total = blocks.sum(axis=1)
    per_stat = [np.ascontiguousarray(blocks[:, :, j]) for j in range(3)]
    del blocks
    halves = list(combinations(range(s), s // 2))
    masks = np.zeros((s, len(halves)))
    for c, h in enumerate(halves):
        masks[list(h), c] = 1.0
    logits, ranks = [], []
    for c0 in range(0, len(halves), chunk):
        m = masks[:, c0:c0 + chunk]
        ins = [b @ m for b in per_stat]                        # (configs, splits) sums per stat
        t_in = t_stat(ins[1], ins[0], ins[2])
        t_in[ins[1] < MIN_TRADES_TRAIN] = np.nan
        t_out = t_stat(total[:, 1:2] - ins[1], total[:, 0:1] - ins[0], total[:, 2:3] - ins[2])
        del ins
        for col in range(m.shape[1]):
            if not np.isfinite(t_in[:, col]).any():
                continue
            k = int(np.nanargmax(t_in[:, col]))
            out = t_out[:, col]
            valid = np.isfinite(out)
            w = (np.sum(out[valid] < out[k]) + 0.5) / (valid.sum() + 1)
            ranks.append(w)
            logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    return {"blocks": s, "splits": comb(s, s // 2), "configs": int(ids.size),
            "pbo": float(np.mean(logits <= 0)), "median_oos_rank": float(np.median(ranks)),
            "logits": logits}


# ---------------------------------------------------------------- Monte Carlo

def monte_carlo(pnl: np.ndarray, n_sims: int = 10_000, seed: int = 0, chunk: int = 500) -> dict:
    """Bootstrap the trade sequence: total PnL and max drawdown distributions."""
    rng = np.random.default_rng(seed)
    pnl = np.asarray(pnl, np.float64)
    totals, dds = [], []
    for start in range(0, n_sims, chunk):
        k = min(chunk, n_sims - start)
        sample = pnl[rng.integers(0, pnl.size, (k, pnl.size))]
        eq = np.cumsum(sample, axis=1)
        dd = (np.maximum.accumulate(np.maximum(eq, 0), axis=1) - eq).max(axis=1)
        totals.append(eq[:, -1])
        dds.append(dd)
    totals, dds = np.concatenate(totals), np.concatenate(dds)
    q = [5, 50, 95]
    return {"p_loss": float(np.mean(totals <= 0)),
            **{f"pnl_p{a}": float(v) for a, v in zip(q, np.percentile(totals, q), strict=True)},
            **{f"max_dd_p{a}": float(v) for a, v in zip(q, np.percentile(dds, q), strict=True)}}
