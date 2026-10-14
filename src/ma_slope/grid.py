"""Full-grid runs: every Fast MA x Slow MA x slope setting x exit model.

Results are dense ``float32`` arrays on disk (memory-mapped ``.npy``):

* ``stats[slow_type, slow_period, slope, fast_type, fast_period, model, group, stat]``
* ``years[slow_type, slow_period, slope, fast_type, fast_period, model, year, (pnl, trades, sum_sq)]``

with groups (is, oos, all) and stats as in :data:`src.ma_slope.kernel.STAT_NAMES`.
``meta.json`` next to them records the axes.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src.ma_slope.averages import MA_TYPES, atr, moving_average
from src.ma_slope.data import Instrument
from src.ma_slope.kernel import GROUPS, NSTAT, STAT_NAMES, YEAR_STATS, run_slow_state
from src.ma_slope.signals import DOWN, UP, cross_events, next_state, slope_state

FAST_PERIODS = tuple(range(2, 100))
SLOW_PERIODS = tuple(range(5, 251))
EXIT_MODELS = ("reverse_cross", "slope_flip")
METHODS = ("pct", "atr")


@dataclass(frozen=True)
class Slope:
    """Slope rule: method 0 = percent change (A), 1 = ATR-normalized (B)."""

    method: int
    k: int
    eps: float
    atr_n: int = 14

    @property
    def label(self) -> str:
        if self.method == 0:
            return f"pct k={self.k} eps={self.eps:g}%"
        return f"atr k={self.k} eps={self.eps:g} atr={self.atr_n}"


# Stage-1 settings, fixed before any backtest from the neutral-zone share on
# in-sample bars (about 10-35% of bars neutral for EMA20..EMA250).
BASELINE = (Slope(0, 10, 0.02), Slope(1, 10, 0.25, 14))

# Stage 2: slope settings swept on a log-spaced period lattice. eps = 0 makes
# both methods identical (sign of the change), so it is listed once, under pct.
EPS_PCT = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1)
ATR_GRID = (14, 100)
EPS_ATR = (0.05, 0.1, 0.25, 0.5, 1.0)


def _slopes(ks):
    return tuple([Slope(0, k, e) for k in ks for e in EPS_PCT]
                 + [Slope(1, k, e, a) for k in ks for a in ATR_GRID for e in EPS_ATR])


# The first sweep (k up to 40) put most optima on its upper edge, so it was
# extended to k = 80, 160 (stage 2b); the extension is appended, keeping the
# indices of the first 96 settings.
K_GRID_BASE = (1, 3, 5, 10, 20, 40)
K_GRID_EXT = (80, 160)
K_GRID = K_GRID_BASE + K_GRID_EXT
SLOPE_GRID_BASE = _slopes(K_GRID_BASE)
SLOPE_GRID_EXT = _slopes(K_GRID_EXT)
SLOPE_GRID = SLOPE_GRID_BASE + SLOPE_GRID_EXT
FAST_LATTICE = (2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 32, 40, 50, 63, 79, 99)
SLOW_LATTICE = (5, 7, 10, 14, 20, 28, 40, 56, 80, 113, 160, 226, 250)


@dataclass
class FastSet:
    """Price/Fast-MA crossings of every fast MA, concatenated."""

    cross_idx: np.ndarray
    cross_dir: np.ndarray
    offsets: np.ndarray
    types: tuple[str, ...]
    periods: tuple[int, ...]


def build_fast(ins: Instrument, types=MA_TYPES, periods=FAST_PERIODS) -> FastSet:
    idx, dirs, offsets = [], [], [0]
    for kind in types:
        for p in periods:
            i, d = cross_events(ins.close, moving_average(kind, ins.close, ins.volume, p))
            idx.append(i)
            dirs.append(d)
            offsets.append(offsets[-1] + i.size)
    return FastSet(np.concatenate(idx), np.concatenate(dirs), np.array(offsets, np.int64),
                   tuple(types), tuple(periods))


def run_grid(ins: Instrument, fast: FastSet, out_dir: Path, *, slopes, slow_types=MA_TYPES,
             slow_periods=SLOW_PERIODS, with_years: bool = True, models=None, log=print,
             extra_meta=None) -> Path:
    """Run the grid for ``ins`` and write ``stats.npy`` / ``years.npy`` / ``meta.json``.

    ``models[s]`` = (reverse_cross, slope_flip) flags to compute for slope ``s``
    (default: both).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    nft, nfp = len(fast.types), len(fast.periods)
    shape = (len(slow_types), len(slow_periods), len(slopes), nft, nfp, 2)
    stats = np.lib.format.open_memmap(out_dir / "stats.npy", "w+", np.float32, (*shape, 3, NSTAT))
    years = (np.lib.format.open_memmap(out_dir / "years.npy", "w+", np.float32, (*shape, ins.n_years, 3))
             if with_years else None)
    atrs = {n: atr(ins.high, ins.low, ins.close, n) for n in {s.atr_n for s in slopes}}
    t0 = time.time()
    for i, kind in enumerate(slow_types):
        for j, p in enumerate(slow_periods):
            ma = moving_average(kind, ins.close, ins.volume, p)
            for s, sl in enumerate(slopes):
                state = slope_state(ma, sl.k, sl.eps, atrs[sl.atr_n], sl.method)
                st, yr = run_slow_state(fast.cross_idx, fast.cross_dir, fast.offsets, state,
                                        next_state(state, DOWN), next_state(state, UP), ins.path,
                                        ins.multiplier, ins.split, ins.minutes, ins.year_of_bar,
                                        ins.n_years, models[s] if models else (True, True))
                stats[i, j, s] = st.reshape(nft, nfp, 2, 3, NSTAT)
                if years is not None:
                    years[i, j, s] = yr.reshape(nft, nfp, 2, ins.n_years, 3)
        log(f"  {ins.ticker}: slow {kind} done ({i + 1}/{len(slow_types)}), {time.time() - t0:.0f}s")
    stats.flush()
    if years is not None:
        years.flush()
    meta = {
        "ticker": ins.ticker, "bars": ins.n, "start": str(ins.index[0]), "end": str(ins.index[-1]),
        "split": ins.split, "split_time": str(ins.split_time), "first_year": ins.first_year,
        "n_years": ins.n_years, "multiplier": ins.multiplier,
        "axes": ["slow_type", "slow_period", "slope", "fast_type", "fast_period", "model"],
        "slow_types": list(slow_types), "slow_periods": list(slow_periods),
        "slopes": [asdict(s) for s in slopes], "fast_types": list(fast.types),
        "fast_periods": list(fast.periods), "models": list(EXIT_MODELS), "groups": list(GROUPS),
        "stats": list(STAT_NAMES), "year_stats": list(YEAR_STATS), "seconds": round(time.time() - t0),
        **(extra_meta or {}),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def run_selected(ins: Instrument, fast: FastSet, out_dir: Path, chosen: dict, *, log=print,
                 extra_meta=None) -> Path:
    """Full grid where each (method, model) slot uses its own slope setting.

    ``chosen[(method, model)]`` is a :class:`Slope` with ``method`` matching the
    slot. Output layout is the stage-1 one with the slope axis = method.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    nft, nfp = len(fast.types), len(fast.periods)
    shape = (len(MA_TYPES), len(SLOW_PERIODS), 2, nft, nfp, 2)
    stats = np.lib.format.open_memmap(out_dir / "stats.npy", "w+", np.float32, (*shape, 3, NSTAT))
    years = np.lib.format.open_memmap(out_dir / "years.npy", "w+", np.float32, (*shape, ins.n_years, 3))
    unique = sorted(set(chosen.values()), key=lambda s: (s.method, s.k, s.eps, s.atr_n))
    feeds = {u: [slot for slot, s in chosen.items() if s == u] for u in unique}
    atrs = {n: atr(ins.high, ins.low, ins.close, n) for n in {s.atr_n for s in unique}}
    t0 = time.time()
    for i, kind in enumerate(MA_TYPES):
        for j, p in enumerate(SLOW_PERIODS):
            ma = moving_average(kind, ins.close, ins.volume, p)
            for u in unique:
                flags = (any(x == 0 for _, x in feeds[u]), any(x == 1 for _, x in feeds[u]))
                state = slope_state(ma, u.k, u.eps, atrs[u.atr_n], u.method)
                st, yr = run_slow_state(fast.cross_idx, fast.cross_dir, fast.offsets, state,
                                        next_state(state, DOWN), next_state(state, UP), ins.path,
                                        ins.multiplier, ins.split, ins.minutes, ins.year_of_bar,
                                        ins.n_years, flags)
                st = st.reshape(nft, nfp, 2, 3, NSTAT)
                yr = yr.reshape(nft, nfp, 2, ins.n_years, 3)
                for method, model in feeds[u]:
                    stats[i, j, method, :, :, model] = st[:, :, model]
                    years[i, j, method, :, :, model] = yr[:, :, model]
        log(f"  {ins.ticker}: slow {kind} done ({i + 1}/{len(MA_TYPES)}), {time.time() - t0:.0f}s")
    stats.flush()
    years.flush()
    meta = {
        "ticker": ins.ticker, "bars": ins.n, "start": str(ins.index[0]), "end": str(ins.index[-1]),
        "split": ins.split, "split_time": str(ins.split_time), "first_year": ins.first_year,
        "n_years": ins.n_years, "multiplier": ins.multiplier,
        "axes": ["slow_type", "slow_period", "method", "fast_type", "fast_period", "model"],
        "slow_types": list(MA_TYPES), "slow_periods": list(SLOW_PERIODS),
        "chosen": {f"{METHODS[m]}/{EXIT_MODELS[x]}": asdict(s) for (m, x), s in chosen.items()},
        "fast_types": list(fast.types), "fast_periods": list(fast.periods),
        "models": list(EXIT_MODELS), "groups": list(GROUPS), "stats": list(STAT_NAMES),
        "year_stats": list(YEAR_STATS), "seconds": round(time.time() - t0), **(extra_meta or {}),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir
