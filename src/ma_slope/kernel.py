"""Event-driven trade simulator for one Slow-MA state against many Fast MAs.

Rules (one contract, no costs, no stops, one position at a time):

* entry on a Fast-MA cross bar ``t`` when flat and the slope state allows the
  direction (long: UP or NEUTRAL, short: DOWN or NEUTRAL); fill at ``open[t+1]``;
* exit model A — the opposite Fast-MA cross; model B — the first bar after
  the entry signal whose slope state is the opposite one (NEUTRAL alone does
  not exit); fill at ``open[x+1]`` of the exit signal bar ``x``;
* no reversal on the exit bar: the next entry needs a new cross on a later bar;
* a position still open at the end of the data is not counted.

PnL is in instrument money: ``direction * (P[exit] - P[entry]) * multiplier``,
where ``P`` is the cumulative raw-price change of the contract actually held
(see :func:`src.ma_slope.data.held_contract_path`), so a position crossing a
roll is rolled at the overlap price instead of jumping.

Trades are attributed to the period of their exit fill: group 0 = in-sample
(exit before ``split``), 1 = out-of-sample, 2 = whole history; and to the
calendar year of the exit for walk-forward buckets. Selection on IS therefore
never sees a price after ``split``.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

STAT_NAMES = ("trades", "wins", "pnl", "gross_profit", "gross_loss", "sum_sq", "max_dd",
              "max_loss_streak", "bars_held", "minutes_held", "long_trades", "long_pnl")
(S_N, S_WIN, S_PNL, S_GP, S_GL, S_SQ, S_DD, S_STREAK, S_BARS, S_MIN, S_NLONG, S_PNLLONG) = range(12)
NSTAT = len(STAT_NAMES)
GROUPS = ("is", "oos", "all")
YEAR_STATS = ("pnl", "trades", "sum_sq")


@njit(cache=True, inline="always")
def _add(out, years, run, g, pnl, bars, minutes, is_long, year):
    for gg in (g, 2):
        o = out[gg]
        o[S_N] += 1.0
        o[S_PNL] += pnl
        o[S_SQ] += pnl * pnl
        if pnl > 0:
            o[S_WIN] += 1.0
            o[S_GP] += pnl
        else:
            o[S_GL] -= pnl
        o[S_BARS] += bars
        o[S_MIN] += minutes
        if is_long:
            o[S_NLONG] += 1.0
            o[S_PNLLONG] += pnl
        eq = run[gg, 0] + pnl
        run[gg, 0] = eq
        if eq > run[gg, 1]:
            run[gg, 1] = eq
        if run[gg, 1] - eq > o[S_DD]:
            o[S_DD] = run[gg, 1] - eq
        if pnl < 0:
            run[gg, 2] += 1.0
            if run[gg, 2] > o[S_STREAK]:
                o[S_STREAK] = run[gg, 2]
        else:
            run[gg, 2] = 0.0
    years[year, 0] += pnl
    years[year, 1] += 1.0
    years[year, 2] += pnl * pnl


@njit(cache=True)
def _simulate(cross_idx, cross_dir, lo, hi, model, state, next_down, next_up,
              path, mult, split, minutes, year_of_bar, out, years, rec):
    """Run one config; when ``rec`` has rows, also log (entry, exit, direction, pnl)."""
    n = state.size
    run = np.zeros((3, 3))  # per group: equity, peak, current loss streak
    m = 0
    i = lo
    if model == 0:
        pos = 0
        e = 0
        while i < hi:
            t = cross_idx[i]
            d = cross_dir[i]
            if pos == 0:
                s = state[t]
                if t + 1 < n and s != 2 and (s == 0 or s == d):
                    pos = d
                    e = t + 1
            else:
                x = t + 1
                if x >= n:
                    break
                pnl = pos * (path[x] - path[e]) * mult
                _add(out, years, run, 0 if x < split else 1, pnl, x - e,
                     minutes[x] - minutes[e], pos == 1, year_of_bar[x])
                if m < rec.shape[0]:
                    rec[m, 0], rec[m, 1], rec[m, 2], rec[m, 3] = e, x, pos, pnl
                m += 1
                pos = 0
            i += 1
    else:
        while i < hi:
            t = cross_idx[i]
            d = cross_dir[i]
            s = state[t]
            if t + 1 < n and s != 2 and (s == 0 or s == d):
                e = t + 1
                j = next_down[e] if d == 1 else next_up[e]
                if j >= n - 1:
                    break
                x = j + 1
                pnl = d * (path[x] - path[e]) * mult
                _add(out, years, run, 0 if x < split else 1, pnl, x - e,
                     minutes[x] - minutes[e], d == 1, year_of_bar[x])
                if m < rec.shape[0]:
                    rec[m, 0], rec[m, 1], rec[m, 2], rec[m, 3] = e, x, d, pnl
                m += 1
                i += 1
                while i < hi and cross_idx[i] <= j:
                    i += 1
                continue
            i += 1
    return m


@njit(parallel=True, cache=True)
def run_slow_state(cross_idx, cross_dir, offsets, state, next_down, next_up, path, mult,
                   split, minutes, year_of_bar, n_years, models=(True, True)):
    """Stats of every Fast MA (``offsets`` slices of the cross arrays) x both exit models.

    Returns ``stats[fast, model, group, stat]`` and ``years[fast, model, year, stat]``;
    a model switched off in ``models`` is left at zero.
    """
    nf = offsets.size - 1
    stats = np.zeros((nf, 2, 3, NSTAT))
    years = np.zeros((nf, 2, n_years, 3))
    no_rec = np.zeros((0, 4))
    for k in prange(nf * 2):
        f = k // 2
        model = k % 2
        if not models[model]:
            continue
        _simulate(cross_idx, cross_dir, offsets[f], offsets[f + 1], model, state, next_down,
                  next_up, path, mult, split, minutes, year_of_bar, stats[f, model], years[f, model],
                  no_rec)
    return stats, years


def trade_list(cross_idx, cross_dir, model, state, next_down, next_up, path, mult, split,
               minutes, year_of_bar, n_years):
    """Trades of one config: ``(entry_bar, exit_bar, direction, pnl)`` rows plus its stats."""
    stats = np.zeros((3, NSTAT))
    years = np.zeros((n_years, 3))
    rec = np.zeros((cross_idx.size // 2 + 1, 4))
    m = _simulate(cross_idx, cross_dir, 0, cross_idx.size, model, state, next_down, next_up, path,
                  mult, split, minutes, year_of_bar, stats, years, rec)
    return rec[:m], stats, years
