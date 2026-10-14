"""Signals of the study: price/Fast-MA crossings and the Slow-MA slope state.

Crossing. ``side[t] = sign(close[t] - fast[t])``; a tie keeps the previous
side, so crossings strictly alternate. A cross up at bar ``t`` is
``side[t-1] = -1, side[t] = +1`` on two consecutive closed bars (cross down
mirrored); bars before the MA is defined have no side. The order fills at
``open[t+1]``.

Slope state of the Slow MA over ``k`` bars:

* method A (percent): ``slope = (MA[t] - MA[t-k]) / MA[t-k] * 100``;
* method B (ATR): ``slope = (MA[t] - MA[t-k]) / ATR[t]``;

``UP`` if ``slope > eps``, ``DOWN`` if ``slope < -eps``, ``NEUTRAL`` otherwise,
``INVALID`` while ``MA[t-k]`` or ``ATR[t]`` is undefined (no entries then).
"""

from __future__ import annotations

import numpy as np
from numba import njit

UP, NEUTRAL, DOWN, INVALID = 1, 0, -1, 2


@njit(cache=True)
def cross_events(close: np.ndarray, fast: np.ndarray):
    """Bars and directions (+1 up, -1 down) of price crossings of ``fast``."""
    n = close.size
    idx = np.empty(n, np.int32)
    direction = np.empty(n, np.int8)
    m = 0
    side = 0
    for t in range(n):
        if np.isnan(fast[t]):
            continue
        diff = close[t] - fast[t]
        new = 1 if diff > 0 else (-1 if diff < 0 else side)
        if side != 0 and new != side:
            idx[m] = t
            direction[m] = new
            m += 1
        side = new
    return idx[:m].copy(), direction[:m].copy()


@njit(cache=True)
def slope_state(ma: np.ndarray, k: int, eps: float, atr: np.ndarray, method: int) -> np.ndarray:
    """State per bar; ``method`` 0 = percent change (A), 1 = ATR-normalized (B)."""
    n = ma.size
    out = np.full(n, INVALID, np.int8)
    for t in range(k, n):
        prev = ma[t - k]
        if np.isnan(prev) or np.isnan(ma[t]):
            continue
        if method == 0:
            slope = (ma[t] - prev) / prev * 100.0
        else:
            a = atr[t]
            if np.isnan(a) or a <= 0:
                continue
            slope = (ma[t] - prev) / a
        if slope > eps:
            out[t] = UP
        elif slope < -eps:
            out[t] = DOWN
        else:
            out[t] = NEUTRAL
    return out


@njit(cache=True)
def next_state(state: np.ndarray, value: int) -> np.ndarray:
    """``out[i]`` = first bar ``j >= i`` with ``state[j] == value`` (``n`` if none)."""
    n = state.size
    out = np.empty(n, np.int32)
    nxt = n
    for i in range(n - 1, -1, -1):
        if state[i] == value:
            nxt = i
        out[i] = nxt
    return out
