"""The six moving averages of the study, as exact numba kernels.

All take ``float64`` arrays and return a ``float64`` array of the same length
with ``NaN`` until the first full window (index ``n - 1``). Formulas, for
period ``n`` and price ``x``:

* SMA  — ``mean(x[t-n+1 .. t])``.
* EMA  — ``e[t] = a*x[t] + (1-a)*e[t-1]``, ``a = 2/(n+1)``, seeded with
  ``e[n-1] = SMA_n`` (TA-Lib convention).
* WMA  — linear weights ``n, n-1, .., 1`` (newest bar heaviest):
  ``sum((n-i) * x[t-i]) / (n(n+1)/2)``, ``i = 0..n-1``.
* SSMA — smoothed (Wilder/MetaTrader SMMA): ``s[t] = (s[t-1]*(n-1) + x[t]) / n``,
  seeded with ``s[n-1] = SMA_n``. Equals an EMA with ``a = 1/n``.
* VWMA — ``sum(x*v) / sum(v)`` over the last ``n`` bars, ``v`` = traded volume.
* TMA  — triangular: ``SMA(SMA(x, ceil(n/2)), floor(n/2) + 1)``; the two
  windows span exactly ``n`` bars with weights ``1, 2, .., peak, .., 2, 1``.

Running sums are recomputed from scratch every ``_RESYNC`` bars so float
drift stays below 1e-9 of the price over millions of bars.
"""

from __future__ import annotations

import numpy as np
from numba import njit

MA_TYPES = ("SMA", "EMA", "WMA", "SSMA", "VWMA", "TMA")
_RESYNC = 4096


@njit(cache=True)
def sma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(x.size, np.nan)
    if n > x.size:
        return out
    s = 0.0
    for t in range(x.size):
        if t >= n and t % _RESYNC == 0:
            s = 0.0
            for j in range(t - n + 1, t + 1):
                s += x[j]
        else:
            s += x[t]
            if t >= n:
                s -= x[t - n]
        if t >= n - 1:
            out[t] = s / n
    return out


@njit(cache=True)
def _exp_smooth(x: np.ndarray, n: int, a: float) -> np.ndarray:
    out = np.full(x.size, np.nan)
    if n > x.size:
        return out
    s = 0.0
    for t in range(n):
        s += x[t]
    e = s / n
    out[n - 1] = e
    for t in range(n, x.size):
        e = a * x[t] + (1.0 - a) * e
        out[t] = e
    return out


@njit(cache=True)
def ema(x: np.ndarray, n: int) -> np.ndarray:
    return _exp_smooth(x, n, 2.0 / (n + 1.0))


@njit(cache=True)
def ssma(x: np.ndarray, n: int) -> np.ndarray:
    return _exp_smooth(x, n, 1.0 / n)


@njit(cache=True)
def wma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(x.size, np.nan)
    if n > x.size:
        return out
    denom = n * (n + 1) / 2.0
    s = 0.0    # plain sum of the window
    num = 0.0  # weighted sum of the window
    for t in range(x.size):
        if t >= n and t % _RESYNC == 0:
            s = 0.0
            num = 0.0
            for i in range(n):
                s += x[t - i]
                num += (n - i) * x[t - i]
        else:
            # Shift the window: every weight drops by one, the new bar gets n.
            num += n * x[t] - s
            s += x[t]
            if t >= n:
                s -= x[t - n]
        if t >= n - 1:
            out[t] = num / denom
    return out


@njit(cache=True)
def vwma(x: np.ndarray, v: np.ndarray, n: int) -> np.ndarray:
    out = np.full(x.size, np.nan)
    if n > x.size:
        return out
    pv = 0.0
    sv = 0.0
    for t in range(x.size):
        if t >= n and t % _RESYNC == 0:
            pv = 0.0
            sv = 0.0
            for j in range(t - n + 1, t + 1):
                pv += x[j] * v[j]
                sv += v[j]
        else:
            pv += x[t] * v[t]
            sv += v[t]
            if t >= n:
                pv -= x[t - n] * v[t - n]
                sv -= v[t - n]
        if t >= n - 1:
            out[t] = pv / sv
    return out


@njit(cache=True)
def tma(x: np.ndarray, n: int) -> np.ndarray:
    n1 = (n + 1) // 2  # ceil(n / 2)
    n2 = n // 2 + 1
    first = sma(x, n1)
    out = np.full(x.size, np.nan)
    if n > x.size:
        return out
    inner = sma(first[n1 - 1:], n2)
    out[n1 - 1:] = inner
    return out


def moving_average(kind: str, close: np.ndarray, volume: np.ndarray, n: int) -> np.ndarray:
    """Dispatch by name (one of :data:`MA_TYPES`)."""
    if kind == "SMA":
        return sma(close, n)
    if kind == "EMA":
        return ema(close, n)
    if kind == "WMA":
        return wma(close, n)
    if kind == "SSMA":
        return ssma(close, n)
    if kind == "VWMA":
        return vwma(close, volume, n)
    if kind == "TMA":
        return tma(close, n)
    raise ValueError(f"unknown moving average {kind!r}; expected one of {MA_TYPES}")


@njit(cache=True)
def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    """Wilder ATR: SSMA (seeded with SMA) of the true range; TR[0] = high - low."""
    tr = np.empty(close.size)
    tr[0] = high[0] - low[0]
    for t in range(1, close.size):
        tr[t] = max(high[t] - low[t], abs(high[t] - close[t - 1]), abs(low[t] - close[t - 1]))
    return ssma(tr, n)
