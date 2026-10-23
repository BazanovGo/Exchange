"""Шесть скользящих средних — точные формулы.

Обозначения: ``C_t`` — Close сигнального ряда, ``V_t`` — объём, ``n`` — период.
Значение не определено (NaN), пока в окне меньше ``n`` баров. Все реализации —
рекуррентные O(N) с точным пересчётом окна каждые ``RESYNC`` баров, чтобы
ошибка округления не накапливалась на 600 тыс. барах.

SMA   ``SMA_t = (1/n) · Σ_{i=0}^{n-1} C_{t-i}``

EMA   ``α = 2/(n+1)``; ``EMA_{n-1} = SMA_{n-1}`` (инициализация простой средней
      первых n баров, как в TA-Lib); ``EMA_t = α·C_t + (1−α)·EMA_{t-1}``.

WMA   линейные веса, самый свежий бар весит n:
      ``WMA_t = Σ_{i=0}^{n-1} (n−i)·C_{t-i} / (n(n+1)/2)``.

SSMA  сглаженная (Wilder, RMA, SMMA): ``α = 1/n``; ``SSMA_{n-1} = SMA_{n-1}``;
      ``SSMA_t = ((n−1)·SSMA_{t-1} + C_t) / n``.

VWMA  ``VWMA_t = Σ_{i=0}^{n-1} C_{t-i}·V_{t-i} / Σ_{i=0}^{n-1} V_{t-i}``; если
      объём в окне нулевой — NaN (на этих данных таких баров нет).

TMA   треугольная, определение TA-Lib TRIMA — SMA от SMA с общим окном n:
      n нечётное: ``m = (n+1)/2``, ``TMA = SMA_m(SMA_m(C))``;
      n чётное:   ``TMA = SMA_{n/2+1}(SMA_{n/2}(C))``.
      Веса треугольные, сумма длин окон минус один равна n.
"""
from __future__ import annotations

import numpy as np
from numba import njit

TYPES = ("SMA", "EMA", "WMA", "SSMA", "VWMA", "TMA")
RESYNC = 2048


@njit(cache=True)
def _first_valid(x):
    for i in range(x.shape[0]):
        if not np.isnan(x[i]):
            return i
    return x.shape[0]


@njit(cache=True)
def sma(x, n):
    out = np.full(x.shape[0], np.nan)
    s0 = _first_valid(x)
    if s0 + n > x.shape[0]:
        return out
    acc = 0.0
    for i in range(s0, s0 + n):
        acc += x[i]
    out[s0 + n - 1] = acc / n
    for t in range(s0 + n, x.shape[0]):
        if (t - s0) % RESYNC == 0:
            acc = 0.0
            for i in range(t - n + 1, t + 1):
                acc += x[i]
        else:
            acc += x[t] - x[t - n]
        out[t] = acc / n
    return out


@njit(cache=True)
def ema(x, n):
    out = np.full(x.shape[0], np.nan)
    s0 = _first_valid(x)
    if s0 + n > x.shape[0]:
        return out
    acc = 0.0
    for i in range(s0, s0 + n):
        acc += x[i]
    prev = acc / n
    out[s0 + n - 1] = prev
    a = 2.0 / (n + 1.0)
    for t in range(s0 + n, x.shape[0]):
        prev = a * x[t] + (1.0 - a) * prev
        out[t] = prev
    return out


@njit(cache=True)
def ssma(x, n):
    out = np.full(x.shape[0], np.nan)
    s0 = _first_valid(x)
    if s0 + n > x.shape[0]:
        return out
    acc = 0.0
    for i in range(s0, s0 + n):
        acc += x[i]
    prev = acc / n
    out[s0 + n - 1] = prev
    for t in range(s0 + n, x.shape[0]):
        prev = ((n - 1.0) * prev + x[t]) / n
        out[t] = prev
    return out


@njit(cache=True)
def wma(x, n):
    """S_t = S_{t-1} + n·C_t − W_{t-1}, где W — простая сумма окна, кончающегося в t-1."""
    out = np.full(x.shape[0], np.nan)
    s0 = _first_valid(x)
    if s0 + n > x.shape[0]:
        return out
    denom = n * (n + 1) / 2.0
    s = 0.0
    w = 0.0
    for j in range(n):
        s += (j + 1.0) * x[s0 + j]
        w += x[s0 + j]
    out[s0 + n - 1] = s / denom
    for t in range(s0 + n, x.shape[0]):
        if (t - s0) % RESYNC == 0:
            s = 0.0
            w = 0.0
            for j in range(n):
                s += (j + 1.0) * x[t - n + 1 + j]
                w += x[t - n + 1 + j]
        else:
            s = s + n * x[t] - w
            w = w + x[t] - x[t - n]
        out[t] = s / denom
    return out


@njit(cache=True)
def vwma(x, v, n):
    out = np.full(x.shape[0], np.nan)
    s0 = _first_valid(x)
    if s0 + n > x.shape[0]:
        return out
    pv = 0.0
    vv = 0.0
    for i in range(s0, s0 + n):
        pv += x[i] * v[i]
        vv += v[i]
    out[s0 + n - 1] = pv / vv if vv > 0 else np.nan
    for t in range(s0 + n, x.shape[0]):
        if (t - s0) % RESYNC == 0:
            pv = 0.0
            vv = 0.0
            for i in range(t - n + 1, t + 1):
                pv += x[i] * v[i]
                vv += v[i]
        else:
            pv += x[t] * v[t] - x[t - n] * v[t - n]
            vv += v[t] - v[t - n]
        out[t] = pv / vv if vv > 0 else np.nan
    return out


@njit(cache=True)
def tma(x, n):
    if n % 2 == 1:
        m = (n + 1) // 2
        return sma(sma(x, m), m)
    return sma(sma(x, n // 2), n // 2 + 1)


def moving_average(kind: str, close: np.ndarray, volume: np.ndarray, n: int) -> np.ndarray:
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
    raise ValueError(kind)


@njit(cache=True)
def atr(high, low, close, p):
    """ATR Уайлдера: TR_t = max(H−L, |H−C_{t-1}|, |L−C_{t-1}|), сглаживание SSMA(p)."""
    n = close.shape[0]
    tr = np.full(n, np.nan)
    for t in range(1, n):
        a = high[t] - low[t]
        b = abs(high[t] - close[t - 1])
        c = abs(low[t] - close[t - 1])
        tr[t] = max(a, max(b, c))
    return ssma(tr, p)
