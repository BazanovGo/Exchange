"""Направление медленной средней: вверх / нейтрально / вниз.

Метод A  ``Slope_t = (MA_t − MA_{t−k}) / MA_{t−k} · 100``   (процент за k баров)
Метод B  ``Slope_t = (MA_t − MA_{t−k}) / ATR_t``            (в долях ATR)

Нейтраль: ``−ε ≤ Slope_t ≤ ε``.

Как задаётся ε
--------------
Порог задаётся через **долю нейтральных баров на in-sample** (coverage):
``ε = quantile(|Slope_t|, coverage)`` по барам до начала out-of-sample.
Это та же оптимизация ε, только в нормированной шкале, и у неё две причины:

1. Масштаб процентного наклона зависит от k, периода и типа средней и от
   инструмента: ε = 0.01% — почти всё нейтрально при k = 1 и почти ничего при
   k = 50. Фиксированная сетка в процентах сравнивала бы не методы, а то, на
   какую часть сетки случайно попал порог.
2. При одинаковой доле нейтрали на IS методы A и B различаются только тем,
   **какие** бары они называют нейтральными. Это и есть честное сравнение.
   На out-of-sample доля нейтрали у метода A плавает вместе с волатильностью,
   у метода B — нормирована ATR; это измеряется и выводится в отчёт.

Квантиль берётся только по in-sample барам — будущее в порог не попадает.
Реальное значение ε (в % или в ATR) сохраняется для каждой конфигурации.
Состояние ``2`` — «не определено» (прогрев): ни входа, ни перехода.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def slope_pct(ma, k):
    n = ma.shape[0]
    out = np.full(n, np.nan)
    for t in range(k, n):
        prev = ma[t - k]
        if not np.isnan(prev) and not np.isnan(ma[t]) and prev != 0:
            out[t] = (ma[t] - prev) / prev * 100.0
    return out


@njit(cache=True)
def slope_atr(ma, atr, k):
    n = ma.shape[0]
    out = np.full(n, np.nan)
    for t in range(k, n):
        prev = ma[t - k]
        a = atr[t]
        if not np.isnan(prev) and not np.isnan(ma[t]) and not np.isnan(a) and a > 0:
            out[t] = (ma[t] - prev) / a
    return out


@njit(cache=True)
def classify(slope, eps):
    n = slope.shape[0]
    st = np.empty(n, dtype=np.int8)
    for t in range(n):
        s = slope[t]
        if np.isnan(s):
            st[t] = 2
        elif s > eps:
            st[t] = 1
        elif s < -eps:
            st[t] = -1
        else:
            st[t] = 0
    return st


def threshold(slope: np.ndarray, oos_bar: int, coverage: float) -> float:
    """ε как квантиль |наклона| на in-sample барах."""
    if coverage <= 0:
        return 0.0
    a = np.abs(slope[:oos_bar])
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0
    return float(np.quantile(a, coverage))


def neutral_share(state: np.ndarray, lo: int, hi: int) -> float:
    s = state[lo:hi]
    s = s[s != 2]
    return float((s == 0).mean()) if s.size else float("nan")
