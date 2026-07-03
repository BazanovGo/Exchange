"""Trend indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import vectorbt as vbt
from numba import njit


def ma(close: pd.Series | pd.DataFrame, window: int, *, ewm: bool = False) -> pd.Series | pd.DataFrame:
    """Moving average (simple or exponential) via ``vbt.MA``."""
    return vbt.MA.run(close, window=window, ewm=ewm).ma


def sma(close: pd.Series | pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    """Simple moving average."""
    return ma(close, window, ewm=False)


def ema(close: pd.Series | pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    """Exponential moving average."""
    return ma(close, window, ewm=True)


def donchian(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
    *,
    exclude_current: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Donchian channel (upper, middle, lower).

    ``exclude_current=True`` uses the previous ``window`` bars, so a close
    above the upper band is a genuine breakout (the current bar cannot be
    its own extreme).
    """
    h = high.shift(1) if exclude_current else high
    l = low.shift(1) if exclude_current else low  # noqa: E741
    upper = h.rolling(window).max()
    lower = l.rolling(window).min()
    middle = (upper + lower) / 2.0
    return upper, middle, lower


def _wilder_smooth(series: pd.Series, window: int) -> pd.Series:
    """Wilder's smoothing = EMA with alpha 1/window."""
    return series.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Average Directional Index: returns ``(adx, plus_di, minus_di)``."""
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)

    atr_ = _wilder_smooth(tr, window)
    plus_di = 100.0 * _wilder_smooth(plus_dm, window) / atr_
    minus_di = 100.0 * _wilder_smooth(minus_dm, window) / atr_
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_ = _wilder_smooth(dx, window)
    return adx_, plus_di, minus_di


@njit(cache=True)
def _supertrend_nb(high: np.ndarray, low: np.ndarray, close: np.ndarray, atr_: np.ndarray, mult: float):
    n = len(close)
    line = np.full(n, np.nan)
    direction = np.zeros(n)  # +1 uptrend, -1 downtrend
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(atr_[i]):
            continue
        mid = (high[i] + low[i]) / 2.0
        ub = mid + mult * atr_[i]
        lb = mid - mult * atr_[i]

        if i == 0 or np.isnan(line[i - 1]):
            upper[i], lower[i] = ub, lb
            direction[i] = 1.0 if close[i] >= mid else -1.0
            line[i] = lb if direction[i] > 0 else ub
            continue

        # Band ratcheting: bands only tighten while the trend persists.
        upper[i] = ub if (ub < upper[i - 1]) or (close[i - 1] > upper[i - 1]) else upper[i - 1]
        lower[i] = lb if (lb > lower[i - 1]) or (close[i - 1] < lower[i - 1]) else lower[i - 1]

        if direction[i - 1] > 0:
            direction[i] = -1.0 if close[i] < lower[i] else 1.0
        else:
            direction[i] = 1.0 if close[i] > upper[i] else -1.0
        line[i] = lower[i] if direction[i] > 0 else upper[i]

    return line, direction


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    """Supertrend: returns ``(line, direction)`` with direction +1/-1."""
    atr_ = vbt.ATR.run(high, low, close, window=window, ewm=True).atr
    line, direction = _supertrend_nb(
        high.to_numpy(float), low.to_numpy(float), close.to_numpy(float), atr_.to_numpy(float), float(multiplier)
    )
    return pd.Series(line, index=close.index), pd.Series(direction, index=close.index)


@njit(cache=True)
def _psar_nb(high: np.ndarray, low: np.ndarray, af0: float, af_step: float, af_max: float):
    n = len(high)
    sar = np.full(n, np.nan)
    trend = np.zeros(n)  # +1 long, -1 short
    if n < 2:
        return sar, trend

    up = high[1] > high[0]
    trend[1] = 1.0 if up else -1.0
    sar[1] = low[0] if up else high[0]
    ep = high[1] if up else low[1]
    af = af0

    for i in range(2, n):
        prev = sar[i - 1]
        cur = prev + af * (ep - prev)
        if trend[i - 1] > 0:
            cur = min(cur, low[i - 1], low[i - 2])
            if low[i] < cur:  # reversal to short
                trend[i] = -1.0
                sar[i] = ep
                ep = low[i]
                af = af0
            else:
                trend[i] = 1.0
                sar[i] = cur
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            cur = max(cur, high[i - 1], high[i - 2])
            if high[i] > cur:  # reversal to long
                trend[i] = 1.0
                sar[i] = ep
                ep = high[i]
                af = af0
            else:
                trend[i] = -1.0
                sar[i] = cur
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)
    return sar, trend


def parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    af0: float = 0.02,
    af_step: float = 0.02,
    af_max: float = 0.2,
) -> tuple[pd.Series, pd.Series]:
    """Parabolic SAR: returns ``(sar, trend)`` with trend +1/-1."""
    sar, trend = _psar_nb(high.to_numpy(float), low.to_numpy(float), float(af0), float(af_step), float(af_max))
    return pd.Series(sar, index=high.index), pd.Series(trend, index=high.index)


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan_window: int = 9,
    kijun_window: int = 26,
    senkou_b_window: int = 52,
) -> dict[str, pd.Series]:
    """Ichimoku components (senkou spans shifted forward by ``kijun_window``)."""

    def midline(w: int) -> pd.Series:
        return (high.rolling(w).max() + low.rolling(w).min()) / 2.0

    tenkan = midline(tenkan_window)
    kijun = midline(kijun_window)
    senkou_a = ((tenkan + kijun) / 2.0).shift(kijun_window)
    senkou_b = midline(senkou_b_window).shift(kijun_window)
    chikou = close.shift(-kijun_window)
    return {"tenkan": tenkan, "kijun": kijun, "senkou_a": senkou_a, "senkou_b": senkou_b, "chikou": chikou}
