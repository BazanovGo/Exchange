"""Семь семейств стратегий. Каждая возвращает сигнал в [-1, +1], несдвинутый:
задержку исполнения добавляет движок.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from exchange.strategies.base import ema, sma


# ---------------------------------------------------------------- 1. TSMOM
def tsmom(bars: pd.DataFrame, *, lookback: int) -> pd.Series:
    """Знак собственной доходности инструмента за lookback баров."""
    return np.sign(bars["close"].pct_change(lookback)).fillna(0.0)


# ------------------------------------------------------ 2. Dual moving average
def dual_ma(bars: pd.DataFrame, *, fast: int, slow: int, kind: str = "sma") -> pd.Series:
    if fast >= slow:
        raise ValueError("fast must be shorter than slow")
    f = (sma if kind == "sma" else ema)(bars["close"], fast)
    s = (sma if kind == "sma" else ema)(bars["close"], slow)
    return np.sign(f - s).fillna(0.0)


# --------------------------------------------------------- 3. Channel breakout
def breakout(bars: pd.DataFrame, *, entry: int, exit_: int) -> pd.Series:
    """Пробой N-дневного экстремума; выход по M-дневному в обратную сторону.

    Каналы считаются по барам СТРОГО до текущего (.shift(1)), иначе касание
    уровня было бы ретроспективным наблюдением, а не торгуемым событием.
    """
    if exit_ >= entry:
        raise ValueError("exit window must be shorter than entry window")
    upper = bars["high"].rolling(entry, min_periods=entry).max().shift(1)
    lower = bars["low"].rolling(entry, min_periods=entry).min().shift(1)
    exit_up = bars["high"].rolling(exit_, min_periods=exit_).max().shift(1)
    exit_dn = bars["low"].rolling(exit_, min_periods=exit_).min().shift(1)

    close = bars["close"].to_numpy(float)
    up, dn = upper.to_numpy(float), lower.to_numpy(float)
    xu, xd = exit_up.to_numpy(float), exit_dn.to_numpy(float)
    state = np.zeros(len(bars))
    position = 0.0
    for i in range(len(bars)):
        if np.isfinite(up[i]) and close[i] > up[i]:
            position = 1.0
        elif np.isfinite(dn[i]) and close[i] < dn[i]:
            position = -1.0
        elif position > 0 and np.isfinite(xd[i]) and close[i] < xd[i]:
            position = 0.0
        elif position < 0 and np.isfinite(xu[i]) and close[i] > xu[i]:
            position = 0.0
        state[i] = position
    return pd.Series(state, index=bars.index)


# ------------------------------------------------- 4. Cross-sectional momentum
def cross_sectional(panel: dict[str, pd.DataFrame], *, lookback: int) -> dict[str, pd.Series]:
    """Длинная позиция в лучшем, короткая в худшем — по доходности за lookback.

    С двумя инструментами это вырожденная версия кросс-секционного моментума:
    ранжирование по двум элементам даёт ровно одну длинную и одну короткую
    ногу, то есть спред NG против BR. Настоящая кросс-секция требует
    универсума из десятков рынков; это нужно держать в голове при чтении
    результата.
    """
    scores = pd.DataFrame(
        {root: bars["close"].pct_change(lookback) for root, bars in panel.items()}
    ).dropna(how="all")
    ranks = scores.rank(axis=1, method="average")
    n = ranks.notna().sum(axis=1)
    # Центрированный ранг: +1 лучшему, -1 худшему при двух активах.
    centred = (ranks.sub((n + 1) / 2.0, axis=0)).div((n - 1).clip(lower=1) / 2.0, axis=0)
    return {
        root: centred[root].reindex(panel[root].index).fillna(0.0).clip(-1.0, 1.0)
        for root in panel
    }


# -------------------------------------------------------------- 5. Carry
def carry(bars: pd.DataFrame, *, smooth: int = 1, threshold: float = 0.0) -> pd.Series:
    """Длинная позиция при бэквордации, короткая при контанго.

    Сигнал — знак сглаженного годового roll yield, наблюдаемого на роллах.
    Порог отсекает слабый наклон, где спред не покрывает издержки.
    """
    series = bars["carry"].rolling(smooth, min_periods=1).mean()
    out = pd.Series(0.0, index=bars.index)
    out[series > threshold] = 1.0
    out[series < -threshold] = -1.0
    return out


# ------------------------------------------- 6. MA по портфелю (две ноги)
def portfolio_ma(panel: dict[str, pd.DataFrame], *, fast: int, slow: int) -> dict[str, pd.Series]:
    """Тот же dual MA, но применённый к каждому рынку портфеля."""
    return {root: dual_ma(bars, fast=fast, slow=slow) for root, bars in panel.items()}


# ------------------------------------------------- 7. Momentum + reversal
def momentum_reversal(bars: pd.DataFrame, *, momentum: int, reversal: int,
                      weight: float = 0.5) -> pd.Series:
    """Среднесрочный моментум против долгосрочного разворота.

    Классическая комбинация: на горизонте 3-12 месяцев доходности
    продолжаются, на горизонте 3-5 лет — разворачиваются. Сигнал складывает
    знак первого и знак второго с обратным знаком.
    """
    if reversal <= momentum:
        raise ValueError("reversal horizon must exceed the momentum horizon")
    mom = np.sign(bars["close"].pct_change(momentum)).fillna(0.0)
    rev = -np.sign(bars["close"].pct_change(reversal)).fillna(0.0)
    combined = weight * mom + (1.0 - weight) * rev
    # Разворот сигнала только при согласии обеих компонент даёт -1/0/+1.
    return combined.clip(-1.0, 1.0)


SINGLE = {
    "1_tsmom": tsmom,
    "2_dual_ma": dual_ma,
    "3_breakout": breakout,
    "5_carry": carry,
    "7_mom_reversal": momentum_reversal,
}
PANEL = {
    "4_cross_sectional": cross_sectional,
    "6_portfolio_ma": portfolio_ma,
}
