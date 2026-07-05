"""Deterministic synthetic OHLCV generation.

Used for tests and as an offline fallback in notebooks so the entire
pipeline can be exercised without network access. The generator produces a
regime-switching geometric random walk with realistic intraday ranges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_ohlcv(
    symbol: str = "SAMPLE",
    *,
    start: str = "2018-01-01",
    periods: int = 2000,
    freq: str = "B",
    seed: int = 42,
    s0: float = 100.0,
    annual_vol: float = 0.30,
    annual_drift: float = 0.07,
) -> pd.DataFrame:
    """Generate a deterministic OHLCV frame with the canonical schema."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, periods=periods, freq=freq)

    dt = 1.0 / 252.0
    # Slow-moving volatility regime makes the series look less like pure GBM.
    regime = np.clip(np.cumsum(rng.normal(0, 0.05, periods)), -1.0, 1.0)
    vol = annual_vol * (1.0 + 0.5 * regime)
    log_ret = (annual_drift - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * rng.standard_normal(periods)
    close = s0 * np.exp(np.cumsum(log_ret))

    open_ = np.empty(periods)
    open_[0] = s0
    open_[1:] = close[:-1] * np.exp(rng.normal(0, 0.15 * vol[1:] * np.sqrt(dt)))
    intraday = np.abs(rng.normal(0, vol * np.sqrt(dt), periods))
    high = np.maximum(open_, close) * (1 + intraday)
    low = np.minimum(open_, close) * (1 - intraday)
    volume = (rng.lognormal(mean=13.0, sigma=0.4, size=periods)).round()

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )
    df.index.name = "Date"
    df.attrs["symbol"] = symbol
    return df


def _wrap_close_into_ohlcv(
    close: np.ndarray,
    index: pd.DatetimeIndex,
    rng: np.random.Generator,
    *,
    intraday_scale: float,
    symbol: str,
) -> pd.DataFrame:
    """Build a plausible OHLCV frame around a given close path."""
    n = len(close)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * np.exp(rng.normal(0, 0.15 * intraday_scale, n - 1))
    intraday = np.abs(rng.normal(0, intraday_scale, n))
    high = np.maximum(open_, close) * (1 + intraday)
    low = np.minimum(open_, close) * (1 - intraday)
    volume = rng.lognormal(mean=13.0, sigma=0.4, size=n).round()

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )
    df.index.name = "Date"
    df.attrs["symbol"] = symbol
    return df


def generate_ou_ohlcv(
    symbol: str = "OU",
    *,
    start: str = "2018-01-01",
    periods: int = 2000,
    freq: str = "B",
    seed: int = 7,
    s0: float = 100.0,
    theta: float = 0.05,
    sigma: float = 0.015,
    anchor_drift: float = 0.00015,
) -> pd.DataFrame:
    """Mean-reverting series: OU process on log price around a slow anchor.

    A positive-control instrument — mean-reversion strategies *should* find
    structure here (negative return autocorrelation by construction).
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, periods=periods, freq=freq)

    x = np.empty(periods)
    x[0] = np.log(s0)
    anchor = np.log(s0) + anchor_drift * np.arange(periods)
    eps = rng.standard_normal(periods)
    for t in range(1, periods):
        x[t] = x[t - 1] + theta * (anchor[t - 1] - x[t - 1]) + sigma * eps[t]
    close = np.exp(x)
    return _wrap_close_into_ohlcv(close, index, rng, intraday_scale=sigma * 0.8, symbol=symbol)


def generate_trending_ohlcv(
    symbol: str = "TREND",
    *,
    start: str = "2018-01-01",
    periods: int = 2000,
    freq: str = "B",
    seed: int = 11,
    s0: float = 100.0,
    persistence: float = 0.985,
    mu_bull: float = 0.0012,
    mu_bear: float = -0.0008,
    sigma: float = 0.012,
) -> pd.DataFrame:
    """Momentum-friendly series: two-state Markov drift with high persistence.

    A positive-control instrument — trend-following strategies *should*
    find structure here (long drift regimes by construction).
    """
    rng = np.random.default_rng(seed)
    index = pd.date_range(start=start, periods=periods, freq=freq)

    bull = np.empty(periods, dtype=bool)
    bull[0] = True
    switches = rng.random(periods) > persistence
    for t in range(1, periods):
        bull[t] = ~bull[t - 1] if switches[t] else bull[t - 1]

    mu = np.where(bull, mu_bull, mu_bear)
    log_ret = mu + sigma * rng.standard_normal(periods)
    close = s0 * np.exp(np.cumsum(log_ret))
    return _wrap_close_into_ohlcv(close, index, rng, intraday_scale=sigma * 0.8, symbol=symbol)
