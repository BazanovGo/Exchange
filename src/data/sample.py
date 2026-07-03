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
