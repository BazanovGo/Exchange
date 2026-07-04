"""Market-state features for pattern discovery.

Everything is computed strictly from past bars (rolling windows, no
centered/forward operations), so a feature value at bar *t* is known at
the close of *t* — the same moment strategy entries execute.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.indicators import adx as adx_ind
from src.indicators import atr as atr_ind
from src.indicators import bollinger_bands, ema, realized_volatility, sma

REGIME_ORDER = ["trend_up", "trend_down", "range_lowvol", "range_highvol"]


def efficiency_ratio(close: pd.Series, window: int = 20) -> pd.Series:
    """Kaufman efficiency ratio in [0, 1]: |net move| / path length."""
    direction = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window).sum()
    return (direction / path).clip(0.0, 1.0)


def higher_tf_trend(close: pd.Series, *, rule: str = "W-FRI", span: int = 10) -> pd.Series:
    """Weekly-EMA slope sign (+1/-1) forward-filled onto the daily index.

    Weekly values are stamped on the period's last bar and forward-filled,
    so a day only ever sees completed (or current-to-date) weekly data.
    """
    weekly = close.resample(rule).last()
    slope = ema(weekly, span).diff()
    daily = slope.reindex(close.index, method="ffill")
    return np.sign(daily).fillna(0.0)


def market_features(
    ohlcv: pd.DataFrame,
    *,
    vol_window: int = 21,
    atr_window: int = 14,
    adx_window: int = 14,
    volume_window: int = 20,
    ma_window: int = 200,
    ema_window: int = 50,
    slope_window: int = 10,
    bb_window: int = 20,
    er_window: int = 20,
    regime_adx_threshold: float = 25.0,
    regime_vol_window: int = 252,
) -> pd.DataFrame:
    """Bar-by-bar market-state features.

    Columns
    -------
    volatility        annualized realized volatility of log returns
    atr_pct           ATR as a fraction of the close
    adx               trend strength, 0..100
    rel_volume        volume vs its own moving average
    hour              bar hour (NaN-degenerate on daily data)
    day_of_week       0=Mon .. 6=Sun
    dist_ma           close / SMA(ma_window) - 1
    bb_width          Bollinger band width relative to the middle band
    ema_slope         EMA(ema_window) percent change per bar over slope_window
    efficiency_ratio  Kaufman ER: straightness of the recent path, 0..1
    htf_trend         weekly-EMA slope sign: +1 up, -1 down
    regime            categorical market regime (see ``label_regime``)
    """
    close, high, low, volume = ohlcv["Close"], ohlcv["High"], ohlcv["Low"], ohlcv["Volume"]

    f = pd.DataFrame(index=ohlcv.index)
    f["volatility"] = realized_volatility(close, vol_window)
    f["atr_pct"] = atr_ind(high, low, close, atr_window) / close
    f["adx"] = adx_ind(high, low, close, adx_window)[0]
    f["rel_volume"] = volume / volume.rolling(volume_window).mean()
    f["hour"] = ohlcv.index.hour
    f["day_of_week"] = ohlcv.index.dayofweek
    f["dist_ma"] = close / sma(close, ma_window) - 1.0
    middle, upper, lower = bollinger_bands(close, bb_window)
    f["bb_width"] = (upper - lower) / middle
    f["ema_slope"] = (ema(close, ema_window).pct_change(slope_window) / slope_window) * 100.0
    f["efficiency_ratio"] = efficiency_ratio(close, er_window)
    f["htf_trend"] = higher_tf_trend(close)
    f["regime"] = label_regime(
        f["adx"], f["dist_ma"], f["volatility"],
        adx_threshold=regime_adx_threshold, vol_window=regime_vol_window,
    )
    return f


def label_regime(
    adx: pd.Series,
    dist_ma: pd.Series,
    volatility: pd.Series,
    *,
    adx_threshold: float = 25.0,
    vol_window: int = 252,
) -> pd.Series:
    """Classify each bar into one of four regimes.

    * ``trend_up`` / ``trend_down`` — ADX confirms a trend; direction by
      the sign of the distance to the long MA;
    * ``range_lowvol`` / ``range_highvol`` — no trend; split by realized
      volatility vs its own rolling median (no full-sample lookahead).
    """
    vol_median = volatility.rolling(vol_window, min_periods=vol_window // 4).median()
    trending = adx >= adx_threshold

    regime = pd.Series("range_lowvol", index=adx.index, dtype="object")
    regime[trending & (dist_ma >= 0)] = "trend_up"
    regime[trending & (dist_ma < 0)] = "trend_down"
    regime[~trending & (volatility > vol_median)] = "range_highvol"
    regime[adx.isna() | dist_ma.isna() | volatility.isna()] = np.nan
    return pd.Series(pd.Categorical(regime, categories=REGIME_ORDER), index=adx.index)


def trade_feature_dataset(
    strategies: dict[str, dict],
    ohlcv: pd.DataFrame,
    features: pd.DataFrame | None = None,
    *,
    config=None,
) -> pd.DataFrame:
    """One row per closed trade of every strategy, joined with the market
    features captured at the entry bar.

    Target columns: ``ret_pct`` (trade return), ``win`` (bool),
    ``duration_days`` (calendar days held).
    """
    from src.backtesting.config import BacktestConfig
    from src.backtesting.engine import run_backtest
    from src.strategies.base import get_strategy

    features = features if features is not None else market_features(ohlcv)
    config = config or BacktestConfig()

    frames = []
    for name, params in strategies.items():
        result = run_backtest(get_strategy(name, **params), ohlcv, config=config)
        trades = result.trades()
        trades = trades[trades["Status"] == "Closed"]
        if trades.empty:
            continue
        rows = pd.DataFrame(
            {
                "strategy": name,
                "entry_date": pd.to_datetime(trades["Entry Timestamp"].to_numpy()),
                "ret_pct": trades["Return"].to_numpy() * 100.0,
                "duration_days": (
                    pd.to_datetime(trades["Exit Timestamp"].to_numpy())
                    - pd.to_datetime(trades["Entry Timestamp"].to_numpy())
                ).days,
            }
        )
        rows["win"] = rows["ret_pct"] > 0
        rows = rows.join(features.reindex(rows["entry_date"]).reset_index(drop=True))
        frames.append(rows)

    if not frames:
        raise ValueError("No closed trades produced by the given strategies")
    return pd.concat(frames, ignore_index=True)


def redundant_pairs(features: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Feature pairs whose absolute correlation exceeds ``threshold``."""
    numeric = features.select_dtypes(include=[np.number]).dropna()
    corr = numeric.corr()
    rows = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = corr.iloc[i, j]
            if abs(value) >= threshold:
                rows.append({"feature_a": cols[i], "feature_b": cols[j], "corr": value})
    return pd.DataFrame(rows, columns=["feature_a", "feature_b", "corr"]).sort_values(
        "corr", key=abs, ascending=False, ignore_index=True
    )
