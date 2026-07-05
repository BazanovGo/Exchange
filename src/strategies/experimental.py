"""Experimental strategies produced by the autonomous research loop.

Each class is one formalized hypothesis; its study lives in
``research/<id>_<name>/`` and its fate (promising / viable / archived) is
tracked in ``research/catalog.csv``. Archived strategies stay here for
reproducibility — the knowledge base forbids re-testing them.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from src.indicators import (
    adx,
    atr,
    bollinger_bands,
    donchian,
    ema,
    roc,
    rolling_vwap,
    rsi,
    sma,
)
from src.strategies.base import BaseStrategy, SignalArray, register_strategy
from src.strategies.signal_utils import rising_edge


@register_strategy
class SqueezeBreakoutStrategy(BaseStrategy):
    """H001: Bollinger Squeeze + ATR expansion breakout."""

    name: ClassVar[str] = "squeeze_breakout"
    category: ClassVar[str] = "momentum"
    description: ClassVar[str] = (
        "Volatility-cycle breakout: wait until Bollinger width falls into its lowest "
        "quantile over a lookback (the squeeze), then buy a close breaking the upper band "
        "while ATR is already expanding; exit on a close back through the middle band."
    )
    default_params: ClassVar[dict[str, Any]] = {
        "bb_window": 20,
        "squeeze_lookback": 120,
        "squeeze_quantile": 0.25,
        "atr_window": 14,
    }
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "bb_window": [15, 20, 30],
        "squeeze_lookback": [80, 120, 180],
        "squeeze_quantile": [0.15, 0.25, 0.35],
        "atr_window": [10, 14],
    }

    def validate_params(self) -> None:
        if not 0 < self.params["squeeze_quantile"] < 1:
            raise ValueError("squeeze_quantile must be in (0, 1)")
        if self.params["squeeze_lookback"] <= self.params["bb_window"]:
            raise ValueError("squeeze_lookback must exceed bb_window")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        p = self.params
        close = ohlcv["Close"]
        middle, upper, _ = bollinger_bands(close, p["bb_window"])
        width = (upper - middle) * 2 / middle

        squeeze_level = width.rolling(p["squeeze_lookback"]).quantile(p["squeeze_quantile"])
        squeezed_recently = (width <= squeeze_level).rolling(5).max().astype(bool)

        atr_ = atr(ohlcv["High"], ohlcv["Low"], close, p["atr_window"])
        atr_expanding = atr_ > atr_.shift(3)

        entries = close.vbt.crossed_above(upper) & squeezed_recently & atr_expanding
        exits = close.vbt.crossed_below(middle)
        return {"entries": entries, "exits": exits}


@register_strategy
class VWAPADXTrendStrategy(BaseStrategy):
    """H002: VWAP trend claim confirmed by ADX."""

    name: ClassVar[str] = "vwap_adx_trend"
    category: ClassVar[str] = "trend"
    description: ClassVar[str] = (
        "Volume-anchored trend following: buy when the close crosses above the rolling VWAP "
        "with ADX above a threshold and rising (the crowd's average position is underwater "
        "and directional pressure is building); exit when price crosses back below VWAP."
    )
    default_params: ClassVar[dict[str, Any]] = {
        "vwap_window": 30,
        "adx_window": 14,
        "adx_threshold": 20.0,
    }
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "vwap_window": [15, 30, 50, 80],
        "adx_window": [10, 14, 20],
        "adx_threshold": [15.0, 20.0, 25.0],
    }

    def validate_params(self) -> None:
        if self.params["adx_threshold"] <= 0:
            raise ValueError("adx_threshold must be positive")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        p = self.params
        close = ohlcv["Close"]
        vwap = rolling_vwap(ohlcv["High"], ohlcv["Low"], close, ohlcv["Volume"], p["vwap_window"])
        adx_, _, _ = adx(ohlcv["High"], ohlcv["Low"], close, p["adx_window"])
        adx_ok = (adx_ > p["adx_threshold"]) & (adx_ > adx_.shift(3))
        return {
            "entries": close.vbt.crossed_above(vwap) & adx_ok,
            "exits": close.vbt.crossed_below(vwap),
        }


@register_strategy
class DonchianRSIPullbackStrategy(BaseStrategy):
    """H003: buy RSI pullbacks inside a Donchian uptrend structure."""

    name: ClassVar[str] = "donchian_rsi_pullback"
    category: ClassVar[str] = "hybrid"
    description: ClassVar[str] = (
        "Market-structure pullback: while price holds in the upper part of its Donchian "
        "channel (an uptrend by construction), buy an RSI dip below the pullback level; "
        "exit when RSI recovers to overbought or price loses the channel midline."
    )
    default_params: ClassVar[dict[str, Any]] = {
        "don_window": 50,
        "structure_position": 0.5,
        "rsi_window": 14,
        "pullback": 40.0,
        "exit_level": 65.0,
    }
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "don_window": [30, 50, 80],
        "structure_position": [0.5, 0.6],
        "pullback": [35.0, 40.0, 45.0],
        "exit_level": [60.0, 65.0, 75.0],
    }

    def validate_params(self) -> None:
        p = self.params
        if not 0 < p["structure_position"] < 1:
            raise ValueError("structure_position must be in (0, 1)")
        if not 0 < p["pullback"] < p["exit_level"] < 100:
            raise ValueError("Expected 0 < pullback < exit_level < 100")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        p = self.params
        close = ohlcv["Close"]
        upper, middle, lower = donchian(ohlcv["High"], ohlcv["Low"], p["don_window"])
        # Position of price inside the channel: 1 = at highs, 0 = at lows.
        channel_pos = (close - lower) / (upper - lower)
        in_structure = channel_pos >= p["structure_position"]

        rsi_ = rsi(close, p["rsi_window"])
        entries = rsi_.vbt.crossed_below(p["pullback"]) & in_structure
        exits = rsi_.vbt.crossed_above(p["exit_level"]) | close.vbt.crossed_below(middle)
        return {"entries": entries, "exits": exits}


@register_strategy
class MomentumVolumeStrategy(BaseStrategy):
    """H004: ROC momentum ignited by a volume expansion."""

    name: ClassVar[str] = "momentum_volume"
    category: ClassVar[str] = "momentum"
    description: ClassVar[str] = (
        "Participation-confirmed momentum: buy when N-bar ROC turns positive while volume "
        "runs above its own average by a multiple (new money, not drift); exit when ROC "
        "falls back below zero."
    )
    default_params: ClassVar[dict[str, Any]] = {
        "roc_window": 20,
        "vol_window": 20,
        "vol_multiplier": 1.3,
    }
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "roc_window": [10, 20, 40],
        "vol_window": [10, 20, 40],
        "vol_multiplier": [1.1, 1.3, 1.6, 2.0],
    }

    def validate_params(self) -> None:
        if self.params["vol_multiplier"] <= 0:
            raise ValueError("vol_multiplier must be positive")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        p = self.params
        close, volume = ohlcv["Close"], ohlcv["Volume"]
        roc_ = roc(close, p["roc_window"])
        vol_hot = volume.rolling(3).mean() > volume.rolling(p["vol_window"]).mean() * p["vol_multiplier"]
        return {
            "entries": roc_.vbt.crossed_above(0.0) & vol_hot,
            "exits": roc_.vbt.crossed_below(0.0),
        }


@register_strategy
class DonchianRSIFixedStrategy(BaseStrategy):
    """H005: H003 with degrees of freedom stripped to the plateau center."""

    name: ClassVar[str] = "donchian_rsi_fixed"
    category: ClassVar[str] = "hybrid"
    description: ClassVar[str] = (
        "Refined H003: the same Donchian-structure RSI pullback but with the fragile "
        "parameters frozen (structure position 0.5, RSI exit 65) — only the channel window "
        "and pullback depth remain free. Tests whether the idea survives with minimal "
        "degrees of freedom."
    )
    default_params: ClassVar[dict[str, Any]] = {
        "don_window": 50,
        "pullback": 40.0,
    }
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "don_window": [30, 40, 50, 65, 80],
        "pullback": [35.0, 40.0, 45.0],
    }

    def validate_params(self) -> None:
        if not 0 < self.params["pullback"] < 65:
            raise ValueError("pullback must be in (0, 65)")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        p = self.params
        close = ohlcv["Close"]
        upper, middle, lower = donchian(ohlcv["High"], ohlcv["Low"], p["don_window"])
        channel_pos = (close - lower) / (upper - lower)
        rsi_ = rsi(close, 14)
        entries = rsi_.vbt.crossed_below(p["pullback"]) & (channel_pos >= 0.5)
        exits = rsi_.vbt.crossed_above(65.0) | close.vbt.crossed_below(middle)
        return {"entries": entries, "exits": exits}


@register_strategy
class EfficientPullbackStrategy(BaseStrategy):
    """H006: buy touches of the fast EMA inside an efficient trend."""

    name: ClassVar[str] = "efficient_pullback"
    category: ClassVar[str] = "hybrid"
    description: ClassVar[str] = (
        "Trend-quality pullback: when the path is efficient (Kaufman ER above a floor) and "
        "price is above the slow EMA, a dip touching the fast EMA is bought; exit when the "
        "close breaks the slow EMA or the trend loses efficiency."
    )
    default_params: ClassVar[dict[str, Any]] = {
        "fast_window": 20,
        "slow_window": 60,
        "er_window": 20,
        "er_floor": 0.3,
    }
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "fast_window": [10, 20, 30],
        "slow_window": [50, 60, 100],
        "er_window": [15, 20, 30],
        "er_floor": [0.2, 0.3, 0.4],
    }

    def validate_params(self) -> None:
        if self.params["fast_window"] >= self.params["slow_window"]:
            raise ValueError("fast_window must be < slow_window")
        if not 0 < self.params["er_floor"] < 1:
            raise ValueError("er_floor must be in (0, 1)")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        from src.analytics.features import efficiency_ratio

        p = self.params
        close, low = ohlcv["Close"], ohlcv["Low"]
        fast = ema(close, p["fast_window"])
        slow = ema(close, p["slow_window"])
        er = efficiency_ratio(close, p["er_window"])

        healthy_trend = (close > slow) & (er >= p["er_floor"])
        touched_fast = (low <= fast) & (close >= fast * 0.99)
        entries = rising_edge(healthy_trend & touched_fast)
        exits = close.vbt.crossed_below(slow) | er.vbt.crossed_below(p["er_floor"] * 0.5)
        return {"entries": entries, "exits": exits}


@register_strategy
class EngulfingTrendStrategy(BaseStrategy):
    """H007: bullish engulfing candle inside a long-term uptrend."""

    name: ClassVar[str] = "engulfing_trend"
    category: ClassVar[str] = "hybrid"
    description: ClassVar[str] = (
        "Candle-pattern entry: a bullish engulfing bar (body swallows the prior red body) "
        "appearing above the long SMA signals dip-buyers taking control; exit when price "
        "closes below the fast SMA."
    )
    default_params: ClassVar[dict[str, Any]] = {
        "trend_window": 150,
        "exit_window": 20,
        "min_body_atr": 0.5,
    }
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "trend_window": [100, 150, 200],
        "exit_window": [10, 20, 40],
        "min_body_atr": [0.3, 0.5, 0.8],
    }

    def validate_params(self) -> None:
        if self.params["min_body_atr"] <= 0:
            raise ValueError("min_body_atr must be positive")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        p = self.params
        o, c = ohlcv["Open"], ohlcv["Close"]
        atr_ = atr(ohlcv["High"], ohlcv["Low"], c, 14)

        prev_red = c.shift(1) < o.shift(1)
        engulfs = (c > o) & (c > o.shift(1)) & (o < c.shift(1))
        big_body = (c - o) >= p["min_body_atr"] * atr_
        uptrend = c > sma(c, p["trend_window"])

        entries = prev_red & engulfs & big_body & uptrend
        exits = c.vbt.crossed_below(sma(c, p["exit_window"]))
        return {"entries": entries, "exits": exits}


@register_strategy
class RegimeSwitcherStrategy(BaseStrategy):
    """H008: trend engine in trending regimes, mean-reversion in volatile ranges."""

    name: ClassVar[str] = "regime_switcher"
    category: ClassVar[str] = "hybrid"
    description: ClassVar[str] = (
        "Regime-routed composite (stage-04 map): when ADX confirms a trend, trade the EMA "
        "crossover engine; when ADX is quiet but volatility is elevated, fade z-score dips. "
        "Positions close when their engine's exit fires or the regime that opened them ends. "
        "The quiet low-volatility range is never traded."
    )
    default_params: ClassVar[dict[str, Any]] = {
        "adx_threshold": 25.0,
        "fast_window": 20,
        "slow_window": 60,
        "z_window": 20,
        "z_entry": 1.5,
    }
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "adx_threshold": [20.0, 25.0, 30.0],
        "fast_window": [10, 20],
        "slow_window": [50, 80],
        "z_window": [15, 20, 30],
        "z_entry": [1.2, 1.5, 2.0],
    }

    def validate_params(self) -> None:
        if self.params["fast_window"] >= self.params["slow_window"]:
            raise ValueError("fast_window must be < slow_window")
        if self.params["z_entry"] <= 0:
            raise ValueError("z_entry must be positive")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        from src.analytics.features import label_regime
        from src.indicators import realized_volatility, zscore

        p = self.params
        close = ohlcv["Close"]
        adx_, _, _ = adx(ohlcv["High"], ohlcv["Low"], close, 14)
        dist_ma = close / sma(close, 200) - 1.0
        vol = realized_volatility(close, 21)
        regime = label_regime(adx_, dist_ma, vol, adx_threshold=p["adx_threshold"])

        trending_up = regime == "trend_up"
        vol_range = regime == "range_highvol"

        fast, slow = ema(close, p["fast_window"]), ema(close, p["slow_window"])
        trend_entry = fast.vbt.crossed_above(slow) & trending_up
        trend_exit = fast.vbt.crossed_below(slow)

        z = zscore(close, p["z_window"])
        mr_entry = z.vbt.crossed_below(-p["z_entry"]) & vol_range
        mr_exit = z.vbt.crossed_above(0.0)

        # Exit when the engine of the current regime fires, or when the
        # market drops into a regime we do not trade (quiet range / downtrend).
        neutral = ~trending_up & ~vol_range
        exits = (trending_up & trend_exit) | (vol_range & mr_exit) | neutral
        return {"entries": trend_entry | mr_entry, "exits": exits}


@register_strategy
class MomentumVolumeRankStrategy(BaseStrategy):
    """H010: H004 reworked — volume percentile rank instead of a multiplier."""

    name: ClassVar[str] = "momentum_volrank"
    category: ClassVar[str] = "momentum"
    description: ClassVar[str] = (
        "Refined H004 (its fragile axis was the volume multiplier): momentum turning positive "
        "is confirmed by the percentile rank of volume over a long lookback — a scale-free "
        "condition. Exit when momentum drops back through zero."
    )
    default_params: ClassVar[dict[str, Any]] = {
        "roc_window": 20,
        "rank_lookback": 120,
        "rank_floor": 0.6,
    }
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "roc_window": [10, 20, 40],
        "rank_lookback": [80, 120, 180],
        "rank_floor": [0.5, 0.6, 0.7, 0.8],
    }

    def validate_params(self) -> None:
        if not 0 < self.params["rank_floor"] < 1:
            raise ValueError("rank_floor must be in (0, 1)")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        p = self.params
        close, volume = ohlcv["Close"], ohlcv["Volume"]
        roc_ = roc(close, p["roc_window"])
        vol_rank = volume.rolling(p["rank_lookback"]).rank(pct=True)
        return {
            "entries": roc_.vbt.crossed_above(0.0) & (vol_rank >= p["rank_floor"]),
            "exits": roc_.vbt.crossed_below(0.0),
        }


@register_strategy
class CLVDipStrategy(BaseStrategy):
    """H011: capitulation closes (low CLV) bought inside an uptrend."""

    name: ClassVar[str] = "clv_dip"
    category: ClassVar[str] = "mean_reversion"
    description: ClassVar[str] = (
        "Candle-position reversion: a close pinned to the bottom of its daily range "
        "(close-location-value below a floor) while the long trend is up marks intraday "
        "capitulation; buy it and exit when price reclaims the short moving average."
    )
    default_params: ClassVar[dict[str, Any]] = {
        "clv_floor": 0.2,
        "trend_window": 150,
        "exit_window": 10,
    }
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "clv_floor": [0.1, 0.2, 0.3],
        "trend_window": [100, 150, 200],
        "exit_window": [5, 10, 20],
    }

    def validate_params(self) -> None:
        if not 0 < self.params["clv_floor"] < 1:
            raise ValueError("clv_floor must be in (0, 1)")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        p = self.params
        h, l, c = ohlcv["High"], ohlcv["Low"], ohlcv["Close"]  # noqa: E741
        clv = (c - l) / (h - l).replace(0.0, pd.NA)
        capitulation = (clv <= p["clv_floor"]).fillna(False).astype(bool)
        uptrend = c > sma(c, p["trend_window"])
        return {
            "entries": rising_edge(capitulation & uptrend),
            "exits": c.vbt.crossed_above(sma(c, p["exit_window"])),
        }
