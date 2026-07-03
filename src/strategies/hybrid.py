"""Hybrid strategies: a signal engine combined with a regime/confirmation filter."""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from src.indicators import adx, atr, donchian, ema, macd, rolling_vwap, rsi, sma
from src.strategies.base import BaseStrategy, SignalArray, register_strategy

_HYBRID = "hybrid"


@register_strategy
class EMARSIStrategy(BaseStrategy):
    name: ClassVar[str] = "ema_rsi"
    category: ClassVar[str] = _HYBRID
    description: ClassVar[str] = (
        "Buy-the-dip in an uptrend: EMA fast>slow defines the regime, an RSI dip below the "
        "pullback level triggers entry; exit on RSI strength or regime loss."
    )
    default_params: ClassVar[dict[str, Any]] = {
        "fast_window": 20,
        "slow_window": 100,
        "rsi_window": 14,
        "pullback": 40.0,
        "exit_level": 70.0,
    }
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "fast_window": [10, 20, 50],
        "slow_window": [100, 150, 200],
        "pullback": [30.0, 40.0, 50.0],
        "exit_level": [60.0, 70.0, 80.0],
    }

    def validate_params(self) -> None:
        p = self.params
        if p["fast_window"] >= p["slow_window"]:
            raise ValueError("fast_window must be < slow_window")
        if not (0 < p["pullback"] < p["exit_level"] < 100):
            raise ValueError("Expected 0 < pullback < exit_level < 100")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        close = ohlcv["Close"]
        uptrend = ema(close, self.params["fast_window"]) > ema(close, self.params["slow_window"])
        rsi_ = rsi(close, self.params["rsi_window"])
        entries = rsi_.vbt.crossed_below(self.params["pullback"]) & uptrend
        exits = rsi_.vbt.crossed_above(self.params["exit_level"]) | (~uptrend & uptrend.shift(1, fill_value=False))
        return {"entries": entries, "exits": exits}


@register_strategy
class EMAADXStrategy(BaseStrategy):
    name: ClassVar[str] = "ema_adx"
    category: ClassVar[str] = _HYBRID
    description: ClassVar[str] = (
        "EMA crossover gated by trend strength: the golden cross only counts when ADX confirms "
        "a trending market; exit on the death cross."
    )
    default_params: ClassVar[dict[str, Any]] = {
        "fast_window": 20,
        "slow_window": 50,
        "adx_window": 14,
        "adx_threshold": 20.0,
    }
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "fast_window": [10, 20, 30],
        "slow_window": [50, 100, 150],
        "adx_window": [10, 14, 20],
        "adx_threshold": [15.0, 20.0, 25.0],
    }

    def validate_params(self) -> None:
        if self.params["fast_window"] >= self.params["slow_window"]:
            raise ValueError("fast_window must be < slow_window")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        close = ohlcv["Close"]
        fast = ema(close, self.params["fast_window"])
        slow = ema(close, self.params["slow_window"])
        adx_, _, _ = adx(ohlcv["High"], ohlcv["Low"], close, self.params["adx_window"])
        entries = fast.vbt.crossed_above(slow) & (adx_ > self.params["adx_threshold"])
        exits = fast.vbt.crossed_below(slow)
        return {"entries": entries, "exits": exits}


@register_strategy
class DonchianATRStrategy(BaseStrategy):
    name: ClassVar[str] = "donchian_atr"
    category: ClassVar[str] = _HYBRID
    description: ClassVar[str] = (
        "Channel breakout with a volatility trailing exit: enter on an N-bar Donchian breakout, "
        "exit when price falls k*ATR below the highest close since entry (chandelier stop)."
    )
    default_params: ClassVar[dict[str, Any]] = {
        "entry_window": 40,
        "atr_window": 14,
        "atr_multiplier": 3.0,
        "trail_window": 22,
    }
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "entry_window": [20, 40, 55, 80],
        "atr_multiplier": [2.0, 3.0, 4.0],
        "trail_window": [10, 22, 40],
    }

    def validate_params(self) -> None:
        if self.params["atr_multiplier"] <= 0:
            raise ValueError("atr_multiplier must be positive")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        close = ohlcv["Close"]
        upper, _, _ = donchian(ohlcv["High"], ohlcv["Low"], self.params["entry_window"])
        atr_ = atr(ohlcv["High"], ohlcv["Low"], close, self.params["atr_window"])
        chandelier = ohlcv["High"].rolling(self.params["trail_window"]).max() - self.params["atr_multiplier"] * atr_
        return {
            "entries": close.vbt.crossed_above(upper),
            "exits": close.vbt.crossed_below(chandelier),
        }


@register_strategy
class VWAPVolumeStrategy(BaseStrategy):
    name: ClassVar[str] = "vwap_volume"
    category: ClassVar[str] = _HYBRID
    description: ClassVar[str] = (
        "Volume-confirmed VWAP reversion: buy a deep deviation below rolling VWAP only when volume "
        "spikes above its average (capitulation), exit when price reverts to VWAP."
    )
    default_params: ClassVar[dict[str, Any]] = {
        "window": 20,
        "threshold": 0.03,
        "vol_window": 20,
        "vol_multiplier": 1.5,
    }
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "window": [10, 20, 40],
        "threshold": [0.02, 0.03, 0.05],
        "vol_multiplier": [1.2, 1.5, 2.0],
    }

    def validate_params(self) -> None:
        if self.params["threshold"] <= 0 or self.params["vol_multiplier"] <= 0:
            raise ValueError("threshold and vol_multiplier must be positive")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        close, volume = ohlcv["Close"], ohlcv["Volume"]
        vwap = rolling_vwap(ohlcv["High"], ohlcv["Low"], close, volume, self.params["window"])
        deviation = close / vwap - 1.0
        volume_spike = volume > volume.rolling(self.params["vol_window"]).mean() * self.params["vol_multiplier"]
        return {
            "entries": deviation.vbt.crossed_below(-self.params["threshold"]) & volume_spike,
            "exits": deviation.vbt.crossed_above(0.0),
        }


@register_strategy
class MACDTrendFilterStrategy(BaseStrategy):
    name: ClassVar[str] = "macd_trend_filter"
    category: ClassVar[str] = _HYBRID
    description: ClassVar[str] = (
        "MACD crossover traded only in the direction of the long-term trend "
        "(price above its long SMA); exit on the bearish MACD cross."
    )
    default_params: ClassVar[dict[str, Any]] = {
        "fast_window": 12,
        "slow_window": 26,
        "signal_window": 9,
        "trend_window": 200,
    }
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "fast_window": [8, 12, 16],
        "slow_window": [21, 26, 40],
        "trend_window": [100, 150, 200],
    }

    def validate_params(self) -> None:
        if self.params["fast_window"] >= self.params["slow_window"]:
            raise ValueError("fast_window must be < slow_window")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        close = ohlcv["Close"]
        macd_line, signal_line, _ = macd(
            close,
            fast_window=self.params["fast_window"],
            slow_window=self.params["slow_window"],
            signal_window=self.params["signal_window"],
        )
        uptrend = close > sma(close, self.params["trend_window"])
        return {
            "entries": macd_line.vbt.crossed_above(signal_line) & uptrend,
            "exits": macd_line.vbt.crossed_below(signal_line),
        }
