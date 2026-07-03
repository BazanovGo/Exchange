"""Momentum / breakout strategies."""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from src.indicators import atr, keltner_channels, macd, momentum, realized_volatility, roc, sma
from src.strategies.base import BaseStrategy, SignalArray, register_strategy
from src.strategies.signal_utils import rising_edge

_MOMENTUM = "momentum"


@register_strategy
class ROCMomentumStrategy(BaseStrategy):
    name: ClassVar[str] = "roc_momentum"
    category: ClassVar[str] = _MOMENTUM
    description: ClassVar[str] = (
        "Rate-of-change momentum: enter when N-bar ROC pushes above a positive threshold, "
        "exit when it falls back through zero."
    )
    default_params: ClassVar[dict[str, Any]] = {"window": 20, "threshold": 2.0}
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "window": [10, 20, 40, 60],
        "threshold": [0.0, 2.0, 5.0, 8.0],
    }

    def validate_params(self) -> None:
        if self.params["threshold"] < 0:
            raise ValueError("threshold must be >= 0")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        roc_ = roc(ohlcv["Close"], self.params["window"])
        return {
            "entries": roc_.vbt.crossed_above(self.params["threshold"]),
            "exits": roc_.vbt.crossed_below(0.0),
        }


@register_strategy
class MomentumStrategy(BaseStrategy):
    name: ClassVar[str] = "momentum"
    category: ClassVar[str] = _MOMENTUM
    description: ClassVar[str] = (
        "Raw momentum with a signal line: enter when N-bar price change crosses above its own "
        "smoothing, exit on the cross back below."
    )
    default_params: ClassVar[dict[str, Any]] = {"window": 20, "signal_window": 10}
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "window": [10, 20, 40, 60],
        "signal_window": [5, 10, 20],
    }

    def validate_params(self) -> None:
        if self.params["signal_window"] < 1:
            raise ValueError("signal_window must be >= 1")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        mom = momentum(ohlcv["Close"], self.params["window"])
        signal_line = mom.rolling(self.params["signal_window"]).mean()
        return {
            "entries": mom.vbt.crossed_above(signal_line) & (mom > 0),
            "exits": mom.vbt.crossed_below(signal_line),
        }


@register_strategy
class MACDMomentumStrategy(BaseStrategy):
    name: ClassVar[str] = "macd_momentum"
    category: ClassVar[str] = _MOMENTUM
    description: ClassVar[str] = (
        "MACD line/signal crossover: enter on the bullish cross, exit on the bearish cross."
    )
    default_params: ClassVar[dict[str, Any]] = {"fast_window": 12, "slow_window": 26, "signal_window": 9}
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "fast_window": [8, 12, 16],
        "slow_window": [21, 26, 40],
        "signal_window": [6, 9, 12],
    }

    def validate_params(self) -> None:
        if self.params["fast_window"] >= self.params["slow_window"]:
            raise ValueError("fast_window must be < slow_window")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        macd_line, signal_line, _ = macd(
            ohlcv["Close"],
            fast_window=self.params["fast_window"],
            slow_window=self.params["slow_window"],
            signal_window=self.params["signal_window"],
        )
        return {
            "entries": macd_line.vbt.crossed_above(signal_line),
            "exits": macd_line.vbt.crossed_below(signal_line),
        }


@register_strategy
class ATRBreakoutStrategy(BaseStrategy):
    name: ClassVar[str] = "atr_breakout"
    category: ClassVar[str] = _MOMENTUM
    description: ClassVar[str] = (
        "Volatility breakout: enter when the close jumps more than k*ATR above the previous close, "
        "exit when price loses its N-bar moving average."
    )
    default_params: ClassVar[dict[str, Any]] = {"atr_window": 14, "multiplier": 1.5, "exit_window": 20}
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "atr_window": [10, 14, 21],
        "multiplier": [1.0, 1.5, 2.0, 3.0],
        "exit_window": [10, 20, 40],
    }

    def validate_params(self) -> None:
        if self.params["multiplier"] <= 0:
            raise ValueError("multiplier must be positive")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        close = ohlcv["Close"]
        atr_ = atr(ohlcv["High"], ohlcv["Low"], close, self.params["atr_window"])
        breakout = close > close.shift(1) + self.params["multiplier"] * atr_.shift(1)
        return {
            "entries": rising_edge(breakout),
            "exits": close.vbt.crossed_below(sma(close, self.params["exit_window"])),
        }


@register_strategy
class KeltnerBreakoutStrategy(BaseStrategy):
    name: ClassVar[str] = "keltner_breakout"
    category: ClassVar[str] = _MOMENTUM
    description: ClassVar[str] = (
        "Keltner channel breakout: enter on a close above the upper channel, "
        "exit when price falls back through the middle EMA."
    )
    default_params: ClassVar[dict[str, Any]] = {"window": 20, "atr_window": 10, "multiplier": 2.0}
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "window": [10, 20, 40],
        "atr_window": [10, 14, 20],
        "multiplier": [1.5, 2.0, 2.5, 3.0],
    }

    def validate_params(self) -> None:
        if self.params["multiplier"] <= 0:
            raise ValueError("multiplier must be positive")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        close = ohlcv["Close"]
        middle, upper, _ = keltner_channels(
            ohlcv["High"], ohlcv["Low"], close,
            window=self.params["window"], atr_window=self.params["atr_window"],
            multiplier=self.params["multiplier"],
        )
        return {
            "entries": close.vbt.crossed_above(upper),
            "exits": close.vbt.crossed_below(middle),
        }


@register_strategy
class VolatilityExpansionStrategy(BaseStrategy):
    name: ClassVar[str] = "volatility_expansion"
    category: ClassVar[str] = _MOMENTUM
    description: ClassVar[str] = (
        "Volatility regime breakout: enter when short-term realized volatility expands above "
        "long-term volatility by a ratio while price trends up; exit when the expansion deflates "
        "or the trend turns down."
    )
    default_params: ClassVar[dict[str, Any]] = {
        "short_window": 10,
        "long_window": 60,
        "ratio": 1.5,
        "trend_window": 50,
    }
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "short_window": [5, 10, 20],
        "long_window": [40, 60, 100],
        "ratio": [1.2, 1.5, 2.0],
        "trend_window": [30, 50, 100],
    }

    def validate_params(self) -> None:
        if self.params["short_window"] >= self.params["long_window"]:
            raise ValueError("short_window must be < long_window")
        if self.params["ratio"] <= 1.0:
            raise ValueError("ratio must be > 1")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        close = ohlcv["Close"]
        vol_ratio = realized_volatility(close, self.params["short_window"]) / realized_volatility(
            close, self.params["long_window"]
        )
        uptrend = close > sma(close, self.params["trend_window"])
        expanding = vol_ratio > self.params["ratio"]
        return {
            "entries": rising_edge(expanding & uptrend),
            "exits": rising_edge(~expanding | ~uptrend),
        }
