"""Trend-following strategies."""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from src.indicators import adx, donchian, ema, ichimoku, parabolic_sar, sma, supertrend
from src.strategies.base import BaseStrategy, SignalArray, register_strategy
from src.strategies.signal_utils import rising_edge

_TREND = "trend"


class _MACrossBase(BaseStrategy):
    """Shared fast/slow crossover logic; subclasses pick the MA type."""

    category: ClassVar[str] = _TREND
    _ewm: ClassVar[bool] = False
    default_params: ClassVar[dict[str, Any]] = {"fast_window": 20, "slow_window": 50, "allow_short": False}
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "fast_window": [5, 10, 20, 30, 50],
        "slow_window": [50, 100, 150, 200],
    }

    def validate_params(self) -> None:
        if self.params["fast_window"] >= self.params["slow_window"]:
            raise ValueError("fast_window must be < slow_window")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        close = ohlcv["Close"]
        fast = ema(close, self.params["fast_window"]) if self._ewm else sma(close, self.params["fast_window"])
        slow = ema(close, self.params["slow_window"]) if self._ewm else sma(close, self.params["slow_window"])
        up = fast.vbt.crossed_above(slow)
        down = fast.vbt.crossed_below(slow)
        sig: dict[str, SignalArray] = {"entries": up, "exits": down}
        if self.params["allow_short"]:
            sig["short_entries"], sig["short_exits"] = down, up
        return sig


@register_strategy
class SMACrossStrategy(_MACrossBase):
    name: ClassVar[str] = "sma_cross"
    description: ClassVar[str] = "Golden/death cross of two simple moving averages."


@register_strategy
class EMACrossStrategy(_MACrossBase):
    name: ClassVar[str] = "ema_cross"
    description: ClassVar[str] = "Golden/death cross of two exponential moving averages."
    _ewm: ClassVar[bool] = True


@register_strategy
class TripleEMAStrategy(BaseStrategy):
    name: ClassVar[str] = "triple_ema"
    category: ClassVar[str] = _TREND
    description: ClassVar[str] = (
        "Three stacked EMAs: enter long when fast crosses above mid while both are above slow "
        "(trend alignment); exit when fast crosses back below mid."
    )
    default_params: ClassVar[dict[str, Any]] = {"fast_window": 10, "mid_window": 25, "slow_window": 60}
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "fast_window": [5, 8, 12],
        "mid_window": [20, 30, 40],
        "slow_window": [60, 100, 150],
    }

    def validate_params(self) -> None:
        p = self.params
        if not (p["fast_window"] < p["mid_window"] < p["slow_window"]):
            raise ValueError("Expected fast_window < mid_window < slow_window")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        close = ohlcv["Close"]
        fast = ema(close, self.params["fast_window"])
        mid = ema(close, self.params["mid_window"])
        slow = ema(close, self.params["slow_window"])
        entries = fast.vbt.crossed_above(mid) & (mid > slow)
        exits = fast.vbt.crossed_below(mid)
        return {"entries": entries, "exits": exits}


@register_strategy
class SupertrendStrategy(BaseStrategy):
    name: ClassVar[str] = "supertrend"
    category: ClassVar[str] = _TREND
    description: ClassVar[str] = (
        "ATR-based Supertrend line: long while price holds above the line, exit (or flip short) "
        "when the trend direction flips."
    )
    default_params: ClassVar[dict[str, Any]] = {"window": 10, "multiplier": 3.0, "allow_short": False}
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "window": [7, 10, 14, 21],
        "multiplier": [1.5, 2.0, 3.0, 4.0],
    }

    def validate_params(self) -> None:
        if self.params["multiplier"] <= 0:
            raise ValueError("multiplier must be positive")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        _, direction = supertrend(
            ohlcv["High"], ohlcv["Low"], ohlcv["Close"],
            window=self.params["window"], multiplier=self.params["multiplier"],
        )
        entries = rising_edge(direction > 0)
        exits = rising_edge(direction < 0)
        sig: dict[str, SignalArray] = {"entries": entries, "exits": exits}
        if self.params["allow_short"]:
            sig["short_entries"], sig["short_exits"] = exits, entries
        return sig


@register_strategy
class DonchianBreakoutStrategy(BaseStrategy):
    name: ClassVar[str] = "donchian"
    category: ClassVar[str] = _TREND
    description: ClassVar[str] = (
        "Classic channel breakout: enter on a close above the N-bar high, "
        "exit on a close below the M-bar low (M < N)."
    )
    default_params: ClassVar[dict[str, Any]] = {"entry_window": 55, "exit_window": 20}
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "entry_window": [20, 40, 55, 80],
        "exit_window": [10, 20, 40],
    }

    def validate_params(self) -> None:
        if self.params["exit_window"] >= self.params["entry_window"]:
            raise ValueError("exit_window must be < entry_window")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        close = ohlcv["Close"]
        upper, _, _ = donchian(ohlcv["High"], ohlcv["Low"], self.params["entry_window"])
        _, _, lower = donchian(ohlcv["High"], ohlcv["Low"], self.params["exit_window"])
        return {"entries": close.vbt.crossed_above(upper), "exits": close.vbt.crossed_below(lower)}


@register_strategy
class ADXTrendStrategy(BaseStrategy):
    name: ClassVar[str] = "adx_trend"
    category: ClassVar[str] = _TREND
    description: ClassVar[str] = (
        "Directional movement: enter when +DI crosses above -DI with ADX confirming trend strength; "
        "exit on the opposite DI cross or when ADX fades below the threshold."
    )
    default_params: ClassVar[dict[str, Any]] = {"window": 14, "adx_threshold": 25.0}
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "window": [10, 14, 20, 28],
        "adx_threshold": [20.0, 25.0, 30.0],
    }

    def validate_params(self) -> None:
        if self.params["adx_threshold"] <= 0:
            raise ValueError("adx_threshold must be positive")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        adx_, plus_di, minus_di = adx(ohlcv["High"], ohlcv["Low"], ohlcv["Close"], self.params["window"])
        th = self.params["adx_threshold"]
        entries = plus_di.vbt.crossed_above(minus_di) & (adx_ > th)
        exits = plus_di.vbt.crossed_below(minus_di) | adx_.vbt.crossed_below(th)
        return {"entries": entries, "exits": exits}


@register_strategy
class ParabolicSARStrategy(BaseStrategy):
    name: ClassVar[str] = "psar"
    category: ClassVar[str] = _TREND
    description: ClassVar[str] = (
        "Parabolic SAR trend flips: long while SAR trails below price, exit (or flip short) on reversal."
    )
    default_params: ClassVar[dict[str, Any]] = {"af0": 0.02, "af_step": 0.02, "af_max": 0.2, "allow_short": False}
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "af0": [0.01, 0.02, 0.03],
        "af_step": [0.01, 0.02, 0.03],
        "af_max": [0.1, 0.2, 0.3],
    }

    def validate_params(self) -> None:
        if not (0 < self.params["af0"] <= self.params["af_max"]):
            raise ValueError("Expected 0 < af0 <= af_max")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        _, trend = parabolic_sar(
            ohlcv["High"], ohlcv["Low"],
            af0=self.params["af0"], af_step=self.params["af_step"], af_max=self.params["af_max"],
        )
        entries = rising_edge(trend > 0)
        exits = rising_edge(trend < 0)
        sig: dict[str, SignalArray] = {"entries": entries, "exits": exits}
        if self.params["allow_short"]:
            sig["short_entries"], sig["short_exits"] = exits, entries
        return sig


@register_strategy
class IchimokuStrategy(BaseStrategy):
    name: ClassVar[str] = "ichimoku"
    category: ClassVar[str] = _TREND
    description: ClassVar[str] = (
        "Ichimoku: enter on tenkan/kijun bullish cross while price is above the cloud; "
        "exit on the bearish cross or a close below the cloud."
    )
    default_params: ClassVar[dict[str, Any]] = {"tenkan_window": 9, "kijun_window": 26, "senkou_b_window": 52}
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "tenkan_window": [7, 9, 12],
        "kijun_window": [22, 26, 34],
        "senkou_b_window": [44, 52, 68],
    }

    def validate_params(self) -> None:
        p = self.params
        if not (p["tenkan_window"] < p["kijun_window"] < p["senkou_b_window"]):
            raise ValueError("Expected tenkan_window < kijun_window < senkou_b_window")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        close = ohlcv["Close"]
        ich = ichimoku(
            ohlcv["High"], ohlcv["Low"], close,
            tenkan_window=self.params["tenkan_window"],
            kijun_window=self.params["kijun_window"],
            senkou_b_window=self.params["senkou_b_window"],
        )
        cloud_top = pd.concat([ich["senkou_a"], ich["senkou_b"]], axis=1).max(axis=1)
        cloud_bottom = pd.concat([ich["senkou_a"], ich["senkou_b"]], axis=1).min(axis=1)
        entries = ich["tenkan"].vbt.crossed_above(ich["kijun"]) & (close > cloud_top)
        exits = ich["tenkan"].vbt.crossed_below(ich["kijun"]) | close.vbt.crossed_below(cloud_bottom)
        return {"entries": entries, "exits": exits}
