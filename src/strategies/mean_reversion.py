"""Counter-trend / mean-reversion strategies."""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from src.indicators import bollinger_bands, cci, rolling_vwap, stochastic, williams_r, zscore
from src.strategies.base import BaseStrategy, SignalArray, register_strategy

_MEANREV = "mean_reversion"


@register_strategy
class BollingerReversionStrategy(BaseStrategy):
    name: ClassVar[str] = "bollinger_reversion"
    category: ClassVar[str] = _MEANREV
    description: ClassVar[str] = (
        "Fade band extremes: buy a close crossing below the lower Bollinger band, "
        "exit when price reverts to the middle band; optionally short the upper band."
    )
    default_params: ClassVar[dict[str, Any]] = {"window": 20, "alpha": 2.0, "allow_short": False}
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "window": [10, 20, 30, 50],
        "alpha": [1.5, 2.0, 2.5, 3.0],
    }

    def validate_params(self) -> None:
        if self.params["alpha"] <= 0:
            raise ValueError("alpha must be positive")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        close = ohlcv["Close"]
        middle, upper, lower = bollinger_bands(close, self.params["window"], self.params["alpha"])
        sig: dict[str, SignalArray] = {
            "entries": close.vbt.crossed_below(lower),
            "exits": close.vbt.crossed_above(middle),
        }
        if self.params["allow_short"]:
            sig["short_entries"] = close.vbt.crossed_above(upper)
            sig["short_exits"] = close.vbt.crossed_below(middle)
        return sig


@register_strategy
class CCIReversionStrategy(BaseStrategy):
    name: ClassVar[str] = "cci_reversion"
    category: ClassVar[str] = _MEANREV
    description: ClassVar[str] = (
        "Commodity Channel Index reversion: buy when CCI drops below the oversold level, "
        "exit when it recovers above zero."
    )
    default_params: ClassVar[dict[str, Any]] = {"window": 20, "level": 100.0}
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "window": [10, 14, 20, 30],
        "level": [80.0, 100.0, 150.0, 200.0],
    }

    def validate_params(self) -> None:
        if self.params["level"] <= 0:
            raise ValueError("level must be positive")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        cci_ = cci(ohlcv["High"], ohlcv["Low"], ohlcv["Close"], self.params["window"])
        return {
            "entries": cci_.vbt.crossed_below(-self.params["level"]),
            "exits": cci_.vbt.crossed_above(0.0),
        }


@register_strategy
class WilliamsRStrategy(BaseStrategy):
    name: ClassVar[str] = "williams_r"
    category: ClassVar[str] = _MEANREV
    description: ClassVar[str] = (
        "Williams %R reversion: buy when %R falls below the oversold floor, "
        "exit when it climbs back above the midpoint."
    )
    default_params: ClassVar[dict[str, Any]] = {"window": 14, "oversold": -80.0, "exit_level": -50.0}
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "window": [10, 14, 20, 28],
        "oversold": [-90.0, -80.0, -70.0],
        "exit_level": [-50.0, -40.0, -30.0],
    }

    def validate_params(self) -> None:
        if not (-100 <= self.params["oversold"] < self.params["exit_level"] <= 0):
            raise ValueError("Expected -100 <= oversold < exit_level <= 0")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        wr = williams_r(ohlcv["High"], ohlcv["Low"], ohlcv["Close"], self.params["window"])
        return {
            "entries": wr.vbt.crossed_below(self.params["oversold"]),
            "exits": wr.vbt.crossed_above(self.params["exit_level"]),
        }


@register_strategy
class StochasticReversionStrategy(BaseStrategy):
    name: ClassVar[str] = "stochastic"
    category: ClassVar[str] = _MEANREV
    description: ClassVar[str] = (
        "Stochastic oscillator: buy a %K/%D bullish cross inside the oversold zone, "
        "exit when %K reaches the overbought zone or a bearish cross occurs there."
    )
    default_params: ClassVar[dict[str, Any]] = {"k_window": 14, "d_window": 3, "lower": 20.0, "upper": 80.0}
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "k_window": [9, 14, 21],
        "d_window": [3, 5],
        "lower": [15.0, 20.0, 30.0],
        "upper": [70.0, 80.0, 85.0],
    }

    def validate_params(self) -> None:
        if not (0 < self.params["lower"] < self.params["upper"] < 100):
            raise ValueError("Expected 0 < lower < upper < 100")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        k, d = stochastic(
            ohlcv["High"], ohlcv["Low"], ohlcv["Close"],
            k_window=self.params["k_window"], d_window=self.params["d_window"],
        )
        entries = k.vbt.crossed_above(d) & (k < self.params["lower"])
        exits = k.vbt.crossed_above(self.params["upper"]) | (k.vbt.crossed_below(d) & (k > self.params["upper"]))
        return {"entries": entries, "exits": exits}


@register_strategy
class VWAPReversionStrategy(BaseStrategy):
    name: ClassVar[str] = "vwap_reversion"
    category: ClassVar[str] = _MEANREV
    description: ClassVar[str] = (
        "Fade deviations from rolling VWAP: buy when price stretches a threshold below VWAP, "
        "exit when the deviation reverts to zero."
    )
    default_params: ClassVar[dict[str, Any]] = {"window": 20, "threshold": 0.03}
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "window": [10, 20, 30, 50],
        "threshold": [0.02, 0.03, 0.05, 0.08],
    }

    def validate_params(self) -> None:
        if self.params["threshold"] <= 0:
            raise ValueError("threshold must be positive")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        vwap = rolling_vwap(ohlcv["High"], ohlcv["Low"], ohlcv["Close"], ohlcv["Volume"], self.params["window"])
        deviation = ohlcv["Close"] / vwap - 1.0
        return {
            "entries": deviation.vbt.crossed_below(-self.params["threshold"]),
            "exits": deviation.vbt.crossed_above(0.0),
        }


@register_strategy
class ZScoreReversionStrategy(BaseStrategy):
    name: ClassVar[str] = "zscore_reversion"
    category: ClassVar[str] = _MEANREV
    description: ClassVar[str] = (
        "Statistical mean reversion: buy when the rolling z-score of price drops below "
        "-z_entry, exit when it recovers above -z_exit; optionally short the mirror side."
    )
    default_params: ClassVar[dict[str, Any]] = {"window": 20, "z_entry": 2.0, "z_exit": 0.5, "allow_short": False}
    opt_grid: ClassVar[dict[str, list[Any]]] = {
        "window": [10, 20, 40, 60],
        "z_entry": [1.5, 2.0, 2.5, 3.0],
        "z_exit": [0.0, 0.5, 1.0],
    }

    def validate_params(self) -> None:
        if not (0 <= self.params["z_exit"] < self.params["z_entry"]):
            raise ValueError("Expected 0 <= z_exit < z_entry")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        z = zscore(ohlcv["Close"], self.params["window"])
        sig: dict[str, SignalArray] = {
            "entries": z.vbt.crossed_below(-self.params["z_entry"]),
            "exits": z.vbt.crossed_above(-self.params["z_exit"]),
        }
        if self.params["allow_short"]:
            sig["short_entries"] = z.vbt.crossed_above(self.params["z_entry"])
            sig["short_exits"] = z.vbt.crossed_below(self.params["z_exit"])
        return sig
