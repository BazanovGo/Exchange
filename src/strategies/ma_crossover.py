"""Moving-average crossover strategy (reference implementation)."""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from src.indicators import ma
from src.strategies.base import BaseStrategy, SignalArray, register_strategy


@register_strategy
class MACrossoverStrategy(BaseStrategy):
    """Long when the fast MA crosses above the slow MA, exit on the reverse
    cross. With ``allow_short=True`` the reverse cross also opens a short
    position, making the strategy always-in-the-market.
    """

    name: ClassVar[str] = "ma_crossover"
    category: ClassVar[str] = "trend"
    description: ClassVar[str] = (
        "Trend-following: fast/slow moving average crossover. "
        "Long on golden cross, flat (or short) on death cross."
    )
    default_params: ClassVar[dict[str, Any]] = {
        "fast_window": 20,
        "slow_window": 50,
        "ewm": False,
        "allow_short": False,
    }

    def validate_params(self) -> None:
        if self.params["fast_window"] >= self.params["slow_window"]:
            raise ValueError(
                f"fast_window ({self.params['fast_window']}) must be < slow_window ({self.params['slow_window']})"
            )
        if self.params["fast_window"] < 1:
            raise ValueError("fast_window must be >= 1")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        close = ohlcv["Close"]
        fast = ma(close, self.params["fast_window"], ewm=self.params["ewm"])
        slow = ma(close, self.params["slow_window"], ewm=self.params["ewm"])

        golden_cross = fast.vbt.crossed_above(slow)
        death_cross = fast.vbt.crossed_below(slow)

        signals: dict[str, SignalArray] = {"entries": golden_cross, "exits": death_cross}
        if self.params["allow_short"]:
            signals["short_entries"] = death_cross
            signals["short_exits"] = golden_cross
        return signals
