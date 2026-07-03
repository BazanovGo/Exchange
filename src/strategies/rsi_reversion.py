"""RSI mean-reversion strategy (reference implementation)."""

from __future__ import annotations

from typing import Any, ClassVar

import pandas as pd

from src.indicators import rsi
from src.strategies.base import BaseStrategy, SignalArray, register_strategy


@register_strategy
class RSIMeanReversionStrategy(BaseStrategy):
    """Buy oversold, sell overbought.

    Long entry when RSI crosses below ``lower``, long exit when RSI crosses
    back above ``exit_level``. With ``allow_short=True`` the mirrored logic
    trades the overbought side.
    """

    name: ClassVar[str] = "rsi_reversion"
    description: ClassVar[str] = (
        "Mean-reversion: enter long on oversold RSI, exit at the midline; "
        "optionally short the overbought side."
    )
    default_params: ClassVar[dict[str, Any]] = {
        "window": 14,
        "lower": 30.0,
        "upper": 70.0,
        "exit_level": 50.0,
        "allow_short": False,
    }

    def validate_params(self) -> None:
        p = self.params
        if not (0 < p["lower"] < p["exit_level"] < p["upper"] < 100):
            raise ValueError("Expected 0 < lower < exit_level < upper < 100, got "
                             f"lower={p['lower']}, exit_level={p['exit_level']}, upper={p['upper']}")
        if p["window"] < 2:
            raise ValueError("window must be >= 2")

    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        p = self.params
        rsi_ = rsi(ohlcv["Close"], window=p["window"])

        signals: dict[str, SignalArray] = {
            "entries": rsi_.vbt.crossed_below(p["lower"]),
            "exits": rsi_.vbt.crossed_above(p["exit_level"]),
        }
        if p["allow_short"]:
            signals["short_entries"] = rsi_.vbt.crossed_above(p["upper"])
            signals["short_exits"] = rsi_.vbt.crossed_below(p["exit_level"])
        return signals
