"""Position management: exit mechanics layered on top of strategy signals.

A single numba simulator walks the bars and applies every exit mechanic in
one place — stop loss, take profit, trailing stop, break-even, time stop,
signal exits and partial profit taking — producing target-position and
execution-price arrays that feed ``vbt.Portfolio.from_orders``. This gives:

* correct intrabar execution prices for stops (gap-aware: if the bar opens
  through the level, the fill is at the open);
* an exit-reason code per bar, so mechanics can be attributed;
* arbitrary combinations of mechanics without double-tracking state.

Execution model (long-only):

* entries fill at the bar close (same convention as ``from_signals``);
* on each subsequent bar, checks run in order: gap-open stop -> intrabar
  stop (SL / trailing / break-even use the *previous* bars' peak, so a bar
  cannot raise a trailing stop and hit it at once) -> intrabar take profit
  (stop wins if both are touched: conservative) -> time stop / signal exit
  at the close;
* partial take profit scales the position down once per trade at its
  level; the remainder keeps trailing under the other rules.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt
from numba import njit

from src.backtesting.config import BacktestConfig

EXIT_REASONS = {
    0: "none",
    1: "signal",
    2: "stop_loss",
    3: "take_profit",
    4: "trailing_stop",
    5: "break_even",
    6: "time_stop",
    7: "partial_tp",
}

_NO_EXIT = 0.0  # sentinel used inside the numba kernel


@dataclass(frozen=True)
class ExitRules:
    """One configuration of position-management mechanics.

    All price-distance parameters are fractions of the entry price
    (``0.05`` = 5%). ``None`` disables a mechanic.
    """

    label: str = "baseline"
    use_signal_exits: bool = True            # exit on the strategy's opposite signal
    sl_stop: float | None = None             # fixed stop loss below entry
    tp_stop: float | None = None             # take profit above entry
    trail_stop: float | None = None          # trailing stop below the post-entry peak
    break_even_trigger: float | None = None  # profit level that arms the break-even stop
    break_even_offset: float = 0.0           # where the BE stop sits relative to entry
    time_stop: int | None = None             # max bars in a trade
    partial_tp: float | None = None          # profit level for scaling out once
    partial_fraction: float = 0.5            # share of the position closed at partial_tp

    def __post_init__(self) -> None:
        for name in ("sl_stop", "tp_stop", "trail_stop", "break_even_trigger", "partial_tp"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.time_stop is not None and self.time_stop < 1:
            raise ValueError("time_stop must be >= 1 bar")
        if not 0 < self.partial_fraction < 1:
            raise ValueError("partial_fraction must be in (0, 1)")
        if self.partial_tp is not None and self.tp_stop is not None and self.partial_tp >= self.tp_stop:
            raise ValueError("partial_tp must be below tp_stop")

    def params(self) -> dict[str, Any]:
        """Active mechanic parameters (for reporting)."""
        out: dict[str, Any] = {}
        for f in fields(self):
            if f.name == "label":
                continue
            value = getattr(self, f.name)
            default = f.default
            if value != default:
                out[f.name] = value
        return out


@njit(cache=True)
def _manage_nb(  # noqa: PLR0912 - a bar-by-bar state machine is clearest inline
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    entries: np.ndarray,
    sig_exits: np.ndarray,
    use_sig_exits: bool,
    sl: float,        # <=0 disables
    tp: float,
    trail: float,
    be_trigger: float,
    be_offset: float,
    time_stop: int,   # <=0 disables
    partial_tp: float,
    partial_fraction: float,
):
    n = len(close)
    target = np.full(n, np.nan)   # target percent of equity, NaN = no order
    price = np.full(n, np.nan)    # execution price, NaN = close
    reason = np.zeros(n, dtype=np.int8)

    in_pos = False
    entry_price = 0.0
    peak = 0.0            # highest high since entry (excluding current bar)
    bars_held = 0
    be_armed = False
    partial_done = False

    for i in range(n):
        if not in_pos:
            if entries[i]:
                in_pos = True
                entry_price = close[i]
                peak = close[i]
                bars_held = 0
                be_armed = False
                partial_done = False
                target[i] = 1.0
                price[i] = close[i]
            continue

        bars_held += 1

        # --- assemble the active protective stop from previous-bar state ---
        stop_price = -1.0
        stop_reason = 0
        if sl > 0.0:
            level = entry_price * (1.0 - sl)
            if level > stop_price:
                stop_price, stop_reason = level, 2
        if trail > 0.0:
            level = peak * (1.0 - trail)
            if level > stop_price:
                stop_price, stop_reason = level, 4
        if be_armed:
            level = entry_price * (1.0 + be_offset)
            if level > stop_price:
                stop_price, stop_reason = level, 5

        exited = False

        # --- gap through the stop: fill at the open ---
        if stop_price > 0.0 and open_[i] <= stop_price:
            target[i], price[i], reason[i] = 0.0, open_[i], stop_reason
            exited = True
        # --- intrabar stop ---
        elif stop_price > 0.0 and low[i] <= stop_price:
            target[i], price[i], reason[i] = 0.0, stop_price, stop_reason
            exited = True

        # --- partial take profit (once per trade) ---
        if not exited and partial_tp > 0.0 and not partial_done:
            level = entry_price * (1.0 + partial_tp)
            if open_[i] >= level:
                target[i], price[i], reason[i] = 1.0 - partial_fraction, open_[i], 7
                partial_done = True
            elif high[i] >= level:
                target[i], price[i], reason[i] = 1.0 - partial_fraction, level, 7
                partial_done = True

        # --- full take profit ---
        if not exited and tp > 0.0:
            level = entry_price * (1.0 + tp)
            if open_[i] >= level:
                target[i], price[i], reason[i] = 0.0, open_[i], 3
                exited = True
            elif high[i] >= level:
                target[i], price[i], reason[i] = 0.0, level, 3
                exited = True

        # --- close-based exits ---
        if not exited and time_stop > 0 and bars_held >= time_stop:
            target[i], price[i], reason[i] = 0.0, close[i], 6
            exited = True
        if not exited and use_sig_exits and sig_exits[i]:
            target[i], price[i], reason[i] = 0.0, close[i], 1
            exited = True

        if exited:
            in_pos = False
            continue

        # --- update state for the next bar ---
        if high[i] > peak:
            peak = high[i]
        if be_trigger > 0.0 and not be_armed and high[i] >= entry_price * (1.0 + be_trigger):
            be_armed = True

    return target, price, reason


def managed_orders(
    ohlcv: pd.DataFrame,
    entries: pd.Series,
    signal_exits: pd.Series,
    rules: ExitRules,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Run the exit simulator; return (target_pct, exec_price, exit_reason)."""
    target, price, reason = _manage_nb(
        ohlcv["Open"].to_numpy(float),
        ohlcv["High"].to_numpy(float),
        ohlcv["Low"].to_numpy(float),
        ohlcv["Close"].to_numpy(float),
        entries.to_numpy(bool),
        signal_exits.to_numpy(bool),
        rules.use_signal_exits,
        rules.sl_stop or _NO_EXIT,
        rules.tp_stop or _NO_EXIT,
        rules.trail_stop or _NO_EXIT,
        rules.break_even_trigger or _NO_EXIT,
        rules.break_even_offset,
        rules.time_stop or 0,
        rules.partial_tp or _NO_EXIT,
        rules.partial_fraction,
    )
    index = ohlcv.index
    return (
        pd.Series(target, index=index, name="target_pct"),
        pd.Series(price, index=index, name="exec_price"),
        pd.Series(reason, index=index, name="exit_reason"),
    )


def managed_portfolio(
    ohlcv: pd.DataFrame,
    entries: pd.Series,
    signal_exits: pd.Series,
    rules: ExitRules,
    config: BacktestConfig | None = None,
) -> tuple[vbt.Portfolio, pd.Series]:
    """Backtest strategy signals under one exit-rule configuration.

    Returns the portfolio and the exit-reason series (nonzero on bars where
    a management order fired).
    """
    config = config or BacktestConfig()
    target, price, reason = managed_orders(ohlcv, entries, signal_exits, rules)
    close = ohlcv["Close"]
    pf = vbt.Portfolio.from_orders(
        close=close,
        size=target,
        size_type="targetpercent",
        price=price.fillna(close),
        fees=config.fees,
        slippage=config.slippage,
        init_cash=config.init_cash,
        freq=config.freq,
    )
    return pf, reason


def exit_reason_counts(reason: pd.Series) -> pd.Series:
    """Human-readable histogram of why positions were closed / scaled."""
    counts = reason[reason > 0].map(EXIT_REASONS).value_counts()
    counts.name = "exits"
    return counts
