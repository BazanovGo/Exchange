"""Order types and the fill rules that make them realistic.

The rule that governs everything in this module
------------------------------------------------
**A price may only be used as a fill if it was reachable after the decision
was made.** Concretely: a decision taken from bar ``t``'s close cannot be
filled at bar ``t``'s close, because at the moment that price printed the
decision did not exist. The earliest legal fill is bar ``t+1``.

The audited project filled at the signal bar's own close — its exit
simulator says so in its own docstring ("entries fill at the bar close") and
``vbt.Portfolio.from_signals`` does the same. That single assumption
flatters every strategy reacting to closing prices, which was all 26 of
them. Removing it typically removes a large part of the apparent edge, which
is the point.

Fill semantics per order type (all evaluated on the execution bar)
------------------------------------------------------------------
``MARKET``
    Fills at the bar's open, plus the adverse cost offset. Always fills.
``LIMIT``
    Fills only if the bar traded through the limit. A buy limit at ``L``
    fills when the bar trades below it; the fill price is ``min(open, L)`` —
    if the bar opened below the limit the trader gets the better open price,
    which is what actually happens. **No spread or slippage is charged**: a
    resting limit order provides liquidity rather than taking it.
    Conservatively, a bar that merely *touches* ``L`` counts as a fill only
    when ``touch_fills`` is set, because queue position at the exact touch
    price is an assumption, not a fact.
``STOP``
    Triggers when the bar trades through the stop level, then behaves as a
    market order: fill at ``min/max(open, stop)`` in the direction that
    hurts, so a gap through the stop fills at the gap price, not the stop.
``STOP_LIMIT``
    Triggers like a stop, then rests like a limit — and can therefore fail
    to fill entirely, which is exactly the risk a real stop-limit carries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from exchange.core.errors import ConfigError


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    @classmethod
    def of(cls, quantity: float) -> Side:
        return cls.BUY if quantity >= 0 else cls.SELL


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    DAY = "day"      # cancelled if unfilled at the end of the execution bar
    GTC = "gtc"      # rests until filled or explicitly cancelled


@dataclass(frozen=True)
class Bar:
    """One bar's tradeable prices, as seen by the fill engine."""

    open: float
    high: float
    low: float
    close: float
    volume: float = float("nan")

    @property
    def range(self) -> float:
        return float(self.high - self.low)

    @property
    def typical_volatility(self) -> float:
        """Bar range as a fraction of its own price — the volatility proxy
        the cost models scale with."""
        mid = 0.5 * (self.high + self.low)
        return float(self.range / mid) if mid > 0 else 0.0


@dataclass
class Order:
    """An instruction submitted at a decision time, filled on a later bar."""

    side: Side
    quantity: float                       # always positive; direction is `side`
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    stop_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    tag: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ConfigError(f"order quantity must be positive, got {self.quantity}")
        if self.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and self.limit_price is None:
            raise ConfigError(f"{self.order_type.value} order requires a limit_price")
        if self.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and self.stop_price is None:
            raise ConfigError(f"{self.order_type.value} order requires a stop_price")

    @property
    def signed_quantity(self) -> float:
        return self.side.sign * self.quantity

    def to_dict(self) -> dict:
        return {
            "side": self.side.value,
            "quantity": self.quantity,
            "type": self.order_type.value,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "tif": self.time_in_force.value,
            "tag": self.tag,
        }


@dataclass(frozen=True)
class Fill:
    """The result of matching one order against one bar."""

    filled: float          # quantity filled (0 when the order did not fill)
    price: float           # execution price *before* cost offsets
    triggered: bool        # stop orders: did the trigger fire?
    reason: str = ""       # why the order did not fill, when it did not

    @property
    def did_fill(self) -> bool:
        return self.filled > 0


NO_FILL = Fill(filled=0.0, price=float("nan"), triggered=False, reason="not filled")


def match(
    order: Order,
    bar: Bar,
    *,
    touch_fills: bool = False,
    max_fillable: float | None = None,
) -> Fill:
    """Match one order against one execution bar.

    Parameters
    ----------
    touch_fills:
        Whether a bar that merely touches a resting limit price counts as a
        fill. Default ``False`` (conservative): queue position at the exact
        touch price is not something a backtest can know.
    max_fillable:
        Liquidity cap for this bar (see
        :meth:`~exchange.execution.costs.ImpactModel.fillable`). ``None``
        means unconstrained.

    Returns
    -------
    Fill
        ``price`` is the raw execution level; the caller applies cost
        offsets, because whether a fill *takes* or *provides* liquidity
        depends on the order type and only the caller knows the cost model.
    """
    cap = order.quantity if max_fillable is None else min(order.quantity, max(0.0, float(max_fillable)))
    if cap <= 0:
        return Fill(0.0, float("nan"), triggered=False, reason="no liquidity in bar")

    if not np.isfinite([bar.open, bar.high, bar.low, bar.close]).all():
        return Fill(0.0, float("nan"), triggered=False, reason="bar has non-finite prices")

    if order.order_type is OrderType.MARKET:
        return Fill(cap, float(bar.open), triggered=True)

    if order.order_type is OrderType.LIMIT:
        return _match_limit(order, bar, cap, touch_fills)

    if not _stop_triggered(order, bar):
        return Fill(0.0, float("nan"), triggered=False, reason="stop not triggered")

    stop = float(order.stop_price)  # type: ignore[arg-type]
    if order.order_type is OrderType.STOP:
        # A gap through the stop fills at the gap, not at the stop level.
        price = max(bar.open, stop) if order.side is Side.BUY else min(bar.open, stop)
        return Fill(cap, float(price), triggered=True)

    # STOP_LIMIT: once triggered it rests as a limit and may never fill.
    #
    # The reference price is the stop level, or the open when the bar gapped
    # through it — the same gap logic as a plain stop, because the trigger
    # cannot have happened at a better price than the one that caused it.
    # Treating the order as a limit resting over the *whole* bar would be
    # wrong in a specific, flattering way: a sell stop at 99.5 on a bar
    # opening at 99.9 would fill at 99.9, a price that was only available
    # before the stop triggered.
    #
    # If the reference is beyond the limit, the market gapped past it and the
    # order does not fill — the classic stop-limit failure in a fast market,
    # and the one a backtest must reproduce. An intrabar recovery back to the
    # limit is not assumed: the path within a bar is unknown, and assuming
    # the favourable path is how a backtest invents fills.
    reference = min(bar.open, stop) if order.side is Side.SELL else max(bar.open, stop)
    limit = float(order.limit_price)  # type: ignore[arg-type]
    reachable = reference >= limit if order.side is Side.SELL else reference <= limit
    if not reachable:
        return Fill(
            0.0, float("nan"), triggered=True,
            reason=f"stop triggered at {reference:g} but the market gapped past the limit {limit:g}",
        )
    return Fill(cap, float(reference), triggered=True)


def _stop_triggered(order: Order, bar: Bar) -> bool:
    stop = float(order.stop_price)  # type: ignore[arg-type]
    # A buy stop is above the market and triggers on the way up; a sell stop
    # is below and triggers on the way down. The open is included: a bar that
    # gaps past the stop has already triggered it.
    return bar.high >= stop if order.side is Side.BUY else bar.low <= stop


def _match_limit(order: Order, bar: Bar, cap: float, touch_fills: bool) -> Fill:
    limit = float(order.limit_price)  # type: ignore[arg-type]
    if order.side is Side.BUY:
        reached = bar.low < limit or (touch_fills and bar.low <= limit)
        if not reached:
            return Fill(0.0, float("nan"), triggered=True, reason="limit not reached")
        # Opening below the limit gives the buyer the better (lower) price.
        return Fill(cap, float(min(bar.open, limit)), triggered=True)

    reached = bar.high > limit or (touch_fills and bar.high >= limit)
    if not reached:
        return Fill(0.0, float("nan"), triggered=True, reason="limit not reached")
    return Fill(cap, float(max(bar.open, limit)), triggered=True)


def takes_liquidity(order_type: OrderType) -> bool:
    """Does this order type cross the spread?

    Market and stop orders take liquidity and pay the spread; resting limit
    orders provide it and do not. Getting this backwards either flatters
    limit strategies or unfairly penalises them.
    """
    return order_type in (OrderType.MARKET, OrderType.STOP)
