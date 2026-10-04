"""Trading cost models.

``fees=0.001`` is not a cost model, and that is all the audited project had
(plus a ``slippage`` scalar). Real cost has at least five components that
scale differently, and collapsing them into one number makes a strategy's
sensitivity to each of them invisible:

============  =====================================  ==================
component     scales with                            typical futures
============  =====================================  ==================
commission    number of contracts                    fixed $/contract
exchange fee  number of contracts                    fixed $/contract
spread        volatility, tick size, liquidity       half the bid/ask
slippage      urgency, volatility, latency           ticks or bps
impact        (order size / volume) ** 0.5           sqrt participation
============  =====================================  ==================

The spread is the one most often forgotten and is usually the largest:
crossing the book costs half the spread *each way* even with a perfect fill.
Note also that a proportional ``fees`` cannot express a per-contract
commission, which is how futures are actually charged.

The square-root impact law is the standard empirical form. It is used here
for its *shape*, not its calibration: it says doubling size does not double
cost, and that impact is proportional to volatility.

Every model is a frozen dataclass, so an experiment record can store exactly
what was assumed and a reproduction can compare it field by field.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np

from exchange.core.errors import ConfigError

BPS = 1e-4


@dataclass(frozen=True)
class CommissionModel:
    """Per-trade fees charged in currency and in basis points of notional.

    Futures desks are charged per contract; equity brokers often charge basis
    points. Both are supported and simply add, because a research system that
    only supports one silently misprices the other asset class.
    """

    per_contract: float = 0.0     # currency per contract traded
    per_trade: float = 0.0        # currency per order, regardless of size
    bps_of_notional: float = 0.0  # basis points of traded notional
    exchange_fee: float = 0.0     # currency per contract (clearing/exchange)
    minimum: float = 0.0          # floor per trade

    def __post_init__(self) -> None:
        for name in ("per_contract", "per_trade", "bps_of_notional", "exchange_fee", "minimum"):
            if getattr(self, name) < 0:
                raise ConfigError(f"{name} must be non-negative")

    def charge(self, quantity: float, price: float, multiplier: float = 1.0) -> float:
        """Currency cost of trading ``|quantity|`` contracts at ``price``."""
        qty = abs(float(quantity))
        if qty == 0:
            return 0.0
        notional = qty * abs(price) * multiplier
        cost = (
            qty * (self.per_contract + self.exchange_fee)
            + self.per_trade
            + notional * self.bps_of_notional * BPS
        )
        return max(cost, self.minimum)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SpreadModel:
    """The bid/ask spread crossed by a marketable order.

    ``half_spread_bps`` is charged on *every* marketable fill, in the
    direction that hurts: a buy pays above mid, a sell receives below mid.
    This is not slippage — it is the unavoidable cost of demanding
    liquidity, present even with instantaneous, zero-impact execution.

    ``ticks`` expresses the same thing in contract ticks, which is how
    futures spreads are actually quoted; when both are set they add.

    ``volatility_multiple`` widens the spread in proportion to realised
    volatility, capturing the fact that spreads blow out precisely when a
    volatility-triggered strategy wants to trade.
    """

    half_spread_bps: float = 0.0
    ticks: float = 0.0
    volatility_multiple: float = 0.0   # extra half-spread as a multiple of bar volatility

    def __post_init__(self) -> None:
        if self.half_spread_bps < 0 or self.ticks < 0 or self.volatility_multiple < 0:
            raise ConfigError("spread parameters must be non-negative")

    def half_cost(self, price: float, *, tick_size: float = 0.0, volatility: float = 0.0) -> float:
        """Price offset paid on one fill, in price units (always positive)."""
        cost = abs(price) * self.half_spread_bps * BPS
        cost += self.ticks * tick_size / 2.0
        if self.volatility_multiple and np.isfinite(volatility):
            cost += self.volatility_multiple * abs(volatility) * abs(price)
        return float(cost)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SlippageModel:
    """Adverse price movement between decision and fill.

    Two components, because they behave differently under stress:

    ``bps``
        A constant adverse offset. Simple, and the right default when
        nothing better is known.
    ``volatility_multiple``
        An offset proportional to the bar's own realised volatility. This is
        the component that matters: slippage is not constant, it explodes in
        fast markets, and a strategy that only trades in fast markets has
        systematically worse fills than its average suggests.

    ``latency_bars`` is separate and handled by the fill engine: it delays
    *when* an order reaches the market, which is a different failure mode
    from paying more at the same moment.
    """

    bps: float = 0.0
    volatility_multiple: float = 0.0
    ticks: float = 0.0

    def __post_init__(self) -> None:
        if self.bps < 0 or self.volatility_multiple < 0 or self.ticks < 0:
            raise ConfigError("slippage parameters must be non-negative")

    def cost(self, price: float, *, tick_size: float = 0.0, volatility: float = 0.0) -> float:
        """Adverse price offset in price units (always positive)."""
        cost = abs(price) * self.bps * BPS + self.ticks * tick_size
        if self.volatility_multiple and np.isfinite(volatility):
            cost += self.volatility_multiple * abs(volatility) * abs(price)
        return float(cost)

    def scaled(self, factor: float) -> SlippageModel:
        """A copy with every component multiplied by ``factor``.

        Used by the falsification engine to ask "does the edge survive 2x or
        3x slippage?" without rebuilding the config.
        """
        if factor < 0:
            raise ConfigError("slippage scale factor must be non-negative")
        return replace(
            self,
            bps=self.bps * factor,
            volatility_multiple=self.volatility_multiple * factor,
            ticks=self.ticks * factor,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ImpactModel:
    """Market impact as a function of participation in the bar's volume.

    ``impact = coefficient * volatility * (quantity / bar_volume) ** exponent``

    The square-root form is the standard empirical regularity (Almgren et
    al.; see also Bouchaud's review of the "square-root law"). Its important
    property here is *shape*, not calibration: it says doubling size does not
    double cost, and that impact is proportional to volatility. Any absolute
    calibration must come from a desk's own fills, so ``coefficient``
    defaults to a deliberately conservative value rather than a fitted one.

    ``max_participation`` is the hard liquidity constraint: an order larger
    than this share of the bar's volume cannot be filled in that bar. This is
    what turns "my backtest trades 10,000 lots of a contract that trades 400"
    from an invisible fantasy into an explicit partial fill.
    """

    coefficient: float = 0.0
    max_participation: float = 0.10   # of the bar's volume
    exponent: float = 0.5

    def __post_init__(self) -> None:
        if self.coefficient < 0:
            raise ConfigError("impact coefficient must be non-negative")
        if not 0 < self.max_participation <= 1.0:
            raise ConfigError("max_participation must be in (0, 1]")
        if not 0 < self.exponent <= 1.0:
            raise ConfigError("impact exponent must be in (0, 1]")

    def cost(self, price: float, quantity: float, bar_volume: float, volatility: float) -> float:
        """Impact cost as a positive price offset."""
        qty = abs(float(quantity))
        if qty == 0 or self.coefficient == 0:
            return 0.0
        if not np.isfinite(bar_volume) or bar_volume <= 0:
            # No volume information: charge the impact of a full-participation
            # order rather than assuming free liquidity.
            participation = self.max_participation
        else:
            participation = min(qty / bar_volume, 1.0)
        vol = abs(volatility) if np.isfinite(volatility) else 0.0
        return float(self.coefficient * vol * abs(price) * participation**self.exponent)

    def fillable(self, quantity: float, bar_volume: float) -> float:
        """Largest quantity that can be filled in a bar of ``bar_volume``.

        Returns ``|quantity|`` when volume is unknown *and* the model is
        disabled; otherwise caps at ``max_participation * bar_volume``.
        """
        qty = abs(float(quantity))
        if not np.isfinite(bar_volume) or bar_volume <= 0:
            return qty if self.coefficient == 0 else 0.0
        return min(qty, self.max_participation * float(bar_volume))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CostModel:
    """The full cost stack applied to every fill."""

    commission: CommissionModel = field(default_factory=CommissionModel)
    spread: SpreadModel = field(default_factory=SpreadModel)
    slippage: SlippageModel = field(default_factory=SlippageModel)
    impact: ImpactModel = field(default_factory=ImpactModel)

    def price_offset(
        self,
        price: float,
        quantity: float,
        *,
        tick_size: float = 0.0,
        volatility: float = 0.0,
        bar_volume: float = float("nan"),
    ) -> float:
        """Total adverse price offset for one fill, always positive."""
        return (
            self.spread.half_cost(price, tick_size=tick_size, volatility=volatility)
            + self.slippage.cost(price, tick_size=tick_size, volatility=volatility)
            + self.impact.cost(price, quantity, bar_volume, volatility)
        )

    def scale_slippage(self, factor: float) -> CostModel:
        return replace(self, slippage=self.slippage.scaled(factor))

    def scale_spread(self, factor: float) -> CostModel:
        return replace(
            self,
            spread=SpreadModel(
                half_spread_bps=self.spread.half_spread_bps * factor,
                ticks=self.spread.ticks * factor,
                volatility_multiple=self.spread.volatility_multiple * factor,
            ),
        )

    def scale_commission(self, factor: float) -> CostModel:
        c = self.commission
        return replace(
            self,
            commission=CommissionModel(
                per_contract=c.per_contract * factor,
                per_trade=c.per_trade * factor,
                bps_of_notional=c.bps_of_notional * factor,
                exchange_fee=c.exchange_fee * factor,
                minimum=c.minimum * factor,
            ),
        )

    def to_dict(self) -> dict:
        return {
            "commission": self.commission.to_dict(),
            "spread": self.spread.to_dict(),
            "slippage": self.slippage.to_dict(),
            "impact": self.impact.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CostModel:
        return cls(
            commission=CommissionModel(**data.get("commission", {})),
            spread=SpreadModel(**data.get("spread", {})),
            slippage=SlippageModel(**data.get("slippage", {})),
            impact=ImpactModel(**data.get("impact", {})),
        )

    def summary(self) -> str:
        return (
            f"commission {self.commission.per_contract:g}/contract + "
            f"{self.commission.bps_of_notional:g}bps | "
            f"half-spread {self.spread.half_spread_bps:g}bps | "
            f"slippage {self.slippage.bps:g}bps + {self.slippage.volatility_multiple:g}x vol | "
            f"impact k={self.impact.coefficient:g}, max {self.impact.max_participation:.0%} of volume"
        )


ZERO_COST = CostModel()
"""No costs at all. The only legitimate use is isolating a signal's raw
structure. A verdict must never be based on a zero-cost run."""
