"""Execution assumptions and the named scenarios.

Every backtest result is a *conditional* statement: "given these execution
assumptions, the strategy did X". Storing the assumptions with the result is
therefore not bookkeeping, it is part of the claim. This module makes the
whole set one serialisable object and defines four standard points on the
optimism axis, so "does the edge survive worse assumptions?" is a one-line
question rather than a re-parameterisation exercise.

The four scenarios
------------------
``optimistic``
    Costs a well-connected desk with good fills might see. Never sufficient
    for a verdict — it exists to bound the *best* case.
``base``
    The default. Realistic retail-to-small-institutional futures costs.
``conservative``
    Wider spreads, doubled slippage, one bar of latency. A strategy that
    still works here has a margin of safety.
``stressed``
    Crisis conditions: 4x spread, 5x slippage, two bars of latency, tight
    liquidity caps. Almost nothing survives this, and that is informative:
    it identifies which strategies are actually liquidity provision in
    disguise.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Literal

from exchange.core.errors import ConfigError
from exchange.execution.costs import (
    CommissionModel,
    CostModel,
    ImpactModel,
    SlippageModel,
    SpreadModel,
)

FillTiming = Literal["next_open", "next_close", "same_close"]
FILL_TIMINGS: tuple[str, ...] = ("next_open", "next_close", "same_close")


@dataclass(frozen=True)
class ExecutionAssumptions:
    """Everything about *how* a decision becomes a position.

    Parameters
    ----------
    fill_timing:
        Where a market order decided from bar ``t`` is filled.

        ``next_open`` (default)
            Bar ``t+1``'s open. The honest default: the decision exists
            before that price prints.
        ``next_close``
            Bar ``t+1``'s close. Legitimate for a strategy that genuinely
            executes on close (e.g. a market-on-close order placed intraday).
        ``same_close``
            Bar ``t``'s own close. **This is a lookahead** — the price used
            to decide is also the price received. It is supported only so
            that the size of the resulting bias can be *measured*, and any
            run using it is flagged and cannot reach a positive verdict.
    latency_bars:
        Extra bars between decision and the order reaching the market, on top
        of the one bar ``fill_timing`` already implies.
    allow_partial_fills:
        When ``True``, an order larger than the bar's liquidity cap fills
        partially and the remainder is re-attempted on the next bar (up to
        ``max_fill_bars``). When ``False``, it fills in full regardless of
        volume — convenient, and a fiction.
    touch_fills:
        Whether a limit order fills when the bar merely touches its price.
    """

    costs: CostModel = field(default_factory=CostModel)
    fill_timing: FillTiming = "next_open"
    latency_bars: int = 0
    allow_partial_fills: bool = True
    max_fill_bars: int = 5
    touch_fills: bool = False
    label: str = "custom"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.fill_timing not in FILL_TIMINGS:
            raise ConfigError(f"fill_timing must be one of {FILL_TIMINGS}, got {self.fill_timing!r}")
        if self.latency_bars < 0:
            raise ConfigError("latency_bars must be non-negative")
        if self.max_fill_bars < 1:
            raise ConfigError("max_fill_bars must be >= 1")

    @property
    def admits_lookahead(self) -> bool:
        """True when the assumptions knowingly use an unreachable price."""
        return self.fill_timing == "same_close"

    @property
    def execution_delay(self) -> int:
        """Bars between the decision bar and the execution bar."""
        return (0 if self.fill_timing == "same_close" else 1) + self.latency_bars

    # -- transformations used by the falsification engine ----------------

    def with_costs(self, costs: CostModel, *, label: str | None = None) -> ExecutionAssumptions:
        return replace(self, costs=costs, label=label or f"{self.label}*")

    def scale_slippage(self, factor: float) -> ExecutionAssumptions:
        return replace(
            self,
            costs=self.costs.scale_slippage(factor),
            label=f"{self.label}+slip{factor:g}x",
        )

    def scale_spread(self, factor: float) -> ExecutionAssumptions:
        return replace(self, costs=self.costs.scale_spread(factor), label=f"{self.label}+spread{factor:g}x")

    def scale_commission(self, factor: float) -> ExecutionAssumptions:
        return replace(self, costs=self.costs.scale_commission(factor), label=f"{self.label}+comm{factor:g}x")

    def with_latency(self, bars: int) -> ExecutionAssumptions:
        return replace(self, latency_bars=bars, label=f"{self.label}+lat{bars}")

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        out = asdict(self)
        out["costs"] = self.costs.to_dict()
        out["execution_delay_bars"] = self.execution_delay
        out["admits_lookahead"] = self.admits_lookahead
        return out

    @classmethod
    def from_dict(cls, data: dict) -> ExecutionAssumptions:
        payload = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        payload["costs"] = CostModel.from_dict(data.get("costs", {}))
        return cls(**payload)

    def summary(self) -> str:
        return (
            f"[{self.label}] fill={self.fill_timing} (+{self.latency_bars} bar latency) | "
            f"{self.costs.summary()}"
        )


def _futures_costs(
    *,
    per_contract: float,
    half_spread_bps: float,
    slippage_bps: float,
    slippage_vol: float,
    impact_k: float,
    max_participation: float,
) -> CostModel:
    return CostModel(
        commission=CommissionModel(per_contract=per_contract, exchange_fee=per_contract * 0.5),
        spread=SpreadModel(half_spread_bps=half_spread_bps),
        slippage=SlippageModel(bps=slippage_bps, volatility_multiple=slippage_vol),
        impact=ImpactModel(coefficient=impact_k, max_participation=max_participation),
    )


OPTIMISTIC = ExecutionAssumptions(
    costs=_futures_costs(
        per_contract=0.50, half_spread_bps=0.5, slippage_bps=0.0,
        slippage_vol=0.0, impact_k=0.0, max_participation=0.20,
    ),
    fill_timing="next_open",
    latency_bars=0,
    allow_partial_fills=False,
    label="optimistic",
    notes="Best case: tight spread, no slippage, no impact. Bounds the upside; never a verdict basis.",
)

BASE = ExecutionAssumptions(
    costs=_futures_costs(
        per_contract=1.50, half_spread_bps=2.0, slippage_bps=1.0,
        slippage_vol=0.05, impact_k=0.10, max_participation=0.10,
    ),
    fill_timing="next_open",
    latency_bars=0,
    allow_partial_fills=True,
    label="base",
    notes="Realistic liquid-futures costs for a small book.",
)

CONSERVATIVE = ExecutionAssumptions(
    costs=_futures_costs(
        per_contract=2.50, half_spread_bps=4.0, slippage_bps=2.0,
        slippage_vol=0.15, impact_k=0.30, max_participation=0.05,
    ),
    fill_timing="next_open",
    latency_bars=1,
    allow_partial_fills=True,
    label="conservative",
    notes="Wider spreads, doubled slippage, one bar of latency. Survival here means a margin of safety.",
)

STRESSED = ExecutionAssumptions(
    costs=_futures_costs(
        per_contract=5.00, half_spread_bps=8.0, slippage_bps=5.0,
        slippage_vol=0.40, impact_k=0.80, max_participation=0.02,
    ),
    fill_timing="next_open",
    latency_bars=2,
    allow_partial_fills=True,
    label="stressed",
    notes="Crisis conditions. Most strategies fail here; which ones fail, and how, is the information.",
)

FRICTIONLESS = ExecutionAssumptions(
    costs=CostModel(),
    fill_timing="next_open",
    latency_bars=0,
    allow_partial_fills=False,
    label="frictionless",
    notes="No costs. Diagnostic only: isolates raw signal structure. Never a verdict basis.",
)

SCENARIOS: dict[str, ExecutionAssumptions] = {
    "frictionless": FRICTIONLESS,
    "optimistic": OPTIMISTIC,
    "base": BASE,
    "conservative": CONSERVATIVE,
    "stressed": STRESSED,
}

STRESS_LADDER: tuple[str, ...] = ("optimistic", "base", "conservative", "stressed")
"""The ordered ladder used by the execution stress test. A real edge decays
monotonically along it; an edge that *improves* under worse assumptions is an
artefact, almost always of an accounting sign error."""


def get_scenario(name: str) -> ExecutionAssumptions:
    key = str(name).lower()
    if key not in SCENARIOS:
        raise ConfigError(f"unknown execution scenario {name!r}; known: {sorted(SCENARIOS)}")
    return SCENARIOS[key]


def build_assumptions(spec: str | dict | ExecutionAssumptions | None) -> ExecutionAssumptions:
    """Build assumptions from config: a scenario name, or an explicit dict.

    A dict may name a scenario to start from and override individual
    fields::

        {"scenario": "base", "latency_bars": 2}
    """
    if spec is None:
        return BASE
    if isinstance(spec, ExecutionAssumptions):
        return spec
    if isinstance(spec, str):
        return get_scenario(spec)
    params = dict(spec)
    base = get_scenario(params.pop("scenario")) if "scenario" in params else BASE
    if "costs" in params:
        params["costs"] = CostModel.from_dict(params["costs"])
    known = {k: v for k, v in params.items() if k in ExecutionAssumptions.__dataclass_fields__}
    unknown = set(params) - set(known)
    if unknown:
        raise ConfigError(f"unknown execution assumption field(s): {sorted(unknown)}")
    known.setdefault("label", f"{base.label}+override" if known else base.label)
    return replace(base, **known)
