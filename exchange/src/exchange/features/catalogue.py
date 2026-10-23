"""Fundamental feature catalogue for natural gas and crude oil.

What this module is, and what it deliberately is not
-----------------------------------------------------
It is a **specification** of the fundamental series a commodity research
programme needs, with the one property that decides whether they can be
used at all: *when each number actually became available*. It is not a data
vendor client. No API key, no vendor SDK, no hard-coded endpoint appears
here or anywhere else in the package. A provider is an interface
(:class:`~exchange.features.providers.FeatureProvider`); whoever has a data
subscription implements it in a few lines, and nothing in the research code
changes.

That separation is not architectural fastidiousness. A research programme
whose validity depends on one vendor's revision policy cannot check its own
results, and a framework that hard-codes a vendor cannot be tested without
one.

The two fields that matter more than the data
----------------------------------------------
``publication_lag``
    The gap between the **end of the period a number describes** and the
    **earliest moment a trader could have acted on it**. EIA's weekly
    natural-gas storage report covers the week ending Friday and is
    released the following Thursday at 10:30 ET: a lag of six days, not
    zero. Indexing that number at the Friday it describes and using it on
    that Friday's close is a lookahead of nearly a week, and it is the most
    common error in published commodity research. Every lag here is
    rounded *up*, because an over-conservative lag costs a little
    statistical power while an under-conservative one invents an edge.

``revision_policy``
    ``"first"`` means the research only ever sees the first print. Most of
    these series are revised — EIA's monthly production (form 914) is
    revised for months, and reserves estimates for years. Using the revised
    value is a lookahead even when the timestamp is right, because the
    revised number did not exist at that timestamp. ``"first"`` is the
    default for every revised series here, and ``"latest"`` appears only
    where the series is genuinely not revised.

Each spec also records the ``mechanism`` — why the series should move the
price — and a ``caveat`` naming the way it is usually misused. A feature
without a mechanism is a column, and the funnel will not accept a
hypothesis built on one.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from exchange.core.errors import ConfigError

DAY = pd.Timedelta(days=1)

MARKETS: tuple[str, ...] = ("natural_gas", "crude_oil", "both")
CATEGORIES: tuple[str, ...] = (
    "inventory",
    "supply",
    "demand",
    "weather",
    "positioning",
    "curve",
    "refining",
    "macro",
)


@dataclass(frozen=True)
class FeatureSpec:
    """One fundamental series, specified so it cannot be misused silently."""

    key: str
    name: str
    market: str
    category: str
    frequency: str                     # daily / weekly / monthly / quarterly
    publication_lag: pd.Timedelta      # end of period -> first availability
    unit: str
    source: str
    mechanism: str                     # why it should move the price
    revision_policy: str = "first"
    caveat: str = ""
    expected_sign: int = 0             # +1 bullish when rising, -1 bearish, 0 ambiguous
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        problems = []
        if self.market not in MARKETS:
            problems.append(f"market must be one of {MARKETS}")
        if self.category not in CATEGORIES:
            problems.append(f"category must be one of {CATEGORIES}")
        if self.revision_policy not in {"first", "latest"}:
            problems.append("revision_policy must be 'first' or 'latest'")
        if self.publication_lag < pd.Timedelta(0):
            problems.append("publication_lag cannot be negative — that is a lookahead by definition")
        if len(self.mechanism.strip()) < 40:
            problems.append(
                "mechanism must say why this series moves the price in at least 40 characters"
            )
        if self.expected_sign not in (-1, 0, 1):
            problems.append("expected_sign must be -1, 0 or +1")
        if problems:
            raise ConfigError(f"feature {self.key!r} is misspecified:\n  - " + "\n  - ".join(problems))

    @property
    def lag_days(self) -> float:
        return self.publication_lag / DAY

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "name": self.name,
            "market": self.market,
            "category": self.category,
            "frequency": self.frequency,
            "publication_lag_days": self.lag_days,
            "unit": self.unit,
            "source": self.source,
            "revision_policy": self.revision_policy,
            "expected_sign": self.expected_sign,
            "mechanism": self.mechanism,
            "caveat": self.caveat,
            "tags": list(self.tags),
        }


# ==========================================================================
# Natural gas
# ==========================================================================

NG_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        key="ng_storage",
        name="Lower-48 working gas in underground storage",
        market="natural_gas", category="inventory", frequency="weekly",
        publication_lag=pd.Timedelta(days=6), unit="Bcf",
        source="EIA Weekly Natural Gas Storage Report (WNGSR)",
        mechanism=(
            "Storage is the only buffer between production and demand. The level relative to "
            "normal determines how much of a weather shock has to be absorbed by price rather "
            "than by inventory, which is why the price response to a given cold snap is far "
            "larger from a low storage base."
        ),
        caveat=(
            "Covers the week ending Friday, published the following Thursday 10:30 ET. Indexing "
            "it at the Friday it describes leaks almost a week. Also revised, so the first print "
            "is the only honest value."
        ),
        expected_sign=-1, tags=("eia", "revised"),
    ),
    FeatureSpec(
        key="ng_storage_change",
        name="Weekly storage injection or withdrawal",
        market="natural_gas", category="inventory", frequency="weekly",
        publication_lag=pd.Timedelta(days=6), unit="Bcf/week",
        source="EIA WNGSR (first difference)",
        mechanism=(
            "The flow, not the level, is what the market trades on release: the surprise versus "
            "consensus is the balance of supply and demand over the week, measured directly."
        ),
        caveat="Only the surprise versus expectations is news; the level change itself is largely seasonal.",
        expected_sign=-1, tags=("eia", "revised"),
    ),
    FeatureSpec(
        key="ng_storage_vs_5y",
        name="Storage surplus or deficit versus the five-year average",
        market="natural_gas", category="inventory", frequency="weekly",
        publication_lag=pd.Timedelta(days=6), unit="Bcf",
        source="EIA WNGSR (derived)",
        mechanism=(
            "Removes the seasonal cycle, which dominates the raw level, leaving the part of "
            "inventory that reflects the current supply-demand balance rather than the calendar."
        ),
        caveat=(
            "The five-year window must be computed from data available at the time, not from the "
            "full sample — a full-sample average is a lookahead."
        ),
        expected_sign=-1, tags=("eia", "derived"),
    ),
    FeatureSpec(
        key="ng_storage_utilisation",
        name="Storage as a share of demonstrated peak capacity",
        market="natural_gas", category="inventory", frequency="weekly",
        publication_lag=pd.Timedelta(days=6), unit="fraction",
        source="EIA WNGSR and capacity estimates",
        mechanism=(
            "Near full capacity the system cannot absorb further injections, so excess supply "
            "must clear through price; this produces the sharply asymmetric autumn price "
            "behaviour that a level-based measure misses entirely."
        ),
        caveat="Capacity estimates are themselves revised and regional constraints bind before national ones.",
        expected_sign=-1, tags=("eia", "derived"),
    ),
    FeatureSpec(
        key="ng_dry_production",
        name="Lower-48 dry gas production",
        market="natural_gas", category="supply", frequency="daily",
        publication_lag=pd.Timedelta(days=2), unit="Bcf/d",
        source="Pipeline scrapes (daily estimate) or EIA-914 (monthly actual)",
        mechanism=(
            "Supply is the slow side of the balance; sustained production growth caps rallies by "
            "refilling storage even in a cold winter, which is why it dominates multi-month moves."
        ),
        caveat=(
            "Daily pipeline-derived estimates are revised heavily for days and are not comparable "
            "with the monthly EIA-914 actuals, which arrive about two months late."
        ),
        expected_sign=-1, tags=("pipeline", "revised"),
    ),
    FeatureSpec(
        key="ng_production_914",
        name="Monthly marketed gas production (EIA-914)",
        market="natural_gas", category="supply", frequency="monthly",
        publication_lag=pd.Timedelta(days=60), unit="Bcf/d",
        source="EIA Form 914 Monthly Crude Oil and Natural Gas Production",
        mechanism=(
            "The authoritative supply measure. Its purpose in research is to validate the daily "
            "estimates, whose biases are only visible against a measured actual."
        ),
        caveat="About two months late and revised afterwards; useless as a trading signal, essential as a benchmark.",
        expected_sign=-1, tags=("eia", "revised"),
    ),
    FeatureSpec(
        key="ng_lng_feedgas",
        name="LNG export terminal feedgas demand",
        market="natural_gas", category="demand", frequency="daily",
        publication_lag=pd.Timedelta(days=1), unit="Bcf/d",
        source="Pipeline nomination scrapes",
        mechanism=(
            "LNG feedgas converted the US market from an isolated system into one linked to "
            "European and Asian prices, so it is the channel through which a foreign cold winter "
            "or a terminal outage reaches Henry Hub."
        ),
        caveat="Nominations are intraday-revised and an unplanned outage appears in the data before it is explained.",
        expected_sign=1, tags=("pipeline",),
    ),
    FeatureSpec(
        key="ng_mexico_exports",
        name="Pipeline exports to Mexico",
        market="natural_gas", category="demand", frequency="daily",
        publication_lag=pd.Timedelta(days=1), unit="Bcf/d",
        source="Pipeline nomination scrapes",
        mechanism=(
            "A structural demand sink concentrated in the summer power season, which changes the "
            "shape of the seasonal balance rather than its level."
        ),
        caveat="Strongly seasonal; a level comparison across years without seasonal adjustment is misleading.",
        expected_sign=1, tags=("pipeline",),
    ),
    FeatureSpec(
        key="ng_power_burn",
        name="Gas consumption for electricity generation",
        market="natural_gas", category="demand", frequency="daily",
        publication_lag=pd.Timedelta(days=2), unit="Bcf/d",
        source="Pipeline scrapes and EIA electricity data",
        mechanism=(
            "The marginal, price-elastic demand: when gas is cheap it displaces coal in the "
            "dispatch stack and when expensive it is displaced, which is the mechanism that makes "
            "the gas price mean-reverting around the coal-switching range."
        ),
        caveat="The switching range itself moves with the coal price and the generation fleet, so it is not a constant.",
        expected_sign=1, tags=("pipeline", "eia"),
    ),
    FeatureSpec(
        key="ng_hdd",
        name="Population-weighted heating degree days",
        market="natural_gas", category="weather", frequency="daily",
        publication_lag=pd.Timedelta(days=1), unit="degree-days",
        source="NOAA / weather vendor, population-weighted",
        mechanism=(
            "Residential and commercial heating is the largest and least price-elastic component "
            "of winter demand, so realised temperature is close to an exogenous demand shock — "
            "one of the few genuinely causal variables available in any market."
        ),
        caveat=(
            "Realised degree days are not tradeable information: the market trades the forecast. "
            "Using realised HDD for the day it describes is a lookahead."
        ),
        expected_sign=1, tags=("noaa", "weather"),
    ),
    FeatureSpec(
        key="ng_cdd",
        name="Population-weighted cooling degree days",
        market="natural_gas", category="weather", frequency="daily",
        publication_lag=pd.Timedelta(days=1), unit="degree-days",
        source="NOAA / weather vendor, population-weighted",
        mechanism=(
            "Summer cooling demand reaches gas through the power sector, which is why the summer "
            "price is far more sensitive to the generation stack than the winter price is."
        ),
        caveat="Same realised-versus-forecast problem as heating degree days.",
        expected_sign=1, tags=("noaa", "weather"),
    ),
    FeatureSpec(
        key="ng_hdd_forecast_15d",
        name="15-day forecast heating degree days",
        market="natural_gas", category="weather", frequency="daily",
        publication_lag=pd.Timedelta(hours=6), unit="degree-days",
        source="Weather vendor ensemble forecast",
        mechanism=(
            "The market prices expected weather, so the tradeable variable is the forecast and, "
            "more precisely, the change in the forecast between model runs."
        ),
        caveat=(
            "Must be the forecast as issued, stored with its issue time. A vendor archive that "
            "silently overwrites old forecasts with later ones destroys the series for research."
        ),
        expected_sign=1, tags=("weather", "forecast"),
    ),
    FeatureSpec(
        key="ng_hdd_forecast_revision",
        name="Change in the 15-day HDD forecast between runs",
        market="natural_gas", category="weather", frequency="daily",
        publication_lag=pd.Timedelta(hours=6), unit="degree-days",
        source="Weather vendor ensemble forecast (derived)",
        mechanism=(
            "A price moves on news, and the news in weather is the revision rather than the "
            "level: an unchanged cold forecast is already in the price."
        ),
        caveat="Requires a true point-in-time forecast archive; it cannot be reconstructed after the fact.",
        expected_sign=1, tags=("weather", "forecast", "derived"),
    ),
    FeatureSpec(
        key="ng_rig_count",
        name="Gas-directed rig count",
        market="natural_gas", category="supply", frequency="weekly",
        publication_lag=pd.Timedelta(days=1), unit="rigs",
        source="Baker Hughes (Friday release)",
        mechanism=(
            "Drilling responds to price with a lag of months and production follows with a "
            "further lag, making the rig count a slow supply-side anticipation variable rather "
            "than a short-horizon signal."
        ),
        caveat="Rigs per unit of production fall continuously with drilling productivity, so levels are not comparable across years.",
        expected_sign=-1, tags=("baker_hughes",),
    ),
    FeatureSpec(
        key="ng_managed_money_net",
        name="Managed-money net position in Henry Hub futures",
        market="natural_gas", category="positioning", frequency="weekly",
        publication_lag=pd.Timedelta(days=3), unit="contracts",
        source="CFTC Commitments of Traders (Tuesday data, Friday 15:30 ET release)",
        mechanism=(
            "Crowded speculative positioning is fuel for a reversal: a market where managed money "
            "is at a positioning extreme needs only a small fundamental surprise to force "
            "liquidation, which is why positioning conditions the size of a move rather than its "
            "direction."
        ),
        caveat=(
            "Reports Tuesday's state three days later, so it is stale on arrival. Treating it as "
            "current positioning is a lookahead of three days plus the staleness."
        ),
        expected_sign=0, tags=("cftc", "positioning"),
    ),
    FeatureSpec(
        key="ng_curve_slope",
        name="Front-to-second-month spread",
        market="natural_gas", category="curve", frequency="daily",
        publication_lag=pd.Timedelta(0), unit="USD/MMBtu",
        source="Exchange settlement prices",
        mechanism=(
            "The curve prices the cost of storage: backwardation means the market pays for "
            "immediate delivery, which is the signature of a physically tight system."
        ),
        caveat=(
            "Observable at the close with no lag, but the front-month spread jumps at every roll, "
            "and that jump is a contract change rather than a market move."
        ),
        revision_policy="latest",   # exchange settlements are final, never restated
        expected_sign=1, tags=("curve",),
    ),
    FeatureSpec(
        key="ng_march_april_spread",
        name="March-April calendar spread (the 'widow-maker')",
        market="natural_gas", category="curve", frequency="daily",
        publication_lag=pd.Timedelta(0), unit="USD/MMBtu",
        source="Exchange settlement prices",
        mechanism=(
            "The spread between the last winter and first summer contract is the market's direct "
            "price of end-of-winter storage scarcity, so it isolates the inventory risk premium "
            "from the level of the price."
        ),
        caveat="Extremely fat-tailed and illiquid in the back years; a Sharpe computed on it understates risk badly.",
        revision_policy="latest",   # exchange settlements are final, never restated
        expected_sign=1, tags=("curve", "seasonal"),
    ),
    FeatureSpec(
        key="ng_cash_basis",
        name="Henry Hub cash price minus prompt futures",
        market="natural_gas", category="curve", frequency="daily",
        publication_lag=pd.Timedelta(days=1), unit="USD/MMBtu",
        source="Price reporting agency and exchange settlements",
        mechanism=(
            "The physical market clears before the futures contract expires, so a persistent cash "
            "premium is direct evidence of physical tightness that has not yet reached the screen."
        ),
        caveat="Index prices are survey-based, published with a lag, and occasionally restated.",
        expected_sign=1, tags=("cash", "curve"),
    ),
)


# ==========================================================================
# Crude oil
# ==========================================================================

CRUDE_FEATURES: tuple[FeatureSpec, ...] = (
    FeatureSpec(
        key="cl_crude_stocks",
        name="US commercial crude oil inventories",
        market="crude_oil", category="inventory", frequency="weekly",
        publication_lag=pd.Timedelta(days=5), unit="thousand barrels",
        source="EIA Weekly Petroleum Status Report (Wednesday 10:30 ET)",
        mechanism=(
            "Inventories are the visible residual of the global balance; a persistent build means "
            "supply exceeds demand at the current price, and the price has to fall far enough to "
            "clear it."
        ),
        caveat=(
            "Covers the week ending Friday, released the following Wednesday. Also revised, and "
            "the API's Tuesday estimate moves the price before the EIA number exists."
        ),
        expected_sign=-1, tags=("eia", "revised"),
    ),
    FeatureSpec(
        key="cl_cushing_stocks",
        name="Cushing, Oklahoma crude inventories",
        market="crude_oil", category="inventory", frequency="weekly",
        publication_lag=pd.Timedelta(days=5), unit="thousand barrels",
        source="EIA WPSR",
        mechanism=(
            "Cushing is the WTI contract's delivery point, so its tank level — not the national "
            "total — determines whether the contract can be physically settled, which is why "
            "operational minimums there have produced the contract's most extreme dislocations."
        ),
        caveat="Operational tank bottoms mean the usable range is far narrower than the reported level suggests.",
        expected_sign=-1, tags=("eia", "delivery_point"),
    ),
    FeatureSpec(
        key="cl_stocks_vs_5y",
        name="Crude inventories versus the five-year average",
        market="crude_oil", category="inventory", frequency="weekly",
        publication_lag=pd.Timedelta(days=5), unit="thousand barrels",
        source="EIA WPSR (derived)",
        mechanism=(
            "Seasonally adjusts the level so that refinery maintenance cycles do not masquerade "
            "as changes in the supply-demand balance."
        ),
        caveat="The five-year baseline must use only data available at each point in time.",
        expected_sign=-1, tags=("eia", "derived"),
    ),
    FeatureSpec(
        key="cl_spr",
        name="Strategic Petroleum Reserve level",
        market="crude_oil", category="inventory", frequency="weekly",
        publication_lag=pd.Timedelta(days=5), unit="thousand barrels",
        source="EIA WPSR",
        mechanism=(
            "A policy-controlled supply source: a release adds physical barrels and removes a tail "
            "risk simultaneously, so it moves both the level and the curve shape."
        ),
        caveat="Driven by policy decisions, not market forces; a statistical relationship fitted across a release period will not generalise.",
        expected_sign=-1, tags=("eia", "policy"),
    ),
    FeatureSpec(
        key="cl_refinery_utilisation",
        name="Refinery capacity utilisation",
        market="crude_oil", category="refining", frequency="weekly",
        publication_lag=pd.Timedelta(days=5), unit="fraction",
        source="EIA WPSR",
        mechanism=(
            "Refineries are the only real demand for crude oil; utilisation determines how fast "
            "crude inventories are drawn, which is why maintenance season mechanically builds "
            "crude stocks without any change in end-user demand."
        ),
        caveat="Highly seasonal, and an unplanned outage is indistinguishable from planned maintenance in the aggregate.",
        expected_sign=1, tags=("eia",),
    ),
    FeatureSpec(
        key="cl_crude_runs",
        name="Gross crude oil inputs to refineries",
        market="crude_oil", category="refining", frequency="weekly",
        publication_lag=pd.Timedelta(days=5), unit="thousand barrels/day",
        source="EIA WPSR",
        mechanism=(
            "The physical flow of crude into demand. Together with imports, exports and "
            "production it closes the balance, so a residual that does not close points to a data "
            "problem rather than a market effect."
        ),
        caveat="The EIA's own adjustment term absorbs the unexplained residual and is sometimes large.",
        expected_sign=1, tags=("eia",),
    ),
    FeatureSpec(
        key="cl_us_production",
        name="US crude oil production (weekly estimate)",
        market="crude_oil", category="supply", frequency="weekly",
        publication_lag=pd.Timedelta(days=5), unit="thousand barrels/day",
        source="EIA WPSR weekly estimate",
        mechanism=(
            "US shale is the marginal global supply source with the shortest investment cycle, so "
            "its response to price is the main stabilising mechanism in the modern oil market."
        ),
        caveat=(
            "The weekly figure is a model-based estimate, not a measurement, and is revised "
            "substantially against the monthly series."
        ),
        expected_sign=-1, tags=("eia", "revised", "estimate"),
    ),
    FeatureSpec(
        key="cl_imports",
        name="US crude oil imports",
        market="crude_oil", category="supply", frequency="weekly",
        publication_lag=pd.Timedelta(days=5), unit="thousand barrels/day",
        source="EIA WPSR",
        mechanism=(
            "Waterborne arrivals are lumpy, so a single week's imports often explains an "
            "inventory surprise that would otherwise be read as a demand signal."
        ),
        caveat="Timing noise from vessel scheduling dominates at weekly frequency.",
        expected_sign=-1, tags=("eia",),
    ),
    FeatureSpec(
        key="cl_exports",
        name="US crude oil exports",
        market="crude_oil", category="demand", frequency="weekly",
        publication_lag=pd.Timedelta(days=5), unit="thousand barrels/day",
        source="EIA WPSR",
        mechanism=(
            "Exports link the US balance to the global one; the arbitrage opens and closes with "
            "the WTI-Brent spread and freight, which is why export volumes and that spread must "
            "be modelled together."
        ),
        caveat="Reported with the same lag as the rest of the WPSR and revised by the monthly trade data.",
        expected_sign=1, tags=("eia",),
    ),
    FeatureSpec(
        key="cl_gasoline_stocks",
        name="US gasoline inventories",
        market="crude_oil", category="inventory", frequency="weekly",
        publication_lag=pd.Timedelta(days=5), unit="thousand barrels",
        source="EIA WPSR",
        mechanism=(
            "Product inventories reveal whether weak crude demand is a refining problem or an "
            "end-user demand problem, which the crude number alone cannot distinguish."
        ),
        caveat="Seasonal specification changes make levels non-comparable across the summer-winter transition.",
        expected_sign=-1, tags=("eia", "products"),
    ),
    FeatureSpec(
        key="cl_distillate_stocks",
        name="US distillate inventories",
        market="crude_oil", category="inventory", frequency="weekly",
        publication_lag=pd.Timedelta(days=5), unit="thousand barrels",
        source="EIA WPSR",
        mechanism=(
            "Distillate is the industrial and freight fuel, so its inventory is the most direct "
            "petroleum read on industrial activity, and it competes with natural gas in heating."
        ),
        caveat="The gas-oil substitution channel means a distillate signal can be a gas signal in disguise.",
        expected_sign=-1, tags=("eia", "products"),
    ),
    FeatureSpec(
        key="cl_days_of_supply",
        name="Crude inventories expressed in days of refinery demand",
        market="crude_oil", category="inventory", frequency="weekly",
        publication_lag=pd.Timedelta(days=5), unit="days",
        source="EIA WPSR (derived)",
        mechanism=(
            "Normalising inventory by the demand it serves makes the measure comparable across "
            "decades of capacity growth, which a raw barrel count is not."
        ),
        caveat="Noisy when refinery runs swing, because the denominator moves as fast as the numerator.",
        expected_sign=-1, tags=("eia", "derived"),
    ),
    FeatureSpec(
        key="cl_rig_count",
        name="Oil-directed rig count",
        market="crude_oil", category="supply", frequency="weekly",
        publication_lag=pd.Timedelta(days=1), unit="rigs",
        source="Baker Hughes (Friday release)",
        mechanism=(
            "The leading indicator of shale supply: rigs respond to price within weeks and "
            "production follows within months, which is the transmission channel for the "
            "price-supply feedback loop."
        ),
        caveat="Drilled-but-uncompleted wells break the rig-to-production relationship for quarters at a time.",
        expected_sign=-1, tags=("baker_hughes",),
    ),
    FeatureSpec(
        key="cl_managed_money_net",
        name="Managed-money net position in WTI futures",
        market="crude_oil", category="positioning", frequency="weekly",
        publication_lag=pd.Timedelta(days=3), unit="contracts",
        source="CFTC Commitments of Traders",
        mechanism=(
            "Positioning extremes precede forced liquidation: when the speculative long is "
            "historically large, the marginal buyer is absent and a small surprise produces an "
            "outsized move."
        ),
        caveat="Three days stale on release, and the trader categories have been redefined over the series' history.",
        expected_sign=0, tags=("cftc", "positioning"),
    ),
    FeatureSpec(
        key="cl_wti_brent_spread",
        name="WTI minus Brent",
        market="crude_oil", category="curve", frequency="daily",
        publication_lag=pd.Timedelta(0), unit="USD/bbl",
        source="Exchange settlement prices",
        mechanism=(
            "The spread is the arbitrage between the US inland balance and the waterborne market; "
            "it is bounded by the cost of moving a barrel to the coast and widens when that "
            "logistics chain binds."
        ),
        caveat="Both legs must use matched contract months, or the spread measures the roll instead of the arbitrage.",
        revision_policy="latest",   # exchange settlements are final, never restated
        expected_sign=0, tags=("curve", "spread"),
    ),
    FeatureSpec(
        key="cl_curve_slope",
        name="Front-to-second-month WTI spread",
        market="crude_oil", category="curve", frequency="daily",
        publication_lag=pd.Timedelta(0), unit="USD/bbl",
        source="Exchange settlement prices",
        mechanism=(
            "Backwardation prices physical scarcity and pays a positive roll yield; contango pays "
            "for storage. The sign of the slope is therefore the market's own statement about "
            "whether inventory is scarce."
        ),
        caveat="The prompt spread is the most roll-contaminated series in the dataset; measure it on fixed contracts.",
        revision_policy="latest",   # exchange settlements are final, never restated
        expected_sign=1, tags=("curve",),
    ),
    FeatureSpec(
        key="cl_crack_spread",
        name="3-2-1 crack spread",
        market="crude_oil", category="refining", frequency="daily",
        publication_lag=pd.Timedelta(0), unit="USD/bbl",
        source="Exchange settlement prices (derived)",
        mechanism=(
            "The refining margin is the price signal that sets crude demand: a wide crack pulls "
            "crude into refineries and a negative one shuts them, so it transmits product demand "
            "back into the crude balance."
        ),
        caveat="The 3-2-1 ratio is a convention, not a refinery; regional margins diverge from it materially.",
        revision_policy="latest",   # exchange settlements are final, never restated
        expected_sign=1, tags=("curve", "derived"),
    ),
)


ALL_FEATURES: tuple[FeatureSpec, ...] = NG_FEATURES + CRUDE_FEATURES
CATALOGUE: dict[str, FeatureSpec] = {spec.key: spec for spec in ALL_FEATURES}

if len(CATALOGUE) != len(ALL_FEATURES):  # pragma: no cover - guards a typo at import time
    raise ConfigError("duplicate feature key in the catalogue")


def feature_spec(key: str) -> FeatureSpec:
    if key not in CATALOGUE:
        raise ConfigError(f"unknown feature {key!r}; {len(CATALOGUE)} known, see catalogue()")
    return CATALOGUE[key]


def features_for(market: str) -> tuple[FeatureSpec, ...]:
    """Every spec for a market, including the ones marked ``both``."""
    if market not in MARKETS:
        raise ConfigError(f"unknown market {market!r}; known: {MARKETS}")
    return tuple(s for s in ALL_FEATURES if s.market in {market, "both"})


def catalogue(
    market: str | None = None,
    category: str | None = None,
    keys: Iterable[str] | None = None,
) -> pd.DataFrame:
    """The catalogue as a frame, for ``exchange features list``."""
    specs = ALL_FEATURES if market is None else features_for(market)
    if category is not None:
        specs = tuple(s for s in specs if s.category == category)
    if keys is not None:
        wanted = set(keys)
        specs = tuple(s for s in specs if s.key in wanted)
    if not specs:
        return pd.DataFrame()
    return pd.DataFrame([s.to_dict() for s in specs]).set_index("key")


@dataclass(frozen=True)
class DerivedFeature:
    """A feature computed from other features, after they are made available.

    The transform runs on a frame of **already point-in-time aligned**
    inputs, so a derived feature cannot be less lagged than its inputs. That
    ordering is the whole point: computing a five-year average from the full
    sample and then lagging the result is a lookahead that no amount of
    shifting afterwards removes.
    """

    key: str
    inputs: tuple[str, ...]
    transform: object                 # Callable[[pd.DataFrame], pd.Series]
    description: str
    expected_sign: int = 0
    tags: tuple[str, ...] = field(default_factory=tuple)

    def compute(self, aligned: pd.DataFrame) -> pd.Series:
        missing = [k for k in self.inputs if k not in aligned.columns]
        if missing:
            raise ConfigError(f"derived feature {self.key!r} needs missing input(s) {missing}")
        out = self.transform(aligned[list(self.inputs)])  # type: ignore[operator]
        if not isinstance(out, pd.Series):
            raise ConfigError(f"derived feature {self.key!r} must produce a Series")
        return out.rename(self.key)
