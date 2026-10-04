"""Continuous futures construction.

Splicing a ladder of dated contracts into one series is the single most
common source of *manufactured* alpha in commodity research. Two mistakes
account for almost all of it:

1. **The roll gap is left in the return series.** Front-month NG can trade
   3.20 while the next month trades 3.55; if the series jumps from one to
   the other, the backtest books a +11% "return" nobody could earn. In
   contango this bleeds a systematic drift into every directional strategy
   — which is exactly the shape of a spurious trend-following edge. Measured
   on the synthetic contango ladder in this package: roughly **+11% of
   manufactured drift over 14 months**, larger than most edges people look
   for.
2. **The roll decision uses future information.** "Roll to whichever
   contract ends up having more volume" is a peek. Every rule here decides
   using data available strictly *before* the bar it takes effect on.

Design
------
Construction is two independent steps, so each can be tested alone:

``RollRule.active_series``
    A causal mapping ``timestamp -> active contract``. Activity-based rolls
    take effect on the bar *after* the signal that triggered them.

``_splice``
    Turns that mapping plus the per-contract bars into one series and
    removes the roll gaps according to the chosen adjustment.

Every series carries :class:`RollProvenance` recording which contracts were
used, when each roll happened, which overlap the gap was measured on, and
what adjustment was applied.

Choosing an adjustment
----------------------
``none``
    Real, tradeable price levels. Returns across a roll boundary are *not*
    tradeable and are marked ``is_roll`` — a return model must drop them.
``backward`` (Panama / difference)
    Historic prices shifted so returns are continuous. **Absolute levels in
    the past are fictitious** and the series can go negative over a long
    contango history. Use for absolute-move (ATR-style) work.
``ratio``
    Historic prices scaled so *percentage* returns are continuous. Levels
    are fictitious but always positive. The default for return-based
    research.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from exchange.core.errors import ConfigError, DataError
from exchange.core.logging import get_logger
from exchange.data.contracts import ContractRegistry, ContractSpec, parse_contract_symbol
from exchange.data.schema import PRICE_COLUMNS, assert_canonical

logger = get_logger(__name__)

Adjustment = Literal["none", "backward", "ratio"]
ADJUSTMENTS: tuple[str, ...] = ("none", "backward", "ratio")


# ==========================================================================
# Roll rules
# ==========================================================================


@dataclass(frozen=True)
class RollRule:
    """Base class: decide the active contract for each bar, causally."""

    name: str = "roll"

    def active_series(
        self,
        panel: Mapping[str, pd.DataFrame],
        order: Sequence[str],
        specs: Mapping[str, ContractSpec],
    ) -> pd.Series:
        """Return ``timestamp -> contract symbol`` over the union index."""
        raise NotImplementedError

    def describe(self) -> dict:
        return {"rule": self.name}


@dataclass(frozen=True)
class CalendarRoll(RollRule):
    """Roll a fixed number of days before the contract's reference date.

    The reference date is the first notice date when known (a physically
    delivered contract must be exited before it), else last trade date, else
    expiration. Purely calendar-driven, so it is trivially causal and fully
    reproducible — the reason it is the default.

    ``business_days=True`` counts business days instead of calendar days.
    """

    days_before: int = 5
    business_days: bool = True
    name: str = "calendar"

    def __post_init__(self) -> None:
        if self.days_before < 0:
            raise ConfigError("days_before must be non-negative")

    def _roll_out_date(self, spec: ContractSpec) -> date:
        ref = spec.roll_reference_date
        if ref is None:
            raise ConfigError(
                f"{spec.symbol}: calendar roll needs a first notice / last trade / expiration date"
            )
        ts = pd.Timestamp(ref)
        offset = (
            pd.tseries.offsets.BDay(self.days_before)
            if self.business_days
            else pd.Timedelta(days=self.days_before)
        )
        return (ts - offset).date()

    def active_series(self, panel, order, specs) -> pd.Series:
        index = _union_index(panel, order)
        active = pd.Series(index=index, dtype=object)
        bar_dates = np.array(index.tz_convert("UTC").date)
        for symbol in order:
            cutoff = self._roll_out_date(specs[symbol])
            unassigned = active.isna().to_numpy()
            eligible = unassigned & (bar_dates < cutoff) & index.isin(panel[symbol].index)
            active.loc[eligible] = symbol
        return active.ffill()

    def describe(self) -> dict:
        unit = "business days" if self.business_days else "calendar days"
        return {"rule": "calendar", "days_before": self.days_before, "unit": unit}


@dataclass(frozen=True)
class ActivityRoll(RollRule):
    """Roll when the deferred contract becomes more active than the front.

    ``column`` is ``volume`` or ``open_interest``. Both series are smoothed
    over ``lookback`` bars; the roll fires once the deferred contract has led
    for ``confirm`` consecutive bars, and — this is the causal part — takes
    effect on the *following* bar. A rule that switched on the same bar it
    observed would be using that bar's own volume to decide a trade priced
    within it.

    Once rolled, the rule never rolls back: contract activity is noisy around
    the crossover and a rule that oscillates books round-trip costs that no
    real desk would pay.
    """

    column: str = "volume"
    lookback: int = 5
    confirm: int = 2
    name: str = "activity"

    def __post_init__(self) -> None:
        if self.column not in {"volume", "open_interest"}:
            raise ConfigError(
                f"activity roll column must be volume or open_interest, got {self.column!r}"
            )
        if self.lookback < 1 or self.confirm < 1:
            raise ConfigError("lookback and confirm must be >= 1")

    def active_series(self, panel, order, specs) -> pd.Series:
        index = _union_index(panel, order)
        smoothed: dict[str, pd.Series] = {}
        for symbol in order:
            bars = panel[symbol]
            if self.column not in bars.columns:
                raise DataError(f"{symbol}: activity roll needs a {self.column!r} column")
            series = bars[self.column].reindex(index)
            smoothed[symbol] = series.rolling(self.lookback, min_periods=1).mean()

        quoted = {symbol: index.isin(panel[symbol].index) for symbol in order}

        active = np.empty(len(index), dtype=object)
        cursor = 0                     # index into `order` of the current front
        streak = 0
        current = order[0]
        for i in range(len(index)):
            # A contract that has stopped quoting must be left, whatever the
            # activity comparison says: holding an expired contract is not a
            # position anybody could have. Expiry is known at the start of
            # the bar, so acting on it immediately is causal.
            while (
                cursor + 1 < len(order)
                and not quoted[current][i]
                and quoted[order[cursor + 1]][i]
            ):
                cursor += 1
                current = order[cursor]
                streak = 0

            active[i] = current
            if cursor + 1 >= len(order):
                continue
            nxt = order[cursor + 1]
            front_activity = smoothed[current].iat[i]
            next_activity = smoothed[nxt].iat[i]
            leads = (
                np.isfinite(front_activity)
                and np.isfinite(next_activity)
                and next_activity > front_activity
            )
            streak = streak + 1 if leads else 0
            if streak >= self.confirm:
                # Effective from the NEXT bar: this bar's own activity is not
                # knowable while this bar is still trading.
                cursor += 1
                current = nxt
                streak = 0
        return pd.Series(active, index=index, dtype=object)

    def describe(self) -> dict:
        return {
            "rule": "activity",
            "column": self.column,
            "lookback": self.lookback,
            "confirm": self.confirm,
            "effective": "next bar",
        }


def volume_roll(lookback: int = 5, confirm: int = 2) -> ActivityRoll:
    """Roll on traded volume."""
    return ActivityRoll(column="volume", lookback=lookback, confirm=confirm)


def open_interest_roll(lookback: int = 5, confirm: int = 2) -> ActivityRoll:
    """Roll on open interest."""
    return ActivityRoll(column="open_interest", lookback=lookback, confirm=confirm)


def n_days_before_expiration(days: int = 5, *, business_days: bool = True) -> CalendarRoll:
    """Roll ``days`` before the contract's reference date."""
    return CalendarRoll(days_before=days, business_days=business_days)


ROLL_RULES: dict[str, type[RollRule]] = {"calendar": CalendarRoll, "activity": ActivityRoll}


def build_roll_rule(spec: str | Mapping | RollRule) -> RollRule:
    """Build a roll rule from a config value.

    Accepts ``"calendar"``, ``"volume"``, ``"open_interest"``, or a mapping
    such as ``{"rule": "calendar", "days_before": 8}``.
    """
    if isinstance(spec, RollRule):
        return spec
    if isinstance(spec, str):
        spec = {"rule": spec}
    params = dict(spec)
    rule = str(params.pop("rule", "calendar")).lower()
    if rule in {"volume", "open_interest"}:
        return ActivityRoll(column=rule, **params)
    if rule == "activity":
        return ActivityRoll(**params)
    if rule in {"calendar", "n_days_before_expiration"}:
        params.pop("unit", None)
        return CalendarRoll(**params)
    raise ConfigError(f"unknown roll rule {rule!r}; known: calendar, volume, open_interest")


# ==========================================================================
# Provenance
# ==========================================================================


@dataclass(frozen=True)
class RollEvent:
    """One splice point."""

    timestamp: pd.Timestamp
    from_contract: str
    to_contract: str
    from_price: float          # old contract's close on the last bar it was active
    to_price: float            # new contract's close on that same bar (the overlap)
    gap: float                 # to_price - from_price
    ratio: float               # to_price / from_price
    overlap_available: bool    # False when both contracts were not quoted together

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "from": self.from_contract,
            "to": self.to_contract,
            "from_price": self.from_price,
            "to_price": self.to_price,
            "gap": self.gap,
            "ratio": self.ratio,
            "overlap_available": self.overlap_available,
        }


@dataclass(frozen=True)
class Segment:
    """A stretch of the continuous series sourced from one contract."""

    contract: str
    start: pd.Timestamp
    end: pd.Timestamp
    n_bars: int

    def to_dict(self) -> dict:
        return {
            "contract": self.contract,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "n_bars": self.n_bars,
        }


@dataclass(frozen=True)
class RollProvenance:
    """The full recipe that produced a continuous series.

    Two continuous series are only comparable if their provenance matches;
    the falsification engine rebuilds a series under a *different* roll rule
    precisely to check that an effect is not an artefact of this one.
    """

    root: str
    adjustment: str
    rule: dict
    segments: tuple[Segment, ...]
    rolls: tuple[RollEvent, ...]
    contracts_used: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    dates_estimated: bool = False

    def to_dict(self) -> dict:
        return {
            "root": self.root,
            "adjustment": self.adjustment,
            "rule": self.rule,
            "contracts_used": list(self.contracts_used),
            "n_rolls": len(self.rolls),
            "segments": [s.to_dict() for s in self.segments],
            "rolls": [r.to_dict() for r in self.rolls],
            "warnings": list(self.warnings),
            "dates_estimated": self.dates_estimated,
        }

    def summary(self) -> str:
        return (
            f"{self.root} continuous | {self.adjustment} adjustment | {self.rule} | "
            f"{len(self.contracts_used)} contracts, {len(self.rolls)} rolls"
        )


@dataclass
class ContinuousSeries:
    """A spliced futures series plus everything needed to audit it."""

    bars: pd.DataFrame           # canonical OHLCV + `contract`, `is_roll`, `adj_factor`
    provenance: RollProvenance
    meta: dict = field(default_factory=dict)

    @property
    def root(self) -> str:
        return self.provenance.root

    @property
    def is_roll(self) -> pd.Series:
        return self.bars["is_roll"].astype(bool)

    def tradeable_returns(self, column: str = "close") -> pd.Series:
        """Simple returns with roll-boundary bars removed when unadjusted.

        For an adjusted series the gap is already gone, so every bar is
        tradeable and nothing is dropped. For an unadjusted series the
        boundary return is the spread between two different instruments and
        is set to ``NaN`` — never to zero, which would silently pretend a
        flat bar happened.
        """
        prices = self.bars[column]
        rets = prices.pct_change()
        if self.provenance.adjustment == "none":
            rets = rets.mask(self.is_roll)
        return rets.rename(f"{self.root}_ret")

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"ContinuousSeries({self.root}, {len(self.bars)} bars, "
            f"{self.provenance.adjustment}, {len(self.provenance.rolls)} rolls)"
        )


# ==========================================================================
# Construction
# ==========================================================================


def _union_index(panel: Mapping[str, pd.DataFrame], order: Sequence[str]) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex([], tz="UTC")
    for symbol in order:
        index = index.union(panel[symbol].index)
    return index


def _sort_contracts(symbols) -> list[str]:
    """Order contract symbols by delivery month."""
    keyed = []
    for symbol in symbols:
        _root, month = parse_contract_symbol(symbol)
        keyed.append(((month.year, month.month), symbol))
    return [s for _, s in sorted(keyed)]


def build_continuous(
    panel: Mapping[str, pd.DataFrame],
    *,
    root: str | None = None,
    roll: str | Mapping | RollRule = "calendar",
    adjustment: Adjustment = "ratio",
    registry: ContractRegistry | None = None,
    specs: Mapping[str, ContractSpec] | None = None,
) -> ContinuousSeries:
    """Splice dated contracts into one continuous series.

    Parameters
    ----------
    panel:
        ``{contract symbol: canonical bars}``. Symbols must share a root.
    root:
        Overrides the root parsed from the symbols.
    roll:
        A :class:`RollRule`, its name, or a mapping (see
        :func:`build_roll_rule`).
    adjustment:
        ``"none"``, ``"backward"`` or ``"ratio"``.
    registry / specs:
        Where contract metadata comes from. ``specs`` wins when both are
        given; otherwise the registry derives specs from the built-in root
        defaults and the result is flagged ``dates_estimated``.
    """
    if adjustment not in ADJUSTMENTS:
        raise ConfigError(f"unknown adjustment {adjustment!r}; expected one of {ADJUSTMENTS}")
    if not panel:
        raise ConfigError("panel is empty; nothing to splice")

    order = _sort_contracts(panel)
    for symbol in order:
        assert_canonical(panel[symbol], name=symbol)

    roots = {parse_contract_symbol(s)[0] for s in order}
    if len(roots) > 1 and root is None:
        raise ConfigError(f"panel mixes roots {sorted(roots)}; splice one root at a time")
    resolved_root = (root or next(iter(roots))).upper()

    registry = registry or ContractRegistry()
    resolved_specs = dict(specs or {})
    dates_estimated = False
    for symbol in order:
        if symbol in resolved_specs:
            continue
        spec = registry.get(symbol)
        if spec.roll_reference_date is None:
            from exchange.data.contracts import estimate_last_trade_date

            spec = spec.with_dates(last_trade_date=estimate_last_trade_date(spec))
            dates_estimated = True
        elif spec.meta.get("dates_estimated"):
            dates_estimated = True
        resolved_specs[symbol] = spec

    rule = build_roll_rule(roll)
    active = rule.active_series(panel, order, resolved_specs).dropna()
    if active.empty:
        raise DataError(f"{resolved_root}: roll rule selected no contract on any bar")

    raw, rolls, warnings = _splice(panel, active, adjustment)

    provenance = RollProvenance(
        root=resolved_root,
        adjustment=adjustment,
        rule=rule.describe(),
        segments=tuple(_segments(raw["contract"])),
        rolls=tuple(rolls),
        contracts_used=tuple(dict.fromkeys(raw["contract"])),
        warnings=tuple(warnings),
        dates_estimated=dates_estimated,
    )
    return ContinuousSeries(bars=raw, provenance=provenance)


def _splice(
    panel: Mapping[str, pd.DataFrame],
    active: pd.Series,
    adjustment: str,
) -> tuple[pd.DataFrame, list[RollEvent], list[str]]:
    """Assemble the series and remove roll gaps."""
    index = active.index
    contracts = active.to_numpy()

    # 1. Pull each bar from its active contract. A bar whose active contract
    #    is not quoted there is dropped rather than filled: fabricating a
    #    price is exactly how a phantom return gets created.
    rows = []
    keep = np.zeros(len(index), dtype=bool)
    for i, (ts, symbol) in enumerate(zip(index, contracts, strict=True)):
        bars = panel[symbol]
        if ts not in bars.index:
            continue
        rows.append(bars.loc[ts])
        keep[i] = True
    if not rows:
        raise DataError("no bar could be sourced from its active contract")

    frame = pd.DataFrame(rows)
    frame.index = index[keep]
    frame.index.name = "timestamp"
    contracts = contracts[keep]
    frame["contract"] = contracts

    changed = np.r_[False, contracts[1:] != contracts[:-1]]
    frame["is_roll"] = changed

    # 2. Measure each roll gap on the overlap bar (the last bar the old
    #    contract was active, where both contracts should be quoted).
    rolls: list[RollEvent] = []
    warnings: list[str] = []
    positions = np.flatnonzero(changed)
    for pos in positions:
        ts_prev = frame.index[pos - 1]
        old, new = contracts[pos - 1], contracts[pos]
        from_price = _price_at(panel[old], ts_prev)
        to_price = _price_at(panel[new], ts_prev)
        overlap = np.isfinite(from_price) and np.isfinite(to_price) and from_price > 0
        if not overlap:
            # No overlap quote: fall back to comparing the two contracts on
            # their own adjacent bars. This is a weaker splice and is
            # recorded as such rather than hidden.
            from_price = float(frame["close"].iat[pos - 1])
            to_price = float(frame["close"].iat[pos])
            warnings.append(
                f"{old}->{new} at {ts_prev.isoformat()}: no overlapping quote; "
                "gap measured across adjacent bars (weaker splice)"
            )
        rolls.append(
            RollEvent(
                timestamp=frame.index[pos],
                from_contract=str(old),
                to_contract=str(new),
                from_price=float(from_price),
                to_price=float(to_price),
                gap=float(to_price - from_price),
                ratio=float(to_price / from_price) if from_price else float("nan"),
                overlap_available=bool(overlap),
            )
        )

    # 3. Apply the adjustment backwards from the most recent roll, so the
    #    latest segment keeps its true, tradeable price levels.
    factor = np.ones(len(frame))
    offset = np.zeros(len(frame))
    if adjustment != "none":
        cum_ratio, cum_gap = 1.0, 0.0
        for pos in positions[::-1]:
            event = next(r for r in rolls if r.timestamp == frame.index[pos])
            if adjustment == "ratio":
                if not np.isfinite(event.ratio) or event.ratio <= 0:
                    warnings.append(
                        f"{event.from_contract}->{event.to_contract}: non-positive ratio, "
                        "roll left unadjusted"
                    )
                else:
                    cum_ratio *= event.ratio
            else:
                cum_gap += event.gap
            factor[:pos] = cum_ratio
            offset[:pos] = cum_gap

    price_cols = [c for c in PRICE_COLUMNS if c in frame.columns]
    for col in price_cols:
        if adjustment == "ratio":
            frame[col] = frame[col].to_numpy(float) * factor
        elif adjustment == "backward":
            frame[col] = frame[col].to_numpy(float) + offset

    if adjustment == "backward" and (frame[price_cols].to_numpy(float) <= 0).any():
        warnings.append(
            "backward adjustment produced non-positive historic prices; "
            "use ratio adjustment for return-based research"
        )

    frame["adj_factor"] = factor if adjustment == "ratio" else 1.0
    frame["adj_offset"] = offset if adjustment == "backward" else 0.0

    ordered = [c for c in (*PRICE_COLUMNS, "volume", "open_interest") if c in frame.columns]
    tail = ["contract", "is_roll", "adj_factor", "adj_offset"]
    extras = [c for c in frame.columns if c not in ordered + tail]
    return frame[ordered + extras + tail], rolls, warnings


def _price_at(bars: pd.DataFrame, ts: pd.Timestamp) -> float:
    if ts in bars.index:
        value = bars["close"].loc[ts]
        return float(value) if np.isfinite(value) else float("nan")
    return float("nan")


def _segments(contracts: pd.Series) -> list[Segment]:
    out: list[Segment] = []
    values = contracts.to_numpy()
    if len(values) == 0:
        return out
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            out.append(
                Segment(
                    contract=str(values[start]),
                    start=contracts.index[start],
                    end=contracts.index[i - 1],
                    n_bars=i - start,
                )
            )
            start = i
    return out


def roll_gap_diagnostics(series: ContinuousSeries) -> pd.DataFrame:
    """Per-roll table used to audit a splice.

    ``gap_bps`` is the size of the raw gap relative to the price. If a
    strategy's edge is of the same order as the mean absolute gap and the
    series is unadjusted, the "edge" is the roll.
    """
    rows = []
    for event in series.provenance.rolls:
        rows.append(
            {
                "timestamp": event.timestamp,
                "from": event.from_contract,
                "to": event.to_contract,
                "from_price": event.from_price,
                "to_price": event.to_price,
                "gap": event.gap,
                "gap_bps": 1e4 * event.gap / event.from_price if event.from_price else np.nan,
                "overlap_available": event.overlap_available,
            }
        )
    return pd.DataFrame(rows)
