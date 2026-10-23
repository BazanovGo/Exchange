"""Continuous series that arrived already spliced.

Some vendors do the splice for you and hand over one series plus a column
naming the contract each bar came from — MOEX AlgoPack's ``secid`` is the
case this module was written for. That is a different problem from
:func:`~exchange.data.continuous.build_continuous`, which starts from a
panel of dated contracts and chooses the roll itself, and it is a *worse*
starting point in two specific ways:

**No overlap.** With a panel, a roll gap can be measured on the bar where
both contracts were quoted, which isolates the spread between the two
instruments. Here the old contract's last bar and the new contract's first
bar are at *different times*, so the measured transition mixes the roll
spread with one bar of genuine market move. Every roll gap this module
reports therefore carries that caveat, and
:attr:`RollEvent.overlap_available` is ``False`` for all of them.

**No choice.** The vendor's roll rule is whatever it was; you cannot ask
what the series would look like rolled three days earlier. The
``roll_methodology`` attack can still vary the *adjustment* — which is the
half of the question that matters most for trend-like results — but not the
roll timing.

Why this matters more than it sounds
------------------------------------
On the MOEX natural-gas hourly series (2020-2026, 80 rolls) the median
transition gap is **+384 basis points** and 95% of rolls exceed 100 bp. The
unadjusted series rises 63% over the period while the tradeable return —
the same series with roll bars excluded — falls 77%. The 140-point
difference is manufactured entirely by splicing a contango curve, and any
trend strategy backtested on the raw series is trading that artefact.

So the default here is ``adjustment="none"`` with roll bars flagged, which
makes :meth:`ContinuousSeries.tradeable_returns` drop them. Choosing an
adjustment is a research decision, and the attack exists to check that the
result does not depend on it.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from exchange.core.errors import DataError
from exchange.core.logging import get_logger
from exchange.data.continuous import (
    Adjustment,
    ContinuousSeries,
    RollEvent,
    RollProvenance,
    Segment,
)
from exchange.data.schema import normalise_bars

logger = get_logger(__name__)


def from_prespliced(
    frame: pd.DataFrame,
    *,
    root: str,
    contract_column: str = "contract",
    adjustment: Adjustment = "none",
    source_tz: str | None = None,
    meta: Mapping | None = None,
) -> ContinuousSeries:
    """Wrap a vendor-spliced series as an auditable :class:`ContinuousSeries`.

    Parameters
    ----------
    frame:
        Bars in any accepted layout, carrying a column that names the
        contract behind each bar.
    root:
        Contract root (``"NG"``, ``"BR"``) for labelling and provenance.
    contract_column:
        The column naming each bar's contract. Its *transitions* define the
        rolls; its values are kept in the result as ``contract``.
    adjustment:
        ``"none"`` keeps vendor prices and flags the roll bars so their
        returns are excluded. ``"ratio"`` and ``"backward"`` remove the
        gaps by rescaling or shifting history, which makes every bar's
        return tradeable at the cost of changing historical price levels.

    Returns
    -------
    ContinuousSeries
        With ``contract``, ``is_roll`` and ``adj_factor`` columns, and a
        provenance record naming every transition.
    """
    if contract_column not in frame.columns:
        raise DataError(
            f"{root}: no {contract_column!r} column. A pre-spliced series without a contract "
            "identifier cannot be audited: the roll points are unknowable, and the roll gaps "
            "would be indistinguishable from market moves."
        )

    contracts = frame[contract_column].astype(str)
    bars = normalise_bars(
        frame.drop(columns=[contract_column]), source_tz=source_tz, symbol=root
    )
    contracts = pd.Series(contracts.to_numpy(), index=bars.index, name="contract")
    if contracts.isna().any() or (contracts == "").any():
        raise DataError(f"{root}: the contract column has missing values")

    values = contracts.to_numpy()
    changed = np.r_[False, values[1:] != values[:-1]]
    positions = np.flatnonzero(changed)

    close = bars["close"].to_numpy(float)
    open_ = bars["open"].to_numpy(float)

    rolls: list[RollEvent] = []
    warnings: list[str] = []
    for pos in positions:
        from_price = float(close[pos - 1])      # old contract's last close
        to_price = float(open_[pos])            # new contract's first open
        if not (np.isfinite(from_price) and np.isfinite(to_price) and from_price > 0):
            warnings.append(
                f"roll at {bars.index[pos].isoformat()} has a non-finite transition price"
            )
            continue
        rolls.append(
            RollEvent(
                timestamp=bars.index[pos],
                from_contract=str(values[pos - 1]),
                to_contract=str(values[pos]),
                from_price=from_price,
                to_price=to_price,
                gap=to_price - from_price,
                ratio=to_price / from_price,
                # Always False: the two contracts were never quoted on the
                # same bar in a pre-spliced series, so this gap includes one
                # bar of market move.
                overlap_available=False,
            )
        )

    adjusted, factors = _adjust(bars, rolls, adjustment)
    adjusted["contract"] = values
    adjusted["is_roll"] = changed
    adjusted["adj_factor"] = factors

    segments = _segments(values, adjusted.index)
    provenance = RollProvenance(
        root=root,
        adjustment=adjustment,
        rule={
            "kind": "vendor_prespliced",
            "contract_column": contract_column,
            "note": (
                "The roll rule is the vendor's and cannot be varied; gaps are measured "
                "between adjacent bars of different contracts, so each includes one bar "
                "of market move in addition to the roll spread."
            ),
        },
        segments=tuple(segments),
        rolls=tuple(rolls),
        contracts_used=tuple(dict.fromkeys(str(v) for v in values)),
        warnings=tuple(warnings),
        dates_estimated=False,
    )
    logger.info(
        "%s: %d bars, %d contract(s), %d roll(s), adjustment=%s",
        root, len(adjusted), len(provenance.contracts_used), len(rolls), adjustment,
    )
    return ContinuousSeries(bars=adjusted, provenance=provenance, meta=dict(meta or {}))


def _adjust(
    bars: pd.DataFrame, rolls: list[RollEvent], adjustment: Adjustment
) -> tuple[pd.DataFrame, np.ndarray]:
    """Apply the adjustment backwards so the latest segment keeps true levels."""
    out = bars.copy()
    n = len(out)
    price_columns = ["open", "high", "low", "close"]

    if adjustment == "none":
        return out, np.ones(n)

    if adjustment not in {"ratio", "backward"}:
        raise DataError(f"unknown adjustment {adjustment!r}")

    index = out.index
    factors = np.ones(n) if adjustment == "ratio" else np.zeros(n)
    # Walk the rolls from the most recent backwards, accumulating the
    # correction applied to everything before each one. The newest segment is
    # untouched, so the last price in the series is the real last price.
    cumulative = 1.0 if adjustment == "ratio" else 0.0
    for event in sorted(rolls, key=lambda e: e.timestamp, reverse=True):
        position = int(index.get_indexer([event.timestamp])[0])
        if position <= 0:
            continue
        if adjustment == "ratio":
            cumulative *= event.ratio
            factors[:position] = cumulative
        else:
            cumulative += event.gap
            factors[:position] = cumulative

    if adjustment == "ratio":
        out[price_columns] = out[price_columns].to_numpy() * factors[:, None]
    else:
        out[price_columns] = out[price_columns].to_numpy() + factors[:, None]
        if (out[price_columns] <= 0).to_numpy().any():
            raise DataError(
                "backward adjustment produced a non-positive price; use 'ratio' on this series "
                "(additive adjustment cannot survive a long run of large roll gaps)"
            )
    return out, factors


def _segments(values: np.ndarray, index: pd.DatetimeIndex) -> list[Segment]:
    out: list[Segment] = []
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            out.append(
                Segment(
                    contract=str(values[start]),
                    start=index[start],
                    end=index[i - 1],
                    n_bars=i - start,
                )
            )
            start = i
    return out


def splice_artefact(series: ContinuousSeries, column: str = "close") -> dict:
    """How much total return the splice itself contributes.

    The difference between the raw series' cumulative return and the
    tradeable return (roll bars excluded). On a contango curve this is
    large, positive and entirely fictitious: it is the step up to the next,
    more expensive contract, repeated at every roll.
    """
    prices = series.bars[column]
    log_returns = np.log(prices).diff()
    raw = float(np.expm1(log_returns.sum()))
    tradeable_mask = ~series.is_roll
    tradeable = float(np.expm1(log_returns[tradeable_mask].sum()))
    gaps_bps = np.array([(e.ratio - 1.0) * 1e4 for e in series.provenance.rolls])
    return {
        "raw_total_return": raw,
        "tradeable_total_return": tradeable,
        "splice_artefact": raw - tradeable,
        "n_rolls": len(series.provenance.rolls),
        "median_gap_bps": float(np.median(gaps_bps)) if gaps_bps.size else float("nan"),
        "mean_gap_bps": float(np.mean(gaps_bps)) if gaps_bps.size else float("nan"),
        "share_above_100bps": (
            float(np.mean(np.abs(gaps_bps) > 100)) if gaps_bps.size else float("nan")
        ),
    }
