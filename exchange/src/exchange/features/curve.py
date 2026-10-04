"""Term-structure features built from dated contracts.

Why this cannot be computed from the continuous series
------------------------------------------------------
The curve slope is the difference between two *different* contracts at the
same moment. A back-adjusted continuous series has had that difference
removed from its history by construction — that is what the adjustment
does — so a slope computed from it measures the splice, not the market. The
same is true of the roll yield, which is the single most important carry
signal in commodities: computing it from adjusted prices produces a series
whose sign flips with the adjustment method.

So this module works only on the panel of dated contracts, and it rebuilds
the tenor ladder **row by row**: for each timestamp it takes the contracts
actually quoted and still alive, sorts them by expiration, and labels them
M1, M2, M3 at that timestamp. A fixed column assignment (treating the first
column as M1 forever) silently turns into nonsense the moment the first
contract expires — the resulting "contango" flag then measures which
contracts happen to be in the file.

Every feature here is observable at the bar's own close with no publication
lag, and none of them uses a contract after its roll reference date.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from exchange.core.errors import DataError
from exchange.data.contracts import ContractRegistry, ContractSpec, parse_contract_symbol

CURVE_COLUMNS: tuple[str, ...] = (
    "m1",
    "m2",
    "m3",
    "slope",
    "slope_pct",
    "roll_yield",
    "curvature",
    "contango",
    "days_to_m1_expiry",
    "n_tenors",
)


def curve_features(
    panel: Mapping[str, pd.DataFrame],
    *,
    specs: Mapping[str, ContractSpec] | None = None,
    registry: ContractRegistry | None = None,
    price_column: str = "close",
    max_tenors: int = 3,
) -> pd.DataFrame:
    """Build M1/M2/M3 term-structure features from a contract panel.

    Parameters
    ----------
    panel:
        ``{contract symbol: canonical bars}``.
    specs, registry:
        Where expiration dates come from. Without either, the dates are
        derived from the symbol's month code, which is an approximation and
        is marked as such in the result's ``attrs``.
    max_tenors:
        How many tenors to label. Three is enough for a slope and a
        curvature; more adds illiquid back months whose settlements are
        often stale.

    Returns
    -------
    pandas.DataFrame
        Indexed by timestamp, with the columns in :data:`CURVE_COLUMNS`.
        ``roll_yield`` is annualised and signed so that **positive means
        backwardation** (a long position earns carry).
    """
    if not panel:
        raise DataError("curve_features needs at least one contract")

    resolved = _resolve_specs(panel, specs=specs, registry=registry)
    frames = {}
    for symbol, bars in panel.items():
        if price_column not in bars.columns:
            raise DataError(f"{symbol}: no {price_column!r} column")
        frames[symbol] = bars[price_column]
    prices = pd.DataFrame(frames).sort_index()

    # Expiry per contract, as a timestamp comparable with the bar index.
    expiry = {}
    for symbol in prices.columns:
        spec = resolved[symbol]
        reference = spec.roll_reference_date or spec.expiration_date or spec.last_trade_date
        if reference is None:
            raise DataError(f"{symbol}: no expiration information, cannot order the ladder")
        expiry[symbol] = pd.Timestamp(reference, tz="UTC")

    rows = []
    for timestamp, quotes in prices.iterrows():
        alive = [
            (expiry[symbol], symbol, float(value))
            for symbol, value in quotes.items()
            if np.isfinite(value) and expiry[symbol] >= timestamp
        ]
        alive.sort()
        rows.append(_row(timestamp, alive, max_tenors=max_tenors))

    out = pd.DataFrame(rows).set_index("timestamp")
    out.index.name = prices.index.name or "timestamp"
    out.attrs["dates_estimated"] = any(
        bool(resolved[s].meta.get("dates_estimated")) for s in prices.columns
    )
    out.attrs["contracts"] = list(prices.columns)
    return out[list(CURVE_COLUMNS)]


def _row(timestamp, alive, *, max_tenors: int) -> dict:
    row: dict = {"timestamp": timestamp, "n_tenors": len(alive)}
    for i in range(max_tenors):
        label = f"m{i + 1}"
        row[label] = alive[i][2] if i < len(alive) else np.nan
    m1, m2, m3 = row.get("m1"), row.get("m2"), row.get("m3")

    if m1 is not None and m2 is not None and np.isfinite(m1) and np.isfinite(m2) and m1 > 0:
        row["slope"] = m2 - m1
        row["slope_pct"] = (m2 - m1) / m1
        days = max((alive[1][0] - alive[0][0]).days, 1)
        # Positive = backwardation = positive carry for a long position.
        row["roll_yield"] = float(np.log(m1 / m2) * 365.0 / days) if m2 > 0 else np.nan
        row["contango"] = float(m2 > m1)
    else:
        row["slope"] = row["slope_pct"] = row["roll_yield"] = row["contango"] = np.nan

    if all(v is not None and np.isfinite(v) for v in (m1, m2, m3)):
        row["curvature"] = m1 - 2.0 * m2 + m3
    else:
        row["curvature"] = np.nan

    row["days_to_m1_expiry"] = (alive[0][0] - timestamp).days if alive else np.nan
    return row


def _resolve_specs(
    panel: Mapping[str, pd.DataFrame],
    *,
    specs: Mapping[str, ContractSpec] | None,
    registry: ContractRegistry | None,
) -> dict[str, ContractSpec]:
    if specs is not None:
        missing = [s for s in panel if s not in specs]
        if missing:
            raise DataError(f"no contract specification for {missing}")
        return dict(specs)
    registry = registry or ContractRegistry()
    out = {}
    for symbol in panel:
        parse_contract_symbol(symbol)      # fail early on a malformed symbol
        out[symbol] = registry.get(symbol, derive=True)
    return out


def seasonal_spread(
    panel: Mapping[str, pd.DataFrame],
    near: str,
    far: str,
    *,
    price_column: str = "close",
) -> pd.Series:
    """The spread between two named contracts, e.g. NGH26 minus NGJ26.

    Calendar spreads between *fixed* contracts are the cleanest fundamental
    series in a futures market: both legs expire, so there is no adjustment
    to choose and no splice to contaminate the history. The natural-gas
    March-April spread is the canonical example — it prices end-of-winter
    storage scarcity directly.
    """
    for symbol in (near, far):
        if symbol not in panel:
            raise DataError(f"{symbol} is not in the panel")
    a = panel[near][price_column]
    b = panel[far][price_column]
    spread = (a - b).dropna()
    return spread.rename(f"{near}-{far}")
