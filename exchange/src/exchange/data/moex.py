"""MOEX AlgoPack files on disk.

A thin reader for one vendor's directory layout, deliberately kept apart
from the rest of the data engine:

    <root>/<ticker>/<timeframe>/<year>.csv
    columns: ticker,secid,begin,end,open,high,low,close,volume,value

Three properties of this source that the reader must handle rather than
hide:

* **Timestamps are Moscow time** as MOEX returns them, and are localised
  explicitly. Guessing UTC would shift every bar by three hours and move
  every session boundary.
* **``value`` is zero for futures.** MOEX does not populate turnover on
  derivatives, so the market-impact model must use ``volume`` in
  contracts. The column is dropped rather than passed on as a field that
  looks like turnover and is not.
* **The series is already spliced by nearest expiry and not adjusted.**
  ``secid`` names the contract behind each bar, which is what makes the
  rolls recoverable; :func:`exchange.data.prespliced.from_prespliced` turns
  that into an auditable continuous series.

This module needs no credentials and no network: it reads files the user
already has.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from exchange.core.errors import DataError
from exchange.core.logging import get_logger
from exchange.data.continuous import Adjustment, ContinuousSeries
from exchange.data.dataset import Dataset, dataset_from_continuous
from exchange.data.prespliced import from_prespliced

logger = get_logger(__name__)

MOEX_TZ = "Europe/Moscow"
TIMEFRAMES = {"1m": "1m", "5m": "5m", "60m": "1h"}
#: AlgoPack ticker -> contract root used for labelling and specifications.
ROOTS = {"ng": "NG", "br": "BR"}
#: MOEX contract terms: NG is 100 MMBtu quoted to 0.001 USD, BR is 10 barrels
#: quoted to 0.01 USD. These are the FORTS contracts, not the NYMEX/ICE ones.
CONTRACTS = {
    "ng": {"multiplier": 100.0, "tick_size": 0.001, "allow_fractional": False},
    "br": {"multiplier": 10.0, "tick_size": 0.01, "allow_fractional": False},
}


def load_moex_algopack(
    directory: str | Path,
    ticker: str,
    timeframe: str = "60m",
    *,
    years: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Read one ticker and timeframe into a single frame.

    Returns the raw rows with ``timestamp`` (UTC) and ``secid``; nothing is
    validated or spliced here.
    """
    folder = Path(directory) / ticker / timeframe
    if not folder.is_dir():
        raise DataError(f"{folder} does not exist")
    files = sorted(folder.glob("*.csv"))
    if years is not None:
        wanted = {str(y) for y in years}
        files = [f for f in files if f.stem in wanted]
    if not files:
        raise DataError(f"no CSV files in {folder}")

    frames = [pd.read_csv(f, parse_dates=["begin"]) for f in files]
    frame = pd.concat(frames, ignore_index=True)
    missing = {"begin", "secid", "open", "high", "low", "close", "volume"} - set(frame.columns)
    if missing:
        raise DataError(f"{folder}: missing column(s) {sorted(missing)}")

    frame = frame.rename(columns={"begin": "timestamp"})
    frame["timestamp"] = (
        pd.DatetimeIndex(frame["timestamp"])
        .tz_localize(MOEX_TZ, ambiguous="NaT", nonexistent="NaT")
        .tz_convert("UTC")
    )
    before = len(frame)
    frame = frame[frame["timestamp"].notna()]
    if len(frame) < before:
        logger.warning(
            "%s/%s: dropped %d row(s) whose Moscow timestamp is ambiguous or does not exist",
            ticker, timeframe, before - len(frame),
        )

    frame = frame.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    # 'value' is zero for every futures row; 'end' duplicates the bar span.
    frame = frame[["timestamp", "open", "high", "low", "close", "volume", "secid"]]
    logger.info(
        "%s/%s: %d bars, %s..%s, %d contract(s)",
        ticker, timeframe, len(frame),
        frame["timestamp"].iloc[0], frame["timestamp"].iloc[-1], frame["secid"].nunique(),
    )
    return frame.reset_index(drop=True)


def moex_continuous(
    directory: str | Path,
    ticker: str,
    timeframe: str = "60m",
    *,
    adjustment: Adjustment = "none",
    years: Sequence[int] | None = None,
) -> ContinuousSeries:
    """Load one AlgoPack series as an auditable continuous series."""
    frame = load_moex_algopack(directory, ticker, timeframe, years=years)
    root = ROOTS.get(ticker.lower(), ticker.upper())
    return from_prespliced(
        frame,
        root=root,
        contract_column="secid",
        adjustment=adjustment,
        meta={"vendor": "moex_algopack", "ticker": ticker, "timeframe": timeframe},
    )


def moex_dataset(
    directory: str | Path,
    ticker: str,
    timeframe: str = "60m",
    *,
    adjustment: Adjustment = "none",
    years: Sequence[int] | None = None,
    strict: bool = False,
) -> Dataset:
    """Load, splice-audit and quality-gate one AlgoPack series.

    ``strict`` defaults to ``False`` here on purpose: a real intraday
    futures series carries holiday gaps and exceptional sessions that are
    warnings, not defects, and refusing to load it would make the honest
    path the inconvenient one. The report still records every finding.
    """
    series = moex_continuous(
        directory, ticker, timeframe, adjustment=adjustment, years=years
    )
    dataset = dataset_from_continuous(series, strict=strict)
    label = TIMEFRAMES.get(timeframe, timeframe)
    dataset.meta.update({"vendor": "moex_algopack", "ticker": ticker, "timeframe": label})
    if ticker.lower() in CONTRACTS:
        dataset.meta["instrument"] = dict(CONTRACTS[ticker.lower()])
    dataset.provenance.update(
        {"source": f"moex_algopack:{ticker}/{timeframe}", "timezone": MOEX_TZ}
    )
    return dataset
