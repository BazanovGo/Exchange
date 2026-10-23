"""The research dataset: bars plus provenance plus a quality verdict.

This is the only object the rest of the pipeline accepts. Constructing one
forces a dataset through normalisation and validation, so nothing can reach
a backtest without a quality report attached to it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from exchange.core.errors import ConfigError, DataQualityError
from exchange.core.hashing import hash_dataframe, short
from exchange.core.logging import get_logger
from exchange.core.timeframe import Timeframe, empirical_bars_per_year, infer_timeframe
from exchange.data.continuous import ContinuousSeries
from exchange.data.providers import BarProvider
from exchange.data.quality import DataQualityReport, validate_bars
from exchange.data.schema import assert_canonical, normalise_bars

logger = get_logger(__name__)


@dataclass
class Dataset:
    """Validated bars for one instrument, with everything needed to cite it."""

    symbol: str
    bars: pd.DataFrame
    timeframe: Timeframe
    quality: DataQualityReport
    provenance: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_canonical(self.bars, name=self.symbol)

    # -- identity -------------------------------------------------------

    @property
    def data_hash(self) -> str:
        return self.quality.data_hash

    @property
    def start(self) -> pd.Timestamp:
        return self.bars.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.bars.index[-1]

    @property
    def n_bars(self) -> int:
        return len(self.bars)

    @property
    def bars_per_year(self) -> float:
        """Empirical annualisation factor for this dataset."""
        return empirical_bars_per_year(self.bars.index)

    # -- access ---------------------------------------------------------

    @property
    def close(self) -> pd.Series:
        return self.bars["close"]

    def returns(self, *, log: bool = False) -> pd.Series:
        """Bar-to-bar returns, with roll boundaries excluded when relevant."""
        close = self.bars["close"]
        rets = np.log(close).diff() if log else close.pct_change()
        if "is_roll" in self.bars.columns and self.provenance.get("adjustment") == "none":
            rets = rets.mask(self.bars["is_roll"].astype(bool))
        return rets.rename(f"{self.symbol}_ret")

    def slice(self, start=None, end=None) -> Dataset:
        """A sub-period, re-validated and re-hashed."""
        bars = self.bars.loc[start:end]
        if bars.empty:
            raise ConfigError(f"{self.symbol}: slice [{start}, {end}] is empty")
        return Dataset(
            symbol=self.symbol,
            bars=bars,
            timeframe=self.timeframe,
            quality=validate_bars(bars, symbol=self.symbol, timeframe=self.timeframe),
            provenance={**self.provenance, "sliced": [str(start), str(end)]},
            meta=dict(self.meta),
        )

    def describe(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe.label,
            "n_bars": self.n_bars,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "bars_per_year": round(self.bars_per_year, 2),
            "data_hash": self.data_hash,
            "quality": "OK" if self.quality.ok else "FAILED",
            "warnings": len(self.quality.warnings),
            "provenance": self.provenance,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Dataset({self.symbol}, {self.timeframe.label}, {self.n_bars} bars, "
            f"{self.start:%Y-%m-%d}..{self.end:%Y-%m-%d}, hash={short(self.data_hash)})"
        )


def make_dataset(
    frame: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str | Timeframe | None = None,
    source_tz: str | None = None,
    provenance: Mapping | None = None,
    strict: bool = True,
) -> Dataset:
    """Normalise, validate and wrap a raw frame.

    ``strict=True`` (the default) refuses to return a dataset with blocking
    quality errors. Turning it off is a deliberate, visible act; the report
    still records every failure.
    """
    bars = normalise_bars(frame, source_tz=source_tz, symbol=symbol)
    tf = Timeframe.parse(timeframe) if timeframe is not None else infer_timeframe(bars.index)
    report = validate_bars(bars, symbol=symbol, timeframe=tf)
    if strict:
        report.raise_for_errors()
    elif not report.ok:
        logger.warning("%s: loaded with %d blocking quality error(s)", symbol, len(report.errors))
    return Dataset(
        symbol=symbol,
        bars=bars,
        timeframe=tf,
        quality=report,
        provenance=dict(provenance or {}),
    )


def load_bars(
    provider: BarProvider,
    symbol: str,
    *,
    timeframe: str = "1d",
    start=None,
    end=None,
    source_tz: str | None = None,
    strict: bool = True,
    **kwargs,
) -> Dataset:
    """Fetch, normalise and validate one symbol in a single call."""
    raw = provider.fetch(symbol, timeframe=timeframe, start=start, end=end, **kwargs)
    tz = source_tz if source_tz is not None else getattr(provider, "source_tz", None)
    dataset = make_dataset(
        raw,
        symbol=symbol,
        timeframe=timeframe,
        source_tz=tz,
        strict=strict,
        provenance={
            "provider": getattr(provider, "name", type(provider).__name__),
            "requested_timeframe": timeframe,
            "requested_start": str(start) if start is not None else None,
            "requested_end": str(end) if end is not None else None,
            "source_tz": tz,
        },
    )
    if start is not None or end is not None:
        lo = pd.Timestamp(start, tz="UTC") if start is not None else None
        hi = pd.Timestamp(end, tz="UTC") if end is not None else None
        dataset = dataset.slice(lo, hi)
    return dataset


def dataset_from_continuous(series: ContinuousSeries, *, strict: bool = True) -> Dataset:
    """Wrap a spliced futures series as a research dataset."""
    bars = series.bars
    tf = infer_timeframe(bars.index)
    report = validate_bars(bars, symbol=series.root, timeframe=tf)
    if strict:
        report.raise_for_errors()
    return Dataset(
        symbol=f"{series.root}_continuous",
        bars=bars,
        timeframe=tf,
        quality=report,
        provenance={
            "continuous": series.provenance.to_dict(),
            "adjustment": series.provenance.adjustment,
        },
        meta={"root": series.root},
    )


def save_dataset(dataset: Dataset, path: str | Path) -> Path:
    """Persist bars to Parquet (or CSV) beside a JSON provenance sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in {".parquet", ".pq"}:
        dataset.bars.to_parquet(path)
    elif path.suffix == ".csv":
        dataset.bars.to_csv(path)
    else:
        raise ConfigError(f"unsupported dataset format {path.suffix!r}; use .parquet or .csv")
    sidecar = path.with_suffix(path.suffix + ".provenance.json")
    sidecar.write_text(json.dumps(dataset.describe(), indent=2, default=str))
    return path


def verify_dataset_hash(dataset: Dataset, expected: str) -> None:
    """Raise unless the dataset still hashes to ``expected``."""
    actual = hash_dataframe(dataset.bars)
    if actual != expected:
        raise DataQualityError(
            f"{dataset.symbol}: data hash mismatch — expected {short(expected)}, got {short(actual)}. "
            "The dataset changed since the experiment was recorded."
        )
