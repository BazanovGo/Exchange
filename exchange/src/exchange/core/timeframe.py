"""Timeframes and annualisation.

Annualisation is one of the most common silent errors in backtesting: an
hourly strategy annualised with ``252`` understates volatility by ~5.9x and
inflates the Sharpe ratio by the same factor. The audited project hardcoded
``freq="1D"`` in its backtest config while advertising intraday timeframes,
so an hourly Sharpe of 0.3 was reported as 1.8 with nothing in the output
revealing it. This module refuses to guess.

Two distinct quantities are kept apart:

``nominal_bars_per_year``
    What the calendar implies for a timeframe if the market never closed
    (e.g. 1h -> 8766). Useful only as a sanity bound.

``empirical_bars_per_year``
    Bars actually observed per year of wall-clock span. This is the correct
    annualisation factor for a return series sampled on trading bars,
    because it automatically absorbs sessions, weekends and holidays.

The rule used throughout the package: **annualise with the empirical count
derived from the index that produced the returns.**
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from exchange.core.errors import ConfigError, NotEnoughDataError

_SECONDS_PER_YEAR = 365.2425 * 24 * 3600

_UNIT_SECONDS = {
    "s": 1,
    "m": 60,
    "h": 3600,
    "d": 86_400,
    "w": 604_800,
}

_TIMEFRAME_RE = re.compile(r"^\s*(\d+)\s*([a-zA-Z]+)\s*$")

# Spellings accepted for a unit, normalised to a key of _UNIT_SECONDS.
_UNIT_ALIASES = {
    "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s",
    "m": "m", "min": "m", "mins": "m", "minute": "m", "minutes": "m", "t": "m",
    "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h",
    "d": "d", "day": "d", "days": "d",
    "w": "w", "wk": "w", "week": "w", "weeks": "w",
    "mo": "mo", "month": "mo", "months": "mo",
    "y": "y", "yr": "y", "year": "y", "years": "y",
}

_PANDAS_FREQ = {"s": "s", "m": "min", "h": "h", "d": "D", "w": "W", "mo": "MS", "y": "YS"}


@dataclass(frozen=True, order=True)
class Timeframe:
    """A bar sampling interval, e.g. ``1h``, ``15m``, ``1d``."""

    n: int
    unit: str  # one of s, m, h, d, w, mo, y

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ConfigError(f"timeframe multiplier must be positive, got {self.n}")
        if self.unit not in _PANDAS_FREQ:
            raise ConfigError(f"unknown timeframe unit {self.unit!r}")

    @classmethod
    def parse(cls, spec: str | Timeframe) -> Timeframe:
        """Parse ``'1h'``, ``'15 min'``, ``'1D'`` … into a :class:`Timeframe`."""
        if isinstance(spec, Timeframe):
            return spec
        match = _TIMEFRAME_RE.match(str(spec))
        if match is None:
            # Allow a bare unit, e.g. "d" meaning "1d".
            key = _UNIT_ALIASES.get(str(spec).strip().lower())
            if key is None:
                raise ConfigError(f"cannot parse timeframe {spec!r}")
            return cls(1, key)
        n, raw_unit = int(match.group(1)), match.group(2).lower()
        key = _UNIT_ALIASES.get(raw_unit)
        if key is None:
            raise ConfigError(f"unknown timeframe unit {raw_unit!r} in {spec!r}")
        return cls(n, key)

    @property
    def label(self) -> str:
        return f"{self.n}{self.unit}"

    @property
    def pandas_freq(self) -> str:
        return f"{self.n}{_PANDAS_FREQ[self.unit]}"

    @property
    def is_intraday(self) -> bool:
        return self.unit in {"s", "m", "h"}

    @property
    def seconds(self) -> float:
        """Nominal length in seconds (calendar approximation for mo/y)."""
        if self.unit in _UNIT_SECONDS:
            return self.n * _UNIT_SECONDS[self.unit]
        if self.unit == "mo":
            return self.n * _SECONDS_PER_YEAR / 12
        return self.n * _SECONDS_PER_YEAR

    @property
    def timedelta(self) -> pd.Timedelta:
        return pd.Timedelta(seconds=self.seconds)

    @property
    def nominal_bars_per_year(self) -> float:
        """Bars per year if the market never closed. A bound, not a factor."""
        return _SECONDS_PER_YEAR / self.seconds

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.label


def infer_timeframe(index: pd.DatetimeIndex) -> Timeframe:
    """Infer the bar interval from the modal spacing of ``index``.

    The mode (not the mean) is used so session gaps and weekends do not
    stretch the estimate.
    """
    index = _as_datetime_index(index)
    if len(index) < 3:
        raise NotEnoughDataError("need at least 3 timestamps to infer a timeframe")
    deltas = np.diff(_epoch_nanos(index)) / 1e9  # seconds
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        raise NotEnoughDataError("index has no strictly increasing timestamps")
    step = float(np.median(_modal_cluster(deltas)))

    for unit in ("w", "d", "h", "m", "s"):
        size = _UNIT_SECONDS[unit]
        if step >= size and abs(step / size - round(step / size)) < 1e-6:
            return Timeframe(round(step / size), unit)
    # Sub-second or irregular: fall back to seconds, rounded up.
    return Timeframe(max(1, round(step)), "s")


def _modal_cluster(deltas: np.ndarray) -> np.ndarray:
    """Values equal to the most common spacing (within float tolerance)."""
    rounded = np.round(deltas, 6)
    values, counts = np.unique(rounded, return_counts=True)
    modal = values[int(np.argmax(counts))]
    return deltas[np.isclose(rounded, modal)]


def _epoch_nanos(index: pd.DatetimeIndex) -> np.ndarray:
    """Integer nanoseconds since the epoch, portable across pandas versions.

    ``DatetimeIndex.asi8`` was removed in pandas 3, and ``.view('int64')``
    before it, so neither can be relied on across the supported range.
    """
    return index.to_numpy(dtype="datetime64[ns]").astype("int64")


def empirical_bars_per_year(index: pd.DatetimeIndex) -> float:
    """Bars observed per year of wall-clock span covered by ``index``.

    This is the annualisation factor to use for return series: it absorbs
    trading sessions, weekends and holidays without any calendar model.

    The span is measured from the first to the last timestamp and extended
    by one bar, so a series of ``n`` bars spanning ``n-1`` intervals is not
    biased upward on short samples.
    """
    index = _as_datetime_index(index)
    if len(index) < 3:
        raise NotEnoughDataError("need at least 3 timestamps to estimate bars per year")
    span_seconds = (index[-1] - index[0]).total_seconds()
    if span_seconds <= 0:
        raise NotEnoughDataError("index spans zero time")
    step = span_seconds / (len(index) - 1)
    span_seconds += step  # count the last bar's own interval
    return len(index) * _SECONDS_PER_YEAR / span_seconds


def annualisation_factor(index: pd.DatetimeIndex, *, override: float | None = None) -> float:
    """Periods per year for annualising returns sampled on ``index``.

    ``override`` short-circuits inference; use it only when the caller
    genuinely knows better (e.g. a synthetic series with no real calendar).
    """
    if override is not None:
        if override <= 0:
            raise ConfigError(f"annualisation factor must be positive, got {override}")
        return float(override)
    return empirical_bars_per_year(index)


def _as_datetime_index(index) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise ConfigError(f"expected a DatetimeIndex, got {type(index).__name__}")
    return index
