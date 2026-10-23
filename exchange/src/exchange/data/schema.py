"""Canonical bar schema and normalisation.

One schema for every bar series in the system:

* index: ``DatetimeIndex``, **timezone-aware, stored in UTC**, strictly
  increasing, unique, named ``timestamp``;
* columns: ``open, high, low, close, volume`` (lower case), float64;
* extra columns are preserved and carried through untouched.

Why UTC-aware rather than naive
-------------------------------
The audited project did ``index.tz_localize(None)``, discarding the zone.
Naive timestamps are the root of an entire family of silent bugs: a DST
transition duplicates or skips an hour, two sources disagree on whether
``09:30`` is exchange local or UTC, and a session filter written for one
assumption silently mis-slices the other. Storing UTC and carrying the
exchange timezone as *metadata* makes every conversion explicit and
reversible. Presentation-time conversion lives in :func:`to_exchange_time`.

Bar timestamp convention
------------------------
A bar's timestamp is the **open** of its interval (left-closed, right-open:
``[t, t + timeframe)``). This matters for lookahead control: the
information in a bar labelled ``t`` is only complete at ``t + timeframe``,
which is when a decision may first use it. See
:mod:`exchange.data.temporal`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from exchange.core.errors import DataError

OHLCV = ("open", "high", "low", "close", "volume")
PRICE_COLUMNS = ("open", "high", "low", "close")
INDEX_NAME = "timestamp"

# Case-insensitive aliases accepted from raw files and APIs.
_ALIASES: Mapping[str, str] = {
    "open": "open", "o": "open", "px_open": "open", "openprice": "open",
    "high": "high", "h": "high", "px_high": "high", "highprice": "high",
    "low": "low", "l": "low", "px_low": "low", "lowprice": "low",
    "close": "close", "c": "close", "px_last": "close", "last": "close",
    "closeprice": "close", "settle": "close", "settlement": "close",
    "volume": "volume", "v": "volume", "vol": "volume", "qty": "volume",
    "openinterest": "open_interest", "open_interest": "open_interest", "oi": "open_interest",
    "adjclose": "adj_close", "adj close": "adj_close", "adj_close": "adj_close",
}

_TIMESTAMP_ALIASES = (
    "timestamp", "datetime", "date", "time", "dt", "begin", "tradedate", "date_time", "ts",
)


def normalise_bars(
    frame: pd.DataFrame,
    *,
    source_tz: str | None = None,
    symbol: str = "",
    require_ohlc: bool = True,
) -> pd.DataFrame:
    """Coerce a raw frame to the canonical bar schema.

    Parameters
    ----------
    frame:
        Raw data with a datetime index or a datetime-like column.
    source_tz:
        Timezone of *naive* input timestamps (e.g. ``"America/New_York"``).
        Ignored when the input is already tz-aware. Passing ``None`` assumes
        UTC, which is exactly the ambiguity this schema exists to remove, so
        it is only allowed explicitly by the caller.
    symbol:
        Used in error messages only.
    require_ohlc:
        When ``True`` all four price columns must be present. Set ``False``
        for close-only series.

    Raises
    ------
    DataError
        If required columns are missing or the index cannot be parsed.
    """
    out = frame.copy()

    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]

    out = _promote_timestamp_index(out, symbol=symbol)
    out = _rename_columns(out)

    missing = [c for c in (PRICE_COLUMNS if require_ohlc else ("close",)) if c not in out.columns]
    if missing:
        raise DataError(
            f"{symbol or 'data'}: missing required column(s) {missing}; "
            f"got {sorted(map(str, frame.columns))}"
        )

    for col in (*OHLCV, "adj_close", "open_interest"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    if "volume" not in out.columns:
        out["volume"] = np.nan

    out.index = _to_utc(out.index, source_tz=source_tz, symbol=symbol)
    out.index.name = INDEX_NAME
    out = out.sort_index(kind="stable")

    ordered = [c for c in (*OHLCV, "adj_close", "open_interest") if c in out.columns]
    extras = [c for c in out.columns if c not in ordered]
    return out[ordered + extras]


def _rename_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Map arbitrary column spellings onto the canonical names."""
    renamed = {}
    for col in frame.columns:
        key = str(col).strip().lower().replace(" ", "_")
        renamed[col] = _ALIASES.get(key, _ALIASES.get(key.replace("_", " "), key))
    return frame.rename(columns=renamed)


def _promote_timestamp_index(frame: pd.DataFrame, *, symbol: str) -> pd.DataFrame:
    """Find the timestamps, or fail loudly.

    The failure mode this guards against is silent and severe: passing an
    integer index to :func:`pandas.to_datetime` succeeds and interprets the
    values as **nanoseconds since the epoch**, so a plain ``RangeIndex``
    becomes a series of timestamps a few nanoseconds apart in 1970. Every
    downstream check then passes — the index is sorted, unique and tz-aware
    — and the data is nonsense. A CSV written without an index name is
    enough to trigger it.
    """
    if isinstance(frame.index, pd.DatetimeIndex):
        return frame

    # 1. A column named like a timestamp.
    for col in frame.columns:
        if str(col).strip().lower() in _TIMESTAMP_ALIASES:
            parsed = _try_parse(frame[col])
            if parsed is not None:
                return frame.drop(columns=[col]).set_index(parsed)

    # 2. An unnamed leading column that parses as datetimes. This is what
    #    `DataFrame.to_csv()` produces for an index with no name, so a
    #    round-trip through CSV must not lose the timestamps.
    if len(frame.columns):
        first = frame.columns[0]
        label = str(first).strip().lower()
        if label.startswith("unnamed") or label == "":
            parsed = _try_parse(frame[first])
            if parsed is not None:
                return frame.drop(columns=[first]).set_index(parsed)

    # 3. An index that is already datetime-like in string form.
    if frame.index.dtype == object:
        parsed = _try_parse(pd.Series(frame.index))
        if parsed is not None:
            return frame.set_index(parsed)

    raise DataError(
        f"{symbol or 'data'}: cannot identify a timestamp index or column. "
        f"Columns are {sorted(map(str, frame.columns))[:8]} and the index is "
        f"{frame.index.dtype}. A numeric index is NOT interpreted as epoch "
        "nanoseconds — name the timestamp column or set it as the index."
    )


def _try_parse(values: pd.Series) -> pd.DatetimeIndex | None:
    """Parse a column of datetimes, or return ``None``.

    Numeric input is refused outright rather than read as epoch offsets: an
    integer column that happens to sit where a timestamp belongs is far more
    likely to be a row counter than a nanosecond count.
    """
    if pd.api.types.is_numeric_dtype(values):
        return None
    for kwargs in ({"format": "mixed"}, {"utc": True}, {}):
        try:
            parsed = pd.to_datetime(values, errors="coerce", **kwargs)
        except (TypeError, ValueError):
            continue
        if parsed.isna().mean() <= 0.01:
            return pd.DatetimeIndex(parsed)
    return None


def _to_utc(index: pd.DatetimeIndex, *, source_tz: str | None, symbol: str) -> pd.DatetimeIndex:
    """Return a UTC-aware index, localising naive input from ``source_tz``."""
    if index.tz is not None:
        return index.tz_convert("UTC")
    tz = source_tz or "UTC"
    try:
        # ambiguous/nonexistent: raise rather than guess — a DST-ambiguous
        # naive timestamp means the source is under-specified.
        return index.tz_localize(tz, ambiguous="raise", nonexistent="raise").tz_convert("UTC")
    except Exception as exc:  # pandas raises several distinct types here
        raise DataError(
            f"{symbol or 'data'}: cannot localise naive timestamps to {tz!r} ({exc}). "
            "The source timezone is ambiguous around a DST transition; supply tz-aware input."
        ) from exc


def to_exchange_time(bars: pd.DataFrame, tz: str) -> pd.DataFrame:
    """View the bars in an exchange's local time (presentation only).

    Never feed the result back into the pipeline; UTC is the storage
    contract and every downstream component assumes it.
    """
    out = bars.copy()
    out.index = out.index.tz_convert(tz)
    return out


def is_canonical(bars: pd.DataFrame) -> bool:
    """Cheap predicate: does ``bars`` already satisfy the schema?"""
    try:
        assert_canonical(bars)
    except DataError:
        return False
    return True


def assert_canonical(bars: pd.DataFrame, *, name: str = "bars") -> None:
    """Raise :class:`DataError` unless ``bars`` satisfies the schema.

    Called at every boundary where data enters a computation, so a schema
    violation surfaces as a named error rather than as a wrong number.
    """
    if not isinstance(bars, pd.DataFrame):
        raise DataError(f"{name}: expected a DataFrame, got {type(bars).__name__}")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise DataError(f"{name}: index must be a DatetimeIndex, got {type(bars.index).__name__}")
    if bars.index.tz is None:
        raise DataError(f"{name}: index must be timezone-aware (UTC); got naive timestamps")
    if str(bars.index.tz) not in {"UTC", "utc"}:
        raise DataError(f"{name}: index must be stored in UTC, got {bars.index.tz}")
    if "close" not in bars.columns:
        raise DataError(f"{name}: missing required column 'close'")
    if not bars.index.is_monotonic_increasing:
        raise DataError(f"{name}: index is not sorted ascending")
    if bars.index.has_duplicates:
        dupes = bars.index[bars.index.duplicated()][:3]
        raise DataError(f"{name}: index has duplicate timestamps, e.g. {list(dupes)}")
