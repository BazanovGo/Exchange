"""Data validation checks.

Each check is a small, independently testable function returning a
:class:`Finding`. :func:`run_all_checks` composes them into a
:class:`~exchange.data.quality.DataQualityReport`.

Severity levels
---------------
``ERROR``
    The data is unusable as-is; any result computed from it is invalid.
    Example: a bar whose ``high`` is below its ``low``.
``WARNING``
    The data is usable but a result may be distorted. Example: a 3-day gap
    in a daily series.
``INFO``
    Descriptive, no action implied.

Checks never mutate the input. Repair is a separate, explicit step so a
silent fix can never hide a bad feed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import pandas as pd

from exchange.core.timeframe import Timeframe, infer_timeframe
from exchange.data.schema import PRICE_COLUMNS


class Severity(IntEnum):
    INFO = 10
    WARNING = 20
    ERROR = 30

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.name


@dataclass(frozen=True)
class Finding:
    """One check's outcome."""

    check: str
    severity: Severity
    passed: bool
    message: str
    count: int = 0
    examples: tuple = ()
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "check": self.check,
            "severity": str(self.severity),
            "passed": self.passed,
            "message": self.message,
            "count": self.count,
            "examples": [str(e) for e in self.examples],
            **({"details": self.details} if self.details else {}),
        }


def _ok(check: str, message: str, severity: Severity = Severity.INFO, **kw) -> Finding:
    return Finding(check=check, severity=severity, passed=True, message=message, **kw)


def _fail(check: str, severity: Severity, message: str, *, count: int = 0, examples=(), **kw) -> Finding:
    return Finding(
        check=check, severity=severity, passed=False, message=message,
        count=count, examples=tuple(examples), **kw,
    )


def _nanos(index: pd.DatetimeIndex) -> np.ndarray:
    return index.to_numpy(dtype="datetime64[ns]").astype("int64")


# --------------------------------------------------------------------------
# Index checks
# --------------------------------------------------------------------------


def check_monotonic(bars: pd.DataFrame) -> Finding:
    """Timestamps must be strictly increasing."""
    idx = bars.index
    if idx.is_monotonic_increasing:
        return _ok("monotonic", "timestamps are sorted ascending")
    bad = int((np.diff(_nanos(idx)) < 0).sum())
    return _fail("monotonic", Severity.ERROR, f"{bad} backwards timestamp step(s)", count=bad)


def check_duplicates(bars: pd.DataFrame) -> Finding:
    """Duplicate timestamps silently double-count a bar."""
    mask = bars.index.duplicated(keep=False)
    n = int(mask.sum())
    if n == 0:
        return _ok("duplicates", "no duplicate timestamps")
    return _fail(
        "duplicates", Severity.ERROR,
        f"{n} row(s) share a timestamp with another row",
        count=n, examples=bars.index[mask][:5],
    )


def check_timezone(bars: pd.DataFrame) -> Finding:
    """Index must be UTC-aware (see :mod:`exchange.data.schema`)."""
    tz = bars.index.tz
    if tz is None:
        return _fail("timezone", Severity.ERROR, "index is timezone-naive; expected UTC-aware")
    if str(tz) not in {"UTC", "utc"}:
        return _fail("timezone", Severity.ERROR, f"index stored in {tz}; expected UTC")
    return _ok("timezone", "index is UTC-aware")


def _spacing_in_bars(bars: pd.DataFrame, tf: Timeframe) -> np.ndarray:
    """Bar-to-bar spacing expressed in units of one bar.

    For daily and weekly series the spacing is counted in **business days**,
    so a Friday-to-Monday step is 1.0 rather than 3.0. Without this, every
    weekend in a normal daily series would be reported as a gap and the
    check would be worthless — the classic "everything is a warning, so
    nobody reads the warnings" failure.

    Holidays still show up (a 2-day business gap over Christmas), which is
    the intended behaviour: they are real absences, just usually benign.
    """
    if tf.unit in {"d", "w"}:
        days = bars.index.tz_convert("UTC").normalize().to_numpy("datetime64[D]")
        steps = np.busday_count(days[:-1], days[1:]).astype(float)
        per_bar = tf.n * (5.0 if tf.unit == "w" else 1.0)
        # A same-day repeat (steps == 0) is a duplicate, handled elsewhere.
        return np.where(steps <= 0, 1.0, steps / per_bar)
    return (np.diff(_nanos(bars.index)) / 1e9) / tf.timedelta.total_seconds()


def check_gaps(bars: pd.DataFrame, *, timeframe: Timeframe | None = None, tolerance: float = 1.5) -> Finding:
    """Report spacing materially longer than one bar.

    Gaps are expected in any real series (holidays, session breaks), so this
    is a WARNING carrying the distribution, not an ERROR. Whether a gap is
    *unexplained* is a question no generic check can answer — hence the
    detail payload for a human.
    """
    if len(bars) < 3:
        return _ok("gaps", "too few bars to assess spacing")
    tf = timeframe or infer_timeframe(bars.index)
    spacing = _spacing_in_bars(bars, tf)
    gaps = spacing > tolerance
    n = int(gaps.sum())
    if n == 0:
        return _ok("gaps", f"no spacing above {tolerance}x the {tf.label} bar")
    largest = float(spacing.max())
    where = bars.index[1:][gaps][:5]
    return _fail(
        "gaps", Severity.WARNING,
        f"{n} gap(s) longer than {tolerance}x one {tf.label} bar; largest is {largest:.1f} bars",
        count=n, examples=where,
        details={"largest_gap_bars": largest, "gap_fraction": float(n / len(spacing))},
    )


def check_missing_bars(bars: pd.DataFrame, *, timeframe: Timeframe | None = None) -> Finding:
    """Bars absent from otherwise-dense stretches.

    Only short interior holes (2 to 5 bars) are counted: those look like a
    feed that dropped rows, whereas a long absence is a market closure.
    """
    if len(bars) < 10:
        return _ok("missing_bars", "too few bars to assess coverage")
    tf = timeframe or infer_timeframe(bars.index)
    spacing = _spacing_in_bars(bars, tf)
    interior = (spacing > 1.0) & (spacing <= 5.0)
    holes = int(interior.sum())
    missing = int(np.round(spacing[interior] - 1.0).sum())
    if holes == 0 or missing == 0:
        return _ok("missing_bars", "no short interior holes")
    frac = missing / (len(bars) + missing)
    ok = frac <= 0.001
    return Finding(
        check="missing_bars",
        severity=Severity.WARNING if not ok else Severity.INFO,
        passed=ok,
        message=f"{missing} bar(s) missing across {holes} short interior hole(s) ({frac:.3%})",
        count=missing, details={"missing_fraction": float(frac)},
    )


# --------------------------------------------------------------------------
# Price checks
# --------------------------------------------------------------------------


def check_ohlc_consistency(bars: pd.DataFrame) -> Finding:
    """``low <= min(open, close) <= max(open, close) <= high`` on every bar."""
    if not all(c in bars.columns for c in PRICE_COLUMNS):
        return _ok("ohlc_consistency", "not an OHLC series; skipped")
    o, h, low, c = (bars[k].to_numpy(float) for k in PRICE_COLUMNS)
    finite = np.isfinite(o) & np.isfinite(h) & np.isfinite(low) & np.isfinite(c)
    bad = finite & ((h < low) | (h < o) | (h < c) | (low > o) | (low > c))
    n = int(bad.sum())
    if n == 0:
        return _ok("ohlc_consistency", "every bar satisfies low <= open/close <= high")
    return _fail(
        "ohlc_consistency", Severity.ERROR,
        f"{n} bar(s) violate the OHLC ordering invariant",
        count=n, examples=bars.index[bad][:5],
    )


def check_positive_prices(bars: pd.DataFrame) -> Finding:
    """Prices must be strictly positive.

    Note: some markets *do* trade negative (WTI in April 2020, power). A
    negative price is reported as an ERROR for the return model's sake — log
    returns are undefined — and the message says so, so a researcher working
    on such a market can override deliberately.
    """
    cols = [c for c in PRICE_COLUMNS if c in bars.columns]
    values = bars[cols].to_numpy(float)
    finite = np.isfinite(values)
    bad_rows = ((values <= 0) & finite).any(axis=1)
    n = int(bad_rows.sum())
    if n == 0:
        return _ok("positive_prices", "all prices are strictly positive")
    return _fail(
        "positive_prices", Severity.ERROR,
        f"{n} bar(s) contain a non-positive price; log returns are undefined there "
        "(legitimate for negative-priced markets — handle explicitly)",
        count=n, examples=bars.index[bad_rows][:5],
    )


def check_nan_prices(bars: pd.DataFrame) -> Finding:
    """Missing close values break every downstream calculation."""
    n = int(bars["close"].isna().sum())
    if n == 0:
        return _ok("nan_prices", "no missing close prices")
    severity = Severity.ERROR if n / len(bars) > 0.01 else Severity.WARNING
    return _fail(
        "nan_prices", severity,
        f"{n} missing close price(s) ({n / len(bars):.2%})",
        count=n, examples=bars.index[bars["close"].isna()][:5],
    )


def check_volume(bars: pd.DataFrame) -> Finding:
    """Negative volume is impossible; all-zero volume means no real feed."""
    if "volume" not in bars.columns or bars["volume"].isna().all():
        return Finding("volume", Severity.INFO, True, "no volume data supplied")
    vol = bars["volume"].to_numpy(float)
    negative = int((vol < 0).sum())
    if negative:
        return _fail(
            "volume", Severity.ERROR, f"{negative} bar(s) with negative volume",
            count=negative, examples=bars.index[vol < 0][:5],
        )
    zero_frac = float(np.nanmean(vol == 0))
    if zero_frac > 0.5:
        return _fail(
            "volume", Severity.WARNING,
            f"{zero_frac:.0%} of bars have zero volume; liquidity-aware execution will be unreliable",
            count=int((vol == 0).sum()), details={"zero_fraction": zero_frac},
        )
    return _ok("volume", f"volume present, {zero_frac:.2%} zero-volume bars")


def check_price_jumps(bars: pd.DataFrame, *, sigma: float = 12.0, min_bars: int = 100) -> Finding:
    """Flag returns far outside the series' own scale.

    A jump is not necessarily an error — real markets gap — so this is a
    WARNING. It is here to surface decimal-point errors, bad ticks and, for
    futures, an unadjusted contract roll left in the series.

    The threshold is on a robust scale (median absolute deviation), so a
    handful of outliers cannot inflate it and hide themselves.
    """
    close = bars["close"].to_numpy(float)
    if len(close) < min_bars:
        return _ok("price_jumps", "too few bars to assess jumps")
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.diff(np.log(np.where(close > 0, close, np.nan)))
    rets = rets[np.isfinite(rets)]
    if rets.size < min_bars // 2:
        return _ok("price_jumps", "too few finite returns to assess jumps")
    med = float(np.median(rets))
    mad = float(np.median(np.abs(rets - med)))
    scale = mad * 1.4826  # MAD -> sigma for a Gaussian
    if scale <= 0:
        return _ok("price_jumps", "return series has zero dispersion")
    z = np.abs(rets - med) / scale
    n = int((z > sigma).sum())
    if n == 0:
        return _ok("price_jumps", f"no return beyond {sigma:g} robust sigma")
    worst = float(z.max())
    return _fail(
        "price_jumps", Severity.WARNING,
        f"{n} return(s) beyond {sigma:g} robust sigma (worst {worst:.0f} sigma); "
        "check for bad ticks, decimal errors or an unadjusted roll",
        count=n, details={"max_sigma": worst, "robust_sigma": scale},
    )


def check_staleness(bars: pd.DataFrame, *, max_run: int = 20) -> Finding:
    """Long runs of an unchanged close usually mean a stalled feed."""
    close = bars["close"].to_numpy(float)
    if len(close) < max_run + 1:
        return _ok("staleness", "too few bars to assess staleness")
    unchanged = np.r_[False, close[1:] == close[:-1]]
    longest, run = 0, 0
    for flag in unchanged:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    if longest < max_run:
        return _ok("staleness", f"longest unchanged-close run is {longest} bar(s)")
    return _fail(
        "staleness", Severity.WARNING,
        f"close price unchanged for {longest} consecutive bars; the feed may be stalled",
        count=longest,
    )


DEFAULT_CHECKS: tuple[Callable[[pd.DataFrame], Finding], ...] = (
    check_timezone,
    check_monotonic,
    check_duplicates,
    check_nan_prices,
    check_ohlc_consistency,
    check_positive_prices,
    check_volume,
    check_gaps,
    check_missing_bars,
    check_price_jumps,
    check_staleness,
)


def run_all_checks(
    bars: pd.DataFrame,
    *,
    checks: Sequence[Callable[[pd.DataFrame], Finding]] | None = None,
) -> list[Finding]:
    """Run every check, isolating failures so one crash cannot hide the rest."""
    out: list[Finding] = []
    for check in checks or DEFAULT_CHECKS:
        name = getattr(check, "__name__", str(check)).removeprefix("check_")
        try:
            out.append(check(bars))
        except Exception as exc:  # a broken check must not mask the others
            out.append(_fail(name, Severity.WARNING, f"check raised {type(exc).__name__}: {exc}"))
    return out


def worst_severity(findings: Iterable[Finding]) -> Severity:
    """Highest severity among failed findings (INFO when all passed)."""
    failed = [f.severity for f in findings if not f.passed]
    return max(failed) if failed else Severity.INFO
