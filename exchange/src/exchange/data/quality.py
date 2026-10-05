"""Data Quality Report.

Wraps the checks in :mod:`exchange.data.validation` into a single object
that is printable, serialisable and — crucially — *gating*: a dataset that
fails an ERROR check must not silently reach a backtest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from exchange.core.errors import DataQualityError
from exchange.core.hashing import hash_dataframe, short
from exchange.core.timeframe import Timeframe, infer_timeframe
from exchange.data.schema import assert_canonical
from exchange.data.validation import Finding, Severity, run_all_checks, worst_severity


@dataclass
class DataQualityReport:
    """The outcome of validating one bar series."""

    symbol: str
    n_bars: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    timeframe: str
    data_hash: str
    findings: list[Finding] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # -- status ---------------------------------------------------------

    @property
    def errors(self) -> list[Finding]:
        return [f for f in self.findings if not f.passed and f.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Finding]:
        return [f for f in self.findings if not f.passed and f.severity is Severity.WARNING]

    @property
    def ok(self) -> bool:
        """True when nothing failed at ERROR severity."""
        return not self.errors

    @property
    def severity(self) -> Severity:
        return worst_severity(self.findings)

    def raise_for_errors(self) -> None:
        """Raise :class:`DataQualityError` if any ERROR check failed."""
        if self.errors:
            detail = "; ".join(f"{f.check}: {f.message}" for f in self.errors)
            raise DataQualityError(
                f"{self.symbol}: {len(self.errors)} blocking data-quality error(s) — {detail}"
            )

    # -- serialisation --------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "n_bars": self.n_bars,
            "start": None if self.start is None else self.start.isoformat(),
            "end": None if self.end is None else self.end.isoformat(),
            "timeframe": self.timeframe,
            "data_hash": self.data_hash,
            "status": "OK" if self.ok else "FAILED",
            "severity": str(self.severity),
            "findings": [f.to_dict() for f in self.findings],
            "meta": self.meta,
        }

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([f.to_dict() for f in self.findings])

    def to_markdown(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        span = "n/a" if self.start is None else f"{self.start:%Y-%m-%d %H:%M} .. {self.end:%Y-%m-%d %H:%M} UTC"
        lines = [
            f"# Data Quality Report — {self.symbol}",
            "",
            f"- **Status**: {status} ({len(self.errors)} error(s), {len(self.warnings)} warning(s))",
            f"- **Bars**: {self.n_bars:,}",
            f"- **Span**: {span}",
            f"- **Timeframe**: {self.timeframe}",
            f"- **Data hash**: `{short(self.data_hash)}`",
            "",
            "| check | severity | result | detail |",
            "|---|---|---|---|",
        ]
        for f in sorted(self.findings, key=lambda f: (f.passed, -f.severity)):
            mark = "pass" if f.passed else "**FAIL**"
            lines.append(f"| `{f.check}` | {f.severity} | {mark} | {f.message} |")
        if not self.ok:
            lines += ["", "> Blocking errors are present. This dataset must not be used for research."]
        return "\n".join(lines) + "\n"

    def save(self, directory: str | Path, *, stem: str = "data_quality") -> Path:
        """Write the report as Markdown next to a JSON payload."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{stem}.md").write_text(self.to_markdown())
        (directory / f"{stem}.json").write_text(json.dumps(self.to_dict(), indent=2))
        return directory / f"{stem}.md"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"DataQualityReport({self.symbol}, {self.n_bars} bars, {'OK' if self.ok else 'FAILED'})"


def _resolve_session(bars: pd.DataFrame, tf, session):
    """Infer, accept or decline a session calendar."""
    if session is None or session == "none":
        return None
    if session != "auto":
        return session
    from exchange.data.sessions import infer_session

    try:
        return infer_session(bars.index, tf)
    except Exception:          # inference must never break validation
        return None


def _bind_session(session):
    """The default checks, with the session bound into the ones that use it."""
    from functools import partial

    from exchange.data.validation import (
        DEFAULT_CHECKS,
        check_gaps,
        check_missing_bars,
        check_session_coverage,
    )

    bound = {
        "check_gaps": partial(check_gaps, session=session),
        "check_missing_bars": partial(check_missing_bars, session=session),
        "check_session_coverage": partial(check_session_coverage, session=session),
    }
    out = []
    for check in DEFAULT_CHECKS:
        name = getattr(check, "__name__", "")
        replacement = bound.get(name)
        if replacement is not None:
            replacement.__name__ = name     # run_all_checks reads __name__
            out.append(replacement)
        else:
            out.append(check)
    return tuple(out)


def validate_bars(
    bars: pd.DataFrame,
    *,
    symbol: str = "data",
    timeframe: Timeframe | str | None = None,
    checks=None,
    session="auto",
) -> DataQualityReport:
    """Validate one canonical bar series and return its quality report.

    ``session`` controls how spacing is judged on intraday data:
    ``"auto"`` (the default) infers a piecewise trading session from the
    timestamps, ``None`` disables it, and a
    :class:`~exchange.data.sessions.SessionSchedule` uses an authoritative
    calendar. Without a session every overnight break on an hourly series
    counts as a seven-bar gap, which buries real holes under thousands of
    warnings.
    """
    assert_canonical(bars, name=symbol)
    tf = None
    if timeframe is None:
        tf = infer_timeframe(bars.index) if len(bars) >= 3 else None
        tf_label = tf.label if tf is not None else "unknown"
    else:
        tf = Timeframe.parse(timeframe)
        tf_label = tf.label

    resolved = _resolve_session(bars, tf, session)
    if checks is None and resolved is not None:
        checks = _bind_session(resolved)

    return DataQualityReport(
        symbol=symbol,
        n_bars=len(bars),
        start=bars.index[0] if len(bars) else None,
        end=bars.index[-1] if len(bars) else None,
        timeframe=tf_label,
        data_hash=hash_dataframe(bars),
        findings=run_all_checks(bars, checks=checks),
    )
