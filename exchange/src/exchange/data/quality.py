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


def validate_bars(
    bars: pd.DataFrame,
    *,
    symbol: str = "data",
    timeframe: Timeframe | str | None = None,
    checks=None,
) -> DataQualityReport:
    """Validate one canonical bar series and return its quality report."""
    assert_canonical(bars, name=symbol)
    if timeframe is None:
        tf = infer_timeframe(bars.index) if len(bars) >= 3 else None
        tf_label = tf.label if tf is not None else "unknown"
    else:
        tf_label = Timeframe.parse(timeframe).label

    return DataQualityReport(
        symbol=symbol,
        n_bars=len(bars),
        start=bars.index[0] if len(bars) else None,
        end=bars.index[-1] if len(bars) else None,
        timeframe=tf_label,
        data_hash=hash_dataframe(bars),
        findings=run_all_checks(bars, checks=checks),
    )
