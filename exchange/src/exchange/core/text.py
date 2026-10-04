"""Small text helpers shared by every report.

The Markdown table renderer is written out here rather than delegated to
``DataFrame.to_markdown``, which needs the optional ``tabulate`` package. A
report that cannot be produced in a minimal environment is a report that
will not be produced in CI, and a reproducibility framework that fails on a
missing formatting library has its priorities wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def format_cell(value, *, digits: int = 4) -> str:
    """One table cell: compact, aligned, and never scientific notation soup."""
    if value is None:
        return ""
    if isinstance(value, bool | np.bool_):
        return str(bool(value))
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float | np.floating):
        number = float(value)
        if not np.isfinite(number):
            return ""
        if number != 0 and (abs(number) >= 1e7 or abs(number) < 1e-6):
            return f"{number:.3e}"
        return f"{number:,.{digits}f}" if abs(number) >= 1000 else f"{number:.{digits}f}"
    if isinstance(value, int | np.integer):
        return f"{int(value):,}" if abs(int(value)) >= 10000 else str(int(value))
    return str(value)


def markdown_table(
    frame: pd.DataFrame,
    *,
    max_rows: int | None = None,
    index: bool = True,
    digits: int = 4,
    empty: str = "_No table produced._",
) -> str:
    """Render a frame as a GitHub-flavoured Markdown table."""
    if frame is None or getattr(frame, "empty", True):
        return empty
    shown = frame if max_rows is None else frame.head(max_rows)
    note = (
        f"\n\n_{len(frame)} rows; first {max_rows} shown._"
        if max_rows is not None and len(frame) > max_rows
        else ""
    )
    header = [str(c) for c in shown.columns]
    if index:
        header = [shown.index.name or "", *header]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for key, row in shown.iterrows():
        cells = [format_cell(v, digits=digits) for v in row]
        if index:
            cells = [format_cell(key, digits=digits), *cells]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + note
