"""Temporal integrity: point-in-time data and lookahead guards.

The rule this module enforces
-----------------------------
**No value published after a decision time may influence that decision.**

Two timestamps are needed to honour it, and conflating them is the most
expensive mistake in this whole system:

``observation_time``
    When the fact was *about*. EIA's weekly gas storage report for the week
    ending 2026-01-02 has observation time 2026-01-02.

``available_time``
    When a market participant could first *act* on it. That same report is
    published on 2026-01-08 at 10:30 ET, so its available time is
    2026-01-08T15:30Z.

Indexing a fundamental series by observation time and joining it onto price
bars is a guaranteed lookahead: on 2026-01-03 the backtest reads a number
nobody had. :class:`PointInTimeSeries` makes the second timestamp
mandatory, and alignment goes through a backward ``merge_asof`` on it.

Bar timing
----------
A bar labelled ``t`` covers ``[t, t + timeframe)``. Its close is knowable
only at ``t + timeframe``. Therefore a signal computed from bar ``t`` may be
acted on no earlier than bar ``t + 1`` — which is why the backtest engine
executes on the next bar and this module provides :func:`assert_causal` to
prove a transform never reaches forward.

Two guards, not one
-------------------
:func:`assert_causal` catches a *transform* that reaches forward.
:func:`detect_column_leakage` catches a *column that arrived already
contaminated*. The second is not redundant: if the leak happened upstream,
every prefix of the data contains it and the prefix test passes. A system
with only one of these two guards has a hole, and that hole was found by a
trap control rather than by inspection.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from exchange.core.errors import DataError, LookaheadError

OBSERVATION_TIME = "observation_time"
AVAILABLE_TIME = "available_time"


@dataclass
class PointInTimeSeries:
    """A fact series that knows when each value became public.

    Parameters
    ----------
    frame:
        Must contain ``available_time`` (UTC-aware) and at least one value
        column. ``observation_time`` is the index.
    name:
        Label used in errors and reports.
    revision_policy:
        ``"latest"`` keeps the most recent revision available at a given
        time (what a live system sees). ``"first"`` keeps the original
        print, ignoring later revisions — useful when testing whether an
        effect survives without the benefit of restatements.
    """

    frame: pd.DataFrame
    name: str = ""
    revision_policy: str = "latest"
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if AVAILABLE_TIME not in self.frame.columns:
            raise DataError(
                f"{self.name or 'series'}: point-in-time data must carry an "
                f"{AVAILABLE_TIME!r} column; observation time alone is a lookahead"
            )
        if not isinstance(self.frame.index, pd.DatetimeIndex):
            raise DataError(f"{self.name or 'series'}: index must be the observation time (DatetimeIndex)")
        if self.frame.index.tz is None:
            raise DataError(f"{self.name or 'series'}: observation_time index must be UTC-aware")
        avail = pd.to_datetime(self.frame[AVAILABLE_TIME], utc=True)
        # pandas 3 parses to microsecond resolution while most bar indexes are
        # nanosecond, and a merge_asof between the two raises rather than
        # converting. Pin the column to the observation index's own unit so
        # every comparison and join downstream sees one resolution.
        avail = avail.astype(self.frame.index.dtype)
        self.frame = self.frame.assign(**{AVAILABLE_TIME: avail}).sort_index(kind="stable")
        if self.revision_policy not in {"latest", "first"}:
            raise DataError(f"unknown revision policy {self.revision_policy!r}")
        early = avail.to_numpy() < self.frame.index.to_numpy()
        if bool(np.any(early)):
            raise LookaheadError(
                f"{self.name or 'series'}: {int(np.sum(early))} row(s) claim to be available "
                "before they were observed"
            )

    @property
    def value_columns(self) -> list[str]:
        return [c for c in self.frame.columns if c != AVAILABLE_TIME]

    def as_of(self, when: pd.Timestamp | str) -> pd.Series:
        """The state of the world as it was known at ``when``.

        Returns the value columns for the latest observation whose
        ``available_time <= when``.
        """
        when = pd.Timestamp(when)
        if when.tz is None:
            raise DataError("as_of requires a timezone-aware timestamp")
        visible = self.frame[self.frame[AVAILABLE_TIME] <= when]
        if visible.empty:
            return pd.Series(index=self.value_columns, dtype="float64")
        keep = "first" if self.revision_policy == "first" else "last"
        visible = visible[~visible.index.duplicated(keep=keep)]
        return visible.iloc[-1][self.value_columns]

    def align_to(self, index: pd.DatetimeIndex, *, decision_lag: pd.Timedelta | None = None) -> pd.DataFrame:
        """Project the series onto bar timestamps without leaking.

        For each bar ``t`` the value returned is the one available at
        ``t + decision_lag``. The default lag is zero, meaning "everything
        published strictly at or before the bar's own start". Because the
        backtest engine acts on bar ``t + 1``, that is already conservative.

        Implemented with a backward :func:`pandas.merge_asof` on
        ``available_time`` — the only join that cannot see forward.
        """
        if index.tz is None:
            raise DataError("align_to requires a UTC-aware bar index")
        lag = decision_lag or pd.Timedelta(0)
        targets = pd.DataFrame({"_bar": index, "_key": index + lag}).sort_values("_key")

        source = (
            self.frame.reset_index(names=OBSERVATION_TIME)
            .sort_values(AVAILABLE_TIME, kind="stable")
        )
        if self.revision_policy == "first":
            source = source.drop_duplicates(subset=OBSERVATION_TIME, keep="first")

        # Same resolution concern as above, now between the bar index and the
        # availability column.
        targets["_key"] = targets["_key"].astype(source[AVAILABLE_TIME].dtype)
        merged = pd.merge_asof(
            targets,
            source,
            left_on="_key",
            right_on=AVAILABLE_TIME,
            direction="backward",
            allow_exact_matches=True,
        )
        merged = merged.set_index("_bar").sort_index()
        merged.index.name = index.name or "timestamp"
        out = merged[[*self.value_columns, OBSERVATION_TIME, AVAILABLE_TIME]]
        return out.rename(
            columns={
                OBSERVATION_TIME: f"{self.name}_observation_time" if self.name else OBSERVATION_TIME,
                AVAILABLE_TIME: f"{self.name}_available_time" if self.name else AVAILABLE_TIME,
            }
        )

    def staleness(self, index: pd.DatetimeIndex) -> pd.Series:
        """Age of the visible observation at each bar, as a Timedelta.

        A feature whose staleness is highly variable (a weekly report seen
        as 1 day old on Thursday and 7 days old on Wednesday) carries a
        calendar signal of its own; reporting it prevents mistaking "it is
        Thursday" for an economic effect.
        """
        aligned = self.align_to(index)
        obs_col = next(c for c in aligned.columns if c.endswith(OBSERVATION_TIME))
        return (index - pd.to_datetime(aligned[obs_col], utc=True)).rename("staleness")


def make_point_in_time(
    values: pd.Series | pd.DataFrame,
    *,
    publication_lag: pd.Timedelta,
    name: str = "",
    revision_policy: str = "latest",
) -> PointInTimeSeries:
    """Build a PIT series from observation-time data plus a fixed lag.

    A constant lag is an approximation — real publication schedules slip —
    so anything built this way is honest only if the lag is conservative
    (i.e. at least the true worst case).
    """
    frame = values.to_frame() if isinstance(values, pd.Series) else values.copy()
    if frame.index.tz is None:
        raise DataError("observation index must be UTC-aware")
    frame[AVAILABLE_TIME] = frame.index + publication_lag
    return PointInTimeSeries(frame=frame, name=name, revision_policy=revision_policy)


# ==========================================================================
# Lookahead guards
# ==========================================================================


def assert_causal(
    transform: Callable[[pd.DataFrame], pd.Series | pd.DataFrame],
    data: pd.DataFrame,
    *,
    name: str = "transform",
    min_prefix: int | None = None,
    atol: float = 1e-9,
) -> None:
    """Prove that ``transform`` never uses information from the future.

    The test: run the transform on the full history, then re-run it on each
    of a set of truncated prefixes. If the transform is causal, the value at
    bar ``k`` must be identical whether or not bars after ``k`` exist. Any
    difference means the future leaked into the past.

    This is a *proof by counterexample* on real data, not a code review, and
    it catches the whole family of leaks that look innocent in source form:
    full-sample z-scores, ``bfill``, centred rolling windows, ``.rank()``
    over the entire series, a scaler fitted on everything.

    Raises
    ------
    LookaheadError
        With the first offending bar and the size of the discrepancy.
    """
    n = len(data)
    if n < 12:
        raise DataError(f"assert_causal needs at least 12 bars, got {n}")
    full = _as_frame(transform(data))
    start = min_prefix if min_prefix is not None else max(5, n // 4)
    cuts = sorted({start, n // 2, (3 * n) // 4, n - 1})

    for cut in cuts:
        if cut < 2 or cut >= n:
            continue
        partial = _as_frame(transform(data.iloc[:cut]))
        if list(partial.columns) != list(full.columns):
            raise LookaheadError(
                f"{name}: output columns depend on the length of the input "
                f"({list(partial.columns)} vs {list(full.columns)})"
            )
        head = full.iloc[:cut]
        for col in full.columns:
            a = pd.to_numeric(head[col], errors="coerce").to_numpy(float)
            b = pd.to_numeric(partial[col], errors="coerce").to_numpy(float)
            both_nan = np.isnan(a) & np.isnan(b)
            diff = np.abs(a - b)
            bad = ~both_nan & ~(diff <= atol)
            # A value that is NaN on the prefix but defined on the full run
            # is also a leak (the transform "knew" more with future data).
            bad |= np.isnan(b) & ~np.isnan(a)
            if bad.any():
                first = int(np.argmax(bad))
                raise LookaheadError(
                    f"{name}: output for column {col!r} at position {first} "
                    f"({data.index[first]}) changes when bars after position {cut} are added "
                    f"(full={a[first]!r}, prefix={b[first]!r}). The transform is not causal."
                )


def _as_frame(obj) -> pd.DataFrame:
    if isinstance(obj, pd.Series):
        return obj.to_frame(obj.name or "value")
    if isinstance(obj, pd.DataFrame):
        return obj
    raise DataError(f"transform must return a Series or DataFrame, got {type(obj).__name__}")


@dataclass(frozen=True)
class LeakageFinding:
    """One column that appears to contain information from the future."""

    column: str
    corr_next: float      # correlation with the NEXT bar's return
    corr_current: float   # correlation with the CURRENT bar's return
    ratio: float          # |corr_next| / |corr_current|
    z_next: float         # corr_next in units of its own null standard error
    verdict: str          # LEAK / SUSPICIOUS / CLEAN

    def to_dict(self) -> dict:
        return {
            "column": self.column,
            "corr_next": self.corr_next,
            "corr_current": self.corr_current,
            "ratio": self.ratio,
            "z_next": self.z_next,
            "verdict": self.verdict,
        }


_SKIP_COLUMNS = {
    "open", "high", "low", "close", "volume", "open_interest", "adj_close",
    "contract", "is_roll", "adj_factor", "adj_offset",
}


def detect_column_leakage(
    bars: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    price_column: str = "close",
    z_threshold: float = 6.0,
    ratio_threshold: float = 2.0,
) -> list[LeakageFinding]:
    """Find feature columns that already contain future information.

    The test compares each column's correlation with the **next** bar's
    return against its correlation with the **current** bar's return. A
    legitimate predictor can correlate with the future, but only weakly and
    never far more strongly than it explains the present. A column that
    predicts the next bar at a magnitude no genuine signal reaches is
    contaminated.

    Thresholds are deliberately loose: this fires on ``|z| > 6`` — a
    correlation six standard errors from zero under the null — so it flags
    outright contamination rather than passing judgement on weak signals. A
    real edge with a per-bar information coefficient of 0.05 will not trip
    it on a few thousand bars; a column holding the next return trips it
    immediately.

    Returns
    -------
    list[LeakageFinding]
        One entry per examined column, ordered worst first.
    """
    if price_column not in bars.columns:
        raise DataError(f"detect_column_leakage needs a {price_column!r} column")
    returns = bars[price_column].pct_change()
    next_returns = returns.shift(-1)

    candidates = columns if columns is not None else [
        c for c in bars.columns
        if c not in _SKIP_COLUMNS and pd.api.types.is_numeric_dtype(bars[c])
    ]

    findings: list[LeakageFinding] = []
    for name in candidates:
        series = pd.to_numeric(bars[name], errors="coerce")
        frame = (
            pd.DataFrame({"x": series, "now": returns, "nxt": next_returns})
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(frame) < 40 or frame["x"].std(ddof=1) == 0:
            continue
        corr_next = float(frame["x"].corr(frame["nxt"]))
        corr_now = float(frame["x"].corr(frame["now"]))
        if not np.isfinite(corr_next):
            continue
        se = 1.0 / np.sqrt(len(frame))
        z_next = corr_next / se
        ratio = abs(corr_next) / max(abs(corr_now), 1e-9)

        if abs(z_next) > z_threshold and ratio > ratio_threshold:
            verdict = "LEAK"
        elif abs(z_next) > z_threshold:
            verdict = "SUSPICIOUS"
        else:
            verdict = "CLEAN"
        findings.append(LeakageFinding(name, corr_next, corr_now, ratio, float(z_next), verdict))
    return sorted(findings, key=lambda f: -abs(f.z_next))


def assert_no_column_leakage(bars: pd.DataFrame, **kwargs) -> None:
    """Raise :class:`LookaheadError` if any column looks contaminated."""
    leaks = [f for f in detect_column_leakage(bars, **kwargs) if f.verdict == "LEAK"]
    if leaks:
        detail = "; ".join(
            f"{f.column} (corr with next bar {f.corr_next:+.3f}, z = {f.z_next:.0f})"
            for f in leaks
        )
        raise LookaheadError(
            f"{len(leaks)} column(s) contain information from the future: {detail}"
        )


def assert_no_future_index(values: pd.Series | pd.DataFrame, decision_time: pd.Timestamp) -> None:
    """Raise if any row is timestamped at or after ``decision_time``."""
    future = values.index >= decision_time
    if bool(np.any(future)):
        raise LookaheadError(
            f"{int(np.sum(future))} row(s) at or after the decision time {decision_time.isoformat()}"
        )


def shift_to_decision(signal: pd.Series | pd.DataFrame, bars: int = 1):
    """Delay a signal by ``bars`` so it can only act on already-closed bars.

    The canonical guard: a signal derived from bar ``t``'s close becomes
    actionable at bar ``t + 1``. Applied by the backtest engine; exposed here
    so strategy code can be written and reasoned about unshifted.
    """
    if bars < 1:
        raise LookaheadError("a signal must be delayed by at least one bar to be actionable")
    return signal.shift(bars)
