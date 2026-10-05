"""Trading sessions inferred from the data.

Why this exists
---------------
A gap check that measures wall-clock time is worthless on intraday data. A
MOEX futures series trades roughly 06:00-23:00 Moscow time, five days a
week, so every night is a 7-hour "gap" and every weekend a 60-hour one. On
hourly natural-gas data that produced 1,800 warnings — and a check that
fires 1,800 times is a check nobody reads, which is how a *real* hole hides
among the noise.

Why the session is inferred rather than configured
--------------------------------------------------
A hard-coded calendar is wrong the moment an exchange changes its hours,
and there is no authoritative calendar for every venue and every year. The
session here is measured from the data: a slot (weekday, time-of-day) is
"open" if bars appear in it on at least ``min_occupancy`` of the days that
weekday occurs.

The consequence is deliberate and must be understood: **a slot that is
*always* empty is not a gap — it is a market that is closed.** This check
cannot tell you that an exchange extended its hours and your vendor never
delivered the new bars. What it *can* tell you is that a normally-traded
slot is missing, which is the failure a feed actually produces.

Sessions change, so the schedule is piecewise
---------------------------------------------
Exchanges move their hours. MOEX added a morning session and extended the
evening one during the sample used to build this module, so a single
calendar averaged over fifteen years fits neither the early years nor the
late ones: it marked 6% of real bars as off-session and still reported
hundreds of gaps. :class:`SessionSchedule` therefore infers one calendar
per year and reports whether the schedule changed — which is itself a
research finding, since an intraday strategy fitted across a session change
is fitted to two different markets.

The inferred session is reported in the quality report so the reading is
auditable, and a caller who does have an authoritative calendar can pass
one in instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from exchange.core.errors import ConfigError, DataError
from exchange.core.timeframe import Timeframe

# Beyond this many expected slots the grid is not materialised: a span that
# large means the series is either enormous or has a hole measured in years,
# and in both cases the raw ratio is the honest answer.
MAX_GRID = 20_000_000


@dataclass(frozen=True)
class SessionCalendar:
    """Which (weekday, time-of-day) slots an instrument normally trades in."""

    timeframe: Timeframe
    slots: frozenset[tuple[int, int]]        # (weekday 0=Mon, slot index within the day)
    min_occupancy: float = 0.5
    n_bars_observed: int = 0
    source: str = "inferred"

    def __post_init__(self) -> None:
        if not self.slots:
            raise ConfigError("a session calendar needs at least one open slot")

    # -- geometry --------------------------------------------------------

    @property
    def step(self) -> pd.Timedelta:
        return self.timeframe.timedelta

    @property
    def slots_per_day(self) -> int:
        return max(round(86_400 / self.timeframe.seconds), 1)

    @property
    def weekdays(self) -> tuple[int, ...]:
        return tuple(sorted({day for day, _ in self.slots}))

    @property
    def slots_per_week(self) -> int:
        return len(self.slots)

    def slot_of(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Slot index within the day for each timestamp."""
        seconds = (
            index.hour.to_numpy() * 3600
            + index.minute.to_numpy() * 60
            + index.second.to_numpy()
        )
        return seconds // int(self.timeframe.seconds)

    def is_open(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Boolean mask: is each timestamp in a normally-traded slot?"""
        days = index.weekday.to_numpy()
        slots = self.slot_of(index)
        lookup = self.slots
        return np.fromiter(
            ((int(d), int(s)) in lookup for d, s in zip(days, slots, strict=True)),
            dtype=bool,
            count=len(index),
        )

    def expected_grid(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        """Every open slot between ``start`` and ``end`` inclusive."""
        full = pd.date_range(start, end, freq=self.step, tz=start.tz)
        if len(full) > MAX_GRID:
            raise DataError(
                f"the session grid between {start} and {end} would need {len(full):,} slots; "
                "the span is too large to materialise"
            )
        return full[self.is_open(full)]

    # -- what the checks need -------------------------------------------

    def spacing(self, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        """Bar-to-bar spacing **in session bars**, and an off-session mask.

        Spacing is the distance between consecutive observed bars measured
        along the session grid, so a Friday close to Monday open is 1.0 and
        an overnight break is 1.0, while a genuinely missing hour is 2.0.

        Returns ``(spacing, off_session)``; ``spacing`` has one entry per
        bar after the first. Off-grid bars (a timestamp in a slot the
        instrument does not normally trade) are reported separately rather
        than silently snapped to a neighbour.
        """
        if len(index) < 2:
            return np.empty(0), np.zeros(len(index), dtype=bool)
        grid = self.expected_grid(index[0], index[-1])
        positions = grid.get_indexer(index)
        off_session = positions < 0
        # Interpolate positions for off-grid bars so the spacing series stays
        # the right length; their own spacing is not meaningful and they are
        # reported through the mask.
        positions = positions.astype(float)
        positions[off_session] = np.nan
        filled = pd.Series(positions).ffill().bfill().to_numpy()
        return np.diff(filled), off_session

    def describe(self) -> str:
        day_names = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
        days = self.weekdays
        span = (
            f"{day_names[days[0]]}-{day_names[days[-1]]}"
            if len(days) > 1 and days == tuple(range(days[0], days[-1] + 1))
            else ",".join(day_names[d] for d in days)
        )
        per_day = len(self.slots) / max(len(days), 1)
        return (
            f"{span}, {per_day:.0f} {self.timeframe.label} slot(s) per day "
            f"({self.slots_per_week} per week), {self.source} from "
            f"{self.n_bars_observed:,} bars"
        )

    def to_dict(self) -> dict:
        return {
            "timeframe": self.timeframe.label,
            "weekdays": list(self.weekdays),
            "slots_per_week": self.slots_per_week,
            "min_occupancy": self.min_occupancy,
            "source": self.source,
            "description": self.describe(),
        }


@dataclass(frozen=True)
class SessionSchedule:
    """One session calendar per period, because exchange hours change."""

    segments: tuple[tuple[pd.Timestamp, SessionCalendar], ...]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ConfigError("a session schedule needs at least one segment")

    @property
    def timeframe(self) -> Timeframe:
        return self.segments[0][1].timeframe

    @property
    def changed(self) -> bool:
        """Did the set of traded slots change during the sample?"""
        return len({c.slots for _, c in self.segments}) > 1

    def calendar_at(self, when: pd.Timestamp) -> SessionCalendar:
        current = self.segments[0][1]
        for start, calendar in self.segments:
            if when >= start:
                current = calendar
            else:
                break
        return current

    def expected_grid(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        """Open slots across the whole span, each period using its own calendar."""
        pieces = []
        for i, (segment_start, calendar) in enumerate(self.segments):
            lo = max(segment_start, start)
            hi = self.segments[i + 1][0] - calendar.step if i + 1 < len(self.segments) else end
            hi = min(hi, end)
            if lo > hi:
                continue
            pieces.append(calendar.expected_grid(lo, hi))
        if not pieces:
            return pd.DatetimeIndex([], tz=start.tz)
        if len(pieces) == 1:
            return pieces[0]
        return pieces[0].append(pieces[1:]).sort_values()

    def spacing(self, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        """Spacing in session bars and an off-session mask (see :meth:`SessionCalendar.spacing`)."""
        if len(index) < 2:
            return np.empty(0), np.zeros(len(index), dtype=bool)
        grid = self.expected_grid(index[0], index[-1])
        if len(grid) == 0:
            return np.full(len(index) - 1, np.nan), np.zeros(len(index), dtype=bool)
        positions = grid.get_indexer(index).astype(float)
        off_session = positions < 0
        positions[off_session] = np.nan
        filled = pd.Series(positions).ffill().bfill().to_numpy()
        return np.diff(filled), off_session

    def describe(self) -> str:
        if len(self.segments) == 1:
            return self.segments[0][1].describe()
        parts = [
            f"{start:%Y-%m}: {calendar.slots_per_week} slot(s)/week"
            for start, calendar in self.segments
        ]
        verdict = "the session CHANGED during the sample" if self.changed else "stable"
        return f"{self.timeframe.label} session, {verdict} — " + "; ".join(parts)

    def to_dict(self) -> dict:
        return {
            "timeframe": self.timeframe.label,
            "n_segments": len(self.segments),
            "changed": self.changed,
            "description": self.describe(),
            "segments": [
                {"from": str(start), **calendar.to_dict()} for start, calendar in self.segments
            ],
        }


@dataclass
class SessionReport:
    """Diagnostics from inferring a session."""

    calendar: SessionCalendar | None
    reason: str = ""
    occupancy: dict = field(default_factory=dict)


def infer_session(
    index: pd.DatetimeIndex,
    timeframe: Timeframe | str | None = None,
    *,
    min_occupancy: float = 0.5,
    min_bars: int = 500,
    by: str | None = "YE",
) -> SessionSchedule | None:
    """Infer the trading session, piecewise, from observed timestamps.

    ``by`` is a pandas period alias (default yearly) over which one
    calendar is inferred; ``None`` infers a single calendar for the whole
    span. A period with too few bars reuses the previous period's calendar
    rather than inferring a flimsy one of its own.

    Returns ``None`` when inference would be unreliable — a daily or
    coarser series (where business-day counting is the right measure
    anyway), or too few bars. Returning ``None`` rather than a guess
    matters: a wrong session calendar would silently reclassify real holes
    as closures.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise DataError("infer_session needs a DatetimeIndex")
    if len(index) < min_bars:
        return None
    tf = Timeframe.parse(timeframe) if timeframe is not None else None
    if tf is None:
        from exchange.core.timeframe import infer_timeframe

        tf = infer_timeframe(index)
    if tf.unit not in {"s", "m", "h"}:
        return None            # daily and above: business-day counting applies

    if by is None:
        calendar = _infer_one(index, tf, min_occupancy=min_occupancy)
        return SessionSchedule(((index[0], calendar),)) if calendar else None

    segments: list[tuple[pd.Timestamp, SessionCalendar]] = []
    previous: SessionCalendar | None = None
    for _period, positions in pd.Series(range(len(index)), index=index).groupby(
        pd.Grouper(freq=by)
    ):
        if len(positions) == 0:
            continue
        subset = index[positions.to_numpy()]
        calendar = (
            _infer_one(subset, tf, min_occupancy=min_occupancy)
            if len(subset) >= max(min_bars // 4, 50)
            else None
        ) or previous
        if calendar is None:
            continue
        start = segments[0][0] if segments else index[0]
        segments.append((subset[0] if segments else start, calendar))
        previous = calendar
    if not segments:
        return None
    return SessionSchedule(tuple(segments))


def _infer_one(
    index: pd.DatetimeIndex, tf: Timeframe, *, min_occupancy: float
) -> SessionCalendar | None:
    """One calendar from one stretch of timestamps."""
    weekday = index.weekday.to_numpy()
    seconds = (
        index.hour.to_numpy() * 3600 + index.minute.to_numpy() * 60 + index.second.to_numpy()
    )
    slot = seconds // int(tf.seconds)
    dates = index.tz_convert("UTC").normalize()

    frame = pd.DataFrame({"weekday": weekday, "slot": slot, "date": dates})
    observed = frame.groupby(["weekday", "slot"])["date"].nunique()

    # The denominator is the number of CALENDAR days of that weekday in the
    # span, not the number of days on which bars happened to appear. Using
    # observed days instead would declare a slot open on the strength of a
    # handful of exceptional sessions: MOEX traded a few weekend days in this
    # sample, and dividing 2 observed Saturdays by 3 observed Saturdays makes
    # Saturday look like a normal trading day.
    calendar_days = pd.date_range(dates.min(), dates.max(), freq="D")
    days_per_weekday = pd.Series(calendar_days.weekday).value_counts()
    denominator = (
        days_per_weekday.reindex(observed.index.get_level_values(0)).to_numpy().astype(float)
    )
    occupancy = observed / np.where(denominator > 0, denominator, np.nan)

    open_slots = frozenset(
        (int(day), int(s)) for (day, s), share in occupancy.items() if share >= min_occupancy
    )
    if not open_slots:
        return None
    return SessionCalendar(
        timeframe=tf,
        slots=open_slots,
        min_occupancy=min_occupancy,
        n_bars_observed=len(index),
        source="inferred",
    )


def session_from_hours(
    timeframe: Timeframe | str,
    *,
    hours: range | tuple[int, ...],
    weekdays: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> SessionCalendar:
    """Build a session from explicit UTC hours — for an authoritative calendar.

    Use this when the exchange's hours are known; inference is for when
    they are not.
    """
    tf = Timeframe.parse(timeframe)
    if tf.unit not in {"s", "m", "h"}:
        raise ConfigError("an explicit session calendar only applies to intraday timeframes")
    per_hour = max(round(3600 / tf.seconds), 1)
    slots = frozenset(
        (day, hour * per_hour + k)
        for day in weekdays
        for hour in hours
        for k in range(per_hour)
    )
    return SessionCalendar(timeframe=tf, slots=slots, source="explicit")
