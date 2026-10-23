"""Trading sessions and session-aware gap detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exchange.core.errors import ConfigError
from exchange.data.quality import validate_bars
from exchange.data.sessions import (
    SessionCalendar,
    SessionSchedule,
    infer_session,
    session_from_hours,
)
from exchange.data.validation import check_gaps, check_session_coverage


def hourly_index(days: int, hours: range, *, start: str = "2024-01-01", weekdays=(0, 1, 2, 3, 4)):
    """A session-shaped hourly index: only the given hours, only weekdays."""
    stamps = []
    day = pd.Timestamp(start, tz="UTC")
    produced = 0
    while produced < days:
        if day.weekday() in weekdays:
            stamps += [day + pd.Timedelta(hours=h) for h in hours]
            produced += 1
        day += pd.Timedelta(days=1)
    return pd.DatetimeIndex(stamps, name="timestamp")


def bars_for(index: pd.DatetimeIndex) -> pd.DataFrame:
    price = 100.0 + np.cumsum(np.random.default_rng(1).normal(0, 0.1, len(index)))
    return pd.DataFrame(
        {
            "open": price, "high": price + 0.2, "low": price - 0.2, "close": price,
            "volume": 1_000.0,
        },
        index=index,
    )


class TestInference:
    def test_daily_series_declines_a_session(self):
        """Business-day counting is already the right measure there."""
        index = pd.date_range("2020-01-01", periods=1000, freq="B", tz="UTC")
        assert infer_session(index, "1d") is None

    def test_too_few_bars_declines(self):
        assert infer_session(hourly_index(5, range(8, 16)), "1h") is None

    def test_the_session_is_recovered(self):
        index = hourly_index(200, range(8, 17))
        schedule = infer_session(index, "1h", by=None)
        calendar = schedule.segments[0][1]
        assert calendar.weekdays == (0, 1, 2, 3, 4)
        assert sorted({slot for _, slot in calendar.slots}) == list(range(8, 17))

    def test_an_exceptional_weekend_session_is_not_declared_open(self):
        """The occupancy denominator is calendar days, not observed days.

        Dividing two observed Saturdays by three observed Saturdays makes
        Saturday look like a normal trading day; dividing by the number of
        calendar Saturdays does not.
        """
        index = hourly_index(200, range(8, 17))
        extra = pd.DatetimeIndex(
            [pd.Timestamp("2024-03-02", tz="UTC") + pd.Timedelta(hours=h) for h in range(8, 17)]
        )
        combined = index.append(extra).sort_values()
        calendar = infer_session(combined, "1h", by=None).segments[0][1]
        assert 5 not in calendar.weekdays            # Saturday stays closed
        assert calendar.is_open(extra).sum() == 0


class TestSpacing:
    def test_an_overnight_break_is_one_bar(self):
        index = hourly_index(60, range(8, 17))
        schedule = infer_session(index, "1h", by=None)
        spacing, off = schedule.spacing(index)
        assert spacing.max() == pytest.approx(1.0)
        assert off.sum() == 0

    def test_a_weekend_is_one_bar(self):
        index = hourly_index(60, range(8, 17))
        schedule = infer_session(index, "1h", by=None)
        spacing, _off = schedule.spacing(index)
        fridays = index[:-1][index[:-1].weekday == 4]
        assert len(fridays) > 5
        assert spacing.max() == pytest.approx(1.0)

    def test_a_real_missing_hour_is_two_bars(self):
        index = hourly_index(60, range(8, 17))
        schedule = infer_session(index, "1h", by=None)
        holed = index.delete(100)
        spacing, _off = schedule.spacing(holed)
        assert spacing.max() == pytest.approx(2.0)

    def test_an_off_session_bar_is_flagged_not_snapped(self):
        index = hourly_index(60, range(8, 17))
        schedule = infer_session(index, "1h", by=None)
        intruder = pd.DatetimeIndex([index[0] + pd.Timedelta(hours=-3)])
        combined = index.append(intruder).sort_values()
        _spacing, off = schedule.spacing(combined)
        assert off.sum() == 1


class TestSchedule:
    def test_a_session_change_is_detected(self):
        """An exchange that extends its hours must not look like a data hole."""
        first = hourly_index(240, range(10, 17), start="2024-01-01")
        second = hourly_index(240, range(7, 17), start="2025-01-01")
        index = first.append(second)
        schedule = infer_session(index, "1h")
        assert schedule.changed
        assert len(schedule.segments) >= 2
        spacing, off = schedule.spacing(index)
        # The extra morning hours are normal in the second period, so they
        # are neither gaps nor off-session bars.
        assert (spacing > 1.5).mean() < 0.01
        assert off.mean() < 0.01

    def test_a_single_calendar_misreads_a_changed_session(self):
        """Why the schedule is piecewise, measured rather than asserted."""
        first = hourly_index(240, range(10, 17), start="2024-01-01")
        second = hourly_index(240, range(7, 17), start="2025-01-01")
        index = first.append(second)
        one = infer_session(index, "1h", by=None)
        piecewise = infer_session(index, "1h")
        off_one = one.spacing(index)[1].mean()
        off_piecewise = piecewise.spacing(index)[1].mean()
        assert off_one > off_piecewise

    def test_explicit_calendar_round_trip(self):
        calendar = session_from_hours("1h", hours=range(8, 17))
        assert calendar.source == "explicit"
        assert calendar.slots_per_week == 9 * 5
        assert "Mon-Fri" in calendar.describe()

    def test_an_empty_calendar_is_refused(self):
        from exchange.core.timeframe import Timeframe

        with pytest.raises(ConfigError):
            SessionCalendar(timeframe=Timeframe.parse("1h"), slots=frozenset())

    def test_an_empty_schedule_is_refused(self):
        with pytest.raises(ConfigError):
            SessionSchedule(())

    def test_schedule_serialises(self):
        index = hourly_index(240, range(9, 17))
        payload = infer_session(index, "1h").to_dict()
        assert payload["timeframe"] == "1h"
        assert "description" in payload


class TestValidationIntegration:
    def test_the_session_removes_the_false_gaps(self):
        """The whole point: 1,800 warnings on real data became 293."""
        index = hourly_index(300, range(8, 17))
        bars = bars_for(index)
        without = check_gaps(bars, session=None)
        schedule = infer_session(index, "1h")
        with_session = check_gaps(bars, session=schedule)
        assert not without.passed                 # every overnight break is a "gap"
        assert with_session.passed

    def test_validate_bars_infers_a_session_automatically(self):
        bars = bars_for(hourly_index(300, range(8, 17)))
        report = validate_bars(bars, symbol="X", timeframe="1h")
        gaps = next(f for f in report.findings if f.check == "gaps")
        coverage = next(f for f in report.findings if f.check == "session_coverage")
        assert gaps.passed
        assert coverage.passed

    def test_session_can_be_switched_off(self):
        bars = bars_for(hourly_index(300, range(8, 17)))
        report = validate_bars(bars, symbol="X", timeframe="1h", session=None)
        gaps = next(f for f in report.findings if f.check == "gaps")
        assert not gaps.passed

    def test_daily_behaviour_is_unchanged(self):
        """Business-day spacing must keep working exactly as before."""
        index = pd.date_range("2020-01-01", periods=600, freq="B", tz="UTC", name="timestamp")
        report = validate_bars(bars_for(index), symbol="D", timeframe="1d")
        gaps = next(f for f in report.findings if f.check == "gaps")
        coverage = next(f for f in report.findings if f.check == "session_coverage")
        assert gaps.passed
        assert coverage.passed
        assert "daily or coarser" in coverage.message

    def test_a_large_off_session_share_is_a_warning(self):
        from exchange.data.validation import Severity

        index = hourly_index(300, range(8, 17))
        schedule = infer_session(index, "1h")
        noon = pd.DatetimeIndex(
            [index[0].normalize() + pd.Timedelta(days=d, hours=2) for d in range(400)]
        )
        combined = index.append(noon).sort_values()
        finding = check_session_coverage(bars_for(combined), session=schedule)
        assert not finding.passed
        assert finding.severity == Severity.WARNING
