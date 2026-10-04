"""Data engine: schema, quality gates, contracts, rolls, datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exchange.core.errors import DataError, DataQualityError
from exchange.data.contracts import (
    BUILTIN_ROOTS,
    ContractRegistry,
    contract_symbol,
    parse_contract_symbol,
)
from exchange.data.continuous import (
    build_continuous,
    n_days_before_expiration,
    roll_gap_diagnostics,
    volume_roll,
)
from exchange.data.dataset import (
    dataset_from_continuous,
    make_dataset,
    save_dataset,
    verify_dataset_hash,
)
from exchange.data.providers import InMemoryProvider, build_provider
from exchange.data.schema import OHLCV, assert_canonical, is_canonical, normalise_bars
from exchange.data.quality import validate_bars
from exchange.data.validation import Severity, run_all_checks, worst_severity
from exchange.synthetic.generators import futures_ladder, random_walk


class TestSchema:
    def test_canonical_output(self, rw_bars):
        assert is_canonical(rw_bars)
        assert list(rw_bars.columns) == list(OHLCV)
        assert rw_bars.index.tz is not None

    def test_column_aliases_are_renamed(self):
        frame = pd.DataFrame(
            {
                "Date": ["2026-01-05", "2026-01-06"],
                "Open": [1.0, 1.1],
                "High": [1.2, 1.3],
                "Low": [0.9, 1.0],
                "Settle": [1.1, 1.2],
                "Volume": [10, 20],
            }
        )
        out = normalise_bars(frame)
        assert list(out.columns)[:5] == list(OHLCV)
        assert out.index[0] == pd.Timestamp("2026-01-05", tz="UTC")
        assert out["close"].tolist() == [1.1, 1.2]   # 'Settle' is the futures close

    def test_adjusted_close_does_not_replace_the_traded_close(self):
        """An adjusted close is a different series and must not be the close.

        Yahoo-style files carry both. Silently using the adjusted column as
        the traded price injects dividend and split adjustments into the
        price a backtest fills at, which is a small, persistent lookahead.
        """
        frame = pd.DataFrame(
            {
                "Date": ["2026-01-05", "2026-01-06"],
                "Open": [1.0, 1.1], "High": [1.2, 1.3], "Low": [0.9, 1.0],
                "Close": [1.1, 1.2], "Adj Close": [0.5, 0.6], "Volume": [10, 20],
            }
        )
        out = normalise_bars(frame)
        assert out["close"].tolist() == [1.1, 1.2]
        assert "adj_close" not in OHLCV

    def test_naive_timestamps_are_localised_explicitly(self):
        frame = pd.DataFrame(
            {
                "timestamp": ["2026-01-05 09:30", "2026-01-05 10:30"],
                "open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0],
                "close": [1.0, 1.0], "volume": [1, 1],
            }
        )
        out = normalise_bars(frame, source_tz="America/New_York")
        assert out.index[0] == pd.Timestamp("2026-01-05 14:30", tz="UTC")

    def test_numeric_index_is_refused_not_silently_converted(self):
        """Regression for a silent-corruption bug.

        ``pd.to_datetime`` on a RangeIndex produces 1970 epoch-nanosecond
        timestamps that pass every downstream quality check, so a CSV whose
        index column was not recognised used to become a valid-looking
        dataset dated 1970.
        """
        frame = pd.DataFrame(
            {"open": [1.0, 1.0], "high": [1.0, 1.0], "low": [1.0, 1.0],
             "close": [1.0, 1.0], "volume": [1, 1]}
        )
        with pytest.raises(DataError, match="timestamp"):
            normalise_bars(frame)

    def test_unnamed_index_column_from_a_csv_round_trip(self, tmp_path, rw_bars):
        """A CSV written with its index must come back with real timestamps."""
        path = tmp_path / "bars.csv"
        rw_bars.reset_index().to_csv(path, index=True)   # creates 'Unnamed: 0'
        out = normalise_bars(pd.read_csv(path))
        assert out.index[0] == rw_bars.index[0]
        assert out.index[-1] == rw_bars.index[-1]

    def test_assert_canonical_rejects_a_missing_close(self, rw_bars):
        with pytest.raises(DataError, match="close"):
            assert_canonical(rw_bars.drop(columns=["close"]))

    def test_assert_canonical_rejects_naive_timestamps(self, rw_bars):
        naive = rw_bars.copy()
        naive.index = naive.index.tz_localize(None)
        with pytest.raises(DataError, match="timezone-aware"):
            assert_canonical(naive)

    def test_assert_canonical_rejects_unsorted_bars(self, rw_bars):
        with pytest.raises(DataError):
            assert_canonical(rw_bars.iloc[::-1])


class TestValidation:
    def test_clean_series_has_no_errors(self, rw_bars):
        findings = run_all_checks(rw_bars)
        assert worst_severity(findings) < Severity.ERROR

    def test_duplicate_timestamps_are_an_error(self, rw_bars):
        broken = pd.concat([rw_bars, rw_bars.iloc[[5]]]).sort_index()
        findings = run_all_checks(broken)
        assert any(f.check == "duplicates" and f.severity >= Severity.ERROR for f in findings)

    def test_ohlc_violation_is_an_error(self, rw_bars):
        broken = rw_bars.copy()
        broken.iloc[10, broken.columns.get_loc("high")] = broken["low"].iloc[10] - 1.0
        findings = run_all_checks(broken)
        assert any(f.check == "ohlc_consistency" and f.severity >= Severity.ERROR for f in findings)

    def test_negative_price_is_an_error(self, rw_bars):
        broken = rw_bars.copy()
        broken.iloc[3, :4] = -1.0
        findings = run_all_checks(broken)
        assert any(f.severity >= Severity.ERROR for f in findings)

    def test_weekends_are_not_reported_as_gaps(self, rw_bars):
        """Business-day data has a two-day spacing every week, by definition."""
        findings = validate_bars(rw_bars, symbol="RW", timeframe="1d").findings
        gaps = [f for f in findings if f.check in {"gaps", "missing_bars"}]
        assert all(f.severity < Severity.ERROR for f in gaps)

    def test_a_real_gap_is_reported(self, rw_bars):
        broken = pd.concat([rw_bars.iloc[:100], rw_bars.iloc[160:]])
        findings = validate_bars(broken, symbol="RW", timeframe="1d").findings
        assert any(f.check == "gaps" and f.severity >= Severity.WARNING for f in findings)

    def test_strict_dataset_refuses_broken_data(self, rw_bars):
        broken = rw_bars.copy()
        broken.iloc[10, broken.columns.get_loc("high")] = broken["low"].iloc[10] - 1.0
        with pytest.raises(DataQualityError):
            make_dataset(broken, symbol="RW")

    def test_non_strict_dataset_still_records_the_failure(self, rw_bars):
        broken = rw_bars.copy()
        broken.iloc[10, broken.columns.get_loc("high")] = broken["low"].iloc[10] - 1.0
        dataset = make_dataset(broken, symbol="RW", strict=False)
        assert not dataset.quality.ok
        assert dataset.quality.errors


class TestDataset:
    def test_hash_is_content_based(self, rw_bars):
        a = make_dataset(rw_bars, symbol="RW")
        b = make_dataset(rw_bars.copy(), symbol="DIFFERENT_NAME")
        assert a.data_hash == b.data_hash

    def test_slice_rehashes(self, rw_dataset):
        part = rw_dataset.slice(end=rw_dataset.bars.index[500])
        assert part.data_hash != rw_dataset.data_hash
        assert part.n_bars < rw_dataset.n_bars

    def test_save_writes_a_provenance_sidecar(self, tmp_path, rw_dataset):
        path = save_dataset(rw_dataset, tmp_path / "rw.csv")
        assert path.with_name(path.name + ".provenance.json").is_file()

    def test_hash_verification_catches_a_changed_dataset(self, rw_dataset):
        from dataclasses import replace

        verify_dataset_hash(rw_dataset, rw_dataset.data_hash)   # the happy path is silent
        tampered = replace(rw_dataset, bars=rw_dataset.bars * 1.0001)
        with pytest.raises(DataQualityError, match="hash mismatch"):
            verify_dataset_hash(tampered, rw_dataset.data_hash)

    def test_bars_per_year_is_empirical(self, rw_dataset):
        assert 258 < rw_dataset.bars_per_year < 264

    def test_provider_round_trip(self, rw_bars):
        provider = InMemoryProvider({"RW": rw_bars})
        assert len(provider.fetch("RW", timeframe="1d")) == len(rw_bars)

    def test_csv_provider_is_buildable_from_a_config_mapping(self, tmp_path, rw_bars):
        rw_bars.to_csv(tmp_path / "RW.csv")
        provider = build_provider({"name": "csv", "base_dir": str(tmp_path)})
        assert len(provider.fetch("RW", timeframe="1d")) == len(rw_bars)


class TestContracts:
    @pytest.mark.parametrize(
        ("symbol", "root", "month", "year"),
        [
            ("NGV26", "NG", 10, 2026),
            ("NGX26", "NG", 11, 2026),
            ("NGZ26", "NG", 12, 2026),
            ("NGF27", "NG", 1, 2027),
            ("CLH25", "CL", 3, 2025),
        ],
    )
    def test_symbol_parsing(self, symbol, root, month, year):
        parsed_root, parsed_month = parse_contract_symbol(symbol)
        assert parsed_root == root
        assert parsed_month.month == month
        assert parsed_month.year == year
        assert contract_symbol(root, year, month) == symbol

    def test_bad_symbol_is_refused(self):
        from exchange.core.errors import ConfigError

        with pytest.raises(ConfigError, match="cannot parse"):
            parse_contract_symbol("NOTACONTRACT")

    def test_natural_gas_specification(self):
        spec = ContractRegistry().get("NGV26")
        assert spec.multiplier == 10_000.0
        assert spec.tick_size == pytest.approx(0.001)
        assert spec.money_per_tick == pytest.approx(10.0)

    def test_money_per_tick_follows_the_declared_tick_value(self):
        """Where a root declares a tick value, that value wins.

        Corn is quoted in cents per bushel, so tick_size * multiplier is
        1,250 while a tick is worth $12.50. Deriving the money per tick
        arithmetically would overstate corn P&L by a factor of 100, which
        is why the specification carries the exchange's own figure.
        """
        registry = ContractRegistry()
        for root, spec in BUILTIN_ROOTS.items():
            month = next(iter(spec.months)) if spec.months else "M"
            built = registry.get(f"{root}{month}26")
            expected = (
                spec.tick_value if spec.tick_value is not None
                else spec.tick_size * spec.multiplier
            )
            assert built.money_per_tick == pytest.approx(expected)
        assert registry.get("ZCZ26").money_per_tick == pytest.approx(12.5)

    def test_rounding_to_the_tick(self):
        spec = ContractRegistry().get("NGV26")
        assert spec.round_to_tick(3.14159) == pytest.approx(3.142)

    def test_unknown_root_is_refused_not_guessed(self):
        """A silently invented multiplier is worse than an error."""
        from exchange.core.errors import ConfigError

        with pytest.raises(ConfigError, match="unknown futures root"):
            ContractRegistry().get("XYZ26")

    def test_a_derived_spec_says_so_and_invents_no_dates(self):
        """A registry-derived spec carries the root's economics, not made-up dates.

        Expiry, first notice and last trade come from the exchange
        calendar. Guessing them would silently move every roll, so the
        derived spec is flagged and leaves them unset until supplied.
        """
        spec = ContractRegistry().get("NGM26")
        assert spec.multiplier == 10_000.0
        assert spec.meta.get("derived") is True
        assert spec.meta.get("dates_estimated") is True


class TestContinuous:
    @pytest.fixture(scope="class")
    @staticmethod
    def ladder():
        return futures_ladder(n_contracts=8, seed=5, term_premium=0.02)

    def test_volume_roll_produces_several_rolls(self, ladder):
        panel, specs, _truth = ladder
        series = build_continuous(panel, roll=volume_roll(), adjustment="ratio", specs=specs)
        assert len(series.provenance.rolls) >= 3
        assert series.bars.index.is_monotonic_increasing

    def test_every_roll_is_recorded_with_its_reason(self, ladder):
        panel, specs, _truth = ladder
        series = build_continuous(panel, roll=volume_roll(), adjustment="ratio", specs=specs)
        for event in series.provenance.rolls:
            assert event.from_contract != event.to_contract
            assert np.isfinite(event.gap)
            assert event.ratio > 0
            assert isinstance(event.overlap_available, bool)

    def test_contango_ladder_has_no_free_trend_after_adjustment(self, ladder):
        """The splice must not manufacture return.

        An unadjusted series of a contango ladder shows a large downward
        drift from the roll gaps alone, and a naively 'adjusted' one can
        show the opposite. A correct ratio adjustment leaves the tradeable
        return close to the underlying's own, which here is near zero.
        """
        panel, specs, _truth = ladder
        adjusted = build_continuous(panel, roll=volume_roll(), adjustment="ratio", specs=specs)
        tradeable = adjusted.tradeable_returns()
        total = float((1.0 + tradeable).prod() - 1.0)
        assert abs(total) < 0.5

    def test_unadjusted_series_masks_roll_bars(self, ladder):
        panel, specs, _truth = ladder
        series = build_continuous(panel, roll=volume_roll(), adjustment="none", specs=specs)
        returns = series.tradeable_returns()
        roll_times = [event.timestamp for event in series.provenance.rolls]
        masked = [t for t in roll_times if t in returns.index]
        assert masked
        # NaN, never zero: a zero would claim a flat bar happened.
        assert returns.loc[masked].isna().all()

    def test_roll_gap_diagnostics_quantify_the_splice(self, ladder):
        panel, specs, _truth = ladder
        series = build_continuous(panel, roll=volume_roll(), adjustment="ratio", specs=specs)
        table = roll_gap_diagnostics(series)
        assert not table.empty
        assert "gap_bps" in table.columns
        assert np.isfinite(table["gap_bps"]).all()

    def test_n_days_rule_never_holds_past_expiry(self, ladder):
        panel, specs, _truth = ladder
        series = build_continuous(
            panel, roll=n_days_before_expiration(5), adjustment="ratio", specs=specs
        )
        for segment in series.provenance.segments:
            reference = specs[segment.contract].roll_reference_date
            assert segment.end.date() <= reference

    def test_adjustment_choice_changes_the_history(self, ladder):
        """Measured, not assumed: this difference is the artefact's size."""
        panel, specs, _truth = ladder
        built = {
            adjustment: build_continuous(
                panel, roll=volume_roll(), adjustment=adjustment, specs=specs
            ).bars["close"]
            for adjustment in ("none", "backward", "ratio")
        }
        totals = {k: float(v.iloc[-1] / v.iloc[0] - 1.0) for k, v in built.items()}
        assert max(totals.values()) - min(totals.values()) > 0.01

    def test_dataset_from_continuous_keeps_provenance(self, ladder):
        panel, specs, _truth = ladder
        series = build_continuous(panel, roll=volume_roll(), adjustment="ratio", specs=specs)
        dataset = dataset_from_continuous(series)
        assert dataset.provenance["adjustment"] == "ratio"
        assert dataset.provenance["continuous"]["rolls"]


class TestSyntheticGenerators:
    def test_random_walk_has_no_edge_by_construction(self):
        series = random_walk(n=500, seed=3)
        assert series.truth.has_edge is False

    def test_signal_plus_noise_states_its_own_sharpe(self, edge_series):
        truth = edge_series.truth
        assert truth.has_edge is True
        assert truth.params["sharpe_per_sqrt_bar"] == pytest.approx(0.25, rel=1e-6)

    def test_generators_are_reproducible(self):
        a = random_walk(n=200, seed=42).bars
        b = random_walk(n=200, seed=42).bars
        pd.testing.assert_frame_equal(a, b)
