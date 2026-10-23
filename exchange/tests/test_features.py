"""Fundamental features, publication lags and curve construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exchange.core.errors import ConfigError, DataError
from exchange.features.catalogue import (
    ALL_FEATURES,
    CRUDE_FEATURES,
    NG_FEATURES,
    DerivedFeature,
    FeatureSpec,
    catalogue,
    feature_spec,
    features_for,
)
from exchange.features.curve import CURVE_COLUMNS, curve_features, seasonal_spread
from exchange.features.providers import (
    CsvFeatureProvider,
    FeatureStore,
    InMemoryFeatureProvider,
    SyntheticFeatureProvider,
    build_feature_provider,
)
from exchange.synthetic.generators import futures_ladder, random_walk


class TestCatalogue:
    def test_both_markets_are_covered(self):
        assert len(NG_FEATURES) >= 16
        assert len(CRUDE_FEATURES) >= 16
        assert len(ALL_FEATURES) == len(NG_FEATURES) + len(CRUDE_FEATURES)

    def test_every_weekly_series_carries_a_real_publication_lag(self):
        """A weekly series with a zero lag is a lookahead of about a week."""
        for spec in ALL_FEATURES:
            if spec.frequency == "weekly":
                assert spec.lag_days >= 1.0, spec.key

    def test_eia_storage_is_lagged_six_days(self):
        """Week ending Friday, released the following Thursday 10:30 ET."""
        assert feature_spec("ng_storage").lag_days == 6.0

    def test_monthly_production_is_lagged_two_months(self):
        assert feature_spec("ng_production_914").lag_days >= 55.0

    def test_revised_series_default_to_the_first_print(self):
        """A revised value did not exist at the timestamp it carries."""
        for key in ("ng_storage", "cl_crude_stocks", "cl_us_production"):
            assert feature_spec(key).revision_policy == "first"

    def test_exchange_settled_series_are_not_revised(self):
        for key in ("ng_curve_slope", "cl_curve_slope", "cl_crack_spread"):
            assert feature_spec(key).revision_policy == "latest"
            assert feature_spec(key).lag_days == 0.0

    def test_every_spec_states_a_mechanism(self):
        for spec in ALL_FEATURES:
            assert len(spec.mechanism) >= 40, spec.key

    def test_a_negative_lag_is_refused(self):
        with pytest.raises(ConfigError, match="lookahead"):
            FeatureSpec(
                key="impossible", name="x", market="natural_gas", category="inventory",
                frequency="weekly", publication_lag=pd.Timedelta(days=-1), unit="Bcf",
                source="nowhere",
                mechanism="a" * 50,
            )

    def test_a_spec_without_a_mechanism_is_refused(self):
        with pytest.raises(ConfigError, match="mechanism"):
            FeatureSpec(
                key="vague", name="x", market="crude_oil", category="supply",
                frequency="daily", publication_lag=pd.Timedelta(days=1), unit="bbl",
                source="somewhere", mechanism="it correlates",
            )

    def test_unknown_feature_is_refused(self):
        with pytest.raises(ConfigError, match="unknown feature"):
            feature_spec("ng_vibes")

    def test_catalogue_filters_by_market_and_category(self):
        table = catalogue(market="natural_gas", category="weather")
        assert not table.empty
        assert (table["market"] == "natural_gas").all()
        assert (table["category"] == "weather").all()

    def test_features_for_rejects_an_unknown_market(self):
        with pytest.raises(ConfigError):
            features_for("uranium")


class TestProviders:
    def test_in_memory_provider_round_trip(self):
        index = pd.date_range("2026-01-05", periods=10, freq="D", tz="UTC")
        values = pd.Series(np.arange(10.0), index=index, name="ng_storage")
        provider = InMemoryFeatureProvider(data={"ng_storage": values})
        assert provider.keys() == ["ng_storage"]
        assert len(provider.fetch("ng_storage")) == 10

    def test_csv_provider_reads_a_directory(self, tmp_path):
        index = pd.date_range("2026-01-05", periods=10, freq="D", tz="UTC")
        frame = pd.DataFrame({"timestamp": index, "value": np.arange(10.0)})
        frame.to_csv(tmp_path / "ng_storage.csv", index=False)
        provider = CsvFeatureProvider(directory=tmp_path)
        assert provider.keys() == ["ng_storage"]
        assert len(provider.fetch("ng_storage")) == 10

    def test_csv_provider_honours_a_real_availability_column(self, tmp_path):
        """A true publication calendar beats a constant lag."""
        index = pd.date_range("2026-01-05", periods=6, freq="D", tz="UTC")
        frame = pd.DataFrame(
            {
                "timestamp": index,
                "value": np.arange(6.0),
                "available_time": index + pd.Timedelta(days=3),
            }
        )
        frame.to_csv(tmp_path / "ng_storage.csv", index=False)
        store = FeatureStore(provider=CsvFeatureProvider(directory=tmp_path))
        pit = store.point_in_time("ng_storage")
        lags = pit.frame["available_time"] - pit.frame.index
        assert (lags == pd.Timedelta(days=3)).all()   # not the catalogue's six

    def test_build_from_a_config_mapping(self):
        provider = build_feature_provider({"kind": "memory", "data": {}})
        assert isinstance(provider, InMemoryFeatureProvider)

    def test_unknown_provider_is_refused(self):
        with pytest.raises(ConfigError, match="unknown feature provider"):
            build_feature_provider("bloomberg_terminal")

    def test_a_leaky_synthetic_feature_needs_an_explicit_opt_in(self):
        bars = random_walk(n=200, seed=1).bars
        with pytest.raises(ConfigError, match="allow_leaky"):
            SyntheticFeatureProvider(bars=bars, kinds={"ng_storage": "leaky"})

    def test_unknown_synthetic_kind_is_refused(self):
        bars = random_walk(n=200, seed=1).bars
        with pytest.raises(ConfigError):
            SyntheticFeatureProvider(bars=bars, kinds={"ng_storage": "magic"})


class TestFeatureStore:
    @pytest.fixture
    def bars(self):
        return random_walk(n=800, seed=3).bars

    def test_alignment_covers_the_bar_index(self, bars):
        store = FeatureStore(
            provider=SyntheticFeatureProvider(
                bars=bars, kinds={"ng_storage": "noise", "ng_hdd": "noise"}
            )
        )
        frame = store.align(bars, ["ng_storage", "ng_hdd"])
        assert frame.index.equals(bars.index)
        assert list(frame.columns) == ["ng_storage", "ng_hdd"]

    def test_a_lagged_feature_repeats_across_bars(self, bars):
        """Six-day-old data cannot change every day."""
        store = FeatureStore(
            provider=SyntheticFeatureProvider(bars=bars, kinds={"ng_storage": "noise"})
        )
        frame = store.align(bars, ["ng_storage"])
        changes = (frame["ng_storage"].diff() != 0).mean()
        assert changes < 0.95

    def test_coverage_reports_what_the_provider_lacks(self, bars):
        store = FeatureStore(
            provider=SyntheticFeatureProvider(bars=bars, kinds={"ng_storage": "noise"})
        )
        coverage = store.coverage("natural_gas")
        assert coverage.loc["ng_storage", "available"]
        assert not coverage.loc["ng_hdd", "available"]

    def test_derived_features_are_computed_on_aligned_inputs(self, bars):
        """A derived feature cannot be less lagged than its inputs."""
        store = FeatureStore(
            provider=SyntheticFeatureProvider(
                bars=bars, kinds={"ng_storage": "noise", "ng_hdd": "noise"}
            )
        )
        derived = DerivedFeature(
            key="storage_minus_hdd",
            inputs=("ng_storage", "ng_hdd"),
            transform=lambda frame: frame["ng_storage"] - frame["ng_hdd"],
            description="difference of two aligned features",
        )
        frame = store.align(bars, ["ng_storage", "ng_hdd"], derived=[derived])
        assert "storage_minus_hdd" in frame.columns
        expected = frame["ng_storage"] - frame["ng_hdd"]
        pd.testing.assert_series_equal(
            frame["storage_minus_hdd"], expected, check_names=False
        )

    def test_a_derived_feature_with_a_missing_input_is_refused(self, bars):
        store = FeatureStore(
            provider=SyntheticFeatureProvider(bars=bars, kinds={"ng_storage": "noise"})
        )
        derived = DerivedFeature(
            key="broken", inputs=("ng_storage", "ng_lng_feedgas"),
            transform=lambda frame: frame.sum(axis=1), description="missing input",
        )
        with pytest.raises(ConfigError, match="missing input"):
            store.align(bars, ["ng_storage"], derived=[derived])

    def test_leakage_report_grades_every_column(self, bars):
        store = FeatureStore(
            provider=SyntheticFeatureProvider(bars=bars, kinds={"ng_storage": "noise"})
        )
        frame = store.align(bars, ["ng_storage"])
        findings = store.leakage_report(bars, frame)
        assert {f.column for f in findings} == {"ng_storage"}


class TestCurveFeatures:
    @pytest.fixture(scope="class")
    @staticmethod
    def ladder():
        return futures_ladder(n_contracts=8, seed=5, term_premium=0.02)

    def test_all_columns_are_produced(self, ladder):
        panel, specs, _truth = ladder
        frame = curve_features(panel, specs=specs)
        assert list(frame.columns) == list(CURVE_COLUMNS)

    def test_contango_is_detected_in_a_contango_ladder(self, ladder):
        panel, specs, _truth = ladder
        frame = curve_features(panel, specs=specs)
        assert frame["contango"].mean() > 0.7
        assert frame["roll_yield"].mean() < 0.0      # negative carry for a long

    def test_backwardation_flips_the_sign(self):
        panel, specs, _truth = futures_ladder(n_contracts=8, seed=5, term_premium=-0.02)
        frame = curve_features(panel, specs=specs)
        assert frame["contango"].mean() < 0.3
        assert frame["roll_yield"].mean() > 0.0

    def test_the_ladder_is_rebuilt_row_by_row(self, ladder):
        """A fixed column assignment breaks the moment a contract expires.

        The front month must change over time, so the number of quoted
        tenors and the days-to-expiry must both vary.
        """
        panel, specs, _truth = ladder
        frame = curve_features(panel, specs=specs)
        assert frame["days_to_m1_expiry"].nunique() > 10
        assert frame["n_tenors"].nunique() > 1
        assert (frame["days_to_m1_expiry"].dropna() >= 0).all()

    def test_an_unknown_contract_specification_is_refused(self, ladder):
        panel, specs, _truth = ladder
        partial = {k: v for k, v in list(specs.items())[:2]}
        with pytest.raises(DataError, match="specification"):
            curve_features(panel, specs=partial)

    def test_an_empty_panel_is_refused(self):
        with pytest.raises(DataError):
            curve_features({})

    def test_seasonal_spread_uses_fixed_contracts(self, ladder):
        panel, _specs, _truth = ladder
        keys = sorted(panel)
        spread = seasonal_spread(panel, keys[0], keys[1])
        assert not spread.empty
        assert spread.name == f"{keys[0]}-{keys[1]}"

    def test_seasonal_spread_refuses_an_absent_contract(self, ladder):
        panel, _specs, _truth = ladder
        with pytest.raises(DataError):
            seasonal_spread(panel, sorted(panel)[0], "NGZ99")
