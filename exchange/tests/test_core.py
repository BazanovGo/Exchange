"""Core: determinism, hashing, timeframes, environment capture."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exchange.core.config import build_config, load_config
from exchange.core.determinism import (
    DEFAULT_SEED,
    derive_seed,
    deterministic_mode,
    rng,
    spawn,
    temporary_seed,
)
from exchange.core.errors import ConfigError
from exchange.core.hashing import canonical_json, hash_config, hash_dataframe, short
from exchange.core.text import format_cell, markdown_table
from exchange.core.timeframe import (
    Timeframe,
    annualisation_factor,
    empirical_bars_per_year,
    infer_timeframe,
)


class TestDeterminism:
    def test_same_seed_same_numbers(self):
        assert rng(5).normal(size=10).tolist() == rng(5).normal(size=10).tolist()

    def test_different_seeds_differ(self):
        assert rng(5).normal(size=10).tolist() != rng(6).normal(size=10).tolist()

    def test_none_means_the_default_not_entropy(self):
        """An unseeded research run is a bug, so None resolves to a fixed seed."""
        assert rng(None).normal(size=5).tolist() == rng(DEFAULT_SEED).normal(size=5).tolist()

    def test_spawn_streams_are_independent_and_reproducible(self):
        first = [g.normal(size=4).tolist() for g in spawn(rng(1), 3)]
        second = [g.normal(size=4).tolist() for g in spawn(rng(1), 3)]
        assert first == second
        assert first[0] != first[1]

    def test_derive_seed_is_stable_across_processes(self):
        """Unlike hash(), which is salted per interpreter."""
        assert derive_seed("attack", base=7) == derive_seed("attack", base=7)
        assert derive_seed("attack", base=7) != derive_seed("attack", base=8)
        assert isinstance(derive_seed("x"), int)

    def test_deterministic_mode_reports_what_it_cannot_pin(self):
        report = deterministic_mode(123)
        assert report.seed == 123
        assert "seed=123" in report.summary()
        assert isinstance(report.fully_deterministic, bool)

    def test_temporary_seed_restores_global_state(self):
        np.random.seed(42)
        before = np.random.random()
        np.random.seed(42)
        with temporary_seed(7):
            np.random.random()
        assert np.random.random() == before


class TestHashing:
    def test_column_order_does_not_change_the_hash(self):
        a = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
        assert hash_dataframe(a) == hash_dataframe(a[["y", "x"]])

    def test_a_single_changed_value_changes_the_hash(self):
        a = pd.DataFrame({"x": [1.0, 2.0]})
        b = pd.DataFrame({"x": [1.0, 2.000000001]})
        assert hash_dataframe(a) != hash_dataframe(b)

    def test_index_participates(self):
        values = {"x": [1.0, 2.0]}
        a = pd.DataFrame(values, index=pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True))
        b = pd.DataFrame(values, index=pd.to_datetime(["2026-01-01", "2026-01-03"], utc=True))
        assert hash_dataframe(a) != hash_dataframe(b)

    def test_config_hash_ignores_key_order(self):
        assert hash_config({"a": 1, "b": 2}) == hash_config({"b": 2, "a": 1})

    def test_nan_round_trips_distinctly(self):
        assert "__nan__" in canonical_json({"x": float("nan")})
        assert hash_config({"x": float("nan")}) != hash_config({"x": None})

    def test_short_is_a_prefix(self):
        digest = hash_config({"a": 1})
        assert digest.startswith(short(digest))


class TestTimeframe:
    @pytest.mark.parametrize(
        ("label", "seconds"),
        [("1m", 60), ("5m", 300), ("1h", 3600), ("1d", 86_400), ("1w", 604_800)],
    )
    def test_parse_round_trip(self, label, seconds):
        tf = Timeframe.parse(label)
        assert tf.seconds == seconds
        assert Timeframe.parse(tf.label) == tf

    def test_rejects_nonsense(self):
        with pytest.raises(ConfigError):
            Timeframe.parse("1 fortnight")

    def test_infer_from_a_business_day_index(self):
        index = pd.date_range("2026-01-05", periods=60, freq="B", tz="UTC")
        assert infer_timeframe(index).label == "1d"

    def test_empirical_bars_per_year_is_measured_not_assumed(self):
        """A business-day index yields ~261 bars a year, not the nominal 365.

        Using 365 on daily bars overstates an annualised Sharpe by about
        18%, which is larger than most of the effects this framework is
        used to judge.
        """
        index = pd.date_range("2020-01-01", periods=1260, freq="B", tz="UTC")
        measured = empirical_bars_per_year(index)
        assert 258 < measured < 264
        assert measured < 300  # nowhere near the nominal calendar year

    def test_annualisation_prefers_the_series_own_index(self):
        index = pd.date_range("2020-01-01", periods=600, freq="B", tz="UTC")
        assert annualisation_factor(index) == pytest.approx(
            empirical_bars_per_year(index), rel=1e-9
        )
        assert annualisation_factor(index, override=252.0) == 252.0

    def test_pandas_3_safe_on_an_hourly_index(self):
        """Regression: nanosecond extraction must not use removed pandas APIs."""
        index = pd.date_range("2026-01-05", periods=200, freq="h", tz="UTC")
        assert infer_timeframe(index).label == "1h"
        assert empirical_bars_per_year(index) > 8000


class TestConfig:
    def test_defaults_load_without_a_file(self):
        config = load_config()
        assert config.seed == DEFAULT_SEED
        assert config.execution_scenario == "base"

    def test_unknown_top_level_key_is_refused(self):
        with pytest.raises(ConfigError, match="unknown configuration key"):
            build_config({"typo": 1})

    def test_unknown_section_key_is_refused(self):
        with pytest.raises(ConfigError, match="unknown key"):
            build_config({"funnel": {"is_fractoin": 0.6}})

    def test_optimistic_cannot_be_the_verdict_basis(self):
        with pytest.raises(ConfigError, match="cannot be the verdict basis"):
            build_config({"execution_scenario": "optimistic"})

    def test_seed_is_propagated_into_the_funnel(self):
        config = build_config({"seed": 99})
        assert config.funnel.seed == 99

    def test_fingerprint_changes_with_content(self):
        assert build_config({"seed": 1}).fingerprint != build_config({"seed": 2}).fingerprint

    def test_missing_explicit_file_is_an_error(self, tmp_path):
        with pytest.raises(ConfigError, match="does not exist"):
            load_config(tmp_path / "nope.yaml")

    def test_yaml_round_trip(self, tmp_path):
        config = build_config({"seed": 5, "funnel": {"is_fraction": 0.7}})
        path = config.save(tmp_path / "c.yaml")
        reloaded = load_config(path)
        assert reloaded.seed == 5
        assert reloaded.funnel.is_fraction == 0.7
        assert reloaded.fingerprint == config.fingerprint


class TestText:
    def test_markdown_table_needs_no_optional_dependency(self):
        frame = pd.DataFrame({"a": [1.5, np.nan], "b": [True, False]}, index=["x", "y"])
        rendered = markdown_table(frame)
        assert rendered.splitlines()[0].startswith("|")
        assert "1.5000" in rendered

    def test_non_finite_renders_blank_not_nan(self):
        assert format_cell(float("inf")) == ""
        assert format_cell(float("nan")) == ""

    def test_truncation_is_announced(self):
        frame = pd.DataFrame({"a": range(100)})
        assert "first 5 shown" in markdown_table(frame, max_rows=5)


class TestEnvironment:
    def test_capture_includes_tracked_packages(self):
        from exchange.core.env import capture_environment

        info = capture_environment()
        assert "numpy" in info.packages
        assert info.python_version

    def test_diff_reports_package_changes(self):
        from dataclasses import replace

        from exchange.core.env import capture_environment

        a = capture_environment()
        b = replace(a, packages={**a.packages, "numpy": "0.0.0"})
        assert "packages.numpy" in a.diff(b)
