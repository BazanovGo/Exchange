"""The experiment registry and the multiple-testing ledger."""

from __future__ import annotations

import json

import pytest

from exchange.core.errors import ConfigError, ReproducibilityError
from exchange.registry.experiment import (
    ExperimentRecord,
    compare_records,
    make_experiment_id,
    validate_record,
)
from exchange.registry.store import ExperimentStore


def _record(experiment_id="EXP-20260902-000001", **overrides):
    payload = {
        "experiment_id": experiment_id,
        "hypothesis": {
            "hypothesis_id": "HYP-1",
            "fingerprint": "f" * 32,
            "strategy": "ts_momentum",
            "param_space": {"lookback": [10, 20]},
            "fixed_params": {},
            "n_configurations": 2,
            "total_trials": 2,
        },
        "dataset": {"symbol": "NG", "data_hash": "d" * 32, "n_bars": 2000},
        "instrument": {"symbol": "NG", "multiplier": 10_000.0},
        "config": {"seed": 1, "execution_scenario": "base", "is_fraction": 0.6},
        "metrics": {"oos_sharpe": 1.25, "pbo": 0.1, "deflated_sharpe_ratio": 0.97},
        "verdict": {"label": "ROBUST", "level": 3},
        "best_params": {"lookback": 20},
        "environment": {"python_version": "3.12.3", "packages": {"numpy": "2.0.0"}},
        "seed": 1,
    }
    payload.update(overrides)
    return ExperimentRecord(**payload)


class TestIdentifiers:
    def test_format(self):
        assert make_experiment_id(157).startswith("EXP-")
        assert make_experiment_id(157).endswith("-000157")

    def test_a_malformed_id_is_refused(self):
        with pytest.raises(ConfigError, match="EXP-"):
            _record("experiment-1")

    def test_ids_sort_chronologically(self):
        ids = [make_experiment_id(i) for i in (3, 1, 2)]
        assert sorted(ids)[0].endswith("-000001")


class TestRecordValidation:
    def test_a_complete_record_validates(self):
        validate_record(_record())

    def test_a_missing_instrument_is_refused(self):
        """Re-running with a default multiplier silently changes the money."""
        with pytest.raises(ReproducibilityError, match="instrument"):
            validate_record(_record(instrument={}))

    def test_a_missing_data_hash_is_refused(self):
        with pytest.raises(ReproducibilityError, match="data"):
            validate_record(_record(dataset={"symbol": "NG"}))

    def test_a_missing_seed_is_refused(self):
        with pytest.raises(ReproducibilityError, match="seed"):
            validate_record(_record(seed=None))

    def test_missing_execution_assumptions_are_refused(self):
        with pytest.raises(ReproducibilityError, match="execution"):
            validate_record(_record(config={"seed": 1, "is_fraction": 0.6}))

    def test_a_missing_environment_is_refused(self):
        with pytest.raises(ReproducibilityError, match="environment"):
            validate_record(_record(environment={}))


class TestReproductionKey:
    def test_the_key_contains_the_inputs(self):
        key = _record().reproduction_key
        assert set(key) == {
            "hypothesis_fingerprint", "strategy", "param_space", "fixed_params",
            "data_hash", "instrument", "config", "seed",
        }

    def test_the_key_excludes_outputs_and_timestamps(self):
        key = _record().reproduction_key
        assert "metrics" not in key
        assert "verdict" not in key
        assert "created_at" not in key

    def test_changing_the_instrument_changes_the_key(self):
        a = _record()
        b = _record(instrument={"symbol": "NG", "multiplier": 1.0})
        assert a.reproduction_hash != b.reproduction_hash

    def test_changing_a_metric_does_not_change_the_key(self):
        a = _record()
        b = _record(metrics={"oos_sharpe": -99.0})
        assert a.reproduction_hash == b.reproduction_hash


class TestComparison:
    def test_identical_records_compare_exact(self):
        assert compare_records(_record(), _record()).status == "EXACT"

    def test_a_tiny_difference_is_equivalent(self):
        b = _record(metrics={"oos_sharpe": 1.25 + 1e-13, "pbo": 0.1,
                             "deflated_sharpe_ratio": 0.97})
        comparison = compare_records(_record(), b)
        assert comparison.status == "EQUIVALENT"
        assert comparison.reproduced

    def test_a_material_difference_is_divergent(self):
        b = _record(metrics={"oos_sharpe": 0.9, "pbo": 0.1, "deflated_sharpe_ratio": 0.97})
        comparison = compare_records(_record(), b)
        assert comparison.status == "DIVERGENT"
        assert not comparison.reproduced
        assert "oos_sharpe" in comparison.metric_differences

    def test_a_changed_verdict_is_divergent(self):
        comparison = compare_records(_record(), _record(verdict={"label": "WEAK", "level": 1}))
        assert comparison.status == "DIVERGENT"
        assert comparison.verdict_changed

    def test_different_parameters_are_divergent(self):
        comparison = compare_records(_record(), _record(best_params={"lookback": 10}))
        assert comparison.status == "DIVERGENT"
        assert comparison.params_changed

    def test_different_inputs_are_incomparable_not_failed(self):
        """A different experiment is not a failed reproduction.

        Conflating the two sends people looking for bugs in the code when
        somebody simply changed an input.
        """
        comparison = compare_records(_record(), _record(dataset={"symbol": "NG", "data_hash": "e" * 32}))
        assert comparison.status == "INCOMPARABLE"
        assert "data_hash" in comparison.input_differences

    def test_environment_differences_are_reported_not_failed(self):
        b = _record(environment={"python_version": "3.13.0", "packages": {"numpy": "2.1.0"}})
        comparison = compare_records(_record(), b)
        assert comparison.status == "EXACT"
        assert comparison.environment_differences


class TestStore:
    def test_save_and_load_round_trip(self, tmp_path):
        store = ExperimentStore(tmp_path)
        record = _record(store.next_id())
        store.save(record)
        assert store.load(record.experiment_id).metrics == record.metrics

    def test_ids_increment(self, tmp_path):
        store = ExperimentStore(tmp_path)
        first = store.next_id()
        store.save(_record(first))
        second = store.next_id()
        assert second != first
        assert int(second.split("-")[-1]) == int(first.split("-")[-1]) + 1

    def test_experiments_are_append_only(self, tmp_path):
        store = ExperimentStore(tmp_path)
        record = _record(store.next_id())
        store.save(record)
        with pytest.raises(ConfigError, match="append-only"):
            store.save(record)

    def test_an_invalid_record_is_never_written(self, tmp_path):
        store = ExperimentStore(tmp_path)
        broken = _record(store.next_id(), instrument={})
        with pytest.raises(ReproducibilityError):
            store.save(broken)
        assert not store.ids()

    def test_tables_and_reports_are_stored_beside_the_record(self, tmp_path):
        import pandas as pd

        store = ExperimentStore(tmp_path)
        record = _record(store.next_id())
        store.save(
            record,
            report="# report\n",
            tables={"sweep": pd.DataFrame({"is_sharpe": [1.0, 2.0]})},
        )
        directory = store.directory(record.experiment_id)
        assert (directory / "report.md").is_file()
        assert (directory / "sweep.csv").is_file()
        assert store.load(record.experiment_id).artefacts["sweep"] == "sweep.csv"

    def test_robustness_can_be_attached_afterwards(self, tmp_path):
        store = ExperimentStore(tmp_path)
        record = _record(store.next_id())
        store.save(record)
        store.attach_robustness(record.experiment_id, {"score": 0.8, "grade": "STURDY"})
        reloaded = store.load(record.experiment_id)
        assert reloaded.robustness["grade"] == "STURDY"
        assert (store.directory(record.experiment_id) / "robustness.json").is_file()

    def test_the_index_matches_the_records(self, tmp_path):
        store = ExperimentStore(tmp_path)
        for _ in range(3):
            store.save(_record(store.next_id()))
        index = store.index()
        assert len(index) == 3
        assert set(index["verdict"]) == {"ROBUST"}

    def test_the_index_can_be_rebuilt_from_the_records(self, tmp_path):
        store = ExperimentStore(tmp_path)
        store.save(_record(store.next_id()))
        store.index_path.unlink()
        store.rebuild_index()
        assert store.index_path.is_file()

    def test_an_unreadable_record_does_not_hide_the_others(self, tmp_path):
        store = ExperimentStore(tmp_path)
        good = _record(store.next_id())
        store.save(good)
        broken_dir = store.directory("EXP-20260902-999999")
        broken_dir.mkdir(parents=True)
        (broken_dir / "record.json").write_text("{not json")
        assert len(list(store.records())) == 1

    def test_find_filters_on_every_field(self, tmp_path):
        store = ExperimentStore(tmp_path)
        store.save(_record(store.next_id()))
        assert store.find(strategy="ts_momentum")
        assert store.find(verdict="ROBUST")
        assert not store.find(verdict="REJECTED")
        assert store.find(data_hash="d" * 32)


class TestLedger:
    def test_prior_trials_accumulate_per_dataset(self, tmp_path):
        """The tenth hypothesis on a dataset carries the first nine's burden."""
        store = ExperimentStore(tmp_path)
        for _ in range(3):
            store.save(_record(store.next_id()))
        assert store.prior_trials("d" * 32) == 6        # 3 experiments x 2 configurations

    def test_prior_trials_can_exclude_the_current_experiment(self, tmp_path):
        store = ExperimentStore(tmp_path)
        record = _record(store.next_id())
        store.save(record)
        assert store.prior_trials("d" * 32) == 2
        assert store.prior_trials("d" * 32, exclude=record.experiment_id) == 0

    def test_other_datasets_are_not_charged(self, tmp_path):
        store = ExperimentStore(tmp_path)
        store.save(_record(store.next_id()))
        assert store.prior_trials("e" * 32) == 0

    def test_trials_by_dataset_summarises_the_ledger(self, tmp_path):
        store = ExperimentStore(tmp_path)
        store.save(_record(store.next_id()))
        store.save(
            _record(store.next_id(), dataset={"symbol": "CL", "data_hash": "c" * 32, "n_bars": 10})
        )
        table = store.trials_by_dataset()
        assert len(table) == 2
        assert set(table["symbol"]) == {"NG", "CL"}

    def test_rejections_are_kept_and_counted(self, tmp_path):
        store = ExperimentStore(tmp_path)
        store.save(_record(store.next_id()))
        store.save(_record(store.next_id(), verdict={"label": "REJECTED", "level": 0}))
        assert store.verdict_counts() == {"REJECTED": 1, "ROBUST": 1}
        assert store.rejection_rate() == pytest.approx(0.5)

    def test_rejection_rate_is_nan_for_an_empty_store(self, tmp_path):
        import numpy as np

        assert np.isnan(ExperimentStore(tmp_path).rejection_rate())

    def test_summary_is_json_serialisable(self, tmp_path):
        store = ExperimentStore(tmp_path)
        store.save(_record(store.next_id()))
        assert json.loads(json.dumps(store.summary(), default=str))["n_experiments"] == 1

    def test_markdown_reports_the_rejection_rate(self, tmp_path):
        store = ExperimentStore(tmp_path)
        store.save(_record(store.next_id(), verdict={"label": "REJECTED", "level": 0}))
        assert "Rejection rate" in store.to_markdown()
