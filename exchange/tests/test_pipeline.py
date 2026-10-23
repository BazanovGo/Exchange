"""The funnel, the verdict logic and the hypothesis contract.

The decisive test in this module is
``test_the_sweep_contains_no_out_of_sample_column``: the audited project's
nineteen published verdicts all rested on selecting a configuration using
an out-of-sample score and then reporting that score as out-of-sample. Here
the separation is structural, and this is where it is proved.
"""

from __future__ import annotations

import pytest

from exchange.backtest.engine import InstrumentSpec
from exchange.core.errors import ConfigError
from exchange.data.dataset import make_dataset
from exchange.funnel.hypothesis import (
    DEFAULT_FALSIFICATION,
    FalsificationCriterion,
    Hypothesis,
    make_hypothesis,
)
from exchange.funnel.pipeline import FunnelConfig, run_funnel
from exchange.funnel.verdict import Gate, VerdictLevel, decide_verdict
from exchange.synthetic.generators import random_walk, signal_plus_noise

FAST = FunnelConfig(n_bootstrap=150, n_monte_carlo=150)


def _hypothesis(**overrides):
    payload = {
        "statement": "The trailing five-bar return sign predicts the next bar positively.",
        "mechanism": (
            "Information reaches slower participants over several bars, so their flow arrives "
            "after the move starts and extends it."
        ),
        "expected_regime": "Trending regimes; it should fail on a random walk.",
        "strategy": "ts_momentum",
        "param_space": {"lookback": [3, 5, 10, 20]},
    }
    payload.update(overrides)
    return make_hypothesis("HYP-PIPE", **payload)


class TestHypothesisContract:
    def test_a_complete_hypothesis_is_accepted(self):
        assert _hypothesis().n_configurations == 4

    def test_a_vague_statement_is_refused(self):
        with pytest.raises(ConfigError, match="statement"):
            _hypothesis(statement="it works")

    def test_a_missing_mechanism_is_refused(self):
        """'The backtest is good' is not a mechanism."""
        with pytest.raises(ConfigError, match="mechanism"):
            _hypothesis(mechanism="because the equity curve rises")

    def test_a_hypothesis_that_predicts_success_everywhere_is_refused(self):
        with pytest.raises(ConfigError, match="expected_regime"):
            _hypothesis(expected_regime="always")

    def test_a_hypothesis_with_no_falsification_criteria_is_refused(self):
        with pytest.raises(ConfigError, match="falsification"):
            Hypothesis(
                hypothesis_id="HYP-X",
                statement="a" * 30,
                rationale="r",
                mechanism="m" * 50,
                expected_regime="e" * 20,
                falsification=(),
                strategy="ts_momentum",
            )

    def test_defaults_are_attached(self):
        metrics = {c.metric for c in _hypothesis().falsification}
        assert {"oos_sharpe", "pbo", "deflated_sharpe_ratio", "n_oos_trades"} <= metrics
        assert len(DEFAULT_FALSIFICATION) >= 5

    def test_trials_include_prior_work_on_the_same_data(self):
        hypothesis = _hypothesis(prior_hypotheses_on_data=96)
        assert hypothesis.n_configurations == 4
        assert hypothesis.total_trials == 100

    def test_the_fingerprint_detects_later_editing(self):
        """A hypothesis edited to match its result must be detectable."""
        original = _hypothesis()
        edited = _hypothesis(statement="A different claim entirely, written afterwards.")
        assert original.fingerprint != edited.fingerprint

    def test_the_fingerprint_ignores_cosmetic_fields(self):
        a = _hypothesis(author="alice")
        b = _hypothesis(author="bob")
        assert a.fingerprint == b.fingerprint

    def test_round_trip_through_a_dict(self):
        original = _hypothesis()
        rebuilt = Hypothesis.from_dict(original.to_dict())
        assert rebuilt.fingerprint == original.fingerprint

    def test_criteria_report_unknown_metrics_as_unevaluated(self):
        criterion = FalsificationCriterion("missing_metric", "<", 0.0)
        assert criterion.is_falsified(None) is None
        assert criterion.is_falsified(float("nan")) is None

    def test_an_unsupported_operator_is_refused(self):
        with pytest.raises(ConfigError):
            FalsificationCriterion("oos_sharpe", "~=", 0.0)

    def test_markdown_states_the_falsification_criteria(self):
        text = _hypothesis().to_markdown()
        assert "Falsification criteria" in text
        assert "oos_sharpe" in text


class TestSelectionSeparation:
    @pytest.fixture(scope="class")
    @staticmethod
    def result():
        dataset = make_dataset(
            signal_plus_noise(n=2000, seed=7, edge_per_bar=0.003, lookback=5).bars, symbol="EDGE"
        )
        return run_funnel(
            dataset, _hypothesis(), config=FAST,
            instrument=InstrumentSpec(symbol="EDGE", multiplier=1000.0),
        )

    def test_the_sweep_contains_no_out_of_sample_column(self, result):
        """The defect that invalidated every verdict in the audited project.

        If an out-of-sample score appears in the table the selection reads
        from, the reported out-of-sample number is an in-sample statistic
        wearing an out-of-sample label, and no later test can detect it.
        """
        assert not any("oos" in column.lower() for column in result.sweep.columns)
        assert not any("out_of_sample" in column.lower() for column in result.sweep.columns)

    def test_the_sweep_covers_the_declared_grid(self, result):
        assert len(result.sweep) == result.hypothesis.n_configurations

    def test_the_in_sample_window_excludes_the_out_of_sample_window(self, result):
        is_end = result.is_result.equity.index[-1]
        oos_start = result.oos_result.equity.index[0]
        assert is_end < oos_start

    def test_an_embargo_separates_the_two_windows(self, result):
        is_end = result.is_result.equity.index[-1]
        oos_start = result.oos_result.equity.index[0]
        gap = result.full_bars.loc[is_end:oos_start]
        assert len(gap) >= result.config.embargo_bars

    def test_the_stages_run_in_order(self, result):
        names = [stage.name for stage in result.stages]
        assert names[0] == "sanity"
        assert names.index("is_search") < names.index("out_of_sample")
        assert names.index("out_of_sample") < names.index("multiple_testing")

    def test_every_metric_is_json_serialisable(self, result):
        import json

        json.dumps(result.to_dict(), default=str)


class TestControls:
    def test_the_positive_control_is_not_rejected(self):
        dataset = make_dataset(
            signal_plus_noise(n=2000, seed=7, edge_per_bar=0.003, lookback=5).bars, symbol="EDGE"
        )
        result = run_funnel(
            dataset, _hypothesis(), config=FAST,
            instrument=InstrumentSpec(symbol="EDGE", multiplier=1000.0),
        )
        assert result.verdict.level >= VerdictLevel.PROMISING
        assert result.metrics["oos_sharpe"] > 0

    def test_the_negative_control_is_rejected(self):
        dataset = make_dataset(random_walk(n=2000, seed=11, s0=3.0, vol=0.02).bars, symbol="RW")
        result = run_funnel(
            dataset, _hypothesis(), config=FAST,
            instrument=InstrumentSpec(symbol="RW", multiplier=10_000.0),
        )
        assert result.verdict.level is VerdictLevel.REJECTED
        assert result.verdict.falsified_by

    def test_the_rejection_names_the_criteria_that_fired(self):
        dataset = make_dataset(random_walk(n=2000, seed=11, s0=3.0, vol=0.02).bars, symbol="RW")
        result = run_funnel(
            dataset, _hypothesis(), config=FAST,
            instrument=InstrumentSpec(symbol="RW", multiplier=10_000.0),
        )
        text = " ".join(result.verdict.falsified_by)
        assert "oos_sharpe" in text or "pbo" in text or "deflated_sharpe_ratio" in text

    def test_costs_decay_monotonically_on_the_stress_ladder(self):
        dataset = make_dataset(
            signal_plus_noise(n=2000, seed=7, edge_per_bar=0.003, lookback=5).bars, symbol="EDGE"
        )
        result = run_funnel(
            dataset, _hypothesis(), config=FAST,
            instrument=InstrumentSpec(symbol="EDGE", multiplier=1000.0),
        )
        assert result.metrics["cost_decay_monotone"] is True
        sharpes = result.stress_table["sharpe"].to_numpy()
        assert list(sharpes) == sorted(sharpes, reverse=True)


class TestFunnelGuards:
    def test_an_absurd_grid_is_refused_rather_than_deflated_away(self):
        dataset = make_dataset(random_walk(n=1500, seed=3).bars, symbol="RW")
        hypothesis = _hypothesis(param_space={"lookback": list(range(2, 500))})
        result = run_funnel(dataset, hypothesis, config=FAST)
        assert result.verdict.level is VerdictLevel.REJECTED
        assert "grid" in result.stage("sanity").summary.lower()

    def test_too_little_data_is_rejected_at_the_sanity_stage(self):
        """Refused before anything is measured, and recorded as a result."""
        dataset = make_dataset(random_walk(n=300, seed=3).bars, symbol="RW")
        result = run_funnel(dataset, _hypothesis(), config=FAST)
        assert result.verdict.level is VerdictLevel.REJECTED
        assert "too short" in result.stage("sanity").summary
        assert result.oos_result is None

    def test_an_invalid_is_fraction_is_refused(self):
        with pytest.raises(ConfigError, match="is_fraction"):
            FunnelConfig(is_fraction=0.95)


class TestVerdictLogic:
    @staticmethod
    def _metrics(**overrides):
        metrics = {
            "oos_sharpe": 1.5,
            "deflated_sharpe_ratio": 0.99,
            "pbo": 0.05,
            "cpcv_median_sharpe": 1.2,
            "cpcv_positive_fraction": 0.9,
            "wf_efficiency": 0.8,
            "wf_profitable_fraction": 0.8,
            "bootstrap_sharpe_ci_low": 0.05,
            "conservative_sharpe": 0.9,
            "stressed_sharpe": 0.4,
            "n_oos_trades": 150,
            "plateau_fraction": 0.8,
            "positive_regime_fraction": 1.0,
            "best_year_return_share": 0.3,
        }
        metrics.update(overrides)
        return metrics

    @staticmethod
    def _stages():
        from exchange.funnel.pipeline import StageResult

        return [StageResult("sanity", True, "fine")]

    def test_a_perfect_result_can_reach_paper_trading(self):
        verdict = decide_verdict(_hypothesis(), self._metrics(), self._stages())
        assert verdict.level is VerdictLevel.PAPER_TRADING_CANDIDATE

    def test_one_spectacular_metric_cannot_rescue_a_fatal_one(self):
        """The audited project's weighted robust_score did exactly that."""
        verdict = decide_verdict(
            _hypothesis(), self._metrics(oos_sharpe=5.0, pbo=0.8), self._stages()
        )
        assert verdict.level is VerdictLevel.REJECTED

    def test_a_failed_gate_caps_the_verdict_at_weak(self):
        verdict = decide_verdict(
            _hypothesis(), self._metrics(wf_efficiency=0.1, deflated_sharpe_ratio=0.99),
            self._stages(),
        )
        assert verdict.level is VerdictLevel.WEAK
        assert verdict.unmet_for_next_level

    def test_an_uncomputable_gate_caps_at_promising(self):
        metrics = self._metrics()
        metrics["cpcv_median_sharpe"] = float("nan")
        verdict = decide_verdict(_hypothesis(), metrics, self._stages())
        assert verdict.level is VerdictLevel.PROMISING
        assert any(gate.passed is None for gate in verdict.gates)

    def test_too_few_trades_blocks_paper_trading(self):
        verdict = decide_verdict(_hypothesis(), self._metrics(n_oos_trades=50), self._stages())
        assert verdict.level is VerdictLevel.ROBUST
        assert any("OOS trades" in reason for reason in verdict.unmet_for_next_level)

    def test_concentration_in_one_year_blocks_paper_trading(self):
        verdict = decide_verdict(
            _hypothesis(), self._metrics(best_year_return_share=0.9), self._stages()
        )
        assert verdict.level is VerdictLevel.ROBUST

    def test_a_sanity_failure_rejects_immediately(self):
        from exchange.funnel.pipeline import StageResult

        verdict = decide_verdict(
            _hypothesis(), self._metrics(), [StageResult("sanity", False, "bad data")]
        )
        assert verdict.level is VerdictLevel.REJECTED
        assert "bad data" in verdict.headline

    def test_non_monotone_costs_are_reported_as_a_caveat(self):
        metrics = self._metrics()
        metrics["cost_decay_monotone"] = False
        verdict = decide_verdict(_hypothesis(), metrics, self._stages())
        assert any("monotonically" in caveat for caveat in verdict.caveats)

    def test_markdown_shows_every_gate(self):
        verdict = decide_verdict(_hypothesis(), self._metrics(), self._stages())
        text = verdict.to_markdown()
        assert text.count("|") > 20
        assert "Verdict" in text

    def test_gate_serialisation(self):
        gate = Gate("example", None, "not evaluated")
        assert gate.to_dict()["passed"] is None

    def test_verdict_levels_are_ordered(self):
        assert (
            VerdictLevel.REJECTED
            < VerdictLevel.WEAK
            < VerdictLevel.PROMISING
            < VerdictLevel.ROBUST
            < VerdictLevel.PAPER_TRADING_CANDIDATE
        )
