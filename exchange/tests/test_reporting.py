"""The standard research report."""

from __future__ import annotations

import re

import pytest

from exchange.backtest.engine import InstrumentSpec
from exchange.data.dataset import make_dataset
from exchange.falsification.engine import falsify
from exchange.funnel.hypothesis import make_hypothesis
from exchange.funnel.pipeline import FunnelConfig, run_funnel
from exchange.reporting.research_report import (
    SECTIONS,
    cost_share_of_gross,
    falsity_threats,
    research_report,
)
from exchange.scanner.report import scan_dataset
from exchange.synthetic.generators import random_walk, signal_plus_noise

FAST = FunnelConfig(n_bootstrap=150, n_monte_carlo=150)


def _hypothesis(hypothesis_id="HYP-REPORT"):
    return make_hypothesis(
        hypothesis_id,
        statement="The trailing five-bar return sign predicts the next bar positively.",
        mechanism=(
            "Slower participants trade after the information arrives, extending the move over "
            "several bars."
        ),
        expected_regime="Trending regimes only; a random walk should refute it.",
        strategy="ts_momentum",
        param_space={"lookback": [3, 5, 10, 20]},
    )


@pytest.fixture(scope="module")
def edge_result():
    dataset = make_dataset(
        signal_plus_noise(n=2000, seed=7, edge_per_bar=0.003, lookback=5).bars, symbol="EDGE"
    )
    return run_funnel(
        dataset, _hypothesis(), config=FAST,
        instrument=InstrumentSpec(symbol="EDGE", multiplier=1000.0),
    )


@pytest.fixture(scope="module")
def noise_result():
    dataset = make_dataset(random_walk(n=2000, seed=11, s0=3.0, vol=0.02).bars, symbol="RW")
    return run_funnel(
        dataset, _hypothesis("HYP-REPORT-NOISE"), config=FAST,
        instrument=InstrumentSpec(symbol="RW", multiplier=10_000.0),
    )


class TestStructure:
    def test_every_declared_section_is_rendered(self, edge_result):
        text = research_report(edge_result)
        headings = re.findall(r"^## \d+\. (.+)$", text, flags=re.M)
        assert headings == list(SECTIONS)

    def test_the_report_needs_no_optional_dependencies(self, edge_result):
        """A report that cannot be produced in CI will not be produced."""
        text = research_report(edge_result)
        assert "|" in text
        assert len(text) > 5000

    def test_a_rejection_fills_in_the_same_sections(self, noise_result):
        text = research_report(noise_result)
        headings = re.findall(r"^## \d+\. (.+)$", text, flags=re.M)
        assert headings == list(SECTIONS)
        assert "REJECTED" in text

    def test_the_hypothesis_and_its_fingerprint_appear(self, edge_result):
        text = research_report(edge_result)
        assert edge_result.hypothesis.hypothesis_id in text
        assert edge_result.hypothesis.fingerprint[:12] in text

    def test_the_search_size_and_noise_benchmark_appear(self, edge_result):
        text = research_report(edge_result)
        assert "Expected best Sharpe from noise alone" in text
        assert "Trials charged to the deflation" in text

    def test_the_experiment_id_produces_a_reproduce_command(self, edge_result):
        text = research_report(edge_result, experiment_id="EXP-20260902-000157")
        assert "exchange reproduce EXP-20260902-000157" in text

    def test_the_scan_is_included_when_supplied(self, edge_result):
        dataset = make_dataset(edge_result.full_bars, symbol="EDGE")
        text = research_report(edge_result, scan=scan_dataset(dataset))
        assert "Structural diagnostics" in text
        assert "variance" in text.lower()

    def test_a_missing_scan_is_stated_not_hidden(self, edge_result):
        text = research_report(edge_result)
        assert "No structural scan supplied" in text

    def test_untested_results_say_so(self, edge_result):
        text = research_report(edge_result)
        assert "has not been attacked" in text

    def test_the_attack_record_is_included_when_supplied(self, edge_result):
        report = falsify(edge_result, n_resamples=150)
        text = research_report(edge_result, robustness=report)
        assert "Robustness score" in text
        assert "`cost_double`" in text


class TestGeneratedThreats:
    def test_threats_are_generated_from_the_numbers(self, edge_result):
        threats = falsity_threats(edge_result)
        assert threats
        assert all(isinstance(t, str) for t in threats)

    def test_an_untested_result_is_listed_as_a_threat(self, edge_result):
        assert any("Untested" in t for t in falsity_threats(edge_result))

    def test_failed_attacks_appear_as_threats(self, noise_result):
        report = falsify(noise_result, n_resamples=150)
        threats = falsity_threats(noise_result, robustness=report)
        assert any("Failed attack" in t for t in threats)

    def test_unevaluated_attacks_are_not_silently_passed(self, edge_result):
        report = falsify(edge_result, n_resamples=150)
        threats = falsity_threats(edge_result, robustness=report)
        assert any("roll_methodology" in t for t in threats)

    def test_missing_provenance_is_a_threat(self, edge_result):
        threats = falsity_threats(edge_result)
        assert any("provenance" in t.lower() for t in threats)

    def test_cost_share_is_measured_against_gross_profit(self, edge_result):
        share = cost_share_of_gross(edge_result)
        assert 0.0 < share < 1.0

    def test_a_short_history_is_flagged(self, edge_result):
        import copy

        shortened = copy.copy(edge_result)
        shortened.dataset_summary = {
            **edge_result.dataset_summary, "n_bars": 500, "bars_per_year": 252.0
        }
        assert any("Short history" in t for t in falsity_threats(shortened))

    def test_a_splice_adjustment_is_flagged(self, edge_result):
        import copy

        spliced = copy.copy(edge_result)
        spliced.dataset_summary = {
            **edge_result.dataset_summary, "provenance": {"adjustment": "backward"}
        }
        assert any("Splice artefact" in t for t in falsity_threats(spliced))
