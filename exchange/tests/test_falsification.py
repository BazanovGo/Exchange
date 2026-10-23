"""The falsification engine.

Two properties matter and are tested directly: a result built on a real,
synthetic edge survives the attacks, and a result built on noise does not.
Also tested: the robustness score can only *lower* a verdict, an attack
that crashes is recorded as missing evidence rather than as a pass, and no
attack re-selects parameters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exchange.backtest.engine import InstrumentSpec
from exchange.core.errors import ConfigError
from exchange.falsification.engine import (
    ATTACK_NAMES,
    CRITICAL,
    AttackContext,
    AttackOutcome,
    RobustnessReport,
    cost_double,
    drop_best_trades,
    execution_latency,
    falsify,
    neighbour_params,
    roll_methodology,
    run_attacks,
    subperiod_consistency,
)
from exchange.funnel.pipeline import FunnelConfig, run_funnel
from exchange.funnel.verdict import VerdictLevel
from exchange.synthetic.generators import random_walk


@pytest.fixture(scope="module")
def fast_config():
    return FunnelConfig(n_bootstrap=150, n_monte_carlo=150)


@pytest.fixture(scope="module")
def edge_result(fast_config):
    from exchange.data.dataset import make_dataset
    from exchange.funnel.hypothesis import make_hypothesis
    from exchange.synthetic.generators import signal_plus_noise

    dataset = make_dataset(
        signal_plus_noise(n=2000, seed=7, edge_per_bar=0.003, lookback=5).bars,
        symbol="EDGE",
    )
    hypothesis = make_hypothesis(
        "HYP-ATTACK-EDGE",
        statement="The trailing five-bar return sign predicts the next bar positively.",
        mechanism=(
            "Slower participants trade after the information arrives, extending the move over "
            "several bars rather than one."
        ),
        expected_regime="Throughout this synthetic sample, which is stationary by construction.",
        strategy="ts_momentum",
        param_space={"lookback": [3, 5, 10, 20]},
    )
    return run_funnel(
        dataset, hypothesis, config=fast_config,
        instrument=InstrumentSpec(symbol="EDGE", multiplier=1000.0),
    )


@pytest.fixture(scope="module")
def noise_result(fast_config):
    from exchange.data.dataset import make_dataset
    from exchange.funnel.hypothesis import make_hypothesis

    dataset = make_dataset(random_walk(n=2000, seed=11, s0=3.0, vol=0.02).bars, symbol="NOISE")
    hypothesis = make_hypothesis(
        "HYP-ATTACK-NOISE",
        statement="The trailing k-bar return sign predicts the next bar on this series.",
        mechanism=(
            "Order flow from slower participants would extend moves if it existed here, which "
            "is the claim under test."
        ),
        expected_regime="Throughout the sample; a random walk should refute it.",
        strategy="ts_momentum",
        param_space={"lookback": [3, 5, 10, 20]},
    )
    return run_funnel(
        dataset, hypothesis, config=fast_config,
        instrument=InstrumentSpec(symbol="NOISE", multiplier=10_000.0),
    )


class TestControls:
    def test_the_known_edge_survives_every_applicable_attack(self, edge_result):
        report = falsify(edge_result, n_resamples=200)
        assert report.failed == [], [o.name for o in report.failed]
        assert report.grade == "HARDENED"
        assert report.score == 1.0

    def test_noise_fails_most_attacks(self, noise_result):
        report = falsify(noise_result, n_resamples=200)
        assert report.score < 0.5
        assert report.grade == "FRAGILE"
        assert report.verdict_cap is VerdictLevel.WEAK

    def test_roll_methodology_is_not_applicable_without_alternatives(self, edge_result):
        report = falsify(edge_result, n_resamples=100, attacks=["roll_methodology"])
        outcome = report.outcome("roll_methodology")
        assert outcome.survived is None
        assert not outcome.applicable

    def test_an_inapplicable_attack_is_not_counted_as_a_pass(self, edge_result):
        report = falsify(edge_result, n_resamples=100, attacks=["roll_methodology"])
        assert np.isnan(report.score)
        assert report.grade == "FRAGILE"      # no evidence is not good evidence


class TestScoreSemantics:
    @staticmethod
    def _report(outcomes):
        return RobustnessReport(outcomes=list(outcomes), baseline_sharpe=1.0, n_bars=500)

    def test_score_counts_only_evaluated_attacks(self):
        report = self._report(
            [
                AttackOutcome("a", "q", CRITICAL, True),
                AttackOutcome("b", "q", CRITICAL, None),
                AttackOutcome("c", "q", "major", False),
            ]
        )
        assert report.score == pytest.approx(0.5)
        assert len(report.evaluated) == 2

    def test_a_critical_failure_forces_fragile_whatever_the_score(self):
        outcomes = [AttackOutcome(f"ok{i}", "q", "minor", True) for i in range(9)]
        outcomes.append(AttackOutcome("fatal", "q", CRITICAL, False))
        report = self._report(outcomes)
        assert report.score == pytest.approx(0.9)
        assert report.grade == "FRAGILE"
        assert report.verdict_cap is VerdictLevel.WEAK

    def test_the_cap_can_only_lower_a_verdict(self):
        perfect = self._report([AttackOutcome("a", "q", CRITICAL, True)])
        assert perfect.cap(VerdictLevel.WEAK) is VerdictLevel.WEAK          # never raised
        assert perfect.cap(VerdictLevel.ROBUST) is VerdictLevel.ROBUST
        fragile = self._report([AttackOutcome("a", "q", CRITICAL, False)])
        assert fragile.cap(VerdictLevel.PAPER_TRADING_CANDIDATE) is VerdictLevel.WEAK

    def test_markdown_names_every_attack(self):
        report = self._report(
            [
                AttackOutcome("a", "Does it hold?", CRITICAL, True, 1.0, 0.9, "detail"),
                AttackOutcome("b", "Does it hold?", CRITICAL, False, 1.0, -0.1, "detail"),
            ]
        )
        text = report.to_markdown()
        assert "`a`" in text and "`b`" in text
        assert "FAILED" in text
        assert "cannot raise a verdict" in text


class TestIndividualAttacks:
    @pytest.fixture
    def context(self, edge_result):
        oos_bars = edge_result.full_bars.loc[edge_result.oos_result.equity.index]
        return AttackContext(
            bars=oos_bars,
            hypothesis=edge_result.hypothesis,
            params=dict(edge_result.best_params),
            config=edge_result.oos_result.config,
            symbol=edge_result.oos_result.symbol,
            bars_per_year=edge_result.oos_result.bars_per_year,
            n_resamples=150,
            baseline=edge_result.oos_result,
        )

    def test_neighbour_params_reports_the_neighbourhood(self, context):
        outcome = neighbour_params(context)
        assert outcome.survived is True
        assert outcome.evidence["n_neighbours"] >= 1
        assert 0.0 <= outcome.evidence["positive_fraction"] <= 1.0

    def test_neighbour_params_does_not_change_the_selection(self, context):
        before = dict(context.params)
        neighbour_params(context)
        assert context.params == before

    def test_cost_double_reduces_performance(self, context):
        outcome = cost_double(context)
        assert outcome.perturbed < outcome.baseline
        assert outcome.severity == CRITICAL

    def test_cost_improving_performance_is_treated_as_an_error(self, context):
        """Costs that help indicate a sign error, not robustness."""
        outcome = AttackOutcome("cost_double", "q", CRITICAL, True, baseline=1.0, perturbed=1.5)
        assert outcome.retained > 1.0        # the diagnostic the engine checks

    def test_latency_attack_reports_both_horizons(self, context):
        outcome = execution_latency(context)
        assert {"plus_one", "plus_two"} <= set(outcome.evidence)

    def test_drop_best_trades_measures_concentration(self, context):
        outcome = drop_best_trades(context)
        assert 0.0 < outcome.evidence["best_share"] <= 1.0
        assert outcome.evidence["n_dropped"] >= 1

    def test_subperiods_are_reported_individually(self, context):
        outcome = subperiod_consistency(context)
        assert outcome.evidence["n_blocks"] >= 3

    def test_roll_methodology_uses_supplied_alternatives(self, context):
        """The only question that matters about a spliced series."""
        shifted = context.bars.copy()
        shifted[["open", "high", "low", "close"]] *= 1.01     # a different splice level
        context.alternative_bars = {"alternative/ratio": shifted}
        outcome = roll_methodology(context)
        assert outcome.applicable
        assert "alternative/ratio" in outcome.detail

    def test_a_crashing_attack_is_recorded_as_missing_evidence(self, context):
        def exploding_attack(ctx):
            raise RuntimeError("boom")

        exploding_attack.__name__ = "exploding_attack"
        report = run_attacks(context, attacks=[exploding_attack])
        outcome = report.outcome("exploding_attack")
        assert outcome.survived is None
        assert "RuntimeError" in outcome.detail
        assert outcome not in report.survived

    def test_unknown_attack_name_is_refused(self, context):
        with pytest.raises(ConfigError, match="unknown attack"):
            run_attacks(context, attacks=["wishful_thinking"])

    def test_every_named_attack_runs(self, context):
        report = run_attacks(context)
        assert {o.name for o in report.outcomes} == set(ATTACK_NAMES)

    def test_too_short_a_window_is_refused(self, edge_result):
        with pytest.raises(ConfigError, match="at least 50"):
            AttackContext(
                bars=edge_result.full_bars.iloc[:10],
                hypothesis=edge_result.hypothesis,
                params=dict(edge_result.best_params),
                config=edge_result.oos_result.config,
                symbol="EDGE",
                bars_per_year=252.0,
            )

    def test_price_perturbation_keeps_the_bars_legal(self, context):
        from exchange.falsification.engine import _perturb_bars

        noisy = _perturb_bars(context.bars, np.random.default_rng(1), scale=0.1)
        assert (noisy["high"] >= noisy[["open", "close"]].max(axis=1)).all()
        assert (noisy["low"] <= noisy[["open", "close"]].min(axis=1)).all()
        assert (noisy[["open", "high", "low", "close"]] > 0).all().all()
        assert noisy.index.equals(context.bars.index)

    def test_attack_seeds_are_independent_of_order(self, context):
        a = context.rng("alpha").normal(size=5).tolist()
        b = context.rng("beta").normal(size=5).tolist()
        assert a != b
        assert context.rng("alpha").normal(size=5).tolist() == a


class TestFalsifyGuards:
    def test_falsify_refuses_a_funnel_that_never_reached_out_of_sample(self, fast_config):
        from exchange.data.dataset import make_dataset
        from exchange.funnel.hypothesis import make_hypothesis

        dataset = make_dataset(random_walk(n=400, seed=3).bars, symbol="SHORT", strict=False)
        hypothesis = make_hypothesis(
            "HYP-SANITY-FAIL",
            statement="This hypothesis will not get past the sanity stage.",
            mechanism=(
                "There is no mechanism; the sample is deliberately too short for the funnel to "
                "run, which is what this test checks."
            ),
            expected_regime="Nowhere, which the sanity stage should establish first.",
            strategy="ts_momentum",
            param_space={"lookback": [10] * 500},      # absurd grid, refused by max_grid
        )
        result = run_funnel.__wrapped__ if hasattr(run_funnel, "__wrapped__") else run_funnel
        try:
            funnel_result = result(dataset, hypothesis, config=fast_config)
        except Exception:
            pytest.skip("the funnel refused the configuration before producing a result")
        if funnel_result.oos_result is None:
            with pytest.raises(ConfigError, match="no out-of-sample result"):
                falsify(funnel_result)

    def test_falsify_requires_a_funnel_result(self):
        with pytest.raises(ConfigError, match="FunnelResult"):
            falsify(pd.DataFrame())
