"""Statistical validation.

Every estimator here is checked three ways, because a statistic that is
merely *implemented* is worthless:

1. against an analytic or published value where one exists;
2. with a **positive control** — synthetic data that contains the effect,
   where the test must detect it;
3. with a **negative control** — synthetic data that cannot contain it,
   where the test must not.

A test that only asserts "the function returns a float between 0 and 1"
documents nothing, and would have passed for every decorative
implementation this framework was written to replace.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from exchange.core.errors import ConfigError
from exchange.stats.bootstrap import (
    block_bootstrap,
    bootstrap_equity_paths,
    monte_carlo_summary,
    moving_block_bootstrap_indices,
    optimal_block_length,
    path_drawdowns,
    stationary_bootstrap_indices,
)
from exchange.stats.cv import (
    CombinatorialPurgedCV,
    PurgedKFold,
    WalkForward,
    assert_no_overlap,
    split_summary,
)
from exchange.stats.multiple_testing import (
    benjamini_hochberg,
    benjamini_yekutieli,
    bonferroni,
    correct,
    effective_n_trials,
    estimate_pi0,
    expected_false_discoveries,
    holm,
)
from exchange.stats.pbo import logit_distribution, probability_of_backtest_overfitting
from exchange.stats.reality_check import (
    reality_check,
    snooping_report,
    superior_predictive_ability,
)
from exchange.stats.sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_confidence_interval,
    sharpe_per_period,
    sharpe_standard_error,
)

RNG = np.random.default_rng(20260902)


# ==========================================================================
# Sharpe inference
# ==========================================================================


class TestSharpeInference:
    def test_sharpe_matches_its_definition(self):
        values = RNG.normal(0.001, 0.01, 5000)
        assert sharpe_per_period(values) == pytest.approx(
            np.mean(values) / np.std(values, ddof=1), rel=1e-12
        )

    def test_standard_error_approaches_the_iid_formula(self):
        """For normal i.i.d. returns, SE ~ sqrt((1 + SR^2/2)/n) (Lo 2002)."""
        values = RNG.normal(0.0005, 0.01, 10_000)
        sr = sharpe_per_period(values)
        analytic = np.sqrt((1.0 + 0.5 * sr**2) / len(values))
        assert sharpe_standard_error(values) == pytest.approx(analytic, rel=0.15)

    def test_negative_skew_raises_the_standard_error(self):
        """Mertens' correction is sign-dependent, which is easy to get wrong.

        For a *positive* Sharpe with left-skewed returns the adjusted
        standard error must exceed the naive i.i.d. one — the estimate is
        less reliable than normality implies, not more. Getting the sign
        wrong makes left-skewed strategies (every short-volatility book
        ever written) look *more* certain than they are.
        """
        raw = -sps.skewnorm.rvs(a=6, size=8000, random_state=1)
        centred = raw - raw.mean()
        values = centred * 0.01 + 0.004           # positive mean, negative skew
        assert sps.skew(values) < -0.3
        assert sharpe_per_period(values) > 0
        adjusted = sharpe_standard_error(values)
        naive = np.sqrt((1.0 + 0.5 * sharpe_per_period(values) ** 2) / len(values))
        assert adjusted > naive

    def test_autocorrelation_inflates_the_standard_error(self):
        """Smoothed returns make a Sharpe look more certain than it is."""
        shocks = RNG.normal(0.0, 0.01, 4000)
        smoothed = np.convolve(shocks, np.ones(5) / 5.0, mode="valid") + 0.001
        naive = sharpe_standard_error(smoothed, autocorrelation_lags=0)
        corrected = sharpe_standard_error(smoothed, autocorrelation_lags=5)
        assert corrected > naive * 1.2

    def test_confidence_interval_brackets_the_estimate(self):
        values = RNG.normal(0.001, 0.01, 2000)
        low, high = sharpe_confidence_interval(values, confidence=0.95)
        assert low < sharpe_per_period(values) < high

    def test_psr_is_calibrated_under_the_null(self):
        """Distributional, not single-seed.

        Under no skill, PSR(0) should be roughly uniform, so its mean sits
        near 0.5 and only a few percent of samples exceed 0.95. A
        single-seed assertion on a quantity like this is a coin flip
        dressed up as a test.
        """
        values = [
            probabilistic_sharpe_ratio(
                np.random.default_rng(seed).normal(0.0, 0.01, 750), benchmark_sr=0.0
            )
            for seed in range(200)
        ]
        values = np.asarray(values)
        assert 0.42 < values.mean() < 0.58
        assert (values > 0.95).mean() < 0.10

    def test_psr_detects_a_real_edge(self):
        values = RNG.normal(0.002, 0.01, 2000)      # SR ~ 0.2 per bar
        assert probabilistic_sharpe_ratio(values, benchmark_sr=0.0) > 0.99

    def test_expected_max_sharpe_matches_the_published_formula(self):
        """Bailey & Lopez de Prado's Gumbel approximation, not sqrt(2 ln N).

        ``(1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N e))`` with g the
        Euler-Mascheroni constant. It is materially lower than the crude
        ``sqrt(2 ln N)`` for small N, which matters because most honest
        searches are small.
        """
        gamma = 0.5772156649015329
        for n in (10, 100, 1000, 10_000):
            expected = (1 - gamma) * sps.norm.ppf(1 - 1 / n) + gamma * sps.norm.ppf(
                1 - 1 / (n * np.e)
            )
            assert expected_max_sharpe(n) == pytest.approx(expected, rel=1e-6)

    def test_expected_max_sharpe_grows_with_the_search_size(self):
        """The reason a 1,000-configuration search needs a higher bar."""
        values = [expected_max_sharpe(n) for n in (2, 10, 100, 1000, 10_000)]
        assert values == sorted(values)
        assert values[-1] < np.sqrt(2.0 * np.log(10_000))   # below the crude bound

    def test_expected_max_sharpe_scales_with_the_trial_spread(self):
        """A tightly clustered grid has less selection potential."""
        assert expected_max_sharpe(100, sharpe_variance=0.04) == pytest.approx(
            expected_max_sharpe(100, sharpe_variance=1.0) * 0.2, rel=1e-9
        )

    def test_expected_max_sharpe_of_one_trial_is_zero(self):
        assert expected_max_sharpe(1) == pytest.approx(0.0, abs=1e-9)


class TestDeflatedSharpe:
    def test_a_single_pre_registered_test_is_barely_deflated(self):
        values = RNG.normal(0.0025, 0.01, 2000)
        one = deflated_sharpe_ratio(values, n_trials=1)
        assert one.deflated_sharpe_ratio == pytest.approx(
            one.probabilistic_sharpe_ratio, abs=0.05
        )

    def test_deflation_is_monotone_in_the_number_of_trials(self):
        values = RNG.normal(0.0006, 0.01, 1500)
        dsrs = [deflated_sharpe_ratio(values, n_trials=n).deflated_sharpe_ratio
                for n in (1, 10, 100, 1000, 10_000)]
        assert dsrs == sorted(dsrs, reverse=True)

    def test_the_best_of_many_noise_draws_is_rejected(self):
        """The whole point of the statistic.

        Take the best of 500 pure-noise strategies. Its raw Sharpe looks
        respectable; after deflating for the search it must not.
        """
        n_trials = 500
        draws = [np.random.default_rng(s).normal(0.0, 0.01, 1000) for s in range(n_trials)]
        sharpes = [sharpe_per_period(d) for d in draws]
        best = draws[int(np.argmax(sharpes))]
        assert sharpe_per_period(best) > 0.05            # looks like something
        naive = probabilistic_sharpe_ratio(best, benchmark_sr=0.0)
        deflated = deflated_sharpe_ratio(best, n_trials=n_trials, trial_sharpes=sharpes)
        assert naive > 0.9                               # the naive test is fooled
        assert deflated.deflated_sharpe_ratio < 0.9      # the deflated one is not
        assert deflated.significant is False

    def test_a_genuine_edge_survives_deflation(self):
        values = RNG.normal(0.004, 0.01, 2000)
        result = deflated_sharpe_ratio(values, n_trials=100)
        assert result.deflated_sharpe_ratio > 0.95

    def test_minimum_track_record_length_is_finite_for_a_real_edge(self):
        values = RNG.normal(0.002, 0.01, 2000)
        length = minimum_track_record_length(values, confidence=0.95)
        assert np.isfinite(length)
        assert length > 10

    def test_minimum_track_record_length_is_infinite_without_an_edge(self):
        values = RNG.normal(-0.001, 0.01, 2000)
        assert not np.isfinite(minimum_track_record_length(values, confidence=0.95))


# ==========================================================================
# Cross-validation
# ==========================================================================


class TestPurging:
    def test_folds_cover_the_sample_without_overlapping(self):
        splits = list(PurgedKFold(n_splits=5, horizon=0, embargo=0).split(1000))
        assert len(splits) == 5
        test_indices = np.concatenate([s.test for s in splits])
        assert len(np.unique(test_indices)) == len(test_indices)
        assert_no_overlap(splits)

    def test_purging_removes_overlapping_labels(self):
        """Without purging, a label spanning the fold boundary leaks.

        A five-bar holding horizon means an observation's outcome depends
        on the next five bars, so training on a bar whose outcome extends
        into the test window trains on the test window.
        """
        unpurged = list(PurgedKFold(n_splits=5, horizon=0, embargo=0).split(500))
        purged = list(PurgedKFold(n_splits=5, horizon=10, embargo=0).split(500))
        assert sum(len(s.train) for s in purged) < sum(len(s.train) for s in unpurged)
        for split in purged:
            gap = min(abs(int(t) - int(v)) for t in split.train for v in split.test[:1])
            assert gap >= 1

    def test_embargo_adds_a_gap_after_the_test_window(self):
        no_embargo = list(PurgedKFold(n_splits=4, horizon=0, embargo=0).split(400))
        embargoed = list(PurgedKFold(n_splits=4, horizon=0, embargo=20).split(400))
        assert sum(len(s.train) for s in embargoed) < sum(len(s.train) for s in no_embargo)
        assert_no_overlap(embargoed)

    def test_train_and_test_never_intersect(self):
        for split in PurgedKFold(n_splits=6, horizon=5, embargo=5).split(800):
            assert not set(split.train.tolist()) & set(split.test.tolist())

    def test_cpcv_path_count_follows_the_formula(self):
        """paths = C(n, k) * k / n (Bailey & Lopez de Prado)."""
        from math import comb

        for groups, test_groups in ((6, 2), (8, 2), (6, 3)):
            cv = CombinatorialPurgedCV(n_groups=groups, n_test_groups=test_groups)
            assert cv.n_splits() == comb(groups, test_groups)
            assert cv.n_paths() == comb(groups, test_groups) * test_groups // groups

    def test_cpcv_splits_are_contiguous_groups(self):
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, horizon=0, embargo=0)
        for split in cv.split(600):
            assert len(split.test) > 0
            assert not set(split.train.tolist()) & set(split.test.tolist())

    def test_walk_forward_windows_move_forward_only(self):
        splits = list(WalkForward(train_size=200, test_size=50).split(600))
        assert splits
        for split in splits:
            assert split.train.max() < split.test.min()
        starts = [s.test.min() for s in splits]
        assert starts == sorted(starts)

    def test_split_summary_is_auditable(self):
        splits = list(PurgedKFold(n_splits=3, horizon=2, embargo=2).split(300))
        table = split_summary(splits)
        assert len(table) == 3
        assert {"n_train", "n_test"} <= set(table.columns)

    def test_too_few_observations_for_the_fold_count_is_refused(self):
        from exchange.core.errors import NotEnoughDataError

        with pytest.raises(NotEnoughDataError):
            list(PurgedKFold(n_splits=10).split(30))

    def test_a_single_fold_is_refused(self):
        """One fold is not cross-validation."""
        with pytest.raises(ConfigError):
            PurgedKFold(n_splits=1)

    def test_an_oversized_purge_leaves_no_training_data(self):
        """Purging wider than the sample must not silently train on nothing."""
        splits = list(PurgedKFold(n_splits=4, horizon=500, embargo=500).split(400))
        assert all(len(split.train) == 0 for split in splits)


# ==========================================================================
# PBO
# ==========================================================================


class TestPBO:
    @staticmethod
    def _matrix(n_obs, n_configs, *, edge_for_first=0.0, seed=0):
        rng = np.random.default_rng(seed)
        matrix = rng.normal(0.0, 0.01, size=(n_obs, n_configs))
        matrix[:, 0] += edge_for_first
        return matrix

    def test_pure_noise_averages_a_pbo_near_one_half(self):
        """Selection from an undifferentiated family is a coin flip.

        Averaged over seeds, not asserted on one: a single CSCV run on
        1,000 observations has a standard deviation of roughly 0.2 in PBO,
        so a single-seed bound would be a flaky test pretending to be a
        calibration.
        """
        values = [
            probability_of_backtest_overfitting(
                self._matrix(1000, 12, seed=seed), n_partitions=12
            ).pbo
            for seed in range(8)
        ]
        assert 0.35 < float(np.mean(values)) < 0.65

    def test_a_dominant_configuration_gives_a_pbo_of_zero(self):
        matrix = self._matrix(1000, 12, edge_for_first=0.004, seed=4)
        result = probability_of_backtest_overfitting(matrix, n_partitions=12)
        assert result.pbo < 0.2
        assert "INFORMATIVE" in result.explain().upper()

    def test_logit_distribution_is_centred_for_noise(self):
        matrix = self._matrix(800, 10, seed=5)
        result = probability_of_backtest_overfitting(matrix, n_partitions=10)
        histogram = logit_distribution(result)
        assert not histogram.empty
        assert abs(float(np.median(result.logits))) < 1.5

    def test_performance_degradation_is_reported(self):
        matrix = self._matrix(1000, 10, seed=6)
        result = probability_of_backtest_overfitting(matrix, n_partitions=10)
        assert np.isfinite(result.performance_degradation)
        assert 0.0 <= result.prob_oos_loss <= 1.0

    def test_too_few_configurations_is_refused(self):
        with pytest.raises(ConfigError):
            probability_of_backtest_overfitting(np.zeros((100, 1)), n_partitions=4)


# ==========================================================================
# Bootstrap
# ==========================================================================


class TestBootstrap:
    def test_optimal_block_length_grows_with_dependence(self):
        independent = RNG.normal(size=3000)
        shocks = RNG.normal(size=3000)
        dependent = np.convolve(shocks, np.ones(20) / 20.0, mode="same")
        assert optimal_block_length(dependent) > optimal_block_length(independent)

    def test_block_length_of_one_for_white_noise(self):
        assert optimal_block_length(RNG.normal(size=5000)) < 8.0

    def test_stationary_bootstrap_indices_are_in_range(self):
        indices = stationary_bootstrap_indices(
            100, block_length=5.0, rng=np.random.default_rng(1), size=100
        )
        assert indices.shape == (100,)
        assert indices.min() >= 0 and indices.max() < 100

    def test_moving_block_preserves_local_structure(self):
        """Resampled blocks must keep neighbouring observations together."""
        indices = moving_block_bootstrap_indices(
            200, block_length=10.0, rng=np.random.default_rng(2), size=200
        )
        steps = np.diff(indices)
        assert (steps == 1).mean() > 0.7

    def test_interval_covers_the_truth_for_a_known_mean(self):
        values = RNG.normal(0.001, 0.01, 2000)
        result = block_bootstrap(values, np.mean, n_resamples=500, seed=1)
        low, high = result.confidence_interval(0.95)
        assert low < 0.001 < high

    def test_interval_excludes_zero_for_a_strong_edge(self):
        values = RNG.normal(0.004, 0.01, 2000)
        result = block_bootstrap(
            values, lambda x: float(np.mean(x) / np.std(x, ddof=1)), n_resamples=500, seed=2
        )
        assert result.confidence_interval(0.95)[0] > 0.0

    def test_interval_includes_zero_for_noise(self):
        values = np.random.default_rng(9).normal(0.0, 0.01, 2000)
        result = block_bootstrap(
            values, lambda x: float(np.mean(x) / np.std(x, ddof=1)), n_resamples=500, seed=3
        )
        low, high = result.confidence_interval(0.95)
        assert low < 0.0 < high

    def test_bootstrap_is_reproducible(self):
        values = RNG.normal(0.001, 0.01, 500)
        a = block_bootstrap(values, np.mean, n_resamples=200, seed=7)
        b = block_bootstrap(values, np.mean, n_resamples=200, seed=7)
        assert a.observed == b.observed
        assert np.allclose(a.samples, b.samples)

    def test_equity_paths_and_drawdowns(self):
        values = RNG.normal(0.001, 0.01, 500)
        paths = bootstrap_equity_paths(values, n_paths=100, seed=5)
        assert paths.shape[0] == 100
        drawdowns = path_drawdowns(paths)
        assert (drawdowns <= 0).all()

    def test_monte_carlo_summary_reports_ruin_probability(self):
        summary = monte_carlo_summary(
            RNG.normal(-0.002, 0.02, 500), n_paths=200, seed=6
        )
        assert 0.0 <= summary["prob_loss"] <= 1.0
        assert summary["prob_loss"] > 0.5


# ==========================================================================
# Multiple testing
# ==========================================================================


class TestMultipleTesting:
    def test_bonferroni_scales_p_values_by_the_count(self):
        result = bonferroni([0.01, 0.04], alpha=0.05)
        assert result.adjusted[0] == pytest.approx(0.02)
        assert result.n_rejected == 1

    def test_holm_is_more_powerful_than_bonferroni(self):
        p_values = [0.001, 0.009, 0.02, 0.5]
        assert holm(p_values, alpha=0.05).n_rejected >= bonferroni(p_values, alpha=0.05).n_rejected

    def test_bh_controls_fdr_and_rejects_more_than_holm(self):
        p_values = [0.001, 0.005, 0.01, 0.02, 0.03, 0.6, 0.7]
        assert (
            benjamini_hochberg(p_values, alpha=0.05).n_rejected
            >= holm(p_values, alpha=0.05).n_rejected
        )

    def test_by_is_more_conservative_than_bh(self):
        p_values = [0.001, 0.005, 0.01, 0.02, 0.03, 0.6, 0.7]
        assert (
            benjamini_yekutieli(p_values, alpha=0.05).n_rejected
            <= benjamini_hochberg(p_values, alpha=0.05).n_rejected
        )

    def test_adjusted_p_values_are_monotone(self):
        p_values = np.sort(np.random.default_rng(4).uniform(size=50))
        for method in ("bonferroni", "holm", "bh", "by"):
            adjusted = correct(p_values, method=method).adjusted
            assert np.all(np.diff(adjusted) >= -1e-12)
            assert np.all(adjusted <= 1.0 + 1e-12)

    def test_familywise_error_is_controlled_under_the_global_null(self):
        """The defining property, measured rather than assumed.

        Across 100 families of 200 uniform p-values, the fraction of
        families with *any* rejection must not exceed alpha by much. One
        family is not a test of a family-wise guarantee.
        """
        rejecting = 0
        for seed in range(100):
            p_values = np.random.default_rng(seed).uniform(size=200)
            if holm(p_values, alpha=0.05).n_rejected > 0:
                rejecting += 1
        assert rejecting / 100 <= 0.10

    def test_true_effects_are_recovered(self):
        rng = np.random.default_rng(6)
        null = rng.uniform(size=200)
        signal = rng.uniform(0.0, 1e-5, size=20)
        result = benjamini_hochberg(np.concatenate([signal, null]), alpha=0.05)
        assert result.n_rejected >= 15

    def test_pi0_estimates_the_null_proportion(self):
        rng = np.random.default_rng(7)
        p_values = np.concatenate([rng.uniform(size=900), rng.uniform(0, 1e-4, size=100)])
        assert 0.8 < estimate_pi0(p_values) < 1.05

    def test_expected_false_discoveries_is_alpha_times_n(self):
        p_values = np.random.default_rng(8).uniform(size=400)
        assert expected_false_discoveries(p_values, alpha=0.05) == pytest.approx(20.0, rel=0.3)

    def test_effective_n_trials_discounts_a_correlated_family(self):
        """A parameter grid is not m independent tests.

        The default refuses to discount anything, because an unjustified
        reduction here weakens the very correction that matters.
        """
        p_values = np.random.default_rng(9).uniform(size=100)
        assert effective_n_trials(p_values) == 100.0
        assert effective_n_trials(p_values, correlation=0.5) < 10.0
        assert effective_n_trials(p_values, correlation=0.0) == 100.0
        assert effective_n_trials(p_values, correlation=0.9) < effective_n_trials(
            p_values, correlation=0.3
        )


# ==========================================================================
# Reality Check and SPA
# ==========================================================================


class TestDataSnooping:
    @staticmethod
    def _performance(n_strategies, n_obs, *, best_edge=0.0, seed=0):
        rng = np.random.default_rng(seed)
        matrix = rng.normal(0.0, 0.01, size=(n_obs, n_strategies))
        matrix[:, 0] += best_edge
        return matrix

    def test_reality_check_does_not_reject_under_the_null(self):
        """100 worthless strategies: the best one must not look significant."""
        result = reality_check(self._performance(100, 1000, seed=1), n_bootstrap=400, seed=1)
        assert result.p_value > 0.10

    def test_reality_check_rejects_a_genuine_edge(self):
        result = reality_check(
            self._performance(100, 1000, best_edge=0.004, seed=2), n_bootstrap=400, seed=2
        )
        assert result.p_value < 0.05

    def test_spa_is_at_least_as_powerful_as_the_reality_check(self):
        """Hansen's recentring removes the dominated strategies' drag."""
        matrix = self._performance(100, 1000, best_edge=0.0025, seed=3)
        rc = reality_check(matrix, n_bootstrap=400, seed=3)
        spa = superior_predictive_ability(matrix, n_bootstrap=400, seed=3)
        assert spa.p_value <= rc.p_value + 1e-9

    def test_spa_bounds_bracket_the_consistent_p_value(self):
        matrix = self._performance(50, 800, best_edge=0.002, seed=4)
        spa = superior_predictive_ability(matrix, n_bootstrap=300, seed=4)
        lower = spa.extra["p_value_lower"]
        upper = spa.extra["p_value_upper"]
        assert lower <= spa.p_value <= upper + 1e-12

    def test_spa_does_not_reject_under_the_null(self):
        spa = superior_predictive_ability(
            self._performance(100, 1000, seed=5), n_bootstrap=400, seed=5
        )
        assert spa.p_value > 0.10

    def test_snooping_report_runs_both_tests(self):
        table = snooping_report(self._performance(20, 500, seed=6), n_bootstrap=200, seed=6)
        tests = {str(v).lower() for v in table.get("test", table.index)}
        assert any("reality" in t for t in tests)
        assert any("spa" in t for t in tests)

    def test_a_single_strategy_still_works(self):
        single = self._performance(1, 500, best_edge=0.004, seed=7)
        assert reality_check(single, n_bootstrap=200, seed=7).p_value < 0.1
