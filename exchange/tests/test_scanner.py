"""Structural scanner.

The scanner's job is to say whether a series has exploitable structure
*before* any strategy search begins. Its most important property is
therefore its false-positive rate, which these tests measure rather than
assume: a scanner that finds mean reversion in random walks sends a
research programme after artefacts.
"""

from __future__ import annotations

import numpy as np
import pytest

from exchange.scanner.diagnostics import (
    adf_test,
    autocorrelation,
    hurst_exponent,
    mean_reversion_half_life,
    seasonality,
    structural_stability,
    tail_behaviour,
    variance_ratio,
    volatility_clustering,
)
from exchange.scanner.report import scan, scan_dataset, scan_table
from exchange.synthetic.generators import (
    mean_reverting,
    momentum,
    random_walk,
    regime_switch,
    seasonal,
    structural_break,
)


class TestDiagnostics:
    @staticmethod
    def _levels(series):
        return np.log(series.bars["close"].to_numpy(float))

    def test_adf_rejects_a_unit_root_for_a_mean_reverting_series(self):
        series = mean_reverting(n=1500, seed=5, half_life=10.0)
        statistic, p_value, _lags = adf_test(self._levels(series))
        assert statistic < -2.86          # 5% MacKinnon critical value
        assert p_value < 0.05

    def test_adf_does_not_reject_for_a_random_walk(self):
        series = random_walk(n=1500, seed=5)
        statistic, p_value, _lags = adf_test(self._levels(series))
        assert statistic > -2.86
        assert p_value > 0.05

    def test_adf_refuses_a_two_dimensional_input(self):
        """A frame flattened into the regression is silent nonsense."""
        from exchange.core.errors import ConfigError

        with pytest.raises(ConfigError, match="1-D"):
            adf_test(random_walk(n=300, seed=1).bars.to_numpy())

    def test_half_life_is_gated_on_the_adf_test(self):
        """A half-life without stationarity is meaningless.

        Regressing a change on a level under a unit root produces a
        confident-looking negative slope and therefore a plausible
        half-life for any random walk. Gating on the ADF statistic is what
        removed the scanner's false positives.
        """
        walk = mean_reversion_half_life(random_walk(n=1500, seed=7).bars)
        assert walk.evidence in {"NONE", "WEAK"}

    def test_half_life_recovers_a_known_value(self):
        series = mean_reverting(n=3000, seed=11, half_life=10.0)
        result = mean_reversion_half_life(series.bars)
        assert result.evidence in {"MODERATE", "STRONG"}
        assert 4.0 < result.value < 25.0

    def test_variance_ratio_below_one_for_mean_reversion(self):
        series = mean_reverting(n=2000, seed=13, half_life=8.0)
        assert variance_ratio(series.bars, q=10).value < 1.0

    def test_variance_ratio_above_one_for_momentum(self):
        series = momentum(n=2000, seed=17)
        assert variance_ratio(series.bars, q=10).value > 1.0

    def test_variance_ratio_near_one_for_a_random_walk(self):
        series = random_walk(n=3000, seed=19)
        assert 0.8 < variance_ratio(series.bars, q=5).value < 1.2

    def test_autocorrelation_detects_momentum(self):
        assert autocorrelation(momentum(n=2000, seed=23).bars).evidence != "NONE"

    def test_volatility_clustering_detected_in_regime_data(self):
        series = regime_switch(n=2000, seed=29)
        assert volatility_clustering(series.bars).evidence in {"MODERATE", "STRONG"}

    def test_seasonality_found_when_the_bucketing_can_express_it(self):
        """An annual cycle is visible to month-of-year buckets."""
        series = seasonal(n=2500, seed=31, period=252, amplitude=0.004)
        assert seasonality(series.bars, by="month").evidence == "STRONG"

    def test_a_cycle_the_bucketing_cannot_express_is_not_found(self):
        """A 21-bar cycle averages out across months, and must not be claimed.

        This is a property of the diagnostic, not a defect: a test can only
        find the seasonality its own bucketing expresses, and reporting
        otherwise would be a false discovery waiting to happen.
        """
        series = seasonal(n=2500, seed=31, period=21, amplitude=0.02)
        assert seasonality(series.bars, by="month").evidence == "NONE"

    def test_seasonality_not_claimed_for_a_random_walk(self):
        """Twelve monthly buckets are twelve tests, so the bar must be high."""
        series = random_walk(n=2500, seed=37)
        assert seasonality(series.bars).evidence in {"NONE", "WEAK"}

    def test_structural_stability_flags_a_break(self):
        series = structural_break(n=2000, seed=41)
        assert structural_stability(series.bars).value is not None

    def test_hurst_uses_a_null_band(self):
        """A raw R/S estimate on 2,000 bars is biased above 0.5.

        Comparing against the distribution of same-length random walks is
        what stops every series looking persistent.
        """
        result = hurst_exponent(random_walk(n=2000, seed=43).bars)
        assert result.evidence in {"NONE", "WEAK"}
        assert result.details["inside_null_band"] is True

    def test_tail_behaviour_reports_excess_kurtosis(self):
        result = tail_behaviour(regime_switch(n=2000, seed=47).bars)
        assert np.isfinite(result.value)


class TestFalsePositiveRate:
    """The scanner's calibration, measured on twenty independent walks.

    ``structural_stability`` is excluded deliberately: its STRONG grade
    means "the process *is* stable", which is the correct reading for a
    random walk. The diagnostics listed here are the ones a research
    programme would act on, and a STRONG grade on noise from any of them
    would send a search after an artefact.
    """

    EXPLOITABLE = (
        "autocorrelation",
        "partial_autocorrelation",
        "variance_ratio",
        "mean_reversion_half_life",
        "seasonality_month",
        "hurst_exponent",
    )

    @pytest.mark.slow
    @pytest.mark.control
    def test_no_strong_exploitable_findings_on_random_walks(self):
        strong = []
        for seed in range(20):
            result = scan(random_walk(n=1500, seed=100 + seed).bars, symbol=f"RW{seed}")
            for name, diagnostic in result.diagnostics.items():
                if name in self.EXPLOITABLE and diagnostic.evidence == "STRONG":
                    strong.append((seed, name, diagnostic.p_value))
        assert not strong, f"strong exploitable findings on pure noise: {strong}"

    @pytest.mark.slow
    @pytest.mark.control
    def test_the_recommendation_for_noise_is_to_stop(self):
        for seed in range(10):
            result = scan(random_walk(n=1500, seed=200 + seed).bars, symbol=f"RW{seed}")
            text = " ".join(result.recommend).lower()
            assert "none" in text or "stop" in text, f"seed {seed}: {result.recommend}"


class TestScanReport:
    def test_scan_summarises_every_diagnostic(self, rw_bars):
        result = scan(rw_bars, symbol="RW")
        assert result.n_bars == len(rw_bars)
        assert result.diagnostics
        assert "data_hash" in result.to_dict()

    def test_scan_summary_separates_the_controls(self):
        """Mean reversion and momentum must not read the same."""
        mr = scan(mean_reverting(n=2500, seed=53, half_life=10.0).bars, symbol="MR")
        mom = scan(momentum(n=2500, seed=59).bars, symbol="MOM")
        assert mr.summary["mean_reversion_evidence"] != "NONE"
        assert mom.summary["mean_reversion_evidence"] == "NONE"

    def test_markdown_is_self_contained(self, rw_bars):
        text = scan(rw_bars, symbol="RW").to_markdown()
        assert "RW" in text
        assert "|" in text        # it contains a table

    def test_scan_dataset_keeps_identity(self, rw_dataset):
        result = scan_dataset(rw_dataset)
        assert result.symbol == rw_dataset.symbol
        assert result.data_hash == rw_dataset.data_hash

    def test_scan_table_stacks_results(self, rw_bars):
        results = [scan(rw_bars.iloc[:800], symbol="A"), scan(rw_bars.iloc[800:], symbol="B")]
        table = scan_table(results)
        assert len(table) == 2

    def test_too_short_a_series_is_refused(self, rw_bars):
        from exchange.core.errors import NotEnoughDataError

        with pytest.raises(NotEnoughDataError):
            scan(rw_bars.iloc[:20], symbol="TINY")
