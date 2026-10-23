"""Statistical validation: Sharpe inference, PBO, purged CV, bootstrap, snooping tests."""

from __future__ import annotations

from exchange.stats.bootstrap import (
    BootstrapResult,
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
    Split,
    WalkForward,
    assert_no_overlap,
    split_summary,
)
from exchange.stats.multiple_testing import (
    METHODS,
    CorrectionResult,
    benjamini_hochberg,
    benjamini_yekutieli,
    bonferroni,
    correct,
    correction_table,
    effective_n_trials,
    estimate_pi0,
    expected_false_discoveries,
    holm,
)
from exchange.stats.pbo import (
    PBOResult,
    logit_distribution,
    probability_of_backtest_overfitting,
)
from exchange.stats.reality_check import (
    SnoopingTestResult,
    reality_check,
    snooping_report,
    superior_predictive_ability,
)
from exchange.stats.sharpe import (
    DeflatedSharpeResult,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_confidence_interval,
    sharpe_per_period,
    sharpe_standard_error,
)

__all__ = [
    "METHODS",
    "BootstrapResult",
    "CombinatorialPurgedCV",
    "CorrectionResult",
    "DeflatedSharpeResult",
    "PBOResult",
    "PurgedKFold",
    "SnoopingTestResult",
    "Split",
    "WalkForward",
    "assert_no_overlap",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "block_bootstrap",
    "bonferroni",
    "bootstrap_equity_paths",
    "correct",
    "correction_table",
    "deflated_sharpe_ratio",
    "effective_n_trials",
    "estimate_pi0",
    "expected_false_discoveries",
    "expected_max_sharpe",
    "holm",
    "logit_distribution",
    "minimum_track_record_length",
    "monte_carlo_summary",
    "moving_block_bootstrap_indices",
    "optimal_block_length",
    "path_drawdowns",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "reality_check",
    "sharpe_confidence_interval",
    "sharpe_per_period",
    "sharpe_standard_error",
    "snooping_report",
    "split_summary",
    "stationary_bootstrap_indices",
    "superior_predictive_ability",
]
