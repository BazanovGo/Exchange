"""White's Reality Check and Hansen's Superior Predictive Ability test.

The question
------------
"I searched over 500 strategies and the best one has a Sharpe of 1.4. Is
*any* of them genuinely better than the benchmark, once I account for the
fact that I looked at 500?"

This is not the same question as the per-strategy Sharpe test, and it is not
the same as a Bonferroni correction either. Bonferroni treats the 500 tests
as independent; they are not — they are 500 highly correlated strategies on
the same data. Both tests here handle that dependence by bootstrapping the
*joint* distribution of all 500 performance series at once, so the
correlation structure is preserved rather than assumed away.

White's Reality Check (2000)
-----------------------------
Test statistic ``V = max_k sqrt(n) * mean(f_k)``, where ``f_k`` is model
``k``'s per-period performance *relative to the benchmark*. The null is that
no model beats the benchmark. The bootstrap distribution is built from
recentred resamples ``V* = max_k sqrt(n) * (mean(f*_k) - mean(f_k))`` and the
p-value is the fraction of ``V*`` exceeding ``V``.

Its known weakness: the least favourable configuration of the null assumes
*every* model is exactly at the benchmark. One hopeless model dragging the
maximum around inflates the p-value, so adding bad strategies to the search
makes the good one *harder* to detect. That is backwards.

Hansen's SPA (2005)
-------------------
Fixes exactly that, in two ways: it studentises each model by its own
standard error, and it recentres only those models that are not too far below
the benchmark to matter (a data-dependent threshold). SPA is therefore more
powerful, and it is the test the funnel uses; the Reality Check is reported
alongside because the gap between the two is diagnostic of how much dead
weight the search contained.

References
----------
White, H. (2000). "A Reality Check for Data Snooping." *Econometrica* 68(5).
Hansen, P. R. (2005). "A Test for Superior Predictive Ability." *Journal of
Business & Economic Statistics* 23(4).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from exchange.core.determinism import rng as make_rng
from exchange.core.errors import ConfigError, NotEnoughDataError
from exchange.stats.bootstrap import optimal_block_length, stationary_bootstrap_indices


@dataclass
class SnoopingTestResult:
    """Outcome of a data-snooping test over a family of strategies."""

    test: str
    statistic: float
    p_value: float
    n_models: int
    n_observations: int
    n_bootstrap: int
    block_length: float
    best_model: int
    best_mean_performance: float
    bootstrap_statistics: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    labels: list[str] | None = None
    extra: dict = field(default_factory=dict)

    @property
    def best_label(self) -> str:
        if self.labels is None:
            return str(self.best_model)
        return self.labels[self.best_model]

    def significant(self, alpha: float = 0.05) -> bool:
        return bool(np.isfinite(self.p_value) and self.p_value <= alpha)

    def to_dict(self) -> dict:
        return {
            "test": self.test,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "n_models": self.n_models,
            "n_observations": self.n_observations,
            "n_bootstrap": self.n_bootstrap,
            "block_length": self.block_length,
            "best_model": self.best_label,
            "best_mean_performance": self.best_mean_performance,
            "significant_at_5pct": self.significant(0.05),
            **self.extra,
        }

    def explain(self) -> str:
        verdict = "beats" if self.significant() else "does NOT beat"
        return (
            f"{self.test}: best of {self.n_models} models ({self.best_label}) {verdict} the "
            f"benchmark after accounting for the search (p = {self.p_value:.4f}, "
            f"statistic = {self.statistic:.3f})."
        )


def _prepare_performance(performance, benchmark=None) -> tuple[np.ndarray, list[str] | None]:
    """Return an ``(n, k)`` matrix of performance relative to the benchmark."""
    labels: list[str] | None = None
    if isinstance(performance, pd.DataFrame):
        labels = [str(c) for c in performance.columns]
        matrix = performance.to_numpy(dtype=float)
    else:
        matrix = np.asarray(performance, dtype=float)
    if matrix.ndim != 2:
        raise ConfigError(f"performance must be 2-D (n_obs x n_models), got shape {matrix.shape}")

    if benchmark is not None:
        bench = np.asarray(pd.Series(benchmark).to_numpy(), dtype=float)
        if bench.size != matrix.shape[0]:
            raise ConfigError("benchmark length does not match the performance matrix")
        matrix = matrix - bench[:, None]

    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    if matrix.shape[0] < 30:
        raise NotEnoughDataError(f"need at least 30 observations, got {matrix.shape[0]}")
    return matrix, labels


def _bootstrap_means(
    matrix: np.ndarray,
    *,
    n_bootstrap: int,
    block_length: float,
    seed: int,
    with_std: bool = False,
):
    """``(n_bootstrap, k)`` matrix of resampled column means.

    One index draw is shared across all columns per resample, which is what
    preserves the cross-sectional correlation between models. With
    ``with_std`` the per-resample sample standard deviations are returned as
    well — the denominator of a bootstrap-t replicate.
    """
    n, k = matrix.shape
    rng = make_rng(seed)
    out = np.empty((n_bootstrap, k), dtype=float)
    sd = np.empty((n_bootstrap, k), dtype=float) if with_std else None
    for b in range(n_bootstrap):
        idx = stationary_bootstrap_indices(n, block_length=block_length, rng=rng)
        sample = matrix[idx]
        out[b] = sample.mean(axis=0)
        if with_std:
            sd[b] = sample.std(axis=0, ddof=1)
    return (out, sd) if with_std else out


def reality_check(
    performance,
    benchmark=None,
    *,
    n_bootstrap: int = 1000,
    block_length: float | None = None,
    seed: int = 0,
) -> SnoopingTestResult:
    """White's Reality Check.

    Parameters
    ----------
    performance:
        ``(n_obs, n_models)`` per-period performance. If ``benchmark`` is
        given, these are raw returns and the benchmark is subtracted;
        otherwise they are already relative (excess) performance.
    block_length:
        Stationary-bootstrap block length. ``None`` selects it from the best
        model's series.
    """
    matrix, labels = _prepare_performance(performance, benchmark)
    n, k = matrix.shape
    means = matrix.mean(axis=0)
    best = int(np.argmax(means))
    b = float(block_length) if block_length is not None else optimal_block_length(matrix[:, best])

    statistic = float(np.sqrt(n) * means[best])
    boot_means = _bootstrap_means(matrix, n_bootstrap=n_bootstrap, block_length=b, seed=seed)
    # Recentre on the null: subtract each model's observed mean.
    boot_stats = np.max(np.sqrt(n) * (boot_means - means[None, :]), axis=1)
    p_value = float((np.sum(boot_stats >= statistic) + 1) / (n_bootstrap + 1))

    return SnoopingTestResult(
        test="White Reality Check",
        statistic=statistic,
        p_value=p_value,
        n_models=k,
        n_observations=n,
        n_bootstrap=n_bootstrap,
        block_length=b,
        best_model=best,
        best_mean_performance=float(means[best]),
        bootstrap_statistics=boot_stats,
        labels=labels,
    )


def superior_predictive_ability(
    performance,
    benchmark=None,
    *,
    n_bootstrap: int = 1000,
    block_length: float | None = None,
    seed: int = 0,
) -> SnoopingTestResult:
    """Hansen's SPA test.

    Improvements over the Reality Check, both implemented here:

    1. **Studentisation.** Each model's mean is divided by its own standard
       deviation (bootstrap-t: each replicate by the deviation of its own
       resample), so a model with a small but very stable edge is not buried
       by a volatile one.
    2. **Data-dependent recentring.** Only models whose mean exceeds
       ``-A_k`` participate in the null, where
       ``A_k = s_k * sqrt(2 * log(log(n)) / n)``. Models far below the
       benchmark are recognised as irrelevant instead of being assumed to sit
       exactly at it. This is what removes the Reality Check's perverse
       incentive to leave bad strategies out of the search.

    ``extra`` carries the lower and upper p-value bounds Hansen defines; the
    consistent p-value is the one reported as ``p_value``.
    """
    matrix, labels = _prepare_performance(performance, benchmark)
    n, k = matrix.shape
    means = matrix.mean(axis=0)
    b = float(block_length) if block_length is not None else optimal_block_length(
        matrix[:, int(np.argmax(means))]
    )

    boot_means, boot_sd = _bootstrap_means(
        matrix, n_bootstrap=n_bootstrap, block_length=b, seed=seed, with_std=True
    )
    # Studentisation is bootstrap-t: the observed mean is divided by the
    # sample standard deviation, each replicate by the standard deviation of
    # its own resample. The first version divided both by the bootstrap
    # standard error omega_k. On small samples that estimate is biased low,
    # so the observed t-statistics were inflated while the replicates had unit
    # variance by construction; the maximum over many models amplified the gap
    # and the test rejected pure noise in 35% of runs at n = 60 (nominal 5%).
    sd = matrix.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-12, sd, np.nan)
    boot_sd = np.where(boot_sd > 1e-12, boot_sd, np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        t_stats = np.sqrt(n) * means / sd
    statistic = float(np.nanmax(np.maximum(t_stats, 0.0)))
    best = int(np.nanargmax(np.where(np.isfinite(t_stats), t_stats, -np.inf)))

    threshold = sd * np.sqrt(2.0 * np.log(max(np.log(n), 1.0001)) / n)
    keep = means >= -threshold                       # consistent recentring
    keep_lower = means >= 0.0                        # lower bound: only good models
    keep_upper = np.ones(k, dtype=bool)              # upper bound: all models

    def _p(mask: np.ndarray) -> float:
        centred = boot_means - np.where(mask, means, 0.0)[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            boot_t = np.sqrt(n) * centred / boot_sd
        boot_stat = np.nan_to_num(np.nanmax(np.maximum(boot_t, 0.0), axis=1))
        return float((np.sum(boot_stat >= statistic) + 1) / (n_bootstrap + 1))

    return SnoopingTestResult(
        test="Hansen SPA",
        statistic=statistic,
        p_value=_p(keep),
        n_models=k,
        n_observations=n,
        n_bootstrap=n_bootstrap,
        block_length=b,
        best_model=best,
        best_mean_performance=float(means[best]),
        labels=labels,
        extra={
            "p_value_lower": _p(keep_lower),
            "p_value_upper": _p(keep_upper),
            "n_models_recentred": int(keep.sum()),
        },
    )


def snooping_report(
    performance,
    benchmark=None,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Both tests side by side.

    A large gap between the Reality Check's p-value and SPA's is a sign the
    candidate set contained many hopeless strategies. That is not itself a
    problem, but it means the Reality Check's verdict should be discounted and
    SPA's taken as the operative one.
    """
    rc = reality_check(performance, benchmark, n_bootstrap=n_bootstrap, seed=seed)
    spa = superior_predictive_ability(performance, benchmark, n_bootstrap=n_bootstrap, seed=seed)
    return pd.DataFrame([rc.to_dict(), spa.to_dict()])
