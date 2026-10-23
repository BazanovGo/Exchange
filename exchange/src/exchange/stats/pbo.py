"""Probability of Backtest Overfitting (PBO) via CSCV.

The question PBO answers
------------------------
Not "is this strategy good?" but "**does my selection procedure work?**"

Given a matrix of returns for N candidate configurations over T periods,
CSCV repeatedly splits the periods into two halves, picks the configuration
that looks best in one half, and checks where that choice lands among all N
in the other half. If selection carries information, the in-sample winner
should rank in the upper half out-of-sample most of the time. If it lands
below median as often as not, the selection procedure is choosing noise.

    ``PBO = P( the IS-optimal configuration ranks below median OOS )``

A PBO above ~0.5 means the search is worse than a coin flip, and the
reported best result is an artefact of the search itself. Critically, PBO is
a property of the *procedure and the grid*, not of any single strategy —
which is why it catches the failure mode that per-strategy statistics
cannot: a huge grid where every point is noise will always produce an
impressive winner.

Method (Bailey, Borwein, López de Prado & Zhu 2017)
---------------------------------------------------
1. Split the T periods into ``S`` disjoint contiguous submatrices.
2. For each of the ``C(S, S/2)`` ways to choose half of them as in-sample:
   a. ``n* = argmax`` in-sample performance across the N configurations;
   b. compute ``n*``'s out-of-sample **relative rank** ``w in (0, 1)``;
   c. the logit ``lambda = ln(w / (1 - w))``.
3. ``PBO`` is the fraction of splits with ``lambda <= 0``.

Contiguous (rather than random) submatrices are used here, departing
slightly from the original paper's presentation: financial returns are
serially correlated, and randomly interleaving periods leaks information
between the halves. The distribution of ``lambda`` is reported alongside PBO
because its shape is informative — a bimodal lambda means the grid contains
two genuinely different regimes rather than one noisy surface.

Reference
---------
Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2017). "The
Probability of Backtest Overfitting." *Journal of Computational Finance*
20(4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from exchange.core.errors import ConfigError, NotEnoughDataError
from exchange.stats.sharpe import sharpe_per_period


@dataclass
class PBOResult:
    """CSCV output."""

    pbo: float                      # probability of backtest overfitting
    n_configurations: int
    n_observations: int
    n_splits: int
    n_partitions: int
    performance_degradation: float  # OLS slope of OOS on IS across splits
    prob_oos_loss: float            # P(selected config loses money OOS)
    logits: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    oos_ranks: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    is_performance: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    oos_performance: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))

    @property
    def median_logit(self) -> float:
        return float(np.median(self.logits)) if self.logits.size else np.nan

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.pbo):
            return "UNDETERMINED"
        if self.pbo <= 0.20:
            return "SELECTION IS INFORMATIVE"
        if self.pbo <= 0.50:
            return "SELECTION IS WEAK"
        return "SELECTION IS OVERFIT"

    def to_dict(self) -> dict:
        return {
            "pbo": self.pbo,
            "verdict": self.verdict,
            "n_configurations": self.n_configurations,
            "n_observations": self.n_observations,
            "n_splits": self.n_splits,
            "n_partitions": self.n_partitions,
            "median_logit": self.median_logit,
            "performance_degradation": self.performance_degradation,
            "prob_oos_loss": self.prob_oos_loss,
        }

    def explain(self) -> str:
        return (
            f"PBO = {self.pbo:.1%} over {self.n_splits} splits of {self.n_configurations} "
            f"configurations ({self.verdict}). The in-sample winner ranked below the OOS median in "
            f"{self.pbo:.0%} of splits; OOS-on-IS slope {self.performance_degradation:+.2f}, "
            f"P(OOS loss) = {self.prob_oos_loss:.1%}."
        )


def probability_of_backtest_overfitting(
    returns_matrix,
    *,
    n_partitions: int = 10,
    performance=None,
    max_splits: int | None = 1000,
) -> PBOResult:
    """Run CSCV on a ``(T, N)`` matrix of per-period returns.

    Parameters
    ----------
    returns_matrix:
        ``T x N`` array or DataFrame: one column per candidate configuration,
        one row per period. Columns must be aligned in time.
    n_partitions:
        ``S``, the number of contiguous blocks. Must be even. Larger ``S``
        gives more splits (``C(S, S/2)``) but shorter blocks; 10 gives 252
        splits and is a good default.
    performance:
        Callable mapping a 1-D return array to a scalar. Defaults to the
        per-period Sharpe ratio. Use something else when the selection
        criterion in the actual research was something else — PBO must
        measure the procedure that was really used.
    max_splits:
        Cap on the number of combinations evaluated (they are subsampled
        deterministically by striding, not randomly, so the result is
        reproducible). ``None`` evaluates all of them.
    """
    matrix = _as_matrix(returns_matrix)
    n_obs, n_cfg = matrix.shape
    if n_cfg < 2:
        raise ConfigError("PBO needs at least 2 configurations to choose between")
    if n_partitions % 2 != 0:
        raise ConfigError("n_partitions must be even so the sample splits into halves")
    if n_obs < n_partitions * 4:
        raise NotEnoughDataError(
            f"{n_obs} observations is too few for {n_partitions} partitions "
            f"(need >= {n_partitions * 4})"
        )

    score = performance or sharpe_per_period
    bounds = np.linspace(0, n_obs, n_partitions + 1).astype(int)
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_partitions)]

    half = n_partitions // 2
    combos = list(combinations(range(n_partitions), half))
    if max_splits is not None and len(combos) > max_splits:
        stride = len(combos) / max_splits
        combos = [combos[int(i * stride)] for i in range(max_splits)]

    logits: list[float] = []
    ranks: list[float] = []
    is_perf: list[float] = []
    oos_perf: list[float] = []

    for combo in combos:
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(n_partitions) if i not in combo])

        is_scores = np.array([score(matrix[is_idx, j]) for j in range(n_cfg)])
        oos_scores = np.array([score(matrix[oos_idx, j]) for j in range(n_cfg)])
        if not np.isfinite(is_scores).any() or not np.isfinite(oos_scores).any():
            continue

        best = int(np.nanargmax(np.where(np.isfinite(is_scores), is_scores, -np.inf)))
        finite_oos = np.where(np.isfinite(oos_scores), oos_scores, -np.inf)

        # Relative rank of the IS winner within the OOS scores, in (0, 1).
        rank = float((finite_oos <= finite_oos[best]).sum()) / (n_cfg + 1)
        rank = min(max(rank, 1.0 / (n_cfg + 1)), n_cfg / (n_cfg + 1))
        logits.append(float(np.log(rank / (1.0 - rank))))
        ranks.append(rank)
        is_perf.append(float(is_scores[best]))
        oos_perf.append(float(oos_scores[best]))

    if not logits:
        raise NotEnoughDataError("no CSCV split produced a finite performance score")

    logit_arr = np.asarray(logits)
    is_arr, oos_arr = np.asarray(is_perf), np.asarray(oos_perf)
    pbo = float(np.mean(logit_arr <= 0.0))

    # How much of the in-sample edge survives: the slope of a line fitted to
    # (IS score, OOS score) pairs across splits. A slope near zero means
    # in-sample performance carries no information about out-of-sample.
    mask = np.isfinite(is_arr) & np.isfinite(oos_arr)
    if mask.sum() >= 3 and np.std(is_arr[mask]) > 0:
        slope = float(np.polyfit(is_arr[mask], oos_arr[mask], 1)[0])
    else:
        slope = np.nan

    return PBOResult(
        pbo=pbo,
        n_configurations=n_cfg,
        n_observations=n_obs,
        n_splits=len(logits),
        n_partitions=n_partitions,
        performance_degradation=slope,
        prob_oos_loss=float(np.mean(oos_arr < 0)) if oos_arr.size else np.nan,
        logits=logit_arr,
        oos_ranks=np.asarray(ranks),
        is_performance=is_arr,
        oos_performance=oos_arr,
    )


def _as_matrix(data) -> np.ndarray:
    arr = data.to_numpy(dtype=float) if isinstance(data, pd.DataFrame) else np.asarray(data, dtype=float)
    if arr.ndim != 2:
        raise ConfigError(f"expected a 2-D (T x N) matrix, got shape {arr.shape}")
    # Non-finite returns break every score; replace with zero and let the
    # per-column scores reflect the reduced information.
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def logit_distribution(result: PBOResult, bins: int = 20) -> pd.DataFrame:
    """Histogram of the CSCV logits, for the report.

    Shape matters: a tight distribution centred well above zero is a healthy
    search; a wide one straddling zero means the ranking is noise even if the
    mean happens to be positive.
    """
    counts, edges = np.histogram(result.logits, bins=bins)
    return pd.DataFrame(
        {
            "lower": edges[:-1],
            "upper": edges[1:],
            "count": counts,
            "fraction": counts / max(counts.sum(), 1),
        }
    )
