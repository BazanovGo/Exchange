"""Multiple-hypothesis correction.

The problem, stated plainly
---------------------------
Test one pre-registered hypothesis at alpha = 0.05 and you accept a 5% chance
of a false positive. Test 1,000 hypotheses at alpha = 0.05 on pure noise and
you should *expect* 50 "discoveries", every one of them false. A parameter
grid of 10 x 10 x 10 is 1,000 hypotheses whether or not anybody wrote them
down as such — and the audited project ran nineteen studies with grids from
four to several hundred combinations each, against one ~2,000-bar dataset,
with no correction of any kind.

This module makes the count explicit and applies one of two families:

**Family-wise error rate (FWER)** — control the probability of *any* false
positive. ``bonferroni`` (simple, conservative) and ``holm`` (uniformly more
powerful than Bonferroni, same guarantee). Use when a single false discovery
is expensive — e.g. deciding to allocate capital.

**False discovery rate (FDR)** — control the *expected proportion* of false
positives among the rejections. ``benjamini_hochberg`` assumes independence
or positive dependence; ``benjamini_yekutieli`` is valid under arbitrary
dependence at the cost of a ``ln(m)`` factor. Use when screening many
candidates and a few false leads are tolerable — the normal situation in
early-stage research.

Which to use here
-----------------
The research funnel applies **BH-FDR at the screening stage** (a wide search
where some false leads are acceptable) and **Holm-FWER at the promotion
stage** (where a false positive means paper trading a non-existent edge).
Both are reported, always, so the reader can see how much of the result
depends on the choice.

References
----------
Holm, S. (1979). *Scandinavian Journal of Statistics* 6(2).
Benjamini, Y. & Hochberg, Y. (1995). *JRSS B* 57(1).
Benjamini, Y. & Yekutieli, D. (2001). *Annals of Statistics* 29(4).
Storey, J. (2002). *JRSS B* 64(3) — for pi0 estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from exchange.core.errors import ConfigError

METHODS = ("bonferroni", "holm", "benjamini_hochberg", "benjamini_yekutieli")


@dataclass
class CorrectionResult:
    """Outcome of correcting a family of p-values."""

    method: str
    alpha: float
    p_values: np.ndarray
    adjusted: np.ndarray
    rejected: np.ndarray
    n_tests: int
    labels: list[str] | None = None

    @property
    def n_rejected(self) -> int:
        return int(self.rejected.sum())

    @property
    def survivors(self) -> list[str]:
        if self.labels is None:
            return [str(i) for i in np.flatnonzero(self.rejected)]
        return [self.labels[i] for i in np.flatnonzero(self.rejected)]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "label": self.labels or list(range(self.n_tests)),
                "p_value": self.p_values,
                "adjusted_p": self.adjusted,
                "rejected": self.rejected,
            }
        ).sort_values("p_value").reset_index(drop=True)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "alpha": self.alpha,
            "n_tests": self.n_tests,
            "n_rejected": self.n_rejected,
            "survivors": self.survivors[:50],
            "min_p": float(np.nanmin(self.p_values)) if self.n_tests else np.nan,
            "min_adjusted_p": float(np.nanmin(self.adjusted)) if self.n_tests else np.nan,
        }

    def explain(self) -> str:
        return (
            f"{self.method}: {self.n_rejected} of {self.n_tests} hypotheses survive at "
            f"alpha={self.alpha}. Smallest raw p={np.nanmin(self.p_values):.2e}, smallest "
            f"adjusted p={np.nanmin(self.adjusted):.2e}."
        )


def _prepare(p_values) -> np.ndarray:
    arr = np.asarray(pd.Series(p_values).to_numpy(), dtype=float)
    if arr.ndim != 1:
        raise ConfigError("p_values must be one-dimensional")
    if arr.size == 0:
        raise ConfigError("no p-values supplied")
    finite = np.isfinite(arr)
    if np.any(finite & ((arr < 0) | (arr > 1))):
        raise ConfigError("p-values must lie in [0, 1]")
    # A non-finite p-value cannot be rejected; treat it as 1.0.
    return np.where(finite, arr, 1.0)


def bonferroni(p_values, alpha: float = 0.05) -> CorrectionResult:
    """Control FWER by scaling every p-value by the number of tests."""
    p = _prepare(p_values)
    adjusted = np.minimum(p * p.size, 1.0)
    return CorrectionResult("bonferroni", alpha, p, adjusted, adjusted <= alpha, p.size)


def holm(p_values, alpha: float = 0.05) -> CorrectionResult:
    """Holm's step-down procedure: FWER control, strictly more powerful than
    Bonferroni and requiring no extra assumptions."""
    p = _prepare(p_values)
    m = p.size
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return CorrectionResult("holm", alpha, p, adjusted, adjusted <= alpha, m)


def benjamini_hochberg(p_values, alpha: float = 0.05) -> CorrectionResult:
    """BH step-up procedure: controls FDR under independence or PRDS.

    The adjusted p-values are the *q-values*: the smallest FDR level at which
    each hypothesis would be rejected.
    """
    p = _prepare(p_values)
    m = p.size
    order = np.argsort(p)
    ranks = np.arange(1, m + 1)
    # Step-up: enforce monotonicity from the largest p downward.
    adjusted_sorted = np.minimum.accumulate((m / ranks * p[order])[::-1])[::-1]
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return CorrectionResult("benjamini_hochberg", alpha, p, adjusted, adjusted <= alpha, m)


def benjamini_yekutieli(p_values, alpha: float = 0.05) -> CorrectionResult:
    """BY procedure: FDR control valid under *arbitrary* dependence.

    Costs a factor of ``sum(1/i)`` ~ ``ln(m) + 0.577`` in power. Appropriate
    for a parameter grid, whose neighbouring configurations are strongly and
    unknowably dependent — which is exactly the case BH does not cover.
    """
    p = _prepare(p_values)
    m = p.size
    c_m = float(np.sum(1.0 / np.arange(1, m + 1)))
    order = np.argsort(p)
    ranks = np.arange(1, m + 1)
    adjusted_sorted = np.minimum.accumulate((c_m * m / ranks * p[order])[::-1])[::-1]
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return CorrectionResult("benjamini_yekutieli", alpha, p, adjusted, adjusted <= alpha, m)


_DISPATCH = {
    "bonferroni": bonferroni,
    "holm": holm,
    "benjamini_hochberg": benjamini_hochberg,
    "bh": benjamini_hochberg,
    "fdr": benjamini_hochberg,
    "benjamini_yekutieli": benjamini_yekutieli,
    "by": benjamini_yekutieli,
}


def correct(p_values, *, method: str = "benjamini_hochberg", alpha: float = 0.05, labels=None):
    """Apply a named correction."""
    key = str(method).lower()
    if key not in _DISPATCH:
        raise ConfigError(f"unknown correction {method!r}; known: {sorted(set(_DISPATCH))}")
    result = _DISPATCH[key](p_values, alpha)
    if labels is not None:
        result.labels = [str(x) for x in labels]
    return result


def correction_table(p_values, *, alpha: float = 0.05, labels=None) -> pd.DataFrame:
    """Every correction side by side.

    Reporting all four is deliberate: the gap between BH and Holm is a direct
    measure of how much a conclusion depends on the reviewer's tolerance for
    false positives, and hiding that choice inside one default is how a
    marginal result gets presented as a finding.
    """
    p = _prepare(p_values)
    out = pd.DataFrame({"label": labels if labels is not None else list(range(p.size)), "p_value": p})
    for method in METHODS:
        res = _DISPATCH[method](p, alpha)
        out[f"{method}_p"] = res.adjusted
        out[f"{method}_reject"] = res.rejected
    return out.sort_values("p_value").reset_index(drop=True)


def estimate_pi0(p_values, *, lambda_: float = 0.5) -> float:
    """Storey's estimate of the fraction of true null hypotheses.

    ``pi0`` near 1.0 means essentially everything tested was noise — a
    useful, sobering summary of a large grid search. Values above 1 are
    clipped to 1.
    """
    p = _prepare(p_values)
    if not 0 < lambda_ < 1:
        raise ConfigError("lambda_ must be in (0, 1)")
    above = float(np.sum(p > lambda_))
    return float(min(above / (p.size * (1.0 - lambda_)), 1.0))


def expected_false_discoveries(p_values, *, alpha: float = 0.05) -> float:
    """Expected number of false positives among the raw rejections.

    ``pi0 * m * alpha``. Put next to the observed rejection count, this is the
    most legible summary of a multiple-testing problem there is: "you found 7;
    you would have expected 12 by chance" ends the discussion.
    """
    p = _prepare(p_values)
    return float(estimate_pi0(p) * p.size * alpha)


def effective_n_trials(p_values, *, correlation: float | None = None) -> float:
    """Effective number of independent tests in a correlated family.

    A parameter grid is not ``m`` independent tests: neighbouring
    configurations are nearly the same strategy. If the average pairwise
    correlation ``r`` of the test statistics is known, the effective count is
    approximately ``m / (1 + (m - 1) * r)``. This is the number to feed to
    :func:`~exchange.stats.sharpe.deflated_sharpe_ratio` when the raw grid
    size would be absurdly punitive.

    Passing ``correlation=None`` returns the raw count — the conservative
    default, since an unjustified reduction here is a way of quietly
    weakening the very correction that matters.
    """
    m = float(_prepare(p_values).size)
    if correlation is None:
        return m
    r = float(np.clip(correlation, 0.0, 0.999))
    return float(m / (1.0 + (m - 1.0) * r))
