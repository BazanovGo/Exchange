"""Bootstrap resampling for dependent data.

The i.i.d. bootstrap destroys exactly the structure that matters
----------------------------------------------------------------
Resampling returns one at a time assumes independence. Financial returns are
not independent — volatility clusters, drawdowns are runs of correlated
losses, and momentum is autocorrelation by definition. An i.i.d. bootstrap
therefore:

* understates the probability of long drawdowns (it breaks up the runs);
* destroys any autocorrelation the strategy was trading;
* gives confidence intervals that are too narrow, sometimes by a lot.

The audited project's Monte Carlo resampled trade returns independently, so
its ruin probabilities were optimistic by construction.

Block bootstraps preserve local dependence by resampling *stretches* of
consecutive observations.

``moving_block_bootstrap``
    Fixed-length overlapping blocks (Künsch 1989). Simple, and the
    dependence structure inside a block is preserved exactly.
``stationary_bootstrap``
    Geometrically distributed block lengths (Politis & Romano 1994). The
    resampled series is strictly stationary, which the fixed-block version is
    not, and it is far less sensitive to the choice of block length — the
    reason it is the default here.

Choosing a block length
-----------------------
Too short and dependence is destroyed; too long and there are too few
independent blocks to resample. :func:`optimal_block_length` implements the
Politis & White (2004) automatic rule, so the choice is data-driven rather
than a magic constant.

References
----------
Künsch, H. (1989). "The Jackknife and the Bootstrap for General Stationary
Observations." *Annals of Statistics* 17(3).
Politis, D. & Romano, J. (1994). "The Stationary Bootstrap." *JASA* 89(428).
Politis, D. & White, H. (2004). "Automatic Block-Length Selection for the
Dependent Bootstrap." *Econometric Reviews* 23(1).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from exchange.core.determinism import rng as make_rng
from exchange.core.errors import ConfigError, NotEnoughDataError


def _clean(values) -> np.ndarray:
    arr = np.asarray(pd.Series(values).to_numpy(), dtype=float)
    return arr[np.isfinite(arr)]


def optimal_block_length(values, *, max_lag: int | None = None) -> float:
    """Politis & White (2004) automatic block length for the stationary bootstrap.

    A pragmatic implementation: the block length is derived from the
    integrated autocorrelation of the series, truncated at the first lag that
    is statistically indistinguishable from zero. Returns at least 1.
    """
    arr = _clean(values)
    n = arr.size
    if n < 20:
        return 1.0
    max_lag = max_lag or min(int(np.ceil(2 * np.sqrt(n))), n // 4)
    demeaned = arr - arr.mean()
    denom = float(np.dot(demeaned, demeaned))
    if denom <= 0:
        return 1.0

    threshold = 2.0 / np.sqrt(n)
    g = 0.0   # sum of k * |rho_k|
    d = 1.0   # 1 + 2 * sum rho_k
    for k in range(1, max_lag + 1):
        rho = float(np.dot(demeaned[:-k], demeaned[k:]) / denom)
        if abs(rho) < threshold and k > 2:
            break
        g += 2.0 * k * abs(rho)
        d += 2.0 * rho
    if d <= 0 or g <= 0:
        return 1.0
    # b_opt = (2 * g^2 / D)^(1/3) * n^(1/3), with D = 2 * d^2 for the
    # stationary bootstrap.
    d_term = 2.0 * d**2
    b = (2.0 * g**2 / d_term) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    return float(min(max(b, 1.0), n / 4.0))


def stationary_bootstrap_indices(
    n: int,
    *,
    block_length: float,
    rng: np.random.Generator,
    size: int | None = None,
) -> np.ndarray:
    """Indices for one stationary-bootstrap resample.

    Each step either continues the current block (probability ``1 - p``) or
    jumps to a new uniformly random start (probability ``p``), where
    ``p = 1 / block_length``. Block lengths are therefore geometric with mean
    ``block_length``. Indices wrap around the end of the sample, which is
    what makes the resampled series stationary.
    """
    if n < 2:
        raise NotEnoughDataError("need at least 2 observations to bootstrap")
    size = size or n
    p = 1.0 / max(block_length, 1.0)
    out = np.empty(size, dtype=np.int64)
    idx = int(rng.integers(0, n))
    for t in range(size):
        out[t] = idx
        idx = int(rng.integers(0, n)) if rng.random() < p else (idx + 1) % n
    return out


def moving_block_bootstrap_indices(
    n: int,
    *,
    block_length: int,
    rng: np.random.Generator,
    size: int | None = None,
) -> np.ndarray:
    """Indices for one moving-block-bootstrap resample."""
    if block_length < 1:
        raise ConfigError("block_length must be >= 1")
    size = size or n
    n_blocks = int(np.ceil(size / block_length))
    starts = rng.integers(0, max(n - block_length + 1, 1), size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_length) % n for s in starts])
    return idx[:size]


@dataclass
class BootstrapResult:
    """Bootstrap distribution of one statistic."""

    observed: float
    samples: np.ndarray
    method: str
    block_length: float
    n_resamples: int

    def confidence_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """Percentile interval."""
        alpha = (1.0 - confidence) / 2.0
        finite = self.samples[np.isfinite(self.samples)]
        if finite.size < 10:
            return (np.nan, np.nan)
        return (float(np.quantile(finite, alpha)), float(np.quantile(finite, 1.0 - alpha)))

    def p_value_greater_than(self, value: float = 0.0) -> float:
        """One-sided p-value for ``H0: statistic <= value``.

        Uses the bootstrap distribution *recentred on the null*, which is the
        correct construction: the raw fraction of resamples above zero is a
        measure of the observed statistic, not a test of it.
        """
        finite = self.samples[np.isfinite(self.samples)]
        if finite.size < 10 or not np.isfinite(self.observed):
            return np.nan
        centred = finite - np.mean(finite)
        return float((np.sum(centred >= self.observed - value) + 1) / (finite.size + 1))

    @property
    def standard_error(self) -> float:
        finite = self.samples[np.isfinite(self.samples)]
        return float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan

    def to_dict(self) -> dict:
        lo, hi = self.confidence_interval()
        return {
            "observed": self.observed,
            "method": self.method,
            "block_length": self.block_length,
            "n_resamples": self.n_resamples,
            "standard_error": self.standard_error,
            "ci_low": lo,
            "ci_high": hi,
            "p_value": self.p_value_greater_than(0.0),
        }


def block_bootstrap(
    values,
    statistic: Callable[[np.ndarray], float],
    *,
    n_resamples: int = 2000,
    block_length: float | None = None,
    method: str = "stationary",
    seed: int = 0,
) -> BootstrapResult:
    """Bootstrap ``statistic`` while preserving serial dependence.

    Parameters
    ----------
    values:
        The observed series (usually per-period returns).
    statistic:
        Function of a 1-D array returning a scalar, e.g. the Sharpe ratio or
        the maximum drawdown of the implied equity curve.
    block_length:
        ``None`` selects it automatically (Politis & White).
    method:
        ``"stationary"`` (default), ``"moving_block"`` or ``"iid"``. The
        i.i.d. option exists so a report can *show* how much narrower — and
        how wrong — the naive interval is.
    """
    arr = _clean(values)
    if arr.size < 20:
        raise NotEnoughDataError(f"bootstrap needs at least 20 observations, got {arr.size}")
    rng = make_rng(seed)
    b = float(block_length) if block_length is not None else optimal_block_length(arr)

    samples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        if method == "stationary":
            idx = stationary_bootstrap_indices(arr.size, block_length=b, rng=rng)
        elif method == "moving_block":
            idx = moving_block_bootstrap_indices(arr.size, block_length=max(1, round(b)), rng=rng)
        elif method == "iid":
            idx = rng.integers(0, arr.size, size=arr.size)
        else:
            raise ConfigError(f"unknown bootstrap method {method!r}")
        try:
            samples[i] = float(statistic(arr[idx]))
        except Exception:
            samples[i] = np.nan

    return BootstrapResult(
        observed=float(statistic(arr)),
        samples=samples,
        method=method,
        block_length=b,
        n_resamples=n_resamples,
    )


def bootstrap_equity_paths(
    returns,
    *,
    n_paths: int = 1000,
    block_length: float | None = None,
    seed: int = 0,
    initial: float = 1.0,
) -> np.ndarray:
    """Resampled equity paths, ``(n_paths, T + 1)``.

    Used for the Monte Carlo stage: the distribution of terminal wealth and
    of maximum drawdown across paths says far more about a strategy's risk
    than the single historical path does. With a block bootstrap the paths
    keep the strategy's own clustering of losses, so the drawdown
    distribution is not artificially benign.
    """
    arr = _clean(returns)
    if arr.size < 20:
        raise NotEnoughDataError("need at least 20 returns to resample equity paths")
    rng = make_rng(seed)
    b = float(block_length) if block_length is not None else optimal_block_length(arr)
    paths = np.empty((n_paths, arr.size + 1), dtype=float)
    paths[:, 0] = initial
    for i in range(n_paths):
        idx = stationary_bootstrap_indices(arr.size, block_length=b, rng=rng)
        paths[i, 1:] = initial * np.cumprod(1.0 + arr[idx])
    return paths


def path_drawdowns(paths: np.ndarray) -> np.ndarray:
    """Maximum drawdown of each row of an equity-path matrix (negative)."""
    peak = np.maximum.accumulate(paths, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, paths / peak - 1.0, np.nan)
    return np.nanmin(dd, axis=1)


def monte_carlo_summary(
    returns,
    *,
    n_paths: int = 1000,
    block_length: float | None = None,
    seed: int = 0,
    ruin_threshold: float = -0.5,
) -> dict:
    """Distribution of outcomes across bootstrapped paths."""
    paths = bootstrap_equity_paths(returns, n_paths=n_paths, block_length=block_length, seed=seed)
    terminal = paths[:, -1] - 1.0
    dd = path_drawdowns(paths)
    return {
        "n_paths": int(n_paths),
        "median_total_return": float(np.median(terminal)),
        "p05_total_return": float(np.quantile(terminal, 0.05)),
        "p95_total_return": float(np.quantile(terminal, 0.95)),
        "prob_loss": float(np.mean(terminal < 0)),
        "median_max_drawdown": float(np.median(dd)),
        "p95_max_drawdown": float(np.quantile(dd, 0.05)),  # 5th pct = worst 5%
        "prob_ruin": float(np.mean(dd <= ruin_threshold)),
        "ruin_threshold": ruin_threshold,
    }
