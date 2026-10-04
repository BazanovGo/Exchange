"""Sharpe-ratio inference: standard errors, PSR and the Deflated Sharpe Ratio.

Why a point estimate of the Sharpe ratio is not evidence
--------------------------------------------------------
A Sharpe ratio estimated from a finite sample is a random variable. Its
standard error under i.i.d. normal returns is roughly
``sqrt((1 + SR^2/2)/n)`` per period — so on three years of daily data
(n ~ 756) an *annual* Sharpe of 1.0 carries a standard error of about 0.37.
Two thirds of the interval [0.26, 1.74] is an ordinary outcome for a
strategy whose true Sharpe is near zero. Reporting "Sharpe 1.0" without
that interval is reporting noise as a finding.

Three corrections matter and are implemented here:

1. **Non-normality** (Lo 2002; Mertens 2002). Negative skew and fat tails —
   the signature of every short-volatility strategy — inflate the naive
   Sharpe and shrink its apparent standard error. Note the correction is
   *sign-dependent*: the ``-g3 * SR`` term widens the interval for a
   positive Sharpe with negative skew, which is the case that matters.
2. **Autocorrelation** (Lo 2002). Serially correlated returns make the naive
   ``sqrt(T)`` annualisation wrong, usually optimistically.
3. **Selection** (Bailey & López de Prado 2014). The Sharpe you report is
   the *maximum* over everything you tried. The expected maximum of N
   independent zero-Sharpe trials is strictly positive and grows with N. The
   Deflated Sharpe Ratio is the correction, and it is the single most
   important number in this package — the audited project had no equivalent
   while running thousands of configurations against one dataset.

References
----------
Lo, A. (2002). "The Statistics of Sharpe Ratios." *Financial Analysts
Journal* 58(4).
Mertens, E. (2002). "Comments on Variance of the IID Estimator in Lo (2002)."
Bailey, D. & López de Prado, M. (2012). "The Sharpe Ratio Efficient
Frontier." *Journal of Risk* 15(2).
Bailey, D. & López de Prado, M. (2014). "The Deflated Sharpe Ratio."
*Journal of Portfolio Management* 40(5).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps

from exchange.core.errors import NotEnoughDataError

EULER_MASCHERONI = 0.5772156649015329


def _clean(returns) -> np.ndarray:
    arr = np.asarray(pd.Series(returns).to_numpy(), dtype=float)
    return arr[np.isfinite(arr)]


def sharpe_per_period(returns) -> float:
    """Sharpe ratio in the sampling frequency's own units (not annualised).

    Every function in this module works in per-period units, because that is
    where the distribution theory lives. Annualise only for display.
    """
    values = _clean(returns)
    if values.size < 2:
        return np.nan
    sd = np.std(values, ddof=1)
    return float(np.mean(values) / sd) if sd > 0 else np.nan


def sharpe_standard_error(
    returns,
    *,
    adjust_for_non_normality: bool = True,
    autocorrelation_lags: int = 0,
) -> float:
    """Standard error of the per-period Sharpe ratio estimate.

    Under i.i.d. normality (Lo 2002, eq. 9)::

        SE(SR) = sqrt((1 + SR^2 / 2) / n)

    With non-normal returns the Mertens (2002) expansion adds skewness and
    excess-kurtosis terms::

        SE(SR) = sqrt((1 - g3*SR + (g4 - 1)/4 * SR^2) / (n - 1))

    where ``g3`` is skewness and ``g4`` is *non-excess* kurtosis.

    ``autocorrelation_lags > 0`` applies Lo's serial-correlation correction,
    which matters for any strategy holding positions across many bars.
    """
    values = _clean(returns)
    n = values.size
    if n < 8:
        raise NotEnoughDataError(f"need at least 8 observations, got {n}")
    sr = sharpe_per_period(values)
    if not np.isfinite(sr):
        return np.nan

    if adjust_for_non_normality:
        skew = float(sps.skew(values, bias=False))
        kurt = float(sps.kurtosis(values, fisher=False, bias=False))  # non-excess
        variance = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2) / (n - 1)
    else:
        variance = (1.0 + 0.5 * sr**2) / n

    se = float(np.sqrt(max(variance, 0.0)))
    if autocorrelation_lags > 0:
        se *= _lo_autocorrelation_inflation(values, autocorrelation_lags)
    return se


def _lo_autocorrelation_inflation(values: np.ndarray, lags: int) -> float:
    """Lo (2002) serial-correlation adjustment factor for the Sharpe SE.

    Lo shows that under serial correlation the ``sqrt(q)`` scaling of a
    Sharpe ratio to a ``q``-period horizon must be replaced by

        ``eta(q) = q / sqrt(q + 2 * sum_{k=1}^{q-1} (q - k) * rho_k)``

    The ratio ``sqrt(q) / eta(q) = sqrt((q + 2*sum(...)) / q)`` is therefore
    how much the i.i.d. formula understates the true dispersion. Positive
    autocorrelation (a slow-moving position book) gives a factor above 1 and
    widens the interval; negative autocorrelation narrows it.
    """
    q = lags + 1
    demeaned = values - values.mean()
    denom = float(np.dot(demeaned, demeaned))
    if denom <= 0:
        return 1.0
    total = float(q)
    for k in range(1, q):
        rho = float(np.dot(demeaned[:-k], demeaned[k:]) / denom)
        total += 2.0 * (q - k) * rho
    if total <= 0:
        return 1.0
    return float(np.sqrt(total / q))


def sharpe_confidence_interval(returns, *, confidence: float = 0.95, **kwargs) -> tuple[float, float]:
    """Two-sided confidence interval for the per-period Sharpe ratio."""
    sr = sharpe_per_period(returns)
    se = sharpe_standard_error(returns, **kwargs)
    if not np.isfinite(sr) or not np.isfinite(se):
        return (np.nan, np.nan)
    z = float(sps.norm.ppf(0.5 + confidence / 2.0))
    return (sr - z * se, sr + z * se)


def probabilistic_sharpe_ratio(returns, *, benchmark_sr: float = 0.0) -> float:
    """PSR: probability that the true Sharpe exceeds ``benchmark_sr``.

    Bailey & López de Prado (2012)::

        PSR = Phi( (SR - SR*) * sqrt(n - 1)
                   / sqrt(1 - g3*SR + (g4 - 1)/4 * SR^2) )

    with ``SR`` and ``SR*`` in per-period units, ``g3`` skewness and ``g4``
    non-excess kurtosis. A PSR of 0.95 means: given this sample's length,
    skew and kurtosis, there is a 95% posterior-like probability that the
    true Sharpe beats the benchmark.

    On its own PSR does **not** correct for selection. Use
    :func:`deflated_sharpe_ratio` for that.
    """
    values = _clean(returns)
    n = values.size
    if n < 8:
        raise NotEnoughDataError(f"PSR needs at least 8 observations, got {n}")
    sr = sharpe_per_period(values)
    if not np.isfinite(sr):
        return np.nan
    skew = float(sps.skew(values, bias=False))
    kurt = float(sps.kurtosis(values, fisher=False, bias=False))
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return np.nan
    z = (sr - benchmark_sr) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(sps.norm.cdf(z))


def expected_max_sharpe(n_trials: int, *, sharpe_variance: float = 1.0) -> float:
    """Expected maximum Sharpe under the null of *no* skill.

    Bailey & López de Prado (2014), eq. 3. If ``N`` independent strategies
    all have a true Sharpe of zero, the best of them still shows

        ``E[max SR] ~ sqrt(V) * [ (1-g) Phi^-1(1 - 1/N) + g Phi^-1(1 - 1/(N e)) ]``

    where ``g`` is the Euler-Mascheroni constant and ``V`` is the variance of
    the Sharpe estimates across trials. This is the benchmark a reported
    Sharpe must beat to mean anything.

    The growth is slow (a maximum of Gaussians, ~sqrt(2 ln N)) but
    relentless: 10 trials gives ~1.54 standard errors, 1,000 gives ~3.24,
    100,000 gives ~4.42. A grid search over 10,000 parameter combinations is
    10,000 trials, whatever the researcher calls it.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if n_trials == 1:
        return 0.0
    sd = np.sqrt(max(sharpe_variance, 0.0))
    g = EULER_MASCHERONI
    a = sps.norm.ppf(1.0 - 1.0 / n_trials)
    b = sps.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(sd * ((1.0 - g) * a + g * b))


@dataclass(frozen=True)
class DeflatedSharpeResult:
    """The DSR verdict for one selected strategy."""

    sharpe: float                 # per-period Sharpe of the selected strategy
    sharpe_annualised: float
    n_observations: int
    n_trials: int
    trial_sharpe_variance: float
    expected_max_sharpe: float    # the benchmark the selection must beat
    deflated_sharpe_ratio: float  # probability the true Sharpe exceeds it
    probabilistic_sharpe_ratio: float  # PSR against zero, for comparison
    skewness: float
    kurtosis: float
    significant: bool             # DSR >= threshold

    def to_dict(self) -> dict:
        return asdict(self)

    def explain(self) -> str:
        return (
            f"Selected Sharpe {self.sharpe_annualised:.2f} (annualised) from {self.n_trials} trial(s). "
            f"Under the null of no skill the best of {self.n_trials} trials would be expected to show "
            f"{self.expected_max_sharpe:.3f} per period; the observed {self.sharpe:.3f} gives "
            f"DSR = {self.deflated_sharpe_ratio:.3f}"
            + (" — significant." if self.significant else " — NOT significant after deflation.")
        )


def _null_sharpe_variance(values: np.ndarray) -> float:
    """Sampling variance of a Sharpe estimate under no skill.

    ``Var(SR_hat) ~ (1 + SR^2 / 2) / n`` (Lo 2002), evaluated at the
    observed Sharpe. Used as the default spread of the trials when the
    caller does not say how dispersed the search actually was.
    """
    n = values.size
    if n < 2:
        return 1.0
    sr = sharpe_per_period(values)
    sr = sr if np.isfinite(sr) else 0.0
    return float((1.0 + 0.5 * sr**2) / n)


def deflated_sharpe_ratio(
    returns,
    *,
    n_trials: int,
    trial_sharpes=None,
    trial_sharpe_variance: float | None = None,
    periods_per_year: float = 252.0,
    threshold: float = 0.95,
) -> DeflatedSharpeResult:
    """Deflated Sharpe Ratio for a strategy selected out of ``n_trials``.

    Parameters
    ----------
    returns:
        Per-period returns of the **selected** strategy.
    n_trials:
        How many configurations were evaluated before this one was chosen.
        Count everything: grid points, timeframes, symbols, exit variants,
        and every earlier idea tested on the same data. Under-counting here
        is the most common way to smuggle an overfit result past this test.
    trial_sharpes:
        Per-period Sharpe ratios of all trials. When supplied, their variance
        is used for ``V`` — far better than assuming 1.0, because a tightly
        clustered grid has much less selection potential than a widely
        dispersed one.
    trial_sharpe_variance:
        Explicit ``V``, overriding ``trial_sharpes``.
    threshold:
        DSR level treated as significant. 0.95 by convention.

    Notes
    -----
    The independence assumption behind :func:`expected_max_sharpe` is
    violated by a parameter grid, whose neighbouring points are highly
    correlated. Using the *observed* variance of the trials' Sharpes (via
    ``trial_sharpes``) partially compensates: a correlated grid produces a
    smaller spread and therefore a smaller deflation benchmark. The result is
    still an approximation, and it errs toward the conservative side — the
    correct direction for a falsification-first system.
    """
    values = _clean(returns)
    n = values.size
    if n < 8:
        raise NotEnoughDataError(f"DSR needs at least 8 observations, got {n}")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")

    sr = sharpe_per_period(values)
    if trial_sharpe_variance is not None:
        variance = float(trial_sharpe_variance)
    elif trial_sharpes is not None:
        arr = _clean(trial_sharpes)
        variance = float(np.var(arr, ddof=1)) if arr.size > 1 else _null_sharpe_variance(values)
    else:
        # No information about how the trials were spread. Fall back to the
        # variance a *null* trial's Sharpe estimate would have on a sample
        # this long, which is the floor: any real grid disperses at least
        # this much. The literature's V = 1.0 is only meaningful for
        # annualised Sharpes; applying it to per-period returns would set a
        # benchmark hundreds of times too high and reject everything, which
        # looks conservative and is in fact just wrong.
        variance = _null_sharpe_variance(values)

    benchmark = expected_max_sharpe(n_trials, sharpe_variance=variance)
    dsr = probabilistic_sharpe_ratio(values, benchmark_sr=benchmark) if np.isfinite(sr) else np.nan
    psr0 = probabilistic_sharpe_ratio(values, benchmark_sr=0.0) if np.isfinite(sr) else np.nan

    return DeflatedSharpeResult(
        sharpe=sr,
        sharpe_annualised=sr * np.sqrt(periods_per_year) if np.isfinite(sr) else np.nan,
        n_observations=n,
        n_trials=int(n_trials),
        trial_sharpe_variance=variance,
        expected_max_sharpe=benchmark,
        deflated_sharpe_ratio=dsr,
        probabilistic_sharpe_ratio=psr0,
        skewness=float(sps.skew(values, bias=False)),
        kurtosis=float(sps.kurtosis(values, fisher=True, bias=False)),
        significant=bool(np.isfinite(dsr) and dsr >= threshold),
    )


def minimum_track_record_length(
    returns,
    *,
    benchmark_sr: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Observations needed for the Sharpe to be significant at ``confidence``.

    Bailey & López de Prado (2012). Answers the practical question "how much
    more data would I need before this result could mean anything?" — and
    very often the answer is decades, which settles the matter.
    """
    values = _clean(returns)
    if values.size < 8:
        raise NotEnoughDataError("need at least 8 observations")
    sr = sharpe_per_period(values)
    if not np.isfinite(sr) or sr <= benchmark_sr:
        return np.inf
    skew = float(sps.skew(values, bias=False))
    kurt = float(sps.kurtosis(values, fisher=False, bias=False))
    z = float(sps.norm.ppf(confidence))
    numerator = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    return float(1.0 + numerator * (z / (sr - benchmark_sr)) ** 2)
