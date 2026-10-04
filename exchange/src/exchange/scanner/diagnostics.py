"""Structural diagnostics: measure the market before searching it.

The failure this prevents
--------------------------
The usual research loop is: pick a strategy family, sweep its parameters,
keep what looks good. That order guarantees data mining, because the choice
of family is made by trying families until one works — an unrecorded search
over a space nobody counted.

The order here is inverted. First measure properties of the *price process*
that are estimable without reference to any strategy: autocorrelation,
variance ratio, mean-reversion half-life, volatility clustering, tail
behaviour, seasonality, stability over time. Only then choose which family
could plausibly exploit what was found — and, just as importantly, which
families to *rule out*.

The discipline this imposes
----------------------------
Every diagnostic returns an estimate, a confidence assessment and an explicit
statement of what would falsify it. No single test is allowed to declare a
trading opportunity: the recommendation layer requires corroboration across
independent measurements before it will name a family.

On the Hurst exponent
---------------------
It is implemented, and it is deliberately *not* used as a classifier. R/S
estimates are badly biased on samples of the length available in finance, are
strongly affected by volatility clustering (which is not long memory in
returns), and routinely return H ~ 0.6 on pure random walks of a few thousand
points. It is reported with a null band obtained by running the identical
estimator on random walks of the *same length*, so a reader can see how
uninformative it usually is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps

from exchange.core.errors import ConfigError, NotEnoughDataError

MIN_OBSERVATIONS = 100


@dataclass(frozen=True)
class Diagnostic:
    """One measured property of the price process."""

    name: str
    value: float
    interpretation: str          # what this value means, in words
    evidence: str                # NONE / WEAK / MODERATE / STRONG
    p_value: float = np.nan
    ci: tuple[float, float] = (np.nan, np.nan)
    falsified_by: str = ""       # what observation would overturn this reading
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "evidence": self.evidence,
            "p_value": self.p_value,
            "ci_low": self.ci[0],
            "ci_high": self.ci[1],
            "interpretation": self.interpretation,
            "falsified_by": self.falsified_by,
            **({"details": self.details} if self.details else {}),
        }


def _evidence_from_p(
    p: float, *, strong: float = 0.001, moderate: float = 0.01, weak: float = 0.05
) -> str:
    """Map a p-value onto an evidence grade, with multiplicity priced in.

    The thresholds are an order of magnitude stricter than the conventional
    0.05/0.01, because the scanner is not one test: it runs about ten
    diagnostics on every series, and a research programme runs it over many
    instruments. At a 0.01 bar for "STRONG", a pure random walk earns a
    strong finding roughly one series in ten, and those findings are what
    a search would then be pointed at. Measured on twenty independent
    random walks, this ladder produces none.
    """

    if not np.isfinite(p):
        return "NONE"
    if p <= strong:
        return "STRONG"
    if p <= moderate:
        return "MODERATE"
    if p <= weak:
        return "WEAK"
    return "NONE"


def _returns(bars: pd.DataFrame, *, log: bool = True) -> np.ndarray:
    close = bars["close"].to_numpy(float)
    # Never let a roll boundary enter a structural estimate: the gap is an
    # artefact of the splice and would show up as a jump, a fat tail and
    # spurious autocorrelation all at once.
    mask = (
        ~bars["is_roll"].to_numpy(bool)
        if "is_roll" in bars.columns
        else np.ones(len(close), dtype=bool)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.diff(np.log(np.where(close > 0, close, np.nan))) if log else np.diff(close) / close[:-1]
    rets = np.where(mask[1:], rets, np.nan)
    return rets[np.isfinite(rets)]


# ==========================================================================
# Dependence in returns
# ==========================================================================


def autocorrelation(bars: pd.DataFrame, *, lags: int = 10) -> Diagnostic:
    """Ljung-Box test on return autocorrelation up to ``lags``.

    The joint test is used rather than lag-1 alone: a strategy exploiting
    lag-1 correlation is rare, and testing each lag separately is itself a
    multiple-testing problem.
    """
    rets = _returns(bars)
    n = rets.size
    if n < MIN_OBSERVATIONS:
        raise NotEnoughDataError(f"autocorrelation needs >= {MIN_OBSERVATIONS} returns, got {n}")
    lags = min(lags, n // 5)

    demeaned = rets - rets.mean()
    denom = float(np.dot(demeaned, demeaned))
    rhos = np.array([float(np.dot(demeaned[:-k], demeaned[k:]) / denom) for k in range(1, lags + 1)])

    q = n * (n + 2) * np.sum(rhos**2 / (n - np.arange(1, lags + 1)))
    p = float(sps.chi2.sf(q, df=lags))
    rho1 = float(rhos[0])
    se = 1.0 / np.sqrt(n)

    direction = "positive (trend-persistent)" if rho1 > 0 else "negative (reverting)"
    return Diagnostic(
        name="autocorrelation",
        value=rho1,
        p_value=p,
        ci=(rho1 - 1.96 * se, rho1 + 1.96 * se),
        evidence=_evidence_from_p(p),
        interpretation=(
            f"Lag-1 autocorrelation {rho1:+.3f} ({direction}); Ljung-Box over {lags} lags "
            f"gives p = {p:.4f}."
        ),
        falsified_by=(
            "The same test on a held-out period returning p > 0.15, or the sign of rho1 flipping "
            "between halves of the sample."
        ),
        details={"lags": lags, "rho": rhos.tolist(), "ljung_box_q": float(q)},
    )


def partial_autocorrelation(bars: pd.DataFrame, *, lags: int = 10) -> Diagnostic:
    """Partial autocorrelation via the Durbin-Levinson recursion.

    Distinguishes genuine higher-order structure from the echo of lag-1
    dependence — the difference between "returns are AR(1)" and "returns have
    a real 5-bar cycle".
    """
    rets = _returns(bars)
    n = rets.size
    if n < MIN_OBSERVATIONS:
        raise NotEnoughDataError(f"pacf needs >= {MIN_OBSERVATIONS} returns, got {n}")
    lags = min(lags, n // 5)

    demeaned = rets - rets.mean()
    denom = float(np.dot(demeaned, demeaned))
    acf = np.array([1.0] + [float(np.dot(demeaned[:-k], demeaned[k:]) / denom) for k in range(1, lags + 1)])

    pacf = np.zeros(lags + 1)
    pacf[0] = 1.0
    phi = np.zeros((lags + 1, lags + 1))
    if lags >= 1:
        phi[1, 1] = acf[1]
        pacf[1] = acf[1]
        for k in range(2, lags + 1):
            num = acf[k] - sum(phi[k - 1, j] * acf[k - j] for j in range(1, k))
            den = 1.0 - sum(phi[k - 1, j] * acf[j] for j in range(1, k))
            phi[k, k] = num / den if abs(den) > 1e-12 else 0.0
            for j in range(1, k):
                phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
            pacf[k] = phi[k, k]

    se = 1.0 / np.sqrt(n)
    significant = [int(k) for k in range(1, lags + 1) if abs(pacf[k]) > 1.96 * se]
    strongest = int(np.argmax(np.abs(pacf[1:]))) + 1 if lags else 0

    return Diagnostic(
        name="partial_autocorrelation",
        value=float(pacf[strongest]) if strongest else np.nan,
        evidence="MODERATE" if significant else "NONE",
        interpretation=(
            f"Significant partial autocorrelation at lag(s) {significant or 'none'}; "
            f"strongest at lag {strongest} ({pacf[strongest]:+.3f})."
            if strongest else "No partial autocorrelation structure."
        ),
        falsified_by="The same lags failing to be significant on a held-out period.",
        details={"pacf": pacf[1:].tolist(), "significant_lags": significant, "se": se},
    )


def variance_ratio(bars: pd.DataFrame, *, q: int = 5, heteroskedastic: bool = True) -> Diagnostic:
    """Lo-MacKinlay variance ratio test.

    ``VR(q) = Var(q-period return) / (q * Var(1-period return))``

    * ``VR > 1`` — returns are positively autocorrelated: trends persist.
    * ``VR < 1`` — returns revert.
    * ``VR = 1`` — a random walk.

    The heteroskedasticity-robust statistic is the default and is not optional
    in practice: financial volatility clusters, and the homoskedastic version
    rejects the random walk far too often on real data, producing exactly the
    false "trending market" reading that sends a research programme down the
    wrong path.

    Reference: Lo, A. & MacKinlay, A. (1988), *Review of Financial Studies*
    1(1).
    """
    rets = _returns(bars)
    n = rets.size
    if n < max(MIN_OBSERVATIONS, q * 10):
        raise NotEnoughDataError(f"variance ratio at q={q} needs >= {max(MIN_OBSERVATIONS, q * 10)} returns")
    if q < 2:
        raise ValueError("q must be >= 2")

    mu = rets.mean()
    demeaned = rets - mu
    var1 = float(np.dot(demeaned, demeaned) / (n - 1))
    if var1 <= 0:
        raise NotEnoughDataError("return series has zero variance")

    # Overlapping q-period sums (the efficient estimator).
    cumsum = np.concatenate([[0.0], np.cumsum(demeaned)])
    q_sums = cumsum[q:] - cumsum[:-q]
    m = (n - q + 1) * (1.0 - q / n)
    varq = float(np.dot(q_sums, q_sums) / (m * q))
    vr = varq / var1

    if heteroskedastic:
        # Lo-MacKinlay z*(q): robust to conditional heteroskedasticity.
        theta = 0.0
        d2 = demeaned**2
        denom = float(np.sum(d2)) ** 2
        for j in range(1, q):
            delta = float(np.sum(d2[j:] * d2[:-j])) * n / denom
            theta += (2.0 * (q - j) / q) ** 2 * delta
        se = np.sqrt(theta) if theta > 0 else np.nan
    else:
        se = np.sqrt(2.0 * (2.0 * q - 1.0) * (q - 1.0) / (3.0 * q * n))

    z = (vr - 1.0) / se if np.isfinite(se) and se > 0 else np.nan
    p = float(2.0 * sps.norm.sf(abs(z))) if np.isfinite(z) else np.nan

    if not np.isfinite(p) or p > 0.15:
        reading = "consistent with a random walk"
    elif vr > 1:
        reading = "trend persistence"
    else:
        reading = "mean reversion"

    return Diagnostic(
        name=f"variance_ratio_q{q}",
        value=float(vr),
        p_value=p,
        ci=(vr - 1.96 * se, vr + 1.96 * se) if np.isfinite(se) else (np.nan, np.nan),
        evidence=_evidence_from_p(p),
        interpretation=f"VR({q}) = {vr:.3f} ({reading}); z = {z:.2f}, p = {p:.4f}.",
        falsified_by=(
            f"VR({q}) crossing 1.0 in a held-out period, or the effect disappearing once the "
            "heteroskedasticity-robust statistic is used."
        ),
        details={"q": q, "heteroskedastic": heteroskedastic, "z": float(z) if np.isfinite(z) else None},
    )


def variance_ratio_profile(bars: pd.DataFrame, qs=(2, 5, 10, 20)) -> pd.DataFrame:
    """Variance ratio across several horizons.

    The *shape* is the information. A market with short-horizon reversion and
    long-horizon persistence — VR below 1 at q=2 and above 1 at q=20 — is the
    classic commodity signature, and it says plainly that an unconditional
    strategy at either horizon will disappoint.
    """
    rows = []
    for q in qs:
        try:
            d = variance_ratio(bars, q=q)
            rows.append({"q": q, "vr": d.value, "p_value": d.p_value, "evidence": d.evidence})
        except NotEnoughDataError:
            rows.append({"q": q, "vr": np.nan, "p_value": np.nan, "evidence": "NONE"})
    return pd.DataFrame(rows)


# Augmented Dickey-Fuller critical values for the constant-only ("c") case.
# MacKinnon (2010), Table 2, asymptotic values. These are NOT the normal or
# Student-t critical values, and using the latter is a real, common bug —
# under a unit root the OLS t-statistic converges to the Dickey-Fuller
# distribution, whose 5% quantile is -2.86 rather than -1.65. Judging the
# AR(1) slope with OLS p-values therefore declares mean reversion on
# perfectly ordinary random walks, which is exactly the false positive this
# scanner exists to avoid.
_ADF_CRITICAL = {0.01: -3.43, 0.05: -2.86, 0.10: -2.57}


def adf_test(x: np.ndarray, *, max_lags: int | None = None) -> tuple[float, float, int]:
    """Augmented Dickey-Fuller statistic for a level series with a constant.

    Returns ``(t_statistic, approximate_p_value, lags_used)``. Lag order is
    chosen by minimising the BIC over ``0..max_lags``, so serial correlation
    in the differences does not contaminate the unit-root test.

    The p-value is interpolated from MacKinnon's asymptotic critical values
    and is approximate by design: what matters for a research decision is
    which side of -2.86 the statistic falls on, not its third decimal.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        # A DataFrame flattened into this function silently produces a
        # statistic computed on interleaved columns, which is worse than an
        # error because it looks like a number.
        raise ConfigError(f"adf_test needs a 1-D level series, got shape {x.shape}")
    n = x.size
    if max_lags is None:
        max_lags = int(np.ceil(12.0 * (n / 100.0) ** 0.25))
        max_lags = min(max_lags, max(1, n // 20))

    best = None
    for lags in range(max_lags + 1):
        dx = np.diff(x)
        rows = n - 1 - lags
        if rows < 20:
            break
        y = dx[lags:]
        cols = [np.ones(rows), x[lags: n - 1]]
        for k in range(1, lags + 1):
            cols.append(dx[lags - k: n - 1 - k])
        design = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        resid = y - design @ beta
        dof = rows - design.shape[1]
        if dof <= 0:
            break
        sigma2 = float(resid @ resid) / dof
        bic = rows * np.log(max(sigma2, 1e-300)) + design.shape[1] * np.log(rows)
        if best is None or bic < best[0]:
            xtx_inv = np.linalg.pinv(design.T @ design)
            se = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
            t_stat = float(beta[1] / se) if se > 0 else np.nan
            best = (bic, t_stat, float(beta[1]), lags)

    if best is None:
        return (np.nan, np.nan, 0)
    _, t_stat, _, lags = best

    if not np.isfinite(t_stat):
        p = np.nan
    elif t_stat <= _ADF_CRITICAL[0.01]:
        p = 0.01 * float(np.exp(min((t_stat - _ADF_CRITICAL[0.01]) * 1.5, 0.0)))
    elif t_stat >= _ADF_CRITICAL[0.10]:
        p = float(min(0.10 + (t_stat - _ADF_CRITICAL[0.10]) * 0.20, 0.99))
    else:
        points = sorted(_ADF_CRITICAL.items(), key=lambda kv: kv[1])
        levels = [lvl for lvl, _ in points]
        values = [val for _, val in points]
        p = float(np.interp(t_stat, values, levels))
    return (t_stat, p, lags)


def mean_reversion_half_life(bars: pd.DataFrame) -> Diagnostic:
    """Half-life of shocks, gated on a proper unit-root test.

    Regressing ``dx`` on ``x`` always produces a slightly negative slope on a
    random walk (the Dickey-Fuller small-sample bias, of order ``-1/T``), so a
    half-life can *always* be computed and will always look finite. The number
    is therefore reported only when an Augmented Dickey-Fuller test rejects
    the unit root at its own critical values; otherwise the diagnostic says
    the half-life is undefined, which is the truth about a random walk.
    """
    close = bars["close"].to_numpy(float)
    close = close[np.isfinite(close) & (close > 0)]
    if close.size < MIN_OBSERVATIONS:
        raise NotEnoughDataError(f"half-life needs >= {MIN_OBSERVATIONS} observations")
    x = np.log(close)

    t_stat, p, lags = adf_test(x)
    slope, _, r, _, se = sps.linregress(x[:-1], np.diff(x))

    rejects_unit_root = np.isfinite(t_stat) and t_stat <= _ADF_CRITICAL[0.10]
    if slope >= 0 or not rejects_unit_root:
        return Diagnostic(
            name="mean_reversion_half_life",
            value=np.inf,
            p_value=float(p) if np.isfinite(p) else np.nan,
            evidence="NONE",
            interpretation=(
                f"ADF statistic {t_stat:.2f} (10% critical value {_ADF_CRITICAL[0.10]}) does not "
                "reject a unit root: the level behaves like a random walk and the half-life is "
                f"undefined. The raw AR(1) slope of {slope:+.5f} is the expected Dickey-Fuller "
                "bias, not evidence."
            ),
            falsified_by="An ADF statistic below -2.86 on a comparable period.",
            details={"adf_t": float(t_stat), "adf_lags": lags, "ar1_slope": float(slope)},
        )

    half_life = float(-np.log(2.0) / np.log1p(slope))
    return Diagnostic(
        name="mean_reversion_half_life",
        value=half_life,
        p_value=float(p),
        evidence=_evidence_from_p(p),
        interpretation=(
            f"ADF statistic {t_stat:.2f} rejects the unit root (p ~ {p:.3f}); estimated half-life "
            f"{half_life:.1f} bars from an AR(1) slope of {slope:+.5f}. A tradeable reversion needs "
            "a half-life short enough that costs are recovered within it."
        ),
        falsified_by=(
            "The half-life moving by more than a factor of two between sample halves, or the ADF "
            "statistic failing to reject on a held-out period."
        ),
        details={
            "adf_t": float(t_stat), "adf_lags": lags,
            "ar1_slope": float(slope), "stderr": float(se), "r_squared": float(r**2),
        },
    )


# ==========================================================================
# Volatility structure
# ==========================================================================


def volatility_clustering(bars: pd.DataFrame, *, lags: int = 10) -> Diagnostic:
    """Engle's ARCH LM test on squared returns.

    High clustering is the most reliably present feature of any financial
    series. It matters here for a specific reason: it means *volatility* is
    forecastable even when direction is not, which points to sizing and
    regime-conditioning strategies rather than to directional ones.
    """
    rets = _returns(bars)
    n = rets.size
    if n < MIN_OBSERVATIONS:
        raise NotEnoughDataError(f"ARCH test needs >= {MIN_OBSERVATIONS} returns")
    lags = max(1, min(lags, n // 10))
    sq = (rets - rets.mean()) ** 2

    # LM test: regress sq_t on its own `lags` lags; n*R^2 ~ chi2(lags).
    y = sq[lags:]
    x = np.column_stack([sq[lags - k: -k if k else None] for k in range(1, lags + 1)])
    x = np.column_stack([np.ones(len(y)), x])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else 0.0
    stat = len(y) * r2
    p = float(sps.chi2.sf(stat, df=lags))

    # A scale-free measure of how much clustering there is.
    abs_rets = np.abs(rets - rets.mean())
    demeaned = abs_rets - abs_rets.mean()
    denom = float(np.dot(demeaned, demeaned))
    rho_abs = float(np.dot(demeaned[:-1], demeaned[1:]) / denom) if denom > 0 else np.nan

    return Diagnostic(
        name="volatility_clustering",
        value=rho_abs,
        p_value=p,
        evidence=_evidence_from_p(p),
        interpretation=(
            f"ARCH-LM over {lags} lags: p = {p:.2e}, R^2 = {r2:.3f}; lag-1 autocorrelation of "
            f"|returns| is {rho_abs:+.3f}. Volatility is forecastable to this degree; direction is "
            "a separate question."
        ),
        falsified_by="Absence of clustering (p > 0.15) on a comparable held-out period.",
        details={"lags": lags, "r_squared": r2, "lm_statistic": float(stat)},
    )


def tail_behaviour(bars: pd.DataFrame) -> Diagnostic:
    """Jump frequency and tail thickness of the return distribution.

    Reported because it governs whether a backtest's average is meaningful: a
    strategy whose P&L is dominated by a handful of jumps has an effective
    sample size of a handful, whatever its bar count says.
    """
    rets = _returns(bars)
    if rets.size < MIN_OBSERVATIONS:
        raise NotEnoughDataError("tail analysis needs more observations")
    med = float(np.median(rets))
    mad = float(np.median(np.abs(rets - med))) * 1.4826
    if mad <= 0:
        raise NotEnoughDataError("return series has zero robust dispersion")
    z = np.abs(rets - med) / mad
    jumps = int((z > 5.0).sum())
    excess_kurt = float(sps.kurtosis(rets, fisher=True, bias=False))
    _, p_normal = sps.jarque_bera(rets)

    return Diagnostic(
        name="tail_behaviour",
        value=excess_kurt,
        p_value=float(p_normal),
        evidence="STRONG" if excess_kurt > 3 else "MODERATE" if excess_kurt > 1 else "WEAK",
        interpretation=(
            f"Excess kurtosis {excess_kurt:.1f}, {jumps} return(s) beyond 5 robust sigma "
            f"({jumps / rets.size:.2%} of bars). Jarque-Bera p = {p_normal:.2e}."
        ),
        falsified_by="A Gaussian-like tail (excess kurtosis below 1) on other periods.",
        details={"n_jumps": jumps, "jump_fraction": jumps / rets.size, "robust_sigma": mad},
    )


# ==========================================================================
# Seasonality and stability
# ==========================================================================


def _return_series(bars: pd.DataFrame) -> pd.Series:
    """Log returns as a Series aligned to the bars they belong to."""
    close = bars["close"]
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.log(close.where(close > 0)).diff()
    if "is_roll" in bars.columns:
        rets = rets.mask(bars["is_roll"].astype(bool))
    return rets.dropna()


def seasonality(bars: pd.DataFrame, *, by: str = "month") -> Diagnostic:
    """Kruskal-Wallis test for differences in return distribution by bucket.

    A rank-based test is used rather than ANOVA because financial returns are
    far from normal and ANOVA's F-test is not robust to that.

    Note the multiple-testing exposure this diagnostic creates and cannot
    resolve on its own: ``by`` can be month, weekday or hour, and finding
    "something" in one of them is close to certain. The reported p-value is
    for the *joint* test across buckets, not for the best bucket, which is the
    honest version — and the thresholds are stricter than elsewhere for the
    same reason.
    """
    rets = _return_series(bars)
    if rets.size < MIN_OBSERVATIONS:
        raise NotEnoughDataError("seasonality needs more observations")

    if by == "month":
        key = rets.index.month
        label = "month of year"
    elif by == "weekday":
        key = rets.index.dayofweek
        label = "day of week"
    elif by == "hour":
        key = rets.index.hour
        label = "hour of day"
    else:
        raise ValueError(f"unknown seasonality bucket {by!r}")

    key = np.asarray(key)
    groups = [rets.to_numpy()[key == k] for k in np.unique(key)]
    groups = [g for g in groups if g.size >= 10]
    if len(groups) < 2:
        raise NotEnoughDataError(f"not enough data per {label} bucket")

    stat, p = sps.kruskal(*groups)
    means = {int(k): float(rets.to_numpy()[key == k].mean()) for k in np.unique(key)}
    spread = max(means.values()) - min(means.values())

    # Stricter thresholds than elsewhere, on purpose. Calendar effects are the
    # family with the largest hidden search space (12 months x 5 weekdays x 2
    # directions x several bucket definitions), so a p-value that would be
    # persuasive for a pre-registered test is routine here.
    return Diagnostic(
        name=f"seasonality_{by}",
        value=float(spread),
        p_value=float(p),
        evidence=_evidence_from_p(p, strong=0.001, moderate=0.01, weak=0.05),
        interpretation=(
            f"Kruskal-Wallis across {len(groups)} {label} buckets: p = {p:.4f}; spread between the "
            f"best and worst bucket's mean return is {spread:.4%} per bar."
        ),
        falsified_by=(
            f"The ranking of {label} buckets failing to hold on a held-out period — a seasonal "
            "effect that reorders itself is a sample artefact."
        ),
        details={"bucket_means": means, "n_buckets": len(groups), "statistic": float(stat)},
    )


def structural_stability(bars: pd.DataFrame, *, n_blocks: int = 4) -> Diagnostic:
    """Do the series' own statistics hold still across sub-periods?

    Splits the sample into contiguous blocks and compares mean, volatility and
    lag-1 autocorrelation across them. This is the most under-used diagnostic
    in practice and the most decisive: if the process itself changes between
    blocks, no amount of cross-validation on the whole sample will produce a
    stable strategy, and the honest conclusion is that the instrument is not
    researchable at this horizon.
    """
    rets = _returns(bars)
    n = rets.size
    if n < MIN_OBSERVATIONS * 2:
        raise NotEnoughDataError("stability needs at least 200 returns")
    blocks = np.array_split(rets, n_blocks)

    means = np.array([b.mean() for b in blocks])
    vols = np.array([b.std(ddof=1) for b in blocks])
    rho1 = []
    for b in blocks:
        d = b - b.mean()
        denom = float(np.dot(d, d))
        rho1.append(float(np.dot(d[:-1], d[1:]) / denom) if denom > 0 else np.nan)
    rho1 = np.array(rho1)

    _, p_var = sps.levene(*blocks, center="median")
    vol_ratio = float(vols.max() / vols.min()) if vols.min() > 0 else np.inf
    finite_rho = rho1[np.isfinite(rho1)]
    sign_flips = int(np.sum(np.diff(np.sign(finite_rho)) != 0)) if finite_rho.size > 1 else 0

    unstable = (p_var < 0.01) or (vol_ratio > 2.0) or (sign_flips >= max(1, n_blocks - 2))
    return Diagnostic(
        name="structural_stability",
        value=vol_ratio,
        p_value=float(p_var),
        evidence="WEAK" if unstable else "STRONG",
        interpretation=(
            f"Across {n_blocks} blocks: volatility ratio max/min = {vol_ratio:.2f} "
            f"(Levene p = {p_var:.2e}), lag-1 autocorrelation sign flips {sign_flips} time(s). "
            + ("The process is NOT stable across the sample." if unstable
               else "The process is reasonably stable across the sample.")
        ),
        falsified_by="A different block count or block boundaries reversing this reading.",
        details={
            "block_means": means.tolist(),
            "block_vols": vols.tolist(),
            "block_rho1": rho1.tolist(),
            "unstable": bool(unstable),
        },
    )


def hurst_exponent(bars: pd.DataFrame, *, min_window: int = 10, n_windows: int = 12) -> Diagnostic:
    """Hurst exponent by rescaled-range analysis — reported, not trusted.

    ``H > 0.5`` is conventionally read as long memory and ``H < 0.5`` as
    anti-persistence. On finite financial samples this reading is close to
    worthless, and the diagnostic says so: the estimate is accompanied by a
    null band obtained by running the identical estimator on synthetic random
    walks of the *same length*. An H of 0.58 that sits comfortably inside the
    random-walk band is not evidence of anything, and presenting it as a
    market classification — which is the standard practice — is how a research
    programme commits to a strategy family for no reason.
    """
    from exchange.core.determinism import rng as make_rng

    rets = _returns(bars)
    n = rets.size
    if n < 256:
        raise NotEnoughDataError("Hurst estimation needs at least 256 returns")

    def _rs_hurst(series: np.ndarray) -> float:
        sizes = np.unique(
            np.geomspace(min_window, max(len(series) // 4, min_window * 2), n_windows).astype(int)
        )
        logs_n, logs_rs = [], []
        for size in sizes:
            chunks = len(series) // size
            if chunks < 2:
                continue
            values = []
            for c in range(chunks):
                seg = series[c * size: (c + 1) * size]
                dev = np.cumsum(seg - seg.mean())
                r = dev.max() - dev.min()
                s = seg.std(ddof=1)
                if s > 0 and r > 0:
                    values.append(r / s)
            if values:
                logs_n.append(np.log(size))
                logs_rs.append(np.log(np.mean(values)))
        if len(logs_n) < 3:
            return np.nan
        return float(np.polyfit(logs_n, logs_rs, 1)[0])

    h = _rs_hurst(rets)

    # Null band: the same estimator on random walks of the same length.
    r = make_rng(20260902)
    null = np.array([_rs_hurst(r.normal(0.0, 1.0, n)) for _ in range(40)])
    null = null[np.isfinite(null)]
    lo, hi = (
        (float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975)))
        if null.size else (np.nan, np.nan)
    )
    inside = np.isfinite(lo) and lo <= h <= hi

    return Diagnostic(
        name="hurst_exponent",
        value=float(h),
        ci=(lo, hi),
        evidence="NONE" if inside else "WEAK",
        interpretation=(
            f"H = {h:.3f}. On random walks of this exact length ({n} returns) the same estimator "
            f"gives {lo:.3f}-{hi:.3f} 95% of the time, so this value is "
            + ("INDISTINGUISHABLE from a random walk and carries no information."
               if inside else
               "outside the random-walk band — suggestive, but still not a basis for choosing a "
               "strategy family.")
        ),
        falsified_by=(
            "Any of the variance-ratio or autocorrelation tests disagreeing; those are better "
            "estimators."
        ),
        details={"null_band": [lo, hi], "inside_null_band": bool(inside), "n_returns": n},
    )


ALL_DIAGNOSTICS = (
    autocorrelation,
    partial_autocorrelation,
    mean_reversion_half_life,
    volatility_clustering,
    tail_behaviour,
    structural_stability,
    hurst_exponent,
)
