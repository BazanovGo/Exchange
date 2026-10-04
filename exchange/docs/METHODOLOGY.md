# Methodology

Every statistic in this package exists to answer one question: *could this
result have arisen without a real effect?* This document states what each one
assumes, what it cannot do, and where the implementation deliberately differs
from the textbook.

---

## 1. Sequence

```
data -> quality gate -> hypothesis (pre-registered, hashed)
     -> structural scan -> in-sample search -> parameter stability
     -> out-of-sample (evaluated once) -> purged / combinatorial CV
     -> walk-forward -> block bootstrap / Monte Carlo
     -> deflation and multiple-testing correction
     -> execution stress ladder -> regime analysis
     -> verdict -> falsification attacks -> registry
```

The order is not cosmetic. Selection happens before the out-of-sample window
is touched; deflation happens after the trial count is known; the attacks
happen last because they re-use the out-of-sample window many times, which is
valid for refutation and not for confirmation.

## 2. Annualisation

Annualisation is measured from the return series' own index
(`empirical_bars_per_year`), never assumed. A business-day series yields ≈261
bars per year, not 252 and certainly not 365; using 365 on daily bars
overstates an annualised Sharpe by about 18%, which is larger than most
effects worth acting on. Where a synthetic series has no real calendar, the
factor is passed explicitly.

## 3. Sharpe inference

* **Standard error.** Lo (2002) with the Mertens (2002) non-normality
  correction. The skewness term is sign-dependent: for a positive Sharpe,
  negative skew *increases* the standard error. Getting that sign wrong makes
  short-volatility strategies look more certain than they are, so the test
  suite fixes it with a left-skewed, positive-Sharpe fixture.
* **Autocorrelation.** Lo's inflation factor is available but off by default
  (`autocorrelation_lags=0`), because applying it blindly to a series with no
  dependence adds noise. Smoothed returns — any strategy holding through
  several bars — need it on.
* **PSR.** The probability that the true Sharpe exceeds a benchmark, given the
  observed sample's skew and kurtosis. Calibrated in the suite over 200 seeds
  rather than asserted on one.

## 4. Deflated Sharpe ratio

`DSR = PSR(expected maximum Sharpe under the null)`, with the expected maximum
from the Bailey–López de Prado Gumbel approximation

```
E[max] = sqrt(V) * [ (1 - g) Z(1 - 1/N) + g Z(1 - 1/(N e)) ]
```

where `g` is Euler–Mascheroni, `N` the number of trials and `V` the variance
of the trials' Sharpe estimates.

Two implementation choices matter:

* **`V` is not defaulted to 1.0.** The literature's `V = 1` is meaningful for
  *annualised* Sharpes. Applied to per-period returns it sets a benchmark
  hundreds of times too high and rejects a true Sharpe of 6. The default here
  is the null sampling variance of the Sharpe estimate, `(1 + SR²/2)/n`, which
  is the floor any real grid exceeds; when the sweep's own Sharpes are
  available the funnel passes them, which is strictly better.
* **`N` is charged from the registry.** The trial count is the current grid
  *plus every configuration already tested on the same dataset hash*. The
  tenth hypothesis on a series carries the burden of the previous nine whether
  or not anyone wants it to.

**What it cannot do.** A parameter grid is not `N` independent tests;
neighbouring cells are nearly the same strategy. Using the observed dispersion
of the trials partially compensates. The result remains an approximation that
errs conservative, which is the correct direction here.

## 5. Probability of backtest overfitting

CSCV (Bailey, Borwein, López de Prado, Zhu): the sample is cut into an even
number of **contiguous** blocks, every half-sized combination in turn becomes
the in-sample set, the in-sample winner's rank among the out-of-sample scores
is logit-transformed, and PBO is the share of splits where that logit is
negative.

Blocks are contiguous, not random, because shuffling destroys the serial
dependence that makes a financial split informative.

**Reading it.** `PBO <= 0.2` the selection procedure is informative;
`<= 0.5` weakly informative; above that, worse than a coin flip, and no
configuration drawn from the search can be trusted. On an undifferentiated
family PBO averages ≈0.5 with a standard deviation around 0.2 on a thousand
observations, which is why the test suite measures it over several seeds.

## 6. Purging, embargo and CPCV

A label whose outcome spans `h` bars contaminates any training bar within `h`
of the test window; purging removes those bars, and the embargo removes a
further block after the test window to account for serial dependence in the
residuals.

CPCV cuts the sample into `n` groups, tests every combination of `k`, and
recombines them into `C(n,k)·k/n` full-sample out-of-sample paths. A single
walk-forward gives one path and therefore one number with no dispersion; CPCV
gives a distribution, and **the dispersion, not the median, is what a
robustness claim rests on**.

## 7. Bootstrap

Stationary bootstrap (Politis–Romano) by default, with the block length from
the Politis–White automatic rule; moving-block (Künsch) available. Both
preserve serial dependence, so the resulting interval is not the i.i.d. one
that systematically overstates significance for autocorrelated returns.

Monte Carlo paths are resampled returns, not simulated prices: the drawdown
distribution it produces is conditional on the observed return distribution,
and it says nothing about a regime the sample does not contain.

## 8. Multiple testing

* **FWER:** Bonferroni and Holm. Holm dominates Bonferroni and is the default
  for "which of these survive".
* **FDR:** Benjamini–Hochberg, and Benjamini–Yekutieli where the tests are
  dependent (a parameter grid always is).
* **Storey's `pi0`** estimates the share of true nulls, and
  `expected_false_discoveries` reports `alpha * n` plainly — the number of
  rejections the search would produce from noise alone.

## 9. Data-snooping tests

White's Reality Check and Hansen's SPA, both on the stationary bootstrap with
a shared index draw per resample so the cross-sectional correlation between
strategies is preserved. SPA is studentised and recentres only the strategies
that are not badly dominated, which is why it is more powerful than the
Reality Check; its lower and upper p-values bracket the consistent one.

## 10. Structural diagnostics

* **ADF** with MacKinnon critical values (−3.43 / −2.86 / −2.57), BIC lag
  selection. **The mean-reversion half-life is gated on it**: regressing a
  change on a level under a unit root produces a confident negative slope and
  therefore a plausible half-life for *any* random walk. Gating removed the
  scanner's false positives.
* **Variance ratio** (Lo–MacKinlay), heteroskedasticity-robust by default,
  because the homoskedastic version rejects the random walk far too often on
  real data.
* **ARCH-LM** (Engle) for volatility clustering, Jarque–Bera and tail ratios
  for tails, Kruskal–Wallis for seasonality, Levene plus autocorrelation sign
  flips for structural stability, and R/S Hurst **compared against the
  distribution of same-length random walks** rather than against 0.5, since
  the raw estimator is biased upward on short samples.
* **Evidence thresholds are 0.001 / 0.01 / 0.05**, an order of magnitude
  stricter than convention, because the scanner runs ~10 tests per series over
  many instruments. Measured on twenty independent random walks, this ladder
  produces no strong exploitable finding.

## 11. Execution

Costs are a stack, each component separately scalable so the attacks can
shock the estimated parts without touching the known ones: commission
(per-contract, per-trade, bps of notional, exchange fee, minimum), half-spread
per side, slippage (constant plus volatility-scaled), square-root market
impact with a participation cap and partial fills.

Fills are on a later bar by construction. `fill_timing="same_close"` exists
only so that a run which uses it can be flagged `admits_lookahead` and its
result marked invalid.

Four scenarios — optimistic, base, conservative, stressed — are always
reported as a ladder, and **performance must fall monotonically across it**.
It usually does; when it does not, that is a sign error in the cost
accounting, and the verdict carries the caveat.

## 12. Verdicts

Five levels, reached through conjunctive gates: the author's pre-registered
falsification criteria first, then sanity, then ten gates covering
out-of-sample edge, deflation, PBO, CPCV, walk-forward, bootstrap interval,
conservative costs, trade count, parameter plateau and regime breadth.

No single metric can raise a verdict past a failed gate. This is the opposite
of a weighted score, and the reason is in the audit: a weighted average lets
one spectacular number pay for a fatal one.

## 13. Falsification

Eleven attacks, each aimed at a specific artefact. The robustness score is an
unweighted count of survivals and **can only lower a verdict**, for two
reasons: a weighted score would reintroduce the defect above, and surviving
thirty looks at the same out-of-sample window is not thirty independent
confirmations. No attack re-selects parameters.

## 14. Point-in-time data

Two timestamps, never one: `observation_time` is the period a number
describes, `available_time` the earliest moment it could have been acted on.
Alignment to bars is a backward `merge_asof` on `available_time` — the only
join that cannot see forward. The default revision policy is **first print**,
because a revised value did not exist at the timestamp it carries.

## 15. What this methodology cannot do

* It cannot make a short sample long. Five years of daily data contains one or
  two commodity regimes, and no statistic recovers what is not there.
* It cannot price a cost model's error. Every result is conditional on the
  assumptions in the stress ladder.
* It cannot detect a lookahead that exists in the *data vendor's* history
  rather than in this code — a restated series delivered as if it were the
  original print is indistinguishable from a real edge.
* It cannot distinguish a genuine effect from one the market has since
  removed. Only forward time does that, which is why the highest verdict is
  "paper trading candidate" and not "deploy".
