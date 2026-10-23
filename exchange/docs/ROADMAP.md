# Roadmap

What exists, what is deliberately absent, and what comes next. The ordering
principle throughout: **the main threat is a false discovery**, so work that
reduces it precedes work that adds surface area.

---

## Shipped

| Area | State |
|---|---|
| Reproducibility core | Determinism, canonical hashing of configs and datasets, environment capture and diff, typed error hierarchy |
| Data engine | UTC canonical schema, severity-graded quality gate, point-in-time series with publication lags and revision policies, two independent lookahead guards |
| Real data | MOEX AlgoPack reader for NG and BR futures (1-, 5- and 60-minute candles, Moscow time): vendor splice recovered into rolls, ratio-adjusted and tradeable continuous series; both studies run on it |
| Futures | Contract symbol grammar and specifications for 8 roots, four roll rules, three adjustment methods, roll provenance, roll-gap diagnostics in basis points |
| Execution | Commission / spread / slippage / square-root impact with a participation cap and partial fills; market, limit, stop and stop-limit matching with gap-aware fills; four scenarios as a ladder |
| Backtest engine | Next-bar execution by construction, average-cost accounting, reversal split into close and open, trade reconciliation |
| Statistics | Lo/Mertens standard errors, PSR, DSR, expected maximum Sharpe, minimum track record length, PBO via CSCV, purged K-fold, CPCV, walk-forward, stationary and moving-block bootstrap, FWER and FDR, Reality Check, SPA |
| Scanner | ADF-gated half-life, variance ratio and profile, ARCH-LM, tails, seasonality, structural stability, Hurst against a null band; evidence thresholds calibrated against twenty random walks |
| Strategies | Typed interface with enforced causality, declared grids and required economic rationale; 11 families |
| Funnel | 11 stages, structurally separated selection, conjunctive verdict gates |
| Falsification | 11 attacks, auditable robustness score that can only lower a verdict |
| Registry | Append-only records with every reproduction input, EXACT / EQUIVALENT / DIVERGENT / INCOMPARABLE comparison, multiple-testing ledger, rejection rate |
| Features | 35 fundamental series for natural gas and crude with conservative publication lags, vendor-free provider interface, row-wise term-structure construction |
| Reporting | 23-section standard report with a generated "what could make this result false?" section |
| CLI | doctor, data validate/import/continuous, scan, strategies, features, backtest, research, falsify, reproduce, experiments, report, controls |
| Tests and CI | 521 tests, 83% coverage; temporal integrity runs first and alone, then tests on three Python versions, then controls, reproducibility and the Docker build |

## Deliberately absent

* **The autonomous research loop.** Designed in `AI_RESEARCHER.md`; blocked on
  the sandbox, allowlists, budgets and append-only enforcement described
  there.
* **A dashboard.** A visual layer on an unstable core teaches people to trust
  numbers nobody has validated. It comes after the first real-data result
  survives the attacks.
* **Hundreds of indicators.** Every indicator is a free parameter in a search
  nobody is counting. The 11 families exist to test specific mechanisms.
* **Machine learning on prices.** A model with thousands of parameters on a
  sample containing one or two regimes is a false-discovery machine. It
  becomes defensible only after the purged-CV and deflation machinery here is
  routinely used, and even then on features with mechanisms.
* **Genetic optimisation of Sharpe.** Maximising the statistic the framework
  exists to distrust.
* **Live trading.** No automated path from a verdict to an order.

## Next, in order

1. **Real spreads.** Both studies ran on MOEX AlgoPack candles; what decides
   whether an intraday effect is tradable is the cost of trading it. Bid/ask
   quotes, not candle opens, are the next data to add.
2. **A real contract calendar.** Specifications currently derive estimated
   dates when none are supplied, and flag them. Exact first-notice and
   last-trade dates per contract move every roll and therefore every result.
3. **Feature providers for the catalogued series.** The specifications and lags
   exist; an implementation against a real EIA/CFTC/NOAA archive is the
   remaining step, and it must preserve first prints rather than restatements.
4. **Cross-sectional research.** Everything here is single-instrument. The
   statistics are ready (SPA already handles families); the dataset layer needs
   an aligned panel abstraction.
5. **The sandbox, then the loop.** In that order.
6. **Paper trading harness.** Once something reaches
   `PAPER_TRADING_CANDIDATE`, the next question is whether the cost model was
   right, which only live quotes answer.

## Known limitations of the current implementation

* Derived contract specifications estimate expiry dates and are flagged as
  such; they must not be used for a result anyone acts on.
* `roll_methodology` is the only attack that cannot run without caller-supplied
  alternatives, so it reads "not evaluated" on any dataset that is not a
  spliced futures panel. That is reported, never counted as a pass.
* The Yahoo provider is unsuitable for futures and documented as such; it
  exists for equity-index sanity checks.
* Coverage is 83%, and the uncovered remainder is mostly rendering and error
  paths. Chasing 100% would add tests that assert formatting rather than
  behaviour.
