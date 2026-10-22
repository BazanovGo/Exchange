# Exchange Research OS

A falsification-first quantitative research framework for futures and
commodities.

**All research in this repository runs on Moscow Exchange futures quotes
obtained through MOEX AlgoPack** — natural gas (NG) and Brent (BR), 1-, 5- and
60-minute candles over their full history (NG from 2020-02, BR from 2011-12,
both to 2026-10). Synthetic series with a known answer are used only to
calibrate the framework itself: its controls and tests.

> **The objective is not to find strategies that backtest well. The objective
> is to find effects that survive attempts to falsify them.**

Every design decision here was made against one question: *does this reduce
the probability of mistaking randomness, overfitting or a data error for a
real trading pattern?* Where the answer was no, the feature is not here — see
[`docs/ROADMAP.md`](docs/ROADMAP.md) for what is deliberately absent and why.

---

## Why this exists

The predecessor framework in this repository was audited line by line
([`docs/ARCHITECTURE_AUDIT.md`](docs/ARCHITECTURE_AUDIT.md)). Its engineering
was sound in places that matter — the stop simulator is more careful than most
commercial backtesters — but one defect invalidated the out-of-sample numbers
behind every verdict it produced: parameters were selected using a score that
*included the out-of-sample window*, and that window's Sharpe was then
reported as an out-of-sample result.

That error is instructive because of where it lived: inside a measure
explicitly designed to prevent overfitting. Good intentions and careful review
did not catch it, and no downstream test could, because the contamination
happened before the tests ran.

So in this framework the honest behaviour is **structural**, not
conventional:

| The mistake | Why it cannot happen here |
|---|---|
| Out-of-sample data leaking into selection | The search function only ever receives the in-sample slice, and its result table has no out-of-sample column at all. |
| Reporting the best of a large search as a single test | The trial count is charged automatically from the registry's ledger and fed to the Deflated Sharpe Ratio. The tenth hypothesis on a dataset inherits the first nine's burden. |
| One great metric offsetting a fatal one | Verdict gates are conjunctive. A Sharpe of 3.0 with a PBO of 0.7 is REJECTED. |
| Filling at the close that produced the signal | The engine executes on a later bar by construction. Same-bar execution exists only so it can be flagged as admitting lookahead and the result marked invalid. |
| A non-causal indicator | `assert_causal` proves causality on real data for every registered strategy, in CI. |
| A pre-contaminated feature column | A separate leakage scan compares each column's correlation with the next bar against the current one. No causality proof can catch this; the two guards are distinct. |
| Alpha manufactured by a futures roll | Roll rules, adjustment methods and roll-gap diagnostics are first-class, and one of the eleven attacks rebuilds the series a different way to see whether the effect survives. |
| An unreproducible result | Every experiment records the dataset content hash, the instrument specification, the execution assumptions, the seed and the environment. `exchange reproduce` re-runs it and compares. |

---

## Quickstart

```bash
cd exchange
uv sync --all-extras

exchange doctor                   # can this installation produce a reproducible result?
exchange controls --quick         # is the framework calibrated? (synthetic positive/negative/leakage)
exchange scan synthetic:mean_reverting            # does this series have exploitable structure at all?
exchange research synthetic:mean_reverting \
    --hypothesis examples/mean_reversion.yaml     # push one pre-registered hypothesis through the funnel
exchange experiments list                         # the registry, with the multiple-testing ledger
exchange reproduce EXP-20260902-000000            # re-run it and compare against the record
```

`exchange --help` lists every command. All of them take `--config`, print
their reasoning rather than only their result, and exit non-zero when
something is wrong.

## The funnel

```
data -> quality gate -> hypothesis (pre-registered, hashed)
     -> structural scan -> in-sample search -> parameter stability
     -> out-of-sample (evaluated once) -> purged / combinatorial CV
     -> walk-forward -> block bootstrap / Monte Carlo
     -> deflation and multiple-testing correction
     -> execution stress ladder -> regime analysis
     -> verdict -> falsification attacks -> registry
```

Verdicts are `REJECTED`, `WEAK`, `PROMISING`, `ROBUST`,
`PAPER_TRADING_CANDIDATE`. **A rejection is a completed experiment**, recorded
with the same weight as a success: it removes a candidate from the search
space permanently and makes the next hypothesis' statistics honest. The
registry's rejection rate is the clearest indicator that the funnel is doing
its job — a programme that rejects almost nothing has either found something
remarkable or disabled its own gates.

The highest verdict is *paper trading candidate*. There is no automated path
from a verdict to an order, and there should not be.

## What is in the box

* **Data engine** — UTC canonical schema; severity-graded quality gate
  (timestamps, duplicates, OHLC consistency, gaps, jumps, staleness);
  point-in-time series with publication lags and first-print revision
  policies; two independent lookahead guards.
* **MOEX AlgoPack** — `exchange.data.moex` reads AlgoPack candle files
  (Moscow time, vendor-spliced by nearest expiry), recovers every roll from
  the contract code and builds an auditable continuous series: ratio-adjusted
  for signals, tradeable for P&L. No network and no API key: it reads files
  already on disk. CSV, Parquet and Yahoo providers remain for other data;
  Yahoo is for equity checks only and unsuitable for futures.
* **Futures** — contract grammar and specifications (multiplier, tick size,
  tick value, expiry, first notice) for eight roots; calendar, volume,
  open-interest and N-days roll rules; unadjusted, backward and ratio
  adjustments; roll provenance and roll-gap diagnostics in basis points.
* **Execution** — commission, exchange fees, half-spread per side, slippage
  (constant and volatility-scaled), square-root market impact with a
  participation cap and partial fills; market, limit, stop and stop-limit
  matching with gap-aware fills; four cost scenarios reported as a ladder on
  which performance must decay monotonically.
* **Statistics** — Lo/Mertens standard errors, PSR, Deflated Sharpe, expected
  maximum Sharpe, minimum track record length, PBO via CSCV, purged K-fold,
  CPCV, walk-forward, stationary and moving-block bootstrap, Holm/BH/BY
  corrections, White's Reality Check and Hansen's SPA. Each one validated
  against an analytic value, a positive control and a negative control.
* **Structural scanner** — ADF-gated mean-reversion half-life, Lo–MacKinlay
  variance ratio, ARCH-LM, tails, seasonality, structural stability, Hurst
  against a null band. Evidence thresholds calibrated so that twenty
  independent random walks produce no strong exploitable finding.
* **Falsification engine** — eleven attacks (neighbouring parameters, 2x and
  3x costs, slippage shock, extra latency, dropping the best trades, shifting
  the window start, sub-period consistency, block bootstrap, price
  perturbation, roll methodology) and an unweighted robustness score that can
  only *lower* a verdict.
* **Commodity research layer** — 35 fundamental series for natural gas and
  crude, each specified with the publication lag measured from the end of the
  period it describes, and the revision policy. Providers are interfaces: no
  API key, SDK or vendor endpoint appears anywhere in the package.
* **Experiment registry** — append-only records, EXACT / EQUIVALENT /
  DIVERGENT / INCOMPARABLE reproduction comparison, and the multiple-testing
  ledger.
* **Reporting** — a 23-section standard report whose section 21, *"What could
  make this result false?"*, is generated from the numbers rather than written
  by the author.

## Tests

521 tests, 83% statement coverage. The ordering in CI is deliberate: temporal
integrity runs **first and alone**, because a lookahead invalidates every
other number in the build. Then the synthetic controls — a random walk that
must not earn a verdict above `WEAK`, a constructed edge that must be found,
and two planted lookaheads that must be caught — then the tests on three
Python versions, the reproducibility check, and the Docker build (whose
image runs `doctor` and the controls during the build itself).

Before every publication `tools/run_checks.py` runs these tests together with
the research tests, a static check and a scan for secrets and e-mail
addresses, and prepends the result to
[`docs/VERIFICATION_LOG.md`](docs/VERIFICATION_LOG.md).

```bash
uv run pytest tests -q                 # everything
uv run pytest tests/test_lookahead.py  # the module whose failure invalidates the rest
uv run exchange controls               # the framework's own calibration
```

## Documentation

* [`docs/ARCHITECTURE_AUDIT.md`](docs/ARCHITECTURE_AUDIT.md) — the audit of
  the predecessor, every finding cited by file and line.
* [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) — what each statistic assumes,
  what it cannot do, and where the implementation departs from the textbook.
* [`docs/AI_RESEARCHER.md`](docs/AI_RESEARCHER.md) — the autonomous loop's
  design, and the sandbox, allowlist, budget and credential requirements that
  must exist before it runs. It is not shipped.
* [`docs/ROADMAP.md`](docs/ROADMAP.md) — what exists, what is deliberately
  absent, and what comes next.
* [`docs/RESULTS_SEVEN_FAMILIES.md`](docs/RESULTS_SEVEN_FAMILIES.md) — the
  first run on real data: seven strategy families on MOEX natural gas and
  Brent, 125 configurations, every one rejected. Written in Russian, generated
  from the pickles by
  [`research/seven_families/`](research/seven_families/README.md).
* [`docs/RESULTS_MA_CROSSOVER_5M.md`](docs/RESULTS_MA_CROSSOVER_5M.md) — a
  price/moving-average crossover with a slow-MA slope filter on 5-minute NG
  and BR: the full 3,471,552-configuration grid per instrument, run twice,
  with walk-forward, PBO, Reality Check/SPA and a path-level null test. A
  real intraday regularity at zero cost; no tradable edge once costs and the
  single-path nature of the evidence are accounted for. In Russian, generated
  by [`research/ma_crossover_5m/`](research/ma_crossover_5m/README.md); charts
  in `notebooks/03_ma_crossover_5m.ipynb`.
* [`docs/figures/investor/`](docs/figures/investor/README.md) — one
  presentation dashboard per strategy (the seven families and the
  moving-average crossover on NG and BR): what was optimised, the map of the
  parameter search, the best in-sample combinations next to their
  out-of-sample result, equity, stability over time and the verdict. Built by
  `research/investor_charts/build.py` from the result files.
* [`research/trade_logs/`](research/trade_logs/README.md) — every trade of
  the selected strategies of both studies, in order, with its profit or loss:
  `trade_log.xlsx` (a sheet per strategy, running total and losing streaks as
  formulas), `trade_log_groups.xlsx` (the best moving-average strategies of
  each exit × slope-method group, 346k trades), `points/` (the outcome of
  every trade at each strategy's best parameters, in price points, one table
  per asset, with statistics) and the same trades as CSV. Each file is reconciled with the
  report it came from when it is exported.
* [`notebooks/00_presentation.ipynb`](notebooks/00_presentation.ipynb) — the
  whole project as one presentation, in Russian: the audit, the framework,
  the data, both studies and what they found. Built from the result files in
  this repository by [`research/presentation/`](research/presentation/build.py);
  every number in its text is read from those files.

## A caution

The framework was first validated on synthetic series with known ground
truth: a framework that cannot find a constructed edge, or that finds edges
in random walks, is broken in a way real data would hide. Both studies since
then ran on real MOEX AlgoPack quotes, and real data did expose what no
generator produces — a vendor splice that had to be unpicked into rolls,
trading sessions that had to be inferred, and two calibration bugs in the
scanner. Calibrating the moving-average study on noise also caught Hansen's
SPA rejecting far above its nominal size. All were fixed and pinned by tests.

Treat a `ROBUST` verdict as a hypothesis that has survived the attacks this
framework knows how to make. The important failure modes are usually the ones
nobody has measured yet.
