# Architecture audit: the existing research framework

**Subject:** `BazanovGo/Exchange` as of the branch point — 47 Python modules,
5,376 lines under `src/`, 12 test modules, an autonomous research loop under
`orchestrator/`, and a catalogue of 19 hypothesis runs under `research/`.

**Method:** every finding below was read off the source and is cited by file
and line. Where a claim is about behaviour rather than code, it was
reproduced by running the code.

**Verdict in one sentence:** the engineering is better than most research
code of this kind — the exit-mechanics simulator is careful, the walk-forward
module is real, the strategies are typed and tested — but a single defect in
the selection step invalidates the out-of-sample numbers that every verdict in
the catalogue rests on, and nothing in the framework corrects for the size of
the search that produced them.

---

## 1. Baseline: does it run?

| Step | Result |
|---|---|
| `pip install -r requirements.txt` | succeeds |
| `import src` | **fails** |
| `pytest` | **9 of 11 test modules fail at collection** |
| `pytest` after pinning `plotly<6` | 99 tests pass in ~90 s |

**Finding A1 (P3, reproducibility).** `requirements.txt` pins floors but no
ceilings: `plotly>=5.18` with `vectorbt>=0.26.0`. Plotly 6 removed
`Scattermapbox`, which vectorbt 0.26 imports at module scope, so a fresh
install of the declared dependency set cannot import the package at all. The
fix is one character of pinning; the lesson is that an unreproducible
environment makes every recorded result unreproducible regardless of the
research code's quality.

---

## 2. The finding that matters

**Finding B1 (P1, invalidates results).** *The out-of-sample window
participates in parameter selection, and the resulting number is then
reported as out-of-sample.*

The chain, verbatim from the source:

1. `src/optimization/research.py:121-127` — `score_combined` computes a
   composite for each window and then
   ```python
   out["robust_score"] = robust_score(out["is_composite"], out["oos_composite"])
   ```
2. `src/optimization/ranking.py:73-80` — the score is
   ```python
   (is_score + oos_score) / 2.0 - consistency_weight * (is_score - oos_score).abs()
   ```
3. `src/optimization/research.py:171` — the representative configuration for
   each strategy is
   ```python
   best_robust = g.loc[g["robust_score"].idxmax()]
   ```
4. `src/optimization/hypothesis.py:119,156` — the top five configurations *by
   `robust_score`* are reported together with their `oos_sharpe_ratio`.

The selected configuration is therefore the one that scored best on a blend of
both windows, and the number presented as its out-of-sample performance was
one of the inputs to choosing it. That number is an in-sample statistic
wearing an out-of-sample label.

Two things make this worse than an ordinary mistake:

* **It is undetectable downstream.** Walk-forward, Monte Carlo and the
  overfitting flags all run *after* selection, on the already-contaminated
  choice. No test in the framework can recover the lost information.
* **The intent was the opposite.** The docstring at `research.py:157` states
  that the representative configuration is chosen by `robust_score` "(NOT the
  best raw return)", and the penalty term exists to punish configurations that
  only shine in-sample. The author was explicitly guarding against
  overfitting, and the guard is the leak. This is the normal shape of this
  error: it arrives inside a measure designed to prevent it.

**Consequence for the catalogue.** All 19 runs in `research/catalog.csv`,
including the five graded `promising`, were produced through this path, so
their reported out-of-sample figures cannot be taken at face value. A
mitigating fact: every run was on synthetic GBM or Ornstein-Uhlenbeck series,
and the `promising` verdicts are on OU data where mean reversion exists by
construction. The catalogue therefore contains no claim about a real market
that this defect could have falsified — but it also contains no clean
out-of-sample measurement of anything.

---

## 3. Statistical validation

**Finding B2 (P1).** There is no correction for the size of the search. A
grep across `src/` for `deflated`, `pbo`, `probability_of_backtest`, `purged`,
`embargo`, `reality_check` and `benjamini` returns nothing. The programme
swept roughly 850 configurations in its classical-strategy stage, 74 more in
the exits stage, and 19 hypotheses with grids on top of that. The best of a
thousand noise draws has an expected Sharpe near `sqrt(2 ln 1000) ≈ 3.7`
standard errors above zero; a framework that reports the winner's p-value as
if it were a single pre-registered test will promote noise at a predictable
rate.

**Finding B3 (P2).** `composite_score` (`ranking.py:33-58`) is a weighted
average of **percentile ranks within the table**. Two consequences:

* A composite of 0.9 means "better than 90% of this grid", not "good". A grid
  in which every configuration loses money still has a 0.9.
* Weighting lets one excellent metric pay for one fatal one. A configuration
  with a spectacular Sharpe and a 90% drawdown can outrank a sound one.

**Finding B4 (P2).** Annualisation is the nominal 252
(`src/risk/metrics.py:9`, `src/indicators/volatility.py:24`,
`src/risk/position_sizing.py:27`) rather than measured from the series' own
index. For a business-day series the empirical figure is ≈261; for anything
irregular — futures sessions, holidays, mixed timeframes — the error is larger
and always in the flattering direction when the sample has gaps.

**What is genuinely good.** `src/optimization/walk_forward.py` re-optimises in
each window and reports walk-forward efficiency and the share of profitable
windows; `summarize_strategies` carries three honest overfitting flags
(degradation, rank instability, "the IS winner flops"). These are the right
diagnostics. They are computed on a contaminated selection.

---

## 4. Execution and lookahead

**Finding C1 (P1).** Entries fill at the close of the bar that generated the
signal. `src/backtesting/engine.py:98` calls `vbt.Portfolio.from_signals` with
`close=ohlcv[price_col]`, and `src/backtesting/position_management.py:15`
documents the convention explicitly: "entries fill at the bar close (same
convention as `from_signals`)". Strategy signals are computed from that same
bar's close. The backtest therefore trades at a price that was being
discovered at the moment of the decision. The bias is small per trade and
systematic across every trade, and it is largest exactly where these
strategies claim their edge: breakouts and reversals, where the close is the
extreme of the bar.

**Finding C2 (P2).** There is no automated lookahead test — no causality
proof, no leakage scan, no control series with a planted future value. The
property is asserted in docstrings and reviewed by eye.

**What is genuinely good.** The stop simulator in `position_management.py` is
the most careful code in the repository: gap-aware fills ("if the bar opens
through the level, the fill is at the open"), trailing stops that use the
*previous* bars' peak so a bar cannot raise a stop and hit it in the same bar,
and stop-wins-over-target when both are touched. That is the correct,
conservative treatment, and it is more rigorous than most commercial
backtesters.

---

## 5. Data

**Finding D1 (P2).** `src/data/base.py:85-86` discards the timezone:
```python
if getattr(out.index, "tz", None) is not None:
    out.index = out.index.tz_localize(None)
```
A naive index cannot express a session boundary, cannot be aligned across
markets, and silently mixes conventions when two sources disagree.

**Finding D2 (P2).** Duplicate timestamps are resolved silently
(`base.py:84`, `keep="last"`). A duplicate bar is a data error; resolving it
without a record hides the fact that the source was wrong.

**Finding D3 (P2).** There is no data-quality gate: no severity-graded checks
for missing bars, OHLC consistency, non-positive prices, impossible jumps or
staleness, and no report attached to a dataset.

**Finding D4 (P1 for the stated goal).** There is no futures support at all.
No contract specification (multiplier, tick size, tick value, expiry, first
notice), no roll rules, no continuous-series construction, no adjustment
method. For a programme whose declared target is natural gas and crude oil
this is the largest functional gap: the instrument's economics are not
representable, and the single most common source of fictitious commodity alpha
— the roll splice — cannot even be measured.

**Finding D5 (P2).** There is no content hashing of datasets, so two results
computed on different data are indistinguishable in the record.

---

## 6. Reproducibility and the record

**Finding E1 (P2).** Results are markdown and CSV under `research/`; there is
no experiment record carrying the hypothesis, the dataset hash, the instrument
specification, the execution assumptions, the seed and the environment, and no
way to re-run a recorded experiment and compare. `research/catalog.csv` lists
verdicts but not the inputs needed to reproduce them.

**Finding E2 (P3).** Seeds are passed where they are needed but there is no
single deterministic mode, and `PYTHONHASHSEED` is not pinned anywhere in the
tooling.

---

## 7. The autonomous loop

**Finding F1 (P1, security).** `orchestrator/config.yaml:42` invokes the agent
with `--dangerously-skip-permissions`:
```yaml
claude_command:
  - claude
  - -p
  - "{prompt}"
  - --model
  - "{model}"
  - --dangerously-skip-permissions
  - --max-turns
  - "120"
```
The loop runs an LLM agent for up to 120 turns with every permission check
disabled, on a machine that also holds the research record. There is no
sandbox, no filesystem or network allowlist, no CPU, memory or wall-clock
budget, and nothing prevents the agent from rewriting an earlier verdict, the
research log or the orchestrator's own configuration. The research risk is as
large as the security risk: a loop that can edit its own history cannot
produce a trustworthy record of what it found.

**Finding F2 (P2).** The loop's output is the research record. There is no
append-only constraint, so the distinction between "this hypothesis was
rejected" and "this hypothesis was never run" is not preserved by the storage.

---

## 8. What the replacement keeps, and what it changes

Kept, because it is right: the exit-mechanics simulator's gap-aware
conservative fills; walk-forward with re-optimisation; the honest overfitting
flags; typed strategy classes with declared parameter grids; synthetic
generators with known ground truth as controls.

Changed structurally rather than by convention, so the honest behaviour is the
only behaviour available:

| Defect | Structural remedy |
|---|---|
| B1 selection contamination | The in-sample search is a function that receives only the in-sample slice, and the sweep table it reads has no out-of-sample column at all (`exchange/funnel/pipeline.py`). |
| B2 no deflation | Deflated Sharpe, PBO, purged and combinatorial CV, FWER and FDR corrections, Reality Check and SPA, with the trial count charged automatically from the registry's ledger. |
| B3 weighted score | Conjunctive verdict gates. No single metric can raise a verdict past a failed gate; the robustness score is an unweighted count that can only lower one. |
| C1 same-bar fills | The engine executes orders on a later bar by construction; `fill_timing="same_close"` exists only so it can be marked as admitting lookahead and the result marked invalid. |
| C2 no lookahead tests | A causality proof run against every registered strategy, a separate leakage scan for pre-contaminated columns, and planted-lookahead controls in CI. |
| D1-D5 data | UTC-aware canonical schema, a severity-graded quality gate, full contract specifications, four roll rules, three adjustment methods with roll-gap diagnostics in basis points, and content hashing of every dataset. |
| E1-E2 reproducibility | An append-only experiment registry recording every input, a `reproduce` command that compares against the record, deterministic mode, and `PYTHONHASHSEED` pinned in CI and the image. |
| F1-F2 the loop | Not shipped. `exchange research-loop` exists only to state the sandbox, allowlist, budget and credential requirements that must exist first. |

The master criterion for every one of those changes was the same question:
*does this reduce the probability of mistaking randomness, overfitting or a
data error for a real trading pattern?* Where the answer was no, the change
was not made.
