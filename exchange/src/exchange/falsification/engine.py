"""The falsification engine.

The funnel asks "does this hypothesis survive the standard validation
sequence?". This module asks the harder question: **what would have to be
true for the result to be an artefact, and does the result still hold when
that possibility is forced?**

Each attack below is a deliberate attempt to destroy the result by changing
something that *should not matter* if the effect is real:

===========================  ===================================================
attack                       the artefact it is designed to expose
===========================  ===================================================
``neighbour_params``         a single lucky cell in the parameter grid
``cost_double``              an edge smaller than the real cost of trading it
``cost_triple``              the same, at a cost level a bad fill day produces
``slippage_shock``           an edge that lives entirely inside the spread
``execution_latency``        a microstructure effect this data cannot measure
``drop_best_trades``         a "strategy" that is a handful of lucky days
``start_date_shift``         dependence on one arbitrary window boundary
``subperiod_consistency``    an effect confined to one stretch of history
``bootstrap_resample``       a Sharpe indistinguishable from sampling noise
``price_perturbation``       a signal fitted to exact prices, i.e. to data errors
``roll_methodology``         alpha manufactured by the futures splice
===========================  ===================================================

Why the score can only lower a verdict
---------------------------------------
:class:`RobustnessReport` reports a score, because the master specification
asks for one, but it is deliberately **not** a weighted average of the
attacks and it can never raise a verdict. Two reasons:

1. A weighted score lets one spectacular survival pay for one fatal failure.
   That is exactly the defect of the audited project's ``robust_score``, and
   it is why an edge that dies when costs double could be labelled robust
   there. Here every attack is reported by name, failures are listed
   individually, and any failed *critical* attack caps the verdict at WEAK
   regardless of the score.
2. The attacks re-use the out-of-sample window many times over. That is
   legitimate for *refutation* — finding that a result breaks is informative
   however many times you looked — but it is not legitimate for
   *confirmation*: surviving thirty looks at the same data is not thirty
   independent confirmations. So survival earns no credit beyond "not yet
   refuted", and :meth:`RobustnessReport.verdict_cap` is a ceiling, never a
   floor.

For the same reason no attack re-selects parameters. ``neighbour_params``
evaluates the neighbourhood of the already-chosen cell and reports whether
it is a plateau or a spike; it never replaces the choice with a better
neighbour, which would be out-of-sample selection by another name.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from exchange.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from exchange.backtest.metrics import sharpe_ratio
from exchange.core.determinism import derive_seed, rng
from exchange.core.errors import ConfigError
from exchange.core.logging import get_logger
from exchange.execution.assumptions import ExecutionAssumptions
from exchange.funnel.hypothesis import Hypothesis
from exchange.funnel.verdict import VerdictLevel
from exchange.stats.bootstrap import block_bootstrap
from exchange.strategies.base import Strategy, strategy_class

logger = get_logger(__name__)

# Severity decides what a failure costs, and nothing else does.
#   critical : failing means the result is almost certainly an artefact.
#   major    : failing means the result is real only under conditions the
#              research programme cannot rely on.
#   minor    : failing weakens the result without refuting it.
Severity = str
CRITICAL: Severity = "critical"
MAJOR: Severity = "major"
MINOR: Severity = "minor"


# ==========================================================================
# Attack results
# ==========================================================================


@dataclass
class AttackOutcome:
    """What one attack found."""

    name: str
    question: str
    severity: Severity
    survived: bool | None          # None = the attack did not apply
    baseline: float = float("nan")
    perturbed: float = float("nan")
    detail: str = ""
    evidence: dict = field(default_factory=dict)

    @property
    def applicable(self) -> bool:
        return self.survived is not None

    @property
    def retained(self) -> float:
        """Fraction of the baseline statistic that survived the attack."""
        if not np.isfinite(self.baseline) or self.baseline == 0:
            return float("nan")
        return float(self.perturbed / self.baseline)

    def to_dict(self) -> dict:
        return {
            "attack": self.name,
            "question": self.question,
            "severity": self.severity,
            "survived": self.survived,
            "baseline": _num(self.baseline),
            "perturbed": _num(self.perturbed),
            "retained": _num(self.retained),
            "detail": self.detail,
            "evidence": {k: _num(v) for k, v in self.evidence.items()},
        }


GRADES = ("FRAGILE", "MIXED", "STURDY", "HARDENED")

# Each grade's ceiling on the funnel verdict. A report never raises a
# verdict, so these are maxima only.
_GRADE_CAP = {
    "FRAGILE": VerdictLevel.WEAK,
    "MIXED": VerdictLevel.PROMISING,
    "STURDY": VerdictLevel.ROBUST,
    "HARDENED": VerdictLevel.PAPER_TRADING_CANDIDATE,
}


@dataclass
class RobustnessReport:
    """The full attack record, and the one number that summarises it."""

    outcomes: list[AttackOutcome] = field(default_factory=list)
    baseline_sharpe: float = float("nan")
    hypothesis_id: str = ""
    n_bars: int = 0

    # -- composition ----------------------------------------------------

    @property
    def evaluated(self) -> list[AttackOutcome]:
        return [o for o in self.outcomes if o.applicable]

    @property
    def survived(self) -> list[AttackOutcome]:
        return [o for o in self.outcomes if o.survived is True]

    @property
    def failed(self) -> list[AttackOutcome]:
        return [o for o in self.outcomes if o.survived is False]

    @property
    def skipped(self) -> list[AttackOutcome]:
        return [o for o in self.outcomes if o.survived is None]

    @property
    def critical_failures(self) -> list[AttackOutcome]:
        return [o for o in self.failed if o.severity == CRITICAL]

    # -- the score ------------------------------------------------------

    @property
    def score(self) -> float:
        """Share of *evaluated* attacks survived.

        Unweighted on purpose: the point of the number is to be auditable,
        and any weighting scheme would be an undeclared theory about which
        artefact matters most. Attacks that did not apply are excluded
        rather than counted as passes — a test that was not run is not
        evidence of anything.
        """
        evaluated = self.evaluated
        if not evaluated:
            return float("nan")
        return len(self.survived) / len(evaluated)

    @property
    def grade(self) -> str:
        if self.critical_failures:
            return "FRAGILE"
        score = self.score
        if not np.isfinite(score):
            return "FRAGILE"
        if score >= 1.0:
            return "HARDENED"
        if score >= 0.8:
            return "STURDY"
        if score >= 0.6:
            return "MIXED"
        return "FRAGILE"

    @property
    def verdict_cap(self) -> VerdictLevel:
        """The highest verdict this attack record permits."""
        return _GRADE_CAP[self.grade]

    def cap(self, level: VerdictLevel) -> VerdictLevel:
        """Apply the ceiling to a funnel verdict level."""
        return VerdictLevel(min(int(level), int(self.verdict_cap)))

    def outcome(self, name: str) -> AttackOutcome | None:
        return next((o for o in self.outcomes if o.name == name), None)

    # -- output ---------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "baseline_sharpe": _num(self.baseline_sharpe),
            "n_bars": self.n_bars,
            "n_attacks": len(self.outcomes),
            "n_evaluated": len(self.evaluated),
            "n_survived": len(self.survived),
            "n_failed": len(self.failed),
            "score": _num(self.score),
            "grade": self.grade,
            "verdict_cap": self.verdict_cap.name,
            "critical_failures": [o.name for o in self.critical_failures],
            "attacks": [o.to_dict() for o in self.outcomes],
        }

    def to_frame(self) -> pd.DataFrame:
        rows = [o.to_dict() for o in self.outcomes]
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows).set_index("attack")
        return frame.drop(columns=["evidence"])

    def to_markdown(self) -> str:
        lines = [
            "## Falsification attacks",
            "",
            f"Baseline out-of-sample Sharpe: **{self.baseline_sharpe:.3f}** "
            f"over {self.n_bars} bars.",
            "",
            f"**Robustness score: {self.score:.2f}** "
            f"({len(self.survived)}/{len(self.evaluated)} attacks survived) — "
            f"grade **{self.grade}**, verdict capped at `{self.verdict_cap.name}`.",
            "",
            "| attack | severity | result | baseline | attacked | detail |",
            "|---|---|---|---|---|---|",
        ]
        for o in self.outcomes:
            mark = {True: "survived", False: "**FAILED**", None: "n/a"}[o.survived]
            lines.append(
                f"| `{o.name}` | {o.severity} | {mark} | {_fmt(o.baseline)} | "
                f"{_fmt(o.perturbed)} | {o.detail} |"
            )
        if self.failed:
            lines += ["", "### What these failures mean", ""]
            for o in self.failed:
                lines.append(f"- **`{o.name}`** — {o.question} Answer: no. {o.detail}")
        if self.skipped:
            lines += ["", "### Not evaluated", ""]
            for o in self.skipped:
                lines.append(f"- `{o.name}` — {o.detail}")
        lines += [
            "",
            "The score is an unweighted count and cannot raise a verdict; see "
            "`exchange.falsification.engine` for why.",
        ]
        return "\n".join(lines) + "\n"


# ==========================================================================
# What an attack is given
# ==========================================================================


@dataclass
class AttackContext:
    """Everything an attack needs, and nothing it should not have.

    Attacks operate on the **out-of-sample window only**. Attacking the
    in-sample window would measure how well the search fitted it, which is
    already known and is not the question.
    """

    bars: pd.DataFrame
    hypothesis: Hypothesis
    params: Mapping[str, Any]
    config: BacktestConfig
    symbol: str
    bars_per_year: float
    seed: int = 20260902
    n_resamples: int = 1000
    alternative_bars: Mapping[str, pd.DataFrame] = field(default_factory=dict)
    baseline: BacktestResult | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if len(self.bars) < 50:
            raise ConfigError(
                f"falsification needs at least 50 out-of-sample bars, got {len(self.bars)}"
            )

    # -- helpers available to every attack ------------------------------

    def strategy(self, overrides: Mapping[str, Any] | None = None) -> Strategy:
        cls = strategy_class(self.hypothesis.strategy)
        return cls(
            **{
                **dict(self.hypothesis.fixed_params),
                **dict(self.params),
                **dict(overrides or {}),
            }
        )

    def run(
        self,
        bars: pd.DataFrame | None = None,
        *,
        params: Mapping[str, Any] | None = None,
        execution: ExecutionAssumptions | None = None,
    ) -> BacktestResult:
        """Re-run the selected configuration under a perturbation."""
        bars = self.bars if bars is None else bars
        strategy = self.strategy(params)
        config = self.config
        if execution is not None:
            config = BacktestConfig(
                initial_cash=config.initial_cash,
                instrument=config.instrument,
                execution=execution,
                volatility_lookback=config.volatility_lookback,
                annualisation=config.annualisation,
            )
        return run_backtest(bars, strategy.positions(bars), config=config, symbol=self.symbol)

    def sharpe(self, result: BacktestResult) -> float:
        return sharpe_ratio(result.returns, periods_per_year=self.bars_per_year)

    def baseline_result(self) -> BacktestResult:
        if self.baseline is None:
            self.baseline = self.run()
        return self.baseline

    def baseline_sharpe(self) -> float:
        return self.sharpe(self.baseline_result())

    def rng(self, label: str) -> np.random.Generator:
        """A generator that depends on the attack name, so attacks are
        independent of each other and of their execution order."""
        return rng(derive_seed(label, base=self.seed))


Attack = Callable[[AttackContext], AttackOutcome]


# ==========================================================================
# The attacks
# ==========================================================================


def neighbour_params(ctx: AttackContext) -> AttackOutcome:
    """Is the chosen cell a plateau, or an isolated spike?

    A real effect is insensitive to small parameter changes, because the
    market process it exploits does not know the difference between a
    20-bar and a 22-bar window. A spike surrounded by failure is the
    signature of a grid search finding one lucky draw.
    """
    question = "Do neighbouring parameter values work too?"
    neighbours = _neighbour_grid(ctx.hypothesis.param_space, ctx.params)
    if not neighbours:
        return AttackOutcome(
            "neighbour_params", question, CRITICAL, None,
            detail="no swept parameters with adjacent grid values",
        )

    baseline = ctx.baseline_sharpe()
    sharpes = []
    for override in neighbours:
        try:
            sharpes.append(ctx.sharpe(ctx.run(params=override)))
        except ConfigError:
            continue
    if not sharpes:
        return AttackOutcome(
            "neighbour_params", question, CRITICAL, None,
            baseline=baseline,
            detail="every neighbouring combination was rejected as invalid",
        )

    arr = np.asarray(sharpes, float)
    median = float(np.nanmedian(arr))
    positive = float(np.mean(arr > 0))
    # Survival: the neighbourhood is positive on balance, and the chosen cell
    # is not a lone spike (it keeps no more than four times the neighbourhood
    # median, or the median is already comfortably positive).
    survived = bool(median > 0.0 and positive >= 0.5)
    detail = (
        f"{len(arr)} neighbour(s): median Sharpe {median:.3f}, "
        f"{positive:.0%} positive (chosen cell {baseline:.3f})"
    )
    if survived and np.isfinite(baseline) and baseline > 0 and median < 0.25 * baseline:
        detail += "; the chosen cell is far above its neighbourhood — treat as a spike"
        survived = False
    return AttackOutcome(
        "neighbour_params", question, CRITICAL, survived,
        baseline=baseline, perturbed=median, detail=detail,
        evidence={"median": median, "positive_fraction": positive, "n_neighbours": len(arr)},
    )


def cost_double(ctx: AttackContext) -> AttackOutcome:
    """Does the edge survive twice the modelled cost of trading it?"""
    return _cost_attack(ctx, "cost_double", 2.0, CRITICAL)


def cost_triple(ctx: AttackContext) -> AttackOutcome:
    """Does the edge survive three times the modelled cost?

    Not a fantasy: three times the modelled cost is an ordinary bad week in
    an illiquid back month, and a strategy that only works at the modelled
    cost is a strategy whose result is a statement about the cost model.
    """
    return _cost_attack(ctx, "cost_triple", 3.0, MAJOR)


def slippage_shock(ctx: AttackContext) -> AttackOutcome:
    """Does the edge survive a slippage and spread shock alone?

    Commissions are known almost exactly; slippage and the spread are
    estimates. Shocking only the estimated part isolates how much of the
    result rests on them.
    """
    question = "Does the edge survive 3x slippage and 2x spread?"
    baseline = ctx.baseline_sharpe()
    execution = ctx.config.execution.scale_slippage(3.0).scale_spread(2.0)
    attacked = ctx.sharpe(ctx.run(execution=execution))
    survived = bool(np.isfinite(attacked) and attacked > 0.0)
    return AttackOutcome(
        "slippage_shock", question, MAJOR, survived,
        baseline=baseline, perturbed=attacked,
        detail=f"Sharpe {baseline:.3f} -> {attacked:.3f} under 3x slippage, 2x spread",
    )


def execution_latency(ctx: AttackContext) -> AttackOutcome:
    """Does the edge survive one and two extra bars of delay?

    The engine already executes on the next bar. If an additional bar of
    latency destroys the result, the effect lives at a horizon shorter than
    the bar size, which means this data cannot measure it and this
    execution model cannot capture it.
    """
    question = "Does the edge survive extra execution delay?"
    baseline = ctx.baseline_sharpe()
    base_latency = ctx.config.execution.latency_bars
    sharpes = {}
    for extra in (1, 2):
        execution = ctx.config.execution.with_latency(base_latency + extra)
        sharpes[extra] = ctx.sharpe(ctx.run(execution=execution))
    one = sharpes[1]
    survived = bool(np.isfinite(one) and one > 0.0)
    detail = (
        f"Sharpe {baseline:.3f} -> {one:.3f} (+1 bar) -> {sharpes[2]:.3f} (+2 bars)"
    )
    if survived and np.isfinite(baseline) and baseline > 0 and one < 0.5 * baseline:
        detail += "; more than half the edge is in the first bar after the signal"
    return AttackOutcome(
        "execution_latency", question, CRITICAL, survived,
        baseline=baseline, perturbed=one, detail=detail,
        evidence={"plus_one": one, "plus_two": sharpes[2]},
    )


def drop_best_trades(ctx: AttackContext) -> AttackOutcome:
    """Is the result the strategy, or is it five lucky days?

    Removing the best trades is not a realistic scenario — it is a
    concentration measure. If the whole profit is in the top handful of
    round trips, the sample contains a few events, not a repeatable effect,
    and the Sharpe is meaningless however large it is.
    """
    question = "Does the result survive removing its best trades?"
    baseline_result = ctx.baseline_result()
    trades = baseline_result.trades
    baseline = ctx.baseline_sharpe()
    if trades is None or trades.empty or "net_pnl" not in trades:
        return AttackOutcome(
            "drop_best_trades", question, CRITICAL, None,
            baseline=baseline, detail="no completed round trips to drop",
        )
    pnl = trades["net_pnl"].to_numpy(float)
    total = float(np.nansum(pnl))
    k = max(1, int(np.ceil(0.05 * len(pnl))))
    order = np.argsort(pnl)[::-1]
    dropped = float(np.nansum(pnl[order[:k]]))
    remaining = total - dropped
    share = dropped / total if total > 0 else float("nan")
    survived = bool(total > 0 and remaining > 0)
    detail = (
        f"top {k} of {len(pnl)} trade(s) contribute {share:.0%} of net P&L; "
        f"{total:,.0f} -> {remaining:,.0f} without them"
    )
    if not survived and total <= 0:
        detail = f"net P&L is {total:,.0f} before any trade is dropped"
    return AttackOutcome(
        "drop_best_trades", question, CRITICAL, survived,
        baseline=total, perturbed=remaining, detail=detail,
        evidence={"n_dropped": k, "n_trades": len(pnl), "best_share": share},
    )


def start_date_shift(ctx: AttackContext) -> AttackOutcome:
    """Does the result depend on where the window happens to start?

    A window boundary is an arbitrary artefact of when the data was pulled.
    An effect that only appears for one starting bar is a property of the
    sample, not of the market.
    """
    question = "Does the result survive moving the window start?"
    baseline = ctx.baseline_sharpe()
    n = len(ctx.bars)
    offsets = [int(f * n) for f in (0.05, 0.10, 0.20, 0.30)]
    sharpes = []
    for offset in offsets:
        sub = ctx.bars.iloc[offset:]
        if len(sub) < 50:
            continue
        sharpes.append(ctx.sharpe(ctx.run(sub)))
    if not sharpes:
        return AttackOutcome(
            "start_date_shift", question, MAJOR, None,
            baseline=baseline, detail="window too short to shift its start",
        )
    arr = np.asarray(sharpes, float)
    positive = float(np.mean(arr > 0))
    worst = float(np.nanmin(arr))
    survived = bool(positive >= 0.75 and np.nanmedian(arr) > 0)
    return AttackOutcome(
        "start_date_shift", question, MAJOR, survived,
        baseline=baseline, perturbed=float(np.nanmedian(arr)),
        detail=(
            f"{len(arr)} shifted start(s): {positive:.0%} positive, "
            f"median {np.nanmedian(arr):.3f}, worst {worst:.3f}"
        ),
        evidence={"positive_fraction": positive, "worst": worst},
    )


def subperiod_consistency(ctx: AttackContext) -> AttackOutcome:
    """Is the effect present throughout, or confined to one stretch?"""
    question = "Is the edge present across sub-periods?"
    baseline = ctx.baseline_sharpe()
    result = ctx.baseline_result()
    returns = result.returns
    blocks = np.array_split(np.arange(len(returns)), 4)
    stats = []
    for block in blocks:
        if block.size < 10:
            continue
        stats.append(sharpe_ratio(returns.iloc[block], periods_per_year=ctx.bars_per_year))
    if len(stats) < 3:
        return AttackOutcome(
            "subperiod_consistency", question, MAJOR, None,
            baseline=baseline, detail="not enough bars for four sub-periods",
        )
    arr = np.asarray(stats, float)
    n_positive = int(np.sum(arr > 0))
    survived = bool(n_positive >= len(arr) - 1)
    return AttackOutcome(
        "subperiod_consistency", question, MAJOR, survived,
        baseline=baseline, perturbed=float(np.nanmin(arr)),
        detail=(
            f"{n_positive}/{len(arr)} sub-periods positive "
            f"(Sharpes {', '.join(f'{s:.2f}' for s in arr)})"
        ),
        evidence={"n_positive": n_positive, "n_blocks": len(arr)},
    )


def bootstrap_resample(ctx: AttackContext) -> AttackOutcome:
    """Is the Sharpe distinguishable from a resampling of its own noise?

    A stationary block bootstrap preserves the serial dependence of the
    return series, so the interval it produces is not the naive i.i.d. one
    that systematically overstates significance for autocorrelated returns.
    """
    question = "Is the Sharpe distinguishable from sampling noise?"
    baseline = ctx.baseline_sharpe()
    returns = ctx.baseline_result().returns
    if returns.abs().sum() == 0:
        return AttackOutcome(
            "bootstrap_resample", question, CRITICAL, None,
            baseline=baseline, detail="the strategy never traded",
        )
    boot = block_bootstrap(
        returns.to_numpy(float),
        lambda x: float(np.mean(x) / np.std(x, ddof=1)) if np.std(x, ddof=1) > 0 else 0.0,
        n_resamples=ctx.n_resamples,
        method="stationary",
        seed=derive_seed("bootstrap_resample", base=ctx.seed),
    )
    lo, hi = boot.confidence_interval(0.95)
    scale = float(np.sqrt(ctx.bars_per_year))
    survived = bool(lo > 0.0)
    return AttackOutcome(
        "bootstrap_resample", question, CRITICAL, survived,
        baseline=baseline, perturbed=lo * scale,
        detail=(
            f"95% block-bootstrap interval on the annualised Sharpe "
            f"[{lo * scale:.3f}, {hi * scale:.3f}]"
        ),
        evidence={"ci_low_per_bar": lo, "ci_high_per_bar": hi, "p_value": boot.p_value_greater_than(0.0)},
    )


def price_perturbation(ctx: AttackContext) -> AttackOutcome:
    """Does the signal depend on exact prices?

    Prices carry errors: bad ticks, stale closes, settlement conventions. A
    signal that changes sign when a price moves by a fraction of its own
    daily range is reading those errors, not the market. The perturbation
    stays inside the bar's own noise scale and the OHLC invariant is
    restored afterwards, so the attacked series is still a legal price
    series.
    """
    question = "Does the result survive small price perturbations?"
    baseline = ctx.baseline_sharpe()
    sharpes = []
    for trial in range(5):
        noisy = _perturb_bars(ctx.bars, ctx.rng(f"price_perturbation/{trial}"), scale=0.1)
        sharpes.append(ctx.sharpe(ctx.run(noisy)))
    arr = np.asarray(sharpes, float)
    positive = float(np.mean(arr > 0))
    median = float(np.nanmedian(arr))
    survived = bool(positive >= 0.8 and median > 0)
    return AttackOutcome(
        "price_perturbation", question, MAJOR, survived,
        baseline=baseline, perturbed=median,
        detail=(
            f"{len(arr)} perturbed series (noise ~10% of a bar range): "
            f"median Sharpe {median:.3f}, {positive:.0%} positive"
        ),
        evidence={"positive_fraction": positive, "median": median},
    )


def roll_methodology(ctx: AttackContext) -> AttackOutcome:
    """Was the edge manufactured by the futures splice?

    Back-adjusting a futures series moves historical prices. Done one way
    the series trends; done another it does not. An apparent trend edge that
    appears under backward adjustment and vanishes under a ratio adjustment
    (or under a volume-based roll) is an artefact of the splice, which is
    the single most common source of fictitious alpha in commodity research.

    Requires alternative constructions of the same window, supplied by the
    caller via :attr:`AttackContext.alternative_bars` — the engine does not
    reach into the data layer to invent them.
    """
    question = "Is the edge independent of the roll and adjustment method?"
    baseline = ctx.baseline_sharpe()
    alternatives = dict(ctx.alternative_bars)
    if not alternatives:
        return AttackOutcome(
            "roll_methodology", question, CRITICAL, None,
            baseline=baseline,
            detail="not a spliced futures series, or no alternative construction supplied",
        )
    sharpes: dict[str, float] = {}
    for label, bars in alternatives.items():
        aligned = bars.reindex(ctx.bars.index).dropna(how="any")
        if len(aligned) < 50:
            continue
        sharpes[label] = ctx.sharpe(ctx.run(aligned))
    if not sharpes:
        return AttackOutcome(
            "roll_methodology", question, CRITICAL, None,
            baseline=baseline, detail="no alternative construction overlapped this window",
        )
    arr = np.asarray(list(sharpes.values()), float)
    worst = float(np.nanmin(arr))
    survived = bool(np.all(arr > 0))
    return AttackOutcome(
        "roll_methodology", question, CRITICAL, survived,
        baseline=baseline, perturbed=worst,
        detail=(
            "alternative constructions: "
            + ", ".join(f"{k} {v:.3f}" for k, v in sorted(sharpes.items()))
            + f" (baseline {baseline:.3f})"
        ),
        evidence={f"sharpe_{k}": v for k, v in sharpes.items()},
    )


ATTACKS: tuple[Attack, ...] = (
    neighbour_params,
    cost_double,
    cost_triple,
    slippage_shock,
    execution_latency,
    drop_best_trades,
    start_date_shift,
    subperiod_consistency,
    bootstrap_resample,
    price_perturbation,
    roll_methodology,
)

ATTACK_NAMES: tuple[str, ...] = tuple(a.__name__ for a in ATTACKS)


# ==========================================================================
# Runner
# ==========================================================================


def run_attacks(
    ctx: AttackContext,
    *,
    attacks: Sequence[Attack] | Sequence[str] | None = None,
) -> RobustnessReport:
    """Run every attack and collect the record.

    An attack that raises is recorded as *not evaluated* with the exception
    text, never silently dropped and never counted as a pass: a crashed
    probe is missing evidence, and reporting it as survival would be the
    exact failure mode this module exists to prevent.
    """
    selected = _resolve_attacks(attacks)
    report = RobustnessReport(hypothesis_id=ctx.hypothesis.hypothesis_id, n_bars=len(ctx.bars))
    try:
        report.baseline_sharpe = ctx.baseline_sharpe()
    except Exception as exc:  # pragma: no cover - defensive
        raise ConfigError(f"the baseline configuration does not run: {exc}") from exc

    for attack in selected:
        name = attack.__name__
        try:
            outcome = attack(ctx)
        except Exception as exc:  # an attack must not abort the audit
            logger.warning("attack %s failed to run: %s", name, exc)
            outcome = AttackOutcome(
                name, "", MINOR, None,
                detail=f"the attack itself raised {type(exc).__name__}: {exc}",
            )
        logger.info(
            "attack %-22s %s", name,
            {True: "survived", False: "FAILED", None: "n/a"}[outcome.survived],
        )
        report.outcomes.append(outcome)
    return report


def falsify(
    result,
    *,
    attacks: Sequence[Attack] | Sequence[str] | None = None,
    alternative_bars: Mapping[str, pd.DataFrame] | None = None,
    n_resamples: int = 1000,
) -> RobustnessReport:
    """Attack a :class:`~exchange.funnel.pipeline.FunnelResult`.

    The attacks use the funnel's own out-of-sample window and the parameters
    the in-sample search selected. Nothing is re-selected.
    """
    from exchange.funnel.pipeline import FunnelResult  # local: avoids a cycle

    if not isinstance(result, FunnelResult):  # pragma: no cover - defensive
        raise ConfigError("falsify() expects a FunnelResult")
    if result.oos_result is None:
        raise ConfigError(
            f"{result.hypothesis.hypothesis_id} has no out-of-sample result to attack "
            "(the funnel stopped at the sanity stage)"
        )

    oos_bars = result.full_bars.loc[result.oos_result.equity.index]
    ctx = AttackContext(
        bars=oos_bars,
        hypothesis=result.hypothesis,
        params=dict(result.best_params),
        config=result.oos_result.config,
        symbol=result.oos_result.symbol,
        bars_per_year=result.oos_result.bars_per_year,
        seed=result.config.seed,
        n_resamples=n_resamples,
        alternative_bars=dict(alternative_bars or {}),
        baseline=result.oos_result,
    )
    return run_attacks(ctx, attacks=attacks)


# ==========================================================================
# Internals
# ==========================================================================


def _resolve_attacks(attacks) -> list[Attack]:
    if attacks is None:
        return list(ATTACKS)
    by_name = {a.__name__: a for a in ATTACKS}
    out: list[Attack] = []
    for item in attacks:
        if callable(item):
            out.append(item)
        elif item in by_name:
            out.append(by_name[item])
        else:
            raise ConfigError(f"unknown attack {item!r}; available: {list(by_name)}")
    return out


def _cost_attack(ctx: AttackContext, name: str, factor: float, severity: Severity) -> AttackOutcome:
    question = f"Does the edge survive {factor:g}x the modelled execution cost?"
    baseline = ctx.baseline_sharpe()
    execution = (
        ctx.config.execution
        .scale_slippage(factor)
        .scale_spread(factor)
        .scale_commission(factor)
    )
    attacked = ctx.sharpe(ctx.run(execution=execution))
    survived = bool(np.isfinite(attacked) and attacked > 0.0)
    detail = f"Sharpe {baseline:.3f} -> {attacked:.3f} at {factor:g}x cost"
    if np.isfinite(baseline) and np.isfinite(attacked) and attacked > baseline:
        detail += "; costs *improved* performance, which indicates a sign error, not robustness"
        survived = False
    return AttackOutcome(
        name, question, severity, survived,
        baseline=baseline, perturbed=attacked, detail=detail,
    )


def _neighbour_grid(
    param_space: Mapping[str, Sequence[Any]],
    chosen: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """One-step neighbours of ``chosen`` in the swept grid.

    Only one parameter moves at a time: the question is whether the cell
    sits on a plateau, and a diagonal move changes two things at once.
    """
    out: list[dict[str, Any]] = []
    for key, values in param_space.items():
        values = list(values)
        if key not in chosen or len(values) < 2:
            continue
        try:
            position = values.index(chosen[key])
        except ValueError:
            continue
        for step in (-1, 1):
            index = position + step
            if 0 <= index < len(values):
                out.append({key: values[index]})
    return out


def _perturb_bars(
    bars: pd.DataFrame,
    generator: np.random.Generator,
    *,
    scale: float = 0.1,
) -> pd.DataFrame:
    """Add noise proportional to each bar's own range, keeping OHLC legal.

    The noise is scaled by the *trailing* median range so the perturbation
    is of the same order as the data's own measurement error, and the
    high/low are widened to contain the perturbed open and close rather
    than being perturbed independently (which would break the invariant and
    be rejected by the data gate).
    """
    out = bars.copy()
    rng_range = (bars["high"] - bars["low"]).rolling(20, min_periods=1).median()
    sigma = (rng_range * scale).to_numpy(float)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, bars["close"].to_numpy(float) * 1e-4)
    for column in ("open", "close"):
        noise = generator.normal(0.0, 1.0, size=len(out)) * sigma
        out[column] = (out[column].to_numpy(float) + noise).clip(min=1e-9)
    out["high"] = np.maximum(bars["high"], out[["open", "close"]].max(axis=1))
    out["low"] = np.minimum(bars["low"], out[["open", "close"]].min(axis=1)).clip(lower=1e-9)
    return out


def _num(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    return value if np.isfinite(value) else None


def _fmt(value, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(value):
        return "n/a"
    return f"{value:,.{digits}f}" if abs(value) >= 1000 else f"{value:.{digits}f}"
