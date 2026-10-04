"""Verdicts.

Five levels, and the ordering is not a score
---------------------------------------------
``REJECTED``
    A pre-registered falsification criterion fired, or a hard gate failed.
    The hypothesis is dead on this data. This is a *result*, recorded with
    the same weight as any other.
``WEAK``
    Nothing decisively falsified it, but the evidence does not clear the bar
    for further work. Most honest research ends here.
``PROMISING``
    Survived every stage with a positive out-of-sample edge, but at least one
    robustness measure is marginal or could not be computed. Worth more data
    or a tighter hypothesis; not worth capital.
``ROBUST``
    Survived every stage including deflation, purged cross-validation,
    walk-forward and conservative execution costs.
``PAPER_TRADING_CANDIDATE``
    ROBUST, plus the additional requirements that only matter when real
    orders are about to be sent: survives *stressed* execution, has enough
    out-of-sample trades to monitor, and is not concentrated in one year.

What is deliberately impossible here
-------------------------------------
No single metric can produce a verdict above WEAK. The gates are
conjunctive: a Sharpe of 3.0 with a PBO of 0.7 is REJECTED, and no amount of
headline performance overrides a failed falsification criterion. This is the
opposite of a weighted score, and it is intentional — the audited project's
``robust_score`` was exactly such a weighted average, which let one
spectacular number compensate for a fatal one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

from exchange.funnel.hypothesis import Hypothesis


class VerdictLevel(IntEnum):
    REJECTED = 0
    WEAK = 1
    PROMISING = 2
    ROBUST = 3
    PAPER_TRADING_CANDIDATE = 4

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.name


@dataclass
class Gate:
    """One named condition a verdict level requires."""

    name: str
    passed: bool | None       # None = could not be evaluated
    detail: str

    def to_dict(self) -> dict:
        return {"gate": self.name, "passed": self.passed, "detail": self.detail}


@dataclass
class Verdict:
    """The funnel's conclusion, with every reason it reached it."""

    level: VerdictLevel
    headline: str
    falsified_by: list[str] = field(default_factory=list)
    gates: list[Gate] = field(default_factory=list)
    unmet_for_next_level: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.level.name

    @property
    def rejected(self) -> bool:
        return self.level is VerdictLevel.REJECTED

    def to_dict(self) -> dict:
        return {
            "level": int(self.level),
            "label": self.label,
            "headline": self.headline,
            "falsified_by": self.falsified_by,
            "gates": [g.to_dict() for g in self.gates],
            "unmet_for_next_level": self.unmet_for_next_level,
            "caveats": self.caveats,
        }

    def to_markdown(self) -> str:
        lines = [f"**Verdict: `{self.label}`** — {self.headline}", ""]
        if self.falsified_by:
            lines += ["Falsified by:", ""] + [f"- {r}" for r in self.falsified_by] + [""]
        lines += ["| gate | result | detail |", "|---|---|---|"]
        for g in self.gates:
            mark = {True: "pass", False: "**FAIL**", None: "n/a"}[g.passed]
            lines.append(f"| {g.name} | {mark} | {g.detail} |")
        if self.unmet_for_next_level:
            lines += ["", "Blocking a higher verdict:", ""] + [
                f"- {r}" for r in self.unmet_for_next_level
            ]
        if self.caveats:
            lines += ["", "Caveats:", ""] + [f"- {c}" for c in self.caveats]
        return "\n".join(lines) + "\n"


def _get(metrics: dict, key: str) -> float | None:
    value = metrics.get(key)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def decide_verdict(hypothesis: Hypothesis, metrics: dict, stages) -> Verdict:
    """Apply the pre-registered criteria, then the standard gates."""
    # ---- 1. The author's own falsification criteria come first. --------
    falsified: list[str] = []
    for criterion in hypothesis.falsification:
        value = _get(metrics, criterion.metric)
        if criterion.is_falsified(value):
            falsified.append(
                f"`{criterion.describe()}` — observed {criterion.metric} = {value:.4g}"
                + (f" ({criterion.rationale})" if criterion.rationale else "")
            )

    # ---- 2. Hard stage failures. --------------------------------------
    stage_map = {s.name: s for s in stages}
    if "sanity" in stage_map and not stage_map["sanity"].passed:
        return Verdict(
            level=VerdictLevel.REJECTED,
            headline=f"Failed sanity checks: {stage_map['sanity'].summary}",
            falsified_by=[stage_map["sanity"].summary],
        )

    # ---- 3. The standard gates. ---------------------------------------
    oos_sharpe = _get(metrics, "oos_sharpe")
    dsr = _get(metrics, "deflated_sharpe_ratio")
    pbo = _get(metrics, "pbo")
    cpcv_median = _get(metrics, "cpcv_median_sharpe")
    cpcv_positive = _get(metrics, "cpcv_positive_fraction")
    wf_eff = _get(metrics, "wf_efficiency")
    wf_profitable = _get(metrics, "wf_profitable_fraction")
    boot_lo = _get(metrics, "bootstrap_sharpe_ci_low")
    conservative = _get(metrics, "conservative_sharpe")
    stressed = _get(metrics, "stressed_sharpe")
    n_oos_trades = _get(metrics, "n_oos_trades")
    plateau = _get(metrics, "plateau_fraction")
    positive_regimes = _get(metrics, "positive_regime_fraction")
    best_year_share = _get(metrics, "best_year_return_share")

    gates = [
        Gate("out-of-sample edge", _ge(oos_sharpe, 0.0),
             f"OOS Sharpe {_fmt(oos_sharpe)} (needs > 0)"),
        Gate("deflated Sharpe", _ge(dsr, 0.90),
             f"DSR {_fmt(dsr)} after {hypothesis.total_trials} trial(s) (needs >= 0.90)"),
        Gate("selection is informative", _le(pbo, 0.5),
             f"PBO {_fmt(pbo)} (needs <= 0.50)"),
        Gate("purged cross-validation", _and(_ge(cpcv_median, 0.0), _ge(cpcv_positive, 0.6)),
             f"CPCV median Sharpe {_fmt(cpcv_median)}, {_fmt_pct(cpcv_positive)} of splits positive"),
        Gate("walk-forward", _and(_ge(wf_eff, 0.3), _ge(wf_profitable, 0.5)),
             f"efficiency {_fmt(wf_eff)}, {_fmt_pct(wf_profitable)} of windows profitable"),
        Gate("bootstrap interval excludes zero", _gt(boot_lo, 0.0),
             f"95% lower bound on per-period Sharpe {_fmt(boot_lo, 5)}"),
        Gate("survives conservative costs", _gt(conservative, 0.0),
             f"conservative-scenario Sharpe {_fmt(conservative)}"),
        Gate("enough out-of-sample trades", _ge(n_oos_trades, 30),
             f"{_fmt(n_oos_trades, 0)} OOS trades (needs >= 30)"),
        Gate("parameter plateau", _ge(plateau, 0.4),
             f"{_fmt_pct(plateau)} of the grid is positive in-sample"),
        Gate("works across regimes", _ge(positive_regimes, 0.5),
             f"{_fmt_pct(positive_regimes)} of volatility regimes positive"),
    ]

    if falsified:
        return Verdict(
            level=VerdictLevel.REJECTED,
            headline=(
                f"{len(falsified)} pre-registered falsification criterion/criteria fired. "
                "The hypothesis was rejected on the terms its author set in advance — a completed "
                "experiment, not a failed one."
            ),
            falsified_by=falsified,
            gates=gates,
        )

    failed = [g for g in gates if g.passed is False]
    unknown = [g for g in gates if g.passed is None]

    caveats = [f"{g.name}: not evaluated ({g.detail})" for g in unknown]
    if metrics.get("cost_decay_monotone") is False:
        caveats.append(
            "Performance does not decay monotonically as execution costs rise. This is not a "
            "feature; it usually indicates a sign error in the cost accounting."
        )

    # Any hard failure caps the verdict at WEAK (not REJECTED — nothing the
    # author pre-committed to has fired, so the honest reading is
    # "insufficient evidence", not "disproved").
    if failed:
        return Verdict(
            level=VerdictLevel.WEAK,
            headline=(
                f"Survived falsification but failed {len(failed)} standard gate(s): "
                + "; ".join(g.name for g in failed)
                + ". Not enough evidence to justify further work."
            ),
            gates=gates,
            unmet_for_next_level=[f"{g.name}: {g.detail}" for g in failed],
            caveats=caveats,
        )

    if unknown:
        return Verdict(
            level=VerdictLevel.PROMISING,
            headline=(
                f"Passed every gate that could be evaluated, but {len(unknown)} could not be "
                "computed on this sample. More data is needed before a robustness claim."
            ),
            gates=gates,
            unmet_for_next_level=[f"{g.name}: {g.detail}" for g in unknown],
            caveats=caveats,
        )

    # All gates passed. Distinguish ROBUST from PAPER_TRADING_CANDIDATE.
    extra_blocking = []
    if _gt(stressed, -0.5) is not True:
        extra_blocking.append(f"stressed-scenario Sharpe {_fmt(stressed)} (needs > -0.5)")
    if _ge(n_oos_trades, 100) is not True:
        extra_blocking.append(
            f"{_fmt(n_oos_trades, 0)} OOS trades (needs >= 100 to monitor a live book meaningfully)"
        )
    if best_year_share is not None and best_year_share > 0.7:
        extra_blocking.append(
            f"the best single year contributes {best_year_share:.0%} of the total return "
            "(needs <= 70%); the edge is concentrated, not persistent"
        )
    if _ge(dsr, 0.95) is not True:
        extra_blocking.append(f"DSR {_fmt(dsr)} (needs >= 0.95 before real orders)")

    if extra_blocking:
        return Verdict(
            level=VerdictLevel.ROBUST,
            headline=(
                "Survived every validation stage, including deflation for the size of the search, "
                "purged cross-validation and conservative execution costs."
            ),
            gates=gates,
            unmet_for_next_level=extra_blocking,
            caveats=caveats,
        )

    return Verdict(
        level=VerdictLevel.PAPER_TRADING_CANDIDATE,
        headline=(
            "Survived every validation stage and the additional requirements for live monitoring: "
            "stressed execution, trade count, and no concentration in a single year. The next step "
            "is paper trading, not capital."
        ),
        gates=gates,
        caveats=caveats,
    )


# -- small tri-state helpers (None propagates as "unknown") -----------------


def _ge(value, threshold):
    return None if value is None else bool(value >= threshold)


def _gt(value, threshold):
    return None if value is None else bool(value > threshold)


def _le(value, threshold):
    return None if value is None else bool(value <= threshold)


def _and(a, b):
    if a is None or b is None:
        return None
    return bool(a and b)


def _fmt(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_pct(value) -> str:
    return "n/a" if value is None else f"{value:.0%}"
