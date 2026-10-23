"""The standard research report.

One fixed structure, 23 numbered sections, generated from the result
objects rather than written by hand. Two reasons it is fixed:

* A report whose shape is chosen per result is a report whose shape is
  chosen to suit the result. The same sections in the same order every time
  make an omission visible — a missing walk-forward section is a missing
  walk-forward, not a stylistic choice.
* It makes rejected hypotheses reportable. A rejection fills in the same
  twenty-three sections as a success, which is what makes the registry's
  record of rejections usable later instead of being a one-line note.

Section 21, "What could make this result false?", is the one that matters
and the one that cannot be written honestly by the person who wants the
result to be true. It is therefore generated from the numbers: every
weakness the data itself reveals is listed automatically, whether or not
the author would have chosen to mention it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from exchange.core.hashing import short
from exchange.core.text import markdown_table
from exchange.falsification.engine import RobustnessReport
from exchange.funnel.pipeline import FunnelResult

SECTIONS: tuple[str, ...] = (
    "Summary and verdict",
    "Pre-registered hypothesis",
    "Pre-registered falsification criteria",
    "Data provenance and quality",
    "Instrument and contract specification",
    "Execution assumptions and cost model",
    "Search space and the multiple-testing budget",
    "In-sample search",
    "Parameter stability",
    "Out-of-sample evaluation",
    "Purged and combinatorial cross-validation",
    "Walk-forward analysis",
    "Bootstrap and Monte Carlo",
    "Deflated Sharpe ratio and multiple-testing adjustment",
    "Probability of backtest overfitting",
    "Execution stress ladder",
    "Regime analysis",
    "Falsification attacks",
    "Structural diagnostics",
    "Verdict gates",
    "What could make this result false?",
    "Reproduction",
    "Environment and limitations",
)


def research_report(
    result: FunnelResult,
    *,
    robustness: RobustnessReport | None = None,
    scan=None,
    experiment_id: str = "",
    prior_trials: int | None = None,
    environment: Mapping | None = None,
    max_table_rows: int = 25,
) -> str:
    """Render the full report for one funnel result as Markdown."""
    h = result.hypothesis
    m = result.metrics
    verdict = result.verdict
    parts: list[str] = []
    n = _counter()

    # ---------------------------------------------------------------- title
    title = f"# Research report — `{h.hypothesis_id}`"
    if experiment_id:
        title += f" (experiment `{experiment_id}`)"
    parts += [
        title,
        "",
        f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} · "
        f"dataset `{short(str(result.dataset_summary.get('data_hash', '')))}` · "
        f"hypothesis fingerprint `{short(h.fingerprint)}`",
        "",
    ]

    # ------------------------------------------------------------------- 1
    parts += [
        f"## {n()}. Summary and verdict",
        "",
        verdict.to_markdown() if verdict else "_The funnel did not complete._",
        "",
        _headline_table(result, robustness),
        "",
    ]

    # ------------------------------------------------------------------- 2
    parts += [f"## {n()}. Pre-registered hypothesis", "", _strip_heading(h.to_markdown()), ""]

    # ------------------------------------------------------------------- 3
    parts += [f"## {n()}. Pre-registered falsification criteria", ""]
    rows = ["| criterion | threshold | observed | fired |", "|---|---|---|---|"]
    for criterion in h.falsification:
        value = _value(m.get(criterion.metric))
        fired = criterion.is_falsified(value)
        mark = {True: "**YES**", False: "no", None: "not evaluated"}[fired]
        rows.append(
            f"| `{criterion.metric}` | {criterion.operator} {criterion.threshold:g} | "
            f"{_fmt(value)} | {mark} |"
        )
    parts += [
        *rows,
        "",
        "These were written before the backtest ran and are hashed into the hypothesis "
        "fingerprint above, so they cannot have been adjusted to the result.",
        "",
    ]

    # ------------------------------------------------------------------- 4
    summary = result.dataset_summary
    parts += [
        f"## {n()}. Data provenance and quality",
        "",
        _definition_list(
            {
                "Symbol": summary.get("symbol"),
                "Timeframe": summary.get("timeframe"),
                "Period": f"{summary.get('start')} to {summary.get('end')}",
                "Bars": summary.get("n_bars"),
                "Empirical bars per year": summary.get("bars_per_year"),
                "Content hash": f"`{summary.get('data_hash')}`",
                "Quality gate": summary.get("quality"),
                "Quality warnings": summary.get("warnings"),
            }
        ),
        "",
    ]
    provenance = summary.get("provenance") or {}
    if provenance:
        parts += ["Provenance:", "", "```", _pretty(provenance), "```", ""]
    else:
        parts += [
            "_No provenance recorded for this dataset._ A series without provenance cannot be "
            "audited later; this is a defect in the input, not a presentational one.",
            "",
        ]

    # ------------------------------------------------------------------- 5
    config = getattr(result.oos_result, "config", None)
    instrument = config.instrument.to_dict() if config is not None else {}
    parts += [
        f"## {n()}. Instrument and contract specification",
        "",
        _definition_list(instrument) if instrument else "_Not recorded._",
        "",
        "The multiplier converts price into money. A result computed with the wrong multiplier "
        "is not wrong by a scale factor — it changes every cost ratio and therefore every "
        "conclusion.",
        "",
    ]

    # ------------------------------------------------------------------- 6
    parts += [f"## {n()}. Execution assumptions and cost model", ""]
    if config is not None:
        execution = config.execution
        parts += [
            _definition_list(
                {
                    "Scenario": execution.label,
                    "Fill timing": execution.fill_timing,
                    "Latency (bars)": execution.latency_bars,
                    "Effective execution delay (bars)": execution.execution_delay,
                    "Partial fills": execution.allow_partial_fills,
                    "Admits lookahead": execution.admits_lookahead,
                }
            ),
            "",
            "```",
            execution.costs.summary(),
            "```",
            "",
            f"Modelled costs out of sample: **{_fmt(m.get('oos_cost_drag_pct'), 2)}%** of "
            f"initial capital, consuming **{_fmt_pct(cost_share_of_gross(result))}** of gross "
            "profit before costs.",
            "",
        ]
    else:
        parts += ["_Not recorded._", ""]

    # ------------------------------------------------------------------- 7
    charged = h.total_trials if prior_trials is None else h.n_configurations + prior_trials
    parts += [
        f"## {n()}. Search space and the multiple-testing budget",
        "",
        _definition_list(
            {
                "Strategy": f"`{h.strategy}`",
                "Configurations in this hypothesis": h.n_configurations,
                "Prior configurations on this dataset": (
                    h.prior_hypotheses_on_data if prior_trials is None else prior_trials
                ),
                "Trials charged to the deflation": charged,
                "Expected best Sharpe from noise alone": _fmt(
                    float(np.sqrt(2.0 * np.log(max(charged, 2))))
                ),
            }
        ),
        "",
        "The last row is the Sharpe the *best* of this many independent random strategies would "
        "show by chance. A result below it is not evidence of anything.",
        "",
    ]

    # ------------------------------------------------------------------- 8
    parts += [f"## {n()}. In-sample search", ""]
    parts += [
        _definition_list(
            {
                "In-sample Sharpe (selected configuration)": _fmt(m.get("is_sharpe")),
                "In-sample return": _fmt_pct(m.get("is_return")),
                "In-sample trades": _fmt(m.get("n_is_trades"), 0),
                "Selected parameters": f"`{result.best_params}`",
            }
        ),
        "",
        "The sweep below contains **no out-of-sample column**, by construction: selection saw "
        "the in-sample window only.",
        "",
        _table(result.sweep, max_table_rows),
        "",
    ]

    # ------------------------------------------------------------------- 9
    parts += [
        f"## {n()}. Parameter stability",
        "",
        _stage_block(result, "parameter_stability"),
        _definition_list(
            {
                "Share of the grid positive in sample": _fmt_pct(m.get("plateau_fraction")),
                "Peak isolation": _fmt(m.get("peak_isolation")),
                "Top-quartile median Sharpe": _fmt(m.get("top_quartile_median_sharpe")),
            }
        ),
        "",
    ]

    # ------------------------------------------------------------------ 10
    parts += [
        f"## {n()}. Out-of-sample evaluation",
        "",
        "Evaluated **once**, with the parameters the in-sample search selected.",
        "",
        _definition_list(
            {
                "Out-of-sample Sharpe": _fmt(m.get("oos_sharpe")),
                "Out-of-sample return": _fmt_pct(m.get("oos_return")),
                "Maximum drawdown": _fmt_pct(m.get("oos_max_drawdown")),
                "Out-of-sample trades": _fmt(m.get("n_oos_trades"), 0),
                "In-sample to out-of-sample Sharpe ratio": _fmt(m.get("is_oos_sharpe_ratio")),
                "Embargo between windows (bars)": result.config.embargo_bars,
            }
        ),
        "",
    ]

    # ------------------------------------------------------------ 11 to 17
    parts += [
        f"## {n()}. Purged and combinatorial cross-validation",
        "",
        _stage_block(result, "cpcv"),
        _table(result.cpcv_table, max_table_rows),
        "",
        f"## {n()}. Walk-forward analysis",
        "",
        _stage_block(result, "walk_forward"),
        _table(result.wf_table, max_table_rows),
        "",
        f"## {n()}. Bootstrap and Monte Carlo",
        "",
        _stage_block(result, "monte_carlo"),
        _definition_list(
            {
                "95% bootstrap interval (per-bar Sharpe)": (
                    f"[{_fmt(m.get('bootstrap_sharpe_ci_low'), 5)}, "
                    f"{_fmt(m.get('bootstrap_sharpe_ci_high'), 5)}]"
                ),
                "Bootstrap p-value (Sharpe > 0)": _fmt(m.get("bootstrap_p_value"), 4),
                "Monte Carlo probability of a loss": _fmt_pct(m.get("mc_prob_loss")),
                "Monte Carlo median maximum drawdown": _fmt_pct(m.get("mc_median_max_drawdown")),
            }
        ),
        "",
        f"## {n()}. Deflated Sharpe ratio and multiple-testing adjustment",
        "",
        _stage_block(result, "multiple_testing"),
        _definition_list(
            {
                "Probabilistic Sharpe ratio": _fmt(m.get("probabilistic_sharpe_ratio"), 4),
                "Deflated Sharpe ratio": _fmt(m.get("deflated_sharpe_ratio"), 4),
                "Trials deflated for": charged,
                "Survivors after Holm-FWER": _fmt(m.get("fwer_survivors"), 0),
                "Expected false discoveries at this alpha": _fmt(
                    m.get("expected_false_discoveries"), 2
                ),
            }
        ),
        "",
        f"## {n()}. Probability of backtest overfitting",
        "",
        _definition_list(
            {
                "PBO": _fmt(m.get("pbo"), 4),
                "Reading": _pbo_reading(m.get("pbo")),
            }
        ),
        "",
        f"## {n()}. Execution stress ladder",
        "",
        _stage_block(result, "execution_stress"),
        _table(result.stress_table, max_table_rows),
        "",
        _cost_monotonicity_note(m),
        "",
        f"## {n()}. Regime analysis",
        "",
        _stage_block(result, "regime"),
        _table(result.regime_table, max_table_rows),
        "",
    ]

    # ------------------------------------------------------------------ 18
    parts += [f"## {n()}. Falsification attacks", ""]
    if robustness is not None:
        parts += [_strip_heading(robustness.to_markdown()), ""]
    else:
        parts += [
            "_Not run._ A result that has not been attacked has not been tested; the verdict "
            "above is therefore an upper bound.",
            "",
        ]

    # ------------------------------------------------------------------ 19
    parts += [f"## {n()}. Structural diagnostics", ""]
    if scan is not None:
        parts += [_strip_heading(scan.to_markdown()), ""]
    else:
        parts += [
            "_No structural scan supplied._ The scanner answers a prior question — whether the "
            "series shows any exploitable structure at all — and a strategy result without it "
            "cannot say whether the effect was predicted or discovered.",
            "",
        ]

    # ------------------------------------------------------------------ 20
    parts += [f"## {n()}. Verdict gates", ""]
    if verdict:
        parts += ["| gate | result | detail |", "|---|---|---|"]
        for gate in verdict.gates:
            mark = {True: "pass", False: "**FAIL**", None: "not evaluated"}[gate.passed]
            parts.append(f"| {gate.name} | {mark} | {gate.detail} |")
        parts += [
            "",
            "The gates are conjunctive. No single metric, however large, can raise a verdict "
            "past a failed gate.",
            "",
        ]

    # ------------------------------------------------------------------ 21
    parts += [
        f"## {n()}. What could make this result false?",
        "",
        "Generated from the numbers above, not from the author's judgement.",
        "",
    ]
    threats = falsity_threats(result, robustness=robustness, prior_trials=prior_trials)
    if threats:
        parts += [f"{i}. {t}" for i, t in enumerate(threats, start=1)]
    else:
        parts.append(
            "No automatic threat was detected. That is not a clean bill of health: the checks "
            "below cover the failure modes this framework knows how to measure, and the "
            "important ones are usually the modes nobody has measured yet."
        )
    parts.append("")

    # ------------------------------------------------------------------ 22
    parts += [
        f"## {n()}. Reproduction",
        "",
        (
            f"```\nexchange reproduce {experiment_id}\n```\n"
            if experiment_id
            else "_Not yet recorded in the experiment registry._\n"
        ),
        _definition_list(
            {
                "Seed": result.config.seed,
                "Dataset hash": f"`{result.dataset_summary.get('data_hash')}`",
                "Hypothesis fingerprint": f"`{h.fingerprint}`",
            }
        ),
        "",
    ]

    # ------------------------------------------------------------------ 23
    parts += [f"## {n()}. Environment and limitations", ""]
    if environment:
        parts += [_definition_list(_environment_rows(environment)), ""]
    parts += [
        "Known limitations of this report:",
        "",
        "- Costs are modelled, not measured. Every number depends on the cost model in section 6.",
        "- A single instrument's history is one realisation. Surviving here is not evidence about "
        "any other market.",
        "- The statistical tests assume the return series' dependence is captured by the block "
        "bootstrap's block length. A regime change longer than the sample cannot be detected "
        "from inside it.",
        "- Out-of-sample means out-of-sample *for the selection step*. Every result in this "
        "programme has now been computed on this data, so the next hypothesis on it starts with "
        "the trials in section 7 already spent.",
        "",
    ]

    return "\n".join(parts)


# ==========================================================================
# Section 21: generated threats
# ==========================================================================


def cost_share_of_gross(result: FunnelResult) -> float | None:
    """Fraction of gross out-of-sample profit consumed by modelled costs.

    The engine's own ``cost_drag_pct`` is costs as a share of *initial
    capital*, which says how expensive the trading was but not how much of
    the edge it ate. This ratio is the one that decides whether a result is
    a market statement or a cost-model statement.
    """
    oos = result.oos_result
    if oos is None or len(oos.equity) < 2:
        return None
    costs = float(oos.total_costs)
    net = float(oos.equity.iloc[-1] - oos.equity.iloc[0])
    gross = net + costs
    if not np.isfinite(gross) or gross <= 0:
        return None
    return costs / gross


def falsity_threats(
    result: FunnelResult,
    *,
    robustness: RobustnessReport | None = None,
    prior_trials: int | None = None,
) -> list[str]:
    """Everything in the result that could make it false, read off the numbers.

    Deliberately pessimistic: each item is phrased as the mechanism by
    which the result could be an artefact, and the thresholds are set to
    fire on conditions a careful reviewer would raise, not only on failures.
    """
    m = result.metrics
    h = result.hypothesis
    out: list[str] = []

    def num(key):
        return _value(m.get(key))

    n_trades = num("n_oos_trades")
    if n_trades is not None and n_trades < 100:
        out.append(
            f"**Sample size.** {n_trades:.0f} out-of-sample trades. The standard error of a "
            f"Sharpe estimate from this many round trips is roughly "
            f"{1.0 / max(np.sqrt(n_trades), 1):.2f} in Sharpe units, so the point estimate is "
            "compatible with a much weaker effect, and with none."
        )

    share = cost_share_of_gross(result)
    if share is not None and share > 0.3:
        out.append(
            f"**Cost dependence.** Modelled costs consume {share:.0%} of gross profit, so the "
            "result is as much a statement about the cost model as about the market. Costs "
            f"{1.0 / max(share, 1e-9):.1f}x the modelled level would remove the edge entirely, "
            "and the cost model is an estimate."
        )

    dsr = num("deflated_sharpe_ratio")
    if dsr is not None and dsr < 0.95:
        out.append(
            f"**Selection.** The deflated Sharpe ratio is {dsr:.3f} after charging "
            f"{h.total_trials if prior_trials is None else h.n_configurations + prior_trials} "
            "trials. That is the probability the true Sharpe is positive *given* the size of the "
            "search; a value below 0.95 means a non-trivial chance the result is the best of "
            "many draws from noise."
        )

    pbo = num("pbo")
    if pbo is not None and pbo > 0.2:
        out.append(
            f"**Overfitting of the selection rule.** PBO is {pbo:.2f}: in {pbo:.0%} of the "
            "combinatorial splits, the configuration chosen in-sample underperformed the median "
            "out-of-sample. The selection procedure itself is only weakly informative."
        )

    plateau = num("plateau_fraction")
    if plateau is not None and plateau < 0.5:
        out.append(
            f"**Narrow parameter region.** Only {plateau:.0%} of the grid is profitable "
            "in-sample. A thin region is what a fitted artefact looks like; a real effect "
            "usually works across a wide band of parameters."
        )

    best_year = num("best_year_return_share")
    if best_year is not None and best_year > 0.5:
        out.append(
            f"**Concentration in time.** The best single year contributes {best_year:.0%} of the "
            "total return. The effect may be one regime rather than a persistent property."
        )

    regimes = num("positive_regime_fraction")
    if regimes is not None and regimes < 1.0:
        out.append(
            f"**Regime dependence.** Only {regimes:.0%} of volatility regimes are profitable. If "
            "the profitable regime does not recur, neither does the edge."
        )

    stressed = num("stressed_sharpe")
    if stressed is not None and stressed <= 0:
        out.append(
            f"**Execution fragility.** Under the stressed scenario the Sharpe is {stressed:.2f}. "
            "A bad fill environment — the one a crowded trade produces — removes the result."
        )

    if m.get("cost_decay_monotone") is False:
        out.append(
            "**Accounting.** Performance does not fall monotonically as costs rise. This usually "
            "means a sign error in the cost or position accounting, and until it is explained "
            "every number here is suspect."
        )

    warnings = result.dataset_summary.get("warnings")
    if warnings:
        out.append(
            f"**Data quality.** The dataset carries {warnings} quality warning(s). Gaps, stale "
            "bars and bad prints create apparent predictability that no statistical test can "
            "distinguish from a market effect."
        )

    if not result.dataset_summary.get("provenance"):
        out.append(
            "**Unverifiable data.** No provenance was recorded, so the series cannot be "
            "re-derived from its source. An unreproducible input makes the result "
            "unreproducible whatever the code does."
        )

    adjustment = (result.dataset_summary.get("provenance") or {}).get("adjustment")
    if adjustment and adjustment != "none":
        out.append(
            f"**Splice artefact.** The series is back-adjusted (`{adjustment}`). Adjustment "
            "moves historical prices, and trend-like results on adjusted futures series are "
            "frequently artefacts of the adjustment rather than of the market. The "
            "`roll_methodology` attack is the direct test."
        )

    if robustness is None:
        out.append(
            "**Untested.** The falsification attacks were not run, so nothing here has been "
            "challenged beyond the standard validation sequence."
        )
    else:
        for outcome in robustness.failed:
            out.append(
                f"**Failed attack `{outcome.name}`.** {outcome.question} Answer: no — "
                f"{outcome.detail}."
            )
        skipped = [o.name for o in robustness.skipped]
        if skipped:
            out.append(
                "**Attacks not evaluated:** "
                + ", ".join(f"`{s}`" for s in skipped)
                + ". A probe that did not run is missing evidence, not a pass."
            )

    bars = result.dataset_summary.get("n_bars") or 0
    ppy = result.dataset_summary.get("bars_per_year") or 0
    if bars and ppy and bars / ppy < 5:
        out.append(
            f"**Short history.** {bars / ppy:.1f} years of data. Commodity markets change "
            "structurally on a multi-year cycle, so a sample this short cannot contain more than "
            "one or two regimes."
        )

    return out


# ==========================================================================
# Formatting helpers
# ==========================================================================


def _counter():
    state = {"i": 0}

    def nxt() -> int:
        state["i"] += 1
        return state["i"]

    return nxt


def _headline_table(result: FunnelResult, robustness: RobustnessReport | None) -> str:
    m = result.metrics
    rows = {
        "Out-of-sample Sharpe": _fmt(m.get("oos_sharpe")),
        "Deflated Sharpe ratio": _fmt(m.get("deflated_sharpe_ratio"), 4),
        "PBO": _fmt(m.get("pbo"), 3),
        "CPCV median Sharpe": _fmt(m.get("cpcv_median_sharpe")),
        "Walk-forward efficiency": _fmt(m.get("wf_efficiency")),
        "Conservative-cost Sharpe": _fmt(m.get("conservative_sharpe")),
        "Out-of-sample trades": _fmt(m.get("n_oos_trades"), 0),
    }
    if robustness is not None:
        rows["Robustness score"] = (
            f"{robustness.score:.2f} ({len(robustness.survived)}/{len(robustness.evaluated)}) "
            f"— {robustness.grade}"
        )
    lines = ["| headline metric | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows.items()]
    return "\n".join(lines)


def _stage_block(result: FunnelResult, name: str) -> str:
    stage = result.stage(name)
    if stage is None:
        return "_Stage not run._\n"
    mark = "passed" if stage.passed else "**did not pass**"
    return f"Stage {mark}: {stage.summary}\n"


def _definition_list(mapping: Mapping) -> str:
    return "\n".join(f"- **{k}:** {v}" for k, v in mapping.items() if v is not None)


def _table(frame: pd.DataFrame, max_rows: int) -> str:
    return markdown_table(frame, max_rows=max_rows)


def _environment_rows(environment: Mapping) -> dict:
    git = environment.get("git") or {}
    rows = {
        "Python": environment.get("python_version"),
        "Platform": environment.get("platform"),
        "Machine": environment.get("machine"),
    }
    for package, version in (environment.get("packages") or {}).items():
        rows[package] = version
    if git.get("commit"):
        rows["Git commit"] = f"`{git['commit']}`" + (" (dirty)" if git.get("dirty") else "")
    return rows


def _pbo_reading(value) -> str:
    value = _value(value)
    if value is None:
        return "not computed"
    if value <= 0.2:
        return "the selection procedure is informative"
    if value <= 0.5:
        return "the selection procedure is weakly informative; treat the choice as provisional"
    return (
        "the selection procedure is worse than a coin flip — no configuration drawn from this "
        "search can be trusted"
    )


def _cost_monotonicity_note(metrics: Mapping) -> str:
    if metrics.get("cost_decay_monotone") is True:
        return (
            "Performance falls monotonically as costs rise, which is the minimum consistency "
            "check on the cost accounting."
        )
    if metrics.get("cost_decay_monotone") is False:
        return (
            "**Performance does not fall monotonically as costs rise.** This is not robustness; "
            "it almost always indicates a sign error in the cost or position accounting."
        )
    return ""


def _strip_heading(markdown: str) -> str:
    """Drop a leading ``#``/``##`` title so nested blocks keep one hierarchy."""
    lines = markdown.splitlines()
    if lines and lines[0].lstrip().startswith("#"):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines)


def _pretty(value, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, Mapping):
        return "\n".join(
            f"{pad}{k}:" + ("\n" + _pretty(v, indent + 2) if isinstance(v, Mapping | list) else f" {v}")
            for k, v in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, str):
        return "\n".join(f"{pad}- {v}" for v in value)
    return f"{pad}{value}"


def _value(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _fmt(value, digits: int = 3) -> str:
    value = _value(value)
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_pct(value) -> str:
    value = _value(value)
    return "n/a" if value is None else f"{value:.1%}"
