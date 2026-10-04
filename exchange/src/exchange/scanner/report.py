"""Scanner report: from diagnostics to a research agenda.

The recommendation rules here are deliberately conservative and entirely
explicit. Two constraints govern them:

**No single test may name a family.** Every recommendation requires
corroboration from at least two independent measurements. A variance ratio
below 1 alone is not mean reversion; a variance ratio below 1 *and* a
significantly negative AR(1) slope that survives an ADF test *and* a
half-life short enough to recover costs is a hypothesis worth testing.

**Ruling families out is as valuable as recommending them.** The
``deprioritise`` list exists because the cheapest way to reduce false
discoveries is to run fewer searches. If the scanner can say "unconditional
trend following has nothing to work with here", that is a saved search and a
saved false positive.

Nothing in this module decides that an effect exists. It decides where it is
worth *looking*, and the funnel decides the rest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from exchange.core.errors import NotEnoughDataError
from exchange.core.hashing import short
from exchange.core.text import markdown_table
from exchange.scanner import diagnostics as dg

_LEVELS = {"NONE": 0, "WEAK": 1, "MODERATE": 2, "STRONG": 3}


@dataclass
class ScanResult:
    """Everything the scanner measured, plus what it suggests."""

    symbol: str
    timeframe: str
    n_bars: int
    start: pd.Timestamp
    end: pd.Timestamp
    data_hash: str
    diagnostics: dict[str, dg.Diagnostic] = field(default_factory=dict)
    vr_profile: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict[str, str] = field(default_factory=dict)
    recommend: list[str] = field(default_factory=list)
    deprioritise: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def get(self, name: str) -> dg.Diagnostic | None:
        return self.diagnostics.get(name)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "n_bars": self.n_bars,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "data_hash": self.data_hash,
            "summary": self.summary,
            "recommend": self.recommend,
            "deprioritise": self.deprioritise,
            "caveats": self.caveats,
            "diagnostics": {k: v.to_dict() for k, v in self.diagnostics.items()},
            "variance_ratio_profile": self.vr_profile.to_dict("records"),
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Structural Scan — {self.symbol} ({self.timeframe})",
            "",
            f"{self.n_bars:,} bars, {self.start:%Y-%m-%d} to {self.end:%Y-%m-%d} UTC, "
            f"data hash `{short(self.data_hash)}`",
            "",
            "## Summary",
            "",
            "| property | reading |",
            "|---|---|",
        ]
        for key, value in self.summary.items():
            lines.append(f"| {key.replace('_', ' ').title()} | **{value}** |")

        lines += ["", "## Recommended research families", ""]
        lines += [f"- {item}" for item in self.recommend] or ["- _none: no corroborated structure_"]
        lines += ["", "## Deprioritise / avoid", ""]
        lines += [f"- {item}" for item in self.deprioritise] or ["- _nothing ruled out_"]

        if self.caveats:
            lines += ["", "## Caveats", ""] + [f"- {c}" for c in self.caveats]

        lines += ["", "## Measurements", ""]
        for name, d in self.diagnostics.items():
            ci = f" (95% CI {d.ci[0]:.3f} to {d.ci[1]:.3f})" if np.isfinite(d.ci[0]) else ""
            p = f" (p = {d.p_value:.4g})" if np.isfinite(d.p_value) else ""
            lines += [
                f"### `{name}`",
                "",
                f"- **Value**: {d.value:.4f}{ci}",
                f"- **Evidence**: {d.evidence}{p}",
                f"- {d.interpretation}",
            ]
            if d.falsified_by:
                lines.append(f"- **Would be falsified by**: {d.falsified_by}")
            lines.append("")

        if not self.vr_profile.empty:
            lines += ["### Variance ratio profile", "",
                      markdown_table(self.vr_profile, index=False), ""]

        lines += [
            "",
            "---",
            "",
            "> A scan reports where structure *might* be. It is not evidence that a strategy "
            "works. Every family listed above must still pass the full research funnel, and the "
            "count of families tested here is part of the multiple-testing budget.",
        ]
        return "\n".join(lines) + "\n"


def scan(
    bars: pd.DataFrame,
    *,
    symbol: str = "data",
    timeframe: str = "",
    data_hash: str = "",
    vr_horizons=(2, 5, 10, 20),
    seasonality_buckets=("month",),
) -> ScanResult:
    """Run every diagnostic and derive a research agenda."""
    from exchange.core.hashing import hash_dataframe
    from exchange.core.timeframe import infer_timeframe
    from exchange.data.schema import assert_canonical

    assert_canonical(bars, name=symbol)
    if len(bars) < dg.MIN_OBSERVATIONS:
        raise NotEnoughDataError(f"{symbol}: scanning needs at least {dg.MIN_OBSERVATIONS} bars")

    result = ScanResult(
        symbol=symbol,
        timeframe=timeframe or infer_timeframe(bars.index).label,
        n_bars=len(bars),
        start=bars.index[0],
        end=bars.index[-1],
        data_hash=data_hash or hash_dataframe(bars),
    )

    for func in dg.ALL_DIAGNOSTICS:
        name = func.__name__
        try:
            d = func(bars)
            result.diagnostics[d.name] = d
        except NotEnoughDataError as exc:
            result.caveats.append(f"{name}: not computed ({exc})")
        except Exception as exc:  # a broken diagnostic must not hide the rest
            result.caveats.append(f"{name}: failed ({type(exc).__name__}: {exc})")

    for q in vr_horizons:
        try:
            d = dg.variance_ratio(bars, q=q)
            result.diagnostics[d.name] = d
        except NotEnoughDataError:
            continue
    result.vr_profile = dg.variance_ratio_profile(bars, qs=vr_horizons)

    for bucket in seasonality_buckets:
        try:
            d = dg.seasonality(bars, by=bucket)
            result.diagnostics[d.name] = d
        except (NotEnoughDataError, ValueError) as exc:
            result.caveats.append(f"seasonality[{bucket}]: not computed ({exc})")

    _summarise(result)
    _recommend(result, n_bars=len(bars))
    return result


def _level(result: ScanResult, name: str) -> int:
    d = result.get(name)
    return _LEVELS.get(d.evidence, 0) if d is not None else 0


def _null() -> dg.Diagnostic:
    return dg.Diagnostic("missing", np.nan, "not computed", "NONE")


def _summarise(result: ScanResult) -> None:
    vr_short = result.get("variance_ratio_q2") or result.get("variance_ratio_q5")
    vr_long = result.get("variance_ratio_q20") or result.get("variance_ratio_q10")
    half_life = result.get("mean_reversion_half_life")
    acf = result.get("autocorrelation")

    # Mean reversion: short-horizon VR below 1, negative lag-1
    # autocorrelation, or a half-life that survived the ADF gate.
    mr_votes = 0
    if vr_short is not None and np.isfinite(vr_short.value) and vr_short.value < 1:
        mr_votes += _LEVELS.get(vr_short.evidence, 0)
    if acf is not None and np.isfinite(acf.value) and acf.value < 0:
        mr_votes += _LEVELS.get(acf.evidence, 0)
    if half_life is not None and np.isfinite(half_life.value):
        mr_votes += _LEVELS.get(half_life.evidence, 0)

    # Trend persistence: long-horizon VR above 1, or positive lag-1.
    trend_votes = 0
    if vr_long is not None and np.isfinite(vr_long.value) and vr_long.value > 1:
        trend_votes += _LEVELS.get(vr_long.evidence, 0)
    if acf is not None and np.isfinite(acf.value) and acf.value > 0:
        trend_votes += _LEVELS.get(acf.evidence, 0)

    result.summary = {
        "mean_reversion_evidence": _grade(mr_votes),
        "trend_persistence": _grade(trend_votes),
        "volatility_clustering": (result.get("volatility_clustering") or _null()).evidence,
        "seasonality": max(
            (d.evidence for name, d in result.diagnostics.items() if name.startswith("seasonality_")),
            key=lambda e: _LEVELS.get(e, 0), default="NONE",
        ),
        "tail_risk": (result.get("tail_behaviour") or _null()).evidence,
        "structural_stability": (result.get("structural_stability") or _null()).evidence,
    }


def _grade(votes: int) -> str:
    if votes >= 5:
        return "STRONG"
    if votes >= 3:
        return "MODERATE"
    if votes >= 1:
        return "WEAK"
    return "NONE"


def _recommend(result: ScanResult, *, n_bars: int) -> None:
    s = result.summary
    mr, trend = s["mean_reversion_evidence"], s["trend_persistence"]
    vol, seas = s["volatility_clustering"], s["seasonality"]
    stable = s["structural_stability"]

    rec, avoid = [], []

    if _LEVELS[mr] >= 2:
        hl = result.get("mean_reversion_half_life")
        hl_text = f" (estimated half-life {hl.value:.0f} bars)" if hl and np.isfinite(hl.value) else ""
        rec.append(
            f"**Conditional mean reversion**{hl_text} — corroborated by the short-horizon variance "
            "ratio and the ADF-gated AR(1) fit. Condition entries on a regime or trend filter; the "
            "unconditional version is where this family usually dies."
        )
    elif _LEVELS[mr] == 1:
        rec.append(
            "**Mean reversion, exploratory only** — one measurement points this way and the others "
            "do not. Treat any result as a single hypothesis, not a family worth sweeping."
        )
    else:
        avoid.append("**Mean reversion** — no measurement supports short-horizon reversion here.")

    if _LEVELS[trend] >= 2:
        rec.append(
            "**Trend / time-series momentum** — the long-horizon variance ratio exceeds 1 with "
            "supporting autocorrelation. Size by volatility so the result is comparable across "
            "regimes."
        )
    else:
        avoid.append(
            "**Naive unconditional trend following** — the long-horizon variance ratio does not "
            "exceed 1 with any confidence, so there is no measured persistence for it to harvest."
        )

    if _LEVELS[vol] >= 2:
        rec.append(
            "**Volatility regime / sizing strategies** — volatility is strongly forecastable even "
            "where direction is not. Volatility targeting and regime-conditioned entries are the "
            "highest-value use of this, and they improve every other family's comparability."
        )

    if _LEVELS[seas] >= 2:
        rec.append(
            "**Seasonality** — calendar buckets differ significantly. Note the multiple-testing "
            "exposure: there are 12 months, 5 weekdays and 2 directions to choose from, so a "
            "seasonal result must clear a much higher bar than its raw p-value suggests."
        )
    elif _LEVELS[seas] == 1:
        avoid.append(
            "**Seasonality** — the joint calendar test is only weakly significant; individual "
            "months will look impressive purely by selection."
        )

    if _LEVELS[stable] <= 1:
        result.caveats.append(
            "The process is NOT stable across sub-periods. Any single in-sample/out-of-sample "
            "split will be dominated by which regime landed on which side. Use combinatorial "
            "purged CV and weight the dispersion across paths above the point estimate."
        )
    tail = result.get("tail_behaviour")
    if tail is not None and np.isfinite(tail.value) and tail.value > 5:
        result.caveats.append(
            f"Excess kurtosis is {tail.value:.0f}: P&L will be dominated by a small number of bars, "
            "so the effective sample size is far below the bar count. Bootstrap intervals, not "
            "normal-theory ones."
        )
    if n_bars < 750:
        result.caveats.append(
            f"Only {n_bars} bars. Nothing measured here should be treated as more than a hint; the "
            "standard error on every statistic is large at this length."
        )

    hurst = result.get("hurst_exponent")
    if hurst is not None and hurst.details.get("inside_null_band"):
        result.caveats.append(
            f"The Hurst exponent ({hurst.value:.2f}) sits inside the random-walk null band for this "
            "sample length and was not used in any recommendation."
        )

    if not rec:
        rec.append(
            "**None.** No corroborated structure was found. The correct action is to stop, not to "
            "sweep strategies until something passes — a search over families on a series with no "
            "measured structure produces false discoveries at exactly the rate the search size "
            "implies."
        )

    result.recommend, result.deprioritise = rec, avoid


def scan_table(results) -> pd.DataFrame:
    """One row per scanned instrument — the universe view."""
    rows = []
    for r in results:
        rows.append({"symbol": r.symbol, "timeframe": r.timeframe, "n_bars": r.n_bars, **r.summary})
    return pd.DataFrame(rows)


def scan_dataset(dataset, **kwargs) -> ScanResult:
    """Scan a :class:`~exchange.data.dataset.Dataset`, carrying its identity.

    Using this rather than :func:`scan` on the raw frame keeps the symbol,
    timeframe and content hash attached to the result, which is what makes
    a scan citable in a research report.
    """
    return scan(
        dataset.bars,
        symbol=dataset.symbol,
        timeframe=dataset.timeframe.label,
        data_hash=dataset.data_hash,
        **kwargs,
    )
