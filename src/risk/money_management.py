"""Capital allocation: Kelly sizing with practical constraints.

The Kelly fraction is estimated from a strategy's closed-trade returns in
two classic forms:

* **discrete Kelly** — from win probability and payoff ratio:
  ``f* = p - (1 - p) / b`` where ``b`` = avg win / avg loss;
* **continuous Kelly** — from the first two moments of trade returns:
  ``f* = mean / variance`` (optimal growth for small returns).

Raw Kelly is notoriously aggressive and estimation-noisy, so the module
exposes fractional variants (half/quarter/custom) plus two hard caps:

* ``max_fraction`` — ceiling on the share of capital in one position;
* ``risk_per_trade`` — portfolio risk limit: the fraction is reduced so a
  tail loss (``loss_quantile`` of historical trade losses) costs no more
  than the given share of equity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KELLY_VARIANTS = {"classical": 1.0, "half": 0.5, "quarter": 0.25}


@dataclass(frozen=True)
class TradeStats:
    """Moments of a closed-trade return sample (returns as fractions)."""

    n_trades: int
    expectancy: float      # mean trade return
    variance: float        # variance of trade returns
    win_prob: float
    avg_win: float         # mean positive return
    avg_loss: float        # mean |negative return|
    payoff_ratio: float    # avg_win / avg_loss
    tail_loss: float       # |loss_quantile| percentile of losing trades

    @classmethod
    def from_returns(cls, returns, *, loss_quantile: float = 0.95) -> TradeStats:
        r = np.asarray(pd.Series(returns).dropna(), dtype=float)
        if len(r) < 2:
            raise ValueError("Need at least 2 trades to estimate statistics")
        wins, losses = r[r > 0], r[r < 0]
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(-losses.mean()) if len(losses) else 0.0
        tail_loss = float(-np.quantile(losses, 1 - loss_quantile)) if len(losses) else 0.0
        return cls(
            n_trades=len(r),
            expectancy=float(r.mean()),
            variance=float(r.var(ddof=1)),
            win_prob=float((r > 0).mean()),
            avg_win=avg_win,
            avg_loss=avg_loss,
            payoff_ratio=avg_win / avg_loss if avg_loss > 0 else np.inf,
            tail_loss=tail_loss,
        )

    def as_series(self) -> pd.Series:
        return pd.Series(vars(self))


def kelly_discrete(stats: TradeStats) -> float:
    """Kelly from win probability and payoff ratio, floored at 0."""
    if not np.isfinite(stats.payoff_ratio) or stats.payoff_ratio <= 0:
        return 1.0 if stats.win_prob == 1.0 else 0.0
    f = stats.win_prob - (1.0 - stats.win_prob) / stats.payoff_ratio
    return max(f, 0.0)


def kelly_continuous(stats: TradeStats) -> float:
    """Kelly from mean/variance of trade returns, floored at 0."""
    if stats.variance <= 0:
        return 0.0
    return max(stats.expectancy / stats.variance, 0.0)


@dataclass(frozen=True)
class SizingDecision:
    """A position fraction with the full derivation trail."""

    variant: str
    kelly_full: float       # raw Kelly estimate the variant scales
    fraction_kelly: float   # after the variant multiplier
    cap_max_fraction: float
    cap_risk: float         # fraction allowed by the portfolio risk limit
    fraction: float         # final: min of the three
    binding: str            # which constraint decided the final value

    def as_series(self) -> pd.Series:
        return pd.Series(vars(self))


def position_fraction(
    stats: TradeStats,
    *,
    variant: str | float = "half",
    basis: str = "continuous",
    max_fraction: float = 0.25,
    risk_per_trade: float = 0.02,
) -> SizingDecision:
    """Kelly-based position fraction under practical constraints.

    Parameters
    ----------
    variant:
        ``"classical"`` / ``"half"`` / ``"quarter"`` or a custom multiplier.
    basis:
        ``"continuous"`` (mean/variance) or ``"discrete"`` (p and payoff).
    max_fraction:
        Hard ceiling on the share of capital in one position.
    risk_per_trade:
        Portfolio risk limit: fraction * tail_loss must not exceed this.
    """
    if isinstance(variant, str):
        if variant not in KELLY_VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; use {sorted(KELLY_VARIANTS)} or a float")
        multiplier, label = KELLY_VARIANTS[variant], variant
    else:
        multiplier, label = float(variant), f"kelly x{variant}"
        if not 0 < multiplier <= 1:
            raise ValueError("Custom Kelly multiplier must be in (0, 1]")

    kelly_full = kelly_continuous(stats) if basis == "continuous" else kelly_discrete(stats)
    fraction_kelly = kelly_full * multiplier
    cap_risk = risk_per_trade / stats.tail_loss if stats.tail_loss > 0 else np.inf

    candidates = {
        "kelly": fraction_kelly,
        "max_fraction": max_fraction,
        "risk_limit": cap_risk,
    }
    binding = min(candidates, key=candidates.get)
    return SizingDecision(
        variant=label,
        kelly_full=kelly_full,
        fraction_kelly=fraction_kelly,
        cap_max_fraction=max_fraction,
        cap_risk=cap_risk,
        fraction=max(candidates[binding], 0.0),
        binding=binding,
    )


def sizing_menu(
    stats: TradeStats,
    *,
    basis: str = "continuous",
    max_fraction: float = 0.25,
    risk_per_trade: float = 0.02,
) -> pd.DataFrame:
    """Comparison table of all Kelly variants under the same constraints."""
    rows = [
        position_fraction(
            stats, variant=v, basis=basis, max_fraction=max_fraction, risk_per_trade=risk_per_trade
        ).as_series()
        for v in ("classical", "half", "quarter")
    ]
    return pd.DataFrame(rows).set_index("variant")
