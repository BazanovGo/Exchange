"""Monte Carlo simulation of trade sequences.

Bootstrap-resamples a strategy's historical trade returns into thousands
of alternative equal-length sequences, compounds them at a given position
fraction, and measures what the single realized backtest path cannot show:
ruin probability, the distribution of terminal returns and of maximum
drawdowns, and how all of it degrades as the Kelly fraction grows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    """Distributions from one (fraction, n_sims) simulation batch."""

    fraction: float
    n_sims: int
    n_trades: int
    ruin_level: float
    final_returns: np.ndarray = field(repr=False)   # terminal return per path
    max_drawdowns: np.ndarray = field(repr=False)   # positive magnitudes per path
    ruin_prob: float = 0.0

    def summary(self) -> pd.Series:
        fr, dd = self.final_returns, self.max_drawdowns
        return pd.Series(
            {
                "fraction": self.fraction,
                "ruin_prob": self.ruin_prob,
                "final_ret_p5": np.quantile(fr, 0.05),
                "final_ret_median": np.median(fr),
                "final_ret_p95": np.quantile(fr, 0.95),
                "prob_loss": float((fr < 0).mean()),
                "maxdd_median": np.median(dd),
                "maxdd_p95": np.quantile(dd, 0.95),
            }
        )


def simulate_trade_sequences(
    trade_returns,
    *,
    fraction: float = 1.0,
    n_sims: int = 10_000,
    n_trades: int | None = None,
    ruin_level: float = 0.5,
    seed: int = 42,
) -> MonteCarloResult:
    """Bootstrap trade sequences and compound them at ``fraction``.

    ``fraction`` scales each trade's return (position share of equity).
    ``ruin_level`` is the equity floor as a share of starting capital:
    a path that ever touches it counts as ruined (and stops compounding
    there conceptually; we only need the event).
    """
    r = np.asarray(pd.Series(trade_returns).dropna(), dtype=float)
    if len(r) < 2:
        raise ValueError("Need at least 2 trades to simulate")
    if not 0 < ruin_level < 1:
        raise ValueError("ruin_level must be in (0, 1)")
    if fraction < 0:
        raise ValueError("fraction must be >= 0")
    n_trades = n_trades or len(r)

    rng = np.random.default_rng(seed)
    sampled = r[rng.integers(0, len(r), size=(n_sims, n_trades))]
    # Leverage above 1 can lose more than the position: floor at total loss.
    growth = np.maximum(1.0 + fraction * sampled, 0.0)
    wealth = np.cumprod(growth, axis=1)

    running_max = np.maximum.accumulate(np.concatenate([np.ones((n_sims, 1)), wealth], axis=1), axis=1)[:, 1:]
    drawdowns = 1.0 - wealth / running_max

    return MonteCarloResult(
        fraction=fraction,
        n_sims=n_sims,
        n_trades=n_trades,
        ruin_level=ruin_level,
        final_returns=wealth[:, -1] - 1.0,
        max_drawdowns=drawdowns.max(axis=1),
        ruin_prob=float((wealth.min(axis=1) <= ruin_level).mean()),
    )


def kelly_robustness(
    trade_returns,
    fractions,
    *,
    n_sims: int = 10_000,
    n_trades: int | None = None,
    ruin_level: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Sweep position fractions; one summary row per fraction.

    The classic Kelly picture: median growth rises, peaks, then collapses
    while ruin probability climbs — the table shows where the estimated
    Kelly sits relative to that cliff.
    """
    rows = [
        simulate_trade_sequences(
            trade_returns, fraction=f, n_sims=n_sims, n_trades=n_trades,
            ruin_level=ruin_level, seed=seed,
        ).summary()
        for f in fractions
    ]
    return pd.DataFrame(rows).set_index("fraction")


def kelly_estimate_stability(
    trade_returns,
    *,
    n_boot: int = 2_000,
    seed: int = 42,
) -> pd.Series:
    """Bootstrap confidence interval of the continuous-Kelly estimate.

    Resamples the trade set itself (estimation risk, not sequence risk):
    a wide interval means the Kelly fraction is barely identified by the
    available number of trades.
    """
    r = np.asarray(pd.Series(trade_returns).dropna(), dtype=float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        sample = r[rng.integers(0, len(r), size=len(r))]
        var = sample.var(ddof=1)
        estimates[i] = max(sample.mean() / var, 0.0) if var > 0 else 0.0
    return pd.Series(
        {
            "kelly_p5": np.quantile(estimates, 0.05),
            "kelly_median": np.median(estimates),
            "kelly_p95": np.quantile(estimates, 0.95),
            "kelly_std": estimates.std(ddof=1),
        }
    )
