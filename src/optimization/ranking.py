"""Integral (composite) ranking of backtest results.

A single metric — especially raw return — is a poor selector: it rewards
lucky, concentrated, overfit configurations. The composite score here
rank-normalizes many metrics at once (risk-adjusted returns, drawdown,
trade quality) and penalizes results built on too few trades.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

# metric -> (weight, higher_is_better)
DEFAULT_WEIGHTS: dict[str, tuple[float, bool]] = {
    "sharpe_ratio": (2.0, True),
    "sortino_ratio": (1.5, True),
    "calmar_ratio": (1.5, True),
    "cagr_pct": (1.0, True),
    "total_return_pct": (0.5, True),
    "max_drawdown_pct": (1.5, False),  # positive magnitude, lower is better
    "profit_factor": (1.0, True),
    "win_rate_pct": (0.5, True),
    "expectancy": (0.5, True),
    "recovery_factor": (1.0, True),
}

MIN_TRADES = 10


def composite_score(
    table: pd.DataFrame,
    *,
    weights: Mapping[str, tuple[float, bool]] | None = None,
    trades_col: str = "total_trades",
    min_trades: int = MIN_TRADES,
) -> pd.Series:
    """Integral score in [0, 1] for each row of a metrics table.

    Each metric is converted to a percentile rank across the table
    (direction-aware), the ranks are combined as a weighted average, and
    the result is shrunk for rows with fewer than ``min_trades`` trades —
    a great-looking backtest with 3 trades is noise, not evidence.
    """
    weights = dict(weights or DEFAULT_WEIGHTS)

    parts: list[pd.Series] = []
    total_weight = 0.0
    for metric, (weight, higher_is_better) in weights.items():
        if metric not in table.columns:
            continue
        values = table[metric].replace([np.inf, -np.inf], np.nan)
        ranks = values.rank(pct=True, na_option="keep")
        if not higher_is_better:
            ranks = 1.0 - ranks
        parts.append(ranks.fillna(0.0) * weight)  # missing metric = worst
        total_weight += weight

    if not parts:
        raise ValueError(f"None of the weighted metrics are present in the table: {sorted(weights)}")

    score = sum(parts) / total_weight

    if trades_col in table.columns and min_trades > 0:
        adequacy = (table[trades_col].fillna(0) / min_trades).clip(0.0, 1.0)
        score = score * np.sqrt(adequacy)  # sqrt: soft penalty, zero trades -> zero score

    return score.rename("composite_score")


def robust_score(is_score: pd.Series, oos_score: pd.Series, *, consistency_weight: float = 0.5) -> pd.Series:
    """Combine in-sample and out-of-sample composites into one robustness score.

    The mean rewards overall quality; the absolute gap subtracts for
    inconsistency, so a configuration that only shines in-sample ranks
    below one that performs evenly in both periods.
    """
    return ((is_score + oos_score) / 2.0 - consistency_weight * (is_score - oos_score).abs()).rename("robust_score")
