"""Mass strategy research: sweep every strategy's parameter grid with an
in-sample / out-of-sample split and rank the results.

Methodology
-----------
* Signals are generated once on the **full** history (indicators only look
  backward, so this is legal and gives the out-of-sample window a realistic
  warm-up instead of a cold start).
* The bar index is split at ``split`` (default 70%): the earlier part is
  in-sample (IS), the later part out-of-sample (OOS). Each parameter
  combination is backtested independently on both windows.
* Composite scores (:func:`src.optimization.ranking.composite_score`) are
  computed across the **global** pool of combinations so strategies are
  directly comparable.
* Overfitting diagnostics per strategy: IS→OOS degradation of the best IS
  combination, Spearman rank correlation of IS vs OOS composites across
  the grid, and the OOS percentile of the IS winner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from src.analytics.report import portfolio_metrics
from src.backtesting.config import BacktestConfig
from src.optimization.grid import param_grid
from src.optimization.ranking import composite_score, robust_score
from src.strategies.base import strategy_class
from src.utils.logging import get_logger

logger = get_logger(__name__)

IS_PREFIX = "is_"
OOS_PREFIX = "oos_"


@dataclass
class ResearchConfig:
    """Settings for a research sweep."""

    split: float = 0.7                 # IS share of the history
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    min_trades_is: int = 5             # combos with fewer IS trades are kept but scored down
    # Overfitting thresholds
    degradation_limit: float = 0.7     # best-IS combo loses >70% of its IS Sharpe in OOS
    rank_corr_limit: float = 0.2       # IS/OOS rank correlation below this = unstable grid
    oos_percentile_limit: float = 0.4  # IS winner lands in the bottom 40% of its own grid OOS


def _sliced_portfolio(close: pd.Series, sig, config: BacktestConfig, start: int, stop: int) -> vbt.Portfolio:
    sl = slice(start, stop)
    return vbt.Portfolio.from_signals(
        close=close.iloc[sl],
        entries=sig.entries.iloc[sl],
        exits=sig.exits.iloc[sl],
        short_entries=sig.short_entries.iloc[sl],
        short_exits=sig.short_exits.iloc[sl],
        **config.to_portfolio_kwargs(),
    )


def evaluate_strategy(
    name: str,
    ohlcv: pd.DataFrame,
    *,
    config: ResearchConfig | None = None,
    grid: dict[str, list[Any]] | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Sweep one strategy's grid; return one row per parameter combination.

    Columns: ``param_*``, ``is_<metric>``, ``oos_<metric>``.
    """
    config = config or ResearchConfig()
    cls = strategy_class(name)
    grid = grid or cls.opt_grid
    if not grid:
        raise ValueError(f"Strategy {name!r} has no optimization grid")

    close = ohlcv["Close"]
    split_at = int(len(ohlcv) * config.split)
    if not 50 <= split_at <= len(ohlcv) - 50:
        raise ValueError(f"split={config.split} leaves too little data on one side ({len(ohlcv)} bars)")

    rows: list[dict[str, Any]] = []
    combos = param_grid(grid)
    iterator = tqdm(combos, desc=name, disable=not show_progress, leave=False)
    for params in iterator:
        try:
            strategy = cls(**params)
        except ValueError:
            continue  # ill-formed combo (e.g. fast >= slow)
        sig = strategy.signals(ohlcv)

        row: dict[str, Any] = {"strategy": name, "category": cls.category}
        row.update({f"param_{k}": v for k, v in params.items()})
        for prefix, start, stop in ((IS_PREFIX, 0, split_at), (OOS_PREFIX, split_at, len(ohlcv))):
            pf = _sliced_portfolio(close, sig, config.backtest, start, stop)
            for metric, value in portfolio_metrics(pf).items():
                row[f"{prefix}{metric}"] = value
        rows.append(row)

    if not rows:
        raise ValueError(f"Grid for {name!r} produced no valid combinations")
    return pd.DataFrame(rows)


def _prefixed(table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """View of one window's metrics with the prefix stripped."""
    cols = {c: c[len(prefix):] for c in table.columns if c.startswith(prefix)}
    return table[list(cols)].rename(columns=cols)


def score_combined(table: pd.DataFrame, *, min_trades: int = 10) -> pd.DataFrame:
    """Attach global composite / robustness scores to a combined sweep table."""
    out = table.copy()
    out["is_composite"] = composite_score(_prefixed(out, IS_PREFIX), min_trades=min_trades)
    out["oos_composite"] = composite_score(_prefixed(out, OOS_PREFIX), min_trades=min_trades)
    out["robust_score"] = robust_score(out["is_composite"], out["oos_composite"])
    return out


def research_universe(
    names: list[str],
    ohlcv: pd.DataFrame,
    *,
    config: ResearchConfig | None = None,
    grids: dict[str, dict[str, list[Any]]] | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Sweep every strategy in ``names``; return the combined scored table."""
    config = config or ResearchConfig()
    tables: list[pd.DataFrame] = []
    for name in tqdm(names, desc="strategies", disable=not show_progress):
        grid = (grids or {}).get(name)
        try:
            tables.append(evaluate_strategy(name, ohlcv, config=config, grid=grid))
        except ValueError as exc:
            logger.warning("Skipping %s: %s", name, exc)
    if not tables:
        raise ValueError("No strategy produced results")
    combined = pd.concat(tables, ignore_index=True)
    return score_combined(combined)


def summarize_strategies(combined: pd.DataFrame, *, config: ResearchConfig | None = None) -> pd.DataFrame:
    """Per-strategy summary with robustness and overfitting diagnostics.

    For each strategy the representative configuration is the one with the
    best ``robust_score`` (NOT the best raw return). Overfitting flags:

    * ``flag_degradation`` — the best *IS* combo loses more than
      ``degradation_limit`` of its IS Sharpe out-of-sample;
    * ``flag_rank_instability`` — Spearman correlation between IS and OOS
      composites across the grid is below ``rank_corr_limit`` (parameter
      quality does not carry over);
    * ``flag_is_winner_flops`` — the IS winner falls into the bottom
      ``oos_percentile_limit`` of its own grid out-of-sample.
    """
    config = config or ResearchConfig()
    rows = []
    for name, group in combined.groupby("strategy"):
        g = group.reset_index(drop=True)
        best_robust = g.loc[g["robust_score"].idxmax()]
        best_is = g.loc[g["is_composite"].idxmax()]

        # Degradation of the IS winner's Sharpe.
        is_sharpe, oos_sharpe = best_is["is_sharpe_ratio"], best_is["oos_sharpe_ratio"]
        if pd.notna(is_sharpe) and is_sharpe > 0.5 and pd.notna(oos_sharpe):
            degradation = 1.0 - oos_sharpe / is_sharpe
        else:
            degradation = np.nan

        # Does grid ranking survive out of sample?
        if len(g) >= 5 and g["is_composite"].nunique() > 1 and g["oos_composite"].nunique() > 1:
            rank_corr = float(spearmanr(g["is_composite"], g["oos_composite"]).statistic)
        else:
            rank_corr = np.nan

        oos_pct_of_is_winner = float((g["oos_composite"] <= best_is["oos_composite"]).mean())

        flag_degradation = bool(pd.notna(degradation) and degradation > config.degradation_limit)
        flag_rank = bool(pd.notna(rank_corr) and rank_corr < config.rank_corr_limit)
        flag_flop = bool(oos_pct_of_is_winner < config.oos_percentile_limit)

        params = {c[len("param_"):]: best_robust[c] for c in g.columns
                  if c.startswith("param_") and pd.notna(best_robust[c])}
        rows.append(
            {
                "strategy": name,
                "category": g["category"].iloc[0],
                "n_combos": len(g),
                "best_params": params,
                "robust_score": best_robust["robust_score"],
                "is_composite": best_robust["is_composite"],
                "oos_composite": best_robust["oos_composite"],
                "is_sharpe": best_robust["is_sharpe_ratio"],
                "oos_sharpe": best_robust["oos_sharpe_ratio"],
                "oos_cagr_pct": best_robust["oos_cagr_pct"],
                "oos_max_dd_pct": best_robust["oos_max_drawdown_pct"],
                "oos_trades": best_robust["oos_total_trades"],
                "sharpe_degradation": degradation,
                "is_oos_rank_corr": rank_corr,
                "is_winner_oos_pctile": oos_pct_of_is_winner,
                "flag_degradation": flag_degradation,
                "flag_rank_instability": flag_rank,
                "flag_is_winner_flops": flag_flop,
                "overfit_flags": int(flag_degradation) + int(flag_rank) + int(flag_flop),
            }
        )
    summary = pd.DataFrame(rows).set_index("strategy")
    return summary.sort_values("robust_score", ascending=False)


def top_configurations(combined: pd.DataFrame, n: int = 20, *, per_strategy: int = 2) -> pd.DataFrame:
    """Top-N configurations by robustness, at most ``per_strategy`` each.

    Capping per strategy keeps the leaderboard a survey of approaches
    rather than twenty near-duplicates of one grid's sweet spot.
    """
    cols = ["strategy", "category", "robust_score", "is_composite", "oos_composite",
            "is_sharpe_ratio", "oos_sharpe_ratio", "oos_cagr_pct", "oos_max_drawdown_pct",
            "oos_profit_factor", "oos_win_rate_pct", "oos_total_trades"]
    param_cols = [c for c in combined.columns if c.startswith("param_")]
    ranked = combined.sort_values("robust_score", ascending=False)
    # Different parameter values sometimes produce identical signals
    # (e.g. an unused filter leg) — keep one representative per outcome.
    ranked = ranked.drop_duplicates(subset=cols)
    capped = ranked.groupby("strategy", sort=False).head(per_strategy)
    return capped.head(n)[cols + param_cols].reset_index(drop=True)
