"""Walk-forward validation.

Rolls a train/test window across the history: in every window the
strategy's parameters are re-optimized on the train segment only and then
evaluated on the untouched test segment. Concatenated test results are the
closest a backtest gets to live behaviour; the IS→OOS efficiency ratio and
the share of profitable windows quantify overfitting risk.

Reuses :func:`src.optimization.research.evaluate_strategy` — each window
is just an IS/OOS split of its own slice, so signals get the window's own
warm-up and never see data beyond it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.backtesting.config import BacktestConfig
from src.optimization.research import ResearchConfig, evaluate_strategy
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class WalkForwardConfig:
    train_bars: int = 1000
    test_bars: int = 250
    step_bars: int = 250          # distance between window starts
    select_by: str = "is_sharpe_ratio"
    min_train_trades: int = 3     # combos with fewer train trades are not candidates
    backtest: BacktestConfig | None = None


def window_bounds(n_bars: int, cfg: WalkForwardConfig) -> list[tuple[int, int, int]]:
    """(start, split, end) triples for every walk-forward window."""
    bounds = []
    start = 0
    while start + cfg.train_bars + cfg.test_bars <= n_bars:
        bounds.append((start, start + cfg.train_bars, start + cfg.train_bars + cfg.test_bars))
        start += cfg.step_bars
    if not bounds:
        raise ValueError(
            f"History of {n_bars} bars is too short for train={cfg.train_bars} + test={cfg.test_bars}"
        )
    return bounds


def walk_forward(
    strategy_name: str,
    ohlcv: pd.DataFrame,
    *,
    grid: dict[str, list[Any]] | None = None,
    cfg: WalkForwardConfig | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """One row per window: params chosen on train, metrics from both sides."""
    cfg = cfg or WalkForwardConfig()
    bounds = window_bounds(len(ohlcv), cfg)

    rows = []
    for w, (start, split, end) in enumerate(tqdm(bounds, desc=strategy_name, disable=not show_progress, leave=False)):
        window = ohlcv.iloc[start:end]
        research_cfg = ResearchConfig(
            split=(split - start) / (end - start),
            backtest=cfg.backtest or BacktestConfig(),
        )
        table = evaluate_strategy(strategy_name, window, config=research_cfg, grid=grid)
        # Zero-trade combos carry meaningless (formerly infinite) ratios and
        # must not be selectable; require a minimum of activity on train.
        candidates = table[table["is_total_trades"] >= cfg.min_train_trades]
        if candidates.empty:
            candidates = table  # degenerate window: fall back, NaNs won't win
        criterion = candidates[cfg.select_by].replace([np.inf, -np.inf], np.nan)
        if criterion.notna().any():
            best = candidates.loc[criterion.idxmax()]
        else:
            best = candidates.iloc[0]

        row = {
            "window": w,
            "train_start": window.index[0],
            "test_start": window.index[split - start],
            "test_end": window.index[-1],
            "params": {
                c[len("param_"):]: best[c]
                for c in table.columns
                if c.startswith("param_") and pd.notna(best[c])
            },
        }
        for metric in ("sharpe_ratio", "total_return_pct", "max_drawdown_pct", "total_trades", "cagr_pct"):
            row[f"is_{metric}"] = best[f"is_{metric}"]
            row[f"oos_{metric}"] = best[f"oos_{metric}"]
        rows.append(row)
    return pd.DataFrame(rows)


def walk_forward_summary(windows: pd.DataFrame) -> pd.Series:
    """Aggregate walk-forward diagnostics.

    * ``wf_efficiency`` — mean OOS Sharpe / mean IS Sharpe of the chosen
      configurations (≈1 healthy, ≪1 overfit, ≤0 broken);
    * ``pct_profitable_windows`` — share of test segments with positive
      return;
    * ``param_stability`` — share of consecutive windows that picked the
      same parameters (1.0 = one global optimum, 0.0 = a new "best" every
      time, which is itself an overfitting sign).
    """
    is_sharpe = windows["is_sharpe_ratio"].mean()
    oos_sharpe = windows["oos_sharpe_ratio"].mean()
    efficiency = oos_sharpe / is_sharpe if is_sharpe > 0 else np.nan

    params = windows["params"].map(lambda p: tuple(sorted(p.items())))
    stability = float((params == params.shift()).iloc[1:].mean()) if len(windows) > 1 else np.nan

    return pd.Series(
        {
            "n_windows": len(windows),
            "mean_is_sharpe": is_sharpe,
            "mean_oos_sharpe": oos_sharpe,
            "wf_efficiency": efficiency,
            "pct_profitable_windows": float((windows["oos_total_return_pct"] > 0).mean()),
            "mean_oos_return_pct": windows["oos_total_return_pct"].mean(),
            "worst_oos_drawdown_pct": windows["oos_max_drawdown_pct"].max(),
            "param_stability": stability,
        }
    )
