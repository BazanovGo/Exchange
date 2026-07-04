"""Entry-filter research: which market-state filters genuinely help.

A filter is a boolean series derived from :mod:`src.analytics.features`
that gates strategy entries (``entries & filter``). Each (strategy,
filter) pair is backtested on the in-sample and out-of-sample windows and
compared with the unfiltered baseline — the same honest protocol as the
exit-mechanic research: a filter only counts as a *real* improvement when
it helps in both windows.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt
from tqdm.auto import tqdm

from src.analytics.report import portfolio_metrics
from src.backtesting.config import BacktestConfig
from src.strategies.base import get_strategy
from src.utils.logging import get_logger

logger = get_logger(__name__)

BASELINE = "baseline"

FilterFn = Callable[[pd.DataFrame], pd.Series]


def default_filters() -> dict[str, FilterFn]:
    """Named entry-filter candidates over the market-feature frame.

    Median-based splits use each feature's *rolling* median (already part
    of regime labelling for volatility) or a fixed conventional threshold,
    never a full-sample statistic, to avoid lookahead.
    """
    return {
        "adx_trending": lambda f: f["adx"] >= 25,
        "adx_quiet": lambda f: f["adx"] < 25,
        "low_vol": lambda f: f["volatility"] <= f["volatility"].rolling(252, min_periods=63).median(),
        "high_vol": lambda f: f["volatility"] > f["volatility"].rolling(252, min_periods=63).median(),
        "htf_up": lambda f: f["htf_trend"] > 0,
        "above_ma": lambda f: f["dist_ma"] > 0,
        "not_stretched": lambda f: f["dist_ma"].abs() < 0.10,
        "bb_narrow": lambda f: f["bb_width"] <= f["bb_width"].rolling(252, min_periods=63).median(),
        "bb_wide": lambda f: f["bb_width"] > f["bb_width"].rolling(252, min_periods=63).median(),
        "ema_slope_up": lambda f: f["ema_slope"] > 0,
        "trend_efficient": lambda f: f["efficiency_ratio"] >= 0.3,
        "volume_active": lambda f: f["rel_volume"] > 1.0,
        # Combinations — regime alignment stacked with confirmation.
        "htf_up+adx_trending": lambda f: (f["htf_trend"] > 0) & (f["adx"] >= 25),
        "htf_up+ema_slope_up": lambda f: (f["htf_trend"] > 0) & (f["ema_slope"] > 0),
        "ema_slope_up+low_vol": lambda f: (f["ema_slope"] > 0)
        & (f["volatility"] <= f["volatility"].rolling(252, min_periods=63).median()),
        "adx_trending+volume_active": lambda f: (f["adx"] >= 25) & (f["rel_volume"] > 1.0),
        "htf_up+trend_efficient": lambda f: (f["htf_trend"] > 0) & (f["efficiency_ratio"] >= 0.3),
    }


def _window_metrics(close, entries, exits, config, split_at, prefix_rows: dict[str, Any]) -> dict[str, Any]:
    for prefix, sl in (("is_", slice(0, split_at)), ("oos_", slice(split_at, None))):
        pf = vbt.Portfolio.from_signals(
            close=close.iloc[sl],
            entries=entries.iloc[sl],
            exits=exits.iloc[sl],
            **config.to_portfolio_kwargs(),
        )
        for metric, value in portfolio_metrics(pf).items():
            prefix_rows[f"{prefix}{metric}"] = value
    return prefix_rows


def filter_research(
    strategies: Mapping[str, dict],
    ohlcv: pd.DataFrame,
    features: pd.DataFrame,
    filters: Mapping[str, FilterFn] | None = None,
    *,
    config: BacktestConfig | None = None,
    split: float = 0.7,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Backtest every (strategy, filter) pair on IS and OOS windows."""
    filters = dict(filters or default_filters())
    config = config or BacktestConfig()
    close = ohlcv["Close"]
    split_at = int(len(ohlcv) * split)
    if not 50 <= split_at <= len(ohlcv) - 50:
        raise ValueError(f"split={split} leaves too little data on one side ({len(ohlcv)} bars)")

    rows: list[dict[str, Any]] = []
    iterator = tqdm(strategies.items(), desc="strategies", disable=not show_progress)
    for name, params in iterator:
        sig = get_strategy(name, **params).signals(ohlcv)
        variants: list[tuple[str, pd.Series]] = [(BASELINE, sig.entries)]
        for fname, fn in filters.items():
            mask = fn(features).fillna(False).astype(bool)
            variants.append((fname, sig.entries & mask))
        for fname, entries in variants:
            row: dict[str, Any] = {
                "strategy": name,
                "filter": fname,
                "entries_kept": int(entries.sum()),
                "entries_total": int(sig.entries.sum()),
            }
            rows.append(_window_metrics(close, entries, sig.exits, config, split_at, row))
    return pd.DataFrame(rows)


def summarize_filters(
    sweep: pd.DataFrame,
    *,
    judge_by: tuple[str, ...] = ("sharpe_ratio", "calmar_ratio"),
    min_entries: int = 5,
) -> pd.DataFrame:
    """Per (strategy, filter) verdict against the unfiltered baseline.

    * ``real improvement`` — better on the judge metrics both IS and OOS;
    * ``false improvement`` — better IS only (the filter fit the past);
    * ``no effect / worse`` — no in-sample edge to begin with;
    * filters that keep fewer than ``min_entries`` trades are marked
      ``too few trades`` — a filter that removes nearly everything proves
      nothing.
    """
    rows = []
    for strategy, group in sweep.groupby("strategy"):
        baseline = group[group["filter"] == BASELINE].iloc[0]
        for _, row in group[group["filter"] != BASELINE].iterrows():
            deltas: dict[str, float] = {}
            is_better, oos_better = [], []
            for metric in ("cagr_pct", "sharpe_ratio", "calmar_ratio", "max_drawdown_pct", "profit_factor"):
                for prefix, bucket in (("is_", is_better), ("oos_", oos_better)):
                    raw = row[f"{prefix}{metric}"] - baseline[f"{prefix}{metric}"]
                    if not np.isfinite(raw):
                        raw = np.nan  # degenerate window (e.g. zero trades left)
                    deltas[f"d_{prefix}{metric}"] = raw
                    improvement = -raw if metric == "max_drawdown_pct" else raw
                    if metric in judge_by:
                        bucket.append(improvement > 0)

            if row["entries_kept"] < min_entries:
                verdict = "too few trades"
            elif all(is_better) and all(oos_better):
                verdict = "real improvement"
            elif all(is_better):
                verdict = "false improvement"
            elif all(oos_better):
                verdict = "oos-only (lucky)"
            else:
                verdict = "no effect / worse"

            rows.append(
                {
                    "strategy": strategy,
                    "filter": row["filter"],
                    "entries_kept": row["entries_kept"],
                    "entries_total": row["entries_total"],
                    "baseline_oos_sharpe": baseline["oos_sharpe_ratio"],
                    "oos_sharpe": row["oos_sharpe_ratio"],
                    **deltas,
                    "verdict": verdict,
                }
            )
    return pd.DataFrame(rows)


def filter_scoreboard(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate filter verdicts across strategies."""
    board = (
        summary.pivot_table(index="filter", columns="verdict", values="strategy", aggfunc="count")
        .fillna(0)
        .astype(int)
    )
    for col in ("real improvement", "false improvement", "oos-only (lucky)", "no effect / worse", "too few trades"):
        if col not in board.columns:
            board[col] = 0
    board["strategies"] = board.sum(axis=1)
    board["real_rate"] = board["real improvement"] / board["strategies"]
    board["mean_d_oos_sharpe"] = summary.groupby("filter")["d_oos_sharpe_ratio"].mean()
    return board.sort_values(["real_rate", "mean_d_oos_sharpe"], ascending=False)


def regime_performance(trades: pd.DataFrame, *, by: str = "regime") -> pd.DataFrame:
    """Aggregate trade outcomes by a categorical feature (regime by default)."""
    grouped = trades.groupby(["strategy", by], observed=False)["ret_pct"]
    table = grouped.agg(trades="count", mean_ret="mean", median_ret="median", win_rate=lambda s: (s > 0).mean())
    return table.reset_index()
