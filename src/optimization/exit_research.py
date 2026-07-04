"""Research runner for position-management mechanics.

For each (strategy, mechanic) pair the runner optimizes the mechanic's
parameters on the in-sample window and judges the result out-of-sample
against the strategy's own baseline (exit on opposite signal only). The
verdict separates mechanics that genuinely improve a strategy from those
that only look better in-sample ("false improvement" = overfit exits).
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.analytics.report import portfolio_metrics
from src.backtesting.config import BacktestConfig
from src.backtesting.position_management import ExitRules, managed_portfolio
from src.strategies.base import get_strategy
from src.utils.logging import get_logger

logger = get_logger(__name__)

BASELINE = "baseline"

# Metrics compared between a mechanic and the baseline (the task's set).
COMPARE_METRICS = [
    "cagr_pct",
    "profit_factor",
    "max_drawdown_pct",
    "expectancy",
    "recovery_factor",
    "sharpe_ratio",
    "calmar_ratio",
]
# Direction: True = higher is better.
_METRIC_BETTER = {m: m != "max_drawdown_pct" for m in COMPARE_METRICS}


def mechanic_universe() -> dict[str, list[ExitRules]]:
    """Default parameter grids for every mechanic and key combinations."""
    sl_levels = [0.03, 0.05, 0.08, 0.12, 0.20]
    tp_levels = [0.05, 0.08, 0.12, 0.20, 0.30]
    trail_levels = [0.04, 0.06, 0.10, 0.15, 0.20]
    be_triggers = [0.03, 0.05, 0.08, 0.12]
    time_stops = [5, 10, 20, 40, 60]
    partial_levels = [0.05, 0.08, 0.12]

    universe: dict[str, list[ExitRules]] = {
        BASELINE: [ExitRules(label=BASELINE)],
        "stop_loss": [ExitRules(label=f"sl={v}", sl_stop=v) for v in sl_levels],
        "take_profit": [ExitRules(label=f"tp={v}", tp_stop=v) for v in tp_levels],
        "trailing_stop": [ExitRules(label=f"trail={v}", trail_stop=v) for v in trail_levels],
        "break_even": [
            ExitRules(label=f"be={t}/{o}", break_even_trigger=t, break_even_offset=o)
            for t in be_triggers
            for o in (0.0, 0.01)
        ],
        "time_stop": [ExitRules(label=f"time={n}", time_stop=n) for n in time_stops],
        # Replace signal exits entirely with a trailing stop: tests whether
        # the strategy's own exit logic adds value over pure price management.
        "trail_replaces_signal": [
            ExitRules(label=f"trail_only={v}", use_signal_exits=False, trail_stop=v) for v in trail_levels
        ],
        "partial_tp": [
            ExitRules(label=f"partial={v}x{f}", partial_tp=v, partial_fraction=f)
            for v in partial_levels
            for f in (1 / 3, 0.5)
        ],
        "combo_sl_tp": [
            ExitRules(label=f"sl={s},tp={t}", sl_stop=s, tp_stop=t)
            for s in (0.05, 0.08, 0.12)
            for t in (0.12, 0.20, 0.30)
            if t > s
        ],
        "combo_sl_trail_be": [
            ExitRules(label=f"sl={s},trail={tr},be={b}", sl_stop=s, trail_stop=tr, break_even_trigger=b)
            for s in (0.08, 0.12)
            for tr in (0.06, 0.10, 0.15)
            for b in (0.05, 0.08)
        ],
        "combo_sl_tp_time": [
            ExitRules(label=f"sl={s},tp={t},time={n}", sl_stop=s, tp_stop=t, time_stop=n)
            for s in (0.05, 0.08)
            for t in (0.12, 0.20)
            for n in (20, 40)
        ],
        "combo_trail_partial": [
            ExitRules(label=f"trail={tr},partial={p}", trail_stop=tr, partial_tp=p)
            for tr in (0.06, 0.10, 0.15)
            for p in (0.05, 0.08)
        ],
    }
    return universe


def _rule_param_columns(rules: ExitRules) -> dict[str, Any]:
    out = {}
    for f in fields(rules):
        if f.name == "label":
            continue
        out[f"rule_{f.name}"] = getattr(rules, f.name)
    return out


def sweep_exit_rules(
    ohlcv: pd.DataFrame,
    entries: pd.Series,
    signal_exits: pd.Series,
    universe: dict[str, list[ExitRules]] | None = None,
    *,
    config: BacktestConfig | None = None,
    split: float = 0.7,
    strategy_name: str = "",
    show_progress: bool = False,
) -> pd.DataFrame:
    """Backtest every exit-rule configuration on IS and OOS windows.

    Signals are precomputed on the full history; the exit simulator runs
    independently inside each window, so a position never straddles the
    IS/OOS boundary.
    """
    universe = universe or mechanic_universe()
    config = config or BacktestConfig()
    split_at = int(len(ohlcv) * split)
    if not 50 <= split_at <= len(ohlcv) - 50:
        raise ValueError(f"split={split} leaves too little data on one side ({len(ohlcv)} bars)")

    windows = {
        "is_": (ohlcv.iloc[:split_at], entries.iloc[:split_at], signal_exits.iloc[:split_at]),
        "oos_": (ohlcv.iloc[split_at:], entries.iloc[split_at:], signal_exits.iloc[split_at:]),
    }

    items = [(mechanic, rules) for mechanic, rule_list in universe.items() for rules in rule_list]
    rows: list[dict[str, Any]] = []
    for mechanic, rules in tqdm(items, desc=strategy_name or "rules", disable=not show_progress, leave=False):
        row: dict[str, Any] = {"strategy": strategy_name, "mechanic": mechanic, "label": rules.label}
        row.update(_rule_param_columns(rules))
        for prefix, (win_ohlcv, win_entries, win_exits) in windows.items():
            pf, _ = managed_portfolio(win_ohlcv, win_entries, win_exits, rules, config)
            for metric, value in portfolio_metrics(pf).items():
                row[f"{prefix}{metric}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def exit_research(
    strategies: dict[str, dict[str, Any]],
    ohlcv: pd.DataFrame,
    universe: dict[str, list[ExitRules]] | None = None,
    *,
    config: BacktestConfig | None = None,
    split: float = 0.7,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Run the mechanics sweep for several strategies.

    ``strategies`` maps a registered strategy name to the parameter dict of
    its (previously researched) best configuration.
    """
    tables = []
    iterator = tqdm(strategies.items(), desc="strategies", disable=not show_progress)
    for name, params in iterator:
        sig = get_strategy(name, **params).signals(ohlcv)
        tables.append(
            sweep_exit_rules(
                ohlcv, sig.entries, sig.exits, universe,
                config=config, split=split, strategy_name=name,
            )
        )
    return pd.concat(tables, ignore_index=True)


def _pick_best_is(group: pd.DataFrame, by: str = "is_sharpe_ratio") -> pd.Series:
    """Best configuration of a mechanic by the in-sample criterion."""
    return group.loc[group[by].fillna(-np.inf).idxmax()]


def summarize_mechanics(
    sweep: pd.DataFrame,
    *,
    select_by: str = "is_sharpe_ratio",
    judge_by: tuple[str, ...] = ("sharpe_ratio", "calmar_ratio"),
) -> pd.DataFrame:
    """Per (strategy, mechanic): optimize IS, judge OOS against the baseline.

    The verdict imitates honest model selection: the mechanic's parameters
    are chosen on IS only (``select_by``), then its OOS metrics are
    compared with the strategy's own baseline:

    * ``real improvement`` — better than baseline both IS and OOS;
    * ``false improvement`` — better IS, worse OOS (the classic overfit exit);
    * ``no effect / worse`` — not better even in-sample.
    """
    rows = []
    for strategy, strat_group in sweep.groupby("strategy"):
        baseline = strat_group[strat_group["mechanic"] == BASELINE].iloc[0]
        for mechanic, group in strat_group.groupby("mechanic"):
            if mechanic == BASELINE:
                continue
            best = _pick_best_is(group, by=select_by)

            row: dict[str, Any] = {
                "strategy": strategy,
                "mechanic": mechanic,
                "best_label": best["label"],
                "n_configs": len(group),
            }
            is_better, oos_better = [], []
            for metric in COMPARE_METRICS:
                for prefix, bucket in (("is_", is_better), ("oos_", oos_better)):
                    # Raw delta (mechanic - baseline); for max_drawdown_pct a
                    # negative delta therefore means "drawdown got smaller".
                    raw = best[f"{prefix}{metric}"] - baseline[f"{prefix}{metric}"]
                    row[f"d_{prefix}{metric}"] = raw
                    improvement = raw if _METRIC_BETTER[metric] else -raw
                    if metric in judge_by:
                        bucket.append(improvement > 0)
            row["baseline_oos_sharpe"] = baseline["oos_sharpe_ratio"]
            row["oos_sharpe"] = best["oos_sharpe_ratio"]
            row["oos_trades"] = best["oos_total_trades"]

            improves_is = all(is_better)
            improves_oos = all(oos_better)
            if improves_is and improves_oos:
                verdict = "real improvement"
            elif improves_is and not improves_oos:
                verdict = "false improvement"
            elif improves_oos:
                verdict = "oos-only (lucky)"
            else:
                verdict = "no effect / worse"
            row["verdict"] = verdict
            rows.append(row)
    return pd.DataFrame(rows)


def mechanic_scoreboard(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate verdicts per mechanic across strategies."""
    board = (
        summary.pivot_table(index="mechanic", columns="verdict", values="strategy", aggfunc="count")
        .fillna(0)
        .astype(int)
    )
    for col in ("real improvement", "false improvement", "oos-only (lucky)", "no effect / worse"):
        if col not in board.columns:
            board[col] = 0
    board["strategies"] = board.sum(axis=1)
    board["real_rate"] = board["real improvement"] / board["strategies"]
    board["mean_d_oos_sharpe"] = summary.groupby("mechanic")["d_oos_sharpe_ratio"].mean()
    board["mean_d_oos_max_dd"] = summary.groupby("mechanic")["d_oos_max_drawdown_pct"].mean()
    return board.sort_values(["real_rate", "mean_d_oos_sharpe"], ascending=False)
