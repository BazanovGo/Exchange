"""Full-cycle hypothesis study: one call runs every validation stage.

The autonomous research loop pushes each new trading idea through the
complete pipeline (idea -> rules -> code -> backtest -> optimization ->
stability -> OOS -> walk-forward -> Monte Carlo -> overfit verdict) and
never moves on before the study is archived in ``research/``.

``run_full_study`` composes the building blocks from previous stages and
adds parameter-plateau analysis plus a machine verdict:

* ``promising`` — OOS edge, healthy walk-forward, wide parameter plateau;
* ``viable``    — survives every gate but without a strong OOS edge;
* ``archived``  — fails at least one gate; the reasons are recorded so the
  idea is never re-tested.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.optimization.research import ResearchConfig, evaluate_strategy, score_combined, summarize_strategies
from src.optimization.walk_forward import WalkForwardConfig, walk_forward, walk_forward_summary
from src.risk.money_management import TradeStats
from src.risk.monte_carlo import kelly_estimate_stability, simulate_trade_sequences
from src.strategies.base import get_strategy, strategy_class
from src.utils.logging import get_logger
from src.utils.paths import PROJECT_ROOT

logger = get_logger(__name__)

RESEARCH_DIR = PROJECT_ROOT / "research"
CATALOG_PATH = RESEARCH_DIR / "catalog.csv"

CATALOG_COLUMNS = [
    "hypothesis_id", "strategy", "category", "idea", "verdict", "reasons",
    "n_combos", "oos_sharpe", "oos_cagr_pct", "oos_max_dd_pct", "oos_trades",
    "plateau_pct_positive", "plateau_top_quartile_oos_median",
    "wf_efficiency", "wf_profitable_windows", "wf_param_stability",
    "mc_ruin_prob", "mc_maxdd_median", "kelly_ci_width",
    "overfit_flags", "is_oos_rank_corr", "studied_at",
]


@dataclass
class StudyResult:
    hypothesis_id: str
    strategy_name: str
    idea: str
    verdict: str                     # promising / viable / archived
    reasons: list[str]               # populated when archived
    best_params: dict[str, Any]
    metrics: pd.Series               # flat summary for the catalog
    directory: Path
    sweep: pd.DataFrame = field(repr=False)
    wf_windows: pd.DataFrame = field(repr=False)


def _plateau_metrics(scored: pd.DataFrame) -> dict[str, float]:
    """How much of the grid works — a plateau, not a single lucky point."""
    oos = scored["oos_sharpe_ratio"].replace([np.inf, -np.inf], np.nan)
    is_ = scored["is_sharpe_ratio"].replace([np.inf, -np.inf], np.nan)
    top_is = scored.nlargest(max(len(scored) // 4, 1), "is_composite")
    return {
        "plateau_pct_positive": float((oos > 0).mean()),
        "plateau_is_pct_positive": float((is_ > 0).mean()),
        "plateau_top_quartile_oos_median": float(
            top_is["oos_sharpe_ratio"].replace([np.inf, -np.inf], np.nan).median()
        ),
        "plateau_oos_iqr": float(oos.quantile(0.75) - oos.quantile(0.25)),
    }


def _save_plots(scored: pd.DataFrame, strategy_name: str, best_params: dict, ohlcv: pd.DataFrame,
                directory: Path) -> None:
    """Equity curve of the best combo + IS/OOS heatmap over the two widest params."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        from src.backtesting.engine import run_backtest
        from src.visualization.charts import _ensure_kaleido_browser, _json_safe

        _ensure_kaleido_browser()
        result = run_backtest(get_strategy(strategy_name, **best_params), ohlcv)
        equity = result.equity()
        fig = go.Figure(go.Scatter(x=equity.index, y=equity, mode="lines", name="equity"))
        fig.update_layout(title=f"{strategy_name}: equity (best robust params)", width=900, height=380)
        _json_safe(fig).write_image(str(directory / "equity.png"), width=900, height=380)

        param_cols = [c for c in scored.columns if c.startswith("param_") and scored[c].nunique() > 1]
        if len(param_cols) >= 2:
            x, y = sorted(param_cols, key=lambda c: scored[c].nunique(), reverse=True)[:2]
            hm = make_subplots(rows=1, cols=2, subplot_titles=("IS Sharpe", "OOS Sharpe"))
            for i, prefix in enumerate(("is_", "oos_"), start=1):
                grid = scored.pivot_table(index=y, columns=x, values=f"{prefix}sharpe_ratio")
                hm.add_trace(
                    go.Heatmap(z=grid.values, x=[str(c) for c in grid.columns],
                               y=[str(r) for r in grid.index], colorscale="RdBu", zmid=0,
                               showscale=i == 2),
                    row=1, col=i,
                )
                hm.update_xaxes(title_text=x.removeprefix("param_"), row=1, col=i)
                hm.update_yaxes(title_text=y.removeprefix("param_"), row=1, col=i)
            hm.update_layout(width=1000, height=420, title=f"{strategy_name}: parameter stability")
            _json_safe(hm).write_image(str(directory / "param_heatmap.png"), width=1000, height=420)
    except Exception as exc:  # plots must never kill a study
        logger.warning("Plot saving failed for %s: %s", strategy_name, exc)


def _write_report(res: StudyResult, scored: pd.DataFrame, wf_sum: pd.Series, notes: str) -> None:
    m = res.metrics
    top5 = scored.nlargest(5, "robust_score")
    param_cols = [c for c in scored.columns if c.startswith("param_")]
    lines = [
        f"# {res.hypothesis_id}: {res.strategy_name}",
        "",
        f"**Вердикт:** `{res.verdict}`" + (f" — {'; '.join(res.reasons)}" if res.reasons else ""),
        f"**Дата:** {m['studied_at']}",
        "",
        "## Идея",
        res.idea,
        "",
        "## Правила",
        strategy_class(res.strategy_name).description,
        "",
        "## Лучшие параметры (по robust score)",
        "```json",
        json.dumps(res.best_params, indent=2, default=str),
        "```",
        "",
        "## Результаты",
        "",
        "| метрика | значение |",
        "|---|---|",
        f"| комбинаций в сетке | {int(m['n_combos'])} |",
        f"| OOS Sharpe (best robust) | {m['oos_sharpe']:.2f} |",
        f"| OOS CAGR % | {m['oos_cagr_pct']:.1f} |",
        f"| OOS MaxDD % | {m['oos_max_dd_pct']:.1f} |",
        f"| OOS сделок | {int(m['oos_trades'])} |",
        f"| плато: доля комбо с OOS Sharpe>0 | {m['plateau_pct_positive']:.0%} |",
        f"| walk-forward efficiency | {m['wf_efficiency']:.2f} |",
        f"| WF прибыльных окон | {m['wf_profitable_windows']:.0%} |",
        f"| MC P(разорения) @f=0.25 | {m['mc_ruin_prob']:.4f} |",
        f"| MC медианная просадка @f=0.25 | {m['mc_maxdd_median']:.1%} |",
        f"| флагов переобучения (этап-02 методика) | {int(m['overfit_flags'])} |",
        "",
        "## Топ-5 конфигураций",
        "",
        top5[param_cols + ["is_sharpe_ratio", "oos_sharpe_ratio", "robust_score"]]
        .round(3).to_markdown(index=False),
        "",
        "## Выводы",
        notes,
        "",
        "Артефакты: `sweep.csv` (все конфигурации), `walk_forward.csv`, `equity.png`, `param_heatmap.png`.",
    ]
    (res.directory / "report.md").write_text("\n".join(lines))


def _update_catalog(metrics: pd.Series, catalog_path: Path = CATALOG_PATH) -> None:
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    row = metrics.reindex(CATALOG_COLUMNS)
    if catalog_path.exists():
        catalog = pd.read_csv(catalog_path)
        catalog = catalog[catalog["hypothesis_id"] != row["hypothesis_id"]]
        catalog = pd.concat([catalog, row.to_frame().T], ignore_index=True)
    else:
        catalog = row.to_frame().T
    catalog.to_csv(catalog_path, index=False)


def run_full_study(
    hypothesis_id: str,
    strategy_name: str,
    idea: str,
    ohlcv: pd.DataFrame,
    *,
    grid: dict[str, list[Any]] | None = None,
    notes: str = "",
    out_root: Path | str = RESEARCH_DIR,
    wf_cfg: WalkForwardConfig | None = None,
    mc_fraction: float = 0.25,
    save_plots: bool = True,
    min_oos_trades: int = 8,
    min_oos_sharpe: float = 0.2,
    min_plateau: float = 0.35,
    min_wf_efficiency: float = 0.3,
    min_wf_profitable: float = 0.25,
) -> StudyResult:
    """Run the complete research pipeline for one registered strategy."""
    cls = strategy_class(strategy_name)
    out_root = Path(out_root)
    directory = out_root / f"{hypothesis_id}_{strategy_name}"
    directory.mkdir(parents=True, exist_ok=True)

    # 1-2. Backtest the whole grid with the IS/OOS split, score it.
    sweep = evaluate_strategy(strategy_name, ohlcv, grid=grid, config=ResearchConfig())
    scored = score_combined(sweep)
    scored.to_csv(directory / "sweep.csv", index=False)

    # 3. Representative configuration + stage-02 overfit diagnostics.
    summary = summarize_strategies(scored).iloc[0]
    best_params = summary["best_params"]

    # 4. Parameter plateau.
    plateau = _plateau_metrics(scored)

    # 5. Walk-forward with per-window re-optimization.
    wf_cfg = wf_cfg or WalkForwardConfig()
    wf_windows = walk_forward(strategy_name, ohlcv, grid=grid, cfg=wf_cfg)
    wf_windows.to_csv(directory / "walk_forward.csv", index=False)
    wf_sum = walk_forward_summary(wf_windows)

    # 6. Monte Carlo on the best combo's trades.
    from src.backtesting.engine import run_backtest

    trades = run_backtest(get_strategy(strategy_name, **best_params), ohlcv).trades()
    trade_rets = trades.loc[trades["Status"] == "Closed", "Return"].to_numpy()
    if len(trade_rets) >= 10:
        mc = simulate_trade_sequences(trade_rets, fraction=mc_fraction, n_sims=5_000, n_trades=100)
        kelly_ci = kelly_estimate_stability(trade_rets, n_boot=1_000)
        mc_ruin, mc_dd = mc.ruin_prob, float(np.median(mc.max_drawdowns))
        kelly_width = float(kelly_ci["kelly_p95"] - kelly_ci["kelly_p5"])
        trade_stats = TradeStats.from_returns(trade_rets).as_series().to_dict()
    else:
        mc_ruin = mc_dd = kelly_width = np.nan
        trade_stats = {}

    # 7. Verdict.
    reasons: list[str] = []
    oos_sharpe = summary["oos_sharpe"]
    if summary["oos_trades"] < min_oos_trades:
        reasons.append(f"too few OOS trades ({int(summary['oos_trades'])} < {min_oos_trades})")
    if summary["overfit_flags"] >= 2:
        reasons.append(f"stage-02 overfit flags = {int(summary['overfit_flags'])}")
    if pd.notna(oos_sharpe) and oos_sharpe < min_oos_sharpe:
        reasons.append(f"no OOS edge (Sharpe {oos_sharpe:.2f} < {min_oos_sharpe})")
    if plateau["plateau_pct_positive"] < min_plateau:
        reasons.append(
            f"no parameter plateau ({plateau['plateau_pct_positive']:.0%} of grid works OOS)"
        )
    wf_eff = wf_sum["wf_efficiency"]
    wf_prof = wf_sum["pct_profitable_windows"]
    if pd.notna(wf_eff) and wf_eff < min_wf_efficiency:
        reasons.append(f"fails walk-forward (efficiency {wf_eff:.2f} < {min_wf_efficiency})")
    if wf_prof < min_wf_profitable:
        reasons.append(
            f"walk-forward windows unprofitable ({wf_prof:.0%} < {min_wf_profitable:.0%})"
        )

    if reasons:
        verdict = "archived"
    elif oos_sharpe >= 0.6 and wf_prof >= 0.5 and (pd.isna(wf_eff) or wf_eff >= 0.5):
        verdict = "promising"
    else:
        verdict = "viable"

    metrics = pd.Series(
        {
            "hypothesis_id": hypothesis_id,
            "strategy": strategy_name,
            "category": cls.category,
            "idea": idea,
            "verdict": verdict,
            "reasons": "; ".join(reasons),
            "n_combos": len(scored),
            "oos_sharpe": oos_sharpe,
            "oos_cagr_pct": summary["oos_cagr_pct"],
            "oos_max_dd_pct": summary["oos_max_dd_pct"],
            "oos_trades": summary["oos_trades"],
            **plateau,
            "wf_efficiency": wf_eff,
            "wf_profitable_windows": wf_sum["pct_profitable_windows"],
            "wf_param_stability": wf_sum["param_stability"],
            "mc_ruin_prob": mc_ruin,
            "mc_maxdd_median": mc_dd,
            "kelly_ci_width": kelly_width,
            "overfit_flags": summary["overfit_flags"],
            "is_oos_rank_corr": summary["is_oos_rank_corr"],
            "studied_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            **{f"trade_{k}": v for k, v in trade_stats.items()},
        }
    )

    result = StudyResult(
        hypothesis_id=hypothesis_id,
        strategy_name=strategy_name,
        idea=idea,
        verdict=verdict,
        reasons=reasons,
        best_params=best_params,
        metrics=metrics,
        directory=directory,
        sweep=scored,
        wf_windows=wf_windows,
    )

    if save_plots:
        _save_plots(scored, strategy_name, best_params, ohlcv, directory)
    _write_report(result, scored, wf_sum, notes or "—")
    _update_catalog(metrics, catalog_path=out_root / "catalog.csv")
    logger.info("%s (%s): %s %s", hypothesis_id, strategy_name, verdict, reasons or "")
    return result


def load_catalog() -> pd.DataFrame:
    """The registry of every hypothesis studied so far."""
    if not CATALOG_PATH.exists():
        return pd.DataFrame(columns=CATALOG_COLUMNS)
    return pd.read_csv(CATALOG_PATH)
