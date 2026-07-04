"""Parameter optimization and mass strategy research."""

from src.optimization.exit_research import (
    COMPARE_METRICS,
    exit_research,
    mechanic_scoreboard,
    mechanic_universe,
    summarize_mechanics,
    sweep_exit_rules,
)
from src.optimization.filter_research import (
    default_filters,
    filter_research,
    filter_scoreboard,
    regime_performance,
    summarize_filters,
)
from src.optimization.grid import grid_search, param_grid
from src.optimization.hypothesis import RESEARCH_DIR, load_catalog, run_full_study
from src.optimization.ranking import DEFAULT_WEIGHTS, composite_score, robust_score
from src.optimization.research import (
    ResearchConfig,
    evaluate_strategy,
    research_universe,
    score_combined,
    summarize_strategies,
    top_configurations,
)
from src.optimization.walk_forward import (
    WalkForwardConfig,
    walk_forward,
    walk_forward_summary,
    window_bounds,
)

__all__ = [
    "grid_search",
    "param_grid",
    "composite_score",
    "robust_score",
    "DEFAULT_WEIGHTS",
    "ResearchConfig",
    "evaluate_strategy",
    "research_universe",
    "score_combined",
    "summarize_strategies",
    "top_configurations",
    "COMPARE_METRICS",
    "mechanic_universe",
    "sweep_exit_rules",
    "exit_research",
    "summarize_mechanics",
    "mechanic_scoreboard",
    "default_filters",
    "filter_research",
    "summarize_filters",
    "filter_scoreboard",
    "regime_performance",
    "WalkForwardConfig",
    "walk_forward",
    "walk_forward_summary",
    "window_bounds",
    "RESEARCH_DIR",
    "run_full_study",
    "load_catalog",
]
