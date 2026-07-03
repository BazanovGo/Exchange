"""Parameter optimization and mass strategy research."""

from src.optimization.grid import grid_search, param_grid
from src.optimization.ranking import DEFAULT_WEIGHTS, composite_score, robust_score
from src.optimization.research import (
    ResearchConfig,
    evaluate_strategy,
    research_universe,
    score_combined,
    summarize_strategies,
    top_configurations,
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
]
