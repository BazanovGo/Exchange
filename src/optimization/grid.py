"""Grid search over strategy parameters.

Runs the unified backtest for every parameter combination and collects the
standard metric set, so any registered strategy can be optimized without
strategy-specific code.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from itertools import product
from typing import Any

import pandas as pd
from tqdm.auto import tqdm

from src.analytics.report import key_metrics
from src.backtesting.config import BacktestConfig
from src.backtesting.engine import run_backtest
from src.data.base import MarketData
from src.utils.logging import get_logger

logger = get_logger(__name__)


def param_grid(grid: Mapping[str, Iterable[Any]]) -> list[dict[str, Any]]:
    """Expand ``{"a": [1, 2], "b": [3]}`` into a list of param dicts."""
    keys = list(grid)
    return [dict(zip(keys, combo, strict=True)) for combo in product(*(grid[k] for k in keys))]


def grid_search(
    strategy_name: str,
    data: MarketData | pd.DataFrame,
    grid: Mapping[str, Iterable[Any]],
    *,
    config: BacktestConfig | None = None,
    symbol: str | None = None,
    sort_by: str = "sharpe_ratio",
    skip_invalid: bool = True,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Backtest every combination in ``grid`` and rank by ``sort_by``.

    Invalid combinations (rejected by the strategy's ``validate_params``)
    are skipped when ``skip_invalid`` is set — convenient for grids like
    fast/slow windows where some pairs are ill-formed.
    """
    combos = param_grid(grid)
    rows: list[pd.Series] = []
    iterator = tqdm(combos, desc=f"grid_search[{strategy_name}]", disable=not show_progress)
    for params in iterator:
        try:
            result = run_backtest(strategy_name, data, params=params, config=config, symbol=symbol)
        except ValueError as exc:
            if skip_invalid:
                logger.debug("Skipping %s: %s", params, exc)
                continue
            raise
        row = key_metrics(result)
        for key, value in params.items():
            row[f"param_{key}"] = value
        rows.append(row)

    if not rows:
        raise ValueError("Grid produced no valid parameter combinations")
    table = pd.DataFrame(rows).reset_index(drop=True)
    param_cols = [c for c in table.columns if c.startswith("param_")]
    metric_cols = [c for c in table.columns if not c.startswith("param_")]
    table = table[param_cols + metric_cols]
    if sort_by in table.columns:
        table = table.sort_values(sort_by, ascending=False).reset_index(drop=True)
    return table
