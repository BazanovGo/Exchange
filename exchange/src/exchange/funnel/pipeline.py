"""The research funnel.

A fixed sequence of stages, each of which can only narrow what survives:

    Hypothesis -> Sanity -> IS search -> Parameter stability -> OOS
    -> Purged CV / CPCV -> Walk-forward -> Bootstrap / Monte Carlo
    -> Multiple-testing adjustment -> Execution stress -> Regime analysis
    -> Verdict

The one rule that fixes the most common error
----------------------------------------------
**Parameters are chosen on the in-sample window only, and the out-of-sample
window is evaluated exactly once, on that choice.**

The audited codebase scored each configuration on both windows, combined the
two into a ``robust_score``, picked the maximum, and then reported that
configuration's out-of-sample Sharpe as an out-of-sample result. It is not
one: the out-of-sample data participated in the selection, so the number
reported is an in-sample statistic wearing an out-of-sample label. That
single mistake can turn a worthless grid into an apparently validated
strategy, and no downstream test will catch it because the contamination
happened before the tests ran. All nineteen verdicts in that project's
results catalogue rest on it.

Here the split is enforced structurally: :func:`_search_in_sample` is only
ever called with the in-sample slice, and the out-of-sample window is
evaluated once with the winner.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from exchange.backtest.engine import BacktestConfig, BacktestResult, InstrumentSpec, run_backtest
from exchange.backtest.metrics import performance_metrics, sharpe_ratio
from exchange.core.errors import ConfigError, NotEnoughDataError
from exchange.core.logging import get_logger
from exchange.data.dataset import Dataset
from exchange.execution.assumptions import STRESS_LADDER, ExecutionAssumptions, get_scenario
from exchange.funnel.hypothesis import Hypothesis
from exchange.funnel.verdict import Verdict, decide_verdict
from exchange.stats.bootstrap import block_bootstrap, monte_carlo_summary
from exchange.stats.cv import CombinatorialPurgedCV, WalkForward, assert_no_overlap
from exchange.stats.multiple_testing import correct, expected_false_discoveries
from exchange.stats.pbo import probability_of_backtest_overfitting
from exchange.stats.sharpe import deflated_sharpe_ratio, sharpe_per_period
from exchange.strategies.base import Strategy, iter_grid, strategy_class

logger = get_logger(__name__)


@dataclass(frozen=True)
class FunnelConfig:
    """Settings for one funnel run."""

    is_fraction: float = 0.6           # share of the sample used for the search
    embargo_bars: int = 10             # bars purged between IS and OOS
    holding_horizon: int = 5           # bars a signal's outcome spans, for purging
    cpcv_groups: int = 6
    cpcv_test_groups: int = 2
    wf_train: int = 500
    wf_test: int = 125
    n_bootstrap: int = 1000
    n_monte_carlo: int = 1000
    execution_scenario: str = "base"
    stress_ladder: tuple[str, ...] = STRESS_LADDER
    initial_cash: float = 100_000.0
    seed: int = 20260902
    alpha: float = 0.05
    max_grid: int = 400                # refuse absurd searches outright

    def __post_init__(self) -> None:
        if not 0.3 <= self.is_fraction <= 0.8:
            raise ConfigError("is_fraction must be between 0.3 and 0.8")

    def to_dict(self) -> dict:
        return {
            "is_fraction": self.is_fraction,
            "embargo_bars": self.embargo_bars,
            "holding_horizon": self.holding_horizon,
            "cpcv_groups": self.cpcv_groups,
            "cpcv_test_groups": self.cpcv_test_groups,
            "wf_train": self.wf_train,
            "wf_test": self.wf_test,
            "n_bootstrap": self.n_bootstrap,
            "n_monte_carlo": self.n_monte_carlo,
            "execution_scenario": self.execution_scenario,
            "stress_ladder": list(self.stress_ladder),
            "initial_cash": self.initial_cash,
            "seed": self.seed,
            "alpha": self.alpha,
            "max_grid": self.max_grid,
        }


@dataclass
class StageResult:
    """One funnel stage's outcome."""

    name: str
    passed: bool
    summary: str
    metrics: dict = field(default_factory=dict)
    artefacts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "stage": self.name,
            "passed": self.passed,
            "summary": self.summary,
            "metrics": {k: _jsonable(v) for k, v in self.metrics.items()},
        }


@dataclass
class FunnelResult:
    """Everything the funnel produced for one hypothesis."""

    hypothesis: Hypothesis
    config: FunnelConfig
    dataset_summary: dict
    stages: list[StageResult] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    best_params: dict = field(default_factory=dict)
    verdict: Verdict | None = None
    is_result: BacktestResult | None = field(default=None, repr=False)
    oos_result: BacktestResult | None = field(default=None, repr=False)
    sweep: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    stress_table: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    cpcv_table: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    wf_table: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    regime_table: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)
    full_bars: pd.DataFrame = field(default_factory=pd.DataFrame, repr=False)

    def stage(self, name: str) -> StageResult | None:
        return next((s for s in self.stages if s.name == name), None)

    def to_dict(self) -> dict:
        return {
            "hypothesis": self.hypothesis.to_dict(),
            "config": self.config.to_dict(),
            "dataset": self.dataset_summary,
            "best_params": _jsonable(self.best_params),
            "metrics": {k: _jsonable(v) for k, v in self.metrics.items()},
            "stages": [s.to_dict() for s in self.stages],
            "verdict": self.verdict.to_dict() if self.verdict else None,
        }

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        v = self.verdict.label if self.verdict else "INCOMPLETE"
        return f"FunnelResult({self.hypothesis.hypothesis_id}, verdict={v})"


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


# ==========================================================================
# The funnel
# ==========================================================================


def run_funnel(
    dataset: Dataset,
    hypothesis: Hypothesis,
    *,
    config: FunnelConfig | None = None,
    instrument: InstrumentSpec | None = None,
) -> FunnelResult:
    """Push one pre-registered hypothesis through every validation stage."""
    config = config or FunnelConfig()
    bars = dataset.bars
    instrument = instrument or InstrumentSpec(symbol=dataset.symbol)
    execution = get_scenario(config.execution_scenario)

    result = FunnelResult(
        hypothesis=hypothesis,
        config=config,
        dataset_summary=dataset.describe(),
        full_bars=bars,
    )

    # ---------------------------------------------------------------- 1
    sanity = _stage_sanity(dataset, hypothesis, config)
    result.stages.append(sanity)
    if not sanity.passed:
        result.verdict = decide_verdict(hypothesis, result.metrics, result.stages)
        return result

    n = len(bars)
    split = int(n * config.is_fraction)
    is_bars = bars.iloc[:split]
    oos_bars = bars.iloc[split + config.embargo_bars:]
    if len(oos_bars) < 100:
        raise NotEnoughDataError(
            f"{dataset.symbol}: only {len(oos_bars)} out-of-sample bars after the embargo"
        )

    ppy = dataset.bars_per_year

    def make_config(assumptions: ExecutionAssumptions) -> BacktestConfig:
        return BacktestConfig(
            initial_cash=config.initial_cash,
            instrument=instrument,
            execution=assumptions,
            annualisation=ppy,
        )

    base_cfg = make_config(execution)

    # ---------------------------------------------------------------- 2
    sweep, best_params, is_result = _search_in_sample(
        is_bars, hypothesis, base_cfg, dataset.symbol, config
    )
    result.sweep = sweep
    result.best_params = best_params
    result.is_result = is_result
    is_metrics = performance_metrics(is_result, periods_per_year=ppy)
    result.metrics["is_sharpe"] = is_metrics.sharpe_ratio
    result.metrics["is_return"] = is_metrics.total_return
    result.metrics["n_is_trades"] = is_metrics.n_trades
    result.metrics["n_configurations"] = len(sweep)
    result.stages.append(
        StageResult(
            "is_search", True,
            f"Searched {len(sweep)} configuration(s) on {len(is_bars)} in-sample bars; "
            f"best in-sample Sharpe {is_metrics.sharpe_ratio:.2f}.",
            {"is_sharpe": is_metrics.sharpe_ratio, "n_configurations": len(sweep)},
        )
    )

    # ---------------------------------------------------------------- 3
    stability = _stage_parameter_stability(sweep)
    result.stages.append(stability)
    result.metrics.update(stability.metrics)

    # ---------------------------------------------------------------- 4
    oos_result = _evaluate(oos_bars, hypothesis, best_params, base_cfg, dataset.symbol)
    result.oos_result = oos_result
    oos_metrics = performance_metrics(oos_result, periods_per_year=ppy)
    result.metrics.update(
        {
            "oos_sharpe": oos_metrics.sharpe_ratio,
            "oos_return": oos_metrics.total_return,
            "oos_max_drawdown": oos_metrics.max_drawdown,
            "n_oos_trades": oos_metrics.n_trades,
            "oos_cost_drag_pct": oos_metrics.cost_drag_pct,
            "is_oos_sharpe_ratio": (
                oos_metrics.sharpe_ratio / is_metrics.sharpe_ratio
                if is_metrics.sharpe_ratio and np.isfinite(is_metrics.sharpe_ratio)
                and is_metrics.sharpe_ratio > 0
                else np.nan
            ),
        }
    )
    result.stages.append(
        StageResult(
            "out_of_sample",
            bool(np.isfinite(oos_metrics.sharpe_ratio) and oos_metrics.sharpe_ratio > 0),
            f"Out-of-sample Sharpe {oos_metrics.sharpe_ratio:.2f} over {len(oos_bars)} bars "
            f"({oos_metrics.n_trades} trades), against in-sample {is_metrics.sharpe_ratio:.2f}. "
            "The configuration was fixed before this window was touched.",
            {
                "oos_sharpe": oos_metrics.sharpe_ratio,
                "oos_return": oos_metrics.total_return,
                "n_oos_trades": oos_metrics.n_trades,
            },
        )
    )

    # ---------------------------------------------------------------- 5
    cpcv = _stage_cpcv(bars, hypothesis, best_params, base_cfg, dataset.symbol, config, ppy)
    result.stages.append(cpcv)
    result.metrics.update(cpcv.metrics)
    result.cpcv_table = cpcv.artefacts.get("table", pd.DataFrame())

    # ---------------------------------------------------------------- 6
    wf = _stage_walk_forward(bars, hypothesis, base_cfg, dataset.symbol, config, ppy)
    result.stages.append(wf)
    result.metrics.update(wf.metrics)
    result.wf_table = wf.artefacts.get("table", pd.DataFrame())

    # ---------------------------------------------------------------- 7
    mc = _stage_monte_carlo(oos_result, config)
    result.stages.append(mc)
    result.metrics.update(mc.metrics)

    # ---------------------------------------------------------------- 8
    mt = _stage_multiple_testing(result, hypothesis, config, ppy)
    result.stages.append(mt)
    result.metrics.update(mt.metrics)

    # ---------------------------------------------------------------- 9
    stress = _stage_execution_stress(
        oos_bars, hypothesis, best_params, instrument, dataset.symbol, config, ppy
    )
    result.stages.append(stress)
    result.metrics.update(stress.metrics)
    result.stress_table = stress.artefacts.get("table", pd.DataFrame())

    # --------------------------------------------------------------- 10
    regime = _stage_regime(oos_result, bars, ppy)
    result.stages.append(regime)
    result.metrics.update(regime.metrics)
    result.regime_table = regime.artefacts.get("table", pd.DataFrame())

    # --------------------------------------------------------------- 11
    result.verdict = decide_verdict(hypothesis, result.metrics, result.stages)
    logger.info("%s: %s (%s)", hypothesis.hypothesis_id, result.verdict.label, result.verdict.headline)
    return result


# ==========================================================================
# Stages
# ==========================================================================


def _stage_sanity(dataset: Dataset, hypothesis: Hypothesis, config: FunnelConfig) -> StageResult:
    """Refuse to start on bad data, an unregistered strategy, or an absurd grid."""
    problems: list[str] = []
    if not dataset.quality.ok:
        problems.append(f"{len(dataset.quality.errors)} blocking data-quality error(s)")
    if len(dataset.bars) < 400:
        problems.append(f"only {len(dataset.bars)} bars; too short for a funnel run")

    try:
        cls = strategy_class(hypothesis.strategy)
    except ConfigError as exc:
        return StageResult("sanity", False, str(exc))

    grid = hypothesis.n_configurations
    if grid > config.max_grid:
        problems.append(
            f"the search space has {grid} configurations (limit {config.max_grid}). A grid this "
            "size is a data-mining exercise; narrow it or split it into separate pre-registered "
            "hypotheses so the multiple-testing cost is visible."
        )

    # The strategy must provably not see the future on this data.
    try:
        probe = cls(**{**dict(hypothesis.fixed_params), **_first_combo(hypothesis)})
        probe.check_causality(dataset.bars.iloc[: min(len(dataset.bars), 600)])
    except Exception as exc:
        problems.append(f"causality check failed: {exc}")

    if problems:
        return StageResult("sanity", False, "; ".join(problems), {"n_problems": len(problems)})
    return StageResult(
        "sanity", True,
        f"Data quality OK ({len(dataset.bars)} bars, {len(dataset.quality.warnings)} warning(s)); "
        f"strategy {hypothesis.strategy!r} is causal; {grid} configuration(s) to search.",
        {"n_bars": len(dataset.bars), "n_configurations": grid},
    )


def _first_combo(hypothesis: Hypothesis) -> dict:
    for combo in iter_grid(hypothesis.param_space):
        return combo
    return {}


def _instantiate(hypothesis: Hypothesis, params: Mapping[str, Any]) -> Strategy:
    cls = strategy_class(hypothesis.strategy)
    return cls(**{**dict(hypothesis.fixed_params), **dict(params)})


def _evaluate(bars, hypothesis, params, cfg: BacktestConfig, symbol: str) -> BacktestResult:
    strategy = _instantiate(hypothesis, params)
    return run_backtest(bars, strategy.positions(bars), config=cfg, symbol=symbol)


def _search_in_sample(is_bars, hypothesis, cfg, symbol, config):
    """Sweep the grid on the in-sample window ONLY.

    Returns the sweep table, the selected parameters, and the winner's
    in-sample result. Nothing here has access to out-of-sample data, and the
    sweep table deliberately contains no out-of-sample column at all.
    """
    rows = []
    results: dict[tuple, BacktestResult] = {}
    for params in iter_grid(hypothesis.param_space):
        try:
            res = _evaluate(is_bars, hypothesis, params, cfg, symbol)
        except ConfigError:
            continue  # an invalid combination, e.g. fast >= slow
        m = performance_metrics(res, periods_per_year=cfg.annualisation)
        results[tuple(sorted(params.items()))] = res
        rows.append(
            {
                **{f"param_{k}": v for k, v in params.items()},
                "is_sharpe": m.sharpe_ratio,
                "is_return": m.total_return,
                "is_max_drawdown": m.max_drawdown,
                "n_trades": m.n_trades,
                "cost_drag_pct": m.cost_drag_pct,
            }
        )
    if not rows:
        raise ConfigError(f"{hypothesis.strategy}: no valid parameter combination in the grid")

    sweep = pd.DataFrame(rows)
    # Selection criterion: in-sample Sharpe, with configurations that barely
    # traded excluded. A Sharpe computed from four trades is not a Sharpe.
    eligible = sweep[sweep["n_trades"] >= 10]
    if eligible.empty:
        eligible = sweep
    scores = eligible["is_sharpe"].replace([np.inf, -np.inf], np.nan)
    idx = scores.idxmax() if scores.notna().any() else eligible.index[0]

    best_params = {
        c[len("param_"):]: _restore_type(sweep.loc[idx, c])
        for c in sweep.columns if c.startswith("param_")
    }
    key = tuple(sorted(best_params.items()))
    winner = results.get(key) or _evaluate(is_bars, hypothesis, best_params, cfg, symbol)
    return sweep, best_params, winner


def _restore_type(value):
    """Undo pandas' float upcasting of integer parameters."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    return value


def _stage_parameter_stability(sweep: pd.DataFrame) -> StageResult:
    """Is the winner on a plateau or on a spike?

    A single configuration standing far above its neighbours is the signature
    of curve fitting: real effects are smooth in their parameters, because the
    underlying market process does not know where the grid points are.
    """
    scores = sweep["is_sharpe"].replace([np.inf, -np.inf], np.nan).dropna()
    if len(scores) < 3:
        return StageResult(
            "parameter_stability", True,
            "Too few configurations to assess a plateau (single-configuration hypothesis).",
            {"plateau_fraction": np.nan, "peak_isolation": np.nan},
        )
    best = float(scores.max())
    plateau = float((scores > 0).mean())
    # How far the best stands above the rest, in units of the grid's spread.
    rest = scores[scores < best]
    isolation = (
        float((best - rest.mean()) / rest.std(ddof=1))
        if len(rest) > 2 and rest.std(ddof=1) > 0 else np.nan
    )
    top_quartile_median = float(scores.nlargest(max(len(scores) // 4, 1)).median())

    ok = plateau >= 0.4 and (not np.isfinite(isolation) or isolation < 3.0)
    return StageResult(
        "parameter_stability", ok,
        f"{plateau:.0%} of configurations have a positive in-sample Sharpe; the best stands "
        f"{isolation:.1f} standard deviations above the rest of the grid"
        + (" — an isolated spike, characteristic of curve fitting."
           if np.isfinite(isolation) and isolation >= 3 else " — consistent with a plateau."),
        {
            "plateau_fraction": plateau,
            "peak_isolation": isolation,
            "top_quartile_median_sharpe": top_quartile_median,
        },
    )


def _stage_cpcv(bars, hypothesis, best_params, cfg, symbol, config: FunnelConfig, ppy) -> StageResult:
    """Combinatorial purged cross-validation of the selected configuration."""
    try:
        cv = CombinatorialPurgedCV(
            n_groups=config.cpcv_groups,
            n_test_groups=config.cpcv_test_groups,
            horizon=config.holding_horizon,
            embargo=config.embargo_bars,
        )
        splits = list(cv.split(len(bars)))
        assert_no_overlap(splits)
    except (NotEnoughDataError, ConfigError) as exc:
        return StageResult("cpcv", True, f"Not run: {exc}", {"cpcv_median_sharpe": np.nan})

    rows = []
    for split in splits:
        test_bars = bars.iloc[split.test]
        if len(test_bars) < 40:
            continue
        try:
            res = _evaluate(test_bars, hypothesis, best_params, cfg, symbol)
        except Exception:
            continue
        m = performance_metrics(res, periods_per_year=ppy)
        rows.append(
            {
                "split": split.label,
                "n_test": split.n_test,
                "purged": split.purged,
                "embargoed": split.embargoed,
                "sharpe": m.sharpe_ratio,
                "return": m.total_return,
                "max_drawdown": m.max_drawdown,
                "n_trades": m.n_trades,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return StageResult("cpcv", True, "No CPCV split produced a usable result.",
                           {"cpcv_median_sharpe": np.nan})

    sharpes = table["sharpe"].replace([np.inf, -np.inf], np.nan).dropna()
    median = float(sharpes.median()) if len(sharpes) else np.nan
    positive = float((sharpes > 0).mean()) if len(sharpes) else np.nan
    p05 = float(sharpes.quantile(0.05)) if len(sharpes) else np.nan
    dispersion = float(sharpes.std(ddof=1)) if len(sharpes) > 1 else np.nan

    ok = bool(np.isfinite(median) and median > 0 and positive >= 0.6)
    return StageResult(
        "cpcv", ok,
        f"{len(sharpes)} purged splits ({cv.n_paths()} reconstructable paths): median Sharpe "
        f"{median:.2f}, {positive:.0%} positive, 5th percentile {p05:.2f}, dispersion "
        f"{dispersion:.2f}. The dispersion, not the median, is what a robustness claim rests on.",
        {
            "cpcv_median_sharpe": median,
            "cpcv_positive_fraction": positive,
            "cpcv_p05_sharpe": p05,
            "cpcv_sharpe_dispersion": dispersion,
            "cpcv_n_splits": len(sharpes),
        },
        {"table": table},
    )


def _stage_walk_forward(bars, hypothesis, cfg, symbol, config: FunnelConfig, ppy) -> StageResult:
    """Re-optimise in every window; measure how much survives out of sample."""
    try:
        wf = WalkForward(
            train_size=config.wf_train,
            test_size=config.wf_test,
            horizon=config.holding_horizon,
            embargo=config.embargo_bars,
        )
        splits = list(wf.split(len(bars)))
    except (NotEnoughDataError, ConfigError) as exc:
        return StageResult("walk_forward", True, f"Not run: {exc}", {"wf_efficiency": np.nan})

    rows = []
    for split in splits:
        train_bars = bars.iloc[split.train]
        test_bars = bars.iloc[split.test]
        try:
            _, params, is_res = _search_in_sample(train_bars, hypothesis, cfg, symbol, config)
            oos_res = _evaluate(test_bars, hypothesis, params, cfg, symbol)
        except Exception:
            continue
        is_m = performance_metrics(is_res, periods_per_year=ppy)
        oos_m = performance_metrics(oos_res, periods_per_year=ppy)
        rows.append(
            {
                "window": split.label,
                "train_start": train_bars.index[0],
                "test_start": test_bars.index[0],
                "test_end": test_bars.index[-1],
                "params": params,
                "is_sharpe": is_m.sharpe_ratio,
                "oos_sharpe": oos_m.sharpe_ratio,
                "oos_return": oos_m.total_return,
                "oos_trades": oos_m.n_trades,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return StageResult("walk_forward", True, "No walk-forward window produced a result.",
                           {"wf_efficiency": np.nan})

    is_mean = float(table["is_sharpe"].replace([np.inf, -np.inf], np.nan).mean())
    oos_mean = float(table["oos_sharpe"].replace([np.inf, -np.inf], np.nan).mean())
    efficiency = oos_mean / is_mean if np.isfinite(is_mean) and is_mean > 0 else np.nan
    profitable = float((table["oos_return"] > 0).mean())

    keys = table["params"].map(lambda p: tuple(sorted(p.items())))
    param_stability = float((keys == keys.shift()).iloc[1:].mean()) if len(keys) > 1 else np.nan

    ok = bool(np.isfinite(efficiency) and efficiency >= 0.3 and profitable >= 0.5)
    return StageResult(
        "walk_forward", ok,
        f"{len(table)} windows: mean in-sample Sharpe {is_mean:.2f}, mean out-of-sample "
        f"{oos_mean:.2f} (efficiency {efficiency:.2f}), {profitable:.0%} of windows profitable, "
        f"parameter choice repeated in {param_stability:.0%} of consecutive windows.",
        {
            "wf_efficiency": efficiency,
            "wf_mean_oos_sharpe": oos_mean,
            "wf_profitable_fraction": profitable,
            "wf_param_stability": param_stability,
            "wf_n_windows": len(table),
        },
        {"table": table},
    )


def _stage_monte_carlo(oos_result: BacktestResult, config: FunnelConfig) -> StageResult:
    """Block-bootstrap the out-of-sample returns."""
    returns = oos_result.returns
    try:
        summary = monte_carlo_summary(
            returns, n_paths=config.n_monte_carlo, seed=config.seed, ruin_threshold=-0.4
        )
        boot = block_bootstrap(
            returns, sharpe_per_period, n_resamples=config.n_bootstrap, seed=config.seed
        )
    except NotEnoughDataError as exc:
        return StageResult("monte_carlo", True, f"Not run: {exc}", {})

    lo, hi = boot.confidence_interval(0.95)
    p = boot.p_value_greater_than(0.0)

    ok = bool(np.isfinite(lo) and lo > 0)
    return StageResult(
        "monte_carlo", ok,
        f"Block-bootstrap 95% interval for the per-period Sharpe: [{lo:.4f}, {hi:.4f}] "
        f"(bootstrap p = {p:.4f}, block length {boot.block_length:.1f}). Across "
        f"{summary['n_paths']} resampled paths: median drawdown "
        f"{summary['median_max_drawdown']:.1%}, 5% worst {summary['p95_max_drawdown']:.1%}, "
        f"P(loss) = {summary['prob_loss']:.1%}.",
        {
            "bootstrap_sharpe_ci_low": lo,
            "bootstrap_sharpe_ci_high": hi,
            "bootstrap_p_value": p,
            "mc_median_max_drawdown": summary["median_max_drawdown"],
            "mc_p95_max_drawdown": summary["p95_max_drawdown"],
            "mc_prob_loss": summary["prob_loss"],
            "mc_prob_ruin": summary["prob_ruin"],
        },
    )


def _stage_multiple_testing(result: FunnelResult, hypothesis, config: FunnelConfig, ppy) -> StageResult:
    """Deflate for the size of the search, then correct the family of p-values."""
    oos = result.oos_result
    if oos is None:
        return StageResult("multiple_testing", False, "No out-of-sample result to deflate.")

    trials = hypothesis.total_trials
    sweep_sharpes = result.sweep["is_sharpe"].replace([np.inf, -np.inf], np.nan).dropna()
    # The sweep's Sharpes are annualised; the DSR works per period.
    per_period = (sweep_sharpes / np.sqrt(ppy)).to_numpy() if len(sweep_sharpes) else None

    metrics: dict[str, float] = {"n_trials": trials}
    parts = []

    try:
        dsr = deflated_sharpe_ratio(
            oos.returns, n_trials=trials, trial_sharpes=per_period, periods_per_year=ppy
        )
        metrics["deflated_sharpe_ratio"] = dsr.deflated_sharpe_ratio
        metrics["probabilistic_sharpe_ratio"] = dsr.probabilistic_sharpe_ratio
        metrics["expected_max_sharpe_under_null"] = dsr.expected_max_sharpe
        parts.append(dsr.explain())
    except NotEnoughDataError as exc:
        parts.append(f"DSR not computed: {exc}")

    # PBO over the grid: does the selection procedure work at all?
    try:
        pbo = _compute_pbo(result, hypothesis)
        if pbo is not None:
            metrics["pbo"] = pbo.pbo
            metrics["pbo_degradation_slope"] = pbo.performance_degradation
            parts.append(pbo.explain())
    except Exception as exc:
        parts.append(f"PBO not computed: {type(exc).__name__}: {exc}")

    # Family-level p-values from the sweep, corrected two ways.
    if len(sweep_sharpes) >= 3:
        from scipy import stats as sps

        z = sweep_sharpes.to_numpy() / np.sqrt(ppy) * np.sqrt(len(oos.returns))
        p_values = np.clip(sps.norm.sf(z), 1e-12, 1.0)
        bh = correct(p_values, method="benjamini_hochberg", alpha=config.alpha)
        holm_res = correct(p_values, method="holm", alpha=config.alpha)
        metrics["fdr_survivors"] = bh.n_rejected
        metrics["fwer_survivors"] = holm_res.n_rejected
        metrics["expected_false_discoveries"] = expected_false_discoveries(p_values, alpha=config.alpha)
        parts.append(
            f"Of {len(p_values)} configurations, {bh.n_rejected} survive BH-FDR and "
            f"{holm_res.n_rejected} survive Holm-FWER at alpha={config.alpha}; "
            f"{metrics['expected_false_discoveries']:.1f} rejections would be expected by chance."
        )

    dsr_value = metrics.get("deflated_sharpe_ratio", np.nan)
    pbo_value = metrics.get("pbo", np.nan)
    ok = bool(
        (not np.isfinite(dsr_value) or dsr_value >= 0.90)
        and (not np.isfinite(pbo_value) or pbo_value <= 0.5)
    )
    return StageResult("multiple_testing", ok, " ".join(parts), metrics)


def _compute_pbo(result: FunnelResult, hypothesis):
    """Build the (T x N) return matrix the CSCV needs and run it.

    Every configuration is re-run over the *full* sample so all columns share
    one time axis — CSCV partitions the time dimension and needs the
    strategies to be comparable bar for bar.
    """
    if len(result.sweep) < 4 or result.full_bars.empty or result.is_result is None:
        return None

    cfg = result.is_result.config
    symbol = result.dataset_summary["symbol"]
    param_cols = [c for c in result.sweep.columns if c.startswith("param_")]

    columns: dict[str, np.ndarray] = {}
    for _, row in result.sweep.iterrows():
        params = {c[len("param_"):]: _restore_type(row[c]) for c in param_cols}
        try:
            res = _evaluate(result.full_bars, hypothesis, params, cfg, symbol)
        except Exception:
            continue
        columns[str(tuple(sorted(params.items())))] = res.returns.to_numpy()
    if len(columns) < 4:
        return None

    matrix = pd.DataFrame(columns)
    partitions = 10 if len(matrix) >= 400 else 6
    return probability_of_backtest_overfitting(matrix, n_partitions=partitions, max_splits=200)


def _stage_execution_stress(
    oos_bars, hypothesis, best_params, instrument, symbol, config, ppy
) -> StageResult:
    """Re-run out-of-sample under each execution scenario.

    A real edge decays monotonically as costs rise. A result that *improves*
    under worse assumptions is an artefact — almost always a sign error in
    the accounting — and is flagged rather than celebrated.
    """
    rows = []
    for name in config.stress_ladder:
        cfg = BacktestConfig(
            initial_cash=config.initial_cash,
            instrument=instrument,
            execution=get_scenario(name),
            annualisation=ppy,
        )
        try:
            res = _evaluate(oos_bars, hypothesis, best_params, cfg, symbol)
        except Exception as exc:
            rows.append({"scenario": name, "sharpe": np.nan, "note": str(exc)})
            continue
        m = performance_metrics(res, periods_per_year=ppy)
        rows.append(
            {
                "scenario": name,
                "sharpe": m.sharpe_ratio,
                "return": m.total_return,
                "max_drawdown": m.max_drawdown,
                "n_trades": m.n_trades,
                "cost_drag_pct": m.cost_drag_pct,
            }
        )
    table = pd.DataFrame(rows)
    sharpes = table.set_index("scenario")["sharpe"]

    stressed = float(sharpes.get("stressed", np.nan))
    conservative = float(sharpes.get("conservative", np.nan))
    optimistic = float(sharpes.get("optimistic", np.nan))
    monotone = bool(np.isfinite(optimistic) and np.isfinite(stressed) and optimistic >= stressed - 1e-9)
    survives = bool(np.isfinite(conservative) and conservative > 0)

    return StageResult(
        "execution_stress", survives,
        "Sharpe across the cost ladder: "
        + ", ".join(f"{s}={sharpes.get(s, np.nan):.2f}" for s in config.stress_ladder)
        + (". Decay is monotone, as a real edge should be." if monotone else
           ". WARNING: performance does not decay monotonically with costs — check the accounting."),
        {
            "conservative_sharpe": conservative,
            "stressed_sharpe": stressed,
            "optimistic_sharpe": optimistic,
            "cost_decay_monotone": monotone,
            "cost_sensitivity": (
                float(optimistic - stressed)
                if np.isfinite(optimistic) and np.isfinite(stressed) else np.nan
            ),
        },
        {"table": table},
    )


def _stage_regime(oos_result: BacktestResult, bars: pd.DataFrame, ppy: float) -> StageResult:
    """Split out-of-sample performance by volatility regime and by year.

    A strategy whose entire out-of-sample edge came from one year, or from one
    volatility regime, has not been validated across regimes — it has been
    validated once, in conditions that may not recur.
    """
    returns = oos_result.returns
    if len(returns) < 60:
        return StageResult("regime_analysis", True, "Too few out-of-sample bars for regime analysis.", {})

    aligned = bars["close"].reindex(returns.index)
    vol = aligned.pct_change().rolling(20, min_periods=10).std().shift(1)
    terciles = pd.qcut(vol.rank(method="first"), 3, labels=["low_vol", "mid_vol", "high_vol"])

    rows = []
    for label in ["low_vol", "mid_vol", "high_vol"]:
        segment = returns[terciles == label]
        if len(segment) < 20:
            continue
        rows.append(
            {
                "regime": label,
                "n_bars": len(segment),
                "sharpe": sharpe_ratio(segment, periods_per_year=ppy),
                "mean_return": float(segment.mean()),
            }
        )
    for year, segment in returns.groupby(returns.index.year):
        if len(segment) < 30:
            continue
        rows.append(
            {
                "regime": f"year_{year}",
                "n_bars": len(segment),
                "sharpe": sharpe_ratio(segment, periods_per_year=ppy),
                "mean_return": float(segment.mean()),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return StageResult("regime_analysis", True, "No regime had enough bars.", {})

    years = table[table["regime"].str.startswith("year_")]
    positive_years = float((years["sharpe"] > 0).mean()) if len(years) else np.nan
    regimes = table[~table["regime"].str.startswith("year_")]
    positive_regimes = float((regimes["sharpe"] > 0).mean()) if len(regimes) else np.nan

    # Concentration: how much of the total return came from the best year?
    total = float(returns.sum())
    best_year_share = np.nan
    if len(years) and total > 0:
        by_year = returns.groupby(returns.index.year).sum()
        best_year_share = float(by_year.max() / total)

    ok = bool(np.isfinite(positive_regimes) and positive_regimes >= 0.5)
    return StageResult(
        "regime_analysis", ok,
        f"{positive_regimes:.0%} of volatility regimes and {positive_years:.0%} of years are positive"
        + (f"; the best single year accounts for {best_year_share:.0%} of the total return."
           if np.isfinite(best_year_share) else "."),
        {
            "positive_regime_fraction": positive_regimes,
            "positive_year_fraction": positive_years,
            "best_year_return_share": best_year_share,
        },
        {"table": table},
    )
