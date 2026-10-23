"""Performance metrics.

Two rules govern this module, both learned from errors in the audited
codebase:

**Annualise with the data's own sampling rate.** ``sqrt(252)`` is not a
universal constant. Every function takes ``periods_per_year`` and the default
comes from the index of the series being measured, so an hourly strategy
cannot silently inherit a daily factor and report a Sharpe ratio nearly six
times too large.

**A ratio without a denominator is not a number.** A portfolio that never
traded has zero return variance; ``mean/0`` is ``inf``, and an ``inf`` sorts
to the top of any ranking. Every ratio here returns ``NaN`` in that case, and
``NaN`` never wins a selection. The audited project handled this for the
Sharpe family but left ``profit_factor`` as ``inf`` when a configuration had
no losing trades.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from exchange.core.timeframe import empirical_bars_per_year

TRADING_DAYS = 252.0


def _clean(returns) -> np.ndarray:
    arr = np.asarray(pd.Series(returns).to_numpy(), dtype=float)
    return arr[np.isfinite(arr)]


def infer_periods_per_year(returns: pd.Series, override: float | None = None) -> float:
    if override is not None:
        return float(override)
    if isinstance(returns, pd.Series) and isinstance(returns.index, pd.DatetimeIndex) and len(returns) >= 3:
        return empirical_bars_per_year(returns.index)
    return TRADING_DAYS


# --------------------------------------------------------------------------
# Return and risk
# --------------------------------------------------------------------------


def total_return(equity: pd.Series) -> float:
    """Cumulative return of an equity curve."""
    values = _clean(equity)
    if values.size < 2 or values[0] == 0:
        return np.nan
    return float(values[-1] / values[0] - 1.0)


def cagr(equity: pd.Series, *, periods_per_year: float | None = None) -> float:
    """Compound annual growth rate.

    Returns ``NaN`` rather than a complex number when the equity curve goes
    non-positive: a blown-up account has no meaningful growth rate, and
    reporting one hides the blow-up.
    """
    values = _clean(equity)
    if values.size < 2 or values[0] <= 0 or values[-1] <= 0:
        return np.nan
    ppy = infer_periods_per_year(equity, periods_per_year)
    years = (values.size - 1) / ppy
    if years <= 0:
        return np.nan
    # Computed in logs: a large growth multiple over a short window overflows
    # the direct power form, and an overflow warning is not an answer.
    with np.errstate(over="ignore", invalid="ignore"):
        growth = float(np.expm1(np.log(values[-1] / values[0]) / years))
    return growth if np.isfinite(growth) else np.nan


def annualised_volatility(returns: pd.Series, *, periods_per_year: float | None = None) -> float:
    values = _clean(returns)
    if values.size < 2:
        return np.nan
    ppy = infer_periods_per_year(returns, periods_per_year)
    return float(np.std(values, ddof=1) * np.sqrt(ppy))


def sharpe_ratio(
    returns: pd.Series,
    *,
    periods_per_year: float | None = None,
    risk_free: float = 0.0,
) -> float:
    """Annualised Sharpe ratio.

    ``risk_free`` is an *annual* rate and is converted to the return series'
    own sampling frequency before subtraction.
    """
    values = _clean(returns)
    if values.size < 2:
        return np.nan
    ppy = infer_periods_per_year(returns, periods_per_year)
    excess = values - risk_free / ppy
    mean = float(np.mean(excess))
    sd = float(np.std(excess, ddof=1))
    # A constant return series has no risk estimate. Floating-point noise
    # leaves its standard deviation at ~1e-19 rather than exactly zero, and
    # dividing by that reports a Sharpe of 1e16 — a number that looks like a
    # spectacular result and is in fact an absent denominator.
    if not np.isfinite(sd) or sd <= 1e-15 or sd <= 1e-10 * abs(mean):
        return np.nan
    return float(mean / sd * np.sqrt(ppy))


def sortino_ratio(
    returns: pd.Series,
    *,
    periods_per_year: float | None = None,
    target: float = 0.0,
) -> float:
    """Annualised Sortino ratio (downside deviation below ``target``).

    The downside deviation is computed over the **full** sample with upside
    deviations set to zero, not over the subset of negative observations.
    Using only the negative subset inflates the ratio for strategies that are
    rarely negative, which is precisely the population this metric is most
    often applied to.
    """
    values = _clean(returns)
    if values.size < 2:
        return np.nan
    ppy = infer_periods_per_year(returns, periods_per_year)
    excess = values - target / ppy
    downside = np.minimum(excess, 0.0)
    dd = np.sqrt(np.mean(downside**2))
    if dd <= 0:
        return np.nan
    return float(np.mean(excess) / dd * np.sqrt(ppy))


def max_drawdown(equity: pd.Series) -> float:
    """Deepest peak-to-trough decline, as a negative fraction."""
    values = _clean(equity)
    if values.size < 2:
        return np.nan
    peak = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, values / peak - 1.0, np.nan)
    return float(np.nanmin(dd))


def drawdown_series(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return (equity / peak - 1.0).rename("drawdown")


def max_drawdown_duration(equity: pd.Series) -> int:
    """Longest run of bars spent below a previous peak."""
    values = _clean(equity)
    if values.size < 2:
        return 0
    peak = np.maximum.accumulate(values)
    under = values < peak
    longest, run = 0, 0
    for flag in under:
        run = run + 1 if flag else 0
        longest = max(longest, run)
    return int(longest)


def calmar_ratio(equity: pd.Series, *, periods_per_year: float | None = None) -> float:
    """CAGR divided by the magnitude of the maximum drawdown."""
    dd = max_drawdown(equity)
    growth = cagr(equity, periods_per_year=periods_per_year)
    if not np.isfinite(dd) or dd >= 0 or not np.isfinite(growth):
        return np.nan
    return float(growth / abs(dd))


def ulcer_index(equity: pd.Series) -> float:
    """RMS drawdown — penalises depth *and* duration, unlike max drawdown."""
    dd = _clean(drawdown_series(equity))
    if dd.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(dd**2)))


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR as a positive loss magnitude."""
    values = _clean(returns)
    if values.size < 20:
        return np.nan
    return float(-np.quantile(values, 1.0 - confidence))


def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
    """Mean loss beyond the VaR threshold, as a positive magnitude."""
    values = _clean(returns)
    if values.size < 20:
        return np.nan
    threshold = np.quantile(values, 1.0 - confidence)
    tail = values[values <= threshold]
    return float(-np.mean(tail)) if tail.size else np.nan


def skewness(returns: pd.Series) -> float:
    values = _clean(returns)
    if values.size < 8:
        return np.nan
    from scipy import stats

    return float(stats.skew(values, bias=False))


def kurtosis(returns: pd.Series) -> float:
    """Excess kurtosis (0 for a Gaussian)."""
    values = _clean(returns)
    if values.size < 8:
        return np.nan
    from scipy import stats

    return float(stats.kurtosis(values, fisher=True, bias=False))


def tail_ratio(returns: pd.Series, q: float = 0.05) -> float:
    """Right tail over left tail. Below 1 means losses are fatter than gains."""
    values = _clean(returns)
    if values.size < 40:
        return np.nan
    left = abs(np.quantile(values, q))
    right = abs(np.quantile(values, 1 - q))
    return float(right / left) if left > 0 else np.nan


# --------------------------------------------------------------------------
# Trade statistics
# --------------------------------------------------------------------------


def trade_stats(trades: pd.DataFrame) -> dict[str, float]:
    """Win rate, profit factor, expectancy and holding period."""
    if trades is None or len(trades) == 0:
        return {
            "n_trades": 0, "win_rate": np.nan, "profit_factor": np.nan,
            "expectancy": np.nan, "avg_win": np.nan, "avg_loss": np.nan,
            "avg_bars_held": np.nan, "best_trade": np.nan, "worst_trade": np.nan,
        }
    pnl = trades["net_pnl"].to_numpy(float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_loss = -losses.sum()
    return {
        "n_trades": len(trades),
        "win_rate": float(len(wins) / len(pnl)),
        # An infinite profit factor (no losers) is not evidence of quality,
        # it is evidence of too few trades. Report NaN so it cannot win.
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else np.nan,
        "expectancy": float(pnl.mean()),
        "avg_win": float(wins.mean()) if wins.size else np.nan,
        "avg_loss": float(losses.mean()) if losses.size else np.nan,
        "avg_bars_held": float(trades["bars_held"].mean()),
        "best_trade": float(pnl.max()),
        "worst_trade": float(pnl.min()),
    }


# --------------------------------------------------------------------------
# The standard metric bundle
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PerformanceMetrics:
    """One uniform metric set for every result in the system."""

    n_bars: int
    periods_per_year: float
    total_return: float
    cagr: float
    annualised_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_bars: int
    ulcer_index: float
    var_95: float
    expected_shortfall_95: float
    skewness: float
    kurtosis: float
    tail_ratio: float
    n_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_bars_held: float
    total_costs: float
    cost_drag_pct: float
    exposure: float
    valid: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    def to_series(self) -> pd.Series:
        return pd.Series(self.to_dict())


def performance_metrics(result, *, periods_per_year: float | None = None) -> PerformanceMetrics:
    """Compute the standard bundle for a :class:`BacktestResult`."""
    equity = result.equity
    returns = result.returns
    ppy = periods_per_year if periods_per_year is not None else result.bars_per_year
    stats = trade_stats(result.trades)

    initial = float(equity.iloc[0]) if len(equity) else np.nan
    costs = result.total_costs
    exposure = float((result.position != 0).mean()) if len(result.position) else np.nan

    return PerformanceMetrics(
        n_bars=len(equity),
        periods_per_year=float(ppy),
        total_return=total_return(equity),
        cagr=cagr(equity, periods_per_year=ppy),
        annualised_volatility=annualised_volatility(returns, periods_per_year=ppy),
        sharpe_ratio=sharpe_ratio(returns, periods_per_year=ppy),
        sortino_ratio=sortino_ratio(returns, periods_per_year=ppy),
        calmar_ratio=calmar_ratio(equity, periods_per_year=ppy),
        max_drawdown=max_drawdown(equity),
        max_drawdown_bars=max_drawdown_duration(equity),
        ulcer_index=ulcer_index(equity),
        var_95=value_at_risk(returns),
        expected_shortfall_95=expected_shortfall(returns),
        skewness=skewness(returns),
        kurtosis=kurtosis(returns),
        tail_ratio=tail_ratio(returns),
        n_trades=stats["n_trades"],
        win_rate=stats["win_rate"],
        profit_factor=stats["profit_factor"],
        expectancy=stats["expectancy"],
        avg_bars_held=stats["avg_bars_held"],
        total_costs=costs,
        cost_drag_pct=float(100.0 * costs / initial) if initial and np.isfinite(initial) else np.nan,
        exposure=exposure,
        valid=result.valid,
    )


def metrics_frame(results: dict) -> pd.DataFrame:
    """Stack several results' metrics into one comparable table."""
    return pd.DataFrame(
        {name: performance_metrics(res).to_series() for name, res in results.items()}
    ).T
