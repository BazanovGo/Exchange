"""Full detail for selected configs: trades, MTM equity via vectorbt, calendar PnL.

The vectorbt portfolio trades one contract on the held-contract price path
(in USD), with orders one bar after the signal at the bar open and
``upon_opposite_entry="ignore"``; its closed-trade PnL reproduces the kernel's
trades exactly, and its bar-close valuation gives the mark-to-market curve
for Sharpe, Sortino and drawdowns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import vectorbt as vbt

from src.ma_slope.averages import atr, moving_average
from src.ma_slope.data import Instrument
from src.ma_slope.grid import Slope
from src.ma_slope.kernel import trade_list
from src.ma_slope.signals import DOWN, NEUTRAL, UP, cross_events, next_state, slope_state


@dataclass
class Config:
    fast_type: str
    fast_period: int
    slow_type: str
    slow_period: int
    slope: Slope
    model: int  # 0 reverse cross, 1 slope flip

    @property
    def label(self) -> str:
        exit_ = ("reverse_cross", "slope_flip")[self.model]
        return (f"{self.fast_type}{self.fast_period} / {self.slow_type}{self.slow_period}, "
                f"{self.slope.label}, exit={exit_}")


def signals(ins: Instrument, cfg: Config):
    fast = moving_average(cfg.fast_type, ins.close, ins.volume, cfg.fast_period)
    slow = moving_average(cfg.slow_type, ins.close, ins.volume, cfg.slow_period)
    state = slope_state(slow, cfg.slope.k, cfg.slope.eps, atr(ins.high, ins.low, ins.close, cfg.slope.atr_n),
                        cfg.slope.method)
    idx, d = cross_events(ins.close, fast)
    return idx, d, state


def trades(ins: Instrument, cfg: Config) -> pd.DataFrame:
    idx, d, state = signals(ins, cfg)
    rec, _, _ = trade_list(idx, d, cfg.model, state, next_state(state, DOWN), next_state(state, UP), ins.path,
                           ins.multiplier, ins.split, ins.minutes, ins.year_of_bar, ins.n_years)
    e, x = rec[:, 0].astype(int), rec[:, 1].astype(int)
    return pd.DataFrame({
        "entry_time": ins.index[e], "exit_time": ins.index[x], "direction": rec[:, 2].astype(int),
        "pnl": rec[:, 3], "bars": x - e, "hours": (ins.minutes[x] - ins.minutes[e]) / 60,
        "sample": np.where(x < ins.split, "is", "oos"),
    })


def portfolio(ins: Instrument, cfg: Config) -> vbt.Portfolio:
    idx, d, state = signals(ins, cfg)
    n = ins.n
    up = np.zeros(n, bool)
    down = np.zeros(n, bool)
    up[idx[d == 1]] = True
    down[idx[d == -1]] = True
    long_entry = up & ((state == UP) | (state == NEUTRAL))
    short_entry = down & ((state == DOWN) | (state == NEUTRAL))
    long_exit, short_exit = (down, up) if cfg.model == 0 else (state == DOWN, state == UP)

    def at_next_open(a):
        return pd.Series(np.r_[False, a[:-1]], index=ins.index)

    usd_open, usd_close = ins.path * ins.multiplier, ins.path_close * ins.multiplier
    # A small offset keeps the synthetic price positive without eating into
    # vectorbt's relative float tolerance (a 1e7 offset rounded 1-tick trades to 0).
    offset = 1.0 + 2.0 * max(np.abs(usd_open).max(), np.abs(usd_close).max())
    price_open = pd.Series(usd_open + offset, index=ins.index)
    price_close = pd.Series(usd_close + offset, index=ins.index)
    return vbt.Portfolio.from_signals(
        close=price_close, price=price_open,
        entries=at_next_open(long_entry), exits=at_next_open(long_exit),
        short_entries=at_next_open(short_entry), short_exits=at_next_open(short_exit),
        size=1.0, size_type="amount", init_cash=1e12, fees=0.0, slippage=0.0,
        upon_opposite_entry="ignore", accumulate=False, freq="5min")


def daily_pnl(pf: vbt.Portfolio) -> pd.Series:
    """Mark-to-market PnL per trading day (bars grouped by calendar date), USD per contract."""
    value = pf.value()
    return value.groupby(value.index.date).last().diff().dropna()


def risk_metrics(pnl_daily: pd.Series) -> dict:
    """Capital-free ratios of one-contract daily PnL, annualized by trading days per year."""
    days = pd.to_datetime(pnl_daily.index)
    span_years = (days[-1] - days[0]).days / 365.25 if len(days) > 1 else np.nan
    per_year = len(days) / span_years if span_years else np.nan
    mean, sd = pnl_daily.mean(), pnl_daily.std(ddof=1)
    downside = np.sqrt(np.mean(np.minimum(pnl_daily, 0) ** 2))
    equity = pnl_daily.cumsum()
    return {"trading_days_per_year": per_year,
            "sharpe": mean / sd * np.sqrt(per_year) if sd > 0 else np.nan,
            "sortino": mean / downside * np.sqrt(per_year) if downside > 0 else np.nan,
            "mtm_max_dd": float((equity.cummax().clip(lower=0) - equity).max())}


def calendar(tr: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """PnL by month (rows = years, columns = months) and by year, attributed to the exit time."""
    t = tr.assign(year=tr["exit_time"].dt.year, month=tr["exit_time"].dt.month)
    monthly = t.pivot_table(index="year", columns="month", values="pnl", aggfunc="sum").reindex(
        columns=range(1, 13))
    return monthly, t.groupby("year")["pnl"].sum()
