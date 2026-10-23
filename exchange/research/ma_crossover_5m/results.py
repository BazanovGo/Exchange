"""Чтение результатов сетки и производные метрики по каждой конфигурации."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import kernel as K

AXES = ("method", "slow_type", "slow_period", "exit", "fast_type", "fast_period")


@dataclass
class Grid:
    ticker: str
    meta: dict
    stats: np.ndarray        # (N, 3, N_STATS) memmap
    monthly: np.ndarray      # (N, n_months) memmap
    shape: tuple
    oos_month: int

    @property
    def n(self) -> int:
        return self.stats.shape[0]

    def decode(self, flat) -> pd.DataFrame:
        flat = np.atleast_1d(np.asarray(flat))
        idx = np.unravel_index(flat, self.shape)
        m = self.meta
        df = pd.DataFrame({
            "config": flat,
            "method": np.asarray(m["methods"])[idx[0]],
            "slow_type": np.asarray(m["types"])[idx[1]],
            "slow_period": idx[2] + m["slow_periods"][0],
            "exit": np.asarray(m["exits"])[idx[3]],
            "fast_type": np.asarray(m["types"])[idx[4]],
            "fast_period": idx[5] + m["fast_periods"][0],
        })
        return df

    def axis_values(self):
        """Значение каждой оси для каждой конфигурации (int-коды)."""
        return np.unravel_index(np.arange(self.n), self.shape)


def load_grid(path: Path) -> Grid:
    meta = json.loads((path / "meta.json").read_text())
    stats = np.load(path / "stats.npy", mmap_mode="r")
    monthly = np.load(path / "monthly.npy", mmap_mode="r")
    return Grid(meta["ticker"], meta, stats, monthly, tuple(meta["shape"]), meta["oos_month"])


def metrics(g: Grid, bucket: int, chunk: int = 500_000) -> pd.DataFrame:
    """Метрики раздела 9 задания для одной корзины (0 full, 1 IS, 2 OOS)."""
    s = np.asarray(g.stats[:, bucket, :], dtype=np.float64)
    n = s[:, K.NTR]
    wins = s[:, K.WINS]
    losses = n - wins
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {
            "pnl": s[:, K.PNL],
            "trades": n,
            "win_rate": np.where(n > 0, wins / n, np.nan),
            "profit_factor": np.where(s[:, K.GL] > 0, s[:, K.GP] / s[:, K.GL], np.nan),
            "avg_trade": np.where(n > 0, s[:, K.PNL] / n, np.nan),
            "avg_win": np.where(wins > 0, s[:, K.GP] / wins, np.nan),
            "avg_loss": np.where(losses > 0, -s[:, K.GL] / losses, np.nan),
            "max_dd": s[:, K.MAXDD],
            "avg_bars": np.where(n > 0, s[:, K.SUMDUR] / n, np.nan),
            "max_loss_streak": s[:, K.MAXLS],
            "long_share": np.where(n > 0, s[:, K.NLONG] / n, np.nan),
            "pnl_long": s[:, K.PNLLONG],
            "open_at_end": s[:, K.OPENEND],
        }
    df = pd.DataFrame({k: v.astype(np.float32) for k, v in out.items()})
    df["avg_trade_ticks"] = (df["avg_trade"] / (g.meta["tick"] * g.meta["multiplier"])).astype(np.float32)
    return df


def monthly_moments(g: Grid, lo: int, hi: int, chunk: int = 200_000):
    """Среднее, std и доля положительных месяцев на [lo, hi) по каждой конфигурации."""
    N = g.n
    mean = np.empty(N, np.float32)
    std = np.empty(N, np.float32)
    pos = np.empty(N, np.float32)
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        x = np.asarray(g.monthly[a:b, lo:hi], dtype=np.float64)
        mean[a:b] = x.mean(axis=1)
        std[a:b] = x.std(axis=1, ddof=1)
        pos[a:b] = (x > 0).mean(axis=1)
    return mean, std, pos


def sharpe_monthly(mean, std):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(std > 0, mean / std * np.sqrt(12.0), np.nan).astype(np.float32)


def summary_frame(g: Grid) -> pd.DataFrame:
    """Одна строка на конфигурацию: оси + IS/OOS/full метрики + месячный Шарп."""
    df = g.decode(np.arange(g.n))
    for b, tag in ((0, "full"), (1, "is"), (2, "oos")):
        m = metrics(g, b)
        m.columns = [f"{c}_{tag}" for c in m.columns]
        df = pd.concat([df, m], axis=1)
    nm = g.monthly.shape[1]
    for lo, hi, tag in ((0, g.oos_month, "is"), (g.oos_month, nm, "oos"), (0, nm, "full")):
        mu, sd, pos = monthly_moments(g, lo, hi)
        df[f"sharpe_m_{tag}"] = sharpe_monthly(mu, sd)
        df[f"pos_months_{tag}"] = pos
    df["fast_ge_slow"] = df["fast_period"] >= df["slow_period"]
    for c in ("method", "slow_type", "exit", "fast_type"):
        df[c] = df[c].astype("category")
    return df
