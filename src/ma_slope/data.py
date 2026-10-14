"""Instrument data for the study: ratio back-adjusted 5m MOEX futures.

Signals use the back-adjusted series, so a roll never creates a crossing or
a slope change. Money PnL uses :func:`held_contract_path`: the raw price change
of the contract actually held, rolled at the overlap price (the new contract's
price at the old contract's last candle, from ``rolls.csv``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data import load_data

# USD per 1.0 of quoted price for one contract (MOEX FORTS specifications).
MULTIPLIERS = {
    "ng": 100.0,     # 100 MMBtu, USD/MMBtu
    "br": 10.0,      # 10 barrels, USD/bbl
    "wti": 10.0,     # 10 barrels, USD/bbl
    "gold": 1.0,     # 1 troy ounce, USD/oz
    "silv": 10.0,    # 10 troy ounces, USD/oz
    "plt": 1.0,      # 1 troy ounce, USD/oz
    "pld": 1.0,      # 1 troy ounce, USD/oz
    "copper": 0.01,  # 10 kg, USD per 1000 kg
}
# Minimum price step (current MOEX specification), for expressing trades in ticks.
TICK_SIZES = {"ng": 0.001, "br": 0.01, "wti": 0.01, "gold": 0.1, "silv": 0.01, "plt": 0.1, "pld": 0.01,
              "copper": 1.0}
TICKERS = tuple(MULTIPLIERS)
IS_FRACTION = 0.7


@dataclass
class Instrument:
    ticker: str
    index: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    secid: np.ndarray
    factor: np.ndarray      # back-adjustment factor (raw = adjusted / factor)
    path: np.ndarray        # cumulative raw-price change of the held contract, at opens
    minutes: np.ndarray     # bar start, minutes since epoch
    year_of_bar: np.ndarray  # calendar year - first_year
    first_year: int
    n_years: int
    split: int              # first out-of-sample bar
    multiplier: float

    @property
    def n(self) -> int:
        return self.close.size

    @property
    def split_time(self) -> pd.Timestamp:
        return self.index[self.split]

    @property
    def path_close(self) -> np.ndarray:
        """:attr:`path` marked at bar closes instead of opens."""
        return self.path + (self.close - self.open) / self.factor


def held_contract_path(open_: np.ndarray, close: np.ndarray, secid: np.ndarray,
                       factor: np.ndarray) -> np.ndarray:
    """Cumulative raw price change of one long contract marked at bar opens.

    Within a contract the step is the raw open-to-open change. Across a roll
    (``secid`` changes between bars ``t-1`` and ``t``) the old contract is held
    to its last close and the new one from the overlap price: in back-adjusted
    terms both legs meet at ``close[t-1]``, each scaled back by its own factor.
    """
    step = np.zeros(open_.size)
    same = secid[1:] == secid[:-1]
    raw_move = (open_[1:] - open_[:-1]) / factor[1:]
    roll_move = (close[:-1] - open_[:-1]) / factor[:-1] + (open_[1:] - close[:-1]) / factor[1:]
    step[1:] = np.where(same, raw_move, roll_move)
    return np.cumsum(step)


def load_instrument(ticker: str, timeframe: str = "5m", is_fraction: float = IS_FRACTION) -> Instrument:
    df = load_data("moex_algopack", ticker, timeframe=timeframe).get()
    if not (df["Volume"] > 0).all():
        raise ValueError(f"{ticker}: bars without volume, VWMA would be undefined")
    o, h, lo, c = (df[k].to_numpy(np.float64) for k in ("Open", "High", "Low", "Close"))
    secid = df["secid"].to_numpy()
    factor = df["adj_factor"].to_numpy(np.float64)
    years = df.index.year.to_numpy()
    first = int(years[0])
    return Instrument(
        ticker=ticker,
        index=df.index,
        open=o, high=h, low=lo, close=c,
        volume=df["Volume"].to_numpy(np.float64),
        secid=secid,
        factor=factor,
        path=held_contract_path(o, c, secid, factor),
        minutes=(df.index.asi8 // 60_000_000_000).astype(np.float64),
        year_of_bar=(years - first).astype(np.int64),
        first_year=first,
        n_years=int(years[-1]) - first + 1,
        split=int(len(df) * is_fraction),
        multiplier=MULTIPLIERS[ticker],
    )
