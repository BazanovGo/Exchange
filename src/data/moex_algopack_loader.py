"""MOEX AlgoPack source: continuous futures from ``data/raw/moex_algopack``.

Prices are ratio back-adjusted by default: at every roll the history before it
is multiplied by the new/old contract price ratio from ``<ticker>/rolls.csv``.
Percentage moves are then those of the contract actually held, the roll gap
disappears and the latest contract keeps its real prices. ``adjust=None``
returns the raw glued prices. Extra columns: ``secid`` (contract of the bar)
and, when adjusted, ``adj_factor`` (raw price = adjusted / adj_factor).

    load_data("moex_algopack", ["ng", "br"], timeframe="60m", start="2022-01-01")
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from src.data.base import BaseDataLoader
from src.data.registry import register_loader
from src.utils.paths import RAW_DATA_DIR

TIMEFRAMES = {"1m": "1m", "1min": "1m", "5m": "5m", "5min": "5m", "60m": "60m", "60min": "60m", "1h": "60m"}
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
_RENAME = {"begin": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}


@register_loader
class MoexAlgopackLoader(BaseDataLoader):
    """Loads ``{base_dir}/{ticker}/{1m|5m|60m}/<year>.csv`` (tickers: ng, br, gold, ...)."""

    source_name: ClassVar[str] = "moex_algopack"

    def __init__(self, base_dir: str | Path = RAW_DATA_DIR / "moex_algopack") -> None:
        self.base_dir = Path(base_dir)

    def _load_symbol(self, symbol: str, *, start=None, end=None, timeframe: str = "60m",
                     adjust: str | None = "ratio", **kwargs) -> pd.DataFrame:
        tf = TIMEFRAMES.get(str(timeframe).lower())
        if tf is None:
            raise ValueError(f"moex_algopack has 1m, 5m and 60m candles, not {timeframe!r}")
        if adjust not in ("ratio", None):
            raise ValueError(f"adjust must be 'ratio' or None, got {adjust!r}")
        ticker = symbol.lower()
        files = sorted((self.base_dir / ticker / tf).glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"no {tf} candles for {symbol!r} in {self.base_dir / ticker}")
        # Year files outside [start, end] are skipped; rolls.csv covers the rest.
        lo = pd.Timestamp(start).year if start is not None else None
        hi = pd.Timestamp(end).year if end is not None else None
        files = [f for f in files if (lo is None or int(f.stem) >= lo) and (hi is None or int(f.stem) <= hi)]
        if not files:
            raise ValueError(f"moex_algopack: no data for {symbol} in [{start}, {end}]")

        df = pd.concat((pd.read_csv(f, usecols=["secid", *_RENAME]) for f in files), ignore_index=True)
        df = df.rename(columns=_RENAME)
        df["Date"] = pd.to_datetime(df["Date"])
        if adjust == "ratio":
            factor = ratio_factors(df["Date"], df["secid"], self.base_dir / ticker / "rolls.csv")
            df[PRICE_COLUMNS] = df[PRICE_COLUMNS].mul(factor, axis=0)
            df["adj_factor"] = factor
        return df.set_index("Date")


def ratio_factors(times: pd.Series, secids: pd.Series, rolls_path: Path) -> np.ndarray:
    """Back-adjustment factor of every bar: product of the ratios of all later rolls.

    A roll applies to the bars up to and including the old contract's last
    candle (``from_time``). Raises if a contract change in the data has no
    matching roll, e.g. after new candles were added without ``--rolls-only``.
    """
    if not rolls_path.exists():
        raise FileNotFoundError(
            f"{rolls_path} not found: run scripts/download_moex_algopack.py --rolls-only")
    rolls = pd.read_csv(rolls_path, parse_dates=["from_time", "to_time"])
    roll_times = rolls["from_time"].to_numpy()
    later = np.append(np.cumprod(rolls["ratio"].to_numpy()[::-1])[::-1], 1.0)
    t = times.to_numpy()
    factor = later[np.searchsorted(roll_times, t, side="left")]

    change = np.flatnonzero(secids.to_numpy()[1:] != secids.to_numpy()[:-1]) + 1
    expected = {(a, b): ft for a, b, ft in zip(rolls["from_secid"], rolls["to_secid"], roll_times, strict=True)}
    for i in change:
        pair = (secids.iat[i - 1], secids.iat[i])
        ft = expected.get(pair)
        if ft is None or not t[i - 1] <= ft < t[i]:
            raise ValueError(f"{rolls_path} is out of sync with the candles at the {pair[0]} -> {pair[1]} "
                             "roll: rebuild it with scripts/download_moex_algopack.py --rolls-only")
    return factor
