"""Данные: 5-минутные свечи MOEX AlgoPack, роллы, два скорректированных ряда.

Почему рядов два
----------------
Вендор склеил контракты по ближайшей экспирации и цены не корректировал: на
каждом переходе в ряду остаётся ступень (NG — медиана +384 б.п., BR — −31 б.п.).
Скользящая средняя, посчитанная по такому ряду, «видит» ступень как движение
цены и рождает пересечения, которых на рынке не было.

* **Сигнальный ряд** — ratio-скорректированный назад: все цены до ролла
  умножаются на ``open_new / close_old``. Последний сегмент совпадает с
  котировками; относительные изменения внутри сегментов сохранены, поэтому
  процентный наклон (метод A) и ATR остаются осмысленными.
* **Ряд для PnL** — аддитивно скорректированный: из цен после ролла вычитается
  накопленная ступень ``open_new − close_old``. Разность двух его значений — это
  ровно результат одного контракта, переложенного на ролле без издержек. Знак и
  уровень этого ряда значения не имеют: в PnL входят только разности.

Ограничение, которое нельзя снять на этих данных: одновременных котировок
старого и нового контракта нет, поэтому ступень измеряется как
``open_new − close_old`` и включает одно ночное движение. Это ошибка порядка
одного ночного движения на ролл (≈ 1 раз в месяц) для позиций, переживающих ролл.

Время — московское, как отдаёт MOEX; месяцы и граница in-sample/out-of-sample
считаются по московскому календарю.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("EXCHANGE_MOEX_DATA", "data/raw/moex_algopack"))
OOS_START = pd.Timestamp("2024-01-01")

# Спецификации MOEX: стандартный BR — 10 баррелей, NG — 100 MMBtu
# (мини-BRM = 1 баррель «в 10 раз меньше», микро-NGM = 1 MMBtu «в 100 раз
# меньше», https://www.moex.com/n75212). Цена в USD за единицу базового актива,
# поэтому денежный PnL одного контракта = Δцены × множитель, в USD.
SPECS = {
    "BR": {"multiplier": 10.0, "tick": 0.01, "unit": "USD/bbl", "name": "Brent"},
    "NG": {"multiplier": 100.0, "tick": 0.001, "unit": "USD/MMBtu", "name": "Natural gas"},
}


@dataclass
class Market:
    """Всё, что нужно ядру, в виде плоских numpy-массивов."""

    ticker: str
    index: pd.DatetimeIndex
    open: np.ndarray            # сигнальный ряд (ratio)
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    pnl_open: np.ndarray        # ряд для PnL (аддитивный): исполнение по Open
    pnl_close: np.ndarray       # он же по Close: принудительное закрытие в конце
    raw_close: np.ndarray       # котировки как есть — только для диагностики
    is_roll: np.ndarray         # True на первом баре нового контракта
    month: np.ndarray           # номер месяца бара, 0..n_months-1
    months: pd.PeriodIndex
    oos_bar: int                # первый бар out-of-sample
    oos_month: int              # первый месяц out-of-sample
    multiplier: float
    tick: float
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.close)


def load_raw(ticker: str, root: Path | None = None) -> pd.DataFrame:
    root = Path(root or DATA)
    files = sorted((root / ticker.lower() / "5m").glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"нет 5m-файлов для {ticker} в {root}")
    df = pd.concat(pd.read_csv(f, parse_dates=["begin"]) for f in files)
    df = df.sort_values("begin").drop_duplicates("begin").set_index("begin")
    return df[["secid", "open", "high", "low", "close", "volume"]]


def load_market(ticker: str, root: Path | None = None) -> Market:
    ticker = ticker.upper()
    df = load_raw(ticker, root)
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    v = df["volume"].to_numpy(float)
    secid = df["secid"].to_numpy()

    roll = np.zeros(len(df), dtype=bool)
    roll[1:] = secid[1:] != secid[:-1]
    r = np.flatnonzero(roll)

    # Ступень на ролле: open нового контракта против close старого.
    gap = o[r] - c[r - 1]
    ratio = o[r] / c[r - 1]

    # Ratio назад: множитель сегмента = произведение отношений всех ПОСЛЕДУЮЩИХ роллов.
    factor = np.ones(len(df))
    seg = np.cumsum(roll)                       # номер сегмента бара
    cum_from_end = np.cumprod(ratio[::-1])[::-1]  # для сегмента s — произведение ratio[s:]
    seg_factor = np.r_[cum_from_end, 1.0]
    factor = seg_factor[seg]

    # Аддитивно вперёд: из сегмента s вычитается сумма ступеней gap[:s].
    seg_shift = np.r_[0.0, np.cumsum(gap)]
    shift = seg_shift[seg]

    idx = pd.DatetimeIndex(df.index)
    months = idx.to_period("M")
    uniq = pd.PeriodIndex(sorted(months.unique()), freq="M")
    month = uniq.get_indexer(months).astype(np.int32)
    oos_bar = int(np.searchsorted(idx.values, OOS_START.to_datetime64()))
    oos_month = int(uniq.get_indexer([OOS_START.to_period("M")])[0])

    spec = SPECS[ticker]
    return Market(
        ticker=ticker, index=idx,
        open=o * factor, high=h * factor, low=l * factor, close=c * factor, volume=v,
        pnl_open=o - shift, pnl_close=c - shift, raw_close=c,
        is_roll=roll, month=month, months=uniq, oos_bar=oos_bar, oos_month=oos_month,
        multiplier=spec["multiplier"], tick=spec["tick"],
        meta={
            "n_rolls": int(roll.sum()),
            "roll_gap_bps_median": float(np.median((ratio - 1) * 1e4)),
            "first_bar": str(idx[0]), "last_bar": str(idx[-1]),
            "zero_volume_bars": int((v <= 0).sum()),
            "unit": spec["unit"], "name": spec["name"],
        },
    )
