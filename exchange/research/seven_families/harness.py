"""Единый стенд: данные, carry-прокси, издержки, сайзинг.

Всё исполнение идёт через движок пакета (next-bar по построению), чтобы
результаты нельзя было получить по цене, которой не было.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from exchange.backtest.engine import BacktestConfig, InstrumentSpec, run_backtest
from exchange.backtest.metrics import performance_metrics
from exchange.core.timeframe import empirical_bars_per_year
from exchange.data.moex import moex_continuous
from exchange.execution.assumptions import ExecutionAssumptions
from exchange.execution.costs import CommissionModel, CostModel, ImpactModel, SlippageModel, SpreadModel

DATA = os.environ.get("EXCHANGE_MOEX_DATA", "data/raw/moex")
CAPITAL = 1_000_000.0
TARGET_VOL = 0.15          # годовая цель волатильности позиции
VOL_WINDOW = 20
MAX_LEVERAGE = 3.0


def bps_costs(scale: float = 1.0) -> CostModel:
    """Издержки в базисных пунктах от ноционала — не зависят от множителя контракта.

    База: полуспред 2 б.п. на сторону, слиппедж 1 б.п. + 5% от волатильности
    бара, комиссия 0.5 б.п. от ноционала. Итого ~7 б.п. за полный оборот.
    Фиксированная часть комиссии FORTS (рубли за контракт) обнулена: на
    ноционале BR это ~0.4 б.п., и её учёт потребовал бы множителя и курса.
    """
    return CostModel(
        commission=CommissionModel(bps_of_notional=0.5 * scale),
        spread=SpreadModel(half_spread_bps=2.0 * scale),
        slippage=SlippageModel(bps=1.0 * scale, volatility_multiple=0.05 * scale),
        impact=ImpactModel(coefficient=0.0, max_participation=1.0),
    )


SCENARIOS = {
    "optimistic": 0.5,
    "base": 1.0,
    "conservative": 2.0,
    "stressed": 3.0,
}


def execution(scenario: str = "base", latency: int = 0) -> ExecutionAssumptions:
    return ExecutionAssumptions(
        costs=bps_costs(SCENARIOS[scenario]),
        fill_timing="next_open",
        latency_bars=latency,
        allow_partial_fills=False,
        label=scenario,
        notes=f"издержки в б.п. от ноционала, масштаб {SCENARIOS[scenario]}x",
    )


def daily_bars(series):
    bars = series.bars.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum", "is_roll": "max"}
    ).dropna(subset=["close"])
    bars["is_roll"] = bars["is_roll"].astype(bool)
    return bars


def carry_series(raw_series, index) -> pd.Series:
    """Прокси carry: наблюдаемый спред между контрактами на переходе.

    На каждом ролле известно отношение цены нового контракта к старому. Это и
    есть наклон срочной структуры в этот момент — единственное, что можно
    извлечь из предсклеенного ряда без панели контрактов.

    carry = -ln(ratio) * 365 / дней_между_роллами: положительный = бэквордация
    = положительный перенос для длинной позиции.

    Значение известно В МОМЕНТ ролла, поэтому переносится вперёд от этой даты.
    Никакого будущего ролла в нём нет.
    """
    rolls = raw_series.provenance.rolls
    if not rolls:
        return pd.Series(0.0, index=index, name="carry")
    stamps = pd.DatetimeIndex([r.timestamp for r in rolls])
    ratios = np.array([r.ratio for r in rolls], dtype=float)
    gaps = np.diff(stamps.to_numpy()).astype("timedelta64[D]").astype(float)
    gaps = np.r_[gaps[0] if len(gaps) else 30.0, gaps]
    annualised = -np.log(ratios) * 365.0 / np.clip(gaps, 1.0, None)
    raw = pd.Series(annualised, index=stamps).sort_index()
    daily = raw.resample("1D").last()
    return daily.reindex(index.union(daily.index)).ffill().reindex(index).fillna(0.0).rename("carry")


def load_universe():
    """NG и BR: дневные бары, флаг ролла, carry-прокси, торгуемые доходности."""
    universe = {}
    for ticker in ("ng", "br"):
        raw = moex_continuous(DATA, ticker, "60m", adjustment="none")
        adjusted = moex_continuous(DATA, ticker, "60m", adjustment="ratio")
        bars = daily_bars(adjusted)              # сигналы считаются по непрерывному ряду
        quoted = daily_bars(raw)
        bars["carry"] = carry_series(raw, bars.index)
        tradeable = (1.0 + raw.tradeable_returns().fillna(0.0)).resample("1D").prod() - 1.0
        universe[raw.root] = {
            "bars": bars,
            "quoted": quoted,
            "returns": tradeable.reindex(bars.index).fillna(0.0),
            "ppy": empirical_bars_per_year(bars.index),
            "raw": raw,
        }
    return universe


def vol_target_units(bars: pd.DataFrame, ppy: float) -> pd.Series:
    """Размер позиции под целевую волатильность. Все входы трейлинговые."""
    vol = bars["close"].pct_change().rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std(ddof=1)
    annual = vol * np.sqrt(ppy)
    units = CAPITAL * TARGET_VOL / (annual.replace(0.0, np.nan) * bars["close"])
    cap = CAPITAL * MAX_LEVERAGE / bars["close"]
    return units.clip(upper=cap).fillna(0.0)


def backtest(bars, target, *, symbol, ppy, scenario="base", latency=0):
    cfg = BacktestConfig(
        initial_cash=CAPITAL,
        instrument=InstrumentSpec(symbol=symbol, multiplier=1.0, allow_fractional=True),
        execution=execution(scenario, latency),
        annualisation=ppy,
    )
    result = run_backtest(bars, target, config=cfg, symbol=symbol)
    return result, performance_metrics(result, periods_per_year=ppy)


REBALANCE_BAND = 0.20       # не трогать позицию, пока отклонение меньше 20%


def apply_band(target: pd.Series, band: float = REBALANCE_BAND) -> pd.Series:
    """Ребалансировать, только когда позиция отошла от цели больше чем на band.

    Волатильностный сайзинг пересчитывает целевой размер каждый бар, и без
    полосы стратегия торгует ежедневно: на 3 828 барах это 1 796 сделок и
    издержки в 16% капитала за период. Ни один систематический фонд так не
    торгует — полоса допуска есть у всех, и её отсутствие превращает
    сравнение семейств в сравнение их оборачиваемости.

    Причинно: решение на баре t зависит только от цели на t и удерживаемой
    позиции, которая известна из прошлого.
    """
    values = target.to_numpy(float)
    held = np.zeros(len(values))
    current = 0.0
    for i, want in enumerate(values):
        if current == 0.0:
            if want != 0.0:
                current = want
        elif want == 0.0 or np.sign(want) != np.sign(current):
            current = want
        elif abs(want - current) > band * abs(current):
            current = want
        held[i] = current
    return pd.Series(held, index=target.index, name="target_position")
