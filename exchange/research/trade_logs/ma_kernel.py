"""Посделочный журнал любой конфигурации сетки скользящих средних — событийным ядром.

Ядро (research/ma_crossover_5m/kernel.py) — реализация правил, независимая от
VectorBT: на нём посчитана вся сетка, а validate_vbt.py сверял его с VectorBT
сделка в сделку. Здесь оно даёт журнал для конфигураций, которые через VectorBT
не прогонялись (лучшие в семействах, шаги walk-forward), и пересчитывает
журналы финалистов для проверки (audit_trades.py).

Нужно окружение исследования (numba, pandas<3) и EXCHANGE_MOEX_DATA.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MA = HERE.parent / "ma_crossover_5m"
sys.path.insert(0, str(MA))

import kernel as K  # noqa: E402
import ma as M  # noqa: E402
from data import load_market, load_raw  # noqa: E402
from grid import slow_state  # noqa: E402

COLUMNS = ["№", "период", "нога", "вход", "выход", "направление", "объём при входе, USD",
           "котировка входа", "котировка выхода", "через ролл", "издержки, USD", "результат, USD",
           "баров в позиции", "примечание"]
OPEN_END_NOTE = "открыта на конец данных: закрыта по последнему Close"
PARAMS = ("method", "slow_type", "slow_period", "exit", "fast_type", "fast_period")


def slopes(stage: str, ticker: str) -> dict:
    """Параметры наклона по методу — те же, с которыми считалась сетка этапа."""
    finalists = json.loads((MA / "results" / stage / ticker / "finalists.json").read_text())
    out = {}
    for f in finalists:
        s = {k: f["slope"][k] for k in ("k", "coverage", "atr_period")}
        if out.setdefault(f["config"]["method"], s) != s:
            raise SystemExit(f"{stage}/{ticker}: разные параметры наклона у одного метода")
    return out


def label(cfg) -> str:
    return (f"{cfg['fast_type']}{int(cfg['fast_period'])} × {cfg['slow_type']}{int(cfg['slow_period'])}, "
            f"наклон {cfg['method']}, выход {cfg['exit']}")


class Kernel:
    """Один инструмент: ряды, кэш средних и пересечений, журнал по конфигурации."""

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.mk = load_market(ticker)
        raw = load_raw(ticker)
        if len(raw) != self.mk.n:
            raise SystemExit(f"{ticker}: число баров в сырых данных и в ряду ядра не совпадает")
        self.raw_open = raw["open"].to_numpy(float)
        self.raw_close = raw["close"].to_numpy(float)
        self.secid = raw["secid"].to_numpy()
        self._ma, self._events, self._atr = {}, {}, {}

    def _average(self, kind: str, n: int) -> np.ndarray:
        key = (kind, n)
        if key not in self._ma:
            self._ma[key] = M.moving_average(kind, self.mk.close, self.mk.volume, n)
        return self._ma[key]

    def simulate(self, cfg, slope) -> tuple[np.ndarray, np.ndarray]:
        """Сделки [бар входа, бар выхода, направление, цена входа, цена выхода, PnL] и статистика."""
        mk = self.mk
        fast_key = (cfg["fast_type"], int(cfg["fast_period"]))
        if fast_key not in self._events:
            self._events[fast_key] = K.cross_events(mk.close, self._average(*fast_key), mk.is_roll)
        ev_bar, ev_dir = self._events[fast_key]
        slow = self._average(cfg["slow_type"], int(cfg["slow_period"]))
        atr = None
        if cfg["method"] == "B_atr":
            p = int(slope["atr_period"])
            if p not in self._atr:
                self._atr[p] = M.atr(mk.high, mk.low, mk.close, p)
            atr = self._atr[p]
        state, _ = slow_state(mk, slow, cfg["method"], int(slope["k"]), float(slope["coverage"]), atr)
        nd, nu = K.next_state_index(state, -1), K.next_state_index(state, 1)
        stats = np.zeros((3, K.N_STATS))
        monthly = np.zeros(len(mk.months))
        log = np.zeros((len(ev_bar) + 1, 6))
        model = 0 if cfg["exit"] == "A_fast_cross" else 1
        n = K.simulate(ev_bar, ev_dir, state, nd, nu, mk.pnl_open, mk.pnl_close, mk.month,
                       mk.oos_bar, mk.multiplier, model, stats, monthly, log)
        return log[:n], stats

    def journal(self, cfg, slope) -> pd.DataFrame:
        """Журнал в колонках trade_logs/csv: котировки — Open бара исполнения."""
        return self.journal_months(cfg, slope)[0]

    def journal_months(self, cfg, slope) -> tuple[pd.DataFrame, np.ndarray]:
        """Журнал и номер месяца выхода каждой сделки (по нему сделка относится к месяцу)."""
        tr, stats = self.simulate(cfg, slope)
        mk = self.mk
        ein, eout = tr[:, 0].astype(np.int64), tr[:, 1].astype(np.int64)
        direction = tr[:, 2]
        open_end = np.zeros(len(tr), bool)
        if stats[0, K.OPENEND] > 0:          # открытой на конец данных может быть только последняя
            open_end[-1] = True
        entry_q = self.raw_open[ein]
        exit_q = np.where(open_end, self.raw_close[eout], self.raw_open[eout])
        rolled = self.secid[ein] != self.secid[eout]
        pnl = np.round(tr[:, 5], 2)
        df = pd.DataFrame({
            "№": np.arange(1, len(tr) + 1),
            "период": np.where(eout < mk.oos_bar, "IS", "OOS"),
            "нога": self.ticker,
            "вход": mk.index[ein], "выход": mk.index[eout],
            "направление": np.where(direction > 0, "long", "short"),
            "объём при входе, USD": entry_q * mk.multiplier,
            "котировка входа": entry_q, "котировка выхода": exit_q,
            "через ролл": np.where(rolled, "да", ""),
            "издержки, USD": 0.0,
            "результат, USD": pnl,
            "баров в позиции": eout - ein,
            "примечание": np.where(open_end, OPEN_END_NOTE, ""),
        })
        # Без ролла результат обязан совпасть с разностью котировок.
        quoted = (exit_q - entry_q) * direction * mk.multiplier
        bad = np.abs(quoted[~rolled] - pnl[~rolled]) > 0.011
        if bad.any():
            raise SystemExit(f"{self.ticker} {label(cfg)}: {bad.sum()} сделок не сходятся с котировками")
        return df[COLUMNS], mk.month[eout]


def config_params(stage: str, ticker: str, configs) -> dict[int, dict]:
    """Номер конфигурации сетки -> её параметры (из grid_<метод>.parquet этапа)."""
    wanted = sorted({int(c) for c in configs})
    out = {}
    for method in ("A_pct", "B_atr"):
        g = pd.read_parquet(MA / "results" / stage / ticker / f"grid_{method}.parquet",
                            columns=["config", *PARAMS], filters=[("config", "in", wanted)])
        for r in g.itertuples(index=False):
            out[int(r.config)] = {k: getattr(r, k) for k in PARAMS}
    missing = set(wanted) - set(out)
    if missing:
        raise SystemExit(f"{stage}/{ticker}: конфигурации не найдены в сетке: {sorted(missing)[:5]}")
    return out
