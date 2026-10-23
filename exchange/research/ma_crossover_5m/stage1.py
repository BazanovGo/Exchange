"""Этап 1: полная базовая сетка 98 × 246 × 36 × 2 × 2 = 3 471 552 на инструмент.

Параметры наклона на этом этапе фиксированы ДО запуска и не подбирались по
результатам: k = 5 баров (25 минут: короче — шум одного бара, длиннее —
запаздывание сверх запаздывания самой средней), доля нейтрали на IS 25%,
ATR Уайлдера с периодом 14 (стандарт). Их оптимизация — этап 2.
"""
import os
import sys
from pathlib import Path

from data import load_market
from grid import run_base_grid

REFERENCE = {"k": 5, "coverage": 0.25, "atr_period": 14}

if __name__ == "__main__":
    work = Path(os.environ.get("MA5M_WORK", "work"))
    for ticker in sys.argv[1:] or ("NG", "BR"):
        mk = load_market(ticker)
        print(f"== {ticker}: {mk.n:,} баров, {len(mk.months)} мес., роллов {mk.meta['n_rolls']}", flush=True)
        meta = run_base_grid(mk, work / "stage1" / ticker, **REFERENCE, log_every=246)
        print(f"== {ticker} готово за {meta['seconds'] / 60:.1f} мин, событий {meta['n_events_total']:,}", flush=True)
