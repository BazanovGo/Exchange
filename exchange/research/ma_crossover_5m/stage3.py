"""Этап 3: полная базовая сетка с параметрами наклона, выбранными на этапе 2.

Для каждого инструмента и метода берётся настройка (k, доля нейтрали, ATR)
с лучшей медианой IS-Шарпа по решётке этапа 2. Выбор сделан только по
in-sample; out-of-sample этой сетки — по-прежнему независимая выборка.
"""
import json
import os
import sys
from pathlib import Path

from data import load_market
from grid import run_base_grid

if __name__ == "__main__":
    work = Path(os.environ.get("MA5M_WORK", "work"))
    params = json.loads((work / "stage3" / "slope_params.json").read_text())
    for t in sys.argv[1:] or ("NG", "BR"):
        mk = load_market(t)
        pm = {m: {k: params[t][m][k] for k in ("k", "coverage", "atr_period")} for m in params[t]}
        print(f"== {t}: параметры наклона {pm}", flush=True)
        meta = run_base_grid(mk, work / "stage3" / t, per_method=pm, log_every=246)
        print(f"== {t} этап 3 готов за {meta['seconds'] / 60:.1f} мин", flush=True)
