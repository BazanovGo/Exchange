import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = next(p for p in Path.cwd().parents if (p / "research" / "ma_crossover_5m").is_dir())
sys.path.insert(0, str(ROOT / "research" / "ma_crossover_5m"))
import plots as P            # noqa: E402

WORK = Path(os.environ.get("MA5M_WORK", "/home/user/ma5m_work"))
STAGE = os.environ.get("MA5M_STAGE", "stage3")
A = WORK / "analysis"
TYPES = ["SMA", "EMA", "WMA", "SSMA", "VWMA", "TMA"]
FAST = np.arange(2, 100)
SLOW = np.arange(5, 251)
TICKERS = ["NG", "BR"]
NAMES = {"NG": "природный газ", "BR": "Brent"}
meta = {t: json.loads((WORK / STAGE / t / "meta.json").read_text()) for t in TICKERS}
res = {t: json.loads((A / STAGE / t / "results.json").read_text()) for t in TICKERS}
cube = lambda t, c: np.load(A / STAGE / t / f"cube_{c}.npy", mmap_mode="r")
picks = {t: pd.read_csv(A / STAGE / t / "picks.csv") for t in TICKERS}
print("этап:", STAGE, "| конфигураций на инструмент:", f"{res['NG']['n_configs']:,}")
for t in TICKERS:
    print(t, "| IS мес.", res[t]["is_months"], "| OOS мес.", res[t]["oos_months"],
          "| параметры наклона:", meta[t].get("per_method") or {k: meta[t][k] for k in ("k", "coverage", "atr_period")})
