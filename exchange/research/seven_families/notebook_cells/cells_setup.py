import os
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Исследовательские скрипты — источник истины: графики показывают ровно те
# сделки, которые попали в отчёт, а не их пересказ.
ROOT = next(p for p in Path.cwd().parents if (p / "research" / "seven_families").is_dir())
sys.path.insert(0, str(ROOT / "research" / "seven_families"))

CANDIDATES = [ROOT / "data" / "raw" / "moex_algopack", Path("/home/user/moex")]
DATA = next((p for p in CANDIDATES if p.is_dir()), None)
if DATA is None:
    raise FileNotFoundError(f"не найден каталог AlgoPack; искал: {[str(p) for p in CANDIDATES]}")
os.environ["EXCHANGE_MOEX_DATA"] = str(DATA)

import families as F          # noqa: E402
import harness as H           # noqa: E402
import run_all as RA          # noqa: E402

print("данные:", DATA)
print("капитал", f"{H.CAPITAL:,.0f}", "| целевая волатильность", H.TARGET_VOL,
      "| полоса ребалансировки", H.REBALANCE_BAND)
