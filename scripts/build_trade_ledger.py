"""Trade ledger of the selected configurations: CSV, Excel workbook, figure and notebook 08.

    python scripts/build_trade_ledger.py
    python scripts/run_notebook.py notebooks/08_trade_ledger.ipynb

Every closed trade of each instrument's stage-3 finalist is re-simulated on
the 5m candles and checked against the simulator (PnL equals the held-contract
move, and the raw price move inside one contract). Outputs:

* ``research/ma_cross_slope_5m/tables/trade_ledger.csv`` — all trades, machine-readable;
* ``research/ma_cross_slope_5m/tables/trade_ledger.xlsx`` — summary, all trades by exit time and one
  sheet per instrument; running totals, drawdowns and streaks are formulas, computed when the file opens;
* ``research/ma_cross_slope_5m/figures/trade_outcomes.png`` and ``notebooks/08_trade_ledger.ipynb``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "08_trade_ledger.ipynb"

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# 08 — Все сделки лучших конфигураций: каждый исход по порядку

Для каждого инструмента взята лучшая конфигурация исследования «пересечение цены с быстрой MA + наклон медленной
MA» (этап 3, лучшая по IS-плато — та же, что в презентации [07](07_project_presentation.ipynb)). Ниже — все её
сделки за всю историю по порядку: вход, выход, цены, результат в долларах и тиках и накопленный итог.

* **Excel:** [`research/ma_cross_slope_5m/tables/trade_ledger.xlsx`](../research/ma_cross_slope_5m/tables/trade_ledger.xlsx).
  Листы: «Сводка», «Все сделки» (все инструменты в порядке закрытия) и по листу на инструмент. Накопленный итог,
  просадка и серии убытков — формулы.
* **CSV:** [`research/ma_cross_slope_5m/tables/trade_ledger.csv`](../research/ma_cross_slope_5m/tables/trade_ledger.csv).

**Как читать.** 1 контракт, без комиссий и проскальзывания (по ТЗ). Вход и выход — по Open свечи после сигнала;
цены — сырые цены контракта. Сделка, прошедшая через ролл, переходит в новый контракт по цене перекрытия, поэтому
её цены входа и выхода относятся к разным контрактам. **IS** — участок, на котором конфигурация выбрана, поэтому
там результат хорош по построению. Честная проверка — только **OOS** (последние 30 % истории).
""")

code("""
from IPython.display import Image, display

from src.ma_slope import ledger as L
from src.ma_slope.presentation import FIGS

led = L.load_ledger()
configs = led.drop_duplicates("ticker").set_index("ticker")["config"]
print(f"Сделок всего: {len(led):,}".replace(",", " "))
""")

md("## Сводка по инструментам")
code("""
import pandas as pd

pd.set_option("display.float_format", "{:,.2f}".format)
L.summary_table(led)
""")

md("""
## Все исходы по порядку

Для каждого инструмента: сверху накопленный итог по номеру сделки, снизу результат каждой сделки (синий — прибыль,
красный — убыток). Вертикальная линия — граница IS/OOS.
""")
code('display(Image(filename=str(FIGS / "trade_outcomes.png")))')

md("""
## Таблицы сделок

Каждая сделка по порядку. Колонки: номер, выборка, сторона, время и контракт входа, цена входа, время и контракт
выхода, цена выхода, длительность в часах, результат в долларах и в тиках, исход и накопленный итог.
""")
for t, name in (("ng", "NG"), ("br", "BR (Brent)"), ("gold", "GOLD (GC)"), ("silv", "SILV (SI)"), ("plt", "PLT (PL)"),
                ("pld", "PLD (PA)"), ("copper", "COPPER (HG)"), ("wti", "WTI (CL), справочно")):
    md(f"### {name}")
    code(f'print(configs["{t}"])\nprint(L.format_trades(led, "{t}"))')

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}

if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    from src.ma_slope import ledger as L
    from src.ma_slope.presentation import FIGS

    ledgers = L.all_ledgers()
    print(f"written {L.write_csv(ledgers).relative_to(ROOT)}")
    print(f"written {L.write_excel(ledgers).relative_to(ROOT)}")
    print(f"written {L.outcomes_figure(L.load_ledger(), FIGS / 'trade_outcomes.png').relative_to(ROOT)}")
    nbf.write(nb, OUT)
    print(f"written {OUT.relative_to(ROOT)}")
