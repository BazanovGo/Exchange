"""Trade ledgers: every trade in order, for the finalists (08) and for walk-forward (09).

    python scripts/build_trade_ledger.py
    python scripts/run_notebook.py notebooks/08_trade_ledger.ipynb
    python scripts/run_notebook.py notebooks/09_walk_forward_ledger.ipynb

Trades are re-simulated on the 5m candles and checked against the study tables (per-trade PnL of the
finalists, trade counts and PnL of every walk-forward test year). Outputs in
``research/ma_cross_slope_5m``:

* ``tables/trade_ledger.{csv,xlsx}`` — every trade of each instrument's stage-3 finalist;
* ``tables/trade_ledger_walk_forward.{csv,xlsx}`` — every trade of the yearly re-selection (stage 1);
* ``figures/trade_outcomes.png``, ``figures/trade_outcomes_walk_forward.png``;
* ``notebooks/08_trade_ledger.ipynb``, ``notebooks/09_walk_forward_ledger.ipynb``.

In the workbooks running totals, drawdowns, streaks and summaries are formulas, computed when the file
opens (the committed files also carry the values, recalculated with LibreOffice).
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_FINALISTS = ROOT / "notebooks" / "08_trade_ledger.ipynb"
NB_WALK_FORWARD = ROOT / "notebooks" / "09_walk_forward_ledger.ipynb"
INSTRUMENTS = (("ng", "NG"), ("br", "BR (Brent)"), ("gold", "GOLD (GC)"), ("silv", "SILV (SI)"),
               ("plt", "PLT (PL)"), ("pld", "PLD (PA)"), ("copper", "COPPER (HG)"), ("wti", "WTI (CL), справочно"))


def notebook(cells) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python",
                                     "name": "python3"}, "language_info": {"name": "python"}}
    return nb


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


FINALISTS = [
    md("""
# 08 — Все сделки лучших конфигураций: каждый исход по порядку

Для каждого инструмента взята лучшая конфигурация исследования «пересечение цены с быстрой MA + наклон медленной
MA» (этап 3, лучшая по IS-плато — та же, что в презентации [07](07_project_presentation.ipynb)). Ниже — все её
сделки за всю историю по порядку: вход, выход, цены, результат в долларах и тиках и накопленный итог. Сделки
ежегодного переотбора параметров (walk-forward) — в ноутбуке [09](09_walk_forward_ledger.ipynb).

* **Excel:** [`research/ma_cross_slope_5m/tables/trade_ledger.xlsx`](../research/ma_cross_slope_5m/tables/trade_ledger.xlsx).
  Листы: «Сводка», «Все сделки» (все инструменты в порядке закрытия) и по листу на инструмент. Накопленный итог,
  просадка и серии убытков — формулы.
* **CSV:** [`research/ma_cross_slope_5m/tables/trade_ledger.csv`](../research/ma_cross_slope_5m/tables/trade_ledger.csv).

**Как читать.** 1 контракт, без комиссий и проскальзывания (по ТЗ). Вход и выход — по Open свечи после сигнала;
цены — сырые цены контракта. Сделка, прошедшая через ролл, переходит в новый контракт по цене перекрытия, поэтому
её цены входа и выхода относятся к разным контрактам. **IS** — участок, на котором конфигурация выбрана, поэтому
там результат хорош по построению. Честная проверка — только **OOS** (последние 30 % истории).
"""),
    code("""
import pandas as pd
from IPython.display import Image, display

from src.ma_slope import ledger as L
from src.ma_slope.presentation import FIGS

led = L.load_ledger(L.LEDGER_CSV)
configs = led.drop_duplicates("ticker").set_index("ticker")["config"]
pd.set_option("display.float_format", "{:,.2f}".format)
print(f"Сделок всего: {len(led):,}".replace(",", " "))
"""),
    md("## Сводка по инструментам"),
    code("L.summary_table(led)"),
    md("""
## Все исходы по порядку

Для каждого инструмента: сверху накопленный итог по номеру сделки, снизу результат каждой сделки (синий — прибыль,
красный — убыток). Вертикальная линия — граница IS/OOS.
"""),
    code('display(Image(filename=str(FIGS / "trade_outcomes.png")))'),
    md("""
## Таблицы сделок

Каждая сделка по порядку. Колонки: номер, выборка, сторона, время и контракт входа, цена входа, время и контракт
выхода, цена выхода, длительность в часах, результат в долларах и в тиках, исход и накопленный итог.
"""),
]
for t, name in INSTRUMENTS:
    FINALISTS += [md(f"### {name}"), code(f'print(configs["{t}"])\nprint(L.format_trades(led, "{t}"))')]

WALK_FORWARD = [
    md("""
# 09 — Walk-forward: все сделки ежегодного переотбора по порядку

Самый сильный результат исследования — ежегодный переотбор параметров. Каждый календарный год торгуется
конфигурация, выбранная по трём предыдущим годам среди всех 3,47 млн конфигураций этапа 1 (наклон медленной MA
задан заранее: k = 10). Критерий выбора — сглаженная по соседним периодам t-статистика средней сделки на обучении.
Ниже — все сделки этих конфигураций по порядку, год за годом.

* **Excel:** [`research/ma_cross_slope_5m/tables/trade_ledger_walk_forward.xlsx`](../research/ma_cross_slope_5m/tables/trade_ledger_walk_forward.xlsx).
  Листы: «Сводка», «По годам» (конфигурация и результат каждого тестового года со сверкой с исследованием),
  «Все сделки» и по листу на инструмент.
* **CSV:** [`research/ma_cross_slope_5m/tables/trade_ledger_walk_forward.csv`](../research/ma_cross_slope_5m/tables/trade_ledger_walk_forward.csv).

**Как читать.**
* Все сделки здесь вне выборки обучения: конфигурация года выбрана только по прошлым годам.
* Тестовые годы: NG — 2023–2026, BR, GOLD, SILV, PLT, PLD — 2014–2026. 2026 год неполный (по 13 октября).
  У COPPER и WTI истории меньше четырёх лет, walk-forward для них не считался.
* Как и в исследовании, сделка относится к году, в котором закрыта. Поэтому несколько сделок открыты
  конфигурацией года ещё в конце предыдущего года; они отмечены отдельно (колонка «Вход до начала года» в Excel).
* 1 контракт, без комиссий и проскальзывания. Цены — сырые цены контракта, вход и выход по Open свечи после
  сигнала. Сделка через ролл переходит в новый контракт по цене перекрытия.
"""),
    code("""
import pandas as pd
from IPython.display import Image, display

from src.ma_slope import ledger as L
from src.ma_slope.presentation import FIGS, NAMES, table

led = L.load_ledger(L.WF_CSV)
pd.set_option("display.float_format", "{:,.2f}".format)
print(f"Сделок всего: {len(led):,}".replace(",", " "), "| с входом до начала тестового года:",
      int(led["carry_in"].sum()), f"(их PnL {led.loc[led['carry_in'], 'pnl_usd'].sum():+,.2f} $)")
"""),
    md("## Сводка по инструментам"),
    code("L.summary_table(led)"),
    md("""
## Конфигурация и результат каждого тестового года

«t на обучении» — сглаженная t-статистика выбранной конфигурации за три года обучения. «PnL по исследованию» —
результат того же года из `tables/walk_forward.csv`: ledger воспроизводит его до округления хранения.
"""),
    code("""
wf = table("walk_forward.csv")
wf = wf[(wf["stage"] == "stage1") & (wf["scope"] == "all")].set_index(["ticker", "test_year"])
years = (led.groupby(["ticker", "test_year"], sort=False)
            .agg(конфигурация=("config", "first"), сделок=("pnl_usd", "size"),
                 прибыльных=("pnl_usd", lambda p: int((p > 0).sum())), pnl=("pnl_usd", "sum")))
years["t на обучении"] = wf["train_plateau_t"]
years["PnL по исследованию, $"] = wf["test_pnl"]
years = years.rename(columns={"pnl": "PnL, $"}).rename(index=NAMES, level=0).rename_axis(["инструмент", "год"])
with pd.option_context("display.max_rows", None, "display.max_colwidth", 70):
    display(years)
"""),
    md("### PnL по годам, $ на контракт"),
    code("""
years["PnL, $"].unstack(0).round(0)
"""),
    md("""
### Проверка учёта на границах лет

Исследование относит сделку к году её закрытия. Поэтому конфигурация года получает и ту часть перенесённой сделки,
что заработана до 1 января, а свои сделки, открытые на конец года, теряет. В реальной торговле конфигурацию
переключили бы ровно на границе года: старую позицию закрыли бы, а текущую позицию новой конфигурации открыли бы по
цене начала года. Ниже — тот же walk-forward, посчитанный так. Если итоги близки, способ учёта вывод не меняет.
"""),
    code("""
tr = L.walk_forward_tradable()
cmp = tr.groupby("instrument", sort=False)[["study_pnl", "tradable_pnl", "carried_in", "open_at_year_end"]].sum()
cmp.loc["Всего"] = cmp.sum()
cmp = cmp.astype({"carried_in": int, "open_at_year_end": int})
cmp["разница, $"] = cmp["tradable_pnl"] - cmp["study_pnl"]
cmp.columns = ["PnL как в исследовании, $", "PnL с переключением 1 января, $", "перенесённых в год сделок",
               "открытых на конец года", "разница, $"]
cmp
"""),
    md("""
## Все исходы по порядку

Сверху накопленный итог по номеру сделки, снизу результат каждой сделки (синий — прибыль, красный — убыток).
Тонкие вертикальные линии — смена тестового года (и конфигурации).
"""),
    code('display(Image(filename=str(FIGS / "trade_outcomes_walk_forward.png")))'),
    md("""
## Таблицы сделок

По каждому инструменту год за годом: заголовок года (конфигурация, годы обучения, число сделок и итог года), затем
все сделки года по порядку. Колонки: номер, сторона, время и контракт входа, цена входа, время и контракт выхода,
цена выхода, длительность в часах, результат в долларах и в тиках, исход и накопленный итог с начала walk-forward.
"""),
]
for t, name in INSTRUMENTS[:6]:
    WALK_FORWARD += [md(f"### {name}"), code(f'print(L.format_walk_forward_trades(led, "{t}"))')]


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    from src.ma_slope import ledger as L
    from src.ma_slope.presentation import FIGS

    finalists = L.all_ledgers()
    for out in (L.write_csv(finalists, L.LEDGER_CSV), L.write_excel(finalists),
                L.outcomes_figure(L.load_ledger(L.LEDGER_CSV), FIGS / "trade_outcomes.png")):
        print(f"written {out.relative_to(ROOT)}")
    wf = L.all_walk_forward_ledgers()
    for out in (L.write_csv(wf, L.WF_CSV), L.write_walk_forward_excel(wf),
                L.outcomes_figure(L.load_ledger(L.WF_CSV), FIGS / "trade_outcomes_walk_forward.png",
                                  "Walk-forward (config re-selected every year on the 3 years before): every trade "
                                  "in order\nPnL per trade (blue gain, red loss) and running total, 1 contract; "
                                  "thin lines = new test year")):
        print(f"written {out.relative_to(ROOT)}")
    nbf.write(notebook(FINALISTS), NB_FINALISTS)
    nbf.write(notebook(WALK_FORWARD), NB_WALK_FORWARD)
    print(f"written {NB_FINALISTS.relative_to(ROOT)}, {NB_WALK_FORWARD.relative_to(ROOT)}")
