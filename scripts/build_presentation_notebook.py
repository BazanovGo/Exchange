"""Build notebooks/07_project_presentation.ipynb: what was done and what came out.

    python scripts/build_presentation_notebook.py            # figures + notebook
    python scripts/run_notebook.py notebooks/07_project_presentation.ipynb
    python scripts/build_presentation_notebook.py --slides   # reveal.js deck from the executed notebook

Cells carry slideshow metadata, so the executed notebook reads top to bottom
on GitHub and also runs as a slide deck. Every number shown in a table is
read from the committed study tables; the three pipeline figures (series
overview, roll example, strategy example) are rebuilt here from the
AlgoPack candles in the repository.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "07_project_presentation.ipynb"

SLIDES_CSS = """<style type="text/css">
.reveal h1 { font-size: 2.1em; line-height: 1.15; }
.reveal h2 { font-size: 1.45em; }
.reveal .jp-RenderedHTMLCommon { font-size: 22px; line-height: 1.45; }
.reveal .jp-RenderedHTMLCommon table { font-size: 18px; }
.reveal .jp-RenderedHTMLCommon blockquote { font-size: 21px; width: auto; margin: 12px 0; }
.reveal .jp-OutputArea-output img { width: 100%; height: 760px; object-fit: contain; }
</style>
"""

cells: list = []


def _add(cell, slide_type):
    cell.metadata["slideshow"] = {"slide_type": slide_type}
    cells.append(cell)


def slide(text: str) -> None:
    _add(nbf.v4.new_markdown_cell(text.strip()), "slide")


def note(text: str) -> None:
    _add(nbf.v4.new_markdown_cell(text.strip()), "-")


def code(text: str, slide_type: str = "-") -> None:
    _add(nbf.v4.new_code_cell(text.strip()), slide_type)


def figure(name: str, width: int | None = None) -> None:
    """A figure on its own slide (in the notebook it simply follows the text)."""
    args = f'"{name}"' + (f", width={width}" if width else "")
    code(f"show({args})", "slide")


slide("""
# Фьючерсы MOEX на сырьё: данные AlgoPack и исследование стратегии «MA + наклон»

**Презентация проделанной работы** · данные по 13.10.2026 · ветка `ccr-cf486678-oggs1n`

| Часть | О чём |
|---|---|
| 1 | Что сделано и главный результат |
| 2 | Данные MOEX AlgoPack: 8 сырьевых фьючерсов |
| 3 | Склейка контрактов и исправление метрик |
| 4 | Исследование: стратегия, пространство поиска, защита от подгонки |
| 5 | Результаты: 11 слайдов |
| 6 | Итог, рекомендации, где что лежит |

Ноутбук уже выполнен и читается на GitHub без запуска. Подробный отчёт —
[`research/ma_cross_slope_5m/README.md`](../research/ma_cross_slope_5m/README.md), интерактивный разбор любой
конфигурации — [`06_ma_slope_research.ipynb`](06_ma_slope_research.ipynb).
""")

code("""
# Подготовка: все цифры ниже читаются из файлов репозитория
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from IPython.display import Markdown, display

from src.ma_slope import presentation as pres
from src.ma_slope.analysis import holm

show, table, NAMES = pres.show, pres.table, pres.NAMES
INF = ["ng", "br", "gold", "silv", "plt", "pld", "copper"]  # WTI: 2 181 бар, только справочно
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:,.2f}".format)
summary = pres.summary()
""", "skip")

slide("""
## 1. Что сделано

1. **Данные.** Загружены котировки 8 сырьевых фьючерсов MOEX **только из AlgoPack**: NG, Brent (BR), CL (WTI),
   GC (GOLD), SI (SILV), HG (COPPER), PL (PLT), PA (PLD). Минутные, 5-минутные и часовые свечи за всю доступную
   историю, с 2011 года по 13.10.2026.
2. **Склейка и метрики.** Контракты склеены **пропорционально**, по вашему решению. Загрузчик отдаёт ряды прямо
   в VectorBT. Исправлен годовой пересчёт метрик: Sharpe на часовых данных был завышен в 1,5–1,9 раза.
3. **Исследование.** Проверена стратегия «цена пересекает быструю MA, направление разрешает наклон медленной MA»
   на 5-минутках: 6 типов средних, все 36 сочетаний, Fast 2–99 × Slow 5–250, два метода наклона, две модели выхода.
   Это **около 9 млн конфигураций на инструмент** и около 64 млн бэктестов всего.
4. **Проверка.** IS/OOS, walk-forward, польза отбора, PBO, Monte Carlo, поправки Холма и Benjamini–Hochberg,
   межрыночная проверка. Ядро симуляции сверено с VectorBT сделка в сделку.

> **Главный результат.** Устойчивого преимущества у конкретных параметров **нет**: после поправок на
> множественное тестирование ни одна конфигурация не подтверждается. Зато есть воспроизводимые закономерности
> правил: выход по смене наклона, горизонт наклона 40–80 баров. Ежегодный переотбор параметров даёт плюс на всех
> 6 рынках с длинной историей. Но средняя сделка всего 0,5–1,8 тика, поэтому без учёта издержек торговать это
> нельзя.
""")

code("""
ov = pres.data_overview()
kpi = {
    "инструментов": len(ov),
    "минутных свечей": f"{ov['свечей 1m'].sum():,}".replace(",", " "),
    "5-минутных баров": f"{ov['баров 5m'].sum():,}".replace(",", " "),
    "контрактов склеено": int(ov["контрактов"].sum()),
    "переходов между контрактами": int(ov["переходов"].sum()),
    "конфигураций проверено": f"{sum(summary['trials_per_instrument'].values()):,}".replace(",", " "),
}
pd.DataFrame(kpi, index=["проект в цифрах"]).T
""")

slide("""
## 2. Данные MOEX AlgoPack

* **Источник** — только AlgoPack (`apim.moex.com`) по вашему токену. Токен в репозиторий не попал. Сертификат
  сервера подписан российским корневым центром, поэтому подключён официальный Russian Trusted Root CA. Проверка
  TLS не отключалась.
* **Список контрактов.** Справочник серий MOEX неполный, поэтому недостающие месяцы найдены перебором кодов
  (`BRQ3`, `BRQ3_2013`…). Всего 510 контрактов.
* **Непрерывный ряд по ближайшей экспирации.** Контракт держится до последнего дня обращения. Следующий берётся
  со следующего дня, а если старый перестал торговаться раньше — сразу после его последней свечи. Так закрыт
  разрыв NG в 2020 году.
* **Таймфреймы.** 1m — из API, 5m — собраны из 1m, 60m — исходные часовые свечи. Хранение: CSV по годам в
  `data/raw/moex_algopack/<тикер>/<таймфрейм>/`.
""")
code("pres.data_overview()")
note("""
**Оговорки.** CL (WTI) торгуется на MOEX только с июля 2026 года, это 2 181 пятиминутный бар: в выводах он не
участвует. У HG (COPPER) два года истории, поэтому для него нет walk-forward и PBO. У NG крупнейший разрыв
(48 %) — переход с январского на февральский контракт 2026 года, когда январский контракт резко вырос перед
экспирацией.
""")
figure("overview_prices.png")


slide("""
## 3. Склейка контрактов: пропорциональная, как вы решили

На каждом переходе в одну и ту же минуту — последнюю свечу старого контракта — берутся цены старого и нового
контракта. Вся история до перехода умножается на их отношение.

* Скачок на ролле исчезает. Значит, нет ложных пересечений средних и ложных изменений наклона.
* Процентные движения ряда — это движения контракта, который реально держали бы.
* Последний контракт сохраняет настоящие цены.
* Коэффициенты всех 502 переходов лежат в `rolls.csv`. Загрузчик `load_data("moex_algopack", ...)` применяет их
  по умолчанию, а `adjust=None` отдаёт сырые цены.

Денежный PnL в исследовании считается иначе: по изменению **сырой** цены удерживаемого контракта × размер
контракта. Позиция переходит в новый контракт по цене перекрытия, без скачка.
""")
figure("roll_example_plt.png", width=1000)

slide("""
## 3. Исправление метрик: годовой пересчёт

VectorBT переводит метрики в годовые через частоту бара. Для часовых свечей это давало 8 760 баров в год
(круглосуточно), хотя MOEX торгует примерно 2 500–4 200 часов в год. Sharpe из-за этого завышался в
√(8760 / факт) раз.

**Исправление:** `BacktestConfig(freq="auto")` берёт среднее календарное время на бар из самих данных. Годовые
Sharpe, Sortino, CAGR и длительности теперь считаются по фактически торгуемым барам. Исправление действует во всех
модулях репозитория: движке бэктестов, исследованиях и управлении позицией.
""")
code("pres.annualization_check()")

slide("""
## 4. Стратегия: правила

* **Вход.** Close пересекает быструю MA (Fast) по двум закрытым свечам, сделка открывается по Open следующей
  свечи. Long — при пересечении вверх, если наклон медленной MA (Slow) вверх или нейтральный. Short — при
  пересечении вниз, если наклон вниз или нейтральный.
* **Наклон Slow MA за k баров.** Метод A: `(MA[t] − MA[t−k]) / MA[t−k] · 100 %`. Метод B: `(MA[t] − MA[t−k]) / ATR`.
  При |наклон| ≤ ε наклон нейтральный.
* **Выход A (`reverse_cross`)** — обратное пересечение Fast MA. **Выход B (`slope_flip`)** — наклон Slow MA стал
  противоположным.
* Одна позиция, 1 контракт, без комиссий и проскальзывания (по ТЗ), без разворота на свече выхода.

Ниже — правила на реальных свечах: лучшая конфигурация NG, три дня октября 2026 года. Окно выбрано автоматически:
последние три будних дня, где есть long и short, прибыль и убыток.
""")
figure("strategy_example_ng.png", width=1150)

slide("""
## 4. Пространство поиска: без произвольных сокращений

| Измерение | Значения | Число |
|---|---|---:|
| Типы средних | SMA, EMA, WMA, SSMA (Wilder), VWMA, TMA | 6 |
| Пары «быстрая × медленная» | все сочетания, включая одинаковые типы | 36 |
| Период Fast | 2–99 | 98 |
| Период Slow | 5–250, включая Fast ≥ Slow как отдельную категорию | 246 |
| Метод наклона | A — в %, B — в ATR | 2 |
| Модель выхода | обратное пересечение / смена наклона | 2 |
| Настройки наклона | k ∈ {1…160}, ε, ATR ∈ {14, 100} | 128 |

Полное произведение — 222 млн конфигураций на инструмент, около 4 суток счёта. Поэтому оптимизация поэтапная, и
это изменение плана задокументировано в отчёте:

1. **Этап 1** — полная сетка 3,47 млн конфигураций при наклоне, заданном заранее (k = 10). Это контроль.
2. **Этап 2** — 128 настроек наклона на решётке периодов. Когда оптимумы упёрлись в k = 40, сетка была расширена
   до k = 160.
3. **Этап 3** — снова полная сетка с наклоном, выбранным только по IS. Это основной результат.

Итого 8 979 840 конфигураций на инструмент с длинной историей, около 6 часов счёта на 4 ядрах (numba).
""")

slide("""
## 4. Как защищались от подгонки

* **IS / OOS.** Первые 70 % баров — обучение, последние 30 % — независимая проверка. Сделка относится к периоду,
  в котором она закрыта.
* **Отбор не по максимальной прибыли**, а по **плато**: средний IS t-stat соседей (Fast ±2, Slow ±5), не меньше
  100 сделок.
* **Множественное тестирование:** поправка Холма по финалистам, Benjamini–Hochberg по лучшим конфигурациям
  групп, сравнение с ожидаемым максимумом t среди 9 млн нулевых тестов.
* **Walk-forward:** 3 года обучения → следующий год, отбор среди всех 3,47 млн конфигураций, 68 тестовых лет.
* **Польза отбора:** перцентиль выбранной конфигурации среди всех конфигураций тестового периода (0,5 — случайный
  выбор).
* **PBO (CSCV)** — вероятность переобучения отбора. **Monte Carlo** — 10 000 бутстрэпов сделок.
* **Межрыночная проверка:** одни правила на всех рынках, перенос лучших параметров с рынка на рынок.
* **Корректность:** ядро симуляции сверено с независимым побарным симулятором и с `vectorbt.Portfolio.from_signals`.
  Совпадают все сделки, расхождение PnL — около 10⁻¹³ USD. На модуль 41 тест.
""")

slide("""
## 5.1 Результат определяет режим рынка, а не параметры

Одни и те же правила зарабатывают или теряют почти всей сеткой сразу. На золоте и серебре с выходом по обратному
пересечению в IS (2012–2022) убыточны 98–100 % конфигураций, а в OOS (2022–2026) прибыльны 67–78 %. У PLT и PLD —
наоборот. Медианный OOS t разных рынков различается на 2,9, а 36 сочетаний типов средних внутри рынка — лишь на
0,7–1,1.
""")
code("""
s = table("grid_summary.csv")
s = s[(s["stage"] == "stage3") & s["ticker"].isin(INF)]
w = s.assign(**{c: s[c] * s["configs"] for c in ("is_share_pos", "oos_share_pos", "is_t_median", "oos_t_median")})
g = w.groupby(["ticker", "model"])[["configs", "is_share_pos", "oos_share_pos", "is_t_median", "oos_t_median"]].sum()
g = g.iloc[:, 1:].div(g["configs"], axis=0)
g.columns = ["IS: доля t>0", "OOS: доля t>0", "IS: медиана t", "OOS: медиана t"]
g.rename(index=NAMES).loc[[NAMES[t] for t in INF]].rename_axis(["инструмент", "выход"])
""")
figure("is_vs_oos.png")

slide("""
## 5.2 Выход по смене наклона лучше выхода по обратному пересечению

Медианный t-stat выхода B выше на IS на всех 7 рабочих рынках и на OOS на 6 из 7 (кроме NG). Все финалисты
используют выход B. Быстрые входы-выходы на 5-минутках систематически проигрывают там, где цена возвращается
к среднему: у COPPER выход A убыточен у 100 % конфигураций на IS.
""")
figure("exit_models_oos.png", width=1000)

slide("""
## 5.3 Главный параметр фильтра — горизонт наклона k

Медианный IS t по всем рынкам растёт с −1,08 при k = 1 до +0,11 при k = 80 баров (≈ 7 часов) и снова падает при
k = 160. Порог нейтральной зоны ε почти ни на что не влияет. Методы A (в %) и B (в ATR) после подбора
равноценны.
""")
code("""
sw = table("slope_sweep.csv")
sw = sw[sw["ticker"].isin(INF)]
display(Markdown("**Медианный IS t по всем рынкам и решётке периодов в зависимости от k**"))
sw.groupby("k")["is_median_t"].mean().to_frame("IS t, медиана").T
""")
figure("slope_sweep_pct_slope_flip.png")

slide("""
## 5.4 Типы средних — эффект второго порядка

Лучшие сочетания — медленная EMA с быстрой TMA, VWMA или SMA: средний медианный OOS t 0,34–0,40. Худшие — быстрая
SSMA с медленной SMA или TMA: около −0,1. Медленные EMA, SSMA и WMA немного лучше медленных SMA, TMA и VWMA.
Но разница между типами меньше, чем между рынками и периодами.

Категория **Fast ≥ Slow** («перевёрнутые» периоды) не хуже стандартной: фильтр наклона и выход B работают при
любом соотношении периодов.
""")
figure("type_pairs_oos_median_t.png")

slide("""
## 5.5 Лучшие конфигурации по инструментам

Для каждого инструмента выбрана лучшая конфигурация по плато IS t, затем она проверена на OOS. Sharpe и Sortino
посчитаны через VectorBT по дневному PnL одного контракта.
""")
code("""
fz = table("finalists.csv")
f = fz[(fz["stage"] == "stage3") & (fz["tag"] == "overall")].set_index("ticker").loc[INF]
pd.DataFrame({
    "Fast": f["fast_type"] + f["fast_period"].astype(str), "Slow": f["slow_type"] + f["slow_period"].astype(str),
    "наклон": f["slope"], "выход": f["model"], "IS сделок": f["is_trades"].astype(int), "IS t": f["is_t"],
    "OOS сделок": f["oos_trades"].astype(int), "OOS PnL, $": f["oos_pnl"].round().astype(int),
    "OOS PF": f["oos_profit_factor"], "OOS t": f["oos_t"], "p (Холм)": f["oos_p_holm"], "Sharpe IS": f["is_sharpe"],
    "Sharpe OOS": f["oos_sharpe"],
}).rename(index=NAMES).rename_axis("инструмент")
""")
note("""
Профиль классический трендовый: около 35 % прибыльных сделок, средняя прибыль примерно вдвое больше среднего
убытка. На OOS все 7 финалистов в плюсе, но OOS t всего 0,09–1,38. После поправки Холма значимых нет: минимальное
p = 0,58. Sharpe на OOS ниже, чем на IS, у всех; почти не изменился только у PLD (1,20 → 1,17).
""")
figure("equity_finalists.png")


slide("""
## 5.6 Множественное тестирование: выдающихся результатов нет

При 9 млн проверок на инструмент даже на случайных данных лучший t был бы около 5,3. Лучшие IS t-плато
финалистов не выше 4,2, то есть ниже этого порога. Поэтому решения принимались только по OOS и walk-forward, с поправками.
""")
code("""
rk = table("ranking_group_bests.csv")
rk3 = rk[(rk["stage"] == "stage3") & rk["ticker"].isin(INF)]
pd.DataFrame({"значение": {
    "ожидаемый максимум t среди 9 млн нулевых тестов": f"{summary['expected_max_t_null']['ng']:.2f}",
    "лучшее IS t-плато финалистов": f"{f['is_plateau_t'].max():.2f}",
    "минимальное p финалистов после Холма": f"{f['oos_p_holm'].min():.2f}",
    "лучших конфигураций групп (тип × тип × метод × выход)": str(len(rk3)),
    "из них с OOS t > 0": str(int((rk3["oos_t"] > 0).sum())),
    "из них значимы после Benjamini–Hochberg (q < 0,05)": str(int((rk3["oos_q_bh"] < 0.05).sum())),
    "минимальный q": f"{rk3['oos_q_bh'].min():.3f}",
}})
""")
figure("finalists_is_vs_oos.png", width=1000)

slide("""
## 5.7 Карта устойчивости: плато, а не пик

Карта Fast × Slow для слоя лучшей конфигурации PLD: IS t, сглаженное плато, доля положительных соседей и OOS t.
Кружок — выбранная конфигурация. Широкие однородные области означают, что результат даёт семейство параметров,
а не одна точка. Карты всех инструментов — `figures/heatmap_stage3_*.png`.
""")
figure("heatmap_stage3_pld.png")

slide("""
## 5.8 Walk-forward: ежегодный переотбор — самый сильный результат

Каждый год параметры заново выбираются по трём предыдущим годам среди всех 3,47 млн конфигураций и торгуются
следующий год. Совокупный результат положителен на всех 6 рынках с длинной историей, после поправки Холма значимы
NG и PLT.
""")
code("""
wf = table("walk_forward_summary.csv")
wf1 = wf[(wf["stage"] == "stage1") & (wf["scope"] == "all")].set_index("ticker").loc[INF[:-1]]
pd.DataFrame({"тестовых лет": wf1["test_years"], "лет в плюсе": wf1["positive_years"],
              "PnL, $ на контракт": wf1["test_pnl"].round().astype(int), "t": wf1["pooled_t"], "p": wf1["pooled_p"],
              "p (Холм)": holm(wf1["pooled_p"])}).rename(index=NAMES).rename_axis("инструмент")
""")
figure("walk_forward_stage1.png")

slide("""
## 5.9 Польза отбора и вероятность переобучения

* В среднем отобранная конфигурация оказывается на **65-м перцентиле** всех конфигураций тестового года
  (68 лет, z = 4,2). Отбор слабо, но надёжно лучше случайного выбора. Однако на NG, BR, PLT и PLD положительна
  и медианная конфигурация, так что заметная часть прибыли — общий трендовый дрейф.
* **PBO** (доля разбиений, где лучшая конфигурация одной половины истории хуже медианы на другой): на BR и PLT
  ≈ 0,5, то есть отбор переобучен; на PLD, SILV, GOLD и NG — 0,07–0,35.
""")
code("""
(table("pbo.csv").pivot(index="ticker", columns="stage", values="pbo").loc[INF[:-1]].rename(index=NAMES)
 .rename(columns={"stage1": "PBO, этап 1", "stage3": "PBO, этап 3"}).rename_axis("инструмент", axis=0)
 .rename_axis(None, axis=1))
""")
figure("selection_skill.png", width=1000)

slide("""
## 5.10 Межрыночная проверка

Одни и те же правила на всех рынках (этап 1). Строка — рынок, где выбрана лучшая по IS конфигурация, столбец —
рынок, где её проверили на OOS. Результат задаёт столбец, то есть режим проверяемого рынка: на золоте положителен
почти любой перенос. Ни одна конфигурация, отобранная честно (только по IS), не прибыльна на OOS на всех рынках:
лучшие дают плюс на 3–6 рынках из 7.
""")
figure("cross_market_transfer.png", width=900)

slide("""
## 5.11 Экономика сделки: преимущество хрупкое

Средняя OOS-сделка финалистов NG, BR и SILV — 0,5–1,8 минимального шага цены. Спред и комиссия обычно стоят 1–2
тика на круг, и они съедят большую часть такой сделки или всю её. Комиссии и проскальзывание по ТЗ не
учитывались. У GOLD, PLT и PLD средняя сделка 16–137 тиков, но у неликвидных PLT и PLD выше и спред, и
проскальзывание.
""")
code("""
pd.DataFrame({"OOS сделок": f["oos_trades"].astype(int), "средняя OOS-сделка, $": f["oos_avg_trade"],
              "в тиках": f["oos_avg_trade_ticks"], "средняя длительность OOS, ч": f["oos_avg_hours"],
              "MC: вероятность убытка на OOS": f["mc_oos_p_loss"]}).rename(index=NAMES).rename_axis("инструмент")
""")

slide("""
## 6. Итог: что получилось

| Вопрос | Ответ |
|---|---|
| Есть ли устойчиво прибыльные параметры? | **Нет.** После поправок на 9 млн проверок ни одна конфигурация не значима на OOS |
| Что важнее всего для результата? | **Режим рынка.** Знак результата меняется сразу у всей сетки |
| Какой выход лучше? | **Смена наклона Slow MA** — на 7 из 7 рынков на IS и на 6 из 7 на OOS |
| Какой горизонт наклона? | **k = 40–80 баров** (3–7 часов); порог ε и метод (% или ATR) почти не важны |
| Какие средние? | Различия второго порядка; медленная EMA, SSMA или WMA немного лучше |
| Fast ≥ Slow? | Не хуже стандартных пар |
| Работает ли регулярный переотбор? | **Да, слабо:** плюс на 6 из 6 рынков, значимо на NG и PLT; отбор на 65-м перцентиле |
| Можно ли торговать? | **Пока нет:** средняя сделка 0,5–1,8 тика на NG, BR, SILV, издержки не учтены |

**Рекомендации и следующие шаги**

1. Повторить лучшие правила (выход B, k = 40–80, walk-forward переотбор) **с комиссией MOEX и спредом**.
2. Проверить на 2026–2027 годах как на новой независимой выборке, без переподбора.
3. Проверить старшие таймфреймы (15m–60m): сделки там длиннее, и средняя сделка в тиках, вероятно, больше.
   Оценить ёмкость по объёмам.
4. Рассматривать трендовую экспозицию как режимный фактор портфеля, а не искать «правильные» периоды средних.
""")

slide("""
## 6. Где всё лежит

| Что | Где |
|---|---|
| Эта презентация | `notebooks/07_project_presentation.ipynb`, слайды — `07_project_presentation.slides.html` |
| Все сделки лучших конфигураций по порядку (16 349 сделок) | `notebooks/08_trade_ledger.ipynb`, `research/ma_cross_slope_5m/tables/trade_ledger.xlsx` |
| Все сделки ежегодного переотбора, walk-forward (35 844 сделки) | `notebooks/09_walk_forward_ledger.ipynb`, `research/ma_cross_slope_5m/tables/trade_ledger_walk_forward.xlsx` |
| Подробный разбор и исследователь любой конфигурации (VectorBT) | `notebooks/06_ma_slope_research.ipynb` |
| Полный отчёт с методологией и ограничениями | `research/ma_cross_slope_5m/README.md` |
| Таблицы: рейтинг, финалисты, сделки, walk-forward, PBO… | `research/ma_cross_slope_5m/tables/` |
| Рисунки (40 PNG) | `research/ma_cross_slope_5m/figures/` |
| Все конфигурации всех этапов (parquet, ~1 ГБ) | `research/ma_cross_slope_5m/grid/` |
| Котировки AlgoPack (1m, 5m, 60m) и коэффициенты склейки | `data/raw/moex_algopack/` |
| Загрузчик котировок и склейка | `scripts/download_moex_algopack.py`, `src/data/moex_algopack_loader.py` |
| Код исследования | `src/ma_slope/`, `scripts/ma_slope_research.py` |

**Воспроизведение.** Каждый этап запускается одной командой (раздел 7 отчёта). Этапы возобновляемые. Сырые сетки
(≈ 17 ГБ) в git не входят из-за лимита GitHub на размер файла, но всё, что из них следует, лежит в репозитории.
Обновить котировки: `MOEX_ALGOPACK_TOKEN=... python scripts/download_moex_algopack.py --from 2026-01-01 ...`.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}



def export_slides() -> Path:
    """reveal.js deck (code hidden) from the executed notebook, with type sized for dense slides."""
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "slides", "--no-input",
                    "--SlidesExporter.reveal_scroll=True", "--SlidesExporter.reveal_number=c/t",
                    "--SlidesExporter.reveal_transition=none", "--SlidesExporter.reveal_width=1400",
                    "--SlidesExporter.reveal_height=900", str(OUT)], check=True)
    html = OUT.with_suffix(".slides.html")
    html.write_text(html.read_text().replace("</head>", SLIDES_CSS + "</head>", 1))
    return html


if __name__ == "__main__":
    if "--slides" in sys.argv:
        print(f"written {export_slides().relative_to(ROOT)}")
        raise SystemExit(0)
    if "--no-figures" not in sys.argv:
        sys.path.insert(0, str(ROOT))
        from src.ma_slope.presentation import build_figures

        for p in build_figures():
            print(f"figure {p.relative_to(ROOT)}")
    nbf.write(nb, OUT)
    print(f"written {OUT.relative_to(ROOT)}")
