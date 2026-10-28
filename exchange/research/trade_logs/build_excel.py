"""trade_log.xlsx и README.md из CSV журнала сделок.

Лист на каждую стратегию: сделки в порядке закрытия, итог нарастающим и
серия убытков подряд — формулами Excel. Лист «Сводка» считает итоги
формулами по листам стратегий и сверяет их с отчётами.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve().parent
CSV = HERE / "csv"
OUT = HERE / "trade_log.xlsx"
OUT_GROUPS = HERE / "trade_log_groups.xlsx"
OUT_FULL = HERE / "trade_log_full.xlsx"

FONT = "Arial"
USD = '#,##0.00;[Red]-#,##0.00;0.00'
USD0 = '#,##0;[Red]-#,##0;0'
HEAD_FILL = PatternFill("solid", fgColor="E8E6E1")
INPUT = Font(name=FONT, size=10, color="0000FF")      # введённые руками значения (из отчётов)
THIN = Border(bottom=Side(style="thin", color="B5B3AB"))

SHEET_COLS = [  # (заголовок, ширина, формат)
    ("№", 7, "0"), ("период", 8, "@"), ("нога", 7, "@"), ("вход", 17, None), ("выход", 17, None),
    ("направление", 12, "@"), ("объём при входе, USD", 15, USD0), ("котировка входа", 12, "0.000"),
    ("котировка выхода", 12, "0.000"), ("через ролл", 9, "@"), ("издержки, USD", 12, USD),
    ("результат, USD", 14, USD), ("итог нарастающим, USD", 17, USD), ("исход", 10, "@"),
    ("серия убытков подряд", 12, "0"), ("баров в позиции", 10, "0"), ("примечание", 60, "@"),
]
CSV_TO_COL = {"№": "A", "период": "B", "нога": "C", "вход": "D", "выход": "E", "направление": "F",
              "объём при входе, USD": "G", "котировка входа": "H", "котировка выхода": "I",
              "через ролл": "J", "издержки, USD": "K", "результат, USD": "L",
              "баров в позиции": "P", "примечание": "Q"}
FIRST = 6   # первая строка данных на листе стратегии


def font(**kw):
    return Font(name=FONT, size=kw.pop("size", 10), **kw)


def strategies(kind: str = "main") -> list[dict]:
    if kind == "full":
        seven = json.loads((CSV / "seven_full_meta.json").read_text())
        for m in seven:
            m["study"] = "Семь семейств (дневные бары), сквозной прогон"
            m["title"] = m["name"] + " · " + m["params"]
            m["ref"], m["ref_kind"] = m["equity_change"], "весь период"
            m["ref_note"] = ("изменение капитала сквозного прогона (export_trades.py seven-full); сделки до "
                             "конца IS совпадают с IS-журналом отчёта")
            m["time_fmt"] = "yyyy-mm-dd"
        return seven
    if kind == "groups":
        ma = json.loads((CSV / "ma_groups_meta.json").read_text())
        _ma_fields(ma, groups=True)
        return ma
    seven = json.loads((CSV / "seven_meta.json").read_text())
    ma = json.loads((CSV / "ma_meta.json").read_text())
    for m in seven:
        m["study"] = "Семь семейств (дневные бары)"
        m["title"] = m["name"] + " · " + m["params"]
        m["ref"], m["ref_kind"] = m["oos_return_report"] * 1_000_000, "OOS"
        m["ref_note"] = (f"docs/RESULTS_SEVEN_FAMILIES.md, раздел 2, «итог. дох.» "
                         f"{m['oos_return_report']:+.1%} × 1 000 000")
        m["time_fmt"] = "yyyy-mm-dd"
    _ma_fields(ma, groups=False)
    return seven + ma


def _ma_fields(ma: list[dict], groups: bool):
    for m in ma:
        stage = "этап 1" if m["stage"] == "stage1" else "этап 3"
        m["study"] = f"Скользящие средние, 5 мин, {stage}" + (", лучшие в группе" if groups else "")
        m["title"] = f"{m['market']} · " + (f"{m['group']} · " if groups else "") + f"{m['rule']} · {m['name']}"
        m["ref"], m["ref_kind"] = m["full_pnl"], "весь период"
        m["ref_note"] = (f"research/ma_crossover_5m/results/{m['stage']}/{m['market']}/finalists.json, "
                         "full.pnl (VectorBT)")
        m["time_fmt"] = "yyyy-mm-dd hh:mm"


def write_strategy(wb: Workbook, m: dict) -> tuple[str, int]:
    df = pd.read_csv(CSV / m["csv"], keep_default_na=False, na_values=[""])
    ws = wb.create_sheet(m["id"])
    ws["A1"] = m["title"]
    ws["A1"].font = font(size=12, bold=True)
    ws["A2"] = f"{m['study']}. Единицы: {m['units']}."
    ws["A3"] = ("Каждая строка — одна сделка в порядке закрытия. «период»: IS — обучающая выборка, "
                "OOS — проверочная; сделка относится к периоду по дате выхода. Итог нарастающим и серия "
                "убытков — формулы.")
    for r in (2, 3):
        ws[f"A{r}"].font = font(color="52514E")
    for j, (name, width, _) in enumerate(SHEET_COLS, start=1):
        c = ws.cell(row=FIRST - 1, column=j, value=name)
        c.font = font(bold=True)
        c.fill = HEAD_FILL
        c.alignment = Alignment(wrap_text=True, vertical="center")
        c.border = THIN
        ws.column_dimensions[get_column_letter(j)].width = width
    ws.row_dimensions[FIRST - 1].height = 30

    for col_name in ("вход", "выход"):
        df[col_name] = pd.to_datetime(df[col_name])
    n = len(df)
    last = FIRST + n - 1
    fmts = {get_column_letter(j): f for j, (_, _, f) in enumerate(SHEET_COLS, start=1)}
    plain = font()
    for i, row in enumerate(df.itertuples(index=False), start=FIRST):
        rec = dict(zip(df.columns, row, strict=True))
        for name, col in CSV_TO_COL.items():
            v = rec[name]
            if pd.isna(v):
                continue
            if name in ("вход", "выход"):
                v = v.to_pydatetime()
            cell = ws[f"{col}{i}"]
            cell.value = v
            cell.font = plain
            cell.number_format = m["time_fmt"] if name in ("вход", "выход") else fmts[col]
        ws[f"M{i}"] = f"=L{i}" if i == FIRST else f"=M{i - 1}+L{i}"
        ws[f"N{i}"] = f'=IF(L{i}>0,"прибыль",IF(L{i}<0,"убыток","ноль"))'
        ws[f"O{i}"] = f"=IF(L{i}<0,1,0)" if i == FIRST else f"=IF(L{i}<0,O{i - 1}+1,0)"
        for col in "MNO":
            ws[f"{col}{i}"].font = plain
            ws[f"{col}{i}"].number_format = fmts[col]
    ws.freeze_panes = f"A{FIRST}"
    ws.auto_filter.ref = f"A{FIRST - 1}:Q{last}"
    return ws.title, last


MAIN_NOTES = [
        "Лист на каждую стратегию: все сделки по порядку, сколько заработано (+) или потеряно (−) на каждой, "
        "итог нарастающим и серия убытков подряд.",
        "Семь семейств: счёт 1 000 000 USD, размер позиции по целевой волатильности 15%, издержки ~7 б.п. за "
        "оборот уже вычтены. IS и OOS — отдельные прогоны, как в отчёте: OOS начинается без позиции.",
        "Скользящие средние: 1 контракт (NG — 100 MMBtu, BR — 10 баррелей), без издержек, как требовало "
        "задание; при 1 тике издержек на сделку (0.1 USD) итог уменьшается на 0.1 USD × число сделок.",
        "Синим — числа из отчётов для сверки; всё остальное на этом листе — формулы по листам стратегий. "
        "У семи семейств отчёт даёт доходность с точностью 0.1% (1 000 USD на счёт), поэтому расхождение "
        "до ±500 USD — округление.",
]
GROUP_NOTES = [
    "Скользящие средние, лучшие R1–R3 внутри каждой из четырёх групп «модель выхода × метод наклона» на "
    "этапах 1 и 3. Если лучший в группе совпал с глобальным отбором, он есть только в trade_log.xlsx.",
    "Лист на каждую стратегию: все сделки по порядку, сколько заработано (+) или потеряно (−) на каждой, "
    "итог нарастающим и серия убытков подряд.",
    "1 контракт (NG — 100 MMBtu, BR — 10 баррелей), без издержек, как требовало задание; при 1 тике издержек "
    "на сделку (0.1 USD) итог уменьшается на 0.1 USD × число сделок.",
    "Синим — числа из отчёта для сверки; всё остальное на этом листе — формулы по листам стратегий.",
]


FULL_NOTES = [
    "Семь семейств: те же победители и параметры, что в trade_log.xlsx, но одним прогоном по всей истории — "
    "все сделки подряд, без разрыва между выборкой и проверкой.",
    "В прогонах отчёта OOS начинается без позиции и с пустыми индикаторами, а между периодами — эмбарго 10 "
    "дней; здесь этих разрывов нет. «период» — по дате выхода: IS, эмбарго или OOS. Итог OOS здесь не равен "
    "итогу отчёта: отчёт оценивает систему, запущенную с нуля в начале OOS.",
    "Счёт 1 000 000 USD, размер позиции по целевой волатильности 15%, издержки ~7 б.п. за оборот вычтены.",
    "Синим — изменение капитала прогона для сверки; всё остальное на этом листе — формулы по листам.",
]


def write_summary(ws, rows: list[tuple[dict, str, int]], title: str, notes: list[str]):
    ws["A1"] = title
    ws["A1"].font = font(size=14, bold=True)
    for k, text in enumerate(notes, start=2):
        ws[f"A{k}"] = text
        ws[f"A{k}"].font = font(color="52514E")
    head = ["Лист", "Исследование", "Рынок", "Стратегия", "Сделок", "Прибыльных", "Убыточных",
            "Доля прибыльных", "Итог IS, USD", "Итог OOS, USD", "Итог всего, USD", "Лучшая сделка, USD",
            "Худшая сделка, USD", "Макс. серия убытков подряд", "Итог по отчёту, USD", "Сверяется с",
            "Расхождение, USD", "Источник числа из отчёта"]
    widths = [22, 30, 8, 58, 9, 11, 11, 11, 15, 15, 15, 14, 14, 13, 16, 12, 13, 70]
    h = 7
    for j, (name, width) in enumerate(zip(head, widths, strict=True), start=1):
        c = ws.cell(row=h, column=j, value=name)
        c.font = font(bold=True)
        c.fill = HEAD_FILL
        c.alignment = Alignment(wrap_text=True, vertical="center")
        c.border = THIN
        ws.column_dimensions[get_column_letter(j)].width = width
    ws.row_dimensions[h].height = 32
    for i, (m, sheet, last) in enumerate(rows, start=h + 1):
        q = f"'{sheet}'!"
        rng = lambda col, q=q, last=last: f"{q}${col}${FIRST}:${col}${last}"  # noqa: E731
        ws[f"A{i}"] = sheet
        ws[f"A{i}"].hyperlink = f"#'{sheet}'!A1"
        ws[f"A{i}"].font = font(color="1C5CAB", underline="single")
        ws[f"B{i}"], ws[f"C{i}"], ws[f"D{i}"] = m["study"], m["market"], m["title"]
        ws[f"E{i}"] = f'=COUNTIFS({rng("D")},"<>")'
        ws[f"F{i}"] = f'=COUNTIFS({rng("L")},">0",{rng("D")},"<>")'
        ws[f"G{i}"] = f'=COUNTIFS({rng("L")},"<0",{rng("D")},"<>")'
        ws[f"H{i}"] = f"=IF(E{i}>0,F{i}/E{i},0)"
        ws[f"I{i}"] = f'=SUMIFS({rng("L")},{rng("B")},"IS")'
        ws[f"J{i}"] = f'=SUMIFS({rng("L")},{rng("B")},"OOS")'
        ws[f"K{i}"] = f"=SUM({rng('L')})"
        ws[f"L{i}"] = f'=_xlfn.MAXIFS({rng("L")},{rng("D")},"<>")'
        ws[f"M{i}"] = f'=_xlfn.MINIFS({rng("L")},{rng("D")},"<>")'
        ws[f"N{i}"] = f"=MAX({rng('O')})"
        ws[f"O{i}"] = round(m["ref"], 2)
        ws[f"P{i}"] = m["ref_kind"]
        ws[f"Q{i}"] = f'=IF(P{i}="OOS",J{i},K{i})-O{i}'
        ws[f"R{i}"] = m["ref_note"]
        for col in "ABCDEFGHIJKLMNOPQR":
            c = ws[f"{col}{i}"]
            if col != "A":
                c.font = INPUT if col == "O" else font()
            c.number_format = {"E": "0", "F": "0", "G": "0", "H": "0.0%", "N": "0"}.get(col, USD if col in "IJKLMOQ" else "@")
    ws.freeze_panes = f"B{h + 1}"


def _streak(values) -> int:
    streak = best = 0
    for v in values:
        streak = streak + 1 if v < 0 else 0
        best = max(best, streak)
    return best


def num(v) -> str:
    return f"{v:+,.0f}".replace(",", " ")


def _readme_rows(rows: list[tuple[dict, str, int]]) -> list[str]:
    out = []
    for m, _sheet, _ in rows:
        df = pd.read_csv(CSV / m["csv"])
        trades = df[df["вход"].notna()]
        is_, oos = (df.loc[df["период"] == p, "результат, USD"].sum() for p in ("IS", "OOS"))
        out.append(f"| {m['study']}: {m['title']} | {len(trades)} | {(trades['результат, USD'] > 0).mean():.0%} "
                   f"| {num(is_)} | {num(oos)} | {_streak(df['результат, USD'])} | [{Path(m['csv']).name}](csv/{m['csv']}) |")
    return out


HEADER = ["| Стратегия | Сделок | Прибыльных | Итог IS, USD | Итог OOS, USD | Макс. убыточных сделок подряд | CSV |",
          "|---|---|---|---|---|---|---|"]


def sp(v: int) -> str:
    return f"{v:,}".replace(",", " ")


def rows_for(kind: str) -> list[tuple[dict, str, int]]:
    """Те же (стратегия, лист, последняя строка), что даёт build, — без сборки книги."""
    out = []
    for m in strategies(kind):
        n = len(pd.read_csv(CSV / m["csv"]))
        out.append((m, m["id"], FIRST + n - 1))
    return out


def _full_rows(rows) -> list[str]:
    out = []
    for m, _sheet, _ in rows:
        df = pd.read_csv(CSV / m["csv"])
        trades = df[df["вход"].notna()]
        part = {p: df.loc[df["период"] == p, "результат, USD"].sum() for p in ("IS", "эмбарго", "OOS")}
        out.append(f"| {m['title']} | {len(trades)} | {(trades['результат, USD'] > 0).mean():.0%} | {num(part['IS'])} "
                   f"| {num(part['эмбарго'])} | {num(part['OOS'])} | {num(df['результат, USD'].sum())} "
                   f"| {_streak(df['результат, USD'])} | [{Path(m['csv']).name}](csv/{m['csv']}) |")
    return out


def _wf_rows() -> list[str]:
    out = ["| Система | Шагов | Сделок | Итог, USD | Сверка с отчётом | CSV |", "|---|---|---|---|---|---|"]
    for m in json.loads((CSV / "seven_wf_meta.json").read_text()):
        r = m["report"]
        out.append(f"| Семь семейств: {m['name']} | {len(m['windows'])} окон | {m['trades']} | {num(m['pnl'])} "
                   f"| эффективность {r['WF эффект.']:+.3f}, окон>0 {r['WF окон>0']:.3f}, стабильность "
                   f"{r['WF стабил. параметров']:.3f} — как в разделе 3 | [{Path(m['csv']).name}](csv/{m['csv']}) |")
    for m in json.loads((CSV / "ma_wf_meta.json").read_text()):
        stage = "этап 1" if m["stage"] == "stage1" else "этап 3"
        out.append(f"| Скользящие средние, {stage}: {m['market']} · критерий «{m['rule']}» | шагов по 3 мес.: {m['steps']}, "
                   f"конфигураций: {m['configs']} | {m['trades']} | {num(m['pnl'])} "
                   f"| итог шагов {m['wf_pnl_reference']:+.1f} USD (`wf_{m['criterion']}.csv`) "
                   f"| [{Path(m['csv']).name}](csv/{m['csv']}) |")
    return out


def _family_rows() -> list[str]:
    meta = pd.DataFrame(json.loads((CSV / "ma_families_meta.json").read_text()))
    out = ["| Этап | Рынок | Семейств | Сделок | Прибыльны в IS | Прибыльны в OOS | Совпали с журналом финалиста |",
           "|---|---|---|---|---|---|---|"]
    for (stage, t), g in meta.groupby(["stage", "market"], sort=False):
        out.append(f"| {'этап 1' if stage == 'stage1' else 'этап 3'} | {t} | {len(g)} | {sp(int(g['trades'].sum()))} "
                   f"| {(g['is_pnl'] > 0).sum()} | {(g['oos_pnl'] > 0).sum()} | {(g['same_as'] != '').sum()} |")
    return out


def write_families_index():
    meta = json.loads((CSV / "ma_families_meta.json").read_text())
    lines = [
        "# Скользящие средние: лучший в каждом из 144 семейств", "",
        "Семейство — сочетание типов быстрой и медленной средней × модель выхода × метод наклона (36 × 2 × 2 = "
        "144). В каждом выбрана конфигурация с лучшим сглаженным IS-Шарпом (правило R3, не меньше 100 сделок в "
        "выборке) — это кандидаты разделов 7 и 12 `docs/RESULTS_MA_CROSSOVER_5M.md`. Все сделки по порядку, "
        "1 контракт, без издержек. Сделки посчитаны событийным ядром исследования и сверены с полной сеткой: "
        "число сделок IS и OOS — точно, PnL — до 0.5 USD (в сетке он округлён).", "",
        "«Совпадает с» — конфигурация уже есть среди финалистов (глобальных или лучших в группе); её сделки в обоих "
        "файлах одинаковы.", "",
        "| Этап | Рынок | Семейство | Конфигурация | Сделок | Итог IS, USD | Итог OOS, USD | Совпадает с | CSV |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in meta:
        lines.append(f"| {'1' if m['stage'] == 'stage1' else '3'} | {m['market']} | {m['family']} | {m['name']} "
                     f"| {m['trades']} | {m['is_pnl']:+.1f} | {m['oos_pnl']:+.1f} | {m['same_as']} "
                     f"| [{Path(m['csv']).name}]({Path(m['csv']).name}) |")
    (CSV / "families" / "README.md").write_text("\n".join(lines) + "\n")


def write_readme(rows, group_rows, full_rows):
    n_main = sum(last - FIRST + 1 for _, _, last in rows)
    n_groups = sum(last - FIRST + 1 for _, _, last in group_rows)
    n_full = sum(last - FIRST + 1 for _, _, last in full_rows)
    families = json.loads((CSV / "ma_families_meta.json").read_text())
    lines = [
        "# Журнал сделок оптимизированных стратегий",
        "",
        "Все сделки по порядку и результат каждой — по каждой системе, которую оптимизация отобрала в отчётах. "
        "Полнота проверена независимым пересчётом: [`AUDIT.md`](AUDIT.md).",
        "",
        f"* **[`trade_log.xlsx`](trade_log.xlsx)** — основные отборы (стратегий: {len(rows)}, строк: {sp(n_main)}): "
        "победители семи семейств и глобальные отборы скользящих средних.",
        f"* **[`trade_log_groups.xlsx`](trade_log_groups.xlsx)** — скользящие средние, лучшие внутри каждой группы "
        f"«модель выхода × метод наклона» (стратегий: {len(group_rows)}, строк: {sp(n_groups)}).",
        f"* **[`trade_log_full.xlsx`](trade_log_full.xlsx)** — семь семейств, **сквозной прогон** по всей истории: "
        f"все сделки подряд, без разрыва между выборкой и проверкой (стратегий: {len(full_rows)}, строк: {sp(n_full)}).",
        f"* **[`csv/families/`](csv/families/README.md)** — скользящие средние, лучший в каждом из 144 семейств на "
        f"этапах 1 и 3 (файлов: {len(families)}, сделок: {sp(sum(m['trades'] for m in families))}); только CSV — "
        "для Excel это слишком много строк.",
        "* **[`csv/walkforward/`](csv/walkforward/)** — walk-forward с переоптимизацией: сделки каждого шага "
        "с параметрами, выбранными на этом шаге.",
        "",
        "* **[`points/`](points/README.md)** — исходы в **пунктах** по выгоднейшим параметрам каждой "
        "стратегии, по таблице на актив, со статистикой; [`points/full/`](points/full/README.md) — то же для "
        "сквозного прогона семи семейств.",
        "",
        "Те же сделки — в CSV: [`csv/`](csv/) для основных отборов, [`csv/groups/`](csv/groups/) "
        "(сжатые `.csv.gz`) для лучших в группах, [`csv/full/`](csv/full/) для сквозного прогона.",
        "",
        "* **Семь семейств** — победители по in-sample из `docs/RESULTS_SEVEN_FAMILIES.md`; счёт "
        "1 000 000 USD, позиция по целевой волатильности 15%, издержки вычтены. В основном журнале IS и OOS — "
        "отдельные прогоны, как в отчёте: OOS начинается без позиции и с пустыми индикаторами, между ними "
        "эмбарго 10 дней, поэтому там нет сделок за несколько месяцев, а позиция на конец IS закрыта "
        "переоценкой. Сквозной прогон этих разрывов не имеет. Позиция с изменяемым размером: уменьшение "
        "позиции записывается движком как отдельное закрытие. Последняя строка периода, если позиция "
        "осталась открытой, — её переоценка по последнему закрытию.",
        "* **Скользящие средние** — правила R1 (макс. PnL), R2 (макс. Шарп), R3 (плато) на этапах 1 и 3: "
        "по всей сетке (глобальные), внутри каждой группы «выход × метод наклона» и в каждом из 144 "
        "семейств. Каждая конфигурация торгует всю историю одним прогоном; сделка относится к IS или OOS по "
        "бару выхода. 1 контракт, без издержек (так в задании). Котировки — Open бара исполнения. «через "
        "ролл» — позиция пережила смену контракта: результат включает перекладку и не равен разности "
        "котировок. Если два правила выбрали одну конфигурацию, её сделки записаны один раз; где именно — "
        "в [`AUDIT.md`](AUDIT.md), раздел «Совпадения».",
        "",
        "Каждый файл сверен при выгрузке (`export_trades.py` падает при расхождении): OOS-итог семи семейств — "
        "с отчётом, итог скользящих средних — с VectorBT-статистикой финалиста, IS/OOS и число сделок — с "
        "полной сеткой, walk-forward — с итогами шагов отчёта.",
        "",
        "## Основные отборы",
        "",
        *HEADER, *_readme_rows(rows),
        "",
        "## Скользящие средние: лучшие в группах",
        "",
        *HEADER, *_readme_rows(group_rows),
        "",
        "## Семь семейств: сквозной прогон по всей истории",
        "",
        "| Стратегия | Сделок | Прибыльных | Итог IS, USD | Эмбарго, USD | Итог OOS, USD | Итог, USD "
        "| Макс. убыточных сделок подряд | CSV |",
        "|---|---|---|---|---|---|---|---|---|",
        *_full_rows(full_rows),
        "",
        "## Walk-forward с переоптимизацией",
        "",
        "Семь семейств: в каждом окне параметры выбираются по лучшему Шарпу на обучающем отрезке, затем отдельный "
        "прогон на тестовом — как в `research/seven_families/run_stats.py`. Скользящие средние: на каждом шаге "
        "лучшая конфигурация всей сетки по 24 месяцам, её сделки за следующие 3 месяца; сделка относится к шагу по "
        "месяцу выхода, поэтому вошедшая до начала шага помечена.",
        "",
        *_wf_rows(),
        "",
        "## Скользящие средние: лучший в каждом из 144 семейств",
        "",
        *_family_rows(),
        "",
        "Полный список с итогами и ссылками — [`csv/families/README.md`](csv/families/README.md).",
        "",
        "Пересборка: `python export_trades.py seven`, `seven-full`, `seven-wf` (окружение пакета `exchange`); "
        "`python export_trades.py ma` (нужен рабочий каталог исследования `MA5M_WORK`), `ma-families`, `ma-wf` "
        "(окружение исследования); `python export_trades.py excel`; проверка — `python audit_trades.py`.",
    ]
    (HERE / "README.md").write_text("\n".join(lines) + "\n")
    write_families_index()


def build(kind: str, out: Path, title: str, notes: list[str]):
    wb = Workbook()
    summary = wb.active
    summary.title = "Сводка"
    rows = []
    for m in strategies(kind):
        sheet, last = write_strategy(wb, m)
        rows.append((m, sheet, last))
        print(f"{sheet:22} строк {last - FIRST + 1:5d}")
    write_summary(summary, rows, title, notes)
    wb.save(out)
    print(f"записан {out} ({out.stat().st_size / 1e6:.1f} МБ)")
    return rows


def main(what: str = "all"):
    """all — все книги и README; full — только trade_log_full.xlsx; readme — только README."""
    if what == "all":
        rows = build("main", OUT, "Журнал сделок отобранных стратегий", MAIN_NOTES)
        group_rows = build("groups", OUT_GROUPS, "Журнал сделок: скользящие средние, лучшие в группах", GROUP_NOTES)
    else:
        rows, group_rows = rows_for("main"), rows_for("groups")
    if what in ("all", "full"):
        full_rows = build("full", OUT_FULL, "Журнал сделок: семь семейств, сквозной прогон по всей истории",
                          FULL_NOTES)
    else:
        full_rows = rows_for("full")
    write_readme(rows, group_rows, full_rows)


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "all")
