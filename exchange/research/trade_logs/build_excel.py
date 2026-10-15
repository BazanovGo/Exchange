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


def strategies() -> list[dict]:
    seven = json.loads((CSV / "seven_meta.json").read_text())
    ma = json.loads((CSV / "ma_meta.json").read_text())
    for m in seven:
        m["study"] = "Семь семейств (дневные бары)"
        m["title"] = m["name"] + " · " + m["params"]
        m["ref"], m["ref_kind"] = m["oos_return_report"] * 1_000_000, "OOS"
        m["ref_note"] = (f"docs/RESULTS_SEVEN_FAMILIES.md, раздел 2, «итог. дох.» "
                         f"{m['oos_return_report']:+.1%} × 1 000 000")
        m["time_fmt"] = "yyyy-mm-dd"
    for m in ma:
        stage = "этап 1" if m["stage"] == "stage1" else "этап 3"
        m["study"] = f"Скользящие средние, 5 мин, {stage}"
        m["title"] = f"{m['market']} · {m['rule']} · {m['name']}"
        m["ref"], m["ref_kind"] = m["full_pnl"], "весь период"
        m["ref_note"] = (f"research/ma_crossover_5m/results/{m['stage']}/{m['market']}/finalists.json, "
                         "full.pnl (VectorBT)")
        m["time_fmt"] = "yyyy-mm-dd hh:mm"
    return seven + ma


def write_strategy(wb: Workbook, m: dict) -> tuple[str, int]:
    df = pd.read_csv(CSV / f"{m['id']}.csv", keep_default_na=False, na_values=[""])
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


def write_summary(ws, rows: list[tuple[dict, str, int]]):
    ws["A1"] = "Журнал сделок отобранных стратегий"
    ws["A1"].font = font(size=14, bold=True)
    notes = [
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


def write_readme(rows: list[tuple[dict, str, int]]):
    lines = [
        "# Журнал сделок отобранных стратегий",
        "",
        "Все сделки по порядку и результат каждой: **[`trade_log.xlsx`](trade_log.xlsx)** — лист «Сводка» "
        "и лист на каждую стратегию с итогом нарастающим и сериями убытков (формулы Excel). Те же сделки — "
        "в CSV в [`csv/`](csv/), по файлу на стратегию.",
        "",
        "* **Семь семейств** — победители по in-sample из `docs/RESULTS_SEVEN_FAMILIES.md`; счёт "
        "1 000 000 USD, позиция по целевой волатильности 15%, издержки вычтены. IS и OOS — отдельные "
        "прогоны, как в отчёте. Позиция с изменяемым размером: уменьшение позиции записывается движком "
        "как отдельное закрытие. Последняя строка периода, если позиция осталась открытой, — её "
        "переоценка по последнему закрытию.",
        "* **Скользящие средние** — глобальные отборы R1 (макс. PnL), R2 (макс. Шарп), R3 (плато) "
        "на этапах 1 и 3; 1 контракт, без издержек (так в задании). Котировки — Open бара исполнения. "
        "«через ролл» — позиция пережила смену контракта: результат включает перекладку и не равен "
        "разности котировок.",
        "",
        "Каждый файл сверен с отчётом при выгрузке (`export_trades.py` падает при расхождении): "
        "OOS-итог семи семейств — с «итог. дох.» отчёта, итог скользящих средних — с VectorBT-статистикой "
        "финалиста, а IS/OOS — с полной сеткой.",
        "",
        "| Стратегия | Сделок | Прибыльных | Итог IS, USD | Итог OOS, USD | Макс. убыточных сделок подряд | CSV |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, _sheet, _ in rows:
        df = pd.read_csv(CSV / f"{m['id']}.csv")
        trades = df[df["вход"].notna()]
        streak = best = 0
        for v in df["результат, USD"]:
            streak = streak + 1 if v < 0 else 0
            best = max(best, streak)
        is_, oos = (df.loc[df["период"] == p, "результат, USD"].sum() for p in ("IS", "OOS"))
        num = lambda v: f"{v:+,.0f}".replace(",", " ")  # noqa: E731
        lines.append(f"| {m['study']}: {m['title']} | {len(trades)} | {(trades['результат, USD'] > 0).mean():.0%} "
                     f"| {num(is_)} | {num(oos)} | {best} | [{m['id']}.csv](csv/{m['id']}.csv) |")
    lines += ["", "Пересборка: `python export_trades.py seven` (окружение пакета `exchange`), "
              "`python export_trades.py ma` (нужен рабочий каталог исследования `MA5M_WORK`), "
              "`python export_trades.py excel`."]
    (HERE / "README.md").write_text("\n".join(lines) + "\n")


def main():
    wb = Workbook()
    summary = wb.active
    summary.title = "Сводка"
    rows = []
    for m in strategies():
        sheet, last = write_strategy(wb, m)
        rows.append((m, sheet, last))
        print(f"{sheet:22} строк {last - FIRST + 1:5d}")
    write_summary(summary, rows)
    wb.save(OUT)
    write_readme(rows)
    print(f"записан {OUT} ({OUT.stat().st_size / 1e6:.1f} МБ)")


if __name__ == "__main__":
    main()
