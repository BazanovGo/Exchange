"""Trade ledgers: every trade in order with its result, for the finalists and for walk-forward.

* **Finalists** — each instrument's stage-3 finalist (best IS plateau, ``tables/finalists.csv``)
  over the whole history.
* **Walk-forward** — the yearly re-selection of stage 1 (``tables/walk_forward.csv``, scope ``all``):
  every test year trades the config chosen on the three years before it. As in the study, a trade
  belongs to the year in which it closes, so a few trades of a year's config were opened late in the
  previous year; they are flagged (``carry_in``).

Configs are re-simulated on the 5m candles. Each trade gets its entry and exit contract and raw price
(the bar open the order fills at), the number of rolls it was carried through, its PnL in USD and ticks
for one contract and the running total. PnL comes from the held-contract path: within one contract it
equals direction x (exit - entry) x contract size; a trade carried through a roll switches contracts at
the overlap price, so its raw prices belong to different contracts. Both ledgers are checked against
the study tables (per-trade PnL, totals, yearly walk-forward results).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ma_slope import finalist as F
from src.ma_slope.data import TICK_SIZES, TICKERS, Instrument, load_instrument
from src.ma_slope.grid import EXIT_MODELS, Slope
from src.ma_slope.presentation import NAMES, TABLES, grid_meta

LEDGER_CSV = TABLES / "trade_ledger.csv"
LEDGER_XLSX = TABLES / "trade_ledger.xlsx"
WF_CSV = TABLES / "trade_ledger_walk_forward.csv"
WF_XLSX = TABLES / "trade_ledger_walk_forward.xlsx"
ORDER = ("ng", "br", "gold", "silv", "plt", "pld", "copper", "wti")
assert set(ORDER) == set(TICKERS)


@dataclass
class Ledger:
    ticker: str
    trades: pd.DataFrame        # one row per trade, in order
    multiplier: float
    split_time: pd.Timestamp
    config: str | None = None   # finalist label; walk-forward configs are per year
    years: pd.DataFrame | None = None  # walk-forward: one row per test year


def _config(ticker, stage, fast_type, fast_period, slow_type, slow_period, method, model) -> F.Config:
    slope = Slope(**grid_meta(ticker, stage)["slopes"][f"{method}/{model}"])
    return F.Config(fast_type, int(fast_period), slow_type, int(slow_period), slope, EXIT_MODELS.index(model))


def finalist_config(ticker: str, stage: str = "stage3") -> F.Config:
    fz = pd.read_csv(TABLES / "finalists.csv")
    r = fz[(fz["stage"] == stage) & (fz["tag"] == "overall") & (fz["ticker"] == ticker)].iloc[0]
    return _config(ticker, stage, r["fast_type"], r["fast_period"], r["slow_type"], r["slow_period"], r["method"],
                   r["model"])


def _trade_frame(ins: Instrument, tr: pd.DataFrame) -> pd.DataFrame:
    """Prices, contracts and PnL of simulator trades, checked against the held-contract path."""
    e = ins.index.get_indexer(tr["entry_time"])
    x = ins.index.get_indexer(tr["exit_time"])
    raw_open = ins.open / ins.factor
    direction = tr["direction"].to_numpy()
    changes = np.r_[0, np.cumsum(ins.secid[1:] != ins.secid[:-1])]
    pnl = tr["pnl"].to_numpy()
    # PnL is the held-contract path move; inside one contract that is the raw price move times the size.
    np.testing.assert_allclose(pnl, direction * (ins.path[x] - ins.path[e]) * ins.multiplier, atol=1e-6)
    same = changes[x] == changes[e]
    np.testing.assert_allclose(pnl[same], (direction * (raw_open[x] - raw_open[e]) * ins.multiplier)[same],
                               atol=1e-6)
    return pd.DataFrame({
        "instrument": NAMES[ins.ticker],
        "ticker": ins.ticker,
        "sample": tr["sample"].str.upper().to_numpy(),
        "direction": np.where(direction == 1, "Long", "Short"),
        "entry_time": tr["entry_time"].to_numpy(),
        "entry_contract": ins.secid[e],
        "entry_price": np.round(raw_open[e], 6),
        "exit_time": tr["exit_time"].to_numpy(),
        "exit_contract": ins.secid[x],
        "exit_price": np.round(raw_open[x], 6),
        "rolls": changes[x] - changes[e],
        "hours": tr["hours"].to_numpy(),
        "pnl_usd": np.round(pnl, 4),
    })


def _number(df: pd.DataFrame, ticker: str, multiplier: float) -> pd.DataFrame:
    pnl = df["pnl_usd"].to_numpy()
    df.insert(2, "trade", np.arange(1, len(df) + 1))
    df["pnl_ticks"] = np.round(pnl / (TICK_SIZES[ticker] * multiplier), 2)
    df["outcome"] = np.where(pnl > 0, "profit", np.where(pnl < 0, "loss", "zero"))
    df["cum_pnl_usd"] = np.round(np.cumsum(pnl), 4)
    return df


def ledger(ticker: str, stage: str = "stage3") -> Ledger:
    """All closed trades of the instrument's finalist, in order."""
    cfg = finalist_config(ticker, stage)
    ins = load_instrument(ticker)
    df = _number(_trade_frame(ins, F.trades(ins, cfg)), ticker, ins.multiplier)
    return Ledger(ticker, df, ins.multiplier, ins.split_time, config=cfg.label)


def walk_forward_ledger(ticker: str, stage: str = "stage1") -> Ledger | None:
    """Trades of the walk-forward selection: each test year, the config picked on the three years before."""
    wf = pd.read_csv(TABLES / "walk_forward.csv")
    wf = wf[(wf["stage"] == stage) & (wf["scope"] == "all") & (wf["ticker"] == ticker)].sort_values("test_year")
    if wf.empty:
        return None
    ins = load_instrument(ticker)
    simulated, parts, years = {}, [], []
    for r in wf.itertuples(index=False):
        key = (r.fast_type, r.fast_period, r.slow_type, r.slow_period, r.method, r.model)
        cfg = _config(ticker, stage, *key)
        if key not in simulated:
            simulated[key] = F.trades(ins, cfg)
        tr = simulated[key]
        tr = tr[tr["exit_time"].dt.year == r.test_year]  # the study books a trade in the year it closes
        part = _trade_frame(ins, tr)
        assert len(part) == r.test_trades and abs(part["pnl_usd"].sum() - r.test_pnl) < 0.05, (ticker, r.test_year)
        part.insert(2, "test_year", r.test_year)
        part.insert(3, "config", cfg.label)
        part["carry_in"] = part["entry_time"].dt.year < r.test_year
        parts.append(part)
        years.append({"instrument": NAMES[ticker], "ticker": ticker, "test_year": r.test_year,
                      "train_years": f"{r.test_year - 3}–{r.test_year - 1}", "config": cfg.label,
                      "train_plateau_t": r.train_plateau_t, "train_t": r.train_t, "train_trades": int(r.train_trades),
                      "train_pnl": r.train_pnl, "study_test_pnl": r.test_pnl, "study_test_trades": int(r.test_trades)})
    df = _number(pd.concat(parts, ignore_index=True), ticker, ins.multiplier)
    return Ledger(ticker, df, ins.multiplier, ins.split_time, years=pd.DataFrame(years))


def walk_forward_tradable(stage: str = "stage1") -> pd.DataFrame:
    """Each test year booked as if the config switched exactly at the year boundary.

    The study books a trade in the year it closes, so a year's config keeps the part of a carried-in
    trade earned before January 1 and drops its own trades still open at year end. Here every
    position of the year's config is opened at the first bar of the year (if one is open then) and
    closed at the first bar of the next year, so the year earns exactly what its config made in it.
    The last test year keeps the study rule for positions open at the end of the data.
    """
    wf = pd.read_csv(TABLES / "walk_forward.csv")
    wf = wf[(wf["stage"] == stage) & (wf["scope"] == "all")]
    rows = []
    for t, g in wf.groupby("ticker", sort=False):
        ins = load_instrument(t)
        bar_year = ins.index.year.to_numpy()
        simulated = {}
        for r in g.sort_values("test_year").itertuples(index=False):
            key = (r.fast_type, r.fast_period, r.slow_type, r.slow_period, r.method, r.model)
            if key not in simulated:
                simulated[key] = F.trades(ins, _config(t, stage, *key))
            tr = simulated[key]
            e, x = ins.index.get_indexer(tr["entry_time"]), ins.index.get_indexer(tr["exit_time"])
            d = tr["direction"].to_numpy()
            b0 = int(np.searchsorted(bar_year, r.test_year))
            b1 = int(np.searchsorted(bar_year, r.test_year + 1))
            live = (e < b1) & (x > b0)
            e2, x2 = np.maximum(e[live], b0), np.minimum(x[live], min(b1, ins.n - 1))
            rows.append({"instrument": NAMES[t], "ticker": t, "test_year": r.test_year, "study_pnl": r.test_pnl,
                         "tradable_pnl": float((d[live] * (ins.path[x2] - ins.path[e2]) * ins.multiplier).sum()),
                         "carried_in": int(((e < b0) & (x >= b0) & (x < b1)).sum()),
                         "open_at_year_end": int(((e < b1) & (x >= b1)).sum())})
    return pd.DataFrame(rows)


def all_ledgers(stage: str = "stage3") -> dict[str, Ledger]:
    return {t: ledger(t, stage) for t in ORDER}


def all_walk_forward_ledgers(stage: str = "stage1") -> dict[str, Ledger]:
    out = {t: walk_forward_ledger(t, stage) for t in ORDER}
    return {t: v for t, v in out.items() if v is not None}


def write_csv(ledgers: dict[str, Ledger], path: Path) -> Path:
    frames = [lg.trades.assign(config=lg.config) if lg.config else lg.trades for lg in ledgers.values()]
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


def load_ledger(path: Path = LEDGER_CSV) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["entry_time", "exit_time"])


# ---------------------------------------------------------------- Excel

SHEET = {"ng": "NG", "br": "BR", "gold": "GOLD", "silv": "SILV", "plt": "PLT", "pld": "PLD", "copper": "COPPER",
         "wti": "WTI"}
PRICE_FORMAT = {"ng": "0.000", "br": "0.00", "gold": "0.0", "silv": "0.00", "plt": "0.0", "pld": "0.00",
                "copper": "#,##0", "wti": "0.00"}
USD = "#,##0.00;[Red]-#,##0.00;0.00"
TICKS = "#,##0.0;[Red]-#,##0.0;0.0"
TIME = "yyyy-mm-dd hh:mm"
FIRST = 9  # first data row on an instrument sheet
NOTE = ("PnL — на 1 контракт, без комиссий и проскальзывания. Вход и выход — по Open свечи после сигнала; цены — "
        "сырые цены контракта. Сделка, прошедшая через ролл, переходит в новый контракт по цене перекрытия, поэтому "
        "её цены входа и выхода относятся к разным контрактам.")
WF_NOTE = ("Каждый тестовый год торгуется конфигурация, выбранная по трём предыдущим годам среди всех 3,47 млн "
           "конфигураций этапа 1 (наклон k = 10 задан заранее). Как и в исследовании, сделка относится к году, в "
           "котором закрыта: несколько сделок открыты конфигурацией года ещё в конце предыдущего года — они "
           "помечены «да» в колонке «Вход до начала года». ")


class _Book:
    """openpyxl workbook with the ledger styles (Arial; inputs blue, cross-sheet links green)."""

    def __init__(self):
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

        self.wb = Workbook()
        self.base = Font(name="Arial", size=10)
        self.bold = Font(name="Arial", size=10, bold=True)
        self.title = Font(name="Arial", size=12, bold=True)
        self.blue = Font(name="Arial", size=10, color="0000FF")
        self.link = Font(name="Arial", size=10, color="008000")
        self.muted = Font(name="Arial", size=9, italic=True, color="595959")
        self.green = Font(name="Arial", size=10, color="1B7A3D")
        self.red = Font(name="Arial", size=10, color="C00000")
        self.head_fill = PatternFill("solid", fgColor="E7E6E1")
        self.head_border = Border(bottom=Side(style="thin", color="7F7F7F"))
        self.top_border = Border(top=Side(style="thin", color="7F7F7F"))
        self.wrap = Alignment(wrap_text=True, vertical="top")
        self.head_align = Alignment(wrap_text=True, vertical="center")

    def put(self, ws, row, col, value, fmt=None, font=None):
        c = ws.cell(row, col, value)
        c.font = font or self.base
        if fmt:
            c.number_format = fmt
        return c

    def header(self, ws, row, columns):
        from openpyxl.utils import get_column_letter

        for j, (name, width) in enumerate(columns, start=1):
            c = ws.cell(row, j, name)
            c.font, c.fill, c.border, c.alignment = self.bold, self.head_fill, self.head_border, self.head_align
            ws.column_dimensions[get_column_letter(j)].width = width
        ws.row_dimensions[row].height = 30

    def note(self, ws, row, text, last_col, height):
        ws.merge_cells(f"A{row}:{last_col}{row}")
        self.put(ws, row, 1, text, font=self.muted).alignment = self.wrap
        ws.row_dimensions[row].height = height

    def outcome_colours(self, ws, col, first, last):
        from openpyxl.formatting.rule import FormulaRule

        rng = f"{col}{first}:{col}{last}"
        ws.conditional_formatting.add(rng, FormulaRule(formula=[f'{col}{first}="Прибыль"'], font=self.green))
        ws.conditional_formatting.add(rng, FormulaRule(formula=[f'{col}{first}="Убыток"'], font=self.red))


def _instrument_sheet(book: _Book, lg: Ledger, title: str, lead: list, note: str) -> dict:
    """Every trade in order: ``lead`` value columns, the trade details, PnL, then formula columns.

    ``lead`` holds (header, width, field, number format) for the columns after "№". Returns the
    column letters of the fields the summary sheets refer to.
    """
    from openpyxl.comments import Comment
    from openpyxl.utils import get_column_letter

    t, df = lg.ticker, lg.trades
    ws = book.wb.create_sheet(SHEET[t])
    put = book.put
    put(ws, 1, 1, title, font=book.title)
    put(ws, 2, 1, "Размер контракта, USD за 1,0 цены")
    put(ws, 2, 2, lg.multiplier, font=book.blue)
    put(ws, 3, 1, "Шаг цены (тик)")
    put(ws, 3, 2, TICK_SIZES[t], font=book.blue)
    put(ws, 4, 1, "Стоимость тика, USD")
    put(ws, 4, 2, "=B2*B3", "0.00")
    put(ws, 5, 1, "Граница IS/OOS исследования")
    put(ws, 5, 2, lg.split_time.to_pydatetime(), TIME)
    ws.cell(2, 3, "спецификация MOEX (src/ma_slope/data.py)").font = book.muted
    ws.cell(5, 3, "первые 70 % 5-минутных баров — IS").font = book.muted

    pf = PRICE_FORMAT[t]
    fields = [("№", 7, "trade", None), *lead,
              ("Направление", 11, "direction", None), ("Вход: время", 17, "entry_time", TIME),
              ("Вход: контракт", 12, "entry_contract", None), ("Вход: цена", 11, "entry_price", pf),
              ("Выход: время", 17, "exit_time", TIME), ("Выход: контракт", 12, "exit_contract", None),
              ("Выход: цена", 11, "exit_price", pf), ("Роллов внутри сделки", 9, "rolls", None),
              ("Длительность, ч", 10, "hours", "0.0"), ("PnL, $", 11, "pnl_usd", USD)]
    col = {f: get_column_letter(j) for j, (_, _, f, _) in enumerate(fields, start=1)}
    n = len(fields)
    L = col["pnl_usd"]
    out_c, tick_c, cum_c, peak_c, dd_c, streak_c = (get_column_letter(n + k) for k in range(1, 7))
    formulas = [("Исход", 10), ("PnL, тиков", 10), ("Накопленный PnL, $", 13), ("Пик накопленного, $", 13),
                ("Просадка от пика, $", 12), ("Убытков подряд", 9)]
    last_col = streak_c
    book.note(ws, 6, note, last_col, 52)
    book.header(ws, FIRST - 1, [(h, w) for h, w, _, _ in fields] + formulas)
    ws.cell(FIRST - 1, n).comment = Comment("Результат сделки из симулятора исследования (держимый контракт, "
                                            "1 контракт).", "ledger")

    def value(v):
        if isinstance(v, (pd.Timestamp, np.datetime64)):
            return pd.Timestamp(v).to_pydatetime()
        if isinstance(v, (bool, np.bool_)):
            return "да" if v else "нет"
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        return v

    last = FIRST + len(df) - 1
    data = df[[f for _, _, f, _ in fields]].to_numpy(object)
    fmts = [fmt for _, _, _, fmt in fields]
    for i, rec in enumerate(data):
        row = FIRST + i
        for j, (v, fmt) in enumerate(zip(rec, fmts, strict=True), start=1):
            put(ws, row, j, value(v), fmt)
        put(ws, row, n + 1, f'=IF({L}{row}>0,"Прибыль",IF({L}{row}<0,"Убыток","Ноль"))')
        put(ws, row, n + 2, f"={L}{row}/$B$4", TICKS)
        put(ws, row, n + 3, f"={L}{row}" if row == FIRST else f"={cum_c}{row - 1}+{L}{row}", USD)
        put(ws, row, n + 4, f"=MAX(0,{cum_c}{row})" if row == FIRST else f"=MAX({peak_c}{row - 1},{cum_c}{row})", USD)
        put(ws, row, n + 5, f"={peak_c}{row}-{cum_c}{row}", USD)
        put(ws, row, n + 6, f"=IF({L}{row}<0,1,0)" if row == FIRST else f"=IF({L}{row}<0,{streak_c}{row - 1}+1,0)")
    book.outcome_colours(ws, out_c, FIRST, last)
    ws.freeze_panes = f"B{FIRST}"
    ws.auto_filter.ref = f"A{FIRST - 1}:{last_col}{last}"
    return {**col, "dd": dd_c, "streak": streak_c, "first": FIRST, "last": last, "sheet": f"'{SHEET[t]}'!"}


def _all_trades_sheet(book: _Book, ledgers: dict[str, Ledger], cols: dict, title: str, note: str, lead: list):
    """All instruments in the order trades closed; PnL links to the instrument sheets."""
    ws = book.wb["Все сделки"]
    put = book.put
    allt = pd.concat([lg.trades.assign(sheet_row=np.arange(FIRST, FIRST + len(lg.trades)))
                      for lg in ledgers.values()])
    allt = allt.sort_values(["exit_time", "entry_time", "ticker"], kind="stable").reset_index(drop=True)
    put(ws, 1, 1, title, font=book.title)
    fields = [("№", 8, None), ("Инструмент", 13, "instrument"), ("№ сделки в инструменте", 11, "trade"),
              *[(h, w, f) for h, w, f, _ in lead], ("Направление", 11, "direction"),
              ("Вход: время", 17, "entry_time"), ("Выход: время", 17, "exit_time"), ("Длительность, ч", 10, "hours")]
    from openpyxl.utils import get_column_letter

    n = len(fields)
    pnl_c, out_c, cum_c = (get_column_letter(n + k) for k in range(1, 4))
    book.note(ws, 2, note, cum_c, 28)
    book.header(ws, 4, [(h, w) for h, w, _ in fields] + [("PnL, $", 11), ("Исход", 10), ("Накопленный PnL, $", 14)])
    fmt = {"entry_time": TIME, "exit_time": TIME, "hours": "0.0"}
    for i, r in enumerate(allt.itertuples(index=False)):
        row = 5 + i
        put(ws, row, 1, i + 1)
        for j, (_, _, f) in enumerate(fields[1:], start=2):
            v = getattr(r, f)
            v = pd.Timestamp(v).to_pydatetime() if f.endswith("_time") else (
                int(v) if isinstance(v, (int, np.integer)) else (float(v) if f == "hours" else v))
            put(ws, row, j, v, fmt.get(f))
        put(ws, row, n + 1, f"={cols[r.ticker]['sheet']}{cols[r.ticker]['pnl_usd']}{r.sheet_row}", USD,
            font=book.link)
        put(ws, row, n + 2, f'=IF({pnl_c}{row}>0,"Прибыль",IF({pnl_c}{row}<0,"Убыток","Ноль"))')
        put(ws, row, n + 3, f"={pnl_c}{row}" if row == 5 else f"={cum_c}{row - 1}+{pnl_c}{row}", USD)
    last = 4 + len(allt)
    book.outcome_colours(ws, out_c, 5, last)
    ws.freeze_panes = "B5"
    ws.auto_filter.ref = f"A4:{cum_c}{last}"


def _money_columns(book, ws, row, start, c, extra_first=()):
    """Shared summary cells from ``start``: counts, win share, total, averages, extremes, streak, drawdown."""
    s, a, b = c["sheet"], c["first"], c["last"]
    pnl = f"{s}${c['pnl_usd']}${a}:${c['pnl_usd']}${b}"
    from openpyxl.utils import get_column_letter as col

    k = start
    n_c, win_c, tot_c = col(k), col(k + 1), col(k + 4)
    cells = [(f"=COUNT({pnl})", "#,##0"), (f'=COUNTIF({pnl},">0")', "#,##0"), (f'=COUNTIF({pnl},"<0")', "#,##0"),
             (f"=IF({n_c}{row}>0,{win_c}{row}/{n_c}{row},0)", "0.0%"), (f"=SUM({pnl})", USD), *extra_first,
             (f"=IF({n_c}{row}>0,{tot_c}{row}/{n_c}{row},0)", USD),
             (f"=IF({n_c}{row}>0,{tot_c}{row}/{n_c}{row}/{s}$B$4,0)", TICKS),
             (f'=IFERROR(AVERAGEIF({pnl},">0"),0)', USD), (f'=IFERROR(AVERAGEIF({pnl},"<0"),0)', USD),
             (f"=MAX({pnl})", USD), (f"=MIN({pnl})", USD),
             (f"=MAX({s}${c['streak']}${a}:${c['streak']}${b})", "0"), (f"=MAX({s}${c['dd']}${a}:${c['dd']}${b})", USD)]
    for j, (formula, fmt) in enumerate(cells):
        book.put(ws, row, k + j, formula, fmt, book.link if "!" in formula else None)


MONEY_HEADERS = [("Сделок", 9), ("Прибыльных", 11), ("Убыточных", 11), ("Доля прибыльных", 11), ("Итого PnL, $", 12)]
MONEY_TAIL = [("Средняя сделка, $", 12), ("Средняя сделка, тиков", 12), ("Средняя прибыль, $", 12),
              ("Средний убыток, $", 12), ("Лучшая сделка, $", 12), ("Худшая сделка, $", 12),
              ("Макс. убытков подряд", 11), ("Макс. просадка, $", 12)]


def _total_row(book, ws, row, n_inst, sum_cols, ratio):
    from openpyxl.utils import get_column_letter

    book.put(ws, row, 1, "Всего", font=book.bold)
    for c in sum_cols:
        letter = get_column_letter(c)
        book.put(ws, row, c, f"=SUM({letter}5:{letter}{4 + n_inst})", ws.cell(5, c).number_format, book.bold)
    win, n, share = ratio
    book.put(ws, row, share, f"=IF({n}{row}>0,{win}{row}/{n}{row},0)", "0.0%", book.bold)
    for c in range(1, ws.max_column + 1):
        ws.cell(row, c).border = book.top_border


def write_excel(ledgers: dict[str, Ledger], path: Path = LEDGER_XLSX) -> Path:
    """Finalists: summary (formulas), all trades by exit time, one sheet per instrument in trade order."""
    book = _Book()
    summary = book.wb.active
    summary.title = "Сводка"
    book.wb.create_sheet("Все сделки")
    lead = [("Выборка", 8, "sample", None)]
    cols = {t: _instrument_sheet(book, lg, f"{NAMES[t]}: {lg.config} — все сделки по порядку", lead,
                                 NOTE + " Выборка (IS/OOS) — по времени выхода.") for t, lg in ledgers.items()}
    _all_trades_sheet(book, ledgers, cols, "Все сделки всех инструментов в порядке закрытия (по 1 контракту каждого)",
                      "Накопленный итог — сумма по всем инструментам в долларах США; WTI торгуется на MOEX с июля "
                      "2026 года и приведён справочно. PnL берётся ссылкой с листа инструмента.", lead)

    put = book.put
    put(summary, 1, 1, "Все сделки лучших конфигураций (этап 3): сводка по инструментам", font=book.title)
    book.note(summary, 2, "Конфигурация каждого инструмента — лучшая по IS-плато (tables/finalists.csv). " + NOTE,
              "Q", 52)
    book.header(summary, 4, [("Инструмент", 13), ("Конфигурация", 44), *MONEY_HEADERS, ("PnL IS, $", 12),
                             ("PnL OOS, $", 12), *MONEY_TAIL])
    for i, (t, lg) in enumerate(ledgers.items()):
        row, c = 5 + i, cols[t]
        smp = f"{c['sheet']}${c['sample']}${c['first']}:${c['sample']}${c['last']}"
        pnl = f"{c['sheet']}${c['pnl_usd']}${c['first']}:${c['pnl_usd']}${c['last']}"
        put(summary, row, 1, NAMES[t], font=book.bold)
        put(summary, row, 2, lg.config)
        _money_columns(book, summary, row, 3, c, extra_first=[(f'=SUMIF({smp},"IS",{pnl})', USD),
                                                              (f'=SUMIF({smp},"OOS",{pnl})', USD)])
    row = 5 + len(ledgers)
    _total_row(book, summary, row, len(ledgers), (3, 4, 5, 7, 8, 9), ("D", "C", 6))
    put(summary, row, 2, f"сумма по {len(ledgers)} инструментам, по 1 контракту каждого", font=book.muted)
    put(summary, row, 10, f"=IF(C{row}>0,G{row}/C{row},0)", USD, book.bold)
    put(summary, row + 2, 1, "WTI (CL) торгуется на MOEX с июля 2026 года (2 181 пятиминутный бар) — справочно; в "
        "выводах исследования не участвует. У COPPER (HG) два года истории.", font=book.muted)
    put(summary, row + 3, 1, "Листы инструментов — каждая сделка по порядку с накопленным итогом, просадкой и "
        "сериями убытков; «Все сделки» — все инструменты вместе в порядке закрытия.", font=book.muted)
    summary.freeze_panes = "C5"
    book.wb.calculation.fullCalcOnLoad = True  # spreadsheet apps compute every formula when the file opens
    book.wb.save(path)
    return path


def write_walk_forward_excel(ledgers: dict[str, Ledger], path: Path = WF_XLSX) -> Path:
    """Walk-forward: summary, results by test year (checked against the study), all trades, sheet per instrument."""
    book = _Book()
    summary = book.wb.active
    summary.title = "Сводка"
    by_year = book.wb.create_sheet("По годам")
    book.wb.create_sheet("Все сделки")
    lead = [("Тестовый год", 9, "test_year", "0"), ("Конфигурация года", 44, "config", None),
            ("Вход до начала года", 9, "carry_in", None)]
    cols = {t: _instrument_sheet(book, lg, f"{NAMES[t]}: walk-forward (переотбор раз в год) — все сделки по порядку",
                                 lead, WF_NOTE + NOTE) for t, lg in ledgers.items()}
    _all_trades_sheet(book, ledgers, cols, "Walk-forward: все сделки всех инструментов в порядке закрытия "
                      "(по 1 контракту каждого)", "Накопленный итог — сумма по всем инструментам в долларах США. "
                      "PnL берётся ссылкой с листа инструмента.", lead[:1])
    put = book.put

    # Results by test year: formulas over the instrument sheets next to the study's own walk-forward numbers.
    put(by_year, 1, 1, "Walk-forward по тестовым годам: выбранная конфигурация и её результат", font=book.title)
    book.note(by_year, 2, "Конфигурация выбрана по трём предыдущим годам (обучение) по сглаженной t-статистике "
              "средней сделки. Сделок и PnL — формулы по листу инструмента; «PnL по исследованию» — "
              "tables/walk_forward.csv, расхождение — проверка (округление до 0,1 $ при хранении).", "L", 40)
    book.header(by_year, 4, [("Инструмент", 13), ("Тестовый год", 9), ("Обучение", 11), ("Конфигурация года", 44),
                             ("t-плато на обучении", 10), ("Сделок на обучении", 10), ("Сделок", 9),
                             ("Прибыльных", 10), ("PnL, $", 12), ("Средняя сделка, $", 11),
                             ("PnL по исследованию, $", 12), ("Расхождение, $", 11)])
    row = 5
    year_rows = {}
    for t, lg in ledgers.items():
        c = cols[t]
        s, a, b = c["sheet"], c["first"], c["last"]
        yr, pnl = f"{s}${c['test_year']}${a}:${c['test_year']}${b}", f"{s}${c['pnl_usd']}${a}:${c['pnl_usd']}${b}"
        start = row
        for y in lg.years.itertuples(index=False):
            put(by_year, row, 1, y.instrument, font=book.bold)
            put(by_year, row, 2, int(y.test_year), "0")
            put(by_year, row, 3, y.train_years)
            put(by_year, row, 4, y.config)
            put(by_year, row, 5, float(y.train_plateau_t), "0.00")
            put(by_year, row, 6, int(y.train_trades), "#,##0")
            put(by_year, row, 7, f"=COUNTIF({yr},B{row})", "#,##0", book.link)
            put(by_year, row, 8, f'=COUNTIFS({yr},B{row},{pnl},">0")', "#,##0", book.link)
            put(by_year, row, 9, f"=SUMIF({yr},B{row},{pnl})", USD, book.link)
            put(by_year, row, 10, f"=IF(G{row}>0,I{row}/G{row},0)", USD)
            put(by_year, row, 11, round(float(y.study_test_pnl), 4), USD, book.blue)
            put(by_year, row, 12, f"=I{row}-K{row}", USD)
            row += 1
        year_rows[t] = (start, row - 1)
    by_year.freeze_panes = "C5"
    by_year.auto_filter.ref = f"A4:L{row - 1}"

    put(summary, 1, 1, "Walk-forward (ежегодный переотбор, этап 1): все сделки — сводка по инструментам",
        font=book.title)
    book.note(summary, 2, WF_NOTE + NOTE, "S", 64)
    book.header(summary, 4, [("Инструмент", 13), ("Тестовые годы", 11), ("Тестовых лет", 9), ("Лет в плюсе", 9),
                             *MONEY_HEADERS, *MONEY_TAIL, ("Сделок с входом до начала года", 11),
                             ("Их PnL, $", 11)])
    for i, (t, lg) in enumerate(ledgers.items()):
        row, c = 5 + i, cols[t]
        a, b = year_rows[t]
        s = c["sheet"]
        carry = f"{s}${c['carry_in']}${c['first']}:${c['carry_in']}${c['last']}"
        pnl = f"{s}${c['pnl_usd']}${c['first']}:${c['pnl_usd']}${c['last']}"
        put(summary, row, 1, NAMES[t], font=book.bold)
        put(summary, row, 2, f"{lg.years['test_year'].min()}–{lg.years['test_year'].max()}")
        put(summary, row, 3, f"=COUNT('По годам'!$B${a}:$B${b})", "0", book.link)
        put(summary, row, 4, f"=COUNTIF('По годам'!$I${a}:$I${b},\">0\")", "0", book.link)
        _money_columns(book, summary, row, 5, c)
        put(summary, row, 18, f'=COUNTIF({carry},"да")', "0", book.link)
        put(summary, row, 19, f'=SUMIF({carry},"да",{pnl})', USD, book.link)
    row = 5 + len(ledgers)
    _total_row(book, summary, row, len(ledgers), (3, 4, 5, 6, 7, 9, 18, 19), ("F", "E", 8))
    put(summary, row, 10, f"=IF(E{row}>0,I{row}/E{row},0)", USD, book.bold)
    put(summary, row + 2, 1, "COPPER (HG) и WTI (CL): истории меньше четырёх лет, walk-forward не считался. Тестовый "
        "2026 год неполный (по 13 октября).", font=book.muted)
    put(summary, row + 3, 1, "«По годам» — конфигурация каждого тестового года и её результат со сверкой с "
        "исследованием; листы инструментов — каждая сделка по порядку.", font=book.muted)
    put(summary, row + 4, 1, "Проверка учёта на границах лет: если переключать конфигурацию ровно 1 января "
        "(старая позиция закрывается, текущая позиция новой конфигурации открывается по цене начала года), итог "
        "почти не меняется — расчёт в notebooks/09_walk_forward_ledger.ipynb.", font=book.muted)
    summary.freeze_panes = "B5"
    book.wb.calculation.fullCalcOnLoad = True
    book.wb.save(path)
    return path


# ---------------------------------------------------------------- notebook helpers

def summary_table(led: pd.DataFrame) -> pd.DataFrame:
    """Per-instrument counts and money, as in the workbooks' summary sheets."""
    rows = {}
    for name, g in led.groupby("instrument", sort=False):
        p, cum = g["pnl_usd"], g["cum_pnl_usd"].to_numpy()
        losing = (p < 0).to_numpy()
        runs = np.diff(np.flatnonzero(np.r_[True, ~losing, True])) - 1
        row = {"сделок": len(g), "прибыльных": int((p > 0).sum()), "убыточных": int((p < 0).sum()),
               "доля прибыльных": (p > 0).mean(), "итого PnL, $": p.sum()}
        if "test_year" in g:
            yearly = g.groupby("test_year")["pnl_usd"].sum()
            row.update({"тестовых лет": len(yearly), "лет в плюсе": int((yearly > 0).sum())})
        else:
            row.update({"PnL IS, $": p[g["sample"] == "IS"].sum(), "PnL OOS, $": p[g["sample"] == "OOS"].sum()})
        row.update({"средняя сделка, $": p.mean(), "средняя сделка, тиков": g["pnl_ticks"].mean(),
                    "лучшая, $": p.max(), "худшая, $": p.min(), "макс. убытков подряд": int(runs.max()),
                    "макс. просадка, $": float((np.maximum.accumulate(np.maximum(cum, 0)) - cum).max())})
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def outcomes_figure(led: pd.DataFrame, path: Path, title: str | None = None) -> Path:
    """Per instrument: every trade's PnL in order (bars) and the running total.

    The finalist ledger marks the IS/OOS boundary; the walk-forward ledger marks the test years.
    """
    from src.ma_slope import plots as P

    plt = P.plt
    names = list(dict.fromkeys(led["instrument"]))
    ncols = 4 if len(names) > 6 else 3
    nrows = int(np.ceil(len(names) / ncols))
    fig = plt.figure(figsize=(4.2 * ncols, 5 * nrows))
    outer = fig.add_gridspec(nrows, ncols, hspace=0.45, wspace=0.3, top=1 - 0.16 / nrows)
    wf = "test_year" in led
    for k, name in enumerate(names):
        g = led[led["instrument"] == name]
        n = np.arange(1, len(g) + 1)
        inner = outer[k // ncols, k % ncols].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.08)
        top, bottom = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        top.plot(n, g["cum_pnl_usd"].to_numpy(), color=P.SERIES[0], lw=1.4)
        pnl = g["pnl_usd"].to_numpy()
        bottom.vlines(n, 0, pnl, colors=np.where(pnl >= 0, P.POS, P.NEG), lw=0.6)
        if wf:
            year = g["test_year"].to_numpy()
            marks = np.flatnonzero(year[1:] != year[:-1]) + 0.5
            lw, colour = 0.6, P.BASE
        else:
            oos = np.flatnonzero(g["sample"].to_numpy() == "OOS")
            marks, lw, colour = oos[:1] + 0.5, 1.0, P.INK2
        for ax in (top, bottom):
            ax.axhline(0, color=P.BASE, lw=0.8)
            ax.set_xlim(0, len(g) + 1)
            for m in marks:
                ax.axvline(m, color=colour, lw=lw)
        span = f", {g['test_year'].min()}–{g['test_year'].max()}" if wf else ""
        top.set_title(f"{name}\n{len(g):,} trades{span}, total {pnl.sum():+,.0f} $")
        top.tick_params(labelbottom=False)
        bottom.set_xlabel("trade number")
        if k % ncols == 0:
            top.set_ylabel("running total, $")
            bottom.set_ylabel("trade PnL, $")
    default = ("Every trade of each instrument's selected config in order: PnL per trade (blue gain, red loss)\n"
               "and running total, 1 contract; line = IS/OOS boundary")
    fig.suptitle(title or default, fontsize=10, color=P.INK2, y=0.995)
    return P._save(fig, path)


DECIMALS = {"ng": 3, "br": 2, "gold": 1, "silv": 2, "plt": 1, "pld": 2, "copper": 0, "wti": 2}
OUTCOME_RU = {"profit": "прибыль", "loss": "убыток", "zero": "ноль"}


def _text_table(g: pd.DataFrame, ticker: str, sample: bool = True) -> str:
    d = DECIMALS[ticker]
    cols = {"№": g["trade"].to_numpy()}
    if sample:
        cols["выборка"] = g["sample"].to_numpy()
    cols.update({
        "сторона": g["direction"].to_numpy(),
        "вход": g["entry_time"].dt.strftime("%Y-%m-%d %H:%M").to_numpy(),
        "контракт": g["entry_contract"].to_numpy(),
        "цена входа": g["entry_price"].map(lambda v: f"{v:,.{d}f}").to_numpy(),
        "выход": g["exit_time"].dt.strftime("%Y-%m-%d %H:%M").to_numpy(),
        "контракт ": g["exit_contract"].to_numpy(),
        "цена выхода": g["exit_price"].map(lambda v: f"{v:,.{d}f}").to_numpy(),
        "часов": g["hours"].map(lambda v: f"{v:.1f}").to_numpy(),
        "PnL, $": g["pnl_usd"].map(lambda v: f"{v:+,.2f}").to_numpy(),
        "тиков": g["pnl_ticks"].map(lambda v: f"{v:+,.1f}").to_numpy(),
        "исход": g["outcome"].map(OUTCOME_RU).to_numpy(),
        "накопленный, $": g["cum_pnl_usd"].map(lambda v: f"{v:+,.2f}").to_numpy(),
    })
    return pd.DataFrame(cols).to_string(index=False)


def format_trades(led: pd.DataFrame, ticker: str) -> str:
    """Every trade of one instrument as a plain-text table (renders compactly in a notebook on GitHub)."""
    return _text_table(led[led["ticker"] == ticker], ticker)


def format_walk_forward_trades(led: pd.DataFrame, ticker: str) -> str:
    """Walk-forward trades of one instrument, year by year, each year headed by its config."""
    g = led[led["ticker"] == ticker]
    blocks = []
    for year, part in g.groupby("test_year"):
        carry = int(part["carry_in"].sum())
        head = (f"== {year}: {part['config'].iloc[0]} · обучение {year - 3}–{year - 1} · сделок {len(part)} · "
                f"за год {part['pnl_usd'].sum():+,.2f} $"
                + (f" · {carry} с входом до начала года" if carry else ""))
        blocks.append(head + "\n" + _text_table(part, ticker, sample=False))
    return "\n\n".join(blocks)
