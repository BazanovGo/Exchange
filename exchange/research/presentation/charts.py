"""Графики и таблицы презентационного ноутбука.

Источник — только файлы репозитория: сгенерированные отчёты в docs/,
машиночитаемые результаты в research/ma_crossover_5m/results/ и графики,
уже сохранённые в ноутбуках и docs/figures/. Ничего не пересчитывается,
поэтому ноутбук исполняется за секунды и без исходных данных.
"""
from __future__ import annotations

import base64
import json
import re
import sys
from functools import cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image

ROOT = Path(__file__).resolve().parents[2]          # каталог exchange/
DOCS = ROOT / "docs"
NOTEBOOKS = ROOT / "notebooks"
MA = ROOT / "research" / "ma_crossover_5m"
RES = MA / "results"
TICKERS = ("NG", "BR")
NAMES = {"NG": "природный газ", "BR": "Brent"}

sys.path.insert(0, str(MA))
import plots as P  # noqa: E402  (палитра и оформление исследования)


# --------------------------------------------------------------- данные
@cache
def res(stage: str, ticker: str | None, name: str):
    path = RES / stage / ticker / name if ticker else RES / stage / name
    return json.loads(path.read_text()) if path.suffix == ".json" else pd.read_csv(path)


@cache
def top(name: str):
    return json.loads((RES / name).read_text())


SHORT = {"1": "TSMOM", "2": "Dual MA", "3": "Breakout", "4": "Cross-sect.",
         "5": "Carry", "6": "Portfolio MA", "7": "Mom+Rev"}


@cache
def seven_families() -> pd.DataFrame:
    text = (DOCS / "RESULTS_SEVEN_FAMILIES.md").read_text()
    a = md_table(text, "конец IS")[["ячейка", "конфигураций", "лучшая"]]
    b = md_table(text, "IS Sharpe")[["ячейка", "IS Sharpe", "OOS Sharpe", "сделок"]]
    c = md_table(text, "DSR программа")[["ячейка", "PBO", "DSR программа"]]
    df = a.merge(b, on="ячейка").merge(c, on="ячейка")
    verdicts = dict(re.findall(r"\*\*(\d [^*]+?)\*\* — (\w+)", text))
    df["вердикт"] = df["ячейка"].map(verdicts)
    parts = df["ячейка"].str.split(" ", n=1)
    df["ячейка"] = parts.str[0].map(SHORT) + " · " + parts.str[1].str.split(" · ").str[-1]
    return df


def md_table(text: str, column: str, first: str = "ячейка") -> pd.DataFrame:
    """Таблица сгенерированного отчёта: первая, где есть столбцы first и column."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(f"| {first} |") and f"| {column} |" in ln)
    rows = []
    for ln in lines[start:]:
        if not ln.startswith("|"):
            break
        rows.append([c.strip() for c in ln.strip("|").split("|")])
    df = pd.DataFrame(rows[2:], columns=rows[0])
    for c in df.columns[1:]:
        conv = pd.to_numeric(df[c].str.replace("%", "", regex=False), errors="coerce")
        if conv.notna().any():
            df[c] = conv
    return df


# --------------------------------------------------------------- готовые картинки
def figure(name: str) -> Image:
    """График из docs/figures/ma_crossover_5m/."""
    return Image(filename=str(DOCS / "figures" / "ma_crossover_5m" / name))


def notebook_image(notebook: str, index: int) -> Image:
    """index-я картинка, сохранённая в другом ноутбуке репозитория."""
    nb = json.loads((NOTEBOOKS / notebook).read_text())
    pngs = [o["data"]["image/png"] for c in nb["cells"] for o in c.get("outputs", [])
            if "image/png" in o.get("data", {})]
    png = pngs[index]
    return Image(data=base64.b64decode("".join(png) if isinstance(png, list) else png))


# --------------------------------------------------------------- графики
def _frame(ax, title, subtitle="", **kw):
    """P.frame с запасом между заголовком и подзаголовком."""
    P.frame(ax, **kw)
    ax.set_title(title, color=P.INK, fontsize=11.5, fontweight="bold", loc="left", pad=24 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.025, subtitle, transform=ax.transAxes, color=P.INK_2, fontsize=9, va="bottom")


def _end_label(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                color=P.INK_2, fontsize=9, fontweight="bold")
    ax.plot([x], [y], "o", color=color, markersize=6, markeredgecolor=P.SURFACE, markeredgewidth=1.5)


def seven_families_chart():
    df = seven_families().sort_values("OOS Sharpe").reset_index(drop=True)
    y = np.arange(len(df))
    _, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True,
                                 gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.08})
    for i, r in df.iterrows():
        a1.plot([r["IS Sharpe"], r["OOS Sharpe"]], [i, i], color=P.AXIS, linewidth=2, zorder=2)
    a1.scatter(df["IS Sharpe"], y, s=64, facecolor=P.SURFACE, edgecolor=P.MUTED, linewidth=1.8,
               zorder=3, label="в выборке (по нему выбирали)")
    pos = df["OOS Sharpe"] > 0
    a1.scatter(df.loc[pos, "OOS Sharpe"], y[pos], s=64, color=P.POS, edgecolor=P.SURFACE,
               linewidth=1.5, zorder=4, label="вне выборки > 0")
    a1.scatter(df.loc[~pos, "OOS Sharpe"], y[~pos], s=64, color=P.NEG, marker="D",
               edgecolor=P.SURFACE, linewidth=1.5, zorder=4, label="вне выборки < 0")
    a1.axvline(0, color=P.AXIS, linewidth=1)
    a1.set_yticks(y, df["ячейка"], color=P.INK_2)
    a1.legend(loc="upper left")
    _frame(a1, "Шарп в выборке и вне её", "лучшая по in-sample конфигурация каждой ячейки",
           xlabel="годовой Шарп", grid_axis="x")

    a2.barh(y, df["DSR программа"], height=0.62, color=P.S1, zorder=2)
    for i, v in enumerate(df["DSR программа"]):
        a2.text(v + 0.015, i, f"{v:.2f}", va="center", color=P.INK_2, fontsize=8.5)
    a2.axvline(0.9, color=P.INK, linewidth=1.2, linestyle=(0, (4, 3)))
    a2.text(0.915, -0.45, "порог\nшлюза 0.90", ha="left", va="bottom", color=P.INK, fontsize=8.5)
    a2.set_xlim(0, 1.12)
    _frame(a2, "Дефлированный Шарп", "поправка на все 125 испытаний программы",
           xlabel="вероятность, что истинный Шарп > 0", grid_axis="x")
    a2.tick_params(axis="y", length=0)
    plt.show()


def cost_ladder_chart(stage: str = "stage3"):
    ticks = ["0.0", "0.5", "1.0", "2.0"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), sharey=True)
    for ax, t in zip(axes, TICKERS, strict=True):
        c = res(stage, t, "extras.json")["costs"]
        x = [float(k) for k in ticks]
        for key, color, label in (("share_pos_is", P.S2, "в выборке"), ("share_pos_oos", P.S1, "вне выборки")):
            yv = [100 * c[k][key] for k in ticks]
            ax.plot(x, yv, color=color, marker="o", markersize=7, markeredgecolor=P.SURFACE,
                    markeredgewidth=1.5, label=label, zorder=3)
            ax.annotate(f"{yv[-1]:.0f}%", (x[-1], yv[-1]), xytext=(8, 0), textcoords="offset points",
                        va="center", color=P.INK_2, fontsize=9, fontweight="bold")
        null = 100 * top("signflip.json")[t]["null_share_profitable_mean"]
        ax.axhline(null, color=P.MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(-0.05, null - 2, f"рынок без\nзакономерности: ~{null:.0f}%", va="top",
                color=P.MUTED, fontsize=8.5)
        ax.set_xticks(x, ["0", "0.5", "1", "2"])
        ax.set_xlim(-0.1, 2.35)
        ax.set_ylim(0, 100)
        _frame(ax, f"{t} · {NAMES[t]}", xlabel="издержки, тиков на полную сделку",
               ylabel="прибыльных конфигураций, %" if t == "NG" else "")
    axes[0].legend(loc="lower left")  # подписи нуля — у левого края, легенда ниже них
    fig.suptitle("Доля прибыльных конфигураций из 3.47 млн при росте издержек", x=0.01, ha="left",
                 color=P.INK, fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    plt.show()


def signflip_chart():
    sf = top("signflip.json")
    _, axes = plt.subplots(1, 2, figsize=(12.5, 3.9))
    for ax, t in zip(axes, TICKERS, strict=True):
        s = sf[t]
        nulls = np.asarray(s["null_means"])
        ax.hist(nulls, bins=30, color=P.AXIS, edgecolor=P.SURFACE, linewidth=1, zorder=2)
        ax.axvline(s["real_mean_trade_bp"], color=P.S1, linewidth=2.2, zorder=3)
        ax.text(s["real_mean_trade_bp"], ax.get_ylim()[1] * 0.92,
                f"реальный ряд  {s['real_mean_trade_bp']:+.1f} б.п.\np = {s['p_value']:.3f}",
                ha="right", va="top", color=P.INK, fontsize=9, fontweight="bold",
                bbox={"facecolor": P.SURFACE, "edgecolor": "none", "pad": 2})
        _frame(ax, f"{t} · {NAMES[t]}", f"серое — {s['n_paths']} траекторий со случайно сменённым "
               "знаком каждой свечи", xlabel="средняя сделка, б.п.",
               ylabel="траекторий" if t == "NG" else "")
    plt.tight_layout()
    plt.show()


def years_chart(stage: str = "stage3"):
    _, ax = plt.subplots(figsize=(12.5, 4.4))
    for t, color in (("NG", P.S1), ("BR", P.S2)):
        y = res(stage, t, "years_grid.csv")
        y = y[y["месяцев"] >= 6]
        share = 100 * y["доля прибыльных конфигураций"]
        ax.plot(y["год"], share, color=color, marker="o", markersize=6, markeredgecolor=P.SURFACE,
                markeredgewidth=1.5, label=f"{t} · {NAMES[t]}", zorder=3)
        _end_label(ax, y["год"].iloc[-1], share.iloc[-1], t, color)
    ax.axvspan(2023.5, 2026.5, color=P.NEUTRAL, zorder=0)
    ax.text(2023.6, 96, "вне выборки", color=P.INK_2, fontsize=9, va="top")
    ax.axhline(50, color=P.MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_ylim(0, 100)
    ax.set_xlim(2011.6, 2026.9)
    ax.set_xticks(range(2012, 2027, 2))
    ax.legend(loc="lower left")
    _frame(ax, "Доля прибыльных конфигураций по календарным годам",
           "без издержек; 2026 — январь–октябрь", ylabel="%")
    plt.show()


# --------------------------------------------------------------- таблицы
def stage_table() -> pd.DataFrame:
    rows = []
    for st, name in (("stage1", "1: исходный наклон"), ("stage3", "3: наклон оптимизирован")):
        for t in TICKERS:
            r, x, o = res(st, t, "results.json"), res(st, t, "extras.json"), res(st, t, "oos_split.json")
            rows.append({
                "этап": name, "рынок": t,
                "прибыльны IS": f"{r['share_pos_is']:.0%}",
                "прибыльны OOS": f"{r['share_pos_oos']:.0%}",
                "OOS без 2026": f"{o['share_pos_early']:.0%}",
                "OOS при 1 тике": f"{x['costs']['1.0']['share_pos_oos']:.0%}",
                "PBO": round(r["pbo_full"]["pbo"], 2),
                "SPA p": round(r["rc_spa_full"]["spa_p"], 3),
            })
    return pd.DataFrame(rows).set_index(["этап", "рынок"])


def nulls_table() -> pd.DataFrame:
    df = pd.DataFrame(top("nulls.json")["rows"])
    df = df[["инструмент", "ряд", "доля прибыльных", "медиана средней сделки, тиков"]]
    df["доля прибыльных"] = (100 * df["доля прибыльных"]).round(0).astype(int).astype(str) + "%"
    df["медиана средней сделки, тиков"] = df["медиана средней сделки, тиков"].round(2)
    return df.set_index(["инструмент", "ряд"])


def topcheck_table() -> pd.DataFrame:
    rows = []
    for st in ("stage1", "stage3"):
        for t in TICKERS:
            c = res(st, t, "topcheck.json")
            for k in ("1.0", "2.0"):
                rows.append({"этап": st[-1], "рынок": t, "издержки, тиков": k[0],
                             "прибыльны в 2024–2025": f"{c[f'share_pos_early_{k}']:.0%}",
                             "USD в месяц на контракт": round(c[f"early_monthly_mean_{k}"], 1),
                             "t-статистика": round(c[f"early_monthly_t_{k}"], 2)})
    return pd.DataFrame(rows).set_index(["этап", "рынок", "издержки, тиков"])


def wf_table() -> pd.DataFrame:
    rows = []
    for st in ("stage1", "stage3"):
        for t in TICKERS:
            w = res(st, t, "extras.json")["walk_forward_vs_average_config"]
            rows.append({"этап": st[-1], "рынок": t,
                         **{f"{k}: t превышения": round(v["excess_t"], 2) for k, v in w.items()}})
    return pd.DataFrame(rows).set_index(["этап", "рынок"]).rename(columns={
        "pnl: t превышения": "отбор по PnL", "sharpe: t превышения": "по Шарпу",
        "plateau: t превышения": "по плато"})


def data_table() -> pd.DataFrame:
    text = (DOCS / "RESULTS_MA_CROSSOVER_5M.md").read_text()
    df = md_table(text, "баров 5m", first="инструмент")
    df = df[["инструмент", "контракт", "баров 5m", "первый бар", "последний бар", "роллов", "мес. IS", "мес. OOS"]]
    return df.set_index("инструмент")
