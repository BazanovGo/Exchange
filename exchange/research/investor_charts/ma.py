"""Дашборд перебора сетки параметров стратегии на пересечении скользящих средних (один рынок)."""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import style as S

ROOT = Path(__file__).resolve().parents[2]
MA = ROOT / "research" / "ma_crossover_5m"
RES = MA / "results"
STAGE = "stage3"
OOS_START = pd.Timestamp("2024-01-01")
TICK_USD = 0.1
MARKET = {"NG": "природный газ", "BR": "нефть Brent"}
RULES = {"R1_max_pnl": "R1 · макс. прибыль", "R2_max_sharpe": "R2 · макс. Шарп", "R3_plateau": "R3 · плато"}
GROUPS = {"NG": [(2020, 2021), (2022, 2023), (2024, 2025), (2026, 2026)],
          "BR": [(2011, 2014), (2015, 2017), (2018, 2020), (2021, 2023), (2024, 2026)]}


def load(t: str):
    sys.path.insert(0, str(MA))
    from export import load_full_grid
    grid = load_full_grid(STAGE, t)
    fin = [x for x in json.loads((RES / STAGE / t / "finalists.json").read_text()) if x["scope"] == "global"]
    fin.sort(key=lambda x: x["rule"])
    j = lambda name: json.loads((RES / STAGE / t / name).read_text())  # noqa: E731
    return {"grid": grid, "finalists": fin, "results": j("results.json"), "extras": j("extras.json"),
            "split": j("oos_split.json"), "slice": j("slice.json"),
            "slice_yearly": pd.read_parquet(RES / STAGE / t / "slice_yearly.parquet"),
            "signflip": json.loads((RES / "signflip.json").read_text())[t],
            "slope": json.loads((RES / "slope_params.json").read_text())[t]}


def wrap(text, width):
    return "\n".join(textwrap.wrap(text, width))


def cloud(b, rect, grid, n=60_000, seed=7):
    rng = np.random.default_rng(seed)
    sample = grid.iloc[rng.choice(len(grid), n, replace=False)]
    sample = sample.iloc[np.argsort(np.abs(sample["sharpe_m_oos"].to_numpy()))]
    norm = S.TwoSlopeNorm(vmin=-2.0, vcenter=0.0, vmax=2.0)
    ax = b.fig.add_axes(rect, projection="3d", computed_zorder=False)
    ax.set_facecolor(S.CARD)
    z = np.log10(sample["trades_is"] + sample["trades_oos"])
    ax.scatter(sample["fast_period"], sample["slow_period"], z, c=sample["sharpe_m_oos"], cmap=S.DIVERGING,
               norm=norm, s=3.2, alpha=0.75, linewidths=0, depthshade=False)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0.06, 0.10, 0.17, 1.0))
        pane._axinfo["grid"]["color"] = S.GRID
        pane.label.set_color(S.INK2)
    ax.set_xlabel("Быстрая средняя, баров", fontsize=10.5, labelpad=6)
    ax.set_ylabel("Медленная средняя, баров", fontsize=10.5, labelpad=8)
    ax.set_zlabel("Сделок за историю", fontsize=10.5, labelpad=6)
    marks = {1000: "1 тыс", 3000: "3 тыс", 10000: "10 тыс", 30000: "30 тыс", 100000: "100 тыс"}
    zt = [m for m in marks if z.min() - 0.05 <= np.log10(m) <= z.max() + 0.05]
    ax.set_zticks(np.log10(zt), [marks[m] for m in zt])
    ax.tick_params(colors=S.INK2, labelsize=9)
    ax.view_init(elev=24, azim=-58)
    ax.set_box_aspect((1.0, 1.25, 0.75), zoom=1.08)
    return norm


def slice_matrix(grid, fam, value):
    sub = grid[(grid["method"] == fam["method"]) & (grid["slow_type"] == fam["slow_type"]) &
               (grid["exit"] == fam["exit"]) & (grid["fast_type"] == fam["fast_type"])]
    mat = sub.pivot(index="slow_period", columns="fast_period", values=value)
    return mat


def heat(b, rect, mat, norm, title, pick, cmap=S.DIVERGING, label_pick=True):
    ax = b.axes(rect)
    xs, ys = mat.columns.to_numpy(), mat.index.to_numpy()
    ax.imshow(mat.to_numpy(), cmap=cmap, norm=norm, aspect="auto", origin="lower", interpolation="nearest",
              extent=(xs[0] - 0.5, xs[-1] + 0.5, ys[0] - 0.5, ys[-1] + 0.5))
    lim = min(xs[-1], ys[-1])
    ax.plot([max(xs[0], ys[0]), lim], [max(xs[0], ys[0]), lim], color=S.INK2, linewidth=0.8,
            linestyle=(0, (3, 3)), alpha=0.7)
    ax.scatter([pick[0]], [pick[1]], s=150, facecolors="none", edgecolors="#ffffff", linewidths=2.2, zorder=5)
    if label_pick:
        ax.annotate("R2", pick, xytext=(9, 6), textcoords="offset points", fontsize=10.5, fontweight="bold",
                    color="#ffffff")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0, labelsize=9.5)
    ax.set_title(title, color=S.INK, fontsize=12, fontweight="bold", loc="left", pad=7)
    return ax


def build(t: str, out: Path):
    d = load(t)
    grid, fin, res, ext, split = d["grid"], d["finalists"], d["results"], d["extras"], d["split"]
    fam = d["slice"]["family"]
    r2 = next(x for x in fin if x["rule"] == "R2_max_sharpe")
    n = len(grid)

    b = S.Board(24, 18)
    b.header(f"Пересечение скользящих средних · {t} ({MARKET[t]}) · перебор {n:,} комбинаций".replace(",", " "),
             "Фьючерс MOEX, 5-минутные свечи · 6 параметров сетки + 3 параметра наклона · отбор только в выборке "
             "(до 2024 г.), проверка вне выборки (2024 — окт. 2026) · 1 контракт")
    hs = 15 / 18

    # --- ряд 1 -----------------------------------------------------------
    y1, h1 = 0.715, 0.195
    inner = b.card(0.012, y1, 0.30, h1, "Как работает система")
    ax = b.text_axes(inner)
    facts = [("Вход", "цена закрытия пересекает быструю среднюю"),
             ("Фильтр", "long — медленная растёт или нейтральна, short — падает или нейтральна"),
             ("Выход A / B", "обратное пересечение / разворот медленной средней"),
             ("Исполнение", "по открытию следующей свечи, 1 контракт, без стопов"),
             ("Издержки", "в сетке нулевые (условие задачи); ниже — пересчёт при 1 тике")]
    for k, (head, body) in enumerate(facts):
        yy = 0.93 - k * 0.2
        ax.plot([0.012], [yy - 0.03], "o", markersize=7, color=S.S1)
        ax.text(0.04, yy, head, va="center", fontsize=11, color=S.INK, fontweight="bold")
        ax.text(0.04, yy - 0.085, body, va="center", fontsize=10.5, color=S.INK2)

    inner = b.card(0.324, y1, 0.352, h1, "Оптимизируемые параметры (9 измерений)")
    ax = b.text_axes(inner)
    sl = d["slope"]
    rows = [["1", "Быстрая средняя, баров", "2 — 99", "98"],
            ["2", "Медленная средняя, баров", "5 — 250", "246"],
            ["3–4", "Тип быстрой / медленной", "SMA EMA WMA SSMA VWMA TMA", "6 × 6"],
            ["5", "Метод наклона медленной", "A: % за k баров · B: в ATR", "2"],
            ["6", "Модель выхода", "A: пересечение · B: разворот", "2"],
            ["7–9", "Наклон: k, нейтраль, ATR", f"k {sl['A_pct']['k']} · нейтраль {sl['A_pct']['coverage']:.0%}/"
             f"{sl['B_atr']['coverage']:.0%} · ATR {sl['B_atr']['atr_period']}", "160"],
            ["", "Комбинаций полной сетки", "на инструмент, два прогона", f"{n:,}".replace(",", " ")]]
    S.table(ax, ["№", "Параметр", "Значения", "Вариантов"], rows, [0.0, 0.07, 0.40, 0.84],
            align=["left", "left", "left", "right"], size=10.5, head_size=10, row_h=0.118, bold_cols=(3,))

    inner = b.card(0.688, y1, 0.30, h1, "Методика")
    S.steps(b.text_axes(inner), [
        "Полная сетка: 3 471 552 комбинации\nна инструмент, ≈174 млрд сделок",
        "Отбор только по выборке (до 2024 г.):\nправила R1, R2, R3",
        "Одна проверка вне выборки (2024–2026)",
        "Издержки, walk-forward, PBO, SPA,\nтест против случайных траекторий",
        "Вердикт по всем проверкам сразу",
    ], gap=0.18)

    # --- ряд 2: облако и срезы ----------------------------------------
    y2, h2 = 0.405, 0.295
    inner = b.card(0.012, y2, 0.40, h2, "5D-пространство перебора",
                   "каждая точка — комбинация (60 000 из 3,47 млн); цвет — Шарп вне выборки, без издержек")
    norm3 = cloud(b, (inner[0] - 0.01, inner[1] - 0.01, inner[2] + 0.0, inner[3] + 0.02), grid)
    S.colorbar(b.fig, (0.392, y2 + 0.03, 0.008, h2 - 0.11), S.DIVERGING, norm3, "Шарп",
               ticks=[-2, -1, 0, 1, 2], fmt="{:+.0f}")

    mat_is = slice_matrix(grid, fam, "sharpe_m_is")
    mat_oos = slice_matrix(grid, fam, "sharpe_m_oos")
    norm = S.sym_norm(np.r_[mat_is.to_numpy().ravel(), mat_oos.to_numpy().ravel()], q=0.99, floor=0.5)
    c = r2["config"]
    pick = (c["fast_period"], c["slow_period"])
    fam_name = f"{fam['fast_type']} × {fam['slow_type']}, наклон {fam['method'][0]}, выход {fam['exit'][0]}"
    for k, (mat, label) in enumerate(((mat_is, "в выборке"), (mat_oos, "вне выборки"))):
        x0 = 0.424 + k * 0.277
        inner = b.card(x0, y2, 0.265, h2, f"2D-срез: быстрая × медленная · {label}",
                       f"{fam_name} — семейство отбора R2; цвет — Шарп")
        ax = heat(b, (inner[0] + 0.03, inner[1] + 0.035, inner[2] - 0.04, inner[3] - 0.045), mat, norm,
                  "", pick)
        ax.set_xlabel("Быстрая средняя, баров", fontsize=10, color=S.INK2)
        ax.set_ylabel("Медленная средняя, баров", fontsize=10, color=S.INK2)
    S.colorbar(b.fig, (0.971, y2 + 0.03, 0.008, h2 - 0.11), S.DIVERGING, norm, "Шарп",
               ticks=np.round(np.linspace(norm.vmin, norm.vmax, 5), 1), fmt="{:+.1f}")
    b.fig.text(0.424, y2 - 0.006, "○ R2 — комбинация, выбранная по выборке; пунктир — быстрая = медленная.",
               fontsize=9.5, color=S.MUTED, va="top")

    # --- ряд 3: капитал, таблица, итог --------------------------------
    y3, h3 = 0.215, 0.17
    inner = b.card(0.012, y3, 0.34, h3, "Накопленный результат отобранных",
                   "USD на 1 контракт, без издержек; справа от пунктира — вне выборки")
    ax = b.axes((inner[0] + 0.045, inner[1] + 0.02, inner[2] - 0.12, inner[3] - 0.015))
    curves = []
    for x, color in zip(fin, (S.S1, S.S2, S.S3), strict=True):
        e = pd.read_parquet(RES / STAGE / t / "finalists" / f"cfg{x['grid_config']}_equity.parquet")["cum_pnl_usd"]
        ax.plot(e.index, e.to_numpy(), color=color, linewidth=2.0)
        curves.append((float(e.iloc[-1]), RULES[x["rule"]].split(" ·")[0], color, e.index[-1]))
    lo, hi = ax.get_ylim()
    hi = hi + 0.22 * (hi - lo)
    ax.set_ylim(lo, hi)
    first = min(pd.read_parquet(RES / STAGE / t / "finalists" / f"cfg{x['grid_config']}_equity.parquet").index[0]
                for x in fin)
    last = max(c[3] for c in curves)
    gap, placed = 0.1 * (hi - lo), []
    for end, name, color, when in sorted(curves):
        y = max(end, placed[-1] + gap) if placed else end
        placed.append(y)
        ax.annotate(f"{name} {end:+,.0f}".replace(",", " "), (when, end),
                    xytext=(when + pd.Timedelta(days=60), y), textcoords="data", va="center", fontsize=10,
                    color=color, fontweight="bold")
    ax.set_xlim(first, last + pd.Timedelta(days=int((last - first).days * 0.24)))
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(first.year + 1, last.year + 1, 2)])
    ax.xaxis.set_major_formatter(S.mpl.dates.DateFormatter("%Y"))
    ax.axvline(OOS_START, color=S.INK2, linestyle=(0, (3, 3)), linewidth=1)
    ax.axvspan(OOS_START, last, color="#ffffff", alpha=0.04)
    ax.text(OOS_START, lo, " вне выборки", va="bottom", fontsize=9.5, color=S.INK2)
    S.frame(ax, ylabel="USD")

    inner = b.card(0.364, y3, 0.36, h3, "Лучшие комбинации по выборке и что было дальше",
                   "вне выборки — без издержек и при 1 тике (0.1 USD) на сделку")
    ax = b.text_axes(inner)
    rows, fills = [], []
    pnl1 = [None] * len(fin)
    for i, x in enumerate(fin):
        g = grid.loc[grid["config"] == x["grid_config"]].iloc[0]
        c = x["config"]
        pnl1[i] = g["pnl_oos"] - TICK_USD * g["trades_oos"]
        rows.append([RULES[x["rule"]], f"{c['fast_type']}{c['fast_period']} × {c['slow_type']}{c['slow_period']} "
                     f"{c['method'][0]}/{c['exit'][0]}", f"{g['sharpe_m_is']:+.2f}", f"{g['sharpe_m_oos']:+.2f}",
                     f"{g['pnl_oos']:+,.0f}".replace(",", " "), f"{pnl1[i]:+,.0f}".replace(",", " "),
                     f"{int(g['trades_oos'])}"])
    med = grid[["sharpe_m_is", "sharpe_m_oos", "pnl_oos", "trades_oos"]].median()
    med1 = float((grid["pnl_oos"] - TICK_USD * grid["trades_oos"]).median())
    rows.append(["Медиана сетки", "все 3,47 млн", f"{med['sharpe_m_is']:+.2f}", f"{med['sharpe_m_oos']:+.2f}",
                 f"{med['pnl_oos']:+,.0f}".replace(",", " "), f"{med1:+,.0f}".replace(",", " "),
                 f"{int(med['trades_oos'])}"])
    pn = S.sym_norm([*pnl1, med1], q=1.0, floor=100)
    fills = [[None] * 5 + [S.DIVERGING(pn(v))] + [None] for v in [*pnl1, med1]]
    S.table(ax, ["Правило", "Комбинация", "Шарп\nв выборке", "Шарп\nвне выб.", "USD вне\nвыборки",
                 "USD вне,\n1 тик", "Сделок\nвне выб."], rows, [0.0, 0.19, 0.48, 0.585, 0.69, 0.80, 0.905],
            colors=fills, size=10.5, head_size=9.5, row_h=0.16,
            align=["left", "left", "right", "right", "right", "right", "right"])

    inner = b.card(0.736, y3, 0.252, h3, "Итог проверки")
    ax = b.text_axes(inner)
    S.badge(ax, 0.0, 0.88, "⚠ Преимущество не доказано", S.WARN, size=13)
    sf = d["signflip"]
    lines = [
        ("Прибыльны без издержек", f"{res['share_pos_is']:.0%} в выборке → {res['share_pos_oos']:.0%} вне"),
        ("При 1 тике на сделку", f"{ext['costs']['1.0']['share_pos_oos']:.0%} прибыльны вне выборки"),
        ("Вне выборки без 2026 г.", f"{split['share_pos_early']:.0%} прибыльны"),
        ("Тест случайных траекторий", f"p = {sf['p_value']:.3f}: закономерность есть"),
        ("Переобучение отбора (PBO)", f"{res['pbo_full']['pbo']:.2f}"),
    ]
    for k, (head, body) in enumerate(lines):
        yy = 0.69 - k * 0.145
        ax.text(0.0, yy, head, va="center", fontsize=10.5, color=S.INK2)
        ax.text(1.0, yy, body, va="center", ha="right", fontsize=10.5, color=S.INK, fontweight="bold")

    # --- ряд 4: устойчивость по времени ---------------------------------
    y4, h4 = 0.025, 0.175
    sy = d["slice_yearly"]
    groups = GROUPS[t]
    inner = b.card(0.012, y4, 0.976, h4, "Устойчивость по времени: тот же 2D-срез, результат по периодам",
                   "USD на 1 контракт без издержек; красный — убыток, синий — прибыль; ○ — комбинация R2")
    mats = []
    for a, z in groups:
        cols = [str(y) for y in range(a, z + 1) if str(y) in sy.columns]
        v = sy[cols].sum(axis=1)
        mats.append(pd.DataFrame({"fast": sy["fast_period"], "slow": sy["slow_period"], "v": v})
                    .pivot(index="slow", columns="fast", values="v"))
    yn = S.sym_norm(np.concatenate([m.to_numpy().ravel() for m in mats]), q=0.99, floor=50)
    w = (inner[2] - 0.05 - 0.012 * (len(groups) - 1)) / len(groups)
    for k, ((a, z), m) in enumerate(zip(groups, mats, strict=True)):
        x0 = inner[0] + 0.02 + k * (w + 0.012)
        period = f"{a}" if a == z else f"{a}–{z}"
        where = "вне выборки" if a >= 2024 else "в выборке"
        ax = heat(b, (x0, inner[1] + 0.02, w, inner[3] - 0.03), m, yn, f"{period} · {where}", pick,
                  label_pick=False)
        ax.tick_params(labelsize=8.5)
        if k:
            ax.set_yticklabels([])
    S.colorbar(b.fig, (0.971, y4 + 0.02, 0.008, h4 - 0.075), S.DIVERGING, yn, "USD",
               ticks=np.round(np.linspace(yn.vmin, yn.vmax, 5), -1), fmt="{:+.0f}")
    b.footer("Источник: research/ma_crossover_5m (сетка, финалисты, срезы) и docs/RESULTS_MA_CROSSOVER_5M.md · "
             "Шарп — годовой по месячному PnL. Прошлые результаты не гарантируют будущих.")
    b.save(out)
    return hs
