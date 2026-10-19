"""Дашборд перебора сетки параметров для одного из семи классических семейств."""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import style as S

ROOT = Path(__file__).resolve().parents[2]
SWEEPS = ROOT / "research" / "seven_families" / "results"
REPORT = ROOT / "docs" / "RESULTS_SEVEN_FAMILIES.md"
MARKET = {"NG": "природный газ", "BR": "нефть Brent", "NG+BR": "портфель NG + BR"}
PARAM = {"lookback": "Окно моментума, дней", "fast": "Быстрая средняя, дней", "slow": "Медленная средняя, дней",
         "entry": "Канал входа, дней", "exit_": "Канал выхода, дней", "smooth": "Сглаживание carry, дней",
         "threshold": "Порог годового roll yield", "momentum": "Горизонт моментума, дней",
         "reversal": "Горизонт разворота, дней"}
SHORT = {"lookback": "окно", "fast": "быстрая", "slow": "медленная", "entry": "вход", "exit_": "выход",
         "smooth": "сглаживание", "threshold": "порог", "momentum": "моментум", "reversal": "разворот"}
GRID_KEYS = {"1_tsmom": ["lookback"], "2_dual_ma": ["fast", "slow"], "3_breakout": ["entry", "exit_"],
             "4_cross_sectional": ["lookback"], "5_carry": ["smooth", "threshold"],
             "6_portfolio_ma": ["fast", "slow"], "7_mom_reversal": ["momentum", "reversal"]}
CRITERIA = {"deflated_sharpe_ratio": "дефлированный Шарп", "oos_sharpe": "Шарп вне выборки", "pbo": "PBO",
            "stressed_sharpe": "Шарп при тройных издержках", "n_oos_trades": "сделок вне выборки"}


def md_table(text: str, first: str, column: str) -> pd.DataFrame:
    lines = text.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith(f"| {first} |") and f"| {column} |" in ln)
    rows = []
    for ln in lines[i:]:
        if not ln.startswith("|"):
            break
        rows.append([c.strip() for c in ln.strip("|").split("|")])
    return pd.DataFrame(rows[2:], columns=rows[0])


def report_facts():
    text = REPORT.read_text()
    rules = md_table(text, "семейство", "правило")
    stats = md_table(text, "ячейка", "DSR программа").merge(
        md_table(text, "ячейка", "IS Sharpe")[["ячейка", "IS Sharpe", "OOS Sharpe"]], on="ячейка")
    verdicts, triggered = {}, {}
    for block in re.split(r"\n(?=\*\*\d )", text.split("### Что именно сработало")[1].split("## 8.")[0]):
        m = re.match(r"\*\*(\d [^*]+?)\*\* — (\w+)", block.strip())
        if m:
            verdicts[m.group(1)] = m.group(2)
            triggered[m.group(1)] = re.findall(r"`(\w+) ([<>]) ([\d.\-]+)`.*?наблюдено \w+ = ([\-\d.e]+)", block)
    return rules, stats, verdicts, triggered


def fmt_value(key, v):
    if key == "threshold":
        return f"{v:.2f}"
    return f"{int(v)}"


def heatmap(board, rect, sub, keys, value, norm, title, winner):
    p1, p2 = keys
    xs, ys = sorted(sub[p1].unique()), sorted(sub[p2].unique())
    mat = np.full((len(ys), len(xs)), np.nan)
    for r in sub.itertuples():
        mat[ys.index(getattr(r, p2)), xs.index(getattr(r, p1))] = getattr(r, value)
    ax = board.axes(rect)
    ax.imshow(np.ma.masked_invalid(mat), cmap=S.DIVERGING, norm=norm, aspect="auto", origin="lower")
    for i in range(len(ys)):
        for j in range(len(xs)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=11.5, fontweight="bold",
                        color=S.ink_on(S.DIVERGING(norm(v))))
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=11, color=S.MUTED)
    wi, wj = ys.index(winner[p2]), xs.index(winner[p1])
    ax.add_patch(S.plt.Rectangle((wj - 0.5, wi - 0.5), 1, 1, fill=False, edgecolor="#ffffff", linewidth=2.6))
    ax.set_xticks(range(len(xs)), [fmt_value(p1, x) for x in xs])
    ax.set_yticks(range(len(ys)), [fmt_value(p2, y) for y in ys])
    ax.set_xlabel(PARAM[p1], color=S.INK2, fontsize=10.5)
    ax.set_ylabel(PARAM[p2], color=S.INK2, fontsize=10.5)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title(title, color=S.INK, fontsize=12.5, fontweight="bold", loc="left", pad=8)


def bars(board, rect, sub, key, title, winner):
    sub = sub.sort_values(key)
    ax = board.axes(rect)
    x = np.arange(len(sub))
    ax.bar(x - 0.2, sub["is_sharpe"], 0.38, color=S.S2, label="в выборке (по ней выбирали)", zorder=2)
    ax.bar(x + 0.2, sub["oos_sharpe"], 0.38, color=S.S1, label="вне выборки", zorder=2)
    for xi, (a, b) in enumerate(zip(sub["is_sharpe"], sub["oos_sharpe"], strict=True)):
        ax.text(xi - 0.2, a + (0.03 if a >= 0 else -0.03), f"{a:+.2f}", ha="center",
                va="bottom" if a >= 0 else "top", fontsize=9.5, color=S.INK2)
        ax.text(xi + 0.2, b + (0.03 if b >= 0 else -0.03), f"{b:+.2f}", ha="center",
                va="bottom" if b >= 0 else "top", fontsize=9.5, color=S.INK, fontweight="bold")
    lo = min(0.0, float(sub[["is_sharpe", "oos_sharpe"]].min().min()))
    hi = max(0.0, float(sub[["is_sharpe", "oos_sharpe"]].max().max()))
    ax.set_ylim(lo - 0.12 * (hi - lo) - 0.05, hi + 0.28 * (hi - lo) + 0.05)
    wx = list(sub[key]).index(winner[key])
    ax.axvspan(wx - 0.5, wx + 0.5, color="#ffffff", alpha=0.07, zorder=0)
    ax.text(wx, ax.get_ylim()[1] - 0.02 * (hi - lo), "★ выбрана", ha="center", va="top", fontsize=10.5,
            color=S.INK, fontweight="bold")
    ax.axhline(0, color=S.MUTED, linewidth=1)
    ax.set_xticks(x, [fmt_value(key, v) for v in sub[key]])
    S.frame(ax, xlabel=PARAM[key], ylabel="годовой Шарп")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=10, borderaxespad=0.2)
    ax.set_title(title, color=S.INK, fontsize=12.5, fontweight="bold", loc="left", pad=8)


def wrap(text, width):
    return "\n".join(textwrap.wrap(text, width))


def build(family: str, out: Path):
    sweeps = pd.read_csv(SWEEPS / "sweeps.csv")
    equity = pd.read_csv(SWEEPS / "winner_equity.csv", parse_dates=["date"])
    rules, stats, verdicts, triggered = report_facts()
    num = family.split("_")[0]
    rule = rules[rules["семейство"].str.startswith(num + " ")].iloc[0]
    name = rule["семейство"].split(" ", 1)[1]
    fam = sweeps[sweeps["family"] == family].copy()
    keys = GRID_KEYS[family]
    markets = [m for m in ("NG", "BR", "NG+BR") if m in set(fam["market"])]
    n_cfg = int(rule["конфигураций"])

    b = S.Board(24, 15)
    b.header(f"{name} · перебор сетки параметров",
             f"Фьючерсы MOEX: {', '.join(f'{m} ({MARKET[m]})' for m in markets)} · дневные бары · "
             f"{n_cfg} комбинаций на рынок · отбор только в выборке, одна проверка вне её · издержки учтены")

    # --- ряд 1: как работает, параметры, методика -----------------------
    r1y, r1h = 0.655, 0.245
    inner = b.card(0.012, r1y, 0.30, r1h, "Как работает система")
    ax = b.text_axes(inner)
    text = re.sub(r"`", "", rule["правило"])
    ax.text(0, 0.98, wrap(text[0].upper() + text[1:] + ".", 50), va="top", fontsize=13, color=S.INK,
            linespacing=1.45, fontweight="bold")
    facts = [("Размер позиции", "целевая волатильность 15% годовых, полоса ребалансировки 20%"),
             ("Исполнение", "по цене открытия следующего дня — без взгляда в будущее"),
             ("Издержки", "полуспред, проскальзывание и комиссия: ~7 б.п. за оборот вычтены"),
             ("Данные", "непрерывный фьючерс с учётом роллов, вся история MOEX")]
    for k, (head, body) in enumerate(facts):
        yy = 0.56 - k * 0.165
        ax.plot([0.012], [yy - 0.01], "o", markersize=7, color=S.S1)
        ax.text(0.04, yy, head, va="center", fontsize=11, color=S.INK, fontweight="bold")
        ax.text(0.04, yy - 0.075, body, va="center", fontsize=10.5, color=S.INK2)

    inner = b.card(0.324, r1y, 0.352, r1h, f"Оптимизируемые параметры ({len(keys)} "
                   f"{'измерение' if len(keys) == 1 else 'измерения'})")
    ax = b.text_axes(inner)
    rows = [[str(k + 1), PARAM[p], ", ".join(fmt_value(p, v) for v in sorted(fam[p].unique()))]
            for k, p in enumerate(keys)]
    rows.append([str(len(keys) + 1), "Результат", "годовой Шарп в выборке и вне её"])
    rows.append(["", "Комбинаций", f"{n_cfg} на рынок · {n_cfg * len(markets)} всего"])
    S.table(ax, ["№", "Параметр", "Значения"], rows, [0.0, 0.07, 0.43], align=["left", "left", "left"],
            size=11.5, row_h=0.15)

    inner = b.card(0.688, r1y, 0.30, r1h, "Методика")
    S.steps(b.text_axes(inner), [
        "Задаём сетку значений каждого параметра",
        "Бэктест каждой комбинации на первых\n60% истории (в выборке)",
        "Выбираем лучшую по Шарпу в выборке —\nбез взгляда на будущее",
        "Один раз проверяем на оставшихся ~40%\n(вне выборки)",
        "Поправка на число испытаний (DSR), PBO\nи вердикт фреймворка",
    ])

    # --- ряд 2: карта перебора ------------------------------------------
    r2y, r2h = 0.345, 0.295
    vals = np.r_[fam["is_sharpe"].to_numpy(), fam["oos_sharpe"].to_numpy()]
    norm = S.sym_norm(vals, q=1.0, floor=0.6)
    if len(keys) == 2:
        panels = [(m, w) for m in markets for w in ("is", "oos")]
        width = (0.976 - 0.012 * (len(panels) - 1) - 0.05) / len(panels)
        for k, (m, which) in enumerate(panels):
            x0 = 0.012 + k * (width + 0.012)
            inner = b.card(x0, r2y, width, r2h, title_h=0.012)
            sub = fam[fam["market"] == m]
            win = sub[sub["is_winner"]].iloc[0]
            label = "в выборке" if which == "is" else "вне выборки"
            heatmap(b, (inner[0] + 0.045, inner[1] + 0.045, inner[2] - 0.055, inner[3] - 0.085), sub, keys,
                    f"{which}_sharpe", norm, f"{m} · Шарп {label}", win)
        S.colorbar(b.fig, (0.958, r2y + 0.03, 0.01, r2h - 0.07), S.DIVERGING, norm, "Шарп",
                   ticks=np.linspace(norm.vmin, norm.vmax, 5), fmt="{:+.1f}")
        b.fig.text(0.012, r2y - 0.012, "□ белая рамка — комбинация, выбранная по выборке. Красный — убыток, "
                   "синий — прибыль; «—» — сочетание вне сетки.", fontsize=10, color=S.MUTED, va="top")
    else:
        width = (0.976 - 0.012 * (len(markets) - 1)) / len(markets)
        for k, m in enumerate(markets):
            x0 = 0.012 + k * (width + 0.012)
            inner = b.card(x0, r2y, width, r2h, title_h=0.012)
            sub = fam[fam["market"] == m]
            win = sub[sub["is_winner"]].iloc[0]
            bars(b, (inner[0] + 0.04, inner[1] + 0.05, inner[2] - 0.05, inner[3] - 0.09), sub, keys[0],
                 f"{m} · Шарп каждой комбинации в выборке и вне её", win)

    # --- ряд 3: капитал, таблица, вердикт --------------------------------
    r3y, r3h = 0.03, 0.29
    inner = b.card(0.012, r3y, 0.34, r3h, "Капитал выбранной комбинации",
                   "счёт 1 млн USD · бледно — в выборке, ярко — вне выборки")
    ax = b.axes((inner[0] + 0.04, inner[1] + 0.03, inner[2] - 0.075, inner[3] - 0.03))
    colors = {"NG": S.S1, "BR": S.S2, "NG+BR": S.S3}
    for m in markets:
        e = equity[(equity["family"] == family) & (equity["market"] == m)]
        v = e["equity"] / 1e6
        is_ = e["period"] == "IS"
        ax.plot(e["date"][is_], v[is_], color=colors[m], alpha=0.4, linewidth=1.8)
        ax.plot(e["date"][~is_], v[~is_], color=colors[m], linewidth=2.6, label=m)
        start = e["date"][~is_].iloc[0]
        ax.axvline(start, color=colors[m], linestyle=(0, (3, 3)), linewidth=1, alpha=0.8)
        oos_ret = float(v[~is_].iloc[-1] / v[~is_].iloc[0] - 1)
        ax.annotate(f"{m}: {oos_ret:+.0%} вне выборки", (e["date"].iloc[-1], v.iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=10.5, color=S.INK, fontweight="bold")
    S.frame(ax, ylabel="капитал, млн USD")
    own = equity[equity["family"] == family]
    first, last = own["date"].min(), own["date"].max()
    ax.set_xlim(right=last + pd.Timedelta(days=900))
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(first.year + 1, last.year + 1, 2)])
    ax.xaxis.set_major_formatter(S.mpl.dates.DateFormatter("%Y"))

    inner = b.card(0.364, r3y, 0.36, r3h, "Лучшие комбинации по выборке",
                   "отсортированы так, как их видел оптимизатор; вне выборки — что вышло на деле")
    ax = b.text_axes(inner)
    top = fam.sort_values("is_sharpe", ascending=False).head(6)
    rows, fills = [], []
    for r in top.itertuples():
        star = "★ " if r.is_winner else ""
        params = ", ".join(f"{SHORT[k]} {fmt_value(k, getattr(r, k))}" for k in keys)
        rows.append([f"{star}{r.market}", params, f"{r.is_sharpe:+.2f}", f"{r.oos_sharpe:+.2f}",
                     f"{r.oos_total_return:+.0%}", f"{r.oos_max_dd:.0%}", f"{int(r.oos_trades)}"])
        fills.append([None, None, None, S.DIVERGING(norm(r.oos_sharpe)), None, None, None])
    S.table(ax, ["Рынок", "Параметры", "Шарп\nв выборке", "Шарп\nвне выборки", "Доход\nвне выб.",
                 "Просадка\nвне выб.", "Сделок\nвне выб."],
            rows, [0.0, 0.11, 0.42, 0.53, 0.665, 0.775, 0.89], colors=fills, size=11, head_size=9.5,
            align=["left", "left", "right", "right", "right", "right", "right"], row_h=0.122)
    ax.text(0, 0.0, "★ — комбинация, выбранная по выборке", fontsize=9.5, color=S.MUTED, va="bottom")

    inner = b.card(0.736, r3y, 0.252, r3h, "Вердикт фреймворка")
    ax = b.text_axes(inner)
    y = 0.93
    for m in markets:
        cell = next(k for k in verdicts if k.startswith(num + " ") and k.endswith("· " + m))
        st = stats[stats["ячейка"] == cell].iloc[0]
        S.badge(ax, 0.0, y - 0.04, f"✖ {m}: отклонено", S.CRIT, size=13)
        ax.text(0.62, y - 0.04, f"Шарп {float(st['IS Sharpe']):+.2f} → {float(st['OOS Sharpe']):+.2f}",
                va="center", ha="left", fontsize=12.5, fontweight="bold", color=S.INK)
        crits = ", ".join(f"{CRITERIA.get(c, c)} {float(obs):.2f} {op} {thr}" for c, op, thr, obs in triggered[cell][:2])
        ax.text(0.0, y - 0.13, wrap(f"DSR {float(st['DSR программа']):.2f} (порог 0.90) · PBO {float(st['PBO']):.2f}", 44),
                va="top", fontsize=11, color=S.INK)
        ax.text(0.0, y - 0.21, wrap("Сработало: " + crits, 46), va="top", fontsize=10, color=S.INK2,
                linespacing=1.3)
        y -= 0.40 if len(markets) > 1 else 0.0
    ax.text(0.0, 0.02, wrap("Отклонение — завершённый эксперимент: кандидат убран из поиска, а число "
                            "испытаний учтено в проверке следующих гипотез.", 50),
            va="bottom", fontsize=9.5, color=S.MUTED, linespacing=1.3)
    b.footer("Источник: research/seven_families/sweep.py и docs/RESULTS_SEVEN_FAMILIES.md · Шарп годовой, по "
             "дневным доходностям. Прошлые результаты не гарантируют будущих.")
    b.save(out)
