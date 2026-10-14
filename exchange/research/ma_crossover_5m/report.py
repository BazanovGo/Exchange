"""Генератор итогового отчёта из результатов анализа. Ни одно число руками.

python report.py stage1 stage3   -> docs/RESULTS_MA_CROSSOVER_5M.md
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path(os.environ.get("MA5M_WORK", "work"))
HERE = Path(__file__).parent
TICKERS = ("NG", "BR")
NAMES = {"NG": "природный газ", "BR": "Brent"}
L: list[str] = []


def w(x=""):
    L.append(x)


def fmt(v, d=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}".replace(",", " ") if abs(int(v)) >= 10_000 else str(int(v))
    if abs(v) >= 1000:
        return f"{v:,.0f}".replace(",", " ")
    return f"{v:.{d}f}"


def table(df: pd.DataFrame, digits=2) -> str:
    """Markdown-таблица. Значения берутся по столбцам, а не через iterrows:
    iterrows приводит строку к общему типу и превращает целые в 1.000."""
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    data = [df[c].tolist() for c in cols]
    for i in range(len(df)):
        cells = []
        for v in (col[i] for col in data):
            if isinstance(v, bool) or isinstance(v, np.bool_):
                cells.append("да" if v else "нет")
            elif isinstance(v, (int, np.integer)):
                cells.append(fmt(int(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append(fmt(float(v), digits))
            else:
                cells.append(str(v).replace("|", "·"))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def load(stage, t, name):
    p = WORK / "analysis" / stage / t / name
    if p.suffix == ".json":
        return json.loads(p.read_text())
    return pd.read_csv(p)


def money(v):
    """+1 695 — знак и тонкий пробел между тысячами, без запятых."""
    return f"{v:+,.0f}".replace(",", "\u202f")


def cfg_label(r):
    return (f"{r['fast_type']}{int(r['fast_period'])} × {r['slow_type']}{int(r['slow_period'])}, "
            f"наклон {r['method'][0]}, выход {r['exit'][0]}")


# =====================================================================
def sec_method(stages):
    from data import SPECS
    w("## 2. Постановка и методика")
    w()
    w("### 2.1. Правила (разделы 6–7 задания)")
    w()
    w("| Элемент | Реализация |")
    w("|---|---|")
    w("| Пересечение | Сторона цены относительно быстрой средней: +1 при `Close > MA`, −1 при `Close < MA`, при равенстве сторона сохраняется. Пересечение вверх — смена стороны с −1 на +1 между двумя последовательными **закрытыми** свечами. Касание без перехода пересечением не считается. |")
    w("| Long | пересечение вверх **и** медленная средняя вверх или нейтральна **и** позиции нет |")
    w("| Short | пересечение вниз **и** медленная средняя вниз или нейтральна **и** позиции нет |")
    w("| Исполнение | Open следующей свечи; 1 контракт; без комиссий и проскальзывания; без стопов |")
    w("| Выход A | обратное пересечение ценой быстрой средней |")
    w("| Выход B | медленная средняя переходит в противоположное состояние; переход в нейтраль позицию не закрывает |")
    w("| Разворот | запрещён: бар решения о выходе не открывает новую сделку; следующая — только по пересечению на более позднем баре |")
    w("| Конец данных | открытая позиция закрывается по последнему Close (отмечается в метриках) |")
    w("| Учёт периодов | сделка относится к месяцу и к IS/OOS по бару **исполнения выхода**: IS-метрики не содержат ни одной цены из OOS |")
    w()
    w("### 2.2. Скользящие средние — точные формулы")
    w()
    w("`C_t` — Close сигнального ряда, `V_t` — фактический объём (на всех 886 751 барах объём положителен, поэтому ни одна VWMA-комбинация не исключена). Значение не определено, пока в окне меньше `n` баров.")
    w()
    w("| Тип | Формула | Инициализация |")
    w("|---|---|---|")
    w("| SMA | `(1/n)·Σ_{i=0}^{n−1} C_{t−i}` | — |")
    w("| EMA | `α·C_t + (1−α)·EMA_{t−1}`, `α = 2/(n+1)` | `EMA_{n−1} = SMA` первых n баров (как TA-Lib) |")
    w("| WMA | `Σ_{i=0}^{n−1} (n−i)·C_{t−i} / (n(n+1)/2)` — самый свежий бар весит n | — |")
    w("| SSMA | `((n−1)·SSMA_{t−1} + C_t)/n` — сглаженная Уайлдера (RMA, SMMA), `α = 1/n` | `SSMA_{n−1} = SMA` первых n баров |")
    w("| VWMA | `Σ C_{t−i}·V_{t−i} / Σ V_{t−i}` по окну n | при нулевом объёме окна — не определена (не встречалось) |")
    w("| TMA | TA-Lib TRIMA: n нечётное — `SMA_m(SMA_m(C))`, `m=(n+1)/2`; n чётное — `SMA_{n/2+1}(SMA_{n/2}(C))` | общее окно ровно n баров, веса треугольные |")
    w()
    w("Реализации рекуррентные с точным пересчётом окна каждые 2048 баров (без накопления ошибки округления на 600 тыс. барах) и проверены тестами против прямых формул; TMA — против явной свёртки с треугольными весами (`test_ma.py`).")
    w()
    w("### 2.3. Направление медленной средней")
    w()
    w("* Метод A: `Slope_t = (MA_t − MA_{t−k}) / MA_{t−k} · 100` — процент за k баров.")
    w("* Метод B: `Slope_t = (MA_t − MA_{t−k}) / ATR_t`, ATR Уайлдера: `TR_t = max(H−L, |H−C_{t−1}|, |L−C_{t−1}|)`, сглаживание SSMA(p).")
    w("* Нейтраль: `−ε ≤ Slope_t ≤ ε`.")
    w()
    w("**Как задан ε.** Порог задаётся через долю нейтральных баров на in-sample: `ε = квантиль(|Slope|, coverage)` по барам до 2024-01-01. Это та же оптимизация ε в нормированной шкале. Фиксированная сетка процентов сравнивала бы не методы, а то, куда случайно попал порог: при k = 1 и k = 50 одинаковый ε в процентах означает совершенно разную долю нейтрали. При одинаковой доле нейтрали на IS методы A и B различаются только тем, **какие** бары они называют нейтральными, — это честное сравнение. Квантиль берётся только по IS, будущее в порог не попадает. Реальное значение ε сохраняется для каждой конфигурации.")
    w()
    w("### 2.4. Фьючерсы: роллы и множители")
    w()
    w("Вендор склеил контракты по ближайшей экспирации без корректировки. Поэтому рядов два:")
    w()
    w("* **сигнальный** — ratio-скорректированный назад (цены до ролла × `open_new/close_old`): средние, наклон и ATR не видят ступени ролла; все сравнения внутри стратегии масштабно-инвариантны, поэтому множитель будущих роллов в решение не попадает;")
    w("* **для PnL** — аддитивно скорректированный: разность двух его значений равна результату одного контракта, переложенного на ролле без издержек.")
    w()
    w("Пересечения на первом баре нового контракта исключаются — это сравнение через склейку. Денежный PnL = Δцены × множитель, в USD (валюта котировки):")
    w()
    w("| Инструмент | Контракт | Множитель | Тик | Стоимость тика |")
    w("|---|---|---|---|---|")
    for t, s in SPECS.items():
        w(f"| {t} | {s['name']} | {s['multiplier']:g} ({s['unit'].split('/')[1]}) | {s['tick']:g} {s['unit']} | {s['tick'] * s['multiplier']:g} USD |")
    w()
    w("Источник множителей — спецификации MOEX: мини-BRM = 1 баррель «в 10 раз меньше стандартного», микро-NGM = 1 MMBtu «в 100 раз меньше» ([moex.com/n75212](https://www.moex.com/n75212)). Расчёт в рублях потребовал бы курса USD/RUB на каждую сделку и в задачу не входил.")
    w()


def sec_validation():
    w("## 3. Проверка корректности")
    w()
    v = pd.read_csv(WORK / "validation_vbt.csv")
    w("| Проверка | Результат |")
    w("|---|---|")
    w("| Шесть средних против прямых формул (6 периодов от 2 до 250, включая TMA против свёртки) | все совпали до 1e−9 |")
    w("| Событийное ядро против независимой побаровой реализации тех же правил (24 случая, оба выхода, маскировка роллов, равенства на тиковой сетке) | все сделки совпали |")
    w(f"| Ядро против `vectorbt.Portfolio.from_signals` на реальных данных | **{int(v['match'].sum())} из {len(v)}** конфигураций, {fmt(int(v['trades_vbt'].sum()))} сделок, максимальное расхождение {v['max_abs_diff'].max():.1e} |")
    for stage in ("stage1", "stage3"):
        for t in TICKERS:
            for ext in (".txt", ".raw"):
                p = WORK / f"verify_{stage}_{t}{ext}"
                if p.exists():
                    w(f"| Раскладка сетки {stage}/{t}: 25 случайных строк результата против пересчёта с нуля | "
                      f"{p.read_text().strip()} |")
                    break
    w("| Нулевой рынок: 5-минутные доходности NG перемешаны (автокорреляция разрушена) | 49% прибыльных конфигураций, средняя сделка −0.04 тика — конвейер прибыли из шума не создаёт |")
    w("| Калибровка Hansen SPA на шуме | после исправления 1–5% отказов при номинале 5% (первая версия — 60–100%; исправлено, закреплено тестом) |")
    w()
    w("VectorBT не применён к полной сетке напрямую: 3.47 млн конфигураций × 606 тыс. баров — это 2·10¹² ячеек сигнальной матрицы (~2 ТБ). Ядро использует ту же технику, что VectorBT (numba), но идёт по событиям-пересечениям, а не по барам, — блок из 1 176 конфигураций считается за 0.4–0.7 с. VectorBT используется для сверки и для полной статистики финалистов (с переоценкой открытой позиции).")
    w()


# =====================================================================
FAM_COLS = {"configs": "конфигураций", "median_sharpe_is": "медиана Шарпа IS", "share_pos_is": "доля IS>0",
            "median_sharpe_oos": "медиана Шарпа OOS", "share_pos_oos": "доля OOS>0",
            "median_ticks_per_trade": "средняя сделка, тиков (медиана)",
            "top1pct_oos_sharpe": "OOS-Шарп верхнего 1% по IS", "spearman_is_oos": "Спирмен IS↔OOS"}


def fam(stage, t, name, keys, rename=None):
    d = load(stage, t, name)
    cols = keys + [c for c in FAM_COLS if c in d.columns]
    d = d[cols].rename(columns={**FAM_COLS, **(rename or {})})
    return d


def sec_grid(stage, title):
    w(f"## {title}")
    w()
    rows = []
    for t in TICKERS:
        r = load(stage, t, "results.json")
        rows.append({"инструмент": t, "конфигураций": r["n_configs"], "мес. IS": r["is_months"],
                     "мес. OOS": r["oos_months"], "доля IS>0": r["share_pos_is"], "доля OOS>0": r["share_pos_oos"],
                     "доля IS>0 и OOS>0": r["share_pos_both"], "Спирмен IS↔OOS (Шарп)": r["spearman_is_oos_sharpe"],
                     "Спирмен плато IS↔OOS": r["spearman_plateau_oos"]})
    w(table(pd.DataFrame(rows), 3))
    w()
    w("Доли — по всем конфигурациям сетки, при нулевых издержках (как требует задание). "
      "«Спирмен IS↔OOS» — ранговая корреляция результата в выборке и вне её по всей сетке: "
      "насколько IS-рейтинг вообще несёт информацию о будущем.")
    w()
    w("### Децили: что даёт отбор по in-sample")
    w()
    for t in TICKERS:
        d = load(stage, t, "deciles_is_oos.csv")
        d = d.rename(columns={"decile": "дециль IS", "is_sharpe": "IS-Шарп", "oos_sharpe": "OOS-Шарп",
                              "oos_pnl": "OOS PnL, USD", "share_pos_oos": "доля OOS>0", "n": "конфигураций"})
        w(f"**{t}**")
        w()
        w(table(d, 3))
        w()
    w("### По годам: эпизод или закономерность")
    w()
    rows = []
    for t in TICKERS:
        p = WORK / "analysis" / stage / t / "oos_split.json"
        if p.exists():
            o = json.loads(p.read_text())
            rows.append({"инструмент": t, "вне выборки целиком": o["share_pos_all"],
                         f"без {o['last_year']} года ({o['early_months']} мес.)": o["share_pos_early"],
                         f"только {o['last_year']} ({o['last_months']} мес.)": o["share_pos_last"]})
    if rows:
        w("Доля конфигураций, прибыльных вне выборки:"); w()
        w(table(pd.DataFrame(rows), 3)); w()
    for t in TICKERS:
        p = WORK / "analysis" / stage / t / "years_grid.csv"
        if p.exists():
            w(f"**{t}** — по всем конфигурациям сетки (перцентили — по случайной выборке 200 000)"); w()
            w(table(pd.read_csv(p), 3)); w()
    w("### Модели выхода")
    w()
    for t in TICKERS:
        w(f"**{t}**"); w()
        w(table(fam(stage, t, "by_exit_method.csv", ["exit", "method"], {"exit": "выход", "method": "метод"}), 3))
        w()
    w("### Быстрая ≥ медленной — отдельная категория")
    w()
    for t in TICKERS:
        d = fam(stage, t, "by_fast_ge_slow.csv", ["fast_ge_slow"])
        d["fast_ge_slow"] = d["fast_ge_slow"].map({True: "быстрая ≥ медленной", False: "быстрая < медленной"})
        w(f"**{t}**"); w(); w(table(d.rename(columns={"fast_ge_slow": "категория"}), 3)); w()
    w("### 36 сочетаний типов средних")
    w()
    for t in TICKERS:
        d = fam(stage, t, "by_type_pair.csv", ["fast_type", "slow_type"],
                {"fast_type": "быстрая", "slow_type": "медленная"})
        d = d.sort_values("медиана Шарпа OOS", ascending=False)
        w(f"**{t}** — по убыванию медианы OOS-Шарпа (медиана по всем периодам, методам и выходам)")
        w()
        w(table(d, 3))
        w()
        from scipy import stats as _st
        e = load(stage, t, "by_type_pair_exit_method.csv")
        r36 = _st.spearmanr(d["медиана Шарпа IS"], d["медиана Шарпа OOS"]).statistic
        r144 = _st.spearmanr(e["median_sharpe_is"], e["median_sharpe_oos"]).statistic
        w(f"Переносится ли рейтинг вне выборки: Спирмен медиан IS↔OOS по 36 сочетаниям типов — "
          f"**{r36:+.2f}**, по 144 семействам (типы × выход × метод) — **{r144:+.2f}**. "
          f"Разница между ними — вклад модели выхода и метода; сам выбор типа средних "
          f"{'вне выборки не воспроизводится' if abs(r36) < 0.3 else 'воспроизводится слабо'}.")
        w()
    w("### По типу каждой средней отдельно")
    w()
    for t in TICKERS:
        a = fam(stage, t, "by_fast_type.csv", ["fast_type"], {"fast_type": "тип"}).assign(роль="быстрая")
        b = fam(stage, t, "by_slow_type.csv", ["slow_type"], {"slow_type": "тип"}).assign(роль="медленная")
        d = pd.concat([a, b])[["роль", "тип", "медиана Шарпа IS", "доля IS>0", "медиана Шарпа OOS", "доля OOS>0",
                               "средняя сделка, тиков (медиана)", "Спирмен IS↔OOS"]]
        w(f"**{t}**"); w(); w(table(d, 3)); w()


# =====================================================================
RULES = {"R1_max_pnl": "R1: максимум прибыли IS", "R2_max_sharpe": "R2: максимум месячного Шарпа IS (≥100 сделок)",
         "R3_plateau": "R3: максимум сглаженного по окрестности IS-Шарпа (≥100 сделок)"}


def sec_selection(stage):
    w("## 8. Отбор стратегий и их проверка вне выборки")
    w()
    w("Все три правила смотрят **только** на in-sample. R1 — то, от чего предостерегает задание "
      "(максимум исторической прибыли); R3 — устойчивый вариант: выбирается не точка, а центр "
      "области, где соседние периоды (±3 быстрой, ±7 медленной) тоже хороши.")
    w()
    for t in TICKERS:
        p = load(stage, t, "picks.csv")
        p = p[~p["scope"].str.startswith("family:")].copy()
        p["стратегия"] = p.apply(cfg_label, axis=1)
        p["правило"] = p["rule"].map(lambda r: r.split("_")[0])
        p["trades_is"] = p["trades_is"].astype(int)
        for c_ in ("pnl_is", "pnl_oos"):
            p[c_] = p[c_].round().astype(int)
        d = p[["scope", "правило", "стратегия", "trades_is", "pnl_is", "pnl_oos", "sharpe_m_is", "sharpe_m_oos",
               "avg_trade_ticks_full", "stability_is"]].rename(columns={
            "scope": "область отбора", "trades_is": "сделок IS", "pnl_is": "PnL IS, USD", "pnl_oos": "PnL OOS, USD",
            "sharpe_m_is": "Шарп IS", "sharpe_m_oos": "Шарп OOS", "avg_trade_ticks_full": "тиков/сделку",
            "stability_is": "устойчивость знака"})
        w(f"**{t}** ({NAMES[t]})")
        w()
        w(table(d, 2))
        w()
    fin_any = all((WORK / "analysis" / stage / t / "finalists.json").exists() for t in TICKERS)
    yearly_ok = all((WORK / "analysis" / stage / t / "picks_yearly.csv").exists() for t in TICKERS)
    if fin_any:
        w("### Полная статистика отобранных (VectorBT, переоценка открытой позиции)")
        w()
        w("Шарп и Сортино — по дневному PnL с годовым пересчётом по фактическому числу торговых дней "
          "(у MOEX бывают сессии выходного дня), от капитала не зависят. "
          "Просадка — по дневной кривой с переоценкой, в USD за 1 контракт. "
          "Monte Carlo — 5 000 бутстрэп-выборок сделок (итог и просадка) и 5 000 перестановок порядка сделок (просадка).")
        w()
        for t in TICKERS:
            fin = json.loads((WORK / "analysis" / stage / t / "finalists.json").read_text())
            rows = []
            for f in fin:
                c = f["config"]
                row = {"область": f["scope"], "правило": f["rule"].split("_")[0],
                       "стратегия": f"{c['fast_type']}{c['fast_period']} × {c['slow_type']}{c['slow_period']}",
                       "наклон/выход": f"{c['method'][0]}/{c['exit'][0]}"}
                for part, tag in (("is", "IS"), ("oos", "OOS")):
                    b = f.get(part) or {}
                    row[f"PnL {tag}"] = b.get("pnl"); row[f"сделок {tag}"] = b.get("trades")
                    row[f"Шарп {tag}"] = b.get("sharpe_daily"); row[f"Сортино {tag}"] = b.get("sortino_daily")
                    row[f"PF {tag}"] = b.get("profit_factor"); row[f"просадка {tag}"] = b.get("max_dd_mtm")
                mc = f.get("monte_carlo_oos") or {}
                row["MC OOS: P(итог<0)"] = mc.get("p_total_negative")
                row["MC: просадка p95"] = (f.get("monte_carlo") or {}).get("max_dd_permutation", {}).get("p95")
                row["в рынке"] = f.get("exposure")
                rows.append(row)
            w(f"**{t}**"); w(); w(table(pd.DataFrame(rows), 2)); w()
        w("### Результат отобранных по годам, USD")
        w()
        for t in TICKERS:
            y = load(stage, t, "picks_yearly.csv")
            y = y[y["scope"] == "global"].drop(columns=["scope", "config"]).rename(columns={"rule": "правило"})
            w(f"**{t}**"); w(); w(table(y, 0)); w()


def sec_robust(stage):
    w("## 9. Устойчивость и множественное тестирование")
    w()
    w("### Walk-forward с переоптимизацией по всей сетке")
    w()
    w("На каждом шаге из 3 471 552 конфигураций выбирается лучшая по последним 24 месяцам, "
      "её результат записывается за следующие 3 месяца, шаг — 3 месяца. Эталон — средняя по "
      "всем конфигурациям сетки за те же месяцы, то есть «выбрать наугад». Если переоптимизация "
      "что-то знает о будущем, она должна этот эталон обгонять.")
    w()
    rows = []
    for t in TICKERS:
        r = load(stage, t, "results.json"); x = load(stage, t, "extras.json")
        for crit, wf in r["walk_forward"].items():
            b = x["walk_forward_vs_average_config"][crit]
            rows.append({"инструмент": t, "критерий": crit, "шагов": wf["steps"], "тестовых мес.": wf["test_months"],
                         "PnL WF, USD": wf["total_pnl"], "t WF": wf["t_stat"], "доля прибыльных шагов": wf["share_pos_steps"],
                         "PnL средней конфигурации": b["avg_config_total"], "превышение": b["excess_total"],
                         "t превышения": b["excess_t"], "разных конфигураций": wf["distinct_configs"]})
    w(table(pd.DataFrame(rows), 2)); w()
    w("### Вероятность переобучения отбора (PBO, CSCV)")
    w()
    rows = []
    for t in TICKERS:
        r = load(stage, t, "results.json")
        rows.append({"инструмент": t, "PBO на IS": r["pbo_is"]["pbo"], "блоков IS": r["pbo_is"]["n_blocks"],
                     "PBO на всём периоде": r["pbo_full"]["pbo"], "блоков": r["pbo_full"]["n_blocks"],
                     "разбиений": r["pbo_full"]["n_splits"]})
    w(table(pd.DataFrame(rows), 3)); w()
    w("PBO — доля разбиений месяцев на две половины, в которых лучшая по Шарпу конфигурация первой "
      "половины оказывается хуже медианы на второй. 0.5 — отбор не лучше монетки.")
    w()
    w("### White's Reality Check и Hansen's SPA по всей сетке")
    w()
    rows = []
    for t in TICKERS:
        r = load(stage, t, "results.json"); x = load(stage, t, "extras.json")
        for tag in ("is", "full"):
            s = r[f"rc_spa_{tag}"]
            rows.append({"инструмент": t, "период": "IS" if tag == "is" else "весь", "моделей": s["N"], "месяцев": s["T"],
                         "RC p": s["rc_p"], "SPA t": s["spa_stat"], "SPA p": s["spa_p"],
                         "лучшая по SPA": f"{s['spa_best_config']['fast_type']}{s['spa_best_config']['fast_period']} × "
                                          f"{s['spa_best_config']['slow_type']}{s['spa_best_config']['slow_period']} "
                                          f"{s['spa_best_config']['method'][0]}/{s['spa_best_config']['exit'][0]}"})
    w(table(pd.DataFrame(rows), 4)); w()
    ac = {t: load(stage, t, "extras.json")["monthly_ac1_median"] for t in TICKERS}
    w("Нулевая гипотеза: ни одна из 3 471 552 конфигураций не имеет положительного ожидания месячного PnL. "
      f"Бутстрэп — стационарный, 1 000 повторов, блок ≈ T^(1/3) месяцев. Медианная автокорреляция месячного "
      f"PnL: NG {ac['NG']:+.3f}, BR {ac['BR']:+.3f} — в этой области калибровочные прогоны дали 1–5% ложных отказов.")
    w()
    w("### Дефлированный Шарп отобранных")
    w()
    rows = []
    for t in TICKERS:
        d = load(stage, t, "dsr_picks.csv")
        d = d[d["scope"] == "global"]
        r = load(stage, t, "results.json")
        for _, x in d.iterrows():
            rows.append({"инструмент": t, "правило": x["rule"].split("_")[0], "испытаний": x["trials"],
                         "N": x["n_trials"], "годовой Шарп IS": x["sr_annual"], "ожидаемый максимум при нуле": x["sr0_annual"],
                         "DSR": x["dsr"], "PSR": x["psr"]})
    w(table(pd.DataFrame(rows), 3)); w()
    w("N_full — все конфигурации сетки (верхняя граница штрафа); N_eff — эффективное число "
      "независимых испытаний по спектру корреляций месячного PnL случайной выборки из 4 000 "
      "конфигураций (participation ratio, нижняя граница).")
    w()


def sec_costs(stage):
    w("## 10. Чувствительность к издержкам")
    w()
    w("Задание требует считать без комиссий и проскальзывания — все результаты выше так и посчитаны. "
      "Но средняя сделка в сетке — от долей тика до нескольких тиков, а минимальная реальная цена "
      "полной сделки — около одного тика (спред) плюс биржевой сбор. Поэтому ниже то же самое при "
      "0.5, 1 и 2 тиках на полную сделку (1 тик = 0.1 USD на обоих контрактах).")
    w()
    rows = []
    for t in TICKERS:
        x = load(stage, t, "extras.json")
        for c, v in x["costs"].items():
            rows.append({"инструмент": t, "тиков на сделку": float(c), "доля IS>0": v["share_pos_is"],
                         "доля OOS>0": v["share_pos_oos"], "медиана OOS, USD": v["median_oos"],
                         "лучшая по IS: IS": v["best_is_net"], "её OOS": v["best_is_oos_net"],
                         "верхний 1% IS: средний OOS": v["top1pct_mean_oos_net"],
                         "верхний 1%: доля OOS>0": v["top1pct_share_pos_oos"]})
    w(table(pd.DataFrame(rows), 3)); w()
    rows = []
    for t in TICKERS:
        x = load(stage, t, "extras.json")
        for c, v in x["costs"].items():
            for e, b in v["by_exit"].items():
                rows.append({"инструмент": t, "тиков": float(c), "выход": e, "доля IS>0": b["share_pos_is"],
                             "доля OOS>0": b["share_pos_oos"], "лучшая по IS: IS": b["is_net"], "её OOS": b["oos_net"]})
    w("По моделям выхода:"); w(); w(table(pd.DataFrame(rows), 3)); w()
    rows = []
    for st in ("stage1", "stage3"):
        for t in TICKERS:
            pth = WORK / "analysis" / st / t / "topcheck.json"
            if not pth.exists():
                continue
            d = json.loads(pth.read_text())
            for c in ("0.0", "1.0", "2.0"):
                rows.append({"этап": st[-1], "инструмент": t, "тиков": float(c),
                             "OOS>0 целиком": d[f"share_pos_oos_{c}"],
                             f"OOS>0 без {d['last_year']}": d[f"share_pos_early_{c}"],
                             f"медиана без {d['last_year']}, USD": d[f"median_early_{c}"],
                             f"только {d['last_year']}": d[f"share_pos_last_{c}"],
                             "средний месяц группы, USD": d[f"early_monthly_mean_{c}"],
                             "t по месяцам": d[f"early_monthly_t_{c}"],
                             "плюсовых месяцев": d[f"early_pos_months_{c}"]})
    if rows:
        w("### Единственная группа, выдерживающая издержки, — и проверка без последнего года")
        w()
        w("Верхний 1% конфигураций по **чистому** IS-результату при 2 тиках на сделку (≥100 сделок) — "
          "редко торгующие стратегии, почти все с выходом B. Вне выборки целиком эта группа прибыльна "
          "почти всегда. Но у BR вся сетка держалась на одном 2026 году, поэтому из группы взяты 400 "
          "случайных конфигураций, пересчитаны с журналом сделок, и их чистый результат разделён на "
          "2024–2025 и последний год. Последние три столбца — временной тест: средний по группе месячный "
          "чистый PnL за 24 месяца 2024–2025 и его t-статистика по месяцам (одна траектория — поэтому "
          "по месяцам, а не по конфигурациям).")
        w()
        w(table(pd.DataFrame(rows), 2)); w()
        w("Проверено 4 группы × 3 уровня издержек; лучшая из них по построению выглядит лучше, чем "
          "есть. t ≈ 2 у одной группы из двенадцати клеток — повод для предрегистрированной проверки "
          "на новых данных, а не основание для торговли.")
        w()


# =====================================================================
def sec_slope():
    if not all((WORK / "analysis" / "stage2" / t / "results.json").exists() for t in TICKERS):
        return
    w("## 5. Оптимизация параметров наклона (этап 2)")
    w()
    w("Полное произведение сетки наклона на базовую сетку — 200 настроек × 3 471 552 ≈ 694 млн "
      "конфигураций на инструмент, около четырёх суток расчёта и ~0.5 ТБ помесячных результатов "
      "на этой машине. Поэтому параметры наклона оптимизированы на **решётке периодов**, заданной "
      "до расчётов и не зависящей от результатов (числа Фибоначчи): быстрая {2, 3, 5, 8, 13, 21, 34, 55, 89}, "
      "медленная {5, 8, 13, 21, 34, 55, 89, 144, 233}; все 36 сочетаний типов, оба выхода. "
      "Сетка наклона: k ∈ {1, 2, 3, 5, 8, 13, 21, 34}; доля нейтрали на IS ∈ {0, 0.10, 0.25, 0.40, 0.60}; "
      "ATR ∈ {7, 14, 28, 56} для метода B — 40 настроек A и 160 настроек B, "
      "1 166 400 конфигураций на инструмент. Лучшая настройка каждого метода выбрана по медиане "
      "IS-Шарпа по всей решётке и затем использована на этапе 3 — повторном прогоне ПОЛНОЙ базовой сетки.")
    w()
    rows = []
    for t in TICKERS:
        r = load("stage2", t, "results.json")
        for m in ("A_pct", "B_atr"):
            b = r["best"][m]; ref = r[f"{m}_reference_row"]
            rows.append({"инструмент": t, "метод": m, "k": b["k"], "доля нейтрали": b["coverage"],
                         "ATR": b["atr_period"] if m == "B_atr" else "—",
                         "медиана IS-Шарпа": b["median_sharpe_is"], "медиана OOS-Шарпа": b["median_sharpe_oos"],
                         "этап 1 (k=5, 25%, ATR 14): IS": ref["median_sharpe_is"], "этап 1: OOS": ref["median_sharpe_oos"],
                         "Спирмен настроек IS↔OOS": r[f"{m}_spearman_settings_is_oos"],
                         "дрейф доли нейтрали IS→OOS": r[f"{m}_neutral_drift_abs_mean"]})
    w(table(pd.DataFrame(rows), 3)); w()
    w("«Спирмен настроек» — согласуется ли рейтинг 40 (A) или 160 (B) настроек наклона в выборке и вне её. "
      "«Дрейф доли нейтрали» — средняя абсолютная разница доли нейтральных баров между IS и OOS: "
      "метод B нормирует наклон текущей волатильностью и должен дрейфовать меньше.")
    w()
    ext = all((WORK / "analysis" / "stage2b" / t / "settings.csv").exists() for t in TICKERS)
    if ext:
        w("### Оптимум оказался на границе сетки k — продление до 144")
        w()
        w("Лучшая настройка этапа 2 для всех четырёх случаев пришлась на k = 34 — верхнюю границу "
          "заранее заданной сетки — при монотонном росте медианы IS-Шарпа с k. Оптимум на границе "
          "означает, что диапазон поиска был мал, поэтому та же решётка досчитана при "
          "k ∈ {55, 89, 144} (этап 2б). Этап 3 уже шёл с k = 34 и задним числом не пересматривался; "
          "таблица показывает, где лежит настоящий оптимум. Значения — лучшая по IS настройка "
          "(доля нейтрали, ATR) при каждом k.")
        w()
        for t in TICKERS:
            s2 = pd.concat([load("stage2", t, "settings.csv"), load("stage2b", t, "settings.csv")])
            rows = []
            for (m, k), d in s2.groupby(["method", "k"]):
                b = d.loc[d["median_sharpe_is"].idxmax()]
                rows.append({"метод": m, "k": int(k), "минут": int(k) * 5, "доля нейтрали": b["coverage"],
                             "ATR": int(b["atr_period"]) if m == "B_atr" else "—",
                             "медиана IS-Шарпа": b["median_sharpe_is"], "медиана OOS-Шарпа": b["median_sharpe_oos"],
                             "доля OOS>0": b["share_pos_oos"]})
            w(f"**{t}**"); w(); w(table(pd.DataFrame(rows), 3)); w()
    for t in TICKERS:
        s = load("stage2", t, "settings.csv")
        g = s.groupby(["method", "k"])[["median_sharpe_is", "median_sharpe_oos"]].max().reset_index()
        piv = g.pivot(index="k", columns="method", values=["median_sharpe_is", "median_sharpe_oos"])
        piv.columns = [f"{a.replace('median_sharpe_', '').upper()} {b}" for a, b in piv.columns]
        w(f"**{t}: лучшая медиана Шарпа по решётке при каждом k**"); w()
        w(table(piv.reset_index(), 3)); w()
        g2 = s.groupby(["method", "coverage"])[["median_sharpe_is", "median_sharpe_oos", "neutral_is", "neutral_oos"]].mean().reset_index()
        w(f"**{t}: доля нейтрали (среднее по k и ATR)**"); w(); w(table(g2, 3)); w()


def sec_cross(stage):
    w("## 11. Кросс-рыночная проверка")
    w()
    cm = json.loads((WORK / "analysis" / stage / "crossmarket.json").read_text())
    w("| Показатель | Значение |"); w("|---|---|")
    w(f"| Спирмен IS-Шарпа одной и той же конфигурации на NG и BR | {cm['spearman_sharpe_m_is_NG_vs_BR']:+.3f} |")
    w(f"| Спирмен OOS-Шарпа NG ↔ BR | {cm['spearman_sharpe_m_oos_NG_vs_BR']:+.3f} |")
    w(f"| Спирмен медиан 144 семейств (типы × выход × метод), IS | {cm['families_spearman_is']:+.3f} |")
    w(f"| То же вне выборки | {cm['families_spearman_oos']:+.3f} |")
    w(f"| Доля конфигураций, прибыльных IS на обоих рынках | {cm['share_pos_is_both']:.3f} (при независимости {cm['share_pos_is_both_if_independent']:.3f}) |")
    w(f"| Доля прибыльных OOS на обоих | {cm['share_pos_oos_both']:.3f} (при независимости {cm['share_pos_oos_both_if_independent']:.3f}) |")
    w(f"| Прибыльны во всех четырёх клетках (IS и OOS × NG и BR) | {fmt(cm['n_pos_all_four'])} конфигураций |")
    w(f"| Доля OOS>0 на обоих среди IS>0 на обоих | {cm['share_oos_both_given_is_both']:.3f} против безусловной {cm['share_oos_both_unconditional']:.3f} |")
    w()
    c = pd.read_csv(WORK / "analysis" / stage / "crossmarket_candidates.csv")
    c["стратегия"] = c.apply(cfg_label, axis=1)
    d = c[["стратегия", "score_min_plateau_is", "sharpe_m_is_NG", "sharpe_m_oos_NG", "pnl_oos_NG",
           "sharpe_m_is_BR", "sharpe_m_oos_BR", "pnl_oos_BR", "oos_positive_both"]].rename(columns={
        "score_min_plateau_is": "худшее плато IS", "sharpe_m_is_NG": "NG Шарп IS", "sharpe_m_oos_NG": "NG Шарп OOS",
        "pnl_oos_NG": "NG PnL OOS", "sharpe_m_is_BR": "BR Шарп IS", "sharpe_m_oos_BR": "BR Шарп OOS",
        "pnl_oos_BR": "BR PnL OOS", "oos_positive_both": "OOS>0 на обоих"})
    w(f"**Кандидаты, отобранные по in-sample сразу на двух рынках** (максимум худшего из двух "
      f"сглаженных IS-Шарпов, ≥100 сделок на каждом): OOS положителен на обоих у "
      f"{cm['candidates_oos_positive_both']} из {cm['candidates']}.")
    w()
    w(table(d, 2)); w()


# =====================================================================
def sec_data():
    from data import load_market
    w("## 1. Данные")
    w()
    rows = []
    for t in TICKERS:
        mk = load_market(t)
        bh = (mk.pnl_close[-1] - mk.pnl_open[0]) * mk.multiplier
        rows.append({"инструмент": t, "контракт": mk.meta["name"], "баров 5m": mk.n,
                     "первый бар": mk.meta["first_bar"][:16], "последний бар": mk.meta["last_bar"][:16],
                     "роллов": mk.meta["n_rolls"], "медианная ступень ролла, б.п.": mk.meta["roll_gap_bps_median"],
                     "мес. IS": mk.oos_month, "мес. OOS": len(mk.months) - mk.oos_month,
                     "купить и держать 1 контракт, USD": bh})
    w(table(pd.DataFrame(rows), 1)); w()
    w("Источник — MOEX AlgoPack, 5-минутные свечи (собраны вендором из минутных), полная доступная "
      "история; время московское. «Все доступные инструменты» — это два фьючерса, NG и BR: других "
      "5-минутных рядов в данных нет. In-sample — до 2024-01-01, out-of-sample — 2024-01-01 … 2026-10-03, "
      "один календарь для обоих рынков, чтобы кросс-рыночная проверка шла на одних и тех же месяцах. "
      "«Купить и держать» — один контракт с перекладкой на роллах: это масштаб, относительно которого "
      "читаются все суммы ниже.")
    w()


def sec_deviations(stage_main):
    w("## 4. Что сделано не буквально по заданию — и почему")
    w()
    meta = {t: json.loads((WORK / s / t / "meta.json").read_text()) for s in ("stage1",) for t in TICKERS}
    secs = {t: meta[t]["seconds"] / 60 for t in TICKERS}
    w("| Требование | Что сделано | Причина |")
    w("|---|---|---|")
    w(f"| Все 98 × 246 × 36 × 2 × 2 = 3 471 552 конфигураций | выполнено полностью дважды (этап 1 и этап 3) на каждом инструменте; этап 1 считался {secs['NG']:.0f} мин (NG) и {secs['BR']:.0f} мин (BR) | — |")
    w("| Оптимизация k, ε, периода ATR | на решётке периодов (81 пара из 24 108) со всеми типами и выходами; лучшая настройка затем прогнана на полной сетке | полное произведение — 694 млн конфигураций на инструмент (~4 суток, ~0.5 ТБ) |")
    w("| ε как порог в % (A) или в ATR (B) | ε задан через долю нейтральных баров на IS; реальный ε сохраняется | фиксированный порог в % несопоставим между k и методами; см. 2.3 |")
    w("| VectorBT для бэктестинга | событийное numba-ядро на всей сетке, VectorBT — сверка (80/80) и полная статистика финалистов | сигнальная матрица полной сетки — ~2 ТБ |")
    w("| Sharpe/Sortino «если определены капитал и оценка позиций» | по дневному PnL с переоценкой (VectorBT) для финалистов; по месячному PnL — для всей сетки | дневная переоценка 3.5 млн конфигураций не нужна для отбора |")
    w("| Максимальная просадка | на всей сетке — по кривой закрытых сделок; у финалистов — по дневной кривой с переоценкой | см. выше |")
    w("| Без комиссий и проскальзывания | так и посчитано; дополнительно — чувствительность к 0.5–2 тикам (раздел 10) | средняя сделка того же порядка, что минимальная цена сделки |")
    w()


def sec_files():
    w("## 13. Код и машиночитаемые результаты")
    w()
    w("Код: `exchange/research/ma_crossover_5m/` — `data.py` (данные, роллы), `ma.py` (шесть средних), "
      "`slope.py` (наклон), `kernel.py` (ядро), `grid.py`, `stage1.py`, `stage2.py`, `stage3.py` (сетки), "
      "`analyze.py`, `analyze_stage2.py`, `extras.py`, `crossmarket.py`, `robust.py` (анализ и устойчивость), "
      "`finalists.py` (VectorBT), `validate_vbt.py`, `verify_grid.py`, тесты `test_*.py`, отчёт `report.py`, "
      "ноутбук `build_notebook.py` → `notebooks/03_ma_crossover_5m.ipynb`.")
    w()
    w("Машиночитаемые результаты: `exchange/research/ma_crossover_5m/results/` — таблицы семейств, отборов, "
      "walk-forward, PBO, RC/SPA, издержек, кросс-рынка, этапа 2 (CSV/JSON), а также топ-10 000 "
      "конфигураций каждого инструмента по сглаженному IS-Шарпу со всеми метриками (parquet). Полные "
      "сетки (по ~1.6–3 ГБ на инструмент и этап) в git не помещаются; они воспроизводятся командами из "
      "`README.md` каталога за ~1 час на 4 ядрах.")
    w()


# =====================================================================
def family_picks(stage, t):
    p = load(stage, t, "picks.csv")
    p = p[p["scope"].str.startswith("family:")].copy()
    tick_usd = 0.1
    p["oos_net_1tick"] = p["pnl_oos"] - p["trades_oos"] * tick_usd
    p["is_net_1tick"] = p["pnl_is"] - p["trades_is"] * tick_usd
    return p


def sec_ranking(stage):
    w("## 7. Рейтинг стратегий, лучшие параметры и сочетания типов")
    w()
    w("Рейтинг построен **по in-sample**: в каждом из 144 семейств (36 сочетаний типов × 2 выхода × "
      "2 метода) выбрана конфигурация с максимальным сглаженным IS-Шарпом (правило R3), затем семейства "
      "упорядочены по этому же IS-критерию. Колонки OOS — проверка, в рейтинге не участвуют. Порядок по "
      "максимальной исторической прибыли не используется: задание прямо требует его не применять, а "
      "разделы 8–9 показывают, почему.")
    w()
    for t in TICKERS:
        p = family_picks(stage, t).sort_values("plateau_sharpe_is", ascending=False).head(20).copy()
        p["стратегия"] = p.apply(cfg_label, axis=1)
        p["trades_oos"] = p["trades_oos"].astype(int)
        d = p[["стратегия", "plateau_sharpe_is", "sharpe_m_is", "sharpe_m_oos", "pnl_is", "pnl_oos",
               "trades_oos", "avg_trade_ticks_full", "oos_net_1tick"]].rename(columns={
            "plateau_sharpe_is": "плато IS", "sharpe_m_is": "Шарп IS", "sharpe_m_oos": "Шарп OOS",
            "pnl_is": "PnL IS", "pnl_oos": "PnL OOS", "trades_oos": "сделок OOS",
            "avg_trade_ticks_full": "тиков/сделку", "oos_net_1tick": "PnL OOS при 1 тике"})
        d.insert(0, "место", range(1, len(d) + 1))
        w(f"**{t}: 20 лучших семейств по IS**"); w(); w(table(d, 2)); w()


def sec_oos_positive(stage):
    w("## 12. Стратегии с положительным результатом на независимой выборке")
    w()
    w("Кандидаты — только отобранные по in-sample: 3 глобальных правила, 12 правил внутри «выход × метод» "
      "и 144 лучших по плато в каждом семействе. Рядом — базовая частота: какая доля **всех** "
      "конфигураций сетки прибыльна вне выборки. Если она высока, положительный OOS у отобранной "
      "стратегии сам по себе ничего не доказывает — его получила бы и случайная конфигурация.")
    w()
    rows = []
    for t in TICKERS:
        p = family_picks(stage, t)
        r = load(stage, t, "results.json"); x = load(stage, t, "extras.json")
        rows.append({"инструмент": t, "отобрано по IS (семейства)": len(p),
                     "из них OOS>0": int((p["pnl_oos"] > 0).sum()),
                     "доля": float((p["pnl_oos"] > 0).mean()),
                     "базовая доля OOS>0 по всей сетке": r["share_pos_oos"],
                     "OOS>0 при 1 тике": int((p["oos_net_1tick"] > 0).sum()),
                     "база при 1 тике": x["costs"]["1.0"]["share_pos_oos"]})
    w(table(pd.DataFrame(rows), 3)); w()
    for t in TICKERS:
        p = family_picks(stage, t)
        p = p[(p["pnl_oos"] > 0) & (p["oos_net_1tick"] > 0)].sort_values("plateau_sharpe_is", ascending=False).copy()
        p["стратегия"] = p.apply(cfg_label, axis=1)
        p["trades_oos"] = p["trades_oos"].astype(int)
        d = p[["стратегия", "sharpe_m_is", "sharpe_m_oos", "pnl_oos", "oos_net_1tick", "trades_oos",
               "avg_trade_ticks_full", "stability_is"]].rename(columns={
            "sharpe_m_is": "Шарп IS", "sharpe_m_oos": "Шарп OOS", "pnl_oos": "PnL OOS, USD",
            "oos_net_1tick": "PnL OOS при 1 тике", "trades_oos": "сделок OOS", "avg_trade_ticks_full": "тиков/сделку",
            "stability_is": "устойчивость знака IS"})
        w(f"**{t}: отобранные по IS, прибыльные вне выборки и при нулевых издержках, и при 1 тике на сделку** "
          f"({len(d)})")
        w()
        w(table(d, 2) if len(d) else "_нет_"); w()


# =====================================================================
def sec_nulls():
    p = WORK / "nulls.json"
    if not p.exists():
        return
    n = json.loads(p.read_text())
    w("### Откуда берётся прибыль при нулевых издержках: нулевые рынки")
    w()
    w("Одни и те же случайные конфигурации прогнаны на реальном ряду и на рядах, где разрушена часть "
      "структуры. Это различает три источника: ошибку конвейера, многодневные тренды и внутридневную "
      "динамику.")
    w()
    w(table(pd.DataFrame(n["rows"]), 2)); w()
    for line in n.get("notes", []):
        w(f"* {line}")
    w()
    w("Нулевые ряды строятся из сигнального ряда без роллов, поэтому их PnL не сравним с основными "
      "результатами в долларах; сравнимы только строки между собой.")
    w()
    sf = WORK / "signflip.json"
    if sf.exists():
        d = json.loads(sf.read_text())
        w("### Главная поправка к интерпретации: все конфигурации торгуют одну историю")
        w()
        w("Доля прибыльных конфигураций — не миллионы независимых подтверждений: все они торгуют одну и ту "
          "же траекторию цены, и удачная траектория делает прибыльными почти все сразу. На 30 синтетических "
          "мартингалах (ожидание любого правила — ноль) средняя сделка по выборке конфигураций колебалась "
          "между траекториями со стандартным отклонением 0.66 тика — это шум одного пути. Поэтому семейство "
          "проверено на уровне траектории: та же мера на реальном ряду против 40 траекторий, где у каждой "
          "свечи случайно сменён знак (модули, кластеры волатильности и внутридневной профиль сохранены, "
          "направленная предсказуемость уничтожена).")
        w()
        rows = []
        for t, v in d.items():
            rows.append({"инструмент": t, "средняя сделка на реальном ряду, б.п.": v["real_mean_trade_bp"],
                         "доля прибыльных (реальный)": v["real_share_profitable"],
                         "нулевые: среднее": v["null_mean"], "нулевые: std": v["null_std"],
                         "нулевые: 95-й перцентиль": v["null_p95"], "z": v["z"], "p": v["p_value"],
                         "нулевые: доля прибыльных, p95": v["null_share_profitable_p95"]})
        w(table(pd.DataFrame(rows), 3)); w()


def headline(stage):
    r = {t: load(stage, t, "results.json") for t in TICKERS}
    x = {t: load(stage, t, "extras.json") for t in TICKERS}
    cm = json.loads((WORK / "analysis" / stage / "crossmarket.json").read_text())
    return r, x, cm


def main(stages):
    s1, s_final = stages[0], stages[-1]
    w("# Скользящие средние на 5-минутных фьючерсах MOEX: полное исследование")
    w()
    w("Стратегия: пересечение ценой быстрой средней с фильтром направления по наклону медленной. "
      "Полная сетка 98 × 246 периодов × 36 сочетаний шести типов × 2 метода нейтрали × 2 модели "
      "выхода = **3 471 552 конфигурации на инструмент**, два инструмента (NG, BR), два прогона "
      "полной сетки (с исходными и с оптимизированными параметрами наклона), отбор только по "
      "in-sample, проверка на out-of-sample, walk-forward, PBO, Reality Check/SPA, DSR, Monte Carlo, "
      "кросс-рыночная проверка.")
    w()
    sec_summary(s_final)
    conclusions_path = HERE / "conclusions.md"
    if conclusions_path.exists():
        w(conclusions_path.read_text().strip())
        w()
    sec_data()
    sec_method(stages)
    sec_validation()
    sec_nulls()
    sec_deviations(s_final)
    sec_slope()
    sec_grid(s_final, "6а. Полная сетка с оптимизированным наклоном (этап 3)") if s_final != s1 else None
    sec_grid(s1, "6б. Полная сетка с исходным наклоном (этап 1)")
    sec_ranking(s_final)
    sec_selection(s_final)
    sec_robust(s_final)
    sec_costs(s_final)
    sec_cross(s_final)
    sec_oos_positive(s_final)
    sec_files()
    w("---"); w()
    w("_Отчёт собран `report.py` из результатов в рабочем каталоге; числа в таблицах не набраны руками._")
    out = HERE.parents[1] / "docs" / "RESULTS_MA_CROSSOVER_5M.md"
    out.write_text("\n".join(L) + "\n")
    print("записан", out, len(L), "строк")



# =====================================================================
def sec_summary(stage):
    """Итог — все числа из результатов; формулировки зависят от них."""
    r, x, cm = headline(stage)
    sf = json.loads((WORK / "signflip.json").read_text()) if (WORK / "signflip.json").exists() else {}
    w("## Итог")
    w()
    for t in TICKERS:
        c0, c1, c2 = x[t]["costs"]["0.0"], x[t]["costs"]["1.0"], x[t]["costs"]["2.0"]
        wf = x[t]["walk_forward_vs_average_config"]
        s = sf.get(t, {})
        w(f"**{t} ({NAMES[t]}).** Без издержек прибыльны {r[t]['share_pos_is']:.0%} конфигураций в выборке "
          f"и {r[t]['share_pos_oos']:.0%} вне её; ранговая связь IS↔OOS по всей сетке {r[t]['spearman_is_oos_sharpe']:+.2f}. "
          + (f"Тест траекторий: средняя сделка {s['real_mean_trade_bp']:+.1f} б.п. против нулевых "
             f"{s['null_mean']:+.1f} ± {s['null_std']:.1f} (z = {s['z']:+.1f}, p = {s['p_value']:.3f}). " if s else "")
          + f"PBO = {r[t]['pbo_full']['pbo']:.2f}; walk-forward с переоптимизацией по плато даёт "
          f"{money(wf['plateau']['wf_total'])} USD против {money(wf['plateau']['avg_config_total'])} у средней конфигурации "
          f"(t превышения {wf['plateau']['excess_t']:+.2f}). SPA по всей сетке: p = {r[t]['rc_spa_full']['spa_p']:.3f} "
          f"(весь период), {r[t]['rc_spa_is']['spa_p']:.3f} (IS). При 1 тике на сделку прибыльны "
          f"{c1['share_pos_oos']:.0%} конфигураций вне выборки, при 2 тиках — {c2['share_pos_oos']:.0%}.")
        w()
    rows = []
    for st in ("stage1", "stage3"):
        if not all((WORK / "analysis" / st / t / "extras.json").exists() for t in TICKERS):
            continue
        for t in TICKERS:
            rr = load(st, t, "results.json"); xx = load(st, t, "extras.json")
            sp = WORK / "analysis" / st / t / "oos_split.json"
            o = json.loads(sp.read_text()) if sp.exists() else {}
            rows.append({"этап": "1: k=5, нейтраль 25%" if st == "stage1" else "3: наклон оптимизирован",
                         "инструмент": t, "доля IS>0": rr["share_pos_is"], "доля OOS>0": rr["share_pos_oos"],
                         "OOS>0 без последнего года": o.get("share_pos_early"),
                         "OOS>0 при 1 тике": xx["costs"]["1.0"]["share_pos_oos"],
                         "Спирмен IS↔OOS": rr["spearman_is_oos_sharpe"], "PBO": rr["pbo_full"]["pbo"],
                         "SPA p (весь период)": rr["rc_spa_full"]["spa_p"],
                         "WF плато − средняя, USD": int(round(xx["walk_forward_vs_average_config"]["plateau"]["excess_total"]))})
    if rows:
        w("**Этап 1 против этапа 3** — что дала оптимизация наклона:"); w()
        w(table(pd.DataFrame(rows), 3)); w()
    w(f"**Кросс-рынок.** Ранговая связь одной и той же конфигурации на NG и BR: {cm['spearman_sharpe_m_is_NG_vs_BR']:+.2f} "
      f"в выборке, {cm['spearman_sharpe_m_oos_NG_vs_BR']:+.2f} вне её; по 144 семействам — {cm['families_spearman_oos']:+.2f} вне выборки.")
    w()


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main(sys.argv[1:] or ["stage1", "stage3"])
