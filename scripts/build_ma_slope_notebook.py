"""Build notebooks/06_ma_slope_research.ipynb (the MA cross + slope study viewer).

    python scripts/build_ma_slope_notebook.py
    python scripts/run_notebook.py notebooks/06_ma_slope_research.ipynb

The notebook reads only files committed to the repository: the study tables,
figures and compact full grids in ``research/ma_cross_slope_5m`` and the
AlgoPack candles (for re-running any config through vectorbt).
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "06_ma_slope_research.ipynb"

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# 06 — Пересечение цены с быстрой MA + наклон медленной MA (5m, фьючерсы MOEX)

Ноутбук-просмотрщик масштабного исследования: ~9 млн конфигураций на инструмент (все 36 пар типов
средних, Fast 2–99 × Slow 5–250, два метода наклона, две модели выхода, подбор параметров наклона) на
NG, BR, WTI, GOLD, SILV, COPPER, PLT, PLD. Полный текстовый отчёт —
[`research/ma_cross_slope_5m/README.md`](../research/ma_cross_slope_5m/README.md).

Всё читается из файлов репозитория, пересчёт сеток не нужен:

* `research/ma_cross_slope_5m/tables/` — сводные таблицы, рейтинг, финалисты, walk-forward, PBO, межрыночная
  проверка;
* `research/ma_cross_slope_5m/grid/` — **все конфигурации** полных сеток (этапы 1 и 3) и решётки подбора
  наклона (этап 2) с метриками IS/OOS;
* `research/ma_cross_slope_5m/figures/` — рисунки отчёта;
* `data/raw/moex_algopack/` — свечи для пересчёта любой конфигурации через VectorBT (раздел 9).

Термины: **IS** — первые 70 % баров инструмента, **OOS** — последние 30 %; **t** — t-статистика средней
сделки; **плато** — средний IS t по соседним периодам (Fast ±2, Slow ±5); выход **reverse_cross** —
обратное пересечение Fast MA, **slope_flip** — смена наклона Slow MA на противоположный; метод наклона
**pct** — изменение MA в %, **atr** — в единицах ATR.
""")

code("""
import json
import tempfile
import warnings
from io import BytesIO
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from IPython.display import Image, Markdown, display
from PIL import Image as PILImage

from src.ma_slope import analysis as A
from src.ma_slope import finalist as F
from src.ma_slope import plots as P
from src.ma_slope.averages import MA_TYPES
from src.ma_slope.data import TICK_SIZES, TICKERS, load_instrument
from src.ma_slope.grid import EXIT_MODELS, Slope
from src.utils.paths import PROJECT_ROOT

STUDY = PROJECT_ROOT / "research" / "ma_cross_slope_5m"
TABLES, FIGS, GRID = STUDY / "tables", STUDY / "figures", STUDY / "grid"
INF = ["ng", "br", "gold", "silv", "plt", "pld", "copper"]  # WTI: 2 181 bars, descriptive only
NAMES = {"ng": "NG", "br": "BR (Brent)", "wti": "WTI (CL)", "gold": "GOLD (GC)", "silv": "SILV (SI)",
         "copper": "COPPER (HG)", "plt": "PLT (PL)", "pld": "PLD (PA)"}
TMP = Path(tempfile.mkdtemp())

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", "{:.3f}".format)


def show(name, width=1300, folder=FIGS):
    \"\"\"Display a figure, downscaled to a 256-colour palette so the executed notebook stays light.\"\"\"
    img = PILImage.open(Path(folder) / name).convert("RGB")
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), PILImage.LANCZOS)
    buf = BytesIO()
    img.quantize(colors=256, method=PILImage.Quantize.MEDIANCUT).save(buf, format="PNG", optimize=True)
    display(Image(data=buf.getvalue()))


def table(name, **kw):
    return pd.read_csv(TABLES / name, **kw)


def grid_meta(stage, ticker):
    return json.loads(pq.read_schema(GRID / f"{stage}_{ticker}.parquet").metadata[b"ma_slope"])


def load_full_grid(ticker, stage="stage3"):
    \"\"\"All 3 471 552 configs of one instrument's full grid with IS/OOS metrics.\"\"\"
    return pd.read_parquet(GRID / f"{stage}_{ticker}.parquet")


summary = json.loads((TABLES / "summary.json").read_text())
print("Файлы сеток:", sorted(p.name for p in GRID.glob("*.parquet")))
""")

md("""
## 1. Главное

* **Устойчивого торгового преимущества у конкретных параметров не найдено.** Ни одна конфигурация не проходит
  поправку на множественное тестирование; лучшие по IS на независимой выборке в среднем не лучше случайной
  конфигурации из той же сетки (кроме PLT и PLD).
* **Результат определяет режим рынка.** Одни и те же правила зарабатывают или теряют почти всей сеткой сразу
  (золото и серебро: минус на IS, плюс на OOS; PLT и PLD — наоборот).
* **Выход по смене наклона (slope_flip) лучше обратного пересечения** на IS на всех 7 рынках и на OOS на 6 из 7.
* **Горизонт наклона k = 40–80 баров (3–7 часов)** — главный параметр фильтра; ширина нейтральной зоны и метод
  (pct/atr) почти не важны.
* **Типы средних — эффект второго порядка**: медленная EMA/SSMA/WMA немного лучше SMA/TMA/VWMA.
* **Walk-forward с ежегодным переотбором** даёт плюс на всех 6 рынках с длинной историей (значимо на NG и PLT после
  поправки Холма), но заметная часть — общий трендовый дрейф; средняя OOS-сделка на NG, BR, SILV — 0,5–1,8 тика.

Цифры ниже — из таблиц исследования.
""")

md("## 2. Данные")
code("""
rows = []
for t in TICKERS:
    m = grid_meta("stage3", t)
    rows.append({"инструмент": NAMES[t], "5m баров": m["bars"], "начало": m["start"][:10], "конец": m["end"][:10],
                 "граница IS/OOS": m["split_time"][:10], "USD за 1 пункт": m["multiplier"], "тик": TICK_SIZES[t]})
pd.DataFrame(rows)
""")
code("""
print("Конфигураций на инструмент (этапы 1 + 2 + 3):", summary["trials_per_instrument"])
print("Ожидаемый максимум t среди стольких нулевых тестов:",
      {t: round(v, 2) for t, v in summary["expected_max_t_null"].items()})
""")

md("""
## 3. Методология в двух абзацах

Сигнал — пересечение Close и Fast MA по двум закрытым свечам, исполнение по Open следующей свечи; long при наклоне
Slow MA вверх или нейтрально, short — вниз или нейтрально; 1 контракт, без издержек, без разворота на свече выхода.
Наклон за k баров: `pct` — `(MA[t]−MA[t−k])/MA[t−k]·100`, `atr` — `(MA[t]−MA[t−k])/ATR[t]`, нейтрально при
|наклон| ≤ ε. Формулы шести средних (SMA, EMA, WMA, SSMA, VWMA, TMA) и способы инициализации — в отчёте, раздел 3.1.

Этап 1 — полная сетка (3 471 552 конфигурации) при заранее заданном наклоне (k = 10, ε = 0,02 % / 0,25·ATR14);
этап 2 — 128 настроек наклона на логарифмической решётке периодов; этап 3 — снова полная сетка с выбранным на IS
наклоном. Отбор — по плато IS t (не по прибыли), проверка — OOS, walk-forward, PBO, Monte Carlo, поправки Холма и
Benjamini–Hochberg, межрыночная проверка. PnL — в USD на контракт по изменению сырой цены удерживаемого контракта,
ролл — по цене перекрытия.
""")

md("## 4. Вся сетка: режим рынка важнее параметров")
code("""
s = table("grid_summary.csv")
s = s[(s["stage"] == "stage3") & s["ticker"].isin(INF)]


def weighted(g, col):
    return (g[col] * g["configs"]).sum() / g["configs"].sum()


rows = []
for (t, model), g in s.groupby(["ticker", "model"]):
    rows.append({"инструмент": NAMES[t], "выход": model, "IS t медиана": weighted(g, "is_t_median"),
                 "IS доля t>0": weighted(g, "is_share_pos"), "OOS t медиана": weighted(g, "oos_t_median"),
                 "OOS доля t>0": weighted(g, "oos_share_pos"), "ρ(IS,OOS)": weighted(g, "spearman_is_oos")})
pd.DataFrame(rows).sort_values(["инструмент", "выход"]).reset_index(drop=True)
""")
code('show("is_vs_oos.png")')

md("## 5. Сравнение 36 сочетаний типов средних")
code('show("type_pairs_oos_median_t.png")')
code('show("type_pairs_best_oos_t.png")')
code("""
tp = table("type_pairs.csv")
tp = tp[(tp["stage"] == "stage3") & tp["ticker"].isin(INF)]
pairs = (tp.groupby(["fast_type", "slow_type"])
           .agg(OOS_t_медиана=("oos_t_median", "mean"), IS_t_медиана=("is_t_median", "mean"),
                OOS_t_лучшей_по_IS=("best_oos_t", "mean"), рынков_с_плюсом=("best_oos_t", lambda x: int((x > 0).sum())))
           .sort_values("OOS_t_медиана", ascending=False))
display(Markdown("**Средние по 7 рынкам, лучшие и худшие сочетания (fast → slow):**"))
pd.concat([pairs.head(8), pairs.tail(5)])
""")
code("""
display(pd.DataFrame({"быстрая (главный эффект)": tp.groupby("fast_type")["oos_t_median"].mean(),
                      "медленная (главный эффект)": tp.groupby("slow_type")["oos_t_median"].mean()}).loc[list(MA_TYPES)])
""")

md("## 6. Модели выхода")
code('show("exit_models_is.png")')
code('show("exit_models_oos.png")')

md("## 7. Методы определения наклона и подбор параметров")
code("""
rows = []
for (t, method), g in s.groupby(["ticker", "method"]):
    rows.append({"инструмент": NAMES[t], "метод": method, "OOS t медиана": weighted(g, "oos_t_median"),
                 "OOS доля t>0": weighted(g, "oos_share_pos")})
pd.DataFrame(rows).pivot(index="инструмент", columns="метод")
""")
code("""
sw = table("slope_sweep.csv")
sw = sw[sw["ticker"].isin(INF)]
display(Markdown("**Медианный IS t по всем рынкам и решётке: горизонт k против порога ε (метод pct)**"))
display(sw.groupby("k")["is_median_t"].mean().to_frame("IS t медиана").T)
display(sw[sw["method"] == "pct"].groupby("eps")["is_median_t"].mean().to_frame("IS t медиана").T)
chosen = []
for (t, method, model), g in sw.groupby(["ticker", "method", "model"]):
    b = g.loc[g["is_median_t"].idxmax()]
    base = g[(g["k"] == 10) & ((g["eps"] == 0.02) if method == "pct" else ((g["eps"] == 0.25) & (g["atr_n"] == 14)))].iloc[0]
    chosen.append({"инструмент": NAMES[t], "метод": method, "выход": EXIT_MODELS[int(model)],
                   "выбор на IS": f"k={int(b['k'])}, eps={b['eps']:g}" + (f", ATR{int(b['atr_n'])}" if method == "atr" else "%"),
                   "IS t: выбор": b["is_median_t"], "IS t: база": base["is_median_t"],
                   "OOS t: выбор": b["oos_median_t"], "OOS t: база": base["oos_median_t"]})
pd.DataFrame(chosen)
""")
code('show("slope_sweep_pct_slope_flip.png")')
code('show("slope_sweep_atr_slope_flip.png")')

md("## 8. Категория Fast ≥ Slow")
code("""
rows = []
for (t, cat), g in s.groupby(["ticker", "category"]):
    rows.append({"инструмент": NAMES[t], "категория": cat, "IS t медиана": weighted(g, "is_t_median"),
                 "OOS t медиана": weighted(g, "oos_t_median")})
pd.DataFrame(rows).pivot(index="инструмент", columns="категория")
""")

md("""
## 9. Рейтинг, лучшие параметры, независимая выборка

Лучшая по IS-плато конфигурация каждого инструмента (этап 3) с метриками IS и OOS, Sharpe/Sortino дневного
PnL одного контракта (VectorBT), Monte Carlo и поправкой Холма:
""")
code("""
fz = table("finalists.csv")
f3 = fz[(fz["stage"] == "stage3") & (fz["tag"] == "overall")].set_index("ticker").loc[INF + ["wti"]]
cols = ["fast_type", "fast_period", "slow_type", "slow_period", "slope", "model", "category", "is_trades", "is_t",
        "is_plateau_t", "is_pnl", "oos_trades", "oos_pnl", "oos_profit_factor", "oos_win_rate", "oos_t", "oos_p",
        "oos_p_holm", "oos_avg_trade_ticks", "oos_avg_hours", "oos_max_dd", "oos_max_loss_streak", "is_sharpe",
        "oos_sharpe", "oos_sortino", "mc_oos_p_loss", "mc_oos_max_dd_p95"]
f3[cols]
""")
code('show("finalists_is_vs_oos.png")')
code("""
display(Markdown("**Этап 1 (наклон k = 10) против этапа 3 (наклон подобран на IS): OOS t финалиста**"))
fz[(fz["tag"] == "overall") & fz["ticker"].isin(INF)].pivot(index="ticker", columns="stage", values="oos_t").loc[INF]
""")
code("""
rk = table("ranking_group_bests.csv")
rk3 = rk[(rk["stage"] == "stage3") & rk["ticker"].isin(INF)]
print(f"Лучших конфигураций групп: {len(rk3)}, с OOS t > 0: {(rk3['oos_t'] > 0).sum()}, "
      f"p < 0.05 без поправки: {(rk3['oos_p'] < 0.05).sum()}, q(BH) < 0.05: {(rk3['oos_q_bh'] < 0.05).sum()}, "
      f"минимальный q: {rk3['oos_q_bh'].min():.3f}")
show_cols = ["ticker", "fast_type", "fast_period", "slow_type", "slow_period", "method", "model", "is_plateau_t",
             "is_t", "oos_trades", "oos_t", "oos_pnl", "oos_profit_factor", "oos_p", "oos_q_bh"]
display(Markdown("**Рейтинг: по 3 лучших группы на инструмент (по IS-плато), с результатом на OOS**"))
rk3.sort_values("is_plateau_t", ascending=False).groupby("ticker").head(3).sort_values(["ticker", "is_plateau_t"],
                                                                                       ascending=[True, False])[show_cols]
""")
code("""
display(Markdown("**Лучшие на OOS среди 990 групп (после отбора по IS) — для информации, q учитывает множественность**"))
rk3.sort_values("oos_q_bh").head(10)[show_cols]
""")

md("### Карты Fast × Slow и устойчивости (слой лучшей конфигурации)")
code("""
for t in INF:
    show(f"heatmap_stage3_{t}.png")
""")

md("### Кривые капитала, распределения сделок, результаты по месяцам и годам")
code('show("equity_finalists.png")')
code('show("trade_distributions.png")')
code("""
yearly = table("finalist_yearly_pnl.csv").pivot(index="year", columns="ticker", values="pnl")
yearly[[t for t in INF + ["wti"] if t in yearly.columns]].round(0)
""")
code("""
monthly = table("finalist_monthly_pnl.csv")
display(Markdown("**PnL по месяцам, PLD (остальные инструменты — в `tables/finalist_monthly_pnl.csv`)**"))
monthly[monthly["ticker"] == "pld"].drop(columns="ticker").set_index("year").round(0)
""")

md("## 10. Устойчивость: walk-forward, польза отбора, PBO, Monte Carlo")
code("""
wf = table("walk_forward_summary.csv")
wf1 = wf[(wf["stage"] == "stage1") & (wf["scope"] == "all")].set_index("ticker")
wf1["p_holm"] = A.holm(wf1["pooled_p"])
wf1
""")
code('show("walk_forward_stage1.png")')
code("""
sk = table("selection_skill.csv")
w = sk[sk["kind"] == "walk_forward"]
display(w.groupby(["stage", "ticker"])[["selected_percentile", "selected_t", "median_config_t"]].mean().unstack(0))
for st in ("stage1", "stage3"):
    v = w[w["stage"] == st]["selected_percentile"].dropna()
    print(f"{st}: средний перцентиль выбора {v.mean():.3f} по {len(v)} тестовым годам, "
          f"z = {(v.mean() - 0.5) / (np.sqrt(1 / 12) / np.sqrt(len(v))):.2f}")
""")
code('show("selection_skill.png")')
code("""
display(table("pbo.csv").pivot(index="ticker", columns="stage", values="pbo"))
show("pbo_logits.png")
""")

md("## 11. Межрыночная проверка (этап 1, одинаковые правила)")
code("""
cm = summary["cross_market"]
print("Конфигураций, валидных на всех рынках:", cm["configs_common"])
print("Доля с OOS t > 0:", {k: round(v, 2) for k, v in cm["positive_rate"].items()})
print("Положительны на OOS везде:", cm["all_positive_oos"], "при ожидании для независимых рынков",
      round(cm["all_positive_oos_expected"]))
print("Положительны на IS везде:", cm["is_positive_everywhere"], "-> их доля с OOS t > 0:",
      {k: round(v, 2) for k, v in cm["oos_positive_rate_given_is_positive_everywhere"].items()})
display(Markdown("**Честный отбор только по IS (худший рынок по IS-плато) и результат на OOS**"))
table("cross_market_is_selected.csv").head(10)
""")
code('show("cross_market_transfer.png")')
code('show("cross_market_counts.png")')

md("""
## 12. Исследователь: любая конфигурация

Полные сетки лежат в `research/ma_cross_slope_5m/grid/`: `stage3_<тикер>.parquet` и `stage1_<тикер>.parquet` —
все 3 471 552 конфигурации инструмента (сделки, PnL, t, Profit Factor, win rate, просадка на IS и OOS, IS-плато);
`stage2_<тикер>.parquet` — решётка периодов под каждой из 128 настроек наклона. Ниже — выборка из сетки, карта
Fast × Slow для любого слоя и пересчёт любой конфигурации на свечах через VectorBT.
""")
code("""
grid = load_full_grid("pld")
print(grid.shape)
q = grid[(grid["model"] == "slope_flip") & (grid["is_trades"] >= 100)]
q.sort_values("is_plateau_t", ascending=False).head(10)
""")
code("""
def slice_maps(grid, ticker, fast_type, slow_type, method, model, stage="stage3"):
    \"\"\"IS t, IS plateau and OOS t over Fast x Slow for one types/method/model layer.\"\"\"
    g = grid[(grid["fast_type"] == fast_type) & (grid["slow_type"] == slow_type)
             & (grid["method"] == method) & (grid["model"] == model)]
    piv = {c: g.pivot(index="fast_period", columns="slow_period", values=c) for c in ("is_t", "is_plateau_t", "oos_t")}
    best = g.loc[g["is_plateau_t"].idxmax()]
    path = TMP / f"slice_{ticker}_{fast_type}_{slow_type}_{method}_{model}.png"
    P.period_heatmaps([("IS t", piv["is_t"].to_numpy()), ("IS plateau t", piv["is_plateau_t"].to_numpy()),
                       ("OOS t", piv["oos_t"].to_numpy())],
                      list(piv["is_t"].index), list(piv["is_t"].columns),
                      (best["fast_period"], best["slow_period"]),
                      f"{NAMES[ticker]}, {stage}: {fast_type} fast x {slow_type} slow, {method}, {model}", path)
    show(path.name, folder=TMP)


slice_maps(grid, "pld", "EMA", "SMA", "atr", "reverse_cross")
""")
code("""
def inspect(ticker, fast_type, fast_period, slow_type, slow_period, method, model, stage="stage3"):
    \"\"\"Re-run one config on the 5m candles: trades from the kernel, equity and stats from vectorbt.\"\"\"
    slope = Slope(**grid_meta(stage, ticker)["slopes"][f"{method}/{model}"])
    cfg = F.Config(fast_type, fast_period, slow_type, slow_period, slope, EXIT_MODELS.index(model))
    ins = load_instrument(ticker)
    tr = F.trades(ins, cfg)
    pf = F.portfolio(ins, cfg)
    daily = F.daily_pnl(pf)
    days = pd.to_datetime(daily.index)
    rows = {}
    for name, part in (("IS", tr[tr["sample"] == "is"]), ("OOS", tr[tr["sample"] == "oos"]), ("все", tr)):
        pnl = part["pnl"].to_numpy()
        n = len(pnl)
        rows[name] = {"сделок": n, "PnL, $": pnl.sum(), "win %": 100 * np.mean(pnl > 0) if n else np.nan,
                      "PF": pnl[pnl > 0].sum() / -pnl[pnl < 0].sum() if (pnl < 0).any() else np.nan,
                      "t": float(A.t_stat(n, pnl.sum(), (pnl ** 2).sum())), "ср. часов": part["hours"].mean()}
    for name, part in (("IS", daily[days < ins.split_time.normalize()]),
                       ("OOS", daily[days >= ins.split_time.normalize()]), ("все", daily)):
        rows[name].update(F.risk_metrics(part))
    closed = pf.trades.records_readable
    closed = closed[closed["Status"] == "Closed"]
    same = len(closed) == len(tr) and np.allclose(closed["PnL"].to_numpy(), tr["pnl"].to_numpy())
    display(Markdown(f"**{NAMES[ticker]}: {cfg.label}** — сделки VectorBT совпадают с ядром: {same}"))
    display(pd.DataFrame(rows).T)
    eq = daily.cumsum()
    path = TMP / f"equity_{ticker}.png"
    P.small_multiples_lines({NAMES[ticker]: [("MTM PnL, USD/contract", pd.to_datetime(eq.index), eq.to_numpy())]},
                            cfg.label, "USD", path, vlines={NAMES[ticker]: ins.split_time}, ncols=1)
    show(path.name, folder=TMP, width=900)
    monthly, _ = F.calendar(tr)
    display(monthly.round(0))
    return tr, pf


tr, pf = inspect("pld", "WMA", 73, "WMA", 50, "pct", "slope_flip")
""")
code("""
tr, pf = inspect("ng", "EMA", 9, "SMA", 100, "atr", "reverse_cross")
""")

md("""
## 13. Выводы

1. Устойчивого преимущества у конкретных параметров нет: после поправок на множественное тестирование ни одна
   конфигурация не подтверждается на OOS, а лучшие по IS в среднем не лучше случайной на NG, BR, GOLD и SILV.
2. Воспроизводимые закономерности касаются семейств правил: удержание до разворота наклона лучше быстрого выхода;
   горизонт наклона 40–80 баров лучше коротких; нейтральная зона и метод её определения почти не важны; медленная
   EMA/SSMA/WMA немного лучше SMA/TMA/VWMA.
3. Знак результата задаёт режим рынка: без издержек это преимущество трендовой экспозиции, а не параметров.
4. Самый сильный результат — ежегодный переотбор (walk-forward): плюс на всех 6 рынках с длинной историей, значимо на
   NG и PLT; но средняя сделка на NG, BR, SILV — 0,5–2 тика, поэтому без учёта комиссий и спреда выводы о
   торгуемости делать нельзя.

Воспроизведение всех этапов — раздел 7 отчёта (`research/ma_cross_slope_5m/README.md`).
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, OUT)
print(f"written {OUT}")
