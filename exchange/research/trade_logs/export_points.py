"""Сделки по выгоднейшим параметрам каждой стратегии: результат каждого исхода в пунктах.

Пункт — единица биржевой котировки: 1 USD за баррель у BR и 1 USD за MMBtu у NG.
Шаг цены — минимальное изменение котировки: 0.01 у BR и 0.001 у NG. Знак результата:
плюс — прибыльный исход, минус — убыточный. Таблица — на каждую пару
«стратегия × актив»; у портфельных стратегий (4 и 6) — отдельно по ногам NG и BR.

Выгоднейшие параметры:
* семь семейств — комбинация, выбранная оптимизатором (лучший Шарп в выборке),
  из docs/RESULTS_SEVEN_FAMILIES.md; прогоны те же, что в отчёте (отдельный IS и
  отдельный OOS), издержки ~7 б.п. за оборот вычтены;
* скользящие средние — правило R1 (максимальная прибыль в выборке) на финальном
  этапе 3; 1 контракт, без издержек, как в задании, плюс столбец «при 1 шаге
  издержек».

    python export_points.py seven    # окружение пакета exchange; EXCHANGE_MOEX_DATA
    python export_points.py ma       # из журнала сделок csv/MA_3_*_R1.csv
    python export_points.py tables   # points/README.md и points/points.xlsx
    python export_points.py seven-full  # семь семейств, сквозной прогон по всей истории -> points/full/
    python export_points.py sequence # points/sequence/ (и sequence/full/): только числа, шаги цены
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "points"
ROOT = HERE.parents[1]
SPECS = {"BR": {"multiplier": 10.0, "tick": 0.01, "unit": "USD за баррель"},
         "NG": {"multiplier": 100.0, "tick": 0.001, "unit": "USD за MMBtu"}}
COLUMNS = ["№", "период", "вход", "выход", "направление", "объём, ед. актива", "цена входа", "цена выхода",
           "через ролл", "результат, пунктов", "результат, шагов цены", "результат, USD", "итог, USD", "примечание"]


ROUND = {"объём, ед. актива": 2, "цена входа": 4, "цена выхода": 4, "результат, пунктов": 5,
         "результат, шагов цены": 2, "результат, USD": 2, "итог, USD": 2, "итог, пунктов": 5,
         "при издержках 1 шаг, пунктов": 5}


def write(df: pd.DataFrame, columns: list[str], path: Path):
    out = df[columns].copy()
    for col, digits in ROUND.items():
        if col in out and out[col].dtype.kind == "f":
            out[col] = out[col].round(digits)
    out.to_csv(path, index=False)


def finish(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df.insert(0, "№", np.arange(1, len(df) + 1))
    df["результат, шагов цены"] = df["результат, пунктов"] / SPECS[ticker]["tick"]
    df["итог, USD"] = df["результат, USD"].cumsum()
    return df


# ------------------------------------------------------------------ семь семейств
def export_seven(mode: str = "report"):
    """mode="report" — прогоны отчёта (отдельный IS и отдельный OOS) -> points/;
    mode="full" — сквозной прогон по всей истории, период сделки по дате выхода -> points/full/."""
    sys.path.insert(0, str(ROOT / "research" / "seven_families"))
    sys.path.insert(0, str(HERE))
    import families as F
    import harness as H
    import run_all as RA
    from export_trades import seven_cells

    universe = H.load_universe()
    panel_all = {r: p["bars"] for r, p in universe.items()}
    ppys = {r: universe[r]["ppy"] for r in panel_all}
    common = panel_all["NG"].index.intersection(panel_all["BR"].index)
    out_dir = OUT if mode == "report" else OUT / "full"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    for cell in seven_cells().itertuples():
        family, market, params = cell.family, cell.market, cell.params
        is_panel = family in F.PANEL
        index = common if is_panel else panel_all[market].index
        is_idx, oos_idx = RA.split(index)
        rows = {}
        oos_usd = 0.0
        if mode == "report":
            runs = [(lambda _ts, p=p: p, idx, p) for p, idx in (("IS", is_idx), ("OOS", oos_idx))]
        else:
            def label(ts, is_end=is_idx[-1], oos_start=oos_idx[0]):
                return "IS" if ts <= is_end else ("эмбарго" if ts < oos_start else "OOS")
            runs = [(label, index, "FULL")]
        for label_of, idx, period in runs:
            if is_panel:
                panel = {r: b.loc[common].loc[idx] for r, b in panel_all.items()}
                signals = F.PANEL[family](panel, **params)
                legs = [(r, panel[r], RA.positions(panel[r], ppys[r], signals[r], legs=len(panel)))
                        for r in panel]
            else:
                bars = panel_all[market].loc[idx]
                legs = [(market, bars, RA.positions(bars, ppys[market], F.SINGLE[family](bars, **params)))]
            for leg, bars, target in legs:
                result, _ = H.backtest(bars, target, symbol=leg, ppy=ppys[leg])
                quoted = universe[leg]["quoted"]
                # Движок считает на ряду, склеенном по отношению цен (adj = котировка × f).
                # Результат на единицу в пунктах котировки контракта, по которому позиция
                # закрыта: net_pnl / количество / f на дату выхода.
                factor = (bars["open"] / quoted["open"].reindex(bars.index)).astype(float)
                rolls = bars["is_roll"].astype(int).cumsum()
                out = rows.setdefault(leg, [])
                for r in result.trades.itertuples():
                    f_in, f_out = factor[r.entry_time], factor[r.exit_time]
                    rolled = rolls[r.exit_time] != rolls[r.entry_time]
                    # Проверки пересчёта: без ролла коэффициент склейки один и тот же, а цена
                    # выхода отличается от биржевой котировки дня только издержками исполнения.
                    if not rolled and abs(f_in / f_out - 1) > 1e-9:
                        raise SystemExit(f"{cell.id}/{leg}: коэффициент склейки менялся без ролла")
                    exit_q = r.exit_price / f_out
                    if abs(exit_q / quoted["open"][r.exit_time] - 1) > 0.01:
                        raise SystemExit(f"{cell.id}/{leg} {r.exit_time}: цена выхода {exit_q} против "
                                         f"котировки {quoted['open'][r.exit_time]}")
                    out.append({"период": label_of(r.exit_time), "вход": r.entry_time, "выход": r.exit_time,
                                "направление": "long" if r.direction > 0 else "short",
                                "объём, ед. актива": r.quantity * f_out,
                                "цена входа": r.entry_price / f_in, "цена выхода": exit_q,
                                "через ролл": "да" if rolled else "",
                                "результат, пунктов": r.net_pnl / r.quantity / f_out,
                                "результат, USD": r.net_pnl,
                                "примечание": ""})
                change = float(result.equity.iloc[-1] - H.CAPITAL)
                residual = change - float(result.trades["net_pnl"].sum() if len(result.trades) else 0.0)
                pos = float(result.position.iloc[-1])
                if pos != 0 or abs(residual) > 0.005:
                    f_last = float(bars["close"].iloc[-1] / quoted["close"].reindex(bars.index).iloc[-1])
                    out.append({"период": label_of(bars.index[-1]), "вход": pd.NaT, "выход": bars.index[-1],
                                "направление": "long" if pos > 0 else "short",
                                "объём, ед. актива": abs(pos) * f_last,
                                "цена входа": np.nan, "цена выхода": float(quoted["close"].iloc[-1]),
                                "через ролл": "",
                                "результат, пунктов": residual / abs(pos) / f_last if pos else 0.0,
                                "результат, USD": residual,
                                "примечание": "позиция открыта на конец периода: переоценка по закрытию"})
                if period == "OOS":
                    oos_usd += change
        got = oos_usd / H.CAPITAL
        if mode == "report" and abs(got - cell.oos_return_report) > 0.0006:
            raise SystemExit(f"{cell.id}: OOS {got:.4%} против отчёта {cell.oos_return_report:.1%}")
        for leg, out in rows.items():
            order = ["период", "выход"] if mode == "report" else ["выход"]
            df = pd.DataFrame(out).sort_values(order, kind="stable")
            for col in ("вход", "выход"):
                df[col] = pd.to_datetime(df[col]).dt.tz_localize(None).dt.date
            df = finish(df, leg)
            # Без ролла пункты — разность цен исполнения минус комиссия на единицу.
            plain = (df["через ролл"] == "") & df["цена входа"].notna()
            sign = np.where(df["направление"] == "long", 1.0, -1.0)
            move = (df["цена выхода"] - df["цена входа"]) * sign
            gap = (move - df["результат, пунктов"])[plain] / df["цена входа"][plain]
            usd = df["объём, ед. актива"] * df["результат, пунктов"]
            if (usd - df["результат, USD"]).abs().max() > 1e-6 * df["результат, USD"].abs().max():
                raise SystemExit(f"{cell.id}/{leg}: объём × пункты не равны результату в USD")
            if len(gap) and (gap.min() < -1e-9 or gap.max() > 0.002):
                raise SystemExit(f"{cell.id}/{leg}: комиссия на единицу вне ожидаемого: {gap.min()}..{gap.max()}")
            sid = f"{cell.id}_{leg}" if is_panel else cell.id
            write(df, COLUMNS, out_dir / f"{sid}.csv")
            meta.append({"id": sid, "study": "seven", "strategy": cell.ячейка.split(" · ")[0],
                         "asset": leg, "params": cell.лучшая,
                         "selection": "лучший Шарп в выборке" + ("" if mode == "report" else
                                                                 "; сквозной прогон по всей истории"),
                         "costs": "издержки ~7 б.п. за оборот вычтены"})
            print(f"{sid:26} сделок {len(df):4d} | пунктов {df['результат, пунктов'].sum():+9.3f} | "
                  f"USD {df['результат, USD'].sum():+12,.0f} | комиссия на единицу ≤ {gap.max() if len(gap) else 0:.4%} цены")
    (out_dir / "seven_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


# ------------------------------------------------------------------ скользящие средние
def export_ma():
    OUT.mkdir(exist_ok=True)
    meta = []
    fin_all = {t: json.loads((ROOT / "research" / "ma_crossover_5m" / "results" / "stage3" / t /
                              "finalists.json").read_text()) for t in SPECS}
    for t in ("NG", "BR"):
        src = pd.read_csv(HERE / "csv" / f"MA_3_{t}_R1.csv")
        f = next(x for x in fin_all[t] if x["scope"] == "global" and x["rule"] == "R1_max_pnl")
        spec = SPECS[t]
        df = pd.DataFrame({"период": src["период"], "вход": src["вход"], "выход": src["выход"],
                           "направление": src["направление"], "объём, ед. актива": spec["multiplier"],
                           "цена входа": src["котировка входа"],
                           "цена выхода": src["котировка выхода"], "через ролл": src["через ролл"].fillna(""),
                           "результат, пунктов": src["результат, USD"] / spec["multiplier"],
                           "результат, USD": src["результат, USD"], "примечание": src["примечание"].fillna("")})
        df = finish(df, t)
        df["результат, шагов цены"] = df["результат, шагов цены"].round().astype(int)
        df["итог, пунктов"] = df["результат, пунктов"].cumsum()
        df["при издержках 1 шаг, пунктов"] = df["результат, пунктов"] - spec["tick"]
        if abs(df["результат, USD"].sum() - f["full"]["pnl"]) > 0.05:
            raise SystemExit(f"{t}: сумма {df['результат, USD'].sum()} против финалиста {f['full']['pnl']}")
        sid = f"MA_3_{t}_R1"
        for col in ("вход", "выход"):
            df[col] = pd.to_datetime(df[col]).dt.strftime("%Y-%m-%d %H:%M")
        write(df, [*COLUMNS[:-1], "итог, пунктов", "при издержках 1 шаг, пунктов", "примечание"],
              OUT / f"{sid}.csv")
        c = f["config"]
        meta.append({"id": sid, "study": "ma", "strategy": "Пересечение скользящих средних, 5 мин",
                     "asset": t, "params": f"{c['fast_type']}{c['fast_period']} × {c['slow_type']}{c['slow_period']}, "
                     f"наклон {c['method']}, выход {c['exit']}", "selection": "R1: максимальная прибыль в выборке, этап 3",
                     "costs": "без издержек (условие задачи); отдельный столбец — при 1 шаге цены на сделку"})
        print(f"{sid:14} сделок {len(df):5d} | пунктов {df['результат, пунктов'].sum():+8.3f} | "
              f"шагов {df['результат, шагов цены'].sum():+7d} | USD {df['результат, USD'].sum():+9.1f}")
    (OUT / "ma_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


# ------------------------------------------------------------------ только числа
def export_sequence(mode: str = "report"):
    """По файлу на стратегию и актив: исходы по порядку, целым числом шагов цены, без заголовка.

    Пункт здесь — минимальный шаг цены (BR 0.01 USD, NG 0.001 USD): плюс — прибыль, минус —
    убыток, 0 — выход по цене входа. Строки переоценки открытой позиции на конец периода
    сделками не являются и не включаются.
    """
    src = OUT if mode == "report" else OUT / "full"
    seq = OUT / "sequence" if mode == "report" else OUT / "sequence" / "full"
    seq.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "seven_meta.json").read_text())
    if mode == "report":
        meta += json.loads((OUT / "ma_meta.json").read_text())
    lines = []
    for m in meta:
        df = pd.read_csv(src / f"{m['id']}.csv")
        steps = df.loc[df["вход"].notna(), "результат, шагов цены"].to_numpy(float)
        whole = (np.sign(steps) * np.floor(np.abs(steps) + 0.5)).astype(int)
        if m["study"] == "ma" and not np.array_equal(whole, np.round(steps).astype(int)):
            raise SystemExit(f"{m['id']}: у скользящих средних шаги должны быть целыми")
        if np.abs(whole - steps).max() > 0.5 + 1e-9:
            raise SystemExit(f"{m['id']}: округление больше полшага")
        (seq / f"{m['id']}.csv").write_text("\n".join(str(v) for v in whole) + "\n")
        lines.append((m, len(whole), int((whole > 0).sum()), int((whole < 0).sum()), int((whole == 0).sum()),
                      int(whole.sum())))
        print(f"{m['id']:26} исходов {len(whole):5d} | + {lines[-1][2]:5d} | − {lines[-1][3]:5d} | 0 {lines[-1][4]:3d}")
    readme = [
        "# Исходы сделок: только числа", "",
        "По файлу на стратегию и актив: результат каждой сделки по порядку, одно число в строке, без "
        "заголовка. **Пункт — минимальный шаг цены**: у BR 0.01 USD за баррель, у NG 0.001 USD за MMBtu. "
        "Плюс — прибыльный исход, минус — убыточный, 0 — выход по цене входа.", "",
        ("Параметры — выгоднейшие по оптимизации: у семи семейств выбранные оптимизатором (лучший Шарп в "
         "выборке), у скользящих средних — R1, максимальная прибыль в выборке (этап 3). У семи семейств "
         "издержки вычтены и результат дан на единицу позиции, округлён до целого шага; у скользящих средних — "
         "1 контракт без издержек, шаги целые точно. Подробные таблицы с датами, ценами и USD — в "
         "[`../`](../README.md).") if mode == "report" else
        ("Параметры — выбранные оптимизатором (лучший Шарп в выборке). Издержки вычтены, результат дан на "
         "единицу позиции и округлён до целого шага. Подробные таблицы с датами, ценами и USD — в "
         "[`../../full/`](../../full/README.md)."), "",
        *([] if mode == "report" else [
            "**Сквозной прогон**: семь семейств торгуют всю историю одним прогоном, без разрыва между "
            "выборкой и проверкой, поэтому здесь все исходы подряд с первого до последнего дня данных. "
            "Исходы прогонов отчёта (отдельный IS и отдельный OOS) — в [`../`](../README.md).", ""]),
        "| Стратегия | Актив | Параметры | Исходов | Плюс | Минус | Ноль | Файл |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m, n, pos, neg, zero, _ in lines:
        readme.append(f"| {m['strategy']} | {m['asset']} | {m['params']} | {n} | {pos} | {neg} | {zero} "
                      f"| [{m['id']}.csv]({m['id']}.csv) |")
    (seq / "README.md").write_text("\n".join(readme) + "\n")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "seven":
        export_seven()
    elif step == "ma":
        export_ma()
    elif step == "seven-full":
        export_seven("full")
    elif step == "sequence":
        export_sequence()
        export_sequence("full")
    elif step == "tables":
        import build_points
        build_points.main()
        build_points.main(OUT / "full")
    else:
        raise SystemExit(__doc__)
