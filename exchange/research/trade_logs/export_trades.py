"""Журнал сделок: каждая сделка отобранных стратегий по порядку и её результат.

Три шага, потому что части живут в разных окружениях:

    python export_trades.py seven   # семь семейств: пересчёт через движок пакета (нужен пакет exchange)
    python export_trades.py ma      # скользящие средние: сделки VectorBT из рабочего каталога
    python export_trades.py excel   # trade_log.xlsx и README.md из CSV

Сделки не отбираются заново: семь семейств — победители из
docs/RESULTS_SEVEN_FAMILIES.md, скользящие средние — глобальные отборы R1–R3
из research/ma_crossover_5m/results/<этап>/<рынок>/finalists.json. Каждый
шаг сверяет сумму сделок с итогом, опубликованным в отчёте, и падает при
расхождении.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                   # каталог exchange/
CSV = HERE / "csv"
SEVEN_DOC = ROOT / "docs" / "RESULTS_SEVEN_FAMILIES.md"
MA = ROOT / "research" / "ma_crossover_5m"
MA_WORK = Path(os.environ.get("MA5M_WORK", "work"))

COLUMNS = ["№", "период", "нога", "вход", "выход", "направление", "объём при входе, USD",
           "котировка входа", "котировка выхода", "через ролл", "издержки, USD", "результат, USD",
           "баров в позиции", "примечание"]

FAMILY_KEYS = {"1": "1_tsmom", "2": "2_dual_ma", "3": "3_breakout", "4": "4_cross_sectional",
               "5": "5_carry", "6": "6_portfolio_ma", "7": "7_mom_reversal"}
SHORT = {"1": "TSMOM", "2": "DualMA", "3": "Breakout", "4": "CrossSect",
         "5": "Carry", "6": "PortfolioMA", "7": "MomRev"}


def md_table(text: str, column: str) -> pd.DataFrame:
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("| ячейка |") and f"| {column} |" in ln)
    rows = []
    for ln in lines[start:]:
        if not ln.startswith("|"):
            break
        rows.append([c.strip() for c in ln.strip("|").split("|")])
    return pd.DataFrame(rows[2:], columns=rows[0])


def parse_params(text: str) -> dict:
    out = {}
    for part in text.split(","):
        k, v = part.split("=")
        out[k] = float(v) if "." in v else int(v)
    return out


def seven_cells() -> pd.DataFrame:
    text = SEVEN_DOC.read_text()
    a = md_table(text, "конец IS")[["ячейка", "лучшая"]]
    b = md_table(text, "итог. дох.")[["ячейка", "итог. дох.", "сделок"]]
    df = a.merge(b, on="ячейка")
    num = df["ячейка"].str.split(" ").str[0]
    df["family"] = num.map(FAMILY_KEYS)
    df["market"] = df["ячейка"].str.split(" · ").str[-1]
    df["id"] = "7F_" + num + "_" + num.map(SHORT) + "_" + df["market"].str.replace("+", "", regex=False)
    df["params"] = df["лучшая"].map(parse_params)
    df["oos_return_report"] = df["итог. дох."].str.rstrip("%").astype(float) / 100
    return df


# ------------------------------------------------------------------ семь семейств
def export_seven():
    sys.path.insert(0, str(ROOT / "research" / "seven_families"))
    import families as F
    import harness as H
    import run_all as RA

    universe = H.load_universe()
    panel_all = {r: p["bars"] for r, p in universe.items()}
    ppys = {r: universe[r]["ppy"] for r in panel_all}
    common = panel_all["NG"].index.intersection(panel_all["BR"].index)
    CSV.mkdir(exist_ok=True)
    meta = []
    for cell in seven_cells().itertuples():
        family, market, params = cell.family, cell.market, cell.params
        is_panel = family in F.PANEL
        index = common if is_panel else panel_all[market].index
        is_idx, oos_idx = RA.split(index)
        rows, totals = [], {}
        for period, idx in (("IS", is_idx), ("OOS", oos_idx)):
            # Ровно те прогоны, по которым считался отчёт: IS-прогон для отбора и
            # отдельный OOS-прогон, начинающийся с пустой позиции (run_all.evaluate_cell).
            if is_panel:
                panel = {r: b.loc[common].loc[idx] for r, b in panel_all.items()}
                signals = F.PANEL[family](panel, **params)
                legs = [(r, panel[r], RA.positions(panel[r], ppys[r], signals[r], legs=len(panel)))
                        for r in panel]
            else:
                bars = panel_all[market].loc[idx]
                legs = [(market, bars, RA.positions(bars, ppys[market], F.SINGLE[family](bars, **params)))]
            period_pnl = 0.0
            for leg, bars, target in legs:
                result, _ = H.backtest(bars, target, symbol=leg, ppy=ppys[leg])
                t = result.trades
                # Движок считает на ряду, склеенном по отношению цен: его цены в прошлом
                # не равны биржевым. В таблицу идут котировки Open дня исполнения
                # и объём позиции в долларах (количество × расчётная цена).
                quoted = universe[leg]["quoted"]
                rolls = bars["is_roll"].astype(int).cumsum()
                for r in t.itertuples():
                    rows.append({"период": period, "нога": leg, "вход": r.entry_time, "выход": r.exit_time,
                                 "направление": "long" if r.direction > 0 else "short",
                                 "объём при входе, USD": r.quantity * r.entry_price,
                                 "котировка входа": quoted["open"].get(r.entry_time, np.nan),
                                 "котировка выхода": quoted["open"].get(r.exit_time, np.nan),
                                 "через ролл": "да" if rolls[r.exit_time] != rolls[r.entry_time] else "",
                                 "издержки, USD": r.fees, "результат, USD": r.net_pnl,
                                 "баров в позиции": r.bars_held, "примечание": ""})
                change = float(result.equity.iloc[-1] - H.CAPITAL)
                residual = change - float(t["net_pnl"].sum()) if len(t) else change
                pos = float(result.position.iloc[-1])
                if abs(residual) > 0.005 or pos != 0:
                    rows.append({"период": period, "нога": leg, "вход": pd.NaT, "выход": bars.index[-1],
                                 "направление": "long" if pos > 0 else "short" if pos < 0 else "",
                                 "объём при входе, USD": np.nan, "котировка входа": np.nan,
                                 "котировка выхода": float(quoted["close"].iloc[-1]), "через ролл": "",
                                 "издержки, USD": np.nan, "результат, USD": residual,
                                 "баров в позиции": np.nan,
                                 "примечание": "позиция открыта на конец периода: переоценка по последнему закрытию"})
                period_pnl += change
            totals[period] = period_pnl
        df = pd.DataFrame(rows)
        df = df.sort_values(["период", "выход", "нога"], kind="stable").reset_index(drop=True)
        for col in ("вход", "выход"):                     # дневные бары: дата без часового пояса
            df[col] = pd.to_datetime(df[col]).dt.tz_localize(None).dt.date
        df.insert(0, "№", np.arange(1, len(df) + 1))
        got = totals["OOS"] / H.CAPITAL
        if abs(got - cell.oos_return_report) > 0.0006:          # отчёт округлён до 0.1%
            raise SystemExit(f"{cell.id}: OOS {got:.4%} против отчёта {cell.oos_return_report:.1%}")
        df[COLUMNS].to_csv(CSV / f"{cell.id}.csv", index=False)
        meta.append({"id": cell.id, "study": "seven", "name": cell.ячейка, "market": market,
                     "params": cell.лучшая, "units": "USD на счёт 1 000 000, размер позиции по целевой волатильности 15%",
                     "oos_return_report": cell.oos_return_report, "is_pnl": totals["IS"], "oos_pnl": totals["OOS"]})
        print(f"{cell.id:22} сделок {len(df):4d} | IS {totals['IS']:+12,.0f} | OOS {totals['OOS']:+12,.0f} "
              f"({got:+.1%}, отчёт {cell.oos_return_report:+.1%})")
    (CSV / "seven_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


# ------------------------------------------------------------------ скользящие средние
def grid_row(stage: str, ticker: str, method: str, config: int) -> pd.Series:
    g = pd.read_parquet(MA / "results" / stage / ticker / f"grid_{method}.parquet",
                        filters=[("config", "==", config)])
    return g.iloc[0]


RULES = {"R1_max_pnl": "R1 макс. PnL", "R2_max_sharpe": "R2 макс. Шарп", "R3_plateau": "R3 плато"}


def export_ma():
    sys.path.insert(0, str(MA))
    import data as D

    CSV.mkdir(exist_ok=True)
    meta = []
    for stage in ("stage1", "stage3"):
        for t in ("NG", "BR"):
            raw = D.load_raw(t)
            secid = raw["secid"].to_numpy()
            pos = pd.Series(np.arange(len(raw)), index=pd.DatetimeIndex(raw.index))
            finalists = json.loads((MA / "results" / stage / t / "finalists.json").read_text())
            for f in (x for x in finalists if x["scope"] == "global"):
                c = f["config"]
                trades = pd.read_parquet(MA_WORK / "analysis" / stage / t / "finalists" /
                                         f"cfg{f['grid_config']}_trades.parquet")
                trades = trades.sort_values("Exit Timestamp", kind="stable").reset_index(drop=True)
                ent = pos.reindex(pd.DatetimeIndex(trades["Entry Timestamp"])).to_numpy()
                ext = pos.reindex(pd.DatetimeIndex(trades["Exit Timestamp"])).to_numpy()
                if np.isnan(ent.astype(float)).any() or np.isnan(ext.astype(float)).any():
                    raise SystemExit(f"{stage}/{t}/{f['grid_config']}: время сделки не найдено в данных")
                is_open = (trades["Status"] == "Open").to_numpy()
                # Котировки, по которым шло исполнение: Open бара входа и выхода;
                # открытая на конец данных позиция закрыта по последнему Close.
                entry_q = raw["open"].to_numpy(float)[ent]
                exit_q = np.where(is_open, raw["close"].to_numpy(float)[ext], raw["open"].to_numpy(float)[ext])
                rolled = np.array([secid[a] != secid[b] for a, b in zip(ent, ext, strict=True)])
                df = pd.DataFrame({
                    "№": np.arange(1, len(trades) + 1),
                    "период": np.where(pd.DatetimeIndex(trades["Exit Timestamp"]) < D.OOS_START, "IS", "OOS"),
                    "нога": t,
                    "вход": trades["Entry Timestamp"], "выход": trades["Exit Timestamp"],
                    "направление": trades["Direction"].str.lower(),
                    "объём при входе, USD": entry_q * D.SPECS[t]["multiplier"],
                    "котировка входа": entry_q, "котировка выхода": exit_q,
                    "через ролл": np.where(rolled, "да", ""),
                    "издержки, USD": 0.0,
                    "результат, USD": trades["PnL"].round(2),
                    "баров в позиции": ext - ent,
                    "примечание": np.where(is_open, "открыта на конец данных: закрыта по последнему Close", ""),
                })
                # Без ролла результат обязан совпасть с разностью котировок.
                sign = np.where(df["направление"] == "long", 1.0, -1.0)
                plain = ~rolled
                quoted = (exit_q - entry_q) * sign * D.SPECS[t]["multiplier"]
                bad = np.abs(quoted[plain] - df["результат, USD"].to_numpy()[plain]) > 0.011
                if bad.any():
                    raise SystemExit(f"{stage}/{t}/{f['grid_config']}: {bad.sum()} сделок не сходятся с котировками")
                # Сверка: весь период — с VectorBT-статистикой финалиста; IS/OOS — с сеткой,
                # где сделка относится к периоду по бару выхода (в finalists.json IS/OOS
                # разделены по дневной переоценке, и сделка через 2024-01-01 делится).
                total = df["результат, USD"].sum()
                if abs(total - f["full"]["pnl"]) > 0.05:
                    raise SystemExit(f"{stage}/{t}/{f['grid_config']}: {total} против {f['full']['pnl']}")
                grid = grid_row(stage, t, c["method"], f["grid_config"])
                by_period = {}
                for period in ("IS", "OOS"):
                    by_period[period] = df.loc[df["период"] == period, "результат, USD"].sum()
                    ref = grid[f"pnl_{period.lower()}"]
                    if abs(by_period[period] - ref) > 0.51:          # в сетке PnL округлён до 1 USD
                        raise SystemExit(f"{stage}/{t}/{f['grid_config']} {period}: {by_period[period]} против {ref}")
                rule = f["rule"]
                sid = f"MA_{stage[-1]}_{t}_{rule.split('_')[0]}"
                name = (f"{c['fast_type']}{c['fast_period']} × {c['slow_type']}{c['slow_period']}, "
                        f"наклон {c['method']}, выход {c['exit']}")
                df[COLUMNS].to_csv(CSV / f"{sid}.csv", index=False)
                meta.append({"id": sid, "study": "ma", "stage": stage, "market": t, "rule": RULES[rule],
                             "name": name, "units": "USD на 1 контракт, без издержек (так в задании)",
                             "is_pnl": by_period["IS"], "oos_pnl": by_period["OOS"], "full_pnl": total,
                             "slope": {k: f["slope"][k] for k in ("k", "coverage", "atr_period")}})
                print(f"{sid:14} {name:50} сделок {len(df):5d} | через ролл {rolled.sum():3d} | "
                      f"IS {by_period['IS']:+8.1f} | OOS {by_period['OOS']:+8.1f}")
    (CSV / "ma_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "seven":
        export_seven()
    elif step == "ma":
        export_ma()
    elif step == "excel":
        import build_excel
        build_excel.main()
    else:
        raise SystemExit(__doc__)
