"""Журнал сделок: каждая сделка отобранных стратегий по порядку и её результат.

Три шага, потому что части живут в разных окружениях:

    python export_trades.py seven   # семь семейств: пересчёт через движок пакета (нужен пакет exchange)
    python export_trades.py seven-full   # семь семейств: сквозной прогон по всей истории -> csv/full/
    python export_trades.py seven-wf     # семь семейств: walk-forward, сделки каждого окна -> csv/walkforward/
    python export_trades.py ma      # скользящие средние: сделки VectorBT из рабочего каталога
                                    # (глобальные отборы — csv/, лучшие в группах — csv/groups/)
    python export_trades.py ma-families  # скользящие средние: лучшие в каждом из 144 семейств
                                         # (событийное ядро исследования) -> csv/families/
    python export_trades.py ma-wf   # скользящие средние: walk-forward, сделки каждого шага -> csv/walkforward/
    python export_trades.py excel   # trade_log.xlsx и README.md из CSV

Сделки не отбираются заново: семь семейств — победители из
docs/RESULTS_SEVEN_FAMILIES.md, скользящие средние — все финалисты из
research/ma_crossover_5m/results/<этап>/<рынок>/finalists.json: глобальные
отборы R1–R3 и лучшие R1–R3 внутри каждой группы «выход × метод наклона». Каждый
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
class Seven:
    """Данные и прогоны семи семейств — ровно так, как их считали run_all и run_stats."""

    def __init__(self):
        sys.path.insert(0, str(ROOT / "research" / "seven_families"))
        import families as F
        import harness as H
        import run_all as RA

        self.F, self.H, self.RA = F, H, RA
        self.universe = H.load_universe()
        self.panel_all = {r: p["bars"] for r, p in self.universe.items()}
        self.ppys = {r: self.universe[r]["ppy"] for r in self.panel_all}
        self.common = self.panel_all["NG"].index.intersection(self.panel_all["BR"].index)

    def index(self, family: str, market: str):
        return self.common if family in self.F.PANEL else self.panel_all[market].index

    def legs(self, family: str, market: str, params: dict, idx):
        """Ноги прогона на отрезке idx: (рынок, бары, целевая позиция). Сигналы — только по idx."""
        F, RA = self.F, self.RA
        if family in F.PANEL:
            panel = {r: b.loc[self.common].loc[idx] for r, b in self.panel_all.items()}
            signals = F.PANEL[family](panel, **params)
            return [(r, panel[r], RA.positions(panel[r], self.ppys[r], signals[r], legs=len(panel)))
                    for r in panel]
        bars = self.panel_all[market].loc[idx]
        return [(market, bars, RA.positions(bars, self.ppys[market], F.SINGLE[family](bars, **params)))]

    def run(self, family: str, market: str, params: dict, idx, period) -> tuple[list[dict], float]:
        """Сделки прогона и изменение капитала; period — строка или функция даты выхода."""
        H = self.H
        label = period if callable(period) else (lambda _ts: period)
        rows, change_total = [], 0.0
        for leg, bars, target in self.legs(family, market, params, idx):
            result, _ = H.backtest(bars, target, symbol=leg, ppy=self.ppys[leg])
            t = result.trades
            # Движок считает на ряду, склеенном по отношению цен: его цены в прошлом
            # не равны биржевым. В таблицу идут котировки Open дня исполнения
            # и объём позиции в долларах (количество × расчётная цена).
            quoted = self.universe[leg]["quoted"]
            rolls = bars["is_roll"].astype(int).cumsum()
            for r in t.itertuples():
                rows.append({"период": label(r.exit_time), "нога": leg, "вход": r.entry_time, "выход": r.exit_time,
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
                rows.append({"период": label(bars.index[-1]), "нога": leg, "вход": pd.NaT, "выход": bars.index[-1],
                             "направление": "long" if pos > 0 else "short" if pos < 0 else "",
                             "объём при входе, USD": np.nan, "котировка входа": np.nan,
                             "котировка выхода": float(quoted["close"].iloc[-1]), "через ролл": "",
                             "издержки, USD": np.nan, "результат, USD": residual,
                             "баров в позиции": np.nan,
                             "примечание": "позиция открыта на конец периода: переоценка по последнему закрытию"})
            change_total += change
        return rows, change_total

    def returns(self, family: str, market: str, params: dict, idx):
        """Доходности прогона на отрезке idx, как run_stats.returns_for."""
        RA = self.RA
        if family in self.F.PANEL:
            sub = {r: b.loc[self.common].loc[idx] for r, b in self.panel_all.items()}
            _, ret, ppy, _, _ = RA.run_panel(family, params, sub, self.ppys)
            return ret, ppy
        res, _ = RA.run_single(family, market, params, self.panel_all[market].loc[idx], self.ppys[market])
        return res.returns, self.ppys[market]


def dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("вход", "выход"):                     # дневные бары: дата без часового пояса
        df[col] = pd.to_datetime(df[col]).dt.tz_localize(None).dt.date
    return df


def export_seven():
    S = Seven()
    CSV.mkdir(exist_ok=True)
    meta = []
    for cell in seven_cells().itertuples():
        family, market, params = cell.family, cell.market, cell.params
        is_idx, oos_idx = S.RA.split(S.index(family, market))
        rows, totals = [], {}
        for period, idx in (("IS", is_idx), ("OOS", oos_idx)):
            # Ровно те прогоны, по которым считался отчёт: IS-прогон для отбора и
            # отдельный OOS-прогон, начинающийся с пустой позиции (run_all.evaluate_cell).
            r, totals[period] = S.run(family, market, params, idx, period)
            rows += r
        df = pd.DataFrame(rows)
        df = dates(df.sort_values(["период", "выход", "нога"], kind="stable").reset_index(drop=True))
        df.insert(0, "№", np.arange(1, len(df) + 1))
        got = totals["OOS"] / S.H.CAPITAL
        if abs(got - cell.oos_return_report) > 0.0006:          # отчёт округлён до 0.1%
            raise SystemExit(f"{cell.id}: OOS {got:.4%} против отчёта {cell.oos_return_report:.1%}")
        df[COLUMNS].to_csv(CSV / f"{cell.id}.csv", index=False)
        meta.append({"id": cell.id, "csv": f"{cell.id}.csv", "study": "seven", "name": cell.ячейка, "market": market,
                     "params": cell.лучшая, "units": "USD на счёт 1 000 000, размер позиции по целевой волатильности 15%",
                     "oos_return_report": cell.oos_return_report, "is_pnl": totals["IS"], "oos_pnl": totals["OOS"]})
        print(f"{cell.id:22} сделок {len(df):4d} | IS {totals['IS']:+12,.0f} | OOS {totals['OOS']:+12,.0f} "
              f"({got:+.1%}, отчёт {cell.oos_return_report:+.1%})")
    (CSV / "seven_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


def same_trades(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Одни и те же сделки: даты, нога, направление и результат."""
    key = ["нога", "вход", "выход", "направление", "результат, USD"]
    a = a[key].sort_values(key, kind="stable").reset_index(drop=True)
    b = b[key].sort_values(key, kind="stable").reset_index(drop=True)
    if len(a) != len(b):
        return False
    same = (a[key[:4]].astype(str) == b[key[:4]].astype(str)).all().all()
    return bool(same and np.allclose(a[key[4]].to_numpy(float), b[key[4]].to_numpy(float), rtol=1e-9, atol=1e-6))


def export_seven_full():
    """Сквозной прогон победителя по всей истории: все сделки подряд, без разрыва.

    Отчёт оценивает IS и OOS отдельными прогонами: OOS начинается без позиции и с
    пустыми индикаторами, а между периодами — эмбарго. Здесь та же система с теми
    же параметрами торгует всю историю одним прогоном (на таком прогоне считались
    PBO и CPCV в run_stats), и сделка помечена периодом по дате выхода: IS,
    эмбарго или OOS. Сверка: сделки, закрытые до конца IS, совпадают с IS-журналом
    отчёта, а сумма сделок с переоценкой равна изменению капитала прогона.
    """
    S = Seven()
    (CSV / "full").mkdir(exist_ok=True)
    meta = []
    for cell in seven_cells().itertuples():
        family, market, params = cell.family, cell.market, cell.params
        index = S.index(family, market)
        is_idx, oos_idx = S.RA.split(index)
        is_end, oos_start = is_idx[-1], oos_idx[0]

        def period(ts, is_end=is_end, oos_start=oos_start):
            return "IS" if ts <= is_end else ("эмбарго" if ts < oos_start else "OOS")

        rows, change = S.run(family, market, params, index, period)
        df = pd.DataFrame(rows)
        if abs(df["результат, USD"].sum() - change) > 0.01:
            raise SystemExit(f"{cell.id}: сумма сделок {df['результат, USD'].sum():.2f} против капитала {change:.2f}")
        df = dates(df.sort_values(["выход", "нога"], kind="stable").reset_index(drop=True))
        df.insert(0, "№", np.arange(1, len(df) + 1))
        report_is = pd.read_csv(CSV / f"{cell.id}.csv", keep_default_na=False, na_values=[""])
        report_is = report_is[(report_is["период"] == "IS") & report_is["вход"].notna()]
        mine_is = df[(df["период"] == "IS") & df["вход"].notna()]
        if not same_trades(mine_is.astype({"вход": str, "выход": str}), report_is):
            raise SystemExit(f"{cell.id}: сделки сквозного прогона до конца IS не совпали с IS-журналом")
        df[COLUMNS].to_csv(CSV / "full" / f"{cell.id}.csv", index=False)
        by = df.groupby("период")["результат, USD"].agg(["sum", "size"])
        meta.append({"id": cell.id, "csv": f"full/{cell.id}.csv", "study": "seven", "name": cell.ячейка,
                     "market": market, "params": cell.лучшая, "is_end": str(is_end.date()),
                     "oos_start": str(oos_start.date()), "trades": len(df),
                     "pnl": {k: float(v) for k, v in by["sum"].items()},
                     "rows": {k: int(v) for k, v in by["size"].items()}, "equity_change": change,
                     "units": "USD на счёт 1 000 000, размер позиции по целевой волатильности 15%"})
        print(f"{cell.id:22} сделок {len(df):4d} | " + " | ".join(
            f"{k} {by.loc[k, 'size']:3d} / {by.loc[k, 'sum']:+12,.0f}" for k in ("IS", "эмбарго", "OOS")
            if k in by.index) + f" | капитал {change:+12,.0f}")
    (CSV / "seven_full_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


WF7_COLUMNS = ["№", "окно", "даты окна", "параметры", *COLUMNS[1:]]


def export_seven_wf():
    """Walk-forward с переоптимизацией, как в run_stats: сделки каждого тестового окна.

    В каждом окне параметры выбираются по лучшему Шарпу на обучающем отрезке
    (35% истории), затем отдельный прогон на тестовом отрезке (8% истории,
    не меньше 120 дней) с этими параметрами — его сделки и записаны. Сверка:
    эффективность, доля прибыльных окон и стабильность параметров совпадают
    с таблицей раздела 3 docs/RESULTS_SEVEN_FAMILIES.md.
    """
    from exchange.backtest.metrics import sharpe_ratio
    from exchange.stats.cv import WalkForward

    S = Seven()
    reference = md_table(SEVEN_DOC.read_text(), "WF эффект.").set_index("ячейка")
    (CSV / "walkforward").mkdir(exist_ok=True)
    meta = []
    for cell in seven_cells().itertuples():
        family, market = cell.family, cell.market
        index = S.index(family, market)
        grid = S.RA.GRIDS[family]
        wf = WalkForward(train_size=max(int(len(index) * 0.35), 250),
                         test_size=max(int(len(index) * 0.08), 120), horizon=5, embargo=S.RA.EMBARGO)
        rows, wf_is, wf_oos, chosen_all, windows = [], [], [], [], []
        for w, sp in enumerate(wf.split(len(index)), 1):
            train, test = index[sp.train], index[sp.test]
            scored = []
            for params in grid:
                try:
                    ret, p = S.returns(family, market, params, train)
                    scored.append((sharpe_ratio(ret, periods_per_year=p), params))
                except Exception:          # как в run_stats: конфигурация без сделок не участвует
                    continue
            scored = [s for s in scored if np.isfinite(s[0])]
            if not scored:
                continue
            best_is, chosen = max(scored, key=lambda t: t[0])
            ret, p = S.returns(family, market, chosen, test)
            wf_is.append(best_is)
            wf_oos.append(sharpe_ratio(ret, periods_per_year=p))
            chosen_all.append(tuple(sorted(chosen.items())))
            span = f"{test[0].date()}…{test[-1].date()}"
            r, change = S.run(family, market, chosen, test, "WF")
            for row in r:
                row.update({"окно": w, "даты окна": span, "параметры": ",".join(f"{k}={v}" for k, v in chosen.items())})
            rows += r
            windows.append({"окно": w, "обучение": f"{train[0].date()}…{train[-1].date()}", "тест": span,
                            "параметры": ",".join(f"{k}={v}" for k, v in chosen.items()),
                            "шарп обучения": float(best_is), "шарп теста": float(wf_oos[-1]), "итог, USD": change})
        wf_is, wf_oos = np.array(wf_is), np.array(wf_oos)
        got = {"WF эффект.": (np.nanmean(wf_oos) / np.nanmean(wf_is)
                              if len(wf_is) and np.nanmean(wf_is) > 0 else np.nan),
               "WF окон>0": float((wf_oos > 0).mean()),
               "WF стабил. параметров": float(np.mean([a == b for a, b in zip(chosen_all, chosen_all[1:])]))}
        for col, value in got.items():
            ref = reference.loc[cell.ячейка, col]
            if (ref == "" and np.isfinite(value)) or (ref != "" and abs(value - float(ref)) > 0.0006):
                raise SystemExit(f"{cell.id}: {col} {value:.3f} против отчёта {ref}")
        df = pd.DataFrame(rows)
        df = dates(df.sort_values(["окно", "выход", "нога"], kind="stable").reset_index(drop=True))
        df.insert(0, "№", np.arange(1, len(df) + 1))
        df[WF7_COLUMNS].to_csv(CSV / "walkforward" / f"{cell.id}_WF.csv", index=False)
        meta.append({"id": f"{cell.id}_WF", "csv": f"walkforward/{cell.id}_WF.csv", "study": "seven",
                     "name": cell.ячейка, "market": market, "windows": windows, "trades": len(df),
                     "pnl": float(df["результат, USD"].sum()), "report": {k: float(v) for k, v in got.items()},
                     "units": "USD на счёт 1 000 000 в каждом окне, размер позиции по целевой волатильности 15%"})
        print(f"{cell.id:22} окон {len(windows)} | сделок {len(df):4d} | итог {df['результат, USD'].sum():+12,.0f} | "
              + " | ".join(f"{k} {v:+.3f}" for k, v in got.items()), flush=True)
    (CSV / "seven_wf_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


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
    (CSV / "groups").mkdir(exist_ok=True)
    meta, groups = [], []
    for stage in ("stage1", "stage3"):
        for t in ("NG", "BR"):
            raw = D.load_raw(t)
            secid = raw["secid"].to_numpy()
            pos = pd.Series(np.arange(len(raw)), index=pd.DatetimeIndex(raw.index))
            finalists = json.loads((MA / "results" / stage / t / "finalists.json").read_text())
            # Если лучший в группе совпал с глобальным отбором, в finalists.json он
            # записан один раз — как глобальный.
            for f in finalists:
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
                r_code = rule.split("_")[0]
                if f["scope"] == "global":
                    sid = f"MA_{stage[-1]}_{t}_{r_code}"
                    path, group = f"{sid}.csv", ""
                else:
                    sid = f"MA_{stage[-1]}_{t}_x{c['exit'][0]}_s{c['method'][0]}_{r_code}"
                    path, group = f"groups/{sid}.csv.gz", f"выход {c['exit']}, наклон {c['method']}"
                name = (f"{c['fast_type']}{c['fast_period']} × {c['slow_type']}{c['slow_period']}, "
                        f"наклон {c['method']}, выход {c['exit']}")
                # mtime=0: одинаковые сделки дают одинаковый файл при каждой выгрузке.
                df[COLUMNS].to_csv(CSV / path, index=False,
                                   compression={"method": "gzip", "mtime": 0} if path.endswith(".gz") else None)
                (meta if f["scope"] == "global" else groups).append({
                    "id": sid, "csv": path, "study": "ma", "stage": stage, "market": t, "rule": RULES[rule],
                    "group": group, "name": name, "units": "USD на 1 контракт, без издержек (так в задании)",
                    "is_pnl": by_period["IS"], "oos_pnl": by_period["OOS"], "full_pnl": total,
                    "slope": {k: f["slope"][k] for k in ("k", "coverage", "atr_period")}})
                print(f"{sid:20} {name:50} сделок {len(df):5d} | через ролл {rolled.sum():3d} | "
                      f"IS {by_period['IS']:+8.1f} | OOS {by_period['OOS']:+8.1f}")
    (CSV / "ma_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    (CSV / "ma_groups_meta.json").write_text(json.dumps(groups, ensure_ascii=False, indent=1))


def write_gz(df: pd.DataFrame, path: Path):
    # mtime=0: одинаковые сделки дают одинаковый файл при каждой выгрузке.
    df.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def finalist_id(stage: str, t: str, f: dict) -> str:
    """Имя журнала финалиста — то же, что даёт export_ma."""
    c, r_code = f["config"], f["rule"].split("_")[0]
    if f["scope"] == "global":
        return f"MA_{stage[-1]}_{t}_{r_code}"
    return f"MA_{stage[-1]}_{t}_x{c['exit'][0]}_s{c['method'][0]}_{r_code}"


def finalist_ids(stage: str, t: str) -> dict[int, str]:
    """Номер конфигурации -> журнал финалиста, где она уже записана."""
    finalists = json.loads((MA / "results" / stage / t / "finalists.json").read_text())
    return {int(f["grid_config"]): finalist_id(stage, t, f) for f in finalists}


def export_ma_families():
    """Лучший по плато в каждом из 144 семейств (сочетание типов × выход × метод).

    Это кандидаты разделов 7 и 12 отчёта: отобраны по in-sample так же, как
    финалисты, но через VectorBT не прогонялись, поэтому сделки даёт ядро.
    Сверка — с сеткой (picks.csv): число сделок IS/OOS точно, PnL до 0.5 USD.
    """
    import ma_kernel as MK

    (CSV / "families").mkdir(exist_ok=True)
    meta = []
    for stage in ("stage1", "stage3"):
        for t in ("NG", "BR"):
            kern, slope, known = MK.Kernel(t), MK.slopes(stage, t), finalist_ids(stage, t)
            picks = pd.read_csv(MA / "results" / stage / t / "picks.csv")
            picks = picks[picks["scope"].str.startswith("family:")]
            for p in picks.itertuples(index=False):
                cfg = {k: getattr(p, k) for k in MK.PARAMS}
                df = kern.journal(cfg, slope[cfg["method"]])
                got = {}
                for period, pnl_ref, n_ref in (("IS", p.pnl_is, p.trades_is), ("OOS", p.pnl_oos, p.trades_oos)):
                    part = df.loc[df["период"] == period, "результат, USD"]
                    got[period] = float(part.sum())
                    if len(part) != int(n_ref) or abs(got[period] - pnl_ref) > 0.51:
                        raise SystemExit(f"{stage}/{t} {MK.label(cfg)} {period}: сделок {len(part)} против "
                                         f"{int(n_ref)}, PnL {got[period]:.1f} против {pnl_ref}")
                sid = (f"MA_{stage[-1]}_{t}_f_{cfg['fast_type']}-{cfg['slow_type']}"
                       f"_x{cfg['exit'][0]}_s{cfg['method'][0]}")
                write_gz(df, CSV / "families" / f"{sid}.csv.gz")
                meta.append({
                    "id": sid, "csv": f"families/{sid}.csv.gz", "study": "ma", "stage": stage, "market": t,
                    "rule": "R3 плато", "family": f"{cfg['fast_type']} × {cfg['slow_type']}, выход {cfg['exit']}, "
                    f"наклон {cfg['method']}", "name": MK.label(cfg), "grid_config": int(p.config),
                    "same_as": known.get(int(p.config), ""), "trades": len(df),
                    "is_pnl": got["IS"], "oos_pnl": got["OOS"], "full_pnl": got["IS"] + got["OOS"],
                    "units": "USD на 1 контракт, без издержек (так в задании)", "slope": slope[cfg["method"]]})
            print(f"{stage} {t}: семейств {len(picks)}, сделок {sum(m['trades'] for m in meta[-len(picks):]):,}",
                  flush=True)
    (CSV / "ma_families_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


WF_RULES = {"pnl": "максимальный PnL", "sharpe": "максимальный месячный Шарп",
            "plateau": "максимальный Шарп, сглаженный по соседним периодам"}
WF_COLUMNS = ["№", "шаг", "месяцы шага", "конфигурация", "номер в сетке", *COLUMNS[1:]]


def export_ma_wf():
    """Walk-forward с переоптимизацией: на каждом шаге — сделки выбранной конфигурации.

    Как в отчёте (robust.walk_forward): конфигурация выбирается по 24 месяцам
    обучения, её результат берётся за следующие 3 месяца; сделка относится к шагу
    по месяцу выхода. Поэтому сделка, открытая до начала шага, входит в шаг целиком
    и помечена. Сверка — с test_pnl шага в wf_<критерий>.csv.
    """
    import ma_kernel as MK

    (CSV / "walkforward").mkdir(exist_ok=True)
    meta = []
    for stage in ("stage1", "stage3"):
        for t in ("NG", "BR"):
            kern, slope = MK.Kernel(t), MK.slopes(stage, t)
            months = kern.mk.months
            for crit in ("pnl", "sharpe", "plateau"):
                wf = pd.read_csv(MA / "results" / stage / t / f"wf_{crit}.csv")
                params = MK.config_params(stage, t, wf["config"])
                parts = []
                for step, r in enumerate(wf.itertuples(index=False), 1):
                    cfg = params[int(r.config)]
                    df, month = kern.journal_months(cfg, slope[cfg["method"]])
                    sel = (month >= r.test_start) & (month < r.test_end)
                    part = df[sel].copy()
                    if abs(part["результат, USD"].sum() - r.test_pnl) > 0.15:
                        raise SystemExit(f"{stage}/{t} WF {crit} шаг {step}: {part['результат, USD'].sum():.2f} "
                                         f"против {r.test_pnl:.2f}")
                    early = pd.DatetimeIndex(part["вход"]).to_period("M") < months[r.test_start]
                    note = "вход до начала шага: сделка относится к шагу по месяцу выхода"
                    part["примечание"] = np.where(early, np.where(part["примечание"] != "",
                                                                  part["примечание"] + "; " + note, note),
                                                  part["примечание"])
                    part.insert(1, "шаг", step)
                    part.insert(2, "месяцы шага", f"{months[r.test_start]}…{months[r.test_end - 1]}")
                    part.insert(3, "конфигурация", MK.label(cfg))
                    part.insert(4, "номер в сетке", int(r.config))
                    parts.append(part)
                out = pd.concat(parts, ignore_index=True)
                out["№"] = np.arange(1, len(out) + 1)
                sid = f"MA_{stage[-1]}_{t}_WF_{crit}"
                write_gz(out[WF_COLUMNS], CSV / "walkforward" / f"{sid}.csv.gz")
                total = float(out["результат, USD"].sum())
                meta.append({"id": sid, "csv": f"walkforward/{sid}.csv.gz", "study": "ma", "stage": stage,
                             "market": t, "criterion": crit, "rule": WF_RULES[crit], "steps": len(wf),
                             "configs": int(wf["config"].nunique()), "trades": len(out),
                             "trades_entered_before_step": int(out["примечание"].str.contains("вход до начала").sum()),
                             "pnl": total, "wf_pnl_reference": float(wf["test_pnl"].sum()),
                             "units": "USD на 1 контракт, без издержек (так в задании)"})
                print(f"{sid:18} шагов {len(wf):3d} | сделок {len(out):6d} | PnL {total:+9.1f} "
                      f"(wf_{crit}.csv {wf['test_pnl'].sum():+9.1f})", flush=True)
    (CSV / "ma_wf_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    if step == "seven":
        export_seven()
    elif step == "seven-full":
        export_seven_full()
    elif step == "seven-wf":
        export_seven_wf()
    elif step == "ma":
        export_ma()
    elif step == "ma-families":
        export_ma_families()
    elif step == "ma-wf":
        export_ma_wf()
    elif step == "excel":
        import build_excel
        build_excel.main()
    else:
        raise SystemExit(__doc__)
