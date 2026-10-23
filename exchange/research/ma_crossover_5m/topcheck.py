"""Выживает ли «торгуемый» верхний процент без последнего года.

Верхний 1% конфигураций по чистому IS-результату (при c тиках на сделку,
≥100 сделок) — единственная группа, которая вне выборки остаётся прибыльной
и при издержках. Проверяем, не сделал ли это один последний год: случайная
выборка из группы пересчитывается ядром с журналом сделок, и чистый PnL
считается отдельно за OOS без последнего года и за последний год.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import kernel as K
import ma as M
from data import load_market
from grid import slow_state
from results import load_grid

WORK = Path(os.environ.get("MA5M_WORK", "work"))


def run(stage, ticker, cost_ticks=2.0, n=400, seed=0):
    g = load_grid(WORK / stage / ticker)
    mk = load_market(ticker)
    tick_usd = mk.tick * mk.multiplier
    s = pd.read_parquet(WORK / "analysis" / stage / ticker / "summary.parquet",
                        columns=["config", "pnl_is", "trades_is"])
    net = s["pnl_is"] - s["trades_is"] * cost_ticks * tick_usd
    ok = s["trades_is"] >= 100
    top = net.where(ok).nlargest(max(1, int(ok.sum() * 0.01))).index
    rng = np.random.default_rng(seed)
    pick = rng.choice(top, size=min(n, len(top)), replace=False)
    pm = g.meta.get("per_method") or {m: {"k": g.meta["k"], "coverage": g.meta["coverage"],
                                          "atr_period": g.meta["atr_period"]} for m in g.meta["methods"]}
    last_year = mk.index[-1].year
    t_last = np.searchsorted(mk.index.values, np.datetime64(f"{last_year}-01-01"))
    atrs = {m: M.atr(mk.high, mk.low, mk.close, int(pm[m]["atr_period"])) for m in pm}
    rows = []
    n_m = len(mk.months)
    group_monthly = {c: np.zeros(n_m) for c in (0.0, 1.0, 2.0)}
    for row in g.decode(pick).itertuples():
        f = M.moving_average(row.fast_type, mk.close, mk.volume, int(row.fast_period))
        eb, ed = K.cross_events(mk.close, f, mk.is_roll)
        sl = M.moving_average(row.slow_type, mk.close, mk.volume, int(row.slow_period))
        p = pm[row.method]
        st, _ = slow_state(mk, sl, row.method, int(p["k"]), float(p["coverage"]), atrs[row.method])
        log = np.zeros((len(eb) + 1, 6)); stt = np.zeros((3, K.N_STATS)); mo = np.zeros(len(mk.months))
        ntr = K.simulate(eb, ed, st, K.next_state_index(st, -1), K.next_state_index(st, 1), mk.pnl_open,
                         mk.pnl_close, mk.month, mk.oos_bar, mk.multiplier, g.meta["exits"].index(row.exit),
                         stt, mo, log)
        tr = log[:ntr]
        ex_month = mk.month[tr[:, 1].astype(int)]
        for c in (0.0, 1.0, 2.0):
            group_monthly[c] += np.bincount(ex_month, weights=tr[:, 5] - c * tick_usd, minlength=n_m)
        out = {"config": int(row.config)}
        for c in (0.0, 1.0, 2.0):
            pnl = tr[:, 5] - c * tick_usd
            ex = tr[:, 1]
            out[f"early_{c}"] = float(pnl[(ex >= mk.oos_bar) & (ex < t_last)].sum())
            out[f"last_{c}"] = float(pnl[ex >= t_last].sum())
            out[f"oos_{c}"] = out[f"early_{c}"] + out[f"last_{c}"]
        rows.append(out)
    d = pd.DataFrame(rows)
    res = {"stage": stage, "ticker": ticker, "group": f"верхний 1% по чистому IS при {cost_ticks} тиках",
           "group_size": int(len(top)), "sample": int(len(d)), "last_year": int(last_year)}
    for c in (0.0, 1.0, 2.0):
        for part in ("oos", "early", "last"):
            res[f"share_pos_{part}_{c}"] = float((d[f"{part}_{c}"] > 0).mean())
            res[f"median_{part}_{c}"] = float(d[f"{part}_{c}"].median())
    # Временной тест: средний по группе месячный чистый PnL за OOS без последнего
    # года — одна траектория, поэтому t-статистика по месяцам, а не по конфигурациям.
    months = mk.months
    early = np.flatnonzero((np.arange(n_m) >= mk.oos_month) & (months.year < last_year))
    for c in (0.0, 1.0, 2.0):
        x = group_monthly[c][early] / len(d)
        res[f"early_monthly_mean_{c}"] = float(x.mean())
        res[f"early_monthly_t_{c}"] = float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if x.std() > 0 else None
        res[f"early_pos_months_{c}"] = float((x > 0).mean())
    res["early_months"] = int(len(early))
    (WORK / "analysis" / stage / ticker / "topcheck.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return res


if __name__ == "__main__":
    for t in sys.argv[2:]:
        run(sys.argv[1], t)
