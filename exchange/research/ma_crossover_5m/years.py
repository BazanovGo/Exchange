"""Устойчивость во времени: по каждому году — доля прибыльных конфигураций
сетки и медианный годовой PnL. Сосредоточенное в одном-двух годах
«преимущество» — это эпизод, а не закономерность."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from results import load_grid

WORK = Path(os.environ.get("MA5M_WORK", "work"))


def run(stage, ticker):
    g = load_grid(WORK / stage / ticker)
    months = pd.PeriodIndex(g.meta["months"], freq="M")
    years = sorted(set(months.year))
    N = g.n
    rows = {y: {"pos": 0, "vals": []} for y in years}
    rng = np.random.default_rng(0)
    sample = np.sort(rng.choice(N, 200_000, replace=False))
    pos = {y: 0 for y in years}
    med_sample = {y: [] for y in years}
    for a in range(0, N, 400_000):
        x = np.asarray(g.monthly[a:a + 400_000], dtype=np.float64)
        for y in years:
            cols = np.flatnonzero(months.year == y)
            v = x[:, cols].sum(1)
            pos[y] += int((v > 0).sum())
        sel = sample[(sample >= a) & (sample < a + 400_000)] - a
        for y in years:
            cols = np.flatnonzero(months.year == y)
            med_sample[y].append(x[sel][:, cols].sum(1))
    out = []
    for y in years:
        v = np.concatenate(med_sample[y])
        out.append({"год": int(y), "месяцев": int((months.year == y).sum()),
                    "доля прибыльных конфигураций": pos[y] / N,
                    "медианный PnL, USD": float(np.median(v)),
                    "PnL 10-го перцентиля": float(np.quantile(v, 0.1)),
                    "PnL 90-го перцентиля": float(np.quantile(v, 0.9))})
    df = pd.DataFrame(out)
    df.to_csv(WORK / "analysis" / stage / ticker / "years_grid.csv", index=False)
    print(ticker, "\n", df.round(3).to_string(index=False))



def oos_split(stage, ticker):
    """Доля прибыльных вне выборки — целиком, без последнего года и только в нём."""
    g = load_grid(WORK / stage / ticker)
    months = pd.PeriodIndex(g.meta["months"], freq="M")
    oos = np.arange(g.oos_month, len(months))
    last = oos[months[oos].year == months[-1].year]
    early = oos[months[oos].year < months[-1].year]
    tot = {"all": 0, "early": 0, "last": 0}
    for a in range(0, g.n, 400_000):
        x = np.asarray(g.monthly[a:a + 400_000], dtype=np.float64)
        tot["all"] += int((x[:, oos].sum(1) > 0).sum())
        tot["early"] += int((x[:, early].sum(1) > 0).sum())
        tot["last"] += int((x[:, last].sum(1) > 0).sum())
    out = {"oos_months": len(oos), "early_months": len(early), "last_year": int(months[-1].year),
           "last_months": len(last), **{f"share_pos_{k}": v / g.n for k, v in tot.items()}}
    (WORK / "analysis" / stage / ticker / "oos_split.json").write_text(json.dumps(out, indent=1))
    print(ticker, out)
    return out


if __name__ == "__main__":
    for t in sys.argv[2:]:
        run(sys.argv[1], t)
        oos_split(sys.argv[1], t)
