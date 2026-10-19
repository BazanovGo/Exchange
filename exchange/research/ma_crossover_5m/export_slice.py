"""Годовой PnL одного среза сетки «быстрая × медленная» — для карт устойчивости во времени.

Срез — семейство (метод наклона, типы средних, модель выхода) финалиста, отобранного
правилом R2 (максимальный месячный Шарп в выборке) на этапе 3: все 98 × 246 периодов.
Помесячный PnL лежит только в рабочем каталоге (monthly.npy, гигабайты), поэтому в
results/ выгружается его годовая сумма для этого среза.

    MA5M_WORK=/path/to/work python export_slice.py
    -> results/stage3/<NG|BR>/slice_yearly.parquet, slice.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from report import WORK

HERE = Path(__file__).parent
RES = HERE / "results"


def main(stage: str = "stage3", rule: str = "R2_max_sharpe"):
    for t in ("NG", "BR"):
        meta = json.loads((WORK / stage / t / "meta.json").read_text())
        monthly = np.load(WORK / stage / t / "monthly.npy", mmap_mode="r")
        pick = next(x for x in json.loads((RES / stage / t / "finalists.json").read_text())
                    if x["scope"] == "global" and x["rule"] == rule)
        c = pick["config"]
        shape = tuple(meta["shape"])
        fast = np.arange(meta["fast_periods"][0], meta["fast_periods"][1] + 1)
        slow = np.arange(meta["slow_periods"][0], meta["slow_periods"][1] + 1)
        ff, ss = np.meshgrid(fast, slow, indexing="ij")
        idx = np.ravel_multi_index((
            np.full(ff.size, meta["methods"].index(c["method"])),
            np.full(ff.size, meta["types"].index(c["slow_type"])),
            ss.ravel() - slow[0],
            np.full(ff.size, meta["exits"].index(c["exit"])),
            np.full(ff.size, meta["types"].index(c["fast_type"])),
            ff.ravel() - fast[0]), shape)
        assert pick["grid_config"] in set(idx.tolist())
        rows = np.asarray(monthly[np.sort(idx)], dtype=np.float64)
        order = np.argsort(idx)
        years = np.array([int(m[:4]) for m in meta["months"]])
        uniq = np.unique(years)
        yearly = np.stack([rows[:, years == y].sum(axis=1) for y in uniq], axis=1)
        df = pd.DataFrame({"config": np.sort(idx).astype(np.int32),
                           "fast_period": ff.ravel()[order].astype(np.int16),
                           "slow_period": ss.ravel()[order].astype(np.int16)})
        for j, y in enumerate(uniq):
            df[str(y)] = np.round(yearly[:, j], 1).astype(np.float32)
        out = RES / stage / t
        df.to_parquet(out / "slice_yearly.parquet", index=False, compression="zstd")
        info = {"stage": stage, "ticker": t, "rule": rule, "grid_config": pick["grid_config"],
                "family": {k: c[k] for k in ("method", "slow_type", "exit", "fast_type")},
                "years": [int(y) for y in uniq],
                "months_per_year": {int(y): int((years == y).sum()) for y in uniq},
                "units": "USD на 1 контракт, без издержек; год — по месяцу бара исполнения выхода"}
        (out / "slice.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))
        # Сверка: сумма по годам равна полному PnL отобранной конфигурации в сетке.
        full = float(df.loc[df["config"] == pick["grid_config"], [str(y) for y in uniq]].sum(axis=1).iloc[0])
        print(f"{t}: срез {c['fast_type']}×{c['slow_type']} {c['method']} {c['exit']} — {len(df)} конфигураций, "
              f"{len(uniq)} лет; PnL отобранной {full:.1f} USD (финалист: {pick['full']['pnl']:.1f})")
        if abs(full - pick["full"]["pnl"]) > 0.6:
            raise SystemExit(f"{t}: годовая сумма {full} не сходится с финалистом {pick['full']['pnl']}")


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
