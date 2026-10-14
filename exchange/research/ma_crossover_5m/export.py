"""Машиночитаемые результаты в git: компактные таблицы и верх сетки.

Полные сетки (stats.npy, monthly.npy, summary.parquet) весят гигабайты и
воспроизводятся скриптами; сюда попадает всё, на чём построен отчёт.
"""
import shutil
import sys
from pathlib import Path

import pandas as pd

from report import WORK

HERE = Path(__file__).parent
OUT = HERE / "results"
SMALL = (".csv", ".json")


def main(stages):
    OUT.mkdir(exist_ok=True)
    for f in ("validation_vbt.csv", "nulls.json", "signflip.json"):
        if (WORK / f).exists():
            shutil.copy(WORK / f, OUT / f)
    for stage in stages:
        base = WORK / "analysis" / stage
        if not base.exists():
            continue
        for f in base.glob("*"):
            if f.suffix in SMALL:
                (OUT / stage).mkdir(parents=True, exist_ok=True)
                shutil.copy(f, OUT / stage / f.name)
        for t in ("NG", "BR"):
            src = base / t
            if not src.exists():
                continue
            dst = OUT / stage / t
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.glob("*"):
                if f.suffix in SMALL:
                    shutil.copy(f, dst / f.name)
            fin = src / "finalists"
            if fin.exists():
                (dst / "finalists").mkdir(exist_ok=True)
                for f in fin.glob("*.csv"):
                    shutil.copy(f, dst / "finalists" / f.name)
            sp = src / "summary.parquet"
            if sp.exists():
                df = pd.read_parquet(sp)
                ok = df["trades_is"] >= 100
                top = df[ok].nlargest(10_000, "plateau_sharpe_is")
                top.to_parquet(dst / "top10000_by_plateau_is.parquet", index=False)
                print(f"{stage}/{t}: топ-10 000 -> {dst / 'top10000_by_plateau_is.parquet'}")
    for t in ("NG", "BR"):
        src = WORK / "analysis" / "stage2" / t
        if src.exists():
            dst = OUT / "stage2" / t
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.glob("*"):
                if f.suffix in SMALL:
                    shutil.copy(f, dst / f.name)
    sp = WORK / "stage3" / "slope_params.json"
    if sp.exists():
        shutil.copy(sp, OUT / "slope_params.json")
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"results/: {total / 1e6:.1f} МБ")


if __name__ == "__main__":
    main(sys.argv[1:] or ["stage1", "stage3"])
