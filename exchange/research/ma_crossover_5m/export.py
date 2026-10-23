"""Машиночитаемые результаты в git: компактные таблицы, верх сетки и
ключевые метрики каждой конфигурации полной сетки.

Сырые сетки (stats.npy, monthly.npy, summary.parquet) весят гигабайты и
воспроизводятся скриптами; сюда попадает всё, на чём построен отчёт.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from report import WORK

HERE = Path(__file__).parent
OUT = HERE / "results"
SMALL = (".csv", ".json")

# Полная сетка: параметры + ключевые метрики каждой конфигурации, целыми
# числами (USD округлены до 1, коэффициенты умножены на 1000) — так файл
# сжимается вдвое лучше float32. Два файла на инструмент и этап (по методу
# нейтрали), каждый ~25 МБ: ниже предупреждения GitHub в 50 МБ.
PARAMS = ["config", "method", "exit", "fast_type", "fast_period", "slow_type", "slow_period"]
USD = ["pnl_is", "pnl_oos", "max_dd_is", "max_dd_oos"]
COUNT = ["trades_is", "trades_oos"]
X1000 = ["sharpe_m_is", "sharpe_m_oos", "plateau_sharpe_is"]


def full_grid(src: Path, dst: Path):
    df = pd.read_parquet(src / "summary.parquet", columns=PARAMS + USD + COUNT + X1000)
    for m, part in df.groupby("method", observed=True, sort=True):
        q = pd.DataFrame({"config": part["config"].astype("int32")})
        for c in PARAMS[1:]:
            q[c] = part[c].astype("int16") if c.endswith("period") else part[c]
        for c in USD:
            v = np.round(part[c].to_numpy())
            assert np.isfinite(v).all() and np.abs(v).max() < 32_000, c
            q[c] = v.astype("int16")
        for c in COUNT:
            q[c] = part[c].astype("int32")
        for c in X1000:
            v = np.round(part[c].to_numpy() * 1000)
            assert np.isfinite(v).all() and np.abs(v).max() < 32_000, c
            q[f"{c}_x1000"] = v.astype("int16")
        f = dst / f"grid_{m}.parquet"
        q.to_parquet(f, compression="zstd", compression_level=19, index=False, row_group_size=1 << 20)
        print(f"{f.relative_to(HERE)}: {len(q):,} конфигураций, {f.stat().st_size / 1e6:.1f} МБ")


def load_full_grid(stage: str, ticker: str, results: Path = OUT) -> pd.DataFrame:
    """Полная сетка из results/ с метриками в исходных единицах."""
    parts = sorted((results / stage / ticker).glob("grid_*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).sort_values("config")
    for c in X1000:
        df[c] = df.pop(f"{c}_x1000") / 1000
    return df.reset_index(drop=True)


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
                # Дневная кривая капитала (VectorBT, с переоценкой) — только текущих
                # финалистов: в рабочем каталоге могут лежать кривые прежних отборов.
                current = src / "finalists.json"
                if current.exists():
                    for x in json.loads(current.read_text()):
                        eq = fin / f"cfg{x['grid_config']}_equity.parquet"
                        if eq.exists():
                            shutil.copy(eq, dst / "finalists" / eq.name)
            meta = WORK / stage / t / "meta.json"
            if meta.exists():
                shutil.copy(meta, dst / "grid_meta.json")
            sp = src / "summary.parquet"
            if sp.exists():
                full_grid(src, dst)
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
