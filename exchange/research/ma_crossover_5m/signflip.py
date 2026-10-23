"""Тест семейства на уровне траектории: случайная смена знака каждой свечи.

Все 3.47 млн конфигураций торгуют ОДНУ историю цены, поэтому доля прибыльных
конфигураций — это не миллионы независимых подтверждений, а одно наблюдение.
Мерой служит средняя по выборке конфигураций лог-доходность сделки (б.п.) на реальном
ряду; нулевое распределение — та же мера на траекториях, где у каждой свечи
случайно сменён знак всех приращений (open, close относительно предыдущего
close; high и low меняются местами). Модули приращений, кластеры
волатильности и её внутридневной профиль сохраняются; направленная
предсказуемость — уничтожается. p — доля нулевых траекторий с мерой не ниже
реальной.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from data import load_market
from grid import TYPES
from nulls import rebuild as _rebuild, rel_bars, run_cfgs


def rebuild(mk, rel, idx):
    """Как в nulls.py, но PnL в лог-единицах: сделка зарабатывает
    d·(ln P_выход − ln P_вход). Без этого траектории без дрейфа уходят по уровню
    цены на порядки, и «тик» на разных путях означает разное."""
    m = _rebuild(mk, rel, idx)
    m.pnl_open, m.pnl_close = np.log(m.open), np.log(m.close)
    m.multiplier = 1.0
    return m

WORK = Path(os.environ.get("MA5M_WORK", "work"))


def flipped(rel, rng):
    s = rng.choice([-1.0, 1.0], size=len(rel))
    out = rel.copy()
    out[:, 0] *= s
    out[:, 3] *= s
    hi, lo = rel[:, 1], rel[:, 2]
    out[:, 1] = np.where(s > 0, hi, -lo)
    out[:, 2] = np.where(s > 0, lo, -hi)
    out[0] = rel[0]
    return out


def stat(x, tick_usd=None):
    """Средняя по конфигурациям лог-доходность сделки, в базисных пунктах."""
    per_trade = x[:, 0] / np.maximum(x[:, 1], 1) * 1e4
    return float(np.mean(per_trade)), float(np.mean(x[:, 0] > 0))


def main(n_paths=40, n_cfg=40):
    out = {}
    for t in ("NG", "BR"):
        mk = load_market(t)
        tick_usd = mk.tick * mk.multiplier
        rng = np.random.default_rng(5)
        cfgs = [(rng.choice(TYPES), int(rng.integers(2, 100)), rng.choice(TYPES), int(rng.integers(5, 251)),
                 str(rng.choice(["A_pct", "B_atr"])), int(rng.integers(0, 2))) for _ in range(n_cfg)]
        rel = rel_bars(mk)
        ident = np.arange(mk.n)
        real_m, real_share = stat(run_cfgs(rebuild(mk, rel, ident), cfgs), tick_usd)
        null_m, null_share = [], []
        for p in range(n_paths):
            m_, s_ = stat(run_cfgs(rebuild(mk, flipped(rel, np.random.default_rng(10_000 + p)), ident), cfgs), tick_usd)
            null_m.append(m_); null_share.append(s_)
            print(f"  {t} нулевая траектория {p + 1}/{n_paths}: средняя сделка {m_:+.3f} б.п., доля прибыльных {s_:.2f}",
                  flush=True)
        null_m = np.array(null_m); null_share = np.array(null_share)
        out[t] = {
            "real_mean_trade_bp": real_m, "real_share_profitable": real_share,
            "null_mean": float(null_m.mean()), "null_std": float(null_m.std(ddof=1)),
            "null_p95": float(np.quantile(null_m, 0.95)), "null_max": float(null_m.max()),
            "p_value": float((np.sum(null_m >= real_m) + 1) / (len(null_m) + 1)),
            "z": float((real_m - null_m.mean()) / null_m.std(ddof=1)),
            "null_share_profitable_mean": float(null_share.mean()),
            "null_share_profitable_p95": float(np.quantile(null_share, 0.95)),
            "n_paths": n_paths, "n_cfg": n_cfg, "null_means": null_m.tolist(),
        }
        print(f"== {t}: реальная средняя сделка {real_m:+.3f} б.п. (доля прибыльных {real_share:.2f}); "
              f"нулевые: {null_m.mean():+.3f} ± {null_m.std(ddof=1):.3f}, p95 {np.quantile(null_m, .95):+.3f}; "
              f"z = {out[t]['z']:+.2f}, p = {out[t]['p_value']:.3f}", flush=True)
    (WORK / "signflip.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
