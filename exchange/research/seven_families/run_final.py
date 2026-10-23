"""Финальная агрегация: вероятности, мультитест, RC/SPA, серии убытков, вердикты."""
from __future__ import annotations

import itertools, pickle, logging, warnings, json
logging.disable(logging.INFO); warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import families as F
import harness as H
from run_all import EMBARGO, GRIDS, split
from exchange.backtest.metrics import sharpe_ratio
from exchange.stats.bootstrap import block_bootstrap, optimal_block_length
from exchange.stats.multiple_testing import (correct, estimate_pi0,
                                             expected_false_discoveries)
from exchange.stats.reality_check import reality_check, superior_predictive_ability
from exchange.funnel.hypothesis import make_hypothesis
from exchange.funnel.verdict import decide_verdict

RNG_SEED = 20261007
R = pickle.load(open("results.pkl", "rb"))
S = pickle.load(open("stats.pkl", "rb"))
table = S["table"].set_index(S["table"]["family"] + "|" + S["table"]["market"])
TOTAL = S["total_trials"]
universe = H.load_universe()
out = []
def say(line=""):
    print(line, flush=True); out.append(line)

def sr_stat(x):
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / sd) if sd > 1e-15 else np.nan

def boot_p(series, n=600, seed=RNG_SEED):
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 50:
        return np.nan
    try:
        return block_bootstrap(arr, sr_stat, n_resamples=n, seed=seed).p_value_greater_than(0.0)
    except Exception:
        return np.nan

# =====================================================================
# 1. ЛЕДЖЕР ПОИСКА: p-value каждой из 125 конфигураций на in-sample
# =====================================================================
import os
if os.path.exists("search.pkl"):
    search = pd.read_pickle("search.pkl")
    say(f"леджер поиска загружен из кэша: {len(search)} конфигураций")
else:
  search = []
  for key, row in table.iterrows():
      matrix = R["matrices"][key]
      is_idx, oos_idx = split(matrix.index)
      ppy = row["ppy"]
      for col in matrix.columns:
          is_r = matrix[col].loc[is_idx].dropna()
          oos_r = matrix[col].loc[oos_idx].dropna()
          search.append({
              "key": key, "family": row["family"], "market": row["market"], "config": col,
              "is_sharpe": sharpe_ratio(is_r, periods_per_year=ppy),
              "oos_sharpe": sharpe_ratio(oos_r, periods_per_year=ppy),
              "p_is": boot_p(is_r), "p_oos": boot_p(oos_r),
              "winner": col == row["best_params"],
          })
  search = pd.DataFrame(search)
  search.to_pickle("search.pkl")
  say(f"конфигураций в леджере: {len(search)}")

# =====================================================================
# 2. МУЛЬТИТЕСТ
# =====================================================================
from scipy import stats as sps


def hac_p(x):
    """Односторонний p для H0: E[r] <= 0, ядро Бартлетта (Newey–West).

    Нужен рядом с бутстрэпом потому, что у бутстрэп-p есть жёсткий порог
    1/(B+1): при B=600 минимальный достижимый p равен 0.0017, а поправка
    Holm на 125 гипотез требует 0.05/125 = 0.0004. Без аналитического p
    «ни одна гипотеза не выжила по Holm» было бы артефактом разрешения, а
    не результатом.
    """
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = x.size
    if n < 50:
        return np.nan
    d = x - x.mean()
    lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    var = float(d @ d) / n
    for k in range(1, lag + 1):
        var += 2.0 * (1.0 - k / (lag + 1.0)) * float(d[k:] @ d[:-k]) / n
    if var <= 0:
        return np.nan
    return float(sps.norm.sf(x.mean() / np.sqrt(var / n)))


if "p_is_hac" not in search.columns:
    raise SystemExit("сначала запустите add_hac.py")

# Победители оцениваются на ЧИСТОМ OOS-прогоне (R["oos_returns"]), а не на
# срезе непрерывного трека: именно этот прогон и есть единственная
# out-of-sample оценка.
conf = []
for key in table.index:
    r = R["oos_returns"][key]
    conf.append({"label": key, "p_hac": hac_p(r), "p_boot": boot_p(r, n=4000)})
conf = pd.DataFrame(conf)
conf.to_pickle("confirm.pkl")

ledgers = {}
for name, p, labels, note in [
    ("поиск: 125 конфигураций, in-sample, HAC",
     search["p_is_hac"].to_numpy(float),
     (search["key"] + " " + search["config"]).tolist(), "основной"),
    ("поиск: 125 конфигураций, in-sample, блочный бутстрэп (B=600)",
     search["p_is"].to_numpy(float),
     (search["key"] + " " + search["config"]).tolist(), "порог разрешения 1/601 = 0.0017"),
    ("подтверждение: 12 победителей, чистый OOS, HAC",
     conf["p_hac"].to_numpy(float), conf["label"].tolist(), "основной"),
    ("подтверждение: 12 победителей, чистый OOS, блочный бутстрэп (B=4000)",
     conf["p_boot"].to_numpy(float), conf["label"].tolist(), "порог разрешения 1/4001"),
]:
    ok = np.isfinite(p)
    pv, lab = p[ok], [l for l, f in zip(labels, ok) if f]
    ledgers[name] = {
        "m": len(pv), "raw": int((pv <= 0.05).sum()), "note": note,
        "holm": correct(pv, method="holm", labels=lab),
        "bh": correct(pv, method="benjamini_hochberg", labels=lab),
        "by": correct(pv, method="benjamini_yekutieli", labels=lab),
        "bonf": correct(pv, method="bonferroni", labels=lab),
        "efd": expected_false_discoveries(pv), "pi0": estimate_pi0(pv),
        "min_p": float(np.min(pv)),
    }

# =====================================================================
# 3. WHITE REALITY CHECK / HANSEN SPA
# =====================================================================
snoop = {}
groups = {
    "NG (все конфигурации односоставных семейств, IS)":
        [(k, r) for k, r in table.iterrows() if r["market"] == "NG"],
    "BR (все конфигурации односоставных семейств, IS)":
        [(k, r) for k, r in table.iterrows() if r["market"] == "BR"],
    "NG+BR (панельные семейства, IS)":
        [(k, r) for k, r in table.iterrows() if r["market"] == "NG+BR"],
}
for gname, cells in groups.items():
    if not cells: continue
    frames = []
    for key, row in cells:
        m = R["matrices"][key]
        is_idx, _ = split(m.index)
        sub = m.loc[is_idx].rename(columns=lambda c, k=key: f"{k} {c}")
        frames.append(sub)
    joint = pd.concat(frames, axis=1).dropna(how="any")
    if joint.shape[0] < 100 or joint.shape[1] < 2: continue
    snoop[gname] = {
        "rc": reality_check(joint, n_bootstrap=1000, seed=RNG_SEED),
        "spa": superior_predictive_ability(joint, n_bootstrap=1000, seed=RNG_SEED),
        "shape": joint.shape,
    }

# победители на OOS — проверка «лучший из 12 бьёт ноль?»
win_oos = pd.concat({k: R["oos_returns"][k] for k in table.index}, axis=1).dropna(how="any")
if win_oos.shape[1] >= 2:
    snoop["12 победителей, общий OOS-период"] = {
        "rc": reality_check(win_oos, n_bootstrap=1000, seed=RNG_SEED),
        "spa": superior_predictive_ability(win_oos, n_bootstrap=1000, seed=RNG_SEED),
        "shape": win_oos.shape,
    }

# =====================================================================
# 4. СЕРИИ НЕГАТИВНЫХ ИСХОДОВ
# =====================================================================
def streaks_of(flags):
    runs, cur = [], 0
    for f in flags:
        if f: cur += 1
        else:
            if cur: runs.append(cur)
            cur = 0
    if cur: runs.append(cur)
    return runs

def streak_analysis(monthly, label, seed=RNG_SEED, n_boot=4000):
    r = np.asarray(monthly, dtype=float); r = r[np.isfinite(r)]
    n = len(r)
    flags = r < 0
    runs = streaks_of(flags)
    obs_max = max(runs) if runs else 0
    q = float(flags.mean())
    # аналитическая оценка при независимости (q — доля убыточных месяцев)
    # P(нет серии >= k) приближённо через цепь Маркова на (k) состояниях
    def p_run_at_least(k, n_=n, q_=q):
        if k <= 0: return 1.0
        if q_ <= 0: return 0.0
        if q_ >= 1: return 1.0
        # ДП: вероятность НЕ встретить серию длины k
        state = np.zeros(k); state[0] = 1.0
        for _ in range(n_):
            nxt = np.zeros(k)
            nxt[0] += state.sum() * (1 - q_)
            nxt[1:] += state[:-1] * q_
            lost = state[-1] * q_          # серия достигла k — поглощение
            state = nxt
            state[0] -= 0.0
            if lost:
                state = state / 1.0
                # вычитаем поглощённую массу
                tot = state.sum() + lost
                state = state  # масса уже не содержит поглощённые пути
        return float(1.0 - state.sum())
    # блочный бутстрэп нулевой гипотезы (нулевое среднее, сохранённая зависимость)
    rng = np.random.default_rng(seed)
    b = max(1.0, optimal_block_length(r)) if n >= 20 else 1.0
    centred = r - r.mean()
    maxes = np.empty(n_boot)
    for i in range(n_boot):
        idx, pos = [], 0
        while pos < n:
            start = rng.integers(0, n)
            length = max(1, int(rng.geometric(1.0 / b)))
            idx.extend(((start + np.arange(length)) % n).tolist())
            pos += length
        sample = centred[np.asarray(idx[:n])]
        rr = streaks_of(sample < 0)
        maxes[i] = max(rr) if rr else 0
    return {
        "label": label, "n_months": n, "neg_months": int(flags.sum()), "share_neg": q,
        "max_streak": obs_max, "mean_streak": float(np.mean(runs)) if runs else 0.0,
        "n_runs": len(runs),
        "p_markov_ge_obs": p_run_at_least(obs_max),
        "p_markov_ge_3": p_run_at_least(3), "p_markov_ge_4": p_run_at_least(4),
        "p_markov_ge_6": p_run_at_least(6),
        "null_mean_max": float(maxes.mean()),
        "null_q95_max": float(np.quantile(maxes, 0.95)),
        "p_boot_ge_obs": float((maxes >= obs_max).mean()),
        "worst_month": float(r.min()), "best_month": float(r.max()),
    }

streak_rows = [streak_analysis(row["monthly"], key) for key, row in table.iterrows()]

# равновзвешенный портфель победителей
port = win_oos.mean(axis=1)
port_monthly = (1.0 + port).resample("ME").prod() - 1.0
streak_rows.append(streak_analysis(port_monthly, "ПОРТФЕЛЬ (12 равных весов)"))
# портфель только из выживших по вердикту — заполним позже
streaks = pd.DataFrame(streak_rows)

# =====================================================================
# 5. ВЕРДИКТЫ ФРЕЙМВОРКА
# =====================================================================
verdicts = {}
for key, row in table.iterrows():
    fam, mkt = row["family"], row["market"]
    matrix = R["matrices"][key]
    is_idx, oos_idx = split(matrix.index)
    ppy = row["ppy"]
    is_sr = matrix.loc[is_idx].apply(lambda c: sharpe_ratio(c.dropna(), periods_per_year=ppy))
    plateau = float((is_sr > 0).mean())
    oos = R["oos_returns"][key]
    # режимы волатильности: терцили 20-дневной реализованной волатильности
    vol = oos.rolling(20).std()
    try:
        bucket = pd.qcut(vol, 3, labels=False, duplicates="drop")
        regime = oos.groupby(bucket).mean()
        pos_regime = float((regime > 0).mean())
    except Exception:
        pos_regime = np.nan
    yearly = (1.0 + oos).resample("YE").prod() - 1.0
    tot = float((1.0 + oos).prod() - 1.0)
    best_share = float(yearly.max() / yearly.sum()) if yearly.sum() > 0 else np.nan
    metrics = {
        "oos_sharpe": row["sharpe_ratio"],
        "deflated_sharpe_ratio": row["dsr_programme"],
        "pbo": row["pbo"],
        "cpcv_median_sharpe": row["cpcv_median"],
        "cpcv_positive_fraction": row["cpcv_positive"],
        "wf_efficiency": row["wf_efficiency"],
        "wf_profitable_fraction": row["wf_profitable"],
        "bootstrap_sharpe_ci_low": row["bootstrap_lo"],
        "conservative_sharpe": row["sharpe_conservative"],
        "stressed_sharpe": row["sharpe_stressed"],
        "n_oos_trades": row["n_trades"],
        "plateau_fraction": plateau,
        "positive_regime_fraction": pos_regime,
        "best_year_return_share": best_share,
    }
    grid = GRIDS[fam]
    space = {k: sorted({g[k] for g in grid}) for k in grid[0]}
    hyp = make_hypothesis(
        f"{fam}@{mkt}",
        statement=f"Семейство {fam} даёт положительную риск-скорректированную доходность на {mkt} после издержек.",
        mechanism=("Трендовые и кэрри-эффекты на фьючерсах связаны с медленной диффузией информации, "
                   "хеджерским давлением и структурой срочной кривой; эти механизмы описаны в литературе "
                   "и проверяемы на независимом периоде."),
        expected_regime="Должно работать в трендовых и волатильных режимах; не должно — в боковых рынках с высокими издержками.",
        strategy=fam, param_space=space, symbol=mkt,
        prior_hypotheses_on_data=max(TOTAL - int(row["n_configs"]), 0),
    )
    verdicts[key] = (decide_verdict(hyp, metrics, []), metrics, plateau, pos_regime, best_share, tot)

pickle.dump({"search": search, "confirm": conf, "streaks": streaks,
             "verdicts": {k: v[0].to_dict() for k, v in verdicts.items()},
             "extra": {k: {"plateau": v[2], "pos_regime": v[3], "best_year_share": v[4],
                           "total_return": v[5]} for k, v in verdicts.items()},
             "ledgers": {k: {kk: (vv.to_dict() if hasattr(vv, "to_dict") else vv)
                             for kk, vv in v.items()} for k, v in ledgers.items()},
             "snoop": {k: {"rc": v["rc"].to_dict(), "spa": v["spa"].to_dict(), "shape": v["shape"]}
                       for k, v in snoop.items()}},
            open("final.pkl", "wb"))

# =====================================================================
# ВЫВОД
# =====================================================================
say()
say("=== МУЛЬТИТЕСТ ===")
for name, L in ledgers.items():
    say(f"{name}: m={L['m']}, p<=0.05 без поправки: {L['raw']}, "
        f"Holm: {L['holm'].n_rejected}, BH: {L['bh'].n_rejected}, BY: {L['by'].n_rejected}, "
        f"Bonferroni: {L['bonf'].n_rejected}, ожидаемо ложных: {L['efd']:.1f}, "
        f"pi0={L['pi0']:.2f}, min p={L['min_p']:.4f}")
    say(f"   выжившие по Holm: {L['holm'].survivors or '—'}")
    say(f"   выжившие по BH:   {L['bh'].survivors or '—'}")

say()
say("=== RC / SPA ===")
for gname, s in snoop.items():
    say(f"{gname}  [{s['shape'][0]}x{s['shape'][1]}]")
    say("   " + s["rc"].explain())
    say("   " + s["spa"].explain())

say()
say("=== СЕРИИ НЕГАТИВНЫХ МЕСЯЦЕВ ===")
for _, r in streaks.iterrows():
    say(f"{r['label']:30} мес {r['n_months']:3.0f} | убыт {r['neg_months']:3.0f} "
        f"({r['share_neg']:.0%}) | макс серия {r['max_streak']:.0f} | нуль: сред {r['null_mean_max']:.1f} "
        f"q95 {r['null_q95_max']:.0f} | p(серия>=набл)={r['p_boot_ge_obs']:.2f} | "
        f"P(>=4 мес подряд)={r['p_markov_ge_4']:.2f} | P(>=6)={r['p_markov_ge_6']:.2f}")

say()
say("=== ВЕРДИКТЫ ===")
for key, (v, m, pl, pr, by, tot) in verdicts.items():
    failed = [g.name for g in v.gates if g.passed is False]
    say(f"{key:28} {v.label:8} | провалено шлюзов {len(failed)}/{len(v.gates)}: {', '.join(failed) or '—'}")
    if v.falsified_by:
        for f in v.falsified_by:
            say(f"      фальсифицировано: {f}")

open("final_log.txt", "w").write("\n".join(out))
print("\nсохранено: final.pkl, search.pkl, final_log.txt")
