"""Устойчивость: плато, отбор, walk-forward, PBO, Reality Check / SPA, DSR."""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import ndimage, stats as sps

# Окрестность для плато: ±3 периода быстрой и ±7 периодов медленной средней
# (7 × 15 соседей). Задана до расчётов; в отчёте есть чувствительность к ней.
PLATEAU = (15, 7)      # (медленная, быстрая) — порядок осей сетки


def plateau(values: np.ndarray, shape: tuple, size=PLATEAU) -> np.ndarray:
    """Среднее значение метрики по окрестности в плоскости (slow_period × fast_period).

    Сетка имеет оси (method, slow_type, slow_period, exit, fast_type, fast_period);
    сглаживание идёт только по осям периодов, внутри одного «семейства»
    (метод × типы × выход). NaN не участвуют в среднем.
    """
    v = values.reshape(shape).astype(np.float64)
    ok = np.isfinite(v)
    filled = np.where(ok, v, 0.0)
    footprint = (1, 1, size[0], 1, 1, size[1])
    # mode="constant": за краем сетки соседей нет. Первая версия использовала
    # "nearest", который размножает крайние значения: у точки на границе
    # окрестность становилась фиктивной, и шум края проходил как «плато» —
    # так на NG была выбрана конфигурация с медленной = 250, ровно на краю.
    s = ndimage.uniform_filter(filled, size=footprint, mode="constant", cval=0.0)
    c = ndimage.uniform_filter(ok.astype(np.float64), size=footprint, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(c > 0, s / c, np.nan)
    return out.reshape(-1).astype(np.float32)


def sign_stability(values: np.ndarray, shape: tuple, size=PLATEAU) -> np.ndarray:
    """Доля соседей того же знака, что и сама конфигурация (карта устойчивости)."""
    pos = (np.nan_to_num(values) > 0).astype(np.float64)
    share = plateau(pos, shape, size)
    return np.where(np.nan_to_num(values) > 0, share, 1.0 - share).astype(np.float32)


# --------------------------------------------------------------- walk-forward
def walk_forward(monthly: np.ndarray, shape: tuple, *, train: int, test: int,
                 criterion: str, min_active: int = 6, chunk: int = 400_000):
    """Переоптимизация на скользящем окне по всей сетке.

    На каждом шаге выбирается конфигурация с лучшим критерием на ``train``
    предыдущих месяцах; её PnL за следующие ``test`` месяцев — результат
    walk-forward. Ничего из тестового окна в выбор не попадает.

    criterion: "pnl" — сумма PnL; "sharpe" — месячный Шарп; "plateau" — Шарп,
    сглаженный по окрестности периодов (устойчивый выбор).
    ``min_active`` — минимум месяцев с ненулевым PnL в окне обучения, чтобы
    почти не торгующая конфигурация не выигрывала за счёт нулевой дисперсии.
    """
    N, T = monthly.shape
    steps = []
    t = train
    while t < T:
        lo, hi = t - train, min(t + test, T)
        s1 = np.empty(N); s2 = np.empty(N); act = np.empty(N)
        for a in range(0, N, chunk):
            b = min(a + chunk, N)
            x = np.asarray(monthly[a:b, lo:t], dtype=np.float64)
            s1[a:b] = x.sum(1); s2[a:b] = (x * x).sum(1); act[a:b] = (x != 0).sum(1)
        mean = s1 / train
        var = np.maximum(s2 / train - mean ** 2, 0.0) * train / (train - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sh = np.where(var > 0, mean / np.sqrt(var), np.nan)
        sh[act < min_active] = np.nan
        if criterion == "pnl":
            score = np.where(act >= min_active, s1, -np.inf)
        elif criterion == "sharpe":
            score = np.nan_to_num(sh, nan=-np.inf)
        elif criterion == "plateau":
            score = np.nan_to_num(plateau(sh, shape), nan=-np.inf)
        else:
            raise ValueError(criterion)
        best = int(np.argmax(score))
        test_pnl = np.asarray(monthly[best, t:hi], dtype=np.float64)
        steps.append({"train_start": lo, "test_start": t, "test_end": hi, "config": best,
                      "train_score": float(score[best]), "train_pnl": float(s1[best]),
                      "test_pnl": float(test_pnl.sum()), "test_months": test_pnl})
        t = hi
    return steps


# ------------------------------------------------------------------- PBO (CSCV)
def pbo_cscv(monthly: np.ndarray, *, n_blocks: int = 16, n_splits: int = 1000,
             seed: int = 0, chunk: int = 400_000, months: slice | None = None):
    """Вероятность переобучения отбора (Bailey, Borwein, López de Prado, Zhu).

    Месяцы режутся на ``n_blocks`` смежных блоков; для случайных ``n_splits``
    из C(n_blocks, n_blocks/2) разбиений лучший по Шарпу на половине блоков
    оценивается рангом на другой половине. PBO — доля разбиений, где он хуже
    медианы. Блоки смежные: перемешивание месяцев разрушило бы автокорреляцию.
    """
    X = monthly if months is None else monthly[:, months]
    N, T = X.shape
    edges = np.linspace(0, T, n_blocks + 1).round().astype(int)
    S1 = np.empty((N, n_blocks)); S2 = np.empty((N, n_blocks))
    cnt = np.diff(edges).astype(np.float64)
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        x = np.asarray(X[a:b], dtype=np.float64)
        for j in range(n_blocks):
            seg = x[:, edges[j]:edges[j + 1]]
            S1[a:b, j] = seg.sum(1); S2[a:b, j] = (seg * seg).sum(1)
    rng = np.random.default_rng(seed)
    combos = list(itertools.combinations(range(n_blocks), n_blocks // 2))
    pick = rng.choice(len(combos), size=min(n_splits, len(combos)), replace=False)
    logits, degr = [], []
    for c in pick:
        w = np.zeros(n_blocks); w[list(combos[c])] = 1.0
        out = []
        for ww in (w, 1.0 - w):
            n_ = cnt @ ww
            m = (S1 @ ww) / n_
            v = np.maximum((S2 @ ww) / n_ - m * m, 0.0) * n_ / (n_ - 1)
            with np.errstate(divide="ignore", invalid="ignore"):
                out.append(np.where(v > 0, m / np.sqrt(v), -np.inf))
        is_sh, oos_sh = out
        best = int(np.argmax(is_sh))
        rank = (oos_sh < oos_sh[best]).sum() + 0.5 * ((oos_sh == oos_sh[best]).sum() - 1)
        omega = (rank + 1) / (N + 1)
        logits.append(math.log(omega / (1 - omega)))
        degr.append((float(is_sh[best]), float(oos_sh[best])))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "logits": logits,
            "is_oos_pairs": np.array(degr), "n_splits": len(pick), "n_blocks": n_blocks}


# ----------------------------------------------------- Reality Check / SPA
def _stationary_counts(T: int, B: int, block: float, rng) -> np.ndarray:
    """Матрица (T, B): сколько раз месяц t вошёл в бутстрэп-выборку b."""
    W = np.zeros((T, B))
    p = 1.0 / block
    for b in range(B):
        t = rng.integers(T)
        for _ in range(T):
            W[t, b] += 1
            t = rng.integers(T) if rng.random() < p else (t + 1) % T
    return W


def reality_check_spa(monthly: np.ndarray, *, months: slice, B: int = 1000,
                      block: float | None = None, seed: int = 0, chunk: int = 40_000):
    """White's Reality Check и Hansen's SPA по ВСЕЙ сетке (бенчмарк — не торговать).

    Средние по бутстрэп-выборкам считаются как X @ W (W — матрица кратностей
    месяцев / T), поэтому 3.5 млн моделей обрабатываются кусками без хранения
    бутстрэп-распределения каждой модели.

    Студентизация SPA — bootstrap-t: наблюдаемая статистика делится на
    выборочное стандартное отклонение модели, а каждая бутстрэп-реплика — на
    стандартное отклонение СВОЕЙ выборки. Первая версия делила и то и другое на
    бутстрэп-оценку ω_k, как в описании Хансена, и на чистом шуме отвергала
    нулевую гипотезу в 60–100% случаев: бутстрэп-дисперсия среднего на малом T
    смещена вниз, наблюдаемый t раздувается, а реплики по построению имеют
    единичную дисперсию; максимум по миллионам моделей усиливает разницу.
    Калибровка проверяется в test_robust.py.
    """
    N = monthly.shape[0]
    T = months.stop - months.start
    block = block or max(1.0, round(T ** (1 / 3)))
    rng = np.random.default_rng(seed)
    W = _stationary_counts(T, B, block, rng) / T
    sqT = math.sqrt(T)
    rc_boot = np.full(B, -np.inf)
    spa_boot = np.zeros(B)
    rc_stat, spa_stat = -np.inf, 0.0
    best_rc = best_spa = -1
    thr_c = math.sqrt(2.0 * math.log(max(math.log(T), 1.0001)) / T)
    corr = T / (T - 1.0)
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        x = np.asarray(monthly[a:b, months], dtype=np.float64)
        mu = x.mean(1)
        sd = x.std(1, ddof=1)
        sd = np.where(sd > 1e-12, sd, np.nan)
        bm = x @ W                                   # (n, B) бутстрэп-средние
        # RC (без студентизации, как у Уайта)
        k = int(np.argmax(mu))
        if sqT * mu[k] > rc_stat:
            rc_stat, best_rc = sqT * mu[k], a + k
        rc_boot = np.maximum(rc_boot, (sqT * (bm - mu[:, None])).max(0))
        # SPA, bootstrap-t
        with np.errstate(invalid="ignore", divide="ignore"):
            t = sqT * mu / sd
        if np.isfinite(t).any():
            k2 = int(np.nanargmax(t))
            if t[k2] > spa_stat:
                spa_stat, best_spa = float(t[k2]), a + k2
        bv = np.maximum((x * x) @ W - bm * bm, 0.0) * corr
        bsd = np.sqrt(bv)
        keep = mu >= -sd * thr_c
        centred = bm - np.where(keep, mu, 0.0)[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            bt = np.where(bsd > 1e-12, sqT * centred / bsd, np.nan)
        spa_boot = np.maximum(spa_boot, np.nan_to_num(np.nanmax(np.maximum(bt, 0.0), axis=0)))
        del bm, bv, bsd, centred, bt
    return {
        "rc_stat": float(rc_stat), "rc_p": float((np.sum(rc_boot >= rc_stat) + 1) / (B + 1)),
        "rc_best": int(best_rc),
        "spa_stat": float(spa_stat), "spa_p": float((np.sum(spa_boot >= spa_stat) + 1) / (B + 1)),
        "spa_best": int(best_spa), "T": T, "B": B, "block": block, "N": int(N),
    }


# ------------------------------------------------------------------------ DSR
def expected_max_sharpe(n_trials: float, var_sr: float) -> float:
    """Ожидаемый максимум n_trials нулевых Шарпов (Bailey & López de Prado, 2014)."""
    g = 0.5772156649
    if n_trials <= 1:
        return 0.0
    return math.sqrt(var_sr) * ((1 - g) * sps.norm.ppf(1 - 1 / n_trials)
                                + g * sps.norm.ppf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(x: np.ndarray, n_trials: float, var_sr: float) -> dict:
    """DSR по помесячному ряду PnL (Шарп на период, не годовой)."""
    x = np.asarray(x, dtype=np.float64)
    T = len(x)
    sd = x.std(ddof=1)
    sr = x.mean() / sd if sd > 0 else 0.0
    skew = float(sps.skew(x)) if sd > 0 else 0.0
    kurt = float(sps.kurtosis(x, fisher=False)) if sd > 0 else 3.0
    sr0 = expected_max_sharpe(n_trials, var_sr)
    den = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr * sr, 1e-12))
    z = (sr - sr0) * math.sqrt(T - 1) / den
    psr0 = (sr - 0.0) * math.sqrt(T - 1) / den
    return {"sr_monthly": sr, "sr_annual": sr * math.sqrt(12), "sr0_monthly": sr0,
            "sr0_annual": sr0 * math.sqrt(12), "dsr": float(sps.norm.cdf(z)),
            "psr": float(sps.norm.cdf(psr0)), "T": T, "n_trials": n_trials}


def effective_trials(monthly: np.ndarray, months: slice, sample: int = 4000, seed: int = 0):
    """Эффективное число независимых испытаний по спектру корреляций выборки.

    Participation ratio (Σλ)²/Σλ² для случайной выборки конфигураций — это
    число «независимых направлений», которое выборка покрывает. Это нижняя
    оценка: полная сетка их не меньше. Отчёт приводит DSR при обеих границах.
    """
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(monthly.shape[0], size=min(sample, monthly.shape[0]), replace=False))
    x = np.asarray(monthly[idx, months], dtype=np.float64)
    x = x[x.std(1) > 0]
    c = np.corrcoef(x)
    lam = np.linalg.eigvalsh(c)
    lam = lam[lam > 0]
    return float(lam.sum() ** 2 / (lam ** 2).sum())
