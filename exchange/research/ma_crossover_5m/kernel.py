"""Событийное ядро бэктеста.

Почему не ``vbt.Portfolio.from_signals`` на всей сетке
-----------------------------------------------------
3 471 552 конфигурации × 605 957 баров — это 2·10¹² ячеек сигнальной матрицы
(~2 ТБ булевых значений) только для одного инструмента. VectorBT сам считает
всё в numba; здесь использована та же техника, но по **событиям**, а не по
барам: сделка в этой стратегии может открыться только на пересечении ценой
быстрой средней, поэтому достаточно пройти по списку пересечений, а не по всем
барам. Корректность ядра проверяется против VectorBT на случайной выборке
конфигураций (``validate_vbt.py``): сделки и PnL должны совпасть.

Правила (раздел 6–7 задания)
----------------------------
* Пересечение: сторона ``side_t = +1`` при ``C_t > MA_t``, ``−1`` при ``C_t < MA_t``,
  при равенстве сторона сохраняется. Пересечение вверх — смена стороны с −1 на
  +1 между двумя последовательными закрытыми свечами (касание без перехода
  пересечением не считается). Пересечения на первом баре нового контракта
  исключаются — это сравнение через склейку.
* Long: пересечение вверх и медленная средняя ``вверх`` или ``нейтрально``;
  short: пересечение вниз и ``вниз`` или ``нейтрально``; только из flat.
* Исполнение — Open следующей свечи. 1 контракт, без издержек.
* Выход A: обратное пересечение быстрой средней.
  Выход B: медленная средняя переходит в противоположное состояние
  (переход в нейтраль позицию не закрывает).
* Разворот запрещён: бар, на котором принято решение о выходе, не может
  открыть новую сделку; следующая сделка — только по пересечению на более
  позднем баре.
* Позиция, открытая на конец данных, закрывается по последнему Close.

Учёт
----
Сделка относится к периоду (месяц, IS/OOS) по бару **исполнения выхода**:
in-sample метрики не используют ни одной цены из out-of-sample.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

# Индексы статистик в последней оси out_stats[config, bucket, stat]
PNL, NTR, WINS, GP, GL, MAXDD, MAXLS, SUMDUR, NLONG, PNLLONG, OPENEND = range(11)
N_STATS = 11
BUCKETS = ("full", "is", "oos")


@njit(cache=True)
def cross_events(close, ma, is_roll):
    """Бары и направления пересечений Close с MA (правило «стороны»)."""
    n = close.shape[0]
    bars = np.empty(n, dtype=np.int32)
    dirs = np.empty(n, dtype=np.int8)
    k = 0
    side = 0
    for t in range(n):
        m = ma[t]
        if np.isnan(m):
            side = 0
            continue
        c = close[t]
        if c > m:
            new = 1
        elif c < m:
            new = -1
        else:
            new = side
        if side != 0 and new != 0 and new != side and not is_roll[t]:
            bars[k] = t
            dirs[k] = new
            k += 1
        side = new
    return bars[:k].copy(), dirs[:k].copy()


@njit(cache=True)
def next_state_index(state, target):
    """nxt[t] = min{j >= t : state[j] == target}, иначе n."""
    n = state.shape[0]
    nxt = np.empty(n + 1, dtype=np.int32)
    nxt[n] = n
    for t in range(n - 1, -1, -1):
        nxt[t] = t if state[t] == target else nxt[t + 1]
    return nxt


@njit(cache=True)
def _record(stats, b, pnl, dur, d, open_end, eq, peak, ls):
    """Добавить сделку в корзину b; eq/peak/ls — состояние кривой корзины."""
    stats[b, PNL] += pnl
    stats[b, NTR] += 1.0
    if pnl > 0:
        stats[b, WINS] += 1.0
        stats[b, GP] += pnl
    else:
        stats[b, GL] -= pnl
    stats[b, SUMDUR] += dur
    if d > 0:
        stats[b, NLONG] += 1.0
        stats[b, PNLLONG] += pnl
    stats[b, OPENEND] += open_end
    eq[b] += pnl
    if eq[b] > peak[b]:
        peak[b] = eq[b]
    dd = peak[b] - eq[b]
    if dd > stats[b, MAXDD]:
        stats[b, MAXDD] = dd
    if pnl < 0:
        ls[b] += 1
        if ls[b] > stats[b, MAXLS]:
            stats[b, MAXLS] = ls[b]
    else:
        ls[b] = 0


@njit(cache=True)
def simulate(ev_bar, ev_dir, state, nxt_down, nxt_up, p_open, p_close, month,
             oos_bar, mult, model, stats, monthly, trades_out):
    """Одна конфигурация. model: 0 — выход A, 1 — выход B.

    stats: (3, N_STATS) — full / IS / OOS; monthly: (n_months,) PnL по месяцу выхода.
    trades_out: (m, 6) или (0, 6) — журнал сделок для проверки против VectorBT:
    [бар входа (исполнение), бар выхода (исполнение), направление, цена входа,
    цена выхода, PnL]. Возвращает число сделок.
    """
    n = p_open.shape[0]
    E = ev_bar.shape[0]
    eq = np.zeros(3)
    peak = np.zeros(3)
    ls = np.zeros(3)
    log = trades_out.shape[0] > 0
    ntr = 0
    i = 0
    while i < E:
        c = ev_bar[i]
        d = ev_dir[i]
        s = state[c]
        if not ((d > 0 and (s == 1 or s == 0)) or (d < 0 and (s == -1 or s == 0))):
            i += 1
            continue
        if c + 1 >= n:
            break
        ein = c + 1
        pin = p_open[ein]
        open_end = 0.0
        # ---- поиск бара решения о выходе ---------------------------------
        if model == 0:
            j = i + 1
            while j < E and ev_dir[j] != -d:
                j += 1
            if j < E:
                xdec = ev_bar[j]
            else:
                xdec = n
            nxt_i = j + 1
        else:
            if d > 0:
                xdec = nxt_down[c + 1]
            else:
                xdec = nxt_up[c + 1]
            # следующая сделка — только по пересечению строго после xdec
            lo = i + 1
            hi = E
            while lo < hi:
                mid = (lo + hi) // 2
                if ev_bar[mid] <= xdec:
                    lo = mid + 1
                else:
                    hi = mid
            nxt_i = lo
        # ---- исполнение выхода ------------------------------------------
        if xdec + 1 < n:
            eout = xdec + 1
            pout = p_open[eout]
        else:
            eout = n - 1
            pout = p_close[n - 1]
            open_end = 1.0
        pnl = d * (pout - pin) * mult
        dur = float(eout - ein)
        _record(stats, 0, pnl, dur, d, open_end, eq, peak, ls)
        b = 1 if eout < oos_bar else 2
        _record(stats, b, pnl, dur, d, open_end, eq, peak, ls)
        monthly[month[eout]] += pnl
        if log and ntr < trades_out.shape[0]:
            trades_out[ntr, 0] = ein
            trades_out[ntr, 1] = eout
            trades_out[ntr, 2] = d
            trades_out[ntr, 3] = pin
            trades_out[ntr, 4] = pout
            trades_out[ntr, 5] = pnl
        ntr += 1
        i = nxt_i
    return ntr


@njit(parallel=True, cache=True)
def run_fast_block(ev_bar_all, ev_dir_all, ev_off, state, nxt_down, nxt_up,
                   p_open, p_close, month, oos_bar, mult, out_stats, out_monthly):
    """Все быстрые средние × оба выхода против одного состояния медленной.

    out_stats: (2, n_fast, 3, N_STATS); out_monthly: (2, n_fast, n_months).
    """
    n_fast = ev_off.shape[0] - 1
    dummy = np.zeros((0, 6))
    for job in prange(2 * n_fast):
        model = job // n_fast
        f = job % n_fast
        a = ev_off[f]
        b = ev_off[f + 1]
        simulate(ev_bar_all[a:b], ev_dir_all[a:b], state, nxt_down, nxt_up,
                 p_open, p_close, month, oos_bar, mult, model,
                 out_stats[model, f], out_monthly[model, f], dummy)
