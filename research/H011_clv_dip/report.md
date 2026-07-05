# H011: clv_dip

**Вердикт:** `archived` — fails walk-forward (efficiency -1.78 < 0.3)
**Дата:** 2026-07-05 05:19 UTC

## Идея
CLV-капитуляция в аптренде. ПЕРЕ-АУДИТ: исходный WF eff -1.78 вычислен с inf-Sharpe багом.

## Правила
Candle-position reversion: a close pinned to the bottom of its daily range (close-location-value below a floor) while the long trend is up marks intraday capitulation; buy it and exit when price reclaims the short moving average.

## Лучшие параметры (по robust score)
```json
{
  "clv_floor": 0.3,
  "trend_window": "100",
  "exit_window": "20"
}
```

## Результаты

| метрика | значение |
|---|---|
| комбинаций в сетке | 27 |
| OOS Sharpe (best robust) | 0.48 |
| OOS CAGR % | 5.2 |
| OOS MaxDD % | 14.2 |
| OOS сделок | 26 |
| плато: доля комбо с OOS Sharpe>0 | 67% |
| walk-forward efficiency | -1.78 |
| WF прибыльных окон | 25% |
| MC P(разорения) @f=0.25 | 0.0000 |
| MC медианная просадка @f=0.25 | 10.0% |
| флагов переобучения (этап-02 методика) | 0 |

## Топ-5 конфигураций

|   param_clv_floor |   param_trend_window |   param_exit_window |   is_sharpe_ratio |   oos_sharpe_ratio |   robust_score |
|------------------:|---------------------:|--------------------:|------------------:|-------------------:|---------------:|
|               0.3 |                  100 |                  20 |            -0.104 |              0.478 |          0.806 |
|               0.3 |                  100 |                  10 |            -0.161 |              0.343 |          0.781 |
|               0.1 |                  100 |                  10 |            -0.386 |              0.499 |          0.739 |
|               0.3 |                  150 |                  20 |            -0.232 |              0.259 |          0.668 |
|               0.2 |                  150 |                  20 |            -0.328 |              0.283 |          0.614 |

## Выводы
Re-audit итерации 4 после фикса WF-отбора.

Артефакты: `sweep.csv` (все конфигурации), `walk_forward.csv`, `equity.png`, `param_heatmap.png`.