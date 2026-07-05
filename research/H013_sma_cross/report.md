# H013: sma_cross

**Вердикт:** `archived` — stage-02 overfit flags = 3; fails walk-forward (efficiency -0.27 < 0.3)
**Дата:** 2026-07-05 05:14 UTC

## Идея
POSITIVE CONTROL (trend following): SMA-кросс на серии с 2-режимным марковским дрейфом (персистентность 0.985). Пайплайн обязан дать positive вердикт.

## Правила
Golden/death cross of two simple moving averages.

## Лучшие параметры (по robust score)
```json
{
  "fast_window": "5",
  "slow_window": "50"
}
```

## Результаты

| метрика | значение |
|---|---|
| комбинаций в сетке | 19 |
| OOS Sharpe (best robust) | 0.41 |
| OOS CAGR % | 5.7 |
| OOS MaxDD % | 15.7 |
| OOS сделок | 11 |
| плато: доля комбо с OOS Sharpe>0 | 68% |
| walk-forward efficiency | -0.27 |
| WF прибыльных окон | 25% |
| MC P(разорения) @f=0.25 | 0.0000 |
| MC медианная просадка @f=0.25 | 12.2% |
| флагов переобучения (этап-02 методика) | 3 |

## Топ-5 конфигураций

|   param_fast_window |   param_slow_window |   is_sharpe_ratio |   oos_sharpe_ratio |   robust_score |
|--------------------:|--------------------:|------------------:|-------------------:|---------------:|
|                   5 |                  50 |             0.766 |              0.415 |          0.758 |
|                  10 |                  50 |             0.781 |              0.411 |          0.642 |
|                   5 |                 100 |             0.707 |              0.315 |          0.542 |
|                  20 |                 100 |             0.582 |              0.227 |          0.415 |
|                  20 |                 150 |             0.289 |              0.577 |          0.374 |

## Выводы
Повторный прогон после фикса inf-Sharpe бага в WF-отборе.

Артефакты: `sweep.csv` (все конфигурации), `walk_forward.csv`, `equity.png`, `param_heatmap.png`.