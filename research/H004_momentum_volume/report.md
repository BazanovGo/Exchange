# H004: momentum_volume

**Вердикт:** `archived` — no parameter plateau (31% of grid works OOS)
**Дата:** 2026-07-04 13:35 UTC

## Идея
Импульс с участием: разворот ROC в плюс на повышенном объеме отделяет приток новых денег от дрейфа. Комбинация 'Momentum + Volume Expansion'.

## Правила
Participation-confirmed momentum: buy when N-bar ROC turns positive while volume runs above its own average by a multiple (new money, not drift); exit when ROC falls back below zero.

## Лучшие параметры (по robust score)
```json
{
  "roc_window": "20",
  "vol_window": "10",
  "vol_multiplier": 1.1
}
```

## Результаты

| метрика | значение |
|---|---|
| комбинаций в сетке | 36 |
| OOS Sharpe (best robust) | 1.26 |
| OOS CAGR % | 13.4 |
| OOS MaxDD % | 5.9 |
| OOS сделок | 9 |
| плато: доля комбо с OOS Sharpe>0 | 31% |
| walk-forward efficiency | nan |
| WF прибыльных окон | 0% |
| MC P(разорения) @f=0.25 | 0.0000 |
| MC медианная просадка @f=0.25 | 5.6% |
| флагов переобучения (этап-02 методика) | 0 |

## Топ-5 конфигураций

|   param_roc_window |   param_vol_window |   param_vol_multiplier |   is_sharpe_ratio |   oos_sharpe_ratio |   robust_score |
|-------------------:|-------------------:|-----------------------:|------------------:|-------------------:|---------------:|
|                 20 |                 10 |                    1.1 |             0.356 |              1.255 |          0.826 |
|                 20 |                 40 |                    1.1 |             0.268 |              1.023 |          0.735 |
|                 20 |                 20 |                    1.1 |             0.268 |              0.351 |          0.614 |
|                 10 |                 40 |                    1.3 |            -0.269 |              0.586 |          0.473 |
|                 10 |                 20 |                    1.1 |            -0.311 |              0.35  |          0.423 |

## Выводы
—

Артефакты: `sweep.csv` (все конфигурации), `walk_forward.csv`, `equity.png`, `param_heatmap.png`.