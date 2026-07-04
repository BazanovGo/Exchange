# H002: vwap_adx_trend

**Вердикт:** `archived` — too few OOS trades (6 < 8); no parameter plateau (19% of grid works OOS)
**Дата:** 2026-07-04 13:34 UTC

## Идея
Объемно-якорный тренд: кросс цены выше rolling VWAP означает, что средняя позиция толпы в минусе; растущий ADX подтверждает направленное давление. Комбинация 'VWAP + ADX'.

## Правила
Volume-anchored trend following: buy when the close crosses above the rolling VWAP with ADX above a threshold and rising (the crowd's average position is underwater and directional pressure is building); exit when price crosses back below VWAP.

## Лучшие параметры (по robust score)
```json
{
  "vwap_window": "50",
  "adx_window": "10",
  "adx_threshold": 15.0
}
```

## Результаты

| метрика | значение |
|---|---|
| комбинаций в сетке | 36 |
| OOS Sharpe (best robust) | 0.48 |
| OOS CAGR % | 4.6 |
| OOS MaxDD % | 9.7 |
| OOS сделок | 6 |
| плато: доля комбо с OOS Sharpe>0 | 19% |
| walk-forward efficiency | nan |
| WF прибыльных окон | 0% |
| MC P(разорения) @f=0.25 | 0.0000 |
| MC медианная просадка @f=0.25 | 4.8% |
| флагов переобучения (этап-02 методика) | 1 |

## Топ-5 конфигураций

|   param_vwap_window |   param_adx_window |   param_adx_threshold |   is_sharpe_ratio |   oos_sharpe_ratio |   robust_score |
|--------------------:|-------------------:|----------------------:|------------------:|-------------------:|---------------:|
|                  50 |                 10 |                    15 |             0.674 |              0.479 |          0.607 |
|                  30 |                 10 |                    15 |             0.8   |              0.334 |          0.583 |
|                  30 |                 10 |                    20 |             0.546 |              0.508 |          0.521 |
|                  50 |                 10 |                    20 |             0.64  |              0.238 |          0.458 |
|                  50 |                 14 |                    15 |             0.176 |              0.397 |          0.376 |

## Выводы
—

Артефакты: `sweep.csv` (все конфигурации), `walk_forward.csv`, `equity.png`, `param_heatmap.png`.