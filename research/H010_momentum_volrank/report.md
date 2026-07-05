# H010: momentum_volrank

**Вердикт:** `archived` — stage-02 overfit flags = 2; no OOS edge (Sharpe 0.04 < 0.2); fails walk-forward (efficiency -0.45 < 0.3)
**Дата:** 2026-07-05 04:57 UTC

## Идея
Доработка H004 (flags=0, но плато 31% из-за хрупкого множителя объема): подтверждение объемом через безмасштабный перцентильный ранг за долгое окно.

## Правила
Refined H004 (its fragile axis was the volume multiplier): momentum turning positive is confirmed by the percentile rank of volume over a long lookback — a scale-free condition. Exit when momentum drops back through zero.

## Лучшие параметры (по robust score)
```json
{
  "roc_window": "10",
  "rank_lookback": "180",
  "rank_floor": 0.6
}
```

## Результаты

| метрика | значение |
|---|---|
| комбинаций в сетке | 36 |
| OOS Sharpe (best robust) | 0.04 |
| OOS CAGR % | 0.0 |
| OOS MaxDD % | 10.6 |
| OOS сделок | 18 |
| плато: доля комбо с OOS Sharpe>0 | 39% |
| walk-forward efficiency | -0.45 |
| WF прибыльных окон | 25% |
| MC P(разорения) @f=0.25 | 0.0000 |
| MC медианная просадка @f=0.25 | 11.0% |
| флагов переобучения (этап-02 методика) | 2 |

## Топ-5 конфигураций

|   param_roc_window |   param_rank_lookback |   param_rank_floor |   is_sharpe_ratio |   oos_sharpe_ratio |   robust_score |
|-------------------:|----------------------:|-------------------:|------------------:|-------------------:|---------------:|
|                 10 |                   180 |                0.6 |            -0.02  |              0.041 |          0.583 |
|                 20 |                   180 |                0.5 |            -0.188 |              0.894 |          0.573 |
|                 10 |                    80 |                0.5 |             0.444 |              0.174 |          0.559 |
|                 20 |                   180 |                0.8 |            -0.276 |              0.478 |          0.496 |
|                 10 |                   180 |                0.7 |             0.304 |             -0.105 |          0.489 |

## Выводы
Урок H004: заменить хрупкую ось сетки на робастную нормировку.

Артефакты: `sweep.csv` (все конфигурации), `walk_forward.csv`, `equity.png`, `param_heatmap.png`.