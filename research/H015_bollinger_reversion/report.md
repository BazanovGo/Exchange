# H015: bollinger_reversion

**Вердикт:** `promising`
**Дата:** 2026-07-05 05:19 UTC

## Идея
MR-направление (открыто H012): какая MR-конструкция устойчивее на MR-структуре? Bollinger-fade на той же OU-серии, где z-score дал promising. Полосы = z-score в ценовых координатах, но с иным выходом (средняя полоса).

## Правила
Fade band extremes: buy a close crossing below the lower Bollinger band, exit when price reverts to the middle band; optionally short the upper band.

## Лучшие параметры (по robust score)
```json
{
  "window": "50",
  "alpha": 1.5
}
```

## Результаты

| метрика | значение |
|---|---|
| комбинаций в сетке | 16 |
| OOS Sharpe (best robust) | 1.15 |
| OOS CAGR % | 16.7 |
| OOS MaxDD % | 9.5 |
| OOS сделок | 8 |
| плато: доля комбо с OOS Sharpe>0 | 75% |
| walk-forward efficiency | 0.69 |
| WF прибыльных окон | 100% |
| MC P(разорения) @f=0.25 | 0.0000 |
| MC медианная просадка @f=0.25 | 0.3% |
| флагов переобучения (этап-02 методика) | 0 |

## Топ-5 конфигураций

|   param_window |   param_alpha |   is_sharpe_ratio |   oos_sharpe_ratio |   robust_score |
|---------------:|--------------:|------------------:|-------------------:|---------------:|
|             50 |           1.5 |             1.734 |              1.146 |          0.657 |
|             50 |           2   |             1.111 |              1.415 |          0.605 |
|             30 |           2   |             1.351 |              1.263 |          0.598 |
|             30 |           1.5 |             1.721 |              1.094 |          0.565 |
|             20 |           2.5 |             0.971 |              1.461 |          0.508 |

## Выводы
Сравнение MR-конструкций на OU: z-score (H012, promising) vs Bollinger.

Артефакты: `sweep.csv` (все конфигурации), `walk_forward.csv`, `equity.png`, `param_heatmap.png`.