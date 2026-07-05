# H014: roc_momentum

**Вердикт:** `archived` — stage-02 overfit flags = 3; no parameter plateau (25% of grid works OOS); fails walk-forward (efficiency 0.22 < 0.3)
**Дата:** 2026-07-05 05:14 UTC

## Идея
POSITIVE CONTROL (momentum, калибровка): ROC-momentum на марковской серии с персистентностью 0.995 (режимы ~200 баров — масштаб реальных бычьих/медвежьих фаз). H013 показал, что режимы ~67 баров короче периодов SMA-кросса; контроль должен быть находимым для подходящего таймфрейма класса.

## Правила
Rate-of-change momentum: enter when N-bar ROC pushes above a positive threshold, exit when it falls back through zero.

## Лучшие параметры (по robust score)
```json
{
  "window": "10",
  "threshold": 5.0
}
```

## Результаты

| метрика | значение |
|---|---|
| комбинаций в сетке | 16 |
| OOS Sharpe (best robust) | 0.91 |
| OOS CAGR % | 10.0 |
| OOS MaxDD % | 8.4 |
| OOS сделок | 8 |
| плато: доля комбо с OOS Sharpe>0 | 25% |
| walk-forward efficiency | 0.22 |
| WF прибыльных окон | 75% |
| MC P(разорения) @f=0.25 | 0.0000 |
| MC медианная просадка @f=0.25 | 3.7% |
| флагов переобучения (этап-02 методика) | 3 |

## Топ-5 конфигураций

|   param_window |   param_threshold |   is_sharpe_ratio |   oos_sharpe_ratio |   robust_score |
|---------------:|------------------:|------------------:|-------------------:|---------------:|
|             10 |                 5 |             1.773 |              0.908 |          0.591 |
|             10 |                 2 |             1.887 |              0.852 |          0.588 |
|             20 |                 0 |             1.498 |             -0.481 |          0.511 |
|             20 |                 8 |             1.476 |             -0.028 |          0.491 |
|             10 |                 0 |             1.664 |              0.675 |          0.381 |

## Выводы
Калибровка positive control: длина внедренных режимов должна соответствовать таймфрейму класса стратегий.

Артефакты: `sweep.csv` (все конфигурации), `walk_forward.csv`, `equity.png`, `param_heatmap.png`.