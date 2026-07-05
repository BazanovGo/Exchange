# H019: zscore_ride

**Вердикт:** `promising`
**Дата:** 2026-07-05 06:10 UTC

## Идея
EXIT-НАПРАВЛЕНИЕ: тот же z-вход, что в H012, но выход не у среднего, а на положительном z-target (доехать до дальнего края колебания). Где живет MR-альфа — во входе или в выходе? Сравнение с H012 (WF eff 0.75) на той же серии.

## Правила
Exit-driven MR variant: identical z-score dip entry, but instead of exiting at the mean, hold until the z-score reaches a positive target on the far side (letting the oscillation complete). Tests whether MR alpha sits in the entry or in the exit.

## Лучшие параметры (по robust score)
```json
{
  "window": "40",
  "z_entry": 1.5,
  "z_target": 0.0
}
```

## Результаты

| метрика | значение |
|---|---|
| комбинаций в сетке | 48 |
| OOS Sharpe (best robust) | 1.52 |
| OOS CAGR % | 22.5 |
| OOS MaxDD % | 10.0 |
| OOS сделок | 11 |
| плато: доля комбо с OOS Sharpe>0 | 85% |
| walk-forward efficiency | 0.57 |
| WF прибыльных окон | 100% |
| MC P(разорения) @f=0.25 | 0.0000 |
| MC медианная просадка @f=0.25 | 0.5% |
| флагов переобучения (этап-02 методика) | 0 |

## Топ-5 конфигураций

|   param_window |   param_z_entry |   param_z_target |   is_sharpe_ratio |   oos_sharpe_ratio |   robust_score |
|---------------:|----------------:|-----------------:|------------------:|-------------------:|---------------:|
|             40 |             1.5 |              0   |             1.678 |              1.517 |          0.684 |
|             40 |             2   |              1.5 |             1.454 |              1.782 |          0.65  |
|             60 |             2   |              0   |             1.468 |              2.494 |          0.638 |
|             60 |             1.5 |              0.5 |             1.706 |              1.581 |          0.622 |
|             40 |             1.2 |              1   |             1.561 |              1.314 |          0.619 |

## Выводы
Единственное открытое направление итерации 1: альфа в выходах.

Артефакты: `sweep.csv` (все конфигурации), `walk_forward.csv`, `equity.png`, `param_heatmap.png`.