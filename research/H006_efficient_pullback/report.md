# H006: efficient_pullback

**Вердикт:** `archived` — fails walk-forward (efficiency -0.21 < 0.3)
**Дата:** 2026-07-05 05:19 UTC

## Идея
Пуллбэк к EMA в эффективном тренде (ER-фильтр). ПЕРЕ-АУДИТ: исходный вердикт (0% прибыльных WF-окон) вычислен с inf-Sharpe багом.

## Правила
Trend-quality pullback: when the path is efficient (Kaufman ER above a floor) and price is above the slow EMA, a dip touching the fast EMA is bought; exit when the close breaks the slow EMA or the trend loses efficiency.

## Лучшие параметры (по robust score)
```json
{
  "fast_window": "30",
  "slow_window": "100",
  "er_window": "30",
  "er_floor": 0.2
}
```

## Результаты

| метрика | значение |
|---|---|
| комбинаций в сетке | 81 |
| OOS Sharpe (best robust) | 1.11 |
| OOS CAGR % | 10.6 |
| OOS MaxDD % | 5.1 |
| OOS сделок | 9 |
| плато: доля комбо с OOS Sharpe>0 | 49% |
| walk-forward efficiency | -0.21 |
| WF прибыльных окон | 50% |
| MC P(разорения) @f=0.25 | 0.0000 |
| MC медианная просадка @f=0.25 | 1.4% |
| флагов переобучения (этап-02 методика) | 0 |

## Топ-5 конфигураций

|   param_fast_window |   param_slow_window |   param_er_window |   param_er_floor |   is_sharpe_ratio |   oos_sharpe_ratio |   robust_score |
|--------------------:|--------------------:|------------------:|-----------------:|------------------:|-------------------:|---------------:|
|                  30 |                 100 |                30 |              0.2 |             1.144 |              1.107 |          0.849 |
|                  30 |                  60 |                30 |              0.2 |             0.996 |              0.769 |          0.795 |
|                  30 |                  50 |                30 |              0.2 |             0.996 |              0.71  |          0.795 |
|                  10 |                 100 |                30 |              0.2 |             0.967 |              0.978 |          0.769 |
|                  20 |                 100 |                30 |              0.2 |             0.894 |              0.871 |          0.721 |

## Выводы
Re-audit итерации 4 после фикса WF-отбора.

Артефакты: `sweep.csv` (все конфигурации), `walk_forward.csv`, `equity.png`, `param_heatmap.png`.