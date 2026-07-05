# H018: williams_r

**Вердикт:** `promising`
**Дата:** 2026-07-05 06:10 UTC

## Идея
РОБАСТНОСТЬ (positive leg): williams_r на ДРУГОЙ реализации OU-процесса (seed 23, те же параметры процесса). Ожидание: promising. Перенос между реализациями = структура, а не запоминание конкретного пути.

## Правила
Williams %R reversion: buy when %R falls below the oversold floor, exit when it climbs back above the midpoint.

## Лучшие параметры (по robust score)
```json
{
  "window": "20",
  "oversold": -90.0,
  "exit_level": -50.0
}
```

## Результаты

| метрика | значение |
|---|---|
| комбинаций в сетке | 36 |
| OOS Sharpe (best robust) | 1.48 |
| OOS CAGR % | 18.1 |
| OOS MaxDD % | 6.0 |
| OOS сделок | 12 |
| плато: доля комбо с OOS Sharpe>0 | 100% |
| walk-forward efficiency | 0.66 |
| WF прибыльных окон | 100% |
| MC P(разорения) @f=0.25 | 0.0000 |
| MC медианная просадка @f=0.25 | 0.5% |
| флагов переобучения (этап-02 методика) | 0 |

## Топ-5 конфигураций

|   param_window |   param_oversold |   param_exit_level |   is_sharpe_ratio |   oos_sharpe_ratio |   robust_score |
|---------------:|-----------------:|-------------------:|------------------:|-------------------:|---------------:|
|             20 |              -90 |                -50 |             2.253 |              1.484 |          0.808 |
|             14 |              -90 |                -50 |             1.594 |              1.683 |          0.659 |
|             14 |              -90 |                -40 |             1.643 |              1.564 |          0.646 |
|             28 |              -70 |                -50 |             1.502 |              1.68  |          0.64  |
|             28 |              -70 |                -40 |             1.405 |              1.886 |          0.634 |

## Выводы
Мульти-инструментный контроль promising-тройки, сторона OU cross-seed.

Артефакты: `sweep.csv` (все конфигурации), `walk_forward.csv`, `equity.png`, `param_heatmap.png`.