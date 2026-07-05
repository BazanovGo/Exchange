# H017: williams_r

**Вердикт:** `archived` — fails walk-forward (efficiency -0.55 < 0.3)
**Дата:** 2026-07-05 06:10 UTC

## Идея
РОБАСТНОСТЬ (negative leg): williams_r — лучшая MR-конструкция (H016 promising на OU) — на бесструктурном GBM. Ожидание: archived. Если 'работает' и здесь — красный флаг методики (ложные открытия).

## Правила
Williams %R reversion: buy when %R falls below the oversold floor, exit when it climbs back above the midpoint.

## Лучшие параметры (по robust score)
```json
{
  "window": "14",
  "oversold": -80.0,
  "exit_level": -40.0
}
```

## Результаты

| метрика | значение |
|---|---|
| комбинаций в сетке | 36 |
| OOS Sharpe (best robust) | 0.23 |
| OOS CAGR % | 1.9 |
| OOS MaxDD % | 12.7 |
| OOS сделок | 16 |
| плато: доля комбо с OOS Sharpe>0 | 42% |
| walk-forward efficiency | -0.55 |
| WF прибыльных окон | 25% |
| MC P(разорения) @f=0.25 | 0.0000 |
| MC медианная просадка @f=0.25 | 11.3% |
| флагов переобучения (этап-02 методика) | 1 |

## Топ-5 конфигураций

|   param_window |   param_oversold |   param_exit_level |   is_sharpe_ratio |   oos_sharpe_ratio |   robust_score |
|---------------:|-----------------:|-------------------:|------------------:|-------------------:|---------------:|
|             14 |              -80 |                -40 |            -0.132 |              0.228 |          0.687 |
|             14 |              -90 |                -30 |            -0.03  |              0.114 |          0.629 |
|             20 |              -80 |                -50 |            -0.136 |             -0.018 |          0.575 |
|             10 |              -90 |                -30 |            -0.139 |             -0.069 |          0.504 |
|             14 |              -70 |                -40 |            -0.21  |              0.27  |          0.47  |

## Выводы
Мульти-инструментный контроль promising-тройки, сторона GBM.

Артефакты: `sweep.csv` (все конфигурации), `walk_forward.csv`, `equity.png`, `param_heatmap.png`.