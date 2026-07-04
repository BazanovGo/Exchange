# H003: donchian_rsi_pullback

**Вердикт:** `archived` — stage-02 overfit flags = 2
**Дата:** 2026-07-04 13:34 UTC

## Идея
Структурный откат: верхняя часть канала Donchian задает аптренд по построению (higher highs), RSI-провал внутри структуры — вход со скидкой. Комбинация 'Donchian + RSI'.

## Правила
Market-structure pullback: while price holds in the upper part of its Donchian channel (an uptrend by construction), buy an RSI dip below the pullback level; exit when RSI recovers to overbought or price loses the channel midline.

## Лучшие параметры (по robust score)
```json
{
  "don_window": "80",
  "structure_position": 0.5,
  "pullback": 45.0,
  "exit_level": 75.0
}
```

## Результаты

| метрика | значение |
|---|---|
| комбинаций в сетке | 54 |
| OOS Sharpe (best robust) | 1.14 |
| OOS CAGR % | 12.0 |
| OOS MaxDD % | 8.2 |
| OOS сделок | 9 |
| плато: доля комбо с OOS Sharpe>0 | 83% |
| walk-forward efficiency | nan |
| WF прибыльных окон | 25% |
| MC P(разорения) @f=0.25 | 0.0000 |
| MC медианная просадка @f=0.25 | 5.1% |
| флагов переобучения (этап-02 методика) | 2 |

## Топ-5 конфигураций

|   param_don_window |   param_structure_position |   param_pullback |   param_exit_level |   is_sharpe_ratio |   oos_sharpe_ratio |   robust_score |
|-------------------:|---------------------------:|-----------------:|-------------------:|------------------:|-------------------:|---------------:|
|                 80 |                        0.5 |               45 |                 75 |             0.777 |              1.142 |          0.648 |
|                 30 |                        0.6 |               45 |                 60 |             0.808 |              1.017 |          0.615 |
|                 80 |                        0.5 |               35 |                 75 |             0.684 |              1.152 |          0.601 |
|                 80 |                        0.6 |               35 |                 75 |             0.842 |              1.555 |          0.595 |
|                 80 |                        0.6 |               45 |                 75 |             0.53  |              1.144 |          0.577 |

## Выводы
—

Артефакты: `sweep.csv` (все конфигурации), `walk_forward.csv`, `equity.png`, `param_heatmap.png`.