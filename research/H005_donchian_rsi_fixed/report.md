# H005: donchian_rsi_fixed

**Вердикт:** `archived` — too few OOS trades (5 < 8); stage-02 overfit flags = 3
**Дата:** 2026-07-04 13:37 UTC

## Идея
Доработка H003 (плато 83%, но нестабильное ранжирование сетки): заморозка хрупких параметров (structure=0.5, exit=65), свободны только окно канала и глубина отката. Проверка выживания идеи при минимуме степеней свободы.

## Правила
Refined H003: the same Donchian-structure RSI pullback but with the fragile parameters frozen (structure position 0.5, RSI exit 65) — only the channel window and pullback depth remain free. Tests whether the idea survives with minimal degrees of freedom.

## Лучшие параметры (по robust score)
```json
{
  "don_window": "50",
  "pullback": 35.0
}
```

## Результаты

| метрика | значение |
|---|---|
| комбинаций в сетке | 15 |
| OOS Sharpe (best robust) | 0.98 |
| OOS CAGR % | 6.3 |
| OOS MaxDD % | 4.8 |
| OOS сделок | 5 |
| плато: доля комбо с OOS Sharpe>0 | 87% |
| walk-forward efficiency | nan |
| WF прибыльных окон | 50% |
| MC P(разорения) @f=0.25 | 0.0000 |
| MC медианная просадка @f=0.25 | 2.8% |
| флагов переобучения (этап-02 методика) | 3 |

## Топ-5 конфигураций

|   param_don_window |   param_pullback |   is_sharpe_ratio |   oos_sharpe_ratio |   robust_score |
|-------------------:|-----------------:|------------------:|-------------------:|---------------:|
|                 50 |               35 |             0.447 |              0.977 |          0.602 |
|                 80 |               35 |             0.409 |              0.728 |          0.564 |
|                 65 |               35 |             0.491 |              0.646 |          0.562 |
|                 30 |               45 |             0.58  |              0.355 |          0.532 |
|                 65 |               45 |             0.343 |              0.623 |          0.506 |

## Выводы
Урок H003: широкое плато + плохой IS-выбор -> резать степени свободы, а не сужать сетку.

Артефакты: `sweep.csv` (все конфигурации), `walk_forward.csv`, `equity.png`, `param_heatmap.png`.