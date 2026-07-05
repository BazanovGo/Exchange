# H009: supertrend

**Вердикт:** `archived` — too few OOS trades (6 < 8); stage-02 overfit flags = 2; fails walk-forward (efficiency -0.44 < 0.3)
**Дата:** 2026-07-05 04:56 UTC

## Идея
Симметрия тренда: supertrend (лидер этапа 02 на статичном сплите) с включенной короткой стороной — always-in-the-market. Проверка направления 3 базы знаний (короткая сторона).

## Правила
ATR-based Supertrend line: long while price holds above the line, exit (or flip short) when the trend direction flips.

## Лучшие параметры (по robust score)
```json
{
  "window": "21",
  "multiplier": 3.0,
  "allow_short": true
}
```

## Результаты

| метрика | значение |
|---|---|
| комбинаций в сетке | 12 |
| OOS Sharpe (best robust) | 0.96 |
| OOS CAGR % | 16.1 |
| OOS MaxDD % | 25.9 |
| OOS сделок | 6 |
| плато: доля комбо с OOS Sharpe>0 | 75% |
| walk-forward efficiency | -0.44 |
| WF прибыльных окон | 25% |
| MC P(разорения) @f=0.25 | 0.0012 |
| MC медианная просадка @f=0.25 | 22.3% |
| флагов переобучения (этап-02 методика) | 2 |

## Топ-5 конфигураций

|   param_window |   param_multiplier | param_allow_short   |   is_sharpe_ratio |   oos_sharpe_ratio |   robust_score |
|---------------:|-------------------:|:--------------------|------------------:|-------------------:|---------------:|
|             21 |                3   | True                |             0.202 |              0.964 |          0.569 |
|             10 |                3   | True                |             0.07  |              1.078 |          0.568 |
|             14 |                3   | True                |             0.122 |              1.016 |          0.553 |
|              7 |                3   | True                |             0.025 |              0.682 |          0.527 |
|             14 |                1.5 | True                |             0.161 |              0.354 |          0.492 |

## Выводы
Существующий класс, новая область пространства параметров (short-леги).

Артефакты: `sweep.csv` (все конфигурации), `walk_forward.csv`, `equity.png`, `param_heatmap.png`.