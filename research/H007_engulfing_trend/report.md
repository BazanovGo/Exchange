# H007: engulfing_trend

**Вердикт:** `archived` — stage-02 overfit flags = 2; no OOS edge (Sharpe 0.15 < 0.2); fails walk-forward (efficiency -0.28 < 0.3)
**Дата:** 2026-07-04 13:38 UTC

## Идея
Свечной паттерн: бычье поглощение с телом >= k*ATR выше долгой SMA — покупатели перехватывают инициативу на откате. Класс 'Candle patterns + Trend filter', ранее не исследован.

## Правила
Candle-pattern entry: a bullish engulfing bar (body swallows the prior red body) appearing above the long SMA signals dip-buyers taking control; exit when price closes below the fast SMA.

## Лучшие параметры (по robust score)
```json
{
  "trend_window": "100",
  "exit_window": "40",
  "min_body_atr": 0.3
}
```

## Результаты

| метрика | значение |
|---|---|
| комбинаций в сетке | 27 |
| OOS Sharpe (best robust) | 0.15 |
| OOS CAGR % | 1.1 |
| OOS MaxDD % | 10.9 |
| OOS сделок | 14 |
| плато: доля комбо с OOS Sharpe>0 | 41% |
| walk-forward efficiency | -0.28 |
| WF прибыльных окон | 25% |
| MC P(разорения) @f=0.25 | 0.0000 |
| MC медианная просадка @f=0.25 | 9.9% |
| флагов переобучения (этап-02 методика) | 2 |

## Топ-5 конфигураций

|   param_trend_window |   param_exit_window |   param_min_body_atr |   is_sharpe_ratio |   oos_sharpe_ratio |   robust_score |
|---------------------:|--------------------:|---------------------:|------------------:|-------------------:|---------------:|
|                  100 |                  40 |                  0.3 |             0.311 |              0.147 |          0.677 |
|                  200 |                  40 |                  0.3 |             0.303 |              0.561 |          0.657 |
|                  150 |                  40 |                  0.3 |             0.262 |              0.242 |          0.635 |
|                  150 |                  40 |                  0.5 |             0.546 |              0.045 |          0.579 |
|                  200 |                  40 |                  0.5 |             0.719 |              0.045 |          0.579 |

## Выводы
—

Артефакты: `sweep.csv` (все конфигурации), `walk_forward.csv`, `equity.png`, `param_heatmap.png`.