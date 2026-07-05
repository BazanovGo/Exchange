# H001: squeeze_breakout

**Вердикт:** `archived` — too few OOS trades (5 < 8); stage-02 overfit flags = 3; fails walk-forward (efficiency -0.82 < 0.3)
**Дата:** 2026-07-05 05:18 UTC

## Идея
Цикл волатильности: сжатие Bollinger + пробой с растущим ATR (см. первый прогон). ПЕРЕ-АУДИТ: исходный WF-вердикт вычислен с inf-Sharpe багом.

## Правила
Volatility-cycle breakout: wait until Bollinger width falls into its lowest quantile over a lookback (the squeeze), then buy a close breaking the upper band while ATR is already expanding; exit on a close back through the middle band.

## Лучшие параметры (по robust score)
```json
{
  "bb_window": "30",
  "squeeze_lookback": "180",
  "squeeze_quantile": 0.25,
  "atr_window": "10"
}
```

## Результаты

| метрика | значение |
|---|---|
| комбинаций в сетке | 54 |
| OOS Sharpe (best robust) | 0.52 |
| OOS CAGR % | 3.9 |
| OOS MaxDD % | 7.6 |
| OOS сделок | 5 |
| плато: доля комбо с OOS Sharpe>0 | 37% |
| walk-forward efficiency | -0.82 |
| WF прибыльных окон | 25% |
| MC P(разорения) @f=0.25 | 0.0000 |
| MC медианная просадка @f=0.25 | 8.0% |
| флагов переобучения (этап-02 методика) | 3 |

## Топ-5 конфигураций

|   param_bb_window |   param_squeeze_lookback |   param_squeeze_quantile |   param_atr_window |   is_sharpe_ratio |   oos_sharpe_ratio |   robust_score |
|------------------:|-------------------------:|-------------------------:|-------------------:|------------------:|-------------------:|---------------:|
|                30 |                      180 |                     0.25 |                 10 |             0.418 |              0.517 |          0.611 |
|                30 |                      180 |                     0.25 |                 14 |             0.418 |              0.517 |          0.611 |
|                30 |                      180 |                     0.35 |                 10 |             0.282 |              0.141 |          0.531 |
|                30 |                      180 |                     0.35 |                 14 |             0.282 |              0.063 |          0.509 |
|                30 |                       80 |                     0.15 |                 14 |             0.275 |              0.229 |          0.503 |

## Выводы
Re-audit итерации 4 после фикса WF-отбора.

Артефакты: `sweep.csv` (все конфигурации), `walk_forward.csv`, `equity.png`, `param_heatmap.png`.