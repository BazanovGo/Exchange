# H012: zscore_reversion

**Вердикт:** `promising`
**Дата:** 2026-07-05 05:13 UTC

## Идея
POSITIVE CONTROL (mean reversion): z-score fade на OU-процессе с внедренным возвратом к среднему. Пайплайн обязан дать positive вердикт — валидация чувствительности.

## Правила
Statistical mean reversion: buy when the rolling z-score of price drops below -z_entry, exit when it recovers above -z_exit; optionally short the mirror side.

## Лучшие параметры (по robust score)
```json
{
  "window": "60",
  "z_entry": 1.5,
  "z_exit": 0.5
}
```

## Результаты

| метрика | значение |
|---|---|
| комбинаций в сетке | 48 |
| OOS Sharpe (best robust) | 1.63 |
| OOS CAGR % | 21.2 |
| OOS MaxDD % | 5.5 |
| OOS сделок | 10 |
| плато: доля комбо с OOS Sharpe>0 | 65% |
| walk-forward efficiency | 0.75 |
| WF прибыльных окон | 100% |
| MC P(разорения) @f=0.25 | 0.0000 |
| MC медианная просадка @f=0.25 | 0.8% |
| флагов переобучения (этап-02 методика) | 0 |

## Топ-5 конфигураций

|   param_window |   param_z_entry |   param_z_exit |   is_sharpe_ratio |   oos_sharpe_ratio |   robust_score |
|---------------:|----------------:|---------------:|------------------:|-------------------:|---------------:|
|             60 |             1.5 |            0.5 |             1.512 |              1.631 |          0.767 |
|             40 |             1.5 |            0   |             1.678 |              1.517 |          0.763 |
|             40 |             1.5 |            0.5 |             1.506 |              1.369 |          0.75  |
|             60 |             2   |            0.5 |             1.075 |              2.624 |          0.743 |
|             60 |             2   |            0   |             1.468 |              2.494 |          0.715 |

## Выводы
Первый прогон поймал баг: WF выбирал конфигурации с 0 сделок (inf Sharpe). После фикса (NaN-метрики пустых портфелей + min_train_trades в WF) — повторный вердикт.

Артефакты: `sweep.csv` (все конфигурации), `walk_forward.csv`, `equity.png`, `param_heatmap.png`.