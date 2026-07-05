# H008: regime_switcher

**Вердикт:** `archived` — stage-02 overfit flags = 2; no OOS edge (Sharpe 0.04 < 0.2); no parameter plateau (31% of grid works OOS); fails walk-forward (efficiency -1.87 < 0.3); walk-forward windows unprofitable (0% < 25%)
**Дата:** 2026-07-05 04:56 UTC

## Идея
Режимный маршрутизатор (карта этапа 04): EMA-кросс в trend_up, z-score fade в range_highvol, тихий диапазон не торгуется вовсе. Первый составной 'переключатель' в проекте.

## Правила
Regime-routed composite (stage-04 map): when ADX confirms a trend, trade the EMA crossover engine; when ADX is quiet but volatility is elevated, fade z-score dips. Positions close when their engine's exit fires or the regime that opened them ends. The quiet low-volatility range is never traded.

## Лучшие параметры (по robust score)
```json
{
  "adx_threshold": 30.0,
  "fast_window": "10",
  "slow_window": "50",
  "z_window": "15",
  "z_entry": 1.2
}
```

## Результаты

| метрика | значение |
|---|---|
| комбинаций в сетке | 108 |
| OOS Sharpe (best robust) | 0.04 |
| OOS CAGR % | 0.1 |
| OOS MaxDD % | 7.8 |
| OOS сделок | 12 |
| плато: доля комбо с OOS Sharpe>0 | 31% |
| walk-forward efficiency | -1.87 |
| WF прибыльных окон | 0% |
| MC P(разорения) @f=0.25 | 0.0000 |
| MC медианная просадка @f=0.25 | 9.2% |
| флагов переобучения (этап-02 методика) | 2 |

## Топ-5 конфигураций

|   param_adx_threshold |   param_fast_window |   param_slow_window |   param_z_window |   param_z_entry |   is_sharpe_ratio |   oos_sharpe_ratio |   robust_score |
|----------------------:|--------------------:|--------------------:|-----------------:|----------------:|------------------:|-------------------:|---------------:|
|                    30 |                  10 |                  50 |               15 |             1.2 |            -0.241 |              0.039 |          0.697 |
|                    30 |                  10 |                  80 |               15 |             1.2 |            -0.4   |              0.039 |          0.644 |
|                    30 |                  10 |                  50 |               20 |             2   |            -0.254 |              0.42  |          0.601 |
|                    30 |                  10 |                  50 |               20 |             1.2 |            -0.306 |             -0.074 |          0.594 |
|                    30 |                  10 |                  50 |               15 |             2   |            -0.303 |              0.447 |          0.546 |

## Выводы
Направление 1 базы знаний: составной режимный переключатель.

Артефакты: `sweep.csv` (все конфигурации), `walk_forward.csv`, `equity.png`, `param_heatmap.png`.