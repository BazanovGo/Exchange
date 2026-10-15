# Журнал сделок отобранных стратегий

Все сделки по порядку и результат каждой: **[`trade_log.xlsx`](trade_log.xlsx)** — лист «Сводка» и лист на каждую стратегию с итогом нарастающим и сериями убытков (формулы Excel). Те же сделки — в CSV в [`csv/`](csv/), по файлу на стратегию.

* **Семь семейств** — победители по in-sample из `docs/RESULTS_SEVEN_FAMILIES.md`; счёт 1 000 000 USD, позиция по целевой волатильности 15%, издержки вычтены. IS и OOS — отдельные прогоны, как в отчёте. Позиция с изменяемым размером: уменьшение позиции записывается движком как отдельное закрытие. Последняя строка периода, если позиция осталась открытой, — её переоценка по последнему закрытию.
* **Скользящие средние** — глобальные отборы R1 (макс. PnL), R2 (макс. Шарп), R3 (плато) на этапах 1 и 3; 1 контракт, без издержек (так в задании). Котировки — Open бара исполнения. «через ролл» — позиция пережила смену контракта: результат включает перекладку и не равен разности котировок.

Каждый файл сверен с отчётом при выгрузке (`export_trades.py` падает при расхождении): OOS-итог семи семейств — с «итог. дох.» отчёта, итог скользящих средних — с VectorBT-статистикой финалиста, а IS/OOS — с полной сеткой.

| Стратегия | Сделок | Прибыльных | Итог IS, USD | Итог OOS, USD | Макс. убыточных сделок подряд | CSV |
|---|---|---|---|---|---|---|
| Семь семейств (дневные бары): 1 Time-Series Momentum · BR · lookback=40 | 344 | 47% | +1 313 344 | +171 453 | 13 | [7F_1_TSMOM_BR.csv](csv/7F_1_TSMOM_BR.csv) |
| Семь семейств (дневные бары): 1 Time-Series Momentum · NG · lookback=90 | 117 | 48% | +444 989 | -181 100 | 7 | [7F_1_TSMOM_NG.csv](csv/7F_1_TSMOM_NG.csv) |
| Семь семейств (дневные бары): 2 Dual Moving Average · BR · fast=20,slow=60 | 172 | 62% | +1 412 319 | +565 401 | 8 | [7F_2_DualMA_BR.csv](csv/7F_2_DualMA_BR.csv) |
| Семь семейств (дневные бары): 2 Dual Moving Average · NG · fast=20,slow=150 | 60 | 62% | +376 022 | -162 324 | 6 | [7F_2_DualMA_NG.csv](csv/7F_2_DualMA_NG.csv) |
| Семь семейств (дневные бары): 3 Channel Breakout · BR · entry=40,exit_=20 | 126 | 69% | +1 593 924 | +494 872 | 7 | [7F_3_Breakout_BR.csv](csv/7F_3_Breakout_BR.csv) |
| Семь семейств (дневные бары): 3 Channel Breakout · NG · entry=20,exit_=10 | 75 | 61% | +374 185 | -54 644 | 6 | [7F_3_Breakout_NG.csv](csv/7F_3_Breakout_NG.csv) |
| Семь семейств (дневные бары): 4 Cross-sectional momentum · NG+BR · lookback=120 | 228 | 45% | +187 566 | -308 723 | 7 | [7F_4_CrossSect_NGBR.csv](csv/7F_4_CrossSect_NGBR.csv) |
| Семь семейств (дневные бары): 5 Carry / term structure · BR · smooth=1,threshold=0.15 | 130 | 56% | +1 158 861 | +366 498 | 7 | [7F_5_Carry_BR.csv](csv/7F_5_Carry_BR.csv) |
| Семь семейств (дневные бары): 5 Carry / term structure · NG · smooth=1,threshold=0.15 | 77 | 56% | -146 662 | +247 829 | 8 | [7F_5_Carry_NG.csv](csv/7F_5_Carry_NG.csv) |
| Семь семейств (дневные бары): 6 MA + portfolio · NG+BR · fast=20,slow=100 | 129 | 63% | +554 789 | -119 040 | 7 | [7F_6_PortfolioMA_NGBR.csv](csv/7F_6_PortfolioMA_NGBR.csv) |
| Семь семейств (дневные бары): 7 Momentum + reversal · BR · momentum=60,reversal=750 | 188 | 51% | +540 269 | +19 126 | 9 | [7F_7_MomRev_BR.csv](csv/7F_7_MomRev_BR.csv) |
| Семь семейств (дневные бары): 7 Momentum + reversal · NG · momentum=60,reversal=500 | 120 | 48% | +7 851 | -35 112 | 8 | [7F_7_MomRev_NG.csv](csv/7F_7_MomRev_NG.csv) |
| Скользящие средние, 5 мин, этап 1: NG · R1 макс. PnL · WMA2 × VWMA69, наклон A_pct, выход B_slow_flip | 4761 | 37% | +1 970 | +823 | 14 | [MA_1_NG_R1.csv](csv/MA_1_NG_R1.csv) |
| Скользящие средние, 5 мин, этап 1: NG · R2 макс. Шарп · VWMA60 × EMA52, наклон B_atr, выход B_slow_flip | 3256 | 30% | +1 436 | +172 | 21 | [MA_1_NG_R2.csv](csv/MA_1_NG_R2.csv) |
| Скользящие средние, 5 мин, этап 1: NG · R3 плато · VWMA81 × EMA81, наклон B_atr, выход B_slow_flip | 2343 | 29% | +1 469 | +211 | 17 | [MA_1_NG_R3.csv](csv/MA_1_NG_R3.csv) |
| Скользящие средние, 5 мин, этап 1: BR · R1 макс. PnL · VWMA7 × TMA78, наклон B_atr, выход B_slow_flip | 5892 | 37% | +2 937 | +303 | 17 | [MA_1_BR_R1.csv](csv/MA_1_BR_R1.csv) |
| Скользящие средние, 5 мин, этап 1: BR · R2 макс. Шарп · WMA88 × EMA120, наклон A_pct, выход B_slow_flip | 4364 | 29% | +2 586 | +61 | 18 | [MA_1_BR_R2.csv](csv/MA_1_BR_R2.csv) |
| Скользящие средние, 5 мин, этап 1: BR · R3 плато · WMA88 × EMA121, наклон A_pct, выход B_slow_flip | 4350 | 29% | +2 564 | +48 | 18 | [MA_1_BR_R3.csv](csv/MA_1_BR_R3.csv) |
| Скользящие средние, 5 мин, этап 3: NG · R1 макс. PnL · EMA12 × WMA131, наклон A_pct, выход B_slow_flip | 1067 | 38% | +1 818 | +518 | 9 | [MA_3_NG_R1.csv](csv/MA_3_NG_R1.csv) |
| Скользящие средние, 5 мин, этап 3: NG · R2 макс. Шарп · TMA21 × EMA20, наклон B_atr, выход B_slow_flip | 3039 | 33% | +1 504 | +533 | 18 | [MA_3_NG_R2.csv](csv/MA_3_NG_R2.csv) |
| Скользящие средние, 5 мин, этап 3: NG · R3 плато · TMA38 × SMA250, наклон A_pct, выход B_slow_flip | 520 | 37% | +1 316 | -305 | 14 | [MA_3_NG_R3.csv](csv/MA_3_NG_R3.csv) |
| Скользящие средние, 5 мин, этап 3: BR · R1 макс. PnL · TMA10 × SMA23, наклон B_atr, выход B_slow_flip | 6353 | 36% | +3 357 | +250 | 17 | [MA_3_BR_R1.csv](csv/MA_3_BR_R1.csv) |
| Скользящие средние, 5 мин, этап 3: BR · R2 макс. Шарп · TMA43 × SMA65, наклон A_pct, выход B_slow_flip | 2188 | 39% | +3 083 | +308 | 12 | [MA_3_BR_R2.csv](csv/MA_3_BR_R2.csv) |
| Скользящие средние, 5 мин, этап 3: BR · R3 плато · TMA40 × SMA65, наклон A_pct, выход B_slow_flip | 2220 | 39% | +2 950 | +268 | 13 | [MA_3_BR_R3.csv](csv/MA_3_BR_R3.csv) |

Пересборка: `python export_trades.py seven` (окружение пакета `exchange`), `python export_trades.py ma` (нужен рабочий каталог исследования `MA5M_WORK`), `python export_trades.py excel`.
