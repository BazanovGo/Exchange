# Журнал сделок оптимизированных стратегий

Все сделки по порядку и результат каждой — по каждой системе, которую оптимизация отобрала в отчётах. Полнота проверена независимым пересчётом: [`AUDIT.md`](AUDIT.md).

* **[`trade_log.xlsx`](trade_log.xlsx)** — основные отборы (стратегий: 24, строк: 42 143): победители семи семейств и глобальные отборы скользящих средних.
* **[`trade_log_groups.xlsx`](trade_log_groups.xlsx)** — скользящие средние, лучшие внутри каждой группы «модель выхода × метод наклона» (стратегий: 35, строк: 345 845).
* **[`trade_log_full.xlsx`](trade_log_full.xlsx)** — семь семейств, **сквозной прогон** по всей истории: все сделки подряд, без разрыва между выборкой и проверкой (стратегий: 12, строк: 1 817).
* **[`csv/families/`](csv/families/README.md)** — скользящие средние, лучший в каждом из 144 семейств на этапах 1 и 3 (файлов: 576, сделок: 5 200 421); только CSV — для Excel это слишком много строк.
* **[`csv/walkforward/`](csv/walkforward/)** — walk-forward с переоптимизацией: сделки каждого шага с параметрами, выбранными на этом шаге.

* **[`points/`](points/README.md)** — исходы в **пунктах** по выгоднейшим параметрам каждой стратегии, по таблице на актив, со статистикой; [`points/full/`](points/full/README.md) — то же для сквозного прогона семи семейств.

Те же сделки — в CSV: [`csv/`](csv/) для основных отборов, [`csv/groups/`](csv/groups/) (сжатые `.csv.gz`) для лучших в группах, [`csv/full/`](csv/full/) для сквозного прогона.

* **Семь семейств** — победители по in-sample из `docs/RESULTS_SEVEN_FAMILIES.md`; счёт 1 000 000 USD, позиция по целевой волатильности 15%, издержки вычтены. В основном журнале IS и OOS — отдельные прогоны, как в отчёте: OOS начинается без позиции и с пустыми индикаторами, между ними эмбарго 10 дней, поэтому там нет сделок за несколько месяцев, а позиция на конец IS закрыта переоценкой. Сквозной прогон этих разрывов не имеет. Позиция с изменяемым размером: уменьшение позиции записывается движком как отдельное закрытие. Последняя строка периода, если позиция осталась открытой, — её переоценка по последнему закрытию.
* **Скользящие средние** — правила R1 (макс. PnL), R2 (макс. Шарп), R3 (плато) на этапах 1 и 3: по всей сетке (глобальные), внутри каждой группы «выход × метод наклона» и в каждом из 144 семейств. Каждая конфигурация торгует всю историю одним прогоном; сделка относится к IS или OOS по бару выхода. 1 контракт, без издержек (так в задании). Котировки — Open бара исполнения. «через ролл» — позиция пережила смену контракта: результат включает перекладку и не равен разности котировок. Если два правила выбрали одну конфигурацию, её сделки записаны один раз; где именно — в [`AUDIT.md`](AUDIT.md), раздел «Совпадения».

Каждый файл сверен при выгрузке (`export_trades.py` падает при расхождении): OOS-итог семи семейств — с отчётом, итог скользящих средних — с VectorBT-статистикой финалиста, IS/OOS и число сделок — с полной сеткой, walk-forward — с итогами шагов отчёта.

## Основные отборы

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

## Скользящие средние: лучшие в группах

| Стратегия | Сделок | Прибыльных | Итог IS, USD | Итог OOS, USD | Макс. убыточных сделок подряд | CSV |
|---|---|---|---|---|---|---|
| Скользящие средние, 5 мин, этап 1, лучшие в группе: NG · выход A_fast_cross, наклон A_pct · R1 макс. PnL · SSMA85 × SSMA129, наклон A_pct, выход A_fast_cross | 5233 | 18% | +1 452 | +182 | 32 | [MA_1_NG_xA_sA_R1.csv.gz](csv/groups/MA_1_NG_xA_sA_R1.csv.gz) |
| Скользящие средние, 5 мин, этап 1, лучшие в группе: NG · выход A_fast_cross, наклон A_pct · R2 макс. Шарп · TMA73 × WMA112, наклон A_pct, выход A_fast_cross | 7852 | 20% | +999 | +547 | 34 | [MA_1_NG_xA_sA_R2.csv.gz](csv/groups/MA_1_NG_xA_sA_R2.csv.gz) |
| Скользящие средние, 5 мин, этап 1, лучшие в группе: NG · выход A_fast_cross, наклон A_pct · R3 плато · TMA73 × WMA108, наклон A_pct, выход A_fast_cross | 7838 | 20% | +934 | +436 | 33 | [MA_1_NG_xA_sA_R3.csv.gz](csv/groups/MA_1_NG_xA_sA_R3.csv.gz) |
| Скользящие средние, 5 мин, этап 1, лучшие в группе: NG · выход A_fast_cross, наклон B_atr · R1 макс. PnL · SMA89 × VWMA250, наклон B_atr, выход A_fast_cross | 7246 | 20% | +1 324 | +214 | 27 | [MA_1_NG_xA_sB_R1.csv.gz](csv/groups/MA_1_NG_xA_sB_R1.csv.gz) |
| Скользящие средние, 5 мин, этап 1, лучшие в группе: NG · выход A_fast_cross, наклон B_atr · R2 макс. Шарп · VWMA59 × EMA52, наклон B_atr, выход A_fast_cross | 10173 | 22% | +1 155 | +113 | 30 | [MA_1_NG_xA_sB_R2.csv.gz](csv/groups/MA_1_NG_xA_sB_R2.csv.gz) |
| Скользящие средние, 5 мин, этап 1, лучшие в группе: NG · выход A_fast_cross, наклон B_atr · R3 плато · SMA98 × EMA186, наклон B_atr, выход A_fast_cross | 6787 | 19% | +1 236 | +326 | 31 | [MA_1_NG_xA_sB_R3.csv.gz](csv/groups/MA_1_NG_xA_sB_R3.csv.gz) |
| Скользящие средние, 5 мин, этап 1, лучшие в группе: NG · выход B_slow_flip, наклон A_pct · R2 макс. Шарп · TMA4 × SMA51, наклон A_pct, выход B_slow_flip | 3602 | 35% | +1 626 | +534 | 17 | [MA_1_NG_xB_sA_R2.csv.gz](csv/groups/MA_1_NG_xB_sA_R2.csv.gz) |
| Скользящие средние, 5 мин, этап 1, лучшие в группе: NG · выход B_slow_flip, наклон A_pct · R3 плато · VWMA2 × WMA88, наклон A_pct, выход B_slow_flip | 2976 | 34% | +1 498 | +847 | 14 | [MA_1_NG_xB_sA_R3.csv.gz](csv/groups/MA_1_NG_xB_sA_R3.csv.gz) |
| Скользящие средние, 5 мин, этап 1, лучшие в группе: NG · выход B_slow_flip, наклон B_atr · R1 макс. PnL · WMA3 × SMA145, наклон B_atr, выход B_slow_flip | 1512 | 37% | +1 818 | +404 | 15 | [MA_1_NG_xB_sB_R1.csv.gz](csv/groups/MA_1_NG_xB_sB_R1.csv.gz) |
| Скользящие средние, 5 мин, этап 1, лучшие в группе: BR · выход A_fast_cross, наклон A_pct · R1 макс. PnL · TMA86 × EMA104, наклон A_pct, выход A_fast_cross | 15565 | 20% | +2 004 | -28 | 27 | [MA_1_BR_xA_sA_R1.csv.gz](csv/groups/MA_1_BR_xA_sA_R1.csv.gz) |
| Скользящие средние, 5 мин, этап 1, лучшие в группе: BR · выход A_fast_cross, наклон A_pct · R2 макс. Шарп · TMA86 × EMA91, наклон A_pct, выход A_fast_cross | 15532 | 20% | +1 957 | -76 | 27 | [MA_1_BR_xA_sA_R2.csv.gz](csv/groups/MA_1_BR_xA_sA_R2.csv.gz) |
| Скользящие средние, 5 мин, этап 1, лучшие в группе: BR · выход A_fast_cross, наклон A_pct · R3 плато · TMA87 × EMA104, наклон A_pct, выход A_fast_cross | 15446 | 20% | +1 942 | -6 | 28 | [MA_1_BR_xA_sA_R3.csv.gz](csv/groups/MA_1_BR_xA_sA_R3.csv.gz) |
| Скользящие средние, 5 мин, этап 1, лучшие в группе: BR · выход A_fast_cross, наклон B_atr · R1 макс. PnL · SMA99 × EMA7, наклон B_atr, выход A_fast_cross | 13959 | 18% | +1 900 | -725 | 38 | [MA_1_BR_xA_sB_R1.csv.gz](csv/groups/MA_1_BR_xA_sB_R1.csv.gz) |
| Скользящие средние, 5 мин, этап 1, лучшие в группе: BR · выход A_fast_cross, наклон B_atr · R2 макс. Шарп · TMA44 × TMA86, наклон B_atr, выход A_fast_cross | 22284 | 22% | +1 397 | +260 | 31 | [MA_1_BR_xA_sB_R2.csv.gz](csv/groups/MA_1_BR_xA_sB_R2.csv.gz) |
| Скользящие средние, 5 мин, этап 1, лучшие в группе: BR · выход A_fast_cross, наклон B_atr · R3 плато · VWMA60 × EMA144, наклон B_atr, выход A_fast_cross | 21176 | 21% | +1 256 | +415 | 35 | [MA_1_BR_xA_sB_R3.csv.gz](csv/groups/MA_1_BR_xA_sB_R3.csv.gz) |
| Скользящие средние, 5 мин, этап 1, лучшие в группе: BR · выход B_slow_flip, наклон A_pct · R1 макс. PnL · TMA86 × EMA104, наклон A_pct, выход B_slow_flip | 4384 | 31% | +2 906 | -44 | 18 | [MA_1_BR_xB_sA_R1.csv.gz](csv/groups/MA_1_BR_xB_sA_R1.csv.gz) |
| Скользящие средние, 5 мин, этап 1, лучшие в группе: BR · выход B_slow_flip, наклон B_atr · R2 макс. Шарп · TMA42 × EMA102, наклон B_atr, выход B_slow_flip | 4945 | 30% | +2 583 | -43 | 18 | [MA_1_BR_xB_sB_R2.csv.gz](csv/groups/MA_1_BR_xB_sB_R2.csv.gz) |
| Скользящие средние, 5 мин, этап 1, лучшие в группе: BR · выход B_slow_flip, наклон B_atr · R3 плато · TMA41 × EMA106, наклон B_atr, выход B_slow_flip | 4869 | 29% | +2 345 | -42 | 17 | [MA_1_BR_xB_sB_R3.csv.gz](csv/groups/MA_1_BR_xB_sB_R3.csv.gz) |
| Скользящие средние, 5 мин, этап 3, лучшие в группе: NG · выход A_fast_cross, наклон A_pct · R1 макс. PnL · SSMA92 × SSMA206, наклон A_pct, выход A_fast_cross | 4940 | 17% | +1 250 | +313 | 27 | [MA_3_NG_xA_sA_R1.csv.gz](csv/groups/MA_3_NG_xA_sA_R1.csv.gz) |
| Скользящие средние, 5 мин, этап 3, лучшие в группе: NG · выход A_fast_cross, наклон A_pct · R2 макс. Шарп · TMA73 × EMA83, наклон A_pct, выход A_fast_cross | 7956 | 20% | +1 005 | +217 | 26 | [MA_3_NG_xA_sA_R2.csv.gz](csv/groups/MA_3_NG_xA_sA_R2.csv.gz) |
| Скользящие средние, 5 мин, этап 3, лучшие в группе: NG · выход A_fast_cross, наклон A_pct · R3 плато · TMA70 × EMA250, наклон A_pct, выход A_fast_cross | 8469 | 20% | +1 101 | +218 | 28 | [MA_3_NG_xA_sA_R3.csv.gz](csv/groups/MA_3_NG_xA_sA_R3.csv.gz) |
| Скользящие средние, 5 мин, этап 3, лучшие в группе: NG · выход A_fast_cross, наклон B_atr · R1 макс. PnL · SSMA92 × SSMA246, наклон B_atr, выход A_fast_cross | 4881 | 17% | +1 224 | +219 | 28 | [MA_3_NG_xA_sB_R1.csv.gz](csv/groups/MA_3_NG_xA_sB_R1.csv.gz) |
| Скользящие средние, 5 мин, этап 3, лучшие в группе: NG · выход A_fast_cross, наклон B_atr · R2 макс. Шарп · SMA95 × SMA200, наклон B_atr, выход A_fast_cross | 6835 | 19% | +985 | +434 | 34 | [MA_3_NG_xA_sB_R2.csv.gz](csv/groups/MA_3_NG_xA_sB_R2.csv.gz) |
| Скользящие средние, 5 мин, этап 3, лучшие в группе: NG · выход A_fast_cross, наклон B_atr · R3 плато · TMA70 × SSMA155, наклон B_atr, выход A_fast_cross | 8433 | 20% | +1 072 | +362 | 25 | [MA_3_NG_xA_sB_R3.csv.gz](csv/groups/MA_3_NG_xA_sB_R3.csv.gz) |
| Скользящие средние, 5 мин, этап 3, лучшие в группе: NG · выход B_slow_flip, наклон A_pct · R2 макс. Шарп · VWMA16 × EMA38, наклон A_pct, выход B_slow_flip | 1929 | 36% | +1 610 | +397 | 13 | [MA_3_NG_xB_sA_R2.csv.gz](csv/groups/MA_3_NG_xB_sA_R2.csv.gz) |
| Скользящие средние, 5 мин, этап 3, лучшие в группе: NG · выход B_slow_flip, наклон B_atr · R1 макс. PnL · SMA2 × SMA79, наклон B_atr, выход B_slow_flip | 1672 | 38% | +1 705 | +649 | 14 | [MA_3_NG_xB_sB_R1.csv.gz](csv/groups/MA_3_NG_xB_sB_R1.csv.gz) |
| Скользящие средние, 5 мин, этап 3, лучшие в группе: NG · выход B_slow_flip, наклон B_atr · R3 плато · VWMA19 × EMA22, наклон B_atr, выход B_slow_flip | 3032 | 34% | +1 495 | +418 | 15 | [MA_3_NG_xB_sB_R3.csv.gz](csv/groups/MA_3_NG_xB_sB_R3.csv.gz) |
| Скользящие средние, 5 мин, этап 3, лучшие в группе: BR · выход A_fast_cross, наклон A_pct · R1 макс. PnL · TMA44 × SMA57, наклон A_pct, выход A_fast_cross | 23398 | 22% | +1 697 | +263 | 34 | [MA_3_BR_xA_sA_R1.csv.gz](csv/groups/MA_3_BR_xA_sA_R1.csv.gz) |
| Скользящие средние, 5 мин, этап 3, лучшие в группе: BR · выход A_fast_cross, наклон A_pct · R2 макс. Шарп · TMA44 × TMA62, наклон A_pct, выход A_fast_cross | 23350 | 23% | +1 640 | +452 | 29 | [MA_3_BR_xA_sA_R2.csv.gz](csv/groups/MA_3_BR_xA_sA_R2.csv.gz) |
| Скользящие средние, 5 мин, этап 3, лучшие в группе: BR · выход A_fast_cross, наклон A_pct · R3 плато · TMA42 × SMA60, наклон A_pct, выход A_fast_cross | 24108 | 22% | +1 490 | +269 | 33 | [MA_3_BR_xA_sA_R3.csv.gz](csv/groups/MA_3_BR_xA_sA_R3.csv.gz) |
| Скользящие средние, 5 мин, этап 3, лучшие в группе: BR · выход A_fast_cross, наклон B_atr · R1 макс. PnL · TMA44 × VWMA33, наклон B_atr, выход A_fast_cross | 21953 | 22% | +1 533 | +456 | 27 | [MA_3_BR_xA_sB_R1.csv.gz](csv/groups/MA_3_BR_xA_sB_R1.csv.gz) |
| Скользящие средние, 5 мин, этап 3, лучшие в группе: BR · выход A_fast_cross, наклон B_atr · R3 плато · TMA43 × VWMA35, наклон B_atr, выход A_fast_cross | 22364 | 22% | +1 220 | +445 | 28 | [MA_3_BR_xA_sB_R3.csv.gz](csv/groups/MA_3_BR_xA_sB_R3.csv.gz) |
| Скользящие средние, 5 мин, этап 3, лучшие в группе: BR · выход B_slow_flip, наклон A_pct · R1 макс. PnL · EMA12 × TMA70, наклон A_pct, выход B_slow_flip | 2855 | 39% | +3 261 | +388 | 11 | [MA_3_BR_xB_sA_R1.csv.gz](csv/groups/MA_3_BR_xB_sA_R1.csv.gz) |
| Скользящие средние, 5 мин, этап 3, лучшие в группе: BR · выход B_slow_flip, наклон B_atr · R2 макс. Шарп · TMA38 × SMA51, наклон B_atr, выход B_slow_flip | 3873 | 36% | +2 710 | +539 | 18 | [MA_3_BR_xB_sB_R2.csv.gz](csv/groups/MA_3_BR_xB_sB_R2.csv.gz) |
| Скользящие средние, 5 мин, этап 3, лучшие в группе: BR · выход B_slow_flip, наклон B_atr · R3 плато · TMA41 × TMA54, наклон B_atr, выход B_slow_flip | 4418 | 39% | +2 370 | +231 | 19 | [MA_3_BR_xB_sB_R3.csv.gz](csv/groups/MA_3_BR_xB_sB_R3.csv.gz) |

## Семь семейств: сквозной прогон по всей истории

| Стратегия | Сделок | Прибыльных | Итог IS, USD | Эмбарго, USD | Итог OOS, USD | Итог, USD | Макс. убыточных сделок подряд | CSV |
|---|---|---|---|---|---|---|---|---|
| 1 Time-Series Momentum · BR · lookback=40 | 348 | 49% | +1 223 472 | +0 | +390 031 | +1 613 503 | 11 | [7F_1_TSMOM_BR.csv](csv/full/7F_1_TSMOM_BR.csv) |
| 1 Time-Series Momentum · NG · lookback=90 | 131 | 47% | +333 382 | +20 510 | -233 358 | +120 534 | 6 | [7F_1_TSMOM_NG.csv](csv/full/7F_1_TSMOM_NG.csv) |
| 2 Dual Moving Average · BR · fast=20,slow=60 | 175 | 63% | +1 349 837 | +0 | +786 050 | +2 135 886 | 8 | [7F_2_DualMA_BR.csv](csv/full/7F_2_DualMA_BR.csv) |
| 2 Dual Moving Average · NG · fast=20,slow=150 | 66 | 61% | +272 985 | +18 697 | -206 357 | +85 325 | 5 | [7F_2_DualMA_NG.csv](csv/full/7F_2_DualMA_NG.csv) |
| 3 Channel Breakout · BR · entry=40,exit_=20 | 130 | 71% | +1 516 085 | +0 | +740 144 | +2 256 229 | 7 | [7F_3_Breakout_BR.csv](csv/full/7F_3_Breakout_BR.csv) |
| 3 Channel Breakout · NG · entry=20,exit_=10 | 79 | 65% | +299 620 | +42 287 | +8 216 | +350 123 | 6 | [7F_3_Breakout_NG.csv](csv/full/7F_3_Breakout_NG.csv) |
| 4 Cross-sectional momentum · NG+BR · lookback=120 | 249 | 46% | +72 355 | +12 243 | -302 371 | -217 773 | 6 | [7F_4_CrossSect_NGBR.csv](csv/full/7F_4_CrossSect_NGBR.csv) |
| 5 Carry / term structure · BR · smooth=1,threshold=0.15 | 135 | 57% | +1 158 861 | +0 | +413 587 | +1 572 448 | 7 | [7F_5_Carry_BR.csv](csv/full/7F_5_Carry_BR.csv) |
| 5 Carry / term structure · NG · smooth=1,threshold=0.15 | 78 | 55% | -210 751 | +10 459 | +186 603 | -13 688 | 8 | [7F_5_Carry_NG.csv](csv/full/7F_5_Carry_NG.csv) |
| 6 MA + portfolio · NG+BR · fast=20,slow=100 | 139 | 60% | +462 011 | +9 289 | -158 613 | +312 686 | 6 | [7F_6_PortfolioMA_NGBR.csv](csv/full/7F_6_PortfolioMA_NGBR.csv) |
| 7 Momentum + reversal · BR · momentum=60,reversal=750 | 170 | 51% | +473 709 | +0 | +268 357 | +742 066 | 9 | [7F_7_MomRev_BR.csv](csv/full/7F_7_MomRev_BR.csv) |
| 7 Momentum + reversal · NG · momentum=60,reversal=500 | 105 | 50% | +7 851 | +0 | -59 178 | -51 327 | 8 | [7F_7_MomRev_NG.csv](csv/full/7F_7_MomRev_NG.csv) |

## Walk-forward с переоптимизацией

Семь семейств: в каждом окне параметры выбираются по лучшему Шарпу на обучающем отрезке, затем отдельный прогон на тестовом — как в `research/seven_families/run_stats.py`. Скользящие средние: на каждом шаге лучшая конфигурация всей сетки по 24 месяцам, её сделки за следующие 3 месяца; сделка относится к шагу по месяцу выхода, поэтому вошедшая до начала шага помечена.

| Система | Шагов | Сделок | Итог, USD | Сверка с отчётом | CSV |
|---|---|---|---|---|---|
| Семь семейств: 1 Time-Series Momentum · BR | 8 окон | 183 | +130 071 | эффективность +0.143, окон>0 0.500, стабильность 0.143 — как в разделе 3 | [7F_1_TSMOM_BR_WF.csv](csv/walkforward/7F_1_TSMOM_BR_WF.csv) |
| Семь семейств: 1 Time-Series Momentum · NG | 8 окон | 93 | +31 991 | эффективность -0.014, окон>0 0.500, стабильность 0.429 — как в разделе 3 | [7F_1_TSMOM_NG_WF.csv](csv/walkforward/7F_1_TSMOM_NG_WF.csv) |
| Семь семейств: 2 Dual Moving Average · BR | 8 окон | 83 | -87 364 | эффективность +0.030, окон>0 0.625, стабильность 0.429 — как в разделе 3 | [7F_2_DualMA_BR_WF.csv](csv/walkforward/7F_2_DualMA_BR_WF.csv) |
| Семь семейств: 2 Dual Moving Average · NG | 8 окон | 15 | -179 846 | эффективность -1.690, окон>0 0.125, стабильность 0.143 — как в разделе 3 | [7F_2_DualMA_NG_WF.csv](csv/walkforward/7F_2_DualMA_NG_WF.csv) |
| Семь семейств: 3 Channel Breakout · BR | 8 окон | 69 | +686 969 | эффективность +0.519, окон>0 0.875, стабильность 0.429 — как в разделе 3 | [7F_3_Breakout_BR_WF.csv](csv/walkforward/7F_3_Breakout_BR_WF.csv) |
| Семь семейств: 3 Channel Breakout · NG | 8 окон | 39 | +15 987 | эффективность -0.043, окон>0 0.500, стабильность 0.714 — как в разделе 3 | [7F_3_Breakout_NG_WF.csv](csv/walkforward/7F_3_Breakout_NG_WF.csv) |
| Семь семейств: 4 Cross-sectional momentum · NG+BR | 8 окон | 59 | -28 106 | эффективность +1.089, окон>0 0.375, стабильность 0.286 — как в разделе 3 | [7F_4_CrossSect_NGBR_WF.csv](csv/walkforward/7F_4_CrossSect_NGBR_WF.csv) |
| Семь семейств: 5 Carry / term structure · BR | 8 окон | 105 | +546 367 | эффективность +0.365, окон>0 0.500, стабильность 0.000 — как в разделе 3 | [7F_5_Carry_BR_WF.csv](csv/walkforward/7F_5_Carry_BR_WF.csv) |
| Семь семейств: 5 Carry / term structure · NG | 8 окон | 50 | +83 778 | эффективность +1.338, окон>0 0.625, стабильность 0.714 — как в разделе 3 | [7F_5_Carry_NG_WF.csv](csv/walkforward/7F_5_Carry_NG_WF.csv) |
| Семь семейств: 6 MA + portfolio · NG+BR | 8 окон | 27 | -38 426 | эффективность -0.269, окон>0 0.250, стабильность 0.286 — как в разделе 3 | [7F_6_PortfolioMA_NGBR_WF.csv](csv/walkforward/7F_6_PortfolioMA_NGBR_WF.csv) |
| Семь семейств: 7 Momentum + reversal · BR | 8 окон | 148 | +225 518 | эффективность +0.364, окон>0 0.625, стабильность 0.714 — как в разделе 3 | [7F_7_MomRev_BR_WF.csv](csv/walkforward/7F_7_MomRev_BR_WF.csv) |
| Семь семейств: 7 Momentum + reversal · NG | 8 окон | 21 | -35 644 | эффективность -1.837, окон>0 0.125, стабильность 0.143 — как в разделе 3 | [7F_7_MomRev_NG_WF.csv](csv/walkforward/7F_7_MomRev_NG_WF.csv) |
| Скользящие средние, этап 1: NG · критерий «максимальный PnL» | шагов по 3 мес.: 19, конфигураций: 17 | 2270 | +143 | итог шагов +143.2 USD (`wf_pnl.csv`) | [MA_1_NG_WF_pnl.csv.gz](csv/walkforward/MA_1_NG_WF_pnl.csv.gz) |
| Скользящие средние, этап 1: NG · критерий «максимальный месячный Шарп» | шагов по 3 мес.: 19, конфигураций: 16 | 2979 | +712 | итог шагов +711.8 USD (`wf_sharpe.csv`) | [MA_1_NG_WF_sharpe.csv.gz](csv/walkforward/MA_1_NG_WF_sharpe.csv.gz) |
| Скользящие средние, этап 1: NG · критерий «максимальный Шарп, сглаженный по соседним периодам» | шагов по 3 мес.: 19, конфигураций: 18 | 2764 | +689 | итог шагов +688.7 USD (`wf_plateau.csv`) | [MA_1_NG_WF_plateau.csv.gz](csv/walkforward/MA_1_NG_WF_plateau.csv.gz) |
| Скользящие средние, этап 1: BR · критерий «максимальный PnL» | шагов по 3 мес.: 52, конфигураций: 45 | 6207 | +1 132 | итог шагов +1132.4 USD (`wf_pnl.csv`) | [MA_1_BR_WF_pnl.csv.gz](csv/walkforward/MA_1_BR_WF_pnl.csv.gz) |
| Скользящие средние, этап 1: BR · критерий «максимальный месячный Шарп» | шагов по 3 мес.: 52, конфигураций: 47 | 7636 | +820 | итог шагов +820.2 USD (`wf_sharpe.csv`) | [MA_1_BR_WF_sharpe.csv.gz](csv/walkforward/MA_1_BR_WF_sharpe.csv.gz) |
| Скользящие средние, этап 1: BR · критерий «максимальный Шарп, сглаженный по соседним периодам» | шагов по 3 мес.: 52, конфигураций: 51 | 5046 | +1 586 | итог шагов +1586.0 USD (`wf_plateau.csv`) | [MA_1_BR_WF_plateau.csv.gz](csv/walkforward/MA_1_BR_WF_plateau.csv.gz) |
| Скользящие средние, этап 3: NG · критерий «максимальный PnL» | шагов по 3 мес.: 19, конфигураций: 19 | 1477 | -155 | итог шагов -155.3 USD (`wf_pnl.csv`) | [MA_3_NG_WF_pnl.csv.gz](csv/walkforward/MA_3_NG_WF_pnl.csv.gz) |
| Скользящие средние, этап 3: NG · критерий «максимальный месячный Шарп» | шагов по 3 мес.: 19, конфигураций: 16 | 3807 | +629 | итог шагов +629.2 USD (`wf_sharpe.csv`) | [MA_3_NG_WF_sharpe.csv.gz](csv/walkforward/MA_3_NG_WF_sharpe.csv.gz) |
| Скользящие средние, этап 3: NG · критерий «максимальный Шарп, сглаженный по соседним периодам» | шагов по 3 мес.: 19, конфигураций: 19 | 3729 | +392 | итог шагов +391.8 USD (`wf_plateau.csv`) | [MA_3_NG_WF_plateau.csv.gz](csv/walkforward/MA_3_NG_WF_plateau.csv.gz) |
| Скользящие средние, этап 3: BR · критерий «максимальный PnL» | шагов по 3 мес.: 52, конфигураций: 45 | 3085 | +130 | итог шагов +129.9 USD (`wf_pnl.csv`) | [MA_3_BR_WF_pnl.csv.gz](csv/walkforward/MA_3_BR_WF_pnl.csv.gz) |
| Скользящие средние, этап 3: BR · критерий «максимальный месячный Шарп» | шагов по 3 мес.: 52, конфигураций: 45 | 5490 | +1 788 | итог шагов +1787.9 USD (`wf_sharpe.csv`) | [MA_3_BR_WF_sharpe.csv.gz](csv/walkforward/MA_3_BR_WF_sharpe.csv.gz) |
| Скользящие средние, этап 3: BR · критерий «максимальный Шарп, сглаженный по соседним периодам» | шагов по 3 мес.: 52, конфигураций: 51 | 5705 | +878 | итог шагов +878.1 USD (`wf_plateau.csv`) | [MA_3_BR_WF_plateau.csv.gz](csv/walkforward/MA_3_BR_WF_plateau.csv.gz) |

## Скользящие средние: лучший в каждом из 144 семейств

| Этап | Рынок | Семейств | Сделок | Прибыльны в IS | Прибыльны в OOS | Совпали с журналом финалиста |
|---|---|---|---|---|---|---|
| этап 1 | NG | 144 | 836 854 | 144 | 144 | 4 |
| этап 1 | BR | 144 | 1 693 512 | 144 | 112 | 4 |
| этап 3 | NG | 144 | 806 587 | 144 | 135 | 4 |
| этап 3 | BR | 144 | 1 863 468 | 144 | 116 | 4 |

Полный список с итогами и ссылками — [`csv/families/README.md`](csv/families/README.md).

Пересборка: `python export_trades.py seven`, `seven-full`, `seven-wf` (окружение пакета `exchange`); `python export_trades.py ma` (нужен рабочий каталог исследования `MA5M_WORK`), `ma-families`, `ma-wf` (окружение исследования); `python export_trades.py excel`; проверка — `python audit_trades.py`.
