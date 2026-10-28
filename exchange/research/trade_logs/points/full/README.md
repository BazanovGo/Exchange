# Исходы сделок в пунктах: семь семейств, сквозной прогон

Те же системы и параметры, что в [`../`](../README.md), но **одним прогоном по всей истории**: все сделки подряд, без разрыва между выборкой и проверкой. В прогонах отчёта OOS начинается без позиции и с пустыми индикаторами, а между периодами — эмбарго 10 дней; поэтому там нет сделок за несколько месяцев, а позиция на конец IS закрыта переоценкой. Здесь этих разрывов нет. Период сделки — по дате выхода: **IS**, **эмбарго** (10 дней между периодами) или **OOS**. Итоги OOS здесь не равны итогам отчёта: отчёт честно оценивает систему, запущенную с нуля в начале OOS.

Тот же файл одним Excel со статистикой на формулах: [`points_full.xlsx`](points_full.xlsx). Только числа, по файлу на актив: [`../sequence/full/`](../sequence/full/README.md).

| Стратегия | Актив | Параметры | Исходов | Прибыльных | Средний +, пунктов | Средний −, пунктов | Убыточных подряд, макс. | Итог IS, USD | Итог OOS, USD | Итог, USD | Таблица |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 Time-Series Momentum | BR | lookback=40 | 348 | 170 (49%) | +7.7009 | -1.9454 | 11 | +1 223 471 | +390 031 | +1 613 503 | [7F_1_TSMOM_BR.csv](7F_1_TSMOM_BR.csv) |
| 1 Time-Series Momentum | NG | lookback=90 | 131 | 61 (47%) | +0.7138 | -0.2084 | 6 | +333 382 | -233 358 | +120 534 | [7F_1_TSMOM_NG.csv](7F_1_TSMOM_NG.csv) |
| 2 Dual Moving Average | BR | fast=20,slow=60 | 175 | 111 (63%) | +11.2331 | -3.3220 | 8 | +1 349 837 | +786 050 | +2 135 886 | [7F_2_DualMA_BR.csv](7F_2_DualMA_BR.csv) |
| 2 Dual Moving Average | NG | fast=20,slow=150 | 66 | 40 (61%) | +0.8226 | -0.5821 | 5 | +272 985 | -206 357 | +85 325 | [7F_2_DualMA_NG.csv](7F_2_DualMA_NG.csv) |
| 3 Channel Breakout | BR | entry=40,exit_=20 | 130 | 92 (71%) | +11.2865 | -3.1731 | 7 | +1 516 085 | +740 144 | +2 256 229 | [7F_3_Breakout_BR.csv](7F_3_Breakout_BR.csv) |
| 3 Channel Breakout | NG | entry=20,exit_=10 | 79 | 51 (65%) | +0.5716 | -0.3805 | 6 | +299 620 | +8 216 | +350 123 | [7F_3_Breakout_NG.csv](7F_3_Breakout_NG.csv) |
| 4 Cross-sectional momentum | NG | lookback=120 | 130 | 66 (51%) | +0.4620 | -0.4939 | 11 | -134 244 | -44 966 | -166 967 | [7F_4_CrossSect_NGBR_NG.csv](7F_4_CrossSect_NGBR_NG.csv) |
| 4 Cross-sectional momentum | BR | lookback=120 | 119 | 48 (40%) | +7.9328 | -5.0048 | 8 | +206 599 | -257 405 | -50 806 | [7F_4_CrossSect_NGBR_BR.csv](7F_4_CrossSect_NGBR_BR.csv) |
| 5 Carry / term structure | BR | smooth=1,threshold=0.15 | 135 | 77 (57%) | +9.0643 | -5.5035 | 7 | +1 158 861 | +413 587 | +1 572 448 | [7F_5_Carry_BR.csv](7F_5_Carry_BR.csv) |
| 5 Carry / term structure | NG | smooth=1,threshold=0.15 | 78 | 43 (55%) | +0.6188 | -0.6836 | 8 | -210 751 | +186 603 | -13 688 | [7F_5_Carry_NG.csv](7F_5_Carry_NG.csv) |
| 6 MA + portfolio | NG | fast=20,slow=100 | 72 | 39 (54%) | +0.8784 | -0.5899 | 4 | +119 303 | -135 018 | -6 426 | [7F_6_PortfolioMA_NGBR_NG.csv](7F_6_PortfolioMA_NGBR_NG.csv) |
| 6 MA + portfolio | BR | fast=20,slow=100 | 67 | 45 (67%) | +12.4050 | -4.4963 | 6 | +342 707 | -23 595 | +319 112 | [7F_6_PortfolioMA_NGBR_BR.csv](7F_6_PortfolioMA_NGBR_BR.csv) |
| 7 Momentum + reversal | BR | momentum=60,reversal=750 | 170 | 87 (51%) | +6.3443 | -1.9716 | 9 | +473 709 | +268 357 | +742 066 | [7F_7_MomRev_BR.csv](7F_7_MomRev_BR.csv) |
| 7 Momentum + reversal | NG | momentum=60,reversal=500 | 105 | 52 (50%) | +0.4504 | -0.2130 | 8 | +7 851 | -59 178 | -51 327 | [7F_7_MomRev_NG.csv](7F_7_MomRev_NG.csv) |

Пересборка: `python export_points.py seven-full` (окружение пакета `exchange`), `python export_points.py tables`, `python export_points.py sequence`.
