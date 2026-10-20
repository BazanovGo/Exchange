# Исходы сделок: только числа

По файлу на стратегию и актив: результат каждой сделки по порядку, одно число в строке, без заголовка. **Пункт — минимальный шаг цены**: у BR 0.01 USD за баррель, у NG 0.001 USD за MMBtu. Плюс — прибыльный исход, минус — убыточный, 0 — выход по цене входа.

Параметры — выгоднейшие по оптимизации: у семи семейств выбранные оптимизатором (лучший Шарп в выборке), у скользящих средних — R1, максимальная прибыль в выборке (этап 3). У семи семейств издержки вычтены и результат дан на единицу позиции, округлён до целого шага; у скользящих средних — 1 контракт без издержек, шаги целые точно. Подробные таблицы с датами, ценами и USD — в [`../`](../README.md).

| Стратегия | Актив | Параметры | Исходов | Плюс | Минус | Ноль | Файл |
|---|---|---|---|---|---|---|---|
| 1 Time-Series Momentum | BR | lookback=40 | 344 | 161 | 181 | 2 | [7F_1_TSMOM_BR.csv](7F_1_TSMOM_BR.csv) |
| 1 Time-Series Momentum | NG | lookback=90 | 117 | 56 | 61 | 0 | [7F_1_TSMOM_NG.csv](7F_1_TSMOM_NG.csv) |
| 2 Dual Moving Average | BR | fast=20,slow=60 | 172 | 107 | 65 | 0 | [7F_2_DualMA_BR.csv](7F_2_DualMA_BR.csv) |
| 2 Dual Moving Average | NG | fast=20,slow=150 | 60 | 37 | 23 | 0 | [7F_2_DualMA_NG.csv](7F_2_DualMA_NG.csv) |
| 3 Channel Breakout | BR | entry=40,exit_=20 | 126 | 87 | 39 | 0 | [7F_3_Breakout_BR.csv](7F_3_Breakout_BR.csv) |
| 3 Channel Breakout | NG | entry=20,exit_=10 | 75 | 46 | 29 | 0 | [7F_3_Breakout_NG.csv](7F_3_Breakout_NG.csv) |
| 4 Cross-sectional momentum | NG | lookback=120 | 120 | 62 | 58 | 0 | [7F_4_CrossSect_NGBR_NG.csv](7F_4_CrossSect_NGBR_NG.csv) |
| 4 Cross-sectional momentum | BR | lookback=120 | 108 | 41 | 67 | 0 | [7F_4_CrossSect_NGBR_BR.csv](7F_4_CrossSect_NGBR_BR.csv) |
| 5 Carry / term structure | BR | smooth=1,threshold=0.15 | 130 | 73 | 57 | 0 | [7F_5_Carry_BR.csv](7F_5_Carry_BR.csv) |
| 5 Carry / term structure | NG | smooth=1,threshold=0.15 | 77 | 43 | 34 | 0 | [7F_5_Carry_NG.csv](7F_5_Carry_NG.csv) |
| 6 MA + portfolio | NG | fast=20,slow=100 | 66 | 38 | 28 | 0 | [7F_6_PortfolioMA_NGBR_NG.csv](7F_6_PortfolioMA_NGBR_NG.csv) |
| 6 MA + portfolio | BR | fast=20,slow=100 | 63 | 43 | 20 | 0 | [7F_6_PortfolioMA_NGBR_BR.csv](7F_6_PortfolioMA_NGBR_BR.csv) |
| 7 Momentum + reversal | BR | momentum=60,reversal=750 | 188 | 95 | 93 | 0 | [7F_7_MomRev_BR.csv](7F_7_MomRev_BR.csv) |
| 7 Momentum + reversal | NG | momentum=60,reversal=500 | 120 | 57 | 63 | 0 | [7F_7_MomRev_NG.csv](7F_7_MomRev_NG.csv) |
| Пересечение скользящих средних, 5 мин | NG | EMA12 × WMA131, наклон A_pct, выход B_slow_flip | 1067 | 405 | 657 | 5 | [MA_3_NG_R1.csv](MA_3_NG_R1.csv) |
| Пересечение скользящих средних, 5 мин | BR | TMA10 × SMA23, наклон B_atr, выход B_slow_flip | 6353 | 2316 | 3990 | 47 | [MA_3_BR_R1.csv](MA_3_BR_R1.csv) |
