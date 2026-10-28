# Исходы сделок: только числа

По файлу на стратегию и актив: результат каждой сделки по порядку, одно число в строке, без заголовка. **Пункт — минимальный шаг цены**: у BR 0.01 USD за баррель, у NG 0.001 USD за MMBtu. Плюс — прибыльный исход, минус — убыточный, 0 — выход по цене входа.

Параметры — выбранные оптимизатором (лучший Шарп в выборке). Издержки вычтены, результат дан на единицу позиции и округлён до целого шага. Подробные таблицы с датами, ценами и USD — в [`../../full/`](../../full/README.md).

**Сквозной прогон**: семь семейств торгуют всю историю одним прогоном, без разрыва между выборкой и проверкой, поэтому здесь все исходы подряд с первого до последнего дня данных. Исходы прогонов отчёта (отдельный IS и отдельный OOS) — в [`../`](../README.md).

| Стратегия | Актив | Параметры | Исходов | Плюс | Минус | Ноль | Файл |
|---|---|---|---|---|---|---|---|
| 1 Time-Series Momentum | BR | lookback=40 | 348 | 169 | 178 | 1 | [7F_1_TSMOM_BR.csv](7F_1_TSMOM_BR.csv) |
| 1 Time-Series Momentum | NG | lookback=90 | 131 | 61 | 70 | 0 | [7F_1_TSMOM_NG.csv](7F_1_TSMOM_NG.csv) |
| 2 Dual Moving Average | BR | fast=20,slow=60 | 175 | 111 | 64 | 0 | [7F_2_DualMA_BR.csv](7F_2_DualMA_BR.csv) |
| 2 Dual Moving Average | NG | fast=20,slow=150 | 66 | 40 | 26 | 0 | [7F_2_DualMA_NG.csv](7F_2_DualMA_NG.csv) |
| 3 Channel Breakout | BR | entry=40,exit_=20 | 130 | 92 | 38 | 0 | [7F_3_Breakout_BR.csv](7F_3_Breakout_BR.csv) |
| 3 Channel Breakout | NG | entry=20,exit_=10 | 79 | 51 | 28 | 0 | [7F_3_Breakout_NG.csv](7F_3_Breakout_NG.csv) |
| 4 Cross-sectional momentum | NG | lookback=120 | 130 | 66 | 64 | 0 | [7F_4_CrossSect_NGBR_NG.csv](7F_4_CrossSect_NGBR_NG.csv) |
| 4 Cross-sectional momentum | BR | lookback=120 | 119 | 48 | 71 | 0 | [7F_4_CrossSect_NGBR_BR.csv](7F_4_CrossSect_NGBR_BR.csv) |
| 5 Carry / term structure | BR | smooth=1,threshold=0.15 | 135 | 77 | 58 | 0 | [7F_5_Carry_BR.csv](7F_5_Carry_BR.csv) |
| 5 Carry / term structure | NG | smooth=1,threshold=0.15 | 78 | 43 | 35 | 0 | [7F_5_Carry_NG.csv](7F_5_Carry_NG.csv) |
| 6 MA + portfolio | NG | fast=20,slow=100 | 72 | 39 | 33 | 0 | [7F_6_PortfolioMA_NGBR_NG.csv](7F_6_PortfolioMA_NGBR_NG.csv) |
| 6 MA + portfolio | BR | fast=20,slow=100 | 67 | 45 | 22 | 0 | [7F_6_PortfolioMA_NGBR_BR.csv](7F_6_PortfolioMA_NGBR_BR.csv) |
| 7 Momentum + reversal | BR | momentum=60,reversal=750 | 170 | 87 | 83 | 0 | [7F_7_MomRev_BR.csv](7F_7_MomRev_BR.csv) |
| 7 Momentum + reversal | NG | momentum=60,reversal=500 | 105 | 52 | 53 | 0 | [7F_7_MomRev_NG.csv](7F_7_MomRev_NG.csv) |
