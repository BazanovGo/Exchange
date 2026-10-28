# Проверка журналов сделок

Вопрос: есть ли **все** сделки **каждой** системы, которую отобрала оптимизация, и идут ли они по порядку. Ответ даёт `audit_trades.py`: каждый журнал пересчитан заново способом, не зависящим от того, как он был выгружен, и сверен построчно.

## Итог

**Расхождений нет.** Систем: 671, сделок: 5 641 196. Каждая сделка каждой системы на месте, по порядку, без пропусков и повторов; производные файлы (Excel, пункты, «только числа») повторяют журналы строка в строку.

| Что проверено | Систем | Сделок | Без расхождений | Против чего сверено |
|---|---|---|---|---|
| Семь семейств: прогоны отчёта (IS и OOS отдельно) | 12 | 1 766 | 12 из 12 | повторный прогон движка; сделок OOS — с отчётом, сделок IS — с `sweeps.csv`, итог OOS — с отчётом |
| Семь семейств: сквозной прогон по всей истории | 12 | 1 805 | 12 из 12 | повторный прогон; сумма сделок = изменение капитала; сделки до конца IS = журнал отчёта |
| Семь семейств: walk-forward, окна | 12 | 811 | 12 из 12 | повторный прогон каждого окна; эффективность, доля окон > 0 и стабильность параметров — с разделом 3 |
| Скользящие средние: глобальные отборы и лучшие в группах | 47 | 386 198 | 47 из 47 | VectorBT-журнал против событийного ядра сделка в сделку; сделки и PnL IS/OOS — с сеткой |
| Скользящие средние: лучший в каждом из 144 семейств | 576 | 5 200 421 | 576 из 576 | пересчёт ядром; сделки и PnL IS/OOS — с сеткой (`picks.csv`) |
| Скользящие средние: walk-forward, шаги | 12 | 50 195 | 12 из 12 | пересчёт ядром по шагам; итог шага — с `wf_<критерий>.csv` |
| Таблицы в пунктах и «только числа» | 30 | 10 991 | 30 из 30 | журналы сделок строка в строку |
| Листы Excel | 101 | 400 832 строк | 101 из 101 | CSV строка в строку |

Проверки каждого журнала:

* **полнота** — повторный расчёт даёт ровно те же сделки: у скользящих средних — событийное ядро исследования против файлов (финалисты выгружены из VectorBT, так что это сверка двух независимых реализаций), у семи семейств — повторный прогон движка; число сделок IS и OOS — с полной сеткой (скользящие средние), с отчётом и `sweeps.csv` (семь семейств);
* **порядок** — нумерация 1..n без пропусков, сделки по дате закрытия, вход не позже выхода; у систем с одной позицией (скользящие средние) следующая сделка открывается только после закрытия предыдущей;
* **повторы** — ни одна сделка не записана дважды;
* **разметка** — период IS/OOS соответствует дате выхода; без ролла результат 1 контракта равен разности котировок × множитель.

## Что было неполным и что добавлено

1. **Семь семейств: разрыв между IS и OOS.** Отчёт оценивает IS и OOS отдельными прогонами: OOS начинается без позиции и с пустыми индикаторами, между периодами — эмбарго 10 дней. Поэтому в журнале отчёта нет сделок за несколько месяцев после конца IS, а открытая позиция на конец IS закрыта переоценкой. Добавлен **сквозной прогон** той же системы по всей истории ([`csv/full/`](csv/full/), [`trade_log_full.xlsx`](trade_log_full.xlsx), [`points/full/`](points/full/README.md)): в нём сделки идут подряд с первого до последнего дня данных. Его сделки до конца IS совпадают с журналом отчёта одна в одну.

| Система | Конец IS | Первая сделка OOS в прогоне отчёта | Сделок сквозного прогона в этом окне | Их итог, USD |
|---|---|---|---|---|
| 7F_1_TSMOM_BR | 2021-02-01 | 2021-04-15 | 4 | +148 344.52 |
| 7F_1_TSMOM_NG | 2024-04-25 | 2024-09-18 | 13 | -27 985.16 |
| 7F_2_DualMA_BR | 2021-02-01 | 2021-05-13 | 4 | +129 588.39 |
| 7F_2_DualMA_NG | 2024-04-25 | 2024-12-10 | 8 | -37 758.43 |
| 7F_3_Breakout_BR | 2021-02-01 | 2021-06-02 | 5 | +158 630.70 |
| 7F_3_Breakout_NG | 2024-04-25 | 2024-06-13 | 3 | +51 126.12 |
| 7F_4_CrossSect_NGBR | 2024-04-25 | 2024-10-30 | 21 | +18 216.36 |
| 7F_5_Carry_BR | 2021-02-01 | 2021-03-18 | 1 | +9 078.74 |
| 7F_5_Carry_NG | 2024-04-25 | 2024-06-11 | 2 | +243.56 |
| 7F_6_PortfolioMA_NGBR | 2024-04-25 | 2024-10-01 | 10 | -30 602.43 |
| 7F_7_MomRev_BR | 2021-02-01 | 2021-05-14 | 4 | +132 380.36 |
| 7F_7_MomRev_NG | 2024-04-25 | 2024-08-07 | 5 | -78 326.44 |

2. **Скользящие средние: 144 семейства.** Разделы 7 и 12 отчёта ранжируют лучшую по плато конфигурацию каждого семейства (сочетание типов × выход × метод), но их сделок не было. Добавлены все: [`csv/families/`](csv/families/README.md), 576 файлов на этапах 1 и 3.
3. **Walk-forward с переоптимизацией** — тоже оптимизированная система: на каждом шаге свои параметры. Добавлены сделки всех шагов: [`csv/walkforward/`](csv/walkforward/), 12 систем семи семейств и 12 — скользящих средних (2 этапа × 2 рынка × 3 критерия). Итоги шагов совпадают с отчётом. У скользящих средних сделка относится к шагу по месяцу выхода, как в отчёте; вошедших до начала своего шага — 294, они помечены в «примечании».

## Совпадения: одна конфигурация — один журнал

Если два правила отбора выбрали одну и ту же конфигурацию, её сделки записаны один раз. Ниже — каждое правило, у которого нет собственного файла, и журнал, где лежат его сделки.

| Этап | Рынок | Отбор | Правило | Конфигурация | Журнал |
|---|---|---|---|---|---|
| 1 | NG | группа B_slow_flip × A_pct | R1_max_pnl | 1233232 | MA_1_NG_R1 |
| 1 | NG | группа B_slow_flip × B_atr | R2_max_sharpe | 2081382 | MA_1_NG_R2 |
| 1 | NG | группа B_slow_flip × B_atr | R3_plateau | 2115507 | MA_1_NG_R3 |
| 1 | BR | группа B_slow_flip × A_pct | R2_max_sharpe | 425406 | MA_1_BR_R2 |
| 1 | BR | группа B_slow_flip × A_pct | R3_plateau | 426582 | MA_1_BR_R3 |
| 1 | BR | группа B_slow_flip × B_atr | R1_max_pnl | 3269089 | MA_1_BR_R1 |
| 3 | NG | группа B_slow_flip × A_pct | R1_max_pnl | 727464 | MA_3_NG_R1 |
| 3 | NG | группа B_slow_flip × A_pct | R3_plateau | 289234 | MA_3_NG_R3 |
| 3 | NG | группа B_slow_flip × B_atr | R2_max_sharpe | 2043809 | MA_3_NG_R2 |
| 3 | BR | группа A_fast_cross × B_atr | R2_max_sharpe | 2926420 | MA_3_BR_xA_sB_R1 |
| 3 | BR | группа B_slow_flip × A_pct | R2_max_sharpe | 71679 | MA_3_BR_R2 |
| 3 | BR | группа B_slow_flip × A_pct | R3_plateau | 71676 | MA_3_BR_R3 |
| 3 | BR | группа B_slow_flip × B_atr | R1_max_pnl | 1758030 | MA_3_BR_R1 |

Кроме того, 16 лучших в семействах совпали с финалистами; у них есть и свой файл, и сделки в нём проверены на равенство журналу финалиста (колонка «Совпадает с» в [`csv/families/README.md`](csv/families/README.md)).

## По системам

Строка на каждую систему — файл, число сделок, первая и последняя сделка, итог и результат проверки: [`audit/systems.csv`](audit/systems.csv).

| Система | Файл | Сделок | Первый вход | Последний выход | Итог, USD | Проверка |
|---|---|---|---|---|---|---|
| MA_1_NG_R1 | [MA_1_NG_R1.csv](csv/MA_1_NG_R1.csv) | 4761 | 2020-02-04 20:05 | 2026-10-03 18:55 | +2 792.40 | ✅ |
| MA_1_NG_R2 | [MA_1_NG_R2.csv](csv/MA_1_NG_R2.csv) | 3256 | 2020-02-05 10:10 | 2026-10-03 18:55 | +1 608.40 | ✅ |
| MA_1_NG_R3 | [MA_1_NG_R3.csv](csv/MA_1_NG_R3.csv) | 2343 | 2020-02-05 10:10 | 2026-10-03 18:55 | +1 680.10 | ✅ |
| MA_1_NG_xA_sA_R1 | [MA_1_NG_xA_sA_R1.csv.gz](csv/groups/MA_1_NG_xA_sA_R1.csv.gz) | 5233 | 2020-02-05 19:35 | 2026-10-02 15:45 | +1 634.20 | ✅ |
| MA_1_NG_xA_sA_R2 | [MA_1_NG_xA_sA_R2.csv.gz](csv/groups/MA_1_NG_xA_sA_R2.csv.gz) | 7852 | 2020-02-05 18:40 | 2026-10-03 18:55 | +1 545.50 | ✅ |
| MA_1_NG_xA_sA_R3 | [MA_1_NG_xA_sA_R3.csv.gz](csv/groups/MA_1_NG_xA_sA_R3.csv.gz) | 7838 | 2020-02-05 18:40 | 2026-10-03 18:55 | +1 370.70 | ✅ |
| MA_1_NG_xA_sB_R1 | [MA_1_NG_xA_sB_R1.csv.gz](csv/groups/MA_1_NG_xA_sB_R1.csv.gz) | 7246 | 2020-02-11 18:10 | 2026-10-03 18:55 | +1 538.40 | ✅ |
| MA_1_NG_xA_sB_R2 | [MA_1_NG_xA_sB_R2.csv.gz](csv/groups/MA_1_NG_xA_sB_R2.csv.gz) | 10173 | 2020-02-05 10:10 | 2026-10-03 18:55 | +1 267.30 | ✅ |
| MA_1_NG_xA_sB_R3 | [MA_1_NG_xA_sB_R3.csv.gz](csv/groups/MA_1_NG_xA_sB_R3.csv.gz) | 6787 | 2020-02-07 10:10 | 2026-10-03 18:55 | +1 562.00 | ✅ |
| MA_1_NG_xB_sA_R2 | [MA_1_NG_xB_sA_R2.csv.gz](csv/groups/MA_1_NG_xB_sA_R2.csv.gz) | 3602 | 2020-02-04 13:10 | 2026-10-03 18:55 | +2 160.20 | ✅ |
| MA_1_NG_xB_sA_R3 | [MA_1_NG_xB_sA_R3.csv.gz](csv/groups/MA_1_NG_xB_sA_R3.csv.gz) | 2976 | 2020-02-05 10:20 | 2026-10-03 18:55 | +2 344.50 | ✅ |
| MA_1_NG_xB_sB_R1 | [MA_1_NG_xB_sB_R1.csv.gz](csv/groups/MA_1_NG_xB_sB_R1.csv.gz) | 1512 | 2020-02-05 22:15 | 2026-10-03 18:55 | +2 221.90 | ✅ |
| MA_1_NG_WF_pnl | [MA_1_NG_WF_pnl.csv.gz](csv/walkforward/MA_1_NG_WF_pnl.csv.gz) | 2270 | 2022-01-31 21:25 | 2026-10-03 18:55 | +143.20 | ✅ |
| MA_1_NG_WF_sharpe | [MA_1_NG_WF_sharpe.csv.gz](csv/walkforward/MA_1_NG_WF_sharpe.csv.gz) | 2979 | 2022-02-01 10:10 | 2026-10-01 21:05 | +711.80 | ✅ |
| MA_1_NG_WF_plateau | [MA_1_NG_WF_plateau.csv.gz](csv/walkforward/MA_1_NG_WF_plateau.csv.gz) | 2764 | 2022-01-31 21:35 | 2026-10-02 15:45 | +688.70 | ✅ |
| MA_1_BR_R1 | [MA_1_BR_R1.csv](csv/MA_1_BR_R1.csv) | 5892 | 2011-12-19 17:10 | 2026-10-03 18:55 | +3 239.80 | ✅ |
| MA_1_BR_R2 | [MA_1_BR_R2.csv](csv/MA_1_BR_R2.csv) | 4364 | 2011-12-20 11:40 | 2026-10-03 18:55 | +2 647.80 | ✅ |
| MA_1_BR_R3 | [MA_1_BR_R3.csv](csv/MA_1_BR_R3.csv) | 4350 | 2011-12-20 11:40 | 2026-10-03 18:55 | +2 612.60 | ✅ |
| MA_1_BR_xA_sA_R1 | [MA_1_BR_xA_sA_R1.csv.gz](csv/groups/MA_1_BR_xA_sA_R1.csv.gz) | 15565 | 2011-12-20 10:05 | 2026-10-03 18:55 | +1 975.80 | ✅ |
| MA_1_BR_xA_sA_R2 | [MA_1_BR_xA_sA_R2.csv.gz](csv/groups/MA_1_BR_xA_sA_R2.csv.gz) | 15532 | 2011-12-20 10:05 | 2026-10-03 18:55 | +1 880.50 | ✅ |
| MA_1_BR_xA_sA_R3 | [MA_1_BR_xA_sA_R3.csv.gz](csv/groups/MA_1_BR_xA_sA_R3.csv.gz) | 15446 | 2011-12-20 10:05 | 2026-10-03 18:55 | +1 935.60 | ✅ |
| MA_1_BR_xA_sB_R1 | [MA_1_BR_xA_sB_R1.csv.gz](csv/groups/MA_1_BR_xA_sB_R1.csv.gz) | 13959 | 2011-12-19 23:10 | 2026-10-03 18:15 | +1 175.80 | ✅ |
| MA_1_BR_xA_sB_R2 | [MA_1_BR_xA_sB_R2.csv.gz](csv/groups/MA_1_BR_xA_sB_R2.csv.gz) | 22284 | 2011-12-19 18:05 | 2026-10-03 18:15 | +1 656.40 | ✅ |
| MA_1_BR_xA_sB_R3 | [MA_1_BR_xA_sB_R3.csv.gz](csv/groups/MA_1_BR_xA_sB_R3.csv.gz) | 21176 | 2011-12-20 16:25 | 2026-10-03 18:55 | +1 671.20 | ✅ |
| MA_1_BR_xB_sA_R1 | [MA_1_BR_xB_sA_R1.csv.gz](csv/groups/MA_1_BR_xB_sA_R1.csv.gz) | 4384 | 2011-12-20 10:05 | 2026-10-03 18:55 | +2 862.00 | ✅ |
| MA_1_BR_xB_sB_R2 | [MA_1_BR_xB_sB_R2.csv.gz](csv/groups/MA_1_BR_xB_sB_R2.csv.gz) | 4945 | 2011-12-20 10:05 | 2026-10-03 18:55 | +2 539.90 | ✅ |
| MA_1_BR_xB_sB_R3 | [MA_1_BR_xB_sB_R3.csv.gz](csv/groups/MA_1_BR_xB_sB_R3.csv.gz) | 4869 | 2011-12-20 10:05 | 2026-10-03 18:55 | +2 303.00 | ✅ |
| MA_1_BR_WF_pnl | [MA_1_BR_WF_pnl.csv.gz](csv/walkforward/MA_1_BR_WF_pnl.csv.gz) | 6207 | 2013-11-29 18:10 | 2026-10-03 18:55 | +1 132.40 | ✅ |
| MA_1_BR_WF_sharpe | [MA_1_BR_WF_sharpe.csv.gz](csv/walkforward/MA_1_BR_WF_sharpe.csv.gz) | 7636 | 2013-11-29 19:50 | 2026-10-03 18:55 | +820.20 | ✅ |
| MA_1_BR_WF_plateau | [MA_1_BR_WF_plateau.csv.gz](csv/walkforward/MA_1_BR_WF_plateau.csv.gz) | 5046 | 2013-12-02 10:45 | 2026-10-02 19:35 | +1 586.00 | ✅ |
| MA_3_NG_R1 | [MA_3_NG_R1.csv](csv/MA_3_NG_R1.csv) | 1067 | 2020-02-06 12:00 | 2026-10-03 18:55 | +2 335.90 | ✅ |
| MA_3_NG_R2 | [MA_3_NG_R2.csv](csv/MA_3_NG_R2.csv) | 3039 | 2020-02-04 16:10 | 2026-10-03 18:55 | +2 036.40 | ✅ |
| MA_3_NG_R3 | [MA_3_NG_R3.csv](csv/MA_3_NG_R3.csv) | 520 | 2020-02-10 15:55 | 2026-10-03 18:55 | +1 010.20 | ✅ |
| MA_3_NG_xA_sA_R1 | [MA_3_NG_xA_sA_R1.csv.gz](csv/groups/MA_3_NG_xA_sA_R1.csv.gz) | 4940 | 2020-02-07 22:35 | 2026-10-02 15:45 | +1 562.80 | ✅ |
| MA_3_NG_xA_sA_R2 | [MA_3_NG_xA_sA_R2.csv.gz](csv/groups/MA_3_NG_xA_sA_R2.csv.gz) | 7956 | 2020-02-05 18:40 | 2026-10-03 18:55 | +1 222.40 | ✅ |
| MA_3_NG_xA_sA_R3 | [MA_3_NG_xA_sA_R3.csv.gz](csv/groups/MA_3_NG_xA_sA_R3.csv.gz) | 8469 | 2020-02-11 17:15 | 2026-10-03 18:55 | +1 319.70 | ✅ |
| MA_3_NG_xA_sB_R1 | [MA_3_NG_xA_sB_R1.csv.gz](csv/groups/MA_3_NG_xA_sB_R1.csv.gz) | 4881 | 2020-02-14 15:20 | 2026-10-02 15:45 | +1 443.50 | ✅ |
| MA_3_NG_xA_sB_R2 | [MA_3_NG_xA_sB_R2.csv.gz](csv/groups/MA_3_NG_xA_sB_R2.csv.gz) | 6835 | 2020-02-07 20:00 | 2026-10-03 18:55 | +1 419.70 | ✅ |
| MA_3_NG_xA_sB_R3 | [MA_3_NG_xA_sB_R3.csv.gz](csv/groups/MA_3_NG_xA_sB_R3.csv.gz) | 8433 | 2020-02-06 19:15 | 2026-10-03 18:55 | +1 434.30 | ✅ |
| MA_3_NG_xB_sA_R2 | [MA_3_NG_xB_sA_R2.csv.gz](csv/groups/MA_3_NG_xB_sA_R2.csv.gz) | 1929 | 2020-02-04 22:25 | 2026-10-03 18:55 | +2 007.20 | ✅ |
| MA_3_NG_xB_sB_R1 | [MA_3_NG_xB_sB_R1.csv.gz](csv/groups/MA_3_NG_xB_sB_R1.csv.gz) | 1672 | 2020-02-05 13:55 | 2026-10-03 18:55 | +2 354.70 | ✅ |
| MA_3_NG_xB_sB_R3 | [MA_3_NG_xB_sB_R3.csv.gz](csv/groups/MA_3_NG_xB_sB_R3.csv.gz) | 3032 | 2020-02-04 16:10 | 2026-10-03 18:55 | +1 913.20 | ✅ |
| MA_3_NG_WF_pnl | [MA_3_NG_WF_pnl.csv.gz](csv/walkforward/MA_3_NG_WF_pnl.csv.gz) | 1477 | 2022-01-31 23:20 | 2026-10-03 18:55 | -155.30 | ✅ |
| MA_3_NG_WF_sharpe | [MA_3_NG_WF_sharpe.csv.gz](csv/walkforward/MA_3_NG_WF_sharpe.csv.gz) | 3807 | 2022-02-01 07:05 | 2026-10-03 18:55 | +629.20 | ✅ |
| MA_3_NG_WF_plateau | [MA_3_NG_WF_plateau.csv.gz](csv/walkforward/MA_3_NG_WF_plateau.csv.gz) | 3729 | 2022-02-01 09:40 | 2026-10-03 18:55 | +391.80 | ✅ |
| MA_3_BR_R1 | [MA_3_BR_R1.csv](csv/MA_3_BR_R1.csv) | 6353 | 2011-12-19 14:55 | 2026-10-03 18:55 | +3 607.10 | ✅ |
| MA_3_BR_R2 | [MA_3_BR_R2.csv](csv/MA_3_BR_R2.csv) | 2188 | 2011-12-20 10:05 | 2026-10-03 18:55 | +3 390.80 | ✅ |
| MA_3_BR_R3 | [MA_3_BR_R3.csv](csv/MA_3_BR_R3.csv) | 2220 | 2011-12-20 10:05 | 2026-10-03 18:55 | +3 218.40 | ✅ |
| MA_3_BR_xA_sA_R1 | [MA_3_BR_xA_sA_R1.csv.gz](csv/groups/MA_3_BR_xA_sA_R1.csv.gz) | 23398 | 2011-12-19 18:05 | 2026-10-03 18:55 | +1 959.30 | ✅ |
| MA_3_BR_xA_sA_R2 | [MA_3_BR_xA_sA_R2.csv.gz](csv/groups/MA_3_BR_xA_sA_R2.csv.gz) | 23350 | 2011-12-19 18:20 | 2026-10-03 18:55 | +2 091.80 | ✅ |
| MA_3_BR_xA_sA_R3 | [MA_3_BR_xA_sA_R3.csv.gz](csv/groups/MA_3_BR_xA_sA_R3.csv.gz) | 24108 | 2011-12-19 18:20 | 2026-10-03 18:55 | +1 759.20 | ✅ |
| MA_3_BR_xA_sB_R1 | [MA_3_BR_xA_sB_R1.csv.gz](csv/groups/MA_3_BR_xA_sB_R1.csv.gz) | 21953 | 2011-12-19 17:10 | 2026-10-03 18:15 | +1 989.50 | ✅ |
| MA_3_BR_xA_sB_R3 | [MA_3_BR_xA_sB_R3.csv.gz](csv/groups/MA_3_BR_xA_sB_R3.csv.gz) | 22364 | 2011-12-19 17:10 | 2026-10-03 18:15 | +1 665.40 | ✅ |
| MA_3_BR_xB_sA_R1 | [MA_3_BR_xB_sA_R1.csv.gz](csv/groups/MA_3_BR_xB_sA_R1.csv.gz) | 2855 | 2011-12-20 10:05 | 2026-10-03 18:55 | +3 649.40 | ✅ |
| MA_3_BR_xB_sB_R2 | [MA_3_BR_xB_sB_R2.csv.gz](csv/groups/MA_3_BR_xB_sB_R2.csv.gz) | 3873 | 2011-12-19 18:20 | 2026-10-03 18:55 | +3 248.20 | ✅ |
| MA_3_BR_xB_sB_R3 | [MA_3_BR_xB_sB_R3.csv.gz](csv/groups/MA_3_BR_xB_sB_R3.csv.gz) | 4418 | 2011-12-19 18:20 | 2026-10-03 18:55 | +2 600.50 | ✅ |
| MA_3_BR_WF_pnl | [MA_3_BR_WF_pnl.csv.gz](csv/walkforward/MA_3_BR_WF_pnl.csv.gz) | 3085 | 2013-11-29 23:15 | 2026-10-03 18:55 | +129.90 | ✅ |
| MA_3_BR_WF_sharpe | [MA_3_BR_WF_sharpe.csv.gz](csv/walkforward/MA_3_BR_WF_sharpe.csv.gz) | 5490 | 2013-12-02 10:50 | 2026-10-03 18:55 | +1 787.90 | ✅ |
| MA_3_BR_WF_plateau | [MA_3_BR_WF_plateau.csv.gz](csv/walkforward/MA_3_BR_WF_plateau.csv.gz) | 5705 | 2013-12-02 10:45 | 2026-10-02 20:00 | +878.10 | ✅ |
| 7F_1_TSMOM_BR | [7F_1_TSMOM_BR.csv](csv/7F_1_TSMOM_BR.csv) | 344 | 2012-02-15 00:00 | 2026-10-03 00:00 | +1 484 797.17 | ✅ |
| 7F_1_TSMOM_BR | [7F_1_TSMOM_BR.csv](csv/full/7F_1_TSMOM_BR.csv) | 348 | 2012-02-15 00:00 | 2026-10-03 00:00 | +1 613 502.58 | ✅ |
| 7F_1_TSMOM_BR_WF | [7F_1_TSMOM_BR_WF.csv](csv/walkforward/7F_1_TSMOM_BR_WF.csv) | 175 | 2017-06-19 00:00 | 2026-08-21 00:00 | +130 071.09 | ✅ |
| 7F_1_TSMOM_NG | [7F_1_TSMOM_NG.csv](csv/7F_1_TSMOM_NG.csv) | 117 | 2020-06-16 00:00 | 2026-10-03 00:00 | +263 889.04 | ✅ |
| 7F_1_TSMOM_NG | [7F_1_TSMOM_NG.csv](csv/full/7F_1_TSMOM_NG.csv) | 131 | 2020-06-16 00:00 | 2026-10-03 00:00 | +120 534.26 | ✅ |
| 7F_1_TSMOM_NG_WF | [7F_1_TSMOM_NG_WF.csv](csv/walkforward/7F_1_TSMOM_NG_WF.csv) | 86 | 2022-08-25 00:00 | 2026-09-08 00:00 | +31 991.44 | ✅ |
| 7F_2_DualMA_BR | [7F_2_DualMA_BR.csv](csv/7F_2_DualMA_BR.csv) | 172 | 2012-03-15 00:00 | 2026-10-03 00:00 | +1 977 720.62 | ✅ |
| 7F_2_DualMA_BR | [7F_2_DualMA_BR.csv](csv/full/7F_2_DualMA_BR.csv) | 175 | 2012-03-15 00:00 | 2026-10-03 00:00 | +2 135 886.44 | ✅ |
| 7F_2_DualMA_BR_WF | [7F_2_DualMA_BR_WF.csv](csv/walkforward/7F_2_DualMA_BR_WF.csv) | 75 | 2017-07-14 00:00 | 2026-08-21 00:00 | -87 364.30 | ✅ |
| 7F_2_DualMA_NG | [7F_2_DualMA_NG.csv](csv/7F_2_DualMA_NG.csv) | 60 | 2020-09-09 00:00 | 2026-10-03 00:00 | +213 698.97 | ✅ |
| 7F_2_DualMA_NG | [7F_2_DualMA_NG.csv](csv/full/7F_2_DualMA_NG.csv) | 66 | 2020-09-09 00:00 | 2026-10-03 00:00 | +85 324.64 | ✅ |
| 7F_2_DualMA_NG_WF | [7F_2_DualMA_NG_WF.csv](csv/walkforward/7F_2_DualMA_NG_WF.csv) | 11 | 2023-07-11 00:00 | 2025-05-08 00:00 | -179 845.84 | ✅ |
| 7F_3_Breakout_BR | [7F_3_Breakout_BR.csv](csv/7F_3_Breakout_BR.csv) | 126 | 2012-02-17 00:00 | 2026-10-03 00:00 | +2 088 796.28 | ✅ |
| 7F_3_Breakout_BR | [7F_3_Breakout_BR.csv](csv/full/7F_3_Breakout_BR.csv) | 130 | 2012-02-17 00:00 | 2026-10-03 00:00 | +2 256 229.01 | ✅ |
| 7F_3_Breakout_BR_WF | [7F_3_Breakout_BR_WF.csv](csv/walkforward/7F_3_Breakout_BR_WF.csv) | 63 | 2017-09-21 00:00 | 2026-08-21 00:00 | +686 969.09 | ✅ |
| 7F_3_Breakout_NG | [7F_3_Breakout_NG.csv](csv/7F_3_Breakout_NG.csv) | 75 | 2020-03-11 00:00 | 2026-10-03 00:00 | +319 541.44 | ✅ |
| 7F_3_Breakout_NG | [7F_3_Breakout_NG.csv](csv/full/7F_3_Breakout_NG.csv) | 79 | 2020-03-11 00:00 | 2026-10-03 00:00 | +350 122.62 | ✅ |
| 7F_3_Breakout_NG_WF | [7F_3_Breakout_NG_WF.csv](csv/walkforward/7F_3_Breakout_NG_WF.csv) | 35 | 2022-09-19 00:00 | 2026-09-08 00:00 | +15 987.04 | ✅ |
| 7F_4_CrossSect_NGBR | [7F_4_CrossSect_NGBR.csv](csv/7F_4_CrossSect_NGBR.csv) | 228 | 2020-07-30 00:00 | 2026-10-03 00:00 | -121 157.11 | ✅ |
| 7F_4_CrossSect_NGBR | [7F_4_CrossSect_NGBR.csv](csv/full/7F_4_CrossSect_NGBR.csv) | 249 | 2020-07-30 00:00 | 2026-10-03 00:00 | -217 772.77 | ✅ |
| 7F_4_CrossSect_NGBR_WF | [7F_4_CrossSect_NGBR_WF.csv](csv/walkforward/7F_4_CrossSect_NGBR_WF.csv) | 51 | 2023-01-16 00:00 | 2026-04-11 00:00 | -28 106.33 | ✅ |
| 7F_5_Carry_BR | [7F_5_Carry_BR.csv](csv/7F_5_Carry_BR.csv) | 130 | 2012-03-20 00:00 | 2026-10-03 00:00 | +1 525 359.47 | ✅ |
| 7F_5_Carry_BR | [7F_5_Carry_BR.csv](csv/full/7F_5_Carry_BR.csv) | 135 | 2012-03-20 00:00 | 2026-10-03 00:00 | +1 572 447.76 | ✅ |
| 7F_5_Carry_BR_WF | [7F_5_Carry_BR_WF.csv](csv/walkforward/7F_5_Carry_BR_WF.csv) | 99 | 2017-05-19 00:00 | 2026-08-21 00:00 | +546 366.56 | ✅ |
| 7F_5_Carry_NG | [7F_5_Carry_NG.csv](csv/7F_5_Carry_NG.csv) | 77 | 2020-03-04 00:00 | 2026-10-03 00:00 | +101 166.49 | ✅ |
| 7F_5_Carry_NG | [7F_5_Carry_NG.csv](csv/full/7F_5_Carry_NG.csv) | 78 | 2020-03-04 00:00 | 2026-10-03 00:00 | -13 688.37 | ✅ |
| 7F_5_Carry_NG_WF | [7F_5_Carry_NG_WF.csv](csv/walkforward/7F_5_Carry_NG_WF.csv) | 42 | 2022-08-25 00:00 | 2026-09-08 00:00 | +83 778.37 | ✅ |
| 7F_6_PortfolioMA_NGBR | [7F_6_PortfolioMA_NGBR.csv](csv/7F_6_PortfolioMA_NGBR.csv) | 129 | 2020-06-30 00:00 | 2026-10-03 00:00 | +435 748.89 | ✅ |
| 7F_6_PortfolioMA_NGBR | [7F_6_PortfolioMA_NGBR.csv](csv/full/7F_6_PortfolioMA_NGBR.csv) | 139 | 2020-06-30 00:00 | 2026-10-03 00:00 | +312 686.05 | ✅ |
| 7F_6_PortfolioMA_NGBR_WF | [7F_6_PortfolioMA_NGBR_WF.csv](csv/walkforward/7F_6_PortfolioMA_NGBR_WF.csv) | 17 | 2023-07-11 00:00 | 2026-09-08 00:00 | -38 425.96 | ✅ |
| 7F_7_MomRev_BR | [7F_7_MomRev_BR.csv](csv/7F_7_MomRev_BR.csv) | 188 | 2012-03-16 00:00 | 2026-08-17 00:00 | +559 394.60 | ✅ |
| 7F_7_MomRev_BR | [7F_7_MomRev_BR.csv](csv/full/7F_7_MomRev_BR.csv) | 170 | 2012-03-16 00:00 | 2026-08-17 00:00 | +742 065.70 | ✅ |
| 7F_7_MomRev_BR_WF | [7F_7_MomRev_BR_WF.csv](csv/walkforward/7F_7_MomRev_BR_WF.csv) | 140 | 2017-07-17 00:00 | 2026-08-21 00:00 | +225 518.18 | ✅ |
| 7F_7_MomRev_NG | [7F_7_MomRev_NG.csv](csv/7F_7_MomRev_NG.csv) | 120 | 2020-04-30 00:00 | 2026-10-03 00:00 | -27 260.92 | ✅ |
| 7F_7_MomRev_NG | [7F_7_MomRev_NG.csv](csv/full/7F_7_MomRev_NG.csv) | 105 | 2020-04-30 00:00 | 2026-10-03 00:00 | -51 326.58 | ✅ |
| 7F_7_MomRev_NG_WF | [7F_7_MomRev_NG_WF.csv](csv/walkforward/7F_7_MomRev_NG_WF.csv) | 17 | 2023-05-16 00:00 | 2025-05-08 00:00 | -35 644.32 | ✅ |

Семейства (576) — в `audit/systems.csv`.

Пересборка: `python audit_trades.py ma` (окружение исследования), `python audit_trades.py seven` (окружение пакета `exchange`), `python audit_trades.py files`.
