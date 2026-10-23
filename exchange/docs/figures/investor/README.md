# Перебор сеток параметров: дашборды по стратегиям

По одному дашборду на стратегию: что оптимизировалось, карта перебора, лучшие комбинации по выборке и их результат вне выборки, капитал, устойчивость и вердикт проверки. Собраны `research/investor_charts/build.py` из файлов результатов репозитория.

Все числа — фактические: отбор только по выборке, результат вне выборки показан рядом; у семи семейств издержки вычтены, у скользящих средних сетка без издержек (условие задачи), а пересчёт при 1 тике на сделку дан в таблице.

## Time-Series Momentum

![Time-Series Momentum](01_tsmom.png)

## Dual Moving Average

![Dual Moving Average](02_dual_ma.png)

## Channel Breakout

![Channel Breakout](03_breakout.png)

## Cross-sectional momentum

![Cross-sectional momentum](04_cross_sectional.png)

## Carry / term structure

![Carry / term structure](05_carry.png)

## MA + portfolio

![MA + portfolio](06_portfolio_ma.png)

## Momentum + reversal

![Momentum + reversal](07_mom_reversal.png)

## Пересечение скользящих средних · NG

![Пересечение скользящих средних · NG](08_ma_crossover_NG.png)

## Пересечение скользящих средних · BR

![Пересечение скользящих средних · BR](09_ma_crossover_BR.png)
