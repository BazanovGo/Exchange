"""Презентационные дашборды перебора сеток параметров — по одному на стратегию.

    python build.py          -> docs/figures/investor/*.png и README.md

Данные — только файлы репозитория: research/seven_families/results/ (полный
перебор семи семейств, sweep.py) и research/ma_crossover_5m/results/ (полная
сетка, финалисты, годовой срез из export_slice.py), плюс сгенерированные отчёты.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import ma
import seven

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = Path(__file__).resolve().parents[2] / "docs" / "figures" / "investor"

CHARTS = [
    ("01_tsmom.png", "Time-Series Momentum", lambda p: seven.build("1_tsmom", p)),
    ("02_dual_ma.png", "Dual Moving Average", lambda p: seven.build("2_dual_ma", p)),
    ("03_breakout.png", "Channel Breakout", lambda p: seven.build("3_breakout", p)),
    ("04_cross_sectional.png", "Cross-sectional momentum", lambda p: seven.build("4_cross_sectional", p)),
    ("05_carry.png", "Carry / term structure", lambda p: seven.build("5_carry", p)),
    ("06_portfolio_ma.png", "MA + portfolio", lambda p: seven.build("6_portfolio_ma", p)),
    ("07_mom_reversal.png", "Momentum + reversal", lambda p: seven.build("7_mom_reversal", p)),
    ("08_ma_crossover_NG.png", "Пересечение скользящих средних · NG", lambda p: ma.build("NG", p)),
    ("09_ma_crossover_BR.png", "Пересечение скользящих средних · BR", lambda p: ma.build("BR", p)),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, title, make in CHARTS:
        make(OUT / name)
        print(f"{name:28} {title:38} {(OUT / name).stat().st_size / 1e6:.1f} МБ")
    lines = ["# Перебор сеток параметров: дашборды по стратегиям", "",
             "По одному дашборду на стратегию: что оптимизировалось, карта перебора, лучшие комбинации "
             "по выборке и их результат вне выборки, капитал, устойчивость и вердикт проверки. "
             "Собраны `research/investor_charts/build.py` из файлов результатов репозитория.", "",
             "Все числа — фактические: отбор только по выборке, результат вне выборки показан рядом; у "
             "семи семейств издержки вычтены, у скользящих средних сетка без издержек (условие задачи), "
             "а пересчёт при 1 тике на сделку дан в таблице.", ""]
    for name, title, _ in CHARTS:
        lines += [f"## {title}", "", f"![{title}]({name})", ""]
    (OUT / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
