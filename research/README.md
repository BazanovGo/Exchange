# research/ — журнал автономного исследовательского цикла

* `catalog.csv` — реестр всех проверенных гипотез (вердикты, метрики, причины).
* `H*/` — полный журнал каждой гипотезы: `report.md` (идея, правила,
  параметры, результаты, выводы), `sweep.csv` (все конфигурации IS/OOS),
  `walk_forward.csv`, `equity.png`, `param_heatmap.png`.
* База знаний (уроки, тупики, направления): [`../knowledge_base.md`](../knowledge_base.md).

Пайплайн одной гипотезы — `src/optimization/hypothesis.py::run_full_study`:
backtest → оптимизация сетки → плато устойчивости → OOS → walk-forward →
Monte Carlo → вердикт (`promising` / `viable` / `archived` + причины).
Код гипотез — `src/strategies/experimental.py`.
