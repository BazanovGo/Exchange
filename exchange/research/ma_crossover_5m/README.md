# Скользящие средние на 5-минутках NG и BR

Полное исследование стратегии «цена пересекает быструю среднюю, направление
разрешает наклон медленной» на всех доступных 5-минутных фьючерсах MOEX
(NG, BR). Отчёт — `../../docs/RESULTS_MA_CROSSOVER_5M.md`, графики —
`../../notebooks/03_ma_crossover_5m.ipynb`.

## Окружение

VectorBT 1.0 несовместим с pandas 3 и plotly 6, поэтому у исследования своё
окружение:

```bash
uv venv .venv-vbt --python 3.12
VIRTUAL_ENV=.venv-vbt uv pip install vectorbt "pandas<3" "plotly<6" pyarrow matplotlib scipy \
    nbformat nbclient ipykernel pytest
```

## Порядок запуска

```bash
export EXCHANGE_MOEX_DATA=/path/to/moex_algopack   # каталоги ng/5m, br/5m
export MA5M_WORK=/path/to/work                      # ~12 ГБ под сетки
python -m pytest -q test_ma.py test_kernel.py test_robust.py
python validate_vbt.py $MA5M_WORK/validation_vbt.csv  # ядро против VectorBT
python stage1.py NG BR          # полная сетка, исходный наклон      (~20 + ~40 мин)
python verify_grid.py stage1 NG # раскладка сетки против пересчёта
python analyze.py stage1 NG BR  # метрики, семейства, отбор, WF, PBO, RC/SPA, DSR
python extras.py stage1 NG BR   # издержки, эталон WF, годы
python crossmarket.py stage1
python stage2.py NG BR          # оптимизация наклона на решётке     (~40 мин)
python analyze_stage2.py NG BR  # -> $MA5M_WORK/stage3/slope_params.json
python stage2b.py NG BR && python analyze_stage2.py stage2b NG BR  # продление k за границу
python stage3.py NG BR          # полная сетка, оптимизированный наклон (~60 мин)
python analyze.py stage3 NG BR && python extras.py stage3 NG BR && python crossmarket.py stage3
python years.py stage1 NG BR && python years.py stage3 NG BR   # годы и OOS без последнего года
python finalists.py stage3 NG BR  # VectorBT: полная статистика, Monte Carlo
python nulls.py                   # нулевые рынки
python -c "import signflip; signflip.main(n_paths=200)"   # тест траекторий
python export.py stage1 stage3    # машиночитаемые результаты в results/
python build_notebook.py          # ноутбук
python export_figures.py          # графики ноутбука -> docs/figures/ma_crossover_5m/
python report.py stage1 stage3    # отчёт (ссылается на графики)
```

## Что лежит в `results/`

- таблицы анализа по этапам и инструментам (CSV/JSON): семейства, отборы, walk-forward, PBO, RC/SPA,
  издержки, годы, кросс-рынок, финалисты VectorBT, нулевые рынки, тест траекторий;
- `top10000_by_plateau_is.parquet` — верх сетки со всеми метриками;
- `grid_A_pct.parquet`, `grid_B_atr.parquet` — **все** 3 471 552 конфигурации на инструмент и этап:
  параметры, PnL и просадка IS/OOS (USD, до 1), сделки IS/OOS, месячный Шарп IS/OOS и плато-Шарп IS
  (×1000, целые); `grid_meta.json` — оси сетки и параметры наклона.
- `finalists/` — помесячный и годовой PnL финалистов (CSV) и их дневная кривая капитала с переоценкой
  (`cfg<номер>_equity.parquet`, VectorBT); все сделки финалистов — в `../trade_logs/`.
- [`runs/`](runs/README.md) — сценарии и журналы исходного прогона, включая сверки сетки с пересчётом.

```python
from export import load_full_grid
g = load_full_grid("stage3", "NG")   # 3.47 млн строк, метрики в исходных единицах
```

Сырые `stats.npy`/`monthly.npy` (~8.5 ГБ) и `summary.parquet` (~2 ГБ) в git не помещаются и
пересчитываются командами выше.

## Устройство

| Файл | Что делает |
|---|---|
| `data.py` | загрузка 5m, роллы по `secid`, ratio-ряд для сигналов, аддитивный — для PnL, множители |
| `ma.py` | SMA, EMA, WMA, SSMA, VWMA, TMA и ATR — формулы в docstring |
| `slope.py` | наклон методами A и B, ε через долю нейтрали на IS |
| `kernel.py` | событийное numba-ядро: правила входа/выхода, учёт по IS/OOS и месяцам |
| `grid.py`, `stage*.py` | пакетный прогон сеток в memmap |
| `results.py`, `analyze*.py`, `extras.py`, `crossmarket.py`, `robust.py` | анализ и проверки устойчивости |
| `finalists.py` | финалисты через `vectorbt.Portfolio.from_signals` |
| `validate_vbt.py`, `verify_grid.py`, `test_*.py`, `nulls.py` | проверки корректности |
| `report.py`, `export.py`, `build_notebook.py`, `export_figures.py`, `nb/`, `plots.py` | отчёт, выгрузка, ноутбук, графики |
