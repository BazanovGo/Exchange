# Exchange — Trading Strategy Research Framework

Промышленный фреймворк для исследования торговых стратегий на базе
[vectorbt](https://vectorbt.dev). Пять этапов исследования — от инфраструктуры
до управления капиталом — с одним сквозным принципом: **любой эффект
подтверждается out-of-sample**, победители никогда не выбираются по одной
доходности.

## Быстрый старт

Требуется Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps

pytest                      # 87 тестов инфраструктуры (сеть не нужна)
```

Ноутбуки выполняются по порядку (каждый самодостаточен и пересчитывает всё,
что ему нужно; артефакты сохраняются в `results/`):

```bash
python scripts/run_notebook.py notebooks/01_repository_setup.ipynb
python scripts/run_notebook.py notebooks/02_strategy_research.ipynb
python scripts/run_notebook.py notebooks/03_position_management.ipynb
python scripts/run_notebook.py notebooks/04_pattern_discovery.ipynb
python scripts/run_notebook.py notebooks/05_final_report.ipynb
```

`scripts/run_notebook.py` выполняет ноутбук и вычищает widget-метаданные,
чтобы все графики (статичные PNG через plotly + kaleido) корректно
отображались на GitHub. Без сети ноутбуки автоматически переключаются с
Yahoo Finance на детерминированный локальный сэмпл `data/raw/SAMPLE.csv` —
через тот же интерфейс загрузки.

## Структура

```
├── data/                  # рыночные данные (в git — только SAMPLE.csv)
├── notebooks/             # пять этапов исследования (см. ниже)
├── reports/               # сохраненные отчеты
├── results/               # артефакты прогонов (метрики, таблицы, кривые)
├── scripts/               # run_notebook.py — запуск ноутбуков для GitHub
├── src/
│   ├── data/              # единый интерфейс загрузки + реестр источников
│   ├── indicators/        # 20+ индикаторов (vectorbt + numba-реализации)
│   ├── strategies/        # 26 стратегий 4 классов, единый контракт сигналов
│   ├── backtesting/       # запуск бэктеста одной функцией + симулятор выходов
│   ├── analytics/         # метрики, отчеты, рыночные признаки без lookahead
│   ├── optimization/      # grid search, IS/OOS research, walk-forward, фильтры
│   ├── risk/              # риск-метрики, Kelly, Monte Carlo, position sizing
│   ├── visualization/     # plotly-графики, GitHub-совместимый PNG-рендеринг
│   └── utils/             # пути, логирование
└── tests/                 # pytest (87 тестов)
```

## Этапы исследования (ноутбуки)

| # | Ноутбук | Содержание |
|---|---------|------------|
| 01 | `01_repository_setup.ipynb` | Инфраструктура end-to-end: данные → график → индикатор → бэктест → отчет |
| 02 | `02_strategy_research.ipynb` | 26 стратегий, ~850 конфигураций, IS/OOS 70/30, интегральный рейтинг по 11 метрикам, ТОП-20, флаги переобучения |
| 03 | `03_position_management.ipynb` | SL/TP/Trailing/BreakEven/TimeStop/частичная фиксация + комбинации; честные вердикты real/false improvement |
| 04 | `04_pattern_discovery.ipynb` | Рыночные признаки, режимы, корреляции, feature importance; проверка фильтров бэктестом IS→OOS |
| 05 | `05_final_report.ipynb` | Kelly (classical/half/quarter + ограничения), Monte Carlo (разорение, распределения), Walk-Forward, сводка переобучения, рекомендации |

## Ключевые интерфейсы

### Данные: `load_data(source, symbols, ...)`

```python
from src.data import load_data

data = load_data("yahoo", "AAPL", start="2020-01-01")   # Yahoo Finance
data = load_data("csv", "SAMPLE")                        # data/raw/SAMPLE.csv
data = load_data("parquet", "BTCUSD")                    # data/processed/
close = data.close                                       # wide-фрейм по символам
```

Все источники возвращают канонический OHLCV (`DatetimeIndex` +
`Open/High/Low/Close/Volume`). Новый источник (MOEX, Polygon, ...) — один
подкласс `BaseDataLoader` с декоратором `@register_loader`, call-sites не
меняются.

### Стратегии: единый контракт

Каждая стратегия возвращает `StrategySignals`: `entries`, `exits`,
`short_entries`, `short_exits`, `params`, `description` — плюс несёт
`category` и сетку оптимизации `opt_grid`:

```python
from src.strategies import get_strategy, strategy_catalog

strategy_catalog()                                   # обзор всех 26 стратегий
strat = get_strategy("ema_rsi", fast_window=20, slow_window=200)
sig = strat.signals(data)
```

Классы: трендовые (sma/ema/triple-ema cross, supertrend, donchian, adx,
psar, ichimoku), контртрендовые (rsi, bollinger, cci, williams %r,
stochastic, vwap- и zscore-reversion), импульсные (roc, momentum, macd,
atr/keltner breakout, volatility expansion), гибридные (ema+rsi, ema+adx,
donchian+atr, vwap+volume, macd+trend-filter).

### Бэктест: одна функция

```python
from src.backtesting import run_backtest, BacktestConfig

result = run_backtest("ema_rsi", data, params={...},
                      config=BacktestConfig(init_cash=100_000, fees=0.001))
result.stats(); result.equity(); result.drawdown(); result.trades()
```

### Массовое исследование (этап 02)

```python
from src.optimization import research_universe, summarize_strategies, top_configurations

combined = research_universe(names, ohlcv)      # IS/OOS по всем сеткам
summary = summarize_strategies(combined)        # рейтинг + флаги переобучения
top20 = top_configurations(combined, n=20)      # устойчивые конфигурации
```

Интегральный рейтинг — перцентильные ранги 11 метрик (CAGR, Return, Sharpe,
Sortino, Calmar, MaxDD, Profit Factor, Win Rate, Expectancy, Recovery
Factor, число сделок) со штрафом за малое число сделок; robust score
наказывает расхождение IS/OOS.

### Управление позицией (этап 03)

```python
from src.backtesting import ExitRules, managed_portfolio

rules = ExitRules(sl_stop=0.08, tp_stop=0.20, partial_tp=0.08, partial_fraction=0.5)
pf, reasons = managed_portfolio(ohlcv, sig.entries, sig.exits, rules)
```

Единый numba-симулятор: SL/TP/Trailing/BreakEven/TimeStop/сигнальные
выходы/частичная фиксация и их комбинации; интрабарные цены стопов
(гэп — по open), код причины каждого выхода. Раннер
`exit_research`/`summarize_mechanics` выносит вердикты real/false
improvement по протоколу IS→OOS.

### Признаки и фильтры (этап 04)

```python
from src.analytics import market_features, trade_feature_dataset
from src.optimization import filter_research, summarize_filters

features = market_features(ohlcv)               # 12 признаков без lookahead
trades = trade_feature_dataset(best_params, ohlcv, features)
fsweep = filter_research(best_params, ohlcv, features)
```

### Капитал: Kelly + Monte Carlo + Walk-Forward (этап 05)

```python
from src.risk import TradeStats, sizing_menu, kelly_robustness, simulate_trade_sequences
from src.optimization import walk_forward, walk_forward_summary, WalkForwardConfig

stats = TradeStats.from_returns(trade_returns)  # E, Var, p, avg_win/avg_loss
sizing_menu(stats, max_fraction=0.25, risk_per_trade=0.02)  # classical/half/quarter + capы
kelly_robustness(trade_returns, [0.1, 0.25, 0.5, 1.0, 2.0]) # MC: разорение, распределения
wf = walk_forward("ema_rsi", ohlcv, cfg=WalkForwardConfig())  # пере-оптимизация в окнах
```

## Главные результаты (на демонстрационных данных)

* **ТОП стратегий**: ema_rsi (гибрид), roc_momentum, ichimoku, triple_ema,
  ema_adx — устойчивые лидеры; supertrend силен на фиксированном сплите,
  но проваливает walk-forward (поучительный кейс).
* **Управление позицией**: реально работают Take Profit и частичная
  фиксация; Time Stop и Break Even — типичные ложные улучшения.
* **Фильтры**: согласованность старшего и младшего тренда
  (`htf_up + ema_slope_up`) и запрет на перерастянутые входы
  (`|dist_ma| < 10%`) — единственные устойчивые; главный резерв —
  не торговать тихий диапазон (`range_lowvol`).
* **Капитал**: полный Kelly на десятках сделок не идентифицирован
  (bootstrap-CI шире оценки); практическое правило —
  `min(quarter Kelly, 25% капитала, 2% риска на сделку)`.

## Внешний оркестратор автономного исследования

`orchestrator/run_research_loop.py` управляет исследовательским циклом
снаружи, не полагаясь на одну сессию Claude Code: строит промпт из шаблона
(`prompts/`) + базы знаний + дайджеста отклонённых идей (защита от
повторов), запускает headless `claude -p`, валидирует контракт артефактов
итерации (`research/pending/iteration_NNN/`: hypothesis.md,
implementation.py, metrics.csv, trades.csv, plots/, notebook.ipynb,
summary.json, decision.md), оценивает улучшение по composite score и
маршрутизирует итерацию в `research/accepted/` или `research/rejected/`,
ведёт `orchestrator/state.json` (переживает перезапуски) и `research_log.md`.

```bash
python orchestrator/run_research_loop.py \
    --max-iterations 100 --patience 15 --min-improvement 0.03 \
    --prompt-file prompts/research_hypothesis.md --results-dir research/

python orchestrator/run_research_loop.py --dry-run --max-iterations 10  # механика без модели
```

Остановка — по любому из критериев: лимит итераций; `patience` итераций без
улучшения; лучшая стратегия стабильна (OOS Sharpe / walk-forward / Monte
Carlo одновременно в норме N итераций); новые гипотезы повторяют уже
проверенные (Jaccard-схожесть идей). Итерация без метрик считается
провальной и повторяется со строгим промптом (`prompts/reject_strategy.md`);
после остановки запрашивается финальный отчёт (`prompts/final_report.md`).
Настройки — `orchestrator/config.yaml`, CLI-аргументы её переопределяют.

## Методологические гарантии

* Признаки и сигналы считаются только из прошлых баров (тест на lookahead);
* сигналы для OOS получают warm-up из истории, но метрики окон изолированы;
* каждая оптимизация (параметры, механики выходов, фильтры) выбирается на
  IS и судится на OOS против своего baseline;
* walk-forward пере-оптимизирует параметры в каждом окне;
* деградационные вердикты и флаги переобучения — часть API, а не ручной
  анализ.

## Дисклеймер

Репозиторий — исследовательский шаблон. Прогоны в ноутбуках выполнены на
синтетических данных и демонстрируют методологию, а не торговые
рекомендации. Прошлые результаты не гарантируют будущих; используйте на
свой риск.
