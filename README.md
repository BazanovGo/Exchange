# Exchange — Trading Strategy Research Framework

Промышленный шаблон репозитория для исследования торговых стратегий на базе
[vectorbt](https://vectorbt.dev). Инфраструктура отделена от исследований:
данные, стратегии, бэктест, аналитика, риск и визуализация — независимые
слои с едиными интерфейсами и реестрами для расширения.

## Структура

```
├── data/                  # рыночные данные (raw/ и processed/, в git — только sample)
├── notebooks/             # исследовательские ноутбуки
│   └── 01_repository_setup.ipynb   # проверка инфраструктуры end-to-end
├── reports/               # сохраненные отчеты
├── results/               # артефакты бэктестов (метрики, сделки, кривые)
├── scripts/               # служебные скрипты (запуск ноутбуков и т.п.)
├── src/
│   ├── data/              # единый интерфейс загрузки данных + реестр источников
│   ├── indicators/        # индикаторы (обертки над vectorbt)
│   ├── strategies/        # единый интерфейс стратегий + реестр
│   ├── backtesting/       # запуск бэктеста одной функцией
│   ├── analytics/         # стандартный отчет: метрики, equity, drawdown, сделки
│   ├── optimization/      # grid search по параметрам любой стратегии
│   ├── risk/              # риск-метрики и position sizing
│   ├── visualization/     # графики (plotly), GitHub-совместимый PNG-рендеринг
│   └── utils/             # пути проекта, логирование
└── tests/                 # pytest-тесты инфраструктуры
```

## Установка

Требуется Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
pytest                      # проверка инфраструктуры
```

## Ключевые интерфейсы

### Данные: `load_data(source, symbols, ...)`

```python
from src.data import load_data

data = load_data("yahoo", "AAPL", start="2020-01-01")        # Yahoo Finance
data = load_data("csv", "SAMPLE")                             # data/raw/SAMPLE.csv
data = load_data("parquet", "BTCUSD", start="2023-01-01")     # data/processed/BTCUSD.parquet

data.get("AAPL")   # канонический OHLCV DataFrame (Open/High/Low/Close/Volume)
data.close         # wide-фрейм Close по всем символам
```

Все источники возвращают одинаковую схему: `DatetimeIndex` + колонки
`Open, High, Low, Close, Volume` (нормализация регистра колонок, дублей и
сортировки — автоматическая).

**Добавление нового источника (MOEX, Polygon, ...):**

```python
from src.data import BaseDataLoader, register_loader

@register_loader
class MoexLoader(BaseDataLoader):
    source_name = "moex"

    def _load_symbol(self, symbol, *, start=None, end=None, timeframe="1d", **kwargs):
        ...  # вернуть сырой DataFrame — нормализация и слайсинг уже в базовом классе
```

После этого `load_data("moex", "SBER")` работает во всем проекте.

### Стратегии: единый контракт `StrategySignals`

Каждая стратегия возвращает стандартный объект:

| Поле | Описание |
|---|---|
| `entries` | входы в лонг (bool) |
| `exits` | выходы из лонга (bool) |
| `short_entries` | входы в шорт (bool, all-False у long-only) |
| `short_exits` | выходы из шорта (bool) |
| `params` | фактические параметры запуска |
| `description` | описание логики |

```python
from src.strategies import get_strategy

strategy = get_strategy("ma_crossover", fast_window=20, slow_window=60)
signals = strategy.signals(data)
```

**Новая стратегия** — подкласс `BaseStrategy` с декоратором
`@register_strategy`: реализуется только `generate_signals(ohlcv)`,
валидация параметров и заполнение short-легов делаются базовым классом.
Встроенные примеры: `ma_crossover`, `rsi_reversion`.

### Бэктест: одна функция для любой стратегии

```python
from src.backtesting import run_backtest, BacktestConfig

config = BacktestConfig(init_cash=100_000, fees=0.001, slippage=0.0005)
result = run_backtest("ma_crossover", data,
                      params={"fast_window": 20, "slow_window": 60},
                      config=config)

result.stats()      # полная таблица vectorbt
result.equity()     # кривая капитала
result.drawdown()   # просадка
result.trades()     # список сделок
```

### Анализ результатов

```python
from src.analytics import PerformanceReport

report = PerformanceReport.from_result(result)
report.summary()            # единый набор метрик (Sharpe, MaxDD, WinRate, ...)
report.save("my_run")       # metrics/trades/curves/params -> results/my_run/
```

### Оптимизация

```python
from src.optimization import grid_search

table = grid_search("ema_cross", data,
                    {"fast_window": [10, 20, 50], "slow_window": [50, 100, 200]},
                    sort_by="sharpe_ratio")
```

Невалидные комбинации (например, `fast >= slow`) пропускаются автоматически.

### Массовое исследование стратегий

Вселенная из **26 стратегий** четырёх классов (трендовые, контртрендовые,
импульсные, гибридные), каждая с собственной сеткой оптимизации (`opt_grid`).
Раннер перебирает все сетки с in-sample/out-of-sample разбиением, строит
интегральный рейтинг по 11 метрикам (CAGR, Return, Sharpe, Sortino, Calmar,
Max Drawdown, Profit Factor, Win Rate, Expectancy, Recovery Factor, число
сделок) и диагностирует переобучение:

```python
from src.optimization import ResearchConfig, research_universe, summarize_strategies, top_configurations
from src.strategies import available_strategies

names = [n for n in available_strategies() if n != "ma_crossover"]
combined = research_universe(names, ohlcv, config=ResearchConfig(split=0.7))
summary = summarize_strategies(combined)          # рейтинг + флаги переобучения
top20 = top_configurations(combined, n=20)        # устойчивые конфигурации
```

Ключевые принципы: композитный скор считается по перцентильным рангам в общем
пуле (никогда — по одной доходности), малое число сделок штрафуется, robust
score наказывает расхождение IS/OOS. Флаги переобучения: деградация Sharpe
IS-победителя, ранговая корреляция Спирмена IS↔OOS по сетке, OOS-перцентиль
IS-победителя. Полный разбор — в `notebooks/02_strategy_research.ipynb`.

## Ноутбуки и GitHub

GitHub рендерит ноутбуки статически — интерактивные plotly-графики (включая
все графики vectorbt) отображаются пустыми ячейками. Поэтому:

1. в начале ноутбука вызывается `setup_github_rendering()`;
2. любые фигуры выводятся через `show(fig)` из `src.visualization` — они
   встраиваются статичными PNG (kaleido);
3. запускать ноутбуки удобно скриптом, который дополнительно вычищает
   тяжелое widget-состояние vectorbt:

```bash
python scripts/run_notebook.py notebooks/01_repository_setup.ipynb
```

Ноутбук `01_repository_setup.ipynb` проверяет инфраструктуру end-to-end:
загрузка данных → график цены → индикатор → бэктест → статистика → графики
vectorbt. Если Yahoo Finance недоступен (офлайн-среда), он прозрачно
переключается на `data/raw/SAMPLE.csv` через тот же интерфейс загрузки.

## Тесты

```bash
pytest              # 29 тестов: загрузчики, контракт стратегий, бэктест, аналитика, риск
```

Тесты не требуют сети: используется детерминированный синтетический генератор
OHLCV (`src.data.sample.generate_ohlcv`).
