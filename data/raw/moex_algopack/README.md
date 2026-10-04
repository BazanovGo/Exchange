# MOEX AlgoPack: NG и BR

Минутные, пятиминутные и часовые свечи фьючерсов NG и BR, полная доступная история.
Источник — `apim.moex.com`, загрузчик `market_data/sources/moex_algopack/loader.py` из репозитория trader.

```
<тикер>/<таймфрейм>/<год>.csv   # ng: 2020–2026, br: 2011–2026; таймфреймы 1m, 5m, 60m
manifest.json                   # состав файлов и число строк
```

Колонки: `ticker,secid,begin,end,open,high,low,close,volume,value`.
Время — московское (UTC+3), как отдаёт MOEX. 5m собраны из 1m; 60m — исходные часовые свечи.

Ряд склеен из контрактов по ближайшей экспирации (переход на следующий день после последнего дня
обращения). Цены не корректируются, на переходах возможны скачки — смотрите `secid`.
`value` для фьючерсов MOEX отдаёт нулевым.

Загрузка непрерывного ряда:

```python
import pandas as pd
from pathlib import Path
files = sorted(Path("data/raw/moex_algopack/ng/1m").glob("*.csv"))
df = pd.concat(pd.read_csv(f, parse_dates=["begin", "end"]) for f in files).set_index("begin")
```
