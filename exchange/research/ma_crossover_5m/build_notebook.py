"""Сборка и исполнение ноутбука 03_ma_crossover_5m.ipynb из файлов nb/c*.py."""
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).parent
OUT = HERE.parents[1] / "notebooks" / "03_ma_crossover_5m.ipynb"

MD = {
    "c00": """# Скользящие средние на 5-минутках NG и BR: полная сетка

Исследование стратегии «цена пересекает быструю среднюю, направление разрешает наклон
медленной». Полная сетка — 98 × 246 периодов × 36 сочетаний типов × 2 метода нейтрали ×
2 модели выхода = **3 471 552 конфигурации на инструмент**; отбор только по in-sample
(до 2024-01-01), проверка — на out-of-sample (2024-01 … 2026-10).

Цифры и выводы — в `docs/RESULTS_MA_CROSSOVER_5M.md`; здесь — девять обязательных
типов графиков и два дополнительных. Ноутбук читает уже посчитанные результаты из
рабочего каталога (`MA5M_WORK`); пересчёт — скрипты в `research/ma_crossover_5m/`.

Цвет: синий — прибыль/«да», красный — убыток/«нет», серый — ноль. Это та же
дивергентная пара, что в ноутбуках 01–02; зелёный против красного не используется,
потому что именно эту пару не различают при дейтеранопии. На картах числа и подписи
дублируют цвет.""",
    "c01": "## 1. Тепловые карты Fast × Slow",
    "c02": "## 2. Все 36 сочетаний типов средних",
    "c03": "## 3. Модели выхода",
    "c04": "## 4. Методы нейтральной зоны",
    "c05": "## 5. Карты устойчивости параметров",
    "c06": "## 6. Кривые накопленного PnL",
    "c07": "## 7. Распределения результатов сделок",
    "c08": "## 8. Сравнение инструментов",
    "c09": "## 9. In-sample против out-of-sample",
    "c10": "## Дополнительно: walk-forward и PBO",
}

cells = []
for f in sorted((HERE / "nb").glob("c*.py")):
    key = f.stem.split("_")[0]
    if key in MD:
        cells.append(nbformat.v4.new_markdown_cell(MD[key]))
    cells.append(nbformat.v4.new_code_cell(f.read_text().rstrip() + "\n"))

nb = nbformat.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python (ma5m, vectorbt)", "language": "python", "name": "ma5m"}})
NotebookClient(nb, timeout=3600, kernel_name="ma5m",
               resources={"metadata": {"path": str(OUT.parent)}}).execute()
OUT.write_text(nbformat.writes(nb))
errs = [c for c in nb.cells if any(o.get("output_type") == "error" for o in c.get("outputs", []))]
print("записан", OUT, "| ячеек", len(nb.cells), "| с ошибкой", len(errs))
for c in errs:
    for o in c["outputs"]:
        if o.get("output_type") == "error":
            print("\n".join(o["traceback"][-4:]))
sys.exit(1 if errs else 0)
