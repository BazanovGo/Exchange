"""Графики ноутбука отдельными PNG: docs/figures/ma_crossover_5m/.

Ноутбук — единственный источник графиков; здесь они только извлекаются,
чтобы отчёт мог на них ссылаться и они открывались на GitHub без Jupyter.

python export_figures.py   (после build_notebook.py)
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

HERE = Path(__file__).parent
NOTEBOOK = HERE.parents[1] / "notebooks" / "03_ma_crossover_5m.ipynb"
OUT = HERE.parents[1] / "docs" / "figures" / "ma_crossover_5m"

# Порядок изображений в ноутбуке: (имя файла, подпись). Ячейки с циклом по
# инструментам выводят сначала NG, затем BR.
FIGURES = [
    ("01_heatmaps_NG.png", "Тепловые карты Fast × Slow, природный газ"),
    ("01_heatmaps_BR.png", "Тепловые карты Fast × Slow, Brent"),
    ("02_type_pairs_NG.png", "36 сочетаний типов средних, природный газ"),
    ("02_type_pairs_BR.png", "36 сочетаний типов средних, Brent"),
    ("03_exits.png", "Модели выхода A и B"),
    ("04_slope_methods.png", "Методы нейтральной зоны A (процент) и B (ATR)"),
    ("04b_slope_period_k.png", "Период определения наклона k"),
    ("05_stability_NG.png", "Карты устойчивости параметров, природный газ"),
    ("05_stability_BR.png", "Карты устойчивости параметров, Brent"),
    ("06_equity_NG.png", "Накопленный PnL отобранных стратегий, природный газ"),
    ("06_equity_BR.png", "Накопленный PnL отобранных стратегий, Brent"),
    ("07_trades_NG.png", "Распределение результатов сделок, природный газ"),
    ("07_trades_BR.png", "Распределение результатов сделок, Brent"),
    ("08_cross_market.png", "Одна и та же конфигурация на NG и BR"),
    ("09_is_vs_oos.png", "In-sample против out-of-sample на всей сетке"),
    ("10_walkforward_pbo.png", "Walk-forward с переоптимизацией и PBO"),
]


def images(nb_path: Path):
    nb = json.loads(nb_path.read_text())
    for cell in nb["cells"]:
        for o in cell.get("outputs", []):
            png = o.get("data", {}).get("image/png")
            if png:
                yield base64.b64decode("".join(png) if isinstance(png, list) else png)


def main():
    imgs = list(images(NOTEBOOK))
    if len(imgs) != len(FIGURES):
        raise SystemExit(f"в ноутбуке {len(imgs)} изображений, ожидалось {len(FIGURES)}: "
                         "обновите FIGURES вместе с nb/")
    OUT.mkdir(parents=True, exist_ok=True)
    for f in OUT.glob("*.png"):
        f.unlink()
    for (name, _), data in zip(FIGURES, imgs, strict=True):
        (OUT / name).write_bytes(data)
    total = sum(f.stat().st_size for f in OUT.glob("*.png"))
    print(f"{len(imgs)} графиков -> {OUT} ({total / 1e6:.1f} МБ)")


if __name__ == "__main__":
    main()
