"""Оформление презентационных дашбордов: тёмная тема, карточки, таблицы, шкалы.

Цвета — проверенные валидатором палитры для тёмной поверхности: категориальные
слоты синий / оранжевый / бирюзовый (все пары различимы при дальтонизме),
дивергентная шкала красный ↔ синий с нейтральной серой серединой: ноль
«исчезает» в фоне, сильные значения в обе стороны светятся. Радуги нет — она
искажает величины. Статус вердикта всегда дублирован знаком и словом.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
from matplotlib.patches import FancyBboxPatch

SURFACE = "#0b1220"
CARD, CARD_EDGE = "#101a2c", "#1f2c46"
INK, INK2, MUTED, GRID = "#eef3fb", "#b4c2d8", "#7d8fab", "#22314d"
S1, S2, S3 = "#3987e5", "#d95926", "#199e70"         # синий, оранжевый, бирюзовый (тёмные шаги)
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"   # статус: только со знаком и словом
MID = "#2c3448"

# Убыток — красный, ноль — серый, прибыль — синий; яркость растёт с модулем.
DIVERGING = LinearSegmentedColormap.from_list("loss_gain", [
    "#ffb4a8", "#f26b5b", "#c43c3c", "#7a2a33", MID, "#1c4f95", "#2a78d6", "#4d9bf0", "#a9d2ff"])

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.facecolor": CARD,
    "font.family": "DejaVu Sans", "font.size": 11, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.frameon": False, "legend.fontsize": 10, "lines.linewidth": 2.2,
    "lines.solid_capstyle": "round",
})


def sym_norm(values, q=0.98, floor=1e-9) -> TwoSlopeNorm:
    v = np.abs(np.asarray(values, float))
    v = v[np.isfinite(v)]
    lim = max(float(np.quantile(v, q)) if v.size else 1.0, floor)
    # Предел — «круглое» число вверх: деления шкалы тогда не выходят за её границы.
    step = 10 ** (np.floor(np.log10(lim)) - 1)
    lim = float(np.ceil(lim / step - 1e-9) * step)
    return TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)


def ink_on(rgba) -> str:
    """Цвет подписи поверх заливки: светлый на тёмной, тёмный на светлой."""
    r, g, b = to_rgb(rgba)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return SURFACE if lum > 0.55 else INK


class Board:
    """Фигура-дашборд: карточки задаются в долях фигуры (x, y, w, h), y — от низа."""

    def __init__(self, width=24.0, height=15.0):
        self.fig = plt.figure(figsize=(width, height))
        self.w, self.h = width, height

    def header(self, title: str, subtitle: str):
        self.fig.text(0.5, 0.975, title, ha="center", va="top", fontsize=30, fontweight="bold", color=INK)
        self.fig.text(0.5, 0.935, subtitle, ha="center", va="top", fontsize=14.5, color=INK2)

    def footer(self, text: str):
        self.fig.text(0.012, 0.006, text, ha="left", va="bottom", fontsize=9.5, color=MUTED)

    def card(self, x, y, w, h, title="", subtitle="", pad=(0.012, 0.012), title_h=None):
        """Рамка карточки и заголовок; возвращает область содержимого (x, y, w, h)."""
        self.fig.patches.append(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.006", transform=self.fig.transFigure,
            facecolor=CARD, edgecolor=CARD_EDGE, linewidth=1.2, zorder=-10))
        top = y + h - 0.012 * 15 / self.h
        if title:
            self.fig.text(x + pad[0], top, title, ha="left", va="top", fontsize=15.5,
                          fontweight="bold", color=INK)
        if subtitle:
            self.fig.text(x + pad[0], top - 0.024 * 15 / self.h, subtitle, ha="left", va="top",
                          fontsize=11, color=INK2)
        used = title_h if title_h is not None else (0.062 if subtitle else 0.038) * 15 / self.h
        return (x + pad[0], y + pad[1], w - 2 * pad[0], h - used - pad[1])

    def axes(self, rect, **kw):
        ax = self.fig.add_axes(rect, **kw)
        ax.set_facecolor(CARD)
        return ax

    def text_axes(self, rect):
        ax = self.fig.add_axes(rect)
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return ax

    def save(self, path, dpi=110):
        self.fig.savefig(path, dpi=dpi)
        plt.close(self.fig)


def frame(ax, xlabel="", ylabel="", grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if grid:
        ax.grid(True, axis=grid, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=11)
    ax.tick_params(length=3, color=GRID)


def table(ax, header, rows, col_x, *, row_h=None, top=0.96, colors=None, align=None, size=11.5,
          head_size=10.5, bold_cols=()):
    """Таблица на осях без рамок: заголовок, строки-полосы, цветные ячейки по запросу.

    colors[i][j] — цвет заливки ячейки или None; текст поверх подбирается по яркости.
    """
    n = len(rows) + 1
    row_h = row_h or min(0.135, (top - 0.02) / n)
    align = align or ["left"] + ["right"] * (len(header) - 1)
    edges = [*col_x, 1.0]
    y = top
    ax.add_patch(plt.Rectangle((0, y - row_h), 1, row_h, color="#17233a", zorder=0))
    for j, h in enumerate(header):
        xx = edges[j] + 0.008 if align[j] == "left" else edges[j + 1] - 0.008
        ax.text(xx, y - row_h / 2, h, ha=align[j], va="center", fontsize=head_size, color=INK2,
                fontweight="bold")
    for i, row in enumerate(rows):
        y -= row_h
        if i % 2 == 1:
            ax.add_patch(plt.Rectangle((0, y - row_h), 1, row_h, color="#131f34", zorder=0))
        for j, v in enumerate(row):
            fill = colors[i][j] if colors and colors[i][j] is not None else None
            if fill is not None:
                ax.add_patch(FancyBboxPatch((edges[j] + 0.004, y - row_h + 0.012), edges[j + 1] - edges[j] - 0.008,
                                            row_h - 0.024, boxstyle="round,pad=0,rounding_size=0.01",
                                            facecolor=fill, edgecolor="none", zorder=1))
            xx = edges[j] + 0.012 if align[j] == "left" else edges[j + 1] - 0.012
            ax.text(xx, y - row_h / 2, v, ha=align[j], va="center", fontsize=size,
                    color=ink_on(fill) if fill is not None else INK,
                    fontweight="bold" if j in bold_cols else "normal", zorder=2)
    return y - row_h


def steps(ax, items, top=0.95, gap=None, color=S1):
    """Нумерованные шаги методики: кружок с номером и текст."""
    gap = gap or (top - 0.03) / len(items)
    for k, text in enumerate(items, start=1):
        yy = top - (k - 0.5) * gap
        ax.plot([0.035], [yy], "o", markersize=24, color=color, zorder=2, clip_on=False)
        ax.text(0.035, yy, str(k), ha="center", va="center", fontsize=12, fontweight="bold", color=INK, zorder=3)
        ax.text(0.09, yy, text, ha="left", va="center", fontsize=11.5, color=INK, linespacing=1.35)


def badge(ax, x, y, text, color, size=15):
    ax.text(x, y, text, ha="left", va="center", fontsize=size, fontweight="bold", color=SURFACE,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": color, "edgecolor": "none"})


def colorbar(fig, rect, cmap, norm, label, ticks=None, fmt="{:+.1f}"):
    cax = fig.add_axes(rect)
    cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="vertical")
    cb.outline.set_edgecolor(GRID)
    cax.tick_params(colors=INK2, labelsize=9.5, length=2)
    if ticks is not None:
        span = norm.vmax - norm.vmin
        ticks = [t for t in ticks if norm.vmin - 1e-9 * span <= t <= norm.vmax + 1e-9 * span]
        cb.set_ticks(ticks)
        cb.set_ticklabels([fmt.format(t) for t in ticks])
    cax.set_title(label, color=INK2, fontsize=10, pad=8, loc="left")
    return cb
