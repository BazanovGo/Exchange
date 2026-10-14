"""Графики исследования. Палитра и правила — из системы, проверенной в
ноутбуках 01/02: синий↔красный — полярность (с серой нейтральной серединой
на дивергентных картах), категориальные слоты 1–2 — идентичность серий,
одна ось на график, тонкие линии, легенда при двух и более сериях.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
POS, NEG = "#2a78d6", "#e34948"
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
NEUTRAL = "#f0efec"
# Дивергентная: красный (убыток) — серый (ноль) — синий (прибыль), равные плечи.
DIVERGING = LinearSegmentedColormap.from_list(
    "rb", ["#a52a2a", "#e34948", "#f2a9a8", NEUTRAL, "#9ec5f4", "#2a78d6", "#104281"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#f0efec", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10, "axes.linewidth": 0.8, "axes.edgecolor": AXIS,
    "lines.linewidth": 2.0, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.frameon": False, "legend.fontsize": 9,
    "figure.dpi": 120,
})


def frame(ax, title="", subtitle="", xlabel="", ylabel="", grid_axis="y"):
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=16 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9.5)
    ax.tick_params(length=3, color=AXIS)


def sym_norm(values, q=0.99):
    v = np.abs(np.asarray(values, float))
    v = v[np.isfinite(v)]
    lim = float(np.quantile(v, q)) if v.size else 1.0
    lim = lim or 1.0
    return TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)


def heatmap(ax, mat, x_vals, y_vals, *, norm, title="", subtitle="", xlabel="", ylabel="",
            cmap=DIVERGING, mark=None):
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap, norm=norm, interpolation="nearest",
                   extent=(x_vals[0] - 0.5, x_vals[-1] + 0.5, y_vals[0] - 0.5, y_vals[-1] + 0.5))
    frame(ax, title, subtitle, xlabel, ylabel, grid_axis=None)
    if mark is not None:
        # двойное кольцо: видно и на тёмно-синем, и на светлом
        ax.plot([mark[0]], [mark[1]], marker="o", markersize=12, markerfacecolor="none",
                markeredgecolor=INK, markeredgewidth=3.2)
        ax.plot([mark[0]], [mark[1]], marker="o", markersize=12, markerfacecolor="none",
                markeredgecolor="#ffffff", markeredgewidth=1.4)
    return im


def matrix6(ax, mat, labels_x, labels_y, *, norm, title="", subtitle="", fmt="{:+.2f}", cmap=DIVERGING):
    im = ax.imshow(mat, cmap=cmap, norm=norm)
    frame(ax, title, subtitle, grid_axis=None)
    ax.set_xticks(range(len(labels_x))); ax.set_xticklabels(labels_x)
    ax.set_yticks(range(len(labels_y))); ax.set_yticklabels(labels_y)
    ax.set_xlabel("тип медленной средней", color=INK_2, fontsize=9.5)
    ax.set_ylabel("тип быстрой средней", color=INK_2, fontsize=9.5)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isfinite(v):
                r, g, b, _ = cmap(norm(v))
                lum = 0.2126 * r + 0.7152 * g + 0.0722 * b      # тёмная клетка — белый текст
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if lum < 0.45 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def step_hist(ax, series: dict, bins, colors=(S1, S2, S3)):
    for (label, x), c in zip(series.items(), colors):
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        ax.hist(x, bins=bins, histtype="step", linewidth=1.8, color=c, label=f"{label} (медиана {np.median(x):+.2f})",
                density=True)
    ax.axvline(0, color=AXIS, linewidth=0.8)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 1.3)          # место под легенду, чтобы она не лежала на пиках
    ax.legend(loc="upper left")
