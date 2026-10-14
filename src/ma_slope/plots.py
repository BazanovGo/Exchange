"""Static figures for the report (matplotlib PNG, light surface).

Palette and chrome follow the project data-viz reference: categorical slots in
fixed order, a blue-gray-red diverging scale for signed t-statistics, one hue
for magnitudes, hairline solid grids, no dual axes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
POS, NEG = "#2a78d6", "#e34948"
DIVERGING = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#8f2424", "#e34948", "#f0efec", "#2a78d6", "#104281"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
DIVERGING.set_bad(SURFACE)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": BASE, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False, "lines.linewidth": 1.6,
    "font.family": "DejaVu Sans", "figure.dpi": 110, "savefig.dpi": 130, "savefig.bbox": "tight",
})


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _norm(values, limit=None):
    v = np.asarray(values, float)
    lim = limit or max(float(np.nanpercentile(np.abs(v), 99)) if np.isfinite(v).any() else 1.0, 0.5)
    return TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)


def period_heatmaps(panels: list[tuple[str, np.ndarray]], fast_periods, slow_periods, mark, title, path):
    """Fast x slow maps of signed statistics; ``mark`` = (fast, slow) of the selected config."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3 * len(panels), 4.2), sharey=True)
    for ax, (name, z) in zip(np.atleast_1d(axes), panels, strict=True):
        share = name.startswith("share")
        im = ax.pcolormesh(slow_periods, fast_periods, z, shading="nearest", rasterized=True,
                           cmap=SEQUENTIAL if share else DIVERGING,
                           norm=None if share else _norm(z), vmin=0 if share else None, vmax=1 if share else None)
        ax.plot(mark[1], mark[0], marker="o", ms=9, mfc="none", mec=INK, mew=1.5)
        ax.plot(slow_periods, slow_periods, color=INK2, lw=0.8)  # fast = slow diagonal
        ax.set_xlim(min(slow_periods), max(slow_periods))
        ax.set_ylim(min(fast_periods), max(fast_periods))
        ax.set_title(name)
        ax.set_xlabel("Slow MA period")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    np.atleast_1d(axes)[0].set_ylabel("Fast MA period")
    fig.suptitle(title, fontsize=10, y=1.02, color=INK2)
    return _save(fig, path)


def matrix_grid(mats: dict[str, np.ndarray], rows, cols, title, path, ncols=4, fmt="{:.1f}", cbar_label="t"):
    """Small multiples of labelled matrices (e.g. Fast type x Slow type) on one diverging scale."""
    names = list(mats)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.1 * nrows), squeeze=False)
    norm = _norm(np.concatenate([np.ravel(m) for m in mats.values()]))
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    for ax, name in zip(axes.ravel(), names, strict=False):
        m = mats[name]
        im = ax.imshow(m, cmap=DIVERGING, norm=norm, aspect="auto")
        for (i, j), v in np.ndenumerate(m):
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.5,
                        color=INK if abs(norm(v) - 0.5) < 0.3 else SURFACE)
        ax.set_xticks(range(len(cols)), cols, rotation=90)
        ax.set_yticks(range(len(rows)), rows)
        ax.set_title(name)
        ax.grid(False)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label=cbar_label)
    fig.suptitle(title, fontsize=10, color=INK2)
    return _save(fig, path)


def ranges(categories, series: dict[str, tuple], title, ylabel, path):
    """Median dot with p10-p90 bar per category for up to 3 series (slots in fixed order)."""
    fig, ax = plt.subplots(figsize=(1.0 * len(categories) + 2.5, 3.6))
    width = 0.8 / len(series)
    for k, (name, (lo, mid, hi)) in enumerate(series.items()):
        x = np.arange(len(categories)) - 0.4 + width * (k + 0.5)
        ax.vlines(x, lo, hi, color=SERIES[k], lw=2.5, alpha=0.55)
        ax.plot(x, mid, "o", color=SERIES[k], ms=6, label=name, mec=SURFACE, mew=1)
    ax.axhline(0, color=BASE, lw=1)
    ax.set_xticks(range(len(categories)), categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", ncols=len(series))
    return _save(fig, path)


def small_multiples_lines(curves: dict[str, list[tuple[str, object, object]]], title, ylabel, path,
                          vlines: dict | None = None, ncols=4):
    """One panel per key; each panel holds named (x, y) lines in fixed slot order."""
    names = list(curves)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.6 * nrows), squeeze=False)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    for ax, name in zip(axes.ravel(), names, strict=False):
        for k, (label, x, y) in enumerate(curves[name]):
            ax.plot(x, y, color=SERIES[k], lw=1.3, label=label)
        if vlines and name in vlines:
            ax.axvline(vlines[name], color=INK2, lw=1)
        ax.axhline(0, color=BASE, lw=0.8)
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=30)
    axes.ravel()[0].set_ylabel(ylabel)
    if any(len(v) > 1 for v in curves.values()):
        axes.ravel()[0].legend(loc="upper left")
    fig.suptitle(title, fontsize=10, color=INK2)
    fig.tight_layout()
    return _save(fig, path)


def small_multiples_hist(data: dict[str, dict[str, np.ndarray]], title, xlabel, path, ncols=4, bins=60):
    names = list(data)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.6 * nrows), squeeze=False)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    for ax, name in zip(axes.ravel(), names, strict=False):
        allv = np.concatenate([v for v in data[name].values() if len(v)]) if data[name] else np.array([0.0])
        lo, hi = np.percentile(allv, [1, 99]) if allv.size else (0, 1)
        edges = np.linspace(lo, hi, bins)
        for k, (label, v) in enumerate(data[name].items()):
            ax.hist(np.clip(v, lo, hi), bins=edges, histtype="step", density=True, color=SERIES[k], lw=1.4,
                    label=f"{label} (n={len(v)})")
        ax.axvline(0, color=BASE, lw=0.8)
        ax.set_title(name)
        ax.legend(loc="upper right")
    axes.ravel()[0].set_xlabel(xlabel)
    fig.suptitle(title, fontsize=10, color=INK2)
    fig.tight_layout()
    return _save(fig, path)


def small_multiples_hexbin(data: dict[str, tuple[np.ndarray, np.ndarray, str]], title, xlabel, ylabel, path,
                           ncols=4):
    names = list(data)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.1 * nrows), squeeze=False)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    for ax, name in zip(axes.ravel(), names, strict=False):
        x, y, note = data[name]
        ax.hexbin(x, y, gridsize=45, cmap=SEQUENTIAL, mincnt=1, bins="log", linewidths=0)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(f"{name}  {note}")
    axes.ravel()[0].set_xlabel(xlabel)
    axes.ravel()[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=10, color=INK2)
    fig.tight_layout()
    return _save(fig, path)


def signed_bars(groups: dict[str, tuple[list, list]], title, ylabel, path, ncols=4):
    """Per panel: bars coloured by sign (blue gain, red loss) on a shared zero line."""
    names = list(groups)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.6 * nrows), squeeze=False)
    for ax in axes.ravel()[len(names):]:
        ax.set_visible(False)
    for ax, name in zip(axes.ravel(), names, strict=False):
        x, y = groups[name]
        y = np.asarray(y, float)
        ax.bar([str(v) for v in x], y, color=np.where(y >= 0, POS, NEG), width=0.7)
        ax.axhline(0, color=BASE, lw=0.8)
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=60)
    axes.ravel()[0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=10, color=INK2)
    fig.tight_layout()
    return _save(fig, path)


def grouped_bars(categories, series: dict[str, list], title, ylabel, path):
    """Two or three series side by side per category, slots in fixed order."""
    fig, ax = plt.subplots(figsize=(1.0 * len(categories) + 2.5, 3.4))
    width = 0.8 / len(series)
    for k, (name, vals) in enumerate(series.items()):
        x = np.arange(len(categories)) - 0.4 + width * (k + 0.5)
        ax.bar(x, vals, width=width * 0.9, color=SERIES[k], label=name)
    ax.axhline(0, color=BASE, lw=1)
    ax.set_xticks(range(len(categories)), categories)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", ncols=len(series))
    return _save(fig, path)
