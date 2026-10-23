# --- Палитра --------------------------------------------------------------
# Дивергентная пара синий<->красный несёт полярность: long против short,
# прибыльная сделка против убыточной. Нейтральный серый — середина (вне рынка).
# Зелёный/красный — отраслевой стандарт и худший выбор: это ровно та пара,
# которую не различает 8% мужчин. Форма маркера (треугольник вверх/вниз)
# дублирует цвет, поэтому направление сделки читается и без цвета.
LONG, SHORT = "#2a78d6", "#e34948"
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"
NEUTRAL = "#f0efec"
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
MUTED, GRID, AXIS = "#898781", "#e1e0d9", "#c3c2b7"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.linewidth": 0.8, "axes.edgecolor": AXIS,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 9,
    "figure.dpi": 130,
})


def frame(ax, title="", subtitle="", ylabel="", grid_axis="y"):
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    if title:
        ax.set_title(title, color=INK, fontsize=12.5, fontweight="semibold", loc="left", pad=16)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9.5, va="bottom")
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9.5)
    ax.tick_params(length=3, color=AXIS)


def date_ticks(ax, index, n=7, x0=0):
    """Даты на порядковой оси: бары равной ширины, без дыр на выходных.

    x0 — смещение окна: данные нарисованы в координатах исходного ряда,
    поэтому и подписи должны стоять там же, иначе они уезжают за пределы оси.
    """
    pos = np.linspace(0, len(index) - 1, n).round().astype(int)
    ax.set_xticks(pos + x0)
    ax.set_xticklabels([index[p].strftime("%m.%y") for p in pos])


def below_legend(ax, ncol=4):
    """Легенда под осью: над данными она закрывает именно то, что показывает."""
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.13), ncol=ncol,
              handletextpad=0.5, columnspacing=1.6, borderaxespad=0)
