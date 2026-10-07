def shade_position(ax, pos, x0):
    """Фон: синий — в длинной позиции, красный — в короткой, пусто — вне рынка."""
    sign = np.sign(pos.to_numpy(float))
    x = np.arange(len(sign)) + x0
    lo, hi = ax.get_ylim()
    for value, colour in ((1.0, LONG), (-1.0, SHORT)):
        ax.fill_between(x, lo, hi, where=(sign == value), color=colour, alpha=0.07,
                        linewidth=0, zorder=0)
    ax.set_ylim(lo, hi)


def mark_trades(ax, trades, index, x0, size=9):
    """Входы — треугольник по направлению, выходы — кольцо.

    Форма маркера дублирует цвет: направление сделки читается и в ч/б.
    """
    loc = {t: i for i, t in enumerate(index)}
    shown = {"long": False, "short": False, "exit": False}
    for _, t in trades.iterrows():
        i, j = loc.get(t["entry_time"]), loc.get(t["exit_time"])
        if i is None:
            continue
        long_ = t["direction"] > 0
        ax.plot([i + x0], [t["entry_price"]], marker="^" if long_ else "v",
                markersize=size, color=LONG if long_ else SHORT, zorder=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5,
                label=("вход long" if long_ else "вход short")
                if not shown["long" if long_ else "short"] else None)
        shown["long" if long_ else "short"] = True
        if j is not None:
            ax.plot([j + x0], [t["exit_price"]], marker="o", markersize=size - 3,
                    markerfacecolor=SURFACE, markeredgecolor=MUTED, markeredgewidth=1.6,
                    zorder=5, label="выход" if not shown["exit"] else None)
            shown["exit"] = True


def trade_pnl(ax, trades, ppy_label=""):
    """Результат каждой сделки: стебель от нуля, цвет и знак совпадают."""
    if not len(trades):
        ax.text(0.5, 0.5, "сделок нет", transform=ax.transAxes, ha="center", color=MUTED)
        return
    pnl = trades["net_pnl"].to_numpy(float) / H.CAPITAL * 100.0
    x = np.arange(len(pnl))
    for mask, colour in ((pnl >= 0, LONG), (pnl < 0, SHORT)):
        if mask.any():
            ax.vlines(x[mask], 0, pnl[mask], color=colour, linewidth=2.0)
            ax.plot(x[mask], pnl[mask], marker="o", markersize=4, linestyle="none",
                    color=colour, markeredgecolor=SURFACE, markeredgewidth=1.0)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    best, worst = int(np.argmax(pnl)), int(np.argmin(pnl))
    for i in (best, worst):
        ax.annotate(f"{pnl[i]:+.1f}%", xy=(i, pnl[i]), xytext=(0, 6 if pnl[i] > 0 else -14),
                    textcoords="offset points", ha="center", color=INK_2, fontsize=9)
    ax.set_xlim(-1, len(pnl))
    span = float(np.nanmax(pnl) - np.nanmin(pnl)) or 1.0
    ax.set_ylim(float(np.nanmin(pnl)) - 0.18 * span, float(np.nanmax(pnl)) + 0.18 * span)
    ax.set_xlabel("сделки в порядке закрытия", color=MUTED, fontsize=9)


def equity_panel(ax, leg, title, subtitle):
    eq = leg["equity"] / H.CAPITAL * 100.0 - 100.0
    x = np.arange(len(eq))
    frame(ax, title, subtitle, ylabel="доходность, %")
    ax.plot(x, eq.to_numpy(float), color=SERIES_1, linewidth=2.0)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    cut = eq.index.get_indexer([leg["is_end"]], method="nearest")[0]
    ax.axvline(cut, color=MUTED, linewidth=1.0, linestyle="-")
    ax.annotate("конец in-sample", xy=(cut, ax.get_ylim()[0]), xytext=(6, 8),
                textcoords="offset points", color=MUTED, fontsize=9, va="bottom",
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax.annotate(f"{eq.iloc[-1]:+.1f}%", xy=(len(eq) - 1, eq.iloc[-1]), xytext=(8, 0),
                textcoords="offset points", va="center", color=INK_2,
                fontsize=9.5, fontweight="semibold")
    date_ticks(ax, eq.index)   # одна серия — легенда не нужна, её называет заголовок


def trades_figure(family, market, leg_name, window=320, overlay=None,
                  title="", mechanism="", extra=None, ylabel="непрерывный ряд (ratio), USD"):
    """Панели: механизм и сделки в окне, (опционально сигнал), капитал, сделки."""
    leg = LEGS[(family, market, leg_name)]
    bars, trades = leg["bars"], leg["trades"]
    x0 = max(len(bars) - window, 0)
    view = bars.iloc[x0:]
    px = view["close"]
    xs = np.arange(len(view)) + x0

    rows = [1.35] + ([0.55] if extra else []) + [1.0, 0.75]
    fig = plt.figure(figsize=(11.5, 3.1 * len(rows) + 1.4))
    gs = fig.add_gridspec(len(rows), 1, height_ratios=rows, hspace=0.75)

    ax = fig.add_subplot(gs[0])
    frame(ax, title or f"{family} · {leg_name}",
          mechanism, ylabel="непрерывный ряд (ratio), USD")
    ax.plot(xs, px.to_numpy(float), color=INK, linewidth=1.4, label="цена (сигнальный ряд)")
    if overlay is not None:
        overlay(ax, leg, x0, view)
    in_view = trades[(trades["entry_time"] >= view.index[0])]
    mark_trades(ax, in_view, bars.index, 0)
    shade_position(ax, leg["position"].iloc[x0:], x0)
    date_ticks(ax, view.index, x0=x0)
    ax.set_xlim(x0 - 2, x0 + len(view) + 1)
    below_legend(ax, ncol=5)

    if extra:
        axe = fig.add_subplot(gs[1])
        extra(axe, leg, x0, view)
        date_ticks(axe, view.index, x0=x0)
        axe.set_xlim(x0 - 2, x0 + len(view) + 1)

    equity_panel(fig.add_subplot(gs[1 + bool(extra)]), leg,
                 "Кривая капитала за весь период",
                 f"издержки базового сценария; отбор параметров закончился "
                 f"{leg['is_end'].date()}")

    ax3 = fig.add_subplot(gs[2 + bool(extra)])
    frame(ax3, "Результат каждой сделки",
          (f"за весь период: {len(trades)} сделок, прибыльных "
           f"{(trades['net_pnl'] > 0).mean():.0%}; в отчёте по этой ячейке "
           f"приведены только сделки вне выборки") if len(trades) else "",
          ylabel="% капитала")
    trade_pnl(ax3, trades)
    plt.show()
