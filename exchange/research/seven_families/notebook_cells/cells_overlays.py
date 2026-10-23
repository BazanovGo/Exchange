def ref_price(label, lag, colour):
    """Цена lag баров назад: сигнал — знак разницы с текущей ценой."""
    def draw(ax, leg, x0, view):
        ref = leg["bars"]["close"].shift(lag).iloc[x0:]
        ax.plot(np.arange(len(view)) + x0, ref.to_numpy(float), color=colour,
                linewidth=1.6, label=label)
    return draw


def ma_overlay(fast, slow):
    def draw(ax, leg, x0, view):
        from exchange.strategies.base import sma
        close = leg["bars"]["close"]
        x = np.arange(len(view)) + x0
        ax.plot(x, sma(close, fast).iloc[x0:].to_numpy(float), color=SERIES_1,
                linewidth=1.8, label=f"SMA {fast}")
        ax.plot(x, sma(close, slow).iloc[x0:].to_numpy(float), color=SERIES_2,
                linewidth=1.8, label=f"SMA {slow}")
    return draw


def channel_overlay(entry, exit_):
    def draw(ax, leg, x0, view):
        bars = leg["bars"]
        x = np.arange(len(view)) + x0
        up = bars["high"].rolling(entry).max().shift(1).iloc[x0:]
        dn = bars["low"].rolling(entry).min().shift(1).iloc[x0:]
        xu = bars["high"].rolling(exit_).max().shift(1).iloc[x0:]
        xd = bars["low"].rolling(exit_).min().shift(1).iloc[x0:]
        ax.plot(x, up.to_numpy(float), color=SERIES_1, linewidth=1.6,
                label=f"канал входа {entry} дн.")
        ax.plot(x, dn.to_numpy(float), color=SERIES_1, linewidth=1.6)
        ax.plot(x, xu.to_numpy(float), color=SERIES_2, linewidth=1.1,
                label=f"канал выхода {exit_} дн.")
        ax.plot(x, xd.to_numpy(float), color=SERIES_2, linewidth=1.1)
    return draw


def rival_overlay(other, lookback):
    """Второй рынок, приведённый к началу окна: сравниваются относительные силы."""
    def draw(ax, leg, x0, view):
        rival = PANEL_ALL[other]["close"].reindex(leg["bars"].index).ffill()
        scaled = rival / float(rival.iloc[x0]) * float(view["close"].iloc[0])
        ax.plot(np.arange(len(view)) + x0, scaled.iloc[x0:].to_numpy(float),
                color=SERIES_2, linewidth=1.6,
                label=f"{other}, приведён к началу окна")
    return draw


def carry_panel(threshold):
    def draw(ax, leg, x0, view):
        carry = leg["bars"]["carry"].iloc[x0:]
        x = np.arange(len(view)) + x0
        frame(ax, "", f"прокси carry по коэффициентам ролла; порог ±{threshold}",
              ylabel="годовой carry")
        ax.plot(x, carry.to_numpy(float), color=INK, linewidth=1.6, label="carry")
        ax.axhline(0, color=AXIS, linewidth=0.8)
        for sign, colour, lab in ((1, LONG, "порог long"), (-1, SHORT, "порог short")):
            ax.axhline(sign * threshold, color=colour, linewidth=1.2,
                       label=lab)
        ax.fill_between(x, threshold, carry.to_numpy(float),
                        where=carry.to_numpy(float) > threshold, color=LONG, alpha=0.12,
                        linewidth=0)
        ax.fill_between(x, -threshold, carry.to_numpy(float),
                        where=carry.to_numpy(float) < -threshold, color=SHORT, alpha=0.12,
                        linewidth=0)
        below_legend(ax, ncol=4)
    return draw


def ref_pair(lag_a, label_a, lag_b, label_b):
    """Два ориентира сразу: горизонт моментума и горизонт реверсии."""
    def draw(ax, leg, x0, view):
        close = leg["bars"]["close"]
        x = np.arange(len(view)) + x0
        for lag, lab, colour in ((lag_a, label_a, SERIES_1), (lag_b, label_b, SERIES_2)):
            ax.plot(x, close.shift(lag).iloc[x0:].to_numpy(float), color=colour,
                    linewidth=1.6, label=lab)
    return draw
