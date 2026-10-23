# 1. Тепловые карты Fast × Slow для семейства отобранной стратегии (правило R3,
#    плато in-sample). Три панели на инструмент: IS-Шарп, OOS-Шарп и
#    сглаженный IS-Шарп, по которому шёл отбор. Шкала общая и симметричная.
for t in TICKERS:
    pk = picks[t].query("scope == 'global' and rule == 'R3_plateau'").iloc[0]
    mi = ["A_pct", "B_atr"].index(pk["method"]); ei = ["A_fast_cross", "B_slow_flip"].index(pk["exit"])
    si, fi = TYPES.index(pk["slow_type"]), TYPES.index(pk["fast_type"])
    mats = {c: np.asarray(cube(t, c)[mi, si, :, ei, fi, :]) for c in ("sharpe_m_is", "sharpe_m_oos", "plateau_sharpe_is")}
    norm = P.sym_norm(np.concatenate([m.ravel() for m in mats.values()]))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    titles = {"sharpe_m_is": "Шарп in-sample", "sharpe_m_oos": "Шарп out-of-sample",
              "plateau_sharpe_is": "IS-Шарп, сглаженный по окрестности"}
    for ax, (c, m) in zip(axes, mats.items()):
        im = P.heatmap(ax, m, FAST, SLOW, norm=norm, title=titles[c],
                       xlabel="период быстрой", ylabel="период медленной" if c == "sharpe_m_is" else "",
                       mark=(pk["fast_period"], pk["slow_period"]))
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01); cb.set_label("годовой Шарп по месячному PnL", color=P.INK_2)
    fig.suptitle(f"{NAMES[t]}: {pk['fast_type']} × {pk['slow_type']}, наклон {pk['method']}, выход {pk['exit']}"
                 f" — кружок: отобранная по IS конфигурация", x=0.01, ha="left", color=P.INK, fontsize=12.5,
                 fontweight="bold", y=1.03)
    plt.show()
