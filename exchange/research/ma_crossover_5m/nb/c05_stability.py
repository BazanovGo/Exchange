# 5. Карты устойчивости: доля соседей (±3 быстрой, ±7 медленной) с тем же
#    знаком IS-Шарпа. 1 — вся окрестность согласна, 0.5 — монетка. Рядом —
#    доля прибыльных вне выборки по той же сетке: совпадают ли области.
from matplotlib.colors import Normalize
for t in TICKERS:
    pk = picks[t].query("scope == 'global' and rule == 'R3_plateau'").iloc[0]
    mi = ["A_pct", "B_atr"].index(pk["method"]); ei = ["A_fast_cross", "B_slow_flip"].index(pk["exit"])
    si, fi = TYPES.index(pk["slow_type"]), TYPES.index(pk["fast_type"])
    stab = np.asarray(cube(t, "stability_is")[mi, si, :, ei, fi, :])
    sh_is = np.asarray(cube(t, "sharpe_m_is")[mi, si, :, ei, fi, :])
    oos_pos = (np.asarray(cube(t, "pnl_oos")[mi, si, :, ei, fi, :]) > 0).astype(float)
    signed = np.where(sh_is > 0, stab, -stab)            # синий — устойчиво прибыльно, красный — устойчиво убыточно
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    from matplotlib.colors import TwoSlopeNorm
    im = P.heatmap(axes[0], signed, FAST, SLOW, norm=TwoSlopeNorm(0, -1, 1), title="устойчивость знака IS",
                   xlabel="период быстрой", ylabel="период медленной", mark=(pk["fast_period"], pk["slow_period"]))
    fig.colorbar(im, ax=axes[0], shrink=0.85, pad=0.01).set_label("± доля соседей того же знака", color=P.INK_2)
    im2 = P.heatmap(axes[1], oos_pos, FAST, SLOW, norm=TwoSlopeNorm(0.5, 0, 1), title="прибыльна ли вне выборки",
                    xlabel="период быстрой", mark=(pk["fast_period"], pk["slow_period"]))
    fig.colorbar(im2, ax=axes[1], shrink=0.85, pad=0.01).set_label("1 — да, 0 — нет", color=P.INK_2)
    fig.suptitle(f"{NAMES[t]}: {pk['fast_type']} × {pk['slow_type']}, {pk['method']}, {pk['exit']}",
                 x=0.01, ha="left", color=P.INK, fontsize=12.5, fontweight="bold", y=1.03)
    plt.show()
