# 4б. Период определения наклона k — самый сильный параметрический эффект.
#    Лучшая по IS настройка (доля нейтрали, ATR) при каждом k на решётке
#    этапа 2 и её продлении (этап 2б, k = 55, 89, 144). Пунктир — граница
#    исходной сетки k: оптимум сначала оказался на ней, продление его нашло.
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=False)
for ax, t in zip(axes, TICKERS):
    parts = [pd.read_csv(A / s / t / "settings.csv") for s in ("stage2", "stage2b") if (A / s / t / "settings.csv").exists()]
    s2 = pd.concat(parts)
    P.frame(ax, NAMES[t], "лучшая по IS настройка при каждом k", xlabel="k, баров (лог-шкала)",
            ylabel="медиана Шарпа по решётке")
    for m, c_ in (("A_pct", P.S1), ("B_atr", P.S2)):
        best = s2[s2["method"] == m].sort_values("median_sharpe_is").groupby("k").tail(1).sort_values("k")
        ax.plot(best["k"], best["median_sharpe_is"], color=c_, marker="o", markersize=5, label=f"{m}, IS")
        ax.plot(best["k"], best["median_sharpe_oos"], color=c_, marker="s", markersize=5, linestyle=":",
                label=f"{m}, OOS")
    ax.set_xscale("log"); ax.set_xticks([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144])
    ax.set_xticklabels(["1", "2", "3", "5", "8", "13", "21", "34", "55", "89", "144"])
    ax.axvline(34, color=P.MUTED, linewidth=1.0, linestyle="--")
    ax.axhline(0, color=P.AXIS, linewidth=0.8)
    ax.legend(ncol=2, loc="upper left")
plt.tight_layout(); plt.show()
