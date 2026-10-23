# 9. In-sample против out-of-sample на всей сетке. Если отбор по IS работает,
#    верхние децили IS должны давать лучший OOS; плоская или падающая лесенка
#    означает, что IS-результат не несёт информации о будущем.
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
for row, t in enumerate(TICKERS):
    a, b = np.asarray(cube(t, "sharpe_m_is")).ravel(), np.asarray(cube(t, "sharpe_m_oos")).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    ax = axes[row, 0]
    P.frame(ax, f"{NAMES[t]}: IS против OOS", f"Спирмен {res[t]['spearman_is_oos_sharpe']:+.3f}",
            xlabel="Шарп in-sample", ylabel="Шарп out-of-sample", grid_axis=None)
    hb = ax.hexbin(a[ok], b[ok], gridsize=70, bins="log", cmap=P.SEQUENTIAL, mincnt=1, extent=(-3, 3, -3, 3))
    ax.axhline(0, color=P.AXIS, linewidth=0.8); ax.axvline(0, color=P.AXIS, linewidth=0.8)
    fig.colorbar(hb, ax=ax, pad=0.01).set_label("число конфигураций (лог)", color=P.INK_2)
    d1 = pd.read_csv(A / STAGE / t / "deciles_is_oos.csv"); d2 = pd.read_csv(A / STAGE / t / "deciles_plateau_oos.csv")
    ax = axes[row, 1]
    P.frame(ax, f"{NAMES[t]}: OOS-Шарп по децилям отбора", "децили 1–10 по IS-критерию", xlabel="дециль",
            ylabel="средний OOS-Шарп")
    x = np.arange(1, 11)
    ax.bar(x - 0.2, d1["oos_sharpe"], width=0.38, color=P.S1, label="по IS-Шарпу")
    ax.bar(x + 0.2, d2["oos_sharpe"], width=0.38, color=P.S2, label="по плато IS-Шарпа")
    ax.axhline(0, color=P.AXIS, linewidth=0.8); ax.set_xticks(x); ax.legend()
plt.tight_layout(); plt.show()
