# 4. Методы нейтральной зоны A (процент) и B (ATR). Слева — распределения на
#    полной сетке; справа — этап 2: медиана IS- и OOS-Шарпа по решётке для
#    каждой пары (k, доля нейтрали), метод B — при лучшем по IS периоде ATR.
fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
bins = np.linspace(-3, 3, 121)
for row, t in enumerate(TICKERS):
    c = np.asarray(cube(t, "sharpe_m_oos"))
    ax = axes[row, 0]
    P.frame(ax, f"{NAMES[t]}: OOS-Шарп по методам", xlabel="годовой Шарп", ylabel="плотность")
    P.step_hist(ax, {"A — процентный наклон": c[0].ravel(), "B — наклон в ATR": c[1].ravel()}, bins)
    ax = axes[row, 1]
    mp_path = A / "stage2" / t / "method_pairs.csv"
    if not mp_path.exists():
        ax.axis("off"); continue
    mp = pd.read_csv(mp_path)
    P.frame(ax, f"{NAMES[t]}: этап 2, медиана Шарпа по решётке", xlabel="доля нейтрали на IS", ylabel="медиана Шарпа")
    for (lab, col, mk_) in (("A, IS", "A_is", "o"), ("B, IS", "B_is", "s"), ("A, OOS", "A_oos", "o"), ("B, OOS", "B_oos", "s")):
        g = mp.groupby("coverage")[col].median()
        ax.plot(g.index, g.values, marker=mk_, color=P.S1 if lab.startswith("A") else P.S2,
                linestyle="-" if "IS" in lab else ":", label=lab)
    ax.axhline(0, color=P.AXIS, linewidth=0.8); ax.legend(ncol=2)
plt.tight_layout(); plt.show()
