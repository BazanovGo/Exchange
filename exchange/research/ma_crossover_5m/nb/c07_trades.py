# 7. Распределения результатов сделок отобранных стратегий в тиках
#    (1 тик = 0.1 USD для обоих контрактов). Вертикаль — средняя сделка.
for t in TICKERS:
    fin = [f for f in json.loads((A / STAGE / t / "finalists.json").read_text()) if f["scope"] == "global"]
    fig, axes = plt.subplots(1, len(fin), figsize=(5.2 * len(fin), 3.6))
    tick_usd = meta[t]["tick"] * meta[t]["multiplier"]
    for ax, f in zip(np.atleast_1d(axes), fin):
        tr = pd.read_parquet(A / STAGE / t / "finalists" / f"cfg{f['grid_config']}_trades.parquet")
        x = tr["PnL"].to_numpy() / tick_usd
        lo, hi = np.percentile(x, [0.5, 99.5])
        P.frame(ax, f["rule"], f"{len(x):,} сделок, средняя {x.mean():+.2f} тика, медиана {np.median(x):+.1f}",
                xlabel="результат сделки, тиков", ylabel="число сделок")
        b = np.linspace(lo, hi, 80)
        ax.hist(x[x >= 0], bins=b, color=P.POS, alpha=0.85, label="прибыльные")
        ax.hist(x[x < 0], bins=b, color=P.NEG, alpha=0.85, label="убыточные")
        ax.axvline(x.mean(), color=P.INK, linewidth=1.2)
        ax.legend()
    fig.suptitle(f"{NAMES[t]}: результаты сделок", x=0.01, ha="left", color=P.INK, fontsize=12.5,
                 fontweight="bold", y=1.06)
    plt.tight_layout(); plt.show()
