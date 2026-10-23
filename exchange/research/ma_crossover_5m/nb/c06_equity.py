# 6. Накопленный PnL отобранных стратегий (VectorBT, переоценка открытой
#    позиции). Каждая панель — одно правило отбора; серый фон — out-of-sample.
for t in TICKERS:
    fin = json.loads((A / STAGE / t / "finalists.json").read_text())
    glob = [f for f in fin if f["scope"] == "global"]
    fig, axes = plt.subplots(1, len(glob), figsize=(5.2 * len(glob), 3.8), sharey=False)
    for ax, f in zip(np.atleast_1d(axes), glob):
        eq = pd.read_parquet(A / STAGE / t / "finalists" / f"cfg{f['grid_config']}_equity.parquet")["cum_pnl_usd"]
        oos0 = pd.Timestamp("2024-01-01")
        P.frame(ax, f"{f['rule']}", f"{f['config']['fast_type']}{f['config']['fast_period']} × "
                f"{f['config']['slow_type']}{f['config']['slow_period']}, {f['config']['method']}, {f['config']['exit']}",
                ylabel="накопленный PnL, USD за 1 контракт")
        ax.axvspan(oos0, eq.index[-1], color=P.NEUTRAL, zorder=0)
        ax.plot(eq.index, eq.values, color=P.S1, linewidth=1.6)
        ax.axhline(0, color=P.AXIS, linewidth=0.8)
        ax.annotate(f"{eq.iloc[-1]:+,.0f}", xy=(eq.index[-1], eq.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", color=P.INK_2, fontsize=9)
        ax.text(0.99, 0.03, "OOS", transform=ax.transAxes, ha="right", color=P.MUTED, fontsize=9)
    fig.suptitle(f"{NAMES[t]}: кривые PnL отобранных по IS стратегий", x=0.01, ha="left", color=P.INK,
                 fontsize=12.5, fontweight="bold", y=1.06)
    plt.tight_layout(); plt.show()
