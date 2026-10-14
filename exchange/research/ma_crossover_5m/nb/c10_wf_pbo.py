# Дополнительно: walk-forward с переоптимизацией по всей сетке и распределение
# логитов PBO. WF: окно обучения 24 месяца, тест 3 месяца, шаг 3 месяца.
fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
for row, t in enumerate(TICKERS):
    ax = axes[row, 0]
    months = pd.PeriodIndex(meta[t]["months"], freq="M")
    P.frame(ax, f"{NAMES[t]}: walk-forward, накопленный PnL", ylabel="USD за 1 контракт")
    for crit, c_ in (("pnl", P.S3), ("sharpe", P.S1), ("plateau", P.S2)):
        m = np.load(A / STAGE / t / f"wf_{crit}_months.npy")
        x = months[-len(m):].to_timestamp()
        wf = res[t]["walk_forward"][crit]
        ax.plot(x, np.cumsum(m), color=c_, linewidth=1.6, label=f"{crit}: {wf['total_pnl']:+,.0f}, t={wf['t_stat']:+.2f}")
    ax.axhline(0, color=P.AXIS, linewidth=0.8); ax.legend(loc="upper left")
    ax = axes[row, 1]
    lg = np.load(A / STAGE / t / "pbo_full_logits.npy")
    P.frame(ax, f"{NAMES[t]}: PBO = {res[t]['pbo_full']['pbo']:.2f}", "логит ранга лучшей IS-конфигурации на OOS-половине",
            xlabel="логит", ylabel="число разбиений")
    ax.hist(lg[lg > 0], bins=40, color=P.POS, label="лучше медианы"); ax.hist(lg[lg <= 0], bins=40, color=P.NEG, label="хуже медианы")
    ax.axvline(0, color=P.INK, linewidth=1); ax.legend()
plt.tight_layout(); plt.show()
