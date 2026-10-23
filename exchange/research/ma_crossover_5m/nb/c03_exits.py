# 3. Модели выхода: распределения Шарпа всех конфигураций (одна серия — один
#    выход, слоты 1 и 2), отдельно IS и OOS.
fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
bins = np.linspace(-3, 3, 121)
for row, t in enumerate(TICKERS):
    for col, per in enumerate(("is", "oos")):
        c = np.asarray(cube(t, f"sharpe_m_{per}"))
        ax = axes[row, col]
        P.frame(ax, f"{NAMES[t]}, {'in-sample' if per == 'is' else 'out-of-sample'}", xlabel="годовой Шарп",
                ylabel="плотность")
        P.step_hist(ax, {"выход A — обратное пересечение": c[:, :, :, 0].ravel(),
                         "выход B — смена наклона": c[:, :, :, 1].ravel()}, bins)
plt.tight_layout(); plt.show()
pd.DataFrame({t: {k: v for k, v in res[t].items() if k.startswith("exit_B_minus_A")} for t in TICKERS})
