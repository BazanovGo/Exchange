# 8. Сравнение инструментов: одна и та же конфигурация на NG и BR.
#    Слева — IS-Шарп всех 3.47 млн конфигураций (плотность), справа — медианы
#    144 семейств (типы × выход × метод) вне выборки.
cm = json.loads((A / STAGE / "crossmarket.json").read_text())
fam = pd.read_csv(A / STAGE / "crossmarket_families.csv")
ng, br = np.asarray(cube("NG", "sharpe_m_is")).ravel(), np.asarray(cube("BR", "sharpe_m_is")).ravel()
ok = np.isfinite(ng) & np.isfinite(br)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
ax = axes[0]
P.frame(ax, "IS-Шарп: NG против BR", f"Спирмен {cm['spearman_sharpe_m_is_NG_vs_BR']:+.3f} по {ok.sum():,} конфигурациям",
        xlabel="Шарп на NG", ylabel="Шарп на BR", grid_axis=None)
hb = ax.hexbin(ng[ok], br[ok], gridsize=70, bins="log", cmap=P.SEQUENTIAL, mincnt=1, extent=(-3, 3, -3, 3))
ax.axhline(0, color=P.AXIS, linewidth=0.8); ax.axvline(0, color=P.AXIS, linewidth=0.8)
fig.colorbar(hb, ax=ax, pad=0.01).set_label("число конфигураций (лог)", color=P.INK_2)
ax = axes[1]
P.frame(ax, "Семейства вне выборки", f"Спирмен {cm['families_spearman_oos']:+.3f} по 144 семействам",
        xlabel="медиана OOS-Шарпа на NG", ylabel="медиана OOS-Шарпа на BR", grid_axis="both")
for e, c_, m_ in (("A_fast_cross", P.S1, "o"), ("B_slow_flip", P.S2, "s")):
    d = fam[fam["exit"] == e]
    ax.scatter(d["med_oos_NG"], d["med_oos_BR"], s=26, color=c_, marker=m_, edgecolor=P.SURFACE, linewidth=0.8,
               label=f"выход {e[0]}")
ax.axhline(0, color=P.AXIS, linewidth=0.8); ax.axvline(0, color=P.AXIS, linewidth=0.8); ax.legend()
plt.tight_layout(); plt.show()
pd.Series({k: v for k, v in cm.items() if not isinstance(v, (list, dict))})
