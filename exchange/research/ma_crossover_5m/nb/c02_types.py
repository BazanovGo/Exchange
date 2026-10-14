# 2. Все 36 сочетаний типов средних. Медиана по всем периодам, обоим методам и
#    выходам; слева — in-sample, справа — out-of-sample. Цвет — знак и сила,
#    число в клетке — сама медиана, поэтому смысл не держится на цвете.
for t in TICKERS:
    fam = pd.read_csv(A / STAGE / t / "by_type_pair.csv")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    m_is = fam.pivot(index="fast_type", columns="slow_type", values="median_sharpe_is").loc[TYPES, TYPES].to_numpy()
    m_oos = fam.pivot(index="fast_type", columns="slow_type", values="median_sharpe_oos").loc[TYPES, TYPES].to_numpy()
    m_pos = fam.pivot(index="fast_type", columns="slow_type", values="share_pos_oos").loc[TYPES, TYPES].to_numpy()
    norm = P.sym_norm(np.r_[m_is.ravel(), m_oos.ravel()], q=1.0)
    P.matrix6(axes[0], m_is, TYPES, TYPES, norm=norm, title="медиана Шарпа, IS")
    P.matrix6(axes[1], m_oos, TYPES, TYPES, norm=norm, title="медиана Шарпа, OOS")
    from matplotlib.colors import TwoSlopeNorm
    P.matrix6(axes[2], m_pos, TYPES, TYPES, norm=TwoSlopeNorm(0.5, 0.0, 1.0), title="доля прибыльных вне выборки",
              fmt="{:.0%}")
    fig.suptitle(f"{NAMES[t]}: 6 × 6 сочетаний типов (строка — быстрая, столбец — медленная)", x=0.01, ha="left",
                 color=P.INK, fontsize=12.5, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.show()
