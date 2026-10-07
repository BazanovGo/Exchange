# Малые кратные: одна панель — одна ячейка, одна серия, один цвет.
# Семь перекрывающихся линий на общей оси были бы нечитаемы, а выбор цвета по
# номеру серии нарушил бы правило «цвет следует за сущностью».
order = [("1_tsmom", "BR"), ("2_dual_ma", "BR"), ("3_breakout", "BR"),
         ("5_carry", "BR"), ("7_mom_reversal", "BR"),
         ("1_tsmom", "NG"), ("2_dual_ma", "NG"), ("3_breakout", "NG"),
         ("5_carry", "NG"), ("7_mom_reversal", "NG"),
         ("4_cross_sectional", "NG+BR"), ("6_portfolio_ma", "NG+BR")]
TITLES = {"1_tsmom": "TSMOM", "2_dual_ma": "Dual MA", "3_breakout": "Breakout",
          "4_cross_sectional": "Cross-sect.", "5_carry": "Carry",
          "6_portfolio_ma": "Portfolio MA", "7_mom_reversal": "Mom+Rev"}

fig, axes = plt.subplots(3, 4, figsize=(13.5, 8.2))
fig.suptitle("Капитал вне выборки: двенадцать ячеек, один масштаб",
             color=INK, fontsize=13.5, fontweight="semibold", x=0.055, ha="left", y=0.99)
fig.text(0.055, 0.945, "доходность в % от капитала с первого бара out-of-sample; "
         "сетка одинакова на всех панелях", color=INK_2, fontsize=9.5, ha="left")

curves = {}
for family, market in order:
    legs = [v for (f, m, _), v in LEGS.items() if (f, m) == (family, market)]
    eq = sum((leg["equity"] - H.CAPITAL for leg in legs), start=0.0) + H.CAPITAL
    oos = RA.split(legs[0]["bars"].index)[1]
    eq = eq.reindex(oos).dropna()
    curves[(family, market)] = (eq / float(eq.iloc[0]) - 1.0) * 100.0

lo = min(float(c.min()) for c in curves.values())
hi = max(float(c.max()) for c in curves.values())
pad = (hi - lo) * 0.12
for ax, key in zip(axes.ravel(), order):
    c = curves[key]
    final = float(c.iloc[-1])
    frame(ax, "", "", ylabel="")
    ax.plot(np.arange(len(c)), c.to_numpy(float),
            color=LONG if final > 0 else SHORT, linewidth=1.8)
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(f"{TITLES[key[0]]} · {key[1]}", color=INK, fontsize=10.5,
                 loc="left", pad=8)
    ax.text(0.97, 0.06, f"{final:+.0f}%", transform=ax.transAxes, ha="right",
            color=INK_2, fontsize=10, fontweight="semibold")
    ax.set_xticks([])
for ax in axes.ravel()[len(order):]:
    ax.axis("off")
fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
plt.show()
