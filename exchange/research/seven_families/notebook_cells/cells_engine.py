UNIVERSE = H.load_universe()
PANEL_ALL = {r: p["bars"] for r, p in UNIVERSE.items()}
COMMON = PANEL_ALL["NG"].index.intersection(PANEL_ALL["BR"].index)

# Конфигурации, отобранные на in-sample в прогоне 2026-10-07
# (docs/RESULTS_SEVEN_FAMILIES.md, раздел 1). Здесь они заданы явно: ноутбук
# показывает сделки уже выбранной системы, а не повторяет отбор.
WINNERS = {
    ("1_tsmom", "BR"): {"lookback": 40},
    ("1_tsmom", "NG"): {"lookback": 90},
    ("2_dual_ma", "BR"): {"fast": 20, "slow": 60},
    ("2_dual_ma", "NG"): {"fast": 20, "slow": 150},
    ("3_breakout", "BR"): {"entry": 40, "exit_": 20},
    ("3_breakout", "NG"): {"entry": 20, "exit_": 10},
    ("4_cross_sectional", "NG+BR"): {"lookback": 120},
    ("5_carry", "BR"): {"smooth": 1, "threshold": 0.15},
    ("5_carry", "NG"): {"smooth": 1, "threshold": 0.15},
    ("6_portfolio_ma", "NG+BR"): {"fast": 20, "slow": 100},
    ("7_mom_reversal", "BR"): {"momentum": 60, "reversal": 750},
    ("7_mom_reversal", "NG"): {"momentum": 60, "reversal": 500},
}


def run_leg(family, market, params):
    """Одна нога системы: сигнал -> целевая позиция -> движок -> сделки.

    Возвращает то, что видел бэктест: бары, по которым считался сигнал,
    сам сигнал, удерживаемую позицию после полосы ребалансировки, журнал
    сделок и кривую капитала. Исполнение — next-open, по построению движка.
    """
    if family in F.PANEL:
        panel = {r: b.loc[COMMON] for r, b in PANEL_ALL.items()}
        signal = F.PANEL[family](panel, **params)[market]
        bars, ppy = panel[market], UNIVERSE[market]["ppy"]
        target = RA.positions(bars, ppy, signal, legs=len(panel))
    else:
        bars = PANEL_ALL[market]
        ppy = UNIVERSE[market]["ppy"]
        signal = F.SINGLE[family](bars, **params)
        target = RA.positions(bars, ppy, signal)
    result, metrics = H.backtest(bars, target, symbol=market, ppy=ppy)
    return {
        "bars": bars, "signal": signal, "target": target, "ppy": ppy,
        "trades": result.trades, "equity": result.equity, "position": result.position,
        "metrics": metrics, "is_end": RA.split(bars.index)[0][-1],
    }


LEGS = {}
for (family, market), params in WINNERS.items():
    for leg in (("NG", "BR") if market == "NG+BR" else (market,)):
        LEGS[(family, market, leg)] = run_leg(family, leg, params)

def direction_changes(position):
    """Сколько раз позиция меняла знак — это и есть «сделки» в обиходном смысле."""
    sign = np.sign(position.to_numpy(float))
    sign = sign[sign != 0]
    return int((np.diff(sign) != 0).sum())


summary = pd.DataFrame([
    {"семейство": f, "рынок": m, "нога": leg,
     "разворотов позиции": direction_changes(v["position"]),
     "записей о сделках": len(v["trades"]),
     "из них long": int((v["trades"]["direction"] > 0).sum()) if len(v["trades"]) else 0,
     "прибыльных": int((v["trades"]["net_pnl"] > 0).sum()) if len(v["trades"]) else 0,
     "медиана, баров": float(v["trades"]["bars_held"].median()) if len(v["trades"]) else np.nan,
     "в рынке, % времени": float((v["position"] != 0).mean() * 100)}
    for (f, m, leg), v in LEGS.items()
])
summary.round(1)
