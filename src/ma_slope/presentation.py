"""Helpers and extra figures for the project presentation notebook (07).

The figures here explain the pipeline rather than the grid results: the
continuous series built from AlgoPack, one roll before and after the ratio
back-adjustment, and the strategy rules on a few days of real 5m candles.
Everything is read from files committed to the repository.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data import load_data
from src.ma_slope import finalist as F
from src.ma_slope import plots as P
from src.ma_slope.averages import moving_average
from src.ma_slope.data import TICK_SIZES, TICKERS, load_instrument
from src.ma_slope.grid import EXIT_MODELS, Slope
from src.ma_slope.signals import DOWN, UP
from src.utils.paths import PROJECT_ROOT, RAW_DATA_DIR

plt = P.plt
STUDY = PROJECT_ROOT / "research" / "ma_cross_slope_5m"
TABLES, FIGS, GRID = STUDY / "tables", STUDY / "figures", STUDY / "grid"
ALGOPACK = RAW_DATA_DIR / "moex_algopack"
NAMES = {"ng": "NG", "br": "BR (Brent)", "wti": "WTI (CL)", "gold": "GOLD (GC)", "silv": "SILV (SI)",
         "copper": "COPPER (HG)", "plt": "PLT (PL)", "pld": "PLD (PA)"}
WORLD = {"ng": "NG", "br": "Brent", "wti": "CL", "gold": "GC", "silv": "SI", "copper": "HG", "plt": "PL",
         "pld": "PA"}
ROLL_EXAMPLE = ("plt", "PTH2", "PTM2")  # March 2022: the largest roll gap in the data (15.5%)
STRATEGY_EXAMPLE = ("ng", "VWMA", 21, "EMA", 42, "pct", "slope_flip")  # NG finalist, stage 3


def show(name, width=1200, folder=FIGS):
    """Display a figure, downscaled to a 256-colour palette so the executed notebook stays light."""
    from IPython.display import Image, display
    from PIL import Image as PILImage

    img = PILImage.open(Path(folder) / name).convert("RGB")
    if img.width > width:
        img = img.resize((width, round(img.height * width / img.width)), PILImage.LANCZOS)
    buf = BytesIO()
    img.quantize(colors=256, method=PILImage.Quantize.MEDIANCUT).save(buf, format="PNG", optimize=True)
    display(Image(data=buf.getvalue()))


def table(name, **kw) -> pd.DataFrame:
    return pd.read_csv(TABLES / name, **kw)


def summary() -> dict:
    return json.loads((TABLES / "summary.json").read_text())


def grid_meta(ticker, stage="stage3") -> dict:
    return json.loads(pq.read_schema(GRID / f"{stage}_{ticker}.parquet").metadata[b"ma_slope"])


def data_overview() -> pd.DataFrame:
    """One row per instrument: history, candle counts, contracts and rolls, IS/OOS split."""
    files = json.loads((ALGOPACK / "manifest.json").read_text())["files"]
    rows = []
    for t in TICKERS:
        rolls = pd.read_csv(ALGOPACK / t / "rolls.csv")
        m = grid_meta(t)
        rows.append({
            "инструмент": NAMES[t], "мировой тикер": WORLD[t], "начало": m["start"][:10], "конец": m["end"][:10],
            "свечей 1m": sum(n for f, n in files.items() if f.startswith(f"{t}/1m/")),
            "баров 5m": m["bars"], "контрактов": len(rolls) + 1, "переходов": len(rolls),
            "макс. разрыв на ролле, %": f"{100 * (rolls['ratio'] - 1).abs().max():.1f}",
            "граница IS/OOS": m["split_time"][:10], "USD за пункт": f"{m['multiplier']:g}",
            "тик": f"{TICK_SIZES[t]:g}"})
    return pd.DataFrame(rows).set_index("инструмент")


def annualization_check() -> pd.DataFrame:
    """Hourly bars a year actually traded vs the 24/7 clock and the Sharpe overstatement it caused."""
    rows = []
    for t in TICKERS:
        idx = load_data("moex_algopack", t, timeframe="60m").get().index
        years = (idx[-1] - idx[0]).total_seconds() / (365 * 86400)
        per_year = (len(idx) - 1) / years
        rows.append({"инструмент": NAMES[t], "часовых свечей в год (факт)": round(per_year),
                     "по часам 24/7": 8760, "завышение Sharpe при 24/7, раз": round(np.sqrt(8760 / per_year), 2)})
    return pd.DataFrame(rows).set_index("инструмент")


def price_overview_figure(path=FIGS / "overview_prices.png") -> Path:
    """Daily close of every continuous ratio-adjusted series, IS/OOS split marked."""
    curves, splits = {}, {}
    for t in TICKERS:
        close = load_data("moex_algopack", t, timeframe="60m").get()["Close"]
        daily = close.groupby(close.index.normalize()).last()
        curves[NAMES[t]] = [("close", daily.index, daily.to_numpy())]
        splits[NAMES[t]] = pd.Timestamp(grid_meta(t)["split_time"])
    return P.small_multiples_lines(
        curves, "Continuous MOEX AlgoPack futures, ratio back-adjusted daily close (USD); line = IS/OOS split",
        "price", path, vlines=splits, zero_line=False)


def roll_example_figure(path=FIGS / "roll_example_plt.png") -> Path:
    """One roll on raw glued 5m prices vs the ratio back-adjusted series (rescaled to the new contract)."""
    t, old, new = ROLL_EXAMPLE
    roll = pd.read_csv(ALGOPACK / t / "rolls.csv", parse_dates=["from_time"])
    roll = roll[(roll["from_secid"] == old) & (roll["to_secid"] == new)].iloc[0]
    lo, hi = roll["from_time"] - pd.Timedelta(days=4), roll["from_time"] + pd.Timedelta(days=4)
    raw = load_data("moex_algopack", t, timeframe="5m", adjust=None, start=lo, end=hi).get().loc[lo:hi]
    adj = load_data("moex_algopack", t, timeframe="5m", start=lo, end=hi).get().loc[lo:hi]
    after = adj["secid"] == new
    # Rescale so the new contract keeps its real prices: only this roll's ratio remains in the history.
    local = adj["Close"] / adj.loc[after, "adj_factor"].iloc[0]
    x = np.arange(len(raw))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(x, raw["Close"].to_numpy(), color=P.SERIES[1], lw=1.6, label="raw glued contracts")
    ax.plot(x, local.to_numpy(), color=P.SERIES[0], lw=1.6, label="ratio back-adjusted")
    first_new = int(np.flatnonzero(after.to_numpy())[0])
    ax.axvline(first_new - 0.5, color=P.INK2, lw=1)
    ax.annotate(f"roll {old} -> {new}: new/old = {roll['ratio']:.3f} at the same minute",
                (first_new - 0.5, ax.get_ylim()[1]), xytext=(6, -14), textcoords="offset points",
                fontsize=8, color=P.INK2)
    days = raw.index.normalize()
    ticks = np.flatnonzero(np.r_[True, days[1:] != days[:-1]])
    ax.set_xticks(ticks, [d.strftime("%d %b") for d in raw.index[ticks]])
    ax.set_ylabel("USD per troy ounce")
    ax.set_title(f"{NAMES[t]}, 5m close around the {roll['from_time']:%B %Y} roll (trading bars, no time gaps)",
                 pad=24)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=2)
    return P._save(fig, path)


def _example_window(tr: pd.DataFrame, ins, days=3, trades=(3, 5)):
    """Latest ``days`` consecutive weekdays in one contract with 3-5 whole trades: long and short, win and loss."""
    dates = pd.Index(ins.index.normalize().unique())
    for end in range(len(dates) - 1, days, -1):
        span = dates[end - days + 1:end + 1]
        if (span.dayofweek >= 5).any() or (span[-1] - span[0]).days != days - 1:
            continue
        lo, hi = span[0], span[-1] + pd.Timedelta(days=1)
        inside = tr[(tr["entry_time"] >= lo) & (tr["exit_time"] < hi)]
        window = (ins.index >= lo) & (ins.index < hi)
        if (trades[0] <= len(inside) <= trades[1] and inside["direction"].nunique() == 2
                and (inside["pnl"] > 0).any() and (inside["pnl"] < 0).any() and len(set(ins.secid[window])) == 1):
            return lo, hi, inside
    raise ValueError("no window with enough trades")


def strategy_example_figure(path=FIGS / "strategy_example_ng.png") -> Path:
    """The rules on real candles: price, Fast and Slow MA, slope state, entries and exits of a finalist."""
    t, ft, fp, st, sp, method, model = STRATEGY_EXAMPLE
    slope = Slope(**grid_meta(t)["slopes"][f"{method}/{model}"])
    cfg = F.Config(ft, fp, st, sp, slope, EXIT_MODELS.index(model))
    ins = load_instrument(t)
    tr = F.trades(ins, cfg)
    _, _, state = F.signals(ins, cfg)
    fast = moving_average(ft, ins.close, ins.volume, fp)
    slow = moving_average(st, ins.close, ins.volume, sp)
    lo, hi, inside = _example_window(tr, ins)
    w = np.flatnonzero((ins.index >= lo) & (ins.index < hi))
    x = np.arange(w.size)
    pos = {ts: i for i, ts in enumerate(ins.index[w])}
    k = slope.k
    slope_pct = (slow[w] - slow[w - k]) / slow[w - k] * 100

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    for a in (ax, ax2):
        for value, colour in ((UP, "#dcebfc"), (DOWN, "#fbe1e1")):
            on = state[w] == value
            edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
            for s, e in zip(edges[::2], edges[1::2], strict=True):
                a.axvspan(s - 0.5, e - 0.5, color=colour, lw=0, zorder=0)
    ax.plot(x, ins.close[w], color=P.INK2, lw=1.0, label="close")
    ax.plot(x, fast[w], color=P.SERIES[0], lw=1.8, label=f"fast {ft}{fp}")
    ax.plot(x, slow[w], color=P.SERIES[1], lw=1.8, label=f"slow {st}{sp}")
    for _, r in inside.iterrows():
        e, xx = pos[r["entry_time"]], pos[r["exit_time"]]
        long = r["direction"] == 1
        ax.plot(e, ins.open[w][e], marker="^" if long else "v", ms=10, color=P.POS if long else P.NEG,
                mec=P.SURFACE, mew=1, zorder=5)
        ax.plot(xx, ins.open[w][xx], marker="o", ms=8, mfc=P.SURFACE, mec=P.INK, mew=1.4, zorder=5)
        ax.plot([e, xx], [ins.open[w][e], ins.open[w][xx]], color=P.MUTED, lw=0.8, ls=":", zorder=4)
        ax.annotate(f"{r['pnl']:+.0f}$", (xx, ins.open[w][xx]), xytext=(6, 8), textcoords="offset points",
                    fontsize=8, color=P.INK, bbox={"boxstyle": "round,pad=0.15", "fc": P.SURFACE, "ec": "none"})
    ax.plot([], [], marker="^", ls="", ms=9, color=P.POS, label="long entry (next open)")
    ax.plot([], [], marker="v", ls="", ms=9, color=P.NEG, label="short entry (next open)")
    ax.plot([], [], marker="o", ls="", ms=8, mfc=P.SURFACE, mec=P.INK, label="exit: slow slope flips")
    ax.set_ylabel("USD / MMBtu")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=6, fontsize=7.5)
    ax.set_title(f"{NAMES[t]} 5m, {lo:%d}–{hi - pd.Timedelta(days=1):%d %b %Y}: {cfg.label}", pad=30)
    ax2.plot(x, slope_pct, color=P.SERIES[1], lw=1.6)
    ax2.axhline(slope.eps, color=P.BASE, lw=1)
    ax2.set_ylabel(f"slow slope, % / {k} bars")
    ax2.text(0.005, 0.92, "background: blue = slope up (longs allowed), red = slope down (shorts allowed)",
             transform=ax2.transAxes, fontsize=7.5, color=P.INK2, va="top")
    days = ins.index[w].normalize()
    ticks = np.flatnonzero(np.r_[True, days[1:] != days[:-1]])
    ax2.set_xticks(ticks, [d.strftime("%d %b") for d in ins.index[w][ticks]])
    fig.tight_layout()
    return P._save(fig, path)


def build_figures() -> list[Path]:
    return [price_overview_figure(), roll_example_figure(), strategy_example_figure()]
