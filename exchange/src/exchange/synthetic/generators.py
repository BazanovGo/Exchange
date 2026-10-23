"""Synthetic markets with known ground truth.

These generators are the measuring instruments of the whole system. A
research pipeline that cannot tell a mean-reverting series from a random
walk is not measuring the market — it is measuring itself. So every
statistical component is exercised against series where the answer is known
in advance:

**Negative controls** — :func:`random_walk`. There is no effect. Anything
the pipeline "finds" here is a false discovery, and the acceptance rate on
these series is a direct estimate of the pipeline's own false-positive rate.

**Positive controls** — :func:`mean_reverting`, :func:`momentum`,
:func:`seasonal`, :func:`signal_plus_noise`. A known effect of known size is
present; ``signal_plus_noise`` has an *analytic* target Sharpe, so a
backtest can be checked against theory rather than against itself.

**Trap controls** — :func:`with_lookahead_trap`. A column literally contains
the next bar's return. Any pipeline stage that can be fooled by this must
fail loudly.

Every generator is seeded and deterministic: the same seed gives the same
series on any machine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from exchange.core.determinism import rng as make_rng
from exchange.core.errors import ConfigError
from exchange.core.timeframe import Timeframe

_DEFAULT_START = "2015-01-05 00:00:00+00:00"


@dataclass(frozen=True)
class GroundTruth:
    """What is actually in a synthetic series."""

    kind: str                       # random_walk / mean_reverting / momentum / ...
    has_edge: bool                  # is there anything to find at all?
    params: dict
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "has_edge": self.has_edge,
            "params": dict(self.params),
            "description": self.description,
        }


@dataclass
class SyntheticSeries:
    """Canonical bars plus the ground truth that produced them."""

    bars: pd.DataFrame
    truth: GroundTruth
    seed: int

    @property
    def close(self) -> pd.Series:
        return self.bars["close"]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"SyntheticSeries({self.truth.kind}, {len(self.bars)} bars, seed={self.seed})"


# ==========================================================================
# Index and bar assembly
# ==========================================================================


def make_index(n: int, *, timeframe: str = "1d", start: str = _DEFAULT_START, business: bool = True):
    """A UTC bar index of ``n`` bars.

    Daily series use business days so the empirical annualisation factor
    lands near 252, matching a real equity/futures calendar.
    """
    tf = Timeframe.parse(timeframe)
    if tf.unit == "d" and business:
        return pd.DatetimeIndex(pd.bdate_range(start=pd.Timestamp(start), periods=n, tz="UTC"))
    return pd.DatetimeIndex(pd.date_range(start=pd.Timestamp(start), periods=n, freq=tf.pandas_freq, tz="UTC"))


def bars_from_close(
    close: np.ndarray,
    index: pd.DatetimeIndex,
    *,
    rng: np.random.Generator,
    intrabar_vol: float = 0.4,
    base_volume: float = 1_000_000.0,
) -> pd.DataFrame:
    """Wrap a close path in a plausible OHLCV frame.

    Open is the previous close (a continuously traded market), and the
    high/low straddle the bar by a fraction of that bar's own move plus
    noise — enough to exercise stop logic without pretending to be a real
    microstructure model.
    """
    close = np.asarray(close, dtype=float)
    n = len(close)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1]

    move = np.abs(close - open_)
    typical = np.median(move[move > 0]) if np.any(move > 0) else 0.0
    scale = intrabar_vol * (move + typical + 1e-12)
    up = np.abs(rng.normal(0.0, 1.0, n)) * scale
    down = np.abs(rng.normal(0.0, 1.0, n)) * scale

    high = np.maximum(open_, close) + up
    low = np.minimum(open_, close) - down
    low = np.minimum(low, np.minimum(open_, close))
    low = np.maximum(low, 1e-8)

    volume = base_volume * np.exp(rng.normal(0.0, 0.35, n))
    frame = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
    # Name the index so a CSV round-trip keeps its timestamps identifiable.
    frame.index.name = "timestamp"
    return frame


# ==========================================================================
# Generators
# ==========================================================================


def random_walk(
    n: int = 2000,
    *,
    seed: int = 0,
    s0: float = 100.0,
    vol: float = 0.012,
    drift: float = 0.0,
    timeframe: str = "1d",
    start: str = _DEFAULT_START,
) -> SyntheticSeries:
    """A geometric random walk. **Negative control: no edge exists.**

    ``drift`` is per bar in log space and defaults to zero. A non-zero drift
    is still not an *edge* — it is a market return any buy-and-hold captures
    — so ``has_edge`` stays ``False``.
    """
    r = make_rng(seed)
    index = make_index(n, timeframe=timeframe, start=start)
    shocks = r.normal(drift - 0.5 * vol**2, vol, n)
    close = s0 * np.exp(np.cumsum(shocks))
    return SyntheticSeries(
        bars=bars_from_close(close, index, rng=r),
        truth=GroundTruth(
            "random_walk", False, {"vol": vol, "drift": drift},
            "Independent increments; no predictable structure at any horizon.",
        ),
        seed=seed,
    )


def mean_reverting(
    n: int = 2000,
    *,
    seed: int = 0,
    s0: float = 100.0,
    half_life: float = 12.0,
    vol: float = 0.012,
    timeframe: str = "1d",
    start: str = _DEFAULT_START,
) -> SyntheticSeries:
    """An Ornstein-Uhlenbeck log-price. **Positive control: mean reversion.**

    The log price follows ``x_{t+1} = phi * x_t + eps`` around a constant
    level, with ``phi = 2 ** (-1 / half_life)``, so the theoretical
    half-life of a shock is exactly ``half_life`` bars. The variance-ratio
    and half-life estimators in the scanner are checked against this.
    """
    if half_life <= 0:
        raise ConfigError("half_life must be positive")
    r = make_rng(seed)
    index = make_index(n, timeframe=timeframe, start=start)
    phi = 2.0 ** (-1.0 / half_life)
    x = np.empty(n)
    x[0] = 0.0
    shocks = r.normal(0.0, vol, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + shocks[t]
    close = s0 * np.exp(x)
    return SyntheticSeries(
        bars=bars_from_close(close, index, rng=r),
        truth=GroundTruth(
            "mean_reverting", True,
            {"half_life": half_life, "phi": phi, "vol": vol},
            f"OU log-price; shock half-life {half_life} bars; variance ratio < 1.",
        ),
        seed=seed,
    )


def momentum(
    n: int = 2000,
    *,
    seed: int = 0,
    s0: float = 100.0,
    persistence: float = 0.15,
    vol: float = 0.012,
    timeframe: str = "1d",
    start: str = _DEFAULT_START,
) -> SyntheticSeries:
    """An AR(1) *return* process. **Positive control: trend persistence.**

    ``r_{t+1} = persistence * r_t + eps``. Positive autocorrelation of
    returns means variance ratio > 1 and a real, if small, trend-following
    edge whose size is known from ``persistence``.
    """
    if not -1 < persistence < 1:
        raise ConfigError("persistence must be in (-1, 1)")
    r = make_rng(seed)
    index = make_index(n, timeframe=timeframe, start=start)
    innovation_vol = vol * np.sqrt(1 - persistence**2)
    rets = np.empty(n)
    rets[0] = r.normal(0.0, vol)
    shocks = r.normal(0.0, innovation_vol, n)
    for t in range(1, n):
        rets[t] = persistence * rets[t - 1] + shocks[t]
    close = s0 * np.exp(np.cumsum(rets - 0.5 * vol**2))
    return SyntheticSeries(
        bars=bars_from_close(close, index, rng=r),
        truth=GroundTruth(
            "momentum", True, {"persistence": persistence, "vol": vol},
            f"AR(1) returns with rho={persistence}; variance ratio > 1.",
        ),
        seed=seed,
    )


def seasonal(
    n: int = 2000,
    *,
    seed: int = 0,
    s0: float = 100.0,
    period: int = 21,
    amplitude: float = 0.02,
    vol: float = 0.012,
    timeframe: str = "1d",
    start: str = _DEFAULT_START,
) -> SyntheticSeries:
    """A deterministic cycle plus noise. **Positive control: seasonality.**

    The seasonal component is in the *return*, so the effect is tradeable
    rather than a level artefact.
    """
    r = make_rng(seed)
    index = make_index(n, timeframe=timeframe, start=start)
    phase = 2 * np.pi * np.arange(n) / period
    rets = amplitude * np.sin(phase) + r.normal(0.0, vol, n)
    close = s0 * np.exp(np.cumsum(rets - 0.5 * vol**2))
    return SyntheticSeries(
        bars=bars_from_close(close, index, rng=r),
        truth=GroundTruth(
            "seasonal", True, {"period": period, "amplitude": amplitude, "vol": vol},
            f"Sinusoidal return component with period {period} bars.",
        ),
        seed=seed,
    )


def regime_switch(
    n: int = 2000,
    *,
    seed: int = 0,
    s0: float = 100.0,
    vols: tuple[float, float] = (0.006, 0.024),
    persistence: float = 0.99,
    timeframe: str = "1d",
    start: str = _DEFAULT_START,
) -> SyntheticSeries:
    """A two-state volatility regime (Markov switching).

    Not an *edge* by itself: the mean is zero in both states, so a naive
    directional strategy has nothing to find. It exists to test regime
    detection and to check that a strategy's performance is not simply a
    function of which regime the sample happened to contain.
    """
    r = make_rng(seed)
    index = make_index(n, timeframe=timeframe, start=start)
    state = np.zeros(n, dtype=int)
    for t in range(1, n):
        state[t] = state[t - 1] if r.random() < persistence else 1 - state[t - 1]
    sigma = np.where(state == 0, vols[0], vols[1])
    rets = r.normal(0.0, 1.0, n) * sigma
    close = s0 * np.exp(np.cumsum(rets - 0.5 * sigma**2))
    bars = bars_from_close(close, index, rng=r)
    bars["regime"] = state
    return SyntheticSeries(
        bars=bars,
        truth=GroundTruth(
            "regime_switch", False, {"vols": list(vols), "persistence": persistence},
            "Two volatility regimes, zero mean in both: volatility clustering without "
            "directional edge.",
        ),
        seed=seed,
    )


def structural_break(
    n: int = 2000,
    *,
    seed: int = 0,
    s0: float = 100.0,
    break_at: float = 0.5,
    before: str = "momentum",
    after: str = "mean_reverting",
    vol: float = 0.012,
    timeframe: str = "1d",
    start: str = _DEFAULT_START,
) -> SyntheticSeries:
    """One process before a break point, a different one after.

    The point of this control: an in-sample fit on the first half must not
    survive out-of-sample on the second. A pipeline that reports such a
    strategy as ROBUST is broken.
    """
    if not 0.1 <= break_at <= 0.9:
        raise ConfigError("break_at must be within [0.1, 0.9]")
    cut = int(n * break_at)
    kinds = {"momentum": momentum, "mean_reverting": mean_reverting, "random_walk": random_walk}
    if before not in kinds or after not in kinds:
        raise ConfigError(f"segments must be one of {sorted(kinds)}")

    first = kinds[before](cut, seed=seed, s0=s0, vol=vol, timeframe=timeframe, start=start)
    level = float(first.close.iloc[-1])
    second = kinds[after](n - cut, seed=seed + 1, s0=level, vol=vol, timeframe=timeframe, start=start)

    index = make_index(n, timeframe=timeframe, start=start)
    bars = pd.concat([first.bars, second.bars], axis=0)
    bars.index = index
    bars.index.name = "timestamp"
    bars["segment"] = np.r_[np.zeros(cut, dtype=int), np.ones(n - cut, dtype=int)]
    return SyntheticSeries(
        bars=bars,
        truth=GroundTruth(
            "structural_break", True,
            {"break_at": break_at, "break_index": cut, "before": before, "after": after},
            f"{before} for {cut} bars, then {after}: an IS fit must not survive OOS.",
        ),
        seed=seed,
    )


def with_lookahead_trap(
    base: SyntheticSeries | None = None,
    *,
    n: int = 2000,
    seed: int = 0,
    column: str = "oracle",
    strength: float = 1.0,
) -> SyntheticSeries:
    """Add a column containing the **next** bar's return. **Trap control.**

    Nothing in the pipeline should ever be able to profit from this without
    a lookahead guard firing. Used two ways:

    * as a positive control for
      :func:`exchange.data.temporal.detect_column_leakage` — a column that
      arrived already contaminated must be caught;
    * as a canary in CI — if a full research run on this dataset produces a
      spectacular Sharpe and no guard trips, a leak has been introduced.
    """
    series = base if base is not None else random_walk(n, seed=seed)
    bars = series.bars.copy()
    future_return = bars["close"].pct_change().shift(-1)
    bars[column] = strength * future_return.fillna(0.0)
    return SyntheticSeries(
        bars=bars,
        truth=GroundTruth(
            "lookahead_trap", True,
            {"column": column, "strength": strength, "base": series.truth.kind},
            f"Column {column!r} holds the next bar's return; any use of it unshifted is a leak.",
        ),
        seed=series.seed,
    )


def signal_plus_noise(
    n: int = 2000,
    *,
    seed: int = 0,
    s0: float = 100.0,
    edge_per_bar: float = 0.0006,
    lookback: int = 5,
    vol: float = 0.012,
    timeframe: str = "1d",
    start: str = _DEFAULT_START,
) -> SyntheticSeries:
    """A calibrated positive control with a *known* achievable Sharpe.

    Construction: the next return contains a component proportional to the
    sign of the trailing ``lookback``-bar return. A strategy that goes long
    on a positive trailing return therefore earns ``edge_per_bar`` on
    average. The theoretical annualised Sharpe of that rule is

        ``edge_per_bar / vol * sqrt(bars_per_year)``

    which is stored in the ground truth, so a backtest can be checked
    against an analytic target instead of against itself.
    """
    r = make_rng(seed)
    index = make_index(n, timeframe=timeframe, start=start)
    rets = np.empty(n)
    logp = np.empty(n)
    rets[0] = r.normal(0.0, vol)
    logp[0] = np.log(s0)
    noise = r.normal(0.0, vol, n)
    for t in range(1, n):
        window = rets[max(0, t - lookback): t]
        direction = np.sign(window.sum()) if window.size else 0.0
        rets[t] = direction * edge_per_bar + noise[t]
        logp[t] = logp[t - 1] + rets[t] - 0.5 * vol**2
    close = np.exp(logp)
    return SyntheticSeries(
        bars=bars_from_close(close, index, rng=r),
        truth=GroundTruth(
            "signal_plus_noise", True,
            {
                "edge_per_bar": edge_per_bar,
                "lookback": lookback,
                "vol": vol,
                "sharpe_per_sqrt_bar": edge_per_bar / vol,
            },
            f"Trailing-{lookback}-bar sign predicts the next return by {edge_per_bar:.2%} per bar.",
        ),
        seed=seed,
    )


# ==========================================================================
# Synthetic futures ladder
# ==========================================================================


def futures_ladder(
    root: str = "NG",
    *,
    n_contracts: int = 12,
    overlap_days: int = 25,
    seed: int = 0,
    s0: float = 3.0,
    vol: float = 0.02,
    term_premium: float = 0.05,
    start: str = "2025-01-01 00:00:00+00:00",
) -> tuple[dict[str, pd.DataFrame], dict, GroundTruth]:
    """A ladder of dated contracts sharing one underlying spot path.

    Contract ``k`` is the front month during calendar month ``k`` and stops
    trading on that month's last business day; its delivery month is the
    following month, mirroring the NG/CL convention. Each contract quotes
    ``spot * (1 + term_premium * years_to_expiry)`` plus small basis noise
    and also trades for ``overlap_days`` business days before its front
    period, so a splice has real overlapping quotes to measure a roll gap on.

    ``term_premium > 0`` is contango: a naively spliced *unadjusted* series
    jumps **up** at every roll (the deferred contract is dearer), inventing
    a positive drift no trader could earn. That artefact is the thing
    :mod:`exchange.data.continuous` exists to remove.

    Ground truth: the spot path is a driftless random walk with **no edge**.
    Any strategy that makes money on the spliced series is trading the
    splice, not the market.

    Returns
    -------
    ``(panel, specs, truth)`` where ``specs`` maps symbol to
    :class:`~exchange.data.contracts.ContractSpec` carrying exact last trade
    dates — no estimation, so a roll-rule test measures the rule rather than
    a date guess.
    """
    from exchange.data.contracts import BUILTIN_ROOTS, ContractMonth, RootSpec

    r = make_rng(seed)
    first_day = pd.Timestamp(start)
    if first_day.tz is not None:
        first_day = first_day.tz_localize(None)
    first_day = first_day.normalize()
    months = [ContractMonth(first_day.year, first_day.month).next(k) for k in range(n_contracts + 2)]

    span_end = pd.Timestamp(year=months[n_contracts].year, month=months[n_contracts].month, day=1)
    index = pd.DatetimeIndex(
        pd.bdate_range(first_day, span_end + pd.Timedelta(days=overlap_days * 2), tz="UTC")
    )
    spot = s0 * np.exp(np.cumsum(r.normal(-0.5 * vol**2, vol, len(index))))
    spot_series = pd.Series(spot, index=index)

    root_spec = BUILTIN_ROOTS.get(root.upper(), RootSpec(root=root.upper()))
    panel: dict[str, pd.DataFrame] = {}
    specs: dict[str, object] = {}

    for k in range(n_contracts):
        front_month, delivery = months[k], months[k + 1]
        symbol = delivery.symbol(root)
        month_start = pd.Timestamp(year=front_month.year, month=front_month.month, day=1, tz="UTC")
        month_end = month_start + pd.offsets.MonthEnd(1)
        candidates = index[index <= month_end]
        if len(candidates) == 0:
            continue
        last_trade = candidates[-1]

        lo = month_start - pd.tseries.offsets.BDay(overlap_days)
        window = index[(index >= lo) & (index <= last_trade)]
        if len(window) < 5:
            continue

        # Years to expiry, decaying to zero at the last trade date.
        ttm = (last_trade - window).days.to_numpy(dtype=float) / 365.25
        basis = 1.0 + term_premium * ttm
        noise = np.exp(r.normal(0.0, vol * 0.05, len(window)))
        close = spot_series.loc[window].to_numpy() * basis * noise

        bars = bars_from_close(close, window, rng=r, base_volume=1.0)
        # Activity peaks while the contract is the front month, so an
        # activity roll has something real to react to.
        centre = month_start + (month_end - month_start) / 2
        age_days = (window - centre).days.to_numpy(dtype=float)
        activity = np.exp(-0.5 * (age_days / 20.0) ** 2)
        bars["volume"] = 1e5 * activity + 1e3
        bars["open_interest"] = 2e5 * activity + 5e3
        panel[symbol] = bars
        specs[symbol] = root_spec.contract(
            delivery.year, delivery.month, last_trade_date=last_trade.date()
        )

    truth = GroundTruth(
        "futures_ladder", False,
        {
            "root": root,
            "n_contracts": len(panel),
            "term_premium": term_premium,
            "structure": "contango" if term_premium > 0 else "backwardation",
        },
        "Common driftless spot path with a linear term structure; the spot path has no edge, "
        "so any spliced-series profit is a roll artefact.",
    )
    return panel, specs, truth


CONTROLS = {
    "random_walk": random_walk,
    "mean_reverting": mean_reverting,
    "momentum": momentum,
    "seasonal": seasonal,
    "regime_switch": regime_switch,
    "structural_break": structural_break,
    "signal_plus_noise": signal_plus_noise,
}


def make_control(kind: str, **kwargs) -> SyntheticSeries:
    """Build a named control series."""
    if kind not in CONTROLS:
        raise ConfigError(f"unknown control {kind!r}; known: {sorted(CONTROLS)}")
    return CONTROLS[kind](**kwargs)
