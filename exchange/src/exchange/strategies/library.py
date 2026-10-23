"""A small reference library of strategies.

Deliberately small. The audited codebase carried 26 strategies across eight
modules (``experimental.py`` alone was 492 lines) with heavy overlap, and the
multiplication of untested variants is precisely what manufactures false
discoveries — each one adds to a multiple-testing burden nothing was
counting.

What is here is a *spanning set*: one clean representative of each major
family, so the structural scanner has something to recommend and the funnel
has something to falsify.

Adding a strategy is cheap. Adding one without an economic rationale is
blocked by :func:`exchange.strategies.base.register`.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from exchange.core.errors import ConfigError
from exchange.strategies.base import (
    Strategy,
    atr,
    ema,
    realised_volatility,
    register,
    rolling_zscore,
    rsi,
    sma,
)


class _SizedStrategy(Strategy):
    """Shared sizing: a unit position scaled by ``size``.

    Keeping sizing out of the signal logic means the signal can be tested for
    causality and structure independently of how much capital it moves.
    """

    def _scale(self, signal: pd.Series) -> pd.Series:
        return (signal * float(self.params.get("size", 1.0))).rename("target_position")


@register
class BuyAndHold(_SizedStrategy):
    """The benchmark every other strategy must beat.

    Included because it is the correct null for most single-instrument work: a
    "strategy" that is long 80% of the time in a rising market has not found
    an effect, it has found beta.
    """

    name = "buy_and_hold"
    family = "benchmark"
    rationale = (
        "Long-only exposure to the instrument's own drift. Not an edge — the benchmark against "
        "which any directional edge must be measured, so that beta is not mistaken for alpha."
    )
    defaults: ClassVar[dict] = {"size": 1.0}
    param_space: ClassVar[dict] = {}

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        return self._scale(pd.Series(1.0, index=bars.index))


@register
class MovingAverageCrossover(_SizedStrategy):
    """Classic dual moving-average trend following."""

    name = "ma_crossover"
    family = "trend"
    rationale = (
        "Slow diffusion of information and the behavioural under-reaction it produces make price "
        "trends persist over intermediate horizons. A moving-average crossover is the crudest "
        "possible estimator of that persistence, and its very crudeness makes it a fair test: if "
        "an instrument has exploitable trend persistence this finds some of it, and if this needs "
        "delicate tuning to work, the persistence is not there."
    )
    reference = "Moskowitz, Ooi & Pedersen (2012), 'Time Series Momentum', JFE 104(2)."
    defaults: ClassVar[dict] = {"fast": 20, "slow": 100, "size": 1.0, "allow_short": True}
    param_space: ClassVar[dict] = {"fast": [10, 20, 40], "slow": [60, 100, 200]}

    def validate(self) -> None:
        if self.params["fast"] >= self.params["slow"]:
            raise ConfigError("fast window must be shorter than slow")

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        fast = ema(bars["close"], self.params["fast"])
        slow = ema(bars["close"], self.params["slow"])
        signal = np.sign(fast - slow)
        if not self.params["allow_short"]:
            signal = signal.clip(lower=0.0)
        return self._scale(signal.fillna(0.0))


@register
class TimeSeriesMomentum(_SizedStrategy):
    """Hold the sign of the trailing return over a fixed lookback."""

    name = "ts_momentum"
    family = "trend"
    rationale = (
        "If returns are positively autocorrelated at horizon k, the sign of the trailing k-bar "
        "return predicts the next one. This is the most direct possible test of that single "
        "statistical claim, with no smoothing to obscure what is being measured — which makes it "
        "the natural strategy to pair with a variance-ratio diagnostic."
    )
    reference = "Moskowitz, Ooi & Pedersen (2012), JFE 104(2)."
    defaults: ClassVar[dict] = {"lookback": 60, "size": 1.0, "allow_short": True}
    param_space: ClassVar[dict] = {"lookback": [10, 20, 60, 120]}

    def validate(self) -> None:
        if self.params["lookback"] < 2:
            raise ConfigError("lookback must be >= 2")

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        signal = np.sign(bars["close"].pct_change(self.params["lookback"]))
        if not self.params["allow_short"]:
            signal = signal.clip(lower=0.0)
        return self._scale(signal.fillna(0.0))


@register
class ZScoreReversion(_SizedStrategy):
    """Fade deviations from a trailing mean."""

    name = "zscore_reversion"
    family = "mean_reversion"
    rationale = (
        "Liquidity demand from non-informational traders pushes price away from fair value; market "
        "makers absorb the imbalance and are compensated as it decays. The trailing z-score "
        "measures the deviation and the position is the liquidity provision. This should work "
        "where the variance ratio is below one and the estimated half-life is short relative to "
        "the holding period, and nowhere else."
    )
    reference = "Nagel (2012), 'Evaporating Liquidity', RFS 25(7)."
    defaults: ClassVar[dict] = {
        "window": 20, "entry_z": 1.5, "exit_z": 0.5, "size": 1.0, "allow_short": True,
    }
    param_space: ClassVar[dict] = {"window": [10, 20, 40], "entry_z": [1.0, 1.5, 2.0]}

    def validate(self) -> None:
        if self.params["window"] < 5:
            raise ConfigError("window must be >= 5")
        if self.params["exit_z"] >= self.params["entry_z"]:
            raise ConfigError("exit_z must be below entry_z")

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        z = rolling_zscore(bars["close"], self.params["window"])
        entry, exit_ = self.params["entry_z"], self.params["exit_z"]
        # A state machine, not a threshold: without hysteresis the position
        # flickers around the entry level and pays the spread every bar.
        position = np.zeros(len(z))
        values = z.to_numpy()
        state = 0.0
        for i, value in enumerate(values):
            if np.isfinite(value):
                if state == 0.0:
                    if value <= -entry:
                        state = 1.0
                    elif value >= entry and self.params["allow_short"]:
                        state = -1.0
                elif (state > 0 and value >= -exit_) or (state < 0 and value <= exit_):
                    state = 0.0
            position[i] = state
        return self._scale(pd.Series(position, index=bars.index))


@register
class RSIReversion(_SizedStrategy):
    """Buy oversold, sell overbought, on a trailing RSI."""

    name = "rsi_reversion"
    family = "mean_reversion"
    rationale = (
        "A second, differently-parameterised estimator of the same short-horizon reversion "
        "hypothesis as the z-score rule. Its purpose in this library is cross-validation of a "
        "mechanism: an effect that shows up under one estimator and vanishes under an equally "
        "reasonable alternative is an artefact of the estimator, not a property of the market."
    )
    defaults: ClassVar[dict] = {
        "window": 14, "low": 30.0, "high": 70.0, "size": 1.0, "allow_short": True,
    }
    param_space: ClassVar[dict] = {"window": [7, 14, 21], "low": [20.0, 30.0]}

    def validate(self) -> None:
        if not 0 < self.params["low"] < self.params["high"] < 100:
            raise ConfigError("require 0 < low < high < 100")

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        value = rsi(bars["close"], self.params["window"])
        long = (value <= self.params["low"]).astype(float)
        short = (value >= self.params["high"]).astype(float) if self.params["allow_short"] else 0.0
        return self._scale(long - short)


@register
class BreakoutChannel(_SizedStrategy):
    """Donchian channel breakout."""

    name = "breakout"
    family = "trend"
    rationale = (
        "A move beyond the recent extreme is informative when it reflects genuine order-flow "
        "imbalance rather than noise, which is the classic account of why breakout systems worked "
        "in commodity futures. The channel is computed from bars strictly before the current one, "
        "so touching the level is a real, tradeable event rather than a retrospective observation."
    )
    defaults: ClassVar[dict] = {"window": 55, "exit_window": 20, "size": 1.0, "allow_short": True}
    param_space: ClassVar[dict] = {"window": [20, 55, 100], "exit_window": [10, 20]}

    def validate(self) -> None:
        if self.params["exit_window"] >= self.params["window"]:
            raise ConfigError("exit_window must be shorter than window")

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        w, xw = self.params["window"], self.params["exit_window"]
        # shift(1): the channel must not include the bar being tested.
        upper = bars["high"].rolling(w, min_periods=w).max().shift(1)
        lower = bars["low"].rolling(w, min_periods=w).min().shift(1)
        exit_up = bars["high"].rolling(xw, min_periods=xw).max().shift(1)
        exit_dn = bars["low"].rolling(xw, min_periods=xw).min().shift(1)

        state = 0.0
        out = np.zeros(len(bars))
        c, u, low_, xu, xd = (
            s.to_numpy() for s in (bars["close"], upper, lower, exit_up, exit_dn)
        )
        for i in range(len(bars)):
            if state == 0.0:
                if np.isfinite(u[i]) and c[i] > u[i]:
                    state = 1.0
                elif np.isfinite(low_[i]) and c[i] < low_[i] and self.params["allow_short"]:
                    state = -1.0
            elif (state > 0 and np.isfinite(xd[i]) and c[i] < xd[i]) or (state < 0 and np.isfinite(xu[i]) and c[i] > xu[i]):
                state = 0.0
            out[i] = state
        return self._scale(pd.Series(out, index=bars.index))


@register
class VolatilityRegimeFilter(_SizedStrategy):
    """Trend following, active only in the low-volatility regime.

    A conditional strategy: the point is not the trend rule but the condition.
    The scanner reports volatility clustering; if it is high, this is the
    family it recommends, and this strategy is the test of whether that
    recommendation was worth anything.
    """

    name = "vol_regime_trend"
    family = "volatility"
    rationale = (
        "Volatility is strongly autocorrelated, so the current regime is forecastable even when "
        "direction is not. Trend-following works badly in high-volatility regimes, where mean "
        "reversion and gap risk dominate and transaction costs rise. Conditioning entry on a "
        "trailing volatility percentile tests whether that conditioning adds anything beyond the "
        "unconditional trend rule it wraps."
    )
    defaults: ClassVar[dict] = {
        "fast": 20, "slow": 100, "vol_window": 20, "vol_percentile": 0.7,
        "lookback": 250, "size": 1.0,
    }
    param_space: ClassVar[dict] = {"vol_percentile": [0.5, 0.7, 0.9], "slow": [60, 100]}

    def validate(self) -> None:
        if not 0 < self.params["vol_percentile"] <= 1:
            raise ConfigError("vol_percentile must be in (0, 1]")
        if self.params["fast"] >= self.params["slow"]:
            raise ConfigError("fast must be shorter than slow")

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        fast = ema(bars["close"], self.params["fast"])
        slow = ema(bars["close"], self.params["slow"])
        trend = np.sign(fast - slow)

        vol = realised_volatility(bars["close"], self.params["vol_window"])
        # Trailing percentile rank: rank within the last `lookback` bars only,
        # and the window excludes nothing of the future by construction.
        rank = vol.rolling(
            self.params["lookback"], min_periods=self.params["vol_window"] * 2
        ).apply(lambda w: float((w[:-1] <= w[-1]).mean()) if len(w) > 1 else np.nan, raw=True)
        allowed = (rank <= self.params["vol_percentile"]).astype(float)
        return self._scale((trend * allowed).fillna(0.0))


@register
class SeasonalCalendar(_SizedStrategy):
    """Hold long during a fixed part of the calendar.

    Present because commodity seasonality is a real, mechanically grounded
    effect — and because it is the family most easily faked. With 12 months
    and two directions there are 24 obvious variants; the funnel counts every
    one of them, which is the whole point.
    """

    name = "seasonal_month"
    family = "seasonal"
    rationale = (
        "Physical commodities have genuine calendar structure: natural gas demand follows heating "
        "and cooling degree days, storage fills and draws on a fixed annual cycle, and refinery "
        "maintenance is scheduled. A month-of-year rule is the simplest expression of that. It is "
        "also the family with the worst multiple-testing hygiene in the literature, so it is "
        "included specifically to be tested against the number of calendar variants available."
    )
    defaults: ClassVar[dict] = {"long_months": (11, 12, 1), "short_months": (), "size": 1.0}
    param_space: ClassVar[dict] = {
        "long_months": [(11, 12, 1), (12, 1, 2), (4, 5, 6), (6, 7, 8), (9, 10, 11)],
    }

    def validate(self) -> None:
        for key in ("long_months", "short_months"):
            months = self.params[key]
            if any(not 1 <= int(m) <= 12 for m in months):
                raise ConfigError(f"{key} must contain months in 1..12")
        overlap = set(self.params["long_months"]) & set(self.params["short_months"])
        if overlap:
            raise ConfigError(f"months {sorted(overlap)} are both long and short")

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        month = np.asarray(bars.index.month)
        long = np.isin(month, list(self.params["long_months"])).astype(float)
        short = np.isin(month, list(self.params["short_months"])).astype(float)
        return self._scale(pd.Series(long - short, index=bars.index))


@register
class VolatilityTargetedMomentum(Strategy):
    """Time-series momentum sized to a constant volatility target.

    The realistic version of :class:`TimeSeriesMomentum`: risk is held
    constant instead of contract count, which is how the effect is actually
    traded and how it must be measured if results across regimes are to be
    comparable.
    """

    name = "vol_target_momentum"
    family = "trend"
    rationale = (
        "Time-series momentum sized to constant risk. Holding contract count fixed makes measured "
        "performance a function of whichever volatility regime the sample contained, which is a "
        "large part of why momentum backtests fail to replicate out of sample. Targeting "
        "volatility removes that confound and makes in-sample and out-of-sample periods "
        "comparable in the only way that matters."
    )
    reference = (
        "Moskowitz, Ooi & Pedersen (2012); Harvey et al. (2018), "
        "'The Impact of Volatility Targeting'."
    )
    defaults: ClassVar[dict] = {
        "lookback": 60, "vol_window": 20, "target_vol": 0.15, "bars_per_year": 252.0,
        "capital": 100_000.0, "multiplier": 1.0, "max_leverage": 3.0, "allow_short": True,
    }
    param_space: ClassVar[dict] = {"lookback": [20, 60, 120], "target_vol": [0.10, 0.15, 0.20]}

    def validate(self) -> None:
        if self.params["target_vol"] <= 0:
            raise ConfigError("target_vol must be positive")
        if self.params["bars_per_year"] <= 0:
            raise ConfigError("bars_per_year must be positive")

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        # `bars_per_year` is a parameter rather than something inferred from
        # the index, and the distinction matters. Inferring it from the full
        # sample makes the strategy's own output depend on how much data
        # follows the bar being sized — which the causality guard correctly
        # flags as non-causal. A trading calendar is a property of the
        # instrument, known in advance, so supplying it from outside is both
        # honest and reproducible; the runner passes the dataset's value.
        from exchange.strategies.base import volatility_target_size

        size = volatility_target_size(
            bars,
            target_annual_vol=self.params["target_vol"],
            bars_per_year=self.params["bars_per_year"],
            capital=self.params["capital"],
            multiplier=self.params["multiplier"],
            window=self.params["vol_window"],
            max_leverage=self.params["max_leverage"],
        )
        signal = np.sign(bars["close"].pct_change(self.params["lookback"]))
        if not self.params["allow_short"]:
            signal = signal.clip(lower=0.0)
        return (signal.fillna(0.0) * size).rename("target_position")


@register
class ATRBreakoutWithStop(_SizedStrategy):
    """Breakout entry with a volatility-scaled trailing stop.

    Exists so exit mechanics are represented in the library at all: exit rules
    change a strategy's return distribution far more than entry rules do, and
    a library with only entry logic systematically understates how much of a
    result depends on choices nobody wrote down.
    """

    name = "atr_breakout"
    family = "trend"
    rationale = (
        "Entries on a channel breakout, exits on a trailing stop set a fixed number of average "
        "true ranges below the running high. Scaling the stop by realised volatility keeps the "
        "probability of a noise-driven exit roughly constant across regimes, which is what "
        "separates a stop that manages risk from one that simply harvests volatility."
    )
    defaults: ClassVar[dict] = {
        "window": 40, "atr_window": 20, "atr_multiple": 3.0, "size": 1.0,
    }
    param_space: ClassVar[dict] = {"window": [20, 40, 80], "atr_multiple": [2.0, 3.0, 4.0]}

    def validate(self) -> None:
        if self.params["atr_multiple"] <= 0:
            raise ConfigError("atr_multiple must be positive")

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        w = self.params["window"]
        upper = bars["high"].rolling(w, min_periods=w).max().shift(1)
        band = atr(bars, self.params["atr_window"]).shift(1) * self.params["atr_multiple"]

        close = bars["close"].to_numpy()
        up = upper.to_numpy()
        stop_width = band.to_numpy()
        out = np.zeros(len(bars))
        state, peak = 0.0, np.nan
        for i in range(len(bars)):
            if state == 0.0:
                if np.isfinite(up[i]) and close[i] > up[i]:
                    state, peak = 1.0, close[i]
            else:
                peak = max(peak, close[i])
                if np.isfinite(stop_width[i]) and close[i] < peak - stop_width[i]:
                    state, peak = 0.0, np.nan
            out[i] = state
        return self._scale(pd.Series(out, index=bars.index))


@register
class MeanReversionInTrendFilter(_SizedStrategy):
    """Buy short-term dips while the long-term trend is up.

    A conditional mean-reversion rule — the family the scanner recommends when
    reversion and trend evidence point in opposite directions at different
    horizons, which is the common case in commodities.
    """

    name = "conditional_reversion"
    family = "mean_reversion"
    rationale = (
        "Short-horizon reversion and long-horizon trend routinely coexist: liquidity effects "
        "dominate over days while information diffusion dominates over months. Buying "
        "short-horizon weakness only while the long-horizon trend is positive tests that joint "
        "structure directly, and it is falsifiable in a specific way — if the long filter adds "
        "nothing, the conditioning is decoration and the unconditional rule should match it."
    )
    defaults: ClassVar[dict] = {
        "trend_window": 200, "z_window": 10, "entry_z": -1.0, "size": 1.0,
    }
    param_space: ClassVar[dict] = {
        "trend_window": [100, 200], "z_window": [5, 10, 20], "entry_z": [-0.5, -1.0, -1.5],
    }

    def validate(self) -> None:
        if self.params["entry_z"] >= 0:
            raise ConfigError("entry_z must be negative (a dip)")

    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        trend_up = bars["close"] > sma(bars["close"], self.params["trend_window"])
        z = rolling_zscore(bars["close"], self.params["z_window"])
        entered = ((z <= self.params["entry_z"]) & trend_up).to_numpy()
        zz = z.to_numpy()

        state = 0.0
        out = np.zeros(len(bars))
        for i in range(len(bars)):
            if state == 0.0 and entered[i]:
                state = 1.0
            elif state > 0 and np.isfinite(zz[i]) and zz[i] >= 0.0:
                # Exit once the dip has been recovered.
                state = 0.0
            out[i] = state
        return self._scale(pd.Series(out, index=bars.index))


FAMILIES = ("trend", "mean_reversion", "seasonal", "volatility", "benchmark")
