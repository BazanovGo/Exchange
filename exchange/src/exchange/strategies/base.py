"""Strategy interface.

A strategy is a pure function from bars to a **target position**, in
contracts, indexed by bar and unshifted. The engine applies the execution
delay (see :mod:`exchange.backtest.engine`), so a strategy never shifts its
own output — forgetting that shift is the single most common way a lookahead
enters a backtest, and this contract removes the opportunity.

Three obligations, all enforced
--------------------------------
1. **Causal.** ``positions(bars[:k])`` must equal ``positions(bars)[:k]`` for
   every ``k``. :meth:`Strategy.check_causality` proves it on real data using
   :func:`exchange.data.temporal.assert_causal`; the test suite runs it
   against every registered strategy.
2. **Declared parameter space.** ``param_space`` names every parameter and
   its candidate values. The funnel counts the grid size as the number of
   hypotheses tested — a strategy cannot hide the size of its own search.
3. **Stated economic rationale.** ``rationale`` is required, non-trivial
   text, and :func:`register` refuses a strategy without one. A strategy with
   no story about *why* the effect should exist is a pattern found in noise,
   and the funnel records the rationale next to the result so the two can be
   judged together.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from exchange.core.errors import ConfigError
from exchange.data.schema import assert_canonical


@dataclass(frozen=True)
class StrategyMeta:
    """Descriptive metadata carried with every result."""

    name: str
    family: str
    rationale: str
    reference: str = ""
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "family": self.family,
            "rationale": self.rationale,
            "reference": self.reference,
            "tags": list(self.tags),
        }


class Strategy(ABC):
    """Base class for every strategy."""

    name: ClassVar[str] = ""
    family: ClassVar[str] = ""          # trend / mean_reversion / seasonal / volatility / carry
    rationale: ClassVar[str] = ""       # WHY this should work, in economic terms
    reference: ClassVar[str] = ""
    tags: ClassVar[tuple[str, ...]] = ()
    defaults: ClassVar[dict[str, Any]] = {}
    param_space: ClassVar[dict[str, Sequence[Any]]] = {}

    def __init__(self, **params: Any) -> None:
        unknown = set(params) - set(self.defaults)
        if unknown:
            raise ConfigError(
                f"{type(self).__name__}: unknown parameter(s) {sorted(unknown)}; "
                f"accepted: {sorted(self.defaults)}"
            )
        self.params: dict[str, Any] = {**self.defaults, **params}
        self.validate()

    def validate(self) -> None:  # noqa: B027 - optional hook; not every strategy has invariants
        """Override to reject impossible parameter combinations."""

    # -- the one method a subclass must write ---------------------------

    @abstractmethod
    def target_position(self, bars: pd.DataFrame) -> pd.Series:
        """Desired holding in contracts after seeing each bar's close.

        Must be causal: the value at bar ``t`` may depend on bars ``0..t``
        only.
        """

    # -- public API ------------------------------------------------------

    def positions(self, bars: pd.DataFrame) -> pd.Series:
        """Validated target positions for ``bars``."""
        assert_canonical(bars, name=self.name or type(self).__name__)
        target = self.target_position(bars)
        if not isinstance(target, pd.Series):
            raise ConfigError(f"{self.name}: target_position must return a Series")
        if not target.index.equals(bars.index):
            raise ConfigError(f"{self.name}: target_position index does not match the bar index")
        return target.astype(float).fillna(0.0).rename("target_position")

    @property
    def meta(self) -> StrategyMeta:
        return StrategyMeta(
            name=self.name or type(self).__name__,
            family=self.family,
            rationale=self.rationale,
            reference=self.reference,
            tags=tuple(self.tags),
        )

    def check_causality(self, bars: pd.DataFrame) -> None:
        """Prove on real data that this strategy cannot see the future."""
        from exchange.data.temporal import assert_causal

        assert_causal(self.positions, bars, name=f"{self.name}.positions")

    def describe(self) -> dict:
        return {**self.meta.to_dict(), "params": dict(self.params)}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        args = ", ".join(f"{k}={v!r}" for k, v in sorted(self.params.items()))
        return f"{type(self).__name__}({args})"


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    """Class decorator: register a strategy under its ``name``.

    Registration enforces the two obligations that cannot be checked at call
    time: a name, and a non-trivial economic rationale.
    """
    if not cls.name:
        raise ConfigError(f"{cls.__name__} must define a non-empty name")
    if len(cls.rationale.strip()) < 30:
        raise ConfigError(
            f"{cls.__name__} must state an economic rationale of at least 30 characters. "
            "A strategy with no mechanism is a pattern, not a hypothesis."
        )
    key = cls.name.lower()
    if key in _REGISTRY and _REGISTRY[key] is not cls:
        raise ConfigError(f"strategy name {cls.name!r} is already registered")
    _REGISTRY[key] = cls
    return cls


def available() -> list[str]:
    return sorted(_REGISTRY)


def strategy_class(name: str) -> type[Strategy]:
    key = str(name).lower()
    if key not in _REGISTRY:
        raise ConfigError(f"unknown strategy {name!r}; available: {available()}")
    return _REGISTRY[key]


def build(name: str, **params: Any) -> Strategy:
    """Instantiate a registered strategy."""
    return strategy_class(name)(**params)


def catalog() -> pd.DataFrame:
    """Overview of every registered strategy, including its search size."""
    rows = []
    for name in available():
        cls = _REGISTRY[name]
        rows.append(
            {
                "name": name,
                "family": cls.family,
                "n_params": len(cls.defaults),
                "grid_size": grid_size(cls.param_space),
                "rationale": cls.rationale,
            }
        )
    return pd.DataFrame(rows).set_index("name")


# --------------------------------------------------------------------------
# Parameter grids
# --------------------------------------------------------------------------


def grid_size(space: Mapping[str, Sequence[Any]]) -> int:
    """Number of combinations in a parameter space.

    This *is* the number of hypotheses a sweep tests, and it feeds directly
    into the Deflated Sharpe Ratio and the multiple-testing correction.
    """
    total = 1
    for values in space.values():
        total *= max(len(list(values)), 1)
    return total if space else 1


def iter_grid(space: Mapping[str, Sequence[Any]]) -> Iterator[dict[str, Any]]:
    """Every combination in a parameter space, in deterministic order."""
    if not space:
        yield {}
        return
    keys = sorted(space)
    for values in product(*(list(space[k]) for k in keys)):
        yield dict(zip(keys, values, strict=True))


def valid_combinations(cls: type[Strategy], space: Mapping[str, Sequence[Any]] | None = None):
    """Instantiable parameter combinations, skipping ones ``validate`` rejects."""
    out = []
    for params in iter_grid(space if space is not None else cls.param_space):
        try:
            cls(**params)
        except ConfigError:
            continue
        out.append(params)
    return out


# --------------------------------------------------------------------------
# Shared causal indicators
#
# Every function here is strictly backward-looking. They live in the strategy
# module rather than a general "indicators" package on purpose: an indicator
# is only meaningful together with the causality contract it is used under,
# and a free-floating indicator library is precisely where non-causal helpers
# (centred windows, full-sample z-scores) accumulate.
# --------------------------------------------------------------------------


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Z-score against a *trailing* window.

    The full-sample alternative (``(x - x.mean()) / x.std()``) is the most
    common lookahead in published strategy code: it standardises today using
    tomorrow's mean.
    """
    mean = series.rolling(window, min_periods=window).mean()
    sd = series.rolling(window, min_periods=window).std(ddof=1)
    return (series - mean) / sd.replace(0.0, np.nan)


def true_range(bars: pd.DataFrame) -> pd.Series:
    prev_close = bars["close"].shift(1)
    ranges = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(bars: pd.DataFrame, window: int = 14) -> pd.Series:
    return true_range(bars).rolling(window, min_periods=window).mean()


def realised_volatility(series: pd.Series, window: int = 20) -> pd.Series:
    return series.pct_change().rolling(window, min_periods=window).std(ddof=1)


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)


def volatility_target_size(
    bars: pd.DataFrame,
    *,
    target_annual_vol: float,
    bars_per_year: float,
    capital: float,
    multiplier: float = 1.0,
    window: int = 20,
    max_leverage: float = 3.0,
) -> pd.Series:
    """Contracts needed for a constant volatility target.

    Sizing by volatility rather than by a fixed contract count is not a
    performance trick: it makes results across regimes comparable. A fixed
    size means a strategy's apparent edge is dominated by whichever regime
    happened to be most volatile, which then fails to replicate.

    All inputs are trailing, so the size for bar ``t`` uses volatility
    estimated through bar ``t`` only.
    """
    vol = realised_volatility(bars["close"], window) * np.sqrt(bars_per_year)
    price = bars["close"]
    notional = capital * target_annual_vol / vol.replace(0.0, np.nan)
    contracts = notional / (price * multiplier)
    cap = capital * max_leverage / (price * multiplier)
    return contracts.clip(upper=cap).fillna(0.0)
