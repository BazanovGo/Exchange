"""Unified strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pandas as pd

from src.data.base import MarketData

SignalArray = pd.Series | pd.DataFrame


@dataclass
class StrategySignals:
    """Standard output of every strategy.

    ``entries``/``exits`` drive the long side, ``short_entries``/
    ``short_exits`` the short side. Long-only strategies leave the short
    legs as all-``False`` (never ``None``), so downstream code can pass the
    object to vectorbt without branching.
    """

    entries: SignalArray
    exits: SignalArray
    short_entries: SignalArray
    short_exits: SignalArray
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        for name in ("entries", "exits", "short_entries", "short_exits"):
            leg = getattr(self, name)
            if not leg.index.equals(self.entries.index):
                raise ValueError(f"{name} index differs from entries index")
            setattr(self, name, leg.fillna(False).astype(bool))

    @property
    def has_shorts(self) -> bool:
        return bool(self.short_entries.to_numpy().any() or self.short_exits.to_numpy().any())

    def stats(self) -> pd.Series:
        """Signal counts — a quick sanity check before backtesting."""
        return pd.Series(
            {
                "entries": int(self.entries.to_numpy().sum()),
                "exits": int(self.exits.to_numpy().sum()),
                "short_entries": int(self.short_entries.to_numpy().sum()),
                "short_exits": int(self.short_exits.to_numpy().sum()),
            },
            name="signal_count",
        )


def _false_like(ref: SignalArray) -> SignalArray:
    if isinstance(ref, pd.DataFrame):
        return pd.DataFrame(False, index=ref.index, columns=ref.columns)
    return pd.Series(False, index=ref.index, name=ref.name)


class BaseStrategy(ABC):
    """Interface every strategy implements.

    Subclasses define ``name``, ``description`` and ``default_params`` as
    class attributes and implement :meth:`generate_signals`, which receives
    a canonical OHLCV frame and returns raw long/short signal legs.

    The public entry point is :meth:`signals` — it accepts either an OHLCV
    ``DataFrame`` or a :class:`MarketData` container, fills missing short
    legs and attaches params/description.
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    category: ClassVar[str] = ""  # trend / mean_reversion / momentum / hybrid
    default_params: ClassVar[dict[str, Any]] = {}
    # Default optimization grid: param name -> candidate values. Used by
    # src.optimization when no explicit grid is supplied, so every strategy
    # is optimizable out of the box.
    opt_grid: ClassVar[dict[str, list[Any]]] = {}

    def __init__(self, **params: Any) -> None:
        unknown = set(params) - set(self.default_params)
        if unknown:
            raise ValueError(
                f"{type(self).__name__}: unknown params {sorted(unknown)}; "
                f"accepted: {sorted(self.default_params)}"
            )
        self.params: dict[str, Any] = {**self.default_params, **params}
        self.validate_params()

    def validate_params(self) -> None:  # noqa: B027 - optional hook, not abstract
        """Override to enforce parameter invariants (raise ``ValueError``)."""

    @abstractmethod
    def generate_signals(self, ohlcv: pd.DataFrame) -> dict[str, SignalArray]:
        """Return a dict with at least ``entries`` and ``exits``.

        ``short_entries`` / ``short_exits`` are optional and default to
        all-``False``.
        """

    def signals(self, data: MarketData | pd.DataFrame, symbol: str | None = None) -> StrategySignals:
        """Produce the standardized :class:`StrategySignals` for ``data``."""
        ohlcv = data.get(symbol) if isinstance(data, MarketData) else data
        raw = self.generate_signals(ohlcv)
        if "entries" not in raw or "exits" not in raw:
            raise ValueError(f"{type(self).__name__}.generate_signals must return 'entries' and 'exits'")
        entries = raw["entries"]
        return StrategySignals(
            entries=entries,
            exits=raw["exits"],
            short_entries=raw.get("short_entries", _false_like(entries)),
            short_exits=raw.get("short_exits", _false_like(entries)),
            params=dict(self.params),
            description=self.description,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"


_STRATEGIES: dict[str, type[BaseStrategy]] = {}


def register_strategy(cls: type[BaseStrategy]) -> type[BaseStrategy]:
    """Class decorator: register a strategy under its ``name``."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} must define a non-empty name")
    _STRATEGIES[cls.name.lower()] = cls
    return cls


def available_strategies() -> list[str]:
    return sorted(_STRATEGIES)


def strategy_class(name: str) -> type[BaseStrategy]:
    """Return the registered strategy class (without instantiating it)."""
    key = name.lower()
    if key not in _STRATEGIES:
        raise KeyError(f"Unknown strategy {name!r}; available: {available_strategies()}")
    return _STRATEGIES[key]


def strategy_catalog() -> pd.DataFrame:
    """Overview table of all registered strategies."""
    rows = []
    for name in available_strategies():
        cls = _STRATEGIES[name]
        grid_size = 1
        for values in cls.opt_grid.values():
            grid_size *= len(values)
        rows.append(
            {
                "name": name,
                "category": cls.category,
                "n_params": len(cls.default_params),
                "grid_size": grid_size if cls.opt_grid else 0,
                "description": cls.description,
            }
        )
    return pd.DataFrame(rows).set_index("name")


def get_strategy(name: str, **params: Any) -> BaseStrategy:
    """Instantiate a registered strategy by name."""
    key = name.lower()
    if key not in _STRATEGIES:
        raise KeyError(f"Unknown strategy {name!r}; available: {available_strategies()}")
    return _STRATEGIES[key](**params)
