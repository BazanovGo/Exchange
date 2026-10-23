"""Feature providers and the point-in-time feature store.

Interfaces, not vendors
------------------------
:class:`FeatureProvider` is a two-method protocol. Implementing it against
EIA's API, a Bloomberg terminal, a vendor CSV dump or an internal warehouse
is a small amount of code in the *caller's* repository, and the research
code never changes. The package ships only providers that need no
credentials: in-memory, CSV directory, and a synthetic one used by the
controls.

This is a deliberate refusal of the usual arrangement, in which research
code imports a vendor client and the two become inseparable. Two things go
wrong when they do: the research cannot be reproduced by anyone without
that subscription, and it cannot be tested at all, because the test needs a
live API. Here every statistical claim in the test suite is made against
the synthetic provider, whose ground truth is known exactly.

A provider returns **observation-time** data
---------------------------------------------
A provider's job is to return the series indexed by the time each value
*describes*, and nothing else. The store attaches the publication lag from
the catalogue and converts it into a
:class:`~exchange.data.temporal.PointInTimeSeries`, so a provider cannot
accidentally create a lookahead, and a provider that already has true
point-in-time data can say so by returning an ``available_time`` column,
which the store then uses instead of the catalogue's conservative lag.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from exchange.core.determinism import derive_seed
from exchange.core.determinism import rng as make_rng
from exchange.core.errors import ConfigError, DataError, LookaheadError
from exchange.core.logging import get_logger
from exchange.data.temporal import (
    AVAILABLE_TIME,
    PointInTimeSeries,
    detect_column_leakage,
    make_point_in_time,
)
from exchange.features.catalogue import (
    CATALOGUE,
    DerivedFeature,
    FeatureSpec,
    feature_spec,
    features_for,
)

logger = get_logger(__name__)


@runtime_checkable
class FeatureProvider(Protocol):
    """Source of one or more fundamental series."""

    def keys(self) -> Sequence[str]:
        """Feature keys this provider can serve."""

    def fetch(
        self,
        key: str,
        *,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.Series | pd.DataFrame:
        """Observation-time data for ``key``.

        Returns a Series indexed by the time each value describes, in UTC.
        A provider with genuine point-in-time knowledge may return a frame
        with a ``value`` column and an ``available_time`` column instead,
        and the store will use those timestamps rather than the
        catalogue's assumed lag.
        """


# ==========================================================================
# Providers that need no credentials
# ==========================================================================


@dataclass
class InMemoryFeatureProvider:
    """Series held in memory. Used by tests and by the CLI's ``--features`` file."""

    data: dict[str, pd.Series | pd.DataFrame] = field(default_factory=dict)
    name: str = "memory"

    def keys(self) -> Sequence[str]:
        return sorted(self.data)

    def fetch(self, key, *, start=None, end=None):
        if key not in self.data:
            raise ConfigError(f"{self.name}: no feature {key!r}; have {self.keys()}")
        return _window(self.data[key], start, end)


@dataclass
class CsvFeatureProvider:
    """One CSV per feature in a directory, named ``<key>.csv``.

    The file must have a timestamp column and a value column. An optional
    ``available_time`` column is honoured, which is how a researcher
    records a real publication calendar instead of a constant lag.
    """

    directory: str | Path
    name: str = "csv"
    timestamp_column: str = "timestamp"
    value_column: str = "value"

    def keys(self) -> Sequence[str]:
        root = Path(self.directory)
        if not root.is_dir():
            return []
        return sorted(p.stem for p in root.glob("*.csv"))

    def fetch(self, key, *, start=None, end=None):
        path = Path(self.directory) / f"{key}.csv"
        if not path.is_file():
            raise ConfigError(f"{self.name}: {path} does not exist")
        frame = pd.read_csv(path)
        if self.timestamp_column not in frame.columns:
            raise DataError(f"{path} has no {self.timestamp_column!r} column")
        index = pd.to_datetime(frame[self.timestamp_column], utc=True)
        frame = frame.drop(columns=[self.timestamp_column]).set_index(index)
        frame.index.name = "observation_time"
        if AVAILABLE_TIME in frame.columns:
            frame[AVAILABLE_TIME] = pd.to_datetime(frame[AVAILABLE_TIME], utc=True)
            keep = [*[c for c in frame.columns if c != AVAILABLE_TIME][:1], AVAILABLE_TIME]
            return _window(frame[keep], start, end)
        if self.value_column in frame.columns:
            return _window(frame[self.value_column], start, end)
        return _window(frame.iloc[:, 0], start, end)


@dataclass
class SyntheticFeatureProvider:
    """Features with a *known* relationship to a price series.

    Three kinds, which is exactly what a control needs:

    ``informative``
        Correlated with the return ``horizon`` bars *ahead of its own
        availability*. A funnel that cannot find this has a bug.
    ``noise``
        Independent of the price. A funnel that finds an edge here is
        producing false discoveries, and that rate is measurable.
    ``leaky``
        Contains the **next** bar's return at full strength — a deliberately
        planted lookahead. The leakage detector must flag it, and the test
        suite fails if it does not. Available only when ``allow_leaky=True``,
        so it can never be built by accident. Note that the trap only
        survives alignment for a feature whose publication lag is zero: for
        a lagged feature the catalogue's own lag removes the leak, which is
        the mechanism working as intended.
    """

    bars: pd.DataFrame
    kinds: Mapping[str, str] = field(default_factory=dict)
    seed: int = 20260902
    information_coefficient: float = 0.08
    horizon: int = 1
    allow_leaky: bool = False
    name: str = "synthetic"

    def __post_init__(self) -> None:
        if "close" not in self.bars:
            raise DataError("SyntheticFeatureProvider needs a close column")
        bad = [k for k, v in self.kinds.items() if v not in {"informative", "noise", "leaky"}]
        if bad:
            raise ConfigError(f"unknown synthetic feature kind(s) for {bad}")
        leaky = [k for k, v in self.kinds.items() if v == "leaky"]
        if leaky and not self.allow_leaky:
            raise ConfigError(
                f"refusing to build deliberately leaky feature(s) {leaky} without allow_leaky=True"
            )

    def keys(self) -> Sequence[str]:
        return sorted(self.kinds)

    def fetch(self, key, *, start=None, end=None):
        kind = self.kinds.get(key)
        if kind is None:
            raise ConfigError(f"{self.name}: no feature {key!r}; have {self.keys()}")
        rng = make_rng(derive_seed(key, kind, base=self.seed))
        returns = self.bars["close"].pct_change().fillna(0.0)
        noise = pd.Series(rng.normal(size=len(returns)), index=returns.index)
        if kind == "noise":
            values = noise
        elif kind == "leaky":
            values = returns.shift(-self.horizon).fillna(0.0)  # planted on purpose
        else:
            target = returns.shift(-self.horizon).fillna(0.0)
            scale = target.std() or 1.0
            rho = self.information_coefficient
            values = rho * target / scale + np.sqrt(max(1.0 - rho**2, 0.0)) * noise
        return _window(values.rename(key), start, end)


_PROVIDERS: dict[str, Callable[..., FeatureProvider]] = {
    "memory": InMemoryFeatureProvider,
    "csv": CsvFeatureProvider,
    "synthetic": SyntheticFeatureProvider,
}


def register_feature_provider(name: str, factory: Callable[..., FeatureProvider]) -> None:
    """Make a provider available to the CLI and config files by name."""
    _PROVIDERS[str(name).lower()] = factory


def available_feature_providers() -> list[str]:
    return sorted(_PROVIDERS)


def build_feature_provider(spec: str | Mapping | FeatureProvider) -> FeatureProvider:
    if isinstance(spec, FeatureProvider):
        return spec
    if isinstance(spec, str):
        spec = {"kind": spec}
    payload = dict(spec)
    kind = str(payload.pop("kind", "")).lower()
    if kind not in _PROVIDERS:
        raise ConfigError(f"unknown feature provider {kind!r}; known: {available_feature_providers()}")
    return _PROVIDERS[kind](**payload)


# ==========================================================================
# The store
# ==========================================================================


@dataclass
class FeatureStore:
    """Turns provider output into leak-free, bar-aligned features.

    The alignment is a backward :func:`pandas.merge_asof` on
    ``available_time``, which is the only join that cannot see forward, and
    every alignment is followed by a leakage scan. A feature that predicts
    the next bar far better than it explains the current one is reported,
    and :meth:`align` refuses to return it unless explicitly told to.
    """

    provider: FeatureProvider
    catalogue: Mapping[str, FeatureSpec] = field(default_factory=lambda: dict(CATALOGUE))
    decision_lag: pd.Timedelta = field(default_factory=lambda: pd.Timedelta(0))

    # -- single series ---------------------------------------------------

    def point_in_time(
        self,
        key: str,
        *,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> PointInTimeSeries:
        """Fetch one feature and attach its availability timestamps."""
        spec = self.catalogue.get(key) or feature_spec(key)
        raw = self.provider.fetch(key, start=start, end=end)

        if isinstance(raw, pd.DataFrame) and AVAILABLE_TIME in raw.columns:
            frame = raw.copy()
            if frame.index.tz is None:
                raise DataError(f"{key}: observation index must be UTC-aware")
            series = PointInTimeSeries(
                frame=frame, name=key, revision_policy=spec.revision_policy
            )
            logger.debug("%s: provider supplied true point-in-time timestamps", key)
            return series

        values = raw if isinstance(raw, pd.Series) else raw.iloc[:, 0]
        return make_point_in_time(
            values.rename(key),
            publication_lag=spec.publication_lag,
            name=key,
            revision_policy=spec.revision_policy,
        )

    # -- many, aligned to bars -------------------------------------------

    def align(
        self,
        bars: pd.DataFrame,
        keys: Iterable[str],
        *,
        derived: Sequence[DerivedFeature] = (),
        strict: bool = True,
    ) -> pd.DataFrame:
        """Feature matrix on the bar index, with no value from the future.

        ``strict`` (the default) raises if the leakage scan flags any
        column. Turning it off is for *studying* a leak — the synthetic
        leaky control does exactly that — and the funnel never does.
        """
        index = bars.index
        if index.tz is None:
            raise DataError("align requires a UTC-aware bar index")

        columns: dict[str, pd.Series] = {}
        staleness: dict[str, pd.Series] = {}
        for key in keys:
            pit = self.point_in_time(key, end=index[-1])
            aligned = pit.align_to(index, decision_lag=self.decision_lag)
            value_columns = [
                c for c in aligned.columns
                if not c.endswith("observation_time") and not c.endswith("available_time")
            ]
            for column in value_columns:
                name = key if len(value_columns) == 1 else f"{key}_{column}"
                columns[name] = aligned[column]
            stale = [c for c in aligned.columns if c.endswith("staleness")]
            if stale:
                staleness[f"{key}_staleness"] = aligned[stale[0]]

        frame = pd.DataFrame(columns, index=index)
        for feature in derived:
            frame[feature.key] = feature.compute(frame)
        frame = frame.join(pd.DataFrame(staleness, index=index)) if staleness else frame

        findings = self.leakage_report(bars, frame)
        leaks = [f for f in findings if f.verdict == "LEAK"]
        if leaks:
            message = "; ".join(
                f"{f.column} (corr with next bar {f.corr_next:+.3f} vs current "
                f"{f.corr_current:+.3f}, z={f.z_next:.1f})"
                for f in leaks
            )
            if strict:
                raise LookaheadError(
                    "feature column(s) contain information from the future: " + message
                )
            logger.warning("leaky feature column(s) returned because strict=False: %s", message)
        return frame

    def leakage_report(self, bars: pd.DataFrame, features: pd.DataFrame) -> list:
        """Run the column-leakage scan over an aligned feature frame."""
        if features.empty:
            return []
        combined = bars.join(features, how="left")
        return detect_column_leakage(combined, columns=list(features.columns))

    # -- description -----------------------------------------------------

    def describe(self, keys: Iterable[str] | None = None) -> pd.DataFrame:
        selected = list(keys) if keys is not None else list(self.provider.keys())
        rows = []
        for key in selected:
            spec = self.catalogue.get(key)
            rows.append(
                spec.to_dict()
                if spec
                else {"key": key, "name": key, "market": "", "category": "",
                      "frequency": "", "publication_lag_days": float("nan"),
                      "unit": "", "source": "provider-defined",
                      "revision_policy": "", "expected_sign": 0,
                      "mechanism": "", "caveat": "not in the catalogue", "tags": []}
            )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("key")

    def coverage(self, market: str) -> pd.DataFrame:
        """Which catalogued features this provider can actually serve."""
        have = set(self.provider.keys())
        rows = [
            {"key": s.key, "name": s.name, "category": s.category, "available": s.key in have}
            for s in features_for(market)
        ]
        return pd.DataFrame(rows).set_index("key")


def _window(data, start, end):
    out = data
    if start is not None:
        out = out.loc[out.index >= pd.Timestamp(start)]
    if end is not None:
        out = out.loc[out.index <= pd.Timestamp(end)]
    return out
