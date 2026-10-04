"""Data providers.

The core never imports a vendor SDK. Everything goes through
:class:`BarProvider`, so swapping Yahoo for a real futures feed is a
configuration change, not a refactor.

Providers return **raw** frames; normalisation and validation happen in
:func:`exchange.data.dataset.load_bars`, so a new provider cannot
accidentally opt out of the quality gate.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import pandas as pd

from exchange.core.errors import ConfigError, DataError
from exchange.core.logging import get_logger

logger = get_logger(__name__)


@runtime_checkable
class BarProvider(Protocol):
    """Anything that can hand back raw bars for a symbol."""

    name: str

    def fetch(self, symbol: str, *, timeframe: str = "1d", start=None, end=None, **kwargs) -> pd.DataFrame:
        """Return a raw frame for one symbol. Normalisation happens later."""
        ...

    def available(self) -> list[str]:
        """Symbols this provider can serve (may be empty if unknowable)."""
        ...


class _BaseProvider:
    name = "base"
    source_tz: str | None = None

    def available(self) -> list[str]:
        return []

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"


class InMemoryProvider(_BaseProvider):
    """Serves frames held in memory. The provider used by tests."""

    name = "memory"

    def __init__(self, frames: Mapping[str, pd.DataFrame], *, source_tz: str | None = None) -> None:
        self._frames = {str(k): v for k, v in frames.items()}
        self.source_tz = source_tz

    def fetch(self, symbol: str, *, timeframe: str = "1d", start=None, end=None, **kwargs) -> pd.DataFrame:
        if symbol not in self._frames:
            raise DataError(f"{self.name}: no data for {symbol!r}; have {sorted(self._frames)}")
        return self._frames[symbol].copy()

    def available(self) -> list[str]:
        return sorted(self._frames)


class _FileProvider(_BaseProvider):
    """Shared logic for one-file-per-symbol providers."""

    suffixes: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        base_dir: str | Path,
        *,
        source_tz: str | None = None,
        pattern: str = "{symbol}.csv",
        read_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.base_dir = Path(base_dir)
        self.source_tz = source_tz
        self.pattern = pattern
        self.read_kwargs = dict(read_kwargs or {})

    def _path(self, symbol: str, override: str | Path | None = None) -> Path:
        path = Path(override) if override else self.base_dir / self.pattern.format(symbol=symbol)
        if not path.exists():
            raise DataError(f"{self.name}: file not found for {symbol!r}: {path}")
        return path

    def available(self) -> list[str]:
        if not self.base_dir.exists():
            return []
        prefix, suffix = self.pattern.split("{symbol}")
        out = []
        for file in sorted(self.base_dir.glob(self.pattern.format(symbol="*"))):
            stem = file.name
            out.append(stem[len(prefix): len(stem) - len(suffix)] if suffix else stem[len(prefix):])
        return out


class CsvBarProvider(_FileProvider):
    """One CSV per symbol under ``base_dir``.

    ``source_tz`` declares the timezone of naive timestamps in the files. It
    has no default, because guessing it wrong shifts every intraday bar by
    hours and no downstream check can detect that.
    """

    name = "csv"

    def fetch(self, symbol: str, *, timeframe: str = "1d", start=None, end=None, path=None, **kwargs) -> pd.DataFrame:
        return pd.read_csv(self._path(symbol, path), **{**self.read_kwargs, **kwargs})


class ParquetBarProvider(_FileProvider):
    """One Parquet file per symbol under ``base_dir``."""

    name = "parquet"

    def __init__(
        self,
        base_dir: str | Path,
        *,
        source_tz: str | None = None,
        pattern: str = "{symbol}.parquet",
        read_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(base_dir, source_tz=source_tz, pattern=pattern, read_kwargs=read_kwargs)

    def fetch(self, symbol: str, *, timeframe: str = "1d", start=None, end=None, path=None, **kwargs) -> pd.DataFrame:
        return pd.read_parquet(self._path(symbol, path), **kwargs)


class YahooBarProvider(_BaseProvider):
    """Yahoo Finance, for equities and index proxies.

    Deliberately *not* recommended for futures research: Yahoo's continuous
    contracts (``NG=F``) are built with an undocumented roll and adjustment,
    which is precisely the provenance this package refuses to leave implicit.
    Use it for equity work or quick sanity checks.
    """

    name = "yahoo"

    _INTERVALS: ClassVar[dict[str, str]] = {
        "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "1d": "1d", "1w": "1wk",
    }

    def __init__(self, *, auto_adjust: bool = True) -> None:
        self.auto_adjust = auto_adjust
        self.source_tz = None  # yfinance returns tz-aware data

    def fetch(self, symbol: str, *, timeframe: str = "1d", start=None, end=None, **kwargs) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ConfigError("YahooBarProvider needs the 'yahoo' extra: pip install '.[yahoo]'") from exc

        interval = self._INTERVALS.get(timeframe.lower())
        if interval is None:
            raise ConfigError(f"Yahoo does not serve timeframe {timeframe!r}")
        logger.info("Yahoo download %s %s %s..%s", symbol, interval, start, end)
        frame = yf.download(
            symbol, start=start, end=end, interval=interval,
            auto_adjust=self.auto_adjust, progress=False, **kwargs,
        )
        if frame is None or frame.empty:
            raise DataError(f"Yahoo returned no data for {symbol!r}")
        return frame


_PROVIDERS: dict[str, type] = {
    "csv": CsvBarProvider,
    "parquet": ParquetBarProvider,
    "yahoo": YahooBarProvider,
}


def register_provider(name: str, cls: type) -> None:
    """Register a provider class under a config-visible name."""
    _PROVIDERS[name.lower()] = cls


def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


def build_provider(spec: str | Mapping[str, Any]) -> BarProvider:
    """Build a provider from config, e.g. ``{"name": "csv", "base_dir": "data/raw"}``."""
    if isinstance(spec, str):
        spec = {"name": spec}
    params = dict(spec)
    name = str(params.pop("name", "")).lower()
    if name not in _PROVIDERS:
        raise ConfigError(f"unknown provider {name!r}; known: {available_providers()}")
    return _PROVIDERS[name](**params)


def fetch_panel(
    provider: BarProvider,
    symbols: Iterable[str],
    *,
    timeframe: str = "1d",
    start=None,
    end=None,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Fetch several symbols, keeping raw frames keyed by symbol."""
    return {
        symbol: provider.fetch(symbol, timeframe=timeframe, start=start, end=end, **kwargs)
        for symbol in symbols
    }
