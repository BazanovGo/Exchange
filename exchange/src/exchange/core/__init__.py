"""Cross-cutting primitives: determinism, timeframes, hashing, environment."""

from __future__ import annotations

from exchange.core.determinism import (
    DEFAULT_SEED,
    DeterminismReport,
    derive_seed,
    deterministic_mode,
    rng,
    spawn,
    temporary_seed,
)
from exchange.core.env import EnvironmentInfo, GitInfo, capture_environment, check_environment
from exchange.core.errors import (
    ConfigError,
    DataError,
    DataQualityError,
    ExchangeError,
    ExecutionError,
    LookaheadError,
    NotEnoughDataError,
    ReproducibilityError,
)
from exchange.core.hashing import canonical_json, hash_config, hash_dataframe, hash_file, short
from exchange.core.logging import configure_logging, get_logger
from exchange.core.timeframe import (
    Timeframe,
    annualisation_factor,
    empirical_bars_per_year,
    infer_timeframe,
)

__all__ = [
    "DEFAULT_SEED",
    "ConfigError",
    "DataError",
    "DataQualityError",
    "DeterminismReport",
    "EnvironmentInfo",
    "ExchangeError",
    "ExecutionError",
    "GitInfo",
    "LookaheadError",
    "NotEnoughDataError",
    "ReproducibilityError",
    "Timeframe",
    "annualisation_factor",
    "canonical_json",
    "capture_environment",
    "check_environment",
    "configure_logging",
    "derive_seed",
    "deterministic_mode",
    "empirical_bars_per_year",
    "get_logger",
    "hash_config",
    "hash_dataframe",
    "hash_file",
    "infer_timeframe",
    "rng",
    "short",
    "spawn",
    "temporary_seed",
]
