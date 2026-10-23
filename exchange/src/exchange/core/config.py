"""Configuration.

One file describes a research programme: where the data comes from, which
execution scenario is the basis for verdicts, how the funnel splits the
sample, which attacks run, and where experiments are recorded. The CLI
reads it, every command takes ``--config``, and the resolved configuration
is **snapshotted into each experiment record**, because a result produced
under settings nobody wrote down is not reproducible.

Three rules this module enforces
---------------------------------
1. **Unknown keys are errors.** A silently ignored key is a setting the
   researcher believes is active and is not — the kind of mistake that
   makes two runs differ for reasons nobody can find.
2. **The seed is always set.** There is no "random" mode. An unseeded
   research run cannot be reproduced, and an irreproducible result is not
   a result.
3. **Verdicts come from one declared scenario.** ``execution_scenario``
   names it; the stress ladder is run in addition, never instead, so a
   verdict can never quietly rest on the optimistic assumptions.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

from exchange.core.determinism import DEFAULT_SEED
from exchange.core.errors import ConfigError
from exchange.core.hashing import hash_config
from exchange.execution.assumptions import SCENARIOS
from exchange.funnel.pipeline import FunnelConfig

DEFAULT_CONFIG_PATH = Path("configs/research.yaml")


@dataclass(frozen=True)
class DataConfig:
    """Where bars come from and how strictly they are gated."""

    provider: dict = field(default_factory=lambda: {"kind": "csv", "directory": "data"})
    symbols: tuple[str, ...] = ()
    timeframe: str = "1d"
    strict: bool = True                 # refuse data with quality errors
    roll: str = "volume"                # continuous-futures roll rule
    adjustment: str = "ratio"           # none / backward / ratio
    max_warnings: int = 50

    def __post_init__(self) -> None:
        if self.adjustment not in {"none", "backward", "ratio"}:
            raise ConfigError(f"unknown adjustment {self.adjustment!r}")


@dataclass(frozen=True)
class FeatureConfig:
    """Fundamental features, off by default."""

    provider: dict = field(default_factory=dict)
    keys: tuple[str, ...] = ()
    strict: bool = True                 # refuse feature columns that fail the leakage scan

    @property
    def enabled(self) -> bool:
        return bool(self.provider) and bool(self.keys)


@dataclass(frozen=True)
class FalsificationConfig:
    """Which attacks run after the funnel."""

    enabled: bool = True
    attacks: tuple[str, ...] = ()       # empty = every attack
    n_resamples: int = 1000

    def __post_init__(self) -> None:
        from exchange.falsification.engine import ATTACK_NAMES

        unknown = [a for a in self.attacks if a not in ATTACK_NAMES]
        if unknown:
            raise ConfigError(f"unknown attack(s) {unknown}; available: {list(ATTACK_NAMES)}")


@dataclass(frozen=True)
class ResearchConfig:
    """The whole programme's settings."""

    seed: int = DEFAULT_SEED
    deterministic: bool = True
    log_level: str = "INFO"
    execution_scenario: str = "base"
    data: DataConfig = field(default_factory=DataConfig)
    funnel: FunnelConfig = field(default_factory=FunnelConfig)
    falsification: FalsificationConfig = field(default_factory=FalsificationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    registry_root: str = "experiments"
    reports_root: str = "reports"

    def __post_init__(self) -> None:
        if self.execution_scenario not in SCENARIOS:
            raise ConfigError(
                f"unknown execution scenario {self.execution_scenario!r}; "
                f"known: {sorted(SCENARIOS)}"
            )
        if self.execution_scenario in {"optimistic", "frictionless"}:
            raise ConfigError(
                f"{self.execution_scenario!r} cannot be the verdict basis: it assumes costs no "
                "live book obtains. Use 'base' or 'conservative'; the optimistic scenario is "
                "reported by the stress ladder either way."
            )
        # The funnel owns the seed during a run, so keep the two in step
        # rather than letting a config file set two different ones.
        if self.funnel.seed != self.seed:
            object.__setattr__(self, "funnel", replace(self.funnel, seed=self.seed))
        if self.funnel.execution_scenario != self.execution_scenario:
            object.__setattr__(
                self, "funnel", replace(self.funnel, execution_scenario=self.execution_scenario)
            )

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "deterministic": self.deterministic,
            "log_level": self.log_level,
            "execution_scenario": self.execution_scenario,
            "data": {**asdict(self.data), "symbols": list(self.data.symbols)},
            "funnel": self.funnel.to_dict(),
            "falsification": {
                "enabled": self.falsification.enabled,
                "attacks": list(self.falsification.attacks),
                "n_resamples": self.falsification.n_resamples,
            },
            "features": {
                "provider": dict(self.features.provider),
                "keys": list(self.features.keys),
                "strict": self.features.strict,
            },
            "registry_root": self.registry_root,
            "reports_root": self.reports_root,
        }

    @property
    def fingerprint(self) -> str:
        """Content hash of the whole configuration, for the experiment record."""
        return hash_config(self.to_dict())

    def to_yaml(self) -> str:
        import yaml

        return yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in {".yaml", ".yml"}:
            path.write_text(self.to_yaml(), encoding="utf-8")
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path

    def summary(self) -> str:
        return (
            f"seed={self.seed} scenario={self.execution_scenario} "
            f"is_fraction={self.funnel.is_fraction} embargo={self.funnel.embargo_bars} "
            f"attacks={'all' if not self.falsification.attacks else len(self.falsification.attacks)} "
            f"registry={self.registry_root} fingerprint={self.fingerprint[:12]}"
        )


# ==========================================================================
# Loading
# ==========================================================================

_SECTIONS = {
    "data": DataConfig,
    "funnel": FunnelConfig,
    "falsification": FalsificationConfig,
    "features": FeatureConfig,
}

_TUPLE_FIELDS = {"symbols", "keys", "attacks", "stress_ladder"}


def build_config(payload: Mapping[str, Any] | None = None) -> ResearchConfig:
    """Build a configuration from a plain mapping, rejecting unknown keys."""
    payload = dict(payload or {})
    top_level = {f.name for f in fields(ResearchConfig)}
    unknown = set(payload) - top_level
    if unknown:
        raise ConfigError(
            f"unknown configuration key(s) {sorted(unknown)}; known: {sorted(top_level)}"
        )

    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        if key in _SECTIONS:
            kwargs[key] = _build_section(key, _SECTIONS[key], value)
        else:
            kwargs[key] = value
    return ResearchConfig(**kwargs)


def _build_section(name: str, cls, value) -> Any:
    if isinstance(value, cls):
        return value
    if value is None:
        return cls()
    if not isinstance(value, Mapping):
        raise ConfigError(f"configuration section {name!r} must be a mapping")
    known = {f.name for f in fields(cls)}
    unknown = set(value) - known
    if unknown:
        raise ConfigError(
            f"unknown key(s) {sorted(unknown)} in section {name!r}; known: {sorted(known)}"
        )
    payload = {
        k: (tuple(v) if k in _TUPLE_FIELDS and v is not None else v) for k, v in value.items()
    }
    return cls(**payload)


def load_config(path: str | Path | None = None, *, overrides: Mapping | None = None) -> ResearchConfig:
    """Load a configuration file, or the defaults when there is none.

    A missing *explicit* path is an error; a missing default path is not,
    because the defaults are a complete, usable configuration and a first
    run should not require a file.
    """
    if path is None:
        candidate = DEFAULT_CONFIG_PATH
        payload = _read(candidate) if candidate.is_file() else {}
    else:
        candidate = Path(path)
        if not candidate.is_file():
            raise ConfigError(f"configuration file {candidate} does not exist")
        payload = _read(candidate)

    if overrides:
        payload = _merge(payload, overrides)
    config = build_config(payload)
    return config


def _read(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text or "{}")
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return dict(data)


def _merge(base: Mapping, overrides: Mapping) -> dict:
    """Shallow-merge per section, so ``--seed`` does not erase a data block."""
    out = {k: (dict(v) if isinstance(v, Mapping) else v) for k, v in base.items()}
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key].update({k: v for k, v in value.items() if v is not None})
        else:
            out[key] = value
    return out
