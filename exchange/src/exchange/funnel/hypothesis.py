"""Pre-registered hypotheses.

Why registration comes before the backtest
-------------------------------------------
The most damaging bias in quantitative research is not in any formula. It is
the sequence: run a search, find a good-looking equity curve, then invent a
market mechanism that explains it. The explanation always arrives, because a
plausible story can be told about any pattern. Once told, it converts a
data-mined artefact into something that feels like an insight, and no amount
of downstream statistics recovers from that.

A :class:`Hypothesis` must therefore be complete — statement, mechanism,
expected regime, and explicit falsification criteria — **before** any
backtest runs. The funnel refuses to start otherwise, and the record is
hashed at registration time so it cannot be quietly edited afterwards to
match the result.

A rejected hypothesis is a successful experiment
-------------------------------------------------
Rejection is recorded with equal weight to acceptance. It has real value: it
removes a candidate from the search space permanently, and the accrued record
of rejections is what makes the multiple-testing count honest for every later
hypothesis on the same data.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from exchange.core.errors import ConfigError
from exchange.core.hashing import hash_config


@dataclass(frozen=True)
class FalsificationCriterion:
    """One pre-committed condition that would kill the hypothesis.

    ``metric`` names a value the funnel produces; ``operator`` and
    ``threshold`` state the failing condition. Because these are written
    before the result exists, they cannot be tuned to whatever the result
    turned out to be.
    """

    metric: str
    operator: str       # "<", "<=", ">", ">=" — the condition that FALSIFIES
    threshold: float
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.operator not in {"<", "<=", ">", ">="}:
            raise ConfigError(f"unsupported operator {self.operator!r}")

    def is_falsified(self, value: float | None) -> bool | None:
        """``True`` if the criterion fires, ``None`` if the metric is absent."""
        if value is None or not math.isfinite(value):
            return None
        return {
            "<": value < self.threshold,
            "<=": value <= self.threshold,
            ">": value > self.threshold,
            ">=": value >= self.threshold,
        }[self.operator]

    def describe(self) -> str:
        return f"{self.metric} {self.operator} {self.threshold:g}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Hypothesis:
    """A complete, pre-registered research hypothesis."""

    hypothesis_id: str
    statement: str                 # what is claimed, testably
    rationale: str                 # why it should be true
    mechanism: str                 # the market process that would produce it
    expected_regime: str           # where it should work, and where it should not
    falsification: tuple[FalsificationCriterion, ...]
    strategy: str
    param_space: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    fixed_params: Mapping[str, Any] = field(default_factory=dict)
    symbol: str = ""
    timeframe: str = ""
    prior_hypotheses_on_data: int = 0   # configurations already tried on this dataset
    registered_at: str = ""
    author: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        problems = []
        if len(self.statement.strip()) < 20:
            problems.append("statement must be a testable claim of at least 20 characters")
        if len(self.mechanism.strip()) < 40:
            problems.append(
                "mechanism must describe the market process in at least 40 characters — "
                "'it works because the backtest is good' is not a mechanism"
            )
        if len(self.expected_regime.strip()) < 15:
            problems.append(
                "expected_regime must say where the effect should and should not appear; "
                "a hypothesis that predicts success everywhere predicts nothing"
            )
        if not self.falsification:
            problems.append(
                "at least one falsification criterion is required — a hypothesis that nothing "
                "could refute is not a hypothesis"
            )
        if not self.strategy:
            problems.append("strategy must name a registered strategy")
        if problems:
            raise ConfigError(
                f"hypothesis {self.hypothesis_id!r} is incomplete:\n  - " + "\n  - ".join(problems)
            )
        if not self.registered_at:
            object.__setattr__(self, "registered_at", datetime.now(UTC).isoformat(timespec="seconds"))

    @property
    def n_configurations(self) -> int:
        """Grid size: the number of hypotheses this single registration tests."""
        from exchange.strategies.base import grid_size

        return grid_size(self.param_space)

    @property
    def total_trials(self) -> int:
        """Configurations here plus everything already tried on this data.

        This is the number that goes into the Deflated Sharpe Ratio. It is
        larger than the grid size on purpose: the tenth hypothesis tested on
        a dataset carries the selection burden of the previous nine, whether
        or not anybody wants it to.
        """
        return self.n_configurations + int(self.prior_hypotheses_on_data)

    @property
    def fingerprint(self) -> str:
        """Hash of the registered content — proof it was not edited later."""
        return hash_config(
            {
                "statement": self.statement,
                "rationale": self.rationale,
                "mechanism": self.mechanism,
                "expected_regime": self.expected_regime,
                "falsification": [c.to_dict() for c in self.falsification],
                "strategy": self.strategy,
                "param_space": {k: list(v) for k, v in self.param_space.items()},
                "fixed_params": dict(self.fixed_params),
                "symbol": self.symbol,
                "timeframe": self.timeframe,
            }
        )

    def to_dict(self) -> dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "rationale": self.rationale,
            "mechanism": self.mechanism,
            "expected_regime": self.expected_regime,
            "falsification": [c.to_dict() for c in self.falsification],
            "strategy": self.strategy,
            "param_space": {k: list(v) for k, v in self.param_space.items()},
            "fixed_params": dict(self.fixed_params),
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "n_configurations": self.n_configurations,
            "prior_hypotheses_on_data": self.prior_hypotheses_on_data,
            "total_trials": self.total_trials,
            "registered_at": self.registered_at,
            "author": self.author,
            "tags": list(self.tags),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> Hypothesis:
        payload = dict(data)
        for derived in ("n_configurations", "total_trials", "fingerprint"):
            payload.pop(derived, None)
        payload["falsification"] = tuple(
            FalsificationCriterion(**c) for c in payload.get("falsification", [])
        )
        payload["param_space"] = {k: list(v) for k, v in payload.get("param_space", {}).items()}
        payload["tags"] = tuple(payload.get("tags", ()))
        known = {k: v for k, v in payload.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def to_markdown(self) -> str:
        lines = [
            f"## Hypothesis `{self.hypothesis_id}`",
            "",
            f"**Registered:** {self.registered_at}  ",
            f"**Fingerprint:** `{self.fingerprint}`",
            "",
            "### Statement", "", self.statement, "",
            "### Rationale", "", self.rationale, "",
            "### Expected market mechanism", "", self.mechanism, "",
            "### Expected regime", "", self.expected_regime, "",
            "### Falsification criteria", "",
            "The hypothesis is REJECTED if any of the following holds:", "",
        ]
        for c in self.falsification:
            lines.append(f"- `{c.describe()}`" + (f" — {c.rationale}" if c.rationale else ""))
        lines += [
            "",
            "### Search space", "",
            f"- Strategy: `{self.strategy}`",
            f"- Configurations in this hypothesis: **{self.n_configurations}**",
            f"- Prior configurations already tested on this data: **{self.prior_hypotheses_on_data}**",
            f"- Total trials charged to the multiple-testing budget: **{self.total_trials}**",
        ]
        return "\n".join(lines) + "\n"


# Criteria applied to every hypothesis unless the author overrides them. They
# encode the minimum a result must clear to be worth discussing at all.
DEFAULT_FALSIFICATION = (
    FalsificationCriterion(
        "oos_sharpe", "<", 0.0,
        rationale="A negative out-of-sample Sharpe under base costs falsifies the claim outright.",
    ),
    FalsificationCriterion(
        "pbo", ">", 0.5,
        rationale="If the selection procedure picks the OOS-worse configuration more often than "
                  "not, the search is choosing noise and no configuration from it can be trusted.",
    ),
    FalsificationCriterion(
        "deflated_sharpe_ratio", "<", 0.90,
        rationale="After deflating for the number of configurations tried, the probability of a "
                  "positive true Sharpe must still be high.",
    ),
    FalsificationCriterion(
        "n_oos_trades", "<", 30,
        rationale="Fewer than 30 out-of-sample trades cannot distinguish skill from luck at any "
                  "effect size this research programme would act on.",
    ),
    FalsificationCriterion(
        "stressed_sharpe", "<", -0.5,
        rationale="A strategy that collapses badly under stressed execution was trading the cost "
                  "assumption rather than the market.",
    ),
)


def make_hypothesis(
    hypothesis_id: str,
    *,
    statement: str,
    mechanism: str,
    expected_regime: str,
    strategy: str,
    rationale: str = "",
    falsification: Sequence[FalsificationCriterion] | None = None,
    include_defaults: bool = True,
    **kwargs,
) -> Hypothesis:
    """Build a hypothesis, attaching the default falsification criteria."""
    criteria = list(falsification or ())
    if include_defaults:
        named = {c.metric for c in criteria}
        criteria += [c for c in DEFAULT_FALSIFICATION if c.metric not in named]
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        statement=statement,
        rationale=rationale or statement,
        mechanism=mechanism,
        expected_regime=expected_regime,
        falsification=tuple(criteria),
        strategy=strategy,
        **kwargs,
    )
