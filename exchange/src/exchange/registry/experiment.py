"""Experiment records.

An experiment is only a result if someone else can obtain it again. That
requires writing down everything that can move a number, which is more than
most research code records:

* the **hypothesis**, with the fingerprint it was registered under, so a
  record cannot be matched to a hypothesis that was edited afterwards;
* the **dataset content hash**, not its filename — two files called
  ``NG_daily.csv`` are not the same data, and the hash is what distinguishes
  a reproduction from a coincidence;
* the **instrument specification**. This one is easy to omit and fatal:
  re-running a natural-gas experiment with the default multiplier of 1.0
  instead of 10,000 produces a different equity curve, a different Sharpe
  and no error message;
* the **execution assumptions**, because a result is a statement about a
  cost model as much as about a market;
* the **seed** and the **environment** (Python, numpy, pandas, git commit),
  because floating-point results move between library versions.

:func:`reproduction_key` is the subset of those that are *inputs*. Outputs —
metrics, selected parameters, the verdict — are compared against a
reproduction, never used to identify it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from exchange.core.env import EnvironmentInfo, capture_environment
from exchange.core.errors import ConfigError, ReproducibilityError
from exchange.core.hashing import hash_config

EXPERIMENT_ID_PATTERN = re.compile(r"^EXP-\d{8}-\d{6}$")

# Outputs compared when checking a reproduction, and the tolerance each is
# allowed. Tight but not zero: a different BLAS can move the last bits of a
# sum of a million floats, and failing a reproduction over 1e-12 teaches
# researchers to ignore the check.
COMPARED_METRICS: tuple[str, ...] = (
    "is_sharpe",
    "oos_sharpe",
    "oos_return",
    "oos_max_drawdown",
    "deflated_sharpe_ratio",
    "pbo",
    "cpcv_median_sharpe",
    "wf_efficiency",
    "conservative_sharpe",
    "stressed_sharpe",
    "n_oos_trades",
)
METRIC_TOLERANCE = 1e-9


def make_experiment_id(sequence: int, *, moment: datetime | None = None) -> str:
    """``EXP-YYYYMMDD-NNNNNN`` — sortable, and unique within a store."""
    moment = moment or datetime.now(UTC)
    if not 0 <= sequence < 1_000_000:
        raise ConfigError(f"experiment sequence out of range: {sequence}")
    return f"EXP-{moment.strftime('%Y%m%d')}-{sequence:06d}"


@dataclass
class ExperimentRecord:
    """One complete, reproducible experiment."""

    experiment_id: str
    hypothesis: dict
    dataset: dict
    instrument: dict
    config: dict
    metrics: dict = field(default_factory=dict)
    verdict: dict = field(default_factory=dict)
    best_params: dict = field(default_factory=dict)
    stages: list = field(default_factory=list)
    robustness: dict | None = None
    # The whole programme's settings (registry paths, logging, feature
    # providers). Recorded for context but kept out of ``reproduction_key``:
    # these cannot change a number, and including them would report a
    # reproduction run with a different registry path as a different
    # experiment.
    programme_config: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    seed: int | None = None
    created_at: str = ""
    notes: str = ""
    tags: list = field(default_factory=list)
    artefacts: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not EXPERIMENT_ID_PATTERN.match(self.experiment_id):
            raise ConfigError(
                f"experiment id {self.experiment_id!r} must look like EXP-20260902-000157"
            )
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat(timespec="seconds")

    # -- identity --------------------------------------------------------

    @property
    def hypothesis_id(self) -> str:
        return str(self.hypothesis.get("hypothesis_id", ""))

    @property
    def hypothesis_fingerprint(self) -> str:
        return str(self.hypothesis.get("fingerprint", ""))

    @property
    def data_hash(self) -> str:
        return str(self.dataset.get("data_hash", ""))

    @property
    def verdict_label(self) -> str:
        return str(self.verdict.get("label", "UNKNOWN"))

    @property
    def total_trials(self) -> int:
        return int(self.hypothesis.get("total_trials", 0) or 0)

    @property
    def reproduction_key(self) -> dict:
        """Every *input* that determines the result.

        Deliberately excludes timestamps, the experiment id, notes, tags and
        the environment: a reproduction on a different machine is still a
        reproduction, and the environment is reported in the comparison
        rather than being part of the identity.
        """
        return {
            "hypothesis_fingerprint": self.hypothesis_fingerprint,
            "strategy": self.hypothesis.get("strategy"),
            "param_space": self.hypothesis.get("param_space", {}),
            "fixed_params": self.hypothesis.get("fixed_params", {}),
            "data_hash": self.data_hash,
            "instrument": self.instrument,
            "config": self.config,
            "seed": self.seed,
        }

    @property
    def reproduction_hash(self) -> str:
        return hash_config(self.reproduction_key)

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        data = asdict(self)
        data["reproduction_hash"] = self.reproduction_hash
        return data

    @classmethod
    def from_dict(cls, data: Mapping) -> ExperimentRecord:
        payload = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**payload)

    def index_row(self) -> dict:
        """The flat row written to the store's index."""
        return {
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "hypothesis_id": self.hypothesis_id,
            "strategy": self.hypothesis.get("strategy", ""),
            "symbol": self.dataset.get("symbol", ""),
            "data_hash": self.data_hash[:16],
            "verdict": self.verdict_label,
            "n_configurations": self.hypothesis.get("n_configurations", 0),
            "total_trials": self.total_trials,
            "is_sharpe": self.metrics.get("is_sharpe"),
            "oos_sharpe": self.metrics.get("oos_sharpe"),
            "deflated_sharpe_ratio": self.metrics.get("deflated_sharpe_ratio"),
            "pbo": self.metrics.get("pbo"),
            "robustness_score": (self.robustness or {}).get("score"),
            "robustness_grade": (self.robustness or {}).get("grade", ""),
            "reproduction_hash": self.reproduction_hash[:16],
            "hypothesis_fingerprint": self.hypothesis_fingerprint[:16],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Experiment `{self.experiment_id}`",
            "",
            f"- **Created:** {self.created_at}",
            f"- **Hypothesis:** `{self.hypothesis_id}` "
            f"(fingerprint `{self.hypothesis_fingerprint[:16]}`)",
            f"- **Dataset:** {self.dataset.get('symbol', '?')} "
            f"({self.dataset.get('n_bars', '?')} bars, hash `{self.data_hash[:16]}`)",
            f"- **Verdict:** `{self.verdict_label}`",
            f"- **Reproduction hash:** `{self.reproduction_hash}`",
            "",
            f"Reproduce with `exchange reproduce {self.experiment_id}`.",
        ]
        return "\n".join(lines) + "\n"


def build_record(
    experiment_id: str,
    funnel_result,
    *,
    instrument: Mapping[str, Any] | None = None,
    robustness: Mapping[str, Any] | None = None,
    notes: str = "",
    tags: list | None = None,
    environment: EnvironmentInfo | None = None,
) -> ExperimentRecord:
    """Turn a :class:`~exchange.funnel.pipeline.FunnelResult` into a record."""
    payload = funnel_result.to_dict()
    spec = instrument
    if spec is None:
        config = getattr(funnel_result.oos_result, "config", None)
        spec = config.instrument.to_dict() if config is not None else {}
    env = environment or capture_environment()
    record = ExperimentRecord(
        experiment_id=experiment_id,
        hypothesis=payload["hypothesis"],
        dataset=payload["dataset"],
        instrument=dict(spec),
        config=payload["config"],
        metrics=payload["metrics"],
        verdict=payload["verdict"] or {},
        best_params=payload["best_params"],
        stages=payload["stages"],
        robustness=dict(robustness) if robustness is not None else None,
        environment=env.to_dict(),
        seed=funnel_result.config.seed,
        notes=notes,
        tags=list(tags or []),
    )
    validate_record(record)
    return record


def validate_record(record: ExperimentRecord) -> None:
    """Refuse a record that could not be reproduced from what it contains.

    Called before anything is written, because a record that is missing an
    input is worse than no record: it looks like evidence.
    """
    problems = []
    if not record.hypothesis_fingerprint:
        problems.append("hypothesis fingerprint is missing")
    if not record.data_hash:
        problems.append("dataset content hash is missing — the data cannot be identified")
    if not record.instrument:
        problems.append(
            "instrument specification is missing — re-running with a default multiplier "
            "silently produces different money"
        )
    if "execution" not in record.config and "execution_scenario" not in record.config:
        problems.append("execution assumptions are missing — the result is a cost-model statement")
    if record.seed is None:
        problems.append("seed is missing — the run is not repeatable")
    if not record.metrics:
        problems.append("no metrics recorded")
    if not record.environment:
        problems.append("environment capture is missing")
    if problems:
        raise ReproducibilityError(
            f"{record.experiment_id} cannot be reproduced from its own record:\n  - "
            + "\n  - ".join(problems)
        )


# ==========================================================================
# Comparison
# ==========================================================================


@dataclass
class Comparison:
    """The outcome of checking one reproduction against its record."""

    status: str                      # EXACT | EQUIVALENT | DIVERGENT | INCOMPARABLE
    experiment_id: str
    metric_differences: dict = field(default_factory=dict)
    input_differences: dict = field(default_factory=dict)
    environment_differences: dict = field(default_factory=dict)
    verdict_changed: bool = False
    params_changed: bool = False
    detail: str = ""

    @property
    def reproduced(self) -> bool:
        return self.status in {"EXACT", "EQUIVALENT"}

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"**Reproduction of `{self.experiment_id}`: {self.status}** — {self.detail}",
            "",
        ]
        if self.input_differences:
            lines += ["Inputs that differ (this is not a reproduction):", ""]
            lines += [f"- `{k}`: {v[0]!r} -> {v[1]!r}" for k, v in self.input_differences.items()]
            lines.append("")
        if self.metric_differences:
            lines += ["| metric | recorded | reproduced | difference |", "|---|---|---|---|"]
            for key, (a, b) in sorted(self.metric_differences.items()):
                delta = "n/a" if a is None or b is None else f"{b - a:+.6g}"
                lines.append(f"| {key} | {_fmt(a)} | {_fmt(b)} | {delta} |")
            lines.append("")
        if self.environment_differences:
            lines += ["Environment differences (reported, not failed on):", ""]
            lines += [
                f"- `{k}`: {v[0]!r} -> {v[1]!r}" for k, v in self.environment_differences.items()
            ]
        return "\n".join(lines) + "\n"


def compare_records(
    recorded: ExperimentRecord,
    reproduced: ExperimentRecord,
    *,
    tolerance: float = METRIC_TOLERANCE,
) -> Comparison:
    """Compare a reproduction against the original record.

    ``EXACT``
        Every compared metric is bit-identical.
    ``EQUIVALENT``
        Metrics agree within ``tolerance`` and the verdict is unchanged.
        Library-level floating-point differences land here.
    ``DIVERGENT``
        A metric moved beyond tolerance, or the verdict or the selected
        parameters changed. This is a bug — in the code, the data, or the
        record — and is never reported as a successful reproduction.
    ``INCOMPARABLE``
        The inputs differ, so the two runs are different experiments. This
        is reported separately because calling it a failed reproduction
        would hide the real problem: somebody changed an input.
    """
    input_diff = _diff_mappings(recorded.reproduction_key, reproduced.reproduction_key)
    env_diff = _diff_mappings(
        recorded.environment.get("packages", {}),
        reproduced.environment.get("packages", {}),
    )
    for key in ("python_version", "platform"):
        a = recorded.environment.get(key)
        b = reproduced.environment.get(key)
        if a != b:
            env_diff[key] = (a, b)

    if input_diff:
        return Comparison(
            status="INCOMPARABLE",
            experiment_id=recorded.experiment_id,
            input_differences=input_diff,
            environment_differences=env_diff,
            detail=(
                f"{len(input_diff)} input(s) differ, so this is a different experiment, "
                "not a reproduction of this one"
            ),
        )

    metric_diff: dict[str, tuple] = {}
    exact = True
    for key in COMPARED_METRICS:
        a = _number(recorded.metrics.get(key))
        b = _number(reproduced.metrics.get(key))
        if a is None and b is None:
            continue
        if a is None or b is None:
            metric_diff[key] = (a, b)
            exact = False
            continue
        if a != b:
            exact = False
            if abs(b - a) > tolerance * max(1.0, abs(a)):
                metric_diff[key] = (a, b)

    verdict_changed = recorded.verdict_label != reproduced.verdict_label
    params_changed = dict(recorded.best_params) != dict(reproduced.best_params)

    if metric_diff or verdict_changed or params_changed:
        reasons = []
        if metric_diff:
            reasons.append(f"{len(metric_diff)} metric(s) outside tolerance")
        if verdict_changed:
            reasons.append(
                f"verdict changed {recorded.verdict_label} -> {reproduced.verdict_label}"
            )
        if params_changed:
            reasons.append("the search selected different parameters")
        return Comparison(
            status="DIVERGENT",
            experiment_id=recorded.experiment_id,
            metric_differences=metric_diff,
            environment_differences=env_diff,
            verdict_changed=verdict_changed,
            params_changed=params_changed,
            detail="; ".join(reasons),
        )

    if exact:
        return Comparison(
            status="EXACT",
            experiment_id=recorded.experiment_id,
            environment_differences=env_diff,
            detail="every compared metric is identical",
        )
    return Comparison(
        status="EQUIVALENT",
        experiment_id=recorded.experiment_id,
        environment_differences=env_diff,
        detail=f"metrics agree within {tolerance:g} and the verdict is unchanged",
    )


def _diff_mappings(a: Mapping, b: Mapping) -> dict[str, tuple]:
    out: dict[str, tuple] = {}
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key), b.get(key)
        if _canonical(va) != _canonical(vb):
            out[key] = (va, vb)
    return out


def _canonical(value):
    if isinstance(value, Mapping):
        return {k: _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return float(value)
    return value


def _number(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value == value else None  # NaN -> None


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.6g}"
