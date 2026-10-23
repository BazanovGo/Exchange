"""What the CLI commands actually do.

Kept separate from the Typer wiring so every operation is callable from a
notebook, a test, or another programme with no command-line parsing in the
way — and so the command surface cannot drift away from the library.

The functions here are the only place that *composes* the pipeline, which
is why two obligations live here rather than being left to callers:

* **The multiple-testing ledger is charged automatically.** Before a funnel
  runs, :func:`run_research` asks the registry how many configurations have
  already been tested on this exact dataset and adds them to the
  hypothesis' trial count. A researcher cannot forget to do it, and cannot
  quietly choose not to.
* **The experiment is recorded whatever the verdict.** A rejection is
  written with the same completeness as a success, because the ledger above
  is only honest if failures are in it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from exchange.backtest.engine import BacktestConfig, InstrumentSpec, run_backtest
from exchange.backtest.metrics import performance_metrics
from exchange.core.config import ResearchConfig
from exchange.core.env import capture_environment
from exchange.core.errors import ConfigError, DataError
from exchange.core.logging import get_logger
from exchange.data.continuous import build_continuous
from exchange.data.contracts import ContractRegistry
from exchange.data.dataset import Dataset, dataset_from_continuous, load_bars, make_dataset
from exchange.data.providers import build_provider
from exchange.data.schema import normalise_bars
from exchange.execution.assumptions import get_scenario
from exchange.falsification.engine import RobustnessReport, falsify
from exchange.funnel.hypothesis import Hypothesis
from exchange.funnel.pipeline import FunnelResult, run_funnel
from exchange.funnel.verdict import VerdictLevel
from exchange.registry.experiment import Comparison, ExperimentRecord, build_record, compare_records
from exchange.registry.store import ExperimentStore
from exchange.reporting.research_report import research_report
from exchange.scanner.report import scan_dataset
from exchange.synthetic.generators import CONTROLS, make_control

logger = get_logger(__name__)


# ==========================================================================
# Data
# ==========================================================================


def _coerce_scalar(value: str):
    """Turn a ``k=v`` string from a source spec into the obvious value."""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def load_dataset(
    source: str,
    *,
    config: ResearchConfig,
    symbol: str | None = None,
    timeframe: str | None = None,
    strict: bool | None = None,
) -> Dataset:
    """Load one dataset from a file, a configured provider, or a control.

    ``source`` is one of

    * a path to a CSV or Parquet file;
    * ``synthetic:<kind>[:seed][:k=v,k=v]`` — a named control from
      :mod:`exchange.synthetic.generators`, which is how every example and
      every test gets data without a vendor. The trailing pairs set
      generator parameters, e.g.
      ``synthetic:signal_plus_noise:7:edge_per_bar=0.003,lookback=5``;
    * a plain symbol, fetched through the provider in the configuration.
    """
    strict = config.data.strict if strict is None else strict
    timeframe = timeframe or config.data.timeframe

    if source.startswith("synthetic:"):
        parts = source.split(":")
        kind = parts[1]
        if kind not in CONTROLS:
            raise ConfigError(f"unknown control {kind!r}; known: {sorted(CONTROLS)}")
        kwargs: dict[str, Any] = {}
        if len(parts) > 2 and parts[2]:
            kwargs["seed"] = int(parts[2])
        for extra in parts[3:]:
            for item in extra.split(","):
                if not item.strip():
                    continue
                if "=" not in item:
                    raise ConfigError(
                        f"generator parameters must be name=value, got {item!r} in {source!r}"
                    )
                key, value = item.split("=", 1)
                kwargs[key.strip()] = _coerce_scalar(value.strip())
        series = make_control(kind, **kwargs)
        return make_dataset(
            series.bars,
            symbol=symbol or f"SYNTH_{kind.upper()}",
            timeframe=timeframe,
            strict=strict,
            provenance={
                # The spec string is the reproduction input, so it is stored
                # verbatim: a generator's ground-truth dict records what the
                # series contains, not the arguments that would rebuild it.
                "source": source,
                "synthetic": series.truth.to_dict() if hasattr(series.truth, "to_dict") else kind,
            },
        )

    path = Path(source)
    if path.is_file():
        frame = pd.read_parquet(path) if path.suffix in {".parquet", ".pq"} else pd.read_csv(path)
        return make_dataset(
            frame,
            symbol=symbol or path.stem,
            timeframe=timeframe,
            strict=strict,
            provenance={
                "source": str(path.resolve()),
                "file": str(path.resolve()),
                "format": path.suffix.lstrip("."),
            },
        )

    provider = build_provider(config.data.provider)
    return load_bars(
        provider, symbol or source, timeframe=timeframe, strict=strict
    )


def validate_file(source: str, *, symbol: str | None = None, timeframe: str | None = None):
    """Quality-check a file without the strict gate, and return the report."""
    path = Path(source)
    if not path.is_file():
        raise DataError(f"{source} does not exist")
    frame = pd.read_parquet(path) if path.suffix in {".parquet", ".pq"} else pd.read_csv(path)
    dataset = make_dataset(
        frame, symbol=symbol or path.stem, timeframe=timeframe, strict=False,
        provenance={"file": str(path.resolve())},
    )
    return dataset, dataset.quality


def build_continuous_dataset(
    directory: str | Path,
    *,
    config: ResearchConfig,
    root: str | None = None,
    roll: str | None = None,
    adjustment: str | None = None,
    pattern: str = "*.csv",
) -> Dataset:
    """Splice a directory of dated contract files into one research dataset."""
    directory = Path(directory)
    files = sorted(directory.glob(pattern))
    if not files:
        raise DataError(f"no files matching {pattern} in {directory}")
    panel = {}
    for path in files:
        frame = pd.read_parquet(path) if path.suffix in {".parquet", ".pq"} else pd.read_csv(path)
        panel[path.stem] = normalise_bars(frame, symbol=path.stem)
    series = build_continuous(
        panel,
        root=root,
        roll=roll or config.data.roll,
        adjustment=adjustment or config.data.adjustment,
        registry=ContractRegistry(),
    )
    return dataset_from_continuous(series, strict=config.data.strict)


def alternative_constructions(
    directory: str | Path,
    *,
    root: str | None = None,
    pattern: str = "*.csv",
    variants: Sequence[tuple[str, str]] = (("calendar", "none"), ("volume", "backward"), ("open_interest", "ratio")),
) -> dict[str, pd.DataFrame]:
    """Build the same series several ways, for the ``roll_methodology`` attack.

    This is what lets the attack ask the only question that matters about a
    spliced series: would the result exist if the splice had been done
    differently?
    """
    directory = Path(directory)
    panel = {}
    for path in sorted(directory.glob(pattern)):
        frame = pd.read_parquet(path) if path.suffix in {".parquet", ".pq"} else pd.read_csv(path)
        panel[path.stem] = normalise_bars(frame, symbol=path.stem)
    if not panel:
        raise DataError(f"no files matching {pattern} in {directory}")
    out = {}
    for roll, adjustment in variants:
        series = build_continuous(
            panel, root=root, roll=roll, adjustment=adjustment, registry=ContractRegistry()
        )
        out[f"{roll}/{adjustment}"] = series.bars
    return out


# ==========================================================================
# Research
# ==========================================================================


@dataclass
class ResearchOutcome:
    """Everything one ``exchange research`` invocation produced."""

    result: FunnelResult
    robustness: RobustnessReport | None
    record: ExperimentRecord | None
    report_path: Path | None
    report: str

    @property
    def verdict(self) -> VerdictLevel:
        """The funnel verdict, lowered by the attack record if it was run."""
        level = self.result.verdict.level if self.result.verdict else VerdictLevel.REJECTED
        return self.robustness.cap(level) if self.robustness is not None else level

    @property
    def verdict_label(self) -> str:
        return self.verdict.name


def run_research(
    dataset: Dataset,
    hypothesis: Hypothesis,
    *,
    config: ResearchConfig,
    instrument: InstrumentSpec | None = None,
    store: ExperimentStore | None = None,
    alternative_bars: Mapping[str, pd.DataFrame] | None = None,
    scan_first: bool = False,
    write_report: bool = True,
    record: bool = True,
) -> ResearchOutcome:
    """The full pipeline for one hypothesis, recorded and reported."""
    store = store if store is not None else ExperimentStore(config.registry_root)
    instrument = instrument or InstrumentSpec(symbol=dataset.symbol)

    # -- charge the ledger before anything is measured ------------------
    prior = store.prior_trials(dataset.data_hash)
    if prior and hypothesis.prior_hypotheses_on_data < prior:
        logger.info(
            "charging %d configuration(s) already tested on %s to this hypothesis",
            prior, dataset.symbol,
        )
        hypothesis = replace(hypothesis, prior_hypotheses_on_data=prior)

    scan_result = scan_dataset(dataset) if scan_first else None

    result = run_funnel(dataset, hypothesis, config=config.funnel, instrument=instrument)

    robustness = None
    if config.falsification.enabled and result.oos_result is not None:
        robustness = falsify(
            result,
            attacks=list(config.falsification.attacks) or None,
            alternative_bars=alternative_bars,
            n_resamples=config.falsification.n_resamples,
        )

    experiment_id = store.next_id() if record else ""
    environment = capture_environment()
    report = research_report(
        result,
        robustness=robustness,
        scan=scan_result,
        experiment_id=experiment_id,
        prior_trials=prior,
        environment=environment.to_dict(),
    )

    stored_record = None
    report_path = None
    if record:
        stored_record = build_record(
            experiment_id,
            result,
            instrument=instrument.to_dict(),
            robustness=robustness.to_dict() if robustness else None,
            environment=environment,
            notes=f"config fingerprint {config.fingerprint}",
            tags=[dataset.symbol, hypothesis.strategy],
        )
        stored_record.programme_config = config.to_dict()
        store.save(
            stored_record,
            report=report if write_report else None,
            tables={
                "sweep": result.sweep,
                "stress": result.stress_table,
                "cpcv": result.cpcv_table,
                "walk_forward": result.wf_table,
                "regimes": result.regime_table,
                "robustness": robustness.to_frame() if robustness else None,
            },
        )
        report_path = store.directory(experiment_id) / "report.md"

    if write_report and not record:
        root = Path(config.reports_root)
        root.mkdir(parents=True, exist_ok=True)
        report_path = root / f"{hypothesis.hypothesis_id}.md"
        report_path.write_text(report, encoding="utf-8")

    return ResearchOutcome(
        result=result,
        robustness=robustness,
        record=stored_record,
        report_path=report_path,
        report=report,
    )


def run_backtest_once(
    dataset: Dataset,
    strategy_name: str,
    *,
    config: ResearchConfig,
    params: Mapping[str, Any] | None = None,
    instrument: InstrumentSpec | None = None,
    scenario: str | None = None,
):
    """One backtest, no search, no verdict.

    Useful for inspection and deliberately *not* a research result: a
    single backtest answers no question that a verdict would need, and the
    CLI says so when it prints the numbers.
    """
    from exchange.strategies.base import build

    strategy = build(strategy_name, **dict(params or {}))
    strategy.check_causality(dataset.bars.iloc[: min(len(dataset.bars), 400)])
    instrument = instrument or InstrumentSpec(symbol=dataset.symbol)
    execution = get_scenario(scenario or config.execution_scenario)
    cfg = BacktestConfig(
        initial_cash=config.funnel.initial_cash,
        instrument=instrument,
        execution=execution,
        annualisation=dataset.bars_per_year,
    )
    result = run_backtest(
        dataset.bars, strategy.positions(dataset.bars), config=cfg, symbol=dataset.symbol
    )
    return result, performance_metrics(result, periods_per_year=dataset.bars_per_year)


# ==========================================================================
# Reproduction
# ==========================================================================


def reproduce_experiment(
    experiment_id: str,
    *,
    config: ResearchConfig,
    store: ExperimentStore | None = None,
    source: str | None = None,
) -> tuple[Comparison, ExperimentRecord, ExperimentRecord]:
    """Re-run a recorded experiment and compare it against the record.

    The data is re-loaded and its content hash checked *before* anything is
    recomputed: a reproduction on different data is not a failed
    reproduction, it is a different experiment, and conflating the two
    sends people looking for bugs in the code.
    """
    store = store if store is not None else ExperimentStore(config.registry_root)
    recorded = store.load(experiment_id)

    source = source or _recorded_source(recorded)
    dataset = load_dataset(source, config=config, symbol=recorded.dataset.get("symbol"))
    if dataset.data_hash != recorded.data_hash:
        raise ConfigError(
            f"{experiment_id}: the data at {source!r} hashes to {dataset.data_hash[:16]} but the "
            f"record was made on {recorded.data_hash[:16]}. Reproduce with the original data, or "
            "register this as a new experiment on the new data."
        )

    hypothesis = Hypothesis.from_dict(recorded.hypothesis)
    instrument = InstrumentSpec(**{k: v for k, v in recorded.instrument.items() if k in InstrumentSpec.__dataclass_fields__})
    funnel_config = replace(config.funnel, **_funnel_overrides(recorded.config))

    result = run_funnel(dataset, hypothesis, config=funnel_config, instrument=instrument)
    fresh = build_record(
        experiment_id,
        result,
        instrument=instrument.to_dict(),
        notes="reproduction",
    )
    # The reproduction inherits the original's seed and inputs by
    # construction; anything that differs is in the outputs.
    fresh.seed = recorded.seed
    return compare_records(recorded, fresh), recorded, fresh


def falsify_experiment(
    experiment_id: str,
    *,
    config: ResearchConfig,
    store: ExperimentStore | None = None,
    source: str | None = None,
    alternative_bars: Mapping[str, pd.DataFrame] | None = None,
    attach: bool = True,
) -> RobustnessReport:
    """Run (or re-run) the attacks on a recorded experiment."""
    store = store if store is not None else ExperimentStore(config.registry_root)
    recorded = store.load(experiment_id)
    source = source or _recorded_source(recorded)
    dataset = load_dataset(source, config=config, symbol=recorded.dataset.get("symbol"))
    if dataset.data_hash != recorded.data_hash:
        raise ConfigError(
            f"{experiment_id}: data hash mismatch; attacking a different dataset would not be "
            "informative about this experiment"
        )
    hypothesis = Hypothesis.from_dict(recorded.hypothesis)
    instrument = InstrumentSpec(**{k: v for k, v in recorded.instrument.items() if k in InstrumentSpec.__dataclass_fields__})
    funnel_config = replace(config.funnel, **_funnel_overrides(recorded.config))
    result = run_funnel(dataset, hypothesis, config=funnel_config, instrument=instrument)
    report = falsify(
        result,
        attacks=list(config.falsification.attacks) or None,
        alternative_bars=alternative_bars,
        n_resamples=config.falsification.n_resamples,
    )
    if attach:
        store.attach_robustness(experiment_id, report.to_dict())
    return report


def _recorded_source(record: ExperimentRecord) -> str:
    provenance = (record.dataset.get("provenance") or {})
    if provenance.get("source"):
        return str(provenance["source"])
    if "file" in provenance:
        return str(provenance["file"])
    symbol = record.dataset.get("symbol")
    if symbol:
        return str(symbol)
    raise ConfigError(
        f"{record.experiment_id}: the record does not say where its data came from; pass --source"
    )


def _funnel_overrides(config_payload: Mapping) -> dict:
    from exchange.funnel.pipeline import FunnelConfig

    known = set(FunnelConfig.__dataclass_fields__)
    payload = {k: v for k, v in config_payload.items() if k in known}
    if "stress_ladder" in payload and payload["stress_ladder"] is not None:
        payload["stress_ladder"] = tuple(payload["stress_ladder"])
    return payload


# ==========================================================================
# Hypotheses from files
# ==========================================================================


def load_hypothesis(path: str | Path) -> Hypothesis:
    """Read a pre-registered hypothesis from YAML or JSON.

    Registering from a file is the intended workflow: the file is written
    and committed *before* the run, so its fingerprint can be checked
    against the record afterwards. A hypothesis assembled from command-line
    flags at run time offers no such proof, which is why the CLI accepts
    one only for throwaway exploration and marks it in the record.
    """
    import json as _json

    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"hypothesis file {path} does not exist")
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        import yaml

        payload = yaml.safe_load(text) or {}
    else:
        payload = _json.loads(text or "{}")
    if not isinstance(payload, Mapping):
        raise ConfigError(f"{path} must contain a mapping")
    payload = dict(payload)
    if "hypothesis_id" not in payload:
        payload["hypothesis_id"] = path.stem.upper()
    if "falsification" in payload:
        return Hypothesis.from_dict(payload)
    from exchange.funnel.hypothesis import make_hypothesis

    known = {
        "statement", "mechanism", "expected_regime", "strategy", "rationale",
        "param_space", "fixed_params", "symbol", "timeframe", "author", "tags",
        "prior_hypotheses_on_data",
    }
    unknown = set(payload) - known - {"hypothesis_id"}
    if unknown:
        raise ConfigError(f"unknown key(s) {sorted(unknown)} in {path}")
    hypothesis_id = payload.pop("hypothesis_id")
    if "tags" in payload:
        payload["tags"] = tuple(payload["tags"])
    return make_hypothesis(hypothesis_id, **payload)


# ==========================================================================
# Controls
# ==========================================================================


@dataclass
class ControlOutcome:
    """One synthetic control's result, and whether it is acceptable."""

    kind: str
    label: str
    role: str                       # negative | positive
    verdict: str
    expectation: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "label": self.label,
            "role": self.role,
            "verdict": self.verdict,
            "expectation": self.expectation,
            "passed": self.passed,
            "detail": self.detail,
        }


DEFAULT_CONTROLS_PATH = Path("configs/controls.yaml")


def run_controls(
    *,
    config: ResearchConfig,
    path: str | Path | None = None,
    quick: bool = False,
) -> list[ControlOutcome]:
    """Run the synthetic controls and judge them against their expectations.

    This is the framework's own calibration, and it is the one test whose
    failure means *the measuring instrument is broken* rather than that a
    hypothesis was wrong:

    * a negative control that scores above ``expect_max_verdict`` is a
      false discovery produced by the framework itself;
    * a positive control that scores below ``expect_min_verdict`` means a
      known, analytically specified edge is invisible to the pipeline, so
      no absence of evidence it reports can be believed.
    """
    import yaml

    path = Path(path) if path is not None else DEFAULT_CONTROLS_PATH
    if not path.is_file():
        raise ConfigError(f"controls file {path} does not exist")
    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    n_bars = int(spec.get("n_bars", 2000))
    if quick:
        n_bars = min(n_bars, 1200)
    funnel = replace(
        config.funnel,
        n_bootstrap=200 if quick else config.funnel.n_bootstrap,
        n_monte_carlo=200 if quick else config.funnel.n_monte_carlo,
        wf_train=min(config.funnel.wf_train, max(int(n_bars * 0.25), 100)),
        wf_test=min(config.funnel.wf_test, max(int(n_bars * 0.06), 30)),
    )
    run_config = replace(
        config,
        funnel=funnel,
        falsification=replace(
            config.falsification,
            enabled=not quick,
            n_resamples=200 if quick else config.falsification.n_resamples,
        ),
    )

    outcomes: list[ControlOutcome] = []
    for role, entries in (("negative", spec.get("negative", [])), ("positive", spec.get("positive", []))):
        for entry in entries or []:
            outcomes.append(_run_one_control(entry, role=role, config=run_config, n_bars=n_bars))
    for entry in spec.get("leakage", []) or []:
        outcomes.append(_run_leakage_control(entry, config=run_config, n_bars=n_bars))
    return outcomes


def _run_leakage_control(entry: Mapping, *, config: ResearchConfig, n_bars: int) -> ControlOutcome:
    """Plant a lookahead deliberately and check that a guard catches it.

    These are the controls that matter most. A false discovery on a random
    walk costs a wasted week; a temporal leak that no guard catches
    invalidates every result the framework has ever produced, silently.
    """
    from exchange.core.errors import LookaheadError
    from exchange.data.temporal import assert_no_column_leakage
    from exchange.execution.assumptions import BASE
    from exchange.strategies.base import build
    from exchange.synthetic.generators import random_walk, with_lookahead_trap

    kind = entry["kind"]
    label = str(entry.get("label", kind))

    if kind == "lookahead_trap":
        series = with_lookahead_trap(random_walk(n=n_bars, seed=101))
        try:
            assert_no_column_leakage(series.bars)
        except LookaheadError as exc:
            return ControlOutcome(
                kind, label, "leakage", "LookaheadError", "guard fires", True, str(exc)[:160]
            )
        return ControlOutcome(
            kind, label, "leakage", "no error", "guard fires", False,
            "THE LEAKAGE GUARD DID NOT FIRE on a planted next-bar column",
        )

    if kind == "same_close_execution":
        dataset = make_dataset(random_walk(n=n_bars, seed=103).bars, symbol="CONTROL_SAME_CLOSE")
        strategy = build("ts_momentum", lookback=20)
        cfg = BacktestConfig(
            instrument=InstrumentSpec(symbol=dataset.symbol, multiplier=1000.0),
            execution=replace(BASE, fill_timing="same_close", label="same_close"),
            annualisation=dataset.bars_per_year,
        )
        result = run_backtest(
            dataset.bars, strategy.positions(dataset.bars), config=cfg, symbol=dataset.symbol
        )
        passed = not result.valid
        return ControlOutcome(
            kind, label, "leakage",
            "invalid_result" if passed else "valid_result",
            "result marked invalid", passed,
            "" if passed else "A RUN THAT FILLS AT THE SIGNAL'S OWN CLOSE WAS REPORTED AS VALID",
        )

    raise ConfigError(f"unknown leakage control {kind!r}")


def _run_one_control(entry: Mapping, *, role: str, config: ResearchConfig, n_bars: int) -> ControlOutcome:
    from exchange.funnel.hypothesis import make_hypothesis

    kind = entry["kind"]
    params = dict(entry.get("params", {}))
    params.setdefault("n", n_bars)
    series = make_control(kind, **params)
    dataset = make_dataset(
        series.bars,
        symbol=f"CONTROL_{kind.upper()}",
        strict=True,
        provenance={"synthetic": series.truth.to_dict() if hasattr(series.truth, "to_dict") else kind},
    )

    strategy = entry.get("strategy", "ts_momentum")
    param_space = {k: list(v) for k, v in (entry.get("param_space") or {"lookback": [10, 20, 60]}).items()}
    hypothesis = make_hypothesis(
        f"CONTROL-{kind.upper()}",
        statement=(
            f"The {kind} control contains an exploitable effect that the {strategy} family can "
            "capture out of sample."
        ),
        mechanism=(
            "This is a synthetic control, so the mechanism is the generator itself: the series "
            "was constructed with a known data-generating process, and the question is only "
            "whether the pipeline reaches the right conclusion about it."
        ),
        expected_regime=(
            "Throughout the sample, since the generator is stationary by construction."
        ),
        strategy=strategy,
        param_space=param_space,
    )
    instrument = InstrumentSpec(
        symbol=dataset.symbol, **{k: v for k, v in (entry.get("instrument") or {}).items()}
    )

    outcome = run_research(
        dataset, hypothesis, config=config, instrument=instrument,
        store=ExperimentStore(config.registry_root), record=False, write_report=False,
    )
    level = outcome.verdict

    if role == "negative":
        ceiling = VerdictLevel[entry.get("expect_max_verdict", "WEAK")]
        passed = level <= ceiling
        expectation = f"<= {ceiling.name}"
        detail = (
            "" if passed else
            f"FALSE DISCOVERY: a series with no exploitable structure reached {level.name}"
        )
    else:
        floor = VerdictLevel[entry.get("expect_min_verdict", "PROMISING")]
        passed = level >= floor
        expectation = f">= {floor.name}"
        detail = (
            "" if passed else
            f"MISSED A KNOWN EDGE: the pipeline could not recover a constructed effect ({level.name})"
        )
    return ControlOutcome(
        kind=kind,
        label=str(entry.get("label", kind)),
        role=role,
        verdict=level.name,
        expectation=expectation,
        passed=bool(passed),
        detail=detail,
    )
