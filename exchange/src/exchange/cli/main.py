"""The ``exchange`` command line.

One entry point for the whole programme. Every command takes ``--config``,
prints its reasoning rather than only its result, and exits non-zero when
something is wrong — so the same commands work in a terminal and in CI
without a second code path.

Deliberately absent: ``research-loop``. The autonomous loop is designed
(see ``docs/AI_RESEARCHER.md``) but not shipped, because an agent that can
run arbitrary code against a research registry needs a sandbox, a
filesystem and network allowlist, and a resource budget first. Shipping the
loop before the sandbox would be the single most dangerous thing in this
package, and the command prints that rather than pretending the feature is
missing by accident.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer

from exchange.core.config import ResearchConfig, load_config
from exchange.core.errors import ExchangeError
from exchange.core.logging import configure_logging, get_logger

app = typer.Typer(
    name="exchange",
    help="Falsification-first quantitative research.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(name="data", help="Load, validate and splice market data.", no_args_is_help=True)
experiments_app = typer.Typer(name="experiments", help="The experiment registry.", no_args_is_help=True)
app.add_typer(data_app)
app.add_typer(experiments_app)

logger = get_logger(__name__)

ConfigOption = Annotated[
    Path | None, typer.Option("--config", "-c", help="Configuration file (default configs/research.yaml).")
]
SeedOption = Annotated[int | None, typer.Option("--seed", help="Override the configured seed.")]
VerboseOption = Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")]


def _config(path: Path | None, *, seed: int | None = None, verbose: bool = False, **overrides) -> ResearchConfig:
    payload = {k: v for k, v in overrides.items() if v is not None}
    if seed is not None:
        payload["seed"] = seed
    if verbose:
        payload["log_level"] = "DEBUG"
    config = load_config(path, overrides=payload or None)
    configure_logging(config.log_level)
    return config


def _echo(text: str = "") -> None:
    typer.echo(text)


def _fail(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _frame(frame, *, max_rows: int = 40) -> None:
    import pandas as pd

    if frame is None or getattr(frame, "empty", True):
        _echo("(nothing to show)")
        return
    with pd.option_context("display.width", 200, "display.max_columns", 50, "display.max_rows", max_rows):
        _echo(frame.to_string())


# ==========================================================================
# doctor
# ==========================================================================


@app.command()
def doctor(config: ConfigOption = None, verbose: VerboseOption = False) -> None:
    """Check that this installation can produce a reproducible result.

    Everything here is a prerequisite for believing a number the rest of
    the package prints, so it runs before anything else is worth doing.
    """
    from exchange.core.determinism import deterministic_mode
    from exchange.core.env import check_environment, environment_summary
    from exchange.registry.store import ExperimentStore
    from exchange.strategies.base import available, strategy_class

    cfg = _config(config, verbose=verbose)
    _echo("exchange doctor")
    _echo("=" * 72)
    _echo(f"environment    : {environment_summary()}")

    checks = check_environment()
    problems = [c for c in checks if not c[1]]
    for name, _ok, detail in checks:
        if (name, _ok) in {(n, o) for n, o, _ in problems}:
            typer.secho(f"  ! {name:16} {detail}", fg=typer.colors.YELLOW)
        else:
            _echo(f"  ok {name:16} {detail}")

    determinism = deterministic_mode(cfg.seed)
    _echo(f"determinism    : {determinism.summary()}")

    _echo(f"configuration  : {cfg.summary()}")
    _echo(f"strategies     : {len(available())} registered")

    # Causality is the one property that must hold for every strategy, so
    # it is checked here rather than only in the test suite.
    from exchange.synthetic.generators import random_walk

    bars = random_walk(n=400, seed=cfg.seed).bars
    failures = []
    for name in available():
        try:
            strategy_class(name)().check_causality(bars)
        except Exception as exc:  # a non-causal strategy must be loud
            failures.append(f"{name}: {exc}")
    if failures:
        for failure in failures:
            typer.secho(f"  ! NON-CAUSAL {failure}", fg=typer.colors.RED)
    else:
        _echo("causality      : every registered strategy is causal on 400 synthetic bars")

    store = ExperimentStore(cfg.registry_root)
    info = store.summary()
    rejection = info["rejection_rate"]
    _echo(
        f"registry       : {info['n_experiments']} experiment(s), "
        f"{info['n_configurations_tested']} configuration(s) tested, "
        f"rejection rate {rejection:.0%}" if info["n_experiments"] else
        f"registry       : empty ({cfg.registry_root})"
    )
    _echo("=" * 72)
    if failures:
        _fail("doctor found a non-causal strategy; fix that before running anything else")
    _echo("Ready. Next: `exchange controls` to verify the framework on synthetic data.")


# ==========================================================================
# data
# ==========================================================================


@data_app.command("validate")
def data_validate(
    source: Annotated[str, typer.Argument(help="CSV or Parquet file to check.")],
    symbol: Annotated[str | None, typer.Option("--symbol")] = None,
    timeframe: Annotated[str | None, typer.Option("--timeframe")] = None,
    save: Annotated[Path | None, typer.Option("--save", help="Write the report as Markdown.")] = None,
    config: ConfigOption = None,
) -> None:
    """Run every data-quality check and print the report.

    Exits non-zero if any check is an error, so it can gate a pipeline.
    """
    from exchange.cli.runner import validate_file

    _config(config)
    try:
        dataset, report = validate_file(source, symbol=symbol, timeframe=timeframe)
    except ExchangeError as exc:
        _fail(str(exc))
        return
    _echo(report.to_markdown())
    _echo(f"content hash: {dataset.data_hash}")
    if save:
        Path(save).write_text(report.to_markdown(), encoding="utf-8")
        _echo(f"written to {save}")
    if not report.ok:
        _fail(f"{len(report.errors)} blocking quality error(s)")


@data_app.command("import")
def data_import(
    source: Annotated[str, typer.Argument(help="File, symbol or synthetic:<kind>.")],
    out: Annotated[Path, typer.Option("--out", help="Where to write the normalised dataset.")],
    symbol: Annotated[str | None, typer.Option("--symbol")] = None,
    timeframe: Annotated[str | None, typer.Option("--timeframe")] = None,
    config: ConfigOption = None,
) -> None:
    """Normalise and validate a series, then save it with its provenance."""
    from exchange.cli.runner import load_dataset
    from exchange.data.dataset import save_dataset

    cfg = _config(config)
    try:
        dataset = load_dataset(source, config=cfg, symbol=symbol, timeframe=timeframe)
        path = save_dataset(dataset, out)
    except ExchangeError as exc:
        _fail(str(exc))
        return
    _echo(f"{dataset}")
    _echo(f"written to {path} (+ provenance sidecar)")


@data_app.command("continuous")
def data_continuous(
    directory: Annotated[Path, typer.Argument(help="Directory of dated contract files.")],
    out: Annotated[Path | None, typer.Option("--out")] = None,
    root: Annotated[str | None, typer.Option("--root", help="Contract root, e.g. NG.")] = None,
    roll: Annotated[str | None, typer.Option("--roll")] = None,
    adjustment: Annotated[str | None, typer.Option("--adjustment")] = None,
    compare: Annotated[bool, typer.Option("--compare", help="Show every roll/adjustment combination.")] = False,
    config: ConfigOption = None,
) -> None:
    """Splice dated futures contracts into one continuous series.

    With ``--compare`` it builds the series several ways and prints how much
    the choice moves the result. That difference is a measurement of how
    much apparent alpha the splice is capable of manufacturing.
    """
    from exchange.cli.runner import alternative_constructions, build_continuous_dataset
    from exchange.data.dataset import save_dataset

    cfg = _config(config)
    try:
        dataset = build_continuous_dataset(
            directory, config=cfg, root=root, roll=roll, adjustment=adjustment
        )
    except ExchangeError as exc:
        _fail(str(exc))
        return
    _echo(f"{dataset}")
    provenance = dataset.provenance.get("continuous", {})
    _echo(f"rolls: {len(provenance.get('events', []))}  adjustment: {provenance.get('adjustment')}")

    if compare:
        _echo("")
        _echo("The same contracts, spliced differently:")
        import numpy as np

        rows = []
        for label, bars in alternative_constructions(directory, root=root).items():
            total = float(bars["close"].iloc[-1] / bars["close"].iloc[0] - 1.0)
            vol = float(bars["close"].pct_change().std() * np.sqrt(252))
            rows.append({"construction": label, "total_return": total, "annual_vol": vol})
        import pandas as pd

        _frame(pd.DataFrame(rows).set_index("construction"))
        _echo(
            "\nA large spread across these rows is not information about the market. It is the "
            "amount of return the adjustment method alone can create."
        )
    if out:
        _echo(f"written to {save_dataset(dataset, out)}")


# ==========================================================================
# scan / strategies / features
# ==========================================================================


@app.command()
def scan(
    source: Annotated[str, typer.Argument(help="File, symbol or synthetic:<kind>.")],
    symbol: Annotated[str | None, typer.Option("--symbol")] = None,
    save: Annotated[Path | None, typer.Option("--save")] = None,
    config: ConfigOption = None,
    seed: SeedOption = None,
) -> None:
    """Ask whether a series has exploitable structure at all.

    This runs *before* strategy search, on purpose. Searching a series that
    shows no structure is how a programme spends a thousand trials to find
    the best of a thousand noise draws.
    """
    from exchange.cli.runner import load_dataset
    from exchange.scanner.report import scan_dataset

    cfg = _config(config, seed=seed)
    try:
        dataset = load_dataset(source, config=cfg, symbol=symbol)
        result = scan_dataset(dataset)
    except ExchangeError as exc:
        _fail(str(exc))
        return
    _echo(result.to_markdown())
    if save:
        Path(save).write_text(result.to_markdown(), encoding="utf-8")
        _echo(f"written to {save}")


@app.command()
def strategies(config: ConfigOption = None) -> None:
    """List registered strategies, with the size of each one's search space."""
    from exchange.strategies.base import catalog

    _config(config)
    table = catalog()
    _frame(table.drop(columns=["rationale"]))
    _echo("")
    for name, row in table.iterrows():
        _echo(f"{name}: {row['rationale']}")
        _echo("")


@app.command()
def features(
    market: Annotated[str | None, typer.Option("--market", help="natural_gas or crude_oil.")] = None,
    category: Annotated[str | None, typer.Option("--category")] = None,
    detail: Annotated[str | None, typer.Option("--detail", help="Show one feature in full.")] = None,
    config: ConfigOption = None,
) -> None:
    """The fundamental feature catalogue, with publication lags."""
    from exchange.features.catalogue import catalogue, feature_spec

    _config(config)
    if detail:
        try:
            spec = feature_spec(detail)
        except ExchangeError as exc:
            _fail(str(exc))
            return
        for key, value in spec.to_dict().items():
            _echo(f"{key:24} {value}")
        return
    table = catalogue(market=market, category=category)
    if table.empty:
        _echo("(no features match)")
        return
    _frame(table[["name", "category", "frequency", "publication_lag_days", "revision_policy"]], max_rows=60)
    _echo(
        "\npublication_lag_days is measured from the END of the period a number describes to the "
        "first moment it could have been traded on. Using a lag of zero on any weekly series is "
        "a lookahead of about a week."
    )


# ==========================================================================
# backtest
# ==========================================================================


@app.command()
def backtest(
    source: Annotated[str, typer.Argument()],
    strategy: Annotated[str, typer.Argument()],
    param: Annotated[list[str] | None, typer.Option("--param", "-p", help="name=value, repeatable.")] = None,
    scenario: Annotated[str | None, typer.Option("--scenario")] = None,
    multiplier: Annotated[float, typer.Option("--multiplier", help="Contract multiplier.")] = 1.0,
    symbol: Annotated[str | None, typer.Option("--symbol")] = None,
    config: ConfigOption = None,
    seed: SeedOption = None,
) -> None:
    """Run a single backtest. **This is not a research result.**

    One backtest of one configuration says nothing about whether an effect
    is real: there is no out-of-sample window, no deflation for the search
    that produced the parameters, and no attack. Use ``exchange research``
    for anything that will inform a decision.
    """
    from exchange.backtest.engine import InstrumentSpec
    from exchange.cli.runner import load_dataset, run_backtest_once

    cfg = _config(config, seed=seed)
    params: dict = {}
    for item in param or []:
        if "=" not in item:
            _fail(f"--param expects name=value, got {item!r}")
        key, value = item.split("=", 1)
        params[key] = _coerce(value)
    try:
        dataset = load_dataset(source, config=cfg, symbol=symbol)
        result, metrics = run_backtest_once(
            dataset, strategy, config=cfg, params=params,
            instrument=InstrumentSpec(symbol=dataset.symbol, multiplier=multiplier),
            scenario=scenario,
        )
    except ExchangeError as exc:
        _fail(str(exc))
        return
    _echo(f"{dataset}")
    _echo(f"strategy: {strategy} {params or '(defaults)'}  scenario: {result.config.execution.label}")
    _echo("")
    _frame(metrics.to_series().to_frame("value"), max_rows=40)
    if not result.valid:
        typer.secho(
            "\nThis run used an execution assumption that admits lookahead; the numbers above "
            "are not usable.", fg=typer.colors.RED,
        )
    _echo(
        "\nReminder: a single backtest has no out-of-sample window and no correction for the "
        "search that chose these parameters."
    )


# ==========================================================================
# research
# ==========================================================================


@app.command()
def research(
    source: Annotated[str, typer.Argument(help="File, symbol or synthetic:<kind>.")],
    hypothesis: Annotated[Path, typer.Option("--hypothesis", "-h", help="Pre-registered hypothesis file.")],
    symbol: Annotated[str | None, typer.Option("--symbol")] = None,
    multiplier: Annotated[float, typer.Option("--multiplier")] = 1.0,
    tick_size: Annotated[float, typer.Option("--tick-size")] = 0.0,
    contracts_dir: Annotated[Path | None, typer.Option("--contracts", help="Dated contracts, for the roll attack.")] = None,
    scan_first: Annotated[bool, typer.Option("--scan/--no-scan", help="Run the structural scanner first.")] = True,
    attacks: Annotated[bool, typer.Option("--attacks/--no-attacks")] = True,
    record: Annotated[bool, typer.Option("--record/--no-record")] = True,
    config: ConfigOption = None,
    seed: SeedOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Push one pre-registered hypothesis through the whole funnel.

    The hypothesis must be written before the run: the file is hashed into
    the record, so a hypothesis edited to match its result can be detected
    afterwards.
    """
    from dataclasses import replace as _replace

    from exchange.backtest.engine import InstrumentSpec
    from exchange.cli.runner import (
        alternative_constructions,
        load_dataset,
        load_hypothesis,
        run_research,
    )
    from exchange.registry.store import ExperimentStore

    cfg = _config(config, seed=seed, verbose=verbose)
    if not attacks:
        cfg = _replace(cfg, falsification=_replace(cfg.falsification, enabled=False))

    try:
        hyp = load_hypothesis(hypothesis)
        dataset = load_dataset(source, config=cfg, symbol=symbol)
        alternatives = (
            alternative_constructions(contracts_dir) if contracts_dir is not None else None
        )
        store = ExperimentStore(cfg.registry_root)
        outcome = run_research(
            dataset, hyp, config=cfg,
            instrument=InstrumentSpec(
                symbol=dataset.symbol, multiplier=multiplier, tick_size=tick_size
            ),
            store=store,
            alternative_bars=alternatives,
            scan_first=scan_first,
            record=record,
        )
    except ExchangeError as exc:
        _fail(str(exc))
        return

    verdict = outcome.result.verdict
    colour = {
        "REJECTED": typer.colors.RED,
        "WEAK": typer.colors.YELLOW,
        "PROMISING": typer.colors.CYAN,
        "ROBUST": typer.colors.GREEN,
        "PAPER_TRADING_CANDIDATE": typer.colors.GREEN,
    }.get(verdict.label if verdict else "REJECTED", typer.colors.WHITE)

    _echo("")
    for stage in outcome.result.stages:
        mark = "ok  " if stage.passed else "FAIL"
        _echo(f"  [{mark}] {stage.name:20} {stage.summary}")
    _echo("")
    typer.secho(f"VERDICT: {verdict.label if verdict else 'INCOMPLETE'}", fg=colour, bold=True)
    if outcome.robustness is not None:
        _echo(
            f"attacks: {len(outcome.robustness.survived)}/{len(outcome.robustness.evaluated)} "
            f"survived — {outcome.robustness.grade}; verdict capped at "
            f"{outcome.robustness.verdict_cap.name}"
        )
        if outcome.verdict_label != (verdict.label if verdict else ""):
            typer.secho(
                f"FINAL VERDICT AFTER ATTACKS: {outcome.verdict_label}", fg=typer.colors.YELLOW, bold=True
            )
    if outcome.record is not None:
        _echo(f"recorded as {outcome.record.experiment_id}")
    if outcome.report_path is not None:
        _echo(f"report: {outcome.report_path}")
    if verdict and verdict.rejected:
        _echo(
            "\nA rejection is a completed experiment. It is recorded with the same weight as any "
            "other result, and its trials are charged to the next hypothesis on this data."
        )


# ==========================================================================
# falsify / reproduce / experiments / report
# ==========================================================================


@app.command()
def falsify(
    experiment_id: Annotated[str, typer.Argument()],
    source: Annotated[str | None, typer.Option("--source", help="Override the recorded data source.")] = None,
    contracts_dir: Annotated[Path | None, typer.Option("--contracts")] = None,
    attach: Annotated[bool, typer.Option("--attach/--no-attach", help="Store the report on the record.")] = True,
    config: ConfigOption = None,
) -> None:
    """Attack a recorded experiment and report what survives."""
    from exchange.cli.runner import alternative_constructions, falsify_experiment

    cfg = _config(config)
    try:
        report = falsify_experiment(
            experiment_id, config=cfg, source=source,
            alternative_bars=(
                alternative_constructions(contracts_dir) if contracts_dir is not None else None
            ),
            attach=attach,
        )
    except ExchangeError as exc:
        _fail(str(exc))
        return
    _echo(report.to_markdown())
    if report.failed:
        typer.secho(
            f"{len(report.failed)} attack(s) failed; the verdict for {experiment_id} is capped at "
            f"{report.verdict_cap.name}",
            fg=typer.colors.YELLOW,
        )


@app.command()
def reproduce(
    experiment_id: Annotated[str, typer.Argument()],
    source: Annotated[str | None, typer.Option("--source")] = None,
    config: ConfigOption = None,
) -> None:
    """Re-run a recorded experiment and compare it with the record.

    Exits non-zero on divergence. A result that cannot be reproduced is not
    a result, so this is the command CI runs.
    """
    from exchange.cli.runner import reproduce_experiment

    cfg = _config(config)
    try:
        comparison, _recorded, _fresh = reproduce_experiment(
            experiment_id, config=cfg, source=source
        )
    except ExchangeError as exc:
        _fail(str(exc))
        return
    _echo(comparison.to_markdown())
    if comparison.status == "EXACT":
        typer.secho(f"{experiment_id} reproduced exactly.", fg=typer.colors.GREEN)
    elif comparison.status == "EQUIVALENT":
        typer.secho(f"{experiment_id} reproduced within tolerance.", fg=typer.colors.GREEN)
    elif comparison.status == "INCOMPARABLE":
        _fail(f"{experiment_id}: inputs differ, so this is a different experiment")
    else:
        _fail(f"{experiment_id} DID NOT REPRODUCE: {comparison.detail}")


@experiments_app.command("list")
def experiments_list(
    verdict: Annotated[str | None, typer.Option("--verdict")] = None,
    config: ConfigOption = None,
) -> None:
    """The registry index, newest last."""
    from exchange.registry.store import ExperimentStore

    cfg = _config(config)
    store = ExperimentStore(cfg.registry_root)
    index = store.index()
    if verdict and not index.empty:
        index = index[index["verdict"] == verdict.upper()]
    _frame(index.drop(columns=["hypothesis_fingerprint", "reproduction_hash"], errors="ignore"))
    _echo("")
    _echo(store.to_markdown())
    _echo("Configurations already tested, per dataset:")
    _frame(store.trials_by_dataset())


@experiments_app.command("show")
def experiments_show(
    experiment_id: Annotated[str, typer.Argument()],
    config: ConfigOption = None,
) -> None:
    """One experiment's record."""
    from exchange.registry.store import ExperimentStore

    cfg = _config(config)
    try:
        record = ExperimentStore(cfg.registry_root).load(experiment_id)
    except ExchangeError as exc:
        _fail(str(exc))
        return
    _echo(record.to_markdown())
    _echo("Metrics:")
    for key, value in sorted(record.metrics.items()):
        _echo(f"  {key:32} {value}")
    if record.robustness:
        _echo(
            f"Robustness: {record.robustness.get('score')} "
            f"({record.robustness.get('grade')}), verdict cap "
            f"{record.robustness.get('verdict_cap')}"
        )


@app.command()
def report(
    experiment_id: Annotated[str, typer.Argument()],
    out: Annotated[Path | None, typer.Option("--out")] = None,
    config: ConfigOption = None,
) -> None:
    """Print (or save) a recorded experiment's research report."""
    from exchange.registry.store import ExperimentStore

    cfg = _config(config)
    store = ExperimentStore(cfg.registry_root)
    path = store.directory(experiment_id) / "report.md"
    if not path.is_file():
        _fail(f"no report stored for {experiment_id} ({path} does not exist)")
        return
    text = path.read_text(encoding="utf-8")
    if out:
        Path(out).write_text(text, encoding="utf-8")
        _echo(f"written to {out}")
    else:
        _echo(text)


# ==========================================================================
# controls
# ==========================================================================


@app.command()
def controls(
    path: Annotated[Path | None, typer.Option("--file", help="Controls file (default configs/controls.yaml).")] = None,
    quick: Annotated[bool, typer.Option("--quick", help="Fewer bars and resamples; for CI and Docker builds.")] = False,
    config: ConfigOption = None,
    seed: SeedOption = None,
) -> None:
    """Verify the framework against synthetic data with known answers.

    Exits non-zero if any control fails. This is the calibration of the
    instrument: a negative control that produces a verdict above WEAK is a
    false discovery generated by the framework itself, and a positive
    control that is missed means the pipeline cannot see a known edge — in
    which case nothing it reports as absent can be believed.
    """
    from exchange.cli.runner import run_controls

    cfg = _config(config, seed=seed)
    try:
        outcomes = run_controls(config=cfg, path=path, quick=quick)
    except ExchangeError as exc:
        _fail(str(exc))
        return

    _echo(f"{'role':9} {'control':22} {'verdict':24} {'expected':16} result")
    _echo("-" * 92)
    for outcome in outcomes:
        mark = "PASS" if outcome.passed else "FAIL"
        colour = typer.colors.GREEN if outcome.passed else typer.colors.RED
        typer.secho(
            f"{outcome.role:9} {outcome.kind:22} {outcome.verdict:24} "
            f"{outcome.expectation:16} {mark}",
            fg=colour,
        )
        if outcome.detail and not outcome.passed:
            typer.secho(f"          {outcome.detail}", fg=typer.colors.RED)
    failures = [o for o in outcomes if not o.passed]
    _echo("-" * 92)
    _echo(f"{len(outcomes) - len(failures)}/{len(outcomes)} controls passed")
    if failures:
        _fail(
            f"{len(failures)} control(s) failed. The framework is not calibrated; no result it "
            "produces can be trusted until this passes."
        )


@app.command("research-loop")
def research_loop() -> None:
    """Not implemented, on purpose. Explains what is missing."""
    typer.secho(
        "The autonomous research loop is designed but deliberately not shipped.",
        fg=typer.colors.YELLOW, bold=True,
    )
    _echo("")
    _echo(
        "An agent that proposes hypotheses, runs them and writes to the registry needs, before\n"
        "it runs even once:\n"
        "  * a sandbox: a container with no credentials and no write access outside the\n"
        "    experiment registry and a scratch directory;\n"
        "  * an explicit filesystem and network allowlist, denying both by default;\n"
        "  * CPU, memory and wall-clock budgets per iteration, enforced externally;\n"
        "  * no access to broker credentials or API keys of any kind;\n"
        "  * append-only registry writes, so a loop cannot rewrite an earlier verdict.\n"
        "\n"
        "Until those exist, the loop would be a process with write access to the research record\n"
        "and no limits, which is a worse outcome than not having it. See docs/AI_RESEARCHER.md\n"
        "for the design, and run the stages by hand in the meantime:\n"
        "  exchange scan <source> && exchange research <source> --hypothesis <file>"
    )
    raise typer.Exit(code=2)


def _coerce(value: str):
    """Turn a command-line string into the obvious Python value."""
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def main() -> None:
    """Console-script entry point."""
    try:
        app()
    except ExchangeError as exc:        # a known, explained failure
        typer.secho(f"{type(exc).__name__}: {exc}", fg=typer.colors.RED, err=True)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
