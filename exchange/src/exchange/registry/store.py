"""The experiment store.

A directory on disk, one folder per experiment, plus a flat index:

    experiments/
        index.csv
        EXP-20260902-000157/
            record.json          <- the complete, reproducible record
            report.md            <- the human-readable research report
            sweep.csv            <- in-sample grid (no OOS column, by design)
            stress.csv, cpcv.csv, walk_forward.csv, regimes.csv
            robustness.json      <- falsification attacks, if run

Why the ledger matters more than the index
-------------------------------------------
:meth:`ExperimentStore.prior_trials` sums the trials already spent on a
dataset and hands the number to the next hypothesis, which passes it to the
Deflated Sharpe Ratio. That is the difference between a research programme
and a sequence of unrelated backtests: the fortieth idea tested on the same
natural-gas series must clear a higher bar than the first, because forty
draws from noise produce a better best than one does. Without a persistent
ledger every researcher starts again at "this is my first hypothesis",
which is how a team collectively tests a thousand configurations and
reports each as a single pre-registered test.

The store keeps rejections. They are the cheapest and most reusable result
a research programme produces: a rejected hypothesis never has to be tested
again, and the rejection rate is the only honest indicator that the funnel
is doing its job. A store whose rejection rate is low is not a store full
of good ideas.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Mapping
from pathlib import Path

import pandas as pd

from exchange.core.errors import ConfigError
from exchange.core.logging import get_logger
from exchange.registry.experiment import (
    ExperimentRecord,
    make_experiment_id,
    validate_record,
)

logger = get_logger(__name__)

INDEX_NAME = "index.csv"
RECORD_NAME = "record.json"


class ExperimentStore:
    """Append-only registry of experiments on disk."""

    def __init__(self, root: str | Path = "experiments") -> None:
        self.root = Path(root)

    # -- layout ----------------------------------------------------------

    @property
    def index_path(self) -> Path:
        return self.root / INDEX_NAME

    def directory(self, experiment_id: str) -> Path:
        return self.root / experiment_id

    def record_path(self, experiment_id: str) -> Path:
        return self.directory(experiment_id) / RECORD_NAME

    def exists(self, experiment_id: str) -> bool:
        return self.record_path(experiment_id).is_file()

    # -- identifiers -----------------------------------------------------

    def next_id(self) -> str:
        """The next sequential id.

        The counter is global to the store rather than per day, so ids sort
        chronologically and never collide when several runs start in the
        same second.
        """
        existing = sorted(self.ids())
        sequence = 0
        if existing:
            sequence = max(int(name.split("-")[-1]) for name in existing) + 1
        return make_experiment_id(sequence)

    def ids(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(
            path.name
            for path in self.root.iterdir()
            if path.is_dir() and (path / RECORD_NAME).is_file()
        )

    # -- writing ---------------------------------------------------------

    def save(
        self,
        record: ExperimentRecord,
        *,
        report: str | None = None,
        tables: Mapping[str, pd.DataFrame] | None = None,
        overwrite: bool = False,
    ) -> Path:
        """Write a record and its artefacts, then update the index."""
        validate_record(record)
        directory = self.directory(record.experiment_id)
        if directory.exists() and not overwrite:
            raise ConfigError(
                f"{record.experiment_id} already exists in {self.root}; experiments are "
                "append-only — register a new one rather than editing a recorded result"
            )
        directory.mkdir(parents=True, exist_ok=True)

        for name, frame in (tables or {}).items():
            if frame is None or getattr(frame, "empty", True):
                continue
            path = directory / f"{name}.csv"
            frame.to_csv(path, index=True)
            record.artefacts[name] = path.name

        if report:
            (directory / "report.md").write_text(report, encoding="utf-8")
            record.artefacts["report"] = "report.md"

        if record.robustness:
            (directory / "robustness.json").write_text(
                json.dumps(record.robustness, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
            record.artefacts["robustness"] = "robustness.json"

        path = self.record_path(record.experiment_id)
        path.write_text(
            json.dumps(record.to_dict(), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        self._append_index(record)
        logger.info(
            "recorded %s: %s on %s -> %s",
            record.experiment_id,
            record.hypothesis_id,
            record.dataset.get("symbol", "?"),
            record.verdict_label,
        )
        return path

    def attach_robustness(self, experiment_id: str, robustness: Mapping) -> Path:
        """Attach a falsification report to an existing record.

        Permitted, unlike editing a result: the attacks are run after the
        funnel and only ever *lower* a verdict, so attaching them cannot
        improve a recorded outcome.
        """
        record = self.load(experiment_id)
        record.robustness = dict(robustness)
        return self.save(record, overwrite=True)

    def _append_index(self, record: ExperimentRecord) -> None:
        row = record.index_row()
        self.root.mkdir(parents=True, exist_ok=True)
        rows = []
        if self.index_path.is_file():
            with self.index_path.open(newline="", encoding="utf-8") as handle:
                rows = [r for r in csv.DictReader(handle) if r.get("experiment_id") != row["experiment_id"]]
        rows.append({k: ("" if v is None else v) for k, v in row.items()})
        fields = list(row)
        with self.index_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for item in sorted(rows, key=lambda r: r.get("experiment_id", "")):
                writer.writerow({k: item.get(k, "") for k in fields})

    def rebuild_index(self) -> Path:
        """Regenerate the index from the records, which are the source of truth."""
        if self.index_path.is_file():
            self.index_path.unlink()
        for record in self.records():
            self._append_index(record)
        return self.index_path

    # -- reading ---------------------------------------------------------

    def load(self, experiment_id: str) -> ExperimentRecord:
        path = self.record_path(experiment_id)
        if not path.is_file():
            raise ConfigError(f"no experiment {experiment_id!r} in {self.root}")
        return ExperimentRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def records(self) -> Iterator[ExperimentRecord]:
        for experiment_id in self.ids():
            try:
                yield self.load(experiment_id)
            except Exception as exc:  # one bad record must not hide the rest
                logger.warning("skipping unreadable record %s: %s", experiment_id, exc)

    def index(self) -> pd.DataFrame:
        """The index as a frame, rebuilt from records if the file is missing."""
        rows = [r.index_row() for r in self.records()]
        if not rows:
            return pd.DataFrame(columns=["experiment_id"]).set_index("experiment_id")
        return pd.DataFrame(rows).set_index("experiment_id").sort_index()

    def latest(self, n: int = 1) -> list[ExperimentRecord]:
        ids = self.ids()[-n:]
        return [self.load(i) for i in ids]

    def find(
        self,
        *,
        hypothesis_fingerprint: str | None = None,
        data_hash: str | None = None,
        strategy: str | None = None,
        verdict: str | None = None,
        reproduction_hash: str | None = None,
    ) -> list[ExperimentRecord]:
        out = []
        for record in self.records():
            if hypothesis_fingerprint and record.hypothesis_fingerprint != hypothesis_fingerprint:
                continue
            if data_hash and record.data_hash != data_hash:
                continue
            if strategy and record.hypothesis.get("strategy") != strategy:
                continue
            if verdict and record.verdict_label != verdict:
                continue
            if reproduction_hash and record.reproduction_hash != reproduction_hash:
                continue
            out.append(record)
        return out

    # -- the multiple-testing ledger -------------------------------------

    def prior_trials(self, data_hash: str, *, exclude: str | None = None) -> int:
        """Configurations already tested on this exact data.

        This is the number the next hypothesis must carry into its Deflated
        Sharpe Ratio. It counts *configurations*, not experiments: a single
        experiment that swept a 400-cell grid spent 400 trials, and
        pretending otherwise is how a grid search is laundered into a
        pre-registered test.
        """
        if not data_hash:
            return 0
        total = 0
        for record in self.records():
            if record.experiment_id == exclude or record.data_hash != data_hash:
                continue
            total += int(record.hypothesis.get("n_configurations", 0) or 0)
        return total

    def trials_by_dataset(self) -> pd.DataFrame:
        rows: dict[str, dict] = {}
        for record in self.records():
            key = record.data_hash
            if not key:
                continue
            entry = rows.setdefault(
                key,
                {
                    "data_hash": key[:16],
                    "symbol": record.dataset.get("symbol", ""),
                    "n_experiments": 0,
                    "n_configurations": 0,
                },
            )
            entry["n_experiments"] += 1
            entry["n_configurations"] += int(record.hypothesis.get("n_configurations", 0) or 0)
        if not rows:
            return pd.DataFrame(columns=["data_hash", "symbol", "n_experiments", "n_configurations"])
        return pd.DataFrame(list(rows.values())).sort_values("n_configurations", ascending=False)

    # -- programme health ------------------------------------------------

    def verdict_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records():
            counts[record.verdict_label] = counts.get(record.verdict_label, 0) + 1
        return dict(sorted(counts.items()))

    def rejection_rate(self) -> float:
        """Share of experiments that ended REJECTED.

        Read it as a diagnostic of the *funnel*, not of the ideas. A
        programme that rejects almost nothing has either found a remarkable
        source of alpha or disabled its own gates, and the second is far
        more common.
        """
        counts = self.verdict_counts()
        total = sum(counts.values())
        if total == 0:
            return float("nan")
        return counts.get("REJECTED", 0) / total

    def summary(self) -> dict:
        counts = self.verdict_counts()
        total = sum(counts.values())
        configurations = sum(
            int(r.hypothesis.get("n_configurations", 0) or 0) for r in self.records()
        )
        return {
            "root": str(self.root),
            "n_experiments": total,
            "n_configurations_tested": configurations,
            "verdicts": counts,
            "rejection_rate": self.rejection_rate(),
            "datasets": int(self.trials_by_dataset().shape[0]),
        }

    def to_markdown(self) -> str:
        info = self.summary()
        lines = [
            "## Experiment registry",
            "",
            f"- Store: `{info['root']}`",
            f"- Experiments: **{info['n_experiments']}**",
            f"- Configurations tested in total: **{info['n_configurations_tested']}**",
            f"- Distinct datasets: **{info['datasets']}**",
            f"- Rejection rate: **{info['rejection_rate']:.0%}**"
            if info["n_experiments"]
            else "- Rejection rate: n/a",
            "",
        ]
        if info["verdicts"]:
            lines += ["| verdict | count |", "|---|---|"]
            lines += [f"| {k} | {v} |" for k, v in info["verdicts"].items()]
        return "\n".join(lines) + "\n"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ExperimentStore({str(self.root)!r}, n={len(self.ids())})"
