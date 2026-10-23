"""The command line, exercised through Typer's test runner.

Every command is checked for the two things a CLI must get right: it exits
non-zero when something is wrong, and it says what is wrong. A research
tool that exits zero on bad data will be wired into a pipeline that
silently produces nothing.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from exchange.cli.main import app

runner = CliRunner()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A temporary working directory with a config and a hypothesis file."""
    from exchange.core.config import build_config

    config = build_config(
        {
            "registry_root": "experiments",
            "reports_root": "reports",
            "funnel": {"n_bootstrap": 120, "n_monte_carlo": 120},
            "falsification": {"n_resamples": 120},
        }
    )
    (tmp_path / "configs").mkdir()
    config.save(tmp_path / "configs" / "research.yaml")
    (tmp_path / "hyp.yaml").write_text(
        "\n".join(
            [
                "statement: The trailing five-bar return sign predicts the next bar positively.",
                "mechanism: >",
                "  Slower participants trade after the information arrives, so their flow",
                "  extends the move over several bars rather than one.",
                "expected_regime: Trending regimes; a random walk should refute it.",
                "strategy: ts_momentum",
                "param_space:",
                "  lookback: [3, 5, 10, 20]",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(*args):
    return runner.invoke(app, list(args), catch_exceptions=False)


class TestHelp:
    def test_top_level_help_lists_the_commands(self):
        result = _run("--help")
        assert result.exit_code == 0
        for command in ("doctor", "scan", "research", "reproduce", "controls", "falsify"):
            assert command in result.stdout

    @pytest.mark.parametrize(
        "command",
        ["doctor", "scan", "research", "backtest", "falsify", "reproduce", "report", "controls",
         "strategies", "features"],
    )
    def test_every_command_has_help(self, command):
        result = _run(command, "--help")
        assert result.exit_code == 0
        assert result.stdout.strip()


class TestDoctor:
    def test_doctor_reports_the_environment_and_causality(self, workspace):
        result = _run("doctor")
        assert result.exit_code == 0
        assert "determinism" in result.stdout
        assert "causal" in result.stdout


class TestDataCommands:
    def test_validate_accepts_clean_data(self, workspace):
        from exchange.synthetic.generators import random_walk

        random_walk(n=400, seed=3).bars.to_csv(workspace / "clean.csv")
        result = _run("data", "validate", "clean.csv")
        assert result.exit_code == 0
        assert "content hash" in result.stdout

    def test_validate_exits_non_zero_on_broken_data(self, workspace):
        from exchange.synthetic.generators import random_walk

        bars = random_walk(n=400, seed=3).bars.copy()
        bars.iloc[10, bars.columns.get_loc("high")] = bars["low"].iloc[10] - 1.0
        bars.to_csv(workspace / "broken.csv")
        result = _run("data", "validate", "broken.csv")
        assert result.exit_code == 1
        assert "quality error" in result.stdout.lower() or "ohlc" in result.stdout.lower()

    def test_import_writes_a_dataset_with_provenance(self, workspace):
        result = _run("data", "import", "synthetic:random_walk:5", "--out", "out.csv")
        assert result.exit_code == 0
        assert (workspace / "out.csv").is_file()
        sidecar = workspace / "out.csv.provenance.json"
        assert sidecar.is_file()
        assert json.loads(sidecar.read_text())["symbol"]

    def test_a_missing_file_is_an_error(self, workspace):
        result = _run("data", "validate", "nope.csv")
        assert result.exit_code == 1


class TestInspectionCommands:
    def test_strategies_lists_the_catalogue(self, workspace):
        result = _run("strategies")
        assert result.exit_code == 0
        assert "ts_momentum" in result.stdout
        assert "grid_size" in result.stdout

    def test_features_shows_publication_lags(self, workspace):
        result = _run("features", "--market", "natural_gas")
        assert result.exit_code == 0
        assert "publication_lag_days" in result.stdout
        assert "lookahead" in result.stdout

    def test_features_detail_shows_one_spec(self, workspace):
        result = _run("features", "--detail", "ng_storage")
        assert result.exit_code == 0
        assert "mechanism" in result.stdout

    def test_an_unknown_feature_is_an_error(self, workspace):
        assert _run("features", "--detail", "ng_vibes").exit_code == 1

    def test_scan_runs_on_a_synthetic_series(self, workspace):
        result = _run("scan", "synthetic:random_walk:7")
        assert result.exit_code == 0
        assert "variance" in result.stdout.lower()

    def test_backtest_warns_that_it_is_not_a_research_result(self, workspace):
        result = _run(
            "backtest", "synthetic:random_walk:7", "ts_momentum",
            "--param", "lookback=20", "--multiplier", "1000",
        )
        assert result.exit_code == 0
        assert "no out-of-sample window" in result.stdout

    def test_a_malformed_parameter_is_an_error(self, workspace):
        result = _run("backtest", "synthetic:random_walk:7", "ts_momentum", "--param", "lookback")
        assert result.exit_code == 1


class TestMoexSource:
    @pytest.fixture
    def algopack(self, workspace, monkeypatch):
        """A miniature AlgoPack directory under $EXCHANGE_MOEX_DATA."""
        import numpy as np
        import pandas as pd

        folder = workspace / "algopack" / "ng" / "60m"
        folder.mkdir(parents=True)
        stamps = pd.date_range("2025-01-03 10:00", periods=600, freq="h")
        close = 3.0 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.004, 600)))
        pd.DataFrame(
            {
                "ticker": "ng",
                "secid": ["NGF5"] * 300 + ["NGG5"] * 300,
                "begin": stamps,
                "end": stamps + pd.Timedelta(minutes=59, seconds=59),
                "open": close, "high": close * 1.002, "low": close * 0.998, "close": close,
                "volume": 1_000, "value": 0,
            }
        ).to_csv(folder / "2025.csv", index=False)
        monkeypatch.setenv("EXCHANGE_MOEX_DATA", str(workspace / "algopack"))
        return workspace / "algopack"

    def test_the_spec_string_is_the_recorded_source(self, algopack):
        from exchange.cli.runner import load_dataset
        from exchange.core.config import build_config

        dataset = load_dataset("moex:ng:60m", config=build_config({}))
        assert dataset.provenance["source"] == "moex:ng:60m"
        assert dataset.provenance["continuous"]["adjustment"] == "ratio"

    def test_the_contract_specification_is_the_default_instrument(self, algopack):
        from exchange.cli.runner import default_instrument, load_dataset
        from exchange.core.config import build_config

        instrument = default_instrument(load_dataset("moex:ng", config=build_config({})))
        assert (instrument.multiplier, instrument.tick_size) == (100.0, 0.001)
        assert instrument.allow_fractional is False

    def test_an_unset_data_directory_is_an_error(self, workspace, monkeypatch):
        monkeypatch.delenv("EXCHANGE_MOEX_DATA", raising=False)
        result = _run("scan", "moex:ng:60m")
        assert result.exit_code == 1
        assert "EXCHANGE_MOEX_DATA" in result.output

    def test_a_malformed_spec_is_an_error(self, algopack):
        assert _run("scan", "moex:").exit_code == 1
        assert _run("scan", "moex:ng:60m:ratio:extra").exit_code == 1

    def test_scan_runs_on_algopack_files(self, algopack):
        result = _run("scan", "moex:ng:60m")
        assert result.exit_code == 0
        assert "variance" in result.stdout.lower()


class TestResearchFlow:
    @pytest.mark.slow
    def test_research_records_reports_and_reproduces(self, workspace):
        research = _run(
            "research",
            "synthetic:signal_plus_noise:7:edge_per_bar=0.003,lookback=5",
            "--hypothesis", "hyp.yaml",
            "--multiplier", "1000",
            "--no-scan",
        )
        assert research.exit_code == 0
        assert "VERDICT" in research.stdout

        experiment_id = next(
            line.split()[-1] for line in research.stdout.splitlines() if "recorded as" in line
        )
        assert (workspace / "experiments" / experiment_id / "report.md").is_file()

        listing = _run("experiments", "list")
        assert listing.exit_code == 0
        assert experiment_id in listing.stdout

        show = _run("experiments", "show", experiment_id)
        assert show.exit_code == 0
        assert "oos_sharpe" in show.stdout

        report = _run("report", experiment_id)
        assert report.exit_code == 0
        assert "What could make this result false?" in report.stdout

        reproduce = _run("reproduce", experiment_id)
        assert reproduce.exit_code == 0
        assert "reproduced" in reproduce.stdout

    @pytest.mark.slow
    def test_a_rejection_still_records_an_experiment(self, workspace):
        result = _run(
            "research", "synthetic:random_walk:11",
            "--hypothesis", "hyp.yaml", "--multiplier", "10000", "--no-scan",
        )
        assert result.exit_code == 0
        assert "REJECTED" in result.stdout
        assert "recorded as" in result.stdout
        assert "completed experiment" in result.stdout

    def test_a_missing_hypothesis_file_is_an_error(self, workspace):
        result = _run("research", "synthetic:random_walk:11", "--hypothesis", "nope.yaml")
        assert result.exit_code == 1

    def test_an_unknown_experiment_is_an_error(self, workspace):
        assert _run("experiments", "show", "EXP-20260902-000999").exit_code == 1
        assert _run("report", "EXP-20260902-000999").exit_code == 1


class TestControlsCommand:
    @pytest.mark.slow
    @pytest.mark.control
    def test_controls_pass_and_exit_zero(self, workspace):
        import shutil
        from pathlib import Path

        source = Path(__file__).resolve().parents[1] / "configs" / "controls.yaml"
        shutil.copy(source, workspace / "configs" / "controls.yaml")
        result = _run("controls", "--quick", "--file", "configs/controls.yaml")
        assert result.exit_code == 0
        assert "controls passed" in result.stdout
        assert "FAIL" not in result.stdout


class TestResearchLoop:
    def test_the_loop_explains_why_it_is_not_shipped(self):
        """The command exists to state the security requirement, not to run."""
        result = _run("research-loop")
        assert result.exit_code == 2
        assert "sandbox" in result.stdout
        assert "credentials" in result.stdout
