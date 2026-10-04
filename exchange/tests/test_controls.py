"""Synthetic controls: the framework's own calibration.

These are marked ``control`` and run in CI on every pull request. A failure
here does not mean a hypothesis was wrong — it means the measuring
instrument is broken, and nothing else the package reports can be believed
until it passes.
"""

from __future__ import annotations

import pytest

from exchange.cli.runner import run_controls
from exchange.funnel.verdict import VerdictLevel

pytestmark = [pytest.mark.control]


@pytest.fixture(scope="module")
def outcomes(tmp_path_factory):
    from exchange.core.config import build_config

    root = tmp_path_factory.mktemp("controls")
    config = build_config(
        {
            "registry_root": str(root / "experiments"),
            "reports_root": str(root / "reports"),
            "funnel": {"n_bootstrap": 150, "n_monte_carlo": 150},
            "falsification": {"n_resamples": 150},
        }
    )
    return run_controls(config=config, path="configs/controls.yaml", quick=True)


class TestControlSuite:
    def test_every_control_passes(self, outcomes):
        failures = [(o.kind, o.role, o.verdict, o.detail) for o in outcomes if not o.passed]
        assert not failures, f"the framework is not calibrated: {failures}"

    def test_the_suite_covers_all_three_roles(self, outcomes):
        assert {o.role for o in outcomes} == {"negative", "positive", "leakage"}

    def test_no_negative_control_exceeds_weak(self, outcomes):
        """A verdict above WEAK on a series with no structure is a false discovery."""
        for outcome in outcomes:
            if outcome.role == "negative":
                assert VerdictLevel[outcome.verdict] <= VerdictLevel.WEAK, outcome.kind

    def test_positive_controls_are_found(self, outcomes):
        """A known, constructed edge must be visible.

        If it is not, the framework's silence about other series means
        nothing: absence of evidence requires a demonstrated ability to
        find evidence.
        """
        positives = [o for o in outcomes if o.role == "positive"]
        assert positives
        for outcome in positives:
            assert VerdictLevel[outcome.verdict] >= VerdictLevel.WEAK, outcome.kind

    def test_the_planted_lookahead_is_caught(self, outcomes):
        trap = next(o for o in outcomes if o.kind == "lookahead_trap")
        assert trap.passed
        assert trap.verdict == "LookaheadError"

    def test_same_close_execution_is_marked_invalid(self, outcomes):
        control = next(o for o in outcomes if o.kind == "same_close_execution")
        assert control.passed
        assert control.verdict == "invalid_result"

    def test_outcomes_are_serialisable(self, outcomes):
        import json

        json.dumps([o.to_dict() for o in outcomes])
