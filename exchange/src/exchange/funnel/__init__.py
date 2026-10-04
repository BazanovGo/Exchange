"""Research funnel: pre-registered hypotheses, staged validation, verdicts."""

from __future__ import annotations

from exchange.funnel.hypothesis import (
    DEFAULT_FALSIFICATION,
    FalsificationCriterion,
    Hypothesis,
    make_hypothesis,
)
from exchange.funnel.pipeline import FunnelConfig, FunnelResult, StageResult, run_funnel
from exchange.funnel.verdict import Gate, Verdict, VerdictLevel, decide_verdict

__all__ = [
    "DEFAULT_FALSIFICATION",
    "FalsificationCriterion",
    "FunnelConfig",
    "FunnelResult",
    "Gate",
    "Hypothesis",
    "StageResult",
    "Verdict",
    "VerdictLevel",
    "decide_verdict",
    "make_hypothesis",
    "run_funnel",
]
