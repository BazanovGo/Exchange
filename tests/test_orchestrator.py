import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchestrator.run_research_loop import (  # noqa: E402
    LoopState,
    decide,
    idea_similarity,
    is_repeat,
    load_summary,
    parse_args,
    run_loop,
    stop_reason,
    validate_iteration,
)

BASE_CFG = Path(__file__).resolve().parents[1] / "orchestrator" / "config.yaml"


def make_env(tmp_path, **cfg_overrides):
    cfg = yaml.safe_load(BASE_CFG.read_text())
    cfg["results_dir"] = str(tmp_path / "research")
    cfg["knowledge_base"] = str(tmp_path / "kb.md")
    cfg["research_log"] = str(tmp_path / "research_log.md")
    cfg["state_file"] = str(tmp_path / "state.json")
    cfg.update(cfg_overrides)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True))
    return cfg_path


def run(tmp_path, cfg_path, **cli):
    argv = ["--dry-run", "--config", str(cfg_path)]
    for key, value in cli.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    run_loop(parse_args(argv))
    return json.loads((tmp_path / "state.json").read_text())


def test_idea_similarity_and_repeat_guard():
    a = "zscore reversion on structural series"
    assert idea_similarity(a, a) == 1.0
    assert idea_similarity(a, "donchian breakout with atr trailing") < 0.2
    assert is_repeat(a, [a], threshold=0.6)
    assert not is_repeat("keltner volatility breakout", [a], threshold=0.6)


def test_decide_thresholds():
    state = LoopState(best_score=0.5)
    accepted, _ = decide({"composite_score": 0.56}, state, 0.03)
    rejected, _ = decide({"composite_score": 0.51}, state, 0.03)
    assert accepted == "accepted"
    assert rejected == "rejected"


def test_state_roundtrip(tmp_path):
    state = LoopState(iteration=3, best_score=0.7, history=[{"iteration": "iteration_003"}])
    state.save(tmp_path / "s.json")
    loaded = LoopState.load(tmp_path / "s.json")
    assert loaded.iteration == 3
    assert loaded.best_score == 0.7


def test_validate_and_summary(tmp_path):
    missing = validate_iteration(tmp_path)
    assert "summary.json" in missing and "metrics.csv" in missing

    (tmp_path / "summary.json").write_text(json.dumps({"hypothesis_id": "X"}))
    assert load_summary(tmp_path) is None  # required fields missing


def test_stop_reasons():
    cfg = yaml.safe_load(BASE_CFG.read_text())
    assert stop_reason(LoopState(iteration=100), cfg, 100, 15)
    assert "patience" in stop_reason(LoopState(no_improvement_streak=15), cfg, 100, 15)
    assert "стабильны" in stop_reason(LoopState(stable_streak=5), cfg, 100, 15)
    assert "повторяют" in stop_reason(LoopState(repeat_streak=3), cfg, 100, 15)
    assert stop_reason(LoopState(iteration=1), cfg, 100, 15) == ""


def test_dry_run_stops_on_stability(tmp_path):
    cfg_path = make_env(tmp_path)
    state = run(tmp_path, cfg_path, max_iterations=20, patience=10, min_improvement=0.03)
    assert state["stopped"]
    assert "стабильны" in state["stop_reason"]
    accepted = list((tmp_path / "research" / "accepted").iterdir())
    rejected = [p for p in (tmp_path / "research" / "rejected").iterdir() if p.is_dir()]
    assert len(accepted) >= 3 and len(rejected) >= 1
    # every routed iteration keeps its full artifact contract
    for it in accepted + rejected:
        assert (it / "summary.json").exists() and (it / "metrics.csv").exists()
    assert (tmp_path / "research_log.md").exists()
    assert "Orchestrator decision" in (accepted[0] / "decision.md").read_text()
    assert (tmp_path / "kb.md").exists()  # KB получает заметки оркестратора


def test_dry_run_stops_on_patience(tmp_path):
    # Disable the stability stop so the score plateau triggers patience.
    cfg_path = make_env(tmp_path, stability={
        "min_oos_sharpe": 99, "min_wf_efficiency": 99,
        "max_mc_ruin_prob": 0.0, "required_streak": 3,
    })
    state = run(tmp_path, cfg_path, max_iterations=30, patience=3, min_improvement=0.05)
    assert state["stopped"]
    assert "подряд" in state["stop_reason"]
    # rejected summary powers the repetition guard of future prompts
    rejected_summary = json.loads((tmp_path / "research" / "rejected" / "summary.json").read_text())
    assert len(rejected_summary) >= 3
    assert all("idea" in e and "reason" in e for e in rejected_summary)


def test_dry_run_repeat_streak_counted(tmp_path):
    cfg_path = make_env(
        tmp_path,
        stability={"min_oos_sharpe": 99, "min_wf_efficiency": 99,
                   "max_mc_ruin_prob": 0.0, "required_streak": 3},
        repetition={"jaccard_threshold": 0.6, "streak_stop": 2},
    )
    state = run(tmp_path, cfg_path, max_iterations=30, patience=10, min_improvement=0.05)
    assert state["stopped"]
    # the synthetic idea stream repeats from iteration 7 on
    assert "повторяют" in state["stop_reason"] or "подряд" in state["stop_reason"]


def test_resume_after_stop_is_noop(tmp_path):
    cfg_path = make_env(tmp_path)
    state = run(tmp_path, cfg_path, max_iterations=20, patience=10, min_improvement=0.03)
    iterations_done = state["iteration"]
    state2 = run(tmp_path, cfg_path, max_iterations=20, patience=10, min_improvement=0.03)
    assert state2["iteration"] == iterations_done  # stopped loop does not restart
