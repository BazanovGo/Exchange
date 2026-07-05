#!/usr/bin/env python3
"""External orchestrator for the autonomous trading-hypothesis research loop.

Drives Claude Code iteration by iteration so the search does not depend on
a single session surviving:

    prompt -> Claude Code run -> artifact contract check -> score ->
    accept/reject routing -> knowledge-base & log update -> next prompt

Every iteration must produce measurable artifacts in
``research/pending/iteration_NNN/`` (hypothesis.md, implementation.py,
metrics.csv, summary.json, ...). An iteration without metrics is a failure
and is retried once more with the strict prompt before being logged as
failed.

Usage::

    python orchestrator/run_research_loop.py \
        --max-iterations 100 --patience 15 --min-improvement 0.03 \
        --prompt-file prompts/research_hypothesis.md --results-dir research/

``--dry-run`` exercises the full loop mechanics (directories, state,
decisions, stopping criteria, log) with synthetic iterations instead of
calling Claude Code.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_ARTIFACTS = ("hypothesis.md", "implementation.py", "metrics.csv", "summary.json", "decision.md")
SUMMARY_FIELDS = ("hypothesis_id", "idea", "composite_score", "oos_sharpe",
                  "wf_efficiency", "mc_ruin_prob", "verdict")


# --------------------------------------------------------------------------
# configuration & state
# --------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _rel(path: Path) -> str:
    """Path relative to the repo root when possible, absolute otherwise."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


@dataclass
class LoopState:
    """Persistent orchestrator state (survives restarts)."""

    iteration: int = 0
    best_score: float = 0.0
    best_iteration: str = ""
    best_summary: dict = field(default_factory=dict)
    no_improvement_streak: int = 0
    repeat_streak: int = 0
    stable_streak: int = 0
    stopped: bool = False
    stop_reason: str = ""
    history: list = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> LoopState:
        if path.exists():
            return cls(**json.loads(path.read_text()))
        return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))


# --------------------------------------------------------------------------
# context for prompt generation (repetition guard included)
# --------------------------------------------------------------------------

def read_knowledge_base(path: Path, max_chars: int = 6000) -> str:
    if not path.exists():
        return "(база знаний ещё пуста)"
    text = path.read_text()
    return text[:max_chars] + ("\n... (truncated)" if len(text) > max_chars else "")


def rejected_digest(results_dir: Path, max_items: int = 40) -> tuple[str, list[str]]:
    """Digest of rejected ideas: prompt text + raw idea strings for the guard."""
    summary_path = results_dir / "rejected" / "summary.json"
    entries: list[dict] = []
    if summary_path.exists():
        entries = json.loads(summary_path.read_text())
    lines = [f"- [{e.get('iteration', '?')}] {e.get('idea', '')} — {e.get('reason', '')}"
             for e in entries[-max_items:]]
    return ("\n".join(lines) or "(пока нет)"), [e.get("idea", "") for e in entries]


def _idea_tokens(text: str) -> set[str]:
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split() if len(w) > 3}


def idea_similarity(a: str, b: str) -> float:
    """Jaccard overlap of idea tokens — the cheap repetition detector."""
    ta, tb = _idea_tokens(a), _idea_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def is_repeat(idea: str, known_ideas: list[str], threshold: float) -> bool:
    return any(idea_similarity(idea, known) >= threshold for known in known_ideas if known)


def build_prompt(template_path: Path, context: dict) -> str:
    template = template_path.read_text()
    # format_map with a default so stray {tokens} in templates never crash.
    class _Safe(dict):
        def __missing__(self, key):  # noqa: D105
            return "{" + key + "}"
    return template.format_map(_Safe(**context))


# --------------------------------------------------------------------------
# running one iteration
# --------------------------------------------------------------------------

def run_claude(prompt: str, cfg: dict, model: str, timeout: int) -> int:
    command = [part.format(prompt=prompt, model=model) for part in cfg["claude_command"]]
    print(f"  -> claude code ({model}), timeout {timeout}s")
    try:
        proc = subprocess.run(command, cwd=ROOT, timeout=timeout,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"  !! claude exited {proc.returncode}: {proc.stderr[-500:]}")
        return proc.returncode
    except subprocess.TimeoutExpired:
        print("  !! claude timed out")
        return -1
    except FileNotFoundError:
        print("  !! claude CLI not found on PATH")
        return -2


def synthesize_iteration(iter_dir: Path, iteration: int) -> None:
    """--dry-run stand-in for a Claude run: deterministic fake artifacts.

    Scores rise early and then plateau/repeat so every stopping criterion
    can be observed without spending a single model token.
    """
    scores = [0.42, 0.55, 0.61, 0.635, 0.64, 0.638, 0.641, 0.639]
    score = scores[min(iteration - 1, len(scores) - 1)]
    ideas = [
        "bollinger squeeze breakout with atr expansion",
        "vwap trend with adx confirmation",
        "donchian pullback with rsi filter",
        "zscore reversion on structural series",
        "williams percent r reversion variant",
        "zscore reversion on structural series with band exit",
        "zscore reversion on structural series",   # repeats begin
        "zscore reversion on structural series",
    ]
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "plots").mkdir(exist_ok=True)
    (iter_dir / "hypothesis.md").write_text(f"# dry-run hypothesis {iteration}\n")
    (iter_dir / "implementation.py").write_text("# dry-run implementation\n")
    (iter_dir / "metrics.csv").write_text("param,is_sharpe,oos_sharpe\n1,1.0,0.9\n")
    (iter_dir / "trades.csv").write_text("entry,exit,ret\n")
    (iter_dir / "notebook.ipynb").write_text('{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}')
    (iter_dir / "decision.md").write_text("dry-run decision\n")
    (iter_dir / "summary.json").write_text(json.dumps({
        "hypothesis_id": f"DRY{iteration:03d}",
        "idea": ideas[min(iteration - 1, len(ideas) - 1)],
        "strategy_name": "dry_run",
        "composite_score": score,
        "oos_sharpe": 0.9 + score, "wf_efficiency": 0.6,
        "wf_profitable_windows": 1.0, "mc_ruin_prob": 0.0,
        "verdict": "viable", "reasons": [],
    }))


def validate_iteration(iter_dir: Path) -> list[str]:
    """Return the list of missing required artifacts (empty = valid)."""
    return [name for name in REQUIRED_ARTIFACTS if not (iter_dir / name).exists()]


def load_summary(iter_dir: Path) -> dict | None:
    try:
        summary = json.loads((iter_dir / "summary.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if any(f not in summary for f in SUMMARY_FIELDS):
        return None
    try:
        summary["composite_score"] = float(summary["composite_score"])
    except (TypeError, ValueError):
        return None
    return summary


# --------------------------------------------------------------------------
# decision, routing, logging
# --------------------------------------------------------------------------

def decide(summary: dict, state: LoopState, min_improvement: float) -> tuple[str, str]:
    score = summary["composite_score"]
    improvement = score - state.best_score
    if improvement >= min_improvement:
        return "accepted", f"composite {score:.3f} улучшил лучший {state.best_score:.3f} на {improvement:.3f}"
    return "rejected", (
        f"composite {score:.3f} не дал улучшения >= {min_improvement} "
        f"против лучшего {state.best_score:.3f}"
    )


def route_iteration(iter_dir: Path, decision: str, results_dir: Path) -> Path:
    target = results_dir / decision / iter_dir.name
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(iter_dir), str(target))
    return target


def update_rejected_summary(results_dir: Path, iteration_name: str, summary: dict, reason: str) -> None:
    path = results_dir / "rejected" / "summary.json"
    entries = json.loads(path.read_text()) if path.exists() else []
    entries.append({
        "iteration": iteration_name,
        "hypothesis_id": summary.get("hypothesis_id", ""),
        "idea": summary.get("idea", ""),
        "score": summary.get("composite_score"),
        "reason": reason,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False))


def append_decision(iter_dir: Path, decision: str, reason: str) -> None:
    with open(iter_dir / "decision.md", "a") as fh:
        fh.write(f"\n\n## Orchestrator decision\n\n**{decision}** — {reason}\n")


def append_log(log_path: Path, iteration_name: str, summary: dict | None,
               decision: str, reason: str, next_hint: str) -> None:
    if not log_path.exists():
        log_path.write_text(
            "# Журнал внешнего исследовательского цикла\n\n"
            "| итерация | идея | score | решение | причина | что дальше |\n"
            "|---|---|---|---|---|---|\n"
        )
    idea = (summary or {}).get("idea", "—")
    score = (summary or {}).get("composite_score")
    score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
    row = (f"| {iteration_name} | {idea} | {score_s} | {decision} | "
           f"{reason} | {next_hint} |\n")
    with open(log_path, "a") as fh:
        fh.write(row)


def append_kb_note(kb_path: Path, iteration_name: str, summary: dict | None,
                   decision: str, reason: str) -> None:
    header = "## Журнал внешнего оркестратора"
    text = kb_path.read_text() if kb_path.exists() else "# База знаний\n"
    if header not in text:
        text += f"\n{header}\n\n"
    idea = (summary or {}).get("idea", "—")
    text += f"* {iteration_name}: {idea} → **{decision}** ({reason})\n"
    kb_path.write_text(text)


# --------------------------------------------------------------------------
# stopping criteria
# --------------------------------------------------------------------------

def check_stability(summary: dict, crit: dict) -> bool:
    """Best-so-far is stable OOS, WF holds, MC risk is acceptable."""
    try:
        return (
            float(summary.get("oos_sharpe", 0)) >= crit["min_oos_sharpe"]
            and float(summary.get("wf_efficiency", 0)) >= crit["min_wf_efficiency"]
            and float(summary.get("mc_ruin_prob", 1)) <= crit["max_mc_ruin_prob"]
        )
    except (TypeError, ValueError):
        return False


def stop_reason(state: LoopState, cfg: dict, max_iterations: int, patience: int) -> str:
    """Return a human-readable stop reason, or '' to continue."""
    if state.iteration >= max_iterations:
        return f"достигнут лимит итераций ({max_iterations})"
    if state.no_improvement_streak >= patience:
        return f"нет улучшений {state.no_improvement_streak} итераций подряд (patience={patience})"
    stability = cfg["stability"]
    if state.stable_streak >= stability["required_streak"]:
        return (
            "лучшие стратегии стабильны: OOS Sharpe >= "
            f"{stability['min_oos_sharpe']}, WF eff >= {stability['min_wf_efficiency']}, "
            f"MC ruin <= {stability['max_mc_ruin_prob']} на протяжении {state.stable_streak} итераций"
        )
    repetition = cfg["repetition"]
    if state.repeat_streak >= repetition["streak_stop"]:
        return f"новые гипотезы повторяют проверенные идеи ({state.repeat_streak} подряд)"
    return ""


# --------------------------------------------------------------------------
# main loop
# --------------------------------------------------------------------------

def run_loop(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config))
    results_dir = (ROOT / (args.results_dir or cfg["results_dir"])).resolve()
    prompts_dir = ROOT / cfg["prompts_dir"]
    kb_path = ROOT / cfg["knowledge_base"]
    log_path = ROOT / cfg["research_log"]
    state_path = ROOT / (args.state_file or cfg["state_file"])
    model = args.model or cfg["model"]
    max_iterations = args.max_iterations or cfg["max_iterations"]
    patience = args.patience or cfg["patience"]
    min_improvement = args.min_improvement if args.min_improvement is not None else cfg["min_improvement"]
    base_prompt = Path(args.prompt_file) if args.prompt_file else prompts_dir / "research_hypothesis.md"

    for sub in ("pending", "accepted", "rejected"):
        (results_dir / sub).mkdir(parents=True, exist_ok=True)

    state = LoopState.load(state_path)
    if state.stopped:
        print(f"Цикл уже остановлен: {state.stop_reason}. Удалите {state_path} для перезапуска.")
        return 0

    last_decision = state.history[-1]["decision"] if state.history else ""

    while True:
        reason = stop_reason(state, cfg, max_iterations, patience)
        if reason:
            finalize(state, cfg, reason, model, args.dry_run, prompts_dir, state_path)
            return 0

        state.iteration += 1
        iteration_name = f"iteration_{state.iteration:03d}"
        iter_dir = results_dir / "pending" / iteration_name
        print(f"\n=== {iteration_name} (best={state.best_score:.3f}, "
              f"no-improve={state.no_improvement_streak}, repeats={state.repeat_streak}) ===")

        kb_text = read_knowledge_base(kb_path)
        rejected_text, rejected_ideas = rejected_digest(results_dir)
        accepted_ideas = [h.get("idea", "") for h in state.history if h.get("decision") == "accepted"]

        template = prompts_dir / "improve_strategy.md" if last_decision == "accepted" else base_prompt
        context = {
            "iteration_id": iteration_name,
            "out_dir": _rel(iter_dir),
            "best_score": f"{state.best_score:.3f}",
            "best_iteration": state.best_iteration or "—",
            "best_summary": json.dumps(state.best_summary, ensure_ascii=False, indent=2),
            "min_improvement": min_improvement,
            "knowledge_base": kb_text,
            "rejected_summary": rejected_text,
            "missing": "",
        }

        # --- run (with strict-prompt retries on missing artifacts) ---------
        summary = None
        for _attempt in range(cfg["retries_per_iteration"] + 1):
            iter_dir.mkdir(parents=True, exist_ok=True)
            if args.dry_run:
                synthesize_iteration(iter_dir, state.iteration)
            else:
                prompt = build_prompt(template, context)
                run_claude(prompt, cfg, model, cfg["iteration_timeout_sec"])
            missing = validate_iteration(iter_dir)
            summary = load_summary(iter_dir) if not missing else None
            if summary is not None:
                break
            print(f"  !! артефакты неполны (missing: {missing or 'summary.json invalid'}) — строгий повтор")
            template = prompts_dir / "reject_strategy.md"
            context["missing"] = ", ".join(missing) or "summary.json невалиден"

        if summary is None:
            decision, reason_text = "rejected", "итерация не создала валидные метрики после повторов"
            append_log(log_path, iteration_name, None, decision, reason_text,
                       "строгий контракт артефактов в следующем промпте")
            update_rejected_summary(results_dir, iteration_name, {}, reason_text)
            if any(iter_dir.iterdir()):
                route_iteration(iter_dir, "rejected", results_dir)
            else:
                iter_dir.rmdir()
            state.no_improvement_streak += 1
            state.history.append({"iteration": iteration_name, "decision": "failed", "idea": ""})
            last_decision = "failed"
            state.save(state_path)
            continue

        # --- repetition guard ----------------------------------------------
        repeated = is_repeat(summary["idea"], rejected_ideas + accepted_ideas,
                             cfg["repetition"]["jaccard_threshold"])
        state.repeat_streak = state.repeat_streak + 1 if repeated else 0

        # --- decision --------------------------------------------------------
        decision, reason_text = decide(summary, state, min_improvement)
        if repeated and decision == "rejected":
            reason_text += "; идея повторяет уже проверенную"
        append_decision(iter_dir, decision, reason_text)

        if decision == "accepted":
            state.best_score = summary["composite_score"]
            state.best_iteration = iteration_name
            state.best_summary = summary
            state.no_improvement_streak = 0
        else:
            state.no_improvement_streak += 1
            update_rejected_summary(results_dir, iteration_name, summary, reason_text)

        state.stable_streak = (
            state.stable_streak + 1
            if state.best_summary and check_stability(state.best_summary, cfg["stability"])
            else 0
        )

        target = route_iteration(iter_dir, decision, results_dir)
        next_hint = ("развить принятую идею (improve_strategy)" if decision == "accepted"
                     else "новая гипотеза вне повторов базы знаний")
        append_log(log_path, iteration_name, summary, decision, reason_text, next_hint)
        append_kb_note(kb_path, iteration_name, summary, decision, reason_text)

        state.history.append({
            "iteration": iteration_name, "decision": decision,
            "idea": summary["idea"], "score": summary["composite_score"],
            "path": _rel(target),
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
        })
        last_decision = decision
        state.save(state_path)
        print(f"  => {decision}: {reason_text}")


def finalize(state: LoopState, cfg: dict, reason: str, model: str,
             dry_run: bool, prompts_dir: Path, state_path: Path) -> None:
    print(f"\n### СТОП: {reason}")
    state.stopped = True
    state.stop_reason = reason
    state.save(state_path)
    run_stats = json.dumps({
        "iterations": state.iteration,
        "best_score": state.best_score,
        "best_iteration": state.best_iteration,
        "accepted": sum(1 for h in state.history if h["decision"] == "accepted"),
        "rejected": sum(1 for h in state.history if h["decision"] == "rejected"),
        "failed": sum(1 for h in state.history if h["decision"] == "failed"),
    }, ensure_ascii=False, indent=2)
    if dry_run:
        print("dry-run: финальный отчёт не запрашивается. run_stats:\n" + run_stats)
        return
    prompt = build_prompt(prompts_dir / "final_report.md",
                          {"stop_reason": reason, "run_stats": run_stats})
    run_claude(prompt, cfg, model, cfg["iteration_timeout_sec"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min-improvement", type=float, default=None)
    parser.add_argument("--prompt-file", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--config", type=str, default=str(ROOT / "orchestrator" / "config.yaml"))
    parser.add_argument("--state-file", type=str, default=None,
                        help="override state.json location (useful for tests)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    sys.exit(run_loop(parse_args()))
