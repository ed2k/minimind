from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from selfchange.kernel.budgets import ComputeProfile, get_profile
from selfchange.kernel.fault_drills import DrillSuiteReport, run_drill_suite
from selfchange.kernel.integrity import KERNEL_MANIFEST_FILENAME, check_git_diff_boundary, verify_kernel_integrity
from selfchange.kernel.sandbox import run_sandboxed, scan_mutable_tree


@dataclass
class GateResult:
    """Outcome of a single immutable verification gate."""

    gate: str
    passed: bool
    score: float
    details: dict


@dataclass
class VerificationReport:
    """Full 4-tier candidate verification scorecard."""

    passed: bool
    total_score: float
    survival_score: float
    eval_score: float
    self_repair_score: float
    efficiency_score: float
    gates: dict[str, GateResult]
    duration_sec: float


def _rep_penalty(text: str, n: int = 3, cap: float = 0.5) -> float:
    toks = re.findall(r"\w+|[^\w\s]", (text or "").lower())
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return 0.0
    return min(cap, (len(grams) - len(set(grams))) * cap * 2.0 / len(grams))


def compute_rlvr_reward(
    turn_outputs: list[str],
    tool_calls: list[dict],
    observed_before_edit: bool,
    first_edit_valid: bool,
    task_passed: bool,
    boundary_violated: bool,
    unfinished: bool,
    over_budget: bool,
    open_thinking: bool = False,
) -> float:
    """Deterministic RLVR reward matching Section 6.2 of selfchange/plan.md."""
    final_text = turn_outputs[-1] if turn_outputs else ""
    tag_errors = 0
    for turn in turn_outputs:
        tag_errors += abs(turn.count("<tool_call>") - turn.count("</tool_call>"))
        for blk in re.findall(r"<tool_call>(.*?)</tool_call>", turn, re.DOTALL):
            try:
                obj = json.loads(blk.strip())
                if not isinstance(obj, dict) or not obj.get("name"):
                    tag_errors += 1
            except Exception:
                tag_errors += 1

    r_fmt = 0.0
    if open_thinking and final_text.count("</think>") == 1:
        think_body = final_text.split("</think>", 1)[0].split("<think>")[-1].strip()
        if 20 <= len(think_body) <= 400:
            r_fmt += 0.25
    if len(tool_calls) >= 1 and tag_errors == 0:
        r_fmt += 0.25
    r_fmt -= 0.5 * tag_errors
    r_fmt = max(-1.0, min(0.5, r_fmt))

    r_act = (0.25 if observed_before_edit else 0.0) + (0.25 if first_edit_valid else 0.0)
    if boundary_violated:
        r_out = -1.0
    elif task_passed:
        r_out = 2.0
    else:
        r_out = -1.0 if tag_errors > 0 else 0.0

    penalties = _rep_penalty(final_text) + (0.5 if unfinished else 0.0) + (0.25 if over_budget else 0.0)
    total = r_fmt + r_act + r_out - penalties
    return round(max(-3.0, min(3.0, total)), 4)


_GATE3_PROBE = """
import json, os, sys
sys.path.insert(0, os.getcwd())
from selfchange.mutable import agent_loop, data_curator, train_recipe

valid_msgs = [
    '<tool_call>{"name": "get_system_state", "arguments": {}}</tool_call>',
    '<think>checking logs</think><tool_call>{"name": "read_log_tail", "arguments": {"log_name": "train.log", "lines": 20}}</tool_call>',
    '<tool_call>{"name": "patch_config", "arguments": {"updates": {"training": {"batch_size": 2}}}}</tool_call>',
]
parsed_ok = sum(1 for m in valid_msgs if len(agent_loop.parse_tool_calls(m)[0]) == 1 and agent_loop.parse_tool_calls(m)[1] == 0)
bad_calls, bad_errs = agent_loop.parse_tool_calls('<tool_call>{broken</tool_call>')
parse_score = (parsed_ok + (1 if not bad_calls and bad_errs >= 1 else 0)) / 4.0

trajs = [
    {"task_id": "t1", "task_family": "oom", "passed": True, "reward": 2.0, "final_text": "batch reduced", "tool_calls": [{"name": "patch_config"}], "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
    {"task_id": "t1", "task_family": "oom", "passed": False, "reward": -1.0, "final_text": "bad", "tool_calls": [], "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "err"}]},
]
rft = data_curator.curate_rft_samples(trajs)
dpo = data_curator.build_dpo_pairs(trajs)
mixed = data_curator.mix_with_anchor(rft, [{"conversations": []}], anchor_ratio=0.25)
curator_ok = len(rft) == 1 and len(dpo) == 1 and len(mixed) >= 2

cfg = {"training": {"batch_size": 8, "learning_rate": 10.0}}
rec = train_recipe.select_training_stage({"pass_at_1": 0.3, "pass_at_4": 0.7}, {"dpo_pairs": 10, "rft_samples": 10}, cfg, safe_mode=True)
bounds_ok = rec["stage"] == "dpo" and rec["training"]["batch_size"] == 4 and rec["training"]["learning_rate"] <= 5e-4

print(json.dumps({"parse_score": parse_score, "curator_ok": curator_ok, "bounds_ok": bounds_ok}))
"""


class Verifier:
    """4-tier immutable promotion gate for candidate worktree and weight updates."""

    def __init__(
        self,
        kernel_dir: Path | None = None,
        manifest_path: Path | None = None,
        profile: ComputeProfile | None = None,
    ):
        self.kernel_dir = Path(kernel_dir).resolve() if kernel_dir is not None else Path(__file__).resolve().parent
        self.manifest_path = (
            Path(manifest_path).resolve()
            if manifest_path is not None
            else self.kernel_dir / KERNEL_MANIFEST_FILENAME
        )
        self.profile = profile if profile is not None else get_profile()

    def run_gate1_boundary_and_static(
        self,
        worktree_dir: Path,
        base_ref: str = "HEAD",
        check_git: bool = True,
    ) -> GateResult:
        """Verify kernel SHA-256 manifest, git diff boundary, AST safety, and py_compile."""
        worktree_dir = Path(worktree_dir).resolve()
        integrity = verify_kernel_integrity(self.kernel_dir, self.manifest_path)
        diff_ok = True
        diff_violations: list[str] = []
        diff_error: str | None = None
        if check_git:
            diff_rep = check_git_diff_boundary(worktree_dir, base_ref=base_ref)
            diff_ok = diff_rep.ok
            diff_violations = diff_rep.violations
            diff_error = diff_rep.error

        mutable_dir = worktree_dir / "selfchange" / "mutable"
        ast_rep = scan_mutable_tree(mutable_dir, run_py_compile=True)
        passed = integrity.ok and diff_ok and ast_rep.ok
        return GateResult(
            gate="gate1_boundary_and_static",
            passed=passed,
            score=1.0 if passed else 0.0,
            details={
                "integrity_ok": integrity.ok,
                "integrity_mismatched": integrity.mismatched,
                "diff_ok": diff_ok,
                "diff_violations": diff_violations,
                "diff_error": diff_error,
                "ast_ok": ast_rep.ok,
                "scanned_files": ast_rep.scanned_files,
                "ast_violations": [asdict(v) for v in ast_rep.violations],
            },
        )

    def run_gate2_fault_drills(
        self,
        worktree_dir: Path,
        seeds: list[int] | None = None,
    ) -> tuple[GateResult, DrillSuiteReport]:
        """Run deterministic operational fault-injection drills inside sandbox."""
        report = run_drill_suite(
            worktree_dir,
            seeds=seeds,
            timeout_sec=min(30.0, float(self.profile.subprocess_timeout_sec)),
        )
        gate = GateResult(
            gate="gate2_fault_drills",
            passed=report.passed_all,
            score=round(report.pass_rate, 4),
            details={
                "pass_rate": report.pass_rate,
                "survival_score": report.survival_score,
                "self_repair_score": report.self_repair_score,
                "healthy_noop_rate": report.healthy_noop_rate,
                "results": [asdict(r) for r in report.results],
            },
        )
        return gate, report

    def run_gate3_regression(
        self,
        worktree_dir: Path,
        extra_eval_fn: Callable[[Path], float] | None = None,
    ) -> GateResult:
        """Verify tool-call parsing, experience curation, recipe bounds, and optional LLM eval."""
        worktree_dir = Path(worktree_dir).resolve()
        res = run_sandboxed(
            [sys.executable, "-c", _GATE3_PROBE],
            cwd=worktree_dir,
            timeout_sec=min(20.0, float(self.profile.subprocess_timeout_sec)),
            extra_env={"PYTHONPATH": str(worktree_dir)},
        )
        if res.timed_out or res.returncode != 0:
            return GateResult(
                gate="gate3_regression",
                passed=False,
                score=0.0,
                details={"error": res.stderr.strip(), "timed_out": res.timed_out},
            )
        probe = json.loads(res.stdout)
        base_score = (probe["parse_score"] + float(probe["curator_ok"]) + float(probe["bounds_ok"])) / 3.0
        extra_score = float(extra_eval_fn(worktree_dir)) if extra_eval_fn is not None else 1.0
        eval_score = round(0.5 * base_score + 0.5 * extra_score, 4) if extra_eval_fn is not None else round(base_score, 4)
        passed = probe["parse_score"] >= 0.95 and probe["curator_ok"] and probe["bounds_ok"] and extra_score >= 0.7
        return GateResult(
            gate="gate3_regression",
            passed=passed,
            score=eval_score,
            details={**probe, "extra_score": extra_score},
        )

    def run_gate4_canary_and_score(
        self,
        _worktree_dir: Path,
        gate2_report: DrillSuiteReport,
        gate3: GateResult,
        elapsed_sec: float,
    ) -> tuple[GateResult, float]:
        """Compute bounded efficiency check and weighted composite fitness score."""
        within_budget = elapsed_sec <= float(self.profile.cycle_timeout_sec)
        efficiency = max(0.0, min(1.0, 1.0 - elapsed_sec / max(float(self.profile.cycle_timeout_sec), 1.0)))
        total_score = round(
            0.45 * gate2_report.survival_score
            + 0.30 * gate3.score
            + 0.15 * gate2_report.self_repair_score
            + 0.10 * efficiency,
            4,
        )
        gate = GateResult(
            gate="gate4_canary_and_score",
            passed=within_budget,
            score=round(efficiency, 4),
            details={
                "elapsed_sec": round(elapsed_sec, 4),
                "cycle_timeout_sec": self.profile.cycle_timeout_sec,
                "total_score": total_score,
            },
        )
        return gate, total_score

    def verify_candidate(
        self,
        worktree_dir: Path,
        base_ref: str = "HEAD",
        check_git: bool = True,
        seeds: list[int] | None = None,
        extra_eval_fn: Callable[[Path], float] | None = None,
    ) -> VerificationReport:
        """Run immutable Gates 1-4 in order, short-circuiting immediately if Gate 1 fails."""
        t0 = time.monotonic()
        gates: dict[str, GateResult] = {}

        gate1 = self.run_gate1_boundary_and_static(worktree_dir, base_ref=base_ref, check_git=check_git)
        gates["gate1"] = gate1
        if not gate1.passed:
            return VerificationReport(
                passed=False,
                total_score=0.0,
                survival_score=0.0,
                eval_score=0.0,
                self_repair_score=0.0,
                efficiency_score=0.0,
                gates=gates,
                duration_sec=round(time.monotonic() - t0, 4),
            )

        gate2, drill_report = self.run_gate2_fault_drills(worktree_dir, seeds=seeds)
        gates["gate2"] = gate2

        gate3 = self.run_gate3_regression(worktree_dir, extra_eval_fn=extra_eval_fn)
        gates["gate3"] = gate3

        elapsed = time.monotonic() - t0
        gate4, total_score = self.run_gate4_canary_and_score(worktree_dir, drill_report, gate3, elapsed)
        gates["gate4"] = gate4

        passed = gate1.passed and gate2.passed and gate3.passed and gate4.passed
        return VerificationReport(
            passed=passed,
            total_score=total_score,
            survival_score=round(drill_report.survival_score, 4),
            eval_score=round(gate3.score, 4),
            self_repair_score=round(drill_report.self_repair_score, 4),
            efficiency_score=round(gate4.score, 4),
            gates=gates,
            duration_sec=round(time.monotonic() - t0, 4),
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Immutable 4-Tier Candidate Verifier")
    parser.add_argument("--worktree", type=Path, default=Path("."))
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--base-ref", type=str, default="HEAD")
    parser.add_argument("--no-git", action="store_true")
    args = parser.parse_args()

    verifier = Verifier(profile=get_profile(args.profile))
    report = verifier.verify_candidate(
        args.worktree,
        base_ref=args.base_ref,
        check_git=not args.no_git,
    )
    print(json.dumps(asdict(report), indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
