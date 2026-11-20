from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from selfchange.kernel.registry import LineageNode, LineageRegistry
from selfchange.kernel.sandbox import run_sandboxed
from selfchange.kernel.tool_runtime import SELFCHANGE_TOOLS, WorktreeToolExecutor


@dataclass
class DrillResult:
    """Outcome of executing a single deterministic fault-injection drill."""

    drill_id: str
    name: str
    passed: bool
    score: float
    turns_used: int
    details: dict


@dataclass
class DrillSuiteReport:
    """Aggregate report across all 5 operational drills plus healthy no-op control."""

    passed_all: bool
    pass_rate: float
    survival_score: float
    self_repair_score: float
    healthy_noop_rate: float
    results: list[DrillResult]


_SANDBOX_DRIVER = """
import json, os, sys
sys.path.insert(0, os.getcwd())
from selfchange.mutable import agent_loop, prompts, recovery_policy

payload = json.loads(os.environ["DRILL_PAYLOAD"])
mode = payload["mode"]
cfg_path = os.path.join(os.getcwd(), "selfchange", "mutable", "config.json")
with open(cfg_path, "r", encoding="utf-8") as f:
    config = json.load(f)

if mode == "plan":
    fault = recovery_policy.classify_fault(payload["state"], payload.get("log_tail", ""))
    action = recovery_policy.plan_recovery_action(fault, payload["state"], config, payload.get("log_tail", ""))
    card = prompts.format_fault_card(payload["state"], payload.get("log_tail", ""))
    out = {"fault": fault, "action": action, "card": card}
    if payload.get("source_lines") is not None:
        out["line_fix"] = recovery_policy.propose_line_fix(payload["source_lines"], payload.get("error_info", {}))
    print(json.dumps(out))
elif mode == "loop_drill":
    turns = iter(payload["scripted_replies"])
    def _mock_model(_messages, _tools):
        return next(turns, "Done.")
    def _mock_dispatch(name, args):
        return {"ok": True, "tool": name, "args": args}
    ep = agent_loop.run_agent_episode(payload["state"], payload["tools"], _mock_model, _mock_dispatch, config)
    print(json.dumps(ep))
"""


def _eval_candidate_policy_in_sandbox(
    worktree_dir: Path,
    payload: dict,
    timeout_sec: float = 15.0,
) -> dict:
    """Run candidate selfchange/mutable policy inside an isolated subprocess."""
    res = run_sandboxed(
        [sys.executable, "-c", _SANDBOX_DRIVER],
        cwd=Path(worktree_dir).resolve(),
        timeout_sec=timeout_sec,
        extra_env={
            "PYTHONPATH": str(Path(worktree_dir).resolve()),
            "DRILL_PAYLOAD": json.dumps(payload),
        },
    )
    if res.timed_out or res.returncode != 0:
        raise RuntimeError(f"Sandboxed candidate probe failed (rc={res.returncode}): {res.stderr.strip()}")
    return json.loads(res.stdout)


def _copy_mutable_to_temp(worktree_dir: Path, temp_root: Path) -> Path:
    src = Path(worktree_dir).resolve() / "selfchange" / "mutable"
    dst = temp_root / "selfchange" / "mutable"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return temp_root


def _run_drill_a_oom(worktree_dir: Path, rng: random.Random, timeout_sec: float) -> DrillResult:
    init_bs = rng.choice([4, 8])
    init_seq = rng.choice([512, 768])
    mem_budget = (init_bs * init_seq) // 2
    state = {
        "profile": "colab_t4",
        "last_fault": "oom",
        "training": {"batch_size": init_bs, "accumulation_steps": 1, "max_seq_len": init_seq},
    }
    log_tail = f"RuntimeError: CUDA out of memory at batch_size={init_bs} max_seq_len={init_seq}"

    with tempfile.TemporaryDirectory() as tmp:
        wt = _copy_mutable_to_temp(worktree_dir, Path(tmp))
        probe = _eval_candidate_policy_in_sandbox(wt, {"mode": "plan", "state": state, "log_tail": log_tail}, timeout_sec)
        executor = WorktreeToolExecutor(wt, autonomy_level=1)
        executor.dispatch("patch_config", {"updates": {"training": state["training"]}})
        action = probe["action"]
        dispatch_ok = False
        if action.get("tool") == "patch_config":
            res = executor.dispatch("patch_config", action.get("arguments", {}))
            dispatch_ok = bool(res.get("ok"))
        cfg = json.loads((wt / "selfchange" / "mutable" / "config.json").read_text(encoding="utf-8"))
        tr = cfg.get("training", {})
        new_bs = int(tr.get("batch_size", init_bs))
        new_seq = int(tr.get("max_seq_len", init_seq))
        passed = (
            probe["fault"] == "oom"
            and dispatch_ok
            and (new_bs < init_bs or new_seq < init_seq)
            and (new_bs * new_seq <= mem_budget)
        )
    return DrillResult("drill_a_oom", "Simulated OOM Recovery", passed, 1.0 if passed else 0.0, 2, {"new_bs": new_bs, "new_seq": new_seq})


def _run_drill_b_nan(worktree_dir: Path, rng: random.Random, timeout_sec: float) -> DrillResult:
    init_lr = rng.choice([1e-4, 2e-4])
    state = {
        "profile": "colab_t4",
        "last_fault": "nan_loss",
        "training": {"learning_rate": init_lr, "grad_clip": 2.0},
    }
    log_tail = f"Epoch:[1/2](12/100), loss: nan, grad_norm: inf, lr: {init_lr}"

    with tempfile.TemporaryDirectory() as tmp:
        wt = _copy_mutable_to_temp(worktree_dir, Path(tmp))
        probe = _eval_candidate_policy_in_sandbox(wt, {"mode": "plan", "state": state, "log_tail": log_tail}, timeout_sec)
        executor = WorktreeToolExecutor(wt, autonomy_level=1)
        executor.dispatch("patch_config", {"updates": {"training": state["training"]}})
        action = probe["action"]
        dispatch_ok = False
        if action.get("tool") == "patch_config":
            res = executor.dispatch("patch_config", action.get("arguments", {}))
            dispatch_ok = bool(res.get("ok"))
        cfg = json.loads((wt / "selfchange" / "mutable" / "config.json").read_text(encoding="utf-8"))
        tr = cfg.get("training", {})
        new_lr = float(tr.get("learning_rate", init_lr))
        new_clip = float(tr.get("grad_clip", 2.0))
        passed = probe["fault"] == "nan_loss" and dispatch_ok and new_lr < init_lr and new_clip <= 1.0
    return DrillResult("drill_b_nan_loss", "NaN Loss Recovery", passed, 1.0 if passed else 0.0, 2, {"new_lr": new_lr, "new_clip": new_clip})


def _run_drill_c_broken_module(worktree_dir: Path, rng: random.Random, timeout_sec: float) -> DrillResult:
    syntax_variant = rng.choice([True, False])
    with tempfile.TemporaryDirectory() as tmp:
        wt = _copy_mutable_to_temp(worktree_dir, Path(tmp))
        broken_file = wt / "selfchange" / "mutable" / "oom_helper.py"
        if syntax_variant:
            broken_src = "def compute_step(x, div=2)\n    return max(1, x // div)\n"
            err_type = "syntaxerror"
        else:
            broken_src = "def compute_step(x, div=2):\n    return max(1, x * div)\n"
            err_type = "assertionerror"
        broken_file.write_text(broken_src, encoding="utf-8")

        test_cmd = [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, '.'); from selfchange.mutable import oom_helper; assert oom_helper.compute_step(8, 2) == 4",
        ]
        fail_run = run_sandboxed(test_cmd, cwd=wt, timeout_sec=timeout_sec, extra_env={"PYTHONPATH": str(wt)})
        state = {
            "last_fault": err_type,
            "target_file": "oom_helper.py",
            "target_line": 1 if syntax_variant else 2,
        }
        executor = WorktreeToolExecutor(wt, autonomy_level=1)
        read_res = executor.dispatch("read_file_lines", {"path": "oom_helper.py", "start": 1, "end": 10})
        source_lines = broken_src.splitlines()
        probe = _eval_candidate_policy_in_sandbox(
            wt,
            {
                "mode": "plan",
                "state": state,
                "log_tail": fail_run.stderr,
                "source_lines": source_lines,
                "error_info": {"line": state["target_line"], "error_type": err_type},
            },
            timeout_sec,
        )
        line_fix = probe.get("line_fix")
        fix_ok = False
        if isinstance(line_fix, dict):
            edit_res = executor.dispatch(
                "replace_line_range",
                {
                    "path": "oom_helper.py",
                    "start_line": line_fix["start_line"],
                    "end_line": line_fix["end_line"],
                    "lines": line_fix["lines"],
                },
            )
            fix_ok = bool(edit_res.get("ok"))
        retest = run_sandboxed(test_cmd, cwd=wt, timeout_sec=timeout_sec, extra_env={"PYTHONPATH": str(wt)})
        checks = executor.dispatch("run_worktree_checks", {})
        passed = (
            fail_run.returncode != 0
            and bool(read_res.get("ok"))
            and probe["fault"] == "syntax_or_test_error"
            and fix_ok
            and retest.returncode == 0
            and bool(checks.get("ok"))
        )
    return DrillResult("drill_c_broken_module", "Broken Module Repair", passed, 1.0 if passed else 0.0, 3, {"syntax_variant": syntax_variant})


def _run_drill_d_corrupt_ckpt(worktree_dir: Path, _rng: random.Random, timeout_sec: float) -> DrillResult:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        wt = _copy_mutable_to_temp(worktree_dir, tmp_path / "wt")
        reg = LineageRegistry(tmp_path / "ws")
        reg.register_node(
            LineageNode("anc_ok", None, "a" * 40, None, 0.9, 0.9, 0.9, 0.9, 0.9, "bootstrap", "ok", "2026-11-19T00:00:00Z")
        )
        reg.register_node(
            LineageNode("bad_curr", "anc_ok", "b" * 40, None, 0.95, 0.95, 0.95, 0.95, 0.95, "archived", "new", "2026-11-19T00:01:00Z")
        )
        corrupt_path = tmp_path / "ws" / "full_sft_128_resume.pth"
        corrupt_path.write_bytes(b"CORRUPT_TRUNCATED")

        state = {"last_fault": "corrupt_checkpoint", "healthy_ancestor_id": "anc_ok"}
        log_tail = f"EOFError: failed loading {corrupt_path.name} (corrupt checksum)"
        probe = _eval_candidate_policy_in_sandbox(wt, {"mode": "plan", "state": state, "log_tail": log_tail}, timeout_sec)

        def _rb(node_id: str) -> dict:
            node = reg.rollback_to_node(node_id)
            return {"ok": True, "champion": node.node_id}

        executor = WorktreeToolExecutor(wt, autonomy_level=1, rollback_handler=_rb)
        action = probe["action"]
        rb_ok = False
        if action.get("tool") == "rollback_to_ancestor":
            res = executor.dispatch("rollback_to_ancestor", action.get("arguments", {}))
            rb_ok = bool(res.get("ok"))
        champ = reg.get_champion()
        passed = probe["fault"] == "corrupt_checkpoint" and rb_ok and champ is not None and champ.node_id == "anc_ok"
    return DrillResult("drill_d_corrupt_checkpoint", "Corrupted Checkpoint Recovery", passed, 1.0 if passed else 0.0, 2, {"champion": champ.node_id if champ else None})


def _run_drill_e_tool_loop(worktree_dir: Path, _rng: random.Random, timeout_sec: float) -> DrillResult:
    scripted_replies = [
        '<think>check</think><tool_call>{"name": "get_system_state", bad json}</tool_call>',
        '<tool_call>{"name": "get_system_state", "arguments": {}}</tool_call>',
        '<tool_call>{"name": "get_system_state", "arguments": {}}</tool_call>',
        '<tool_call>{"name": "get_system_state", "arguments": {}}</tool_call>',
    ]
    with tempfile.TemporaryDirectory() as tmp:
        wt = _copy_mutable_to_temp(worktree_dir, Path(tmp))
        ep = _eval_candidate_policy_in_sandbox(
            wt,
            {
                "mode": "loop_drill",
                "state": {"last_fault": "malformed_tool_loop", "log_tail": "invalid json loop"},
                "tools": SELFCHANGE_TOOLS[:2],
                "scripted_replies": scripted_replies,
            },
            timeout_sec,
        )
        passed = (
            ep.get("tag_errors", 0) >= 1
            and len(ep.get("tool_calls_made", [])) == 1
            and ep.get("turns_used", 99) == 3
        )
    return DrillResult("drill_e_malformed_tool_loop", "Malformed Tool-Loop Guard", passed, 1.0 if passed else 0.0, ep.get("turns_used", 0), {"tag_errors": ep.get("tag_errors", 0)})


def _run_drill_noop_healthy(worktree_dir: Path, _rng: random.Random, timeout_sec: float) -> DrillResult:
    state = {"last_fault": "none", "training": {"batch_size": 2, "learning_rate": 5e-5}}
    log_tail = "Epoch:[1/1](10/10), loss: 1.4200, status: ok"
    with tempfile.TemporaryDirectory() as tmp:
        wt = _copy_mutable_to_temp(worktree_dir, Path(tmp))
        before_cfg = (wt / "selfchange" / "mutable" / "config.json").read_text(encoding="utf-8")
        probe = _eval_candidate_policy_in_sandbox(wt, {"mode": "plan", "state": state, "log_tail": log_tail}, timeout_sec)
        after_cfg = (wt / "selfchange" / "mutable" / "config.json").read_text(encoding="utf-8")
        passed = probe["fault"] == "healthy" and probe["action"].get("tool") == "noop" and before_cfg == after_cfg
    return DrillResult("drill_noop_healthy", "Healthy No-Op Preservation", passed, 1.0 if passed else 0.0, 1, {})


def run_drill_suite(
    worktree_dir: Path,
    seeds: list[int] | None = None,
    timeout_sec: float = 15.0,
) -> DrillSuiteReport:
    """Run all 5 deterministic fault-injection drills plus healthy no-op across seeds."""
    seed_list = seeds if seeds is not None else [42]
    results: list[DrillResult] = []
    for seed in seed_list:
        rng = random.Random(seed)
        results.append(_run_drill_a_oom(worktree_dir, rng, timeout_sec))
        results.append(_run_drill_b_nan(worktree_dir, rng, timeout_sec))
        results.append(_run_drill_c_broken_module(worktree_dir, rng, timeout_sec))
        results.append(_run_drill_d_corrupt_ckpt(worktree_dir, rng, timeout_sec))
        results.append(_run_drill_e_tool_loop(worktree_dir, rng, timeout_sec))
        results.append(_run_drill_noop_healthy(worktree_dir, rng, timeout_sec))

    survival_items = [r.score for r in results if r.drill_id != "drill_c_broken_module"]
    repair_items = [r.score for r in results if r.drill_id == "drill_c_broken_module"]
    noop_items = [1.0 if r.passed else 0.0 for r in results if r.drill_id == "drill_noop_healthy"]

    survival_score = sum(survival_items) / max(1, len(survival_items))
    self_repair_score = sum(repair_items) / max(1, len(repair_items))
    healthy_noop_rate = sum(noop_items) / max(1, len(noop_items))
    pass_rate = sum(1 for r in results if r.passed) / max(1, len(results))
    passed_all = all(r.passed for r in results)

    return DrillSuiteReport(
        passed_all=passed_all,
        pass_rate=pass_rate,
        survival_score=survival_score,
        self_repair_score=self_repair_score,
        healthy_noop_rate=healthy_noop_rate,
        results=results,
    )
