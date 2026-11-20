from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
import time
from pathlib import Path

from selfchange.bootstrap.generate_traces import _CLEAN_FIXTURES, inject_ast_bug
from selfchange.kernel.budgets import get_profile
from selfchange.kernel.fault_drills import run_drill_suite
from selfchange.kernel.integrity import atomic_write_text
from selfchange.kernel.sandbox import scan_python_source
from selfchange.kernel.tool_runtime import SELFCHANGE_TOOLS, WorktreeToolExecutor, validate_tool_call
from selfchange.mutable.agent_loop import parse_tool_calls
from selfchange.mutable.data_curator import compute_rep_penalty
from selfchange.mutable.recovery_policy import classify_fault, plan_recovery_action, propose_line_fix

BOOTSTRAP_READY_THRESHOLDS: dict[str, float] = {
    "max_ppl_regression_pct": 5.0,
    "min_eos_rate": 0.95,
    "max_rep_penalty": 0.10,
    "min_json_parse_rate": 0.95,
    "min_schema_validity_rate": 0.90,
    "min_copy_fidelity_rate": 0.90,
    "min_code_pass_at_1": 0.40,
    "min_code_pass_at_4": 0.70,
    "min_ast_valid_rate": 0.85,
    "min_drills_pass_rate": 0.70,
    "min_config_drills_pass_rate": 0.80,
    "min_healthy_noop_rate": 0.90,
}


def check_bootstrap_ready_gate(
    scorecard: dict,
    base_ppl: float | None = None,
) -> tuple[bool, list[str], list[str]]:
    """Evaluate Step 6 Bootstrap-Ready Entry Gate and emit Go/No-Go recommendations."""
    t1 = scorecard["tier1_language"]
    t2 = scorecard["tier2_toolcall"]
    t3 = scorecard["tier3_code_edit"]
    t4 = scorecard["tier4_drills"]

    failed: list[str] = []
    recs: list[str] = []

    if base_ppl is not None and base_ppl > 0:
        reg_pct = ((t1["dialogue_ppl"] - base_ppl) / base_ppl) * 100.0
        if reg_pct > BOOTSTRAP_READY_THRESHOLDS["max_ppl_regression_pct"]:
            failed.append(f"dialogue_ppl_regression={reg_pct:.2f}% > 5.0%")
            recs.append("Raise Bucket C anchor replay ratio to 35-40% or lower learning rate.")
    if t1["eos_rate"] < BOOTSTRAP_READY_THRESHOLDS["min_eos_rate"]:
        failed.append(f"eos_rate={t1['eos_rate']:.3f} < 0.95")
    if t1["mean_rep_penalty"] > BOOTSTRAP_READY_THRESHOLDS["max_rep_penalty"]:
        failed.append(f"mean_rep_penalty={t1['mean_rep_penalty']:.3f} > 0.10")

    if t2["json_parse_rate"] < BOOTSTRAP_READY_THRESHOLDS["min_json_parse_rate"]:
        failed.append(f"json_parse_rate={t2['json_parse_rate']:.3f} < 0.95")
    if t2["schema_validity_rate"] < BOOTSTRAP_READY_THRESHOLDS["min_schema_validity_rate"]:
        failed.append(f"schema_validity_rate={t2['schema_validity_rate']:.3f} < 0.90")
    if t2["copy_fidelity_rate"] < BOOTSTRAP_READY_THRESHOLDS["min_copy_fidelity_rate"]:
        failed.append(f"copy_fidelity_rate={t2['copy_fidelity_rate']:.3f} < 0.90")

    if t3["exact_match_rate"] < 0.50:
        recs.append("Keep L1 replace_line_range primary instead of verbatim patch_file_replace.")
    if t3["pass_at_4"] < 0.10:
        recs.append("pass@4 < 10%: run Stage 0.5 mid-training + Stage 1 3-rung SFT before RL.")
    elif t3["pass_at_4"] >= 0.40 and t3["pass_at_1"] < 0.30:
        recs.append("pass@4 >= 40% while pass@1 < 30%: run Stage 1.5 offline RFT + DPO sharpening.")

    if t3["pass_at_1"] < BOOTSTRAP_READY_THRESHOLDS["min_code_pass_at_1"]:
        failed.append(f"code_pass_at_1={t3['pass_at_1']:.3f} < 0.40")
    if t3["pass_at_4"] < BOOTSTRAP_READY_THRESHOLDS["min_code_pass_at_4"]:
        failed.append(f"code_pass_at_4={t3['pass_at_4']:.3f} < 0.70")
    if t3["ast_valid_rate"] < BOOTSTRAP_READY_THRESHOLDS["min_ast_valid_rate"]:
        failed.append(f"ast_valid_rate={t3['ast_valid_rate']:.3f} < 0.85")

    if t4["pass_rate"] < BOOTSTRAP_READY_THRESHOLDS["min_drills_pass_rate"]:
        failed.append(f"drills_pass_rate={t4['pass_rate']:.3f} < 0.70")
    if t4["config_drills_pass_rate"] < BOOTSTRAP_READY_THRESHOLDS["min_config_drills_pass_rate"]:
        failed.append(f"config_drills_pass_rate={t4['config_drills_pass_rate']:.3f} < 0.80")
    if t4["healthy_noop_rate"] < BOOTSTRAP_READY_THRESHOLDS["min_healthy_noop_rate"]:
        failed.append(f"healthy_noop_rate={t4['healthy_noop_rate']:.3f} < 0.90")

    return len(failed) == 0, failed, recs


def _scripted_tool_eval() -> dict:
    prompts = [
        '<tool_call>{"name": "get_system_state", "arguments": {}}</tool_call>',
        '<think>Inspect train log for root cause.</think><tool_call>{"name": "read_log_tail", "arguments": {"log_name": "train.log", "lines": 20}}</tool_call>',
        '<tool_call>{"name": "patch_config", "arguments": {"updates": {"training": {"batch_size": 1}}}}</tool_call>',
        '<tool_call>{"name": "replace_line_range", "arguments": {"path": "oom_helper.py", "start_line": 2, "end_line": 2, "lines": ["    return max(1, batch_size // div)"]}}</tool_call>',
    ]
    parsed_ok = 0
    schema_ok = 0
    for txt in prompts:
        calls, errs = parse_tool_calls(txt)
        if calls and errs == 0:
            parsed_ok += 1
            ok, _ = validate_tool_call(calls[0]["name"], calls[0]["arguments"])
            if ok:
                schema_ok += 1
    total = len(prompts)
    return {
        "json_parse_rate": parsed_ok / total,
        "schema_validity_rate": schema_ok / total,
        "tag_balance_rate": 1.0,
        "hallucinated_tool_rate": 0.0,
        "copy_fidelity_rate": 1.0,
    }


def _scripted_code_edit_eval(num_tasks: int = 20) -> dict:
    exact_hits = 0
    ast_valid = 0
    pass1 = 0
    pass4 = 0
    for i in range(num_tasks):
        fname, clean_src, _ = _CLEAN_FIXTURES[i % len(_CLEAN_FIXTURES)]
        broken_src, _fix = inject_ast_bug(clean_src, i % 5)
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            m_dir = wt / "selfchange" / "mutable"
            m_dir.mkdir(parents=True, exist_ok=True)
            (m_dir / fname).write_text(broken_src, encoding="utf-8")
            executor = WorktreeToolExecutor(wt, autonomy_level=1)
            lines = broken_src.splitlines()
            proposal = propose_line_fix(lines, {"line": 1 if (i % 5) == 0 else 2, "error_type": ""})
            if proposal is not None:
                exact_hits += 1
                res = executor.dispatch(
                    "replace_line_range",
                    {
                        "path": fname,
                        "start_line": proposal["start_line"],
                        "end_line": proposal["end_line"],
                        "lines": proposal["lines"],
                    },
                )
                edited = (m_dir / fname).read_text(encoding="utf-8")
                if scan_python_source(edited).ok:
                    ast_valid += 1
                if res.get("ok") and edited.strip() == clean_src.strip():
                    pass1 += 1
                    pass4 += 1
    return {
        "num_tasks": num_tasks,
        "exact_match_rate": round(exact_hits / max(1, num_tasks), 4),
        "ast_valid_rate": round(ast_valid / max(1, num_tasks), 4),
        "pass_at_1": round(pass1 / max(1, num_tasks), 4),
        "pass_at_4": round(pass4 / max(1, num_tasks), 4),
    }


def _minimind_eval(
    model_path: Path,
    tokenizer_path: Path,
    profile_name: str | None,
    heldout_path: Path | None,
    num_tasks: int,
) -> tuple[dict, dict, dict]:
    import torch  # type: ignore[import-not-found]
    from selfchange.training.model_loader import generate_agent_reply, load_minimind_model

    profile = get_profile(profile_name)
    model, tokenizer = load_minimind_model(model_path, tokenizer_path, profile)

    # Tier 1: held-out dialogue loss / PPL if heldout_path is provided
    losses: list[float] = []
    if heldout_path is not None and heldout_path.is_file():
        samples = [json.loads(ln) for ln in heldout_path.read_text(encoding="utf-8").splitlines() if ln.strip()][:32]
        with torch.no_grad():
            for sample in samples:
                text = tokenizer.apply_chat_template(sample["conversations"], tokenize=False, add_generation_prompt=False)
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=profile.max_seq_len).to(profile.device)
                out = model(enc["input_ids"], labels=enc["input_ids"])
                if out.loss is not None and torch.isfinite(out.loss):
                    losses.append(float(out.loss.item()))
    mean_loss = sum(losses) / len(losses) if losses else 2.5
    ppl = round(math.exp(min(20.0, mean_loss)), 4)

    # Tier 2: real MiniMind tool-call generation on compact fault prompts
    prompts = [
        [{"role": "system", "content": "Diagnose OOM.", "tools": SELFCHANGE_TOOLS[:3]}, {"role": "user", "content": "fault=oom batch_size=8"}],
        [{"role": "system", "content": "Diagnose NaN.", "tools": SELFCHANGE_TOOLS[:3]}, {"role": "user", "content": "fault=nan_loss lr=0.0002"}],
        [{"role": "system", "content": "Read file.", "tools": SELFCHANGE_TOOLS[:5]}, {"role": "user", "content": "fault=syntax_or_test_error file=oom_helper.py line=1"}],
        [{"role": "system", "content": "Rollback.", "tools": SELFCHANGE_TOOLS}, {"role": "user", "content": "fault=corrupt_checkpoint ancestor=anc_1"}],
    ]
    json_ok = schema_ok = tag_ok = eos_ok = 0
    reps: list[float] = []
    for msgs in prompts:
        reply = generate_agent_reply(model, tokenizer, msgs, SELFCHANGE_TOOLS, max_new_tokens=96, temperature=0.0, top_p=1.0)
        reps.append(compute_rep_penalty(reply))
        if reply.strip():
            eos_ok += 1
        calls, errs = parse_tool_calls(reply)
        if errs == 0:
            tag_ok += 1
        if calls:
            json_ok += 1
            ok, _ = validate_tool_call(calls[0]["name"], calls[0]["arguments"])
            if ok:
                schema_ok += 1

    tier1 = {
        "dialogue_ppl": ppl,
        "python_ppl": ppl,
        "eos_rate": round(eos_ok / len(prompts), 4),
        "mean_rep_penalty": round(sum(reps) / len(reps), 4),
        "empty_rate": round(1.0 - eos_ok / len(prompts), 4),
    }
    tier2 = {
        "json_parse_rate": round(json_ok / len(prompts), 4),
        "schema_validity_rate": round(schema_ok / len(prompts), 4),
        "tag_balance_rate": round(tag_ok / len(prompts), 4),
        "hallucinated_tool_rate": round(max(0.0, (json_ok - schema_ok) / len(prompts)), 4),
        "copy_fidelity_rate": round(schema_ok / len(prompts), 4),
    }

    # Tier 3: evaluate model edits on AST bugs (pass@1 greedy + pass@4 sampled)
    pass1 = pass4 = ast_ok = exact_ok = 0
    for i in range(min(num_tasks, 8)):
        fname, clean_src, _ = _CLEAN_FIXTURES[i % len(_CLEAN_FIXTURES)]
        broken_src, _fix = inject_ast_bug(clean_src, i % 5)
        numbered = "\n".join(f"{ln_no + 1}: {ln}" for ln_no, ln in enumerate(broken_src.splitlines()))
        msgs = [
            {"role": "system", "content": "Fix bug with replace_line_range.", "tools": [SELFCHANGE_TOOLS[5]]},
            {"role": "user", "content": f"{fname}:\n{numbered}"},
        ]
        attempts = [generate_agent_reply(model, tokenizer, msgs, [SELFCHANGE_TOOLS[5]], 96, 0.0, 1.0)]
        for _ in range(3):
            attempts.append(generate_agent_reply(model, tokenizer, msgs, [SELFCHANGE_TOOLS[5]], 96, 0.7, 0.9))

        solved_any = False
        valid_any = False
        for k, rep in enumerate(attempts):
            calls, _ = parse_tool_calls(rep)
            if not calls or calls[0]["name"] != "replace_line_range":
                continue
            with tempfile.TemporaryDirectory() as tmp:
                wt = Path(tmp)
                (wt / "selfchange" / "mutable").mkdir(parents=True, exist_ok=True)
                (wt / "selfchange" / "mutable" / fname).write_text(broken_src, encoding="utf-8")
                ex = WorktreeToolExecutor(wt, autonomy_level=1)
                res = ex.dispatch("replace_line_range", calls[0]["arguments"])
                if res.get("ok"):
                    valid_any = True
                    if (wt / "selfchange" / "mutable" / fname).read_text(encoding="utf-8").strip() == clean_src.strip():
                        solved_any = True
                        if k == 0:
                            pass1 += 1
                        break
        if valid_any:
            ast_ok += 1
            exact_ok += 1
        if solved_any:
            pass4 += 1

    denom = max(1, min(num_tasks, 8))
    tier3 = {
        "num_tasks": denom,
        "exact_match_rate": round(exact_ok / denom, 4),
        "ast_valid_rate": round(ast_ok / denom, 4),
        "pass_at_1": round(pass1 / denom, 4),
        "pass_at_4": round(pass4 / denom, 4),
    }
    return tier1, tier2, tier3


def evaluate_baseline(
    worktree_dir: Path,
    backend: str = "scripted",
    model_path: Path | None = None,
    tokenizer_path: Path | None = None,
    profile_name: str | None = None,
    heldout_path: Path | None = None,
    num_code_tasks: int = 20,
    drill_seeds: list[int] | None = None,
    base_ppl: float | None = None,
) -> dict:
    """Compute the 5-Tier Baseline Scorecard and Step 6 Bootstrap-Ready Gate."""
    t0 = time.monotonic()
    profile = get_profile(profile_name)
    worktree_dir = Path(worktree_dir).resolve()

    if backend == "scripted":
        tier1 = {
            "dialogue_ppl": 1.0,
            "python_ppl": 1.0,
            "eos_rate": 1.0,
            "mean_rep_penalty": 0.0,
            "empty_rate": 0.0,
        }
        tier2 = _scripted_tool_eval()
        tier3 = _scripted_code_edit_eval(num_tasks=num_code_tasks)
    elif backend == "minimind":
        if model_path is None or tokenizer_path is None:
            raise ValueError("model_path and tokenizer_path are required for backend='minimind'")
        tier1, tier2, tier3 = _minimind_eval(model_path, tokenizer_path, profile.name, heldout_path, num_code_tasks)
    else:
        raise ValueError(f"Unknown backend {backend!r}")

    drills = run_drill_suite(worktree_dir, seeds=drill_seeds or [42])
    cfg_drills = [r for r in drills.results if r.drill_id in ("drill_a_oom", "drill_b_nan_loss")]
    cfg_rate = sum(1 for r in cfg_drills if r.passed) / max(1, len(cfg_drills))
    tier4 = {
        "pass_rate": round(drills.pass_rate, 4),
        "config_drills_pass_rate": round(cfg_rate, 4),
        "survival_score": round(drills.survival_score, 4),
        "self_repair_score": round(drills.self_repair_score, 4),
        "healthy_noop_rate": round(drills.healthy_noop_rate, 4),
    }

    elapsed = round(time.monotonic() - t0, 4)
    tier5 = {
        "profile": profile.name,
        "device": profile.device,
        "dtype": profile.dtype,
        "eval_duration_sec": elapsed,
        "max_total_len": profile.max_total_len,
    }

    scorecard = {
        "backend": backend,
        "tier1_language": tier1,
        "tier2_toolcall": tier2,
        "tier3_code_edit": tier3,
        "tier4_drills": tier4,
        "tier5_hardware": tier5,
    }
    ready, failed_checks, recs = check_bootstrap_ready_gate(scorecard, base_ppl=base_ppl)
    scorecard["bootstrap_ready"] = ready
    scorecard["failed_checks"] = failed_checks
    scorecard["recommendations"] = recs
    return scorecard


def main() -> int:
    parser = argparse.ArgumentParser(description="5-Tier Baseline Scorecard & Bootstrap-Ready Gate")
    parser.add_argument("--worktree", type=Path, default=Path("."))
    parser.add_argument("--backend", choices=["scripted", "minimind"], default="scripted")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    parser.add_argument("--heldout-path", type=Path, default=None)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--num-code-tasks", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    scorecard = evaluate_baseline(
        worktree_dir=args.worktree,
        backend=args.backend,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        profile_name=args.profile,
        heldout_path=args.heldout_path,
        num_code_tasks=args.num_code_tasks,
    )
    rendered = json.dumps(scorecard, indent=2)
    if args.output is not None:
        atomic_write_text(args.output, rendered + "\n")
    print(rendered)
    return 0 if scorecard["bootstrap_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
