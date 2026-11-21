from __future__ import annotations

import argparse
import json
import math
import random
import sys
import tempfile
import time
from pathlib import Path

from selfchange.bootstrap.data_quality import score_thinking_turn
from selfchange.bootstrap.generate_traces import _CLEAN_FIXTURES, inject_ast_bug
from selfchange.kernel.budgets import ComputeProfile, get_profile
from selfchange.kernel.fault_drills import run_drill_suite
from selfchange.kernel.integrity import atomic_write_text
from selfchange.kernel.sandbox import run_sandboxed, scan_python_source
from selfchange.kernel.tool_runtime import SELFCHANGE_TOOLS, WorktreeToolExecutor, validate_tool_call
from selfchange.mutable.agent_loop import parse_tool_calls
from selfchange.mutable.data_curator import compute_rep_penalty
from selfchange.mutable.recovery_policy import classify_fault, plan_recovery_action, propose_line_fix

_TOOLS_BY_NAME = {t["function"]["name"]: t for t in SELFCHANGE_TOOLS}


def _tools_by_names(names: list[str], compact: bool = True) -> list[dict]:
    selected: list[dict] = []
    for name in names:
        tool = _TOOLS_BY_NAME.get(name)
        if not tool:
            continue
        fn = tool["function"]
        if compact:
            selected.append({"type": "function", "function": {"name": fn["name"], "parameters": fn["parameters"]}})
        else:
            selected.append(tool)
    return selected


def _extract_worktree_system_prompt(worktree_dir: Path) -> str:
    res = run_sandboxed(
        [sys.executable, "-c", "from selfchange.mutable.prompts import SYSTEM_PROMPT; print(SYSTEM_PROMPT)"],
        cwd=worktree_dir,
        timeout_sec=5.0,
        extra_env={"PYTHONPATH": str(worktree_dir)},
    )
    if not res.timed_out and res.returncode == 0 and res.stdout.strip():
        return res.stdout.strip()
    from selfchange.mutable.prompts import SYSTEM_PROMPT

    return SYSTEM_PROMPT

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
    if t3["pass_at_4"] >= 0.40 and t3["pass_at_1"] < 0.30:
        recs.append("pass@4 >= 40% while pass@1 < 30%: run Stage 1.5 offline RFT + DPO sharpening.")

    t35 = scorecard.get("tier3_5_thinking")
    if isinstance(t35, dict):
        if t35.get("observation_grounding_rate", 1.0) < 0.85:
            failed.append(f"think_grounding={t35['observation_grounding_rate']:.3f} < 0.85")
            recs.append("Upgrade SFT traces to 4-step grounded CoT + Contrastive Thinking DPO.")
        if t35.get("think_action_consistency_rate", 1.0) < 0.90:
            failed.append(f"think_consistency={t35['think_action_consistency_rate']:.3f} < 0.90")

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


def _scripted_tool_eval() -> tuple[dict, dict]:
    probes = [
        (
            "fault=healthy profile=local_intel_cpu",
            '<think>\nObserve: fault=healthy profile=local_intel_cpu. Plan: call get_system_state.\n</think><tool_call>{"name": "get_system_state", "arguments": {}}</tool_call>',
        ),
        (
            "fault=nan_loss train.log step=20",
            '<think>\nObserve: fault=nan_loss train.log step=20. Plan: call read_log_tail.\n</think><tool_call>{"name": "read_log_tail", "arguments": {"log_name": "train.log", "lines": 20}}</tool_call>',
        ),
        (
            "fault=oom batch_size=2",
            '<think>\nObserve: fault=oom batch_size=2. Plan: call patch_config batch_size=1.\n</think><tool_call>{"name": "patch_config", "arguments": {"updates": {"training": {"batch_size": 1}}}}</tool_call>',
        ),
        (
            "oom_helper.py:\n1: def compute_step(batch_size: int, div: int = 2) -> int:\n2:     return min(1, batch_size // div)",
            '<think>\nObserve: oom_helper.py line 2 uses min instead of max. Plan: call replace_line_range line 2.\n</think><tool_call>{"name": "replace_line_range", "arguments": {"path": "oom_helper.py", "start_line": 2, "end_line": 2, "lines": ["    return max(1, batch_size // div)"]}}</tool_call>',
        ),
    ]
    parsed_ok = 0
    schema_ok = 0
    ground_ok = 0
    consist_ok = 0
    think_chars: list[int] = []
    for ctx, txt in probes:
        calls, errs = parse_tool_calls(txt)
        if calls and errs == 0:
            parsed_ok += 1
            ok, _ = validate_tool_call(calls[0]["name"], calls[0]["arguments"])
            if ok:
                schema_ok += 1
        ts = score_thinking_turn(ctx, txt)
        if ts["grounded"]:
            ground_ok += 1
        if ts["action_consistent"]:
            consist_ok += 1
        think_chars.append(ts["think_chars"])
    total = len(probes)
    tier2 = {
        "json_parse_rate": parsed_ok / total,
        "schema_validity_rate": schema_ok / total,
        "tag_balance_rate": 1.0,
        "hallucinated_tool_rate": 0.0,
        "copy_fidelity_rate": 1.0,
    }
    tier3_5 = {
        "observation_grounding_rate": round(ground_ok / total, 4),
        "think_action_consistency_rate": round(consist_ok / total, 4),
        "mean_think_chars": round(sum(think_chars) / max(1, len(think_chars)), 1),
        "thinking_efficiency": round((schema_ok / total) / max(0.25, (sum(think_chars) / max(1, len(think_chars))) / 250.0), 4),
    }
    return tier2, tier3_5


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


def _run_tool_and_code_probes(
    model,
    tokenizer,
    num_code_tasks: int,
    system_prefix: str | None = None,
    sample_pass4: bool = True,
) -> tuple[dict, dict, dict, dict, float]:
    from selfchange.training.model_loader import generate_agent_reply

    eos_tok = tokenizer.eos_token or "<|im_end|>"
    pad_tok = tokenizer.pad_token or "<|endoftext|>"

    def _strip_special(text: str) -> str:
        return text.replace(eos_tok, "").replace(pad_tok, "").strip()

    prefix = f"{system_prefix}\n" if system_prefix else ""
    prompts = [
        ([{"role": "system", "content": f"{prefix}Diagnose OOM."}, {"role": "user", "content": "fault=oom batch_size=8"}], _tools_by_names(["read_log_tail", "patch_config"])),
        ([{"role": "system", "content": f"{prefix}Diagnose NaN."}, {"role": "user", "content": "fault=nan_loss lr=0.0002"}], _tools_by_names(["read_log_tail", "patch_config"])),
        ([{"role": "system", "content": f"{prefix}Read file."}, {"role": "user", "content": "fault=syntax_or_test_error file=oom_helper.py line=1"}], _tools_by_names(["read_file_lines", "replace_line_range"])),
        ([{"role": "system", "content": f"{prefix}Rollback."}, {"role": "user", "content": "fault=corrupt_checkpoint ancestor=anc_1"}], _tools_by_names(["get_archive_summary", "rollback_to_ancestor"])),
    ]
    json_ok = schema_ok = tag_ok = eos_ok = non_empty = 0
    think_total = think_ground_ok = think_consist_ok = gen_tokens = 0
    gen_sec = 0.0
    think_chars: list[int] = []
    reps: list[float] = []
    for msgs, tools_arg in prompts:
        t_g0 = time.monotonic()
        raw_reply = generate_agent_reply(model, tokenizer, msgs, tools_arg, max_new_tokens=96, temperature=0.0, top_p=1.0)
        gen_sec += max(1e-4, time.monotonic() - t_g0)
        try:
            gen_tokens += len(tokenizer(raw_reply, add_special_tokens=False).input_ids)
        except Exception:
            gen_tokens += max(1, int(len(raw_reply) / 2.8))
        if eos_tok in raw_reply:
            eos_ok += 1
        reply = _strip_special(raw_reply)
        reps.append(compute_rep_penalty(reply))
        if reply:
            non_empty += 1
        calls, errs = parse_tool_calls(reply)
        if errs == 0:
            tag_ok += 1
        if calls:
            json_ok += 1
            ok, _ = validate_tool_call(calls[0]["name"], calls[0]["arguments"])
            if ok:
                schema_ok += 1
        ts = score_thinking_turn(msgs[-1]["content"], reply)
        think_total += 1
        if ts["grounded"]:
            think_ground_ok += 1
        if ts["action_consistent"]:
            think_consist_ok += 1
        think_chars.append(ts["think_chars"])

    tier1_rates = {
        "eos_rate": round(eos_ok / len(prompts), 4),
        "mean_rep_penalty": round(sum(reps) / len(reps), 4),
        "empty_rate": round(1.0 - non_empty / len(prompts), 4),
    }
    tier2 = {
        "json_parse_rate": round(json_ok / len(prompts), 4),
        "schema_validity_rate": round(schema_ok / len(prompts), 4),
        "tag_balance_rate": round(tag_ok / len(prompts), 4),
        "hallucinated_tool_rate": round(max(0.0, (json_ok - schema_ok) / len(prompts)), 4),
        "copy_fidelity_rate": round(schema_ok / len(prompts), 4),
    }

    denom = max(1, min(num_code_tasks, 8))
    code_tools = _tools_by_names(["replace_line_range"])
    pass1 = pass4 = ast_ok = exact_ok = 0
    for i in range(denom):
        fname, clean_src, _ = _CLEAN_FIXTURES[i % len(_CLEAN_FIXTURES)]
        broken_src, _fix = inject_ast_bug(clean_src, i % 5)
        numbered = "\n".join(f"{ln_no + 1}: {ln}" for ln_no, ln in enumerate(broken_src.splitlines()))
        msgs = [
            {"role": "system", "content": f"{prefix}Fix bug with replace_line_range."},
            {"role": "user", "content": f"{fname}:\n{numbered}"},
        ]
        t_g0 = time.monotonic()
        g_first = generate_agent_reply(model, tokenizer, msgs, code_tools, 96, 0.0, 1.0)
        gen_sec += max(1e-4, time.monotonic() - t_g0)
        try:
            gen_tokens += len(tokenizer(g_first, add_special_tokens=False).input_ids)
        except Exception:
            gen_tokens += max(1, int(len(g_first) / 2.8))
        ts_c = score_thinking_turn(msgs[-1]["content"], _strip_special(g_first))
        think_total += 1
        if ts_c["grounded"]:
            think_ground_ok += 1
        if ts_c["action_consistent"]:
            think_consist_ok += 1
        think_chars.append(ts_c["think_chars"])

        attempts = [g_first]
        if sample_pass4:
            for _ in range(3):
                attempts.append(generate_agent_reply(model, tokenizer, msgs, code_tools, 96, 0.7, 0.9))

        solved_any = False
        valid_any = False
        for k, raw_rep in enumerate(attempts):
            rep = _strip_special(raw_rep)
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

    tier3 = {
        "num_tasks": denom,
        "exact_match_rate": round(exact_ok / denom, 4),
        "ast_valid_rate": round(ast_ok / denom, 4),
        "pass_at_1": round(pass1 / denom, 4),
        "pass_at_4": round(pass4 / denom, 4),
    }
    mean_tc = round(sum(think_chars) / max(1, len(think_chars)), 1)
    solve_avg = 0.5 * (tier2["schema_validity_rate"] + tier3["pass_at_1"])
    tier3_5 = {
        "observation_grounding_rate": round(think_ground_ok / max(1, think_total), 4),
        "think_action_consistency_rate": round(think_consist_ok / max(1, think_total), 4),
        "mean_think_chars": mean_tc,
        "thinking_efficiency": round(solve_avg / max(0.25, mean_tc / 250.0), 4),
    }
    tok_s = round(gen_tokens / max(1e-4, gen_sec), 2)
    return tier1_rates, tier2, tier3, tier3_5, tok_s


def evaluate_minimind_candidate(
    worktree_dir: Path,
    model,
    tokenizer,
    profile: ComputeProfile,
    num_code_tasks: int = 4,
) -> dict:
    """Run lightweight greedy MiniMind-3 probes for Verifier Gate 3 candidate scoring."""
    _ = profile
    sys_prompt = _extract_worktree_system_prompt(Path(worktree_dir))
    t1_rates, tier2, tier3, tier3_5, _tok_s = _run_tool_and_code_probes(
        model,
        tokenizer,
        num_code_tasks=num_code_tasks,
        system_prefix=sys_prompt,
        sample_pass4=False,
    )
    extra_score = round(
        0.15 * tier2["json_parse_rate"]
        + 0.25 * tier2["schema_validity_rate"]
        + 0.15 * tier3["ast_valid_rate"]
        + 0.30 * tier3["pass_at_1"]
        + 0.075 * t1_rates["eos_rate"]
        + 0.075 * max(0.0, 1.0 - 2.0 * t1_rates["mean_rep_penalty"]),
        4,
    )
    return {
        "extra_score": extra_score,
        "json_parse_rate": tier2["json_parse_rate"],
        "schema_validity_rate": tier2["schema_validity_rate"],
        "ast_valid_rate": tier3["ast_valid_rate"],
        "pass_at_1": tier3["pass_at_1"],
        "eos_rate": t1_rates["eos_rate"],
        "mean_rep_penalty": t1_rates["mean_rep_penalty"],
        "observation_grounding_rate": tier3_5["observation_grounding_rate"],
        "think_action_consistency_rate": tier3_5["think_action_consistency_rate"],
    }


def _minimind_eval(
    model_path: Path,
    tokenizer_path: Path,
    profile_name: str | None,
    heldout_path: Path | None,
    num_tasks: int,
    model_config: dict | None = None,
    quantize_int8: bool = False,
) -> tuple[dict, dict, dict, dict, float]:
    import torch  # type: ignore[import-not-found]
    from selfchange.training.model_loader import load_minimind_model, normalize_messages_for_template

    profile = get_profile(profile_name)
    model, tokenizer = load_minimind_model(
        model_path,
        tokenizer_path,
        profile,
        model_config=model_config,
        quantize_int8=quantize_int8,
    )

    losses: list[float] = []
    if heldout_path is not None and heldout_path.is_file():
        samples = [json.loads(ln) for ln in heldout_path.read_text(encoding="utf-8").splitlines() if ln.strip()][:32]
        with torch.no_grad():
            for sample in samples:
                norm_msgs, sample_tools = normalize_messages_for_template(sample["conversations"])
                text = tokenizer.apply_chat_template(
                    norm_msgs,
                    tokenize=False,
                    add_generation_prompt=False,
                    tools=sample_tools or None,
                )
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=profile.max_seq_len).to(profile.device)
                out = model(enc["input_ids"], labels=enc["input_ids"])
                if out.loss is not None and torch.isfinite(out.loss):
                    losses.append(float(out.loss.item()))
    mean_loss = sum(losses) / len(losses) if losses else 2.5
    ppl = round(math.exp(min(20.0, mean_loss)), 4)

    t1_rates, tier2, tier3, tier3_5, tok_s = _run_tool_and_code_probes(model, tokenizer, num_code_tasks=num_tasks, sample_pass4=True)
    tier1 = {
        "dialogue_ppl": ppl,
        "python_ppl": ppl,
        **t1_rates,
    }
    return tier1, tier2, tier3, tier3_5, tok_s


def _resolve_eval_model_path(repo_root: Path, profile_name: str | None, requested: Path | None) -> Path | None:
    if requested is not None and str(requested) != "auto":
        return requested
    profile = get_profile(profile_name)
    moe_suffix = "_moe" if profile.use_moe else ""
    hs = profile.hidden_size
    repo_root = Path(repo_root).resolve()
    ws = repo_root / "selfchange" / "workspace"
    candidates = [
        ws / "promoted_models" / f"champion_{hs}{moe_suffix}.pth",
        ws / "models" / "candidates" / f"offline_dpo_r1_{hs}{moe_suffix}.pth",
        ws / "models" / "candidates" / f"offline_rft_r1_merged_{hs}{moe_suffix}.pth",
        ws / "models" / "candidates" / f"warmstart_rung3_{hs}{moe_suffix}.pth",
        ws / "models" / "candidates" / f"warmstart_rung2_{hs}{moe_suffix}.pth",
        ws / "models" / "candidates" / f"warmstart_rung1_{hs}{moe_suffix}.pth",
        ws / "models" / "base" / f"minimind3_{hs}{moe_suffix}.pth",
        repo_root / "out" / f"full_sft_{hs}{moe_suffix}.pth",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


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
    model_config: dict | None = None,
    quantize_int8: bool = False,
) -> dict:
    """Compute the Baseline Scorecard (including Tier 3.5 thinking) and Step 6 Bootstrap-Ready Gate."""
    t0 = time.monotonic()
    profile = get_profile(profile_name)
    worktree_dir = Path(worktree_dir).resolve()
    cpu_decode_tok_s = 0.0

    if backend == "scripted":
        tier1 = {
            "dialogue_ppl": 1.0,
            "python_ppl": 1.0,
            "eos_rate": 1.0,
            "mean_rep_penalty": 0.0,
            "empty_rate": 0.0,
        }
        tier2, tier3_5 = _scripted_tool_eval()
        tier3 = _scripted_code_edit_eval(num_tasks=num_code_tasks)
        cpu_decode_tok_s = 100.0
    elif backend == "minimind":
        if model_path is None or tokenizer_path is None:
            raise ValueError("model_path and tokenizer_path are required for backend='minimind'")
        tier1, tier2, tier3, tier3_5, cpu_decode_tok_s = _minimind_eval(
            model_path,
            tokenizer_path,
            profile.name,
            heldout_path,
            num_code_tasks,
            model_config=model_config,
            quantize_int8=quantize_int8,
        )
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
        "cpu_decode_tok_s": cpu_decode_tok_s,
        "max_total_len": profile.max_total_len,
    }

    scorecard = {
        "backend": backend,
        "tier1_language": tier1,
        "tier2_toolcall": tier2,
        "tier3_code_edit": tier3,
        "tier3_5_thinking": tier3_5,
        "tier4_drills": tier4,
        "tier5_hardware": tier5,
    }
    ready, failed_checks, recs = check_bootstrap_ready_gate(scorecard, base_ppl=base_ppl)
    scorecard["bootstrap_ready"] = ready
    scorecard["failed_checks"] = failed_checks
    scorecard["recommendations"] = recs
    return scorecard


def main() -> int:
    parser = argparse.ArgumentParser(description="Baseline Scorecard (Tiers 1-5 + Tier 3.5 Thinking) & Bootstrap Gate")
    parser.add_argument("--worktree", type=Path, default=Path("."))
    parser.add_argument("--backend", choices=["scripted", "minimind"], default="scripted")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    parser.add_argument("--heldout-path", type=Path, default=None)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--quantize-int8", action="store_true")
    parser.add_argument("--num-code-tasks", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    resolved_model = args.model_path
    if args.backend == "minimind":
        resolved_model = _resolve_eval_model_path(args.worktree, args.profile, args.model_path)
        if resolved_model is None or not resolved_model.is_file():
            raise FileNotFoundError(f"Could not find MiniMind checkpoint for evaluation (requested={args.model_path})")

    cfg_dict = None
    if args.model_config is not None and args.model_config.is_file():
        cfg_dict = json.loads(args.model_config.read_text(encoding="utf-8"))

    scorecard = evaluate_baseline(
        worktree_dir=args.worktree,
        backend=args.backend,
        model_path=resolved_model,
        tokenizer_path=args.tokenizer_path,
        profile_name=args.profile,
        heldout_path=args.heldout_path,
        num_code_tasks=args.num_code_tasks,
        model_config=cfg_dict,
        quantize_int8=args.quantize_int8,
    )
    if resolved_model is not None:
        scorecard["evaluated_model_path"] = str(resolved_model)
    rendered = json.dumps(scorecard, indent=2)
    if args.output is not None:
        atomic_write_text(args.output, rendered + "\n")
    print(rendered)
    return 0 if scorecard["bootstrap_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
