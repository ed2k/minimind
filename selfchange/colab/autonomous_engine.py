from __future__ import annotations

import ast
import json
import random
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from selfchange.bootstrap.data_quality import (
    _extract_prompt_keys,
    _tc_xml,
    estimate_tokens,
    review_and_record_dataset,
    score_thinking_turn,
)
from selfchange.bootstrap.limitation_to_fix import (
    build_limitation_fix_and_meta_dataset,
    record_verified_selfchange_episode,
)
from selfchange.kernel.autonomous_cycle import AutonomousCycleRunner, CycleConfig
from selfchange.kernel.budgets import ComputeProfile
from selfchange.kernel.integrity import atomic_write_text
from selfchange.kernel.sandbox import scan_python_source
from selfchange.kernel.supervisor import CycleContext
from selfchange.kernel.tool_runtime import SELFCHANGE_TOOLS, WorktreeToolExecutor, validate_tool_call
from selfchange.kernel.verifier import compute_rlvr_reward
from selfchange.mutable.agent_loop import parse_tool_calls
from selfchange.mutable.data_curator import compute_rep_penalty, curate_rft_samples, build_dpo_pairs, mix_with_anchor
from selfchange.mutable.prompts import SYSTEM_PROMPT
from selfchange.mutable.train_recipe import select_training_stage
from selfchange.training.model_loader import load_model_config
from selfchange.training.self_train_runner import (
    format_dpo_record,
    format_sft_record,
    promote_lora_adapter,
    run_training_stage,
    stage_from_weight_for_trainer,
    write_dpo_jsonl,
    write_sft_jsonl,
)

_TOOLS_BY_NAME = {t["function"]["name"]: t for t in SELFCHANGE_TOOLS}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compact_tools(names: list[str]) -> list[dict]:
    selected: list[dict] = []
    for name in names:
        tool = _TOOLS_BY_NAME.get(name)
        if not tool:
            continue
        fn = tool["function"]
        selected.append({"type": "function", "function": {"name": fn["name"], "parameters": fn["parameters"]}})
    return selected


def _apply_replace_line_range_strict(worktree_dir: Path, filename: str, broken_src: str, clean_src: str, args: dict) -> tuple[bool, bool, bool]:
    """Strictly validate integer line bounds, replay via WorktreeToolExecutor, check AST & exact match."""
    s_val = args.get("start_line")
    e_val = args.get("end_line")
    lines_arg = args.get("lines")
    if isinstance(s_val, bool) or isinstance(e_val, bool) or not isinstance(s_val, int) or not isinstance(e_val, int):
        return False, False, False
    if not isinstance(lines_arg, list) or not all(isinstance(x, str) for x in lines_arg):
        return False, False, False

    m_dir = Path(worktree_dir) / "selfchange" / "mutable"
    m_dir.mkdir(parents=True, exist_ok=True)
    target = m_dir / Path(filename).name
    target.write_text(broken_src, encoding="utf-8")
    executor = WorktreeToolExecutor(worktree_dir, autonomy_level=1)
    res = executor.dispatch("replace_line_range", args)
    if not res.get("ok"):
        return False, False, False
    edited = target.read_text(encoding="utf-8")
    try:
        ast.parse(edited)
    except SyntaxError:
        return True, False, False
    if not scan_python_source(edited).ok:
        return True, False, False
    return True, True, edited.strip() == clean_src.strip()


STRICT_EVAL_SUITE: list[dict] = [
    # 4 operational tool-call tasks
    {
        "id": "eval_1_tool_oom",
        "category": "tool_call_op",
        "check": "tool_oom",
        "messages": [
            {"role": "system", "content": "Diagnose OOM and call read_log_tail or patch_config."},
            {"role": "user", "content": "fault=oom module=train_pretrain.py batch_size=12 accumulation_steps=1"},
        ],
        "tools": _compact_tools(["read_log_tail", "patch_config"]),
    },
    {
        "id": "eval_2_tool_nan",
        "category": "tool_call_op",
        "check": "tool_nan",
        "messages": [
            {"role": "system", "content": "Stabilize NaN loss by halving lr to 0.0001 and setting grad_clip=0.5 via patch_config."},
            {"role": "user", "content": "fault=nan_loss step=40 lr=0.0002 grad_norm=inf"},
        ],
        "tools": _compact_tools(["patch_config", "read_log_tail"]),
    },
    {
        "id": "eval_3_tool_rollback",
        "category": "tool_call_op",
        "check": "tool_rollback",
        "ancestor_id": "cand_884",
        "messages": [
            {"role": "system", "content": "Corrupt checkpoint detected. Roll back to verified ancestor using rollback_to_ancestor."},
            {"role": "user", "content": "fault=corrupt_checkpoint file=ckpt_bad_768_resume.pth verified_ancestor=cand_884"},
        ],
        "tools": _compact_tools(["rollback_to_ancestor", "get_archive_summary"]),
    },
    {
        "id": "eval_4_tool_malformed",
        "category": "tool_call_op",
        "check": "tool_malformed",
        "messages": [
            {"role": "system", "content": "Stabilize decoding temperature <= 0.1 using patch_config."},
            {"role": "user", "content": "fault=malformed_tool_loop temperature=0.9 tag_errors=3"},
        ],
        "tools": _compact_tools(["patch_config", "get_system_state"]),
    },
    # 6 Python AST code repair tasks (3 classic + 3 multi-line float literals)
    {
        "id": "eval_5_code_colon",
        "category": "code_edit",
        "check": "code_edit",
        "filename": "eval_oom_util.py",
        "broken": "def compute_eval_step(batch_size: int, div: int = 2) -> int\n    return max(1, batch_size // div)\n",
        "clean": "def compute_eval_step(batch_size: int, div: int = 2) -> int:\n    return max(1, batch_size // div)\n",
        "messages": [
            {"role": "system", "content": "Fix the Python syntax bug using replace_line_range."},
            {"role": "user", "content": "eval_oom_util.py:\n1: def compute_eval_step(batch_size: int, div: int = 2) -> int\n2:     return max(1, batch_size // div)"},
        ],
        "tools": _compact_tools(["replace_line_range"]),
    },
    {
        "id": "eval_6_code_cmp",
        "category": "code_edit",
        "check": "code_edit",
        "filename": "eval_step_cmp.py",
        "broken": "def within_token_cap(tokens: int, limit: int = 691) -> bool:\n    return tokens >= limit\n",
        "clean": "def within_token_cap(tokens: int, limit: int = 691) -> bool:\n    return tokens <= limit\n",
        "messages": [
            {"role": "system", "content": "Fix comparison bug on line 2 using replace_line_range (should be <=)."},
            {"role": "user", "content": "eval_step_cmp.py:\n1: def within_token_cap(tokens: int, limit: int = 691) -> bool:\n2:     return tokens >= limit"},
        ],
        "tools": _compact_tools(["replace_line_range"]),
    },
    {
        "id": "eval_7_code_div",
        "category": "code_edit",
        "check": "code_edit",
        "filename": "eval_seq_div.py",
        "broken": "def next_eval_batch(batch_size: int, factor: int = 2) -> int:\n    return max(1, batch_size * factor)\n",
        "clean": "def next_eval_batch(batch_size: int, factor: int = 2) -> int:\n    return max(1, batch_size // factor)\n",
        "messages": [
            {"role": "system", "content": "Fix floor-division bug on line 2 using replace_line_range (halve batch_size with //)."},
            {"role": "user", "content": "eval_seq_div.py:\n1: def next_eval_batch(batch_size: int, factor: int = 2) -> int:\n2:     return max(1, batch_size * factor)"},
        ],
        "tools": _compact_tools(["replace_line_range"]),
    },
    {
        "id": "eval_8_code_clamp_float1",
        "category": "code_edit",
        "check": "code_edit",
        "filename": "eval_lr_guard.py",
        "broken": "def clamp_lr(lr: float, max_lr: float = 5e-4) -> float:\n    if lr < 1e-8:\n        return 1e-8\n    return max(lr, max_lr)\n",
        "clean": "def clamp_lr(lr: float, max_lr: float = 5e-4) -> float:\n    if lr < 1e-8:\n        return 1e-8\n    return min(lr, max_lr)\n",
        "messages": [
            {"role": "system", "content": "Fix the logic bug on line 4 using replace_line_range (should cap lr with min, not max)."},
            {"role": "user", "content": "eval_lr_guard.py:\n1: def clamp_lr(lr: float, max_lr: float = 5e-4) -> float:\n2:     if lr < 1e-8:\n3:         return 1e-8\n4:     return max(lr, max_lr)"},
        ],
        "tools": _compact_tools(["replace_line_range"]),
    },
    {
        "id": "eval_9_code_clamp_float2",
        "category": "code_edit",
        "check": "code_edit",
        "filename": "eval_clip_cap.py",
        "broken": "def bound_clip(gn: float, top_clip: float = 2.5e-3) -> float:\n    if gn < 1e-7:\n        return 1e-7\n    return max(gn, top_clip)\n",
        "clean": "def bound_clip(gn: float, top_clip: float = 2.5e-3) -> float:\n    if gn < 1e-7:\n        return 1e-7\n    return min(gn, top_clip)\n",
        "messages": [
            {"role": "system", "content": "Fix line 4 with integer line indices via replace_line_range (cap with min, not max)."},
            {"role": "user", "content": "eval_clip_cap.py:\n1: def bound_clip(gn: float, top_clip: float = 2.5e-3) -> float:\n2:     if gn < 1e-7:\n3:         return 1e-7\n4:     return max(gn, top_clip)"},
        ],
        "tools": _compact_tools(["replace_line_range"]),
    },
    {
        "id": "eval_10_code_clamp_float3",
        "category": "code_edit",
        "check": "code_edit",
        "filename": "eval_wd_bound.py",
        "broken": "def floor_wd(wd: float, floor_val: float = 1e-5) -> float:\n    if wd > 0.85:\n        return 0.85\n    return min(wd, floor_val)\n",
        "clean": "def floor_wd(wd: float, floor_val: float = 1e-5) -> float:\n    if wd > 0.85:\n        return 0.85\n    return max(wd, floor_val)\n",
        "messages": [
            {"role": "system", "content": "Fix line 4 lower bound with replace_line_range (use max, not min)."},
            {"role": "user", "content": "eval_wd_bound.py:\n1: def floor_wd(wd: float, floor_val: float = 1e-5) -> float:\n2:     if wd > 0.85:\n3:         return 0.85\n4:     return min(wd, floor_val)"},
        ],
        "tools": _compact_tools(["replace_line_range"]),
    },
    # 4 exact multi-step math tasks (no substring loopholes)
    {
        "id": "eval_11_math_oom_2step",
        "category": "reasoning_math",
        "check": "math_pair",
        "expected_bs": 4,
        "expected_acc": 8,
        "messages": [
            {"role": "user", "content": "If batch_size=16 and accumulation_steps=2, and we halve batch_size twice after two OOM errors while doubling accumulation_steps each time, what are the final batch_size and accumulation_steps?"},
        ],
        "tools": None,
    },
    {
        "id": "eval_12_math_oom_3step",
        "category": "reasoning_math",
        "check": "math_pair",
        "expected_bs": 4,
        "expected_acc": 8,
        "messages": [
            {"role": "user", "content": "Start with batch_size=32 and accumulation_steps=1. After 3 OOM halving step(s) where batch_size halves and accumulation_steps doubles each step, what are final batch_size and accumulation_steps?"},
        ],
        "tools": None,
    },
    {
        "id": "eval_13_math_lr_2step",
        "category": "reasoning_math",
        "check": "math_token",
        "expected_patterns": [r"\b0\.0001\b|\b1e-0?4\b"],
        "forbidden_patterns": [r"\bfinal\s+learning_rate\s*(?:=|is|:)?\s*0\.0002\b"],
        "messages": [
            {"role": "user", "content": "Starting at learning_rate=0.0004, if NaN recovery halves the learning rate 2 times, what is the final learning_rate?"},
        ],
        "tools": None,
    },
    {
        "id": "eval_14_math_seq_budget",
        "category": "reasoning_math",
        "check": "math_token",
        "expected_patterns": [r"\b691\b"],
        "forbidden_patterns": [],
        "messages": [
            {"role": "user", "content": "If max_seq_len=768 and the safe pre-truncation budget ratio is 0.9, what is the maximum allowed training sequence token count?"},
        ],
        "tools": None,
    },
    # 3 healthy no-op restraint tasks
    {
        "id": "eval_15_noop_c12",
        "category": "noop",
        "check": "healthy_noop",
        "messages": [
            {"role": "system", "content": "Monitor system health. Only call tools if there is an actual fault."},
            {"role": "user", "content": "fault=healthy cycle=12 loss=0.842 consecutive_failures=0"},
        ],
        "tools": _compact_tools(["get_system_state", "patch_config"]),
    },
    {
        "id": "eval_16_noop_c27",
        "category": "noop",
        "check": "healthy_noop",
        "messages": [
            {"role": "system", "content": "Inspect system state safely; never mutate config when fault=healthy."},
            {"role": "user", "content": "fault=healthy cycle=27 loss=0.514 consecutive_failures=0"},
        ],
        "tools": _compact_tools(["get_system_state", "patch_config"]),
    },
    {
        "id": "eval_17_noop_c44",
        "category": "noop",
        "check": "healthy_noop",
        "messages": [
            {"role": "system", "content": "Check training metrics. Do not call patch_config on healthy runs."},
            {"role": "user", "content": "health_report status=healthy cycle=44 loss=0.391 consecutive_failures=0"},
        ],
        "tools": _compact_tools(["get_system_state", "patch_config"]),
    },
    # 3 core technical grounding QA tasks
    {
        "id": "eval_18_qa_zh_accum",
        "category": "core_tech_qa",
        "check": "qa_zh_accum",
        "messages": [
            {"role": "user", "content": "什么是梯度累积？它在显存受限时有什么作用？"},
        ],
        "tools": None,
    },
    {
        "id": "eval_19_qa_en_clip",
        "category": "core_tech_qa",
        "check": "qa_en_clip",
        "messages": [
            {"role": "user", "content": "What does torch.nn.utils.clip_grad_norm_ prevent during fp16 training?"},
        ],
        "tools": None,
    },
    {
        "id": "eval_20_qa_ast_worktree",
        "category": "core_tech_qa",
        "check": "qa_ast_worktree",
        "messages": [
            {"role": "user", "content": "Why validate edited Python code in a git worktree with ast.parse before execution?"},
        ],
        "tools": None,
    },
]

STRICT_HELDOUT_PROMPTS: list[dict] = [{"id": t["id"], "messages": t["messages"]} for t in STRICT_EVAL_SUITE]


def score_strict_task(task: dict, reply: str, eos_ok: bool) -> dict:
    """Deterministically verify one strict eval or exploration rollout without substring loopholes."""
    text = reply or ""
    calls, tag_errs = parse_tool_calls(text)
    has_nonempty_think = "<think>" in text and "</think>" in text and "<think>\n</think>" not in text and "<think></think>" not in text
    ctx_text = "\n".join(str(m.get("content", "")) for m in task.get("messages", []))
    ts = score_thinking_turn(ctx_text, text)
    ans_part = text.split("</think>")[-1].strip() if "</think>" in text else text.strip()
    ans_low = ans_part.lower()
    rep = compute_rep_penalty(ans_part or text)

    chk = task["check"]
    passed = False
    note = ""

    if chk == "tool_oom":
        cl = calls[0] if calls else None
        if cl and cl["name"] == "read_log_tail":
            ok_v, _ = validate_tool_call("read_log_tail", cl["arguments"])
            passed = ok_v and isinstance(cl["arguments"].get("log_name"), str)
        elif cl and cl["name"] == "patch_config":
            ok_v, _ = validate_tool_call("patch_config", cl["arguments"])
            passed = ok_v and isinstance(cl["arguments"].get("updates"), dict)
        passed = passed and tag_errs == 0 and eos_ok
        note = f"calls={calls}"
    elif chk == "tool_nan":
        pc = next((cl for cl in calls if cl["name"] == "patch_config"), None)
        upd = pc["arguments"].get("updates", {}) if (pc and isinstance(pc.get("arguments"), dict)) else {}
        tr = upd.get("training", upd) if isinstance(upd, dict) else {}
        ok_v = validate_tool_call("patch_config", pc["arguments"])[0] if pc else False
        passed = tag_errs == 0 and eos_ok and ok_v and isinstance(upd, dict) and (tr.get("learning_rate") == 0.0001 or tr.get("lr") == 0.0001)
        note = f"calls={calls}"
    elif chk == "tool_rollback":
        rb = next((cl for cl in calls if cl["name"] == "rollback_to_ancestor"), None)
        want = str(task.get("ancestor_id", ""))
        passed = tag_errs == 0 and eos_ok and rb is not None and rb.get("arguments", {}).get("node_id") == want
        note = f"calls={calls}"
    elif chk == "tool_malformed":
        pc = next((cl for cl in calls if cl["name"] == "patch_config"), None)
        upd = pc["arguments"].get("updates", {}) if (pc and isinstance(pc.get("arguments"), dict)) else {}
        temp_val = upd.get("temperature") if isinstance(upd, dict) else None
        passed = tag_errs == 0 and eos_ok and isinstance(temp_val, (int, float)) and not isinstance(temp_val, bool) and float(temp_val) <= 0.1
        note = f"calls={calls}"
    elif chk == "code_edit":
        ed = next((cl for cl in calls if cl["name"] == "replace_line_range"), None)
        dispatched, ast_ok, exact_ok = False, False, False
        if ed is not None:
            with tempfile.TemporaryDirectory() as tmp:
                dispatched, ast_ok, exact_ok = _apply_replace_line_range_strict(
                    Path(tmp), task["filename"], task["broken"], task["clean"], ed["arguments"]
                )
        passed = tag_errs == 0 and eos_ok and dispatched and ast_ok and exact_ok
        note = f"dispatched={dispatched} ast={ast_ok} exact={exact_ok}"
    elif chk == "math_pair":
        ebs = int(task["expected_bs"])
        eacc = int(task["expected_acc"])
        has_bs = bool(re.search(rf"\bbatch_size\s*(?:=|is|:)\s*{ebs}\b", ans_low))
        has_acc = bool(re.search(rf"\b(?:accumulation_steps|accum)\s*(?:=|is|:)\s*{eacc}\b", ans_low))
        swapped = bool(ebs != eacc and re.search(rf"\bbatch_size\s*(?:=|is|:)\s*{eacc}\b", ans_low))
        passed = eos_ok and has_bs and has_acc and (not swapped)
        note = f"has_bs={has_bs} has_acc={has_acc} swapped={swapped}"
    elif chk == "math_token":
        ok_all = all(bool(re.search(pat, ans_low)) for pat in task.get("expected_patterns", []))
        bad_any = any(bool(re.search(pat, ans_low)) for pat in task.get("forbidden_patterns", []))
        passed = eos_ok and ok_all and (not bad_any)
        note = f"ok_all={ok_all} bad_any={bad_any}"
    elif chk == "healthy_noop":
        mutated = any(cl["name"] in ("patch_config", "replace_line_range", "patch_file_replace") for cl in calls)
        passed = tag_errs == 0 and eos_ok and (not mutated)
        note = f"calls={[c['name'] for c in calls]}"
    elif chk == "qa_zh_accum":
        passed = eos_ok and rep < 0.20 and ("累积" in ans_part or "梯度" in ans_part) and ("梯度下降" not in ans_part and "gradient descent" not in ans_low)
        note = f"rep={rep:.3f} eos={eos_ok}"
    elif chk == "qa_en_clip":
        passed = eos_ok and rep < 0.15 and any(k in ans_low for k in ("explod", "overflow", "nan", "norm", "gradient")) and ("ios and android" not in ans_low)
        note = f"rep={rep:.3f} eos={eos_ok}"
    elif chk == "qa_ast_worktree":
        passed = eos_ok and rep < 0.15 and any(k in ans_low for k in ("ast", "syntax", "isolat", "worktree", "verify"))
        note = f"rep={rep:.3f} eos={eos_ok}"

    return {
        "id": task["id"],
        "category": task["category"],
        "passed": bool(passed),
        "eos_ok": bool(eos_ok),
        "nonempty_think": bool(has_nonempty_think),
        "think_grounded": bool(ts["grounded"]),
        "think_action_consistent": bool(ts["action_consistent"]),
        "tag_errors": tag_errs,
        "rep_penalty": round(rep, 4),
        "tool_calls": calls,
        "note": note,
        "reply": text,
    }


def _generate_with_model(model, tokenizer, profile: ComputeProfile, messages: list[dict], tools: list[dict] | None, temperature: float) -> tuple[str, bool]:
    import torch  # type: ignore[import-not-found]
    from selfchange.training.model_loader import normalize_messages_for_template, restore_open_thinking_prefix

    norm_messages, extracted_tools = normalize_messages_for_template(messages)
    effective_tools = tools if tools is not None else extracted_tools
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if effective_tools:
        kwargs["tools"] = effective_tools
    try:
        prompt_text = tokenizer.apply_chat_template(norm_messages, open_thinking=True, **kwargs)
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(norm_messages, **kwargs)

    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True).to(model.device)
    do_sample = temperature > 0.0
    gen_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": min(160, profile.max_gen_len),
        "do_sample": do_sample,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9
    with torch.no_grad():
        out_ids = model.generate(**gen_kwargs)
    new_ids = out_ids[0][inputs["input_ids"].size(1) :]
    decoded = tokenizer.decode(new_ids, skip_special_tokens=False)
    eos_tok = tokenizer.eos_token or "<|im_end|>"
    eos_ok = (eos_tok in decoded) or (len(new_ids) > 0 and int(new_ids[-1]) == int(tokenizer.eos_token_id or -1))
    clean = decoded.replace(eos_tok, "").replace(tokenizer.pad_token or "<|endoftext|>", "").strip()
    return restore_open_thinking_prefix(clean, True), eos_ok


def evaluate_model_strict_suite(
    model,
    tokenizer,
    profile: ComputeProfile,
    tasks: list[dict] | None = None,
    max_tasks: int = 20,
    k_rollouts: int = 1,
    reply_fn: Callable[[list[dict], list[dict] | None, float], tuple[str, bool]] | None = None,
) -> dict:
    """Evaluate a live model across STRICT_EVAL_SUITE (no timing noise, no substring math loopholes)."""
    eval_tasks = list(tasks if tasks is not None else STRICT_EVAL_SUITE)[:max_tasks]
    gen = reply_fn if reply_fn is not None else (lambda msgs, tls, temp: _generate_with_model(model, tokenizer, profile, msgs, tls, temp))

    results: list[dict] = []
    pass_k_hits = 0
    for task in eval_tasks:
        first_res: dict | None = None
        any_passed = False
        temps = [0.0] if k_rollouts <= 1 else [0.0, 0.35, 0.55, 0.70][:k_rollouts]
        for idx, temp in enumerate(temps):
            reply, eos_ok = gen(task["messages"], task.get("tools"), temp)
            scored = score_strict_task(task, reply, eos_ok)
            if idx == 0:
                first_res = scored
            if scored["passed"]:
                any_passed = True
        assert first_res is not None
        first_res["pass_k"] = any_passed
        if any_passed:
            pass_k_hits += 1
        results.append(first_res)

    n = max(1, len(results))
    pass_at_1 = round(sum(1 for r in results if r["passed"]) / n, 4)
    pass_at_k = round(pass_k_hits / n, 4)
    eos_rate = round(sum(1 for r in results if r["eos_ok"]) / n, 4)
    think_ground = round(sum(1 for r in results if r["think_grounded"]) / n, 4)
    think_consist = round(sum(1 for r in results if r["think_action_consistent"]) / n, 4)

    tool_items = [r for r in results if r["category"] == "tool_call_op"]
    code_items = [r for r in results if r["category"] == "code_edit"]
    math_items = [r for r in results if r["category"] == "reasoning_math"]
    noop_items = [r for r in results if r["category"] == "noop"]
    schema_items = [r for r in results if r["category"] in ("tool_call_op", "code_edit")]

    schema_rate = round(sum(1 for r in schema_items if r["tag_errors"] == 0 and r["tool_calls"]) / max(1, len(schema_items)), 4)
    tool_pass = round(sum(1 for r in tool_items if r["passed"]) / max(1, len(tool_items)), 4)
    code_pass = round(sum(1 for r in code_items if r["passed"]) / max(1, len(code_items)), 4)
    math_pass = round(sum(1 for r in math_items if r["passed"]) / max(1, len(math_items)), 4)
    noop_pass = round(sum(1 for r in noop_items if r["passed"]) / max(1, len(noop_items)), 4)

    overall_score = round(
        0.10 * eos_rate
        + 0.25 * tool_pass
        + 0.25 * code_pass
        + 0.15 * math_pass
        + 0.10 * noop_pass
        + 0.15 * (0.5 * think_ground + 0.5 * think_consist),
        4,
    )

    return {
        "evaluated_at": _utc_now(),
        "passed_count": sum(1 for r in results if r["passed"]),
        "total_count": len(results),
        "pass_at_1": pass_at_1,
        "pass_at_4": pass_at_k if k_rollouts > 1 else pass_at_1,
        "eos_rate": eos_rate,
        "schema_validity_rate": schema_rate,
        "tool_pass_at_1": tool_pass,
        "code_pass_at_1": code_pass,
        "math_pass_at_1": math_pass,
        "noop_pass_rate": noop_pass,
        "think_grounding_rate": think_ground,
        "think_action_consistency_rate": think_consist,
        "nonempty_think_rate": round(sum(1 for r in results if r["nonempty_think"]) / n, 4),
        "overall_score": overall_score,
        "failing_task_ids": [r["id"] for r in results if not r["passed"]],
        "examples": results,
    }


def _procedural_exploration_bank(cycle_index: int, rng: random.Random) -> dict[str, list[dict]]:
    step = 20 + cycle_index * 7 + rng.randint(1, 9)
    bs = rng.choice([6, 8, 12])
    anc = f"anc_exp_{300 + cycle_index * 11 + rng.randint(1, 7)}"
    lo_val = rng.choice(["1e-6", "2e-6", "1e-5"])
    hi_val = rng.choice(["3e-4", "4e-4", "0.25", "0.45"])
    b_start, a_start, halves, f_b, f_a = rng.choice([(24, 1, 2, 6, 4), (48, 1, 3, 6, 8), (20, 2, 2, 5, 8)])
    cyc = 60 + cycle_index * 5 + rng.randint(1, 4)

    code_clip_clean = f"def exp_bound_val(val: float, hi: float = {hi_val}) -> float:\n    if val < {lo_val}:\n        return {lo_val}\n    return min(val, hi)\n"
    code_clip_broken = code_clip_clean.replace("return min(val, hi)", "return max(val, hi)")
    code_fn = f"exp_bound_{cycle_index}.py"
    code_num = "\n".join(f"{i + 1}: {ln}" for i, ln in enumerate(code_clip_broken.splitlines()))

    return {
        "tool_call_op": [
            {
                "id": f"exp_c{cycle_index}_nan",
                "category": "tool_call_op",
                "check": "tool_nan",
                "messages": [
                    {"role": "system", "content": "Stabilize NaN loss by halving lr to 0.0001 and setting grad_clip=0.5 via patch_config."},
                    {"role": "user", "content": f"fault=nan_loss step={step} lr=0.0002 grad_norm=inf"},
                ],
                "tools": _compact_tools(["patch_config", "read_log_tail"]),
            },
            {
                "id": f"exp_c{cycle_index}_rb",
                "category": "tool_call_op",
                "check": "tool_rollback",
                "ancestor_id": anc,
                "messages": [
                    {"role": "system", "content": "Corrupt checkpoint detected. Roll back to verified ancestor using rollback_to_ancestor."},
                    {"role": "user", "content": f"fault=corrupt_checkpoint file=resume_c{cycle_index}.pth verified_ancestor={anc}"},
                ],
                "tools": _compact_tools(["rollback_to_ancestor", "get_archive_summary"]),
            },
        ],
        "code_edit": [
            {
                "id": f"exp_c{cycle_index}_code",
                "category": "code_edit",
                "check": "code_edit",
                "filename": code_fn,
                "broken": code_clip_broken,
                "clean": code_clip_clean,
                "messages": [
                    {"role": "system", "content": "Fix line 4 with integer line indices via replace_line_range (cap with min, not max)."},
                    {"role": "user", "content": f"{code_fn}:\n{code_num}"},
                ],
                "tools": _compact_tools(["replace_line_range"]),
            },
        ],
        "reasoning_math": [
            {
                "id": f"exp_c{cycle_index}_math",
                "category": "reasoning_math",
                "check": "math_pair",
                "expected_bs": f_b,
                "expected_acc": f_a,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Start with batch_size={b_start} and accumulation_steps={a_start}. "
                            f"After {halves} OOM halving step(s) where batch_size halves and accumulation_steps doubles each step, "
                            f"what are final batch_size and accumulation_steps?"
                        ),
                    }
                ],
                "tools": None,
            }
        ],
        "noop": [
            {
                "id": f"exp_c{cycle_index}_noop",
                "category": "noop",
                "check": "healthy_noop",
                "messages": [
                    {"role": "system", "content": "Monitor system health. Do not call patch_config on healthy runs."},
                    {"role": "user", "content": f"health_report status=healthy cycle={cyc} loss=0.412 consecutive_failures=0"},
                ],
                "tools": _compact_tools(["get_system_state", "patch_config"]),
            }
        ],
    }


def build_exploration_tasks(scorecard: dict | None, cycle_index: int, max_tasks: int = 4, seed: int = 42) -> list[dict]:
    """Select procedural non-heldout exploration tasks biased toward live scorecard weaknesses."""
    rng = random.Random(seed + cycle_index * 31)
    bank = _procedural_exploration_bank(cycle_index, rng)
    scores = {
        "code_edit": float((scorecard or {}).get("code_pass_at_1", 0.5)),
        "reasoning_math": float((scorecard or {}).get("math_pass_at_1", 0.5)),
        "tool_call_op": float((scorecard or {}).get("tool_pass_at_1", 0.5)),
    }
    ordered_cats = sorted(scores.keys(), key=lambda k: scores[k])
    tasks: list[dict] = []
    for cat in ordered_cats:
        for item in bank.get(cat, []):
            if len(tasks) < max_tasks:
                tasks.append(item)
    if len(tasks) < max_tasks and float((scorecard or {}).get("noop_pass_rate", 1.0)) < 1.0:
        tasks.append(bank["noop"][0])
    # Guarantee diversity across tool/code/math if max_tasks allows
    for cat in ("tool_call_op", "code_edit", "reasoning_math"):
        for item in bank.get(cat, []):
            if len(tasks) >= max_tasks:
                break
            if item["id"] not in {t["id"] for t in tasks}:
                tasks.append(item)
    return tasks[:max_tasks]


def run_multi_rollout_exploration(
    model,
    tokenizer,
    profile: ComputeProfile,
    tasks: list[dict],
    k_rollouts: int = 4,
    reply_fn: Callable[[list[dict], list[dict] | None, float], tuple[str, bool]] | None = None,
) -> dict:
    """Sample K rollouts per shared task_id to produce live pass@1, pass@4, RFT winners, and same-task DPO pairs."""
    gen = reply_fn if reply_fn is not None else (lambda msgs, tls, temp: _generate_with_model(model, tokenizer, profile, msgs, tls, temp))
    temps = [0.0, 0.35, 0.55, 0.70]
    trajectories: list[dict] = []
    p1_hits = 0
    pk_hits = 0

    for task in tasks:
        any_pass = False
        for k_idx in range(k_rollouts):
            temp = temps[min(k_idx, len(temps) - 1)]
            reply, eos_ok = gen(task["messages"], task.get("tools"), temp)
            scored = score_strict_task(task, reply, eos_ok)
            if k_idx == 0 and scored["passed"]:
                p1_hits += 1
            if scored["passed"]:
                any_pass = True

            convs: list[dict] = []
            for idx, m in enumerate(task["messages"]):
                msg_copy = dict(m)
                if idx == 0 and msg_copy.get("role") == "system" and task.get("tools"):
                    msg_copy["tools"] = json.dumps(task["tools"], ensure_ascii=False)
                convs.append(msg_copy)
            convs.append({"role": "assistant", "content": reply})

            reward = compute_rlvr_reward(
                turn_outputs=[reply],
                tool_calls=scored["tool_calls"],
                observed_before_edit=scored["think_grounded"],
                first_edit_valid=scored["passed"] or scored["tag_errors"] == 0,
                task_passed=scored["passed"],
                boundary_violated=False,
                unfinished=not scored["eos_ok"],
                over_budget=False,
                open_thinking=True,
            )
            trajectories.append(
                {
                    "task_id": task["id"],
                    "task_family": f"explore_{task['category']}",
                    "passed": scored["passed"],
                    "reward": reward,
                    "tool_calls": scored["tool_calls"],
                    "messages": convs,
                    "final_text": reply,
                    "eos_ok": scored["eos_ok"],
                    "rollout_index": k_idx,
                    "temperature": temp,
                }
            )
        if any_pass:
            pk_hits += 1

    n_t = max(1, len(tasks))
    pass1 = round(p1_hits / n_t, 4)
    passk = round(pk_hits / n_t, 4)
    return {
        "tasks_count": len(tasks),
        "k_rollouts": k_rollouts,
        "pass_at_1": pass1,
        "pass_at_4": passk,
        "trajectories": trajectories,
    }


def diagnose_weakness_state(scorecard: dict | None, cycle_index: int, parent_node_id: str | None, cfg: dict, safe_mode: bool, config_only: bool) -> dict:
    """Convert real live scorecard failures into an actionable fault card instead of fault=none."""
    fault = "none"
    target_hint = f"cycle={cycle_index} safe_mode={safe_mode} config_only={config_only}"
    if scorecard:
        if float(scorecard.get("code_pass_at_1", 1.0)) < 0.85:
            fault = "code_edit_line_index_or_syntax"
            target_hint += f" code_pass1={scorecard.get('code_pass_at_1')} failing={','.join(scorecard.get('failing_task_ids', [])[:3])}"
        elif float(scorecard.get("math_pass_at_1", 1.0)) < 0.75:
            fault = "oom_math_halving_precision"
            target_hint += f" math_pass1={scorecard.get('math_pass_at_1')}"
        elif float(scorecard.get("tool_pass_at_1", 1.0)) < 1.0:
            fault = "tool_schema_or_arg_mismatch"
            target_hint += f" tool_pass1={scorecard.get('tool_pass_at_1')}"
        elif float(scorecard.get("eos_rate", 1.0)) < 0.95:
            fault = "eos_truncation_or_loop"
            target_hint += f" eos_rate={scorecard.get('eos_rate')}"

    return {
        "profile": str(cfg.get("profile", "colab_t4")),
        "last_fault": fault if safe_mode is False else "safe_mode_check",
        "champion_id": parent_node_id,
        "consecutive_failures": 0 if fault == "none" else 1,
        "training": cfg.get("training", {}),
        "log_tail": target_hint,
    }


class SelfImprovingCycleRunner(AutonomousCycleRunner):
    """Autonomous runner using live strict eval, multi-rollout exploration, weakness-targeted synthesis, and gated promotion."""

    def __init__(
        self,
        repo_root: Path,
        workspace_dir: Path,
        profile: ComputeProfile,
        proposer_fn=None,
        kernel_dir: Path | None = None,
        manifest_path: Path | None = None,
        cycle_config: CycleConfig | None = None,
        extra_eval_fn=None,
        session=None,
        live_self_improve: bool = True,
        model_config: dict | None = None,
        curated_episodes_dir: Path | None = None,
    ):
        super().__init__(
            repo_root=repo_root,
            workspace_dir=workspace_dir,
            profile=profile,
            proposer_fn=proposer_fn,
            kernel_dir=kernel_dir,
            manifest_path=manifest_path,
            cycle_config=cycle_config,
            extra_eval_fn=extra_eval_fn,
        )
        self.session = session
        self.live_self_improve = live_self_improve
        self.model_config = dict(model_config) if model_config else None
        self.curated_episodes_dir = Path(curated_episodes_dir) if curated_episodes_dir else (self.workspace_dir / "curated_data")
        self._ep_counter = 0

    def _collect_live_metrics(self, ctx: CycleContext, rft: list[dict], dpo: list[dict]) -> dict:
        sc = getattr(self.session, "last_scorecard", None) or {}
        exp = getattr(self.session, "last_exploration", None) or {}
        exp_p1 = float(exp.get("pass_at_1", 0.0))
        exp_p4 = float(exp.get("pass_at_4", 0.0))
        sc_p1 = float(sc.get("pass_at_1", 0.5))
        p1 = exp_p1 if exp else sc_p1
        p4 = max(p1, exp_p4 if exp else float(sc.get("pass_at_4", sc_p1)))
        eos = float(sc.get("eos_rate", 1.0))
        return {
            "pass_at_1": p1,
            "pass_at_4": p4,
            "eos_rate": eos,
            "rl_ready": bool((not ctx.safe_mode) and (p4 > p1 > 0.0) and eos >= 0.85 and len(rft) >= 4),
            "overall_score": float(sc.get("overall_score", 0.5)),
            "code_pass_at_1": float(sc.get("code_pass_at_1", 0.5)),
            "math_pass_at_1": float(sc.get("math_pass_at_1", 0.5)),
            "noop_pass_rate": float(sc.get("noop_pass_rate", 1.0)),
        }

    def _run_periodic_training(self, ctx: CycleContext, worktree_dir: Path) -> str | None:
        if self.cycle_config.dry_run_training or not self.live_self_improve or self.session is None:
            return super()._run_periodic_training(ctx, worktree_dir)

        trajs = (
            [json.loads(ln) for ln in self.trajectories_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if self.trajectories_path.is_file()
            else []
        )
        rft = curate_rft_samples(trajs, max_per_family=40)
        dpo = build_dpo_pairs(trajs)
        live_metrics = self._collect_live_metrics(ctx, rft, dpo)

        cfg_path = worktree_dir / "selfchange" / "mutable" / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        before_sc = getattr(self.session, "last_scorecard", None) or self.session.eval_strict_scorecard()

        # Synthesize weakness-targeted data + replay carried-over episodes without heldout leakage
        diag_eval_path = self.workspace_dir / f"live_scorecard_c{ctx.cycle_index:04d}.json"
        atomic_write_text(diag_eval_path, json.dumps(before_sc, indent=2, ensure_ascii=False) + "\n")
        cur_ep_sft = self.curated_episodes_dir / "verified_selfchange_episodes.jsonl"
        cur_ep_dpo = self.curated_episodes_dir / "verified_selfchange_dpo.jsonl"
        synth_dir = self.workspace_dir / "training_data" / f"synth_c{ctx.cycle_index:04d}"
        max_seq = min(768, self.profile.max_seq_len)

        synth_rep = build_limitation_fix_and_meta_dataset(
            repo_root=self.repo_root,
            output_dir=synth_dir,
            eval_report_path=diag_eval_path,
            sft_count=320,
            dpo_count=80,
            max_seq_len=max_seq,
            seed=42 + ctx.cycle_index * 13,
            extra_sft_paths=[cur_ep_sft] if cur_ep_sft.is_file() else None,
            extra_dpo_paths=[cur_ep_dpo] if cur_ep_dpo.is_file() else None,
            extra_heldout_prompts=STRICT_HELDOUT_PROMPTS,
        )

        heldout_path = Path(synth_rep["heldout_path"])
        heldout_lines = [json.loads(ln) for ln in heldout_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        h_exact = {_extract_prompt_keys(r["conversations"])[0] for r in heldout_lines}
        h_norm = {_extract_prompt_keys(r["conversations"])[1] for r in heldout_lines}

        synth_sft = [json.loads(ln) for ln in Path(synth_rep["sft_path"]).read_text(encoding="utf-8").splitlines() if ln.strip()]
        synth_dpo = (
            [json.loads(ln) for ln in Path(synth_rep["dpo_path"]).read_text(encoding="utf-8").splitlines() if ln.strip()]
            if Path(synth_rep["dpo_path"]).is_file()
            else []
        )

        # Merge non-leaking, tag-clean & grounded on-policy RFT winners + same-task DPO pairs
        safe_rft: list[dict] = []
        non_noop_rft: list[dict] = []
        noop_rft: list[dict] = []
        for s in rft:
            rec = format_sft_record(s)
            convs = rec["conversations"]
            ex_u, nm_u = _extract_prompt_keys(convs)
            if (ex_u and ex_u in h_exact) or (nm_u and nm_u in h_norm):
                continue
            asst_msgs = [m for m in convs if m.get("role") == "assistant"]
            if not asst_msgs:
                continue
            asst_txt = str(asst_msgs[-1].get("content", ""))
            ctx_txt = "\n".join(str(m.get("content", "")) for m in convs if m.get("role") != "assistant")
            _, tag_errs = parse_tool_calls(asst_txt)
            ts = score_thinking_turn(ctx_txt, asst_txt)
            if tag_errs != 0 or not ts["has_think"] or ts["think_chars"] <= 0 or not ts["grounded"] or not ts["action_consistent"]:
                continue
            if estimate_tokens(convs) > max_seq - 32:
                continue
            if "no edits required" in asst_txt.lower() or "system is healthy" in asst_txt.lower():
                noop_rft.append(rec)
            else:
                non_noop_rft.append(rec)
        safe_rft.extend(non_noop_rft)
        safe_rft.extend(noop_rft[: max(1, len(non_noop_rft) // 3)])
        merged_sft = safe_rft + synth_sft

        safe_on_dpo: list[dict] = []
        for p in dpo:
            rec = format_dpo_record(p)
            ch_convs = rec["chosen"]
            ex_u, nm_u = _extract_prompt_keys(ch_convs)
            if (ex_u and ex_u in h_exact) or (nm_u and nm_u in h_norm):
                continue
            ch_asst = [m for m in ch_convs if m.get("role") == "assistant"]
            rej_asst = [m for m in rec["rejected"] if m.get("role") == "assistant"]
            if not ch_asst or not rej_asst:
                continue
            ch_txt = str(ch_asst[-1].get("content", ""))
            rej_txt = str(rej_asst[-1].get("content", ""))
            if ch_txt.strip() == rej_txt.strip():
                continue
            ctx_txt = "\n".join(str(m.get("content", "")) for m in ch_convs if m.get("role") != "assistant")
            _, tag_errs = parse_tool_calls(ch_txt)
            ts = score_thinking_turn(ctx_txt, ch_txt)
            if tag_errs != 0 or not ts["has_think"] or ts["think_chars"] <= 0 or not ts["grounded"] or not ts["action_consistent"]:
                continue
            if estimate_tokens(ch_convs) > max_seq - 32 or estimate_tokens(rec["rejected"]) > max_seq - 32:
                continue
            safe_on_dpo.append(rec)
        merged_dpo = safe_on_dpo + synth_dpo

        recipe = select_training_stage(
            metrics=live_metrics,
            buffer_stats={"rft_samples": len(merged_sft), "dpo_pairs": len(merged_dpo)},
            config=cfg,
            safe_mode=ctx.safe_mode,
        )
        stage = recipe["stage"]
        if stage == "none":
            stage = "dpo" if (len(merged_dpo) >= 8 and ctx.safe_mode) else "lora"
        if stage == "rlvr" and len(rft) < 4:
            stage = "dpo" if len(merged_dpo) >= 8 else "lora"

        data_dir = self.workspace_dir / "training_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        if stage == "dpo":
            data_file = data_dir / f"dpo_c{ctx.cycle_index:04d}.jsonl"
            write_dpo_jsonl(data_file, merged_dpo)
        else:
            data_file = data_dir / f"rft_c{ctx.cycle_index:04d}.jsonl"
            write_sft_jsonl(data_file, merged_sft)

        from_weight, from_weight_path = self._resolve_training_from_weight(ctx)
        eff_arch = self.model_config or (load_model_config(from_weight_path) if from_weight_path else None) or getattr(self.session, "model_config", None)
        train_cfg = dict(recipe["training"])
        train_cfg["heldout_path"] = str(heldout_path)
        train_cfg["max_seq_len"] = max_seq
        train_cfg.setdefault("epochs", 1)
        train_cfg.setdefault("batch_size", min(4, self.profile.batch_size))
        train_cfg.setdefault("accumulation_steps", 2)
        if stage == "dpo":
            train_cfg["dpo_learning_rate"] = max(2e-7, float(train_cfg.get("dpo_learning_rate", 2e-7)))
        elif stage == "lora":
            train_cfg["learning_rate"] = min(5e-5, max(2e-5, float(train_cfg.get("learning_rate", 3e-5))))

        res = run_training_stage(
            stage=stage,
            repo_root=self.repo_root,
            workspace_dir=self.workspace_dir,
            data_path=data_file,
            profile=self.profile,
            training_cfg=train_cfg,
            from_weight=from_weight,
            save_weight=f"cand_c{ctx.cycle_index:04d}",
            dry_run=False,
            from_weight_path=from_weight_path,
            model_config=eff_arch,
        )

        final_weight_path: str | None = res.output_weight_path if res.ok else None
        merge_error: str | None = None
        if stage == "lora" and res.ok and res.output_weight_path:
            try:
                base_pth = stage_from_weight_for_trainer(
                    from_weight,
                    self.repo_root,
                    self.workspace_dir,
                    self.profile,
                    from_weight_path=from_weight_path,
                    model_config=eff_arch,
                )
                if base_pth is None:
                    raise FileNotFoundError(f"Base weight {from_weight!r} not found for LoRA merge")
                cfg_eff = eff_arch or {}
                hs = int(cfg_eff.get("hidden_size", self.profile.hidden_size))
                nl = int(cfg_eff.get("num_hidden_layers", self.profile.num_hidden_layers))
                um = bool(cfg_eff.get("use_moe", self.profile.use_moe))
                moe_suffix = "_moe" if um else ""
                merged_path = (
                    self.workspace_dir
                    / "models"
                    / "candidates"
                    / f"cand_c{ctx.cycle_index:04d}_merged_{hs}{moe_suffix}.pth"
                )
                lora_rank = int(train_cfg.get("lora_rank", 16))
                promote_lora_adapter(
                    base_weight=base_pth,
                    lora_weight=Path(res.output_weight_path),
                    merged_out=merged_path,
                    hidden_size=hs,
                    num_hidden_layers=nl,
                    use_moe=um,
                    lora_rank=lora_rank,
                    target_all_linear=True,
                    model_config=eff_arch,
                )
                final_weight_path = str(merged_path)
            except Exception as exc:
                merge_error = repr(exc)
                final_weight_path = None

        accepted = False
        after_sc: dict | None = None
        if final_weight_path and Path(final_weight_path).is_file():
            after_sc = self.session.eval_strict_scorecard(Path(final_weight_path))
            score_up = float(after_sc["overall_score"]) > float(before_sc["overall_score"]) + 0.005
            pass_up = (
                int(after_sc["passed_count"]) > int(before_sc["passed_count"])
                and float(after_sc["eos_rate"]) >= float(before_sc["eos_rate"]) - 0.05
            )
            accepted = bool(score_up or pass_up)
            if accepted:
                self._ep_counter += 1
                self.curated_episodes_dir.mkdir(parents=True, exist_ok=True)
                record_verified_selfchange_episode(
                    before_eval=before_sc,
                    after_eval=after_sc,
                    fix_summary={"steps_taken": ["live_strict_eval", "multi_rollout_explore", stage], "learning_rate": train_cfg.get("learning_rate", 3e-5)},
                    output_sft_jsonl=cur_ep_sft,
                    output_dpo_jsonl=cur_ep_dpo,
                    episode_tag=f"live_c{ctx.cycle_index}_{self._ep_counter}",
                )
                atomic_write_text(self.workspace_dir / "live_scorecard_end.json", json.dumps(after_sc, indent=2, ensure_ascii=False) + "\n")
                self.session._ensure_loaded(Path(final_weight_path))
                self.session.last_scorecard = after_sc
            else:
                # Reload previous baseline weight into session if rejected
                if from_weight_path:
                    self.session._ensure_loaded(from_weight_path)
                    self.session.last_scorecard = before_sc
                final_weight_path = None

        self._append_train_history(
            {
                "timestamp": _utc_now(),
                "cycle_index": ctx.cycle_index,
                "stage": stage,
                "reason": recipe.get("reason"),
                "live_metrics": live_metrics,
                "on_policy_rft": len(safe_rft),
                "on_policy_dpo": len(safe_on_dpo),
                "total_batch_rows": len(merged_dpo) if stage == "dpo" else len(merged_sft),
                "ok": res.ok,
                "accepted": accepted,
                "before_score": before_sc.get("overall_score"),
                "after_score": after_sc.get("overall_score") if after_sc else None,
                "before_passed": f"{before_sc.get('passed_count')}/{before_sc.get('total_count')}",
                "after_passed": f"{after_sc.get('passed_count')}/{after_sc.get('total_count')}" if after_sc else None,
                "exit_code": res.exit_code,
                "final_loss": res.final_loss,
                "duration_sec": res.duration_sec,
                "output_weight_path": final_weight_path,
                "merge_error": merge_error,
                "log_tail": res.log_tail,
            }
        )
        return final_weight_path
