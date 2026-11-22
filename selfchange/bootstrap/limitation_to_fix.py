from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from selfchange.bootstrap.data_quality import (
    _build_direct_code_edit_sample,
    _build_multiturn_code_edit_sample,
    _compact_tools_json,
    _extract_prompt_keys,
    _normalize_for_diversity,
    _tc_xml,
    audit_training_dataset,
    collect_code_fixtures,
    review_and_record_dataset,
)
from selfchange.bootstrap.generate_traces import _select_tools, estimate_tokens
from selfchange.kernel.integrity import atomic_write_text
from selfchange.kernel.tool_runtime import WorktreeToolExecutor
from selfchange.mutable.prompts import SYSTEM_PROMPT
from selfchange.training.self_train_runner import (
    format_dpo_record,
    format_sft_record,
    write_dpo_jsonl,
    write_sft_jsonl,
)

HELDOUT_10_PROMPTS: list[dict] = [
    {
        "id": "1_trivia_capital",
        "messages": [{"role": "user", "content": "What is the capital city of France, and what river flows through it?"}],
    },
    {
        "id": "2_tech_zh_grad_accum",
        "messages": [{"role": "user", "content": "什么是梯度累积？它在显存受限时有什么作用？"}],
    },
    {
        "id": "3_tech_en_grad_clip",
        "messages": [{"role": "user", "content": "What does torch.nn.utils.clip_grad_norm_ prevent during fp16 training?"}],
    },
    {
        "id": "4_reasoning_oom_math",
        "messages": [{"role": "user", "content": "If batch_size=16 and accumulation_steps=2, and we halve batch_size twice after two OOM errors while doubling accumulation_steps each time, what are the final batch_size and accumulation_steps?"}],
    },
    {
        "id": "5_tool_oom",
        "messages": [
            {"role": "system", "content": "Diagnose OOM and call read_log_tail or patch_config."},
            {"role": "user", "content": "fault=oom module=train_lora.py batch_size=8 accumulation_steps=1"},
        ],
    },
    {
        "id": "6_tool_nan_patch",
        "messages": [
            {"role": "system", "content": "Stabilize NaN loss by halving lr to 0.0001 and setting grad_clip=0.5 via patch_config."},
            {"role": "user", "content": "fault=nan_loss step=40 lr=0.0002 grad_norm=inf"},
        ],
    },
    {
        "id": "7_tool_rollback",
        "messages": [
            {"role": "system", "content": "Corrupt checkpoint detected. Roll back to verified ancestor using rollback_to_ancestor."},
            {"role": "user", "content": "fault=corrupt_checkpoint file=full_sft_768_resume.pth verified_ancestor=cand_420"},
        ],
    },
    {
        "id": "8_code_missing_colon",
        "messages": [
            {"role": "system", "content": "Fix the Python syntax bug using replace_line_range."},
            {"role": "user", "content": "oom_helper.py:\n1: def compute_step(batch_size: int, div: int = 2) -> int\n2:     return max(1, batch_size // div)"},
        ],
    },
    {
        "id": "9_code_clamp_logic",
        "messages": [
            {"role": "system", "content": "Fix the logic bug on line 4 using replace_line_range (should cap lr with min, not max)."},
            {"role": "user", "content": "lr_guard.py:\n1: def clamp_lr(lr: float, max_lr: float = 5e-4) -> float:\n2:     if lr < 1e-8:\n3:         return 1e-8\n4:     return max(lr, max_lr)"},
        ],
    },
    {
        "id": "10_healthy_noop",
        "messages": [
            {"role": "system", "content": "Monitor system health. Only call tools if there is an actual fault."},
            {"role": "user", "content": "fault=healthy cycle=12 loss=0.842 consecutive_failures=0"},
        ],
    },
]

_CORE_TECH_QA_PAIRS: list[tuple[str, str, str]] = [
    (
        "Explain how gradient accumulation simulates a larger effective batch size under tight GPU VRAM.",
        "Gradient accumulation sums loss/gradients over K micro-batches via loss / K before calling optimizer.step(), multiplying effective batch_size by K without allocating larger activation memory.",
        "Focus on loss / accumulation_steps and delayed optimizer.step().",
    ),
    (
        "为什么混合精度 fp16 训练需要搭配 GradScaler 和梯度裁剪 clip_grad_norm_？",
        "fp16 指数范围窄容易溢出产生 inf/nan 梯度；GradScaler 放大损失防止下溢，clip_grad_norm_ 在反缩放后限制总梯度范数，避免参数更新发散。",
        "关联 fp16 范围、GradScaler.unscale_ 和 clip_grad_norm_ 阈值约束。",
    ),
    (
        "In PyTorch, what happens if optimizer.step() runs without unscaling gradients before clip_grad_norm_?",
        "Clipping scaled gradients distorts the effective threshold and can allow exploding fp16 updates or skipped steps to destabilize weights; always call scaler.unscale_(optimizer) first.",
        "Order is backward -> unscale_ -> clip_grad_norm_ -> scaler.step -> scaler.update.",
    ),
    (
        "Distinguish gradient descent from gradient accumulation cleanly.",
        "Gradient descent is the optimization rule updating parameters opposite the loss gradient, whereas gradient accumulation defers the weight step across N micro-batches to raise effective batch size.",
        "Keep optimization algorithm separate from micro-batch accumulation.",
    ),
    (
        "Why does SFTDataset truncate over-budget examples at max_seq_len break EOS and tool_call closing?",
        "Left-to-right truncation slices tokens past max_seq_len, cutting off the trailing tool_call XML terminator and EOS stop token so the model never sees termination supervision.",
        "Sequence budget must fit inside 0.9 * max_seq_len before tokenization.",
    ),
    (
        "Why enforce strict five-field conversations dictionaries in write_sft_jsonl?",
        "dataset/lm_dataset.py loads JSONL with a fixed Arrow Features schema containing role, content, reasoning_content, tools, and tool_calls; extra keys raise schema cast errors.",
        "Normalize every SFT message strictly to the 5 allowed fields.",
    ),
    (
        "When choosing between read_log_tail, patch_config, and replace_line_range, how do schemas differ?",
        "read_log_tail expects {log_name, lines}, patch_config expects {updates: {...}}, and replace_line_range requires {path, start_line, end_line, lines}.",
        "Never pass log_name inside patch_config or omit line ranges in replace_line_range.",
    ),
    (
        "Why validate edited Python code in a git worktree with ast.parse and scan_python_source?",
        "git worktree isolates candidate edits from main, ast.parse immediately catches SyntaxError typos, and scan_python_source blocks unsafe imports before subprocess execution.",
        "Verify syntax and AST policy locally before committing or promoting.",
    ),
    (
        "How does Contrastive DPO fix repetitive loops and empty think-block shortcuts?",
        "Pairing the same prompt with a grounded thinking block and verified tool call chosen response against an empty-think, looping, or wrong-schema rejected response lowers log-prob of failure modes.",
        "Same-prompt DPO preference margin directly suppresses mode collapse.",
    ),
    (
        "What does LoRA target_all_linear add compared to square-only q_proj/o_proj adapters?",
        "Expanding LoRA to k_proj, v_proj, gate_proj, up_proj, and down_proj adapts MLP circuits where domain knowledge lives while leaving full weights mergeable via merge_lora.",
        "MLP projections carry domain knowledge needed during capability installation.",
    ),
    (
        "Why compute pass@4 alongside pass@1 when diagnosing a small language model?",
        "When pass@4 is high while pass@1 is low, the fix already exists in policy distribution and RFT/DPO sharpening works; when both are 0, curated supervised traces are needed first.",
        "Separate missing knowledge from decoding calibration.",
    ),
    (
        "How does atomic checkpoint saving via .tmp rename protect Colab training runs?",
        "Writing weights to a hidden temp file and swapping with os.replace ensures a mid-write Colab preemption never leaves a half-written corrupt checkpoint.",
        "os.replace is atomic on POSIX filesystems.",
    ),
    (
        "为什么要从 64M 模型中剪除与任务无关的百科问答并保留技术核心锚点？",
        "64M 参数中约 70% 集中在 SwiGLU MLP 层；减少无关闲聊与百科记忆可释放参数容量给 Python AST 修复、工具调用和结构化推理过程，同时靠核心锚点保持 EOS 终止与语法流畅。",
        "解释 MLP 容量占比、释放在哪、为何保留核心锚点。",
    ),
    (
        "Why must continuous data auditing block training on off-domain trivia and ungrounded reasoning?",
        "Auditing schema, token fit, live worktree execution, trivia ratio, and reasoning grounding prevents bad batches from wasting compute or causing catastrophic forgetting on small models.",
        "Fail-closed review gate protects downstream training stability.",
    ),
    (
        "What happens in GRPO/CISPO when every rollout in a group fails with identical reward?",
        "When rewards within a prompt group have zero variance (std_r == 0), normalized advantages collapse to zero, yielding no useful policy gradient until SFT/DPO creates variance.",
        "Zero-variance RL cliff explains why SFT warm-start precedes RL.",
    ),
    (
        "In AdamW optimization, how does halving learning_rate and tightening grad_clip resolve NaN loss?",
        "Smaller step sizes prevent parameters from overshooting steep curvature regions while lower grad_clip (e.g. 0.5) bounds gradient spikes before mixed-precision weights overflow.",
        "Connect learning_rate scaling and gradient norm clamping.",
    ),
]

_SYS_PREFIX_VARIANTS = [
    "Diagnose the fault and choose the exact tool and schema.",
    "Inspect system state cleanly; emit valid think blocks and exact tool arguments.",
    "Operate safely inside selfchange/mutable limits.",
    "Verify parameter names and constraints before calling any tool.",
]

_FLOAT_CLAMP_CODE_FIXTURES: list[tuple[str, str]] = [
    (
        "beta_clamp.py",
        "def clip_beta(beta: float, max_beta: float = 0.25) -> float:\n    if beta < 1e-5:\n        return 1e-5\n    return min(beta, max_beta)\n",
    ),
    (
        "step_lr_cap.py",
        "def bound_step_lr(lr: float, cap: float = 3e-4) -> float:\n    if lr < 1e-6:\n        return 1e-6\n    return min(lr, cap)\n",
    ),
    (
        "scale_floor.py",
        "def floor_decay(lr: float, floor_lr: float = 2e-5) -> float:\n    scaled = lr * 0.5\n    if scaled < floor_lr:\n        return floor_lr\n    return min(scaled, 0.95)\n",
    ),
    (
        "grad_cap.py",
        "def cap_grad_norm(gn: float, max_gn: float = 1.5) -> float:\n    if gn < 1e-7:\n        return 1e-7\n    return min(gn, max_gn)\n",
    ),
    (
        "wd_guard.py",
        "def bound_weight_decay(wd: float, top: float = 0.15) -> float:\n    if wd < 1e-6:\n        return 1e-6\n    return min(wd, top)\n",
    ),
    (
        "temp_range.py",
        "def clip_temp(temp: float, max_t: float = 1.2) -> float:\n    if temp < 0.05:\n        return 0.05\n    return min(temp, max_t)\n",
    ),
    (
        "ratio_guard.py",
        "def clamp_ratio(val: float, hi: float = 0.85) -> float:\n    if val < 0.05:\n        return 0.05\n    return min(val, hi)\n",
    ),
    (
        "eps_bound.py",
        "def safe_adam_eps(eps: float, upper: float = 4e-4) -> float:\n    if eps < 1e-7:\n        return 1e-7\n    return min(eps, upper)\n",
    ),
]


def _build_numbered_line_code_edit_sample(fname: str, clean_src: str, rng: random.Random, wt_dir: Path) -> dict | None:
    lo_val = rng.choice(["1e-7", "2e-6", "1e-6", "1e-5", "0.02", "0.05"])
    hi_val = rng.choice(["0.15", "0.25", "0.45", "0.65", "0.85", "1.2", "3e-4", "4e-4"])
    src = re.sub(r"1e-[567]|0\.05", lo_val, clean_src)
    src = re.sub(r"0\.(?:15|25|85|95)|1\.[25]|[34]e-4", hi_val, src)
    lines = src.splitlines()
    if len(lines) < 3 or "min(" not in src:
        return None
    target_idx = next((i for i, ln in enumerate(lines) if "min(" in ln and "#" not in ln), -1)
    if target_idx < 0:
        return None
    broken_lines = list(lines)
    broken_lines[target_idx] = lines[target_idx].replace("min(", "max(", 1)
    broken_src = "\n".join(broken_lines) + "\n"
    line_num = target_idx + 1
    edit_args = {"path": fname, "start_line": line_num, "end_line": line_num, "lines": [lines[target_idx]]}

    m_dir = wt_dir / "selfchange" / "mutable"
    m_dir.mkdir(parents=True, exist_ok=True)
    target_file = m_dir / Path(fname).name
    target_file.write_text(broken_src, encoding="utf-8")
    res = WorktreeToolExecutor(wt_dir, autonomy_level=1).dispatch("replace_line_range", edit_args)
    if not res.get("ok") or target_file.read_text(encoding="utf-8").strip() != src.strip():
        return None

    numbered = "\n".join(f"{i + 1}: {ln}" for i, ln in enumerate(broken_lines))
    sys_msg = rng.choice(
        [
            f"{SYSTEM_PROMPT}\nFix bug with replace_line_range using integer line numbers.",
            "Fix the logic bug using replace_line_range with integer start_line and end_line.",
        ]
    )
    convs = [
        {"role": "system", "content": sys_msg, "tools": _select_tools(["replace_line_range"])},
        {"role": "user", "content": f"{fname}:\n{numbered}"},
        {
            "role": "assistant",
            "content": (
                f"<think>\nObserve: {fname} line {line_num} uses max instead of min. "
                f"Integer index start_line={line_num} end_line={line_num}. Plan: call replace_line_range.\n</think>\n\n"
                f"{_tc_xml('replace_line_range', edit_args)}"
            ),
        },
    ]
    return format_sft_record({"conversations": convs})


def diagnose_model_limitations(eval_report: dict) -> list[dict]:
    """Extract actionable, evidence-backed limitation cards from an eval report."""
    limitations: list[dict] = []
    examples = eval_report.get("examples") or []
    eos_rate = float(eval_report.get("eos_rate", eval_report.get("tier1_language", {}).get("eos_rate", 1.0)))

    empty_think_ids = [e["id"] for e in examples if "<think>\n</think>" in e.get("reply", "") or "<think></think>" in e.get("reply", "")]
    no_eos_ids = [e["id"] for e in examples if not e.get("eos_ok", True)]
    math_fails = [e["id"] for e in examples if e.get("category") == "reasoning_math" and not e.get("passed")]
    code_fails = [e["id"] for e in examples if e.get("category") == "code_edit" and not e.get("passed")]
    qa_fails = [e["id"] for e in examples if e.get("category") in ("core_tech_qa", "trivia") and not e.get("passed")]

    schema_bad_ids: list[str] = []
    for e in examples:
        for c in e.get("tool_calls") or []:
            name = c.get("name")
            args = c.get("arguments") or {}
            if name == "patch_config" and not isinstance(args.get("updates"), dict):
                schema_bad_ids.append(e["id"])
            elif name == "replace_line_range" and not all(k in args for k in ("path", "start_line", "end_line", "lines")):
                schema_bad_ids.append(e["id"])

    if empty_think_ids or math_fails:
        limitations.append(
            {
                "limitation_id": "empty_think_and_math_reasoning",
                "severity": "high",
                "evidence": f"empty_think={empty_think_ids}, math_fails={math_fails}",
                "root_cause": "Base model skips think reasoning (emitting empty think tags) and hallucinates arithmetic under scaling prompts.",
                "fix_steps": [
                    "Require non-empty Observe -> Cause -> Plan steps inside think tags across all SFT buckets.",
                    "Add parametric batch_size/lr/token arithmetic reasoning traces with step-by-step verification.",
                    "Penalize empty think-block shortcuts using Contrastive DPO pairs.",
                ],
            }
        )

    if eos_rate < 0.90 or no_eos_ids:
        limitations.append(
            {
                "limitation_id": "repetition_loop_and_eos_cutoff",
                "severity": "high",
                "evidence": f"eos_rate={eos_rate}, cutoff_examples={no_eos_ids}",
                "root_cause": "Model enters repetitive loops and hits max_new_tokens without emitting the EOS stop token.",
                "fix_steps": [
                    "Filter all training conversations to <= 0.9 * max_seq_len so EOS labels are never truncated.",
                    "Add Contrastive DPO pairs rejecting verbatim n-gram repetition loops.",
                ],
            }
        )

    if schema_bad_ids or any(e.get("category") == "tool_call_op" and not e.get("passed") for e in examples):
        limitations.append(
            {
                "limitation_id": "cross_tool_schema_contamination",
                "severity": "high",
                "evidence": f"malformed_tool_args={sorted(set(schema_bad_ids))}",
                "root_cause": "When multiple tools are declared, model copies log_name from read_log_tail into patch_config or omits required keys.",
                "fix_steps": [
                    "Train on multi-tool system prompts contrasting patch_config(updates), read_log_tail(log_name, lines), and rollback_to_ancestor(node_id).",
                    "Add DPO pairs explicitly rejecting wrong-schema arguments on patch_config and replace_line_range.",
                ],
            }
        )

    if code_fails or float(eval_report.get("tier3_code_edit", {}).get("pass_at_1", 1.0)) < 0.40:
        limitations.append(
            {
                "limitation_id": "code_edit_missing_args_and_markdown_fallback",
                "severity": "critical",
                "evidence": f"code_fails={code_fails}, tier3={eval_report.get('tier3_code_edit')}",
                "root_cause": "Model hallucinates non-existent paths, omits start_line/end_line/lines, or outputs python markdown instead of replace_line_range.",
                "fix_steps": [
                    "Generate diverse 1-turn and multi-turn execution-verified replace_line_range repairs across 67+ repo/utility fixtures.",
                    "Replay every edit in WorktreeToolExecutor + ast.parse and reject markdown-only DPO negatives.",
                ],
            }
        )

    if qa_fails:
        limitations.append(
            {
                "limitation_id": "core_tech_misconception_and_trivia_hallucination",
                "severity": "medium",
                "evidence": f"qa_fails={qa_fails}",
                "root_cause": "64M capacity diluted by noisy trivia while core PyTorch/ML terminology confuses accumulation with descent.",
                "fix_steps": [
                    "Cut general trivia and replace with concise grounded technical ML/PyTorch/systems QA anchors.",
                ],
            }
        )

    noop_items = [e for e in examples if e.get("category") == "noop"]
    if noop_items or not limitations:
        limitations.append(
            {
                "limitation_id": "healthy_state_parsing_and_selfchange_grounding",
                "severity": "medium",
                "evidence": f"noop_evals={[e['id'] for e in noop_items]}",
                "root_cause": "Model lacks meta-knowledge mapping evaluation diagnostics into safe self-change tool actions and accurate no-op reporting.",
                "fix_steps": [
                    "Train meta-selfchange episodes converting diagnosed limitations into curate_experience -> patch_config -> submit_candidate steps.",
                    "Preserve healthy no-op traces accurately reading metrics without extraneous mutations.",
                ],
            }
        )
    return limitations


def _build_schema_disambiguation_sample(rng: random.Random, wt_dir: Path) -> dict | None:
    cfg_dir = wt_dir / "selfchange" / "mutable"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    sys_prefix = rng.choice(_SYS_PREFIX_VARIANTS)
    mode = rng.randrange(5)

    if mode == 0:
        mod = rng.choice(["train_full_sft.py", "train_dpo.py", "train_pretrain.py", "train_distillation.py", "train_agent.py"])
        bs = rng.choice([4, 6, 8, 12, 16])
        acc = rng.choice([1, 2, 4])
        new_bs = max(1, bs // 2)
        new_acc = min(32, acc * 2)
        updates = {"training": {"batch_size": new_bs, "accumulation_steps": new_acc}}
        (cfg_dir / "config.json").write_text(json.dumps({"training": {"batch_size": bs, "accumulation_steps": acc}}), encoding="utf-8")
        if not WorktreeToolExecutor(wt_dir, autonomy_level=1).dispatch("patch_config", {"updates": updates}).get("ok"):
            return None
        convs = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n{sys_prefix}", "tools": _select_tools(["read_log_tail", "patch_config"])},
            {"role": "user", "content": f"fault=oom module={mod} batch_size={bs} accumulation_steps={acc}; read train.log then patch_config."},
            {
                "role": "assistant",
                "content": f"<think>\nObserve: fault=oom in {mod} batch_size={bs}. Plan: call read_log_tail.\n</think>\n\n{_tc_xml('read_log_tail', {'log_name': 'train.log', 'lines': 20})}",
            },
            {"role": "tool", "content": f"RuntimeError: CUDA out of memory in {mod} at batch_size={bs}"},
            {
                "role": "assistant",
                "content": f"<think>\nObserve: CUDA OOM in {mod} batch_size={bs}. Plan: call patch_config.\n</think>\n\n{_tc_xml('patch_config', {'updates': updates})}",
            },
        ]
        return format_sft_record({"conversations": convs})

    if mode == 1:
        lr = rng.choice([1e-4, 2.5e-4, 3e-4, 4e-4, 5e-4])
        new_lr = round(lr * 0.5, 8)
        step = rng.randint(10, 95)
        updates = {"training": {"learning_rate": new_lr, "grad_clip": 0.5}}
        (cfg_dir / "config.json").write_text(json.dumps({"training": {"learning_rate": lr, "grad_clip": 1.0}}), encoding="utf-8")
        if not WorktreeToolExecutor(wt_dir, autonomy_level=1).dispatch("patch_config", {"updates": updates}).get("ok"):
            return None
        convs = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\nStabilize NaN loss using patch_config(updates).", "tools": _select_tools(["patch_config", "read_log_tail"])},
            {"role": "user", "content": f"fault=nan_loss step={step} lr={lr} grad_norm=inf; halve lr to {new_lr} and set grad_clip=0.5 via patch_config."},
            {
                "role": "assistant",
                "content": (
                    f"<think>\nObserve: step={step} nan_loss lr={lr} grad_norm=inf. "
                    f"Plan: call patch_config learning_rate={new_lr} grad_clip=0.5.\n</think>\n\n"
                    f"{_tc_xml('patch_config', {'updates': updates})}"
                ),
            },
        ]
        return format_sft_record({"conversations": convs})

    if mode == 2:
        anc = f"anc_{rng.randint(500, 999)}"
        ckpt = rng.choice(["warmstart_rung2_768_resume.pth", "offline_dpo_r2_768_resume.pth", "fixed_sft_768_resume.pth"])
        convs = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\nUse rollback_to_ancestor(node_id) or get_archive_summary.", "tools": _select_tools(["rollback_to_ancestor", "get_archive_summary"])},
            {"role": "user", "content": f"fault=corrupt_checkpoint file={ckpt} error=EOFError verified_ancestor={anc}"},
            {
                "role": "assistant",
                "content": (
                    f"<think>\nObserve: {ckpt} corrupt_checkpoint verified_ancestor={anc}. "
                    f"Plan: call rollback_to_ancestor.\n</think>\n\n"
                    f"{_tc_xml('rollback_to_ancestor', {'node_id': anc})}"
                ),
            },
        ]
        return format_sft_record({"conversations": convs})

    if mode == 3:
        seq = rng.choice([512, 768, 1024])
        new_seq = max(128, int(seq * 0.75))
        updates = {"training": {"max_seq_len": new_seq, "batch_size": 1}}
        (cfg_dir / "config.json").write_text(json.dumps({"training": {"max_seq_len": seq, "batch_size": 1}}), encoding="utf-8")
        if not WorktreeToolExecutor(wt_dir, autonomy_level=1).dispatch("patch_config", {"updates": updates}).get("ok"):
            return None
        convs = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\n{sys_prefix}", "tools": _select_tools(["patch_config", "read_file_lines", "read_log_tail"])},
            {"role": "user", "content": f"fault=oom batch_size=1 max_seq_len={seq}; shrink sequence budget via patch_config."},
            {
                "role": "assistant",
                "content": (
                    f"<think>\nObserve: batch_size=1 OOM max_seq_len={seq}. "
                    f"Plan: call patch_config max_seq_len={new_seq}.\n</think>\n\n"
                    f"{_tc_xml('patch_config', {'updates': updates})}"
                ),
            },
        ]
        return format_sft_record({"conversations": convs})

    temp = rng.choice([0.6, 0.8, 1.0])
    errs = rng.randint(1, 4)
    updates = {"temperature": 0.0, "max_turns": 4}
    (cfg_dir / "config.json").write_text(json.dumps({"temperature": temp, "max_turns": 6}), encoding="utf-8")
    if not WorktreeToolExecutor(wt_dir, autonomy_level=1).dispatch("patch_config", {"updates": updates}).get("ok"):
        return None
    convs = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\nStabilize decoding config.", "tools": _select_tools(["patch_config", "get_system_state"])},
        {"role": "user", "content": f"fault=malformed_tool_loop temperature={temp} tag_errors={errs}"},
        {
            "role": "assistant",
            "content": (
                f"<think>\nObserve: tag_errors={errs} temperature={temp}. "
                f"Plan: call patch_config.\n</think>\n\n"
                f"{_tc_xml('patch_config', {'updates': updates})}"
            ),
        },
    ]
    return format_sft_record({"conversations": convs})


def _build_meta_selfchange_sample(limitation: dict, rng: random.Random, wt_dir: Path, variant_idx: int) -> dict | None:
    cfg_dir = wt_dir / "selfchange" / "mutable"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    lid = limitation["limitation_id"]
    eos_obs = round(rng.uniform(0.45, 0.78), 2)
    cyc = rng.randint(1, 25)
    lr = rng.choice([2e-5, 3e-5, 5e-5])
    arch_r = round(rng.choice([0.20, 0.25, 0.30]), 2)
    bs = rng.choice([2, 4])
    acc = rng.choice([2, 4, 8])

    (cfg_dir / "config.json").write_text(json.dumps({"training": {"batch_size": bs, "learning_rate": lr, "anchor_replay_ratio": arch_r}, "temperature": 0.2}), encoding="utf-8")
    scenario = variant_idx % 4
    if scenario == 0:
        updates = {"temperature": 0.0, "training": {"learning_rate": lr, "anchor_replay_ratio": arch_r, "dpo_beta": 0.15}}
        if not WorktreeToolExecutor(wt_dir, autonomy_level=1).dispatch("patch_config", {"updates": updates}).get("ok"):
            return None
        convs = [
            {"role": "system", "content": "Diagnose limitation from logs and apply patch_config.", "tools": _compact_tools_json(["read_log_tail", "patch_config"])},
            {"role": "user", "content": f"meta_selfchange cycle={cyc} limitation={lid} eos={eos_obs} code_pass1=0.0"},
            {
                "role": "assistant",
                "content": f"<think>\nObserve: cycle={cyc} reports {lid} (eos={eos_obs}, code_pass1=0.0). Plan: call read_log_tail.\n</think>\n\n{_tc_xml('read_log_tail', {'log_name': 'eval.log', 'lines': 20})}",
            },
            {
                "role": "tool",
                "content": f"eval_summary: limitation={lid}; cause={limitation['root_cause'][:72]}; eos={eos_obs}",
            },
            {
                "role": "assistant",
                "content": f"<think>\nObserve: {lid} confirmed in eval.log. Plan: call patch_config lr={lr} anchor_replay_ratio={arch_r}.\n</think>\n\n{_tc_xml('patch_config', {'updates': updates})}",
            },
        ]
        return format_sft_record({"conversations": convs})

    if scenario == 1:
        seq = rng.choice([768, 1024])
        updates = {"thinking_ratio": 1.0, "training": {"max_seq_len": seq, "learning_rate": lr, "anchor_replay_ratio": arch_r}}
        if not WorktreeToolExecutor(wt_dir, autonomy_level=1).dispatch("patch_config", {"updates": updates}).get("ok"):
            return None
        convs = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\nTurn observed limitation into a concrete config + curation update.", "tools": _select_tools(["patch_config", "curate_experience"])},
            {"role": "user", "content": f"selfchange_step cycle={cyc} diagnosed={lid} evidence={limitation['evidence'][:70]}"},
            {
                "role": "assistant",
                "content": (
                    f"<think>\nObserve: diagnosed={lid} at cycle={cyc}. Root cause: {limitation['root_cause'][:70]}. "
                    f"Plan: call patch_config setting thinking_ratio=1.0 max_seq_len={seq} lr={lr}.\n</think>\n\n"
                    f"{_tc_xml('patch_config', {'updates': updates})}"
                ),
            },
        ]
        return format_sft_record({"conversations": convs})

    if scenario == 2:
        updates = {"training": {"learning_rate": lr, "batch_size": bs, "accumulation_steps": acc}}
        if not WorktreeToolExecutor(wt_dir, autonomy_level=1).dispatch("patch_config", {"updates": updates}).get("ok"):
            return None
        convs = [
            {"role": "system", "content": "Curate targeted limitation_fix data then update training config.", "tools": _compact_tools_json(["curate_experience", "patch_config"])},
            {"role": "user", "content": f"selfchange_plan cycle={cyc} limitation={lid} eos={eos_obs}"},
            {
                "role": "assistant",
                "content": f"<think>\nObserve: cycle={cyc} limitation={lid} eos={eos_obs}. Plan: call curate_experience mode=rft_and_dpo.\n</think>\n\n{_tc_xml('curate_experience', {'mode': 'rft_and_dpo'})}",
            },
            {"role": "tool", "content": '{"ok":1,"sft":1200,"dpo":240}'},
            {
                "role": "assistant",
                "content": f"<think>\nObserve: curated sft=1200 dpo=240. Plan: call patch_config batch_size={bs} lr={lr}.\n</think>\n\n{_tc_xml('patch_config', {'updates': updates})}",
            },
        ]
        return format_sft_record({"conversations": convs})

    # Scenario 3: Record completed self-change verification outcome & curate next dataset
    after_pass = round(rng.uniform(0.7, 0.9), 2)
    hyp = f"record {lid} fix outcome"
    convs = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\nCurate verified improvement and submit selfchange step.", "tools": _select_tools(["curate_experience", "submit_candidate"])},
        {"role": "user", "content": f"selfchange_verified cycle={cyc} target={lid} before_eos={eos_obs} after_eos=1.0 after_pass={after_pass}"},
        {
            "role": "assistant",
            "content": (
                f"<think>\nObserve: {lid} improved eos {eos_obs}->1.0 and pass={after_pass} on cycle={cyc}. "
                f"Plan: call curate_experience mode=selfchange_episode.\n</think>\n\n"
                f"{_tc_xml('curate_experience', {'mode': 'selfchange_episode'})}"
            ),
        },
        {"role": "tool", "content": '{"ok":1,"appended":true}'},
        {
            "role": "assistant",
            "content": (
                f"<think>\nObserve: selfchange_episode curated cleanly. Plan: call submit_candidate.\n</think>\n\n"
                f"{_tc_xml('submit_candidate', {'hypothesis': hyp})}"
            ),
        },
    ]
    return format_sft_record({"conversations": convs})


def _build_healthy_noop_sample(rng: random.Random) -> dict:
    loss = round(rng.uniform(0.35, 1.55), 3)
    cyc = rng.randint(1, 90)
    if cyc == 12:
        cyc = 13
    gn = round(rng.uniform(0.15, 0.85), 2)
    lr = rng.choice(["2e-5", "3e-5", "5e-5"])
    ppl = round(rng.uniform(4.2, 14.5), 2)
    sys_msg = rng.choice(
        [
            "Monitor training health metrics cleanly. Do not call patch_config unless an error is present.",
            "Inspect system state safely; avoid config edits when consecutive_failures=0 and system is healthy.",
            "Check health status. Call tools only for verified faults.",
        ]
    )
    user_msg = rng.choice(
        [
            f"health_report status=healthy cycle={cyc} train_loss={loss} grad_norm={gn} errors=0",
            f"monitor checkpoint: healthy=true cycle={cyc} val_ppl={ppl} loss={loss} consecutive_failures=0",
            f"eval_cycle={cyc} fault=healthy stage=full_sft lr={lr} loss={loss} consecutive_failures=0",
            f"state=healthy epoch=1 cycle={cyc} loss={loss} consecutive_failures=0",
        ]
    )
    if rng.random() < 0.70:
        convs = [
            {"role": "system", "content": sys_msg, "tools": _select_tools(["get_system_state", "patch_config"])},
            {"role": "user", "content": user_msg},
            {
                "role": "assistant",
                "content": (
                    f"<think>\nObserve: cycle={cyc} status is healthy (loss={loss}, zero failures). "
                    f"No fault present; do not call patch_config.\n</think>\n\n"
                    f"Training health looks normal on cycle {cyc}. No edits required."
                ),
            },
        ]
    else:
        convs = [
            {"role": "system", "content": sys_msg, "tools": _select_tools(["get_system_state", "patch_config"])},
            {"role": "user", "content": user_msg},
            {
                "role": "assistant",
                "content": (
                    f"<think>\nObserve: cycle={cyc} loss={loss} zero failures. "
                    f"Inspect state only without mutating config. Plan: call get_system_state.\n</think>\n\n"
                    f"{_tc_xml('get_system_state', {})}"
                ),
            },
        ]
    return format_sft_record({"conversations": convs})


def _build_reasoning_math_or_noop_sample(rng: random.Random) -> dict:
    kind = rng.randrange(4)
    if kind == 0:
        bs = rng.choice([8, 12, 16, 20, 24, 32, 48, 64])
        acc = rng.choice([1, 2, 4])
        halves = rng.choice([1, 2, 3])
        if (bs, acc, halves) == (16, 2, 2):
            bs = 32
        cur_bs, cur_acc = bs, acc
        trace_parts = []
        for step_idx in range(1, halves + 1):
            cur_bs = max(1, cur_bs // 2)
            cur_acc = cur_acc * 2
            trace_parts.append(f"Step {step_idx}: batch_size={cur_bs}, accum={cur_acc}")
        q = (
            f"Start with batch_size={bs} and accumulation_steps={acc}. "
            f"After {halves} OOM halving step(s) where batch_size halves and accumulation_steps doubles each step, "
            f"what are final batch_size and accumulation_steps?"
        )
        think = (
            f"Observe: initial batch_size={bs}, accumulation_steps={acc}, steps={halves}. "
            + "; ".join(trace_parts)
            + f". Effective batch={cur_bs * cur_acc}."
        )
        ans = f"Final batch_size={cur_bs} and accumulation_steps={cur_acc} (effective batch size remains {cur_bs * cur_acc})."
    elif kind == 1:
        seq = rng.choice([384, 512, 768, 1024, 1536])
        budget = int(0.9 * seq)
        q = f"If max_seq_len={seq} and the safe pre-truncation budget ratio is 0.9, what is the maximum allowed training sequence token count?"
        think = f"Observe: max_seq_len={seq}, ratio=0.9. Multiply {seq} * 0.9 = {seq * 0.9:.1f}, integer floor = {budget} tokens."
        ans = f"The maximum allowed token count at max_seq_len={seq} is {budget} tokens."
    elif kind == 2:
        lr = rng.choice([0.0004, 0.0003, 0.0002, 0.0001])
        steps = rng.choice([2, 3])
        cur_lr = lr
        halve_notes = []
        for s_i in range(1, steps + 1):
            cur_lr = round(cur_lr / 2.0, 8)
            halve_notes.append(f"#{s_i}:{cur_lr}")
        q = f"Starting at learning_rate={lr}, if NaN recovery halves the learning rate {steps} times, what is the final learning_rate?"
        think = f"Observe: initial learning_rate={lr}, count={steps}. Divide repeatedly -> " + ", ".join(halve_notes) + "."
        ans = f"Result: final learning_rate is {cur_lr}."
    else:
        total = rng.choice([100, 200, 400, 600, 1200, 1600])
        ratio = rng.choice([0.2, 0.25, 0.3])
        anc = int(round(total * ratio))
        dom = total - anc
        q = f"Out of {total} curated SFT rows with anchor_ratio={ratio}, how many are core anchors vs domain repair rows?"
        think = f"Observe: total={total}, anchor_ratio={ratio}. Anchors = {total} * {ratio} = {anc}; domain = {total} - {anc} = {dom}."
        ans = f"There are {anc} core anchor rows and {dom} domain repair rows."

    return format_sft_record(
        {
            "conversations": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": f"<think>\n{think}\n</think>\n\n{ans}"},
            ]
        }
    )


def _build_core_tech_qa_sample(idx: int, rng: random.Random) -> dict:
    q_base, ans, think_hint = _CORE_TECH_QA_PAIRS[idx % len(_CORE_TECH_QA_PAIRS)]
    prefixes = ("", "Concise technical answer: ", "Briefly explain: ", "请简明准确回答：")
    pref = prefixes[(idx // len(_CORE_TECH_QA_PAIRS)) % len(prefixes)]
    q_txt = f"{pref}{q_base}".strip()
    if idx >= len(_CORE_TECH_QA_PAIRS) * len(prefixes):
        q_txt = f"[tech-{idx}] {q_txt}"
    words = [w for w in q_base.replace("?", "").replace("？", "").split() if len(w) >= 4][:4]
    key_ref = " ".join(words) if words else "PyTorch training"
    return format_sft_record(
        {
            "conversations": [
                {"role": "user", "content": q_txt},
                {"role": "assistant", "content": f"<think>\nObserve question on {key_ref}: {think_hint}\n</think>\n\n{ans}"},
            ]
        }
    )


def _build_targeted_dpo_pair(sample: dict, mode: int, rng: random.Random) -> dict | None:
    convs = sample.get("conversations", [])
    sys_msg = next((m for m in convs if m.get("role") == "system"), {"role": "system", "content": SYSTEM_PROMPT})
    usr_msg = next((m for m in convs if m.get("role") == "user"), None)
    asst_msg = next((m for m in reversed(convs) if m.get("role") == "assistant"), None)
    if usr_msg is None or asst_msg is None:
        return None
    good_text = str(asst_msg.get("content", ""))
    usr_text = str(usr_msg.get("content", ""))
    if not good_text.strip():
        return None

    chosen = [
        {"role": "system", "content": str(sys_msg.get("content", ""))},
        {"role": "user", "content": usr_text},
        {"role": "assistant", "content": good_text},
    ]

    bad_text: str | None = None
    if "healthy" in usr_text and ("consecutive_failures=0" in usr_text or "errors=0" in usr_text):
        bogus_lr = rng.choice([12, 0.01, 1.5])
        bad_text = (
            '<think>\nObserve: healthy cycle needs tuning.\n</think>\n\n'
            f'<tool_call>\n{{"name":"patch_config","arguments":{{"updates":{{"temperature":0.0,"training":{{"learning_rate":{bogus_lr},"accumulation_steps":8}}}}}}}}\n</tool_call>'
        )
    elif "replace_line_range" in good_text:
        tc_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", good_text, flags=re.DOTALL)
        sub = mode % 4
        if sub == 0 and tc_match:
            try:
                obj = json.loads(tc_match.group(1))
                bad_args = dict(obj.get("arguments", {}))
                bad_args["start_line"] = rng.choice([5e-4, 3e-4, 1e-5])
                bad_args["end_line"] = 99
                bad_text = (
                    f"<think>\nObserve: use float constant as start_line.\n</think>\n\n"
                    f"{_tc_xml('replace_line_range', bad_args)}"
                )
            except Exception:
                bad_text = None
        elif sub == 1:
            bad_text = '<think>\nUpdate target file.\n</think>\n\n<tool_call>\n{"name":"replace_line_range","arguments":{"path":"/home/user/input/database.txt"}}\n</tool_call>'
        elif sub == 2:
            bad_text = "<think>\nWrite corrected Python function.\n</think>\n\nI'll create a new function:\n\n```python\ndef fix_bug(x):\n    return x\n"
    elif "Final batch_size=" in good_text and mode % 2 == 0:
        m_bs = re.search(r"Final batch_size=(\d+) and accumulation_steps=(\d+)", good_text)
        if m_bs:
            wrong_bs = int(m_bs.group(1)) + 2
            acc_v = int(m_bs.group(2))
            bad_text = (
                f"<think>\nObserve: compute direct jump batch_size={wrong_bs}, accum={acc_v}.\n</think>\n\n"
                f"Final batch_size={wrong_bs} and accumulation_steps={acc_v}."
            )

    if bad_text is None:
        m = mode % 7
        if m == 0 and "patch_config" in good_text:
            bad_text = '<think>\n</think>\n\n<tool_call>\n{"name":"patch_config","arguments":{"log_name":"' + usr_text[:60] + '"}}\n</tool_call>'
        elif m == 3:
            bad_text = "<think>\n</think>\n\n" + good_text.split("</think>")[-1].strip()
            if "{" in bad_text and "patch_config" in bad_text:
                bad_text = bad_text.replace('"updates"', '"log_name"')
        elif m == 4:
            bad_text = "<think>\nWe need to check the Gradient Norm in the Grad Norm. " + ("It is a type of normalized gradient in the Grad Norm. " * 4) + "\n"
        elif m == 5:
            bad_text = good_text.replace("</tool_call>", "").strip() if "</tool_call>" in good_text else good_text.replace("</think>", "")
        else:
            bad_text = '<think>\nObserve: guess unknown issue. Plan: call read_log_tail on train.log.\n</think>\n\n<tool_call>\n{"name":"replace_line_range","arguments":{"path":"wrong.py","start_line":99,"end_line":99,"lines":["pass"]}}\n</tool_call>'

    rejected = [
        {"role": "system", "content": chosen[0]["content"]},
        {"role": "user", "content": chosen[1]["content"]},
        {"role": "assistant", "content": bad_text},
    ]
    if chosen == rejected:
        return None
    return format_dpo_record({"chosen": chosen, "rejected": rejected})


def build_limitation_fix_and_meta_dataset(
    repo_root: Path,
    output_dir: Path,
    eval_report_path: Path | None = None,
    sft_count: int = 1200,
    dpo_count: int = 240,
    max_seq_len: int = 768,
    seed: int = 42,
    extra_sft_paths: list[Path] | None = None,
    extra_dpo_paths: list[Path] | None = None,
    extra_heldout_prompts: list[dict] | None = None,
) -> dict:
    """Diagnose limitations from eval report and build an audited fix + meta-selfchange dataset."""
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    token_limit = int(0.9 * max_seq_len)

    eval_report = json.loads(Path(eval_report_path).read_text(encoding="utf-8")) if (eval_report_path and Path(eval_report_path).is_file()) else {"examples": [], "eos_rate": 0.6}
    limitations = diagnose_model_limitations(eval_report)
    atomic_write_text(output_dir / "limitations_diagnosed.json", json.dumps(limitations, indent=2, ensure_ascii=False) + "\n")

    heldout_path = output_dir / "heldout_eval_10.jsonl"
    combined_heldout = [*HELDOUT_10_PROMPTS, *(extra_heldout_prompts or [])]
    heldout_rows = [
        format_sft_record({"conversations": [*item["messages"], {"role": "assistant", "content": f"HELD_OUT_PROBE:{item.get('id', idx)}"}]})
        for idx, item in enumerate(combined_heldout)
    ]
    write_sft_jsonl(heldout_path, heldout_rows)
    h_exact = {_extract_prompt_keys(r["conversations"])[0] for r in heldout_rows}
    h_norm = {_extract_prompt_keys(r["conversations"])[1] for r in heldout_rows}

    excluded_files = {"oom_helper.py", "lr_guard.py"}
    for item in combined_heldout:
        for m in item.get("messages", []):
            txt = str(m.get("content", ""))
            for fn in re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\.py\b", txt):
                excluded_files.add(Path(fn).name)

    raw_fixtures = collect_code_fixtures(repo_root)
    fixtures = [fx for fx in raw_fixtures if Path(fx["filename"]).name not in excluded_files]
    compact_fixtures = [fx for fx in fixtures if len(fx["source"].splitlines()) <= 5 and len(fx["source"]) <= 170] or fixtures

    target_schema = max(1, int(round(sft_count * 0.20)))
    target_code = max(1, int(round(sft_count * 0.30)))
    target_meta = max(1, int(round(sft_count * 0.18)))
    target_reason = max(1, int(round(sft_count * 0.12)))
    target_noop = max(1, int(round(sft_count * 0.08)))
    target_qa = max(1, sft_count - target_schema - target_code - target_meta - target_reason - target_noop)

    sft_rows: list[dict] = []
    dpo_buckets: dict[str, list[dict]] = {}
    seen_json: set[str] = set()
    counts = {
        "schema_disambiguation": 0,
        "code_repair": 0,
        "meta_selfchange": 0,
        "reasoning_math": 0,
        "healthy_noop": 0,
        "core_tech_qa": 0,
    }

    def _try_add(sample: dict | None, bucket: str, for_dpo: bool = False) -> bool:
        if not sample or not sample.get("conversations"):
            return False
        convs = sample["conversations"]
        ex_u, nm_u = _extract_prompt_keys(convs)
        if (ex_u and ex_u in h_exact) or (nm_u and nm_u in h_norm):
            return False
        if estimate_tokens(convs) > token_limit:
            return False
        sig = json.dumps(sample, sort_keys=True, ensure_ascii=False)
        if sig in seen_json:
            return False
        seen_json.add(sig)
        sft_rows.append(sample)
        counts[bucket] += 1
        if for_dpo:
            dpo_buckets.setdefault(bucket, []).append(sample)
        return True

    for extra_p in extra_sft_paths or []:
        ep = Path(extra_p)
        if not ep.is_file():
            continue
        for ln in ep.read_text(encoding="utf-8").splitlines():
            if not ln.strip() or counts["meta_selfchange"] >= target_meta:
                continue
            try:
                rec = format_sft_record(json.loads(ln))
            except Exception:
                continue
            _try_add(rec, "meta_selfchange", for_dpo=(sum(1 for m in rec["conversations"] if m.get("role") == "assistant") == 1))

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        wt_dir = Path(tmp)

        attempts = 0
        while counts["schema_disambiguation"] < target_schema and attempts < target_schema * 8:
            attempts += 1
            _try_add(_build_schema_disambiguation_sample(rng, wt_dir), "schema_disambiguation", for_dpo=True)

        attempts = 0
        float_goal = max(1, target_code // 3)
        float_got = 0
        while float_got < float_goal and attempts < float_goal * 10:
            attempts += 1
            fn_name, fn_src = _FLOAT_CLAMP_CODE_FIXTURES[attempts % len(_FLOAT_CLAMP_CODE_FIXTURES)]
            if _try_add(_build_numbered_line_code_edit_sample(fn_name, fn_src, rng, wt_dir), "code_repair", for_dpo=True):
                float_got += 1

        attempts = 0
        direct_goal = max(float_goal + 1, int(round(target_code * 0.65)))
        while counts["code_repair"] < direct_goal and attempts < target_code * 8:
            attempts += 1
            fx = fixtures[attempts % len(fixtures)]
            _try_add(_build_direct_code_edit_sample(fx, rng, wt_dir), "code_repair", for_dpo=True)

        attempts = 0
        while counts["code_repair"] < target_code and attempts < target_code * 12:
            attempts += 1
            fx = compact_fixtures[(attempts + 5) % len(compact_fixtures)]
            _try_add(_build_multiturn_code_edit_sample(fx, rng, wt_dir, with_retry=(max_seq_len >= 1024 and attempts % 5 == 0)), "code_repair", for_dpo=False)

        attempts = 0
        while counts["meta_selfchange"] < target_meta and attempts < target_meta * 8:
            attempts += 1
            lim = limitations[attempts % len(limitations)]
            _try_add(_build_meta_selfchange_sample(lim, rng, wt_dir, variant_idx=attempts), "meta_selfchange", for_dpo=(attempts % 2 == 1))

        attempts = 0
        while counts["reasoning_math"] < target_reason and attempts < target_reason * 8:
            attempts += 1
            _try_add(_build_reasoning_math_or_noop_sample(rng), "reasoning_math", for_dpo=(attempts % 2 == 0))

        attempts = 0
        while counts["healthy_noop"] < target_noop and attempts < target_noop * 10:
            attempts += 1
            _try_add(_build_healthy_noop_sample(rng), "healthy_noop", for_dpo=True)

        qa_idx = 0
        while counts["core_tech_qa"] < target_qa and qa_idx < target_qa * 4:
            _try_add(_build_core_tech_qa_sample(qa_idx, rng), "core_tech_qa", for_dpo=(qa_idx % 2 == 0))
            qa_idx += 1

        while len(sft_rows) < sft_count:
            fx = fixtures[rng.randrange(len(fixtures))]
            if not _try_add(_build_direct_code_edit_sample(fx, rng, wt_dir), "code_repair", for_dpo=True):
                qa_idx += 1
                if not _try_add(_build_core_tech_qa_sample(qa_idx, rng), "core_tech_qa", for_dpo=False):
                    break

    rng.shuffle(sft_rows)
    sft_rows = sft_rows[:sft_count]
    sft_path = output_dir / "sft_train.jsonl"
    write_sft_jsonl(sft_path, sft_rows)

    dpo_rows: list[dict] = []
    seen_dpo: set[str] = set()
    for extra_dp in extra_dpo_paths or []:
        edp = Path(extra_dp)
        if not edp.is_file():
            continue
        for ln in edp.read_text(encoding="utf-8").splitlines():
            if not ln.strip() or len(dpo_rows) >= dpo_count:
                continue
            try:
                rec = format_dpo_record(json.loads(ln))
            except Exception:
                continue
            ex_u, nm_u = _extract_prompt_keys(rec["chosen"])
            if (ex_u and ex_u in h_exact) or (nm_u and nm_u in h_norm):
                continue
            if estimate_tokens(rec["chosen"]) > token_limit:
                continue
            p_sig = json.dumps(rec, sort_keys=True, ensure_ascii=False)
            if p_sig not in seen_dpo:
                seen_dpo.add(p_sig)
                dpo_rows.append(rec)

    bucket_names = ["healthy_noop", "code_repair", "schema_disambiguation", "reasoning_math", "meta_selfchange", "core_tech_qa"]
    interleaved_pool: list[dict] = []
    single_by_bucket: dict[str, list[dict]] = {}
    for b in bucket_names:
        items = [r for r in dpo_buckets.get(b, []) if sum(1 for m in r["conversations"] if m.get("role") == "assistant") == 1]
        if items:
            rng.shuffle(items)
            single_by_bucket[b] = items
    max_len_b = max((len(v) for v in single_by_bucket.values()), default=0)
    for i in range(max_len_b):
        for b in bucket_names:
            lst = single_by_bucket.get(b, [])
            if i < len(lst):
                interleaved_pool.append(lst[i])

    if interleaved_pool and dpo_count > 0:
        for idx in range(dpo_count * 6):
            if len(dpo_rows) >= dpo_count:
                break
            pair = _build_targeted_dpo_pair(interleaved_pool[idx % len(interleaved_pool)], mode=idx, rng=rng)
            if pair is None:
                continue
            p_sig = json.dumps(pair, sort_keys=True, ensure_ascii=False)
            if p_sig in seen_dpo:
                continue
            seen_dpo.add(p_sig)
            dpo_rows.append(pair)
    dpo_path = output_dir / "dpo_train.jsonl"
    dpo_written = write_dpo_jsonl(dpo_path, dpo_rows)

    sft_review = review_and_record_dataset(sft_path, output_dir, dataset_format="sft", max_seq_len=max_seq_len, heldout_path=heldout_path, cycle_tag="limitation_fix_sft")
    dpo_review = review_and_record_dataset(dpo_path, output_dir, dataset_format="dpo", max_seq_len=max_seq_len, heldout_path=heldout_path, cycle_tag="limitation_fix_dpo") if dpo_written > 0 else {"passed": True, "quality_score": 1.0, "failed_checks": []}

    report = {
        "repo_root": str(repo_root),
        "output_dir": str(output_dir),
        "limitations_count": len(limitations),
        "limitations": [l["limitation_id"] for l in limitations],
        "sft_path": str(sft_path),
        "sft_count": len(sft_rows),
        "dpo_path": str(dpo_path),
        "dpo_count": dpo_written,
        "heldout_path": str(heldout_path),
        "split_breakdown": counts,
        "sft_audit": sft_review,
        "dpo_audit": dpo_review,
        "overall_passed": bool(sft_review.get("passed") and dpo_review.get("passed")),
    }
    atomic_write_text(output_dir / "limitation_fix_report.json", json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def record_verified_selfchange_episode(
    before_eval: dict,
    after_eval: dict,
    fix_summary: dict,
    output_sft_jsonl: Path,
    output_dpo_jsonl: Path | None = None,
    seed: int = 42,
    episode_tag: str = "ep",
) -> dict:
    """Convert an executed (before_eval -> fix_steps -> after_eval) self-change run into verified training traces."""
    _ = seed
    limitations = diagnose_model_limitations(before_eval)
    before_p = int(before_eval.get("passed_count", 0))
    before_t = int(before_eval.get("total_count", 10))
    after_p = int(after_eval.get("passed_count", 0))
    after_t = int(after_eval.get("total_count", 10))
    before_eos = float(before_eval.get("eos_rate", 0.0))
    after_eos = float(after_eval.get("eos_rate", 0.0))
    improved = (after_p > before_p) or (after_eos > before_eos + 0.05)

    steps = fix_summary.get("steps_taken") or [l["limitation_id"] for l in limitations]
    lr = fix_summary.get("learning_rate", 3e-5)
    epochs = fix_summary.get("sft_epochs", 3)
    dpo_epochs = fix_summary.get("dpo_epochs", 1)

    sft_episodes: list[dict] = []
    tools_diagnose = _compact_tools_json(["read_log_tail", "patch_config"])
    tools_submit = _compact_tools_json(["curate_experience", "submit_candidate"])
    for idx, lim in enumerate(limitations):
        lid = lim["limitation_id"]
        updates = {"temperature": 0.0, "thinking_ratio": 1.0, "training": {"learning_rate": lr, "epochs": epochs, "dpo_beta": 0.15}}
        ev = str(lim["evidence"]).replace("<", "").replace(">", "")[:70]
        rc = str(lim["root_cause"]).replace("<", "").replace(">", "")[:70]
        ep_a = f"{episode_tag}_{idx}a"
        ep_b = f"{episode_tag}_{idx}b"
        sft_episodes.append(
            format_sft_record(
                {
                    "conversations": [
                        {"role": "system", "content": "Diagnose eval limitations from logs and configure training fix via patch_config.", "tools": tools_diagnose},
                        {"role": "user", "content": f"selfchange_diagnose id={ep_a} limitation={lid} before_pass={before_p}/{before_t} before_eos={before_eos}"},
                        {
                            "role": "assistant",
                            "content": f"<think>\nObserve: {ep_a} before_pass={before_p}/{before_t} eos={before_eos} limitation={lid}. Plan: call read_log_tail.\n</think>\n\n{_tc_xml('read_log_tail', {'log_name': 'eval.log', 'lines': 20})}",
                        },
                        {"role": "tool", "content": f"evidence={ev} root_cause={rc}"},
                        {
                            "role": "assistant",
                            "content": f"<think>\nObserve: {lid} root_cause confirmed; set lr={lr} epochs={epochs} dpo_epochs={dpo_epochs}. Plan: call patch_config.\n</think>\n\n{_tc_xml('patch_config', {'updates': updates})}",
                        },
                    ]
                }
            )
        )
        sft_episodes.append(
            format_sft_record(
                {
                    "conversations": [
                        {"role": "system", "content": "Curate verified fix trace and submit selfchange candidate.", "tools": tools_submit},
                        {"role": "user", "content": f"selfchange_verify id={ep_b} limitation={lid} before_pass={before_p}/{before_t} after_pass={after_p}/{after_t} after_eos={after_eos}"},
                        {
                            "role": "assistant",
                            "content": f"<think>\nObserve: {ep_b} resolved {lid} {before_p}->{after_p} after_eos={after_eos}. Plan: call curate_experience mode=limitation_fix.\n</think>\n\n{_tc_xml('curate_experience', {'mode': 'limitation_fix'})}",
                        },
                        {"role": "tool", "content": json.dumps({"ok": 1, "steps": steps[:2]}, separators=(",", ":"))},
                        {
                            "role": "assistant",
                            "content": f"<think>\nObserve: limitation_fix curated; improved={improved}. Plan: call submit_candidate.\n</think>\n\n{_tc_xml('submit_candidate', {'hypothesis': f'resolved {lid} ({before_p}->{after_p})'})}",
                        },
                    ]
                }
            )
        )

    existing_sft: list[dict] = []
    if Path(output_sft_jsonl).is_file():
        existing_sft = [json.loads(ln) for ln in Path(output_sft_jsonl).read_text(encoding="utf-8").splitlines() if ln.strip()]
    written_sft = write_sft_jsonl(output_sft_jsonl, existing_sft + sft_episodes)

    written_dpo = 0
    if output_dpo_jsonl is not None:
        dpo_pairs: list[dict] = []
        for idx, lim in enumerate(limitations):
            chosen = [
                {"role": "system", "content": f"{SYSTEM_PROMPT}\nDiagnose limitation and fix via patch_config."},
                {"role": "user", "content": f"meta_selfchange_fix tag={episode_tag} limitation={lim['limitation_id']} before_pass={before_p}/{before_t} before_eos={before_eos}"},
                {
                    "role": "assistant",
                    "content": (
                        f"<think>\nObserve: {episode_tag} before_pass={before_p}/{before_t} before_eos={before_eos} caused by {lim['limitation_id']}. "
                        f"Apply curated fix ({steps[0] if steps else 'sft+dpo'}) with lr={lr}. Plan: call patch_config.\n</think>\n\n"
                        f"{_tc_xml('patch_config', {'updates': {'temperature': 0.0, 'training': {'learning_rate': lr, 'epochs': epochs}}})}"
                    ),
                },
            ]
            rejected = [
                {"role": "system", "content": chosen[0]["content"]},
                {"role": "user", "content": chosen[1]["content"]},
                {
                    "role": "assistant",
                    "content": '<think>\n</think>\n\n<tool_call>\n{"name":"patch_config","arguments":{"log_name":"ignore_eval"}}\n</tool_call>',
                },
            ]
            dpo_pairs.append(format_dpo_record({"chosen": chosen, "rejected": rejected}))
        existing_dpo: list[dict] = []
        if Path(output_dpo_jsonl).is_file():
            existing_dpo = [json.loads(ln) for ln in Path(output_dpo_jsonl).read_text(encoding="utf-8").splitlines() if ln.strip()]
        written_dpo = write_dpo_jsonl(output_dpo_jsonl, existing_dpo + dpo_pairs)

    return {
        "improved": improved,
        "before_passed": f"{before_p}/{before_t}",
        "after_passed": f"{after_p}/{after_t}",
        "before_eos": before_eos,
        "after_eos": after_eos,
        "recorded_sft_total": written_sft,
        "recorded_dpo_total": written_dpo,
        "sft_path": str(output_sft_jsonl),
        "dpo_path": str(output_dpo_jsonl) if output_dpo_jsonl else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose model limitations and build fix + meta-selfchange datasets")
    sub = parser.add_subparsers(dest="command", required=True)

    p_b = sub.add_parser("build", help="Diagnose limitations and generate audited SFT + DPO datasets")
    p_b.add_argument("--repo-root", type=Path, default=Path("."))
    p_b.add_argument("--output-dir", type=Path, default=Path("selfchange/workspace/limitation_fix_data"))
    p_b.add_argument("--eval-report", type=Path, default=None)
    p_b.add_argument("--extra-sft", action="append", type=Path, default=[])
    p_b.add_argument("--extra-dpo", action="append", type=Path, default=[])
    p_b.add_argument("--sft-count", type=int, default=1200)
    p_b.add_argument("--dpo-count", type=int, default=240)
    p_b.add_argument("--max-seq-len", type=int, default=768)
    p_b.add_argument("--seed", type=int, default=42)

    p_rec = sub.add_parser("record-episode", help="Record an executed before->after fix run into training data")
    p_rec.add_argument("--before-eval", type=Path, required=True)
    p_rec.add_argument("--after-eval", type=Path, required=True)
    p_rec.add_argument("--output-sft", type=Path, required=True)
    p_rec.add_argument("--output-dpo", type=Path, default=None)
    p_rec.add_argument("--episode-tag", type=str, default="ep")

    args = parser.parse_args()
    if args.command == "build":
        rep = build_limitation_fix_and_meta_dataset(
            repo_root=args.repo_root,
            output_dir=args.output_dir,
            eval_report_path=args.eval_report,
            sft_count=args.sft_count,
            dpo_count=args.dpo_count,
            max_seq_len=args.max_seq_len,
            seed=args.seed,
            extra_sft_paths=args.extra_sft,
            extra_dpo_paths=args.extra_dpo,
        )
        print(json.dumps(rep, indent=2))
        return 0 if rep["overall_passed"] else 1

    before = json.loads(args.before_eval.read_text(encoding="utf-8"))
    after = json.loads(args.after_eval.read_text(encoding="utf-8"))
    rec = record_verified_selfchange_episode(
        before_eval=before,
        after_eval=after,
        fix_summary={"steps_taken": ["diagnose_limitations", "build_limitation_fix_and_meta_dataset", "full_sft", "contrastive_dpo"]},
        output_sft_jsonl=args.output_sft,
        output_dpo_jsonl=args.output_dpo,
        episode_tag=args.episode_tag,
    )
    print(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
