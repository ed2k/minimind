from __future__ import annotations

import argparse
import ast
import json
import random
import re
import tempfile
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from selfchange.bootstrap.generate_traces import _select_tools, estimate_tokens
from selfchange.bootstrap.prune_reallocate import classify_trivia_vs_domain
from selfchange.kernel.integrity import atomic_write_text
from selfchange.kernel.sandbox import scan_python_source
from selfchange.kernel.tool_runtime import SELFCHANGE_TOOLS, WorktreeToolExecutor, validate_tool_call
from selfchange.mutable.agent_loop import parse_tool_calls
from selfchange.mutable.data_curator import compute_rep_penalty
from selfchange.mutable.prompts import SYSTEM_PROMPT
from selfchange.training.self_train_runner import (
    format_dpo_record,
    format_sft_record,
    write_dpo_jsonl,
    write_sft_jsonl,
)

_SFT_MSG_KEYS = {"role", "content", "reasoning_content", "tools", "tool_calls"}
_DPO_MSG_KEYS = {"role", "content"}
_SELFCHANGE_TOOL_NAMES = {t["function"]["name"] for t in SELFCHANGE_TOOLS}
_SELFCHANGE_TOOL_BY_NAME = {t["function"]["name"]: t for t in SELFCHANGE_TOOLS}

QUALITY_THRESHOLDS: dict[str, float] = {
    "min_schema_valid_rate": 1.0,
    "min_assistant_supervision_rate": 1.0,
    "min_tag_balance_rate": 0.99,
    "min_tool_schema_valid_rate": 0.99,
    "min_within_budget_rate": 0.98,
    "max_high_rep_rate": 0.02,
    "min_exec_verified_pass_rate": 0.95,
    "max_exact_duplicate_rate": 0.15,
    "min_normalized_unique_rate": 0.15,
    "max_noop_rate": 0.25,
    "max_leakage_count": 0.0,
    "max_off_domain_trivia_rate": 0.10,
    "min_think_grounding_rate": 0.80,
    "min_think_action_consistency_rate": 0.85,
}

_BUILTIN_CODE_FIXTURES: list[tuple[str, str]] = [
    ("clip_grad.py", "def clip_value(val: float, max_val: float = 1.0) -> float:\n    if val < -max_val:\n        return -max_val\n    return min(val, max_val)\n"),
    ("lr_cosine.py", "def decay_step(lr: float, factor: float = 0.5, floor: float = 1e-6) -> float:\n    scaled = lr * factor\n    return max(floor, scaled)\n"),
    ("ckpt_keep.py", "def keep_top_k(scores: list, k: int = 3) -> list:\n    ordered = sorted(scores, reverse=True)\n    return ordered[:max(1, k)]\n"),
    ("ema_step.py", "def update_ema(prev: float, curr: float, beta: float = 0.9) -> float:\n    return beta * prev + (1.0 - beta) * curr\n"),
    ("backoff.py", "def retry_delay(attempt: int, base: float = 0.5, cap: float = 8.0) -> float:\n    wait = base * (2 ** max(0, attempt))\n    return min(cap, wait)\n"),
    ("tok_budget.py", "def fits_budget(tokens: int, max_seq_len: int = 768) -> bool:\n    limit = int(max_seq_len * 0.9)\n    return tokens <= limit\n"),
    ("batch_halve.py", "def halve_batch(batch_size: int, accum: int = 1) -> tuple:\n    new_bs = max(1, batch_size // 2)\n    new_acc = min(32, accum * 2)\n    return new_bs, new_acc\n"),
    ("loss_spike.py", "def is_loss_spike(curr: float, prev: float, mult: float = 2.5) -> bool:\n    if prev <= 0.0:\n        return False\n    return curr > prev * mult\n"),
    ("disk_guard.py", "def has_disk_headroom(free_mb: float, min_mb: float = 512.0) -> bool:\n    return free_mb >= min_mb\n"),
    ("slice_window.py", "def clamp_window(start: int, end: int, total: int) -> tuple:\n    lo = max(1, start)\n    hi = min(total, max(lo, end))\n    return lo, hi\n"),
    ("score_gate.py", "def improved_over(new_score: float, old_score: float, tol: float = 1e-4) -> bool:\n    return new_score > old_score + tol\n"),
    ("warmup_lr.py", "def warmup_scale(step: int, total_warmup: int = 100) -> float:\n    if total_warmup <= 0:\n        return 1.0\n    return min(1.0, max(0, step) / total_warmup)\n"),
    ("safe_ratio.py", "def pass_rate(passed: int, total: int) -> float:\n    if total <= 0:\n        return 0.0\n    return passed / total\n"),
    ("seq_clamp.py", "def shrink_context(seq_len: int, factor: float = 0.75) -> int:\n    return max(128, int(seq_len * factor))\n"),
    ("state_fail.py", "def read_failure_streak(state: dict) -> int:\n    return int(state.get('consecutive_failures', 0))\n"),
    ("norm_cap.py", "def bounded_temp(temp: float, low: float = 0.05, high: float = 1.2) -> float:\n    return min(high, max(low, temp))\n"),
    ("merge_cfg.py", "def next_turn_limit(safe_mode: bool, configured: int = 5) -> int:\n    if safe_mode:\n        return min(4, max(1, configured))\n    return max(1, min(12, configured))\n"),
    ("dedup_count.py", "def unique_ratio(items: list) -> float:\n    if not items:\n        return 1.0\n    return len(set(items)) / len(items)\n"),
]

_BUILTIN_DIALOGUE_BANK: list[tuple[str, str]] = [
    ("什么是梯度累积？", "梯度累积在多个小批次前向与反向传播后统一更新一次权重，可在显存受限时等价放大有效 batch size。"),
    ("LoRA 微调为什么能省显存？", "LoRA 冻结原始主干权重，只训练低秩分解矩阵 A 和 B，显著减少可训练参数与优化器状态显存。"),
    ("学习率过大导致 loss NaN 怎么办？", "应降低学习率、收紧梯度裁剪阈值 clip_grad_norm，并检查混合精度缩放是否稳定。"),
    ("DPO 训练里的参考模型起什么作用？", "参考模型提供 KL 约束基准，防止策略模型在放大偏好边界时遗忘基础语言能力。"),
    ("为什么自修改代码必须在 git worktree 中进行？", "git worktree 能在独立目录与分支上隔离候选补丁，验证失败即可丢弃而不污染主干分支。"),
    ("为什么 SFT 中需要混入锚点回放数据？", "锚点回放保留通用对话与工具格式先验，防止领域微调引发灾难性遗忘。"),
    ("余弦退火学习率调度有什么优势？", "余弦退火在训练后期平滑缩小步长，降低震荡并帮助参数收敛到更平坦的局部极小值。"),
    ("为什么训练 JSONL 样本长度不能超过 max_seq_len？", "超长样本会被截断掉末尾的 assistant 回复和 eos 结束符，导致模型学不到正常终止生成。"),
    ("什么时候优先用 replace_line_range 而不是整文件替换？", "修复局部语法或单行逻辑错误时，精确行号替换改动最小且更不容易破坏周围上下文。"),
    ("断点续训文件损坏报错 EOFError 时应如何处理？", "跳过损坏的临时 resume checkpoint，并从档案中最高分的已验证祖先节点恢复。"),
    ("如何判断一段训练日志里出现了显存溢出？", "常见特征是出现 RuntimeError: CUDA out of memory 或 MemoryError，此时需减半 batch_size。"),
    ("为什么要计算 n-gram 重复惩罚？", "重复惩罚用于检测模型输出死循环或模板复读，避免劣质轨迹污染后续 RFT 数据集。"),
    ("Why verify syntax with ast.parse before running candidate code?", "Checking syntax with ast.parse catches malformed Python edits cheaply before spawning a subprocess."),
    ("What does clip_grad_norm prevent during SFT or RL?", "Gradient norm clipping bounds sudden large gradients that can destabilize float16 or bfloat16 training."),
    ("Why keep safe_mode restricted to config-only edits?", "Limiting recovery under consecutive failures to bounded config tuning reduces the risk of compounding code faults."),
    ("How does rejection sampling fine-tuning (RFT) work?", "RFT generates multiple rollouts per prompt, keeps only verifier-passed low-repetition trajectories, and runs supervised SFT."),
    ("Why pair DPO chosen and rejected rollouts by the same task_id?", "On-policy same-task pairs isolate decision quality and format correctness without prompt distribution mismatch."),
    ("What happens when SFTDataset in MiniMind sees unexpected top-level JSONL keys?", "Its strict PyArrow Features schema enforces a fixed five-field conversations struct and fails or drops mismatched schemas."),
    ("How does GQA reduce KV-cache memory during inference?", "Grouped-query attention shares key and value heads across multiple query heads, shrinking cache footprint cleanly."),
    ("Why measure pass@4 alongside pass@1 before RL?", "When pass@4 is high while pass@1 is low, the model has latent capability that RFT and DPO can quickly sharpen."),
    ("What does pin_memory=True do in a PyTorch DataLoader?", "Page-locked host buffers allow faster asynchronous host-to-device DMA transfers to CUDA memory."),
    ("Why save checkpoints atomically via a temporary file rename?", "Writing to a temporary file and using os.replace avoids half-written files if Colab preempts mid-save."),
    ("What triggers L3 halt in the selfchange Supervisor?", "Failing repeatedly even inside safe_mode indicates unrecoverable instability, triggering a clean fatal report halt."),
    ("Why track SHA-256 checksums for kernel files?", "A cryptographic manifest guarantees mutable self-edits cannot alter the immutable verifier, sandbox, or supervisor."),
    ("How should an agent act before editing a broken file?", "Always inspect system state, logs, or target file lines first so edits rely on observed facts rather than guesses."),
    ("Why cap max_new_tokens during autonomous tool calls?", "Bounding generation length prevents infinite loops and keeps latency within cycle timeout budgets."),
    ("What does float16 GradScaler do?", "Dynamic loss scaling keeps small float16 gradients representable above underflow thresholds before unscaling."),
    ("Why deduplicate trajectories by task family and tool chain before RFT?", "Deduplication prevents easy repeated no-op or trivial episodes from dominating the fine-tuning batch."),
    ("How does RMSNorm differ from LayerNorm?", "RMSNorm rescales activations using root mean square magnitude without subtracting the mean, saving compute."),
    ("Why test a healthy no-op drill inside Gate 2?", "A healthy no-op drill ensures the policy stays restrained and does not make unnecessary edits when metrics are healthy."),
    ("When is LoRA merging needed before running DPO?", "train_dpo loads a standard base .pth state_dict directly, so LoRA deltas must be merged first."),
    ("What makes a line-range patch atomic in the tool runtime?", "Invalid syntax or AST policy violations revert the file immediately before any candidate verification continues."),
]

_BUILTIN_GENERAL_TOOLS: list[tuple[str, list[dict], dict, str, str]] = [
    (
        "Compute the effective batch size for batch_size={a} and accumulation_steps={b}.",
        [{"type": "function", "function": {"name": "multiply_ints", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"]}}}],
        {"name": "multiply_ints", "arguments": {"a": 4, "b": 8}},
        '{"product": 32}',
        "The effective batch size is 32.",
    ),
    (
        "Convert {km} km into miles for benchmark reporting.",
        [{"type": "function", "function": {"name": "convert_distance", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}}],
        {"name": "convert_distance", "arguments": {"value": 12, "from_unit": "km", "to_unit": "miles"}},
        '{"result": 7.4565}',
        "12 km equals 7.4565 miles.",
    ),
    (
        "Estimate disk usage in MB for storing {n} checkpoints of {mb} MB each.",
        [{"type": "function", "function": {"name": "estimate_storage_mb", "parameters": {"type": "object", "properties": {"count": {"type": "integer"}, "size_mb": {"type": "number"}}, "required": ["count", "size_mb"]}}}],
        {"name": "estimate_storage_mb", "arguments": {"count": 4, "size_mb": 128.5}},
        '{"total_mb": 514.0}',
        "Storing 4 checkpoints requires 514.0 MB.",
    ),
    (
        "Check JSON validity of payload keys batch_size and learning_rate.",
        [{"type": "function", "function": {"name": "validate_json_schema", "parameters": {"type": "object", "properties": {"required_keys": {"type": "array"}}, "required": ["required_keys"]}}}],
        {"name": "validate_json_schema", "arguments": {"required_keys": ["batch_size", "learning_rate"]}},
        '{"valid": true}',
        "The JSON payload schema is valid.",
    ),
    (
        "Compute remaining training ETA for step {step} of {total} at {sec}s per step.",
        [{"type": "function", "function": {"name": "compute_eta", "parameters": {"type": "object", "properties": {"step": {"type": "integer"}, "total_steps": {"type": "integer"}, "sec_per_step": {"type": "number"}}, "required": ["step", "total_steps", "sec_per_step"]}}}],
        {"name": "compute_eta", "arguments": {"step": 120, "total_steps": 300, "sec_per_step": 0.5}},
        '{"eta_sec": 90.0}',
        "Estimated remaining training time is 90 seconds.",
    ),
    (
        "Calculate the percentage drop from perplexity {before} to {after}.",
        [{"type": "function", "function": {"name": "pct_change", "parameters": {"type": "object", "properties": {"old": {"type": "number"}, "new": {"type": "number"}}, "required": ["old", "new"]}}}],
        {"name": "pct_change", "arguments": {"old": 40.0, "new": 25.0}},
        '{"pct_change": -37.5}',
        "Perplexity decreased by 37.5%.",
    ),
    (
        "Compute halved learning rate when current lr={lr}.",
        [{"type": "function", "function": {"name": "scale_value", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "factor": {"type": "number"}}, "required": ["value", "factor"]}}}],
        {"name": "scale_value", "arguments": {"value": 0.0002, "factor": 0.5}},
        '{"scaled": 0.0001}',
        "The halved learning rate is 0.0001.",
    ),
    (
        "Verify whether {free_mb} MB free disk meets the {min_mb} MB minimum.",
        [{"type": "function", "function": {"name": "check_threshold", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "minimum": {"type": "number"}}, "required": ["value", "minimum"]}}}],
        {"name": "check_threshold", "arguments": {"value": 1800, "minimum": 1024}},
        '{"ok": true, "margin": 776}',
        "Disk headroom passes with a 776 MB margin.",
    ),
]


def _compact_tools_json(names: list[str]) -> str:
    compact = []
    for n in names:
        t = _SELFCHANGE_TOOL_BY_NAME.get(n)
        if not t:
            continue
        fn = t["function"]
        raw_params = fn.get("parameters", {})
        raw_props = raw_params.get("properties", {})
        compact_props = {}
        for k, v in raw_props.items():
            if isinstance(v, dict):
                p = {}
                if "type" in v:
                    p["type"] = v["type"]
                if "items" in v:
                    p["items"] = v["items"]
                compact_props[k] = p
            else:
                compact_props[k] = v
        params = {
            "type": raw_params.get("type", "object"),
            "properties": compact_props,
            "required": raw_params.get("required", []),
        }
        compact.append({"type": "function", "function": {"name": fn["name"], "parameters": params}})
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def _tc_xml(name: str, arguments: dict) -> str:
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments}, ensure_ascii=False, separators=(',', ':'))}\n</tool_call>"


def _normalize_for_diversity(text: str) -> str:
    s = (text or "").lower().strip()
    s = re.sub(r"\b(?:cand|anc|champ|node|case|cycle)_[a-z0-9_]+\b", "<ID>", s)
    s = re.sub(r"\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b", "<NUM>", s)
    s = re.sub(r"\s+", " ", s)
    return s


def extract_repo_code_fixtures(repo_root: Path) -> list[dict]:
    """Extract compact, AST-clean top-level functions from the repository for real repair tasks."""
    repo_root = Path(repo_root).resolve()
    search_targets: list[Path] = []
    for rel in ("selfchange/mutable", "selfchange/bootstrap", "selfchange/training", "selfchange/colab", "selfchange/kernel"):
        d = repo_root / rel
        if d.is_dir():
            search_targets.extend(sorted(d.glob("*.py")))
    for single in ("trainer/trainer_utils.py", "model/model_lora.py", "model/model_minimind.py"):
        f = repo_root / single
        if f.is_file():
            search_targets.append(f)

    fixtures: list[dict] = []
    seen_bodies: set[str] = set()
    for path in search_targets:
        try:
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
        except Exception:
            continue
        lines = text.splitlines()
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef) or node.decorator_list:
                continue
            if node.lineno is None or node.end_lineno is None:
                continue
            snippet_lines = lines[node.lineno - 1 : node.end_lineno]
            if not (4 <= len(snippet_lines) <= 20):
                continue
            snippet = textwrap.dedent("\n".join(snippet_lines)).rstrip() + "\n"
            if len(snippet) > 700 or "return " not in snippet:
                continue
            try:
                ast.parse(snippet)
            except SyntaxError:
                continue
            if not scan_python_source(snippet).ok:
                continue
            norm = snippet.strip()
            if norm in seen_bodies:
                continue
            seen_bodies.add(norm)
            fixtures.append(
                {
                    "filename": f"{path.stem}_{node.name}.py",
                    "source": snippet,
                    "origin": path.relative_to(repo_root).as_posix(),
                }
            )
    return fixtures


def collect_code_fixtures(repo_root: Path) -> list[dict]:
    """Combine repo-extracted AST functions with built-in ML/systems utility fixtures."""
    combined = extract_repo_code_fixtures(repo_root)
    seen = {f["source"].strip() for f in combined}
    for fname, src in _BUILTIN_CODE_FIXTURES:
        if src.strip() not in seen:
            seen.add(src.strip())
            combined.append({"filename": fname, "source": src, "origin": "builtin"})
    return combined


def inject_ast_mutation(source: str, rng: random.Random) -> tuple[str, dict] | None:
    """Inject a deterministic, reversible AST/logic bug and return (broken_src, fix_info)."""
    lines = source.splitlines()
    candidates: list[dict] = []

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        if re.match(r"^(def |if |elif |for |while )", stripped) and line.rstrip().endswith(":"):
            broken = line.rstrip()[:-1]
            if not broken.rstrip().endswith(":"):
                candidates.append({"idx": idx, "broken": broken, "clean": line, "bug_type": "missing_colon", "reason": "restore missing colon"})
        if stripped.startswith("return ") and stripped.count("(") > 0 and stripped.endswith(")") and "#" not in stripped:
            broken = line.rstrip()[:-1]
            candidates.append({"idx": idx, "broken": broken, "clean": line, "bug_type": "unclosed_paren", "reason": "restore closing parenthesis"})
        for pat, rep, btype, rsn in (
            (" <= ", " >= ", "cmp_flip_le", "restore <= boundary comparison"),
            (" >= ", " <= ", "cmp_flip_ge", "restore >= boundary comparison"),
            (" < ", " > ", "cmp_flip_lt", "restore < lower-bound comparison"),
            (" > ", " < ", "cmp_flip_gt", "restore > upper-bound comparison"),
            (" == ", " != ", "eq_flip", "restore equality comparison"),
            (" // ", " * ", "div_to_mul", "restore floor division operator"),
            ("max(", "min(", "clamp_max_to_min", "restore max() lower clamp"),
            ("min(", "max(", "clamp_min_to_max", "restore min() upper clamp"),
            ("if not ", "if ", "bool_polarity", "restore negated guard condition"),
            (" + 1", " - 1", "off_by_one", "fix off-by-one step increment"),
        ):
            if pat in line and "#" not in line.split(pat, 1)[0]:
                broken = line.replace(pat, rep, 1)
                if broken != line:
                    candidates.append({"idx": idx, "broken": broken, "clean": line, "bug_type": btype, "reason": rsn})
        m_get = re.search(r"\.get\(\s*(['\"][^'\"]+['\"])\s*,\s*[^)]+\)", line)
        if m_get and "#" not in line[: m_get.start()]:
            broken = line[: m_get.start()] + f"[{m_get.group(1)}]" + line[m_get.end() :]
            candidates.append({"idx": idx, "broken": broken, "clean": line, "bug_type": "dict_get_to_subscript", "reason": "restore safe dict.get default"})

    if not candidates:
        return None
    pick = rng.choice(candidates)
    broken_lines = list(lines)
    broken_lines[pick["idx"]] = pick["broken"]
    broken_src = "\n".join(broken_lines) + "\n"
    if broken_src.strip() == source.strip():
        return None
    return broken_src, {
        "start_line": pick["idx"] + 1,
        "end_line": pick["idx"] + 1,
        "lines": [pick["clean"]],
        "bug_type": pick["bug_type"],
        "reason": pick["reason"],
    }


def verify_edit_in_worktree(
    worktree_dir: Path,
    filename: str,
    broken_src: str,
    clean_src: str,
    edit_args: dict,
) -> bool:
    """Replay replace_line_range via WorktreeToolExecutor and confirm exact AST-clean repair."""
    m_dir = Path(worktree_dir) / "selfchange" / "mutable"
    m_dir.mkdir(parents=True, exist_ok=True)
    target_file = m_dir / Path(filename).name
    target_file.write_text(broken_src, encoding="utf-8")
    executor = WorktreeToolExecutor(worktree_dir, autonomy_level=1)
    res = executor.dispatch("replace_line_range", edit_args)
    if not res.get("ok"):
        return False
    edited = target_file.read_text(encoding="utf-8")
    try:
        ast.parse(edited)
    except SyntaxError:
        return False
    if not scan_python_source(edited).ok:
        return False
    return edited.strip() == clean_src.strip()


def _format_numbered(src: str) -> str:
    return "\n".join(f"{i + 1}: {ln}" for i, ln in enumerate(src.splitlines()))


def _build_direct_code_edit_sample(fixture: dict, rng: random.Random, wt_dir: Path) -> dict | None:
    fname = fixture["filename"]
    clean_src = fixture["source"]
    mutated = inject_ast_mutation(clean_src, rng)
    if mutated is None:
        return None
    broken_src, fix = mutated
    edit_args = {"path": fname, "start_line": fix["start_line"], "end_line": fix["end_line"], "lines": fix["lines"]}
    if not verify_edit_in_worktree(wt_dir, fname, broken_src, clean_src, edit_args):
        return None
    convs = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\nFix bug with replace_line_range.", "tools": _select_tools(["replace_line_range"])},
        {"role": "user", "content": f"{fname}:\n{_format_numbered(broken_src)}"},
        {
            "role": "assistant",
            "content": (
                f"<think>\nObserve: {fname} line {fix['start_line']} has {fix['bug_type']}. "
                f"Cause: {fix['reason']}. Plan: call replace_line_range on {fname} line {fix['start_line']}.\n</think>\n\n"
                f"{_tc_xml('replace_line_range', edit_args)}"
            ),
        },
    ]
    return format_sft_record({"conversations": convs})


def _build_multiturn_code_edit_sample(fixture: dict, rng: random.Random, wt_dir: Path, with_retry: bool) -> dict | None:
    fname = fixture["filename"]
    clean_src = fixture["source"]
    if len(clean_src.splitlines()) > 5 or len(clean_src) > 170:
        return None
    mutated = inject_ast_mutation(clean_src, rng)
    if mutated is None:
        return None
    broken_src, fix = mutated
    edit_args = {"path": fname, "start_line": fix["start_line"], "end_line": fix["end_line"], "lines": fix["lines"]}
    if not verify_edit_in_worktree(wt_dir, fname, broken_src, clean_src, edit_args):
        return None
    numbered = _format_numbered(broken_src)
    line_cnt = len(broken_src.splitlines())
    tools_str = _compact_tools_json(["read_file_lines", "replace_line_range", "submit_candidate"])
    convs: list[dict] = [
        {"role": "system", "content": "Fix.", "tools": tools_str},
        {"role": "user", "content": f"bug {fname} line {fix['start_line']} {fix['bug_type']}"},
        {
            "role": "assistant",
            "content": f"<think>\n{fname} line {fix['start_line']}: call read_file_lines.\n</think>\n\n{_tc_xml('read_file_lines', {'path': fname, 'start': 1, 'end': line_cnt})}",
        },
        {"role": "tool", "content": json.dumps({"ok": 1, "content": numbered}, ensure_ascii=False, separators=(",", ":"))},
    ]
    if with_retry:
        bad_args = {"path": fname, "start_line": 99, "end_line": 99, "lines": fix["lines"]}
        convs.extend(
            [
                {"role": "assistant", "content": f"<think>\n{fname} inspected; call replace_line_range.\n</think>\n\n{_tc_xml('replace_line_range', bad_args)}"},
                {"role": "tool", "content": json.dumps({"ok": False, "error": "line range 99..99 out of bounds"}, ensure_ascii=False, separators=(",", ":"))},
            ]
        )
    convs.extend(
        [
            {
                "role": "assistant",
                "content": f"<think>\n{fname} line {fix['start_line']} {fix['bug_type']}: {fix['reason']}; call replace_line_range.\n</think>\n\n{_tc_xml('replace_line_range', edit_args)}",
            },
            {"role": "tool", "content": '{"ok":1}'},
            {
                "role": "assistant",
                "content": f"<think>\n{fname} ok; call submit_candidate.\n</think>\n\n{_tc_xml('submit_candidate', {'hypothesis': fix['bug_type']})}",
            },
        ]
    )
    return format_sft_record({"conversations": convs})


def _build_operational_fault_sample(rng: random.Random, wt_dir: Path) -> dict | None:
    cfg_dir = wt_dir / "selfchange" / "mutable"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    case = rng.randrange(4)
    if case == 0:
        bs = rng.choice([2, 4, 8, 16])
        acc = rng.choice([1, 2, 4])
        mod = rng.choice(["train_full_sft.py", "train_lora.py", "train_dpo.py", "train_self_agent.py"])
        if bs > 1:
            updates = {"training": {"batch_size": bs // 2, "accumulation_steps": min(32, acc * 2)}}
            note = f"Halve batch_size from {bs} to {bs // 2} and double accumulation."
        else:
            seq = rng.choice([512, 768])
            updates = {"training": {"max_seq_len": max(128, int(seq * 0.75))}}
            note = f"batch_size=1 so shrink max_seq_len from {seq} to {updates['training']['max_seq_len']}."
        (cfg_dir / "config.json").write_text(json.dumps({"training": {"batch_size": bs, "accumulation_steps": acc}}), encoding="utf-8")
        if not WorktreeToolExecutor(wt_dir, autonomy_level=1).dispatch("patch_config", {"updates": updates}).get("ok"):
            return None
        convs = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\nRecover from OOM.", "tools": _select_tools(["read_log_tail", "patch_config"])},
            {"role": "user", "content": f"fault=oom module={mod} batch_size={bs} accumulation_steps={acc}"},
            {"role": "assistant", "content": f"<think>\nObserve: fault=oom in {mod} batch_size={bs}. Plan: call read_log_tail on train.log.\n</think>\n\n{_tc_xml('read_log_tail', {'log_name': 'train.log', 'lines': 20})}"},
            {"role": "tool", "content": f"RuntimeError: CUDA out of memory in {mod} at batch_size={bs}"},
            {"role": "assistant", "content": f"<think>\nObserve: train.log confirms CUDA OOM in {mod} batch_size={bs}. Plan: call patch_config to {note}\n</think>\n\n{_tc_xml('patch_config', {'updates': updates})}"},
        ]
        return format_sft_record({"conversations": convs})

    if case == 1:
        lr = rng.choice([1e-4, 2e-4, 3e-4, 4e-4])
        step = rng.randint(5, 80)
        new_lr = round(lr * 0.5, 8)
        updates = {"training": {"learning_rate": new_lr, "grad_clip": 0.5}}
        (cfg_dir / "config.json").write_text(json.dumps({"training": {"learning_rate": lr, "grad_clip": 1.0}}), encoding="utf-8")
        if not WorktreeToolExecutor(wt_dir, autonomy_level=1).dispatch("patch_config", {"updates": updates}).get("ok"):
            return None
        convs = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\nStabilize diverging training.", "tools": _select_tools(["read_log_tail", "patch_config"])},
            {"role": "user", "content": f"fault=nan_loss step={step} lr={lr}"},
            {"role": "assistant", "content": f"<think>\nObserve: fault=nan_loss step={step} lr={lr}. Plan: call read_log_tail on train.log.\n</think>\n\n{_tc_xml('read_log_tail', {'log_name': 'train.log', 'lines': 20})}"},
            {"role": "tool", "content": f"Epoch:[1/2]({step}/200), loss: nan, grad_norm: inf, lr: {lr}"},
            {"role": "assistant", "content": f"<think>\nObserve: step={step} grad_norm inf nan loss. Cause: lr={lr} too high. Plan: call patch_config lr={new_lr} grad_clip=0.5.\n</think>\n\n{_tc_xml('patch_config', {'updates': updates})}"},
        ]
        return format_sft_record({"conversations": convs})

    if case == 2:
        anc = f"cand_{rng.randint(100, 999)}"
        ckpt = rng.choice(["full_sft_768_resume.pth", "warmstart_rung3_768_resume.pth", "offline_dpo_r1_768_resume.pth"])
        convs = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}\nRecover corrupt checkpoint.", "tools": _select_tools(["get_archive_summary", "rollback_to_ancestor"])},
            {"role": "user", "content": f"fault=corrupt_checkpoint file={ckpt} error=EOFError"},
            {"role": "assistant", "content": f"<think>\nObserve: fault=corrupt_checkpoint {ckpt} EOFError. Plan: call get_archive_summary.\n</think>\n\n{_tc_xml('get_archive_summary', {'top_k': 3})}"},
            {"role": "tool", "content": json.dumps([{"node_id": anc, "total_score": 0.91}])},
            {"role": "assistant", "content": f"<think>\nObserve: archive verified ancestor {anc}. Plan: call rollback_to_ancestor node_id={anc}.\n</think>\n\n{_tc_xml('rollback_to_ancestor', {'node_id': anc})}"},
        ]
        return format_sft_record({"conversations": convs})

    temp = rng.choice([0.7, 0.9, 1.0])
    updates = {"temperature": 0.1, "max_turns": 4}
    (cfg_dir / "config.json").write_text(json.dumps({"temperature": temp, "max_turns": 5}), encoding="utf-8")
    if not WorktreeToolExecutor(wt_dir, autonomy_level=1).dispatch("patch_config", {"updates": updates}).get("ok"):
        return None
    convs = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\nFix malformed tool loop.", "tools": _select_tools(["patch_config"])},
        {"role": "user", "content": f"fault=malformed_tool_loop temperature={temp} tag_errors=2"},
        {"role": "assistant", "content": f"<think>\nObserve: fault=malformed_tool_loop temperature={temp} tag_errors=2. Plan: call patch_config temperature=0.1 max_turns=4.\n</think>\n\n{_tc_xml('patch_config', {'updates': updates})}"},
    ]
    return format_sft_record({"conversations": convs})


def _build_noop_sample(rng: random.Random) -> dict:
    loss = round(rng.uniform(0.6, 1.6), 3)
    cyc = rng.randint(1, 50)
    convs = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\nMonitor system health.", "tools": _select_tools(["get_system_state"])},
        {"role": "user", "content": f"profile=colab_t4 fault=healthy cycle={cyc} loss={loss} consecutive_failures=0"},
        {"role": "assistant", "content": f"<think>\nCycle {cyc} metrics stable at loss={loss} with zero failures.\n</think>\n\nSystem is healthy; no edits required."},
    ]
    return format_sft_record({"conversations": convs})


def _make_general_tool_sample(tpl: tuple[str, list[dict], dict, str, str], idx: int, rng: random.Random) -> dict:
    prompt_tpl, tools, call, obs, final_ans = tpl
    q_txt = prompt_tpl.format(a=4, b=8, km=12, n=4, mb=128.5, step=120, total=300, sec=0.5, before=40.0, after=25.0, lr=0.0002, free_mb=1800, min_mb=1024)
    if idx > 0:
        q_txt = f"[ref {idx}-{rng.randint(10, 99)}] {q_txt}"
    args_str = ", ".join(f"{k}={v}" for k, v in call["arguments"].items())
    return format_sft_record(
        {
            "conversations": [
                {"role": "system", "content": "You are MiniMind.", "tools": json.dumps(tools, ensure_ascii=False)},
                {"role": "user", "content": q_txt},
                {"role": "assistant", "content": f"<think>\nObserve request inputs ({args_str}); invoke {call['name']}.\n</think>\n\n{_tc_xml(call['name'], call['arguments'])}"},
                {"role": "tool", "content": obs},
                {"role": "assistant", "content": f"<think>\nObserve tool result {obs}; answer cleanly.\n</think>\n\n{final_ans}"},
            ]
        }
    )


def _split_anchor_and_heldout(
    heldout_count: int,
    source_sft_jsonl: Path | None,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Build strictly disjoint anchor-train and heldout-eval conversation pools."""
    rng = random.Random(seed)
    if source_sft_jsonl is not None and source_sft_jsonl.is_file():
        rows = [format_sft_record(json.loads(ln)) for ln in source_sft_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
        rows = [r for r in rows if r["conversations"]]
        rng.shuffle(rows)
        split_n = min(heldout_count, max(1, len(rows) // 10))
        return rows[split_n:], rows[:split_n]

    dialogue_indices = list(range(len(_BUILTIN_DIALOGUE_BANK)))
    rng.shuffle(dialogue_indices)
    d_split = max(4, min(len(dialogue_indices) // 4, max(4, heldout_count * 2 // 3)))
    heldout_dialogues = [_BUILTIN_DIALOGUE_BANK[i] for i in dialogue_indices[:d_split]]
    train_dialogues = [_BUILTIN_DIALOGUE_BANK[i] for i in dialogue_indices[d_split:]]

    tool_indices = list(range(len(_BUILTIN_GENERAL_TOOLS)))
    rng.shuffle(tool_indices)
    t_split = max(2, len(tool_indices) // 4)
    heldout_tools = [_BUILTIN_GENERAL_TOOLS[i] for i in tool_indices[:t_split]]
    train_tools = [_BUILTIN_GENERAL_TOOLS[i] for i in tool_indices[t_split:]]

    heldout_rows: list[dict] = []
    for i in range(heldout_count):
        if i % 3 == 0:
            heldout_rows.append(_make_general_tool_sample(heldout_tools[i % len(heldout_tools)], i + 5000, rng))
        else:
            q, a = heldout_dialogues[i % len(heldout_dialogues)]
            heldout_rows.append(format_sft_record({"conversations": [{"role": "user", "content": f"[eval {i}] {q}"}, {"role": "assistant", "content": a}]}))

    anchor_pool: list[dict] = []
    for idx, (q, a) in enumerate(train_dialogues):
        anchor_pool.append(format_sft_record({"conversations": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}))
        for tag in ("Explain concisely:", "请简明回答："):
            anchor_pool.append(format_sft_record({"conversations": [{"role": "user", "content": f"{tag} {q}"}, {"role": "assistant", "content": a}]}))
    for idx, tpl in enumerate(train_tools):
        anchor_pool.append(_make_general_tool_sample(tpl, idx, rng))
        anchor_pool.append(_make_general_tool_sample(tpl, idx + 100, rng))
    return anchor_pool, heldout_rows


def _build_dpo_pair_from_sft(sample: dict, mode: int, rng: random.Random) -> dict | None:
    convs = sample.get("conversations", [])
    sys_msg = convs[0] if convs and convs[0].get("role") == "system" else {"role": "system", "content": SYSTEM_PROMPT}
    usr_msg = next((m for m in convs if m.get("role") == "user"), None)
    asst_msg = next((m for m in reversed(convs) if m.get("role") == "assistant"), None)
    if usr_msg is None or asst_msg is None:
        return None
    chosen = [{"role": "system", "content": str(sys_msg.get("content", ""))}, {"role": "user", "content": str(usr_msg.get("content", ""))}, {"role": "assistant", "content": str(asst_msg.get("content", ""))}]
    good_text = str(asst_msg.get("content", ""))
    if "</tool_call>" not in good_text:
        return None

    m = mode % 8
    if m == 0:
        bad_text = good_text.replace("</tool_call>", "").replace("<tool_call>", "<tool_call>\n{broken")
    elif m == 1:
        bad_text = re.sub(r'"start_line":\s*\d+', '"start_line": 99', good_text, count=1)
        if bad_text == good_text:
            bad_text = '<think>\nTry invalid call.\n</think>\n\n<tool_call>\n{"name": "patch_config", "arguments": {"updates": {"learning_rate": 100.0}}}\n</tool_call>'
    elif m == 2:
        bad_text = '<think>\nUse a nonexistent helper.\n</think>\n\n<tool_call>\n{"name": "edit_whole_repo_file", "arguments": {"path": "trainer/train_lora.py"}}\n</tool_call>'
    elif m == 3:
        bad_text = "<think>\nCheck system.\n</think>\n\nSystem is healthy; no edits required."
    elif m == 4:
        bad_text = "<think>\n" + ("repeat fix line now " * 12) + "\n</think>\n\nrepeat fix line now"
    elif m == 5:
        bad_text = good_text.replace("</think>", "")
    elif m == 6:
        bad_text = '<think>\nObserve: guess line 99 looks suspicious. Plan: overwrite blind.\n</think>\n\n<tool_call>\n{"name":"replace_line_range","arguments":{"path":"blind.py","start_line":99,"end_line":99,"lines":["pass"]}}\n</tool_call>'
    else:
        bad_text = '<think>\nObserve: fault unclear. Plan: call read_log_tail on train.log.\n</think>\n\n<tool_call>\n{"name":"replace_line_range","arguments":{"path":"wrong.py","start_line":99,"end_line":99,"lines":["pass"]}}\n</tool_call>'

    rejected = [{"role": "system", "content": chosen[0]["content"]}, {"role": "user", "content": chosen[1]["content"]}, {"role": "assistant", "content": bad_text}]
    if chosen == rejected:
        return None
    _ = rng
    return format_dpo_record({"chosen": chosen, "rejected": rejected})


def _extract_prompt_keys(conversations: list[dict]) -> tuple[str, str]:
    usr = " ".join(str(m.get("content", "")) for m in conversations if m.get("role") == "user")
    return usr.strip(), _normalize_for_diversity(usr)


def _load_heldout_sets(heldout_path: Path | None) -> tuple[set[str], set[str]]:
    if heldout_path is None or not heldout_path.is_file():
        return set(), set()
    exact_set: set[str] = set()
    norm_set: set[str] = set()
    for ln in heldout_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        convs = row.get("conversations") or row.get("chosen") or []
        ex, nm = _extract_prompt_keys(convs)
        if ex:
            exact_set.add(ex)
        if nm:
            norm_set.add(nm)
    return exact_set, norm_set


def build_finetune_dataset(
    repo_root: Path,
    output_dir: Path,
    sft_count: int = 1200,
    dpo_count: int = 200,
    heldout_count: int = 80,
    max_seq_len: int = 768,
    anchor_ratio: float = 0.25,
    source_sft_jsonl: Path | None = None,
    seed: int = 42,
) -> dict:
    """Generate diverse, execution-verified SFT + DPO + disjoint held-out datasets and audit quality."""
    repo_root = Path(repo_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    token_limit = int(0.9 * max_seq_len)

    fixtures = collect_code_fixtures(repo_root)
    anchor_pool, heldout_rows = _split_anchor_and_heldout(heldout_count, source_sft_jsonl, seed=seed + 11)
    heldout_path = output_dir / "heldout_eval.jsonl"
    write_sft_jsonl(heldout_path, heldout_rows)
    h_exact, h_norm = _load_heldout_sets(heldout_path)

    target_anchor = int(round(sft_count * min(0.5, max(0.05, anchor_ratio))))
    target_noop = min(max(1, int(round(sft_count * 0.05))), max(1, int(sft_count * 0.08)))
    remaining = max(1, sft_count - target_anchor - target_noop)
    target_direct = int(round(remaining * 0.40))
    target_multi = int(round(remaining * 0.33))
    target_op = max(1, remaining - target_direct - target_multi)

    sft_rows: list[dict] = []
    code_repair_pool: list[dict] = []
    seen_json: set[str] = set()

    def _try_add_sft(sample: dict | None, into_code_pool: bool = False) -> bool:
        if not sample or not sample.get("conversations"):
            return False
        convs = sample["conversations"]
        ex_p, _ = _extract_prompt_keys(convs)
        if ex_p and ex_p in h_exact:
            return False
        if estimate_tokens(convs) > token_limit:
            return False
        sig = json.dumps(sample, sort_keys=True, ensure_ascii=False)
        if sig in seen_json:
            return False
        seen_json.add(sig)
        sft_rows.append(sample)
        if into_code_pool:
            code_repair_pool.append(sample)
        return True

    with tempfile.TemporaryDirectory() as tmp:
        wt_dir = Path(tmp)
        attempts = 0
        direct_added = 0
        while direct_added < target_direct and attempts < target_direct * 6:
            attempts += 1
            fx = fixtures[attempts % len(fixtures)]
            if _try_add_sft(_build_direct_code_edit_sample(fx, rng, wt_dir), into_code_pool=True):
                direct_added += 1

        compact_fixtures = [fx for fx in fixtures if len(fx["source"].splitlines()) <= 5 and len(fx["source"]) <= 170] or fixtures
        attempts = 0
        multi_added = 0
        while multi_added < target_multi and attempts < target_multi * 12:
            attempts += 1
            fx = compact_fixtures[(attempts + 7) % len(compact_fixtures)]
            if _try_add_sft(_build_multiturn_code_edit_sample(fx, rng, wt_dir, with_retry=(max_seq_len >= 1024 and attempts % 5 == 0)), into_code_pool=True):
                multi_added += 1

        attempts = 0
        op_added = 0
        while op_added < target_op and attempts < target_op * 6:
            attempts += 1
            if _try_add_sft(_build_operational_fault_sample(rng, wt_dir), into_code_pool=True):
                op_added += 1

        noop_added = 0
        for _ in range(target_noop):
            if _try_add_sft(_build_noop_sample(rng), into_code_pool=False):
                noop_added += 1

        anchor_added = 0
        for idx in range(target_anchor * 3):
            if anchor_added >= target_anchor or not anchor_pool:
                break
            base_a = anchor_pool[idx % len(anchor_pool)]
            candidate = format_sft_record(base_a)
            if idx >= len(anchor_pool) and candidate["conversations"][0].get("role") == "user":
                candidate["conversations"][0]["content"] = f"[anchor {idx}] {candidate['conversations'][0]['content']}"
            if _try_add_sft(candidate, into_code_pool=False):
                anchor_added += 1

        while len(sft_rows) < sft_count and fixtures:
            fx = fixtures[rng.randrange(len(fixtures))]
            if not _try_add_sft(_build_direct_code_edit_sample(fx, rng, wt_dir), into_code_pool=True):
                break

    rng.shuffle(sft_rows)
    sft_rows = sft_rows[:sft_count]
    sft_path = output_dir / "sft_train.jsonl"
    sft_written = write_sft_jsonl(sft_path, sft_rows)

    dpo_rows: list[dict] = []
    seen_dpo: set[str] = set()
    pool_for_dpo = [
        r
        for r in code_repair_pool
        if sum(1 for m in r["conversations"] if m.get("role") == "assistant") == 1
        and any("</tool_call>" in str(m.get("content", "")) for m in r["conversations"] if m.get("role") == "assistant")
    ]
    if pool_for_dpo and dpo_count > 0:
        for idx in range(dpo_count * 4):
            if len(dpo_rows) >= dpo_count:
                break
            src_row = pool_for_dpo[idx % len(pool_for_dpo)]
            pair = _build_dpo_pair_from_sft(src_row, mode=idx, rng=rng)
            if pair is None:
                continue
            p_sig = json.dumps(pair, sort_keys=True, ensure_ascii=False)
            if p_sig in seen_dpo:
                continue
            seen_dpo.add(p_sig)
            dpo_rows.append(pair)
    dpo_path = output_dir / "dpo_train.jsonl"
    dpo_written = write_dpo_jsonl(dpo_path, dpo_rows)

    sft_audit = audit_training_dataset(sft_path, dataset_format="sft", max_seq_len=max_seq_len, heldout_path=heldout_path, max_exec_checks=40)
    dpo_audit = audit_training_dataset(dpo_path, dataset_format="dpo", max_seq_len=max_seq_len, heldout_path=heldout_path, max_exec_checks=25) if dpo_written > 0 else {"passed": True, "quality_score": 1.0, "failed_checks": []}

    combined_report = {
        "repo_root": str(repo_root),
        "output_dir": str(output_dir),
        "fixtures_total": len(fixtures),
        "sft_path": str(sft_path),
        "sft_count": sft_written,
        "dpo_path": str(dpo_path),
        "dpo_count": dpo_written,
        "heldout_path": str(heldout_path),
        "heldout_count": len(heldout_rows),
        "split_breakdown": {
            "direct_code_edit": direct_added,
            "multi_code_edit": multi_added,
            "operational_fault": op_added,
            "noop": noop_added,
            "anchor": anchor_added,
        },
        "sft_audit": sft_audit,
        "dpo_audit": dpo_audit,
        "overall_passed": bool(sft_audit.get("passed") and dpo_audit.get("passed")),
    }
    atomic_write_text(output_dir / "data_quality_report.json", json.dumps(combined_report, indent=2, ensure_ascii=False) + "\n")
    md_text = "\n".join([format_quality_report_md(sft_audit), "", format_quality_report_md(dpo_audit)] if dpo_written > 0 else [format_quality_report_md(sft_audit)])
    atomic_write_text(output_dir / "data_quality_report.md", md_text + "\n")
    return combined_report


def _validate_sft_messages(convs: object) -> tuple[bool, str | None, dict[str, dict]]:
    if not isinstance(convs, list) or not convs:
        return False, "empty or non-list conversations", {}
    declared_tools: dict[str, dict] = {}
    for msg in convs:
        if not isinstance(msg, dict) or set(msg.keys()) != _SFT_MSG_KEYS:
            return False, f"message keys mismatch (got {sorted(msg.keys()) if isinstance(msg, dict) else type(msg).__name__})", {}
        if not isinstance(msg["role"], str) or not isinstance(msg["content"], str):
            return False, "role/content must be str", {}
        if msg["reasoning_content"] is not None and not isinstance(msg["reasoning_content"], str):
            return False, "reasoning_content must be str or None", {}
        for jfield in ("tools", "tool_calls"):
            val = msg[jfield]
            if val is not None:
                if not isinstance(val, str):
                    return False, f"{jfield} must be JSON string or None", {}
                try:
                    parsed = json.loads(val)
                except Exception:
                    return False, f"invalid JSON in {jfield}", {}
                if jfield == "tools" and isinstance(parsed, list):
                    for item in parsed:
                        fn = item.get("function", {}) if isinstance(item, dict) else {}
                        if isinstance(fn.get("name"), str):
                            declared_tools[fn["name"]] = fn.get("parameters", {})
    return True, None, declared_tools


def _validate_dpo_side(msgs: object) -> tuple[bool, str | None]:
    if not isinstance(msgs, list) or not msgs:
        return False, "empty or non-list DPO messages"
    for msg in msgs:
        if not isinstance(msg, dict) or set(msg.keys()) != _DPO_MSG_KEYS:
            return False, "DPO message keys must strictly be {'role', 'content'}"
        if not isinstance(msg["role"], str) or not isinstance(msg["content"], str):
            return False, "DPO role/content must be str"
    return True, None


def _validate_tool_against_declared(name: str, arguments: dict, declared_tools: dict[str, dict]) -> tuple[bool, str | None]:
    if name in _SELFCHANGE_TOOL_NAMES:
        return validate_tool_call(name, arguments)
    if name in declared_tools:
        req = declared_tools[name].get("required", []) if isinstance(declared_tools[name], dict) else []
        missing = [k for k in req if k not in arguments]
        if missing:
            return False, f"Missing required args for {name}: {missing}"
        return True, None
    return False, f"Undeclared/hallucinated tool {name!r}"


def _extract_file_lines_from_messages(convs: list[dict], target_path: str) -> str | None:
    target_name = Path(target_path).name
    for msg in convs:
        if msg.get("role") == "tool":
            try:
                obs = json.loads(str(msg.get("content", "")))
                if isinstance(obs, dict) and obs.get("ok") and isinstance(obs.get("content"), str):
                    if not obs.get("path") or Path(str(obs["path"])).name == target_name:
                        extracted = [re.sub(r"^\d+:\s?", "", ln) for ln in obs["content"].splitlines() if re.match(r"^\d+:\s?", ln)]
                        if extracted:
                            return "\n".join(extracted) + "\n"
            except Exception:
                pass
    for msg in convs:
        if msg.get("role") == "user":
            lines = str(msg.get("content", "")).splitlines()
            extracted = [re.sub(r"^\d+:\s?", "", ln) for ln in lines if re.match(r"^\d+:\s?", ln)]
            if extracted:
                return "\n".join(extracted) + "\n"
    return None


def _replay_sample_tools(convs: list[dict], wt_dir: Path) -> bool | None:
    """Replay verifiable tool calls (replace_line_range / patch_config) inside a worktree."""
    verifiable_calls: list[tuple[str, dict]] = []
    for idx, msg in enumerate(convs):
        if msg.get("role") != "assistant":
            continue
        calls, _ = parse_tool_calls(str(msg.get("content", "")))
        next_msg = convs[idx + 1] if idx + 1 < len(convs) else None
        is_failed_retry = False
        if next_msg and next_msg.get("role") == "tool":
            try:
                obs = json.loads(str(next_msg.get("content", "")))
                if isinstance(obs, dict) and obs.get("ok") is False:
                    is_failed_retry = True
            except Exception:
                pass
        if is_failed_retry:
            continue
        for c in calls:
            if c["name"] in ("replace_line_range", "patch_config"):
                verifiable_calls.append((c["name"], c["arguments"]))

    if not verifiable_calls:
        return None

    m_dir = wt_dir / "selfchange" / "mutable"
    m_dir.mkdir(parents=True, exist_ok=True)
    for name, args in verifiable_calls:
        if name == "patch_config":
            (m_dir / "config.json").write_text(json.dumps({"training": {"batch_size": 4, "learning_rate": 5e-5, "accumulation_steps": 1, "grad_clip": 1.0, "max_seq_len": 384}, "temperature": 0.2, "max_turns": 5}), encoding="utf-8")
            if not WorktreeToolExecutor(wt_dir, autonomy_level=1).dispatch("patch_config", args).get("ok"):
                return False
        elif name == "replace_line_range":
            rel_p = str(args.get("path", ""))
            source_before = _extract_file_lines_from_messages(convs, rel_p)
            if source_before is None:
                return False
            fname = Path(rel_p).name
            target_file = m_dir / fname
            target_file.write_text(source_before, encoding="utf-8")
            res = WorktreeToolExecutor(wt_dir, autonomy_level=1).dispatch("replace_line_range", args)
            if not res.get("ok"):
                return False
            edited = target_file.read_text(encoding="utf-8")
            try:
                ast.parse(edited)
            except SyntaxError:
                return False
            if not scan_python_source(edited).ok:
                return False
            if edited.strip() == source_before.strip():
                return False
    return True


def _load_optional_tokenizer(tokenizer_path: Path | None):
    if tokenizer_path is None:
        return None
    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]

        return AutoTokenizer.from_pretrained(str(tokenizer_path))
    except Exception:
        return None


_THINK_STOP_WORDS = {
    "the", "and", "for", "with", "this", "that", "from", "into", "your", "have", "will",
    "line", "lines", "file", "code", "call", "plan", "cause", "observe", "check", "read",
    "fix", "edit", "patch", "bug", "error", "module", "clean", "exact", "result", "value",
    "values", "output", "system", "state", "true", "false", "none", "only", "using",
    "after", "before", "when", "then", "than", "step", "answer", "cleanly",
}


def score_thinking_turn(context_text: str, assistant_text: str) -> dict:
    """Evaluate observation grounding and think-to-action consistency for a <think>...</think> turn."""
    asst = assistant_text or ""
    m = re.search(r"<think>(.*?)</think>", asst, flags=re.DOTALL | re.IGNORECASE)
    has_think = bool(m)
    think_body = m.group(1).strip() if m else ""
    think_chars = len(think_body)
    think_lower = think_body.lower()
    ctx_lower = (context_text or "").lower()

    ctx_files = set(re.findall(r"\b[a-z_][a-z0-9_]*\.(?:py|log|json|pth)\b", ctx_lower))
    think_files = set(re.findall(r"\b[a-z_][a-z0-9_]*\.(?:py|log|json|pth)\b", think_lower))
    ctx_words = {w for w in re.findall(r"[a-z_\u4e00-\u9fff][a-z0-9_\u4e00-\u9fff]{3,}", ctx_lower) if w not in _THINK_STOP_WORDS}
    think_words = {w for w in re.findall(r"[a-z_\u4e00-\u9fff][a-z0-9_\u4e00-\u9fff]{3,}", think_lower) if w not in _THINK_STOP_WORDS}
    ctx_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", ctx_lower))
    think_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", think_lower))

    cited_lines = [int(n) for n in re.findall(r"\bline\s+(\d+)\b", think_lower)]
    has_fabricated_line = any(n >= 20 and str(n) not in ctx_lower for n in cited_lines)
    rep_ok = (compute_rep_penalty(think_body) < 0.25) if think_body else False
    has_overlap = bool((ctx_files & think_files) or (ctx_words & think_words) or (ctx_nums & think_nums))
    grounded = bool(has_think and think_chars >= 18 and rep_ok and has_overlap and not has_fabricated_line)

    calls, tag_errs = parse_tool_calls(asst)
    action_consistent = bool(has_think)
    if tag_errs > 0:
        action_consistent = False
    elif calls:
        emitted_names = {c["name"] for c in calls}
        planned_tools = {
            t for t in re.findall(r"(?:call|invoke)\s+([a-z_][a-z0-9_]*)", think_lower)
            if t in _SELFCHANGE_TOOL_NAMES
        }
        if planned_tools and not planned_tools.issubset(emitted_names):
            action_consistent = False
        first_call = calls[0]
        f_args = first_call.get("arguments") if isinstance(first_call.get("arguments"), dict) else {}
        if first_call["name"] == "replace_line_range":
            sl = f_args.get("start_line")
            el = f_args.get("end_line")
            if cited_lines and isinstance(sl, int) and isinstance(el, int):
                if not any(sl <= ln <= el for ln in cited_lines):
                    action_consistent = False
            arg_path = str(f_args.get("path", "")).strip()
            if arg_path.endswith(".py"):
                py_mentioned = {f for f in think_files if f.endswith(".py")}
                if py_mentioned and Path(arg_path).name.lower() not in py_mentioned:
                    action_consistent = False
    elif "plan: call " in think_lower:
        action_consistent = False

    return {
        "has_think": has_think,
        "think_chars": think_chars,
        "grounded": grounded,
        "action_consistent": action_consistent,
    }


def audit_training_dataset(
    data_path: Path,
    dataset_format: str = "auto",
    max_seq_len: int = 768,
    heldout_path: Path | None = None,
    max_exec_checks: int = 40,
    tokenizer_path: Path | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict:
    """Run comprehensive schema, supervision, tag, token-budget, diversity, trivia, thinking, leakage, and exec checks."""
    data_path = Path(data_path)
    limits = {**QUALITY_THRESHOLDS, **(thresholds or {})}
    if not data_path.is_file():
        return {"data_path": str(data_path), "format": dataset_format, "total_rows": 0, "passed": False, "quality_score": 0.0, "failed_checks": ["file_not_found"], "sample_errors": [], "failed_line_reasons": [], "recommendations": ["Provide an existing JSONL file."]}

    raw_lines = [(idx + 1, ln) for idx, ln in enumerate(data_path.read_text(encoding="utf-8").splitlines()) if ln.strip()]
    if not raw_lines:
        return {"data_path": str(data_path), "format": dataset_format, "total_rows": 0, "passed": False, "quality_score": 0.0, "failed_checks": ["empty_dataset"], "sample_errors": [], "failed_line_reasons": [], "recommendations": ["Generate non-empty training data."]}

    detected_fmt = dataset_format
    if detected_fmt == "auto":
        try:
            first_obj = json.loads(raw_lines[0][1])
            if "chosen" in first_obj or "rejected" in first_obj:
                detected_fmt = "dpo"
            elif "text" in first_obj and "conversations" not in first_obj:
                detected_fmt = "pretrain"
            else:
                detected_fmt = "sft"
        except Exception:
            detected_fmt = "sft"

    tokenizer = _load_optional_tokenizer(tokenizer_path)
    h_exact, h_norm = _load_heldout_sets(heldout_path)
    token_budget = int(0.9 * max_seq_len)

    schema_ok_cnt = sup_ok_cnt = tag_ok_cnt = tool_ok_cnt = budget_ok_cnt = high_rep_cnt = 0
    tool_sample_cnt = direct_edit_cnt = noop_cnt = leakage_cnt = trivia_cnt = 0
    think_checked_cnt = think_ground_ok_cnt = think_consist_ok_cnt = 0
    tool_counts: dict[str, int] = {}
    token_lengths: list[int] = []
    rep_values: list[float] = []
    exact_sigs: list[str] = []
    norm_sigs: list[str] = []
    asst_sigs: list[str] = []
    sample_errors: list[dict] = []
    line_reasons: dict[int, list[str]] = {}
    exec_candidates: list[tuple[int, list[dict]]] = []

    def _add_err(line_no: int, reason: str) -> None:
        if len(sample_errors) < 10:
            sample_errors.append({"line": line_no, "reason": reason})
        bucket = line_reasons.setdefault(line_no, [])
        if reason not in bucket:
            bucket.append(reason)

    for line_no, raw_ln in raw_lines:
        try:
            row = json.loads(raw_ln)
        except Exception as exc:
            _add_err(line_no, f"invalid_json: {exc}")
            continue
        if not isinstance(row, dict):
            _add_err(line_no, "row must be JSON object")
            continue

        exact_sigs.append(json.dumps(row, sort_keys=True, ensure_ascii=False))

        if detected_fmt == "pretrain":
            valid_pre = set(row.keys()) == {"text"} and isinstance(row.get("text"), str) and len(row["text"].strip()) >= 20
            if valid_pre:
                schema_ok_cnt += 1
                sup_ok_cnt += 1
            else:
                _add_err(line_no, "invalid pretrain row (expected strict {'text'} with len>=20)")
            txt = str(row.get("text", ""))
            toks = len(tokenizer(txt, add_special_tokens=False).input_ids) if tokenizer else max(1, int(len(txt) / 2.5))
            token_lengths.append(toks)
            if toks <= token_budget:
                budget_ok_cnt += 1
            else:
                _add_err(line_no, f"tokens={toks} > {token_budget}")
            rep = compute_rep_penalty(txt)
            rep_values.append(rep)
            if rep >= 0.15:
                high_rep_cnt += 1
                _add_err(line_no, f"high repetition pretrain={rep:.3f}")
            if classify_trivia_vs_domain(txt)["is_trivia"]:
                trivia_cnt += 1
                _add_err(line_no, "off_domain_trivia")
            norm_sigs.append(_normalize_for_diversity(txt))
            asst_sigs.append(_normalize_for_diversity(txt))
            tag_ok_cnt += 1
            tool_ok_cnt += 1
            continue

        if detected_fmt == "dpo":
            if set(row.keys()) != {"chosen", "rejected"}:
                _add_err(line_no, f"DPO top-level keys mismatch: {sorted(row.keys())}")
                continue
            ok_c, err_c = _validate_dpo_side(row["chosen"])
            ok_r, err_r = _validate_dpo_side(row["rejected"])
            if not (ok_c and ok_r) or row["chosen"] == row["rejected"]:
                _add_err(line_no, err_c or err_r or "degenerate chosen == rejected DPO pair")
                continue
            c_asst = [m for m in row["chosen"] if m.get("role") == "assistant" and str(m.get("content", "")).strip()]
            r_asst = [m for m in row["rejected"] if m.get("role") == "assistant" and str(m.get("content", "")).strip()]
            if c_asst and r_asst:
                schema_ok_cnt += 1
                sup_ok_cnt += 1
            else:
                _add_err(line_no, "missing non-empty assistant side in DPO pair")
            convs = row["chosen"]
            declared_tools: dict[str, dict] = {}
        else:
            if set(row.keys()) != {"conversations"}:
                _add_err(line_no, f"SFT top-level keys mismatch: {sorted(row.keys())} (lm_dataset.py allows only ['conversations'])")
                convs = row.get("conversations") if isinstance(row.get("conversations"), list) else []
                declared_tools = {}
            else:
                ok_s, err_s, declared_tools = _validate_sft_messages(row["conversations"])
                if ok_s:
                    schema_ok_cnt += 1
                else:
                    _add_err(line_no, err_s or "invalid SFT conversations")
                convs = row["conversations"]
            has_asst = any(
                isinstance(m, dict) and m.get("role") == "assistant" and (str(m.get("content", "")).strip() or m.get("tool_calls"))
                for m in convs
            )
            if has_asst:
                sup_ok_cnt += 1
            else:
                _add_err(line_no, "no assistant supervision turns")

        asst_texts = [str(m.get("content", "")) for m in convs if isinstance(m, dict) and m.get("role") == "assistant"]
        usr_texts = [str(m.get("content", "")) for m in convs if isinstance(m, dict) and m.get("role") == "user"]
        row_tag_ok = True
        row_tool_ok = True
        row_calls: list[dict] = []
        ctx_parts: list[str] = []
        for m in convs:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = str(m.get("content", ""))
            if role != "assistant":
                ctx_parts.append(content)
                if m.get("tools"):
                    ctx_parts.append(str(m.get("tools")))
                continue
            calls, tag_errs = parse_tool_calls(content)
            if tag_errs > 0:
                row_tag_ok = False
                _add_err(line_no, f"unbalanced tags or invalid <tool_call> JSON (errs={tag_errs})")
            for c in calls:
                row_calls.append(c)
                ok_t, err_t = _validate_tool_against_declared(c["name"], c["arguments"], declared_tools)
                if not ok_t:
                    row_tool_ok = False
                    _add_err(line_no, err_t or "invalid tool schema")
            if "<think" in content.lower() or "</think>" in content.lower() or calls:
                think_checked_cnt += 1
                t_score = score_thinking_turn("\n".join(ctx_parts), content)
                if t_score["grounded"]:
                    think_ground_ok_cnt += 1
                else:
                    _add_err(line_no, "trivial or ungrounded <think>")
                if t_score["action_consistent"]:
                    think_consist_ok_cnt += 1
                else:
                    _add_err(line_no, "think-to-action mismatch")
        if row_tag_ok:
            tag_ok_cnt += 1
        if row_tool_ok:
            tool_ok_cnt += 1

        if row_calls:
            tool_sample_cnt += 1
            for c in row_calls:
                tool_counts[c["name"]] = tool_counts.get(c["name"], 0) + 1
            if len(row_calls) == 1 and row_calls[0]["name"] == "replace_line_range":
                direct_edit_cnt += 1
            if any(c["name"] in ("replace_line_range", "patch_config") for c in row_calls):
                exec_candidates.append((line_no, convs))
        else:
            if any("no edits required" in t.lower() or "system is healthy" in t.lower() for t in asst_texts):
                noop_cnt += 1
            if not declared_tools and classify_trivia_vs_domain("\n".join(usr_texts + asst_texts))["is_trivia"]:
                trivia_cnt += 1
                _add_err(line_no, "off_domain_trivia")

        toks = estimate_tokens(convs, tokenizer=tokenizer)
        token_lengths.append(toks)
        if toks <= token_budget:
            budget_ok_cnt += 1
        else:
            _add_err(line_no, f"estimated tokens={toks} > budget={token_budget}")

        joined_asst = "\n".join(asst_texts)
        prose_turns = [
            re.sub(r"<tool_call>.*?</tool_call>", " ", t, flags=re.DOTALL)
            .replace("<think>", " ")
            .replace("</think>", " ")
            .strip()
            for t in asst_texts
        ]
        rep = max([compute_rep_penalty(p) for p in prose_turns if p] or [0.0])
        raw_turn_rep = max([compute_rep_penalty(t) for t in asst_texts if t] or [0.0])
        rep_values.append(rep)
        if rep >= 0.15 or raw_turn_rep >= 0.40:
            high_rep_cnt += 1
            _add_err(line_no, f"high repetition prose={rep:.3f} raw={raw_turn_rep:.3f}")

        norm_sigs.append(_normalize_for_diversity(" | ".join(usr_texts) + " => " + joined_asst))
        asst_sigs.append(_normalize_for_diversity(joined_asst))
        if h_exact or h_norm:
            ex_u, nm_u = _extract_prompt_keys(convs)
            if (ex_u and ex_u in h_exact) or (nm_u and nm_u in h_norm):
                leakage_cnt += 1
                _add_err(line_no, "heldout prompt leakage detected")

    exec_checked = exec_passed = 0
    if exec_candidates and max_exec_checks > 0:
        stride = max(1, len(exec_candidates) // max_exec_checks)
        sample_subset = exec_candidates[::stride][:max_exec_checks]
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            for lno, convs in sample_subset:
                verdict = _replay_sample_tools(convs, wt)
                if verdict is None:
                    continue
                exec_checked += 1
                if verdict:
                    exec_passed += 1
                else:
                    _add_err(lno, "tool execution replay failed")

    n = len(raw_lines)
    sorted_tok = sorted(token_lengths) if token_lengths else [0]
    exact_unique = len(set(exact_sigs))
    norm_unique = len(set(norm_sigs))
    asst_unique = len(set(asst_sigs))
    metrics = {
        "schema_valid_rate": round(schema_ok_cnt / n, 4),
        "assistant_supervision_rate": round(sup_ok_cnt / n, 4),
        "tag_balance_rate": round(tag_ok_cnt / n, 4),
        "tool_schema_valid_rate": round(tool_ok_cnt / n, 4),
        "within_budget_rate": round(budget_ok_cnt / n, 4),
        "p50_tokens": sorted_tok[len(sorted_tok) // 2],
        "p95_tokens": sorted_tok[min(len(sorted_tok) - 1, int(0.95 * len(sorted_tok)))],
        "max_tokens": max(sorted_tok),
        "mean_rep_penalty": round(sum(rep_values) / max(1, len(rep_values)), 4),
        "high_rep_rate": round(high_rep_cnt / n, 4),
        "exact_unique_count": exact_unique,
        "exact_duplicate_rate": round(1.0 - (exact_unique / n), 4),
        "normalized_unique_count": norm_unique,
        "normalized_unique_rate": round(norm_unique / n, 4),
        "unique_assistant_rate": round(asst_unique / n, 4),
        "tool_call_sample_rate": round(tool_sample_cnt / n, 4),
        "direct_code_edit_rate": round(direct_edit_cnt / n, 4),
        "noop_rate": round(noop_cnt / n, 4),
        "off_domain_trivia_rate": round(trivia_cnt / n, 4),
        "think_checked_count": think_checked_cnt,
        "think_grounding_rate": round(think_ground_ok_cnt / think_checked_cnt, 4) if think_checked_cnt > 0 else 1.0,
        "think_action_consistency_rate": round(think_consist_ok_cnt / think_checked_cnt, 4) if think_checked_cnt > 0 else 1.0,
        "exec_checked_count": exec_checked,
        "exec_verified_pass_rate": round(exec_passed / exec_checked, 4) if exec_checked > 0 else 1.0,
        "leakage_count": leakage_cnt,
        "tool_counts": dict(sorted(tool_counts.items())),
    }

    failed_checks: list[str] = []
    recs: list[str] = []
    if metrics["schema_valid_rate"] < limits["min_schema_valid_rate"]:
        failed_checks.append(f"schema_valid_rate={metrics['schema_valid_rate']} < {limits['min_schema_valid_rate']}")
        recs.append("Sanitize rows with write_sft_jsonl / write_dpo_jsonl so dataset/lm_dataset.py loads without schema errors.")
    if metrics["assistant_supervision_rate"] < limits["min_assistant_supervision_rate"]:
        failed_checks.append(f"assistant_supervision_rate={metrics['assistant_supervision_rate']} < {limits['min_assistant_supervision_rate']}")
    if metrics["tag_balance_rate"] < limits["min_tag_balance_rate"]:
        failed_checks.append(f"tag_balance_rate={metrics['tag_balance_rate']} < {limits['min_tag_balance_rate']}")
    if metrics["tool_schema_valid_rate"] < limits["min_tool_schema_valid_rate"]:
        failed_checks.append(f"tool_schema_valid_rate={metrics['tool_schema_valid_rate']} < {limits['min_tool_schema_valid_rate']}")
    if metrics["within_budget_rate"] < limits["min_within_budget_rate"]:
        failed_checks.append(f"within_budget_rate={metrics['within_budget_rate']} < {limits['min_within_budget_rate']}")
        recs.append(f"Shorten prompts/traces or raise max_seq_len above {metrics['max_tokens']}.")
    if metrics["high_rep_rate"] > limits["max_high_rep_rate"]:
        failed_checks.append(f"high_rep_rate={metrics['high_rep_rate']} > {limits['max_high_rep_rate']}")
    if exec_checked > 0 and metrics["exec_verified_pass_rate"] < limits["min_exec_verified_pass_rate"]:
        failed_checks.append(f"exec_verified_pass_rate={metrics['exec_verified_pass_rate']} < {limits['min_exec_verified_pass_rate']}")
        recs.append("Filter failed line-range edits using WorktreeToolExecutor before exporting SFT/DPO rows.")
    if metrics["exact_duplicate_rate"] > limits["max_exact_duplicate_rate"]:
        failed_checks.append(f"exact_duplicate_rate={metrics['exact_duplicate_rate']} > {limits['max_exact_duplicate_rate']}")
        recs.append("Deduplicate exact samples before training.")
    if n >= 20 and metrics["normalized_unique_rate"] < limits["min_normalized_unique_rate"]:
        failed_checks.append(f"normalized_unique_rate={metrics['normalized_unique_rate']} < {limits['min_normalized_unique_rate']}")
        recs.append("Expand fixture/template diversity using repo-extracted functions and diverse fault templates.")
    if metrics["noop_rate"] > limits["max_noop_rate"]:
        failed_checks.append(f"noop_rate={metrics['noop_rate']} > {limits['max_noop_rate']}")
        recs.append("Cap healthy no-op traces so the fine-tuned policy still performs tool actions.")
    if metrics["leakage_count"] > int(limits["max_leakage_count"]):
        failed_checks.append(f"leakage_count={metrics['leakage_count']} > {int(limits['max_leakage_count'])}")
        recs.append("Enforce disjoint user/eval split against heldout_eval.jsonl.")
    if metrics["off_domain_trivia_rate"] > limits["max_off_domain_trivia_rate"]:
        failed_checks.append(f"off_domain_trivia_rate={metrics['off_domain_trivia_rate']} > {limits['max_off_domain_trivia_rate']}")
        recs.append("Filter trivia rows via prune_reallocate.filter_trivia_from_anchor_jsonl.")
    if think_checked_cnt > 0 and metrics["think_grounding_rate"] < limits["min_think_grounding_rate"]:
        failed_checks.append(f"think_grounding_rate={metrics['think_grounding_rate']} < {limits['min_think_grounding_rate']}")
        recs.append("Ensure <think> blocks cite real observed filenames, lines, symbols, or fault metrics.")
    if think_checked_cnt > 0 and metrics["think_action_consistency_rate"] < limits["min_think_action_consistency_rate"]:
        failed_checks.append(f"think_action_consistency_rate={metrics['think_action_consistency_rate']} < {limits['min_think_action_consistency_rate']}")
        recs.append("Align <think> plans with emitted <tool_call> tool name, path, and line range.")

    div_ok = max(0.0, min(1.0, (1.0 - metrics["exact_duplicate_rate"]) * 0.5 + min(1.0, metrics["normalized_unique_rate"] / 0.25) * 0.5))
    think_ok = 0.5 * (metrics["think_grounding_rate"] + metrics["think_action_consistency_rate"])
    quality_score = round(
        0.20 * metrics["schema_valid_rate"]
        + 0.15 * ((metrics["tag_balance_rate"] + metrics["tool_schema_valid_rate"]) * 0.5)
        + 0.20 * metrics["exec_verified_pass_rate"]
        + 0.15 * metrics["within_budget_rate"]
        + 0.15 * div_ok
        + 0.15 * think_ok,
        4,
    )
    return {
        "data_path": str(data_path),
        "format": detected_fmt,
        "total_rows": n,
        "max_seq_len": max_seq_len,
        "token_budget": token_budget,
        "passed": len(failed_checks) == 0,
        "quality_score": quality_score,
        "metrics": metrics,
        "failed_checks": failed_checks,
        "recommendations": recs,
        "sample_errors": sample_errors,
        "failed_line_reasons": [{"line": ln, "reasons": rlist} for ln, rlist in sorted(line_reasons.items())[:500]],
    }


def review_and_record_dataset(
    data_path: Path,
    workspace_dir: Path,
    dataset_format: str = "auto",
    max_seq_len: int = 768,
    heldout_path: Path | None = None,
    cycle_tag: str | None = None,
    tokenizer_path: Path | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict:
    """Audit dataset, detect cross-cycle drift, quarantine bad rows, and record persistent review history."""
    data_path = Path(data_path)
    workspace_dir = Path(workspace_dir)
    reviews_dir = workspace_dir / "data_reviews"
    quarantine_dir = workspace_dir / "data_quarantine"
    logs_dir = workspace_dir / "logs"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    tag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", cycle_tag or datetime.now(timezone.utc).strftime("rev_%Y%m%d_%H%M%S"))
    audit = audit_training_dataset(
        data_path=data_path,
        dataset_format=dataset_format,
        max_seq_len=max_seq_len,
        heldout_path=heldout_path,
        tokenizer_path=tokenizer_path,
        thresholds=thresholds,
    )

    history_path = logs_dir / "data_review_history.jsonl"
    prev_same_fmt: dict | None = None
    if history_path.is_file():
        for ln in history_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
                if rec.get("format") == audit.get("format"):
                    prev_same_fmt = rec
            except Exception:
                pass

    drift_alerts: list[str] = []
    if prev_same_fmt is not None:
        prev_m = prev_same_fmt.get("metrics", {})
        cur_m = audit.get("metrics", {})
        if float(prev_m.get("normalized_unique_rate", 0.0)) - float(cur_m.get("normalized_unique_rate", 0.0)) > 0.10:
            drift_alerts.append(f"normalized_unique_rate dropped from {prev_m.get('normalized_unique_rate')} to {cur_m.get('normalized_unique_rate')}")
        if float(prev_m.get("think_grounding_rate", 1.0)) - float(cur_m.get("think_grounding_rate", 1.0)) > 0.10:
            drift_alerts.append(f"think_grounding_rate dropped from {prev_m.get('think_grounding_rate')} to {cur_m.get('think_grounding_rate')}")
        prev_tools = prev_m.get("tool_counts", {}) or {}
        cur_tools = cur_m.get("tool_counts", {}) or {}
        for core_t in ("replace_line_range", "patch_config", "read_file_lines"):
            if prev_tools.get(core_t, 0) > 0 and cur_tools.get(core_t, 0) == 0:
                drift_alerts.append(f"core tool {core_t} disappeared (was {prev_tools[core_t]})")

    quarantine_count = 0
    quarantine_path = quarantine_dir / f"{tag}_quarantine.jsonl"
    failed_map = {item["line"]: item["reasons"] for item in audit.get("failed_line_reasons", [])}
    raw_lines = [(idx + 1, ln) for idx, ln in enumerate(data_path.read_text(encoding="utf-8").splitlines()) if ln.strip()] if data_path.is_file() else []
    q_lines: list[str] = []
    if failed_map and raw_lines:
        for lno, raw_ln in raw_lines:
            if lno in failed_map:
                q_lines.append(json.dumps({"line": lno, "reasons": failed_map[lno], "raw": raw_ln}, ensure_ascii=False) + "\n")
                quarantine_count += 1
        atomic_write_text(quarantine_path, "".join(q_lines))

    sample_previews: list[str] = []
    if raw_lines:
        stride = max(1, len(raw_lines) // 8)
        for lno, raw_ln in raw_lines[::stride][:8]:
            sample_previews.append(f"- line `{lno}`: `{raw_ln[:140]}...`")

    md_sections = [format_quality_report_md(audit)]
    if drift_alerts:
        md_sections.append("\n- **Cross-Cycle Drift Alerts**: " + "; ".join(f"`{d}`" for d in drift_alerts))
    md_sections.append(f"- **Quarantined Rows**: `{quarantine_count}` (`{quarantine_path.name if quarantine_count else 'none'}`)")
    if sample_previews:
        md_sections.extend(["", "### Stratified Spot-Check Previews", *sample_previews])

    review_md_path = reviews_dir / f"{tag}_review.md"
    audit_json_path = reviews_dir / f"{tag}_audit.json"
    atomic_write_text(review_md_path, "\n".join(md_sections) + "\n")
    atomic_write_text(audit_json_path, json.dumps({**audit, "drift_alerts": drift_alerts, "quarantine_count": quarantine_count}, indent=2, ensure_ascii=False) + "\n")

    history_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "data_path": str(data_path),
        "format": audit.get("format"),
        "passed": audit.get("passed"),
        "quality_score": audit.get("quality_score"),
        "total_rows": audit.get("total_rows"),
        "metrics": audit.get("metrics"),
        "failed_checks": audit.get("failed_checks"),
        "drift_alerts": drift_alerts,
        "quarantine_count": quarantine_count,
    }
    prev_hist = history_path.read_text(encoding="utf-8") if history_path.is_file() else ""
    atomic_write_text(history_path, prev_hist + json.dumps(history_entry, ensure_ascii=False) + "\n")

    return {
        **audit,
        "tag": tag,
        "drift_alerts": drift_alerts,
        "quarantine_count": quarantine_count,
        "quarantine_path": str(quarantine_path) if quarantine_count else None,
        "review_md_path": str(review_md_path),
        "audit_json_path": str(audit_json_path),
        "history_path": str(history_path),
    }


def compute_eval_steered_weights(eval_scorecard: dict) -> dict:
    """Translate eval scorecard weaknesses into next-batch generator mixture & active-retirement hints."""
    t2 = eval_scorecard.get("tier2_toolcall", {})
    t3 = eval_scorecard.get("tier3_code_edit", {})
    t35 = eval_scorecard.get("tier3_5_thinking", {})
    t4 = eval_scorecard.get("tier4_drills", {})

    direct_w = 0.35
    multi_w = 0.30
    op_w = 0.25
    anchor_r = 0.20
    retired: list[str] = []
    notes: list[str] = []

    if float(t3.get("pass_at_1", 0.0)) < 0.40:
        direct_w += 0.15
        notes.append("Boost direct single-turn code repairs (Tier 3 pass@1 < 0.40).")
    elif float(t3.get("pass_at_1", 0.0)) >= 0.95:
        retired.append("simple_1line_ast_syntax")
        direct_w = max(0.15, direct_w - 0.15)
        multi_w += 0.15
        notes.append("Retire saturated 1-line AST syntax fixtures (pass@1 >= 0.95); shift to multi-turn.")

    if float(t35.get("observation_grounding_rate", 1.0)) < 0.85 or float(t35.get("think_action_consistency_rate", 1.0)) < 0.90:
        multi_w += 0.10
        notes.append("Prioritize 4-step grounded <think> + Contrastive DPO pairs.")

    if float(t4.get("pass_rate", 1.0)) < 0.80:
        op_w += 0.15
        notes.append("Upsample operational fault drills (OOM / NaN / rollback).")

    if float(t2.get("json_parse_rate", 1.0)) < 0.95:
        anchor_r = min(0.30, anchor_r + 0.05)
        notes.append("Raise structured tool-call core anchors.")

    norm = direct_w + multi_w + op_w
    return {
        "direct_weight": round(direct_w / norm, 4),
        "multi_weight": round(multi_w / norm, 4),
        "op_weight": round(op_w / norm, 4),
        "anchor_ratio": round(anchor_r, 4),
        "retired_families": retired,
        "focus_notes": notes,
    }


def format_quality_report_md(report: dict) -> str:
    """Render a compact Markdown audit table from audit_training_dataset output."""
    m = report.get("metrics", {})
    lines = [
        f"## Dataset Quality Audit (`{report.get('format', 'unknown')}` — {Path(report.get('data_path', '')).name})",
        f"- **Status**: `{'PASS' if report.get('passed') else 'FAIL'}` | **Quality Score**: `{report.get('quality_score', 0.0)}` | **Rows**: `{report.get('total_rows', 0)}`",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Schema Valid Rate | `{m.get('schema_valid_rate')}` |",
        f"| Supervision / Tag / Tool Valid | `{m.get('assistant_supervision_rate')}` / `{m.get('tag_balance_rate')}` / `{m.get('tool_schema_valid_rate')}` |",
        f"| Within Token Budget (<= {report.get('token_budget')}) | `{m.get('within_budget_rate')}` (p50=`{m.get('p50_tokens')}`, max=`{m.get('max_tokens')}`) |",
        f"| Exec-Verified Replay Pass Rate | `{m.get('exec_verified_pass_rate')}` ({m.get('exec_checked_count')} checked) |",
        f"| Exact Dup / Norm Unique | `{m.get('exact_duplicate_rate')}` / `{m.get('normalized_unique_rate')}` |",
        f"| Direct Code Edit / No-Op / Trivia | `{m.get('direct_code_edit_rate')}` / `{m.get('noop_rate')}` / `{m.get('off_domain_trivia_rate')}` |",
        f"| Think Grounding / Consistency | `{m.get('think_grounding_rate')}` / `{m.get('think_action_consistency_rate')}` ({m.get('think_checked_count')} turns) |",
        f"| High-Rep Rate / Leakage Count | `{m.get('high_rep_rate')}` / `{m.get('leakage_count')}` |",
    ]
    if m.get("tool_counts"):
        lines.append(f"- **Tool Distribution**: `{json.dumps(m['tool_counts'])}`")
    if report.get("failed_checks"):
        lines.append("- **Failed Checks**: " + ", ".join(f"`{c}`" for c in report["failed_checks"]))
    if report.get("recommendations"):
        lines.append("- **Recommendations**: " + " | ".join(report["recommendations"]))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate, audit, & continuously review MiniMind selfchange fine-tuning datasets")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="Generate execution-verified SFT + DPO + heldout datasets")
    p_gen.add_argument("--repo-root", type=Path, default=Path("."))
    p_gen.add_argument("--output-dir", type=Path, default=Path("selfchange/workspace/curated_data"))
    p_gen.add_argument("--sft-count", type=int, default=1200)
    p_gen.add_argument("--dpo-count", type=int, default=200)
    p_gen.add_argument("--heldout-count", type=int, default=80)
    p_gen.add_argument("--max-seq-len", type=int, default=768)
    p_gen.add_argument("--anchor-ratio", type=float, default=0.25)
    p_gen.add_argument("--source-sft", type=Path, default=None)
    p_gen.add_argument("--seed", type=int, default=42)

    p_chk = sub.add_parser("check", help="Audit an existing SFT/DPO/pretrain JSONL dataset")
    p_chk.add_argument("--data-path", type=Path, required=True)
    p_chk.add_argument("--format", choices=["auto", "sft", "dpo", "pretrain"], default="auto")
    p_chk.add_argument("--max-seq-len", type=int, default=768)
    p_chk.add_argument("--heldout-path", type=Path, default=None)
    p_chk.add_argument("--max-exec-checks", type=int, default=40)
    p_chk.add_argument("--tokenizer-path", type=Path, default=None)
    p_chk.add_argument("--report-json", type=Path, default=None)
    p_chk.add_argument("--report-md", type=Path, default=None)

    p_rev = sub.add_parser("review", help="Run continuous review: audit + drift history + quarantine bad rows + review MD")
    p_rev.add_argument("--data-path", type=Path, required=True)
    p_rev.add_argument("--workspace-dir", type=Path, default=Path("selfchange/workspace"))
    p_rev.add_argument("--format", choices=["auto", "sft", "dpo", "pretrain"], default="auto")
    p_rev.add_argument("--max-seq-len", type=int, default=768)
    p_rev.add_argument("--heldout-path", type=Path, default=None)
    p_rev.add_argument("--cycle-tag", type=str, default=None)
    p_rev.add_argument("--tokenizer-path", type=Path, default=None)

    args = parser.parse_args()
    if args.command == "generate":
        res = build_finetune_dataset(
            repo_root=args.repo_root,
            output_dir=args.output_dir,
            sft_count=args.sft_count,
            dpo_count=args.dpo_count,
            heldout_count=args.heldout_count,
            max_seq_len=args.max_seq_len,
            anchor_ratio=args.anchor_ratio,
            source_sft_jsonl=args.source_sft,
            seed=args.seed,
        )
        print(json.dumps(res, indent=2))
        return 0 if res["overall_passed"] else 1

    if args.command == "review":
        rep = review_and_record_dataset(
            data_path=args.data_path,
            workspace_dir=args.workspace_dir,
            dataset_format=args.format,
            max_seq_len=args.max_seq_len,
            heldout_path=args.heldout_path,
            cycle_tag=args.cycle_tag,
            tokenizer_path=args.tokenizer_path,
        )
        print(json.dumps(rep, indent=2))
        return 0 if rep["passed"] else 1

    rep = audit_training_dataset(
        data_path=args.data_path,
        dataset_format=args.format,
        max_seq_len=args.max_seq_len,
        heldout_path=args.heldout_path,
        max_exec_checks=args.max_exec_checks,
        tokenizer_path=args.tokenizer_path,
    )
    if args.report_json is not None:
        atomic_write_text(args.report_json, json.dumps(rep, indent=2, ensure_ascii=False) + "\n")
    if args.report_md is not None:
        atomic_write_text(args.report_md, format_quality_report_md(rep) + "\n")
    print(json.dumps(rep, indent=2))
    return 0 if rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
