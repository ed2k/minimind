from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from pathlib import Path

from selfchange.kernel.integrity import atomic_write_text
from selfchange.kernel.sandbox import run_sandboxed
from selfchange.kernel.tool_runtime import SELFCHANGE_TOOLS, validate_tool_call
from selfchange.mutable.agent_loop import parse_tool_calls

_TOOL_BY_NAME = {t["function"]["name"]: t for t in SELFCHANGE_TOOLS}

_CLEAN_FIXTURES: list[tuple[str, str, str]] = [
    (
        "oom_helper.py",
        "def compute_step(batch_size: int, div: int = 2) -> int:\n    return max(1, batch_size // div)\n",
        "from oom_helper import compute_step; assert compute_step(8, 2) == 4",
    ),
    (
        "lr_guard.py",
        "def clamp_lr(lr: float, max_lr: float = 5e-4) -> float:\n    if lr < 1e-8:\n        return 1e-8\n    return min(lr, max_lr)\n",
        "from lr_guard import clamp_lr; assert clamp_lr(1e-8) == 1e-8 and clamp_lr(1e-2) == 5e-4",
    ),
    (
        "seq_guard.py",
        "def shrink_seq(seq_len: int, factor: float = 0.75) -> int:\n    return max(128, int(seq_len * factor))\n",
        "from seq_guard import shrink_seq; assert shrink_seq(512) == 384 and shrink_seq(128) == 128",
    ),
    (
        "state_reader.py",
        "def read_failures(state: dict) -> int:\n    return int(state.get('consecutive_failures', 0))\n",
        "from state_reader import read_failures; assert read_failures({}) == 0 and read_failures({'consecutive_failures': 2}) == 2",
    ),
]


def _select_tools(names: list[str]) -> str:
    compact = []
    for n in names:
        t = _TOOL_BY_NAME.get(n)
        if not t:
            continue
        fn = t["function"]
        compact.append({"type": "function", "function": {"name": fn["name"], "parameters": fn["parameters"]}})
    return json.dumps(compact, ensure_ascii=False)


def _tc(name: str, arguments: dict) -> str:
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': arguments}, ensure_ascii=False)}\n</tool_call>"


def estimate_tokens(messages: list[dict], tokenizer=None) -> int:
    """Estimate token count accurately with tokenizer or unescaped rendered length."""
    if tokenizer is not None:
        try:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return len(tokenizer(rendered, add_special_tokens=False).input_ids)
        except Exception:
            pass
    total_chars = 0
    for msg in messages:
        total_chars += len(str(msg.get("content", ""))) + len(str(msg.get("tools", ""))) + 24
    return max(1, int(total_chars / 2.5))


def inject_ast_bug(source: str, variant: int) -> tuple[str, dict]:
    """Deterministically inject a single reversible bug into a short Python module."""
    lines = source.splitlines()
    mode = variant % 5

    if mode == 0:
        for idx, line in enumerate(lines):
            if line.strip().startswith("def ") and line.rstrip().endswith(":"):
                broken_line = line.rstrip()[:-1]
                broken_lines = list(lines)
                broken_lines[idx] = broken_line
                return "\n".join(broken_lines) + "\n", {
                    "start_line": idx + 1,
                    "end_line": idx + 1,
                    "lines": [line],
                    "target": broken_line,
                    "replacement": line,
                    "reason": "restore missing colon in function header",
                }
    elif mode == 1:
        for idx, line in enumerate(lines):
            if " // " in line:
                broken_line = line.replace(" // ", " * ", 1)
                broken_lines = list(lines)
                broken_lines[idx] = broken_line
                return "\n".join(broken_lines) + "\n", {
                    "start_line": idx + 1,
                    "end_line": idx + 1,
                    "lines": [line],
                    "target": broken_line,
                    "replacement": line,
                    "reason": "restore floor division instead of multiplication",
                }
    elif mode == 2:
        for idx, line in enumerate(lines):
            if " < " in line:
                broken_line = line.replace(" < ", " > ", 1)
                broken_lines = list(lines)
                broken_lines[idx] = broken_line
                return "\n".join(broken_lines) + "\n", {
                    "start_line": idx + 1,
                    "end_line": idx + 1,
                    "lines": [line],
                    "target": broken_line,
                    "replacement": line,
                    "reason": "fix inverted lower-bound comparison",
                }
    elif mode == 3:
        for idx, line in enumerate(lines):
            if "max(" in line:
                broken_line = line.replace("max(", "min(", 1)
                broken_lines = list(lines)
                broken_lines[idx] = broken_line
                return "\n".join(broken_lines) + "\n", {
                    "start_line": idx + 1,
                    "end_line": idx + 1,
                    "lines": [line],
                    "target": broken_line,
                    "replacement": line,
                    "reason": "restore lower clamp max() instead of min()",
                }
    else:
        for idx, line in enumerate(lines):
            if ".get('consecutive_failures', 0)" in line:
                broken_line = line.replace(".get('consecutive_failures', 0)", "['consecutive_failures']", 1)
                broken_lines = list(lines)
                broken_lines[idx] = broken_line
                return "\n".join(broken_lines) + "\n", {
                    "start_line": idx + 1,
                    "end_line": idx + 1,
                    "lines": [line],
                    "target": broken_line,
                    "replacement": line,
                    "reason": "use dict.get default to avoid KeyError",
                }

    # Fallback deterministic colon bug on line 1
    broken_line = lines[0].rstrip()[:-1] if lines[0].rstrip().endswith(":") else lines[0] + " )"
    broken_lines = [broken_line] + lines[1:]
    return "\n".join(broken_lines) + "\n", {
        "start_line": 1,
        "end_line": 1,
        "lines": [lines[0]],
        "target": broken_line,
        "replacement": lines[0],
        "reason": "restore valid function definition",
    }


def _capture_real_traceback(filename: str, broken_src: str, assertion_cmd: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        (tmp_dir / filename).write_text(broken_src, encoding="utf-8")
        res = run_sandboxed(
            [sys.executable, "-c", assertion_cmd],
            cwd=tmp_dir,
            timeout_sec=5.0,
            extra_env={"PYTHONPATH": str(tmp_dir)},
        )
        tail = [ln.strip() for ln in res.stderr.splitlines() if ln.strip()][-3:]
        return " | ".join(tail) if tail else "SyntaxError or AssertionError"


def generate_ast_repair_trace(
    rng: random.Random,
    edit_tool: str = "replace_line_range",
    run_subprocess: bool = True,
    with_error_recovery: bool = False,
) -> dict:
    """Generate a verified multi-turn Python AST bug-repair trajectory."""
    fname, clean_src, assert_cmd = _CLEAN_FIXTURES[rng.randrange(len(_CLEAN_FIXTURES))]
    variant = rng.randrange(5)
    broken_src, fix = inject_ast_bug(clean_src, variant)
    tb = _capture_real_traceback(fname, broken_src, assert_cmd) if run_subprocess else f"Error in {fname} line {fix['start_line']}"
    if len(tb) > 160:
        tb = tb[-160:]

    numbered = "\n".join(f"{i + 1}: {ln}" for i, ln in enumerate(broken_src.splitlines()))
    tool_names = ["read_file_lines", edit_tool, "submit_candidate"]
    edit_args = (
        {"path": fname, "start_line": fix["start_line"], "end_line": fix["end_line"], "lines": fix["lines"]}
        if edit_tool == "replace_line_range"
        else {"path": fname, "target": fix["target"], "replacement": fix["replacement"]}
    )

    convs = [
        {"role": "system", "content": "Repair broken mutable module.", "tools": _select_tools(tool_names)},
        {"role": "user", "content": f"fault=syntax_or_test_error target={fname} err={tb}"},
        {"role": "assistant", "content": f"<think>Read {fname} near line {fix['start_line']}.</think>\n{_tc('read_file_lines', {'path': fname, 'start': 1, 'end': 20})}"},
        {"role": "tool", "content": json.dumps({"ok": True, "path": fname, "content": numbered}, ensure_ascii=False)},
    ]

    if with_error_recovery:
        bad_args = (
            {"path": fname, "start_line": 99, "end_line": 99, "lines": fix["lines"]}
            if edit_tool == "replace_line_range"
            else {"path": fname, "target": "NON_EXISTENT", "replacement": fix["replacement"]}
        )
        convs.extend(
            [
                {"role": "assistant", "content": f"<think>Apply fix.</think>\n{_tc(edit_tool, bad_args)}"},
                {"role": "tool", "content": json.dumps({"ok": False, "error": "line range mismatch"})},
            ]
        )

    convs.extend(
        [
            {"role": "assistant", "content": f"<think>{fix['reason']} at line {fix['start_line']}.</think>\n{_tc(edit_tool, edit_args)}"},
            {"role": "tool", "content": json.dumps({"ok": True, "path": fname})},
            {"role": "assistant", "content": f"<think>Submit repaired module.</think>\n{_tc('submit_candidate', {'hypothesis': fix['reason']})}"},
        ]
    )
    return {"conversations": convs, "rung": 3, "task_family": "ast_repair"}


def _generate_rung1_trace(rng: random.Random) -> dict:
    case = rng.randrange(4)
    if case == 0:
        bs = rng.choice([4, 8, 16])
        new_bs = max(1, bs // 2)
        convs = [
            {"role": "system", "content": "Diagnose fault.", "tools": _select_tools(["patch_config"])},
            {"role": "user", "content": f"fault=oom batch_size={bs} log=CUDA out of memory"},
            {"role": "assistant", "content": f"<think>OOM at batch_size={bs}; halve batch_size to {new_bs}.</think>\n{_tc('patch_config', {'updates': {'training': {'batch_size': new_bs}}})}"},
        ]
        return {"conversations": convs, "rung": 1, "task_family": "atomic_oom"}
    if case == 1:
        lr = rng.choice([1e-4, 2e-4, 4e-4])
        new_lr = lr * 0.5
        convs = [
            {"role": "system", "content": "Diagnose fault.", "tools": _select_tools(["patch_config"])},
            {"role": "user", "content": f"fault=nan_loss lr={lr} log=loss: nan grad_norm: inf"},
            {"role": "assistant", "content": f"<think>NaN loss; halve learning_rate to {new_lr} and clamp grad_clip.</think>\n{_tc('patch_config', {'updates': {'training': {'learning_rate': new_lr, 'grad_clip': 0.5}}})}"},
        ]
        return {"conversations": convs, "rung": 1, "task_family": "atomic_nan"}
    if case == 2:
        line_no = rng.randint(2, 18)
        target = f"    return max(1, batch * {rng.choice([2, 4])})"
        convs = [
            {"role": "user", "content": f"Copy line {line_no} exactly:\n{line_no}: {target}"},
            {"role": "assistant", "content": f"<think>Exact line {line_no}.</think>\n{target}"},
        ]
        return {"conversations": convs, "rung": 1, "task_family": "copy_fidelity"}

    anc = f"anc_{rng.randint(1, 9)}"
    convs = [
        {"role": "system", "content": "Recover checkpoint.", "tools": _select_tools(["rollback_to_ancestor"])},
        {"role": "user", "content": f"fault=corrupt_checkpoint ancestor={anc} log=EOFError truncated _resume.pth"},
        {"role": "assistant", "content": f"<think>Corrupt resume checkpoint; roll back to {anc}.</think>\n{_tc('rollback_to_ancestor', {'node_id': anc})}"},
    ]
    return {"conversations": convs, "rung": 1, "task_family": "atomic_corrupt"}


def _generate_rung2_trace(rng: random.Random) -> dict:
    kind = rng.randrange(3)
    if kind == 0:
        bs = rng.choice([4, 8])
        acc = rng.choice([1, 2])
        convs = [
            {"role": "system", "content": "Observe logs then fix config.", "tools": _select_tools(["read_log_tail", "patch_config"])},
            {"role": "user", "content": f"Training exited rc=1 batch_size={bs} accumulation_steps={acc}."},
            {"role": "assistant", "content": f"<think>Inspect train.log first.</think>\n{_tc('read_log_tail', {'log_name': 'train.log', 'lines': 20})}"},
            {"role": "tool", "content": f"RuntimeError: CUDA out of memory at batch_size={bs}"},
            {"role": "assistant", "content": f"<think>OOM confirmed; halve batch_size and double accumulation.</think>\n{_tc('patch_config', {'updates': {'training': {'batch_size': bs // 2, 'accumulation_steps': acc * 2}}})}"},
        ]
        return {"conversations": convs, "rung": 2, "task_family": "oom_2turn"}
    if kind == 1:
        lr = rng.choice([1e-4, 2e-4])
        convs = [
            {"role": "system", "content": "Observe logs then stabilize LR.", "tools": _select_tools(["read_log_tail", "patch_config"])},
            {"role": "user", "content": f"Training diverged lr={lr}."},
            {"role": "assistant", "content": f"<think>Read train.log tail.</think>\n{_tc('read_log_tail', {'log_name': 'train.log', 'lines': 20})}"},
            {"role": "tool", "content": f"Epoch:[1/2](12/100), loss: nan, grad_norm: inf, lr: {lr}"},
            {"role": "assistant", "content": f"<think>NaN detected; reduce LR to {lr * 0.5} and clip grads at 0.5.</think>\n{_tc('patch_config', {'updates': {'training': {'learning_rate': lr * 0.5, 'grad_clip': 0.5}}})}"},
        ]
        return {"conversations": convs, "rung": 2, "task_family": "nan_2turn"}

    anc = f"champ_{rng.randint(10, 99)}"
    convs = [
        {"role": "system", "content": "Check archive and recover.", "tools": _select_tools(["get_archive_summary", "rollback_to_ancestor"])},
        {"role": "user", "content": "Resume failed: EOFError in full_sft_768_resume.pth."},
        {"role": "assistant", "content": f"<think>Find verified ancestor.</think>\n{_tc('get_archive_summary', {'top_k': 3})}"},
        {"role": "tool", "content": json.dumps([{"node_id": anc, "total_score": 0.91}])},
        {"role": "assistant", "content": f"<think>Roll back to {anc}.</think>\n{_tc('rollback_to_ancestor', {'node_id': anc})}"},
    ]
    return {"conversations": convs, "rung": 2, "task_family": "corrupt_2turn"}


def _generate_noop_or_rollback_trace(rng: random.Random) -> dict:
    if rng.random() < 0.6:
        loss = round(rng.uniform(0.9, 1.8), 3)
        convs = [
            {"role": "system", "content": "Monitor system.", "tools": _select_tools(["get_system_state"])},
            {"role": "user", "content": f"fault=healthy loss={loss} consecutive_failures=0"},
            {"role": "assistant", "content": f"<think>Metrics healthy at loss={loss}; no changes needed.</think>\nSystem is healthy; no edits required."},
        ]
        return {"conversations": convs, "rung": 3, "task_family": "healthy_noop"}

    anc = f"anc_{rng.randint(1, 5)}"
    convs = [
        {"role": "system", "content": "Recover from failure streak.", "tools": _select_tools(["get_archive_summary", "rollback_to_ancestor"])},
        {"role": "user", "content": "consecutive_failures=3 safe_mode=true"},
        {"role": "assistant", "content": f"<think>Streak hit 3; fetch top archive node.</think>\n{_tc('get_archive_summary', {'top_k': 3})}"},
        {"role": "tool", "content": json.dumps([{"node_id": anc, "total_score": 0.88}])},
        {"role": "assistant", "content": f"<think>Rollback to verified ancestor {anc}.</think>\n{_tc('rollback_to_ancestor', {'node_id': anc})}"},
    ]
    return {"conversations": convs, "rung": 3, "task_family": "streak_rollback"}


def validate_trace_record(record: dict, max_seq_len: int = 768) -> tuple[bool, str | None]:
    """Validate balanced tags, tool-call schemas, and <= 0.9 * max_seq_len token budget."""
    convs = record.get("conversations", [])
    if not convs:
        return False, "empty conversations"
    for msg in convs:
        if msg.get("role") == "assistant":
            calls, tag_errs = parse_tool_calls(msg.get("content", ""))
            if tag_errs > 0:
                return False, f"tag_errors={tag_errs}"
            for call in calls:
                ok, err = validate_tool_call(call["name"], call["arguments"])
                if not ok:
                    return False, err
    if estimate_tokens(convs) > int(0.9 * max_seq_len):
        return False, "exceeds 0.9 * max_seq_len"
    return True, None


def generate_traces(
    output_path: Path,
    count: int = 500,
    seed: int = 42,
    max_seq_len: int = 768,
    rung: str = "all",
    real_ast_subprocesses: int = 12,
) -> dict:
    """Generate validated multi-rung synthetic operational and AST-repair traces."""
    rng = random.Random(seed)
    records: list[dict] = []
    skipped = 0
    ast_subprocs_used = 0
    family_counts: dict[str, int] = {}

    for idx in range(count * 2):
        if len(records) >= count:
            break
        if rung == "1":
            rec = _generate_rung1_trace(rng)
        elif rung == "2":
            rec = _generate_rung2_trace(rng)
        elif rung == "3":
            if rng.random() < 0.25:
                rec = _generate_noop_or_rollback_trace(rng)
            else:
                use_sub = ast_subprocs_used < real_ast_subprocesses
                if use_sub:
                    ast_subprocs_used += 1
                rec = generate_ast_repair_trace(
                    rng,
                    edit_tool="replace_line_range",
                    run_subprocess=use_sub,
                    with_error_recovery=(idx % 5 == 0),
                )
        else:
            bucket = idx % 10
            if bucket < 3:
                rec = _generate_rung1_trace(rng)
            elif bucket < 6:
                rec = _generate_rung2_trace(rng)
            elif bucket < 8 or bucket == 8:
                use_sub = ast_subprocs_used < real_ast_subprocesses
                if use_sub:
                    ast_subprocs_used += 1
                rec = generate_ast_repair_trace(
                    rng,
                    edit_tool="replace_line_range",
                    run_subprocess=use_sub,
                    with_error_recovery=(bucket == 8),
                )
            else:
                rec = _generate_noop_or_rollback_trace(rng)

        ok, _ = validate_trace_record(rec, max_seq_len=max_seq_len)
        if not ok:
            skipped += 1
            continue
        fam = rec["task_family"]
        family_counts[fam] = family_counts.get(fam, 0) + 1
        records.append(rec)

    output_path = Path(output_path)
    atomic_write_text(output_path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    return {
        "output_path": str(output_path),
        "generated": len(records),
        "skipped_length_or_schema": skipped,
        "real_ast_subprocesses": ast_subprocs_used,
        "families": family_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate verified synthetic selfchange bootstrap traces")
    parser.add_argument("--output", type=Path, default=Path("selfchange/workspace/bootstrap_traces.jsonl"))
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-len", type=int, default=768)
    parser.add_argument("--rung", choices=["all", "1", "2", "3"], default="all")
    args = parser.parse_args()

    stats = generate_traces(args.output, count=args.count, seed=args.seed, max_seq_len=args.max_seq_len, rung=args.rung)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
