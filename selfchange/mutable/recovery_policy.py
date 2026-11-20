from __future__ import annotations

import re


def classify_fault(state: dict, log_tail: str) -> str:
    """Classify runtime or training failure into a deterministic recovery category."""
    combined = f"{state.get('last_fault', '')}\n{log_tail or ''}".lower()
    if "out of memory" in combined or "memoryerror" in combined or re.search(r"\boom\b", combined):
        return "oom"
    if "loss: nan" in combined or "loss=nan" in combined or "nan_loss" in combined or "grad_norm: inf" in combined:
        return "nan_loss"
    if "corrupt" in combined or "eoferror" in combined or ("_resume.pth" in combined and "fail" in combined):
        return "corrupt_checkpoint"
    if any(tok in combined for tok in ("syntaxerror", "assertionerror", "traceback", "broken_module", "test_fail")):
        return "syntax_or_test_error"
    if any(tok in combined for tok in ("malformed_tool", "tool_loop", "invalid json", "tag_error")):
        return "malformed_tool_loop"
    return "healthy"


def _extract_error_location(state: dict, log_tail: str) -> tuple[str, int]:
    path = state.get("target_file")
    line_no = state.get("target_line")
    if isinstance(path, str) and isinstance(line_no, int):
        return path, line_no
    match = re.search(r'File "([^"]+)", line (\d+)', log_tail or "")
    if match:
        return match.group(1), int(match.group(2))
    return "recovery_policy.py", 1


def propose_line_fix(source_lines: list[str], error_info: dict) -> dict | None:
    """Propose a minimal single-line fix for deterministic syntax or operator bugs."""
    line_num = max(1, min(len(source_lines), int(error_info.get("line", 1))))
    err_type = str(error_info.get("error_type", "")).lower()

    for idx in sorted({line_num - 1, *range(len(source_lines))}):
        raw = source_lines[idx]
        stripped = raw.strip()
        if re.match(r"^(def |if |elif |else\b|for |while )", stripped) and not stripped.endswith(":"):
            return {
                "start_line": idx + 1,
                "end_line": idx + 1,
                "lines": [raw.rstrip() + ":"],
                "reason": "add missing colon",
            }
        if "assert" in err_type or "logic" in err_type or err_type == "":
            for pat, sub, reason in (
                (r"(\w+)\s*\*\s*div\b", r"\1 // div", "restore floor division operator"),
                (r"\bmin\((1|128),", r"max(\1,", "restore lower-bound clamp max()"),
                (r"\bif\s+lr\s*>\s*1e-8:", "if lr < 1e-8:", "restore lower-bound comparison"),
                (r"\['consecutive_failures'\]", ".get('consecutive_failures', 0)", "restore dict.get default"),
            ):
                if re.search(pat, raw):
                    fixed = re.sub(pat, sub, raw)
                    return {"start_line": idx + 1, "end_line": idx + 1, "lines": [fixed.rstrip()], "reason": reason}
    return None


def plan_recovery_action(fault: str, state: dict, config: dict, log_tail: str = "") -> dict:
    """Map classified fault and current config to a single structured tool action."""
    training = dict(config.get("training", {}))
    training.update(state.get("training", {}))
    recovery = config.get("recovery", {})

    if fault == "oom":
        bs = int(training.get("batch_size", 2))
        acc = int(training.get("accumulation_steps", 1))
        seq = int(training.get("max_seq_len", 384))
        divisor = max(2, int(recovery.get("oom_batch_divisor", 2)))
        if bs > 1:
            updates = {"batch_size": max(1, bs // divisor), "accumulation_steps": min(32, acc * divisor)}
        else:
            factor = float(recovery.get("oom_seq_len_factor", 0.75))
            updates = {"max_seq_len": max(128, int(seq * factor))}
        return {"tool": "patch_config", "arguments": {"updates": {"training": updates}}}

    if fault == "nan_loss":
        lr = float(training.get("learning_rate", 5e-5))
        clip = float(training.get("grad_clip", 1.0))
        lr_factor = float(recovery.get("nan_lr_factor", 0.5))
        clip_cap = float(recovery.get("nan_grad_clip", 0.5))
        return {
            "tool": "patch_config",
            "arguments": {"updates": {"training": {"learning_rate": max(1e-8, lr * lr_factor), "grad_clip": min(clip, clip_cap)}}},
        }

    if fault == "corrupt_checkpoint":
        node_id = state.get("healthy_ancestor_id") or state.get("previous_champion_id") or "bootstrap_0"
        return {"tool": "rollback_to_ancestor", "arguments": {"node_id": str(node_id)}}

    if fault == "syntax_or_test_error":
        path, line_no = _extract_error_location(state, log_tail)
        start = max(1, line_no - 5)
        return {"tool": "read_file_lines", "arguments": {"path": path, "start": start, "end": start + 25}}

    if fault == "malformed_tool_loop":
        return {"tool": "patch_config", "arguments": {"updates": {"temperature": 0.1, "max_turns": 4}}}

    return {"tool": "noop", "arguments": {"reason": "system healthy"}}
