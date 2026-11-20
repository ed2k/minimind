from __future__ import annotations

import json

SYSTEM_PROMPT = (
    "You operate MiniMind selfchange inside a strict sandbox. "
    "Observe state, logs, or file lines before editing. "
    "Keep patches minimal, stay within selfchange/mutable/, "
    "and emit valid <tool_call> JSON arguments."
)


def format_fault_card(state: dict, log_tail: str, max_log_lines: int = 8) -> str:
    """Render a compact 5-10 line diagnostic summary within 64M context limits."""
    lines = [line.rstrip() for line in (log_tail or "").splitlines() if line.strip()]
    excerpt = " | ".join(lines[-max(1, max_log_lines) :]) if lines else "none"
    if len(excerpt) > 320:
        excerpt = excerpt[-320:]
    training = state.get("training", {})
    card_lines = [
        f"profile={state.get('profile', 'local_intel_cpu')} fault={state.get('last_fault', 'unknown')}",
        f"champion={state.get('champion_id', 'none')} failures={state.get('consecutive_failures', 0)}",
        f"batch_size={training.get('batch_size', state.get('batch_size', 2))} lr={training.get('learning_rate', state.get('learning_rate', 5e-5))}",
        f"max_seq_len={training.get('max_seq_len', state.get('max_seq_len', 384))} grad_clip={training.get('grad_clip', 1.0)}",
        f"log_excerpt={excerpt}",
    ]
    return "\n".join(card_lines)


def format_observation(tool_name: str, result: dict, max_chars: int = 600) -> str:
    """Format a compact JSON tool observation truncated safely for context budgets."""
    payload = json.dumps({"tool": tool_name, "result": result}, ensure_ascii=False)
    if len(payload) > max_chars:
        return payload[: max_chars - 3] + "..."
    return payload
