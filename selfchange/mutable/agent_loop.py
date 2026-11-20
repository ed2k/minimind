from __future__ import annotations

import json
import re

from selfchange.mutable.prompts import SYSTEM_PROMPT, format_fault_card, format_observation


def parse_tool_calls(text: str) -> tuple[list[dict], int]:
    """Parse <tool_call> JSON blocks and count unbalanced tags or malformed JSON."""
    content = text or ""
    unbalanced = abs(content.count("<tool_call>") - content.count("</tool_call>"))
    unbalanced += abs(content.count("<think>") - content.count("</think>"))

    valid_calls: list[dict] = []
    malformed = 0
    for block in re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL):
        try:
            obj = json.loads(block.strip())
        except Exception:
            malformed += 1
            continue
        if isinstance(obj, dict) and isinstance(obj.get("name"), str) and obj.get("name"):
            args = obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    malformed += 1
                    continue
            if not isinstance(args, dict):
                malformed += 1
                continue
            valid_calls.append({"name": obj["name"], "arguments": args})
        else:
            malformed += 1
    return valid_calls, unbalanced + malformed


def run_agent_episode(
    initial_state: dict,
    tools: list[dict],
    model_fn,
    tool_dispatch,
    config: dict,
) -> dict:
    """Execute a bounded multi-turn observe-plan-act-verify episode."""
    max_turns = max(1, min(12, int(config.get("max_turns", 5))))
    card = format_fault_card(initial_state, str(initial_state.get("log_tail", "")))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": card},
    ]

    turn_outputs: list[str] = []
    tool_calls_made: list[dict] = []
    tool_results: list[dict] = []
    seen_signatures: set[tuple[str, str]] = set()
    total_tag_errors = 0
    consecutive_malformed = 0
    submitted = False
    unfinished = True

    for _ in range(max_turns):
        reply = str(model_fn(messages, tools) or "")
        turn_outputs.append(reply)
        messages.append({"role": "assistant", "content": reply})

        calls, tag_errors = parse_tool_calls(reply)
        total_tag_errors += tag_errors

        if not calls and tag_errors > 0:
            consecutive_malformed += 1
            messages.append({"role": "user", "content": "Invalid <tool_call> JSON. Retry with valid JSON."})
            if consecutive_malformed >= 2:
                break
            continue

        if not calls:
            unfinished = False
            break

        consecutive_malformed = 0
        turn_sigs = [(c["name"], json.dumps(c["arguments"], sort_keys=True, ensure_ascii=False)) for c in calls]
        if all(sig in seen_signatures for sig in turn_sigs):
            break

        for call, sig in zip(calls, turn_sigs):
            seen_signatures.add(sig)
            result = tool_dispatch(call["name"], call["arguments"])
            tool_calls_made.append(call)
            tool_results.append(result)
            messages.append({"role": "tool", "content": format_observation(call["name"], result)})
            if call["name"] in ("submit_candidate", "rollback_to_ancestor") and result.get("ok", True):
                submitted = True
                unfinished = False
        if submitted:
            break

    final_text = turn_outputs[-1] if turn_outputs else ""
    return {
        "messages": messages,
        "turn_outputs": turn_outputs,
        "tool_calls_made": tool_calls_made,
        "tool_results": tool_results,
        "tag_errors": total_tag_errors,
        "turns_used": len(turn_outputs),
        "submitted": submitted,
        "final_text": final_text,
        "unfinished": unfinished,
    }
