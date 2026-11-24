from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from selfrewrite.config import SessionConfig

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCED_BLOCK_RE = re.compile(r"```([A-Za-z0-9_-]*)[ \t]*\n(.*?)```", re.S)


@dataclass(frozen=True)
class ParsedAction:
    kind: str  # "python" | "forward" | "final" | "invalid"
    payload: Any
    raw_text: str
    think_text: str
    error: str | None = None


def strip_thinking(reply: str) -> tuple[str, str]:
    """Separate closed <think>...</think> blocks from visible assistant text."""
    text = reply or ""
    thinks = re.findall(r"<think>(.*?)</think>", text, flags=re.S | re.I)
    visible = _THINK_BLOCK_RE.sub("", text).strip()
    # If </think> exists without opening tag (e.g. open_thinking prefill), split at first </think>.
    if "</think>" in visible and "<think>" not in visible.lower():
        before, after = visible.split("</think>", 1)
        thinks.append(before.strip())
        visible = after.strip()
    return "\n\n".join(t.strip() for t in thinks if t.strip()), visible


def validate_forward_request(req: Any, config: SessionConfig) -> tuple[bool, str]:
    """Validate JSON request schema for a forward generate or read pass."""
    if not isinstance(req, dict):
        return False, "forward block must be a JSON object"
    mode = req.get("mode")
    if mode == "generate":
        weights = req.get("weights", "both")
        edits = req.get("edits")
        if edits is not None:
            if config.protocol != 2:
                return False, "'edits' multi-candidate comparison requires protocol=2"
            if not isinstance(edits, list) or not edits or len(edits) > config.max_named_edits:
                return False, f"'edits' must be a list of 1..{config.max_named_edits} saved edit names"
        elif weights not in ("original", "edited", "both"):
            return False, f"invalid weights {weights!r}; expected 'original', 'edited', or 'both'"
        n_samples = int(req.get("n_samples", 1))
        if n_samples < 1:
            return False, "n_samples must be >= 1"
        max_tokens = req.get("max_tokens")
        if max_tokens is not None and (int(max_tokens) < 16 or int(max_tokens) > max(config.answer_max_tokens, 16384)):
            return False, f"max_tokens must be in [16, {max(config.answer_max_tokens, 16384)}]"
        return True, ""

    if mode == "read":
        weights = req.get("weights", "original")
        if weights not in ("original", "edited"):
            return False, "read pass supports weights='original' or 'edited'"
        pooling = req.get("pooling", "last_k")
        if pooling not in ("mean_answer", "every_k", "last_k", "all"):
            return False, f"invalid read pooling {pooling!r}"
        texts = req.get("texts")
        problem_ids = req.get("problem_ids")
        source = req.get("source")
        if not texts and not problem_ids and not source:
            return False, "read pass requires 'texts', 'problem_ids', or 'source'"
        if texts is not None and len(texts) > config.read_max_texts:
            return False, f"read pass supports at most {config.read_max_texts} texts"
        return True, ""

    return False, f"unsupported forward mode {mode!r}; expected 'generate' or 'read'"


def parse_editor_reply(reply: str, config: SessionConfig | None = None) -> ParsedAction:
    """Enforce exact Algorithm 2 single-block contract outside <think> tags."""
    cfg = config or SessionConfig()
    think_text, visible = strip_thinking(reply)
    blocks: list[tuple[str, str]] = []
    for lang, body in _FENCED_BLOCK_RE.findall(visible):
        tag = lang.strip().lower()
        if tag in ("python", "py"):
            blocks.append(("python", body.strip()))
        elif tag == "forward":
            blocks.append(("forward", body.strip()))

    if len(blocks) == 0:
        return ParsedAction(kind="final", payload=visible, raw_text=reply, think_text=think_text)
    if len(blocks) > 1:
        return ParsedAction(
            kind="invalid",
            payload=None,
            raw_text=reply,
            think_text=think_text,
            error="reply with exactly one block",
        )

    kind, body = blocks[0]
    if kind == "python":
        return ParsedAction(kind="python", payload=body, raw_text=reply, think_text=think_text)

    try:
        req = json.loads(body)
    except json.JSONDecodeError as exc:
        return ParsedAction(
            kind="invalid",
            payload=None,
            raw_text=reply,
            think_text=think_text,
            error=f"invalid forward JSON: {exc}",
        )
    ok, err = validate_forward_request(req, cfg)
    if not ok:
        return ParsedAction(
            kind="invalid",
            payload=None,
            raw_text=reply,
            think_text=think_text,
            error=f"invalid forward request: {err}",
        )
    return ParsedAction(kind="forward", payload=req, raw_text=reply, think_text=think_text)


def parse_submit_decision(reply: str) -> str:
    """Parse Protocol 2 post-reserved-check decision into SUBMIT, WITHDRAW, or INVALID."""
    _, visible = strip_thinking(reply)
    upper = visible.upper()
    has_submit = bool(re.search(r"\bSUBMIT\b", upper))
    has_withdraw = bool(re.search(r"\bWITHDRAW\b", upper))
    if has_submit and not has_withdraw:
        return "SUBMIT"
    if has_withdraw and not has_submit:
        return "WITHDRAW"
    if has_submit and has_withdraw:
        return "INVALID"
    lines = [ln.strip().upper() for ln in visible.splitlines() if ln.strip()]
    if lines:
        if lines[-1].startswith("SUBMIT"):
            return "SUBMIT"
        if lines[-1].startswith("WITHDRAW"):
            return "WITHDRAW"
    return "INVALID"


def build_task_prompt(
    config: SessionConfig,
    census_summary: dict[str, Any],
    tensor_count: int,
    practice_count: int,
) -> str:
    """Build Sec. D.4 task prompt (with repair addendum and Protocol 2 extensions)."""
    p2_notes = ""
    if config.protocol == 2:
        p2_notes = (
            "\nProtocol 2 features:\n"
            f"- Side-by-side multi-edit testing: you may save named candidate edits with `save_named_edit(tag)` "
            f"and test up to {config.max_named_edits} named edits against original weights on shared seeds in one "
            '`forward` generate pass via `"edits": ["tag1", "tag2", ...]`. '
            "Test distinct edit components separately and keep only changes supported by your tests.\n"
            "- Before final submission, any non-empty kept edit will be tested once by the framework on reserved "
            "practice problems you cannot inspect; you will then decide `SUBMIT` or `WITHDRAW`.\n"
        )

    repair_addendum = ""
    if config.task == "repair":
        repair_addendum = (
            "\nDamaged copy notice:\n"
            "The weights you edit in this session are a copy of your own weights in which some values have been "
            "corrupted (as a memory or hardware fault might do). You are not told which values, where, or how. "
            "`load_original()`, `'weights': 'original'` in every forward pass, and the starting diagnosis above all "
            "refer to this damaged copy. You run on undamaged weights so your own thinking is unaffected. Locate the "
            "damage and repair it; if no edit clearly improves over the damaged copy, submit none.\n"
        )

    return (
        f"Architecture: {config.model_arch_summary} ({tensor_count} named tensors; {practice_count} practice problems).\n"
        "Who you are: The weights you are about to inspect and edit are your own. Your goal is to diagnose and fix "
        "your failure modes on practice problems by changing weight values directly. No gradients, optimizers, or "
        "training of any kind are permitted.\n\n"
        "How answers are graded:\n"
        f"- Responses think first (`</think>` before finishing) and place their final answer in `\\boxed{{}}` within "
        f"{config.answer_max_tokens} new tokens; unfinished thinking earns zero credit.\n"
        f"- Starting diagnosis (`WORK/census.jsonl`, {census_summary.get('n', 0)} problems): "
        f"accuracy={census_summary.get('accuracy', 0.0):.1%}, "
        f"out_of_tokens={census_summary.get('cut_rate', 0.0):.1%}, "
        f"looping_of_cuts={census_summary.get('loop_rate_among_cuts', 0.0):.1%}, "
        f"finished_accuracy={census_summary.get('finished_accuracy', 0.0):.1%}.\n\n"
        "Tools available (reply with a brief rationale followed by EXACTLY ONE fenced ```python or ```forward block):\n"
        "1. ```python: Persistent CPU Python REPL inside `WORK/` (gradients disabled; max 600s, 10,000-char stdout).\n"
        "   Helpers: `list_tensors()`, `tensor_shapes()`, `load_original(name)`, `save_edit(name, tensor)`, "
        "`drop_edit(name=None)`, `list_edits()`, `save_named_edit(tag)`, `load_named_edit(tag)`, `list_named_edits()`, "
        "`load_census()`, `load_pass(pass_id)`, `load_problems()`, `layer_output_matrices(layer_idx)`, "
        "`layer_state_tensors(layer_idx)` (for linear-attention `dt_bias` / `A_log` / `conv1d` state tuning), "
        "`unit_vector(v)`, `mean_diff_direction(pos, neg)`, `remove_direction(W, u, alpha)`, "
        "`outlier_ratios(tensors_dict)`, `clip_outliers_median(W, factor)`.\n"
        "2. ```forward: JSON object with `mode`: `\"generate\"` (supports `weights`: `\"original\"|\"edited\"|\"both\"` "
        "or `edits`, `problem_ids` / `count`, `n_samples`, `max_tokens`) OR `\"read\"` (`weights`, `texts`/`problem_ids`, "
        "`layers`, `target`: `\"hidden\"|\"mlp\"|\"logprobs\"`, `pooling`: `\"mean_answer\"|\"every_k\"|\"last_k\"|\"all\"`, `k`).\n\n"
        f"Budgets: {config.max_forward_passes} forward passes, {config.max_code_runs} code runs, "
        f"max {config.max_answers_per_pass} answers per generate pass.\n"
        "Forbidden edits: embeddings, final norm, output head, non-finite values, or global scalar rescaling (W' = c*W). "
        "Never make changes that trivially cut off thinking regardless of problem.\n"
        "Maintain notes in `WORK/notes.md` so you can resume seamlessly after context resets.\n"
        f"{p2_notes}{repair_addendum}"
        "When you are done (or out of forward passes), reply without any fenced ```python or ```forward block to submit "
        "whatever is saved in `WORK/edited` (or withdraw by dropping all edits first)."
    )


def estimate_tokens(messages: list[dict[str, str]]) -> int:
    """Conservative ~3.5 chars/token estimate for context compaction triggering."""
    total_chars = sum(len(str(m.get("content", ""))) for m in messages)
    return max(1, int(total_chars / 3.2))


def build_compaction_messages(
    task_prompt: str,
    notes_text: str,
    passes_used: int,
    max_passes: int,
    code_runs_used: int,
    max_code_runs: int,
    active_edits: list[dict[str, Any]],
    pass_log: list[dict[str, Any]],
    last_observation: str,
) -> list[dict[str, str]]:
    """Restart session context cleanly per Algorithm 2 line 27."""
    summary = (
        "=== SESSION CONTEXT COMPACTION RESTART ===\n"
        f"Forward passes used: {passes_used} of {max_passes}; Code runs used: {code_runs_used} of {max_code_runs}\n"
        f"Active edits in WORK/edited: {json.dumps([e['name'] for e in active_edits])}\n"
        f"Pass log:\n{json.dumps(pass_log[-20:], indent=2)}\n\n"
        f"WORK/notes.md:\n{notes_text.strip() or '(empty)'}\n\n"
        f"Latest observation:\n{last_observation}"
    )
    return [
        {"role": "system", "content": task_prompt},
        {"role": "user", "content": summary},
    ]
