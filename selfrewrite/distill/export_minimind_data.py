from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

from selfrewrite.config import atomic_write_text, compute_file_sha256
from selfrewrite.harvest.audit import audit_distill_bundle
from selfrewrite.tasks.corruption import split_train_eval_layers
from selfrewrite.tasks.datasets import load_jsonl_problems

_EOS_MARKER = "<|im_end|>"

_CENSUS_STATS_RE = re.compile(
    r"accuracy=([0-9.]+%),\s*out_of_tokens=([0-9.]+%),\s*looping_of_cuts=([0-9.]+%),\s*finished_accuracy=([0-9.]+%)"
)
_PASS1_JSON_RE = re.compile(
    r"(Pass 1 read stored at WORK/read_1\.npz:\n)(\{.*?\})(\n\[Status:)",
    flags=re.S,
)
_COMPACTED_WINDOW_RE = re.compile(
    r"(=== COMPACTED SESSION WINDOW ===\nActive edits: )(\[[^\n]*\])(\nRecent pass log:\n)(\[.*?\])(\n\nWORK/notes\.md:)",
    flags=re.S,
)
_DIFF_LIST_RE = re.compile(r'("per_problem_diffs": )\[(.*?)\]', flags=re.S)
_LAYER_MENTION_RE = re.compile(r"\b(layers?\s+)(\d+)\b", flags=re.IGNORECASE)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    if body:
        body += "\n"
    return atomic_write_text(path, body)


def _repo_relative_str(path: Path, repo_root: Path | None = None) -> str:
    root = repo_root or Path(__file__).resolve().parents[2]
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _strip_eos_tokens(text: str) -> str:
    cleaned = str(text or "").strip()
    while cleaned.endswith(_EOS_MARKER):
        cleaned = cleaned[: -len(_EOS_MARKER)].strip()
    return cleaned


def _extract_think_and_answer(raw_content: str) -> tuple[str | None, str]:
    """Split `<think>...</think>` reasoning from assistant reply body cleanly."""
    cleaned = _strip_eos_tokens(raw_content)
    if "</think>" not in cleaned:
        return None, cleaned
    pre, post = cleaned.split("</think>", 1)
    if "<think>" in pre:
        reasoning = pre.split("<think>")[-1].strip("\n")
    else:
        reasoning = pre.strip("\n")
    return reasoning, post.lstrip("\n")


def _format_for_dpo_message_content(raw_content: str) -> str:
    """Normalize assistant message to canonical `<think>\\n...\\n</think>\\n...` without EOS suffix."""
    reasoning, answer = _extract_think_and_answer(raw_content)
    if reasoning is None:
        return answer
    return f"<think>\n{reasoning}\n</think>\n{answer}".rstrip()


def _to_sft_conversation(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Format messages to exact `dataset/lm_dataset.py` `SFTDataset` PyArrow `Features` schema."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = str(msg["role"])
        raw = str(msg.get("content", ""))
        if role == "assistant":
            reasoning, answer = _extract_think_and_answer(raw)
            out.append(
                {
                    "role": role,
                    "content": answer,
                    "reasoning_content": reasoning,
                    "tools": None,
                    "tool_calls": None,
                }
            )
        else:
            out.append(
                {
                    "role": role,
                    "content": _strip_eos_tokens(raw),
                    "reasoning_content": None,
                    "tools": None,
                    "tool_calls": None,
                }
            )
    return out


def _compact_read_summary(summary: dict[str, Any], target_layers: int = 8) -> dict[str, Any]:
    out = dict(summary)
    out["layers"] = list(range(target_layers))
    out["keys"] = [f"layer_{i}_hidden" for i in range(target_layers)]
    return out


def _compact_json_inline_arrays(text: str) -> str:
    """Collapse short JSON numeric arrays onto one line to save context tokens."""
    def _collapse(match: re.Match[str]) -> str:
        inner = " ".join(part.strip() for part in match.group(2).splitlines())
        return f"{match.group(1)}[{inner}]"

    return _DIFF_LIST_RE.sub(_collapse, text)


def _format_pass1_inline(summary: dict[str, Any]) -> str:
    compact_layers = json.dumps(summary["layers"])
    compact_keys = json.dumps(summary["keys"])
    return (
        "{\n"
        f'  "file": {json.dumps(summary["file"])},\n'
        f'  "layers": {compact_layers},\n'
        f'  "num_texts": {json.dumps(summary.get("num_texts", 8))},\n'
        f'  "target": {json.dumps(summary.get("target", "hidden"))},\n'
        f'  "pooling": {json.dumps(summary.get("pooling", "mean_answer"))},\n'
        f'  "keys": {compact_keys}\n'
        "}"
    )


def _remap_layer_index(idx_str: str, target_layers: int = 8) -> str:
    idx = int(idx_str)
    if idx == 62:
        return "6"
    if idx == 2:
        return "1"
    if idx < target_layers:
        return idx_str
    return str(min(target_layers - 1, max(0, round(idx * (target_layers - 1) / 63.0))))


def _build_compact_minimind_system_prompt(orig_system: str, target_layers: int = 8, hidden_size: int = 768) -> str:
    """Condense Protocol 2 system prompt (~350 tokens) for MiniMind context budgets."""
    is_damaged = "[damaged]" in orig_system
    damaged_tag = " [damaged]" if is_damaged else ""
    m = _CENSUS_STATS_RE.search(orig_system)
    if m:
        census_line = (
            f"accuracy={m.group(1)}, out_of_tokens={m.group(2)}, "
            f"looping_of_cuts={m.group(3)}, finished_accuracy={m.group(4)}"
        )
    else:
        census_line = "see WORK/census.jsonl"

    damaged_note = (
        "\nDamaged copy: `original` weights contain corrupted values; diagnose and clip outliers or withdraw if no gain."
        if is_damaged
        else ""
    )
    return (
        f"Architecture: minimind-{hidden_size} (layers={target_layers}, hidden_size={hidden_size}){damaged_tag}.\n"
        "You diagnose and edit your own weights directly without gradients.\n"
        f"Grading: think inside `</think>` then place final answer in `\\boxed{{}}` within 1536 tokens. Census: {census_line}.\n"
        "Reply with `<think>...</think>` followed by ONE fenced ```python or ```forward block:\n"
        "- ```python helpers: `list_tensors()`, `tensor_shapes()`, `load_original(name)`, `save_edit(name, tensor)`, "
        "`drop_edit(name=None)`, `save_named_edit(tag)`, `load_named_edit(tag)`, `list_named_edits()`, "
        "`load_census()`, `load_pass(id)`, `layer_output_matrices(layer_idx)`, `mean_diff_direction(pos, neg)`, "
        "`remove_direction(W, u, alpha)`, `outlier_ratios(d)`, `clip_outliers_median(W, factor)`.\n"
        "- ```forward JSON: `{\"mode\": \"read\", \"weights\": \"original\", \"source\": \"census\", \"target\": \"hidden\", \"pooling\": \"mean_answer\"}` "
        "or `{\"mode\": \"generate\", \"edits\": [...], \"count\": 16, \"n_samples\": 1}`.\n"
        "Forbidden edits: embeddings, final norm, lm_head, non-finite values, or global scaling. "
        "Test edits on practice sets; on Framework Reserved-Practice Check reply strictly `SUBMIT` or `WITHDRAW`."
        f"{damaged_note}"
    )


def adapt_text_to_minimind(text: str, target_layers: int = 8, hidden_size: int = 768) -> str:
    """Remap Qwen/Qwen3.5-27B layer/tensor identifiers and 64-layer logs to MiniMind-8L."""
    out = _strip_eos_tokens(text)

    # 1. Exact tensor path remapping matching harvest_improve_01 / harvest_repair_01 and Qwen3.5-27B structure.
    replacements = (
        ("Qwen/Qwen3.5-27B (layers=64, hidden_size=5120, profile=colab_l4)", f"minimind-{hidden_size} (layers={target_layers}, hidden_size={hidden_size})"),
        ("Qwen/Qwen3.5-27B", f"minimind-{hidden_size}"),
        ("model.language_model.layers.62.linear_attn.out_proj.weight", "model.layers.6.self_attn.o_proj.weight"),
        ("model.language_model.layers.62.mlp.down_proj.weight", "model.layers.6.mlp.down_proj.weight"),
        ("model.language_model.layers.2.mlp.down_proj.weight", "model.layers.1.mlp.down_proj.weight"),
        ("model.language_model.layers.", "model.layers."),
        (".linear_attn.out_proj.weight", ".self_attn.o_proj.weight"),
        ("TARGET_LAYER 62", "TARGET_LAYER 6"),
        ("layer_62_hidden", "layer_6_hidden"),
        ("layer_2_hidden", "layer_1_hidden"),
    )
    for src, dst in replacements:
        out = out.replace(src, dst)

    out = re.sub(
        r"model\.layers\.(\d+)\.",
        lambda m: f"model.layers.{_remap_layer_index(m.group(1), target_layers)}.",
        out,
    )
    out = re.sub(
        r"layer_(\d+)_hidden",
        lambda m: f"layer_{_remap_layer_index(m.group(1), target_layers)}_hidden",
        out,
    )
    out = _LAYER_MENTION_RE.sub(
        lambda m: f"{m.group(1)}{_remap_layer_index(m.group(2), target_layers)}"
        if int(m.group(2)) >= target_layers or int(m.group(2)) == 2
        else m.group(0),
        out,
    )

    # 2. Compact 64-layer JSON read summaries and compacted session logs to 8 MiniMind layers.
    def _replace_pass1(match: re.Match[str]) -> str:
        try:
            payload = json.loads(match.group(2))
            compacted = _format_pass1_inline(_compact_read_summary(payload, target_layers=target_layers))
            return f"{match.group(1)}{compacted}{match.group(3)}"
        except Exception:
            return match.group(0)

    out = _PASS1_JSON_RE.sub(_replace_pass1, out)

    def _replace_window(match: re.Match[str]) -> str:
        try:
            log = json.loads(match.group(4))
            for entry in log:
                summ = entry.get("summary")
                if isinstance(summ, dict) and "layers" in summ and "keys" in summ:
                    entry["summary"] = _compact_read_summary(summ, target_layers=target_layers)
            parts: list[str] = []
            for entry in log:
                summ = entry.get("summary")
                if isinstance(summ, dict) and "layers" in summ and "keys" in summ:
                    entry["summary"] = _compact_read_summary(summ, target_layers=target_layers)
                parts.append("  " + json.dumps(entry, ensure_ascii=False))
            log_txt = "[\n" + ",\n".join(parts) + "\n]"
            return f"{match.group(1)}{match.group(2)}{match.group(3)}{log_txt}{match.group(5)}"
        except Exception:
            return match.group(0)

    out = _COMPACTED_WINDOW_RE.sub(_replace_window, out)
    out = _compact_json_inline_arrays(out)
    return out


def _adapt_messages_to_minimind(
    messages: list[dict[str, Any]],
    target_layers: int = 8,
    hidden_size: int = 768,
) -> list[dict[str, str]]:
    adapted: list[dict[str, str]] = []
    for idx, msg in enumerate(messages):
        role = str(msg["role"])
        content = str(msg.get("content", ""))
        if idx == 0 and role == "system":
            new_content = _build_compact_minimind_system_prompt(content, target_layers=target_layers, hidden_size=hidden_size)
        else:
            new_content = adapt_text_to_minimind(content, target_layers=target_layers, hidden_size=hidden_size)
        adapted.append({"role": role, "content": new_content})
    return adapted


def _solve_math_reasoning(question: str, answer: str) -> str:
    """Create deterministic step-by-step CoT reasoning for distill_train arithmetic anchors."""
    q = question.strip()
    m_mul = re.search(r"Compute the exact value of\s+(\d+)\s*\*\s*(\d+)\s*\+\s*(\d+)", q)
    if m_mul:
        a, b, c = int(m_mul.group(1)), int(m_mul.group(2)), int(m_mul.group(3))
        prod = a * b
        return f"First compute {a} * {b} = {prod}. Then add {c}: {prod} + {c} = {prod + c}."

    m_lin = re.search(r"Solve for x:\s*(\d+)\s*\*\s*x\s*\+\s*(\d+)\s*=\s*(\d+)", q)
    if m_lin:
        a, b, rhs = int(m_lin.group(1)), int(m_lin.group(2)), int(m_lin.group(3))
        diff = rhs - b
        return f"Subtract {b} from both sides: {a}*x = {rhs} - {b} = {diff}. Divide by {a}: x = {diff} / {a} = {diff // a}."

    m_seq = re.search(r"sum of the\s+(\d+)\s+consecutive integers starting at\s+(\d+)", q)
    if m_seq:
        k, start = int(m_seq.group(1)), int(m_seq.group(2))
        end = start + k - 1
        total = k * (start + end) // 2
        return f"The integers run from {start} to {end}. Their sum is {k} * ({start} + {end}) / 2 = {total}."

    return f"Solving the expression carefully yields {answer}."


def _build_math_anchor_conversations(splits_dir: Path, limit: int = 16) -> list[dict[str, Any]]:
    train_problems = load_jsonl_problems(splits_dir / "distill_train.jsonl")
    anchors: list[dict[str, Any]] = []
    for prob in train_problems[:limit]:
        reason = _solve_math_reasoning(prob.question, str(prob.answer))
        anchors.append(
            {
                "conversations": _to_sft_conversation(
                    [
                        {"role": "user", "content": prob.question},
                        {
                            "role": "assistant",
                            "content": f"<think>\n{reason}\n</think>\n\\boxed{{{prob.answer}}}",
                        },
                    ]
                )
            }
        )
    return anchors


def _extract_verified_notes(turn_primitives: list[dict[str, Any]]) -> dict[str, str]:
    """Recover recorded WORK/notes.md snippets embedded in harvested compacted windows."""
    notes_by_sid: dict[str, str] = {}
    pat = re.compile(r"WORK/notes\.md:\n(.*?)\n\nLatest observation:", flags=re.S)
    for row in turn_primitives:
        sid = str(row.get("session_id", ""))
        if not sid or sid in notes_by_sid:
            continue
        for msg in row.get("messages", []):
            if msg.get("role") == "user":
                m = pat.search(str(msg.get("content", "")))
                if m:
                    notes_by_sid[sid] = m.group(1).strip() + "\n"
                    break
    return notes_by_sid


def _prepare_work_snapshots(
    workspace_dir: Path,
    output_dir: Path,
    rlvr_rows: list[dict[str, Any]],
    turn_primitives: list[dict[str, Any]],
    repo_root: Path,
) -> list[dict[str, Any]]:
    notes_by_sid = _extract_verified_notes(turn_primitives)
    snaps_root = output_dir / "work_snapshots"
    snaps_root.mkdir(parents=True, exist_ok=True)
    normalized_rlvr: list[dict[str, Any]] = []
    raw_root = workspace_dir / "raw_sessions"

    for row in rlvr_rows:
        out_row = dict(row)
        sid = str(row.get("session_id", ""))
        dest_snap = snaps_root / sid
        dest_snap.mkdir(parents=True, exist_ok=True)

        src_work_snap = raw_root / sid / "WORK_snapshot"
        copied_any = False
        if src_work_snap.is_dir():
            for fname in ("census.jsonl", "census_summary.json", "notes.md", "practice_problems.jsonl"):
                src = src_work_snap / fname
                if src.is_file():
                    shutil.copy2(src, dest_snap / fname)
                    copied_any = True

        if not copied_any:
            # Reconstruct strictly from verified session_result.json + embedded WORK/notes.md
            res_file = raw_root / sid / "session_result.json"
            if res_file.is_file():
                res = json.loads(res_file.read_text(encoding="utf-8"))
                if isinstance(res.get("census_summary"), dict):
                    atomic_write_text(
                        dest_snap / "census_summary.json",
                        json.dumps(res["census_summary"], indent=2) + "\n",
                    )
            elif isinstance(row.get("census_summary"), dict):
                atomic_write_text(
                    dest_snap / "census_summary.json",
                    json.dumps(row["census_summary"], indent=2) + "\n",
                )
            if sid in notes_by_sid:
                atomic_write_text(dest_snap / "notes.md", notes_by_sid[sid])

        out_row["work_snapshot_dir"] = _repo_relative_str(dest_snap, repo_root=repo_root)
        normalized_rlvr.append(out_row)

    return normalized_rlvr


def _rewrite_leaderboard_paths(data: Any, src_bundle_prefix: str, dst_bundle_rel: str) -> Any:
    if isinstance(data, str):
        return data.replace(src_bundle_prefix, dst_bundle_rel)
    if isinstance(data, list):
        return [_rewrite_leaderboard_paths(x, src_bundle_prefix, dst_bundle_rel) for x in data]
    if isinstance(data, dict):
        return {k: _rewrite_leaderboard_paths(v, src_bundle_prefix, dst_bundle_rel) for k, v in data.items()}
    return data


def _build_readme_markdown(
    manifest: dict[str, Any],
    export_meta: dict[str, Any],
    audit_report: dict[str, Any],
) -> str:
    return (
        "# Qwen/Qwen3.5-27B Self-Rewrite Dataset (`qwen35_27b_v1`)\n\n"
        "Verified mechanistic weight self-rewriting steps extracted from live Colab L4 runs on `Qwen/Qwen3.5-27B`,\n"
        "packaged with leak-audited splits and `dataset/lm_dataset.py`-compatible MiniMind training exports.\n\n"
        "## Directory Layout\n\n"
        "- `canonical_27b/` — Verified 27B bundle (`sft_gold_trajectories.jsonl`, `sft_turn_primitives.jsonl`, "
        "`dpo_contrastive_pairs.jsonl`, normalized `rlvr_episodes.jsonl`, `manifest.json`, `quality_audit.json`, "
        "plus `stage1_27b_leaderboard.json` and `stage2_downsize_leaderboard.json`).\n"
        "- `splits/` — Disjoint problem splits (`distill_train.jsonl`, `practice_eval.jsonl`, "
        "`reserved_eval.jsonl`, `heldout_eval.jsonl`, `transfer_eval.jsonl`).\n"
        "- `work_snapshots/` — Lightweight verified `census_summary.json` and `notes.md` state for RLVR episodes.\n"
        "- `minimind_ready/` — Exact schemas consumed by `dataset/lm_dataset.py` (`SFTDataset`, `DPODataset`, `RLAIFDataset`) "
        "and MiniMind's Jinja chat template (`model/tokenizer_config.json`).\n\n"
        "## Dataset Counts & Integrity\n\n"
        f"- Canonical 27B counts: `{json.dumps(manifest['counts'])}`\n"
        f"- Audit status: `passed={audit_report['passed']}`, `violations={audit_report['num_violations']}` "
        f"(checked against `{audit_report['eval_problems_checked']}` eval problems and `len={len(audit_report['eval_layers_checked'])}` held-out layers)\n"
        f"- MiniMind-ready counts: `{json.dumps(export_meta['minimind_ready_counts'])}`\n\n"
        "## Canonical SHA-256 Checksums\n\n"
        + "\n".join(f"- `{k}`: `{v}`" for k, v in sorted(manifest["sha256"].items()))
        + "\n\n## Re-Verify Anti-Leakage Audit & Regenerate\n\n"
        "```bash\n"
        "python3 -m selfrewrite.cli audit-bundle \\\n"
        "  --bundle-dir selfrewrite/data/qwen35_27b_v1/canonical_27b \\\n"
        "  --splits-dir selfrewrite/data/qwen35_27b_v1/splits \\\n"
        "  --num-layers 64\n"
        "```\n"
    )


def export_qwen27b_dataset(
    workspace_dir: Path = Path("selfrewrite/workspace"),
    output_dir: Path = Path("selfrewrite/data/qwen35_27b_v1"),
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Extract verified 27B self-rewrite traces from workspace into git-tracked canonical & MiniMind-ready datasets."""
    ws = Path(workspace_dir)
    out = Path(output_dir)
    root = repo_root or Path(__file__).resolve().parents[2]

    src_bundle = ws / "distill_bundle" / "v1"
    src_splits = ws / "splits"
    required_files = (
        src_bundle / "sft_gold_trajectories.jsonl",
        src_bundle / "sft_turn_primitives.jsonl",
        src_bundle / "dpo_contrastive_pairs.jsonl",
        src_bundle / "rlvr_episodes.jsonl",
        src_bundle / "manifest.json",
        src_splits / "distill_train.jsonl",
        src_splits / "practice_eval.jsonl",
        src_splits / "reserved_eval.jsonl",
        src_splits / "heldout_eval.jsonl",
        src_splits / "transfer_eval.jsonl",
    )
    missing = [str(p) for p in required_files if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Cannot export dataset; missing required source files: {missing}")

    canonical_dir = out / "canonical_27b"
    splits_out_dir = out / "splits"
    mm_dir = out / "minimind_ready"
    for d in (canonical_dir, splits_out_dir, mm_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Copy disjoint splits
    for split_file in sorted(src_splits.glob("*.jsonl")):
        shutil.copy2(split_file, splits_out_dir / split_file.name)

    # 2. Copy byte-identical canonical JSONL files and normalize RLVR paths
    for fn in ("sft_gold_trajectories.jsonl", "sft_turn_primitives.jsonl", "dpo_contrastive_pairs.jsonl"):
        shutil.copy2(src_bundle / fn, canonical_dir / fn)

    gold_rows = _load_jsonl(canonical_dir / "sft_gold_trajectories.jsonl")
    prim_rows = _load_jsonl(canonical_dir / "sft_turn_primitives.jsonl")
    dpo_rows = _load_jsonl(canonical_dir / "dpo_contrastive_pairs.jsonl")
    rlvr_raw = _load_jsonl(src_bundle / "rlvr_episodes.jsonl")

    normalized_rlvr = _prepare_work_snapshots(ws, out, rlvr_raw, prim_rows, repo_root=root)
    _write_jsonl(canonical_dir / "rlvr_episodes.jsonl", normalized_rlvr)

    # 3. Copy & path-normalize experimental leaderboards
    canonical_rel = _repo_relative_str(canonical_dir, repo_root=root)
    lb1 = ws / "leaderboard.json"
    if lb1.is_file():
        lb1_data = json.loads(lb1.read_text(encoding="utf-8"))
        atomic_write_text(canonical_dir / "stage1_27b_leaderboard.json", json.dumps(lb1_data, indent=2) + "\n")
    lb2 = ws / "downsize_leaderboard.json"
    if lb2.is_file():
        lb2_data = json.loads(lb2.read_text(encoding="utf-8"))
        lb2_clean = _rewrite_leaderboard_paths(
            lb2_data,
            "/content/minimind/selfrewrite/workspace/distill_bundle/v1",
            canonical_rel,
        )
        atomic_write_text(canonical_dir / "stage2_downsize_leaderboard.json", json.dumps(lb2_clean, indent=2) + "\n")

    # 4. Run hard anti-leakage audit & recompute canonical manifest
    eval_texts = [
        item.question
        for sname in ("heldout_eval.jsonl", "reserved_eval.jsonl", "practice_eval.jsonl")
        for item in load_jsonl_problems(splits_out_dir / sname)
    ]
    _, eval_layers = split_train_eval_layers(64)
    audit_report = audit_distill_bundle(
        bundle_dir=canonical_dir,
        forbidden_eval_texts=eval_texts,
        forbidden_eval_layers=eval_layers,
        raw_sessions_dir=ws / "raw_sessions",
    )
    if not audit_report.get("passed"):
        raise ValueError(f"Canonical 27B bundle failed audit: {audit_report}")

    src_manifest = json.loads((src_bundle / "manifest.json").read_text(encoding="utf-8"))
    bundle_files = (
        "sft_gold_trajectories.jsonl",
        "sft_turn_primitives.jsonl",
        "dpo_contrastive_pairs.jsonl",
        "rlvr_episodes.jsonl",
    )
    manifest = {
        "source_model": "Qwen/Qwen3.5-27B",
        "counts": {
            "sft_gold_trajectories": len(gold_rows),
            "sft_turn_primitives": len(prim_rows),
            "dpo_contrastive_pairs": len(dpo_rows),
            "rlvr_episodes": len(normalized_rlvr),
        },
        "included_sessions": src_manifest.get("included_sessions", []),
        "skipped_or_negative_sessions": src_manifest.get("skipped_or_negative_sessions", []),
        "sha256": {fn: compute_file_sha256(canonical_dir / fn) for fn in bundle_files},
        "audit_passed": True,
    }
    atomic_write_text(canonical_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")

    # 5. Export MiniMind-ready JSONLs strictly matching dataset/lm_dataset.py schemas
    raw_arch_sft = [{"conversations": _to_sft_conversation(r["messages"])} for r in [*prim_rows, *gold_rows]]
    _write_jsonl(mm_dir / "sft_27b_steps_raw_arch.jsonl", raw_arch_sft)

    adapted_prims_msgs = [_adapt_messages_to_minimind(r["messages"], target_layers=8, hidden_size=768) for r in prim_rows]
    adapted_gold_msgs = [_adapt_messages_to_minimind(r["messages"], target_layers=8, hidden_size=768) for r in gold_rows]
    math_anchors = _build_math_anchor_conversations(splits_out_dir, limit=16)

    minimind_sft = (
        [{"conversations": _to_sft_conversation(msgs)} for msgs in [*adapted_prims_msgs, *adapted_gold_msgs]]
        + math_anchors
    )
    _write_jsonl(mm_dir / "sft_minimind_adapted.jsonl", minimind_sft)

    minimind_dpo = []
    for row in dpo_rows:
        prompt_adapted = _adapt_messages_to_minimind(row["prompt"], target_layers=8, hidden_size=768)
        chosen_adapted = _format_for_dpo_message_content(
            adapt_text_to_minimind(row["chosen"], target_layers=8, hidden_size=768)
        )
        rejected_adapted = _format_for_dpo_message_content(
            adapt_text_to_minimind(row["rejected"], target_layers=8, hidden_size=768)
        )
        minimind_dpo.append(
            {
                "chosen": [*prompt_adapted, {"role": "assistant", "content": chosen_adapted}],
                "rejected": [*prompt_adapted, {"role": "assistant", "content": rejected_adapted}],
            }
        )
    _write_jsonl(mm_dir / "dpo_minimind_pairs.jsonl", minimind_dpo)

    # RLAIFDataset takes conversations[:-1] as prompt and conversations[-1] as reference answer
    rlaif_rows = []
    for msgs in [*adapted_prims_msgs, *adapted_gold_msgs]:
        if len(msgs) >= 2 and msgs[-1]["role"] == "assistant" and msgs[-2]["role"] == "user":
            rlaif_rows.append({"conversations": _to_sft_conversation(msgs)})
    _write_jsonl(mm_dir / "rlaif_minimind_prompts.jsonl", rlaif_rows)

    def _max_chars(rows_list: list[dict[str, Any]], key: str = "conversations") -> int:
        if not rows_list:
            return 0
        if key == "conversations":
            return max(sum(len(str(m.get("content", ""))) + len(str(m.get("reasoning_content") or "")) for m in r[key]) for r in rows_list)
        return max(sum(len(str(m.get("content", ""))) for m in r[key]) for r in rows_list)

    export_meta = {
        "schema_targets": {
            "sft_27b_steps_raw_arch.jsonl": "dataset.lm_dataset.SFTDataset",
            "sft_minimind_adapted.jsonl": "dataset.lm_dataset.SFTDataset",
            "dpo_minimind_pairs.jsonl": "dataset.lm_dataset.DPODataset",
            "rlaif_minimind_prompts.jsonl": "dataset.lm_dataset.RLAIFDataset",
        },
        "minimind_ready_counts": {
            "sft_27b_steps_raw_arch": len(raw_arch_sft),
            "sft_minimind_adapted": len(minimind_sft),
            "dpo_minimind_pairs": len(minimind_dpo),
            "rlaif_minimind_prompts": len(rlaif_rows),
        },
        "max_char_lengths": {
            "sft_27b_steps_raw_arch": _max_chars(raw_arch_sft),
            "sft_minimind_adapted": _max_chars(minimind_sft),
            "dpo_minimind_pairs_chosen": _max_chars(minimind_dpo, key="chosen"),
            "dpo_minimind_pairs_rejected": _max_chars(minimind_dpo, key="rejected"),
            "rlaif_minimind_prompts": _max_chars(rlaif_rows),
        },
        "sha256": {
            fn: compute_file_sha256(mm_dir / fn)
            for fn in (
                "sft_27b_steps_raw_arch.jsonl",
                "sft_minimind_adapted.jsonl",
                "dpo_minimind_pairs.jsonl",
                "rlaif_minimind_prompts.jsonl",
            )
        },
    }
    atomic_write_text(mm_dir / "export_metadata.json", json.dumps(export_meta, indent=2) + "\n")
    atomic_write_text(out / "README.md", _build_readme_markdown(manifest, export_meta, audit_report))

    return {
        "output_dir": str(out),
        "manifest": manifest,
        "audit": audit_report,
        "export_metadata": export_meta,
    }


def _cli_main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Export verified 27B self-rewrite traces into repo & MiniMind schemas.")
    p.add_argument("--workspace-dir", default="selfrewrite/workspace")
    p.add_argument("--output-dir", default="selfrewrite/data/qwen35_27b_v1")
    args = p.parse_args()
    res = export_qwen27b_dataset(workspace_dir=Path(args.workspace_dir), output_dir=Path(args.output_dir))
    print(json.dumps(res, indent=2))
    return 0 if res["audit"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(_cli_main())
