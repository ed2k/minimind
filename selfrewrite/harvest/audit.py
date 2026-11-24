from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from selfrewrite.config import atomic_write_text
from selfrewrite.tasks.datasets import normalize_problem_text

_LAYER_REF_RE = re.compile(r"layers\.(\d+)\.")
_NOISE_PATTERNS = ("np.random.randn", "np.random.normal", "torch.randn", "rng.gauss")
_DELTA_CITE_RE = re.compile(r"delta_hat\s*=\s*([+-]?\d+\.\d+)")


def _extract_strings(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, list):
        return "\n".join(_extract_strings(x) for x in obj)
    if isinstance(obj, dict):
        return "\n".join(_extract_strings(v) for v in obj.values())
    return str(obj)


def _check_text_leak(blob: str, forbidden_norms: set[str]) -> list[str]:
    if not blob or not forbidden_norms:
        return []
    hits: list[str] = []
    # Check normalized whole chunks / lines and substrings for verbatim or near-verbatim question overlap.
    blob_norm = normalize_problem_text(blob)
    for fn in forbidden_norms:
        if len(fn) >= 18 and fn in blob_norm:
            hits.append(fn[:80])
            if len(hits) >= 5:
                break
    return hits


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _load_recorded_session_deltas(raw_sessions_dir: Path | None) -> dict[str, list[float]]:
    if raw_sessions_dir is None or not Path(raw_sessions_dir).is_dir():
        return {}
    by_sid: dict[str, list[float]] = {}
    for sdir in sorted(Path(raw_sessions_dir).iterdir()):
        if not sdir.is_dir():
            continue
        deltas: list[float] = []
        res_file = sdir / "session_result.json"
        if res_file.is_file():
            try:
                res = json.loads(res_file.read_text(encoding="utf-8"))
                rc = res.get("reserved_check")
                if isinstance(rc, dict) and "delta_hat" in rc:
                    deltas.append(float(rc["delta_hat"]))
                for item in res.get("paired_test_history") or []:
                    if isinstance(item, dict) and "delta_hat" in item:
                        deltas.append(float(item["delta_hat"]))
            except Exception:
                pass
        fwd_file = sdir / "forward_passes.jsonl"
        if fwd_file.is_file():
            for row in _load_jsonl(fwd_file):
                summ = row.get("summary")
                if not isinstance(summ, dict):
                    continue
                if "delta_hat" in summ:
                    deltas.append(float(summ["delta_hat"]))
                comps = summ.get("comparisons")
                if isinstance(comps, dict):
                    for comp in comps.values():
                        if isinstance(comp, dict) and "delta_hat" in comp:
                            deltas.append(float(comp["delta_hat"]))
        if deltas:
            by_sid[sdir.name] = deltas
    return by_sid


def audit_distill_bundle(
    bundle_dir: Path,
    forbidden_eval_texts: Iterable[str] = (),
    forbidden_eval_layers: Iterable[int] = (),
    raw_sessions_dir: Path | None = None,
) -> dict[str, Any]:
    """Run hard anti-leakage, selection-leakage, layer-disjointness, grounding, and format checks."""
    root = Path(bundle_dir).resolve()
    if raw_sessions_dir is None:
        auto_raw = root.parent.parent / "raw_sessions"
        if auto_raw.is_dir():
            raw_sessions_dir = auto_raw
    recorded_deltas_by_session = _load_recorded_session_deltas(raw_sessions_dir)

    eval_norms = {normalize_problem_text(t) for t in forbidden_eval_texts if normalize_problem_text(t)}
    eval_layer_set = {int(x) for x in forbidden_eval_layers}

    violations: list[dict[str, Any]] = []
    stats: dict[str, Any] = {}

    files = (
        "sft_gold_trajectories.jsonl",
        "sft_turn_primitives.jsonl",
        "dpo_contrastive_pairs.jsonl",
        "rlvr_episodes.jsonl",
    )

    for fname in files:
        rows = _load_jsonl(root / fname)
        token_lens: list[int] = []
        stats[fname] = {"count": len(rows)}

        for idx, row in enumerate(rows):
            blob = _extract_strings(row)
            token_lens.append(max(1, len(blob) // 3))

            # 0. Reject heldout selection metadata in distillation rows
            if fname in ("sft_gold_trajectories.jsonl", "rlvr_episodes.jsonl"):
                if "eval_delta" in row or "heldout_eval" in row:
                    violations.append(
                        {
                            "file": fname,
                            "row": idx,
                            "rule": "heldout_selection_field_leakage",
                            "detail": "held-out selection metadata present in training row",
                        }
                    )

            # 1. Problem leakage check against reserved / heldout eval problems
            leaks = _check_text_leak(blob, eval_norms)
            if leaks:
                violations.append(
                    {
                        "file": fname,
                        "row": idx,
                        "rule": "eval_problem_leakage",
                        "detail": leaks[0],
                    }
                )

            # 2. Corruption-layer leakage check for repair examples
            task = str(row.get("task", ""))
            if task == "repair":
                corrupt_layer = row.get("corruption_layer")
                if corrupt_layer is not None and int(corrupt_layer) in eval_layer_set:
                    violations.append(
                        {
                            "file": fname,
                            "row": idx,
                            "rule": "eval_layer_leakage",
                            "detail": f"repair row uses held-out layer {corrupt_layer}",
                        }
                    )
                for edited_name in row.get("target_tensors", []) or []:
                    m = _LAYER_REF_RE.search(str(edited_name))
                    if m and int(m.group(1)) in eval_layer_set:
                        violations.append(
                            {
                                "file": fname,
                                "row": idx,
                                "rule": "eval_layer_tensor_leakage",
                                "detail": f"target tensor {edited_name!r} touches eval layer",
                            }
                        )

            # 3. Chosen turns must not contain ungrounded random noise perturbations
            if fname == "dpo_contrastive_pairs.jsonl":
                chosen_txt = str(row.get("chosen", ""))
                if not chosen_txt or not row.get("rejected"):
                    violations.append({"file": fname, "row": idx, "rule": "empty_dpo_field", "detail": "missing chosen/rejected"})
                elif any(pat in chosen_txt for pat in _NOISE_PATTERNS):
                    violations.append(
                        {
                            "file": fname,
                            "row": idx,
                            "rule": "noise_in_chosen_dpo",
                            "detail": "chosen DPO turn contains random-noise weight perturbation",
                        }
                    )

            # 4. Schema, <think> sanity & numerical grounding audit on SFT conversations
            if fname in ("sft_gold_trajectories.jsonl", "sft_turn_primitives.jsonl"):
                msgs = row.get("messages")
                sid = str(row.get("session_id", ""))
                rec_deltas = recorded_deltas_by_session.get(sid) or []
                if not isinstance(msgs, list) or len(msgs) < 2:
                    violations.append({"file": fname, "row": idx, "rule": "invalid_messages", "detail": "needs >=2 messages"})
                else:
                    assistants = [m.get("content", "") for m in msgs if m.get("role") == "assistant"]
                    if not assistants:
                        violations.append({"file": fname, "row": idx, "rule": "missing_assistant", "detail": "no assistant reply"})
                    for a in assistants:
                        if "<think>" in a and "</think>" not in a:
                            violations.append({"file": fname, "row": idx, "rule": "unclosed_think", "detail": "assistant open <think>"})
                        if rec_deltas:
                            for m_val in _DELTA_CITE_RE.findall(str(a)):
                                cited = float(m_val)
                                if min(abs(cited - d) for d in rec_deltas) > 0.02:
                                    violations.append(
                                        {
                                            "file": fname,
                                            "row": idx,
                                            "rule": "numerical_grounding_mismatch",
                                            "detail": f"cited delta_hat={m_val} not in recorded session deltas {rec_deltas}",
                                        }
                                    )

        if token_lens:
            token_lens_sorted = sorted(token_lens)
            stats[fname]["approx_mean_tokens"] = round(sum(token_lens) / len(token_lens), 1)
            stats[fname]["approx_max_tokens"] = token_lens_sorted[-1]

    report = {
        "passed": len(violations) == 0,
        "num_violations": len(violations),
        "violations": violations[:50],
        "stats": stats,
        "eval_problems_checked": len(eval_norms),
        "eval_layers_checked": sorted(eval_layer_set),
    }
    atomic_write_text(root / "quality_audit.json", json.dumps(report, indent=2) + "\n")
    return report
