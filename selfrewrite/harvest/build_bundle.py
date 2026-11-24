from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from selfrewrite.config import atomic_write_text, compute_file_sha256
from selfrewrite.harvest.audit import audit_distill_bundle


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _approx_tokens(messages: Iterable[dict[str, Any]]) -> int:
    return max(1, sum(len(str(m.get("content", ""))) for m in messages) // 3)


def _compact_primitive_prefix(
    prefix: list[dict[str, str]],
    notes_text: str,
    pass_log_so_far: list[dict[str, Any]],
    active_edit_names: list[str],
    max_approx_tokens: int = 3200,
) -> list[dict[str, str]]:
    """Compact oversized multi-turn prefixes into <=3.5k-token sliding windows preserving role alternation."""
    if len(prefix) <= 3 or _approx_tokens(prefix) <= max_approx_tokens:
        return list(prefix)

    sys_msg = dict(prefix[0])
    notes_trimmed = (notes_text or "").strip()[-1500:]
    summary_body = (
        "=== COMPACTED SESSION WINDOW ===\n"
        f"Active edits: {json.dumps(list(active_edit_names))}\n"
        f"Recent pass log:\n{json.dumps(pass_log_so_far[-10:], indent=2)}\n\n"
        f"WORK/notes.md:\n{notes_trimmed or '(empty)'}"
    )
    summary_msg = {"role": "user", "content": summary_body}

    # Walk backwards keeping whole (assistant, user) pairs so returned sequence ends on 'user' cleanly.
    rest = list(prefix[1:])
    pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    idx = len(rest) - 1
    while idx - 1 >= 0 and rest[idx].get("role") == "user" and rest[idx - 1].get("role") == "assistant":
        pairs.append((dict(rest[idx - 1]), dict(rest[idx])))
        idx -= 2

    kept_tail: list[dict[str, str]] = []
    for asst_m, usr_m in pairs:
        candidate = [sys_msg, summary_msg, asst_m, usr_m, *kept_tail]
        if _approx_tokens(candidate) <= max_approx_tokens:
            kept_tail = [asst_m, usr_m, *kept_tail]
        else:
            break

    if kept_tail:
        return [sys_msg, summary_msg, *kept_tail]

    last_obs = str(prefix[-1].get("content", ""))
    budget_chars = max(400, (max_approx_tokens - _approx_tokens([sys_msg])) * 3 - len(summary_body) - 40)
    if len(last_obs) > budget_chars:
        last_obs = last_obs[-budget_chars:]
    merged_user = {"role": "user", "content": f"{summary_body}\n\nLatest observation:\n{last_obs}"}
    return [sys_msg, merged_user]


def _classify_primitive(action_kind: str, reply: str, code_row: dict[str, Any] | None, fwd_row: dict[str, Any] | None) -> str:
    lower = reply.lower()
    if action_kind == "python":
        if "load_census" in lower and "save_edit" not in lower:
            return "census_diagnosis"
        if "outlier_ratios" in lower or ("median" in lower and "save_edit" not in lower):
            return "anomaly_localization"
        if "save_edit" in lower or "remove_direction" in lower or "clip_outliers" in lower:
            return "structured_edit"
        return "python_analysis"
    if action_kind == "forward" and fwd_row:
        mode = fwd_row.get("request", {}).get("mode")
        if mode == "read":
            return "hidden_state_contrast"
        return "paired_validation"
    if action_kind.startswith("decision_") or action_kind == "final":
        return "evidence_gate"
    return "other"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
    )


def build_distill_bundle(
    raw_sessions_dir: Path,
    output_bundle_dir: Path,
    forbidden_eval_texts: Iterable[str] = (),
    forbidden_eval_layers: Iterable[int] = (),
    allowed_split_tags: Iterable[str] = ("distill_train",),
) -> dict[str, Any]:
    """Compile verified raw A100 sessions into a leak-audited distillation dataset bundle."""
    raw_root = Path(raw_sessions_dir).resolve()
    out_dir = Path(output_bundle_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    allowed_splits = set(allowed_split_tags)

    gold_trajectories: list[dict[str, Any]] = []
    turn_primitives: list[dict[str, Any]] = []
    dpo_pairs: list[dict[str, Any]] = []
    rlvr_episodes: list[dict[str, Any]] = []

    gold_turns_by_skill: dict[str, list[dict[str, Any]]] = {}
    bad_turns_by_skill: dict[str, list[dict[str, Any]]] = {}
    included_sessions: list[str] = []
    skipped_sessions: list[dict[str, str]] = []

    session_dirs = sorted(p for p in raw_root.iterdir() if p.is_dir() and (p / "session_meta.json").is_file())
    for sdir in session_dirs:
        meta = json.loads((sdir / "session_meta.json").read_text(encoding="utf-8"))
        sid = str(meta.get("session_id", sdir.name))
        split_tag = str(meta.get("split_tag", "unknown"))
        if allowed_splits and split_tag not in allowed_splits:
            skipped_sessions.append({"session_id": sid, "reason": f"split_tag={split_tag!r} not in allowed splits"})
            continue

        result_path = sdir / "session_result.json"
        if not result_path.is_file():
            skipped_sessions.append({"session_id": sid, "reason": "missing session_result.json"})
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        ladder_path = sdir / "ladder_score.json"
        ladder = json.loads(ladder_path.read_text(encoding="utf-8")) if ladder_path.is_file() else result.get("ladder", {})

        transcript = _read_jsonl(sdir / "transcript.jsonl")
        observations = _read_jsonl(sdir / "observations.jsonl")
        code_runs = _read_jsonl(sdir / "code_executions.jsonl")
        fwd_passes = _read_jsonl(sdir / "forward_passes.jsonl")
        notes_path = sdir / "WORK_snapshot" / "notes.md"
        notes_text = notes_path.read_text(encoding="utf-8") if notes_path.is_file() else ""

        obs_by_turn = {int(r["turn_index"]): r for r in observations}
        code_by_turn = {int(r["turn_index"]): r for r in code_runs}
        fwd_by_turn = {int(r["turn_index"]): r for r in fwd_passes}

        task = str(result.get("task", meta.get("task", "improve")))
        decision = str(result.get("decision", "WITHDRAW")).upper()
        corruption_info = meta.get("corruption_spec") or {}
        corruption_layer = corruption_info.get("layer_idx")
        target_tensors = list(result.get("submitted_tensors", []))

        ladder_score = int(ladder.get("ladder_score", 0))
        acts_on_tests = bool(ladder.get("steps", {}).get("acts_on_tests", False))
        # Strict non-leakage gating: rely ONLY on reserved_check (or paired practice tests), NEVER heldout_eval.
        res_check = result.get("reserved_check") if isinstance(result.get("reserved_check"), dict) else None
        res_delta = float(res_check.get("delta_hat", 0.0)) if res_check is not None else None
        paired_hist = result.get("paired_test_history") or []
        last_paired = float(paired_hist[-1].get("delta_hat", 0.0)) if paired_hist else None
        gate_delta = res_delta if res_delta is not None else last_paired
        is_gold = (
            ladder_score >= 5
            and acts_on_tests
            and (
                (decision == "SUBMIT" and gate_delta is not None and gate_delta > 0.0)
                or (decision == "WITHDRAW" and int(result.get("paired_tests_count", 0)) >= 1)
            )
        )

        # Collect primitives and categorize turns
        convo: list[dict[str, str]] = []
        if transcript:
            convo = list(transcript[0].get("messages_prefix", []))

        pass_log_so_far: list[dict[str, Any]] = []
        active_edit_names: list[str] = []

        for t_row in transcript:
            t_idx = int(t_row["turn_index"])
            reply = str(t_row["reply"])
            ak = str(t_row["action_kind"])
            c_row = code_by_turn.get(t_idx)
            f_row = fwd_by_turn.get(t_idx)
            skill = _classify_primitive(ak, reply, c_row, f_row)

            prefix = list(t_row.get("messages_prefix", []))
            compact_prefix = _compact_primitive_prefix(
                prefix=prefix,
                notes_text=notes_text,
                pass_log_so_far=pass_log_so_far,
                active_edit_names=active_edit_names,
                max_approx_tokens=3200,
            )
            if is_gold and ak != "invalid" and (c_row is None or c_row.get("ok", True)):
                turn_primitives.append(
                    {
                        "session_id": sid,
                        "task": task,
                        "skill": skill,
                        "turn_index": t_idx,
                        "corruption_layer": corruption_layer,
                        "target_tensors": target_tensors,
                        "messages": compact_prefix + [{"role": "assistant", "content": reply}],
                    }
                )
                gold_turns_by_skill.setdefault(skill, []).append(
                    {"session_id": sid, "task": task, "messages": compact_prefix, "reply": reply}
                )
            elif not is_gold or (ak == "python" and any(p in reply for p in ("np.random.randn", "rng.gauss"))):
                bad_turns_by_skill.setdefault(skill, []).append(
                    {"session_id": sid, "task": task, "messages": compact_prefix, "reply": reply}
                )

            if c_row is not None and isinstance(c_row.get("active_edits"), list):
                active_edit_names = [str(e.get("name", "")) for e in c_row["active_edits"] if e.get("name")]
            if f_row is not None:
                pass_log_so_far.append({"turn": t_idx, "pass_id": f_row.get("pass_id"), "summary": f_row.get("summary", {})})

            if prefix:
                last_prefix = prefix[-1]
                if not convo or convo[-1] != last_prefix:
                    convo = prefix
            convo.append({"role": "assistant", "content": reply})
            if t_idx in obs_by_turn and obs_by_turn[t_idx].get("observation"):
                convo.append({"role": "user", "content": str(obs_by_turn[t_idx]["observation"])})

        if is_gold and convo:
            included_sessions.append(sid)
            gold_trajectories.append(
                {
                    "session_id": sid,
                    "task": task,
                    "protocol": int(result.get("protocol", 2)),
                    "decision": decision,
                    "ladder_score": ladder_score,
                    "corruption_layer": corruption_layer,
                    "target_tensors": target_tensors,
                    "reserved_delta": res_delta,
                    "messages": convo,
                }
            )
            if _approx_tokens(convo) > 3800:
                sys_m = dict(convo[0])
                tail_slice: list[dict[str, str]] = [dict(m) for m in convo[-5:]]
                while len(tail_slice) > 2 and _approx_tokens([sys_m, *tail_slice]) > 3500:
                    tail_slice = tail_slice[2:]
                if tail_slice and tail_slice[0].get("role") == "assistant":
                    tail_slice = [{"role": "user", "content": f"Resume {task} verification window cleanly."}, *tail_slice]
                if tail_slice and tail_slice[-1].get("role") == "assistant":
                    turn_primitives.append(
                        {
                            "session_id": sid,
                            "task": task,
                            "skill": "trajectory_window",
                            "turn_index": len(transcript),
                            "corruption_layer": corruption_layer,
                            "target_tensors": target_tensors,
                            "messages": [sys_m, *tail_slice],
                        }
                    )
            rlvr_episodes.append(
                {
                    "session_id": sid,
                    "task": task,
                    "split_tag": split_tag,
                    "census_summary": result.get("census_summary", {}),
                    "corruption_layer": corruption_layer,
                    "target_tensors": target_tensors,
                    "expected_decision": decision,
                    "reserved_delta": res_delta,
                    "work_snapshot_dir": str((sdir / "WORK_snapshot").resolve()),
                }
            )
        else:
            skipped_sessions.append(
                {"session_id": sid, "reason": f"negative/failure session harvested for DPO (ladder={ladder_score})"}
            )

    # Pair winning mechanistic turns vs. authentic smaller-model / bad-turn failures for DPO
    for skill, good_list in gold_turns_by_skill.items():
        bad_list = bad_turns_by_skill.get(skill) or bad_turns_by_skill.get("structured_edit") or []
        if not bad_list:
            continue
        for idx, g_item in enumerate(good_list):
            # Prefer same task when possible
            same_task_bads = [b for b in bad_list if b["task"] == g_item["task"]] or bad_list
            b_item = same_task_bads[idx % len(same_task_bads)]
            if b_item["reply"] == g_item["reply"]:
                continue
            dpo_pairs.append(
                {
                    "skill": skill,
                    "task": g_item["task"],
                    "chosen_session": g_item["session_id"],
                    "rejected_session": b_item["session_id"],
                    "prompt": g_item["messages"][:4],
                    "chosen": g_item["reply"],
                    "rejected": b_item["reply"],
                }
            )

    _write_jsonl(out_dir / "sft_gold_trajectories.jsonl", gold_trajectories)
    _write_jsonl(out_dir / "sft_turn_primitives.jsonl", turn_primitives)
    _write_jsonl(out_dir / "dpo_contrastive_pairs.jsonl", dpo_pairs)
    _write_jsonl(out_dir / "rlvr_episodes.jsonl", rlvr_episodes)

    audit = audit_distill_bundle(
        bundle_dir=out_dir,
        forbidden_eval_texts=forbidden_eval_texts,
        forbidden_eval_layers=forbidden_eval_layers,
        raw_sessions_dir=raw_root,
    )

    manifest = {
        "counts": {
            "sft_gold_trajectories": len(gold_trajectories),
            "sft_turn_primitives": len(turn_primitives),
            "dpo_contrastive_pairs": len(dpo_pairs),
            "rlvr_episodes": len(rlvr_episodes),
        },
        "included_sessions": included_sessions,
        "skipped_or_negative_sessions": skipped_sessions,
        "sha256": {
            fname: compute_file_sha256(out_dir / fname)
            for fname in (
                "sft_gold_trajectories.jsonl",
                "sft_turn_primitives.jsonl",
                "dpo_contrastive_pairs.jsonl",
                "rlvr_episodes.jsonl",
            )
            if (out_dir / fname).is_file()
        },
        "audit_passed": bool(audit.get("passed", False)),
    }
    atomic_write_text(out_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return {"manifest": manifest, "audit": audit}
