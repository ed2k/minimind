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
        eval_path = sdir / "eval_result.json"
        eval_res = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.is_file() else {}

        transcript = _read_jsonl(sdir / "transcript.jsonl")
        observations = _read_jsonl(sdir / "observations.jsonl")
        code_runs = _read_jsonl(sdir / "code_executions.jsonl")
        fwd_passes = _read_jsonl(sdir / "forward_passes.jsonl")

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
        eval_delta = float(eval_res.get("delta_hat", 0.0)) if eval_res else None
        is_gold = (
            ladder_score >= 5
            and acts_on_tests
            and (
                (decision == "SUBMIT" and (eval_delta is None or eval_delta > 0.0))
                or (decision == "WITHDRAW" and int(result.get("paired_tests_count", 0)) >= 1)
            )
        )

        # Collect primitives and categorize turns
        convo: list[dict[str, str]] = []
        if transcript:
            convo = list(transcript[0].get("messages_prefix", []))

        for t_row in transcript:
            t_idx = int(t_row["turn_index"])
            reply = str(t_row["reply"])
            ak = str(t_row["action_kind"])
            c_row = code_by_turn.get(t_idx)
            f_row = fwd_by_turn.get(t_idx)
            skill = _classify_primitive(ak, reply, c_row, f_row)

            prefix = list(t_row.get("messages_prefix", []))
            if is_gold and ak != "invalid" and (c_row is None or c_row.get("ok", True)):
                turn_primitives.append(
                    {
                        "session_id": sid,
                        "task": task,
                        "skill": skill,
                        "turn_index": t_idx,
                        "corruption_layer": corruption_layer,
                        "target_tensors": target_tensors,
                        "messages": prefix + [{"role": "assistant", "content": reply}],
                    }
                )
                gold_turns_by_skill.setdefault(skill, []).append(
                    {"session_id": sid, "task": task, "messages": prefix, "reply": reply}
                )
            elif not is_gold or (ak == "python" and any(p in reply for p in ("np.random.randn", "rng.gauss"))):
                bad_turns_by_skill.setdefault(skill, []).append(
                    {"session_id": sid, "task": task, "messages": prefix, "reply": reply}
                )

            if t_idx == 0 or (len(convo) < len(prefix) + 1):
                pass
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
                    "eval_delta": eval_delta,
                    "messages": convo,
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
