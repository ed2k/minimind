from __future__ import annotations

import json
import re


def compute_rep_penalty(text: str, n: int = 3, cap: float = 0.5) -> float:
    """Compute n-gram repetition penalty matching trainer/train_agent.py."""
    toks = re.findall(r"\w+|[^\w\s]", (text or "").lower())
    grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
    if not grams:
        return 0.0
    return min(cap, (len(grams) - len(set(grams))) * cap * 2.0 / len(grams))


def _tool_signature(traj: dict) -> str:
    calls = traj.get("tool_calls", [])
    names = [str(c.get("name", "")) for c in calls if isinstance(c, dict)]
    return ">".join(names)


def curate_rft_samples(
    trajectories: list[dict],
    max_rep: float = 0.15,
    max_per_family: int = 50,
) -> list[dict]:
    """Select verified low-repetition passed episodes and deduplicate by family and tool chain."""
    seen: set[tuple[str, str]] = set()
    family_counts: dict[str, int] = {}
    selected: list[dict] = []

    for traj in trajectories:
        if not traj.get("passed"):
            continue
        final_text = str(traj.get("final_text", ""))
        if compute_rep_penalty(final_text) >= max_rep:
            continue
        family = str(traj.get("task_family", "default"))
        sig = _tool_signature(traj)
        key = (family, sig)
        if key in seen:
            continue
        if family_counts.get(family, 0) >= max_per_family:
            continue
        conversations = traj.get("messages") or traj.get("conversations")
        if not conversations:
            continue
        seen.add(key)
        family_counts[family] = family_counts.get(family, 0) + 1
        selected.append({"conversations": conversations, "task_family": family, "score": traj.get("reward", 1.0)})
    return selected


def build_dpo_pairs(trajectories: list[dict]) -> list[dict]:
    """Pair verified winners against failed trajectories on identical task_id values."""
    grouped: dict[str, list[dict]] = {}
    for traj in trajectories:
        task_id = str(traj.get("task_id", ""))
        if task_id:
            grouped.setdefault(task_id, []).append(traj)

    pairs: list[dict] = []
    for task_id in sorted(grouped):
        items = grouped[task_id]
        winners = [t for t in items if t.get("passed") and (t.get("messages") or t.get("conversations"))]
        losers = [t for t in items if not t.get("passed") and (t.get("messages") or t.get("conversations"))]
        if not winners or not losers:
            continue
        best_win = max(winners, key=lambda t: float(t.get("reward", 0.0)))
        worst_loss = min(losers, key=lambda t: float(t.get("reward", 0.0)))
        pairs.append(
            {
                "task_id": task_id,
                "chosen": best_win.get("messages") or best_win.get("conversations"),
                "rejected": worst_loss.get("messages") or worst_loss.get("conversations"),
            }
        )
    return pairs


def mix_with_anchor(
    op_samples: list[dict],
    anchor_samples: list[dict],
    anchor_ratio: float = 0.25,
) -> list[dict]:
    """Mix operational samples with anchor replay samples at the target anchor_ratio."""
    if not op_samples:
        return list(anchor_samples)
    if not anchor_samples or anchor_ratio <= 0.0:
        return list(op_samples)
    clamped = min(0.8, max(0.0, float(anchor_ratio)))
    target_anchors = max(1, int(round(len(op_samples) * clamped / max(1e-6, 1.0 - clamped))))
    chosen_anchors = [anchor_samples[i % len(anchor_samples)] for i in range(target_anchors)]
    mixed = list(op_samples) + chosen_anchors
    return json.loads(json.dumps(mixed))
