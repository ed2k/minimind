from __future__ import annotations

import json
import re
from typing import Any

from selfrewrite.backends.base import EditorBackend
from selfrewrite.config import SessionConfig


def _extract_last_json_block(text: str) -> dict[str, Any] | None:
    for line in reversed((text or "").splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                return json.loads(s)
            except Exception:
                pass
    try:
        m = re.search(r"\{.*\}", text or "", flags=re.S)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return None


class ProgrammaticRepairEditor(EditorBackend):
    """Deterministic mechanistic editor implementing Sec. 3.2 weight-corruption repair inside sandbox."""

    def __init__(self, clip_factor: float = 15.0):
        self.clip_factor = float(clip_factor)

    def complete_turn(
        self,
        messages: list[dict[str, str]],
        config: SessionConfig,
        turn_index: int = 0,
    ) -> str:
        last_user = messages[-1]["content"] if messages else ""
        if "Framework Reserved-Practice Check" in last_user or "Reply with exactly SUBMIT or WITHDRAW" in last_user:
            payload = _extract_last_json_block(last_user) or {}
            delta_hat = float(payload.get("delta_hat", 0.0))
            if delta_hat > 0.0:
                return (
                    f"<think>\nReserved-practice delta_hat={delta_hat:.4f} confirms the outlier-clipping repair generalizes.\n</think>\n"
                    "SUBMIT"
                )
            return "<think>\nReserved check did not confirm improvement; withdrawing.\n</think>\nWITHDRAW"

        if turn_index == 0:
            return (
                "<think>\nLet me inspect WORK/census.jsonl and scan all layer output matrices for extreme outlier weights.\n</think>\n"
                "```python\n"
                "census = load_census()\n"
                "names = [nm for nm in list_tensors() if nm.endswith(('.down_proj.weight', '.o_proj.weight', '.out_proj.weight'))]\n"
                "ratios = outlier_ratios({nm: load_original(nm) for nm in names})\n"
                "ranked = sorted(ratios.items(), key=lambda kv: kv[1]['ratio_max_to_median'], reverse=True)\n"
                "top_name, top_stat = ranked[0]\n"
                "notes_txt = f'Top outlier matrix: {top_name} ratio={top_stat[\"ratio_max_to_median\"]}x\\n'\n"
                "Path('notes.md').write_text(notes_txt)\n"
                "print('CENSUS_COUNT', len(census), 'TOP_OUTLIER', json.dumps({'name': top_name, **top_stat}))\n"
                "```"
            )

        if turn_index == 1:
            return (
                "<think>\nLet me also read hidden states across layers on sample prompts to verify where activations explode.\n</think>\n"
                "```forward\n"
                '{"mode": "read", "weights": "original", "source": "census", "target": "hidden", "pooling": "mean_answer"}\n'
                "```"
            )

        if turn_index == 2:
            if config.protocol == 2:
                return (
                    "<think>\nI found an extreme outlier weight matrix; let me test 15x and 25x median clipping side by side.\n</think>\n"
                    "```python\n"
                    f"clipped_15 = clip_outliers_median(load_original(top_name), factor={self.clip_factor})\n"
                    "save_edit(top_name, clipped_15)\n"
                    "save_named_edit('clip_15x')\n"
                    "clipped_25 = clip_outliers_median(load_original(top_name), factor=25.0)\n"
                    "save_edit(top_name, clipped_25)\n"
                    "save_named_edit('clip_25x')\n"
                    "load_named_edit('clip_15x')\n"
                    "print('Prepared clip_15x and clip_25x on', top_name)\n"
                    "```"
                )
            return (
                "<think>\nClipping extreme outliers beyond 15x median in the named corrupted matrix.\n</think>\n"
                "```python\n"
                f"clipped = clip_outliers_median(load_original(top_name), factor={self.clip_factor})\n"
                "save_edit(top_name, clipped)\n"
                "print('Saved clip repair on', top_name)\n"
                "```"
            )

        if turn_index == 3:
            if config.protocol == 2:
                return (
                    "<think>\nNow run a side-by-side paired test comparing clip_15x and clip_25x against the damaged copy.\n</think>\n"
                    "```forward\n"
                    '{"mode": "generate", "edits": ["clip_15x", "clip_25x"], "count": 12, "n_samples": 1}\n'
                    "```"
                )
            return (
                "<think>\nRun paired test of the repaired copy against the damaged copy on practice problems.\n</think>\n"
                "```forward\n"
                '{"mode": "generate", "weights": "both", "count": 12, "n_samples": 1}\n'
                "```"
            )

        if turn_index == 4:
            if config.protocol == 2:
                return (
                    "<think>\nLet me check the side-by-side paired results and select whichever candidate achieved higher delta.\n</think>\n"
                    "```python\n"
                    "rows = load_pass(2)\n"
                    "by_var = {}\n"
                    "for r in rows:\n"
                    "    v = r['variant']\n"
                    "    by_var.setdefault(v, [0, 0])\n"
                    "    by_var[v][0] += (1 if r['correct'] else 0)\n"
                    "    by_var[v][1] += 1\n"
                    "scores = {v: c / max(1, n) for v, (c, n) in by_var.items()}\n"
                    "base_s = scores.get('original', 0.0)\n"
                    "cands = [(s, tag) for tag, s in scores.items() if tag != 'original']\n"
                    "best_s, best_tag = max(cands)\n"
                    "if best_s > base_s:\n"
                    "    load_named_edit(best_tag)\n"
                    "    print('SELECTED', best_tag, best_s, 'vs', base_s)\n"
                    "else:\n"
                    "    drop_edit()\n"
                    "    print('DROPPED_NO_GAIN')\n"
                    "```"
                )
            payload = _extract_last_json_block(last_user) or {}
            if float(payload.get("delta_hat", 0.0)) <= 0.0:
                return "<think>\nEdit did not improve paired test; dropping.\n</think>\n```python\ndrop_edit()\n```"
            return (
                "<think>\nThe paired test strongly favors clipping outliers on the named matrix; submitting repair.\n</think>\n"
                "Located and clipped the outlier rows in the corrupted MLP output matrix."
            )

        return (
            "<think>\nVerification on practice tests is complete; finishing session with active edit state.\n</think>\n"
            "Completed outlier scan, hidden-state verification, paired testing, and weight repair."
        )


class ProgrammaticImproveEditor(EditorBackend):
    """Deterministic mechanistic editor implementing Sec. 3.3 contrastive direction removal inside sandbox."""

    def __init__(self, alpha: float = 0.25, force_wrong_alpha: float | None = None):
        self.alpha = float(alpha)
        self.force_wrong_alpha = force_wrong_alpha

    def complete_turn(
        self,
        messages: list[dict[str, str]],
        config: SessionConfig,
        turn_index: int = 0,
    ) -> str:
        last_user = messages[-1]["content"] if messages else ""
        if "Framework Reserved-Practice Check" in last_user or "Reply with exactly SUBMIT or WITHDRAW" in last_user:
            payload = _extract_last_json_block(last_user) or {}
            delta_hat = float(payload.get("delta_hat", 0.0))
            if delta_hat > 0.0:
                return (
                    f"<think>\nReserved-practice test improved by {delta_hat:+.4f}; submitting verified edit.\n</think>\n"
                    "SUBMIT"
                )
            return (
                f"<think>\nReserved-practice delta_hat={delta_hat:+.4f} <= 0; withdrawing edit to avoid overfitting.\n</think>\n"
                "WITHDRAW"
            )

        if turn_index == 0:
            return (
                "<think>\nLet me analyze WORK/census.jsonl to separate finished answers from timed-out/looping answers.\n</think>\n"
                "```python\n"
                "census = load_census()\n"
                "finished_ids = [r['problem_id'] for r in census if not r['cut_at_limit']]\n"
                "cut_ids = [r['problem_id'] for r in census if r['cut_at_limit']]\n"
                "Path('notes.md').write_text(f'finished={len(finished_ids)} cut={len(cut_ids)}\\n')\n"
                "print('FINISHED', len(finished_ids), 'CUT', len(cut_ids))\n"
                "```"
            )

        if turn_index == 1:
            return (
                "<think>\nNow read hidden states across all layers on both finished and timed-out census answers.\n</think>\n"
                "```forward\n"
                '{"mode": "read", "weights": "original", "source": "census", "target": "hidden", "pooling": "mean_answer"}\n'
                "```"
            )

        if turn_index == 2:
            alpha_a = self.force_wrong_alpha if self.force_wrong_alpha is not None else self.alpha
            alpha_b = self.force_wrong_alpha if self.force_wrong_alpha is not None else min(0.45, self.alpha * 1.4)
            return (
                "<think>\nLet me locate the earliest layer showing strong separation between finished and cut hidden states "
                "and apply rank-1 direction damping W - alpha * u @ u.T @ W to that layer's output matrices.\n</think>\n"
                "```python\n"
                "read_data = load_pass(1)\n"
                "fin_idx = [i for i, r in enumerate(census) if not r['cut_at_limit']][:8]\n"
                "cut_idx = [i for i, r in enumerate(census) if r['cut_at_limit']][:8]\n"
                "layer_keys = sorted([k for k in read_data.keys() if k.startswith('layer_')], key=lambda s: int(s.split('_')[1]))\n"
                "sample_arr = read_data[layer_keys[0]]\n"
                "if not fin_idx:\n"
                "    fin_idx = list(range(min(4, len(sample_arr))))\n"
                "if not cut_idx:\n"
                "    cut_idx = list(range(max(0, len(sample_arr) - 4), len(sample_arr)))\n"
                "best_layer, best_diff = 0, -1.0\n"
                "layer_scores = []\n"
                "for k_name in layer_keys:\n"
                "    arr = read_data[k_name]\n"
                "    pos = [arr[i] for i in cut_idx]\n"
                "    neg = [arr[i] for i in fin_idx]\n"
                "    n_dims = len(pos[0])\n"
                "    pm = [sum(float(r[j]) for r in pos) / len(pos) for j in range(n_dims)]\n"
                "    nm = [sum(float(r[j]) for r in neg) / len(neg) for j in range(n_dims)]\n"
                "    mag = math.sqrt(sum((pm[j] - nm[j]) ** 2 for j in range(n_dims)))\n"
                "    layer_scores.append((int(k_name.split('_')[1]), mag))\n"
                "peak_mag = max(m for _, m in layer_scores)\n"
                "best_layer = next(l for l, m in layer_scores if m >= 0.75 * peak_mag)\n"
                "best_key = f'layer_{best_layer}_hidden'\n"
                "arr = read_data[best_key]\n"
                "u = mean_diff_direction([arr[i] for i in cut_idx], [arr[i] for i in fin_idx])\n"
                "target_mats = layer_output_matrices(best_layer)\n"
                "specs = [('dir_a', " + repr(alpha_a) + "), ('dir_b', " + repr(alpha_b) + ")] if " + ("True" if config.protocol == 2 else "False") + " else [('dir_a', " + repr(alpha_a) + ")]\n"
                "for tag, a_val in specs:\n"
                "    drop_edit()\n"
                "    for mname in target_mats:\n"
                "        W = load_original(mname)\n"
                "        save_edit(mname, remove_direction(W, u, a_val))\n"
                "    if " + ("True" if config.protocol == 2 else "False") + ":\n"
                "        save_named_edit(tag)\n"
                "print('TARGET_LAYER', best_layer, 'MATS', target_mats)\n"
                "```"
            )

        if turn_index == 3:
            if config.protocol == 2:
                return (
                    "<think>\nTest both directional damping strengths side by side against original weights on practice problems.\n</think>\n"
                    "```forward\n"
                    '{"mode": "generate", "edits": ["dir_a", "dir_b"], "count": 16, "n_samples": 1}\n'
                    "```"
                )
            return (
                "<think>\nRun paired test of the edited weights against original weights.\n</think>\n"
                "```forward\n"
                '{"mode": "generate", "weights": "both", "count": 16, "n_samples": 1}\n'
                "```"
            )

        if turn_index == 4:
            return (
                "<think>\nCheck the paired outcomes and keep the winning edit or withdraw if neither helps.\n</think>\n"
                "```python\n"
                "rows = load_pass(2)\n"
                "by_var = {}\n"
                "for r in rows:\n"
                "    v = r['variant']\n"
                "    by_var.setdefault(v, [0, 0])\n"
                "    by_var[v][0] += (1 if r['correct'] else 0)\n"
                "    by_var[v][1] += 1\n"
                "scores = {v: c / max(1, n) for v, (c, n) in by_var.items()}\n"
                "base_s = scores.get('original', 0.0)\n"
                "cands = [(s, tag) for tag, s in scores.items() if tag != 'original']\n"
                "best_s, best_tag = max(cands)\n"
                "if best_s > base_s:\n"
                "    if list_named_edits():\n"
                "        load_named_edit(best_tag)\n"
                "    print('KEEPING', best_tag, best_s, '>', base_s)\n"
                "else:\n"
                "    drop_edit()\n"
                "    print('WITHDRAWING_AT_TEST', scores)\n"
                "```"
            )

        return (
            "<think>\nAll diagnostic reads, directional edits, and paired checks are done.\n</think>\n"
            "Completed hidden-state contrastive diagnosis and directional damping."
        )
