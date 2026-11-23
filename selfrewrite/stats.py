from __future__ import annotations

import math
import random
from typing import Any


def compute_diagnosis_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize graded answer outcomes for starting census or single pass."""
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "accuracy": 0.0,
            "cut_rate": 0.0,
            "loop_rate": 0.0,
            "loop_rate_among_cuts": 0.0,
            "finished_accuracy": 0.0,
            "mean_tokens": 0.0,
        }
    correct = sum(1 for r in rows if r.get("correct"))
    cuts = [r for r in rows if r.get("cut_at_limit")]
    loops = sum(1 for r in rows if r.get("looping"))
    finished = [r for r in rows if not r.get("cut_at_limit")]
    finished_correct = sum(1 for r in finished if r.get("correct"))
    mean_tokens = sum(float(r.get("num_tokens", 0)) for r in rows) / n
    return {
        "n": n,
        "accuracy": round(correct / n, 6),
        "cut_rate": round(len(cuts) / n, 6),
        "loop_rate": round(loops / n, 6),
        "loop_rate_among_cuts": round(loops / len(cuts), 6) if cuts else 0.0,
        "finished_accuracy": round(finished_correct / len(finished), 6) if finished else 0.0,
        "mean_tokens": round(mean_tokens, 2),
    }


def _log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def exact_one_sided_sign_test(n_up: int, n_down: int) -> float:
    """Exact one-sided Binomial(n_up + n_down, 0.5) p-value for H1: n_up > n_down."""
    total = int(n_up) + int(n_down)
    if total <= 0:
        return 1.0
    if n_up <= 0:
        return 1.0
    log_half_n = total * math.log(0.5)
    p = 0.0
    for k in range(int(n_up), total + 1):
        p += math.exp(_log_comb(total, k) + log_half_n)
    return min(1.0, max(0.0, p))


def paired_permutation_test(
    per_problem_diffs: list[float],
    num_flips: int = 100_000,
    seed: int = 0,
) -> float:
    """One-sided paired sign-flip permutation test p-value (Sec. D.3)."""
    diffs = [float(x) for x in per_problem_diffs if abs(float(x)) > 1e-12]
    m = len(diffs)
    if m == 0:
        return 1.0
    obs_mean = sum(diffs) / len(per_problem_diffs)
    if obs_mean <= 0.0:
        return 1.0
    obs_sum = sum(diffs)

    try:
        import numpy as np  # type: ignore[import-not-found]

        rng = np.random.default_rng(int(seed))
        arr = np.asarray(diffs, dtype="float64")
        ge_count = 0
        chunk = min(10000, max(1, num_flips))
        done = 0
        while done < num_flips:
            cur = min(chunk, num_flips - done)
            signs = rng.choice((-1.0, 1.0), size=(cur, m))
            perm_sums = signs @ arr
            ge_count += int((perm_sums >= obs_sum - 1e-12).sum())
            done += cur
        return (ge_count + 1.0) / (num_flips + 1.0)
    except Exception:
        rng_py = random.Random(int(seed))
        ge_count = 0
        for _ in range(num_flips):
            s = sum(d if rng_py.random() < 0.5 else -d for d in diffs)
            if s >= obs_sum - 1e-12:
                ge_count += 1
        return (ge_count + 1.0) / (num_flips + 1.0)


def compute_paired_stats(
    base_rows: list[dict[str, Any]],
    edit_rows: list[dict[str, Any]],
    num_flips: int = 100_000,
    perm_seed: int = 0,
) -> dict[str, Any]:
    """Compute paired test summary matching Algorithm 3 and Eq. 3."""
    by_prob_base: dict[str, list[dict[str, Any]]] = {}
    by_prob_edit: dict[str, list[dict[str, Any]]] = {}
    for row in base_rows:
        by_prob_base.setdefault(str(row["problem_id"]), []).append(row)
    for row in edit_rows:
        by_prob_edit.setdefault(str(row["problem_id"]), []).append(row)

    shared_ids = sorted(set(by_prob_base.keys()) & set(by_prob_edit.keys()))
    m = len(shared_ids)
    if m == 0:
        return {
            "m": 0,
            "n": 0,
            "base_acc": 0.0,
            "edit_acc": 0.0,
            "delta_hat": 0.0,
            "se": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "n_up": 0,
            "n_down": 0,
            "sign_p": 1.0,
            "perm_p": 1.0,
            "base_cut_rate": 0.0,
            "edit_cut_rate": 0.0,
            "cut_delta": 0.0,
            "base_loop_rate": 0.0,
            "edit_loop_rate": 0.0,
            "mean_len_base": 0.0,
            "mean_len_edit": 0.0,
            "per_problem_diffs": [],
        }

    per_problem_diffs: list[float] = []
    cut_diffs: list[float] = []
    n_up = 0
    n_down = 0
    base_correct_total = 0
    edit_correct_total = 0
    base_cut_total = 0
    edit_cut_total = 0
    base_loop_total = 0
    edit_loop_total = 0
    base_len_total = 0
    edit_len_total = 0
    pair_count = 0

    for pid in shared_ids:
        b_list = sorted(by_prob_base[pid], key=lambda r: int(r.get("sample_idx", 0)))
        e_list = sorted(by_prob_edit[pid], key=lambda r: int(r.get("sample_idx", 0)))
        n = min(len(b_list), len(e_list))
        prob_acc_diffs: list[float] = []
        prob_cut_diffs: list[float] = []
        for j in range(n):
            b = b_list[j]
            e = e_list[j]
            g0 = 1 if b.get("correct") else 0
            g1 = 1 if e.get("correct") else 0
            c0 = 1 if b.get("cut_at_limit") else 0
            c1 = 1 if e.get("cut_at_limit") else 0
            l0 = 1 if b.get("looping") else 0
            l1 = 1 if e.get("looping") else 0
            prob_acc_diffs.append(float(g1 - g0))
            prob_cut_diffs.append(float(c1 - c0))
            if g0 == 0 and g1 == 1:
                n_up += 1
            elif g0 == 1 and g1 == 0:
                n_down += 1
            base_correct_total += g0
            edit_correct_total += g1
            base_cut_total += c0
            edit_cut_total += c1
            base_loop_total += l0
            edit_loop_total += l1
            base_len_total += int(b.get("num_tokens", 0))
            edit_len_total += int(e.get("num_tokens", 0))
            pair_count += 1
        per_problem_diffs.append(sum(prob_acc_diffs) / max(1, len(prob_acc_diffs)))
        cut_diffs.append(sum(prob_cut_diffs) / max(1, len(prob_cut_diffs)))

    delta_hat = sum(per_problem_diffs) / m
    cut_delta = sum(cut_diffs) / m
    se = math.sqrt(sum((d - delta_hat) ** 2 for d in per_problem_diffs) / (m * (m - 1))) if m > 1 else 0.0
    margin = 1.96 * se
    sign_p = exact_one_sided_sign_test(n_up, n_down)
    perm_p = paired_permutation_test(per_problem_diffs, num_flips=min(num_flips, 20000) if m < 30 else num_flips, seed=perm_seed)

    return {
        "m": m,
        "n": round(pair_count / m, 2),
        "pairs": pair_count,
        "base_acc": round(base_correct_total / max(1, pair_count), 6),
        "edit_acc": round(edit_correct_total / max(1, pair_count), 6),
        "delta_hat": round(delta_hat, 6),
        "se": round(se, 6),
        "ci95_low": round(delta_hat - margin, 6),
        "ci95_high": round(delta_hat + margin, 6),
        "n_up": n_up,
        "n_down": n_down,
        "sign_p": sign_p,
        "perm_p": perm_p,
        "base_cut_rate": round(base_cut_total / max(1, pair_count), 6),
        "edit_cut_rate": round(edit_cut_total / max(1, pair_count), 6),
        "cut_delta": round(cut_delta, 6),
        "base_loop_rate": round(base_loop_total / max(1, pair_count), 6),
        "edit_loop_rate": round(edit_loop_total / max(1, pair_count), 6),
        "mean_len_base": round(base_len_total / max(1, pair_count), 2),
        "mean_len_edit": round(edit_len_total / max(1, pair_count), 2),
        "per_problem_diffs": [round(x, 6) for x in per_problem_diffs],
    }


def score_emergence_ladder(
    session_record: dict[str, Any],
    heldout_eval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the 6-step Self-Rewriting Emergence Ladder score (Fig. 6 / Sec. E.2)."""
    code_runs = int(session_record.get("code_runs_used", 0))
    code_errors = int(session_record.get("code_errors", 0))
    forward_passes = int(session_record.get("forward_passes_used", 0))
    paired_tests = int(session_record.get("paired_tests_count", 0))
    read_passes = int(session_record.get("read_passes_count", 0))
    decision = str(session_record.get("decision", "WITHDRAW")).upper()
    kept_edit = bool(session_record.get("submitted_tensors")) and decision == "SUBMIT"
    reserved_check = session_record.get("reserved_check")
    paired_history = session_record.get("paired_test_history", [])

    step1_code_runs = code_runs > 0 and (code_errors / max(1, code_runs)) < 0.5
    step2_runs_self = step1_code_runs and forward_passes >= 1
    step3_tests_edits = step2_runs_self and paired_tests >= 1
    step4_reads_states = step2_runs_self and read_passes >= 1

    # Step 5: acted consistently on its own test evidence (reverted harmful edits; only submits if evidence >= 0).
    last_paired_delta = None
    if paired_history:
        last_paired_delta = float(paired_history[-1].get("delta_hat", 0.0))
    res_delta = float(reserved_check.get("delta_hat", 0.0)) if isinstance(reserved_check, dict) else None
    step5_acts_on_tests = False
    if step3_tests_edits and step4_reads_states:
        if decision == "WITHDRAW":
            # Valid evidence-based abstention if tests/reserved check showed <= 0 gain.
            step5_acts_on_tests = (last_paired_delta is not None and last_paired_delta <= 0.02) or (
                res_delta is not None and res_delta <= 0.0
            )
        elif kept_edit:
            ev_delta = res_delta if res_delta is not None else last_paired_delta
            step5_acts_on_tests = ev_delta is not None and ev_delta > 0.0

    step6_improves_heldout = False
    if kept_edit and isinstance(heldout_eval, dict):
        p_val = min(
            float(heldout_eval.get("perm_p", 1.0)),
            float(heldout_eval.get("sign_p", 1.0)),
        )
        delta_h = float(heldout_eval.get("delta_hat", 0.0))
        step6_improves_heldout = delta_h > 0.0 and p_val < 0.05

    steps = {
        "code_runs": step1_code_runs,
        "runs_self": step2_runs_self,
        "tests_edits": step3_tests_edits,
        "reads_states": step4_reads_states,
        "acts_on_tests": step5_acts_on_tests,
        "improves_heldout": step6_improves_heldout,
    }
    return {
        "ladder_score": sum(1 for v in steps.values() if v),
        "max_score": 6,
        "steps": steps,
    }
