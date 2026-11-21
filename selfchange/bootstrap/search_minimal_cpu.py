from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from selfchange.bootstrap.data_quality import (
    build_finetune_dataset,
    compute_eval_steered_weights,
    review_and_record_dataset,
)
from selfchange.bootstrap.eval_baseline import evaluate_baseline
from selfchange.bootstrap.prune_reallocate import prune_mlp_width
from selfchange.kernel.budgets import get_profile
from selfchange.kernel.integrity import atomic_write_text
from selfchange.training.model_loader import estimate_minimind_params, save_model_config
from selfchange.training.self_train_runner import run_training_stage

_RAW_LADDER_SPECS: list[tuple[str, int, int, int, int, int, str]] = [
    ("L0_teacher_64m", 768, 8, 2432, 8, 4, "Domain-specialized Colab teacher"),
    ("L1_pruned_42m", 768, 8, 1216, 8, 4, "-50% MLP width slice"),
    ("L2_compact_30m", 512, 8, 1664, 8, 4, "Conservative distilled student"),
    ("L3_slim_mlp_22m", 512, 8, 1024, 8, 4, "High-headroom CPU candidate"),
    ("L4_deep_narrow_17m", 384, 8, 1216, 6, 3, "Balanced CPU thinking candidate"),
    ("L5_deep_slim_13m", 384, 8, 768, 6, 3, "Primary CPU target (~10-15M band)"),
    ("L6_tiny_deep_10m", 384, 6, 768, 6, 3, "Minimum 6-layer CPU floor probe"),
    ("L7_ultra_tiny_7m", 256, 8, 640, 4, 2, "Cache-resident extreme floor probe"),
]


def _build_ladder() -> list[dict]:
    items: list[dict] = []
    for rung_id, hs, nl, inter, nh, nkv, note in _RAW_LADDER_SPECS:
        info = estimate_minimind_params(
            hidden_size=hs,
            num_hidden_layers=nl,
            intermediate_size=inter,
            num_attention_heads=nh,
            num_key_value_heads=nkv,
        )
        items.append({"rung_id": rung_id, "role": note, **info})
    return items


MINIMAL_CPU_LADDER: list[dict] = _build_ladder()
_LADDER_BY_ID: dict[str, dict] = {r["rung_id"]: r for r in MINIMAL_CPU_LADDER}

MINIMUM_VIABLE_CPU_FLOOR: dict[str, float] = {
    "min_eos_rate": 0.90,
    "min_json_parse_rate": 0.90,
    "min_code_pass_at_1": 0.30,
    "min_code_pass_at_4": 0.60,
    "min_think_grounding_rate": 0.80,
    "min_think_action_consistency_rate": 0.85,
    "min_drills_pass_rate": 0.70,
}


def check_minimal_cpu_floor(scorecard: dict, thresholds: dict[str, float] | None = None) -> tuple[bool, list[str]]:
    """Verify whether a compressed student passes the Minimum Viable Thinking & Self-Change Floor."""
    limits = {**MINIMUM_VIABLE_CPU_FLOOR, **(thresholds or {})}
    t1 = scorecard.get("tier1_language", {})
    t2 = scorecard.get("tier2_toolcall", {})
    t3 = scorecard.get("tier3_code_edit", {})
    t35 = scorecard.get("tier3_5_thinking", {})
    t4 = scorecard.get("tier4_drills", {})

    failed: list[str] = []
    if float(t1.get("eos_rate", 0.0)) < limits["min_eos_rate"]:
        failed.append(f"eos_rate={t1.get('eos_rate', 0.0)} < {limits['min_eos_rate']}")
    if float(t2.get("json_parse_rate", 0.0)) < limits["min_json_parse_rate"]:
        failed.append(f"json_parse_rate={t2.get('json_parse_rate', 0.0)} < {limits['min_json_parse_rate']}")
    if float(t3.get("pass_at_1", 0.0)) < limits["min_code_pass_at_1"]:
        failed.append(f"pass_at_1={t3.get('pass_at_1', 0.0)} < {limits['min_code_pass_at_1']}")
    if float(t3.get("pass_at_4", 0.0)) < limits["min_code_pass_at_4"]:
        failed.append(f"pass_at_4={t3.get('pass_at_4', 0.0)} < {limits['min_code_pass_at_4']}")
    if float(t35.get("observation_grounding_rate", 0.0)) < limits["min_think_grounding_rate"]:
        failed.append(f"think_grounding={t35.get('observation_grounding_rate', 0.0)} < {limits['min_think_grounding_rate']}")
    if float(t35.get("think_action_consistency_rate", 0.0)) < limits["min_think_action_consistency_rate"]:
        failed.append(f"think_consistency={t35.get('think_action_consistency_rate', 0.0)} < {limits['min_think_action_consistency_rate']}")
    if float(t4.get("pass_rate", 0.0)) < limits["min_drills_pass_rate"]:
        failed.append(f"drills_pass_rate={t4.get('pass_rate', 0.0)} < {limits['min_drills_pass_rate']}")
    return len(failed) == 0, failed


def select_next_bracket_rung(results_by_id: dict[str, dict], ladder: list[dict] = MINIMAL_CPU_LADDER) -> dict | None:
    """Choose the next rung using binary/bracket search: L2 -> L5 -> (down to L6/L7 or up to L3/L4 or L1)."""
    by_id = {r["rung_id"]: r for r in ladder}

    def _pick(rid: str) -> dict | None:
        return by_id.get(rid) if rid not in results_by_id else None

    if "L2_compact_30m" not in results_by_id:
        return _pick("L2_compact_30m")

    l2_ok = bool(results_by_id["L2_compact_30m"].get("passed_floor"))
    if not l2_ok:
        return _pick("L1_pruned_42m")

    if "L5_deep_slim_13m" not in results_by_id:
        return _pick("L5_deep_slim_13m")

    if bool(results_by_id["L5_deep_slim_13m"].get("passed_floor")):
        if "L6_tiny_deep_10m" not in results_by_id:
            return _pick("L6_tiny_deep_10m")
        if bool(results_by_id["L6_tiny_deep_10m"].get("passed_floor")):
            return _pick("L7_ultra_tiny_7m")
        return None

    if "L3_slim_mlp_22m" not in results_by_id:
        return _pick("L3_slim_mlp_22m")
    if bool(results_by_id["L3_slim_mlp_22m"].get("passed_floor")):
        return _pick("L4_deep_narrow_17m")
    return None


def _candidate_quality_score(entry: dict) -> tuple[float, float]:
    sc = entry.get("scorecard", {})
    t2 = sc.get("tier2_toolcall", {})
    t3 = sc.get("tier3_code_edit", {})
    t35 = sc.get("tier3_5_thinking", {})
    t5 = sc.get("tier5_hardware", {})
    q = (
        0.25 * float(t2.get("schema_validity_rate", 0.0))
        + 0.35 * float(t3.get("pass_at_1", 0.0))
        + 0.20 * float(t35.get("observation_grounding_rate", 0.0))
        + 0.20 * float(t35.get("think_action_consistency_rate", 0.0))
    )
    return round(q, 4), float(t5.get("cpu_decode_tok_s", 0.0))


def select_pareto_champion(results: list[dict]) -> dict | None:
    """Return the smallest candidate clearing MINIMUM_VIABLE_CPU_FLOOR (tie-break by quality, CPU tok/s)."""
    passing = [r for r in results if r.get("passed_floor")]
    if not passing:
        return None
    return min(
        passing,
        key=lambda r: (
            int(r["arch"]["total_params"]),
            -_candidate_quality_score(r)[0],
            -_candidate_quality_score(r)[1],
        ),
    )


def _format_frontier_md(report: dict) -> str:
    lines = [
        "# Minimal-CPU Pareto Search Report",
        f"- **Profile**: `{report.get('profile')}` | **Dry Run**: `{report.get('dry_run')}` | **Evaluated Rungs**: `{len(report.get('results', []))}`",
    ]
    champ = report.get("pareto_champion")
    if champ:
        a = champ["arch"]
        lines.append(
            f"- **Pareto Champion**: `{champ['rung_id']}` (`{a['total_params'] / 1e6:.2f}M` params, "
            f"`H={a['hidden_size']}, L={a['num_hidden_layers']}, I={a['intermediate_size']}`, FP32 `{a['fp32_mb']}MB` / INT8 `{a['int8_mb']}MB`)"
        )
    else:
        lines.append("- **Pareto Champion**: `NONE` (no probed rung cleared Minimum Viable CPU Floor)")
    lines.extend(["", "| Rung | Params | Pass@1 / Pass@4 | Think Ground / Consist | Floor |", "|---|---:|---|---|---|"])
    for r in report.get("results", []):
        a = r["arch"]
        sc = r.get("scorecard", {})
        t3 = sc.get("tier3_code_edit", {})
        t35 = sc.get("tier3_5_thinking", {})
        lines.append(
            f"| `{r['rung_id']}` | `{a['total_params'] / 1e6:.2f}M` | "
            f"`{t3.get('pass_at_1')}` / `{t3.get('pass_at_4')}` | "
            f"`{t35.get('observation_grounding_rate')}` / `{t35.get('think_action_consistency_rate')}` | "
            f"`{'PASS' if r.get('passed_floor') else 'FAIL'}` |"
        )
    return "\n".join(lines) + "\n"


def run_minimal_cpu_search(
    repo_root: Path,
    workspace_dir: Path,
    teacher_weight: str = "full_sft",
    teacher_weight_path: Path | None = None,
    profile_name: str | None = None,
    explicit_rungs: list[str] | None = None,
    sft_count: int = 400,
    dpo_count: int = 80,
    heldout_count: int = 40,
    max_rungs: int = 4,
    quantize_int8_eval: bool = False,
    dry_run: bool = False,
    seed: int = 42,
) -> dict:
    """Bracket-search compressed student architectures from a 64M teacher and promote the CPU Pareto champion."""
    repo_root = Path(repo_root).resolve()
    workspace_dir = Path(workspace_dir).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    profile = get_profile(profile_name)

    data_dir = workspace_dir / "search_data"
    gen_rep = build_finetune_dataset(
        repo_root=repo_root,
        output_dir=data_dir,
        sft_count=sft_count,
        dpo_count=dpo_count,
        heldout_count=heldout_count,
        max_seq_len=max(512, profile.max_seq_len),
        seed=seed,
    )
    sft_path = Path(gen_rep["sft_path"])
    dpo_path = Path(gen_rep["dpo_path"])
    heldout_path = Path(gen_rep["heldout_path"])
    sft_review = review_and_record_dataset(sft_path, workspace_dir, dataset_format="sft", max_seq_len=max(512, profile.max_seq_len), heldout_path=heldout_path, cycle_tag="cpu_search_sft")
    dpo_review = review_and_record_dataset(dpo_path, workspace_dir, dataset_format="dpo", max_seq_len=max(512, profile.max_seq_len), heldout_path=heldout_path, cycle_tag="cpu_search_dpo") if dpo_path.is_file() and dpo_count > 0 else {"passed": True}
    if not sft_review.get("passed"):
        report = {
            "profile": profile.name,
            "dry_run": dry_run,
            "data_audit_sft_passed": False,
            "data_audit_dpo_passed": bool(dpo_review.get("passed")),
            "results": [],
            "pareto_champion": None,
            "promoted_latest_champion_json": None,
            "failed_checks": sft_review.get("failed_checks", []),
        }
        atomic_write_text(workspace_dir / "search_frontier_report.json", json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        return report

    teacher_arch = dict(_LADDER_BY_ID["L0_teacher_64m"])
    results_by_id: dict[str, dict] = {}
    ordered_results: list[dict] = []
    explicit_queue = [dict(_LADDER_BY_ID[rid]) for rid in (explicit_rungs or []) if rid in _LADDER_BY_ID]

    for step_idx in range(max_rungs):
        if explicit_rungs is not None:
            rung = explicit_queue[step_idx] if step_idx < len(explicit_queue) else None
        else:
            rung = select_next_bracket_rung(results_by_id)
        if rung is None:
            break

        rid = rung["rung_id"]
        cand_save = f"search_{rid}"
        distill_res = None
        dpo_res = None
        out_weight_path: str | None = None

        if rid == "L1_pruned_42m" and not dry_run and teacher_weight_path and Path(teacher_weight_path).is_file():
            pruned_out = workspace_dir / "models" / "candidates" / f"{cand_save}_pruned_{rung['hidden_size']}.pth"
            prune_mlp_width(
                Path(teacher_weight_path),
                repo_root / "model",
                pruned_out,
                target_intermediate_size=int(rung["intermediate_size"]),
                profile_name=profile.name,
            )
            distill_res = run_training_stage(
                stage="full_sft",
                repo_root=repo_root,
                workspace_dir=workspace_dir,
                data_path=sft_path,
                profile=profile,
                training_cfg={"epochs": 1, "learning_rate": 2e-5, "batch_size": profile.batch_size, "max_seq_len": profile.max_seq_len, "heldout_path": str(heldout_path), "model_config": rung},
                from_weight=f"{cand_save}_pruned",
                save_weight=cand_save,
                dry_run=dry_run,
                from_weight_path=pruned_out,
                model_config=rung,
            )
        else:
            distill_res = run_training_stage(
                stage="distill",
                repo_root=repo_root,
                workspace_dir=workspace_dir,
                data_path=sft_path,
                profile=profile,
                training_cfg={
                    "epochs": 1,
                    "learning_rate": 5e-5,
                    "batch_size": profile.batch_size,
                    "max_seq_len": profile.max_seq_len,
                    "heldout_path": str(heldout_path),
                    "enforce_data_gate": bool((not dry_run) and profile.max_seq_len >= 512 and sft_review.get("passed")),
                    "model_config": rung,
                    "teacher_config": teacher_arch,
                    "from_student_weight": "none",
                    "from_teacher_weight": teacher_weight,
                    "teacher_weight_path": str(teacher_weight_path) if teacher_weight_path else None,
                    "distill_alpha": 0.5,
                    "distill_temperature": 1.5,
                },
                from_weight="none",
                save_weight=cand_save,
                dry_run=dry_run,
                model_config=rung,
            )

        out_weight_path = distill_res.output_weight_path if distill_res else None
        if distill_res and distill_res.ok and out_weight_path and dpo_count > 0 and dpo_review.get("passed"):
            dpo_save = f"{cand_save}_dpo"
            dpo_res = run_training_stage(
                stage="dpo",
                repo_root=repo_root,
                workspace_dir=workspace_dir,
                data_path=dpo_path,
                profile=profile,
                training_cfg={"epochs": 1, "dpo_learning_rate": 4e-8, "dpo_beta": 0.15, "batch_size": max(1, profile.batch_size // 2), "max_seq_len": profile.max_seq_len, "heldout_path": str(heldout_path), "enforce_data_gate": False, "model_config": rung},
                from_weight=cand_save,
                save_weight=dpo_save,
                dry_run=dry_run,
                from_weight_path=Path(out_weight_path),
                model_config=rung,
            )
            if dpo_res.ok and dpo_res.output_weight_path:
                out_weight_path = dpo_res.output_weight_path

        eval_backend = "scripted" if dry_run else "minimind"
        scorecard = evaluate_baseline(
            worktree_dir=repo_root,
            backend=eval_backend,
            model_path=Path(out_weight_path) if (eval_backend == "minimind" and out_weight_path) else None,
            tokenizer_path=repo_root / "model",
            profile_name=profile.name,
            heldout_path=heldout_path,
            num_code_tasks=8,
            drill_seeds=[42],
            model_config=rung,
            quantize_int8=quantize_int8_eval,
        )
        passed_floor, failed_floor = check_minimal_cpu_floor(scorecard)
        entry = {
            "rung_id": rid,
            "arch": rung,
            "train_ok": bool(distill_res and distill_res.ok),
            "dpo_ok": bool(dpo_res.ok) if dpo_res else None,
            "weight_path": out_weight_path,
            "scorecard": scorecard,
            "passed_floor": passed_floor,
            "failed_floor": failed_floor,
            "steered_next_batch": compute_eval_steered_weights(scorecard),
        }
        results_by_id[rid] = entry
        ordered_results.append(entry)

    champion = select_pareto_champion(ordered_results)
    promoted_meta_path: str | None = None
    if champion and champion.get("weight_path"):
        prom_dir = workspace_dir / "promoted_models"
        prom_dir.mkdir(parents=True, exist_ok=True)
        src_w = Path(champion["weight_path"])
        if src_w.is_file():
            dst_w = prom_dir / src_w.name
            shutil.copy2(src_w, dst_w)
            cfg_p = save_model_config(prom_dir / "model_config.json", champion["arch"])
            save_model_config(dst_w.with_suffix(".config.json"), champion["arch"])
            latest_meta = {
                "champion_rung_id": champion["rung_id"],
                "champion_weight_path": str(dst_w),
                "model_config_path": str(cfg_p),
                "arch": champion["arch"],
                "passed_floor": True,
                "dry_run": dry_run,
            }
            atomic_write_text(prom_dir / "latest_champion.json", json.dumps(latest_meta, indent=2) + "\n")
            promoted_meta_path = str(prom_dir / "latest_champion.json")

    report = {
        "profile": profile.name,
        "dry_run": dry_run,
        "data_audit_sft_passed": bool(sft_review.get("passed")),
        "data_audit_dpo_passed": bool(dpo_review.get("passed")),
        "results": ordered_results,
        "pareto_champion": champion,
        "promoted_latest_champion_json": promoted_meta_path,
    }
    atomic_write_text(workspace_dir / "search_frontier_report.json", json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    atomic_write_text(workspace_dir / "search_frontier_report.md", _format_frontier_md(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Colab bracket search for the minimal CPU selfchange thinking model")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--workspace-dir", type=Path, default=Path("selfchange/workspace"))
    parser.add_argument("--teacher-weight", type=str, default="full_sft")
    parser.add_argument("--teacher-weight-path", type=Path, default=None)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--rungs", type=str, default=None, help="Optional comma-separated explicit rung IDs")
    parser.add_argument("--sft-count", type=int, default=400)
    parser.add_argument("--dpo-count", type=int, default=80)
    parser.add_argument("--heldout-count", type=int, default=40)
    parser.add_argument("--max-rungs", type=int, default=4)
    parser.add_argument("--quantize-int8-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    explicit = [r.strip() for r in args.rungs.split(",") if r.strip()] if args.rungs else None
    rep = run_minimal_cpu_search(
        repo_root=args.repo_root,
        workspace_dir=args.workspace_dir,
        teacher_weight=args.teacher_weight,
        teacher_weight_path=args.teacher_weight_path,
        profile_name=args.profile,
        explicit_rungs=explicit,
        sft_count=args.sft_count,
        dpo_count=args.dpo_count,
        heldout_count=args.heldout_count,
        max_rungs=args.max_rungs,
        quantize_int8_eval=args.quantize_int8_eval,
        dry_run=args.dry_run,
        seed=args.seed,
    )
    print(json.dumps(rep, indent=2))
    return 0 if rep.get("pareto_champion") is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
