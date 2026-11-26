from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from selfrewrite.backends.mock_backend import MockFailureEditor, MockTargetBackend, MockTransformerModel
from selfrewrite.colab.colab_launcher import _evaluate_on_split, _make_target_for_repair, _prepare_splits
from selfrewrite.colab.drive_sync import DriveMirrorSync
from selfrewrite.config import DOWNSIZE_RUNGS, SessionConfig, atomic_write_text, get_profile
from selfrewrite.distill.programmatic_teacher import ProgrammaticImproveEditor, ProgrammaticRepairEditor
from selfrewrite.distill.train_student import StudentTrainConfig, train_downsize_student, verify_bundle_audit_gate
from selfrewrite.harvest.build_bundle import build_distill_bundle
from selfrewrite.harvest.collector import SessionHarvester
from selfrewrite.session import SessionOrchestrator
from selfrewrite.stats import score_emergence_ladder
from selfrewrite.tasks.corruption import split_train_eval_layers


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _release_torch_cache() -> None:
    gc.collect()
    gc.collect()
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _destroy_backend(obj: Any) -> None:
    """Break circular references inside target/damaged backends so CUDA weights free immediately."""
    if obj is None:
        return
    overlay = getattr(obj, "overlay", None)
    if overlay is not None:
        if hasattr(overlay, "_param_backups") and isinstance(overlay._param_backups, dict):
            overlay._param_backups.clear()
        if hasattr(overlay, "_module_backups") and isinstance(overlay._module_backups, dict):
            overlay._module_backups.clear()
        try:
            overlay.model = None
        except Exception:
            pass
    creader = getattr(obj, "canonical_reader", None)
    if creader is not None:
        for attr in ("_model", "_state_dict"):
            if hasattr(creader, attr):
                try:
                    setattr(creader, attr, None)
                except Exception:
                    pass
    layers = getattr(obj, "_layers", None)
    if isinstance(layers, list):
        layers.clear()
    for attr in ("editor_model", "model", "_ov_t", "_inner"):
        if hasattr(obj, attr):
            try:
                setattr(obj, attr, None)
            except Exception:
                pass


def _load_rung_backend(rung_id: str, model_id_or_path: str, profile: Any) -> tuple[Any, dict[str, Any], int]:
    """Instantiate MiniMindBackend for R4m or HFGPUBackend for Qwen3.5/Qwen3 rungs."""
    target_str = str(model_id_or_path)
    if rung_id == "R4m" or Path(target_str).name.startswith("minimind") or target_str.startswith("minimind"):
        from selfrewrite.backends.minimind_backend import MiniMindBackend

        p_obj = Path(target_str)
        w_path = (p_obj / "pytorch_model.bin") if p_obj.is_dir() else (p_obj if p_obj.is_file() else None)
        tok_dir = Path(__file__).resolve().parents[2] / "model"
        mm = MiniMindBackend.from_checkpoint(
            weight_path=w_path,
            tokenizer_dir=tok_dir,
            profile=profile,
            hidden_size=512,
            num_hidden_layers=8,
        )
        return mm, {"_mm_backend": mm}, mm.num_layers

    from selfrewrite.backends.hf_gpu_backend import HFGPUBackend

    hf = HFGPUBackend.from_pretrained(target_str, profile=profile)
    return hf, {"_hf_backend": hf}, hf.num_layers


@dataclass(frozen=True)
class PromotionGateConfig:
    """Criteria from plan.md Sec. 5.3 for promoting a frozen downsized student rung."""

    min_ladder_score: int = 6
    min_repair_recovery_ratio: float = 0.40
    max_repair_p: float = 0.02
    require_positive_transfer: bool = True
    max_improve_p: float = 0.05
    max_harmful_submit_drop: float = 0.01
    max_base_forgetting: float = 0.02


def evaluate_promotion_gates(rung_result: dict[str, Any], gates: PromotionGateConfig = PromotionGateConfig()) -> dict[str, Any]:
    """Evaluate frozen self-rewriting capability, unseen repair, and retention gates."""
    base_acc = float(rung_result.get("base_unmodified_acc", 0.0))
    student_unedited_acc = float(rung_result.get("student_unedited_acc", 0.0))
    forgetting = base_acc - student_unedited_acc

    improve_runs = list(rung_result.get("improve_sessions", []))
    repair_run = rung_result.get("repair_session") or {}

    best_ladder = max([int(r.get("ladder_score", 0)) for r in [*improve_runs, repair_run]], default=0)
    ladder_ok = best_ladder >= gates.min_ladder_score

    submitted_improve_deltas = [float(r["heldout_delta"]) for r in improve_runs if r.get("decision") == "SUBMIT"]
    no_harmful_submit = all(d >= -gates.max_harmful_submit_drop for d in submitted_improve_deltas)
    sig_improve = any(
        r.get("decision") == "SUBMIT"
        and float(r.get("heldout_delta", 0.0)) > 0.0
        and min(float(r.get("perm_p", 1.0)), float(r.get("sign_p", 1.0))) <= gates.max_improve_p
        for r in improve_runs
    )
    improve_ok = no_harmful_submit and (sig_improve or bool(improve_runs and not submitted_improve_deltas))

    rep_edit_acc = float(repair_run.get("heldout_edit_acc", 0.0))
    rep_p = min(float(repair_run.get("perm_p", 1.0)), float(repair_run.get("sign_p", 1.0)))
    rep_recovery_ratio = rep_edit_acc / max(1e-6, base_acc if base_acc > 0 else student_unedited_acc)
    rep_transfer = float(repair_run.get("transfer_delta", 0.0))
    repair_ok = (
        repair_run.get("decision") == "SUBMIT"
        and rep_p <= gates.max_repair_p
        and rep_recovery_ratio >= gates.min_repair_recovery_ratio
        and (not gates.require_positive_transfer or rep_transfer > 0.0)
    )

    retention_ok = forgetting <= gates.max_base_forgetting

    checks = {
        "ladder_gate": ladder_ok,
        "improve_gate": improve_ok,
        "repair_gate": repair_ok,
        "retention_gate": retention_ok,
    }
    reasons: list[str] = []
    if not ladder_ok:
        reasons.append(f"ladder_score {best_ladder} < {gates.min_ladder_score}")
    if not improve_ok:
        reasons.append("improve sessions failed significance or harmed held-out")
    if not repair_ok:
        reasons.append(f"repair recovery={rep_recovery_ratio:.2f} p={rep_p:.4g} transfer={rep_transfer:+.3f}")
    if not retention_ok:
        reasons.append(f"base_forgetting {forgetting:.3f} > {gates.max_base_forgetting}")

    return {
        "promoted": all(checks.values()),
        "gates": checks,
        "best_ladder_score": best_ladder,
        "repair_recovery_ratio": round(rep_recovery_ratio, 4),
        "base_forgetting": round(forgetting, 4),
        "reasons": reasons,
    }


def _baseline_strategy_for_rung(rung_id: str) -> str:
    if rung_id == "R1":
        return "bad_submit_9b"
    if rung_id == "R2":
        return "random_noise_4b"
    return "no_edit_2b"


def run_downsize_bracket(
    workspace_dir: Path,
    bundle_dir: Path,
    rungs: Iterable[str] = ("R1", "R2", "R3", "R4"),
    backend_kind: str = "mock",
    profile_name: str | None = None,
    drive_dir: Path | None = None,
    improve_sessions: int = 2,
    starting_diagnosis_n: int | None = None,
    answer_max_tokens: int | None = None,
    editor_max_new_tokens: int | None = None,
    max_turns: int | None = None,
    rlvr_steps: int = 8,
    cleanup_student_weights: bool = False,
    bootstrap_next_rung: bool = False,
    seed: int = 2026,
    gates: PromotionGateConfig = PromotionGateConfig(),
) -> dict[str, Any]:
    """Train & frozen-evaluate students down the parameter ladder to find the smallest 6/6 rung."""
    ws = Path(workspace_dir).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    active_bundle = Path(bundle_dir).resolve()
    verify_bundle_audit_gate(active_bundle)
    profile = get_profile(profile_name)
    sync = DriveMirrorSync(ws, drive_dir)
    splits = _prepare_splits(ws, use_deepscaler=(backend_kind == "hf"), seed=seed)

    rung_entries: list[dict[str, Any]] = []
    completed_rungs: list[str] = []
    raw_root = ws / "raw_sessions"

    for r_idx, rung_id in enumerate(rungs):
        if rung_id not in DOWNSIZE_RUNGS:
            raise KeyError(f"Unknown downsize rung {rung_id!r}")
        spec = DOWNSIZE_RUNGS[rung_id]
        rung_ws = ws / "bracket" / rung_id
        rung_ws.mkdir(parents=True, exist_ok=True)
        rung_res_path = rung_ws / "rung_result.json"
        if rung_res_path.is_file() and (rung_ws / "frozen_student" / "student_provenance.json").is_file():
            entry = json.loads(rung_res_path.read_text(encoding="utf-8"))
            rung_entries.append(entry)
            completed_rungs.append(rung_id)
            atomic_write_text(
                ws / "HEARTBEAT.json",
                json.dumps(
                    {
                        "updated_at": _utc_now(),
                        "profile": profile.name,
                        "status": "running_bracket",
                        "current_rung": rung_id,
                        "completed_rungs": list(completed_rungs),
                    },
                    indent=2,
                )
                + "\n",
            )
            continue
        atomic_write_text(
            ws / "HEARTBEAT.json",
            json.dumps(
                {
                    "updated_at": _utc_now(),
                    "profile": profile.name,
                    "status": "running_bracket",
                    "current_rung": rung_id,
                    "completed_rungs": list(completed_rungs),
                },
                indent=2,
            )
            + "\n",
        )

        if (
            backend_kind != "mock"
            and spec.eval_quantization == "none"
            and profile.vram_gb >= int(spec.params_b * 2.5 + 4)
            and profile.quantization != "none"
        ):
            rung_profile = replace(profile, quantization="none")
        else:
            rung_profile = profile

        # 1. Establish base target & baseline ladder before student specialization
        if backend_kind == "mock":
            mock_model = MockTransformerModel(num_layers=8, hidden_dim=16, inter_dim=24, verbosity_layer=2, seed=7)
            base_target: Any = MockTargetBackend(mock_model, damage_mode=False)
            num_layers = mock_model.num_layers
            clean_ctx = (base_target, dict(mock_model.raw_tensors), num_layers)
        else:
            clean_ctx = _load_rung_backend(rung_id, spec.model_id, rung_profile)
            base_target, _, num_layers = clean_ctx

        train_layers, eval_layers = split_train_eval_layers(num_layers)
        eff_max_tokens = (
            int(answer_max_tokens)
            if answer_max_tokens is not None
            else min(rung_profile.answer_max_tokens, 512 if backend_kind == "mock" else rung_profile.answer_max_tokens)
        )
        eff_diag_n = min(
            len(splits["practice_eval"]),
            int(starting_diagnosis_n) if starting_diagnosis_n is not None else rung_profile.starting_diagnosis_n,
        )
        eff_res_n = min(
            len(splits["reserved_eval"]),
            int(starting_diagnosis_n) if starting_diagnosis_n is not None else rung_profile.starting_diagnosis_n,
        )
        eff_max_turns = int(max_turns) if max_turns is not None else 160
        eff_editor_max_new_tokens = int(editor_max_new_tokens) if editor_max_new_tokens is not None else 12000

        eval_cfg = SessionConfig(
            task="improve",
            protocol=2,
            max_turns=eff_max_turns,
            answer_max_tokens=eff_max_tokens,
            editor_max_new_tokens=eff_editor_max_new_tokens,
            starting_diagnosis_n=eff_diag_n,
            reserved_check_n=eff_res_n,
            reserved_check_samples=1 if backend_kind == "mock" else 2,
            model_arch_summary=base_target.arch_summary,
            session_id=f"{rung_id}_base_audit",
            seed=seed + r_idx,
        )
        base_unmodified_acc = float(_evaluate_on_split(base_target, None, splits["heldout_eval"], eval_cfg)["base_acc"])

        base_ed: Any = MockFailureEditor(_baseline_strategy_for_rung(rung_id)) if backend_kind == "mock" else base_target
        base_orch = SessionOrchestrator(
            work_dir=rung_ws / "base_audit" / "WORK",
            target=base_target,
            editor=base_ed,
            practice_problems=splits["practice_eval"],
            reserved_problems=splits["reserved_eval"],
            config=eval_cfg,
        )
        base_audit_res = base_orch.run()
        if backend_kind != "mock":
            _destroy_backend(base_target)
            del base_ed, base_orch, base_target, clean_ctx
            _release_torch_cache()

        # 2. Offline Student Specialization from verified bundle -> Merge & Freeze
        student_out = rung_ws / "frozen_student"
        train_cfg = StudentTrainConfig(
            rung_id=rung_id,
            base_model_id=spec.model_id,
            bundle_dir=active_bundle,
            output_dir=student_out,
            splits_dir=ws / "splits",
            backend=backend_kind,
            profile_name=rung_profile.name,
            run_rlvr=("rlvr" in spec.train_strategy),
            rlvr_steps=int(rlvr_steps),
            seed=seed + r_idx,
        )
        train_meta = train_downsize_student(train_cfg)

        # Load frozen student checkpoint cleanly after training memory is released
        if backend_kind == "mock":
            student_target: Any = MockTargetBackend(mock_model, damage_mode=False)
            student_clean_ctx = (student_target, dict(mock_model.raw_tensors), num_layers)
        else:
            _release_torch_cache()
            student_clean_ctx = _load_rung_backend(rung_id, str(student_out), rung_profile)
            student_target = student_clean_ctx[0]

        student_unedited_acc = float(_evaluate_on_split(student_target, None, splits["heldout_eval"], eval_cfg)["base_acc"])

        # 3. Frozen Zero-Gradient Evaluation: Protocol 2 improve + unseen-layer repair
        # In mock mode, R4 models capacity floor exhaustion (< 2B threshold).
        hits_floor = backend_kind == "mock" and rung_id == "R4"
        improve_summaries: list[dict[str, Any]] = []
        for s_idx in range(improve_sessions):
            sid = f"bracket_{rung_id}_improve_s{s_idx:02d}"
            s_cfg = SessionConfig(
                task="improve",
                protocol=2,
                max_turns=eff_max_turns,
                answer_max_tokens=eval_cfg.answer_max_tokens,
                editor_max_new_tokens=eff_editor_max_new_tokens,
                starting_diagnosis_n=eval_cfg.starting_diagnosis_n,
                reserved_check_n=eval_cfg.reserved_check_n,
                reserved_check_samples=eval_cfg.reserved_check_samples,
                model_arch_summary=student_target.arch_summary,
                session_id=sid,
                seed=seed + 100 * (r_idx + 1) + s_idx,
            )
            if backend_kind == "mock":
                imp_ed: Any = MockFailureEditor("random_noise_4b") if hits_floor else ProgrammaticImproveEditor(alpha=0.25 if s_idx == 0 else 0.30)
            else:
                imp_ed = student_target
            orch = SessionOrchestrator(
                work_dir=rung_ws / sid / "WORK",
                target=student_target,
                editor=imp_ed,
                practice_problems=splits["practice_eval"],
                reserved_problems=splits["reserved_eval"],
                config=s_cfg,
            )
            res = orch.run()
            ed_t = orch._host_edit_store.load_edit_tensors(target_torch=(backend_kind != "mock")) if res["decision"] == "SUBMIT" else None
            held = _evaluate_on_split(student_target, ed_t, splits["heldout_eval"], s_cfg, seed_base=9100 + s_idx)
            lad = score_emergence_ladder(res, heldout_eval=held)
            improve_summaries.append(
                {
                    "session_id": sid,
                    "decision": res["decision"],
                    "ladder_score": lad["ladder_score"],
                    "heldout_delta": held["delta_hat"],
                    "perm_p": held["perm_p"],
                    "sign_p": held["sign_p"],
                }
            )
            del orch, imp_ed, ed_t
            _release_torch_cache()

        # Unseen eval_layers repair session
        rep_layer = eval_layers[len(eval_layers) // 2]
        rep_hidden = rung_ws / "repair_eval" / "hidden"
        dmg_target, _, dmg_spec = _make_target_for_repair(backend_kind, student_clean_ctx, rep_layer, rep_hidden, seed=seed + r_idx)
        r_cfg = SessionConfig(
            task="repair",
            protocol=2,
            max_turns=eff_max_turns,
            answer_max_tokens=eval_cfg.answer_max_tokens,
            editor_max_new_tokens=eff_editor_max_new_tokens,
            starting_diagnosis_n=eval_cfg.starting_diagnosis_n,
            reserved_check_n=eval_cfg.reserved_check_n,
            reserved_check_samples=eval_cfg.reserved_check_samples,
            model_arch_summary=dmg_target.arch_summary,
            session_id=f"bracket_{rung_id}_repair",
            seed=seed + 500 + r_idx,
        )
        rep_ed: Any = (MockFailureEditor("random_noise_4b") if hits_floor else ProgrammaticRepairEditor()) if backend_kind == "mock" else dmg_target
        rep_orch = SessionOrchestrator(
            work_dir=rung_ws / "repair_eval" / "WORK",
            target=dmg_target,
            editor=rep_ed,
            practice_problems=splits["practice_eval"],
            reserved_problems=splits["reserved_eval"],
            config=r_cfg,
        )
        rep_res = rep_orch.run()
        rep_t = rep_orch._host_edit_store.load_edit_tensors(target_torch=(backend_kind != "mock")) if rep_res["decision"] == "SUBMIT" else None
        rep_math = _evaluate_on_split(dmg_target, rep_t, splits["heldout_eval"], r_cfg, seed_base=9400 + r_idx)
        rep_trans = _evaluate_on_split(dmg_target, rep_t, splits["transfer_eval"], r_cfg, seed_base=9600 + r_idx)
        rep_lad = score_emergence_ladder(rep_res, heldout_eval=rep_math)
        repair_summary = {
            "session_id": r_cfg.session_id,
            "corrupted_eval_layer": rep_layer,
            "decision": rep_res["decision"],
            "ladder_score": rep_lad["ladder_score"],
            "heldout_base_acc": rep_math["base_acc"],
            "heldout_edit_acc": rep_math["edit_acc"],
            "heldout_delta": rep_math["delta_hat"],
            "perm_p": rep_math["perm_p"],
            "sign_p": rep_math["sign_p"],
            "transfer_delta": rep_trans["delta_hat"],
        }

        entry = {
            "rung_id": rung_id,
            "model_id": spec.model_id,
            "params_b": spec.params_b,
            "frozen_for_eval": bool(train_meta.get("frozen_for_eval", False)),
            "base_unmodified_acc": base_unmodified_acc,
            "base_ladder_score": int(base_audit_res["ladder"]["ladder_score"]),
            "student_unedited_acc": student_unedited_acc,
            "improve_sessions": improve_summaries,
            "repair_session": repair_summary,
        }
        entry["promotion"] = evaluate_promotion_gates(entry, gates=gates)
        atomic_write_text(rung_ws / "rung_result.json", json.dumps(entry, indent=2) + "\n")
        rung_entries.append(entry)

        # Optional self-bootstrapping: promoted student harvests fresh disjoint traces for smaller rung
        if bootstrap_next_rung and entry["promotion"]["promoted"] and r_idx + 1 < len(rungs):
            bs_sid = f"bootstrap_{rung_id}_to_next"
            bs_harvester = SessionHarvester(
                raw_root=raw_root,
                session_id=bs_sid,
                meta={"session_id": bs_sid, "split_tag": "distill_train", "task": "improve", "model_id": spec.model_id},
            )
            bs_ed: Any = ProgrammaticImproveEditor() if backend_kind == "mock" else student_target
            distill_p = splits["distill_train"]
            bs_orch = SessionOrchestrator(
                work_dir=rung_ws / bs_sid / "WORK",
                target=student_target,
                editor=bs_ed,
                practice_problems=distill_p[: max(4, len(distill_p) // 2)],
                reserved_problems=distill_p[max(4, len(distill_p) // 2) :],
                config=eval_cfg,
                harvester=bs_harvester,
            )
            bs_orch.run()
            forbidden_texts = [p.question for k in ("heldout_eval", "reserved_eval", "practice_eval") for p in splits[k]]
            bs_bundle = ws / "distill_bundle" / f"v_{rung_id}_bootstrap"
            build_res = build_distill_bundle(raw_root, bs_bundle, forbidden_texts, eval_layers, ("distill_train",))
            if build_res.get("audit", {}).get("passed"):
                active_bundle = bs_bundle

        if backend_kind != "mock":
            _destroy_backend(dmg_target)
            _destroy_backend(student_target)
            del rep_orch, rep_ed, rep_t, dmg_target, student_target, student_clean_ctx
            _release_torch_cache()
            if cleanup_student_weights and student_out.is_dir():
                for sf in student_out.glob("*.safetensors"):
                    sf.unlink(missing_ok=True)
        completed_rungs.append(rung_id)
        atomic_write_text(
            ws / "HEARTBEAT.json",
            json.dumps(
                {
                    "updated_at": _utc_now(),
                    "profile": profile.name,
                    "status": "running_bracket",
                    "current_rung": rung_id,
                    "completed_rungs": list(completed_rungs),
                },
                indent=2,
            )
            + "\n",
        )

    promoted = [e for e in rung_entries if e["promotion"]["promoted"]]
    winner = min(promoted, key=lambda e: float(e["params_b"])) if promoted else None

    summary = {
        "updated_at": _utc_now(),
        "profile": profile.name,
        "backend": backend_kind,
        "bundle_dir": str(active_bundle),
        "rungs": rung_entries,
        "pareto_winner": {
            "rung_id": winner["rung_id"],
            "model_id": winner["model_id"],
            "params_b": winner["params_b"],
        }
        if winner
        else None,
    }
    atomic_write_text(ws / "downsize_leaderboard.json", json.dumps(summary, indent=2) + "\n")

    lines = [
        f"# selfrewrite Downsizing Bracket Report (`backend={backend_kind}`, `profile={profile.name}`)",
        "",
        "| Rung | Model | Params (B) | Base Ladder | Frozen Ladder | Improve Δ | Repair Recovery | Transfer Δ | Promoted |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for e in rung_entries:
        best_imp = max((float(x["heldout_delta"]) for x in e["improve_sessions"]), default=0.0)
        rep = e["repair_session"]
        prom = e["promotion"]
        lines.append(
            f"| {e['rung_id']} | `{e['model_id']}` | {e['params_b']} | {e['base_ladder_score']}/6 | "
            f"{prom['best_ladder_score']}/6 | {best_imp:+.4f} | {prom['repair_recovery_ratio']:.1%} | "
            f"{float(rep['transfer_delta']):+.4f} | {'YES' if prom['promoted'] else 'NO'} |"
        )
    if winner:
        lines.extend(["", f"**Pareto-Smallest Promoted Model**: `{winner['rung_id']}` (`{winner['model_id']}`, `{winner['params_b']}B`)"])
    atomic_write_text(ws / "downsize_report.md", "\n".join(lines) + "\n")

    atomic_write_text(
        ws / "HEARTBEAT.json",
        json.dumps(
            {
                "updated_at": _utc_now(),
                "profile": profile.name,
                "status": "completed_bracket",
                "completed_rungs": list(completed_rungs),
            },
            indent=2,
        )
        + "\n",
    )

    if sync.enabled:
        sync.sync_all()
        if sync.drive_mirror_dir is not None:
            for fname in ("downsize_leaderboard.json", "downsize_report.md"):
                src = ws / fname
                if src.is_file():
                    sync._copy_atomic(src, sync.drive_mirror_dir / fname)

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Downsizing bracket runner over 27B -> 9B -> 4B -> 2B -> MiniMind")
    parser.add_argument("--workspace-dir", default="selfrewrite/workspace")
    parser.add_argument("--bundle-dir", default="selfrewrite/workspace/distill_bundle/v1")
    parser.add_argument("--rungs", default="R1,R2,R3,R4")
    parser.add_argument("--backend", choices=("hf", "mock"), default="hf")
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--drive-dir", default=None)
    parser.add_argument("--improve-sessions", type=int, default=2)
    parser.add_argument("--starting-diagnosis-n", type=int, default=None)
    parser.add_argument("--answer-max-tokens", type=int, default=None)
    parser.add_argument("--editor-max-new-tokens", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--rlvr-steps", type=int, default=8)
    parser.add_argument("--cleanup-student-weights", action="store_true")
    parser.add_argument("--bootstrap-next-rung", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)

    rungs = tuple(r.strip() for r in str(args.rungs).split(",") if r.strip())
    res = run_downsize_bracket(
        workspace_dir=Path(args.workspace_dir),
        bundle_dir=Path(args.bundle_dir),
        rungs=rungs,
        backend_kind=args.backend,
        profile_name=None if args.profile == "auto" else args.profile,
        drive_dir=Path(args.drive_dir) if args.drive_dir else None,
        improve_sessions=args.improve_sessions,
        starting_diagnosis_n=args.starting_diagnosis_n,
        answer_max_tokens=args.answer_max_tokens,
        editor_max_new_tokens=args.editor_max_new_tokens,
        max_turns=args.max_turns,
        rlvr_steps=args.rlvr_steps,
        cleanup_student_weights=bool(args.cleanup_student_weights),
        bootstrap_next_rung=bool(args.bootstrap_next_rung),
        seed=args.seed,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
