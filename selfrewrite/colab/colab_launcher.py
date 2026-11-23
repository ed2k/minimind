from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from selfrewrite.backends.mock_backend import MockFailureEditor, MockTargetBackend, MockTransformerModel
from selfrewrite.colab.drive_sync import DriveMirrorSync
from selfrewrite.config import ComputeProfile, SessionConfig, atomic_write_text, get_profile
from selfrewrite.distill.programmatic_teacher import ProgrammaticImproveEditor, ProgrammaticRepairEditor
from selfrewrite.harvest.build_bundle import build_distill_bundle
from selfrewrite.harvest.collector import SessionHarvester
from selfrewrite.session import SessionOrchestrator
from selfrewrite.stats import compute_paired_stats, score_emergence_ladder
from selfrewrite.tasks.corruption import create_damaged_workspace, split_train_eval_layers
from selfrewrite.tasks.datasets import (
    Problem,
    build_builtin_suites,
    load_deepscaler_splits,
    load_jsonl_problems,
    save_jsonl_problems,
)
from selfrewrite.tasks.grader import grade_response


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _evaluate_on_split(
    target: Any,
    edit_tensors: dict[str, Any] | None,
    probs: list[Problem],
    config: SessionConfig,
    seed_base: int = 9999,
    n_samples: int = 1,
) -> dict[str, Any]:
    base_recs = target.generate_batch(probs, None, n_samples, config.answer_max_tokens, config, seed_base=seed_base)
    by_id = {p.problem_id: p for p in probs}

    def _grade(recs):
        rows = []
        for rec in recs:
            p = by_id[rec.problem_id]
            g = grade_response(p, rec.response_text, rec.num_tokens, config.answer_max_tokens, config.require_thinking)
            rows.append({"problem_id": rec.problem_id, "sample_idx": rec.sample_idx, **g})
        return rows

    base_rows = _grade(base_recs)
    if not edit_tensors:
        stats = compute_paired_stats(base_rows, base_rows, num_flips=2000, perm_seed=seed_base)
        return {k: v for k, v in stats.items() if k != "per_problem_diffs"}
    edit_recs = target.generate_batch(probs, edit_tensors, n_samples, config.answer_max_tokens, config, seed_base=seed_base)
    edit_rows = _grade(edit_recs)
    stats = compute_paired_stats(base_rows, edit_rows, num_flips=10000, perm_seed=seed_base)
    return {k: v for k, v in stats.items() if k != "per_problem_diffs"}


def _prepare_splits(workspace_dir: Path, use_deepscaler: bool, seed: int = 2026) -> dict[str, list[Problem]]:
    splits_dir = workspace_dir / "splits"
    names = ("heldout_eval", "reserved_eval", "practice_eval", "distill_train", "transfer_eval")
    if all((splits_dir / f"{n}.jsonl").is_file() for n in names):
        return {n: load_jsonl_problems(splits_dir / f"{n}.jsonl") for n in names}

    if use_deepscaler:
        splits = load_deepscaler_splits(seed=seed)
        builtin = build_builtin_suites(seed=seed, practice_n=8, reserved_n=8, heldout_n=8, distill_n=8, transfer_n=32)
        splits["transfer_eval"] = builtin["transfer_eval"]
    else:
        splits = build_builtin_suites(
            seed=seed,
            practice_n=28,
            reserved_n=16,
            heldout_n=28,
            distill_n=28,
            transfer_n=16,
        )

    for n, plist in splits.items():
        save_jsonl_problems(splits_dir / f"{n}.jsonl", plist)
    return splits


def _build_backends(
    backend_kind: str,
    model_id: str,
    profile: ComputeProfile,
    shared_hf_backend: Any | None = None,
) -> tuple[Any, dict[str, Any], int]:
    """Instantiate clean target backend & clean weights dictionary for either HF GPU or mock mode."""
    if backend_kind == "mock":
        mock_model = MockTransformerModel(num_layers=8, hidden_dim=16, inter_dim=24, verbosity_layer=2, seed=7)
        target = MockTargetBackend(mock_model, damage_mode=False)
        return target, dict(mock_model.raw_tensors), mock_model.num_layers

    from selfrewrite.backends.hf_gpu_backend import HFGPUBackend

    hf = shared_hf_backend or HFGPUBackend.from_pretrained(model_id=model_id, profile=profile)
    # Only cache down_proj / output matrix keys lazily via reader when corrupting a repair target.
    return hf, {"_hf_backend": hf}, hf.num_layers


def _make_target_for_repair(
    backend_kind: str,
    clean_ctx: tuple[Any, dict[str, Any], int],
    layer_idx: int,
    hidden_dir: Path,
    seed: int,
    sigma_mult: float = 200.0,
    num_rows: int = 8,
    mode: str = "gaussian_noise",
) -> tuple[Any, dict[str, Any], Any]:
    clean_target, raw_map, _ = clean_ctx
    if backend_kind == "mock":
        tname = f"model.layers.{layer_idx}.mlp.down_proj.weight"
        out_dim = len(raw_map[tname])
        eff_rows = min(num_rows, max(1, out_dim // 6))
        damaged_w, spec = create_damaged_workspace(
            raw_map,
            hidden_dir=hidden_dir,
            layer_idx=layer_idx,
            tensor_name=tname,
            num_rows=eff_rows,
            sigma_mult=sigma_mult,
            mode=mode,
            seed=seed,
        )
        damaged_target = MockTargetBackend(clean_target._base_model, target_tensors=damaged_w, damage_mode=True)
        return damaged_target, damaged_w, spec

    # HF GPU Backend: materialize ONLY the single corrupted layer matrix into a CanonicalWeightReader overlay.
    from selfrewrite.backends.hf_gpu_backend import CanonicalWeightReader, HFGPUBackend, HybridWeightOverlay
    from selfrewrite.tasks.corruption import apply_block_corruption, CorruptionSpec

    hf: HFGPUBackend = raw_map["_hf_backend"]
    candidates = [nm for nm in hf.tensor_shapes() if f"layers.{layer_idx}." in nm and nm.endswith("mlp.down_proj.weight")]
    if not candidates:
        candidates = [nm for nm in hf.tensor_shapes() if f"layers.{layer_idx}." in nm and nm.endswith(".down_proj.weight")]
    tname = candidates[0]
    orig_t = hf.canonical_reader.load_tensor(tname)
    out_dim = int(orig_t.shape[0])
    r0 = random.Random(seed).randint(0, max(0, out_dim - num_rows))
    init_spec = CorruptionSpec(layer_idx=layer_idx, tensor_name=tname, row_start=r0, num_rows=min(num_rows, out_dim), sigma_mult=sigma_mult, mode=mode, seed=seed)
    corrupted_t, final_spec = apply_block_corruption(orig_t, init_spec)
    hidden_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(hidden_dir / "damage_manifest.json", json.dumps(final_spec.to_dict(), indent=2) + "\n")

    # Build a target view whose load_original reads canonical except tname -> corrupted_t, and generate runs with corrupted_t overlay!
    class _CorruptedCanonicalReader(CanonicalWeightReader):
        def __init__(self, base_reader: CanonicalWeightReader, override_name: str, override_tensor: Any):
            self._base = base_reader
            self._ov_name = override_name
            self._ov_t = override_tensor

        def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
            return self._base.tensor_shapes()

        def load_tensor(self, name: str) -> Any:
            if name == self._ov_name:
                return self._ov_t.detach().cpu().clone()
            return self._base.load_tensor(name)

        def export_weights_spec(self, dest_dir: Path) -> Path:
            # Materialize damaged tensor shard beside canonical spec safely outside WORK/
            base_spec_path = self._base.export_weights_spec(dest_dir)
            spec = json.loads(base_spec_path.read_text(encoding="utf-8"))
            from safetensors.torch import save_file  # type: ignore[import-not-found]

            dmg_file = Path(dest_dir) / "damaged_override.safetensors"
            save_file({self._ov_name: self._ov_t.detach().cpu()}, str(dmg_file))
            if spec["mode"] == "safetensors_dir":
                spec["weight_map"][self._ov_name] = dmg_file.name
                if Path(spec["base_dir"]) != Path(dest_dir).resolve():
                    # Point relative paths back cleanly or write merged spec: all non-overridden shards keep absolute/relative?
                    # Keep canonical shard copy-free by copying weight_map paths only if same dir; else materialize damaged only.
                    spec["base_dir"] = "/"
                    new_map = {}
                    base_d = Path(self._base.weight_dir) if self._base.weight_dir else Path(dest_dir)
                    for k, rel in spec["weight_map"].items():
                        new_map[k] = str((Path(dest_dir) / dmg_file.name).resolve()).lstrip("/") if k == self._ov_name else str((base_d / rel).resolve()).lstrip("/")
                    spec["weight_map"] = new_map
            atomic_write_text(base_spec_path, json.dumps(spec, indent=2))
            return base_spec_path

    class _DamagedHFTarget(HFGPUBackend):
        def generate_batch(self, problems, edit_tensors, n_samples, max_tokens, config, seed_base=0):
            merged = {tname: corrupted_t}
            if edit_tensors:
                merged.update(edit_tensors)
            return super().generate_batch(problems, merged, n_samples, max_tokens, config, seed_base=seed_base)

        def read_activations(self, texts, layers, target, pooling, k, edit_tensors, max_tokens, pass_id):
            merged = {tname: corrupted_t}
            if edit_tensors:
                merged.update(edit_tensors)
            return super().read_activations(texts, layers, target, pooling, k, merged, max_tokens, pass_id)

    dmg_reader = _CorruptedCanonicalReader(hf.canonical_reader, tname, corrupted_t)
    dmg_target = _DamagedHFTarget(
        model=hf.model,
        tokenizer=hf.tokenizer,
        profile=hf.profile,
        canonical_reader=dmg_reader,
        editor_model=hf.editor_model,
        editor_tokenizer=hf.editor_tokenizer,
        model_id=f"{hf.model_id}-damaged",
    )
    return dmg_target, {}, final_spec


def run_reproduce_and_harvest_pipeline(
    workspace_dir: Path,
    mode: str = "smoke",  # "smoke" | "reproduce" | "harvest" | "reproduce-and-harvest" | "build-bundle"
    backend_kind: str = "mock",
    model_id: str = "Qwen/Qwen3.5-27B",
    profile_name: str | None = None,
    drive_dir: Path | None = None,
    num_reproduce_improve: int = 2,
    num_harvest_improve: int = 2,
    num_harvest_repair: int = 2,
    use_programmatic_teacher: bool = False,
    seed: int = 2026,
) -> dict[str, Any]:
    """Execute Colab A100 reproduction and/or disjoint distillation trace harvesting."""
    ws = Path(workspace_dir).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    profile = get_profile(profile_name)
    sync = DriveMirrorSync(ws, drive_dir)
    if sync.enabled:
        sync.restore_from_drive()

    splits = _prepare_splits(ws, use_deepscaler=(backend_kind == "hf"), seed=seed)
    raw_root = ws / "raw_sessions"
    sessions_root = ws / "sessions"
    clean_ctx = _build_backends(backend_kind=backend_kind, model_id=model_id, profile=profile)
    clean_target, _, num_layers = clean_ctx
    train_layers, eval_layers = split_train_eval_layers(num_layers)

    atomic_write_text(
        ws / "HEARTBEAT.json",
        json.dumps({"updated_at": _utc_now(), "profile": profile.name, "mode": mode, "status": "running"}, indent=2) + "\n",
    )

    recorded_sessions: list[dict[str, Any]] = []

    def _run_one(
        sid: str,
        split_tag: str,
        task: str,
        target_backend: Any,
        editor_backend: Any,
        practice_list: list[Problem],
        reserved_list: list[Problem] | None,
        eval_list: list[Problem],
        transfer_list: list[Problem] | None = None,
        corruption_spec: Any = None,
        s_seed: int = 0,
    ) -> dict[str, Any]:
        s_work = sessions_root / sid / "WORK"
        cfg = SessionConfig(
            task=task,
            protocol=2,
            answer_max_tokens=min(profile.answer_max_tokens, 512 if backend_kind == "mock" else profile.answer_max_tokens),
            starting_diagnosis_n=min(len(practice_list), profile.starting_diagnosis_n),
            reserved_check_n=min(len(reserved_list or []), profile.starting_diagnosis_n),
            reserved_check_samples=1 if backend_kind == "mock" else 2,
            model_arch_summary=target_backend.arch_summary,
            session_id=sid,
            seed=s_seed,
            wall_time_limit_sec=float(profile.max_wall_time_sec),
        )
        harvester = SessionHarvester(
            raw_root=raw_root,
            session_id=sid,
            meta={
                "session_id": sid,
                "split_tag": split_tag,
                "task": task,
                "model_id": model_id,
                "backend_kind": backend_kind,
                "profile": profile.name,
                "corruption_spec": corruption_spec.to_dict() if hasattr(corruption_spec, "to_dict") else corruption_spec,
            },
        )
        orch = SessionOrchestrator(
            work_dir=s_work,
            target=target_backend,
            editor=editor_backend,
            practice_problems=practice_list,
            reserved_problems=reserved_list,
            config=cfg,
            harvester=harvester,
        )
        res = orch.run()
        edit_tensors = orch._host_edit_store.load_edit_tensors(target_torch=(backend_kind != "mock")) if res["decision"] == "SUBMIT" else None
        held_eval = _evaluate_on_split(target_backend, edit_tensors, eval_list, cfg, seed_base=9000 + s_seed)
        if transfer_list:
            held_eval["transfer"] = _evaluate_on_split(target_backend, edit_tensors, transfer_list, cfg, seed_base=9500 + s_seed)
        updated_ladder = score_emergence_ladder(res, heldout_eval=held_eval)
        res["heldout_eval"] = held_eval
        res["ladder"] = updated_ladder
        harvester.record_eval(held_eval, updated_ladder)
        recorded_sessions.append(res)
        return res

    do_reproduce = mode in ("reproduce", "reproduce-and-harvest", "smoke")
    do_harvest = mode in ("harvest", "reproduce-and-harvest", "smoke")

    # Batch 1: Reference Reproduction on unseen practice_eval / reserved_eval / heldout_eval / eval_layers
    if do_reproduce:
        for i in range(num_reproduce_improve):
            ed = ProgrammaticImproveEditor(alpha=0.25 if i % 2 == 0 else 0.0) if (backend_kind == "mock" or use_programmatic_teacher) else clean_target
            if isinstance(ed, ProgrammaticImproveEditor) and i > 0:
                # Session 1 calibrated withdraw example if alpha causes negative/flat outcome or test different layer/alpha
                ed = ProgrammaticImproveEditor(alpha=0.28)
            _run_one(
                sid=f"reproduce_improve_s{i:02d}",
                split_tag="practice_eval",
                task="improve",
                target_backend=clean_target,
                editor_backend=ed,
                practice_list=splits["practice_eval"],
                reserved_list=splits["reserved_eval"],
                eval_list=splits["heldout_eval"],
                s_seed=seed + i,
            )

        rep_layer = eval_layers[len(eval_layers) // 2]
        rep_hidden = sessions_root / "reproduce_repair_eval" / "hidden"
        dmg_target, _, dmg_spec = _make_target_for_repair(backend_kind, clean_ctx, rep_layer, rep_hidden, seed=seed)
        ed_rep = ProgrammaticRepairEditor() if (backend_kind == "mock" or use_programmatic_teacher) else clean_target
        _run_one(
            sid="reproduce_repair_eval",
            split_tag="practice_eval",
            task="repair",
            target_backend=dmg_target,
            editor_backend=ed_rep,
            practice_list=splits["practice_eval"],
            reserved_list=splits["reserved_eval"],
            eval_list=splits["heldout_eval"],
            transfer_list=splits["transfer_eval"],
            corruption_spec=dmg_spec,
            s_seed=seed + 50,
        )

    # Batch 2: Disjoint Distillation Data Harvesting on distill_train & train_layers
    if do_harvest:
        distill_all = splits["distill_train"]
        mid = max(4, len(distill_all) // 2)
        distill_practice = distill_all[:mid]
        distill_reserved = distill_all[mid:]

        for i in range(num_harvest_improve):
            # Mix genuine positive edits and deliberate calibrated-withdraw sessions (force_wrong_alpha=-0.3)
            if backend_kind == "mock" or use_programmatic_teacher:
                ed = ProgrammaticImproveEditor(alpha=0.25) if i < max(1, num_harvest_improve - 1) else ProgrammaticImproveEditor(force_wrong_alpha=-0.35)
            else:
                ed = clean_target
            _run_one(
                sid=f"harvest_improve_{i:02d}",
                split_tag="distill_train",
                task="improve",
                target_backend=clean_target,
                editor_backend=ed,
                practice_list=distill_practice,
                reserved_list=distill_reserved,
                eval_list=splits["heldout_eval"],
                s_seed=seed + 200 + i,
            )

        for i in range(num_harvest_repair):
            t_layer = train_layers[i % len(train_layers)]
            h_hidden = sessions_root / f"harvest_repair_{i:02d}" / "hidden"
            dmg_t, _, h_spec = _make_target_for_repair(
                backend_kind,
                clean_ctx,
                t_layer,
                h_hidden,
                seed=seed + 300 + i,
                sigma_mult=200.0 if i % 2 == 0 else 150.0,
            )
            ed_r = ProgrammaticRepairEditor() if (backend_kind == "mock" or use_programmatic_teacher) else clean_target
            _run_one(
                sid=f"harvest_repair_{i:02d}",
                split_tag="distill_train",
                task="repair",
                target_backend=dmg_t,
                editor_backend=ed_r,
                practice_list=distill_practice,
                reserved_list=distill_reserved,
                eval_list=splits["heldout_eval"],
                corruption_spec=h_spec,
                s_seed=seed + 300 + i,
            )

        # Capture authentic small-model failure sessions on distill_train for DPO negative pairs
        for strat in ("random_noise_4b", "bad_submit_9b"):
            _run_one(
                sid=f"harvest_fail_{strat}",
                split_tag="distill_train",
                task="improve",
                target_backend=clean_target,
                editor_backend=MockFailureEditor(strategy=strat),
                practice_list=distill_practice,
                reserved_list=distill_reserved,
                eval_list=splits["heldout_eval"],
                s_seed=seed + 500,
            )

    bundle_res: dict[str, Any] = {}
    if mode in ("harvest", "reproduce-and-harvest", "build-bundle", "smoke"):
        eval_forbidden_texts = [p.question for k in ("heldout_eval", "reserved_eval", "practice_eval") for p in splits[k]]
        bundle_dir = ws / "distill_bundle" / "v1"
        bundle_res = build_distill_bundle(
            raw_sessions_dir=raw_root,
            output_bundle_dir=bundle_dir,
            forbidden_eval_texts=eval_forbidden_texts,
            forbidden_eval_layers=eval_layers,
            allowed_split_tags=("distill_train",),
        )

    leaderboard = [
        {
            "session_id": r["session_id"],
            "task": r["task"],
            "decision": r["decision"],
            "ladder_score": r["ladder"]["ladder_score"],
            "delta_hat": r.get("heldout_eval", {}).get("delta_hat"),
            "perm_p": r.get("heldout_eval", {}).get("perm_p"),
            "sign_p": r.get("heldout_eval", {}).get("sign_p"),
            "transfer_delta": (r.get("heldout_eval", {}).get("transfer") or {}).get("delta_hat"),
        }
        for r in recorded_sessions
    ]
    atomic_write_text(ws / "leaderboard.json", json.dumps(leaderboard, indent=2) + "\n")

    md_lines = [
        f"# selfrewrite Report ({mode}, profile={profile.name}, backend={backend_kind})",
        "",
        "| Session | Task | Decision | Ladder | Held-Out Δ | p-val | Transfer Δ |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in leaderboard:
        p_min = min(float(row["perm_p"] or 1.0), float(row["sign_p"] or 1.0)) if row["perm_p"] is not None else None
        p_s = f"{p_min:.4g}" if p_min is not None else "-"
        d_s = f"{row['delta_hat']:+.4f}" if row["delta_hat"] is not None else "-"
        t_s = f"{row['transfer_delta']:+.4f}" if row["transfer_delta"] is not None else "-"
        md_lines.append(f"| `{row['session_id']}` | {row['task']} | {row['decision']} | {row['ladder_score']}/6 | {d_s} | {p_s} | {t_s} |")

    if bundle_res:
        md_lines.extend(
            [
                "",
                "## Harvested Distillation Bundle (`distill_bundle/v1`)",
                f"- **Audit Passed**: `{bundle_res['audit']['passed']}`",
                f"- **Counts**: `{json.dumps(bundle_res['manifest']['counts'])}`",
            ]
        )
    atomic_write_text(ws / "report.md", "\n".join(md_lines) + "\n")
    atomic_write_text(
        ws / "HEARTBEAT.json",
        json.dumps({"updated_at": _utc_now(), "profile": profile.name, "mode": mode, "status": "completed"}, indent=2) + "\n",
    )
    sync_res = sync.sync_all()
    return {
        "workspace_dir": str(ws),
        "profile": profile.name,
        "leaderboard": leaderboard,
        "bundle": bundle_res,
        "drive_sync": sync_res,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Colab A100 self-rewriting reproduce & harvest runner")
    parser.add_argument("--workspace-dir", default="selfrewrite/workspace")
    parser.add_argument(
        "--mode",
        choices=("smoke", "reproduce", "harvest", "reproduce-and-harvest", "build-bundle"),
        default="reproduce-and-harvest",
    )
    parser.add_argument("--backend", choices=("hf", "mock"), default="hf")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--drive-dir", default=None)
    parser.add_argument("--num-reproduce-improve", type=int, default=2)
    parser.add_argument("--num-harvest-improve", type=int, default=4)
    parser.add_argument("--num-harvest-repair", type=int, default=4)
    parser.add_argument("--programmatic-teacher", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)

    out = run_reproduce_and_harvest_pipeline(
        workspace_dir=Path(args.workspace_dir),
        mode=args.mode,
        backend_kind=args.backend,
        model_id=args.model_id,
        profile_name=None if args.profile == "auto" else args.profile,
        drive_dir=Path(args.drive_dir) if args.drive_dir else None,
        num_reproduce_improve=args.num_reproduce_improve,
        num_harvest_improve=args.num_harvest_improve,
        num_harvest_repair=args.num_harvest_repair,
        use_programmatic_teacher=bool(args.programmatic_teacher),
        seed=args.seed,
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
