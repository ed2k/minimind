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


def _prepare_splits(
    workspace_dir: Path,
    use_deepscaler: bool,
    seed: int = 2026,
    max_problems_per_split: int | None = None,
) -> dict[str, list[Problem]]:
    splits_dir = workspace_dir / "splits"
    names = ("heldout_eval", "reserved_eval", "practice_eval", "distill_train", "transfer_eval")
    if all((splits_dir / f"{n}.jsonl").is_file() for n in names):
        return {n: load_jsonl_problems(splits_dir / f"{n}.jsonl") for n in names}

    if use_deepscaler:
        try:
            splits = load_deepscaler_splits(seed=seed)
            builtin = build_builtin_suites(seed=seed, practice_n=8, reserved_n=8, heldout_n=8, distill_n=8, transfer_n=32)
            splits["transfer_eval"] = builtin["transfer_eval"]
        except Exception as exc:
            print(f"[colab_launcher] DeepScaleR load failed ({exc}); falling back to builtin suites", flush=True)
            splits = build_builtin_suites(
                seed=seed,
                practice_n=28,
                reserved_n=16,
                heldout_n=28,
                distill_n=28,
                transfer_n=16,
            )
    else:
        splits = build_builtin_suites(
            seed=seed,
            practice_n=28,
            reserved_n=16,
            heldout_n=28,
            distill_n=28,
            transfer_n=16,
        )

    if max_problems_per_split is not None and int(max_problems_per_split) > 0:
        cap = int(max_problems_per_split)
        splits = {
            k: (v[: max(8, int(cap * 4 // 3))] if k == "distill_train" else v[:cap])
            for k, v in splits.items()
        }

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

    # HF / MiniMind Backend: materialize ONLY the single corrupted layer matrix into a CanonicalWeightReader overlay.
    from selfrewrite.backends.base import EditorBackend, GenerationRecord, ReadResult, TargetBackend
    from selfrewrite.backends.hf_gpu_backend import CanonicalWeightReader
    from selfrewrite.tasks.corruption import CorruptionSpec, apply_block_corruption

    base_backend = raw_map.get("_hf_backend") or raw_map.get("_mm_backend")
    candidates = [nm for nm in base_backend.tensor_shapes() if f"layers.{layer_idx}." in nm and nm.endswith("mlp.down_proj.weight")]
    if not candidates:
        candidates = [nm for nm in base_backend.tensor_shapes() if f"layers.{layer_idx}." in nm and nm.endswith(".down_proj.weight")]
    tname = candidates[0]
    orig_t = base_backend.canonical_reader.load_tensor(tname)
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
                    spec["base_dir"] = "/"
                    new_map = {}
                    base_d = Path(self._base.weight_dir) if self._base.weight_dir else Path(dest_dir)
                    for k, rel in spec["weight_map"].items():
                        new_map[k] = str((Path(dest_dir) / dmg_file.name).resolve()).lstrip("/") if k == self._ov_name else str((base_d / rel).resolve()).lstrip("/")
                    spec["weight_map"] = new_map
            atomic_write_text(base_spec_path, json.dumps(spec, indent=2))
            return base_spec_path

    dmg_reader = _CorruptedCanonicalReader(base_backend.canonical_reader, tname, corrupted_t)

    class _DamagedBackendProxy(TargetBackend, EditorBackend):
        def __init__(self, inner: Any, reader: _CorruptedCanonicalReader, ov_name: str, ov_tensor: Any):
            self._inner = inner
            self.canonical_reader = reader
            self._ov_name = ov_name
            self._ov_t = ov_tensor

        @property
        def arch_summary(self) -> str:
            return f"{self._inner.arch_summary} [damaged]"

        @property
        def num_layers(self) -> int:
            return int(self._inner.num_layers)

        def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
            return self.canonical_reader.tensor_shapes()

        def export_sandbox_weights_spec(self, dest_dir: Path) -> Path:
            return self.canonical_reader.export_weights_spec(dest_dir)

        def generate_batch(self, problems: list[Any], edit_tensors: dict[str, Any] | None, n_samples: int, max_tokens: int, config: SessionConfig, seed_base: int = 0) -> list[GenerationRecord]:
            merged = {self._ov_name: self._ov_t}
            if edit_tensors:
                merged.update(edit_tensors)
            return self._inner.generate_batch(problems, merged, n_samples, max_tokens, config, seed_base=seed_base)

        def read_activations(self, texts: list[str], layers: list[int] | None, target: str, pooling: str, k: int, edit_tensors: dict[str, Any] | None, max_tokens: int, pass_id: int) -> ReadResult:
            merged = {self._ov_name: self._ov_t}
            if edit_tensors:
                merged.update(edit_tensors)
            return self._inner.read_activations(texts, layers, target, pooling, k, merged, max_tokens, pass_id)

        def complete_turn(self, messages: list[dict[str, str]], config: SessionConfig, turn_index: int = 0) -> str:
            return self._inner.complete_turn(messages, config, turn_index=turn_index)

        def __getattr__(self, item: str) -> Any:
            return getattr(self._inner, item)

    dmg_target = _DamagedBackendProxy(base_backend, dmg_reader, tname, corrupted_t)
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
    hybrid_harvest_teacher: bool = False,
    starting_diagnosis_n: int | None = None,
    answer_max_tokens: int | None = None,
    editor_max_new_tokens: int | None = None,
    max_turns: int | None = None,
    max_problems_per_split: int | None = None,
    use_builtin_splits: bool = False,
    canary_max_problems: int | None = None,
    fast_harvest_profile: bool = False,
    seed: int = 2026,
) -> dict[str, Any]:
    """Execute Colab A100 reproduction and/or disjoint distillation trace harvesting."""
    ws = Path(workspace_dir).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    profile = get_profile(profile_name)
    harvest_profile = (
        get_profile(f"{profile.name}_harvest")
        if fast_harvest_profile and profile.name in ("colab_a100_80g", "colab_a100_40g", "colab_l4")
        else profile
    )
    sync = DriveMirrorSync(ws, drive_dir)
    restored_from_drive = False
    if sync.enabled:
        restore_res = sync.restore_from_drive()
        restored_from_drive = bool(restore_res.get("restored"))

    splits = _prepare_splits(
        ws,
        use_deepscaler=(backend_kind == "hf" and not use_builtin_splits),
        seed=seed,
        max_problems_per_split=max_problems_per_split,
    )
    if canary_max_problems is not None and int(canary_max_problems) > 0:
        cap = int(canary_max_problems)
        splits = {
            k: (v[: max(4, cap)] if k == "distill_train" else v[:cap])
            for k, v in splits.items()
        }
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
        run_heldout_eval: bool = True,
        step_profile: ComputeProfile | None = None,
        editor_mode: str = "autonomous_llm",
    ) -> dict[str, Any]:
        eff_prof = step_profile or profile
        s_dir = raw_root / sid
        res_file = s_dir / "session_result.json"
        eval_file = s_dir / "eval_result.json"
        ladder_file = s_dir / "ladder_score.json"
        if res_file.is_file() and (not run_heldout_eval or eval_file.is_file()):
            res = json.loads(res_file.read_text(encoding="utf-8"))
            if eval_file.is_file() and run_heldout_eval:
                res["heldout_eval"] = json.loads(eval_file.read_text(encoding="utf-8"))
            else:
                res.setdefault("heldout_eval", None)
            if ladder_file.is_file():
                res["ladder"] = json.loads(ladder_file.read_text(encoding="utf-8"))
            recorded_sessions.append(res)
            return res

        s_work = sessions_root / sid / "WORK"
        eff_max_tokens = (
            int(answer_max_tokens)
            if answer_max_tokens is not None
            else min(eff_prof.answer_max_tokens, 512 if backend_kind == "mock" else eff_prof.answer_max_tokens)
        )
        eff_diag_n = min(
            len(practice_list),
            int(starting_diagnosis_n) if starting_diagnosis_n is not None else eff_prof.starting_diagnosis_n,
        )
        eff_res_n = min(
            len(reserved_list or []),
            int(starting_diagnosis_n) if starting_diagnosis_n is not None else eff_prof.starting_diagnosis_n,
        )
        eff_max_turns = int(max_turns) if max_turns is not None else 160
        eff_editor_max_new_tokens = int(editor_max_new_tokens) if editor_max_new_tokens is not None else 12000
        cfg = SessionConfig(
            task=task,
            protocol=2,
            max_turns=eff_max_turns,
            answer_max_tokens=eff_max_tokens,
            editor_max_new_tokens=eff_editor_max_new_tokens,
            starting_diagnosis_n=eff_diag_n,
            reserved_check_n=eff_res_n,
            reserved_check_samples=1 if (backend_kind == "mock" or eff_prof.name.endswith("_harvest")) else 2,
            model_arch_summary=target_backend.arch_summary,
            session_id=sid,
            seed=s_seed,
            wall_time_limit_sec=float(eff_prof.max_wall_time_sec),
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
                "profile": eff_prof.name,
                "editor_mode": editor_mode,
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
        res = orch.run(resume=(restored_from_drive and (s_work / "session_state.json").is_file()))
        if run_heldout_eval:
            edit_tensors = orch._host_edit_store.load_edit_tensors(target_torch=(backend_kind != "mock")) if res["decision"] == "SUBMIT" else None
            held_eval = _evaluate_on_split(target_backend, edit_tensors, eval_list, cfg, seed_base=9000 + s_seed)
            if transfer_list:
                held_eval["transfer"] = _evaluate_on_split(target_backend, edit_tensors, transfer_list, cfg, seed_base=9500 + s_seed)
            updated_ladder = score_emergence_ladder(res, heldout_eval=held_eval)
            res["heldout_eval"] = held_eval
            res["ladder"] = updated_ladder
            harvester.record_eval(held_eval, updated_ladder)
        else:
            updated_ladder = score_emergence_ladder(res, heldout_eval=None)
            res["heldout_eval"] = None
            res["ladder"] = updated_ladder
            atomic_write_text(s_dir / "ladder_score.json", json.dumps(updated_ladder, indent=2) + "\n")

        atomic_write_text(
            ws / "HEARTBEAT.json",
            json.dumps(
                {
                    "updated_at": _utc_now(),
                    "profile": eff_prof.name,
                    "mode": mode,
                    "status": "running",
                    "last_completed_session": sid,
                },
                indent=2,
            )
            + "\n",
        )
        if sync.enabled:
            sync.sync_all()
        recorded_sessions.append(res)
        return res

    do_reproduce = mode in ("reproduce", "reproduce-and-harvest", "smoke")
    do_harvest = mode in ("harvest", "reproduce-and-harvest", "smoke")

    # Batch 1: Reference Reproduction on unseen practice_eval / reserved_eval / heldout_eval / eval_layers
    if do_reproduce:
        rep_editor_mode = (
            "mechanistic_teacher_on_real_target"
            if (backend_kind == "mock" or use_programmatic_teacher)
            else "autonomous_llm"
        )
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
                editor_mode=rep_editor_mode,
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
            editor_mode=rep_editor_mode,
        )

    # Batch 2: Disjoint Distillation Data Harvesting on distill_train & train_layers
    if do_harvest:
        distill_all = splits["distill_train"]
        mid = max(4, len(distill_all) // 2)
        distill_practice = distill_all[:mid]
        distill_reserved = distill_all[mid:]

        for i in range(num_harvest_improve):
            # Mix genuine positive edits and deliberate calibrated-withdraw sessions (force_wrong_alpha=-0.35)
            if backend_kind == "mock" or use_programmatic_teacher:
                ed = ProgrammaticImproveEditor(alpha=0.25) if i < max(1, num_harvest_improve - 1) else ProgrammaticImproveEditor(force_wrong_alpha=-0.35)
                ed_mode = "mechanistic_teacher_on_real_target"
            elif hybrid_harvest_teacher:
                if i == 0:
                    ed = clean_target
                    ed_mode = "autonomous_llm"
                else:
                    ed = (
                        ProgrammaticImproveEditor(force_wrong_alpha=-0.35)
                        if (num_harvest_improve >= 3 and i == num_harvest_improve - 1)
                        else ProgrammaticImproveEditor(alpha=0.25)
                    )
                    ed_mode = "mechanistic_teacher_on_real_target"
            else:
                ed = clean_target
                ed_mode = "autonomous_llm"
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
                run_heldout_eval=False,
                step_profile=harvest_profile,
                editor_mode=ed_mode,
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
            if backend_kind == "mock" or use_programmatic_teacher:
                ed_r: Any = ProgrammaticRepairEditor()
                ed_r_mode = "mechanistic_teacher_on_real_target"
            elif hybrid_harvest_teacher:
                if i == 0:
                    ed_r = dmg_t
                    ed_r_mode = "autonomous_llm"
                else:
                    ed_r = ProgrammaticRepairEditor()
                    ed_r_mode = "mechanistic_teacher_on_real_target"
            else:
                ed_r = dmg_t
                ed_r_mode = "autonomous_llm"
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
                run_heldout_eval=False,
                step_profile=harvest_profile,
                editor_mode=ed_r_mode,
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
                run_heldout_eval=False,
                step_profile=harvest_profile,
                editor_mode="failure_negative",
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
            "delta_hat": (r.get("heldout_eval") or {}).get("delta_hat"),
            "perm_p": (r.get("heldout_eval") or {}).get("perm_p"),
            "sign_p": (r.get("heldout_eval") or {}).get("sign_p"),
            "transfer_delta": ((r.get("heldout_eval") or {}).get("transfer") or {}).get("delta_hat"),
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
    parser = argparse.ArgumentParser(description="Colab A100 self-rewriting reproduce, harvest & downsize runner")
    parser.add_argument("--workspace-dir", default="selfrewrite/workspace")
    parser.add_argument(
        "--mode",
        choices=("smoke", "reproduce", "harvest", "reproduce-and-harvest", "build-bundle", "train-student", "bracket"),
        default="reproduce-and-harvest",
    )
    parser.add_argument("--backend", choices=("hf", "mock"), default="hf")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--drive-dir", default=None)
    parser.add_argument("--bundle-dir", default=None)
    parser.add_argument("--rung-id", default="R2")
    parser.add_argument("--rungs", default="R1,R2,R3,R4")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-reproduce-improve", type=int, default=2)
    parser.add_argument("--num-harvest-improve", type=int, default=4)
    parser.add_argument("--num-harvest-repair", type=int, default=4)
    parser.add_argument("--programmatic-teacher", action="store_true")
    parser.add_argument("--hybrid-harvest-teacher", action="store_true")
    parser.add_argument("--starting-diagnosis-n", type=int, default=None)
    parser.add_argument("--answer-max-tokens", type=int, default=None)
    parser.add_argument("--editor-max-new-tokens", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--max-problems-per-split", type=int, default=None)
    parser.add_argument("--use-builtin-splits", action="store_true")
    parser.add_argument("--fast-harvest-profile", action="store_true")
    parser.add_argument("--bootstrap-next-rung", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)

    ws = Path(args.workspace_dir).resolve()
    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else (ws / "distill_bundle" / "v1")
    drive_dir = Path(args.drive_dir) if args.drive_dir else None

    if args.mode == "train-student":
        from selfrewrite.distill.train_student import StudentTrainConfig, train_downsize_student

        out_dir = Path(args.output_dir) if args.output_dir else (ws / "students" / args.rung_id)
        out = train_downsize_student(
            StudentTrainConfig(
                rung_id=args.rung_id,
                base_model_id="" if args.model_id == "Qwen/Qwen3.5-27B" else args.model_id,
                bundle_dir=bundle_dir,
                output_dir=out_dir,
                splits_dir=ws / "splits",
                backend=args.backend,
                profile_name=args.profile,
                seed=args.seed,
            )
        )
        print(json.dumps(out, indent=2))
        return 0

    if args.mode == "bracket":
        from selfrewrite.colab.downsize_bracket import run_downsize_bracket

        rungs = tuple(r.strip() for r in str(args.rungs).split(",") if r.strip())
        out = run_downsize_bracket(
            workspace_dir=ws,
            bundle_dir=bundle_dir,
            rungs=rungs,
            backend_kind=args.backend,
            profile_name=None if args.profile == "auto" else args.profile,
            drive_dir=drive_dir,
            starting_diagnosis_n=args.starting_diagnosis_n,
            answer_max_tokens=args.answer_max_tokens,
            editor_max_new_tokens=args.editor_max_new_tokens,
            max_turns=args.max_turns,
            bootstrap_next_rung=bool(args.bootstrap_next_rung),
            seed=args.seed,
        )
        print(json.dumps(out, indent=2))
        return 0

    out = run_reproduce_and_harvest_pipeline(
        workspace_dir=ws,
        mode=args.mode,
        backend_kind=args.backend,
        model_id=args.model_id,
        profile_name=None if args.profile == "auto" else args.profile,
        drive_dir=drive_dir,
        num_reproduce_improve=args.num_reproduce_improve,
        num_harvest_improve=args.num_harvest_improve,
        num_harvest_repair=args.num_harvest_repair,
        use_programmatic_teacher=bool(args.programmatic_teacher),
        hybrid_harvest_teacher=bool(args.hybrid_harvest_teacher),
        starting_diagnosis_n=args.starting_diagnosis_n,
        answer_max_tokens=args.answer_max_tokens,
        editor_max_new_tokens=args.editor_max_new_tokens,
        max_turns=args.max_turns,
        max_problems_per_split=args.max_problems_per_split,
        use_builtin_splits=bool(args.use_builtin_splits),
        fast_harvest_profile=bool(args.fast_harvest_profile),
        seed=args.seed,
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
