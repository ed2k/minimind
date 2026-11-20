from __future__ import annotations

import inspect
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from selfchange.kernel.budgets import ComputeProfile
from selfchange.kernel.integrity import KERNEL_MANIFEST_FILENAME, check_git_diff_boundary
from selfchange.kernel.registry import LineageNode, LineageRegistry
from selfchange.kernel.supervisor import CycleContext, CycleOutcome
from selfchange.kernel.tool_runtime import WorktreeToolExecutor
from selfchange.kernel.verifier import Verifier, compute_rlvr_reward
from selfchange.kernel.worktree import BoundaryViolationError, WorktreeHandle, WorktreeManager
from selfchange.kernel.sandbox import SandboxPolicyError


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CycleConfig:
    """Configuration controlling autonomous candidate generation and training cadence."""

    autonomy_level: int = 1
    train_every_n_cycles: int = 3
    dry_run_training: bool = False
    min_improvement: float = 1e-6
    drill_seeds: tuple[int, ...] = (42,)
    anchor_path: Path | None = None
    allow_small_batch_train: bool = True
    base_weight_path: Path | None = None
    extra_eval_max_regression: float = 0.05
    extra_eval_floor: float = 0.0


def _default_proposer(ctx: CycleContext, _handle: WorktreeHandle, executor: WorktreeToolExecutor) -> dict:
    state_res = executor.dispatch("get_system_state", {})
    temp = 0.15 if (ctx.cycle_index % 2 == 1) else 0.2
    if ctx.safe_mode:
        temp = 0.1
    edit_res = executor.dispatch("patch_config", {"updates": {"temperature": temp}})
    sub_res = executor.dispatch("submit_candidate", {"hypothesis": f"cycle {ctx.cycle_index} temperature={temp}"})
    tool_calls = [
        {"name": "get_system_state", "arguments": {}},
        {"name": "patch_config", "arguments": {"updates": {"temperature": temp}}},
        {"name": "submit_candidate", "arguments": {"hypothesis": f"cycle {ctx.cycle_index} temperature={temp}"}},
    ]
    messages = [
        {"role": "user", "content": f"cycle={ctx.cycle_index} safe_mode={ctx.safe_mode}"},
        {"role": "assistant", "content": f"<think>Calibrate temperature to {temp}.</think>\n<tool_call>{json.dumps(tool_calls[1])}</tool_call>"},
    ]
    return {
        "task_id": f"cycle_{ctx.cycle_index}",
        "task_family": "config_tune",
        "observed_before_edit": bool(state_res.get("ok")),
        "first_edit_valid": bool(edit_res.get("ok")),
        "tool_calls": tool_calls,
        "turn_outputs": [messages[1]["content"]],
        "messages": messages,
        "final_text": f"Submitted cycle {ctx.cycle_index} with temperature={temp} (sub_ok={sub_res.get('ok')}).",
    }


class AutonomousCycleRunner:
    """Executes one complete isolated worktree -> edit -> verify -> archive -> self-train cycle."""

    def __init__(
        self,
        repo_root: Path,
        workspace_dir: Path,
        profile: ComputeProfile,
        proposer_fn: Callable[[CycleContext, WorktreeHandle, WorktreeToolExecutor], dict] | None = None,
        kernel_dir: Path | None = None,
        manifest_path: Path | None = None,
        cycle_config: CycleConfig | None = None,
        extra_eval_fn: Callable[..., float] | None = None,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.workspace_dir = Path(workspace_dir).resolve()
        self.profile = profile
        self.proposer_fn = proposer_fn if proposer_fn is not None else _default_proposer
        self.kernel_dir = Path(kernel_dir).resolve() if kernel_dir is not None else Path(__file__).resolve().parent
        self.manifest_path = (
            Path(manifest_path).resolve()
            if manifest_path is not None
            else self.kernel_dir / KERNEL_MANIFEST_FILENAME
        )
        self.cycle_config = cycle_config if cycle_config is not None else CycleConfig()
        self.extra_eval_fn = extra_eval_fn
        self.worktree_mgr = WorktreeManager(self.repo_root, self.workspace_dir / "worktrees")
        self.verifier = Verifier(kernel_dir=self.kernel_dir, manifest_path=self.manifest_path, profile=self.profile)
        self.registry = LineageRegistry(self.workspace_dir)
        self.trajectories_path = self.workspace_dir / "trajectories.jsonl"
        self.train_history_path = self.workspace_dir / "logs" / "train_history.jsonl"
        self._best_extra_score: float | None = None

    def _invoke_extra_eval(self, worktree_dir: Path, weight_path: Path | None) -> float:
        assert self.extra_eval_fn is not None
        sig = inspect.signature(self.extra_eval_fn)
        params = sig.parameters.values()
        positional = [
            p for p in params if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        accepts_two = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params) or len(positional) >= 2
        if accepts_two:
            return float(self.extra_eval_fn(worktree_dir, weight_path))
        return float(self.extra_eval_fn(worktree_dir))

    def _compute_min_extra_score(self) -> float:
        if self._best_extra_score is not None:
            return round(
                max(
                    self.cycle_config.extra_eval_floor,
                    self._best_extra_score - self.cycle_config.extra_eval_max_regression,
                ),
                4,
            )
        champ = self.registry.get_champion()
        if champ is not None and champ.eval_score < 0.9999:
            champ_extra = max(0.0, 2.0 * float(champ.eval_score) - 1.0)
            return round(
                max(
                    self.cycle_config.extra_eval_floor,
                    champ_extra - self.cycle_config.extra_eval_max_regression,
                ),
                4,
            )
        return float(self.cycle_config.extra_eval_floor)

    def _resolve_base_ref(self, parent: LineageNode | None) -> str:
        if parent and parent.commit_sha and parent.commit_sha != "HEAD":
            proc = subprocess.run(
                ["git", "cat-file", "-e", f"{parent.commit_sha}^{{commit}}"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                return parent.commit_sha
        return "HEAD"

    def _append_trajectory(self, record: dict) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        with self.trajectories_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _append_train_history(self, record: dict) -> None:
        self.train_history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.train_history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _resolve_training_from_weight(self, ctx: CycleContext) -> tuple[str, Path | None]:
        cand: Path | None = None
        if ctx.parent and ctx.parent.weight_path and Path(ctx.parent.weight_path).is_file():
            cand = Path(ctx.parent.weight_path)
        elif self.cycle_config.base_weight_path is not None and Path(self.cycle_config.base_weight_path).is_file():
            cand = Path(self.cycle_config.base_weight_path)
        if cand is None:
            return "full_sft", None
        moe_suffix = "_moe" if self.profile.use_moe else ""
        suffix = f"_{self.profile.hidden_size}{moe_suffix}"
        stem = cand.stem
        if stem.endswith(suffix) and len(stem) > len(suffix):
            return stem[: -len(suffix)], cand
        return "full_sft", cand

    def _run_periodic_training(self, ctx: CycleContext, worktree_dir: Path) -> str | None:
        from selfchange.mutable.data_curator import build_dpo_pairs, curate_rft_samples, mix_with_anchor
        from selfchange.mutable.train_recipe import select_training_stage
        from selfchange.training.self_train_runner import (
            promote_lora_adapter,
            run_training_stage,
            stage_from_weight_for_trainer,
            write_dpo_jsonl,
            write_sft_jsonl,
        )

        if not self.trajectories_path.is_file():
            return None
        trajs = [json.loads(ln) for ln in self.trajectories_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        rft = curate_rft_samples(trajs)
        dpo = build_dpo_pairs(trajs)

        cfg_path = worktree_dir / "selfchange" / "mutable" / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        recipe = select_training_stage(
            metrics={"pass_at_1": 0.5, "pass_at_4": 0.7, "rl_ready": False},
            buffer_stats={"rft_samples": len(rft), "dpo_pairs": len(dpo)},
            config=cfg,
            safe_mode=ctx.safe_mode,
        )
        stage = recipe["stage"]
        if stage == "none" and self.cycle_config.allow_small_batch_train:
            if dpo:
                stage = "dpo"
            elif rft:
                stage = "lora"
        if stage == "none":
            self._append_train_history(
                {
                    "timestamp": _utc_now(),
                    "cycle_index": ctx.cycle_index,
                    "stage": "none",
                    "rft_count": len(rft),
                    "dpo_count": len(dpo),
                    "ok": False,
                    "skipped_reason": "no_training_stage",
                }
            )
            return None

        anchors = []
        if self.cycle_config.anchor_path and Path(self.cycle_config.anchor_path).is_file():
            anchors = [json.loads(ln) for ln in Path(self.cycle_config.anchor_path).read_text(encoding="utf-8").splitlines() if ln.strip()]

        data_dir = self.workspace_dir / "training_data"
        data_dir.mkdir(parents=True, exist_ok=True)
        if stage == "dpo":
            data_file = data_dir / f"dpo_c{ctx.cycle_index:04d}.jsonl"
            write_dpo_jsonl(data_file, dpo)
        else:
            mixed = mix_with_anchor(rft, anchors, anchor_ratio=recipe["training"].get("anchor_replay_ratio", 0.25))
            data_file = data_dir / f"rft_c{ctx.cycle_index:04d}.jsonl"
            write_sft_jsonl(data_file, mixed)

        from_weight, from_weight_path = self._resolve_training_from_weight(ctx)
        res = run_training_stage(
            stage=stage,
            repo_root=self.repo_root,
            workspace_dir=self.workspace_dir,
            data_path=data_file,
            profile=self.profile,
            training_cfg=recipe["training"],
            from_weight=from_weight,
            save_weight=f"cand_c{ctx.cycle_index:04d}",
            dry_run=self.cycle_config.dry_run_training,
            from_weight_path=from_weight_path,
        )

        final_weight_path: str | None = res.output_weight_path if res.ok else None
        merge_error: str | None = None
        if stage == "lora" and res.ok and res.output_weight_path and not self.cycle_config.dry_run_training:
            try:
                base_pth = stage_from_weight_for_trainer(
                    from_weight,
                    self.repo_root,
                    self.workspace_dir,
                    self.profile,
                    from_weight_path=from_weight_path,
                )
                if base_pth is None:
                    raise FileNotFoundError(f"Base weight {from_weight!r} not found for LoRA merge")
                moe_suffix = "_moe" if self.profile.use_moe else ""
                merged_path = (
                    self.workspace_dir
                    / "models"
                    / "candidates"
                    / f"cand_c{ctx.cycle_index:04d}_merged_{self.profile.hidden_size}{moe_suffix}.pth"
                )
                lora_rank = int(recipe["training"].get("lora_rank", 16))
                promote_lora_adapter(
                    base_weight=base_pth,
                    lora_weight=Path(res.output_weight_path),
                    merged_out=merged_path,
                    hidden_size=self.profile.hidden_size,
                    num_hidden_layers=self.profile.num_hidden_layers,
                    use_moe=self.profile.use_moe,
                    lora_rank=lora_rank,
                )
                final_weight_path = str(merged_path)
            except Exception as exc:
                merge_error = repr(exc)
                final_weight_path = None

        self._append_train_history(
            {
                "timestamp": _utc_now(),
                "cycle_index": ctx.cycle_index,
                "stage": stage,
                "from_weight": from_weight,
                "from_weight_path": str(from_weight_path) if from_weight_path else None,
                "rft_count": len(rft),
                "dpo_count": len(dpo),
                "data_file": str(data_file),
                "ok": res.ok,
                "exit_code": res.exit_code,
                "final_loss": res.final_loss,
                "duration_sec": res.duration_sec,
                "oom_detected": res.oom_detected,
                "nan_detected": res.nan_detected,
                "timed_out": res.timed_out,
                "output_weight_path": final_weight_path,
                "merge_error": merge_error,
                "log_tail": res.log_tail,
            }
        )
        return final_weight_path

    def __call__(self, ctx: CycleContext) -> CycleOutcome:
        cand_id = f"cand_c{ctx.cycle_index:04d}_{int(time.monotonic() * 1000) % 100000}"
        base_ref = self._resolve_base_ref(ctx.parent)
        handle = self.worktree_mgr.create_worktree(cand_id, base_ref=base_ref)
        effective_level = 0 if (ctx.config_only or ctx.safe_mode) else self.cycle_config.autonomy_level
        promote_commit = False

        try:
            executor = WorktreeToolExecutor(
                handle.path,
                autonomy_level=effective_level,
                state_provider=lambda: {
                    "cycle_index": ctx.cycle_index,
                    "profile": ctx.profile.name,
                    "champion_id": ctx.parent.node_id if ctx.parent else None,
                    "safe_mode": ctx.safe_mode,
                    "config_only": ctx.config_only,
                },
                archive_provider=lambda k: [n.to_dict() for n in self.registry.load_archive()[-k:]],
                base_ref=handle.base_commit,
                trajectories_path=self.trajectories_path,
            )

            ep_meta = self.proposer_fn(ctx, handle, executor) or {}
            parent_weight_path = (
                Path(ctx.parent.weight_path)
                if (ctx.parent and ctx.parent.weight_path and Path(ctx.parent.weight_path).is_file())
                else self.cycle_config.base_weight_path
            )
            verify_extra_fn = (
                (lambda wt: self._invoke_extra_eval(wt, parent_weight_path))
                if self.extra_eval_fn is not None
                else None
            )
            min_extra = self._compute_min_extra_score() if self.extra_eval_fn is not None else 0.7
            report = self.verifier.verify_candidate(
                handle.path,
                base_ref=handle.base_commit,
                check_git=True,
                seeds=list(self.cycle_config.drill_seeds),
                extra_eval_fn=verify_extra_fn,
                min_extra_score=min_extra,
            )
            extra_score = report.gates["gate3"].details.get("extra_score") if "gate3" in report.gates else None
            diff_rep = check_git_diff_boundary(handle.path, base_ref=handle.base_commit)
            boundary_violated = not diff_rep.ok
            reward = compute_rlvr_reward(
                turn_outputs=ep_meta.get("turn_outputs", []),
                tool_calls=ep_meta.get("tool_calls", []),
                observed_before_edit=bool(ep_meta.get("observed_before_edit", True)),
                first_edit_valid=bool(ep_meta.get("first_edit_valid", True)),
                task_passed=report.passed,
                boundary_violated=boundary_violated,
                unfinished=bool(ep_meta.get("unfinished", False)),
                over_budget=False,
                open_thinking=True,
            )

            self._append_trajectory(
                {
                    "cycle_index": ctx.cycle_index,
                    "task_id": ep_meta.get("task_id", f"cycle_{ctx.cycle_index}"),
                    "task_family": ep_meta.get("task_family", "autonomous_cycle"),
                    "passed": report.passed,
                    "reward": reward,
                    "tool_calls": ep_meta.get("tool_calls", []),
                    "messages": ep_meta.get("messages", []),
                    "final_text": ep_meta.get("final_text", ""),
                    "total_score": report.total_score,
                    "extra_score": extra_score,
                }
            )

            if not report.passed:
                err_reason = next((k for k, g in report.gates.items() if not g.passed), "verification_failed")
                rej_node = LineageNode(
                    node_id=cand_id,
                    parent_id=ctx.parent.node_id if ctx.parent else None,
                    commit_sha=handle.base_commit,
                    weight_path=ctx.parent.weight_path if ctx.parent else None,
                    survival_score=report.survival_score,
                    eval_score=report.eval_score,
                    self_repair_score=report.self_repair_score,
                    efficiency_score=report.efficiency_score,
                    total_score=report.total_score,
                    status="rejected",
                    hypothesis=executor.hypothesis or ep_meta.get("final_text", "rejected"),
                    created_at=_utc_now(),
                )
                return CycleOutcome(passed=False, total_score=report.total_score, candidate_node=rej_node, error=err_reason)

            commit_sha = handle.base_commit
            if diff_rep.allowed:
                try:
                    commit_sha = self.worktree_mgr.commit_mutable_changes(
                        handle,
                        message=f"selfchange cycle {ctx.cycle_index}: {executor.hypothesis or 'verified update'}",
                    )
                    promote_commit = True
                except (BoundaryViolationError, SandboxPolicyError) as exc:
                    return CycleOutcome(passed=False, error=str(exc))

            weight_path = ctx.parent.weight_path if ctx.parent else (str(self.cycle_config.base_weight_path) if self.cycle_config.base_weight_path else None)
            if self.cycle_config.train_every_n_cycles > 0 and ctx.cycle_index % self.cycle_config.train_every_n_cycles == 0:
                trained_w = self._run_periodic_training(ctx, handle.path)
                if trained_w is not None:
                    if self.extra_eval_fn is not None and not self.cycle_config.dry_run_training:
                        post_report = self.verifier.verify_candidate(
                            handle.path,
                            base_ref=handle.base_commit,
                            check_git=True,
                            seeds=list(self.cycle_config.drill_seeds),
                            extra_eval_fn=lambda wt: self._invoke_extra_eval(wt, Path(trained_w)),
                            min_extra_score=min_extra,
                        )
                        if post_report.passed:
                            weight_path = trained_w
                            report = post_report
                            self._append_train_history(
                                {
                                    "timestamp": _utc_now(),
                                    "event": "post_train_eval",
                                    "cycle_index": ctx.cycle_index,
                                    "accepted": True,
                                    "weight_path": trained_w,
                                    "eval_score": post_report.eval_score,
                                    "total_score": post_report.total_score,
                                }
                            )
                        else:
                            self._append_train_history(
                                {
                                    "timestamp": _utc_now(),
                                    "event": "post_train_eval",
                                    "cycle_index": ctx.cycle_index,
                                    "accepted": False,
                                    "rejected_weight_path": trained_w,
                                    "eval_score": post_report.eval_score,
                                    "total_score": post_report.total_score,
                                }
                            )
                    else:
                        weight_path = trained_w

            if self.extra_eval_fn is not None and "gate3" in report.gates:
                final_extra = float(report.gates["gate3"].details.get("extra_score", 0.0))
                self._best_extra_score = max(self._best_extra_score or 0.0, final_extra)

            node = LineageNode(
                node_id=cand_id,
                parent_id=ctx.parent.node_id if ctx.parent else None,
                commit_sha=commit_sha,
                weight_path=weight_path,
                survival_score=report.survival_score,
                eval_score=report.eval_score,
                self_repair_score=report.self_repair_score,
                efficiency_score=report.efficiency_score,
                total_score=report.total_score,
                status="archived",
                hypothesis=executor.hypothesis or f"cycle {ctx.cycle_index}",
                created_at=_utc_now(),
            )
            return CycleOutcome(passed=True, total_score=report.total_score, candidate_node=node)
        finally:
            self.worktree_mgr.remove_worktree(handle, delete_branch=not promote_commit)
