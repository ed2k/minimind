from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from selfchange.kernel.autonomous_cycle import AutonomousCycleRunner, CycleConfig
from selfchange.kernel.budgets import ComputeProfile, get_profile
from selfchange.kernel.colab_sync import ColabDriveSync
from selfchange.kernel.integrity import KERNEL_MANIFEST_FILENAME, atomic_write_text
from selfchange.kernel.registry import LineageNode
from selfchange.kernel.supervisor import CycleContext, Supervisor
from selfchange.kernel.tool_runtime import SELFCHANGE_TOOLS, WorktreeToolExecutor
from selfchange.kernel.worktree import WorktreeHandle
from selfchange.mutable.agent_loop import run_agent_episode

_TOOLS_BY_NAME = {t["function"]["name"]: t for t in SELFCHANGE_TOOLS}


def _compact_tools(names: list[str]) -> list[dict]:
    selected: list[dict] = []
    for name in names:
        tool = _TOOLS_BY_NAME.get(name)
        if not tool:
            continue
        fn = tool["function"]
        selected.append({"type": "function", "function": {"name": fn["name"], "parameters": fn["parameters"]}})
    return selected


def _is_real_weight_file(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("rb") as f:
            return f.read(1) != b"{"
    except OSError:
        return False


def resolve_model_checkpoint(
    repo_root: Path,
    workspace_dir: Path,
    profile: ComputeProfile,
    explicit_model_path: Path | None = None,
) -> tuple[Path | None, str]:
    """Resolve the best available MiniMind checkpoint in priority order without silent fallback."""
    repo_root = Path(repo_root).resolve()
    workspace_dir = Path(workspace_dir).resolve()
    moe_suffix = "_moe" if profile.use_moe else ""
    hs = profile.hidden_size

    champ_file = workspace_dir / "champion.json"
    if champ_file.is_file():
        try:
            champ_w = json.loads(champ_file.read_text(encoding="utf-8")).get("champion", {}).get("weight_path")
            if champ_w:
                cand_champ = Path(champ_w)
                if not cand_champ.is_absolute():
                    cand_champ = workspace_dir / cand_champ
                if _is_real_weight_file(cand_champ):
                    return cand_champ.resolve(), "champion"
        except Exception:
            pass

    if explicit_model_path is not None and str(explicit_model_path) != "auto":
        cand_exp = Path(explicit_model_path)
        if not cand_exp.is_absolute():
            cand_exp = repo_root / cand_exp
        if _is_real_weight_file(cand_exp):
            return cand_exp.resolve(), "explicit"

    candidates = [
        ("offline_dpo_r1", workspace_dir / "models" / "candidates" / f"offline_dpo_r1_{hs}{moe_suffix}.pth"),
        ("offline_rft_r1_merged", workspace_dir / "models" / "candidates" / f"offline_rft_r1_merged_{hs}{moe_suffix}.pth"),
        ("warmstart_rung3", workspace_dir / "models" / "candidates" / f"warmstart_rung3_{hs}{moe_suffix}.pth"),
        ("warmstart_rung2", workspace_dir / "models" / "candidates" / f"warmstart_rung2_{hs}{moe_suffix}.pth"),
        ("warmstart_rung1", workspace_dir / "models" / "candidates" / f"warmstart_rung1_{hs}{moe_suffix}.pth"),
        ("base_minimind3", workspace_dir / "models" / "base" / f"minimind3_{hs}{moe_suffix}.pth"),
        ("out_full_sft", repo_root / "out" / f"full_sft_{hs}{moe_suffix}.pth"),
    ]
    for label, path in candidates:
        if _is_real_weight_file(path):
            return path.resolve(), label
    return None, "none"


class MiniMindProposerSession:
    """Reusable cached MiniMind-3 session supporting dynamic weight hot-reloading and eval."""

    def __init__(
        self,
        initial_model_path: Path,
        tokenizer_path: Path,
        profile: ComputeProfile,
        autonomy_level: int = 1,
    ):
        self.initial_model_path = Path(initial_model_path).resolve()
        self.tokenizer_path = Path(tokenizer_path).resolve()
        self.profile = profile
        self.autonomy_level = autonomy_level
        self.loaded_path: Path | None = None
        self.model = None
        self.tokenizer = None

    def _ensure_loaded(self, target_path: Path | None = None) -> None:
        desired = (
            Path(target_path).resolve()
            if (target_path is not None and _is_real_weight_file(Path(target_path)))
            else self.initial_model_path
        )
        if self.model is not None and self.loaded_path == desired:
            return
        from selfchange.training.model_loader import load_minimind_model

        if self.model is not None:
            self.model = None
            try:
                import torch  # type: ignore[import-not-found]

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        self.model, self.tokenizer = load_minimind_model(desired, self.tokenizer_path, self.profile)
        self.loaded_path = desired

    def propose(self, ctx: CycleContext, handle: WorktreeHandle, executor: WorktreeToolExecutor) -> dict:
        from selfchange.training.model_loader import generate_agent_reply

        parent_w = Path(ctx.parent.weight_path) if (ctx.parent and ctx.parent.weight_path) else None
        self._ensure_loaded(parent_w)
        assert self.model is not None and self.tokenizer is not None
        eos_tok = self.tokenizer.eos_token or "<|im_end|>"
        pad_tok = self.tokenizer.pad_token or "<|endoftext|>"

        cfg_path = handle.path / "selfchange" / "mutable" / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {"max_turns": 4}
        if ctx.safe_mode or ctx.config_only:
            cfg["max_turns"] = min(int(cfg.get("max_turns", 4)), 4)

        effective_level = 0 if (ctx.safe_mode or ctx.config_only) else self.autonomy_level
        tool_names = (
            ["get_system_state", "patch_config", "submit_candidate"]
            if effective_level == 0
            else ["get_system_state", "read_file_lines", "patch_config", "replace_line_range", "submit_candidate"]
        )
        compact_tools = _compact_tools(tool_names)

        def _model_fn(messages: list[dict], tools: list[dict]) -> str:
            raw = generate_agent_reply(
                self.model,
                self.tokenizer,
                messages,
                tools,
                max_new_tokens=min(160, self.profile.max_gen_len),
                temperature=float(cfg.get("temperature", 0.2)),
                top_p=float(cfg.get("top_p", 0.9)),
                open_thinking=True,
            )
            return raw.replace(eos_tok, "").replace(pad_tok, "").strip()

        initial_state = {
            "profile": ctx.profile.name,
            "last_fault": "none" if not ctx.safe_mode else "safe_mode_check",
            "champion_id": ctx.parent.node_id if ctx.parent else None,
            "consecutive_failures": 0,
            "training": cfg.get("training", {}),
            "log_tail": f"cycle={ctx.cycle_index} safe_mode={ctx.safe_mode} config_only={ctx.config_only}",
        }
        ep = run_agent_episode(initial_state, compact_tools, _model_fn, executor.dispatch, cfg)
        if not executor.submitted:
            executor.dispatch("submit_candidate", {"hypothesis": f"minimind cycle {ctx.cycle_index}"})
        calls = ep["tool_calls_made"]
        results = ep["tool_results"]
        observed = any(c["name"] in ("get_system_state", "read_log_tail", "read_file_lines") for c in calls)
        first_edit_ok = any(
            res.get("ok")
            for c, res in zip(calls, results)
            if c["name"] in ("patch_config", "replace_line_range", "patch_file_replace")
        )
        return {
            "task_id": f"cycle_{ctx.cycle_index}",
            "task_family": "minimind_autonomous",
            "model_weight": str(self.loaded_path) if self.loaded_path else None,
            "observed_before_edit": observed,
            "first_edit_valid": first_edit_ok,
            "tool_calls": calls,
            "turn_outputs": ep["turn_outputs"],
            "messages": ep["messages"],
            "final_text": ep["final_text"],
            "unfinished": ep["unfinished"],
        }

    def eval_candidate(self, worktree_dir: Path, weight_path: Path | None = None) -> float:
        from selfchange.bootstrap.eval_baseline import evaluate_minimind_candidate

        self._ensure_loaded(weight_path)
        assert self.model is not None and self.tokenizer is not None
        res = evaluate_minimind_candidate(worktree_dir, self.model, self.tokenizer, self.profile, num_code_tasks=4)
        return float(res["extra_score"])


def make_minimind_proposer(
    model_path: Path,
    tokenizer_path: Path,
    profile: ComputeProfile,
):
    """Build a proposer callback powered by a loaded MiniMindForCausalLM checkpoint."""
    return MiniMindProposerSession(model_path, tokenizer_path, profile).propose


def generate_run_report(workspace_dir: Path) -> Path:
    """Write a concise Markdown report summarizing session config, baselines, lineage, and training."""
    workspace_dir = Path(workspace_dir).resolve()
    archive_path = workspace_dir / "archive.jsonl"
    champ_path = workspace_dir / "champion.json"
    traj_path = workspace_dir / "trajectories.jsonl"
    state_path = workspace_dir / "supervisor_state.json"
    fatal_path = workspace_dir / "logs" / "fatal_report.json"
    session_path = workspace_dir / "session_info.json"
    history_path = workspace_dir / "logs" / "train_history.jsonl"

    nodes = [json.loads(ln) for ln in archive_path.read_text(encoding="utf-8").splitlines() if ln.strip()] if archive_path.is_file() else []
    champ = json.loads(champ_path.read_text(encoding="utf-8")).get("champion") if champ_path.is_file() else None
    traj_cnt = sum(1 for ln in traj_path.read_text(encoding="utf-8").splitlines() if ln.strip()) if traj_path.is_file() else 0
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    fatal = json.loads(fatal_path.read_text(encoding="utf-8")) if fatal_path.is_file() else None
    session_info = json.loads(session_path.read_text(encoding="utf-8")) if session_path.is_file() else None
    train_rows = [json.loads(ln) for ln in history_path.read_text(encoding="utf-8").splitlines() if ln.strip()] if history_path.is_file() else []

    lines = [
        "# MiniMind SelfChange Autonomous Run Report",
        "",
        f"- **Cycle Index**: `{state.get('cycle_index', 0)}`",
        f"- **Safe Mode**: `{state.get('safe_mode', False)}` | **Halted Reason**: `{state.get('halted_reason')}`",
        f"- **Trajectories Logged**: `{traj_cnt}` | **Archive Nodes**: `{len(nodes)}`",
    ]

    if session_info:
        lines.extend(
            [
                "",
                "## Session Configuration",
                f"- **Proposer Mode**: `{session_info.get('proposer_mode')}` | **Model Source**: `{session_info.get('model_source')}`",
                f"- **Active Model**: `{session_info.get('active_model_path')}`",
                f"- **Eval With Model**: `{session_info.get('eval_with_model')}` | **Anchor Path**: `{session_info.get('anchor_path')}`",
            ]
        )

    baseline_entries = [
        ("Untouched", workspace_dir / "baseline_untouched.json"),
        ("Prepared", workspace_dir / "baseline_prepared.json"),
    ]
    baseline_rows: list[str] = []
    for label, bpath in baseline_entries:
        if bpath.is_file():
            try:
                card = json.loads(bpath.read_text(encoding="utf-8"))
                t1 = card.get("tier1_language", {})
                t2 = card.get("tier2_toolcall", {})
                t3 = card.get("tier3_code_edit", {})
                baseline_rows.append(
                    f"| {label} | `{t1.get('dialogue_ppl')}` | `{t2.get('json_parse_rate')}` | "
                    f"`{t2.get('schema_validity_rate')}` | `{t3.get('pass_at_1')}` / `{t3.get('pass_at_4')}` | `{card.get('bootstrap_ready')}` |"
                )
            except Exception:
                pass
    if baseline_rows:
        lines.extend(
            [
                "",
                "## Baseline Scorecard Summary",
                "| Baseline | PPL | JSON Parse | Schema Valid | AST pass@1 / pass@4 | Bootstrap Ready |",
                "|---|---|---|---|---|---|",
                *baseline_rows,
            ]
        )

    lines.extend(["", "## Active Champion"])
    if champ:
        champ_w = Path(champ["weight_path"]).name if champ.get("weight_path") else "none"
        lines.append(
            f"- **ID**: `{champ['node_id']}` | **Total Score**: `{champ['total_score']}` "
            f"(survival=`{champ['survival_score']}`, eval=`{champ['eval_score']}`, self_repair=`{champ['self_repair_score']}`) | **Weight**: `{champ_w}`"
        )
    else:
        lines.append("- None")

    lines.extend(["", "## Lineage History", "| Node | Status | Score | Eval | Weight | Hypothesis |", "|---|---|---|---|---|---|"])
    for n in nodes[-15:]:
        wname = Path(n["weight_path"]).name if n.get("weight_path") else "-"
        lines.append(f"| `{n['node_id']}` | `{n['status']}` | `{n['total_score']}` | `{n.get('eval_score')}` | `{wname}` | {n['hypothesis']} |")

    if train_rows:
        lines.extend(
            [
                "",
                "## Periodic Self-Training History",
                "| Cycle | Stage / Event | OK | Exit | RFT / DPO | Loss | Weight / Note |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for tr in train_rows[-15:]:
            kind = tr.get("event") or tr.get("stage", "-")
            ok_val = tr["accepted"] if "accepted" in tr else tr.get("ok")
            exit_c = tr.get("exit_code", "-")
            counts = f"{tr.get('rft_count', '-')}/{tr.get('dpo_count', '-')}"
            loss_val = tr.get("final_loss", tr.get("total_score", "-"))
            note = tr.get("output_weight_path") or tr.get("weight_path") or tr.get("rejected_weight_path") or tr.get("merge_error") or tr.get("skipped_reason") or "-"
            if isinstance(note, str) and "/" in note and not tr.get("merge_error"):
                note = Path(note).name
            lines.append(f"| `{tr.get('cycle_index', '-')}` | `{kind}` | `{ok_val}` | `{exit_c}` | `{counts}` | `{loss_val}` | `{note}` |")

    if fatal:
        lines.extend(["", "## Last Fatal Report", f"- Reason: `{fatal.get('reason')}` at `{fatal.get('timestamp')}`"])

    out_path = workspace_dir / "report.md"
    atomic_write_text(out_path, "\n".join(lines) + "\n")
    return out_path


def launch_colab_session(
    repo_root: Path,
    workspace_dir: Path,
    drive_dir: Path | None = None,
    profile_name: str | None = None,
    max_cycles: int = 5,
    autonomy_level: int = 1,
    train_every_n_cycles: int = 3,
    dry_run: bool = False,
    kernel_dir: Path | None = None,
    manifest_path: Path | None = None,
    proposer_fn=None,
    model_path: Path | None = None,
    tokenizer_path: Path = Path("model"),
    require_model: bool = False,
    anchor_path: Path | None = None,
    eval_with_model: bool = False,
    extra_eval_fn=None,
) -> dict:
    """Re-entrant Colab session bootstrap: restore Drive -> run cycles -> report -> sync Drive."""
    repo_root = Path(repo_root).resolve()
    workspace_dir = Path(workspace_dir).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    profile = get_profile(profile_name)

    kdir = Path(kernel_dir).resolve() if kernel_dir is not None else repo_root / "selfchange" / "kernel"
    mpath = Path(manifest_path).resolve() if manifest_path is not None else kdir / KERNEL_MANIFEST_FILENAME

    drive_sync = ColabDriveSync(workspace_dir, drive_dir) if drive_dir is not None else None
    restore_info = {"restored": False, "reason": "no_drive_dir"}
    if drive_sync is not None:
        restore_info = drive_sync.restore_from_drive(
            verify_checksums=True,
            mutable_restore_dir=repo_root / "selfchange" / "mutable",
        )
        if restore_info.get("reason") == "checksum_mismatch":
            raise RuntimeError(f"Drive restore aborted due to checksum mismatch: {restore_info}")

    resolved_model: Path | None = None
    model_source = "dry_run" if dry_run else ("custom" if proposer_fn is not None else "none")
    session: MiniMindProposerSession | None = None

    if proposer_fn is None and not dry_run:
        resolved_model, model_source = resolve_model_checkpoint(
            repo_root,
            workspace_dir,
            profile,
            explicit_model_path=model_path,
        )
        if resolved_model is None:
            if require_model:
                raise FileNotFoundError(
                    f"MiniMind checkpoint required (--require-model) but none found "
                    f"(requested={model_path}, workspace={workspace_dir})."
                )
            print(
                f"[WARN] No MiniMind checkpoint found (requested={model_path}); falling back to scripted _default_proposer.",
                file=sys.stderr,
            )
        else:
            resolved_tok = Path(tokenizer_path) if Path(tokenizer_path).is_absolute() else (repo_root / tokenizer_path)
            session = MiniMindProposerSession(
                resolved_model,
                resolved_tok,
                profile,
                autonomy_level=autonomy_level,
            )
            proposer_fn = session.propose
            if eval_with_model and extra_eval_fn is None:
                extra_eval_fn = session.eval_candidate

    proposer_mode = "minimind" if session is not None else ("custom" if proposer_fn is not None else ("dry_run" if dry_run else "scripted_fallback"))

    resolved_anchor: Path | None = None
    if anchor_path is not None:
        cand_anchor = Path(anchor_path) if Path(anchor_path).is_absolute() else (repo_root / anchor_path)
        if cand_anchor.is_file():
            resolved_anchor = cand_anchor.resolve()
    else:
        default_anchor = workspace_dir / "models" / "base" / "data" / "anchor_sft.jsonl"
        if default_anchor.is_file():
            resolved_anchor = default_anchor.resolve()

    session_info = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile.name,
        "proposer_mode": proposer_mode,
        "model_source": model_source,
        "active_model_path": str(resolved_model) if resolved_model else None,
        "eval_with_model": bool(extra_eval_fn is not None),
        "anchor_path": str(resolved_anchor) if resolved_anchor else None,
        "require_model": require_model,
        "dry_run": dry_run,
    }
    atomic_write_text(workspace_dir / "session_info.json", json.dumps(session_info, indent=2, ensure_ascii=False) + "\n")

    runner = AutonomousCycleRunner(
        repo_root=repo_root,
        workspace_dir=workspace_dir,
        profile=profile,
        proposer_fn=proposer_fn,
        kernel_dir=kdir,
        manifest_path=mpath,
        cycle_config=CycleConfig(
            autonomy_level=autonomy_level,
            train_every_n_cycles=train_every_n_cycles,
            dry_run_training=dry_run,
            anchor_path=resolved_anchor,
            base_weight_path=resolved_model,
        ),
        extra_eval_fn=extra_eval_fn,
    )

    if not runner.registry.load_archive():
        head_proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True)
        head_sha = head_proc.stdout.strip() if head_proc.returncode == 0 else "0" * 40
        surv_s, eval_s, rep_s, eff_s, tot_s = 1.0, 1.0, 1.0, 1.0, 0.95
        if extra_eval_fn is not None:
            cal = runner.verifier.verify_candidate(
                repo_root,
                check_git=False,
                extra_eval_fn=lambda wt: runner._invoke_extra_eval(wt, resolved_model),
                min_extra_score=0.0,
            )
            surv_s, eval_s, rep_s, eff_s, tot_s = (
                cal.survival_score,
                cal.eval_score,
                cal.self_repair_score,
                cal.efficiency_score,
                cal.total_score,
            )
            if "gate3" in cal.gates:
                runner._best_extra_score = float(cal.gates["gate3"].details.get("extra_score", 0.0))
        runner.registry.register_node(
            LineageNode(
                node_id="bootstrap_0",
                parent_id=None,
                commit_sha=head_sha,
                weight_path=str(resolved_model) if resolved_model else None,
                survival_score=surv_s,
                eval_score=eval_s,
                self_repair_score=rep_s,
                efficiency_score=eff_s,
                total_score=tot_s,
                status="bootstrap",
                hypothesis="initial verified baseline",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        )

    def _on_cycle_end(_st, _out) -> None:
        generate_run_report(workspace_dir)
        if drive_sync is not None:
            drive_sync.sync_to_drive(repo_root / "selfchange" / "mutable")

    sup = Supervisor(
        repo_root=repo_root,
        workspace_dir=workspace_dir,
        profile=profile,
        cycle_runner=runner,
        kernel_dir=kdir,
        manifest_path=mpath,
        register_signals=False,
        on_cycle_end=_on_cycle_end,
    )
    sup.load_state()
    final_state = sup.run(max_cycles=max_cycles)

    report_path = generate_run_report(workspace_dir)
    if drive_sync is not None:
        drive_sync.sync_to_drive(repo_root / "selfchange" / "mutable")
    champ = runner.registry.get_champion()
    return {
        "restore": restore_info,
        "proposer_mode": proposer_mode,
        "active_model_path": str(resolved_model) if resolved_model else None,
        "model_source": model_source,
        "eval_with_model": bool(extra_eval_fn is not None),
        "state": asdict(final_state),
        "champion": champ.to_dict() if champ else None,
        "report_path": str(report_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-entrant Colab launcher for MiniMind selfchange")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--workspace-dir", type=Path, default=Path(__file__).resolve().parents[1] / "workspace")
    parser.add_argument("--drive-dir", type=Path, default=None)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    parser.add_argument("--anchor-path", type=Path, default=None)
    parser.add_argument("--max-cycles", type=int, default=5)
    parser.add_argument("--autonomy-level", type=int, default=1)
    parser.add_argument("--train-every-n-cycles", type=int, default=3)
    parser.add_argument("--require-model", action="store_true")
    parser.add_argument("--eval-with-model", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    res = launch_colab_session(
        repo_root=args.repo_root,
        workspace_dir=args.workspace_dir,
        drive_dir=args.drive_dir,
        profile_name=args.profile,
        max_cycles=args.max_cycles,
        autonomy_level=args.autonomy_level,
        train_every_n_cycles=args.train_every_n_cycles,
        dry_run=args.dry_run,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        require_model=args.require_model,
        anchor_path=args.anchor_path,
        eval_with_model=args.eval_with_model,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
