from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from selfchange.kernel.autonomous_cycle import AutonomousCycleRunner, CycleConfig
from selfchange.kernel.budgets import get_profile
from selfchange.kernel.colab_sync import ColabDriveSync
from selfchange.kernel.integrity import KERNEL_MANIFEST_FILENAME, atomic_write_text
from selfchange.kernel.registry import LineageNode
from selfchange.kernel.supervisor import Supervisor


def generate_run_report(workspace_dir: Path) -> Path:
    """Write a concise Markdown report summarizing lineage, trajectories, and supervisor state."""
    workspace_dir = Path(workspace_dir).resolve()
    archive_path = workspace_dir / "archive.jsonl"
    champ_path = workspace_dir / "champion.json"
    traj_path = workspace_dir / "trajectories.jsonl"
    state_path = workspace_dir / "supervisor_state.json"
    fatal_path = workspace_dir / "logs" / "fatal_report.json"

    nodes = [json.loads(ln) for ln in archive_path.read_text(encoding="utf-8").splitlines() if ln.strip()] if archive_path.is_file() else []
    champ = json.loads(champ_path.read_text(encoding="utf-8")).get("champion") if champ_path.is_file() else None
    traj_cnt = sum(1 for ln in traj_path.read_text(encoding="utf-8").splitlines() if ln.strip()) if traj_path.is_file() else 0
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    fatal = json.loads(fatal_path.read_text(encoding="utf-8")) if fatal_path.is_file() else None

    lines = [
        "# MiniMind SelfChange Autonomous Run Report",
        "",
        f"- **Cycle Index**: `{state.get('cycle_index', 0)}`",
        f"- **Safe Mode**: `{state.get('safe_mode', False)}` | **Halted Reason**: `{state.get('halted_reason')}`",
        f"- **Trajectories Logged**: `{traj_cnt}` | **Archive Nodes**: `{len(nodes)}`",
        "",
        "## Active Champion",
    ]
    if champ:
        lines.append(
            f"- **ID**: `{champ['node_id']}` | **Total Score**: `{champ['total_score']}` "
            f"(survival=`{champ['survival_score']}`, eval=`{champ['eval_score']}`, self_repair=`{champ['self_repair_score']}`)"
        )
    else:
        lines.append("- None")

    lines.extend(["", "## Lineage History", "| Node | Status | Score | Hypothesis |", "|---|---|---|---|"])
    for n in nodes[-15:]:
        lines.append(f"| `{n['node_id']}` | `{n['status']}` | `{n['total_score']}` | {n['hypothesis']} |")

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
) -> dict:
    """Re-entrant Colab session bootstrap: restore Drive -> run cycles -> sync Drive -> report."""
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
        ),
    )

    if not runner.registry.load_archive():
        head_proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True)
        head_sha = head_proc.stdout.strip() if head_proc.returncode == 0 else "0" * 40
        runner.registry.register_node(
            LineageNode(
                node_id="bootstrap_0",
                parent_id=None,
                commit_sha=head_sha,
                weight_path=None,
                survival_score=1.0,
                eval_score=1.0,
                self_repair_score=1.0,
                efficiency_score=1.0,
                total_score=0.95,
                status="bootstrap",
                hypothesis="initial verified baseline",
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        )

    on_end = (
        (lambda _st, _out: drive_sync.sync_to_drive(repo_root / "selfchange" / "mutable"))
        if drive_sync is not None
        else None
    )
    sup = Supervisor(
        repo_root=repo_root,
        workspace_dir=workspace_dir,
        profile=profile,
        cycle_runner=runner,
        kernel_dir=kdir,
        manifest_path=mpath,
        register_signals=False,
        on_cycle_end=on_end,
    )
    sup.load_state()
    final_state = sup.run(max_cycles=max_cycles)

    if drive_sync is not None:
        drive_sync.sync_to_drive(repo_root / "selfchange" / "mutable")
    report_path = generate_run_report(workspace_dir)
    champ = runner.registry.get_champion()
    return {
        "restore": restore_info,
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
    parser.add_argument("--max-cycles", type=int, default=5)
    parser.add_argument("--autonomy-level", type=int, default=1)
    parser.add_argument("--train-every-n-cycles", type=int, default=3)
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
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
