from __future__ import annotations

import argparse
import json
import signal
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from selfchange.kernel.budgets import ComputeProfile, check_resource_headroom, get_profile
from selfchange.kernel.integrity import (
    KERNEL_MANIFEST_FILENAME,
    atomic_write_text,
    generate_manifest,
    verify_kernel_integrity,
)
from selfchange.kernel.registry import LineageNode, LineageRegistry


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SupervisorState:
    """Mutable runtime counters and escalation state tracked by Supervisor."""

    cycle_index: int = 0
    consecutive_failures: int = 0
    safe_mode: bool = False
    safe_mode_failures: int = 0
    last_heartbeat: str = ""
    halted_reason: str | None = None
    config_only: bool = False


@dataclass
class CycleContext:
    """Read-only context passed into a single autonomous cycle execution."""

    cycle_index: int
    profile: ComputeProfile
    workspace_dir: Path
    parent: LineageNode | None
    safe_mode: bool
    config_only: bool


@dataclass
class CycleOutcome:
    """Structured result returned by a cycle_runner callback."""

    passed: bool
    total_score: float = 0.0
    candidate_node: LineageNode | None = None
    error: str | None = None
    fatal: bool = False


class Supervisor:
    """Immutable outer loop enforcing integrity, budgets, kill switches, and L1/L2/L3 escalation."""

    def __init__(
        self,
        repo_root: Path,
        workspace_dir: Path,
        profile: ComputeProfile | None = None,
        cycle_runner: Callable[[CycleContext], CycleOutcome] | None = None,
        kernel_dir: Path | None = None,
        manifest_path: Path | None = None,
        l2_failure_threshold: int = 3,
        l3_safe_mode_threshold: int = 2,
        register_signals: bool = False,
        on_cycle_end: Callable[[SupervisorState, CycleOutcome], None] | None = None,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.workspace_dir = Path(workspace_dir).resolve()
        self.profile = profile if profile is not None else get_profile()
        self.cycle_runner = cycle_runner
        self.on_cycle_end = on_cycle_end
        self.kernel_dir = Path(kernel_dir).resolve() if kernel_dir is not None else Path(__file__).resolve().parent
        self.manifest_path = (
            Path(manifest_path).resolve()
            if manifest_path is not None
            else self.kernel_dir / KERNEL_MANIFEST_FILENAME
        )
        self.l2_failure_threshold = l2_failure_threshold
        self.l3_safe_mode_threshold = l3_safe_mode_threshold
        self.registry = LineageRegistry(self.workspace_dir)
        self.state = SupervisorState()
        self.state_path = self.workspace_dir / "supervisor_state.json"
        self._signal_received = False

        if register_signals:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)

    def save_state(self) -> None:
        """Atomically persist SupervisorState for Colab preemption recovery."""
        atomic_write_text(self.state_path, json.dumps(asdict(self.state), indent=2, ensure_ascii=False) + "\n")

    def load_state(self) -> bool:
        """Restore persisted SupervisorState if present; returns True when loaded."""
        if not self.state_path.is_file():
            return False
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.state = SupervisorState(
            cycle_index=int(data.get("cycle_index", 0)),
            consecutive_failures=int(data.get("consecutive_failures", 0)),
            safe_mode=bool(data.get("safe_mode", False)),
            safe_mode_failures=int(data.get("safe_mode_failures", 0)),
            last_heartbeat=str(data.get("last_heartbeat", "")),
            halted_reason=data.get("halted_reason"),
            config_only=bool(data.get("config_only", False)),
        )
        return True

    def _handle_signal(self, _signum: int, _frame: object) -> None:
        self._signal_received = True

    def stop_requested(self) -> str | None:
        """Return shutdown trigger reason if STOP sentinel or termination signal is active."""
        if self._signal_received:
            return "signal"
        if (self.workspace_dir / "STOP").exists():
            return "stop_file"
        return None

    def write_heartbeat(self, phase: str) -> None:
        """Atomically persist current supervisor state to HEARTBEAT.json."""
        ts = _utc_now()
        self.state.last_heartbeat = ts
        payload = {
            "phase": phase,
            "timestamp": ts,
            "profile": self.profile.name,
            "state": asdict(self.state),
        }
        atomic_write_text(
            self.workspace_dir / "HEARTBEAT.json",
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        )

    def write_fatal_report(self, reason: str, details: dict) -> Path:
        """Atomically persist a diagnostic fatal report before clean halt."""
        report_path = self.workspace_dir / "logs" / "fatal_report.json"
        payload = {
            "timestamp": _utc_now(),
            "reason": reason,
            "profile": self.profile.name,
            "state": asdict(self.state),
            "details": details,
        }
        atomic_write_text(report_path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return report_path

    def preflight(self) -> tuple[bool, str | None, dict]:
        """Run stop-switch check, kernel SHA-256 verification, and disk headroom validation."""
        stop_reason = self.stop_requested()
        if stop_reason is not None:
            return False, stop_reason, {"stop_reason": stop_reason}

        integrity = verify_kernel_integrity(self.kernel_dir, self.manifest_path)
        if not integrity.ok:
            return False, "kernel_integrity_violation", asdict(integrity)

        disk_ok, disk_info = check_resource_headroom(self.workspace_dir, self.profile)
        if not disk_ok:
            self.registry.garbage_collect_checkpoints()
            disk_ok, disk_info = check_resource_headroom(self.workspace_dir, self.profile)
            if not disk_ok:
                return False, "resource_exhausted", disk_info

        return True, None, {"integrity": asdict(integrity), "resources": disk_info}

    def _safe_select_parent(self) -> LineageNode | None:
        archive = self.registry.load_archive()
        if not any(n.status != "rejected" for n in archive):
            return None
        return self.registry.select_parent()

    def _enter_safe_mode(self) -> None:
        self.state.safe_mode = True
        self.state.config_only = True
        self.state.safe_mode_failures = 0

        eligible = [n for n in self.registry.load_archive() if n.status != "rejected"]
        if not eligible:
            return
        current = self.registry.get_champion()
        alternatives = [n for n in eligible if current is None or n.node_id != current.node_id]
        pool = alternatives if alternatives else eligible
        best_ancestor = max(pool, key=lambda n: n.total_score)
        if current is None or best_ancestor.node_id != current.node_id:
            self.registry.rollback_to_node(best_ancestor.node_id)

    def _record_outcome(self, outcome: CycleOutcome) -> None:
        if outcome.passed:
            self.state.consecutive_failures = 0
            self.state.safe_mode = False
            self.state.safe_mode_failures = 0
            self.state.config_only = False
            if outcome.candidate_node is not None:
                self.registry.register_node(outcome.candidate_node, promote_if_best=True)
                self.registry.garbage_collect_checkpoints(
                    keep_top_k=max(1, self.profile.checkpoint_retention_count // 2)
                )
            return

        self.state.consecutive_failures += 1
        if outcome.candidate_node is not None:
            rejected = LineageNode(
                node_id=outcome.candidate_node.node_id,
                parent_id=outcome.candidate_node.parent_id,
                commit_sha=outcome.candidate_node.commit_sha,
                weight_path=outcome.candidate_node.weight_path,
                survival_score=outcome.candidate_node.survival_score,
                eval_score=outcome.candidate_node.eval_score,
                self_repair_score=outcome.candidate_node.self_repair_score,
                efficiency_score=outcome.candidate_node.efficiency_score,
                total_score=outcome.candidate_node.total_score,
                status="rejected",
                hypothesis=outcome.candidate_node.hypothesis,
                created_at=outcome.candidate_node.created_at,
            )
            self.registry.register_node(rejected, promote_if_best=False)

        if outcome.fatal:
            reason = outcome.error or "fatal_cycle_error"
            self.state.halted_reason = reason
            self.write_fatal_report(
                reason,
                {"cycle_index": self.state.cycle_index, "error": outcome.error},
            )
            return

        if self.state.safe_mode:
            self.state.safe_mode_failures += 1
            if self.state.safe_mode_failures >= self.l3_safe_mode_threshold:
                self.state.halted_reason = "safe_mode_exhausted"
                self.write_fatal_report(
                    "safe_mode_exhausted",
                    {
                        "cycle_index": self.state.cycle_index,
                        "consecutive_failures": self.state.consecutive_failures,
                        "safe_mode_failures": self.state.safe_mode_failures,
                        "last_error": outcome.error,
                    },
                )
        elif self.state.consecutive_failures >= self.l2_failure_threshold:
            self._enter_safe_mode()

    def run_once(self) -> SupervisorState:
        """Execute preflight checks and a single supervised cycle."""
        self.write_heartbeat("preflight")
        ok, reason, details = self.preflight()
        if not ok:
            assert reason is not None
            if reason in ("kernel_integrity_violation", "resource_exhausted"):
                self.write_fatal_report(reason, details)
            self.state.halted_reason = reason
            self.write_heartbeat("halted")
            return self.state

        if self.cycle_runner is None:
            raise RuntimeError("Supervisor cannot run a cycle without a cycle_runner")

        parent = self._safe_select_parent()
        self.state.cycle_index += 1
        ctx = CycleContext(
            cycle_index=self.state.cycle_index,
            profile=self.profile,
            workspace_dir=self.workspace_dir,
            parent=parent,
            safe_mode=self.state.safe_mode,
            config_only=self.state.config_only,
        )

        self.write_heartbeat("running_cycle")
        try:
            outcome = self.cycle_runner(ctx)
        except Exception as exc:
            outcome = CycleOutcome(passed=False, error=repr(exc), fatal=False)

        self._record_outcome(outcome)
        self.save_state()
        self.write_heartbeat("halted" if self.state.halted_reason else "idle")
        if self.on_cycle_end is not None:
            self.on_cycle_end(self.state, outcome)
        return self.state

    def run(self, max_cycles: int | None = None) -> SupervisorState:
        """Run bounded autonomous cycles until limit, stop switch, or L3 halt."""
        limit = self.profile.max_cycles_per_run if max_cycles is None else max_cycles
        for _ in range(limit):
            if self.state.halted_reason or self.stop_requested():
                if not self.state.halted_reason:
                    self.state.halted_reason = self.stop_requested()
                break
            self.run_once()
            if self.state.halted_reason:
                break
        return self.state


def main() -> int:
    parser = argparse.ArgumentParser(description="Self-Change Immutable Kernel Supervisor")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "workspace",
    )
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--generate-manifest", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--autonomy-level", type=int, default=1)
    parser.add_argument("--dry-run-training", action="store_true")
    parser.add_argument("--drive-dir", type=Path, default=None)
    args = parser.parse_args()

    kernel_dir = Path(__file__).resolve().parent
    manifest_path = kernel_dir / KERNEL_MANIFEST_FILENAME

    if args.generate_manifest:
        entries = generate_manifest(kernel_dir, manifest_path=manifest_path, write=True)
        print(json.dumps({"generated": str(manifest_path), "files": sorted(entries)}, indent=2))
        return 0

    profile = get_profile(args.profile)
    drive_sync = None
    if args.drive_dir is not None:
        from selfchange.kernel.colab_sync import ColabDriveSync

        drive_sync = ColabDriveSync(args.workspace_dir, args.drive_dir)
        drive_sync.restore_from_drive(verify_checksums=True)

    runner = None
    on_end = None
    if args.run:
        from selfchange.kernel.autonomous_cycle import AutonomousCycleRunner, CycleConfig

        runner = AutonomousCycleRunner(
            repo_root=args.repo_root,
            workspace_dir=args.workspace_dir,
            profile=profile,
            kernel_dir=kernel_dir,
            manifest_path=manifest_path,
            cycle_config=CycleConfig(
                autonomy_level=args.autonomy_level,
                dry_run_training=args.dry_run_training,
            ),
        )
        if drive_sync is not None:
            on_end = lambda _st, _out: drive_sync.sync_to_drive(args.repo_root / "selfchange" / "mutable")

    sup = Supervisor(
        repo_root=args.repo_root,
        workspace_dir=args.workspace_dir,
        profile=profile,
        cycle_runner=runner,
        kernel_dir=kernel_dir,
        manifest_path=manifest_path,
        register_signals=True,
        on_cycle_end=on_end,
    )
    sup.load_state()

    if args.check_only:
        ok, reason, details = sup.preflight()
        print(json.dumps({"ok": ok, "reason": reason, "details": details}, indent=2))
        return 0 if ok else 1

    if args.run:
        final_state = sup.run(max_cycles=args.max_cycles)
        print(json.dumps(asdict(final_state), indent=2))
        return 0 if final_state.halted_reason is None else 1

    print("Use --generate-manifest, --check-only, or --run.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
