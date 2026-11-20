from __future__ import annotations

import json
import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from selfchange.kernel.budgets import get_profile
from selfchange.kernel.integrity import KERNEL_MANIFEST_FILENAME
from selfchange.kernel.registry import LineageNode, LineageRegistry
from selfchange.kernel.sandbox import SandboxPolicyError
from selfchange.kernel.supervisor import CycleContext, CycleOutcome, Supervisor
from selfchange.kernel.worktree import BoundaryViolationError, WorktreeManager


def _init_temp_git_repo(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.name", "test-user"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@local"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    (root / "trainer").mkdir(parents=True, exist_ok=True)
    (root / "trainer" / "train_lora.py").write_text("# base trainer\n", encoding="utf-8")
    (root / "selfchange" / "mutable").mkdir(parents=True, exist_ok=True)
    (root / "selfchange" / "mutable" / "policy.py").write_text("THRESHOLD = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


def _make_node(
    node_id: str,
    total_score: float,
    parent_id: str | None = None,
    status: str = "archived",
    weight_path: str | None = None,
) -> LineageNode:
    return LineageNode(
        node_id=node_id,
        parent_id=parent_id,
        commit_sha="deadbeef" * 5,
        weight_path=weight_path,
        survival_score=total_score,
        eval_score=total_score,
        self_repair_score=total_score,
        efficiency_score=total_score,
        total_score=total_score,
        status=status,
        hypothesis=f"hypothesis for {node_id}",
        created_at="2026-11-19T00:00:00+00:00",
    )


class TestWorktreeRegistrySupervisor(unittest.TestCase):
    def setUp(self) -> None:
        self.live_kernel_dir = Path(__file__).resolve().parents[1] / "kernel"

    def _copy_valid_kernel(self, tmp_path: Path) -> Path:
        kdir = tmp_path / "kernel"
        shutil.copytree(
            self.live_kernel_dir,
            kdir,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        return kdir

    def test_worktree_create_commit_and_remove(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            repo = tmp_path / "repo"
            repo.mkdir()
            _init_temp_git_repo(repo)
            wt_root = tmp_path / "worktrees"
            mgr = WorktreeManager(repo, wt_root)

            base_sha = mgr.resolve_commit("HEAD")
            handle = mgr.create_worktree("cand_001")
            self.assertTrue(handle.path.is_dir())
            (handle.path / "selfchange" / "mutable" / "policy.py").write_text(
                "THRESHOLD = 2\n", encoding="utf-8"
            )
            new_sha = mgr.commit_mutable_changes(handle, "bump threshold")
            self.assertNotEqual(new_sha, base_sha)

            mgr.remove_worktree(handle, delete_branch=True)
            self.assertFalse(handle.path.exists())
            branch_check = subprocess.run(
                ["git", "rev-parse", "--verify", handle.branch],
                cwd=repo,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(branch_check.returncode, 0)

    def test_worktree_rejects_out_of_bounds_edit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            repo = tmp_path / "repo"
            repo.mkdir()
            _init_temp_git_repo(repo)
            mgr = WorktreeManager(repo, tmp_path / "worktrees")
            handle = mgr.create_worktree("cand_oob")
            try:
                (handle.path / "trainer" / "train_lora.py").write_text("# hacked\n", encoding="utf-8")
                with self.assertRaises(BoundaryViolationError):
                    mgr.commit_mutable_changes(handle, "should fail")
            finally:
                mgr.remove_worktree(handle)

    def test_worktree_rejects_ast_policy_violation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            repo = tmp_path / "repo"
            repo.mkdir()
            _init_temp_git_repo(repo)
            mgr = WorktreeManager(repo, tmp_path / "worktrees")
            handle = mgr.create_worktree("cand_ast")
            try:
                (handle.path / "selfchange" / "mutable" / "policy.py").write_text(
                    "import subprocess\n", encoding="utf-8"
                )
                with self.assertRaises(SandboxPolicyError):
                    mgr.commit_mutable_changes(handle, "unsafe import")
            finally:
                mgr.remove_worktree(handle)

    def test_prune_stale_worktrees(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            repo = tmp_path / "repo"
            repo.mkdir()
            _init_temp_git_repo(repo)
            mgr = WorktreeManager(repo, tmp_path / "worktrees")
            h1 = mgr.create_worktree("cand_keep")
            h2 = mgr.create_worktree("cand_drop")
            removed = mgr.prune_stale_worktrees(keep_ids={"cand_keep"})
            self.assertEqual(removed, ["cand_drop"])
            self.assertTrue(h1.path.exists())
            self.assertFalse(h2.path.exists())
            mgr.remove_worktree(h1)

    def test_lineage_registry_promote_select_rollback_gc(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp).resolve() / "workspace"
            reg = LineageRegistry(ws)
            reg.candidates_dir.mkdir(parents=True, exist_ok=True)

            paths = {}
            for name in ("n0", "n1", "n2", "n3", "n_rej"):
                p = reg.candidates_dir / f"{name}.pth"
                p.write_bytes(b"ckpt-" + name.encode("ascii"))
                paths[name] = str(p)

            n0, prom0 = reg.register_node(_make_node("n0", 0.50, status="bootstrap", weight_path=paths["n0"]))
            self.assertTrue(prom0)
            self.assertEqual(reg.get_champion().node_id, "n0")

            _, prom_worse = reg.register_node(_make_node("n1", 0.40, parent_id="n0", weight_path=paths["n1"]))
            self.assertFalse(prom_worse)

            n2, prom_better = reg.register_node(_make_node("n2", 0.80, parent_id="n0", weight_path=paths["n2"]))
            self.assertTrue(prom_better)
            self.assertEqual(n2.status, "champion")
            self.assertEqual(reg.get_champion().node_id, "n2")

            _, prom_mid = reg.register_node(_make_node("n3", 0.65, parent_id="n0", weight_path=paths["n3"]))
            self.assertFalse(prom_mid)

            _, prom_rej = reg.register_node(
                _make_node("n_rej", 0.99, parent_id="n2", status="rejected", weight_path=paths["n_rej"])
            )
            self.assertFalse(prom_rej)
            self.assertEqual(reg.get_champion().node_id, "n2")

            rng = random.Random(123)
            picks = {reg.select_parent(rng=rng, exploit_prob=0.5, top_k=3).node_id for _ in range(40)}
            self.assertNotIn("n_rej", picks)
            self.assertGreater(len(picks), 1)

            rolled = reg.rollback_to_node("n0")
            self.assertEqual(rolled.node_id, "n0")
            self.assertEqual(reg.get_champion().node_id, "n0")

            # Protected weights: current champion (n0), previous champion (n2), and top-1 non-rejected (n2/n3)
            deleted = reg.garbage_collect_checkpoints(keep_top_k=2)
            deleted_names = {p.name for p in deleted}
            self.assertIn("n1.pth", deleted_names)
            self.assertIn("n_rej.pth", deleted_names)
            self.assertTrue(Path(paths["n0"]).exists())
            self.assertTrue(Path(paths["n2"]).exists())
            self.assertTrue(Path(paths["n3"]).exists())

    def test_supervisor_stop_file_and_integrity_halt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            kdir = self._copy_valid_kernel(tmp_path)
            ws = tmp_path / "workspace"
            ws.mkdir()
            (ws / "STOP").write_text("halt\n", encoding="utf-8")

            called = {"count": 0}

            def _runner(_ctx: CycleContext) -> CycleOutcome:
                called["count"] += 1
                return CycleOutcome(passed=True, total_score=1.0)

            sup_stop = Supervisor(
                repo_root=tmp_path,
                workspace_dir=ws,
                profile=get_profile("local_intel_cpu"),
                cycle_runner=_runner,
                kernel_dir=kdir,
                manifest_path=kdir / KERNEL_MANIFEST_FILENAME,
            )
            state_stop = sup_stop.run(max_cycles=3)
            self.assertEqual(state_stop.halted_reason, "stop_file")
            self.assertEqual(called["count"], 0)

            (ws / "STOP").unlink()
            (kdir / "budgets.py").write_text("# corrupted\n", encoding="utf-8")
            sup_bad = Supervisor(
                repo_root=tmp_path,
                workspace_dir=ws,
                profile=get_profile("local_intel_cpu"),
                cycle_runner=_runner,
                kernel_dir=kdir,
                manifest_path=kdir / KERNEL_MANIFEST_FILENAME,
            )
            state_bad = sup_bad.run(max_cycles=3)
            self.assertEqual(state_bad.halted_reason, "kernel_integrity_violation")
            self.assertEqual(called["count"], 0)
            fatal_path = ws / "logs" / "fatal_report.json"
            self.assertTrue(fatal_path.is_file())
            fatal_data = json.loads(fatal_path.read_text(encoding="utf-8"))
            self.assertEqual(fatal_data["reason"], "kernel_integrity_violation")

    def test_supervisor_escalation_l1_l2_l3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            kdir = self._copy_valid_kernel(tmp_path)
            ws = tmp_path / "workspace"
            ws.mkdir()

            seen_safe_modes: list[tuple[bool, bool]] = []

            def _always_fail(ctx: CycleContext) -> CycleOutcome:
                seen_safe_modes.append((ctx.safe_mode, ctx.config_only))
                return CycleOutcome(passed=False, error="drill_failed")

            sup = Supervisor(
                repo_root=tmp_path,
                workspace_dir=ws,
                profile=get_profile("local_intel_cpu"),
                cycle_runner=_always_fail,
                kernel_dir=kdir,
                manifest_path=kdir / KERNEL_MANIFEST_FILENAME,
                l2_failure_threshold=3,
                l3_safe_mode_threshold=2,
            )
            reg = sup.registry
            reg.register_node(_make_node("anc_good", 0.90, status="bootstrap"))
            reg.register_node(_make_node("curr_worse", 0.95, parent_id="anc_good"))

            final_state = sup.run(max_cycles=10)
            self.assertEqual(len(seen_safe_modes), 5)
            self.assertEqual(seen_safe_modes[:3], [(False, False)] * 3)
            self.assertEqual(seen_safe_modes[3:], [(True, True)] * 2)
            self.assertEqual(final_state.halted_reason, "safe_mode_exhausted")
            self.assertEqual(reg.get_champion().node_id, "anc_good")
            self.assertTrue((ws / "logs" / "fatal_report.json").is_file())


if __name__ == "__main__":
    unittest.main()
