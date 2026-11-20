from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from selfchange.kernel.budgets import get_profile
from selfchange.kernel.fault_drills import run_drill_suite
from selfchange.kernel.integrity import KERNEL_MANIFEST_FILENAME
from selfchange.kernel.registry import LineageNode
from selfchange.kernel.sandbox import scan_mutable_tree
from selfchange.kernel.supervisor import CycleContext, CycleOutcome, Supervisor
from selfchange.kernel.tool_runtime import WorktreeToolExecutor
from selfchange.kernel.verifier import Verifier, compute_rlvr_reward
from selfchange.kernel.worktree import WorktreeManager


def _init_repo_with_mutable(repo: Path, live_mutable_dir: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "test-user"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "test@local"], cwd=repo, check=True, capture_output=True, text=True)
    dst_mutable = repo / "selfchange" / "mutable"
    dst_mutable.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        live_mutable_dir,
        dst_mutable,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "initial mutable"], cwd=repo, check=True, capture_output=True, text=True)


class TestFaultDrillsAndVerifier(unittest.TestCase):
    def setUp(self) -> None:
        self.selfchange_dir = Path(__file__).resolve().parents[1]
        self.repo_root = self.selfchange_dir.parent
        self.live_kernel_dir = self.selfchange_dir / "kernel"
        self.live_mutable_dir = self.selfchange_dir / "mutable"

    def _copy_valid_kernel(self, dest_dir: Path) -> Path:
        kdir = dest_dir / "kernel"
        shutil.copytree(
            self.live_kernel_dir,
            kdir,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        return kdir

    def test_mutable_baseline_passes_ast_and_line_budget(self) -> None:
        report = scan_mutable_tree(self.live_mutable_dir, run_py_compile=True)
        self.assertTrue(report.ok, msg=f"AST/compile violations: {report.violations}")
        self.assertGreaterEqual(report.scanned_files, 6)
        for py_file in sorted(self.live_mutable_dir.glob("*.py")):
            line_count = len(py_file.read_text(encoding="utf-8").splitlines())
            self.assertLessEqual(line_count, 120, msg=f"{py_file.name} has {line_count} lines (> 120)")

    def test_all_five_fault_drills_and_noop_pass_on_baseline(self) -> None:
        report = run_drill_suite(self.repo_root, seeds=[42, 7])
        self.assertTrue(report.passed_all, msg=f"Drill failures: {report.results}")
        self.assertEqual(report.survival_score, 1.0)
        self.assertEqual(report.self_repair_score, 1.0)
        self.assertEqual(report.healthy_noop_rate, 1.0)
        self.assertEqual(len(report.results), 12)

    def test_broken_policy_fails_drills_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_wt = Path(tmp).resolve()
            dst_mutable = tmp_wt / "selfchange" / "mutable"
            dst_mutable.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                self.live_mutable_dir,
                dst_mutable,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            (dst_mutable / "recovery_policy.py").write_text(
                "def classify_fault(state, log_tail):\n    return 'healthy'\n\n"
                "def propose_line_fix(source_lines, error_info):\n    return None\n\n"
                "def plan_recovery_action(fault, state, config, log_tail=''):\n    return {'tool': 'noop', 'arguments': {}}\n",
                encoding="utf-8",
            )
            report = run_drill_suite(tmp_wt, seeds=[42])
            self.assertFalse(report.passed_all)
            by_id = {r.drill_id: r for r in report.results}
            self.assertFalse(by_id["drill_a_oom"].passed)
            self.assertFalse(by_id["drill_b_nan_loss"].passed)
            self.assertFalse(by_id["drill_c_broken_module"].passed)
            self.assertFalse(by_id["drill_d_corrupt_checkpoint"].passed)

    def test_tool_executor_blocks_escape_and_reverts_invalid_edits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_wt = Path(tmp).resolve()
            dst_mutable = tmp_wt / "selfchange" / "mutable"
            dst_mutable.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                self.live_mutable_dir,
                dst_mutable,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            executor_l1 = WorktreeToolExecutor(tmp_wt, autonomy_level=1)

            escape_res = executor_l1.dispatch("read_file_lines", {"path": "../kernel/supervisor.py"})
            self.assertFalse(escape_res["ok"])

            syntax_res = executor_l1.dispatch(
                "replace_line_range",
                {"path": "prompts.py", "start_line": 1, "end_line": 1, "lines": ["def broken(:"]},
            )
            self.assertFalse(syntax_res["ok"])
            self.assertTrue(syntax_res.get("reverted"))

            ast_res = executor_l1.dispatch(
                "replace_line_range",
                {"path": "prompts.py", "start_line": 1, "end_line": 1, "lines": ["import subprocess"]},
            )
            self.assertFalse(ast_res["ok"])
            self.assertTrue(ast_res.get("reverted"))

            l1_block = executor_l1.dispatch(
                "patch_file_replace",
                {"path": "prompts.py", "target": "SYSTEM_PROMPT", "replacement": "SYSTEM_PROMPT"},
            )
            self.assertFalse(l1_block["ok"])

            executor_l2 = WorktreeToolExecutor(tmp_wt, autonomy_level=2)
            l2_bad = executor_l2.dispatch(
                "patch_file_replace",
                {
                    "path": "prompts.py",
                    "target": "from __future__ import annotations",
                    "replacement": "import subprocess",
                },
            )
            self.assertFalse(l2_bad["ok"])
            self.assertTrue(l2_bad.get("reverted"))

            cfg_res = executor_l1.dispatch(
                "patch_config",
                {"updates": {"training": {"learning_rate": 10.0, "batch_size": 999}}},
            )
            self.assertTrue(cfg_res["ok"])
            self.assertLessEqual(cfg_res["config"]["training"]["learning_rate"], 5e-4)
            self.assertLessEqual(cfg_res["config"]["training"]["batch_size"], 16)

    def test_four_tier_verifier_gates_and_supervisor_integration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            kdir = self._copy_valid_kernel(tmp_path)
            repo = tmp_path / "repo"
            repo.mkdir()
            _init_repo_with_mutable(repo, self.live_mutable_dir)
            ws = tmp_path / "workspace"
            ws.mkdir()

            profile = get_profile("local_intel_cpu")
            verifier = Verifier(kernel_dir=kdir, manifest_path=kdir / KERNEL_MANIFEST_FILENAME, profile=profile)
            mgr = WorktreeManager(repo, ws / "worktrees")

            # 1. Good candidate passes all 4 gates and promotes via Supervisor
            h_good = mgr.create_worktree("cand_good")
            try:
                exec_good = WorktreeToolExecutor(h_good.path, autonomy_level=1, base_ref=h_good.base_commit)
                exec_good.dispatch("patch_config", {"updates": {"temperature": 0.15}})
                rep_good = verifier.verify_candidate(h_good.path, base_ref=h_good.base_commit, check_git=True, seeds=[42])
                self.assertTrue(rep_good.passed, msg=f"Good candidate failed: {rep_good}")
                self.assertGreater(rep_good.total_score, 0.8)
                new_sha = mgr.commit_mutable_changes(h_good, "tune temperature")

                sup = Supervisor(
                    repo_root=repo,
                    workspace_dir=ws,
                    profile=profile,
                    kernel_dir=kdir,
                    manifest_path=kdir / KERNEL_MANIFEST_FILENAME,
                    cycle_runner=lambda ctx: CycleOutcome(
                        passed=True,
                        total_score=rep_good.total_score,
                        candidate_node=LineageNode(
                            node_id="good_1",
                            parent_id=None,
                            commit_sha=new_sha,
                            weight_path=None,
                            survival_score=rep_good.survival_score,
                            eval_score=rep_good.eval_score,
                            self_repair_score=rep_good.self_repair_score,
                            efficiency_score=rep_good.efficiency_score,
                            total_score=rep_good.total_score,
                            status="archived",
                            hypothesis="tune temp",
                            created_at="2026-11-19T00:00:00Z",
                        ),
                    ),
                )
                sup.run(max_cycles=1)
                self.assertEqual(sup.registry.get_champion().node_id, "good_1")
            finally:
                mgr.remove_worktree(h_good)

            # 2. Unsafe AST edit fails Gate 1 immediately without running Gate 2
            h_unsafe = mgr.create_worktree("cand_unsafe")
            try:
                p = h_unsafe.path / "selfchange" / "mutable" / "prompts.py"
                p.write_text(p.read_text(encoding="utf-8") + "\nimport socket\n", encoding="utf-8")
                rep_unsafe = verifier.verify_candidate(h_unsafe.path, base_ref=h_unsafe.base_commit, check_git=True)
                self.assertFalse(rep_unsafe.passed)
                self.assertFalse(rep_unsafe.gates["gate1"].passed)
                self.assertNotIn("gate2", rep_unsafe.gates)
            finally:
                mgr.remove_worktree(h_unsafe)

            # 3. Broken recovery policy fails Gate 2 and leaves champion unchanged
            h_bad = mgr.create_worktree("cand_bad_policy")
            try:
                (h_bad.path / "selfchange" / "mutable" / "recovery_policy.py").write_text(
                    "def classify_fault(state, log_tail):\n    return 'healthy'\n\n"
                    "def propose_line_fix(source_lines, error_info):\n    return None\n\n"
                    "def plan_recovery_action(fault, state, config, log_tail=''):\n    return {'tool': 'noop', 'arguments': {}}\n",
                    encoding="utf-8",
                )
                rep_bad = verifier.verify_candidate(h_bad.path, base_ref=h_bad.base_commit, check_git=True, seeds=[42])
                self.assertFalse(rep_bad.passed)
                self.assertTrue(rep_bad.gates["gate1"].passed)
                self.assertFalse(rep_bad.gates["gate2"].passed)
            finally:
                mgr.remove_worktree(h_bad)

            # 4. STOP file halts supervisor cleanly
            (ws / "STOP").write_text("stop\n", encoding="utf-8")
            sup_stop = Supervisor(
                repo_root=repo,
                workspace_dir=ws,
                profile=profile,
                kernel_dir=kdir,
                manifest_path=kdir / KERNEL_MANIFEST_FILENAME,
                cycle_runner=lambda _ctx: CycleOutcome(passed=True),
            )
            st = sup_stop.run(max_cycles=1)
            self.assertEqual(st.halted_reason, "stop_file")
            self.assertEqual(sup_stop.registry.get_champion().node_id, "good_1")

    def test_rlvr_reward_formula_and_clipping(self) -> None:
        good_turn = (
            "<think>" + ("diagnosing oom error cleanly " * 2) + "</think>"
            '<tool_call>{"name": "patch_config", "arguments": {"updates": {"training": {"batch_size": 1}}}}</tool_call>'
        )
        r_good = compute_rlvr_reward(
            turn_outputs=[good_turn],
            tool_calls=[{"name": "patch_config", "arguments": {}}],
            observed_before_edit=True,
            first_edit_valid=True,
            task_passed=True,
            boundary_violated=False,
            unfinished=False,
            over_budget=False,
            open_thinking=True,
        )
        self.assertGreaterEqual(r_good, 2.5)
        self.assertLessEqual(r_good, 3.0)

        rep_spam = "repeat word " * 40 + "<tool_call>{bad json"
        r_bad = compute_rlvr_reward(
            turn_outputs=[rep_spam],
            tool_calls=[],
            observed_before_edit=False,
            first_edit_valid=False,
            task_passed=False,
            boundary_violated=True,
            unfinished=True,
            over_budget=True,
            open_thinking=True,
        )
        self.assertGreaterEqual(r_bad, -3.0)
        self.assertLessEqual(r_bad, -2.0)


if __name__ == "__main__":
    unittest.main()
