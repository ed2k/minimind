from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from selfchange.colab.colab_launcher import launch_colab_session, resolve_model_checkpoint
from selfchange.kernel.autonomous_cycle import AutonomousCycleRunner, CycleConfig
from selfchange.kernel.budgets import get_profile
from selfchange.kernel.colab_sync import ColabDriveSync
from selfchange.kernel.integrity import KERNEL_MANIFEST_FILENAME
from selfchange.kernel.registry import LineageNode
from selfchange.kernel.supervisor import CycleContext, Supervisor
from selfchange.kernel.tool_runtime import WorktreeToolExecutor
from selfchange.kernel.worktree import WorktreeHandle


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
    subprocess.run(["git", "commit", "-m", "init selfchange"], cwd=repo, check=True, capture_output=True, text=True)


class TestEndToEndAutonomousLoop(unittest.TestCase):
    def setUp(self) -> None:
        self.selfchange_dir = Path(__file__).resolve().parents[1]
        self.live_kernel_dir = self.selfchange_dir / "kernel"
        self.live_mutable_dir = self.selfchange_dir / "mutable"

    def _copy_valid_kernel(self, dest: Path) -> Path:
        kdir = dest / "kernel"
        shutil.copytree(
            self.live_kernel_dir,
            kdir,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        return kdir

    def test_full_autonomous_loop_promote_reject_train_drive_wipe_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            kdir = self._copy_valid_kernel(tmp_path)
            manifest = kdir / KERNEL_MANIFEST_FILENAME
            repo = tmp_path / "repo"
            repo.mkdir()
            _init_repo_with_mutable(repo, self.live_mutable_dir)

            ws = tmp_path / "workspace"
            ws.mkdir()
            drive_dir = tmp_path / "drive_mirror"
            profile = get_profile("local_intel_cpu")
            drive_sync = ColabDriveSync(ws, drive_dir)

            def _cycle_proposer(ctx: CycleContext, handle: WorktreeHandle, executor: WorktreeToolExecutor) -> dict:
                if ctx.cycle_index == 1:
                    # Bounded L1 line edit + config adjustment -> should pass Gates 1-4 and commit
                    r1 = executor.dispatch("read_file_lines", {"path": "prompts.py", "start": 1, "end": 12})
                    edit_ok = executor.dispatch(
                        "replace_line_range",
                        {
                            "path": "prompts.py",
                            "start_line": 4,
                            "end_line": 4,
                            "lines": ["SYSTEM_PROMPT = (  # cycle 1 verified edit"],
                        },
                    )
                    cfg_ok = executor.dispatch("patch_config", {"updates": {"temperature": 0.15}})
                    executor.dispatch("submit_candidate", {"hypothesis": "cycle 1 L1 edit + temp tune"})
                    return {
                        "task_id": "e2e_task_1",
                        "task_family": "l1_edit",
                        "observed_before_edit": bool(r1.get("ok")),
                        "first_edit_valid": bool(edit_ok.get("ok")) and bool(cfg_ok.get("ok")),
                        "tool_calls": [{"name": "replace_line_range", "arguments": {}}],
                        "turn_outputs": ['<think>Applying safe L1 prompt comment edit.</think><tool_call>{"name": "submit_candidate", "arguments": {"hypothesis": "ok"}}</tool_call>'],
                        "messages": [{"role": "user", "content": "tune cycle 1"}, {"role": "assistant", "content": "updated prompts.py"}],
                        "final_text": "cycle 1 passed",
                    }
                if ctx.cycle_index == 2:
                    # Direct unsafe mutation bypassing tool -> Gate 1 must reject and preserve champion
                    p = handle.path / "selfchange" / "mutable" / "prompts.py"
                    p.write_text(p.read_text(encoding="utf-8") + "\nimport socket\n", encoding="utf-8")
                    return {
                        "task_id": "e2e_task_1",
                        "task_family": "unsafe_edit",
                        "observed_before_edit": False,
                        "first_edit_valid": False,
                        "tool_calls": [],
                        "turn_outputs": ["bad edit"],
                        "messages": [{"role": "user", "content": "tune cycle 2"}, {"role": "assistant", "content": "import socket"}],
                        "final_text": "unsafe edit",
                    }
                # Cycle 3: valid config update -> triggers periodic dry-run self-training
                executor.dispatch("patch_config", {"updates": {"temperature": 0.18}})
                executor.dispatch("submit_candidate", {"hypothesis": f"cycle {ctx.cycle_index} config tune"})
                return {
                    "task_id": f"e2e_task_{ctx.cycle_index}",
                    "task_family": "config_tune",
                    "observed_before_edit": True,
                    "first_edit_valid": True,
                    "tool_calls": [{"name": "patch_config", "arguments": {"updates": {"temperature": 0.18}}}],
                    "turn_outputs": ['<think>Tuning config cleanly.</think><tool_call>{"name": "patch_config", "arguments": {"updates": {"temperature": 0.18}}}</tool_call>'],
                    "messages": [{"role": "user", "content": "tune cycle 3"}, {"role": "assistant", "content": "tuned config"}],
                    "final_text": "cycle 3 passed",
                }

            runner = AutonomousCycleRunner(
                repo_root=repo,
                workspace_dir=ws,
                profile=profile,
                proposer_fn=_cycle_proposer,
                kernel_dir=kdir,
                manifest_path=manifest,
                cycle_config=CycleConfig(autonomy_level=1, train_every_n_cycles=3, dry_run_training=True),
            )
            runner.registry.register_node(
                LineageNode(
                    node_id="bootstrap_0",
                    parent_id=None,
                    commit_sha=runner.worktree_mgr.resolve_commit("HEAD"),
                    weight_path=None,
                    survival_score=1.0,
                    eval_score=1.0,
                    self_repair_score=1.0,
                    efficiency_score=0.5,
                    total_score=0.90,
                    status="bootstrap",
                    hypothesis="initial",
                    created_at="2026-11-19T00:00:00Z",
                )
            )
            sup = Supervisor(
                repo_root=repo,
                workspace_dir=ws,
                profile=profile,
                cycle_runner=runner,
                kernel_dir=kdir,
                manifest_path=manifest,
                on_cycle_end=lambda _st, _out: drive_sync.sync_to_drive(repo / "selfchange" / "mutable"),
            )
            state3 = sup.run(max_cycles=3)
            self.assertEqual(state3.cycle_index, 3)
            self.assertIsNone(state3.halted_reason)

            archive_nodes = runner.registry.load_archive()
            statuses = [n.status for n in archive_nodes]
            self.assertIn("rejected", statuses)
            champ3 = runner.registry.get_champion()
            self.assertIsNotNone(champ3)
            self.assertIsNotNone(champ3.weight_path)
            self.assertTrue(Path(champ3.weight_path).is_file())
            self.assertTrue((ws / "logs" / "train.log").is_file())
            self.assertTrue((ws / "logs" / "train_history.jsonl").is_file())
            hist_rows = [json.loads(ln) for ln in (ws / "logs" / "train_history.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(hist_rows), 1)
            self.assertTrue(hist_rows[0]["ok"])
            self.assertEqual(hist_rows[0]["cycle_index"], 3)
            self.assertTrue(Path(hist_rows[0]["data_file"]).is_file())
            self.assertEqual([p for p in (ws / "worktrees").glob("*") if p.is_dir()], [])

            # Simulate Colab VM wipe: delete local workspace completely and resume into fresh ws_new
            shutil.rmtree(ws)
            ws_new = tmp_path / "workspace_restored"
            session_res = launch_colab_session(
                repo_root=repo,
                workspace_dir=ws_new,
                drive_dir=drive_dir,
                profile_name="local_intel_cpu",
                max_cycles=1,
                autonomy_level=1,
                train_every_n_cycles=3,
                dry_run=True,
                kernel_dir=kdir,
                manifest_path=manifest,
            )
            self.assertTrue(session_res["restore"]["restored"])
            self.assertEqual(session_res["state"]["cycle_index"], 4)
            self.assertTrue(Path(session_res["report_path"]).is_file())
            self.assertTrue((drive_dir / "report.md").is_file())
            self.assertTrue((drive_dir / "session_info.json").is_file())
            self.assertTrue((drive_dir / "logs" / "train_history.jsonl").is_file())
            self.assertIn("MiniMind SelfChange Autonomous Run Report", Path(session_res["report_path"]).read_text(encoding="utf-8"))

    def test_drive_sync_and_restore_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            ws = tmp_path / "ws"
            drive = tmp_path / "drive"
            (ws / "logs").mkdir(parents=True)
            (ws / "archive.jsonl").write_text('{"node_id": "n0"}\n', encoding="utf-8")
            (ws / "report.md").write_text("# Report\n", encoding="utf-8")
            (ws / "session_info.json").write_text('{"proposer_mode": "minimind"}\n', encoding="utf-8")
            (ws / "baseline_untouched.json").write_text('{"tier1_language": {}}\n', encoding="utf-8")
            (ws / "baseline_prepared.json").write_text('{"tier1_language": {}}\n', encoding="utf-8")
            (ws / "logs" / "train.log").write_text("loss: 0.5\n", encoding="utf-8")
            (ws / "logs" / "train_history.jsonl").write_text('{"cycle_index": 3}\n', encoding="utf-8")

            sync = ColabDriveSync(ws, drive)
            sync.sync_to_drive()
            for rel in ("report.md", "session_info.json", "baseline_untouched.json", "baseline_prepared.json", "logs/train.log", "logs/train_history.jsonl"):
                self.assertTrue((drive / rel).is_file(), msg=f"Missing mirrored diagnostic: {rel}")

            ws_restored = tmp_path / "ws_restored"
            res = ColabDriveSync(ws_restored, drive).restore_from_drive(verify_checksums=True)
            self.assertTrue(res["restored"])
            for rel in ("report.md", "session_info.json", "baseline_untouched.json", "baseline_prepared.json", "logs/train.log", "logs/train_history.jsonl"):
                self.assertTrue((ws_restored / rel).is_file(), msg=f"Missing restored diagnostic: {rel}")

    def test_resolve_checkpoint_priority_and_require_model_fails_loudly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            kdir = self._copy_valid_kernel(tmp_path)
            manifest = kdir / KERNEL_MANIFEST_FILENAME
            repo = tmp_path / "repo"
            repo.mkdir()
            _init_repo_with_mutable(repo, self.live_mutable_dir)
            ws = tmp_path / "ws"
            ws.mkdir()
            profile = get_profile("local_intel_cpu")
            hs = profile.hidden_size

            path, src = resolve_model_checkpoint(repo, ws, profile)
            self.assertIsNone(path)
            self.assertEqual(src, "none")

            with self.assertRaises(FileNotFoundError):
                launch_colab_session(
                    repo_root=repo,
                    workspace_dir=ws,
                    profile_name="local_intel_cpu",
                    dry_run=False,
                    require_model=True,
                    kernel_dir=kdir,
                    manifest_path=manifest,
                )

            # Dry-run JSON stub must not be mistaken for a real binary checkpoint
            stub = ws / "models" / "candidates" / f"offline_dpo_r1_{hs}.pth"
            stub.parent.mkdir(parents=True, exist_ok=True)
            stub.write_text('{"dry_run": true}\n', encoding="utf-8")
            self.assertEqual(resolve_model_checkpoint(repo, ws, profile), (None, "none"))
            stub.unlink()

            out_sft = repo / "out" / f"full_sft_{hs}.pth"
            out_sft.parent.mkdir(parents=True, exist_ok=True)
            out_sft.write_bytes(b"PT0")
            self.assertEqual(resolve_model_checkpoint(repo, ws, profile)[1], "out_full_sft")

            rung3 = ws / "models" / "candidates" / f"warmstart_rung3_{hs}.pth"
            rung3.write_bytes(b"PT1")
            self.assertEqual(resolve_model_checkpoint(repo, ws, profile)[1], "warmstart_rung3")

            dpo = ws / "models" / "candidates" / f"offline_dpo_r1_{hs}.pth"
            dpo.write_bytes(b"PT2")
            self.assertEqual(resolve_model_checkpoint(repo, ws, profile)[1], "offline_dpo_r1")

    def test_calibrated_extra_eval_promotes_and_blocks_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            kdir = self._copy_valid_kernel(tmp_path)
            manifest = kdir / KERNEL_MANIFEST_FILENAME
            repo = tmp_path / "repo"
            repo.mkdir()
            _init_repo_with_mutable(repo, self.live_mutable_dir)
            ws = tmp_path / "ws"
            ws.mkdir()

            eval_scores = [0.40, 0.55, 0.20]
            eval_calls = {"i": 0}

            def _extra_eval(_wt: Path, _wpath: Path | None) -> float:
                idx = min(eval_calls["i"], len(eval_scores) - 1)
                eval_calls["i"] += 1
                return eval_scores[idx]

            def _proposer(ctx: CycleContext, _handle: WorktreeHandle, executor: WorktreeToolExecutor) -> dict:
                temp = 0.12 if ctx.cycle_index == 1 else 0.90
                executor.dispatch("get_system_state", {})
                executor.dispatch("patch_config", {"updates": {"temperature": temp}})
                executor.dispatch("submit_candidate", {"hypothesis": f"calibrated cycle {ctx.cycle_index}"})
                return {
                    "task_id": f"cal_{ctx.cycle_index}",
                    "task_family": "calibrated",
                    "observed_before_edit": True,
                    "first_edit_valid": True,
                    "tool_calls": [{"name": "patch_config", "arguments": {"updates": {"temperature": temp}}}],
                    "turn_outputs": ["<think>Test calibrated eval</think>"],
                    "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
                    "final_text": f"cal cycle {ctx.cycle_index}",
                }

            res = launch_colab_session(
                repo_root=repo,
                workspace_dir=ws,
                profile_name="local_intel_cpu",
                max_cycles=2,
                autonomy_level=1,
                train_every_n_cycles=0,
                dry_run=False,
                kernel_dir=kdir,
                manifest_path=manifest,
                proposer_fn=_proposer,
                extra_eval_fn=_extra_eval,
            )
            archive = [json.loads(ln) for ln in (ws / "archive.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(archive), 3)
            self.assertLess(archive[0]["eval_score"], archive[1]["eval_score"])
            self.assertLess(archive[1]["eval_score"], 0.9999)
            self.assertEqual(archive[2]["status"], "rejected")
            self.assertEqual(res["champion"]["hypothesis"], "calibrated cycle 1")

    def test_drive_restore_rejects_corrupted_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            ws = tmp_path / "ws"
            drive_dir = tmp_path / "drive"
            ws.mkdir()
            (ws / "archive.jsonl").write_text('{"node_id": "n0"}\n', encoding="utf-8")
            sync = ColabDriveSync(ws, drive_dir)
            sync.sync_to_drive()

            # Tamper with Drive archive.jsonl
            (drive_dir / "archive.jsonl").write_text('{"node_id": "tampered"}\n', encoding="utf-8")
            ws_target = tmp_path / "ws_target"
            res = ColabDriveSync(ws_target, drive_dir).restore_from_drive(verify_checksums=True)
            self.assertFalse(res["restored"])
            self.assertEqual(res["reason"], "checksum_mismatch")
            self.assertIn("archive.jsonl", res["mismatched"])
            self.assertFalse((ws_target / "archive.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
