from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from selfrewrite.cli import main as cli_main
from selfrewrite.colab.drive_sync import DriveMirrorSync
from selfrewrite.colab.run_colab_cli import run_colab_cli_pipeline
from selfrewrite.harvest.audit import audit_distill_bundle


class TestHarvestAuditAndColab(unittest.TestCase):
    def test_full_smoke_harvest_bundle_audit_and_leak_detection(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td) / "ws"
            rc = cli_main(["smoke", "--work-root", str(ws), "--seed", "2026"])
            self.assertEqual(rc, 0)

            bundle_dir = ws / "distill_bundle" / "v1"
            audit = json.loads((bundle_dir / "quality_audit.json").read_text(encoding="utf-8"))
            manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(audit["passed"], audit)
            self.assertGreater(manifest["counts"]["sft_gold_trajectories"], 0)
            self.assertGreater(manifest["counts"]["sft_turn_primitives"], 0)
            self.assertGreater(manifest["counts"]["dpo_contrastive_pairs"], 0)
            self.assertGreater(manifest["counts"]["rlvr_episodes"], 0)

            # Batch 2 harvest_* sessions must NEVER run heldout_eval; Batch 1 reproduce_* MUST run it.
            self.assertFalse((ws / "raw_sessions" / "harvest_improve_00" / "eval_result.json").exists())
            self.assertFalse((ws / "raw_sessions" / "harvest_repair_00" / "eval_result.json").exists())
            self.assertTrue((ws / "raw_sessions" / "reproduce_repair_eval" / "eval_result.json").is_file())

            # Verify no eval_delta key exists and reserved_delta + work_snapshot_dir exist
            first_traj = json.loads((bundle_dir / "sft_gold_trajectories.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertNotIn("eval_delta", first_traj)
            self.assertIn("reserved_delta", first_traj)
            first_rlvr = json.loads((bundle_dir / "rlvr_episodes.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(first_rlvr.get("work_snapshot_dir"))

            # Verify DriveMirrorSync round-trip restore
            sync = DriveMirrorSync(ws, ws / "mock_drive_mirror")
            self.assertTrue((ws / "mock_drive_mirror" / "drive_manifest.json").is_file())
            # Delete local bundle, restore from Drive mirror, confirm exact recovery
            (bundle_dir / "manifest.json").unlink()
            rest = sync.restore_from_drive()
            self.assertTrue(rest["restored"])
            self.assertTrue((bundle_dir / "manifest.json").is_file())

            # Hard negative test 1: inject a held-out eval problem into sft_gold_trajectories -> audit MUST FAIL
            held_row = json.loads((ws / "splits" / "heldout_eval.jsonl").read_text(encoding="utf-8").splitlines()[0])
            sft_path = bundle_dir / "sft_gold_trajectories.jsonl"
            orig_sft = sft_path.read_text(encoding="utf-8")
            leaked_row = {
                "session_id": "leak_test",
                "task": "improve",
                "messages": [{"role": "user", "content": held_row["question"]}, {"role": "assistant", "content": "<think>ok</think>ok"}],
            }
            sft_path.write_text(orig_sft + json.dumps(leaked_row) + "\n", encoding="utf-8")
            rc_fail = cli_main(
                [
                    "audit-bundle",
                    "--bundle-dir",
                    str(bundle_dir),
                    "--splits-dir",
                    str(ws / "splits"),
                    "--num-layers",
                    "8",
                ]
            )
            self.assertNotEqual(rc_fail, 0)
            # Restore clean SFT file and test held-out layer leakage failure (layer 3 is in eval_layers of 8-layer mock)
            sft_path.write_text(orig_sft, encoding="utf-8")
            rlvr_path = bundle_dir / "rlvr_episodes.jsonl"
            orig_rlvr = rlvr_path.read_text(encoding="utf-8")
            rlvr_path.write_text(
                orig_rlvr + json.dumps({"session_id": "leak_layer", "task": "repair", "corruption_layer": 3, "target_tensors": ["model.layers.3.mlp.down_proj.weight"]}) + "\n",
                encoding="utf-8",
            )
            audit_layer = audit_distill_bundle(bundle_dir, forbidden_eval_texts=[], forbidden_eval_layers=[3])
            self.assertFalse(audit_layer["passed"])
            self.assertEqual(audit_layer["violations"][0]["rule"], "eval_layer_leakage")
            rlvr_path.write_text(orig_rlvr, encoding="utf-8")

            # Hard negative test 3: heldout selection field leakage rejected
            rlvr_path.write_text(
                orig_rlvr + json.dumps({"session_id": "sel_leak", "task": "improve", "eval_delta": 0.25}) + "\n",
                encoding="utf-8",
            )
            audit_sel = audit_distill_bundle(bundle_dir, forbidden_eval_texts=[], forbidden_eval_layers=[])
            self.assertFalse(audit_sel["passed"])
            self.assertEqual(audit_sel["violations"][0]["rule"], "heldout_selection_field_leakage")
            rlvr_path.write_text(orig_rlvr, encoding="utf-8")

            # Hard negative test 4: fabricated delta_hat citation caught by numerical_grounding_mismatch
            bad_cite_row = {
                "session_id": "harvest_repair_00",
                "task": "repair",
                "messages": [
                    {"role": "user", "content": "Check results."},
                    {"role": "assistant", "content": "<think>Fake delta_hat=0.9876 looks huge.</think>\nSUBMIT"},
                ],
            }
            sft_path.write_text(orig_sft + json.dumps(bad_cite_row) + "\n", encoding="utf-8")
            audit_num = audit_distill_bundle(
                bundle_dir,
                forbidden_eval_texts=[],
                forbidden_eval_layers=[],
                raw_sessions_dir=ws / "raw_sessions",
            )
            self.assertFalse(audit_num["passed"])
            self.assertTrue(any(v["rule"] == "numerical_grounding_mismatch" for v in audit_num["violations"]))
            sft_path.write_text(orig_sft, encoding="utf-8")

    def test_compact_primitive_prefix_and_detached_colab_dry_run(self) -> None:
        from selfrewrite.harvest.build_bundle import _approx_tokens, _compact_primitive_prefix

        long_prefix = [{"role": "system", "content": "System prompt instructions."}]
        long_prefix.append({"role": "user", "content": "Begin session."})
        for t in range(40):
            long_prefix.append({"role": "assistant", "content": f"Thinking {t} " + ("x" * 600)})
            long_prefix.append({"role": "user", "content": f"Obs {t} " + ("y" * 600)})
        self.assertGreater(_approx_tokens(long_prefix), 15000)

        compacted = _compact_primitive_prefix(
            prefix=long_prefix,
            notes_text="Layer 12 outlier observed.",
            pass_log_so_far=[{"pass_id": 1, "delta_hat": 0.2}],
            active_edit_names=["model.layers.12.mlp.down_proj.weight"],
            max_approx_tokens=3200,
        )
        self.assertLessEqual(_approx_tokens(compacted), 3200)
        self.assertEqual(compacted[0]["role"], "system")
        self.assertEqual(compacted[1]["role"], "user")
        self.assertEqual(compacted[-1]["role"], "user")

        with tempfile.TemporaryDirectory() as td:
            res = run_colab_cli_pipeline(
                workspace_dir=Path(td) / "ws",
                gpu="A100",
                high_mem=True,
                overlay_local_code=True,
                repo_root=Path("."),
                fast_harvest_profile=True,
                detached=True,
                dry_run=True,
            )
            self.assertTrue(res["dry_run"])
            self.assertTrue(res["detached"])
            self.assertIn("step2_detached_launch.py", res["scripts"])
            self.assertIn("step2_detached_poll.py", res["scripts"])
            cmds_flat = " ".join(" ".join(c) for c in res["planned_commands"])
            self.assertIn("step2_detached_launch.py", cmds_flat)
            self.assertIn("step2_detached_poll.py", cmds_flat)

    def test_colab_cli_a100_dry_run_plan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            res = run_colab_cli_pipeline(
                workspace_dir=Path(td) / "ws",
                gpu="A100",
                high_mem=True,
                overlay_local_code=True,
                repo_root=Path("."),
                dry_run=True,
            )
            self.assertTrue(res["dry_run"])
            cmds = res["planned_commands"]
            self.assertEqual(cmds[0][:5], ["colab", "new", "-s", "minimind-selfrewrite-a100", "--gpu"])
            self.assertIn("A100", cmds[0])
            self.assertIn("--high-mem", cmds[0])


if __name__ == "__main__":
    unittest.main()
