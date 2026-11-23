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
