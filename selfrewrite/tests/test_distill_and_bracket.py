from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from selfrewrite.cli import main as cli_main
from selfrewrite.colab.downsize_bracket import PromotionGateConfig, evaluate_promotion_gates, run_downsize_bracket
from selfrewrite.distill.compress_model import estimate_compressed_params, plan_compression
from selfrewrite.distill.train_student import (
    StudentTrainConfig,
    build_dpo_dataset,
    build_sft_curriculum,
    compute_rlvr_turn_reward,
    train_downsize_student,
    verify_bundle_audit_gate,
)
from selfrewrite.tasks.datasets import build_builtin_suites, normalize_problem_text


class TestDistillAndBracket(unittest.TestCase):
    def test_audit_gate_curriculum_rlvr_and_compression_plan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ws = root / "ws"
            self.assertEqual(cli_main(["smoke", "--work-root", str(ws), "--seed", "2026"]), 0)
            bundle_dir = ws / "distill_bundle" / "v1"

            # Verify valid bundle passes gate
            gate_info = verify_bundle_audit_gate(bundle_dir)
            self.assertTrue(gate_info["audit"]["passed"])

            # Tamper quality_audit.json -> gate MUST raise ValueError
            audit_file = bundle_dir / "quality_audit.json"
            orig_audit = audit_file.read_text(encoding="utf-8")
            audit_file.write_text(json.dumps({"passed": False, "num_violations": 1, "violations": [{"rule": "leak"}]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_bundle_audit_gate(bundle_dir)
            audit_file.write_text(orig_audit, encoding="utf-8")

            # Curriculum respects ratios and contains zero heldout/reserved problems
            suites = build_builtin_suites(seed=2026)
            cfg = StudentTrainConfig(
                rung_id="R2",
                bundle_dir=bundle_dir,
                output_dir=ws / "student_r2",
                splits_dir=ws / "splits",
                backend="mock",
            )
            sft = build_sft_curriculum(bundle_dir, suites["distill_train"], cfg)
            dpo = build_dpo_dataset(bundle_dir, output_dir=cfg.output_dir)
            self.assertGreater(len(sft), 0)
            self.assertGreater(len(dpo), 0)
            sources = {r["source"] for r in sft}
            self.assertIn("primitive", sources)
            self.assertIn("gold_trajectory", sources)
            self.assertIn("distill_anchor", sources)

            eval_norms = {normalize_problem_text(p.question) for p in [*suites["heldout_eval"], *suites["reserved_eval"]]}
            for row in sft:
                blob = normalize_problem_text(json.dumps(row))
                for en in eval_norms:
                    self.assertNotIn(en, blob)

            res = train_downsize_student(cfg)
            self.assertTrue(res["frozen_for_eval"])
            self.assertTrue((cfg.output_dir / "student_provenance.json").is_file())

        # RLVR reward shaping check
        good_rew = compute_rlvr_turn_reward(
            reply="<think>Inspecting state diffs</think>\n```python\nr = outlier_ratios(load_original('x'))\n```",
            action_kind="python",
            sandbox_result={"ok": True},
            paired_stats={"delta_hat": 0.4},
            decision="SUBMIT",
            reserved_stats={"delta_hat": 0.35},
        )
        bad_rew = compute_rlvr_turn_reward(
            reply="<think>Adding noise</think>\n```python\nw += rng.gauss(0, 0.5)\n```",
            action_kind="python",
            sandbox_result={"ok": True},
            paired_stats={"delta_hat": -0.3},
            decision="SUBMIT",
            reserved_stats={"delta_hat": -0.3},
        )
        self.assertGreater(good_rew["total_reward"], 1.5)
        self.assertLess(bad_rew["total_reward"], 0.0)

        # Compression math check
        est = estimate_compressed_params(hidden_size=2560, num_layers=36, intermediate_size=6912, vocab_size=151936)
        self.assertGreater(est["total_params"], 1e9)
        plan = plan_compression(
            {"hidden_size": 2560, "num_layers": 36, "intermediate_size": 6912, "vocab_size": 151936},
            target_mlp_ratio=0.75,
            keep_layer_ratio=0.75,
        )
        self.assertGreater(plan["param_reduction_pct"], 15.0)
        self.assertEqual(plan["keep_layer_indices"][0], 0)
        self.assertEqual(plan["keep_layer_indices"][-1], 35)

    def test_downsize_bracket_promotes_r1_r2_r3_and_rejects_r4_floor(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = Path(td) / "ws"
            self.assertEqual(cli_main(["smoke", "--work-root", str(ws), "--seed", "2026"]), 0)
            drive_dir = ws / "mock_drive_mirror"

            summary = run_downsize_bracket(
                workspace_dir=ws,
                bundle_dir=ws / "distill_bundle" / "v1",
                rungs=("R1", "R2", "R3", "R4"),
                backend_kind="mock",
                drive_dir=drive_dir,
                improve_sessions=2,
                bootstrap_next_rung=True,
                seed=2026,
            )
            by_rung = {r["rung_id"]: r for r in summary["rungs"]}
            for r_id in ("R1", "R2", "R3"):
                self.assertTrue(by_rung[r_id]["promotion"]["promoted"], by_rung[r_id])
                self.assertEqual(by_rung[r_id]["promotion"]["best_ladder_score"], 6)
            self.assertFalse(by_rung["R4"]["promotion"]["promoted"])
            self.assertEqual(summary["pareto_winner"]["rung_id"], "R3")
            self.assertTrue((ws / "downsize_leaderboard.json").is_file())
            self.assertTrue((ws / "downsize_report.md").is_file())
            self.assertTrue((drive_dir / "downsize_report.md").is_file())

            # Explicit promotion gate rejection check (catastrophic forgetting rejected)
            bad_gate = evaluate_promotion_gates(
                {**by_rung["R2"], "student_unedited_acc": 0.0},
                gates=PromotionGateConfig(),
            )
            self.assertFalse(bad_gate["promoted"])
            self.assertFalse(bad_gate["gates"]["retention_gate"])

            # CLI smoke check for compress, train-student, and bracket
            self.assertEqual(
                cli_main(["compress", "--hidden-size", "2560", "--num-layers", "36", "--intermediate-size", "6912", "--mlp-ratio", "0.75"]),
                0,
            )
            self.assertEqual(
                cli_main(
                    [
                        "train-student",
                        "--rung-id", "R2",
                        "--bundle-dir", str(ws / "distill_bundle" / "v1"),
                        "--output-dir", str(ws / "cli_student_r2"),
                        "--splits-dir", str(ws / "splits"),
                        "--backend", "mock",
                    ]
                ),
                0,
            )


if __name__ == "__main__":
    unittest.main()
