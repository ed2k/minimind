from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from selfrewrite.backends.mock_backend import MockTransformerModel
from selfrewrite.helpers import outlier_ratios
from selfrewrite.tasks.corruption import create_damaged_workspace, split_train_eval_layers
from selfrewrite.tasks.datasets import build_builtin_suites, deduplicate_splits, normalize_problem_text
from selfrewrite.tasks.grader import extract_last_boxed, grade_response, is_looping_answer


class TestGraderAndDamage(unittest.TestCase):
    def test_boxed_nested_and_thinking_and_loop_grader(self) -> None:
        resp_ok = "<think>\nFirst 11/125 = 88/1000.\n</think>\nAnswer: \\boxed{\\frac{11}{125}}"
        self.assertEqual(extract_last_boxed(resp_ok), "\\frac{11}{125}")
        g = grade_response({"answer": "0.088", "problem_type": "math"}, resp_ok, num_tokens=50, token_budget=512)
        self.assertTrue(g["correct"])
        self.assertFalse(g["looping"])

        # Open <think> earns zero credit even if boxed is present inside thinking
        resp_open = "<think>\nWait \\boxed{42}"
        g_open = grade_response({"answer": "42", "problem_type": "math"}, resp_open, num_tokens=512, token_budget=512)
        self.assertFalse(g_open["correct"])
        self.assertTrue(g_open["cut_at_limit"])

        # Zlib compression detects repetitive loops
        loop_txt = "<think>\n" + ("I will write the response. " * 400)
        self.assertTrue(is_looping_answer(loop_txt, cut_at_limit=True))

    def test_dataset_splits_zero_overlap_and_layer_isolation(self) -> None:
        suites = build_builtin_suites(seed=123, practice_n=20, reserved_n=12, heldout_n=20, distill_n=20, transfer_n=10)
        seen_by_split = {k: {normalize_problem_text(p.question) for p in plist} for k, plist in suites.items()}
        self.assertEqual(len(seen_by_split["heldout_eval"] & seen_by_split["distill_train"]), 0)
        self.assertEqual(len(seen_by_split["heldout_eval"] & seen_by_split["practice_eval"]), 0)
        self.assertEqual(len(seen_by_split["reserved_eval"] & seen_by_split["distill_train"]), 0)

        # Explicitly inject duplicate from heldout into distill_train and verify deduplicate_splits strips it
        dup_train = list(suites["distill_train"]) + [suites["heldout_eval"][0]]
        cleaned = deduplicate_splits({**suites, "distill_train": dup_train})
        self.assertEqual(len(cleaned["distill_train"]), len(suites["distill_train"]))

        # Verify layer 30 of 64 sits strictly in eval_layers (excluded from train_layers)
        train_l, eval_l = split_train_eval_layers(64)
        self.assertIn(30, eval_l)
        self.assertNotIn(30, train_l)
        self.assertEqual(set(train_l) & set(eval_l), set())

    def test_corruption_isolation_and_outlier_ratio(self) -> None:
        model = MockTransformerModel(num_layers=8, hidden_dim=12, inter_dim=16)
        with tempfile.TemporaryDirectory() as td:
            hidden_dir = Path(td) / "hidden"
            dmg_w, spec = create_damaged_workspace(
                model.raw_tensors,
                hidden_dir=hidden_dir,
                layer_idx=4,
                tensor_name="model.layers.4.mlp.down_proj.weight",
                num_rows=4,
                sigma_mult=200.0,
                seed=11,
            )
            manifest = json.loads((hidden_dir / "damage_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["layer_idx"], 4)
            ratios = outlier_ratios({k: v for k, v in dmg_w.items() if k.endswith(".down_proj.weight")})
            ranked = sorted(ratios.items(), key=lambda kv: kv[1]["ratio_max_to_median"], reverse=True)
            self.assertEqual(ranked[0][0], "model.layers.4.mlp.down_proj.weight")
            self.assertGreater(ranked[0][1]["ratio_max_to_median"], 100.0)


if __name__ == "__main__":
    unittest.main()
