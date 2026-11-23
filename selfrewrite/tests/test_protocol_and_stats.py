from __future__ import annotations

import unittest

from selfrewrite.config import SessionConfig
from selfrewrite.protocol import (
    build_compaction_messages,
    build_task_prompt,
    parse_editor_reply,
    parse_submit_decision,
    strip_thinking,
)
from selfrewrite.stats import (
    compute_paired_stats,
    exact_one_sided_sign_test,
    paired_permutation_test,
    score_emergence_ladder,
)


class TestProtocolAndStats(unittest.TestCase):
    def test_single_block_parser_ignores_blocks_inside_think(self) -> None:
        cfg = SessionConfig(protocol=2)
        reply = (
            "<think>\nLet's draft code:\n```python\nx = 1\n```\nNow output actual block.\n</think>\n"
            "I will inspect the census.\n"
            "```python\nprint('real')\n```"
        )
        think, vis = strip_thinking(reply)
        self.assertIn("x = 1", think)
        self.assertNotIn("x = 1", vis)
        act = parse_editor_reply(reply, cfg)
        self.assertEqual(act.kind, "python")
        self.assertEqual(act.payload, "print('real')")

        # Two visible blocks rejected without consuming budget
        two_blocks = (
            "<think>ok</think>\n```python\na=1\n```\n```forward\n{\"mode\":\"generate\"}\n```"
        )
        act2 = parse_editor_reply(two_blocks, cfg)
        self.assertEqual(act2.kind, "invalid")
        self.assertIn("exactly one block", act2.error or "")

        # Zero visible blocks -> final reply
        fin = parse_editor_reply("<think>done</think>\nSubmitting my final edit.", cfg)
        self.assertEqual(fin.kind, "final")

    def test_submit_withdraw_and_compaction(self) -> None:
        self.assertEqual(parse_submit_decision("<think>good</think>\nSUBMIT"), "SUBMIT")
        self.assertEqual(parse_submit_decision("<think>bad</think>\nWITHDRAW"), "WITHDRAW")
        self.assertEqual(parse_submit_decision("<think>unclear</think>\nSUBMIT or WITHDRAW"), "INVALID")

        cfg = SessionConfig(task="repair", protocol=2)
        prompt = build_task_prompt(cfg, {"n": 64, "accuracy": 0.0, "cut_rate": 0.3, "loop_rate_among_cuts": 1.0, "finished_accuracy": 0.0}, 100, 64)
        self.assertIn("Damaged copy notice", prompt)
        msgs = build_compaction_messages(prompt, "layer 30 outlier", 4, 20, 10, 200, [{"name": "a"}], [], "obs")
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("layer 30 outlier", msgs[1]["content"])

    def test_paired_stats_sign_and_permutation_tests_and_ladder(self) -> None:
        base_rows = [
            {"problem_id": f"p{i}", "sample_idx": 0, "correct": i < 2, "cut_at_limit": i >= 10, "looping": False, "num_tokens": 200}
            for i in range(20)
        ]
        edit_rows = [
            {"problem_id": f"p{i}", "sample_idx": 0, "correct": i < 15, "cut_at_limit": False, "looping": False, "num_tokens": 120}
            for i in range(20)
        ]
        stats = compute_paired_stats(base_rows, edit_rows, num_flips=5000, perm_seed=0)
        self.assertEqual(stats["n_up"], 13)
        self.assertEqual(stats["n_down"], 0)
        self.assertAlmostEqual(stats["delta_hat"], 0.65, places=4)
        self.assertLess(exact_one_sided_sign_test(13, 0), 1e-3)
        self.assertLess(paired_permutation_test(stats["per_problem_diffs"], num_flips=5000, seed=1), 1e-3)

        ladder_ok = score_emergence_ladder(
            {
                "code_runs_used": 4,
                "code_errors": 0,
                "forward_passes_used": 2,
                "paired_tests_count": 1,
                "read_passes_count": 1,
                "decision": "SUBMIT",
                "submitted_tensors": ["w"],
                "reserved_check": {"delta_hat": 0.25},
                "paired_test_history": [{"delta_hat": 0.30}],
            },
            heldout_eval={"delta_hat": 0.30, "perm_p": 0.002, "sign_p": 0.004},
        )
        self.assertEqual(ladder_ok["ladder_score"], 6)


if __name__ == "__main__":
    unittest.main()
