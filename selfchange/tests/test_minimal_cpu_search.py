from __future__ import annotations

import json
import py_compile
import tempfile
import unittest
from pathlib import Path

from selfchange.bootstrap.data_quality import (
    audit_training_dataset,
    build_finetune_dataset,
    compute_eval_steered_weights,
    review_and_record_dataset,
    score_thinking_turn,
)
from selfchange.bootstrap.eval_baseline import evaluate_baseline
from selfchange.bootstrap.prune_reallocate import (
    classify_trivia_vs_domain,
    filter_trivia_from_anchor_jsonl,
    rank_mlp_channels,
)
from selfchange.bootstrap.search_minimal_cpu import (
    MINIMAL_CPU_LADDER,
    check_minimal_cpu_floor,
    run_minimal_cpu_search,
    select_next_bracket_rung,
    select_pareto_champion,
)
from selfchange.colab.run_colab_cli import build_minimal_cpu_search_script
from selfchange.kernel.budgets import get_profile
from selfchange.training.model_loader import estimate_minimind_params, load_model_config, save_model_config
from selfchange.training.self_train_runner import build_trainer_command, run_training_stage, write_sft_jsonl


class TestMinimalCpuAndContinuousReview(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]

    def test_param_estimates_ladder_and_model_config_io(self) -> None:
        expected_millions = {
            "L0_teacher_64m": 63.91,
            "L1_pruned_42m": 41.50,
            "L2_compact_30m": 30.03,
            "L3_slim_mlp_22m": 22.16,
            "L4_deep_narrow_17m": 17.21,
            "L5_deep_slim_13m": 13.08,
            "L6_tiny_deep_10m": 10.43,
            "L7_ultra_tiny_7m": 7.15,
        }
        self.assertEqual(len(MINIMAL_CPU_LADDER), 8)
        for item in MINIMAL_CPU_LADDER:
            self.assertAlmostEqual(item["total_params"] / 1e6, expected_millions[item["rung_id"]], places=1)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "cand_384.pth"
            p.write_bytes(b"pt")
            cfg = estimate_minimind_params(hidden_size=384, num_hidden_layers=8, intermediate_size=768, num_attention_heads=6, num_key_value_heads=3)
            save_model_config(p.with_suffix(".config.json"), cfg)
            loaded = load_model_config(p)
            self.assertEqual(loaded["intermediate_size"], 768)
            self.assertEqual(loaded["num_key_value_heads"], 3)

    def test_trivia_filter_and_mlp_channel_ranking(self) -> None:
        ranked = rank_mlp_channels(
            domain_acts=[0.1, 2.0, 0.05, 1.5],
            trivia_acts=[3.0, 0.1, 0.2, 0.5],
            down_norms=[1.0, 1.0, 1.0, 1.0],
        )
        self.assertEqual(ranked[0], 1)
        self.assertEqual(ranked[-1], 0)

        self.assertTrue(classify_trivia_vs_domain("What is the capital of France?")["is_trivia"])
        self.assertFalse(classify_trivia_vs_domain("Why check syntax with ast.parse before editing python code?")["is_trivia"])

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            src = p / "raw_anchors.jsonl"
            clean = p / "clean_anchors.jsonl"
            quar = p / "quarantine_anchors.jsonl"
            write_sft_jsonl(
                src,
                [
                    {"conversations": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "Paris."}]},
                    {"conversations": [{"role": "user", "content": "今天晚饭想做红烧肉，需要准备哪些菜谱食材？"}, {"role": "assistant", "content": "五花肉等。"}]},
                    {"conversations": [{"role": "user", "content": "Why use clip_grad_norm in PyTorch training?"}, {"role": "assistant", "content": "To bound gradients."}]},
                    {
                        "conversations": [
                            {"role": "system", "content": "Tools", "tools": '[{"type":"function","function":{"name":"read_log_tail"}}]'},
                            {"role": "user", "content": "Check capital of France in train.log"},
                            {"role": "assistant", "content": '<tool_call>{"name":"read_log_tail","arguments":{"log_name":"train.log"}}</tool_call>'},
                        ]
                    },
                ],
            )
            rep = filter_trivia_from_anchor_jsonl(src, clean, quar)
            self.assertEqual(rep["kept"], 2)
            self.assertEqual(rep["quarantined_trivia"], 2)
            self.assertTrue(clean.is_file() and quar.is_file())

    def test_think_grounding_trivia_gate_and_continuous_review_ops(self) -> None:
        good_ctx = "clip_grad.py:\n1: def clip_value(val: float):\n2:     return max(val, 1.0)"
        good_turn = (
            "<think>\nObserve: clip_grad.py line 2 uses max. Cause: upper bound uses min. "
            "Plan: call replace_line_range line 2.\n</think>\n\n"
            '<tool_call>{"name":"replace_line_range","arguments":{"path":"clip_grad.py","start_line":2,"end_line":2,"lines":["    return min(val, 1.0)"]}}</tool_call>'
        )
        g_res = score_thinking_turn(good_ctx, good_turn)
        self.assertTrue(g_res["grounded"])
        self.assertTrue(g_res["action_consistent"])

        bad_turn = (
            "<think>\nObserve: guess line 99 looks wrong. Plan: call read_log_tail.\n</think>\n\n"
            '<tool_call>{"name":"replace_line_range","arguments":{"path":"clip_grad.py","start_line":2,"end_line":2,"lines":["    return min(val, 1.0)"]}}</tool_call>'
        )
        b_res = score_thinking_turn(good_ctx, bad_turn)
        self.assertFalse(b_res["grounded"])
        self.assertFalse(b_res["action_consistent"])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            ws = tmp_p / "ws"
            data_dir = tmp_p / "data"
            gen = build_finetune_dataset(self.repo_root, data_dir, sft_count=40, dpo_count=12, heldout_count=10, seed=42)
            self.assertTrue(gen["overall_passed"], msg=gen)
            sft_metrics = gen["sft_audit"]["metrics"]
            self.assertEqual(sft_metrics["off_domain_trivia_rate"], 0.0)
            self.assertGreaterEqual(sft_metrics["think_grounding_rate"], 0.95)
            self.assertGreaterEqual(sft_metrics["think_action_consistency_rate"], 0.95)

            rev1 = review_and_record_dataset(Path(gen["sft_path"]), ws, heldout_path=Path(gen["heldout_path"]), cycle_tag="cycle_1")
            self.assertTrue(rev1["passed"])
            self.assertTrue(Path(rev1["review_md_path"]).is_file())
            self.assertTrue(Path(rev1["history_path"]).is_file())

            # Inject a failing trivia + ungrounded batch and verify fail-closed trainer gate + quarantine + drift
            bad_jsonl = tmp_p / "bad_batch.jsonl"
            write_sft_jsonl(
                bad_jsonl,
                [
                    {"conversations": [{"role": "user", "content": "Who won the FIFA World Cup in 1998?"}, {"role": "assistant", "content": "France."}]},
                    {
                        "conversations": [
                            {"role": "user", "content": "fault=oom batch_size=8"},
                            {"role": "assistant", "content": "<think>bad</think><tool_call>{\"name\":\"patch_config\",\"arguments\":{\"updates\":{\"training\":{\"batch_size\":4}}}}</tool_call>"},
                        ]
                    },
                ],
            )
            blocked = run_training_stage(
                stage="lora",
                repo_root=self.repo_root,
                workspace_dir=ws,
                data_path=bad_jsonl,
                profile=get_profile("local_intel_cpu"),
                training_cfg={"enforce_data_gate": True, "epochs": 1},
                dry_run=True,
            )
            self.assertFalse(blocked.ok)
            self.assertEqual(blocked.exit_code, 3)
            q_files = list((ws / "data_quarantine").glob("*_quarantine.jsonl"))
            self.assertGreaterEqual(len(q_files), 1)

    def test_trainer_command_plumbing_and_compile(self) -> None:
        for rel in ("train_pretrain.py", "train_full_sft.py", "train_lora.py", "train_dpo.py", "train_distillation.py"):
            with tempfile.TemporaryDirectory() as tmp:
                py_compile.compile(str(self.repo_root / "trainer" / rel), cfile=str(Path(tmp) / f"{rel}.pyc"), doraise=True)

        prof = get_profile("local_intel_cpu")
        student_cfg = {"hidden_size": 384, "num_hidden_layers": 8, "intermediate_size": 768, "num_attention_heads": 6, "num_key_value_heads": 3, "use_moe": False}
        teacher_cfg = {"hidden_size": 768, "num_hidden_layers": 8, "intermediate_size": 2432, "num_attention_heads": 8, "num_key_value_heads": 4, "use_moe": False}
        cmd, out_p = build_trainer_command(
            stage="distill",
            repo_root=self.repo_root,
            data_path=self.repo_root / "x.jsonl",
            save_dir=self.repo_root / "out",
            save_weight="student_13m",
            from_weight="none",
            profile=prof,
            training_cfg={"model_config": student_cfg, "teacher_config": teacher_cfg},
        )
        cmd_str = " ".join(cmd)
        self.assertIn("train_distillation.py", cmd_str)
        self.assertIn("--student_intermediate_size 768", cmd_str)
        self.assertIn("--teacher_hidden_size 768", cmd_str)
        self.assertTrue(out_p.name.endswith("_384.pth"))

    def test_bracket_search_pareto_and_colab_search_scripts(self) -> None:
        # Path 1: L2 pass -> L5 pass -> L6 pass -> L7
        seen: dict[str, dict] = {}
        r1 = select_next_bracket_rung(seen)
        self.assertEqual(r1["rung_id"], "L2_compact_30m")
        seen["L2_compact_30m"] = {"passed_floor": True}
        r2 = select_next_bracket_rung(seen)
        self.assertEqual(r2["rung_id"], "L5_deep_slim_13m")
        seen["L5_deep_slim_13m"] = {"passed_floor": True}
        r3 = select_next_bracket_rung(seen)
        self.assertEqual(r3["rung_id"], "L6_tiny_deep_10m")
        seen["L6_tiny_deep_10m"] = {"passed_floor": True}
        self.assertEqual(select_next_bracket_rung(seen)["rung_id"], "L7_ultra_tiny_7m")

        # Path 2: L5 fails while L2 passes -> L3 -> L4
        seen2 = {"L2_compact_30m": {"passed_floor": True}, "L5_deep_slim_13m": {"passed_floor": False}}
        self.assertEqual(select_next_bracket_rung(seen2)["rung_id"], "L3_slim_mlp_22m")

        # Path 3: L2 fails -> L1
        self.assertEqual(select_next_bracket_rung({"L2_compact_30m": {"passed_floor": False}})["rung_id"], "L1_pruned_42m")

        scorecard = evaluate_baseline(self.repo_root, backend="scripted", profile_name="local_intel_cpu", num_code_tasks=6)
        self.assertIn("tier3_5_thinking", scorecard)
        ok_fl, failed_fl = check_minimal_cpu_floor(scorecard)
        self.assertTrue(ok_fl, msg=failed_fl)
        steered = compute_eval_steered_weights(scorecard)
        self.assertIn("simple_1line_ast_syntax", steered["retired_families"])

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            rep = run_minimal_cpu_search(
                repo_root=self.repo_root,
                workspace_dir=ws,
                profile_name="local_intel_cpu",
                sft_count=40,
                dpo_count=12,
                heldout_count=10,
                max_rungs=3,
                dry_run=True,
            )
            self.assertIsNotNone(rep["pareto_champion"])
            self.assertEqual(rep["pareto_champion"]["rung_id"], "L6_tiny_deep_10m")
            self.assertTrue((ws / "search_frontier_report.json").is_file())
            self.assertTrue((ws / "search_frontier_report.md").is_file())
            self.assertTrue((ws / "promoted_models" / "latest_champion.json").is_file())
            self.assertTrue((ws / "promoted_models" / "model_config.json").is_file())

        scripts = build_minimal_cpu_search_script(branch="mychanges", rungs="L2_compact_30m,L5_deep_slim_13m", max_rungs=2)
        self.assertIn("step2_search_minimal_cpu.py", scripts)
        self.assertIn("selfchange.bootstrap.search_minimal_cpu", scripts["step2_search_minimal_cpu.py"])


if __name__ == "__main__":
    unittest.main()
