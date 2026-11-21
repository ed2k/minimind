from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from selfchange.bootstrap.data_quality import audit_training_dataset
from selfchange.bootstrap.limitation_to_fix import (
    HELDOUT_10_PROMPTS,
    build_limitation_fix_and_meta_dataset,
    diagnose_model_limitations,
    record_verified_selfchange_episode,
)


class TestLimitationToFixPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.sample_before = {
            "passed_count": 2,
            "total_count": 10,
            "eos_rate": 0.6,
            "examples": [
                {"id": "1_trivia_capital", "category": "trivia", "passed": False, "eos_ok": True, "reply": "<think>\n</think>\nParis longest river."},
                {"id": "2_tech_zh_grad_accum", "category": "core_tech_qa", "passed": False, "eos_ok": False, "reply": "<think>\n</think>\nGradient Descent"},
                {"id": "3_tech_en_grad_clip", "category": "core_tech_qa", "passed": False, "eos_ok": False, "reply": "<think>\nGrad Norm " * 10},
                {"id": "4_reasoning_oom_math", "category": "reasoning_math", "passed": False, "eos_ok": False, "reply": "<think>\n</think>\n16*2=32"},
                {"id": "5_tool_oom", "category": "tool_call_op", "passed": True, "eos_ok": True, "tool_calls": [{"name": "patch_config", "arguments": {"log_name": "bad"}}], "reply": "<think>\n</think>"},
                {"id": "6_tool_nan_patch", "category": "tool_call_op", "passed": False, "eos_ok": True, "tool_calls": [{"name": "patch_config", "arguments": {"log_name": "bad2"}}], "reply": "<think>\n</think>"},
                {"id": "8_code_missing_colon", "category": "code_edit", "passed": False, "eos_ok": True, "tool_calls": [{"name": "replace_line_range", "arguments": {"path": "db.txt"}}], "reply": "<think>bad</think>"},
                {"id": "9_code_clamp_logic", "category": "code_edit", "passed": False, "eos_ok": False, "tool_calls": [], "reply": "<think>plan</think>```python\ndef l"},
                {"id": "10_healthy_noop", "category": "noop", "passed": True, "eos_ok": True, "tool_calls": [{"name": "get_system_state", "arguments": {}}], "reply": "<think>ok</think>"},
            ],
        }

    def test_diagnose_limitations_extracts_all_failure_classes(self) -> None:
        lims = diagnose_model_limitations(self.sample_before)
        ids = {x["limitation_id"] for x in lims}
        self.assertIn("empty_think_and_math_reasoning", ids)
        self.assertIn("repetition_loop_and_eos_cutoff", ids)
        self.assertIn("cross_tool_schema_contamination", ids)
        self.assertIn("code_edit_missing_args_and_markdown_fallback", ids)
        self.assertIn("core_tech_misconception_and_trivia_hallucination", ids)
        self.assertIn("healthy_state_parsing_and_selfchange_grounding", ids)

    def test_build_limitation_fix_and_meta_dataset_passes_all_gates_without_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            eval_json = tmp_p / "base_eval.json"
            eval_json.write_text(json.dumps(self.sample_before), encoding="utf-8")
            out_dir = tmp_p / "fix_data"
            rep = build_limitation_fix_and_meta_dataset(
                repo_root=self.repo_root,
                output_dir=out_dir,
                eval_report_path=eval_json,
                sft_count=120,
                dpo_count=40,
                max_seq_len=768,
                seed=42,
            )
            self.assertTrue(rep["overall_passed"], msg=json.dumps(rep, indent=2))
            self.assertGreater(rep["split_breakdown"]["schema_disambiguation"], 0)
            self.assertGreater(rep["split_breakdown"]["meta_selfchange"], 0)
            self.assertGreater(rep["split_breakdown"]["code_repair"], 0)

            sft_m = rep["sft_audit"]["metrics"]
            self.assertEqual(sft_m["leakage_count"], 0)
            self.assertEqual(sft_m["off_domain_trivia_rate"], 0.0)
            self.assertEqual(sft_m["exec_verified_pass_rate"], 1.0)
            self.assertGreaterEqual(sft_m["think_grounding_rate"], 0.90)
            self.assertGreaterEqual(sft_m["think_action_consistency_rate"], 0.90)

            dpo_m = rep["dpo_audit"]["metrics"]
            self.assertTrue(rep["dpo_audit"]["passed"])
            self.assertEqual(dpo_m["schema_valid_rate"], 1.0)

            # Confirm none of the 10 held-out prompt strings appear verbatim in sft_train.jsonl
            sft_raw = Path(rep["sft_path"]).read_text(encoding="utf-8")
            for item in HELDOUT_10_PROMPTS:
                usr = item["messages"][-1]["content"]
                self.assertNotIn(usr, sft_raw, msg=f"Heldout leakage for {item['id']}")

    def test_record_verified_selfchange_episode_passes_audit(self) -> None:
        sample_after = {"passed_count": 8, "total_count": 10, "eos_rate": 1.0, "examples": []}
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            sft_p = tmp_p / "episodes.jsonl"
            dpo_p = tmp_p / "episodes_dpo.jsonl"
            res = record_verified_selfchange_episode(
                before_eval=self.sample_before,
                after_eval=sample_after,
                fix_summary={"steps_taken": ["diagnose", "curate", "full_sft", "dpo"], "learning_rate": 3e-5, "sft_epochs": 3, "dpo_epochs": 1},
                output_sft_jsonl=sft_p,
                output_dpo_jsonl=dpo_p,
            )
            self.assertTrue(res["improved"])
            self.assertGreaterEqual(res["recorded_sft_total"], 5)
            self.assertGreaterEqual(res["recorded_dpo_total"], 5)

            audit_sft = audit_training_dataset(
                sft_p,
                dataset_format="sft",
                max_seq_len=768,
                thresholds={"min_normalized_unique_rate": 0.05},
            )
            self.assertTrue(audit_sft["passed"], msg=audit_sft)


if __name__ == "__main__":
    unittest.main()
