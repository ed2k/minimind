from __future__ import annotations

import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from selfchange.bootstrap.data_quality import (
    audit_training_dataset,
    build_finetune_dataset,
    collect_code_fixtures,
    extract_repo_code_fixtures,
    format_quality_report_md,
    inject_ast_mutation,
    verify_edit_in_worktree,
)
from selfchange.kernel.sandbox import scan_python_source
from selfchange.training.self_train_runner import write_sft_jsonl


class TestDataQualityGeneratorAndAuditor(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]

    def test_extract_repo_code_fixtures_and_ast_mutations(self) -> None:
        repo_fixtures = extract_repo_code_fixtures(self.repo_root)
        all_fixtures = collect_code_fixtures(self.repo_root)
        self.assertGreaterEqual(len(all_fixtures), 20)
        for fx in repo_fixtures:
            self.assertTrue(scan_python_source(fx["source"]).ok, msg=f"Unsafe fixture: {fx['filename']}")

        rng = random.Random(42)
        verified_cnt = 0
        bug_types: set[str] = set()
        with tempfile.TemporaryDirectory() as tmp:
            wt = Path(tmp)
            for fx in all_fixtures[:24]:
                mut = inject_ast_mutation(fx["source"], rng)
                if mut is None:
                    continue
                broken_src, fix = mut
                bug_types.add(fix["bug_type"])
                edit_args = {
                    "path": fx["filename"],
                    "start_line": fix["start_line"],
                    "end_line": fix["end_line"],
                    "lines": fix["lines"],
                }
                if verify_edit_in_worktree(wt, fx["filename"], broken_src, fx["source"], edit_args):
                    verified_cnt += 1
        self.assertGreaterEqual(verified_cnt, 15)
        self.assertGreaterEqual(len(bug_types), 3)

    def test_build_finetune_dataset_passes_all_quality_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "curated"
            res = build_finetune_dataset(
                repo_root=self.repo_root,
                output_dir=out_dir,
                sft_count=80,
                dpo_count=24,
                heldout_count=20,
                max_seq_len=768,
                seed=42,
            )
            self.assertTrue(res["overall_passed"], msg=json.dumps(res, indent=2))
            for fname in ("sft_train.jsonl", "dpo_train.jsonl", "heldout_eval.jsonl", "data_quality_report.json", "data_quality_report.md"):
                self.assertTrue((out_dir / fname).is_file(), msg=f"Missing {fname}")

            sft_rep = res["sft_audit"]
            self.assertEqual(sft_rep["format"], "sft")
            self.assertEqual(sft_rep["metrics"]["schema_valid_rate"], 1.0)
            self.assertEqual(sft_rep["metrics"]["exec_verified_pass_rate"], 1.0)
            self.assertEqual(sft_rep["metrics"]["leakage_count"], 0)
            self.assertGreater(sft_rep["metrics"]["direct_code_edit_rate"], 0.0)
            self.assertGreater(res["split_breakdown"]["multi_code_edit"], 0)
            self.assertLessEqual(sft_rep["metrics"]["noop_rate"], 0.25)
            self.assertIn("read_file_lines", sft_rep["metrics"]["tool_counts"])
            self.assertIn("submit_candidate", sft_rep["metrics"]["tool_counts"])

            dpo_rep = res["dpo_audit"]
            self.assertEqual(dpo_rep["format"], "dpo")
            self.assertTrue(dpo_rep["passed"])
            self.assertEqual(dpo_rep["metrics"]["schema_valid_rate"], 1.0)
            self.assertIn("**Status**: `PASS`", format_quality_report_md(sft_rep))

    def test_audit_detects_all_failure_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)

            # 1) Bad schema (extra top-level key matching Run 1 regression)
            bad_schema = p / "bad_schema.jsonl"
            bad_schema.write_text(
                json.dumps({"conversations": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}], "task_family": "bad"}) + "\n",
                encoding="utf-8",
            )
            r_schema = audit_training_dataset(bad_schema, dataset_format="sft")
            self.assertFalse(r_schema["passed"])
            self.assertEqual(r_schema["metrics"]["schema_valid_rate"], 0.0)

            # 2) Unbalanced tags + hallucinated tool name
            bad_tool = p / "bad_tool.jsonl"
            write_sft_jsonl(
                bad_tool,
                [
                    {
                        "conversations": [
                            {"role": "system", "content": "Sys"},
                            {"role": "user", "content": "fix"},
                            {"role": "assistant", "content": "<think>oops\n<tool_call>{\"name\": \"hallucinated_tool\", \"arguments\": {}}</tool_call>"},
                        ]
                    }
                ],
            )
            r_tool = audit_training_dataset(bad_tool, dataset_format="sft")
            self.assertFalse(r_tool["passed"])
            self.assertLess(r_tool["metrics"]["tag_balance_rate"], 1.0)
            self.assertLess(r_tool["metrics"]["tool_schema_valid_rate"], 1.0)

            # 3) Over-budget truncation
            over_budget = p / "over_budget.jsonl"
            write_sft_jsonl(
                over_budget,
                [{"conversations": [{"role": "user", "content": "x " * 1200}, {"role": "assistant", "content": "ok"}]}],
            )
            r_budget = audit_training_dataset(over_budget, dataset_format="sft", max_seq_len=384)
            self.assertFalse(r_budget["passed"])
            self.assertEqual(r_budget["metrics"]["within_budget_rate"], 0.0)

            # 4) Template collapse / high duplicate rate
            cloned = p / "cloned.jsonl"
            row = {"conversations": [{"role": "user", "content": "same question"}, {"role": "assistant", "content": "same answer"}]}
            write_sft_jsonl(cloned, [row] * 30)
            r_clone = audit_training_dataset(cloned, dataset_format="sft")
            self.assertFalse(r_clone["passed"])
            self.assertGreater(r_clone["metrics"]["exact_duplicate_rate"], 0.15)

            # 5) Broken code edit execution replay
            bad_exec = p / "bad_exec.jsonl"
            write_sft_jsonl(
                bad_exec,
                [
                    {
                        "conversations": [
                            {"role": "user", "content": "foo.py:\n1: def f(x):\n2:     return x\n"},
                            {
                                "role": "assistant",
                                "content": '<think>bad</think><tool_call>{"name": "replace_line_range", "arguments": {"path": "foo.py", "start_line": 99, "end_line": 99, "lines": ["pass"]}}</tool_call>',
                            },
                        ]
                    }
                ],
            )
            r_exec = audit_training_dataset(bad_exec, dataset_format="sft")
            self.assertFalse(r_exec["passed"])
            self.assertEqual(r_exec["metrics"]["exec_verified_pass_rate"], 0.0)

            # 6) Held-out leakage
            heldout = p / "heldout.jsonl"
            write_sft_jsonl(heldout, [{"conversations": [{"role": "user", "content": "secret heldout prompt"}, {"role": "assistant", "content": "ans"}]}])
            leaked = p / "leaked.jsonl"
            write_sft_jsonl(leaked, [{"conversations": [{"role": "user", "content": "secret heldout prompt"}, {"role": "assistant", "content": "ans"}]}])
            r_leak = audit_training_dataset(leaked, dataset_format="sft", heldout_path=heldout)
            self.assertFalse(r_leak["passed"])
            self.assertGreaterEqual(r_leak["metrics"]["leakage_count"], 1)

            # 7) Repetitive prose spam caught even inside <think>
            rep_spam = p / "rep_spam.jsonl"
            write_sft_jsonl(
                rep_spam,
                [
                    {
                        "conversations": [
                            {"role": "user", "content": "fix bug"},
                            {"role": "assistant", "content": "<think>\nrepeat this phrase over and over " * 8 + "\n</think>\n\nok"},
                        ]
                    }
                ],
            )
            r_rep = audit_training_dataset(rep_spam, dataset_format="sft")
            self.assertFalse(r_rep["passed"])
            self.assertEqual(r_rep["metrics"]["high_rep_rate"], 1.0)

            # 8) Degenerate DPO pair & valid Pretrain row
            bad_dpo = p / "bad_dpo.jsonl"
            same = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "same"}]
            bad_dpo.write_text(json.dumps({"chosen": same, "rejected": same}) + "\n", encoding="utf-8")
            r_dpo = audit_training_dataset(bad_dpo, dataset_format="dpo")
            self.assertFalse(r_dpo["passed"])

            good_pre = p / "good_pre.jsonl"
            lines = [json.dumps({"text": f"# module_{i}\ndef step_{i}(x):\n    return max(1, x + {i})\n"}) + "\n" for i in range(25)]
            good_pre.write_text("".join(lines), encoding="utf-8")
            r_pre = audit_training_dataset(good_pre, dataset_format="auto")
            self.assertEqual(r_pre["format"], "pretrain")
            self.assertTrue(r_pre["passed"], msg=r_pre)

    def test_cli_generate_and_check_subcommands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "cli_out"
            g_proc = subprocess.run(
                [sys.executable, "-m", "selfchange.bootstrap.data_quality", "generate", "--output-dir", str(out_dir), "--sft-count", "40", "--dpo-count", "12", "--heldout-count", "10"],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
            )
            self.assertEqual(g_proc.returncode, 0, msg=g_proc.stderr or g_proc.stdout)

            c_ok = subprocess.run(
                [sys.executable, "-m", "selfchange.bootstrap.data_quality", "check", "--data-path", str(out_dir / "sft_train.jsonl"), "--heldout-path", str(out_dir / "heldout_eval.jsonl")],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
            )
            self.assertEqual(c_ok.returncode, 0, msg=c_ok.stderr or c_ok.stdout)

            bad_file = Path(tmp) / "corrupt.jsonl"
            bad_file.write_text('{"bad": 1}\n', encoding="utf-8")
            c_fail = subprocess.run(
                [sys.executable, "-m", "selfchange.bootstrap.data_quality", "check", "--data-path", str(bad_file)],
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
            )
            self.assertEqual(c_fail.returncode, 1)


if __name__ == "__main__":
    unittest.main()
