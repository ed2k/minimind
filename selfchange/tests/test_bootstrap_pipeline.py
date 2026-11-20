from __future__ import annotations

import json
import py_compile
import tempfile
import unittest
from pathlib import Path

from selfchange.bootstrap.eval_baseline import check_bootstrap_ready_gate, evaluate_baseline
from selfchange.bootstrap.generate_traces import estimate_tokens, generate_traces, validate_trace_record
from selfchange.bootstrap.midtrain_code import build_midtrain_corpus
from selfchange.bootstrap.offline_rft_dpo import collect_offline_rollouts, run_offline_rft_dpo_round
from selfchange.bootstrap.setup_base import build_anchor_split, setup_micro_model
from selfchange.bootstrap.warmstart_sft import run_warmstart_curriculum
from selfchange.kernel.budgets import get_profile
from selfchange.training.model_loader import normalize_messages_for_template
from selfchange.training.self_train_runner import stage_from_weight_for_trainer
from selfchange.training.train_self_agent import compute_group_advantages


class TestBootstrapPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.selfchange_dir = Path(__file__).resolve().parents[1]
        self.repo_root = self.selfchange_dir.parent

    def test_setup_base_micro_and_anchor_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            meta = setup_micro_model(tmp_path / "base", seed=42)
            self.assertEqual(meta["mode"], "micro")
            anchor_p = Path(meta["anchor_path"])
            heldout_p = Path(meta["heldout_path"])
            self.assertTrue(anchor_p.is_file())
            self.assertTrue(heldout_p.is_file())

            train_lines = set(anchor_p.read_text(encoding="utf-8").splitlines())
            heldout_lines = set(heldout_p.read_text(encoding="utf-8").splitlines())
            self.assertTrue(train_lines.isdisjoint(heldout_lines))

            src = tmp_path / "source.jsonl"
            src.write_text(
                "".join(json.dumps({"conversations": [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]}) + "\n" for i in range(40)),
                encoding="utf-8",
            )
            split = build_anchor_split(src, tmp_path / "split_a.jsonl", tmp_path / "split_h.jsonl", train_limit=20, heldout_limit=10)
            self.assertEqual(split["heldout_count"], 4)
            self.assertEqual(split["train_anchor_count"], 20)
            s_a = set((tmp_path / "split_a.jsonl").read_text(encoding="utf-8").splitlines())
            s_h = set((tmp_path / "split_h.jsonl").read_text(encoding="utf-8").splitlines())
            self.assertTrue(s_a.isdisjoint(s_h))

    def test_generate_traces_all_rungs_valid_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "traces.jsonl"
            stats = generate_traces(out_path, count=60, seed=42, max_seq_len=768, rung="all", real_ast_subprocesses=4)
            self.assertEqual(stats["generated"], 60)
            self.assertGreaterEqual(stats["real_ast_subprocesses"], 4)
            rows = [json.loads(ln) for ln in out_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            self.assertEqual(len(rows), 60)
            rungs_seen = {r["rung"] for r in rows}
            self.assertEqual(rungs_seen, {1, 2, 3})
            for row in rows:
                ok, err = validate_trace_record(row, max_seq_len=768)
                self.assertTrue(ok, msg=f"Invalid trace {err}: {row}")
                self.assertLessEqual(estimate_tokens(row["conversations"]), int(0.9 * 768))

    def test_eval_baseline_five_tiers_and_bootstrap_gate(self) -> None:
        scorecard = evaluate_baseline(
            self.repo_root,
            backend="scripted",
            profile_name="local_intel_cpu",
            num_code_tasks=10,
            drill_seeds=[42],
        )
        for key in ("tier1_language", "tier2_toolcall", "tier3_code_edit", "tier4_drills", "tier5_hardware"):
            self.assertIn(key, scorecard)
        self.assertTrue(scorecard["bootstrap_ready"], msg=f"Failed checks: {scorecard['failed_checks']}")

        weak_card = {
            "tier1_language": {"dialogue_ppl": 25.0, "eos_rate": 0.70, "mean_rep_penalty": 0.25},
            "tier2_toolcall": {"json_parse_rate": 0.40, "schema_validity_rate": 0.30, "copy_fidelity_rate": 0.50},
            "tier3_code_edit": {"exact_match_rate": 0.10, "ast_valid_rate": 0.20, "pass_at_1": 0.0, "pass_at_4": 0.0},
            "tier4_drills": {"pass_rate": 0.20, "config_drills_pass_rate": 0.30, "healthy_noop_rate": 0.50},
        }
        ready, failed, recs = check_bootstrap_ready_gate(weak_card, base_ppl=20.0)
        self.assertFalse(ready)
        self.assertGreaterEqual(len(failed), 8)
        self.assertTrue(any("replace_line_range" in r for r in recs))
        self.assertTrue(any("pass@4 < 10%" in r for r in recs))

    def test_midtrain_curriculum_offline_rft_dpo_and_group_adv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            anchor_pre = tmp_path / "anchor_pre.jsonl"
            anchor_pre.write_text('{"text": "Hello MiniMind"}\n' * 10, encoding="utf-8")
            mid_stats = build_midtrain_corpus(
                self.repo_root,
                tmp_path / "midtrain.jsonl",
                anchor_pretrain_jsonl=anchor_pre,
                max_samples=20,
            )
            self.assertEqual(mid_stats["code_samples"], 10)
            self.assertEqual(mid_stats["anchor_samples"], 10)

            ws = tmp_path / "workspace"
            warm_res = run_warmstart_curriculum(
                repo_root=self.repo_root,
                workspace_dir=ws,
                work_dir=tmp_path / "curr",
                profile_name="local_intel_cpu",
                total_traces=60,
                dry_run=True,
            )
            self.assertEqual(len(warm_res["stages"]), 3)
            self.assertTrue(all(s["ok"] for s in warm_res["stages"]))

            trajs = collect_offline_rollouts(self.repo_root, num_tasks=6, num_rollouts=4)
            rft_dpo_res = run_offline_rft_dpo_round(
                trajs,
                repo_root=self.repo_root,
                workspace_dir=ws,
                work_dir=tmp_path / "rft_dpo",
                anchor_samples=[],
                profile=get_profile("local_intel_cpu"),
                dry_run=True,
            )
            self.assertGreater(rft_dpo_res["rft_count"], 0)
            self.assertGreater(rft_dpo_res["dpo_pairs_count"], 0)
            self.assertTrue(Path(rft_dpo_res["rft_path"]).is_file())
            self.assertTrue(Path(rft_dpo_res["dpo_path"]).is_file())
            self.assertTrue(rft_dpo_res["rft_ok"])
            self.assertTrue(rft_dpo_res["dpo_ok"])

            adv_varied, zero_v = compute_group_advantages([1.0, -1.0, 2.0, 0.0], 2)
            self.assertEqual(zero_v, 0)
            self.assertEqual(len(adv_varied), 4)
            adv_flat, zero_f = compute_group_advantages([-1.0, -1.0, -1.0, -1.0], 4)
            self.assertEqual(zero_f, 1)
            self.assertEqual(adv_flat, [0.0, 0.0, 0.0, 0.0])

    def test_training_and_colab_modules_compile_and_notebook_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for folder in ("bootstrap", "training", "colab"):
                for py_file in sorted((self.selfchange_dir / folder).glob("*.py")):
                    cfile = str(Path(tmp) / f"{folder}_{py_file.stem}.pyc")
                    py_compile.compile(str(py_file), cfile=cfile, doraise=True)

        nb_path = self.selfchange_dir / "colab" / "run_selfchange.ipynb"
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        self.assertEqual(nb.get("nbformat"), 4)
        self.assertGreaterEqual(len(nb.get("cells", [])), 5)
        joined_src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
        self.assertIn("-b mychanges", joined_src)
        self.assertIn("--backend minimind", joined_src)

    def test_stage_weight_and_normalize_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = tmp_path / "repo"
            ws = tmp_path / "ws"
            profile = get_profile("local_intel_cpu")
            cand = ws / "models" / "candidates" / f"warmstart_rung3_{profile.hidden_size}.pth"
            cand.parent.mkdir(parents=True, exist_ok=True)
            cand.write_bytes(b"fake-ckpt")

            staged = stage_from_weight_for_trainer("warmstart_rung3", repo, ws, profile)
            self.assertIsNotNone(staged)
            self.assertTrue((repo / "out" / f"warmstart_rung3_{profile.hidden_size}.pth").is_file())

            raw_msgs = [
                {"role": "system", "content": "Sys", "tools": json.dumps([{"type": "function", "function": {"name": "foo"}}]), "tool_calls": None},
                {"role": "assistant", "content": "", "tool_calls": json.dumps([{"name": "foo", "arguments": {}}]), "reasoning_content": None},
            ]
            norm, tools = normalize_messages_for_template(raw_msgs)
            self.assertIsInstance(tools, list)
            self.assertEqual(tools[0]["function"]["name"], "foo")
            self.assertNotIn("tools", norm[0])
            self.assertIsInstance(norm[1]["tool_calls"], list)
            self.assertNotIn("reasoning_content", norm[1])


if __name__ == "__main__":
    unittest.main()
