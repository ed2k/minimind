from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from selfchange.bootstrap.limitation_to_fix import build_limitation_fix_and_meta_dataset
from selfchange.colab.autonomous_engine import (
    STRICT_EVAL_SUITE,
    STRICT_HELDOUT_PROMPTS,
    SelfImprovingCycleRunner,
    build_exploration_tasks,
    diagnose_weakness_state,
    evaluate_model_strict_suite,
    run_multi_rollout_exploration,
    score_strict_task,
)
from selfchange.colab.run_colab_cli import inspect_resume_seed, run_colab_cli_pipeline
from selfchange.kernel.autonomous_cycle import CycleConfig
from selfchange.kernel.budgets import get_profile
from selfchange.kernel.registry import LineageNode
from selfchange.kernel.supervisor import CycleContext
from selfchange.mutable.data_curator import build_dpo_pairs, curate_rft_samples
from selfchange.training.model_loader import estimate_minimind_params, save_model_config
from selfchange.training.self_train_runner import stage_from_weight_for_trainer


class TestAutonomousSelfImprovementEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.profile = get_profile("local_intel_cpu")

    def test_strict_eval_rejects_math_loophole_and_float_line_index(self) -> None:
        math_task = next(t for t in STRICT_EVAL_SUITE if t["id"] == "eval_11_math_oom_2step")
        bad_math = "<think>\nObserve: batch_size=16, accumulation_steps=2.\n</think>\n\nFinal batch_size=8 and accumulation_steps=4 (effective batch size remains 32)."
        good_math = "<think>\nObserve: batch_size=16, accumulation_steps=2. Step 1: batch_size=8, accum=4; Step 2: batch_size=4, accum=8.\n</think>\n\nFinal batch_size=4 and accumulation_steps=8."
        self.assertFalse(score_strict_task(math_task, bad_math, True)["passed"])
        self.assertTrue(score_strict_task(math_task, good_math, True)["passed"])

        code_task = next(t for t in STRICT_EVAL_SUITE if t["id"] == "eval_8_code_clamp_float1")
        bad_float = '<think>\nObserve: eval_lr_guard.py line 4.\n</think>\n\n<tool_call>\n{"name":"replace_line_range","arguments":{"path":"eval_lr_guard.py","start_line":5e-05,"end_line":5,"lines":["    return min(lr, max_lr)"]}}\n</tool_call>'
        good_code = '<think>\nObserve: eval_lr_guard.py line 4 uses max instead of min. Plan: call replace_line_range line 4.\n</think>\n\n<tool_call>\n{"name":"replace_line_range","arguments":{"path":"eval_lr_guard.py","start_line":4,"end_line":4,"lines":["    return min(lr, max_lr)"]}}\n</tool_call>'
        self.assertFalse(score_strict_task(code_task, bad_float, True)["passed"])
        self.assertTrue(score_strict_task(code_task, good_code, True)["passed"])

    def test_strict_heldout_prompts_exclude_all_20_tasks_without_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rep = build_limitation_fix_and_meta_dataset(
                repo_root=self.repo_root,
                output_dir=Path(tmp) / "fix_strict",
                sft_count=120,
                dpo_count=30,
                max_seq_len=768,
                seed=42,
                extra_heldout_prompts=STRICT_HELDOUT_PROMPTS,
            )
            self.assertTrue(rep["overall_passed"], msg=rep)
            self.assertEqual(rep["sft_audit"]["metrics"]["leakage_count"], 0)
            sft_raw = Path(rep["sft_path"]).read_text(encoding="utf-8")
            for item in STRICT_HELDOUT_PROMPTS:
                usr = item["messages"][-1]["content"]
                self.assertNotIn(usr, sft_raw)

    def test_multi_rollout_exploration_produces_rft_and_same_task_dpo_pairs(self) -> None:
        sc = {"code_pass_at_1": 0.3, "math_pass_at_1": 0.5, "tool_pass_at_1": 0.9, "noop_pass_rate": 1.0, "eos_rate": 0.95}
        tasks = build_exploration_tasks(sc, cycle_index=2, max_tasks=3)
        self.assertEqual(len(tasks), 3)
        self.assertNotEqual(diagnose_weakness_state(sc, 2, "p0", {}, False, False)["last_fault"], "none")

        call_counts: dict[str, int] = {}

        def _mock_reply(msgs, _tools, _temp):
            usr = msgs[-1]["content"]
            call_counts[usr] = call_counts.get(usr, 0) + 1
            attempt = call_counts[usr]
            if "exp_bound_2.py" in usr:
                if attempt == 1:
                    return '<think>\nObserve: exp_bound_2.py.\n</think>\n\n<tool_call>\n{"name":"replace_line_range","arguments":{"path":"exp_bound_2.py","start_line":5e-4,"end_line":4,"lines":["    return min(val, hi)"]}}\n</tool_call>', True
                return '<think>\nObserve: exp_bound_2.py line 4 uses max instead of min. Plan: call replace_line_range.\n</think>\n\n<tool_call>\n{"name":"replace_line_range","arguments":{"path":"exp_bound_2.py","start_line":4,"end_line":4,"lines":["    return min(val, hi)"]}}\n</tool_call>', True
            if "batch_size=" in usr:
                if attempt <= 2:
                    return "<think>\nObserve: batch_size wrong.\n</think>\n\nFinal batch_size=99 and accumulation_steps=1.", True
                for t in tasks:
                    if t["category"] == "reasoning_math":
                        return f"<think>\nObserve: {t['expected_bs']} {t['expected_acc']}.\n</think>\n\nFinal batch_size={t['expected_bs']} and accumulation_steps={t['expected_acc']}.", True
            if "fault=nan_loss" in usr:
                return '<think>\nObserve: nan_loss lr=0.0002 grad_norm=inf. Plan: call patch_config.\n</think>\n\n<tool_call>\n{"name":"patch_config","arguments":{"updates":{"training":{"learning_rate":0.0001,"grad_clip":0.5}}}}\n</tool_call>', True
            return "<think>\nObserve: fallback.\n</think>\n\nDone", True

        res = run_multi_rollout_exploration(None, None, self.profile, tasks, k_rollouts=4, reply_fn=_mock_reply)
        self.assertGreater(res["pass_at_4"], res["pass_at_1"])
        rft = curate_rft_samples(res["trajectories"])
        dpo = build_dpo_pairs(res["trajectories"])
        self.assertGreaterEqual(len(rft), 2)
        self.assertGreaterEqual(len(dpo), 1)

        ev = evaluate_model_strict_suite(None, None, self.profile, tasks=STRICT_EVAL_SUITE[:2], k_rollouts=1, reply_fn=lambda m, t, tp: ('<think>\nObserve: fault.\n</think>\n\n<tool_call>\n{"name":"patch_config","arguments":{"updates":{"training":{"learning_rate":0.0001,"grad_clip":0.5}}}}\n</tool_call>', True))
        self.assertIn("overall_score", ev)

    def test_self_improving_runner_live_metrics_config_propagation_and_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            ws = tmp_p / "ws"
            (ws / "models" / "candidates").mkdir(parents=True)
            base_w = ws / "models" / "candidates" / "pruned_42m_768.pth"
            base_w.write_bytes(b"W0")
            arch42 = estimate_minimind_params(hidden_size=768, num_hidden_layers=8, intermediate_size=1216)
            save_model_config(base_w.with_suffix(".config.json"), arch42)

            staged = stage_from_weight_for_trainer("pruned_42m", self.repo_root, ws, self.profile, from_weight_path=base_w, model_config=arch42)
            try:
                self.assertIsNotNone(staged)
                self.assertTrue(staged.with_suffix(".config.json").is_file())
            finally:
                if staged and staged.exists():
                    staged.unlink()
                if staged and staged.with_suffix(".config.json").exists():
                    staged.with_suffix(".config.json").unlink()

            class _FakeSession:
                def __init__(self, w_path):
                    self.model_config = arch42
                    self.last_scorecard = {
                        "passed_count": 5,
                        "total_count": 20,
                        "pass_at_1": 0.25,
                        "pass_at_4": 0.50,
                        "eos_rate": 0.90,
                        "code_pass_at_1": 0.33,
                        "math_pass_at_1": 0.50,
                        "tool_pass_at_1": 0.75,
                        "noop_pass_rate": 1.0,
                        "overall_score": 0.52,
                        "examples": [],
                    }
                    self.last_exploration = {"pass_at_1": 0.25, "pass_at_4": 0.75}
                    self._calls = 0
                    self._base_w = Path(w_path)
                    self._w = w_path

                def _ensure_loaded(self, p):
                    self._w = p

                def eval_strict_scorecard(self, p=None):
                    self._calls += 1
                    if p is None or Path(p) == self._base_w:
                        return self.last_scorecard
                    return {**self.last_scorecard, "passed_count": 8, "pass_at_1": 0.40, "overall_score": 0.71, "eos_rate": 0.95}

            fake_sess = _FakeSession(base_w)
            sample_trajs = [
                {
                    "task_id": f"exp_{i}",
                    "task_family": f"fam_{i}",
                    "passed": True,
                    "reward": 1.0,
                    "tool_calls": [{"name": "patch_config", "arguments": {"updates": {"grad_clip": 0.5}}}],
                    "messages": [
                        {"role": "user", "content": f"step={10 + i} fault=nan_loss grad_norm=inf"},
                        {
                            "role": "assistant",
                            "content": f"<think>Saw step={10 + i} fault=nan_loss; call patch_config.</think>\n<tool_call>{{\"name\": \"patch_config\", \"arguments\": {{\"updates\": {{\"grad_clip\": 0.5}}}}}}</tool_call>",
                        },
                    ],
                    "final_text": f"Patched grad_clip at step={10 + i}.",
                }
                for i in range(4)
            ]
            (ws / "trajectories.jsonl").write_text("\n".join(json.dumps(r) for r in sample_trajs) + "\n", encoding="utf-8")
            wt = tmp_p / "wt"
            (wt / "selfchange" / "mutable").mkdir(parents=True)
            (wt / "selfchange" / "mutable" / "config.json").write_text('{"training":{"batch_size":2,"learning_rate":3e-5}}', encoding="utf-8")

            runner = SelfImprovingCycleRunner(
                repo_root=self.repo_root,
                workspace_dir=ws,
                profile=self.profile,
                cycle_config=CycleConfig(dry_run_training=False, base_weight_path=base_w),
                session=fake_sess,
                live_self_improve=True,
                model_config=arch42,
            )
            ctx = CycleContext(cycle_index=3, profile=self.profile, workspace_dir=ws, parent=LineageNode("b0", None, "HEAD", str(base_w), 1.0, 0.7, 1.0, 1.0, 0.9, "bootstrap", "b0", "t"), safe_mode=False, config_only=False)

            captured = {}
            out_w = ws / "models" / "candidates" / "cand_c0003_768.pth"

            def _fake_stage(**kwargs):
                captured["model_config"] = kwargs["model_config"]
                captured["stage"] = kwargs["stage"]
                out_w.write_bytes(b"W1")
                return type("Res", (), {"ok": True, "exit_code": 0, "final_loss": 0.42, "duration_sec": 1.0, "output_weight_path": str(out_w), "log_tail": "ok"})()

            with patch("selfchange.colab.autonomous_engine.select_training_stage", return_value={"stage": "dpo", "training": {"dpo_learning_rate": 2e-7}, "reason": "live_test"}) as sel_mock, \
                 patch("selfchange.colab.autonomous_engine.run_training_stage", side_effect=_fake_stage):
                chosen_w = runner._run_periodic_training(ctx, wt)

            self.assertEqual(chosen_w, str(out_w))
            passed_metrics = sel_mock.call_args.kwargs["metrics"]
            self.assertEqual(passed_metrics["pass_at_1"], 0.25)
            self.assertEqual(passed_metrics["pass_at_4"], 0.75)
            self.assertTrue(passed_metrics["rl_ready"])
            self.assertEqual(captured["model_config"]["intermediate_size"], 1216)
            self.assertTrue((ws / "curated_data" / "verified_selfchange_episodes.jsonl").is_file())

    def test_run_colab_cli_overlay_and_episode_carryover(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            prom = ws / "promoted_models" / "r1"
            cur = ws / "curated_data"
            prom.mkdir(parents=True)
            cur.mkdir(parents=True)
            w = prom / "champ_768.pth"
            w.write_bytes(b"W")
            save_model_config(w.with_suffix(".config.json"), estimate_minimind_params(768, 8, 1216))
            (cur / "verified_selfchange_episodes.jsonl").write_text('{"conversations":[{"role":"user","content":"u"},{"role":"assistant","content":"a"}]}\n', encoding="utf-8")
            (ws / "promoted_models" / "latest_champion.json").write_text(json.dumps({"run_id": "r1", "primary_weight": str(w), "weights": [str(w)]}), encoding="utf-8")

            plan = inspect_resume_seed(ws / "promoted_models")
            self.assertEqual(len(plan["extra_uploads"]), 2)

            cmds = []

            def _mock(cmd, **_kwargs):
                cmds.append(cmd)
                return type("Proc", (), {"returncode": 0})()

            res = run_colab_cli_pipeline(session="ov", workspace_dir=ws, resume=True, overlay_local_code=True, run_cmd=_mock)
            self.assertIsNone(res["error"])
            uploads = [c[5] for c in cmds if c[:2] == ["colab", "upload"]]
            self.assertIn("/content/local_code_overlay.tar.gz", uploads)
            self.assertTrue(any("verified_selfchange_episodes.jsonl" in u for u in uploads))


if __name__ == "__main__":
    unittest.main()
