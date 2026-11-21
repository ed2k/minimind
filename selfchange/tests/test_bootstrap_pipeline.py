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
from selfchange.colab.run_colab_cli import (
    build_remote_phase_scripts,
    inspect_resume_seed,
    persist_improved_weights_from_colab,
    run_colab_cli_pipeline,
)
from selfchange.kernel.budgets import get_profile
from selfchange.training.model_loader import normalize_messages_for_template, restore_open_thinking_prefix
from selfchange.training.self_train_runner import (
    format_dpo_record,
    format_sft_record,
    stage_from_weight_for_trainer,
    write_dpo_jsonl,
    write_sft_jsonl,
)
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
            expected_msg_keys = {"role", "content", "reasoning_content", "tools", "tool_calls"}
            for idx in (1, 2, 3):
                cpath = Path(warm_res["curriculum"][f"rung{idx}"]["mixed_path"])
                rows = [json.loads(ln) for ln in cpath.read_text(encoding="utf-8").splitlines() if ln.strip()]
                self.assertGreater(len(rows), 0)
                for row in rows:
                    self.assertEqual(set(row.keys()), {"conversations"})
                    for msg in row["conversations"]:
                        self.assertEqual(set(msg.keys()), expected_msg_keys)

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
            for rrow in [json.loads(ln) for ln in Path(rft_dpo_res["rft_path"]).read_text(encoding="utf-8").splitlines() if ln.strip()]:
                self.assertEqual(set(rrow.keys()), {"conversations"})
            for drow in [json.loads(ln) for ln in Path(rft_dpo_res["dpo_path"]).read_text(encoding="utf-8").splitlines() if ln.strip()]:
                self.assertEqual(set(drow.keys()), {"chosen", "rejected"})
                self.assertTrue(all(set(m.keys()) == {"role", "content"} for m in drow["chosen"]))
                self.assertTrue(all(set(m.keys()) == {"role", "content"} for m in drow["rejected"]))

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
        self.assertIn("--require-model", joined_src)
        self.assertIn("--eval-with-model", joined_src)

    def test_stage_weight_and_normalize_messages_and_strict_formatters(self) -> None:
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

            custom_w = tmp_path / "custom.pth"
            custom_w.write_bytes(b"custom-ckpt")
            staged_custom = stage_from_weight_for_trainer("custom_base", repo, ws, profile, from_weight_path=custom_w)
            self.assertIsNotNone(staged_custom)
            self.assertEqual((repo / "out" / f"custom_base_{profile.hidden_size}.pth").read_bytes(), b"custom-ckpt")

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

            sft_rec = format_sft_record({"conversations": raw_msgs, "task_family": "drop_me", "score": 2.0})
            self.assertEqual(set(sft_rec.keys()), {"conversations"})
            self.assertEqual(set(sft_rec["conversations"][0].keys()), {"role", "content", "reasoning_content", "tools", "tool_calls"})
            self.assertEqual(write_sft_jsonl(tmp_path / "sft.jsonl", [{"messages": raw_msgs, "score": 1.0}]), 1)
            self.assertEqual(write_dpo_jsonl(tmp_path / "dpo.jsonl", [{"chosen": raw_msgs, "rejected": raw_msgs, "task_id": "x"}]), 1)

            self.assertEqual(restore_open_thinking_prefix("check</think>\nok", True), "<think>\ncheck</think>\nok")
            self.assertEqual(restore_open_thinking_prefix("<think>\ncheck</think>", True), "<think>\ncheck</think>")
            self.assertEqual(restore_open_thinking_prefix("plain reply", False), "plain reply")

    def test_run_colab_cli_conditional_persistence_and_script_generation(self) -> None:
        import ast

        scripts = build_remote_phase_scripts(branch="mychanges", total_traces=120, max_cycles=3)
        self.assertEqual(set(scripts.keys()), {"step1_setup.py", "step2_bootstrap.py", "step3_autonomous.py"})
        for src in scripts.values():
            ast.parse(src)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            diag_no = tmp_path / "diag_no"
            diag_no.mkdir()
            (diag_no / "session_info.json").write_text(
                json.dumps({"persistence_decision": {"improved": False, "persist_weights": False, "reason": "no_improvement_over_baseline", "champion_weight_path": "/content/w.pth"}}),
                encoding="utf-8",
            )
            (diag_no / "manifest.json").write_text(
                json.dumps({"weight_rel_map": {"weights/w_768.pth": "models/candidates/w_768.pth"}}),
                encoding="utf-8",
            )
            called_cmds: list[list[str]] = []

            def _mock_run(cmd, **_kwargs):
                called_cmds.append(cmd)
                if len(cmd) >= 5 and cmd[:2] == ["colab", "download"]:
                    Path(cmd[5]).write_bytes(b"PT_GOOD")
                return type("Proc", (), {"returncode": 0, "stdout": "{}"})()

            res_no = persist_improved_weights_from_colab(
                session="sess",
                diagnostics_dir=diag_no,
                promoted_dir=tmp_path / "promoted_no",
                run_id="run_1",
                run_cmd=_mock_run,
            )
            self.assertFalse(res_no["persisted"])
            self.assertEqual(called_cmds, [])

            diag_yes = tmp_path / "diag_yes"
            diag_yes.mkdir()
            (diag_yes / "session_info.json").write_text(
                json.dumps({"persistence_decision": {"improved": True, "persist_weights": True, "reason": "autonomous_cycle_improved", "champion_weight_path": "/content/w_768.pth"}}),
                encoding="utf-8",
            )
            (diag_yes / "manifest.json").write_text(
                json.dumps({"weight_rel_map": {"weights/w_768.pth": "models/candidates/w_768.pth"}}),
                encoding="utf-8",
            )
            promoted_yes = tmp_path / "promoted_yes"
            res_yes = persist_improved_weights_from_colab(
                session="sess",
                diagnostics_dir=diag_yes,
                promoted_dir=promoted_yes,
                run_id="run_2",
                run_cmd=_mock_run,
            )
            self.assertTrue(res_yes["persisted"])
            self.assertEqual(len(called_cmds), 1)
            self.assertEqual(called_cmds[0][:3], ["colab", "download", "-s"])
            self.assertTrue((promoted_yes / "run_2" / "w_768.pth").is_file())
            self.assertTrue((promoted_yes / "latest_champion.json").is_file())

    def test_run_colab_cli_resume_seeding_and_skip_bootstrap(self) -> None:
        import ast
        import tarfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ws = tmp_path / "ws"
            promoted = ws / "promoted_models"
            self.assertEqual(inspect_resume_seed(promoted)["mode"], "none")

            # Seed prior diagnostics + promoted binary weight -> full_mirror mode
            diag = tmp_path / "prev_diag"
            (diag / "logs").mkdir(parents=True)
            (diag / "archive.jsonl").write_text('{"node_id": "n1"}\n', encoding="utf-8")
            (diag / "champion.json").write_text('{"champion": {"node_id": "n1"}}\n', encoding="utf-8")
            (diag / "logs" / "train.log").write_text("loss: 0.2\n", encoding="utf-8")
            (diag / "manifest.json").write_text(
                json.dumps(
                    {
                        "files": {
                            "archive.jsonl": "h1",
                            "champion.json": "h2",
                            "logs/train.log": "h3",
                            "weights/cand_768.pth": "h4",
                        },
                        "weight_rel_map": {"weights/cand_768.pth": "models/candidates/cand_768.pth"},
                    }
                ),
                encoding="utf-8",
            )
            (promoted / "run_prev").mkdir(parents=True)
            w_prev = promoted / "run_prev" / "cand_768.pth"
            w_prev.write_bytes(b"PT_PREV")
            (promoted / "latest_champion.json").write_text(
                json.dumps(
                    {
                        "run_id": "run_prev",
                        "diagnostics_dir": str(diag),
                        "primary_weight": str(w_prev),
                        "weights": [str(w_prev)],
                    }
                ),
                encoding="utf-8",
            )

            plan_full = inspect_resume_seed(promoted)
            self.assertEqual(plan_full["mode"], "full_mirror")
            self.assertEqual(len(plan_full["weight_uploads"]), 1)

            def _make_mock_runner():
                recorded: list[list[str]] = []

                def _mock_run(cmd, **_kwargs):
                    recorded.append(cmd)
                    if len(cmd) >= 5 and cmd[:2] == ["colab", "exec"] and cmd[-2] == "-f":
                        ast.parse(Path(cmd[-1]).read_text(encoding="utf-8"))
                    if len(cmd) >= 5 and cmd[:2] == ["colab", "download"] and cmd[4] == "/content/run_diagnostics.tar.gz":
                        out_tar = Path(cmd[5])
                        out_tar.parent.mkdir(parents=True, exist_ok=True)
                        stub_dir = tmp_path / "stub_tar"
                        stub_dir.mkdir(exist_ok=True)
                        (stub_dir / "session_info.json").write_text(
                            json.dumps({"persistence_decision": {"improved": False, "persist_weights": False, "reason": "no_improvement_over_baseline"}}),
                            encoding="utf-8",
                        )
                        with tarfile.open(out_tar, "w:gz") as tf:
                            tf.add(stub_dir / "session_info.json", arcname="session_info.json")
                    return type("Proc", (), {"returncode": 0, "stdout": "{}"})()

                return recorded, _mock_run

            # 1) resume=True, skip_bootstrap_on_resume=True -> uploads seed tar + .pth, skips step2_bootstrap.py, stops VM
            cmds1, runner1 = _make_mock_runner()
            res1 = run_colab_cli_pipeline(
                session="sess1",
                workspace_dir=ws,
                resume=True,
                skip_bootstrap_on_resume=True,
                run_cmd=runner1,
            )
            self.assertTrue(res1["bootstrap_skipped"])
            self.assertTrue(res1["resume_seed"]["seeded"])
            self.assertEqual(res1["resume_seed"]["mode"], "full_mirror")
            uploads1 = [c for c in cmds1 if c[:2] == ["colab", "upload"]]
            self.assertTrue(any(c[5] == "/content/seed_mirror_state.tar.gz" for c in uploads1))
            self.assertTrue(any(c[5] == "/content/drive_mirror/weights/cand_768.pth" for c in uploads1))
            exec_scripts1 = [Path(c[-1]).name for c in cmds1 if c[:2] == ["colab", "exec"] and c[-2] == "-f"]
            self.assertEqual(exec_scripts1, ["step1_setup.py", "step_unpack_seed.py", "step3_autonomous.py"])
            self.assertEqual(cmds1[-1], ["colab", "stop", "-s", "sess1"])

            # 2) resume=True, skip_bootstrap_on_resume=False -> seeds VM AND still runs step2_bootstrap.py
            cmds2, runner2 = _make_mock_runner()
            res2 = run_colab_cli_pipeline(
                session="sess2",
                workspace_dir=ws,
                resume=True,
                skip_bootstrap_on_resume=False,
                run_cmd=runner2,
            )
            self.assertFalse(res2["bootstrap_skipped"])
            self.assertTrue(res2["resume_seed"]["seeded"])
            exec_scripts2 = [Path(c[-1]).name for c in cmds2 if c[:2] == ["colab", "exec"] and c[-2] == "-f"]
            self.assertEqual(exec_scripts2, ["step1_setup.py", "step_unpack_seed.py", "step2_bootstrap.py", "step3_autonomous.py"])

            # 3) resume=False -> no uploads, runs full step1 -> step2 -> step3
            cmds3, runner3 = _make_mock_runner()
            res3 = run_colab_cli_pipeline(
                session="sess3",
                workspace_dir=ws,
                resume=False,
                run_cmd=runner3,
            )
            self.assertFalse(res3["bootstrap_skipped"])
            self.assertEqual([c for c in cmds3 if c[:2] == ["colab", "upload"]], [])
            exec_scripts3 = [Path(c[-1]).name for c in cmds3 if c[:2] == ["colab", "exec"] and c[-2] == "-f"]
            self.assertEqual(exec_scripts3, ["step1_setup.py", "step2_bootstrap.py", "step3_autonomous.py"])

            # 4) Missing manifest falls back cleanly to weight_only seeding with explicit --model-path in step3
            (diag / "manifest.json").unlink()
            plan_wo = inspect_resume_seed(promoted)
            self.assertEqual(plan_wo["mode"], "weight_only")
            cmds4, runner4 = _make_mock_runner()
            res4 = run_colab_cli_pipeline(
                session="sess4",
                workspace_dir=ws,
                resume=True,
                skip_bootstrap_on_resume=True,
                run_cmd=runner4,
            )
            self.assertEqual(res4["resume_seed"]["mode"], "weight_only")
            uploads4 = [c for c in cmds4 if c[:2] == ["colab", "upload"]]
            self.assertTrue(any(c[5].endswith("/cand_768.pth") for c in uploads4))
            step3_file = [Path(c[-1]) for c in cmds4 if c[:2] == ["colab", "exec"] and Path(c[-1]).name == "step3_autonomous.py"][0]
            self.assertIn("--model-path", step3_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
