from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from unittest.mock import patch

from selfchange.bootstrap.data_quality import _extract_prompt_keys
from selfchange.colab.autonomous_engine import (
    STRICT_HELDOUT_PROMPTS,
    SelfImprovingCycleRunner,
    build_exploration_tasks,
    diagnose_weakness_state,
    prune_forever_workspace,
    score_strict_task,
)
from selfchange.colab.colab_launcher import MiniMindProposerSession
from selfchange.colab.forever_daemon import run_forever_daemon
from selfchange.kernel.autonomous_cycle import CycleConfig
from selfchange.kernel.budgets import get_profile
from selfchange.kernel.registry import LineageNode
from selfchange.kernel.supervisor import CycleContext
from selfchange.training.llamacpp_backend import LlamaCppServerSession
from selfchange.training.model_loader import estimate_minimind_params, save_model_config
from selfchange.training.self_train_runner import TrainingRunResult


class TestForeverDaemonAndLearning(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.profile = get_profile("local_intel_cpu")

    def test_prune_forever_workspace_protects_champion_and_compacts_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "ws"
            gguf_dir = ws / "promoted_models" / "gguf"
            cand_dir = gguf_dir / "candidates"
            train_dir = ws / "training_data"
            rev_dir = ws / "data_reviews"
            cand_dir.mkdir(parents=True)
            train_dir.mkdir(parents=True)
            rev_dir.mkdir(parents=True)

            champ = gguf_dir / "champion.gguf"
            champ.write_bytes(b"GGUF_CHAMP")
            prot_cand = cand_dir / "cand_c0001-q4_k_m.gguf"
            prot_cand.write_bytes(b"GGUF_PROT")
            (prot_cand.with_suffix(".config.json")).write_text("{}", encoding="utf-8")
            (ws / "champion.json").write_text(json.dumps({"champion": {"weight_path": str(prot_cand)}}), encoding="utf-8")

            cands = [prot_cand]
            for i in range(2, 6):
                p = cand_dir / f"cand_c000{i}-q4_k_m.gguf"
                p.write_bytes(b"GGUF")
                p.with_suffix(".config.json").write_text("{}", encoding="utf-8")
                os.utime(p, (1000 + i, 1000 + i))
                cands.append(p)
            os.utime(prot_cand, (500, 500))

            for i in range(1, 6):
                d = train_dir / f"synth_c000{i}"
                d.mkdir()
                (d / "sft.jsonl").write_text("{}\n", encoding="utf-8")
                os.utime(d, (1000 + i, 1000 + i))
                rft = train_dir / f"rft_c000{i}.jsonl"
                rft.write_text("{}\n", encoding="utf-8")
                os.utime(rft, (1000 + i, 1000 + i))
                sc = ws / f"live_scorecard_c000{i}.json"
                sc.write_text("{}\n", encoding="utf-8")
                os.utime(sc, (1000 + i, 1000 + i))

            for i in range(1, 13):
                rf = rev_dir / f"rev_{i:02d}.json"
                rf.write_text("{}\n", encoding="utf-8")
                os.utime(rf, (1000 + i, 1000 + i))

            (ws / "trajectories.jsonl").write_text(
                "\n".join(json.dumps({"idx": i}) for i in range(450)) + "\n",
                encoding="utf-8",
            )

            rep = prune_forever_workspace(ws, keep_gguf_candidates=2, keep_synth_dirs=3, keep_scorecards=4, keep_reviews=10, max_trajectories=400)
            self.assertTrue(champ.is_file())
            self.assertTrue(prot_cand.is_file())
            self.assertFalse((cand_dir / "cand_c0002-q4_k_m.gguf").exists())
            self.assertFalse((cand_dir / "cand_c0002-q4_k_m.config.json").exists())
            self.assertEqual(len(list(train_dir.glob("synth_c*"))), 3)
            self.assertEqual(len(list(train_dir.glob("rft_c*.jsonl"))), 3)
            self.assertEqual(len(list(ws.glob("live_scorecard_c*.json"))), 4)
            self.assertEqual(len(list(rev_dir.iterdir())), 10)
            self.assertEqual(rep["compacted_trajectories"], 50)
            self.assertEqual(rep["retained_trajectories"], 400)

    def test_exploration_targeting_no_heldout_leakage_and_diagnose_feedback(self) -> None:
        heldout_exact = {_extract_prompt_keys(t["messages"])[0] for t in STRICT_HELDOUT_PROMPTS}
        heldout_norm = {_extract_prompt_keys(t["messages"])[1] for t in STRICT_HELDOUT_PROMPTS}
        sc = {
            "code_pass_at_1": 0.8333,
            "math_pass_at_1": 0.5,
            "tool_pass_at_1": 0.75,
            "noop_pass_rate": 1.0,
            "eos_rate": 0.9,
            "failing_task_ids": [
                "eval_3_tool_rollback",
                "eval_10_code_clamp_float3",
                "eval_13_math_lr_2step",
                "eval_18_qa_zh_accum",
            ],
        }
        tasks = build_exploration_tasks(sc, cycle_index=7, max_tasks=4)
        self.assertEqual(len(tasks), 4)
        ids = {t["id"] for t in tasks}
        self.assertIn("exp_c7_rb", ids)
        self.assertIn("exp_c7_code_floor", ids)
        self.assertIn("exp_c7_math_lr", ids)
        self.assertIn("exp_c7_qa_zh", ids)
        for t in tasks:
            ex_u, nm_u = _extract_prompt_keys(t["messages"])
            self.assertNotIn(ex_u, heldout_exact)
            self.assertNotIn(nm_u, heldout_norm)

        qa_zh = next(t for t in tasks if t["id"] == "exp_c7_qa_zh")
        ok_reply = "<think>\nExplain gradient accumulation carefully.\n</think>\n\n梯度累积在微批受限时分步累加梯度，等同于增大有效 batch_size 而不增加单步显存。"
        self.assertTrue(score_strict_task(qa_zh, ok_reply, True)["passed"])

        diag = diagnose_weakness_state(
            sc,
            7,
            "p0",
            {},
            False,
            False,
            recent_train_rows=[{"stage": "lora", "ok": True, "accepted": False, "final_loss": 0.0006, "before_score": 0.7996, "after_score": 0.7858}],
        )
        self.assertIn("last_train=lora", diag["log_tail"])
        self.assertIn("overfit_hint=lower_lr_or_switch_dpo", diag["log_tail"])

    def test_gguf_local_cpu_adaptive_backoff_anchor_mix_and_rejected_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp).resolve()
            ws = tmp_p / "ws"
            repo = tmp_p / "repo"
            (ws / "logs").mkdir(parents=True)
            (ws / "promoted_models" / "gguf").mkdir(parents=True)
            (repo / "out").mkdir(parents=True)
            (repo / "selfchange" / "mutable").mkdir(parents=True)

            arch = estimate_minimind_params(128, 4, 384, 4, 2)
            champ_gguf = ws / "promoted_models" / "gguf" / "champion.gguf"
            champ_gguf.write_bytes(b"GGUF_BASE")
            save_model_config(champ_gguf.with_suffix(".config.json"), arch)
            base_pth = repo / "out" / "minimal_cpu_champion.pth"
            base_pth.write_bytes(b"PTH" * 500)
            save_model_config(base_pth.with_suffix(".config.json"), arch)

            wt = tmp_p / "wt"
            (wt / "selfchange" / "mutable").mkdir(parents=True)
            (wt / "selfchange" / "mutable" / "config.json").write_text('{"training":{"learning_rate":3e-5}}', encoding="utf-8")

            anchor_f = ws / "anchor.jsonl"
            anchor_rows = [
                {
                    "conversations": [
                        {"role": "user", "content": STRICT_HELDOUT_PROMPTS[0]["messages"][-1]["content"]},
                        {"role": "assistant", "content": "Leaked heldout should be stripped"},
                    ]
                }
            ]
            for i in range(5):
                anchor_rows.append(
                    {
                        "conversations": [
                            {"role": "user", "content": f"Anchor safe operational check #{i}: batch_size=8, steps=2"},
                            {"role": "assistant", "content": "<think>\nObserved batch_size=8 steps=2.\n</think>\n\nHalve batch_size cleanly."},
                        ]
                    }
                )
            anchor_f.write_text("\n".join(json.dumps(r) for r in anchor_rows) + "\n", encoding="utf-8")

            # Seed 1 rejected low-loss LoRA row so cycle 2 alternates to DPO (rej_streak=1)
            (ws / "logs" / "train_history.jsonl").write_text(
                json.dumps({"cycle_index": 1, "stage": "lora", "ok": True, "accepted": False, "final_loss": 0.002}) + "\n",
                encoding="utf-8",
            )

            class _FakeGGUFEngineSession:
                def __init__(self):
                    self.is_gguf_mode = True
                    self.loaded_path = champ_gguf
                    self.model_config = arch
                    self.tokenizer_path = repo / "model"
                    self.last_scorecard = {
                        "passed_count": 14,
                        "total_count": 20,
                        "pass_at_1": 0.7,
                        "pass_at_4": 0.7,
                        "eos_rate": 0.9,
                        "overall_score": 0.7996,
                        "code_pass_at_1": 0.83,
                        "math_pass_at_1": 0.5,
                        "noop_pass_rate": 1.0,
                    }

                def close(self):
                    return None

                def _ensure_loaded(self, p):
                    self.loaded_path = Path(p)

                def eval_strict_scorecard(self, p=None):
                    if p is not None and Path(p) != champ_gguf:
                        return {**self.last_scorecard, "passed_count": 13, "overall_score": 0.78}
                    return self.last_scorecard

            sess = _FakeGGUFEngineSession()
            runner = SelfImprovingCycleRunner(
                repo_root=repo,
                workspace_dir=ws,
                profile=self.profile,
                cycle_config=CycleConfig(dry_run_training=False, anchor_path=anchor_f, base_weight_path=champ_gguf),
                session=sess,
                live_self_improve=True,
                model_config=arch,
                enable_local_cpu_train=True,
                local_train_base_pth=base_pth,
                local_cpu_sft_count=20,
                local_cpu_dpo_count=12,
            )

            captured_calls: list[dict] = []

            def _mock_train(**kwargs):
                captured_calls.append({"stage": kwargs["stage"], **kwargs["training_cfg"], "data_path": kwargs["data_path"]})
                c_dir = ws / "models" / "candidates"
                c_dir.mkdir(parents=True, exist_ok=True)
                w = c_dir / f"{kwargs['save_weight']}_{arch['hidden_size']}.pth"
                w.write_bytes(b"PTH" * 400)
                return TrainingRunResult(kwargs["stage"], True, 0, str(w), None, 0.001, False, False, False, 1.0, "ok", False)

            def _mock_merge(**kwargs):
                m = Path(kwargs["merged_out"])
                m.parent.mkdir(parents=True, exist_ok=True)
                m.write_bytes(b"MERGED" * 400)
                return m

            def _mock_export(**kwargs):
                g = Path(kwargs["out_gguf_path"])
                g.parent.mkdir(parents=True, exist_ok=True)
                g.write_bytes(b"GGUF_CAND")
                save_model_config(g.with_suffix(".config.json"), arch)
                return g

            with (
                patch("selfchange.colab.autonomous_engine.run_training_stage", side_effect=_mock_train),
                patch("selfchange.colab.autonomous_engine.promote_lora_adapter", side_effect=_mock_merge),
                patch("selfchange.colab.export_gguf.export_pth_to_single_gguf", side_effect=_mock_export),
            ):
                ctx2 = CycleContext(2, self.profile, ws, LineageNode("b0", None, "HEAD", str(champ_gguf), 1.0, 0.9, 1.0, 1.0, 0.96, "champion", "h", "t"), False, False)
                out2 = runner._run_periodic_training(ctx2, wt)
                self.assertIsNone(out2)
                self.assertEqual(captured_calls[-1]["stage"], "dpo")
                self.assertEqual(captured_calls[-1]["dpo_learning_rate"], 1e-7)
                self.assertFalse((ws / "promoted_models" / "gguf" / "candidates" / "cand_c0002-q4_k_m.gguf").exists())

                # Cycle 3 now sees rej_streak=2 -> switches back to lora with reduced LR <= 1e-5, rank=4, anchor replay mixed, no heldout leak
                ctx3 = CycleContext(3, self.profile, ws, LineageNode("b0", None, "HEAD", str(champ_gguf), 1.0, 0.9, 1.0, 1.0, 0.96, "champion", "h", "t"), False, False)
                out3 = runner._run_periodic_training(ctx3, wt)
                self.assertIsNone(out3)
                self.assertEqual(captured_calls[-1]["stage"], "lora")
                self.assertLessEqual(captured_calls[-1]["learning_rate"], 1e-5)
                self.assertEqual(captured_calls[-1]["lora_rank"], 4)
                sft_txt = Path(captured_calls[-1]["data_path"]).read_text(encoding="utf-8")
                self.assertIn("Anchor safe operational check", sft_txt)
                self.assertNotIn("Leaked heldout should be stripped", sft_txt)
                self.assertFalse((ws / "promoted_models" / "gguf" / "candidates" / "cand_c0003-q4_k_m.gguf").exists())

    def test_run_forever_daemon_chunks_log_rotation_recovery_and_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp).resolve()
            repo = tmp_p / "repo"
            repo.mkdir()

            # A: 2 chunks + log rotation (>1 MB) + DAEMON_STATUS.json
            ws_a = tmp_p / "ws_a"
            ws_a.mkdir()
            (ws_a / "daemon.log").write_bytes(b"X" * (1200 * 1024))
            cmds_a: list[list[str]] = []
            res_a = run_forever_daemon(
                repo_root=repo,
                workspace_dir=ws_a,
                max_chunks=2,
                cooldown_sec=0.0,
                max_log_mb=1.0,
                child_runner=lambda cmd, _cwd, _log: (cmds_a.append(cmd), 0)[1],
            )
            self.assertEqual(res_a["chunks_completed"], 2)
            self.assertEqual(len(cmds_a), 2)
            self.assertTrue((ws_a / "daemon.log.1").is_file())
            st_a = json.loads((ws_a / "DAEMON_STATUS.json").read_text(encoding="utf-8"))
            self.assertEqual(st_a["chunk_index"], 2)
            self.assertEqual(st_a["status"], "idle")

            # B: integrity violation halts immediately without clearing state
            ws_b = tmp_p / "ws_b"
            ws_b.mkdir()
            (ws_b / "supervisor_state.json").write_text(json.dumps({"halted_reason": "kernel_integrity_violation"}), encoding="utf-8")
            cmds_b: list[list[str]] = []
            res_b = run_forever_daemon(
                repo_root=repo,
                workspace_dir=ws_b,
                max_chunks=2,
                cooldown_sec=0.0,
                child_runner=lambda cmd, _cwd, _log: (cmds_b.append(cmd), 0)[1],
            )
            self.assertEqual(res_b["status"], "halted")
            self.assertEqual(cmds_b, [])

            # C: safe_mode_exhausted recovers with safe_mode preserved
            ws_c = tmp_p / "ws_c"
            ws_c.mkdir()
            (ws_c / "supervisor_state.json").write_text(
                json.dumps({"cycle_index": 5, "consecutive_failures": 5, "safe_mode": True, "safe_mode_failures": 2, "halted_reason": "safe_mode_exhausted"}),
                encoding="utf-8",
            )
            cmds_c: list[list[str]] = []
            res_c = run_forever_daemon(
                repo_root=repo,
                workspace_dir=ws_c,
                max_chunks=1,
                cooldown_sec=0.0,
                recovery_cooldown_sec=0.01,
                child_runner=lambda cmd, _cwd, _log: (cmds_c.append(cmd), 0)[1],
            )
            self.assertEqual(res_c["chunks_completed"], 1)
            self.assertEqual(len(cmds_c), 1)
            st_c = json.loads((ws_c / "supervisor_state.json").read_text(encoding="utf-8"))
            self.assertIsNone(st_c["halted_reason"])
            self.assertTrue(st_c["safe_mode"])
            self.assertIn("safe_mode_recovery", (ws_c / "logs" / "daemon_events.jsonl").read_text(encoding="utf-8"))

            # D: STOP file exits cleanly before running any chunk
            ws_d = tmp_p / "ws_d"
            ws_d.mkdir()
            (ws_d / "STOP").write_text("stop\n", encoding="utf-8")
            cmds_d: list[list[str]] = []
            res_d = run_forever_daemon(
                repo_root=repo,
                workspace_dir=ws_d,
                max_chunks=2,
                cooldown_sec=0.0,
                child_runner=lambda cmd, _cwd, _log: (cmds_d.append(cmd), 0)[1],
            )
            self.assertEqual(res_d["chunks_completed"], 0)
            self.assertEqual(cmds_d, [])

            # E: resource_exhausted recovers after pruning when free disk > 1024 MB
            ws_e = tmp_p / "ws_e"
            ws_e.mkdir()
            (ws_e / "supervisor_state.json").write_text(json.dumps({"halted_reason": "resource_exhausted"}), encoding="utf-8")
            cmds_e: list[list[str]] = []
            res_e = run_forever_daemon(
                repo_root=repo,
                workspace_dir=ws_e,
                max_chunks=1,
                cooldown_sec=0.0,
                child_runner=lambda cmd, _cwd, _log: (cmds_e.append(cmd), 0)[1],
            )
            self.assertEqual(res_e["chunks_completed"], 1)
            self.assertEqual(len(cmds_e), 1)
            self.assertIn("resource_recovery", (ws_e / "logs" / "daemon_events.jsonl").read_text(encoding="utf-8"))

    def test_bounded_caches_scorecard_and_tps(self) -> None:
        class _H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", "0"))
                _ = self.rfile.read(n)
                body = json.dumps({"content": "ok.\n</think>\n\nDone.", "stopped_eos": True, "timings": {"predicted_per_second": 50.0}}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                return

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
        url = f"http://127.0.0.1:{srv.server_address[1]}"
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_p = Path(tmp)
                g0 = tmp_p / "m0.gguf"
                g0.write_bytes(b"GGUF\x03\x00\x00\x00")
                sess = LlamaCppServerSession(g0, base_url=url)
                for i in range(210):
                    sess.generate_reply([{"role": "user", "content": f"q{i}"}])
                self.assertEqual(len(sess.tok_per_sec_samples), 200)

                proposer = MiniMindProposerSession(g0, self.repo_root / "model", self.profile, llama_base_url=url)
                with patch("selfchange.colab.colab_launcher.evaluate_model_strict_suite", return_value={"passed_count": 20, "total_count": 20, "overall_score": 1.0, "eos_rate": 1.0}):
                    for i in range(20):
                        gi = tmp_p / f"m{i}.gguf"
                        gi.write_bytes(b"GGUF\x03\x00\x00\x00" + bytes([i]))
                        proposer.eval_strict_scorecard(gi)
                self.assertLessEqual(len(proposer._scorecard_cache), 16)
        finally:
            srv.shutdown()
            srv.server_close()
            t.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
