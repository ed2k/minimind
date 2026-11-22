from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from selfchange.colab.colab_launcher import launch_colab_session, resolve_model_checkpoint
from selfchange.colab.run_colab_cli import run_colab_cli_pipeline
from selfchange.kernel.budgets import get_profile
from selfchange.kernel.integrity import KERNEL_MANIFEST_FILENAME
from selfchange.training.llamacpp_backend import (
    LlamaCppServerSession,
    find_llama_binary,
    render_minimind_chat_prompt,
    restore_open_thinking_text,
)
from selfchange.training.model_loader import estimate_minimind_params, save_model_config


def _start_mock_llama_server(handler_fn) -> tuple[ThreadingHTTPServer, str, threading.Thread]:
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                body = b'{"status":"ok"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            req = json.loads(raw)
            resp_obj = handler_fn(req)
            body = json.dumps(resp_obj).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            return

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{port}", t


class TestLowEndIntelMacGGUF(unittest.TestCase):
    def test_render_minimind_chat_prompt_matches_chatml_and_tools(self) -> None:
        tools = [{"type": "function", "function": {"name": "patch_config", "parameters": {"type": "object"}}}]
        msgs = [
            {"role": "system", "content": "Operate safely.", "tools": json.dumps(tools)},
            {"role": "user", "content": "fault=oom batch_size=8"},
            {
                "role": "assistant",
                "content": "<think>\nSaw batch_size=8.\n</think>\n\nCalling tool.",
                "tool_calls": json.dumps([{"name": "patch_config", "arguments": {"updates": {"batch_size": 4}}}]),
            },
            {"role": "tool", "content": '{"ok": true}'},
        ]
        prompt = render_minimind_chat_prompt(msgs, open_thinking=True)
        self.assertIn("<|im_start|>system\nOperate safely.\n\n# Tools", prompt)
        self.assertIn("<tools>\n", prompt)
        self.assertIn("<|im_start|>user\nfault=oom batch_size=8<|im_end|>\n", prompt)
        self.assertIn('<tool_call>\n{"name": "patch_config", "arguments": {"updates": {"batch_size": 4}}}\n</tool_call><|im_end|>\n', prompt)
        self.assertIn("<|im_start|>user\n<tool_response>\n{\"ok\": true}\n</tool_response><|im_end|>\n", prompt)
        self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n"))

        closed = render_minimind_chat_prompt([{"role": "user", "content": "hi"}], open_thinking=False)
        self.assertTrue(closed.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
        self.assertEqual(restore_open_thinking_text("Reason.\n</think>\nOK", True), "<think>\nReason.\n</think>\nOK")

    def test_llamacpp_server_session_mock_and_tok_s(self) -> None:
        captured: list[dict] = []

        def _handle(req: dict) -> dict:
            captured.append(req)
            return {
                "content": "Observe LoRA freezes base weights.\n</think>\n\nLoRA freezes base weights.",
                "stopped_eos": True,
                "timings": {"predicted_per_second": 72.5},
            }

        srv, url, thread = _start_mock_llama_server(_handle)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                gguf = Path(tmp) / "test.gguf"
                gguf.write_bytes(b"GGUF\x03\x00\x00\x00MOCK")
                with LlamaCppServerSession(gguf, base_url=url) as sess:
                    reply, eos_ok = sess.generate_reply([{"role": "user", "content": "Explain LoRA"}], temperature=0.0, max_new_tokens=64)
                    self.assertTrue(eos_ok)
                    self.assertTrue(reply.startswith("<think>\n"))
                    self.assertEqual(sess.mean_tok_per_sec(), 72.5)
                    self.assertEqual(captured[0]["top_k"], 1)
                    self.assertIn("<|im_end|>", captured[0]["stop"])
        finally:
            srv.shutdown()
            srv.server_close()
            thread.join(timeout=2)

    def test_resolve_gguf_without_torch_and_launch_session(self) -> None:
        def _handle(req: dict) -> dict:
            p = req.get("prompt", "")
            if "get_system_state" in p and "submit_candidate" in p and "Observed" not in p:
                txt = (
                    "Observe workspace status from context.\n</think>\n\n"
                    '<tool_call>\n{"name": "get_system_state", "arguments": {}}\n</tool_call>'
                )
            elif "<tool_response>" in p and "patch_config" in p and '"patched"' not in p:
                txt = (
                    "Apply minimal config update.\n</think>\n\n"
                    '<tool_call>\n{"name": "patch_config", "arguments": {"updates": {"temperature": 0.15}}}\n</tool_call>'
                )
            elif "batch_size=4" in p and "accumulation_steps=1" in p:
                txt = "Batch=4->2->1, accum=1->2->4.\n</think>\n\nFinal batch_size=1, accumulation_steps=4."
            elif "cap(" in p or "clamp" in p or "exp_bound_val" in p:
                txt = (
                    "Fix max to min on line 4.\n</think>\n\n"
                    '<tool_call>\n{"name": "replace_line_range", "arguments": {"path": "x.py", "start_line": 4, "end_line": 4, "lines": ["    return min(val, hi)"]}}\n</tool_call>'
                )
            elif "status=healthy" in p:
                txt = "Healthy run, no edits required.\n</think>\n\nSystem is healthy. No edits required."
            elif "submit_candidate" in p:
                txt = (
                    "Submitting verified update.\n</think>\n\n"
                    '<tool_call>\n{"name": "submit_candidate", "arguments": {"hypothesis": "gguf local test"}}\n</tool_call>'
                )
            else:
                txt = (
                    "Checked context carefully.\n</think>\n\nVerified response."
                )
            return {"content": txt, "stopped_eos": True, "timings": {"predicted_per_second": 64.0}}

        srv, url, thread = _start_mock_llama_server(_handle)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_p = Path(tmp).resolve()
                real_repo = Path(__file__).resolve().parents[2]
                kdir = tmp_p / "kernel"
                shutil.copytree(real_repo / "selfchange" / "kernel", kdir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

                repo = tmp_p / "repo"
                repo.mkdir(parents=True)
                subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
                subprocess.run(["git", "config", "user.name", "test-user"], cwd=repo, check=True, capture_output=True, text=True)
                subprocess.run(["git", "config", "user.email", "test@local"], cwd=repo, check=True, capture_output=True, text=True)
                shutil.copytree(
                    real_repo / "selfchange" / "mutable",
                    repo / "selfchange" / "mutable",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
                subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
                subprocess.run(["git", "commit", "-m", "init"], cwd=repo, check=True, capture_output=True, text=True)

                ws = tmp_p / "ws"
                gguf_dir = ws / "promoted_models" / "gguf"
                gguf_dir.mkdir(parents=True)
                champ_gguf = gguf_dir / "champion.gguf"
                champ_gguf.write_bytes(b"GGUF\x03\x00\x00\x00TEST")
                save_model_config(champ_gguf.with_suffix(".config.json"), estimate_minimind_params(768, 8, 1216))
                (gguf_dir / "latest_gguf.json").write_text(json.dumps({"champion_gguf": str(champ_gguf)}), encoding="utf-8")

                prof = get_profile("local_intel_cpu")
                with patch("selfchange.colab.colab_launcher._torch_available", return_value=False):
                    resolved, source = resolve_model_checkpoint(repo, ws, prof)
                    self.assertEqual(resolved, champ_gguf.resolve())
                    self.assertEqual(source, "gguf_latest")

                    out = launch_colab_session(
                        repo_root=repo,
                        workspace_dir=ws,
                        profile_name="local_intel_cpu",
                        max_cycles=1,
                        eval_with_model=True,
                        require_model=True,
                        kernel_dir=kdir,
                        manifest_path=kdir / KERNEL_MANIFEST_FILENAME,
                        llama_base_url=url,
                    )
                self.assertEqual(out["backend"], "llamacpp_gguf")
                self.assertEqual(out["proposer_mode"], "minimind")
                self.assertTrue((ws / "live_scorecard_start.json").is_file())
                self.assertTrue((ws / "trajectories.jsonl").is_file())
                self.assertTrue((ws / "report.md").is_file())
        finally:
            srv.shutdown()
            srv.server_close()
            thread.join(timeout=2)

    def test_run_colab_cli_export_gguf_and_sync_train_gguf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_p = Path(tmp)
            ws = tmp_p / "ws"
            prom = ws / "promoted_models" / "r1"
            prom.mkdir(parents=True)
            w = prom / "minimal_cpu_champion.pth"
            w.write_bytes(b"PTHDATA")
            save_model_config(w.with_suffix(".config.json"), estimate_minimind_params(768, 8, 1216))
            (ws / "promoted_models" / "latest_champion.json").write_text(
                json.dumps({"run_id": "r1", "primary_weight": str(w), "weights": [str(w)]}),
                encoding="utf-8",
            )
            (ws / "trajectories.jsonl").write_text('{"task_id":"t1","passed":true}\n', encoding="utf-8")

            cmds: list[list[str]] = []

            def _mock(cmd, **_kwargs):
                cmds.append(cmd)
                if len(cmd) >= 5 and cmd[:2] == ["colab", "download"] and cmd[4] == "/content/run_gguf_artifacts.tar.gz":
                    out_tar = Path(cmd[5])
                    out_tar.parent.mkdir(parents=True, exist_ok=True)
                    stub = tmp_p / "gguf_stub"
                    stub.mkdir(exist_ok=True)
                    (stub / "champion.gguf").write_bytes(b"GGUF\x03\x00\x00\x00QUANT")
                    (stub / "latest_gguf.json").write_text(
                        json.dumps(
                            {
                                "champion_gguf": "/content/minimind/selfchange/workspace/promoted_models/gguf/champion.gguf",
                                "selected_artifact": {"path": "/content/minimind/selfchange/workspace/promoted_models/gguf/champion.gguf", "size_mb": 22.5},
                                "reports": [],
                            }
                        ),
                        encoding="utf-8",
                    )
                    with tarfile.open(out_tar, "w:gz") as tf:
                        tf.add(stub / "champion.gguf", arcname="champion.gguf")
                        tf.add(stub / "latest_gguf.json", arcname="latest_gguf.json")
                elif len(cmd) >= 5 and cmd[:2] == ["colab", "download"] and cmd[4] == "/content/run_diagnostics.tar.gz":
                    out_tar = Path(cmd[5])
                    out_tar.parent.mkdir(parents=True, exist_ok=True)
                    stub_d = tmp_p / "diag_stub"
                    stub_d.mkdir(exist_ok=True)
                    (stub_d / "session_info.json").write_text(
                        json.dumps({"persistence_decision": {"improved": False, "persist_weights": False, "reason": "no_improvement"}}),
                        encoding="utf-8",
                    )
                    with tarfile.open(out_tar, "w:gz") as tf:
                        tf.add(stub_d / "session_info.json", arcname="session_info.json")
                return type("Proc", (), {"returncode": 0})()

            res_exp = run_colab_cli_pipeline(
                session="gguf-exp",
                workspace_dir=ws,
                mode="export_gguf",
                export_pth=str(w),
                gguf_quants="Q8_0,Q4_K_M",
                run_cmd=_mock,
            )
            self.assertIsNone(res_exp["error"])
            self.assertIn("gguf", res_exp)
            local_gguf = ws / "promoted_models" / "gguf" / "champion.gguf"
            self.assertTrue(local_gguf.is_file())
            latest_meta = json.loads((ws / "promoted_models" / "gguf" / "latest_gguf.json").read_text(encoding="utf-8"))
            self.assertEqual(latest_meta["champion_gguf"], str(local_gguf.resolve()))
            self.assertEqual(cmds[-1], ["colab", "stop", "-s", "gguf-exp"])

            cmds.clear()
            res_sync = run_colab_cli_pipeline(
                session="gguf-sync",
                workspace_dir=ws,
                mode="sync_train_gguf",
                resume=True,
                run_cmd=_mock,
            )
            self.assertIsNone(res_sync["error"])
            uploads = [c[5] for c in cmds if c[:2] == ["colab", "upload"]]
            self.assertTrue(any("trajectories.jsonl" in u for u in uploads))
            exec_scripts = [Path(c[-1]).name for c in cmds if c[:2] == ["colab", "exec"] and c[-2] == "-f"]
            self.assertEqual(exec_scripts[-2:], ["step3_autonomous.py", "step4_export_gguf.py"])
            self.assertEqual(cmds[-1], ["colab", "stop", "-s", "gguf-sync"])

    def test_find_llama_binary_nested_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "llama-b12626" / "bin"
            nested.mkdir(parents=True)
            fake_bin = nested / "llama-server"
            fake_bin.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
            fake_bin.chmod(0o755)
            found = find_llama_binary("llama-server", search_dirs=[Path(tmp)])
            self.assertEqual(found, fake_bin.resolve())


if __name__ == "__main__":
    unittest.main()
