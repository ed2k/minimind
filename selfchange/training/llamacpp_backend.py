from __future__ import annotations

import argparse
import atexit
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

_LLAMA_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=5"
_DEFAULT_FALLBACK_TAG = "b12626"
_DEFAULT_BIN_NAMES = ("llama-server", "llama-cli", "llama-quantize")


def detect_llamacpp_platform_tag() -> str:
    """Return the official ggml-org/llama.cpp prebuilt archive suffix for this OS and CPU."""
    sys_name = platform.system().lower()
    mach = platform.machine().lower()
    is_arm = mach in ("arm64", "aarch64")
    if sys_name == "darwin":
        return "macos-arm64" if is_arm else "macos-x64"
    if sys_name == "linux":
        return "ubuntu-arm64" if is_arm else "ubuntu-x64"
    raise RuntimeError(f"Unsupported OS/arch for prebuilt llama.cpp archive: {sys_name}/{mach}")


def _extract_thinking_and_content(raw_content: str) -> tuple[str, str]:
    text = raw_content or ""
    if "</think>" in text:
        before_end, after_end = text.split("</think>", 1)
        reasoning = before_end.split("<think>")[-1].lstrip("\n").rstrip("\n")
        return reasoning, after_end.lstrip("\n")
    return "", text


def render_minimind_chat_prompt(
    messages: list[dict],
    tools: list[dict] | None = None,
    open_thinking: bool = True,
) -> str:
    """Render MiniMind-3 ChatML + tool + <think> prompt in pure Python stdlib matching tokenizer_config.json."""
    norm_messages: list[dict] = []
    extracted_tools: list[dict] | None = None
    for raw in messages:
        msg = dict(raw)
        if msg.get("role") == "system" and msg.get("tools"):
            raw_tools = msg.pop("tools")
            extracted_tools = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools
        elif "tools" in msg and msg["tools"] is None:
            msg.pop("tools")
        if msg.get("tool_calls") and isinstance(msg["tool_calls"], str):
            msg["tool_calls"] = json.loads(msg["tool_calls"])
        elif "tool_calls" in msg and msg["tool_calls"] is None:
            msg.pop("tool_calls")
        if "reasoning_content" in msg and msg["reasoning_content"] is None:
            msg.pop("reasoning_content")
        norm_messages.append(msg)

    effective_tools = tools if tools is not None else extracted_tools
    parts: list[str] = []

    if effective_tools:
        parts.append("<|im_start|>system\n")
        if norm_messages and norm_messages[0].get("role") == "system":
            parts.append(str(norm_messages[0].get("content", "")) + "\n\n")
        parts.append(
            "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        )
        for tool in effective_tools:
            parts.append("\n" + json.dumps(tool))
        parts.append(
            "\n</tools>\n\nFor each function call, return a json object with function name and arguments "
            "within <tool_call></tool_call> XML tags:\n<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call><|im_end|>\n"
        )
    elif norm_messages and norm_messages[0].get("role") == "system":
        parts.append(f"<|im_start|>system\n{norm_messages[0].get('content', '')}<|im_end|>\n")

    n_msgs = len(norm_messages)
    for idx, msg in enumerate(norm_messages):
        role = msg.get("role")
        content = msg.get("content") if isinstance(msg.get("content"), str) else ""
        if role == "user" or (role == "system" and idx != 0):
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        elif role == "assistant":
            if isinstance(msg.get("reasoning_content"), str):
                reasoning = msg["reasoning_content"]
                visible = content
            else:
                reasoning, visible = _extract_thinking_and_content(content)
            parts.append(
                f"<|im_start|>assistant\n<think>\n{reasoning.strip(chr(10))}\n</think>\n\n{visible.lstrip(chr(10))}"
            )
            tcalls = msg.get("tool_calls") or []
            for t_idx, tc in enumerate(tcalls):
                if (t_idx == 0 and visible) or t_idx > 0:
                    parts.append("\n")
                fn = tc.get("function") if isinstance(tc, dict) and isinstance(tc.get("function"), dict) else tc
                fn_name = fn.get("name", "") if isinstance(fn, dict) else ""
                fn_args = fn.get("arguments", {}) if isinstance(fn, dict) else {}
                args_json = fn_args if isinstance(fn_args, str) else json.dumps(fn_args)
                parts.append(f'<tool_call>\n{{"name": "{fn_name}", "arguments": {args_json}}}\n</tool_call>')
            parts.append("<|im_end|>\n")
        elif role == "tool":
            if idx == 0 or norm_messages[idx - 1].get("role") != "tool":
                parts.append("<|im_start|>user")
            parts.append(f"\n<tool_response>\n{content}\n</tool_response>")
            if idx == n_msgs - 1 or norm_messages[idx + 1].get("role") != "tool":
                parts.append("<|im_end|>\n")

    parts.append("<|im_start|>assistant\n")
    parts.append("<think>\n" if open_thinking else "<think>\n\n</think>\n\n")
    return "".join(parts)


def restore_open_thinking_text(decoded: str, open_thinking: bool = True) -> str:
    """Restore the prompt-prefilled <think> opening tag matching MiniMind's PyTorch helper."""
    text = decoded or ""
    if open_thinking and "<think>" not in text and ("</think>" in text or text.strip()):
        return "<think>\n" + text
    return text


def find_llama_binary(name: str = "llama-server", search_dirs: list[Path] | None = None) -> Path | None:
    """Locate a llama.cpp executable across env overrides, workspace bin dirs, and PATH."""
    candidates_roots: list[Path] = []
    env_dir = os.environ.get("LLAMA_CPP_BIN_DIR")
    if env_dir:
        candidates_roots.append(Path(env_dir))
    for d in search_dirs or []:
        candidates_roots.append(Path(d))
    repo_guess = Path(__file__).resolve().parents[2]
    candidates_roots.extend(
        [
            repo_guess / "selfchange" / "workspace" / "bin" / "llama.cpp",
            Path("/usr/local/bin"),
            Path("/opt/homebrew/bin"),
        ]
    )

    for root in candidates_roots:
        if not root.exists():
            continue
        direct = root / name
        if direct.is_file() and os.access(direct, os.X_OK):
            return direct.resolve()
        try:
            for hit in sorted(root.rglob(name)):
                if hit.is_file() and os.access(hit, os.X_OK):
                    return hit.resolve()
        except OSError:
            continue

    which_hit = shutil.which(name)
    return Path(which_hit).resolve() if which_hit else None


def _resolve_latest_prebuilt_url(platform_tag: str, fallback_tag: str = _DEFAULT_FALLBACK_TAG) -> tuple[str, str]:
    target_suffix = f"-bin-{platform_tag}.tar.gz"
    req = urllib.request.Request(
        _LLAMA_RELEASES_API,
        headers={"User-Agent": "minimind-selfchange/1.0", "Accept": "application/vnd.github+json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            releases = json.loads(resp.read().decode("utf-8"))
        for rel in releases:
            tag = str(rel.get("tag_name") or "")
            for asset in rel.get("assets", []):
                aname = str(asset.get("name") or "")
                if aname.startswith("llama-") and aname.endswith(target_suffix) and "cudart-" not in aname:
                    return tag, str(asset["browser_download_url"])
    except Exception:
        pass
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{fallback_tag}/llama-{fallback_tag}-bin-{platform_tag}.tar.gz"
    return fallback_tag, url


def download_llamacpp_prebuilt(
    dest_dir: Path,
    platform_tag: str | None = None,
    fallback_tag: str = _DEFAULT_FALLBACK_TAG,
) -> dict:
    """Download and extract an official prebuilt llama.cpp archive into dest_dir (no compiler required)."""
    dest_dir = Path(dest_dir).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    tag_name = platform_tag or detect_llamacpp_platform_tag()

    existing_server = find_llama_binary("llama-server", search_dirs=[dest_dir])
    existing_quant = find_llama_binary("llama-quantize", search_dirs=[dest_dir])
    if existing_server is not None and existing_quant is not None:
        return {
            "ok": True,
            "cached": True,
            "release_tag": "cached",
            "platform_tag": tag_name,
            "bin_dir": str(existing_server.parent),
            "llama_server": str(existing_server),
            "llama_cli": str(find_llama_binary("llama-cli", search_dirs=[dest_dir]) or ""),
            "llama_quantize": str(existing_quant),
        }

    rel_tag, url = _resolve_latest_prebuilt_url(tag_name, fallback_tag=fallback_tag)
    archive_path = dest_dir / f"llama-{rel_tag}-{tag_name}.tar.gz"
    req = urllib.request.Request(url, headers={"User-Agent": "minimind-selfchange/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, archive_path.open("wb") as wf:
        shutil.copyfileobj(resp, wf)

    with tarfile.open(archive_path, "r:gz") as tf:
        tf.extractall(dest_dir, filter="data")
    archive_path.unlink(missing_ok=True)

    for p in dest_dir.rglob("*"):
        if not p.is_file():
            continue
        if p.name in _DEFAULT_BIN_NAMES or p.suffix in (".dylib", "") or ".so" in p.name:
            try:
                p.chmod(p.stat().st_mode | 0o755)
            except OSError:
                pass

    if platform.system().lower() == "darwin":
        subprocess.run(["xattr", "-dr", "com.apple.quarantine", str(dest_dir)], check=False, capture_output=True)

    srv = find_llama_binary("llama-server", search_dirs=[dest_dir])
    cli = find_llama_binary("llama-cli", search_dirs=[dest_dir])
    quant = find_llama_binary("llama-quantize", search_dirs=[dest_dir])
    return {
        "ok": srv is not None,
        "cached": False,
        "release_tag": rel_tag,
        "platform_tag": tag_name,
        "url": url,
        "bin_dir": str(srv.parent if srv else dest_dir),
        "llama_server": str(srv) if srv else None,
        "llama_cli": str(cli) if cli else None,
        "llama_quantize": str(quant) if quant else None,
    }


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class LlamaCppServerSession:
    """Stdlib-only client managing a local llama-server subprocess for GGUF inference."""

    def __init__(
        self,
        model_path: Path,
        ctx_size: int = 4096,
        threads: int | None = None,
        bin_dir: Path | None = None,
        base_url: str | None = None,
        startup_timeout_sec: float = 30.0,
    ):
        self.model_path = Path(model_path).resolve()
        self.ctx_size = int(ctx_size)
        self.threads = int(threads) if threads is not None else max(1, min(4, os.cpu_count() or 2))
        self.bin_dir = Path(bin_dir).resolve() if bin_dir is not None else None
        self.startup_timeout_sec = float(startup_timeout_sec)
        self._external_url = base_url.rstrip("/") if base_url else None
        self.base_url: str = self._external_url or ""
        self.proc: subprocess.Popen | None = None
        self._log_file = None
        self.last_tok_per_sec: float = 0.0
        self.tok_per_sec_samples: list[float] = []
        atexit.register(self.close)
        if self._external_url is None:
            self._start_server()

    def _start_server(self) -> None:
        search = [self.bin_dir] if self.bin_dir else None
        srv_bin = find_llama_binary("llama-server", search_dirs=search)
        if srv_bin is None:
            raise FileNotFoundError(
                "llama-server executable not found. Run `python3 -m selfchange.training.llamacpp_backend --install-bin` "
                "or `bash scripts/setup_lowend_intel_mac.sh` first."
            )
        if not self.model_path.is_file():
            raise FileNotFoundError(f"GGUF model file not found: {self.model_path}")

        port = _pick_free_port()
        self.base_url = f"http://127.0.0.1:{port}"
        cmd = [
            str(srv_bin),
            "-m",
            str(self.model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "-c",
            str(self.ctx_size),
            "-t",
            str(self.threads),
            "-np",
            "1",
            "--no-webui",
        ]
        env = os.environ.copy()
        lib_dir = str(srv_bin.parent)
        if platform.system().lower() == "darwin":
            env["DYLD_LIBRARY_PATH"] = lib_dir + (os.pathsep + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else "")
        else:
            env["LD_LIBRARY_PATH"] = lib_dir + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        deadline = time.monotonic() + self.startup_timeout_sec
        last_err = ""
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                err_txt = (self.proc.stderr.read() if self.proc.stderr else "")[-1500:]
                raise RuntimeError(f"llama-server exited prematurely (code={self.proc.returncode}): {err_txt}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=2.0) as resp:
                    if resp.status == 200:
                        return
            except Exception as exc:
                last_err = repr(exc)
            time.sleep(0.25)

        self.close()
        raise TimeoutError(f"Timed out waiting {self.startup_timeout_sec}s for llama-server /health ({last_err})")

    def _post_completion(self, payload: dict, max_new_tokens: int) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/completion",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=max(30.0, float(max_new_tokens) * 1.5)) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _trim_messages_for_retry(messages: list[dict]) -> list[dict]:
        if len(messages) > 3:
            kept = [dict(messages[0])] + [dict(m) for m in messages[-2:]]
        else:
            kept = [dict(m) for m in messages]
        for idx, msg in enumerate(kept):
            if idx == 0 and msg.get("role") == "system":
                continue
            txt = str(msg.get("content", ""))
            if len(txt) > 1200:
                msg["content"] = txt[:1200]
        return kept

    def generate_reply(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_new_tokens: int = 160,
        top_p: float = 0.9,
        open_thinking: bool = True,
    ) -> tuple[str, bool]:
        """Generate a completion via /completion and return (reply_text, eos_ok)."""
        def _build_payload(msgs: list[dict]) -> dict:
            prompt = render_minimind_chat_prompt(msgs, tools=tools, open_thinking=open_thinking)
            pl: dict = {
                "prompt": prompt,
                "n_predict": int(max_new_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p) if temperature > 0.0 else 1.0,
                "stop": ["<|im_end|>", "<|endoftext|>"],
                "cache_prompt": True,
            }
            if temperature <= 0.0:
                pl["top_k"] = 1
            return pl

        payload = _build_payload(messages)
        t0 = time.monotonic()
        try:
            data = self._post_completion(payload, max_new_tokens)
        except urllib.error.HTTPError as exc:
            if exc.code != 400:
                err_body = exc.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(f"llama-server HTTP {exc.code}: {err_body}") from exc
            trimmed = self._trim_messages_for_retry(messages)
            try:
                data = self._post_completion(_build_payload(trimmed), max_new_tokens)
            except urllib.error.HTTPError as retry_exc:
                err_body = retry_exc.read().decode("utf-8", errors="replace")[:500]
                raise RuntimeError(f"llama-server HTTP {retry_exc.code}: {err_body}") from retry_exc
        elapsed = max(1e-4, time.monotonic() - t0)

        timings = data.get("timings") if isinstance(data.get("timings"), dict) else {}
        tps = timings.get("predicted_per_second")
        if tps is None:
            pred_n = timings.get("predicted_n", data.get("tokens_predicted", 0))
            pred_ms = timings.get("predicted_ms")
            if pred_n and pred_ms and pred_ms > 0:
                tps = float(pred_n) / (float(pred_ms) / 1000.0)
            elif pred_n:
                tps = float(pred_n) / elapsed
        if tps is not None:
            self.last_tok_per_sec = round(float(tps), 2)
            self.tok_per_sec_samples.append(self.last_tok_per_sec)

        raw_text = str(data.get("content", ""))
        stop_word = str(data.get("stopping_word") or "")
        stop_type = str(data.get("stop_type") or "")
        eos_ok = bool(
            data.get("stopped_eos")
            or stop_type == "eos"
            or (data.get("stopped_word") and "<|im_end|>" in stop_word)
            or (stop_type == "word" and (not stop_word or "<|im_end|>" in stop_word))
            or "<|im_end|>" in raw_text
        )
        clean = raw_text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        return restore_open_thinking_text(clean, open_thinking=open_thinking), eos_ok

    def mean_tok_per_sec(self) -> float:
        if not self.tok_per_sec_samples:
            return 0.0
        return round(sum(self.tok_per_sec_samples) / len(self.tok_per_sec_samples), 2)

    def reload(self, new_model_path: Path) -> None:
        resolved = Path(new_model_path).resolve()
        if resolved == self.model_path and (self.proc is not None or self._external_url):
            return
        self.close()
        self.model_path = resolved
        if self._external_url is None:
            self._start_server()

    def close(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        finally:
            if proc.stderr is not None:
                try:
                    proc.stderr.close()
                except Exception:
                    pass

    def __enter__(self) -> "LlamaCppServerSession":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Stdlib-only llama.cpp GGUF backend for low-end Intel Mac")
    parser.add_argument("--install-bin", action="store_true", help="Download prebuilt llama.cpp archive")
    parser.add_argument("--dest", type=Path, default=Path("selfchange/workspace/bin/llama.cpp"))
    parser.add_argument("--platform-tag", type=str, default=None)
    parser.add_argument("--smoke-test", type=Path, default=None, help="Run 1-prompt smoke test on GGUF path")
    parser.add_argument("--eval-strict", type=Path, default=None, help="Run STRICT_EVAL_SUITE on GGUF path")
    parser.add_argument("--max-tasks", type=int, default=20)
    parser.add_argument("--ctx-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    out: dict = {}
    if args.install_bin:
        out["install"] = download_llamacpp_prebuilt(args.dest, platform_tag=args.platform_tag)

    target_gguf = args.smoke_test or args.eval_strict
    if target_gguf is not None:
        with LlamaCppServerSession(target_gguf, ctx_size=args.ctx_size, threads=args.threads, bin_dir=args.dest if args.dest.exists() else None) as sess:
            if args.smoke_test is not None:
                reply, eos_ok = sess.generate_reply(
                    [{"role": "user", "content": "What does LoRA freeze during fine-tuning?"}],
                    temperature=0.0,
                    max_new_tokens=96,
                )
                out["smoke_test"] = {
                    "gguf_path": str(Path(target_gguf).resolve()),
                    "file_mb": round(Path(target_gguf).stat().st_size / 1e6, 2),
                    "eos_ok": eos_ok,
                    "tok_per_sec": sess.last_tok_per_sec,
                    "reply": reply,
                }
            if args.eval_strict is not None:
                from selfchange.colab.autonomous_engine import evaluate_model_strict_suite
                from selfchange.kernel.budgets import get_profile

                prof = get_profile("local_intel_cpu")
                card = evaluate_model_strict_suite(
                    None,
                    None,
                    prof,
                    max_tasks=args.max_tasks,
                    k_rollouts=1,
                    reply_fn=lambda msgs, tls, temp: sess.generate_reply(msgs, tools=tls, temperature=temp, max_new_tokens=160),
                )
                card["mean_tok_per_sec"] = sess.mean_tok_per_sec()
                card["gguf_path"] = str(Path(target_gguf).resolve())
                card["file_mb"] = round(Path(target_gguf).stat().st_size / 1e6, 2)
                out["eval_strict"] = card

    if args.output is not None and out:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
