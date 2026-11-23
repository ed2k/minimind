from __future__ import annotations

import argparse
import ast
import builtins
import contextlib
import io
import json
import math
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
import traceback
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from selfrewrite.edit_store import EditStore
from selfrewrite.helpers import (
    clip_outliers_median,
    layer_output_matrices,
    mean_diff_direction,
    outlier_ratios,
    remove_direction,
    unit_vector,
)


class SandboxPolicyError(RuntimeError):
    """Raised when Python sandbox code violates Algorithm 4 safety constraints."""


@dataclass(frozen=True)
class AstViolation:
    line: int
    rule: str
    detail: str


FORBIDDEN_MODULES = {
    "subprocess",
    "socket",
    "urllib",
    "http",
    "requests",
    "httpx",
    "aiohttp",
    "ftplib",
    "smtplib",
    "telnetlib",
    "ssl",
    "ctypes",
    "cffi",
    "multiprocessing",
    "pty",
    "fcntl",
    "signal",
    "shutil",
    "webbrowser",
    "importlib",
    "runpy",
    "code",
    "codeop",
    "sysconfig",
    "pip",
    "builtins",
    "vllm",
    "sglang",
    "accelerate",
    "peft",
    "trl",
    "bitsandbytes",
    "inspect",
    "gc",
}

FORBIDDEN_MODULE_PREFIXES = (
    "torch.autograd",
    "torch.optim",
    "torch.cuda",
    "torch.distributed",
    "transformers.models",
    "transformers.trainer",
)

FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "breakpoint",
    "input",
    "open_unjailed",
}

FORBIDDEN_ATTRS = {
    "backward",
    "grad",
    "requires_grad",
    "requires_grad_",
    "enable_grad",
    "set_grad_enabled",
    "inference_mode",
    "autograd",
    "optim",
    "from_pretrained",
    "AutoModel",
    "AutoModelForCausalLM",
    "AutoModelForImageTextToText",
    "system",
    "popen",
    "fork",
    "forkpty",
    "kill",
    "killpg",
    "environ",
    "getenv",
    "putenv",
    "chdir",
    "spawnl",
    "spawnv",
    "spawnvp",
    "execl",
    "execv",
    "execvp",
    "subclasses",
    "mro",
}

FORBIDDEN_DUNDERS = {
    "__subclasses__",
    "__globals__",
    "__builtins__",
    "__import__",
    "__loader__",
    "__spec__",
    "__code__",
    "__closure__",
    "__mro__",
    "__bases__",
    "__dict__",
}

_SENTINEL_PREFIX = "__SELFREWRITE_IPC__:"


def scan_sandbox_code(source: str, ast_parse_fn: Any = None) -> list[AstViolation]:
    """Static AST policy scanner blocking gradients, optimizers, loaders, network, and escapes."""
    parser_fn = ast_parse_fn or ast.parse
    try:
        tree = parser_fn(source, "<sandbox>")
    except SyntaxError as exc:
        return [AstViolation(exc.lineno or 1, "syntax_error", str(exc))]

    violations: list[AstViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                root = mod.split(".", 1)[0]
                if root in FORBIDDEN_MODULES or any(mod == p or mod.startswith(p + ".") for p in FORBIDDEN_MODULE_PREFIXES):
                    violations.append(AstViolation(node.lineno, "forbidden_import", f"module {mod!r} is forbidden"))
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".", 1)[0]
            if root in FORBIDDEN_MODULES or any(mod == p or mod.startswith(p + ".") for p in FORBIDDEN_MODULE_PREFIXES):
                violations.append(AstViolation(node.lineno, "forbidden_import", f"from-import {mod!r} is forbidden"))
            for alias in node.names:
                if alias.name in FORBIDDEN_ATTRS or alias.name in FORBIDDEN_CALLS:
                    violations.append(AstViolation(node.lineno, "forbidden_import_name", f"name {alias.name!r} is forbidden"))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_CALLS:
                    violations.append(AstViolation(node.lineno, "forbidden_call", f"call {node.func.id!r} is forbidden"))
                elif node.func.id in FORBIDDEN_ATTRS:
                    violations.append(AstViolation(node.lineno, "forbidden_call", f"call {node.func.id!r} is forbidden"))
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_DUNDERS:
                violations.append(AstViolation(node.lineno, "forbidden_dunder", f"attribute {node.attr!r} is forbidden"))
            elif node.attr in FORBIDDEN_ATTRS:
                violations.append(AstViolation(node.lineno, "forbidden_attr", f"attribute {node.attr!r} is forbidden"))
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_DUNDERS:
                violations.append(AstViolation(node.lineno, "forbidden_dunder", f"symbol {node.id!r} is forbidden"))
    return violations


def _make_original_loader(weights_spec_path: Path, raw_open) -> tuple[Callable[[str], Any], dict[str, tuple[int, ...]]]:
    with raw_open(str(weights_spec_path), "r", encoding="utf-8") as f:
        spec = json.load(f)
    mode = spec.get("mode", "json_tensors")
    shapes = {k: tuple(int(d) for d in v) for k, v in spec["shapes"].items()}

    if mode == "json_tensors":
        with raw_open(spec["data_path"], "r", encoding="utf-8") as df:
            tensors = json.load(df)

        def load_original(name: str) -> Any:
            if name not in tensors:
                raise KeyError(f"Unknown weight tensor {name!r}")
            return tensors[name]

        return load_original, shapes

    if mode == "safetensors_dir":
        base_dir = Path(spec["base_dir"])
        weight_map = spec["weight_map"]

        def load_original(name: str) -> Any:
            if name not in shapes:
                raise KeyError(f"Unknown weight tensor {name!r}")
            rel = weight_map[name]
            shard_path = base_dir / rel
            from safetensors import safe_open  # type: ignore[import-not-found]

            with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
                return sf.get_tensor(name).clone()

        return load_original, shapes

    raise ValueError(f"Unsupported weights spec mode: {mode!r}")


def _install_work_jail(work_dir: Path) -> dict[str, Any]:
    """Enforce WORK/ filesystem confinement at runtime while returning raw primitives for trusted helpers."""
    work_root = work_dir.resolve()
    raw_eval = builtins.eval
    raw_exec = builtins.exec
    raw_compile_fn = builtins.compile
    raw_open = builtins.open
    raw_path_open = Path.open
    raw_read_text = Path.read_text
    raw_write_text = Path.write_text
    raw_read_bytes = Path.read_bytes
    raw_write_bytes = Path.write_bytes
    raw_iterdir = Path.iterdir
    raw_glob = Path.glob
    raw_rglob = Path.rglob
    raw_exists = Path.exists
    raw_is_file = Path.is_file
    raw_is_dir = Path.is_dir
    raw_mkdir = Path.mkdir
    raw_unlink = Path.unlink
    raw_replace = Path.replace
    raw_rename = Path.rename
    raw_touch = Path.touch
    raw_rmdir = Path.rmdir
    raw_os_listdir = os.listdir
    raw_os_scandir = os.scandir
    raw_os_open = os.open
    raw_os_stat = os.stat
    raw_os_replace = os.replace
    raw_os_rename = os.rename
    raw_os_unlink = os.unlink
    raw_os_remove = os.remove
    raw_getattr = builtins.getattr
    raw_import = builtins.__import__

    def _assert_inside_work(p: Any) -> Path:
        candidate = Path(os.fspath(p))
        if not candidate.is_absolute():
            candidate = work_root / candidate
        resolved = candidate.resolve()
        if resolved != work_root and not resolved.is_relative_to(work_root):
            raise PermissionError(f"Refused filesystem access outside WORK/: {resolved}")
        return resolved

    def _jailed_open(file, mode="r", *args, **kwargs):
        resolved = _assert_inside_work(file)
        return raw_open(resolved, mode, *args, **kwargs)

    def _jailed_path_open(self, mode="r", *args, **kwargs):
        resolved = _assert_inside_work(self)
        return raw_open(resolved, mode, *args, **kwargs)

    def _jailed_read_text(self, *args, **kwargs):
        return raw_read_text(_assert_inside_work(self), *args, **kwargs)

    def _jailed_write_text(self, *args, **kwargs):
        return raw_write_text(_assert_inside_work(self), *args, **kwargs)

    def _jailed_read_bytes(self, *args, **kwargs):
        return raw_read_bytes(_assert_inside_work(self), *args, **kwargs)

    def _jailed_write_bytes(self, *args, **kwargs):
        return raw_write_bytes(_assert_inside_work(self), *args, **kwargs)

    def _jailed_iterdir(self):
        return raw_iterdir(_assert_inside_work(self))

    def _jailed_glob(self, pattern):
        return raw_glob(_assert_inside_work(self), pattern)

    def _jailed_rglob(self, pattern):
        return raw_rglob(_assert_inside_work(self), pattern)

    def _jailed_exists(self):
        return raw_exists(_assert_inside_work(self))

    def _jailed_is_file(self):
        return raw_is_file(_assert_inside_work(self))

    def _jailed_is_dir(self):
        return raw_is_dir(_assert_inside_work(self))

    def _jailed_mkdir(self, *args, **kwargs):
        return raw_mkdir(_assert_inside_work(self), *args, **kwargs)

    def _jailed_listdir(path="."):
        return raw_os_listdir(_assert_inside_work(path))

    def _jailed_scandir(path="."):
        return raw_os_scandir(_assert_inside_work(path))

    def _jailed_os_open(path, flags, *args, **kwargs):
        return raw_os_open(_assert_inside_work(path), flags, *args, **kwargs)

    def _jailed_os_stat(path, *args, **kwargs):
        return raw_os_stat(_assert_inside_work(path), *args, **kwargs)

    def _safe_getattr(obj, name, *default):
        if name in FORBIDDEN_DUNDERS or name in FORBIDDEN_ATTRS:
            raise SandboxPolicyError(f"Dynamic access to {name!r} is forbidden")
        return raw_getattr(obj, name, *default)

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = str(name).split(".", 1)[0]
        if root in FORBIDDEN_MODULES or any(str(name) == p or str(name).startswith(p + ".") for p in FORBIDDEN_MODULE_PREFIXES):
            raise SandboxPolicyError(f"Dynamic import of {name!r} is forbidden")
        return raw_import(name, globals, locals, fromlist, level)

    def _blocked_fn(fn_name: str):
        def _raiser(*args, **kwargs):
            raise SandboxPolicyError(f"Call to {fn_name!r} is forbidden")
        return _raiser

    def _safe_compile(source, filename, mode, *args, **kwargs):
        flags = args[0] if args else kwargs.get("flags", 0)
        if int(flags) & ast.PyCF_ONLY_AST:
            return raw_compile_fn(source, filename, mode, *args, **kwargs)
        raise SandboxPolicyError("Call to 'compile' is forbidden")

    builtins.open = _jailed_open
    builtins.getattr = _safe_getattr
    builtins.__import__ = _safe_import
    builtins.eval = _blocked_fn("eval")
    builtins.exec = _blocked_fn("exec")
    builtins.compile = _safe_compile  # type: ignore[assignment]
    builtins.breakpoint = _blocked_fn("breakpoint")
    builtins.input = _blocked_fn("input")

    Path.open = _jailed_path_open  # type: ignore[method-assign]
    Path.read_text = _jailed_read_text  # type: ignore[method-assign]
    Path.write_text = _jailed_write_text  # type: ignore[method-assign]
    Path.read_bytes = _jailed_read_bytes  # type: ignore[method-assign]
    Path.write_bytes = _jailed_write_bytes  # type: ignore[method-assign]
    Path.iterdir = _jailed_iterdir  # type: ignore[method-assign]
    Path.glob = _jailed_glob  # type: ignore[method-assign]
    Path.rglob = _jailed_rglob  # type: ignore[method-assign]
    Path.exists = _jailed_exists  # type: ignore[method-assign]
    Path.is_file = _jailed_is_file  # type: ignore[method-assign]
    Path.is_dir = _jailed_is_dir  # type: ignore[method-assign]
    Path.mkdir = _jailed_mkdir  # type: ignore[method-assign]
    Path.unlink = lambda self, *a, **kw: raw_unlink(_assert_inside_work(self), *a, **kw)  # type: ignore[method-assign]
    Path.replace = lambda self, dst: raw_replace(_assert_inside_work(self), _assert_inside_work(dst))  # type: ignore[method-assign]
    Path.rename = lambda self, dst: raw_rename(_assert_inside_work(self), _assert_inside_work(dst))  # type: ignore[method-assign]
    Path.touch = lambda self, *a, **kw: raw_touch(_assert_inside_work(self), *a, **kw)  # type: ignore[method-assign]
    Path.rmdir = lambda self: raw_rmdir(_assert_inside_work(self))  # type: ignore[method-assign]

    os.listdir = _jailed_listdir
    os.scandir = _jailed_scandir
    os.open = _jailed_os_open
    os.stat = _jailed_os_stat  # type: ignore[assignment]
    os.replace = lambda src, dst: raw_os_replace(_assert_inside_work(src), _assert_inside_work(dst))
    os.rename = lambda src, dst: raw_os_rename(_assert_inside_work(src), _assert_inside_work(dst))
    os.unlink = lambda p: raw_os_unlink(_assert_inside_work(p))
    os.remove = lambda p: raw_os_remove(_assert_inside_work(p))
    for dangerous_os in ("system", "popen", "fork", "kill", "chdir"):
        if hasattr(os, dangerous_os):
            setattr(os, dangerous_os, _blocked_fn(f"os.{dangerous_os}"))

    return {
        "raw_open": raw_open,
        "work_root": work_root,
    }


class SealedPythonSandbox:
    """Host-side manager for a persistent, AST-validated, jailed CPU Python REPL worker."""

    def __init__(
        self,
        work_dir: Path,
        weights_spec_path: Path,
        timeout_sec: float = 600.0,
        max_output_chars: int = 10000,
    ):
        self.work_dir = Path(work_dir).resolve()
        self.weights_spec_path = Path(weights_spec_path).resolve()
        self.timeout_sec = float(timeout_sec)
        self.max_output_chars = int(max_output_chars)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._proc: subprocess.Popen[str] | None = None
        self._repo_root = Path(__file__).resolve().parents[1]

    def _start_worker(self) -> None:
        self.close()
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.work_dir),
            "LANG": os.environ.get("LANG", "C.UTF-8"),
            "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
            "TMPDIR": str(self.work_dir),
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": str(self._repo_root),
        }
        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "selfrewrite.sandbox",
                "--worker",
                "--work-dir",
                str(self.work_dir),
                "--weights-spec",
                str(self.weights_spec_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(self.work_dir),
            env=env,
        )

    def _read_response(self, timeout_sec: float) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdout is not None
        out_q: queue.Queue[str | None] = queue.Queue()

        def _reader():
            try:
                assert self._proc is not None and self._proc.stdout is not None
                for line in self._proc.stdout:
                    if line.startswith(_SENTINEL_PREFIX):
                        out_q.put(line[len(_SENTINEL_PREFIX):])
                        return
                out_q.put(None)
            except Exception:
                out_q.put(None)

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        try:
            payload = out_q.get(timeout=timeout_sec)
        except queue.Empty as exc:
            self.close()
            raise TimeoutError(f"Sandbox Python execution exceeded {timeout_sec:.1f}s timeout") from exc

        if payload is None:
            stderr_tail = ""
            if self._proc and self._proc.stderr:
                stderr_tail = self._proc.stderr.read()[-2000:]
            self.close()
            raise RuntimeError(f"Sandbox worker exited unexpectedly: {stderr_tail}")
        return json.loads(payload)

    def run(self, code: str) -> dict[str, Any]:
        """Validate AST safety and run code inside persistent jailed CPU worker."""
        t0 = time.monotonic()
        violations = scan_sandbox_code(code)
        if violations:
            msg = "refused: " + "; ".join(f"L{v.line} [{v.rule}] {v.detail}" for v in violations)
            return {
                "ok": False,
                "refused": True,
                "timed_out": False,
                "output": msg[: self.max_output_chars],
                "error": msg,
                "duration_sec": round(time.monotonic() - t0, 4),
            }

        if self._proc is None or self._proc.poll() is not None:
            self._start_worker()
        assert self._proc is not None and self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps({"cmd": "exec", "code": code}) + "\n")
            self._proc.stdin.flush()
            resp = self._read_response(self.timeout_sec)
        except TimeoutError as exc:
            return {
                "ok": False,
                "refused": False,
                "timed_out": True,
                "output": f"ExecutionError: {exc}",
                "error": str(exc),
                "duration_sec": round(time.monotonic() - t0, 4),
            }
        except Exception as exc:
            return {
                "ok": False,
                "refused": False,
                "timed_out": False,
                "output": f"WorkerError: {exc}",
                "error": str(exc),
                "duration_sec": round(time.monotonic() - t0, 4),
            }

        output = str(resp.get("output", ""))
        if len(output) > self.max_output_chars:
            output = output[: self.max_output_chars] + f"\n...[truncated to {self.max_output_chars} chars]"
        return {
            "ok": bool(resp.get("ok", False)),
            "refused": bool(resp.get("refused", False)),
            "timed_out": False,
            "output": output,
            "error": resp.get("error"),
            "duration_sec": round(time.monotonic() - t0, 4),
        }

    def close(self) -> None:
        if self._proc is None:
            return
        proc = self._proc
        self._proc = None
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=2.0)
        except Exception:
            pass
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=2.0)
            except Exception:
                pass
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass


def _run_worker_loop(work_dir: Path, weights_spec: Path) -> int:
    try:
        import torch  # type: ignore[import-not-found]

        torch.set_grad_enabled(False)
    except Exception:
        torch = None  # type: ignore[assignment]

    try:
        import numpy as np  # type: ignore[import-not-found]
    except Exception:
        np = None  # type: ignore[assignment]

    work_root = work_dir.resolve()
    raw_eval = builtins.eval
    raw_exec = builtins.exec
    raw_compile_fn = builtins.compile
    raw_open = builtins.open
    raw_compile = builtins.compile
    raw_exec = builtins.exec
    load_original, shapes = _make_original_loader(weights_spec.resolve(), raw_open)
    store = EditStore(work_root, load_original, shapes)

    for key in list(os.environ.keys()):
        if key not in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "CUDA_VISIBLE_DEVICES"):
            os.environ.pop(key, None)
    sys.argv = ["sandbox_worker"]

    _install_work_jail(work_root)

    def _load_census() -> list[dict[str, Any]]:
        census_file = work_root / "census.jsonl"
        if not census_file.is_file():
            return []
        return [json.loads(line) for line in census_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _load_pass(pass_id: int) -> Any:
        gen_file = work_root / f"gen_{int(pass_id)}.jsonl"
        if gen_file.is_file():
            return [json.loads(line) for line in gen_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        read_npz = work_root / f"read_{int(pass_id)}.npz"
        if read_npz.is_file() and np is not None:
            return dict(np.load(str(read_npz), allow_pickle=False))
        read_json = work_root / f"read_{int(pass_id)}.json"
        if read_json.is_file():
            return json.loads(read_json.read_text(encoding="utf-8"))
        raise FileNotFoundError(f"No pass output found for pass_id={pass_id}")

    def _load_problems() -> list[dict[str, Any]]:
        pfile = work_root / "practice_problems.jsonl"
        if not pfile.is_file():
            return []
        return [json.loads(line) for line in pfile.read_text(encoding="utf-8").splitlines() if line.strip()]

    sorted_names = sorted(shapes.keys())
    namespace: dict[str, Any] = {
        "WORK": work_root,
        "math": math,
        "json": json,
        "statistics": statistics,
        "zlib": zlib,
        "Path": Path,
        "list_tensors": lambda: list(sorted_names),
        "tensor_shapes": lambda: {k: tuple(v) for k, v in shapes.items()},
        "load_original": load_original,
        "save_edit": store.save_edit,
        "drop_edit": store.drop_edit,
        "list_edits": store.list_edits,
        "save_named_edit": store.save_named_edit,
        "load_named_edit": store.load_named_edit,
        "list_named_edits": store.list_named_edits,
        "load_census": _load_census,
        "load_pass": _load_pass,
        "load_problems": _load_problems,
        "layer_output_matrices": lambda layer_idx: layer_output_matrices(layer_idx, sorted_names),
        "unit_vector": unit_vector,
        "mean_diff_direction": mean_diff_direction,
        "remove_direction": remove_direction,
        "outlier_ratios": outlier_ratios,
        "clip_outliers_median": clip_outliers_median,
    }
    if np is not None:
        namespace["np"] = np
    if torch is not None:
        namespace["torch"] = torch

    proto_in = sys.stdin
    proto_out = sys.stdout
    def _raw_ast_parse(src: str, filename: str = "<sandbox>"):
        return raw_compile(src, filename, "exec", ast.PyCF_ONLY_AST)

    for raw_line in proto_in:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        req = json.loads(raw_line)
        if req.get("cmd") == "shutdown":
            proto_out.write(_SENTINEL_PREFIX + json.dumps({"ok": True}) + "\n")
            proto_out.flush()
            return 0
        code = str(req.get("code", ""))
        violations = scan_sandbox_code(code, ast_parse_fn=_raw_ast_parse)
        if violations:
            msg = "refused: " + "; ".join(f"L{v.line} [{v.rule}] {v.detail}" for v in violations)
            proto_out.write(_SENTINEL_PREFIX + json.dumps({"ok": False, "refused": True, "output": msg, "error": msg}) + "\n")
            proto_out.flush()
            continue

        buf = io.StringIO()
        ok = True
        refused = False
        error_text = None
        if torch is not None:
            torch.set_grad_enabled(False)
        try:
            compiled = raw_compile(code, "<sandbox>", "exec")
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                raw_exec(compiled, namespace, namespace)
        except SandboxPolicyError as exc:
            ok = False
            refused = True
            error_text = f"refused: {exc}"
            buf.write(f"\n{error_text}\n")
        except Exception:
            ok = False
            tb = traceback.format_exc()
            error_text = tb.strip().splitlines()[-1] if tb.strip() else "ExecutionError"
            buf.write(f"\n{tb}")
        proto_out.write(
            _SENTINEL_PREFIX + json.dumps({"ok": ok, "refused": refused, "output": buf.getvalue(), "error": error_text}) + "\n"
        )
        proto_out.flush()
    return 0


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Sealed CPU Python sandbox worker")
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--weights-spec", required=True)
    args = parser.parse_args()
    return _run_worker_loop(Path(args.work_dir), Path(args.weights_spec))


if __name__ == "__main__":
    raise SystemExit(_cli())
