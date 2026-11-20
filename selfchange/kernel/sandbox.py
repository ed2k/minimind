from __future__ import annotations

import ast
import os
import py_compile
import resource
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


class SandboxPolicyError(RuntimeError):
    """Raised when mutable code violates static AST or syntax safety rules."""


@dataclass
class AstViolation:
    """Single static safety violation discovered in candidate Python code."""

    file: str
    line: int
    rule: str
    detail: str


@dataclass
class AstScanReport:
    """Aggregate static scan result across one or more Python files."""

    ok: bool
    scanned_files: int
    violations: list[AstViolation]


@dataclass
class SandboxResult:
    """Outcome of executing a command inside an isolated subprocess sandbox."""

    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_sec: float


FORBIDDEN_MODULES = {
    "subprocess",
    "socket",
    "urllib",
    "http",
    "requests",
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
    "pickle",
    "marshal",
    "importlib",
    "runpy",
    "code",
    "codeop",
    "sysconfig",
    "pip",
}

FORBIDDEN_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "breakpoint",
    "input",
}

FORBIDDEN_OS_ATTRS = {
    "system",
    "popen",
    "fork",
    "forkpty",
    "kill",
    "killpg",
    "remove",
    "unlink",
    "rmdir",
    "removedirs",
    "chmod",
    "chown",
    "symlink",
    "link",
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
}

FORBIDDEN_PATH_WRITE_ATTRS = {
    "write_text",
    "write_bytes",
    "unlink",
    "rmdir",
    "chmod",
    "symlink_to",
    "hardlink_to",
    "rename",
    "replace",
    "mkdir",
    "touch",
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
}

_WRITE_OPEN_FLAGS = ("w", "a", "x", "+")
_ENV_ALLOWLIST = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR")


def _check_open_mode(call_node: ast.Call, filename: str, violations: list[AstViolation]) -> None:
    """Permit open() only with omitted or literal read-only mode strings."""
    mode_node: ast.expr | None = None
    if len(call_node.args) >= 2:
        mode_node = call_node.args[1]
    for kw in call_node.keywords:
        if kw.arg == "mode":
            mode_node = kw.value
            break

    if mode_node is None:
        return
    if not (isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str)):
        violations.append(
            AstViolation(filename, call_node.lineno, "forbidden_open_mode", "non-literal open() mode")
        )
        return
    if any(flag in mode_node.value for flag in _WRITE_OPEN_FLAGS):
        violations.append(
            AstViolation(
                filename,
                call_node.lineno,
                "forbidden_open_mode",
                f"write-capable open() mode {mode_node.value!r}",
            )
        )


def scan_python_source(source: str, filename: str = "<string>") -> AstScanReport:
    """Statically inspect Python source against the mutable-code AST policy."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        return AstScanReport(
            ok=False,
            scanned_files=1,
            violations=[
                AstViolation(
                    file=filename,
                    line=exc.lineno or 1,
                    rule="syntax_error",
                    detail=exc.msg,
                )
            ],
        )

    violations: list[AstViolation] = []
    os_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_mod = alias.name.split(".")[0]
                if top_mod in FORBIDDEN_MODULES:
                    violations.append(
                        AstViolation(filename, node.lineno, "forbidden_import", f"import {alias.name}")
                    )
                if alias.name == "os":
                    os_aliases.add(alias.asname or "os")

        elif isinstance(node, ast.ImportFrom):
            top_mod = node.module.split(".")[0] if node.module else ""
            if top_mod in FORBIDDEN_MODULES:
                violations.append(
                    AstViolation(filename, node.lineno, "forbidden_import", f"from {node.module} import ...")
                )
            elif node.module == "os":
                for alias in node.names:
                    if alias.name == "*" or alias.name in FORBIDDEN_OS_ATTRS:
                        violations.append(
                            AstViolation(
                                filename,
                                node.lineno,
                                "forbidden_os_import",
                                f"from os import {alias.name}",
                            )
                        )

        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_DUNDERS:
                violations.append(
                    AstViolation(filename, node.lineno, "forbidden_dunder", f"access to {node.attr}")
                )
            elif isinstance(node.value, ast.Name) and node.value.id in os_aliases and node.attr in FORBIDDEN_OS_ATTRS:
                violations.append(
                    AstViolation(
                        filename,
                        node.lineno,
                        "forbidden_os_attr",
                        f"{node.value.id}.{node.attr}",
                    )
                )

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_CALLS:
                    violations.append(
                        AstViolation(filename, node.lineno, "forbidden_call", f"call to {node.func.id}()")
                    )
                elif node.func.id == "open":
                    _check_open_mode(node, filename, violations)
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in FORBIDDEN_PATH_WRITE_ATTRS:
                    violations.append(
                        AstViolation(
                            filename,
                            node.lineno,
                            "forbidden_path_write",
                            f"call to .{node.func.attr}()",
                        )
                    )
                elif node.func.attr == "open":
                    _check_open_mode(node, filename, violations)

    return AstScanReport(ok=len(violations) == 0, scanned_files=1, violations=violations)


def scan_python_file(path: Path) -> AstScanReport:
    """Read and AST-scan a single Python file."""
    path = Path(path)
    source = path.read_text(encoding="utf-8")
    return scan_python_source(source, filename=str(path))


def scan_mutable_tree(mutable_dir: Path, run_py_compile: bool = True) -> AstScanReport:
    """Scan all Python files under mutable_dir for AST safety and optional compilation."""
    mutable_dir = Path(mutable_dir)
    if not mutable_dir.exists():
        return AstScanReport(ok=True, scanned_files=0, violations=[])

    py_files = sorted(
        p for p in mutable_dir.rglob("*.py") if "__pycache__" not in p.parts and p.is_file()
    )
    all_violations: list[AstViolation] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, py_file in enumerate(py_files):
            report = scan_python_file(py_file)
            all_violations.extend(report.violations)
            if run_py_compile and not any(v.rule == "syntax_error" for v in report.violations):
                try:
                    cfile = os.path.join(tmp_dir, f"mod_{idx}.pyc")
                    py_compile.compile(str(py_file), cfile=cfile, doraise=True)
                except py_compile.PyCompileError as exc:
                    all_violations.append(
                        AstViolation(str(py_file), 1, "py_compile_error", str(exc))
                    )

    return AstScanReport(
        ok=len(all_violations) == 0,
        scanned_files=len(py_files),
        violations=all_violations,
    )


def build_scrubbed_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """Construct a minimal, proxy-blocked environment dictionary for sandboxed children."""
    env = {k: os.environ[k] for k in _ENV_ALLOWLIST if k in os.environ}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    env["NO_PROXY"] = "*"
    env["http_proxy"] = "http://127.0.0.1:0"
    env["https_proxy"] = "http://127.0.0.1:0"
    if extra_env:
        env.update(extra_env)
    return env


def _set_limit_clamped(limit_const: int, target_val: int) -> None:
    """Apply a POSIX resource limit safely clamped to the current hard limit."""
    try:
        _, hard = resource.getrlimit(limit_const)
        if hard != resource.RLIM_INFINITY:
            target_val = min(target_val, hard)
        resource.setrlimit(limit_const, (target_val, hard))
    except (ValueError, OSError):
        pass


def run_sandboxed(
    cmd: list[str],
    cwd: Path,
    timeout_sec: float = 30.0,
    memory_limit_mb: int | None = None,
    file_size_limit_mb: int | None = 64,
    extra_env: dict[str, str] | None = None,
    max_output_bytes: int = 65536,
) -> SandboxResult:
    """Execute a command in a scrubbed, resource-limited, session-isolated subprocess."""
    env = build_scrubbed_env(extra_env)

    def _apply_limits() -> None:
        cpu_limit = max(1, int(timeout_sec) + 2)
        _set_limit_clamped(resource.RLIMIT_CPU, cpu_limit)
        if file_size_limit_mb is not None:
            _set_limit_clamped(resource.RLIMIT_FSIZE, file_size_limit_mb * 1024 * 1024)
        # Apply RLIMIT_AS only on Linux; macOS rejects or breaks Python allocators under it.
        if memory_limit_mb is not None and sys.platform.startswith("linux"):
            _set_limit_clamped(resource.RLIMIT_AS, memory_limit_mb * 1024 * 1024)

    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        preexec_fn=_apply_limits,
    )

    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec)
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        returncode = proc.returncode if proc.returncode is not None else -9

    duration = round(time.monotonic() - t0, 4)
    return SandboxResult(
        returncode=returncode,
        stdout=(stdout or "")[:max_output_bytes],
        stderr=(stderr or "")[:max_output_bytes],
        timed_out=timed_out,
        duration_sec=duration,
    )
