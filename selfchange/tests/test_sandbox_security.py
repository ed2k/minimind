from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

from selfchange.kernel.sandbox import (
    run_sandboxed,
    scan_mutable_tree,
    scan_python_source,
)


class TestSandboxSecurity(unittest.TestCase):
    def test_ast_allows_clean_mutable_code(self) -> None:
        clean_code = """
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

@dataclass
class RecoveryStep:
    batch_size: int
    lr: float

def diagnose(log_text: str) -> dict:
    if re.search(r"out of memory", log_text, re.IGNORECASE):
        return {"batch_size": max(1, 4 // 2), "scale": math.sqrt(4.0)}
    with open("readonly.log", "r", encoding="utf-8") as f:
        data = f.read()
    return json.loads(data or "{}")
"""
        report = scan_python_source(clean_code, filename="clean.py")
        self.assertTrue(report.ok, msg=f"Unexpected violations: {report.violations}")

    def test_ast_blocks_forbidden_imports(self) -> None:
        snippets = [
            "import subprocess",
            "from socket import socket",
            "import urllib.request",
            "import ctypes",
            "import shutil",
            "import importlib",
        ]
        for snippet in snippets:
            with self.subTest(snippet=snippet):
                report = scan_python_source(snippet)
                self.assertFalse(report.ok)
                self.assertEqual(report.violations[0].rule, "forbidden_import")

    def test_ast_blocks_forbidden_calls_and_dunders(self) -> None:
        snippets = [
            ("eval('1+1')", "forbidden_call"),
            ("exec('x = 1')", "forbidden_call"),
            ("__import__('os')", "forbidden_call"),
            ("import os\nos.system('ls')", "forbidden_os_attr"),
            ("import os as _o\n_o.popen('id')", "forbidden_os_attr"),
            ("from os import system\nsystem('ls')", "forbidden_os_import"),
            ("open('a.txt', 'w')", "forbidden_open_mode"),
            ("from pathlib import Path\nPath('a').write_text('x')", "forbidden_path_write"),
            ("().__class__.__base__.__subclasses__()", "forbidden_dunder"),
        ]
        for snippet, expected_rule in snippets:
            with self.subTest(snippet=snippet):
                report = scan_python_source(snippet)
                self.assertFalse(report.ok)
                rules = {v.rule for v in report.violations}
                self.assertIn(expected_rule, rules)

    def test_ast_handles_syntax_error_cleanly(self) -> None:
        report = scan_python_source("def broken(:\n    pass\n", filename="broken.py")
        self.assertFalse(report.ok)
        self.assertEqual(report.violations[0].rule, "syntax_error")

    def test_scan_mutable_tree_avoids_pycache_and_catches_bad_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mutable_dir = Path(tmp) / "mutable"
            mutable_dir.mkdir()
            (mutable_dir / "good.py").write_text("def f(x: int) -> int:\n    return x + 1\n", encoding="utf-8")
            ok_report = scan_mutable_tree(mutable_dir, run_py_compile=True)
            self.assertTrue(ok_report.ok)
            self.assertEqual(ok_report.scanned_files, 1)
            self.assertFalse((mutable_dir / "__pycache__").exists())

            (mutable_dir / "bad.py").write_text("import socket\n", encoding="utf-8")
            bad_report = scan_mutable_tree(mutable_dir, run_py_compile=True)
            self.assertFalse(bad_report.ok)
            self.assertEqual(bad_report.scanned_files, 2)

    def test_run_sandboxed_success_and_env_scrubbing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_secret = os.environ.get("SECRET_TOKEN")
            os.environ["SECRET_TOKEN"] = "top-secret-value"
            try:
                res = run_sandboxed(
                    [
                        sys.executable,
                        "-c",
                        "import os; print(os.environ.get('SECRET_TOKEN', 'CLEAN'))",
                    ],
                    cwd=Path(tmp),
                    timeout_sec=5.0,
                )
            finally:
                if old_secret is None:
                    os.environ.pop("SECRET_TOKEN", None)
                else:
                    os.environ["SECRET_TOKEN"] = old_secret

            self.assertFalse(res.timed_out)
            self.assertEqual(res.returncode, 0, msg=res.stderr)
            self.assertEqual(res.stdout.strip(), "CLEAN")

    def test_run_sandboxed_timeout_kills_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            res = run_sandboxed(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                cwd=Path(tmp),
                timeout_sec=0.5,
            )
            self.assertTrue(res.timed_out)
            self.assertLess(res.duration_sec, 3.0)


if __name__ == "__main__":
    unittest.main()
