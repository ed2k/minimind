from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from selfchange.kernel.integrity import (
    KERNEL_MANIFEST_FILENAME,
    check_git_diff_boundary,
    validate_changed_paths,
    verify_kernel_integrity,
)


def _init_temp_git_repo(root: Path) -> None:
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.name", "test-user"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@local"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    (root / "trainer").mkdir(parents=True, exist_ok=True)
    (root / "trainer" / "train_lora.py").write_text("# base\n", encoding="utf-8")
    (root / "selfchange" / "mutable").mkdir(parents=True, exist_ok=True)
    (root / "selfchange" / "mutable" / "policy.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )


class TestKernelIntegrity(unittest.TestCase):
    def setUp(self) -> None:
        self.live_kernel_dir = Path(__file__).resolve().parents[1] / "kernel"

    def _copy_kernel(self, dest: Path) -> Path:
        target = dest / "kernel"
        shutil.copytree(
            self.live_kernel_dir,
            target,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        return target

    def test_live_kernel_manifest_is_valid(self) -> None:
        report = verify_kernel_integrity(self.live_kernel_dir)
        self.assertTrue(report.ok, msg=f"Unexpected integrity failure: {report}")
        self.assertEqual(report.mismatched, [])
        self.assertEqual(report.missing, [])
        self.assertEqual(report.unexpected, [])

    def test_tampered_kernel_file_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kdir = self._copy_kernel(Path(tmp))
            budgets_path = kdir / "budgets.py"
            budgets_path.write_text(budgets_path.read_text(encoding="utf-8") + "\n# tamper\n", encoding="utf-8")
            report = verify_kernel_integrity(kdir)
            self.assertFalse(report.ok)
            self.assertIn("budgets.py", report.mismatched)

    def test_unexpected_kernel_file_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kdir = self._copy_kernel(Path(tmp))
            (kdir / "evil.py").write_text("x = 1\n", encoding="utf-8")
            report = verify_kernel_integrity(kdir)
            self.assertFalse(report.ok)
            self.assertIn("evil.py", report.unexpected)

    def test_missing_kernel_file_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kdir = self._copy_kernel(Path(tmp))
            (kdir / "sandbox.py").unlink()
            report = verify_kernel_integrity(kdir)
            self.assertFalse(report.ok)
            self.assertIn("sandbox.py", report.missing)

    def test_missing_manifest_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kdir = self._copy_kernel(Path(tmp))
            (kdir / KERNEL_MANIFEST_FILENAME).unlink()
            report = verify_kernel_integrity(kdir)
            self.assertFalse(report.ok)
            self.assertIsNotNone(report.error)

    def test_diff_boundary_allows_mutable_only(self) -> None:
        allowed_report = validate_changed_paths(
            [
                "selfchange/mutable/config.json",
                "selfchange/mutable/recovery_policy.py",
            ]
        )
        self.assertTrue(allowed_report.ok)
        self.assertEqual(len(allowed_report.allowed), 2)
        self.assertEqual(allowed_report.violations, [])

        bad_paths = [
            "trainer/train_lora.py",
            "selfchange/kernel/supervisor.py",
            "selfchange/mutable/../kernel/supervisor.py",
            "/etc/passwd",
        ]
        bad_report = validate_changed_paths(bad_paths)
        self.assertFalse(bad_report.ok)
        self.assertEqual(bad_report.allowed, [])
        self.assertEqual(len(bad_report.violations), len(bad_paths))

    def test_git_diff_boundary_rejects_tracked_and_untracked_outside_mutable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            repo.mkdir()
            _init_temp_git_repo(repo)

            (repo / "selfchange" / "mutable" / "policy.py").write_text("VALUE = 2\n", encoding="utf-8")
            ok_report = check_git_diff_boundary(repo)
            self.assertTrue(ok_report.ok)
            self.assertEqual(ok_report.allowed, ["selfchange/mutable/policy.py"])

            (repo / "trainer" / "train_lora.py").write_text("# modified\n", encoding="utf-8")
            (repo / "evil.py").write_text("# untracked\n", encoding="utf-8")
            bad_report = check_git_diff_boundary(repo)
            self.assertFalse(bad_report.ok)
            self.assertIn("trainer/train_lora.py", bad_report.violations)
            self.assertIn("evil.py", bad_report.violations)


if __name__ == "__main__":
    unittest.main()
