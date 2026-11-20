from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

KERNEL_MANIFEST_FILENAME = "kernel_manifest.sha256"
ALLOWED_MUTABLE_PREFIX = "selfchange/mutable/"
_HEX_CHARS = set("0123456789abcdefABCDEF")


@dataclass
class IntegrityReport:
    """Result of verifying kernel Python files against their pinned SHA-256 manifest."""

    ok: bool
    mismatched: list[str]
    missing: list[str]
    unexpected: list[str]
    error: str | None = None


@dataclass
class DiffBoundaryReport:
    """Result of checking changed paths against the allowed mutable directory."""

    ok: bool
    allowed: list[str]
    violations: list[str]
    error: str | None = None


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write text to a file via a sibling temporary file and os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def compute_file_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file using 64KB streaming reads."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def collect_kernel_files(kernel_dir: Path) -> list[Path]:
    """Collect all Python files under kernel_dir, ignoring __pycache__, sorted by POSIX relpath."""
    kernel_dir = Path(kernel_dir).resolve()
    files = [
        p
        for p in kernel_dir.rglob("*.py")
        if "__pycache__" not in p.parts and p.is_file()
    ]
    return sorted(files, key=lambda p: p.relative_to(kernel_dir).as_posix())


def generate_manifest(
    kernel_dir: Path,
    manifest_path: Path | None = None,
    write: bool = True,
) -> dict[str, str]:
    """Compute (and optionally atomically write) SHA-256 manifest for kernel Python files."""
    kernel_dir = Path(kernel_dir).resolve()
    target = Path(manifest_path) if manifest_path is not None else kernel_dir / KERNEL_MANIFEST_FILENAME
    entries: dict[str, str] = {}
    for file_path in collect_kernel_files(kernel_dir):
        rel = file_path.relative_to(kernel_dir).as_posix()
        entries[rel] = compute_file_sha256(file_path)

    if write:
        lines = [f"{digest}  {rel}\n" for rel, digest in entries.items()]
        atomic_write_text(target, "".join(lines))
    return entries


def load_manifest(manifest_path: Path) -> dict[str, str]:
    """Load and validate a SHA-256 manifest file."""
    manifest_path = Path(manifest_path)
    entries: dict[str, str] = {}
    for lineno, raw_line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Malformed manifest line {lineno} in {manifest_path}: {raw_line!r}")
        digest, rel = parts[0], parts[1].strip()
        if len(digest) != 64 or not all(ch in _HEX_CHARS for ch in digest):
            raise ValueError(f"Invalid SHA-256 digest on line {lineno} in {manifest_path}: {digest!r}")
        entries[rel] = digest.lower()
    return entries


def verify_kernel_integrity(
    kernel_dir: Path,
    manifest_path: Path | None = None,
) -> IntegrityReport:
    """Verify kernel Python files match the pinned SHA-256 manifest with no extras or gaps."""
    kernel_dir = Path(kernel_dir).resolve()
    target = Path(manifest_path) if manifest_path is not None else kernel_dir / KERNEL_MANIFEST_FILENAME
    if not target.is_file():
        return IntegrityReport(
            ok=False,
            mismatched=[],
            missing=[],
            unexpected=[],
            error=f"Kernel manifest not found: {target}",
        )

    try:
        expected = load_manifest(target)
    except Exception as exc:
        return IntegrityReport(
            ok=False,
            mismatched=[],
            missing=[],
            unexpected=[],
            error=f"Failed to load manifest: {exc}",
        )

    actual_files = {
        p.relative_to(kernel_dir).as_posix(): p
        for p in collect_kernel_files(kernel_dir)
    }

    missing: list[str] = []
    mismatched: list[str] = []
    for rel, expected_digest in sorted(expected.items()):
        actual_path = actual_files.get(rel)
        if actual_path is None:
            missing.append(rel)
        elif compute_file_sha256(actual_path).lower() != expected_digest:
            mismatched.append(rel)

    unexpected = sorted(rel for rel in actual_files if rel not in expected)
    ok = not (missing or mismatched or unexpected)
    return IntegrityReport(
        ok=ok,
        mismatched=mismatched,
        missing=missing,
        unexpected=unexpected,
    )


def validate_changed_paths(
    changed_paths: Iterable[str],
    allowed_prefix: str = ALLOWED_MUTABLE_PREFIX,
) -> DiffBoundaryReport:
    """Validate that all changed paths are strictly relative to allowed_prefix without traversal."""
    allowed: list[str] = []
    violations: list[str] = []
    for raw in changed_paths:
        norm = raw.strip().replace("\\", "/")
        if not norm:
            continue
        is_abs = norm.startswith("/") or (len(norm) >= 2 and norm[1] == ":")
        has_traversal = ".." in norm.split("/")
        if is_abs or has_traversal or not norm.startswith(allowed_prefix):
            violations.append(norm)
        else:
            allowed.append(norm)
    return DiffBoundaryReport(ok=len(violations) == 0, allowed=allowed, violations=violations)


def check_git_diff_boundary(
    repo_dir: Path,
    base_ref: str = "HEAD",
    allowed_prefix: str = ALLOWED_MUTABLE_PREFIX,
) -> DiffBoundaryReport:
    """Inspect tracked diffs and untracked files in a git repo against allowed_prefix."""
    repo_dir = Path(repo_dir).resolve()
    try:
        diff_proc = subprocess.run(
            ["git", "diff", "--name-only", base_ref],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
        untracked_proc = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        return DiffBoundaryReport(
            ok=False,
            allowed=[],
            violations=[],
            error=f"git diff inspection failed: {exc.stderr.strip() or exc}",
        )

    paths = sorted(
        {
            line.strip()
            for out in (diff_proc.stdout, untracked_proc.stdout)
            for line in out.splitlines()
            if line.strip()
        }
    )
    return validate_changed_paths(paths, allowed_prefix=allowed_prefix)
