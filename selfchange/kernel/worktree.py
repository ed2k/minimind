from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from selfchange.kernel.integrity import ALLOWED_MUTABLE_PREFIX, check_git_diff_boundary
from selfchange.kernel.sandbox import SandboxPolicyError, scan_mutable_tree

_CANDIDATE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class BoundaryViolationError(RuntimeError):
    """Raised when a candidate patch modifies files outside selfchange/mutable/."""


class WorktreeError(RuntimeError):
    """Raised when git worktree creation, commit, or cleanup fails."""


@dataclass
class WorktreeHandle:
    """Metadata for an isolated candidate git worktree."""

    candidate_id: str
    path: Path
    branch: str
    base_commit: str
    repo_root: Path


class WorktreeManager:
    """Manages ephemeral git worktree sandboxes for candidate self-modifications."""

    def __init__(self, repo_root: Path, worktrees_root: Path):
        self.repo_root = Path(repo_root).resolve()
        self.worktrees_root = Path(worktrees_root).resolve()

    def _git(self, args: list[str], cwd: Path | None = None) -> str:
        target = self.repo_root if cwd is None else cwd
        proc = subprocess.run(
            ["git", *args],
            cwd=target,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise WorktreeError(
                f"git {' '.join(args)} failed in {target}: {proc.stderr.strip() or proc.stdout.strip()}"
            )
        return proc.stdout.strip()

    def resolve_commit(self, ref: str = "HEAD") -> str:
        """Resolve a git ref to its full 40-char commit SHA."""
        return self._git(["rev-parse", ref])

    def create_worktree(self, candidate_id: str, base_ref: str = "HEAD") -> WorktreeHandle:
        """Create an isolated git worktree branched from base_ref for a candidate."""
        if not _CANDIDATE_ID_RE.fullmatch(candidate_id):
            raise ValueError(f"Invalid candidate_id {candidate_id!r}")
        base_commit = self.resolve_commit(base_ref)
        branch = f"selfchange/cand/{candidate_id}"
        self.worktrees_root.mkdir(parents=True, exist_ok=True)
        wt_path = (self.worktrees_root / candidate_id).resolve()
        self._git(["worktree", "add", "-b", branch, str(wt_path), base_commit])
        return WorktreeHandle(
            candidate_id=candidate_id,
            path=wt_path,
            branch=branch,
            base_commit=base_commit,
            repo_root=self.repo_root,
        )

    def commit_mutable_changes(
        self,
        handle: WorktreeHandle,
        message: str,
        allowed_prefix: str = ALLOWED_MUTABLE_PREFIX,
    ) -> str:
        """Gate diff boundary and AST safety, then commit mutable changes inside the worktree."""
        diff_report = check_git_diff_boundary(
            handle.path, base_ref=handle.base_commit, allowed_prefix=allowed_prefix
        )
        if diff_report.error:
            raise WorktreeError(diff_report.error)
        if not diff_report.ok:
            raise BoundaryViolationError(
                f"Candidate modified paths outside {allowed_prefix}: {diff_report.violations}"
            )
        if not diff_report.allowed:
            raise WorktreeError("No mutable changes to commit")

        mutable_dir = handle.path / allowed_prefix.rstrip("/")
        if mutable_dir.is_dir():
            scan_report = scan_mutable_tree(mutable_dir, run_py_compile=True)
            if not scan_report.ok:
                summary = "; ".join(
                    f"{v.file}:{v.line} [{v.rule}] {v.detail}" for v in scan_report.violations
                )
                raise SandboxPolicyError(f"AST/compile gate failed: {summary}")

        self._git(["add", "-A", "--", allowed_prefix.rstrip("/")], cwd=handle.path)
        self._git(
            [
                "-c",
                "user.name=selfchange",
                "-c",
                "user.email=selfchange@local",
                "commit",
                "-m",
                message,
            ],
            cwd=handle.path,
        )
        return self._git(["rev-parse", "HEAD"], cwd=handle.path)

    def remove_worktree(self, handle: WorktreeHandle, delete_branch: bool = True) -> None:
        """Remove a candidate worktree directory and optionally delete its temporary branch."""
        proc = subprocess.run(
            ["git", "worktree", "remove", "--force", str(handle.path)],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0 or handle.path.exists():
            shutil.rmtree(handle.path, ignore_errors=True)
            subprocess.run(
                ["git", "worktree", "prune"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
            )
        if delete_branch:
            subprocess.run(
                ["git", "branch", "-D", handle.branch],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
            )

    def prune_stale_worktrees(self, keep_ids: set[str] | None = None) -> list[str]:
        """Delete stale worktree directories under worktrees_root not listed in keep_ids."""
        keep = keep_ids or set()
        removed: list[str] = []
        if self.worktrees_root.is_dir():
            for child in sorted(self.worktrees_root.iterdir()):
                if child.name in keep:
                    continue
                if child.is_dir():
                    proc = subprocess.run(
                        ["git", "worktree", "remove", "--force", str(child)],
                        cwd=self.repo_root,
                        capture_output=True,
                        text=True,
                    )
                    if proc.returncode != 0 or child.exists():
                        shutil.rmtree(child, ignore_errors=True)
                    removed.append(child.name)
                elif child.is_file():
                    child.unlink(missing_ok=True)
                    removed.append(child.name)
        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=self.repo_root,
            capture_output=True,
            text=True,
        )
        return removed
