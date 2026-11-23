from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from selfrewrite.config import atomic_write_text, compute_file_sha256


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DriveMirrorSync:
    """Atomic SHA-256 verified Google Drive mirror for raw_sessions and distill_bundle."""

    def __init__(self, local_workspace: Path, drive_mirror_dir: Path | None):
        self.local_workspace = Path(local_workspace).resolve()
        self.drive_mirror_dir = Path(drive_mirror_dir).resolve() if drive_mirror_dir else None
        self.manifest_path = (self.drive_mirror_dir / "drive_manifest.json") if self.drive_mirror_dir else None

    @property
    def enabled(self) -> bool:
        return self.drive_mirror_dir is not None

    def _copy_atomic(self, src: Path, dst: Path) -> str:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{dst.name}.tmp")
        try:
            shutil.copy2(src, tmp)
            tmp.replace(dst)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        return compute_file_sha256(dst)

    def _mirror_tree(self, rel_root: str) -> dict[str, str]:
        if not self.enabled:
            return {}
        assert self.drive_mirror_dir is not None
        src_root = self.local_workspace / rel_root
        if not src_root.exists():
            return {}
        hashes: dict[str, str] = {}
        for item in sorted(src_root.rglob("*")):
            if not item.is_file() or "__pycache__" in item.parts:
                continue
            rel = item.relative_to(self.local_workspace).as_posix()
            dst = self.drive_mirror_dir / rel
            if dst.is_file() and dst.stat().st_size == item.stat().st_size:
                if compute_file_sha256(dst) == compute_file_sha256(item):
                    hashes[rel] = compute_file_sha256(dst)
                    continue
            hashes[rel] = self._copy_atomic(item, dst)
        return hashes

    def sync_all(self) -> dict[str, Any]:
        """Mirror HEARTBEAT, reports, raw_sessions/, and distill_bundle/ atomically to Drive."""
        if not self.enabled or self.drive_mirror_dir is None:
            return {"enabled": False, "synced_files": 0}
        self.drive_mirror_dir.mkdir(parents=True, exist_ok=True)
        all_hashes: dict[str, str] = {}
        for fname in ("HEARTBEAT.json", "report.md", "leaderboard.json"):
            src = self.local_workspace / fname
            if src.is_file():
                all_hashes[fname] = self._copy_atomic(src, self.drive_mirror_dir / fname)
        for subtree in ("splits", "raw_sessions", "distill_bundle"):
            all_hashes.update(self._mirror_tree(subtree))

        manifest = {
            "updated_at": _utc_now(),
            "synced_files": len(all_hashes),
            "sha256": all_hashes,
        }
        assert self.manifest_path is not None
        atomic_write_text(self.manifest_path, json.dumps(manifest, indent=2) + "\n")
        return {"enabled": True, **manifest}

    def restore_from_drive(self) -> dict[str, Any]:
        """Restore interrupted raw_sessions, splits, and distill_bundle from Drive with SHA checks."""
        if not self.enabled or self.drive_mirror_dir is None or not self.drive_mirror_dir.is_dir():
            return {"restored": False, "reason": "no_drive_mirror"}
        expected_sha: dict[str, str] = {}
        if self.manifest_path and self.manifest_path.is_file():
            try:
                expected_sha = dict(json.loads(self.manifest_path.read_text(encoding="utf-8")).get("sha256", {}))
            except Exception:
                expected_sha = {}

        restored_files = 0
        verified = 0
        for item in sorted(self.drive_mirror_dir.rglob("*")):
            if not item.is_file() or item.name == "drive_manifest.json":
                continue
            rel = item.relative_to(self.drive_mirror_dir).as_posix()
            exp = expected_sha.get(rel)
            if exp and compute_file_sha256(item) != exp:
                continue
            if exp:
                verified += 1
            dst = self.local_workspace / rel
            self._copy_atomic(item, dst)
            restored_files += 1
        return {"restored": restored_files > 0, "restored_files": restored_files, "verified_sha": verified}
