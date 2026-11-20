from __future__ import annotations

import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from selfchange.kernel.integrity import atomic_write_text, compute_file_sha256

_STATE_FILES = (
    "archive.jsonl",
    "champion.json",
    "supervisor_state.json",
    "HEARTBEAT.json",
    "trajectories.jsonl",
)

_DIAGNOSTIC_FILES = (
    "report.md",
    "session_info.json",
    "baseline_untouched.json",
    "baseline_prepared.json",
    "logs/train.log",
    "logs/train_history.jsonl",
    "logs/fatal_report.json",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ColabDriveSync:
    """Atomic Google Drive mirror and checksum-verified recovery manager."""

    def __init__(self, local_workspace: Path, drive_mirror_dir: Path):
        self.local_workspace = Path(local_workspace).resolve()
        self.drive_mirror_dir = Path(drive_mirror_dir).resolve()
        self.manifest_path = self.drive_mirror_dir / "manifest.json"

    def _copy_atomic(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{dst.name}.tmp")
        try:
            shutil.copy2(src, tmp)
            tmp.replace(dst)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def _pack_mutable_tar(self, mutable_dir: Path, tar_out: Path) -> None:
        tar_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = tar_out.with_name(f".{tar_out.name}.tmp")
        try:
            with tarfile.open(tmp, "w:gz") as tf:
                for item in sorted(mutable_dir.rglob("*")):
                    if "__pycache__" in item.parts or item.suffix == ".pyc":
                        continue
                    rel = item.relative_to(mutable_dir).as_posix()
                    tf.add(item, arcname=rel, recursive=False)
            tmp.replace(tar_out)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def sync_to_drive(
        self,
        mutable_dir: Path | None = None,
        keep_top_k_weights: int = 2,
    ) -> dict:
        """Mirror workspace state, verified weights, and mutable snapshot to Drive atomically."""
        self.drive_mirror_dir.mkdir(parents=True, exist_ok=True)
        files_sha: dict[str, str] = {}
        weight_rel_map: dict[str, str] = {}

        for fname in (*_STATE_FILES, *_DIAGNOSTIC_FILES):
            src = self.local_workspace / fname
            if src.is_file():
                dst = self.drive_mirror_dir / fname
                self._copy_atomic(src, dst)
                files_sha[fname] = compute_file_sha256(dst)

        # Mirror champion, previous champion, and top-K archive weights
        weights_to_mirror: list[Path] = []
        champ_id: str | None = None
        champ_file = self.local_workspace / "champion.json"
        if champ_file.is_file():
            champ_data = json.loads(champ_file.read_text(encoding="utf-8"))
            champ_node = champ_data.get("champion", {})
            champ_id = champ_node.get("node_id")
            if champ_node.get("weight_path"):
                weights_to_mirror.append(Path(champ_node["weight_path"]))

        archive_file = self.local_workspace / "archive.jsonl"
        if archive_file.is_file():
            nodes = [json.loads(ln) for ln in archive_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
            eligible = [n for n in nodes if n.get("status") != "rejected" and n.get("weight_path")]
            for n in sorted(eligible, key=lambda x: float(x.get("total_score", 0.0)), reverse=True)[: max(1, keep_top_k_weights)]:
                weights_to_mirror.append(Path(n["weight_path"]))

        mirrored_weight_names: set[str] = set()
        weights_dir = self.drive_mirror_dir / "weights"
        for wpath in weights_to_mirror:
            resolved_w = wpath.resolve()
            if not resolved_w.is_file() or resolved_w.name in mirrored_weight_names:
                continue
            mirrored_weight_names.add(resolved_w.name)
            rel_drive = f"weights/{resolved_w.name}"
            dst_w = self.drive_mirror_dir / rel_drive
            self._copy_atomic(resolved_w, dst_w)
            files_sha[rel_drive] = compute_file_sha256(dst_w)
            try:
                local_rel = resolved_w.relative_to(self.local_workspace).as_posix()
            except ValueError:
                local_rel = f"models/candidates/{resolved_w.name}"
            weight_rel_map[rel_drive] = local_rel

        if weights_dir.is_dir():
            for existing in list(weights_dir.glob("*.pth")):
                if existing.name not in mirrored_weight_names:
                    existing.unlink(missing_ok=True)

        if mutable_dir is not None and Path(mutable_dir).is_dir():
            tar_dst = self.drive_mirror_dir / "mutable_snapshot.tar.gz"
            self._pack_mutable_tar(Path(mutable_dir), tar_dst)
            files_sha["mutable_snapshot.tar.gz"] = compute_file_sha256(tar_dst)

        manifest = {
            "synced_at": _utc_now(),
            "source_workspace": str(self.local_workspace),
            "champion_id": champ_id,
            "files": files_sha,
            "weight_rel_map": weight_rel_map,
        }
        atomic_write_text(self.manifest_path, json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        return manifest

    def restore_from_drive(
        self,
        verify_checksums: bool = True,
        mutable_restore_dir: Path | None = None,
    ) -> dict:
        """Verify Drive manifest checksums and atomically restore workspace state."""
        if not self.manifest_path.is_file():
            return {"restored": False, "reason": "no_manifest"}

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        expected_files: dict[str, str] = manifest.get("files", {})
        if verify_checksums:
            mismatched: list[str] = []
            for rel, expected_sha in expected_files.items():
                fpath = self.drive_mirror_dir / rel
                if not fpath.is_file() or compute_file_sha256(fpath) != expected_sha:
                    mismatched.append(rel)
            if mismatched:
                return {"restored": False, "reason": "checksum_mismatch", "mismatched": mismatched}

        self.local_workspace.mkdir(parents=True, exist_ok=True)
        restored_files: list[str] = []
        for fname in (*_STATE_FILES, *_DIAGNOSTIC_FILES):
            if fname in expected_files:
                src = self.drive_mirror_dir / fname
                dst = self.local_workspace / fname
                self._copy_atomic(src, dst)
                restored_files.append(fname)

        weight_rel_map: dict[str, str] = manifest.get("weight_rel_map", {})
        old_ws = manifest.get("source_workspace", "")
        for rel_drive, rel_local in weight_rel_map.items():
            src = self.drive_mirror_dir / rel_drive
            dst = self.local_workspace / rel_local
            self._copy_atomic(src, dst)
            restored_files.append(rel_local)

        # Rewrite stale source workspace prefixes in champion.json if restored to a new VM path.
        champ_file = self.local_workspace / "champion.json"
        if old_ws and str(self.local_workspace) != old_ws and champ_file.is_file():
            raw = champ_file.read_text(encoding="utf-8").replace(old_ws, str(self.local_workspace))
            atomic_write_text(champ_file, raw)

        if mutable_restore_dir is not None and "mutable_snapshot.tar.gz" in expected_files:
            tar_src = self.drive_mirror_dir / "mutable_snapshot.tar.gz"
            mutable_restore_dir = Path(mutable_restore_dir)
            mutable_restore_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_src, "r:gz") as tf:
                tf.extractall(mutable_restore_dir, filter="data")

        return {
            "restored": True,
            "champion_id": manifest.get("champion_id"),
            "restored_files": restored_files,
        }
