from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from selfchange.kernel.integrity import atomic_write_text


@dataclass
class LineageNode:
    """Single candidate or champion record in the DGM-style evolution archive."""

    node_id: str
    parent_id: str | None
    commit_sha: str
    weight_path: str | None
    survival_score: float
    eval_score: float
    self_repair_score: float
    efficiency_score: float
    total_score: float
    status: str  # "bootstrap" | "champion" | "archived" | "rejected"
    hypothesis: str
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> LineageNode:
        return cls(
            node_id=data["node_id"],
            parent_id=data.get("parent_id"),
            commit_sha=data["commit_sha"],
            weight_path=data.get("weight_path"),
            survival_score=float(data["survival_score"]),
            eval_score=float(data["eval_score"]),
            self_repair_score=float(data["self_repair_score"]),
            efficiency_score=float(data["efficiency_score"]),
            total_score=float(data["total_score"]),
            status=data["status"],
            hypothesis=data["hypothesis"],
            created_at=data["created_at"],
        )


class LineageRegistry:
    """Manages archive.jsonl, atomic champion pointers, parent sampling, and checkpoint GC."""

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = Path(workspace_dir)
        self.archive_path = self.workspace_dir / "archive.jsonl"
        self.champion_path = self.workspace_dir / "champion.json"
        self.candidates_dir = self.workspace_dir / "models" / "candidates"

    def load_archive(self) -> list[LineageNode]:
        """Load all recorded lineage nodes in chronological append order."""
        if not self.archive_path.is_file():
            return []
        nodes: list[LineageNode] = []
        for line in self.archive_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                nodes.append(LineageNode.from_dict(json.loads(stripped)))
        return nodes

    def get_node(self, node_id: str) -> LineageNode | None:
        """Look up the latest archive entry matching node_id."""
        for node in reversed(self.load_archive()):
            if node.node_id == node_id:
                return node
        return None

    def _read_champion_payload(self) -> dict | None:
        if not self.champion_path.is_file():
            return None
        return json.loads(self.champion_path.read_text(encoding="utf-8"))

    def get_champion(self) -> LineageNode | None:
        """Return the active champion node, if promoted."""
        payload = self._read_champion_payload()
        if not payload or "champion" not in payload:
            return None
        return LineageNode.from_dict(payload["champion"])

    def _write_champion(self, node: LineageNode, previous_champion_id: str | None) -> None:
        payload = {
            "champion": node.to_dict(),
            "previous_champion_id": previous_champion_id,
        }
        atomic_write_text(self.champion_path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    def register_node(
        self,
        node: LineageNode,
        promote_if_best: bool = True,
        min_improvement: float = 1e-6,
    ) -> tuple[LineageNode, bool]:
        """Append a node to archive.jsonl and atomically promote champion if eligible."""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        current_champion = self.get_champion()

        should_promote = False
        if node.status != "rejected":
            if node.status == "bootstrap" and current_champion is None:
                should_promote = True
            elif promote_if_best and (
                current_champion is None
                or node.total_score > current_champion.total_score + min_improvement
                or (
                    current_champion.weight_path is None
                    and node.weight_path is not None
                    and node.total_score >= current_champion.total_score - 1e-3
                )
            ):
                should_promote = True

        recorded = (
            replace(node, status="champion")
            if should_promote and node.status not in ("bootstrap", "champion")
            else node
        )

        with self.archive_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(recorded.to_dict(), ensure_ascii=False) + "\n")

        if should_promote:
            prev_id = current_champion.node_id if current_champion is not None else None
            self._write_champion(recorded, previous_champion_id=prev_id)
            return recorded, True

        return recorded, False

    def select_parent(
        self,
        rng: random.Random | None = None,
        exploit_prob: float = 0.7,
        top_k: int = 5,
    ) -> LineageNode:
        """Select parent node via champion exploitation or DGM inverse-child-count exploration."""
        picker = rng if rng is not None else random.Random()
        archive = self.load_archive()
        eligible = [n for n in archive if n.status != "rejected"]
        if not eligible:
            raise RuntimeError("Cannot select parent from empty or all-rejected lineage archive")

        champion = self.get_champion()
        if champion is not None and champion.status != "rejected" and picker.random() < exploit_prob:
            return champion

        child_counts: dict[str, int] = {}
        for item in archive:
            if item.parent_id:
                child_counts[item.parent_id] = child_counts.get(item.parent_id, 0) + 1

        ranked = sorted(eligible, key=lambda n: n.total_score, reverse=True)[: max(1, top_k)]
        weights = [max(n.total_score, 0.05) / (1.0 + child_counts.get(n.node_id, 0)) for n in ranked]
        return picker.choices(ranked, weights=weights, k=1)[0]

    def rollback_to_node(self, node_id: str) -> LineageNode:
        """Atomically revert champion pointer to a verified non-rejected ancestor node."""
        target = self.get_node(node_id)
        if target is None:
            raise KeyError(f"Lineage node {node_id!r} not found")
        if target.status == "rejected":
            raise ValueError(f"Cannot roll back to rejected node {node_id!r}")

        current_champion = self.get_champion()
        prev_id = current_champion.node_id if current_champion is not None else None
        self._write_champion(target, previous_champion_id=prev_id)
        return target

    def garbage_collect_checkpoints(self, keep_top_k: int = 2) -> list[Path]:
        """Delete stale .pth files under candidates_dir while protecting champion and top-K weights."""
        if not self.candidates_dir.is_dir():
            return []

        protected: set[Path] = set()
        payload = self._read_champion_payload()
        if payload:
            champ_weight = payload.get("champion", {}).get("weight_path")
            if champ_weight:
                protected.add(Path(champ_weight).resolve())
            prev_id = payload.get("previous_champion_id")
            if prev_id:
                prev_node = self.get_node(prev_id)
                if prev_node and prev_node.weight_path:
                    protected.add(Path(prev_node.weight_path).resolve())

        eligible = [n for n in self.load_archive() if n.status != "rejected" and n.weight_path]
        top_nodes = sorted(eligible, key=lambda n: n.total_score, reverse=True)[: max(0, keep_top_k)]
        for n in top_nodes:
            if n.weight_path:
                protected.add(Path(n.weight_path).resolve())

        deleted: list[Path] = []
        for ckpt in sorted(self.candidates_dir.glob("*.pth")):
            if ckpt.resolve() not in protected:
                ckpt.unlink(missing_ok=True)
                deleted.append(ckpt)
        return deleted
