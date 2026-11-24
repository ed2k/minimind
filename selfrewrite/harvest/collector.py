from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from selfrewrite.config import atomic_write_text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SessionHarvester:
    """Appends lossless turn/code/forward/edit telemetry into raw_sessions/<session_id>/."""

    def __init__(self, raw_root: Path, session_id: str, meta: dict[str, Any]):
        self.session_dir = Path(raw_root).resolve() / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.meta = dict(meta)
        self.meta.setdefault("session_id", session_id)
        self.meta.setdefault("started_at", _utc_now())
        atomic_write_text(self.session_dir / "session_meta.json", json.dumps(self.meta, indent=2) + "\n")
        self._transcript_rows: list[dict[str, Any]] = []
        self._code_rows: list[dict[str, Any]] = []
        self._forward_rows: list[dict[str, Any]] = []
        self._edit_rows: list[dict[str, Any]] = []

    def _append_jsonl(self, filename: str, row: dict[str, Any]) -> None:
        target = self.session_dir / filename
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def log_turn(
        self,
        turn_index: int,
        messages_prefix: list[dict[str, str]],
        reply: str,
        action_kind: str,
        think_text: str,
        wall_sec: float,
    ) -> None:
        row = {
            "turn_index": int(turn_index),
            "messages_prefix": messages_prefix,
            "reply": reply,
            "action_kind": action_kind,
            "think_text": think_text,
            "wall_sec": float(wall_sec),
            "timestamp": _utc_now(),
        }
        self._transcript_rows.append(row)
        self._append_jsonl("transcript.jsonl", row)

    def log_observation(self, turn_index: int, action_kind: str, observation: str) -> None:
        self._append_jsonl(
            "observations.jsonl",
            {
                "turn_index": int(turn_index),
                "action_kind": action_kind,
                "observation": observation,
                "timestamp": _utc_now(),
            },
        )

    def log_code_run(
        self,
        turn_index: int,
        code: str,
        result: dict[str, Any],
        edits: list[dict[str, Any]],
    ) -> None:
        row = {
            "turn_index": int(turn_index),
            "code": code,
            "ok": bool(result.get("ok")),
            "refused": bool(result.get("refused")),
            "timed_out": bool(result.get("timed_out")),
            "output": result.get("output", ""),
            "error": result.get("error"),
            "duration_sec": result.get("duration_sec", 0.0),
            "active_edits": edits,
            "timestamp": _utc_now(),
        }
        self._code_rows.append(row)
        self._append_jsonl("code_executions.jsonl", row)
        if edits:
            edit_row = {
                "turn_index": int(turn_index),
                "event": "edits_snapshot",
                "edits": edits,
                "timestamp": _utc_now(),
            }
            self._edit_rows.append(edit_row)
            self._append_jsonl("edits_lineage.jsonl", edit_row)

    def log_forward_pass(
        self,
        turn_index: int,
        pass_id: int,
        request: dict[str, Any],
        summary: dict[str, Any],
    ) -> None:
        row = {
            "turn_index": int(turn_index),
            "pass_id": int(pass_id),
            "request": request,
            "summary": summary,
            "timestamp": _utc_now(),
        }
        self._forward_rows.append(row)
        self._append_jsonl("forward_passes.jsonl", row)

    def record_eval(self, eval_result: dict[str, Any], updated_ladder: dict[str, Any] | None = None) -> None:
        atomic_write_text(self.session_dir / "eval_result.json", json.dumps(eval_result, indent=2) + "\n")
        if updated_ladder is not None:
            atomic_write_text(self.session_dir / "ladder_score.json", json.dumps(updated_ladder, indent=2) + "\n")

    def finalize(self, result: dict[str, Any], work_dir: Path | None = None) -> None:
        atomic_write_text(self.session_dir / "session_result.json", json.dumps(result, indent=2) + "\n")
        if "ladder" in result:
            atomic_write_text(self.session_dir / "ladder_score.json", json.dumps(result["ladder"], indent=2) + "\n")
        if work_dir is not None and Path(work_dir).is_dir():
            work_dst = self.session_dir / "WORK_snapshot"
            work_dst.mkdir(parents=True, exist_ok=True)
            for fname in ("census.jsonl", "census_summary.json", "notes.md", "pass_log.jsonl", "practice_problems.jsonl", "session_state.json"):
                src = Path(work_dir) / fname
                if src.is_file():
                    shutil.copy2(src, work_dst / fname)
            edited_src = Path(work_dir) / "edited" / "index.json"
            if edited_src.is_file():
                shutil.copy2(edited_src, work_dst / "edited_index.json")
