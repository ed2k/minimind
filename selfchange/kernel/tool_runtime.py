from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable

from selfchange.kernel.integrity import ALLOWED_MUTABLE_PREFIX, atomic_write_text, check_git_diff_boundary, compute_file_sha256
from selfchange.kernel.sandbox import scan_mutable_tree, scan_python_source

SELFCHANGE_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_system_state",
            "description": "Get profile, resources, champion id, failures, and last fault.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_log_tail",
            "description": "Read the last N lines of a workspace log.",
            "parameters": {
                "type": "object",
                "properties": {
                    "log_name": {"type": "string"},
                    "lines": {"type": "integer"},
                },
                "required": ["log_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_archive_summary",
            "description": "Get top-K verified lineage nodes from archive.",
            "parameters": {
                "type": "object",
                "properties": {"top_k": {"type": "integer"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_mutable_files",
            "description": "List files inside selfchange/mutable/ with line counts and hashes.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_lines",
            "description": "Read 1-indexed line range (max 80 lines) from selfchange/mutable/.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_line_range",
            "description": "Replace 1-indexed inclusive lines inside selfchange/mutable/*.py.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                    "lines": {"type": "array"},
                },
                "required": ["path", "start_line", "end_line", "lines"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patch_file_replace",
            "description": "Exact unique search-and-replace in selfchange/mutable/*.py (level 2).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "target": {"type": "string"},
                    "replacement": {"type": "string"},
                },
                "required": ["path", "target", "replacement"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patch_config",
            "description": "Schema-validated partial update to selfchange/mutable/config.json.",
            "parameters": {
                "type": "object",
                "properties": {"updates": {"type": "object"}},
                "required": ["updates"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "revert_file",
            "description": "Revert a file in selfchange/mutable/ to parent state.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_worktree_checks",
            "description": "Run AST safety, py_compile, and git boundary checks on worktree.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "curate_experience",
            "description": "Summarize or curate experience buffers from trajectories.",
            "parameters": {
                "type": "object",
                "properties": {"mode": {"type": "string"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_candidate",
            "description": "Freeze worktree edits and submit candidate for 4-tier verification.",
            "parameters": {
                "type": "object",
                "properties": {"hypothesis": {"type": "string"}},
                "required": ["hypothesis"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rollback_to_ancestor",
            "description": "Roll back active parent/champion to a verified archive node.",
            "parameters": {
                "type": "object",
                "properties": {"node_id": {"type": "string"}},
                "required": ["node_id"],
            },
        },
    },
]

_TOOL_PARAM_SCHEMAS: dict[str, dict] = {
    t["function"]["name"]: t["function"]["parameters"] for t in SELFCHANGE_TOOLS
}

CONFIG_CLAMP_BOUNDS: dict[str, tuple[float, float, bool]] = {
    "autonomy_level": (0, 2, True),
    "max_turns": (1, 12, True),
    "temperature": (0.0, 1.5, False),
    "top_p": (0.1, 1.0, False),
    "thinking_ratio": (0.0, 1.0, False),
    "learning_rate": (1e-8, 5e-4, False),
    "dpo_learning_rate": (1e-9, 1e-5, False),
    "rl_learning_rate": (1e-9, 1e-4, False),
    "batch_size": (1, 16, True),
    "accumulation_steps": (1, 32, True),
    "max_seq_len": (128, 2048, True),
    "grad_clip": (0.1, 5.0, False),
    "lora_rank": (4, 64, True),
    "dpo_beta": (0.01, 1.0, False),
    "kl_beta": (0.0, 1.0, False),
    "anchor_replay_ratio": (0.1, 0.8, False),
}


def _check_json_type(val: object, expected_type: str) -> bool:
    if expected_type == "string":
        return isinstance(val, str)
    if expected_type == "integer":
        return isinstance(val, int) and not isinstance(val, bool)
    if expected_type == "number":
        return isinstance(val, (int, float)) and not isinstance(val, bool)
    if expected_type == "array":
        return isinstance(val, list)
    if expected_type == "object":
        return isinstance(val, dict)
    if expected_type == "boolean":
        return isinstance(val, bool)
    return True


def validate_tool_call(
    name: str,
    arguments: dict,
    allowed_tools: list[dict] | None = None,
) -> tuple[bool, str | None]:
    """Validate tool name, required parameters, and basic JSON types."""
    schemas = (
        {t["function"]["name"]: t["function"]["parameters"] for t in allowed_tools}
        if allowed_tools is not None
        else _TOOL_PARAM_SCHEMAS
    )
    if name not in schemas:
        return False, f"Unknown tool: {name!r}"
    if not isinstance(arguments, dict):
        return False, "Tool arguments must be a JSON object"

    schema = schemas[name]
    for req in schema.get("required", []):
        if req not in arguments:
            return False, f"Missing required parameter {req!r} for {name}"

    props = schema.get("properties", {})
    for key, val in arguments.items():
        if key in props and "type" in props[key]:
            if not _check_json_type(val, props[key]["type"]):
                return False, f"Invalid type for {name}.{key}: expected {props[key]['type']}"
    return True, None


def _clamp_config_dict(data: dict) -> dict:
    out: dict = {}
    for key, val in data.items():
        if isinstance(val, dict):
            out[key] = _clamp_config_dict(val)
        elif key in CONFIG_CLAMP_BOUNDS and isinstance(val, (int, float)) and not isinstance(val, bool):
            low, high, is_int = CONFIG_CLAMP_BOUNDS[key]
            bounded = max(low, min(high, float(val)))
            out[key] = int(round(bounded)) if is_int else bounded
        else:
            out[key] = val
    return out


def _deep_merge(base: dict, updates: dict) -> dict:
    merged = dict(base)
    for key, val in updates.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


class WorktreeToolExecutor:
    """Bounded, path-jailed tool dispatcher operating inside a candidate worktree."""

    def __init__(
        self,
        worktree_dir: Path,
        autonomy_level: int = 1,
        state_provider: Callable[[], dict] | None = None,
        log_provider: Callable[[str, int], str] | None = None,
        archive_provider: Callable[[int], list[dict]] | None = None,
        rollback_handler: Callable[[str], dict] | None = None,
        base_ref: str = "HEAD",
        trajectories_path: Path | None = None,
    ):
        self.worktree_dir = Path(worktree_dir).resolve()
        self.mutable_dir = (self.worktree_dir / "selfchange" / "mutable").resolve()
        self.autonomy_level = int(autonomy_level)
        self.state_provider = state_provider
        self.log_provider = log_provider
        self.archive_provider = archive_provider
        self.rollback_handler = rollback_handler
        self.base_ref = base_ref
        self.trajectories_path = Path(trajectories_path) if trajectories_path is not None else None
        self.submitted = False
        self.hypothesis: str | None = None
        self.rolled_back_to: str | None = None
        self._snapshots: dict[Path, str] = {}
        if self.mutable_dir.is_dir():
            for p in self.mutable_dir.rglob("*"):
                if p.is_file() and "__pycache__" not in p.parts:
                    self._snapshots[p.resolve()] = p.read_text(encoding="utf-8")

    def _resolve_mutable_path(self, rel_path: str) -> Path:
        norm = str(rel_path).strip().replace("\\", "/")
        if not norm or norm.startswith("/") or (len(norm) >= 2 and norm[1] == ":"):
            raise ValueError(f"Absolute or empty path rejected: {rel_path!r}")
        if ".." in norm.split("/"):
            raise ValueError(f"Path traversal rejected: {rel_path!r}")
        if norm.startswith(ALLOWED_MUTABLE_PREFIX):
            norm = norm[len(ALLOWED_MUTABLE_PREFIX) :]
        candidate = (self.mutable_dir / norm).resolve()
        if not candidate.is_relative_to(self.mutable_dir):
            raise ValueError(f"Path escaped mutable root: {rel_path!r}")
        return candidate

    def _validate_file_content(self, path: Path, content: str) -> str | None:
        if path.suffix == ".py":
            report = scan_python_source(content, filename=str(path))
            if not report.ok:
                v = report.violations[0]
                return f"{v.rule} at line {v.line}: {v.detail}"
        elif path.suffix == ".json":
            try:
                json.loads(content)
            except json.JSONDecodeError as exc:
                return f"json_error at line {exc.lineno}: {exc.msg}"
        return None

    def dispatch(self, name: str, arguments: dict) -> dict:
        """Validate and execute one structured tool call inside the worktree boundary."""
        ok, err = validate_tool_call(name, arguments)
        if not ok:
            return {"ok": False, "error": err}

        if name == "get_system_state":
            return {"ok": True, "state": self.state_provider() if self.state_provider else {}}

        if name == "read_log_tail":
            lines = max(1, min(80, int(arguments.get("lines", 40))))
            text = self.log_provider(arguments["log_name"], lines) if self.log_provider else ""
            return {"ok": True, "log_tail": text}

        if name == "get_archive_summary":
            top_k = max(1, min(20, int(arguments.get("top_k", 5))))
            nodes = self.archive_provider(top_k) if self.archive_provider else []
            return {"ok": True, "nodes": nodes}

        if name == "list_mutable_files":
            files = []
            if self.mutable_dir.is_dir():
                for p in sorted(self.mutable_dir.rglob("*")):
                    if p.is_file() and "__pycache__" not in p.parts:
                        line_cnt = len(p.read_text(encoding="utf-8").splitlines())
                        files.append(
                            {
                                "path": p.relative_to(self.mutable_dir).as_posix(),
                                "lines": line_cnt,
                                "sha256": compute_file_sha256(p)[:8],
                            }
                        )
            return {"ok": True, "files": files}

        if name == "read_file_lines":
            try:
                target = self._resolve_mutable_path(arguments["path"])
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            if not target.is_file():
                return {"ok": False, "error": f"File not found: {arguments['path']}"}
            all_lines = target.read_text(encoding="utf-8").splitlines()
            start = max(1, int(arguments.get("start", 1)))
            end = min(len(all_lines), max(start, int(arguments.get("end", start + 59))))
            if end - start + 1 > 80:
                end = start + 79
            numbered = [f"{idx}: {all_lines[idx - 1]}" for idx in range(start, end + 1)]
            return {"ok": True, "path": target.relative_to(self.mutable_dir).as_posix(), "total_lines": len(all_lines), "content": "\n".join(numbered)}

        if name == "replace_line_range":
            if self.autonomy_level < 1:
                return {"ok": False, "error": "replace_line_range disabled at autonomy_level < 1"}
            try:
                target = self._resolve_mutable_path(arguments["path"])
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            if not target.is_file():
                return {"ok": False, "error": f"File not found: {arguments['path']}"}
            new_lines = arguments["lines"]
            if not all(isinstance(x, str) for x in new_lines):
                return {"ok": False, "error": "lines must be a list of strings"}
            orig_text = target.read_text(encoding="utf-8")
            orig_lines = orig_text.splitlines()
            s_line, e_line = int(arguments["start_line"]), int(arguments["end_line"])
            if s_line < 1 or e_line < s_line or e_line > len(orig_lines):
                return {"ok": False, "error": f"Invalid line range {s_line}..{e_line} for {len(orig_lines)} lines"}
            updated = "\n".join(orig_lines[: s_line - 1] + new_lines + orig_lines[e_line:])
            if orig_text.endswith("\n") or updated:
                updated += "\n"
            err_msg = self._validate_file_content(target, updated)
            if err_msg:
                atomic_write_text(target, orig_text)
                return {"ok": False, "reverted": True, "error": err_msg}
            atomic_write_text(target, updated)
            return {"ok": True, "path": target.relative_to(self.mutable_dir).as_posix()}

        if name == "patch_file_replace":
            if self.autonomy_level < 2:
                return {"ok": False, "error": "patch_file_replace disabled at autonomy_level < 2"}
            try:
                target = self._resolve_mutable_path(arguments["path"])
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            if not target.is_file():
                return {"ok": False, "error": f"File not found: {arguments['path']}"}
            orig_text = target.read_text(encoding="utf-8")
            needle = arguments["target"]
            count = orig_text.count(needle) if needle else 0
            if count != 1:
                return {"ok": False, "error": f"target occurred {count} times (must be exactly 1)"}
            updated = orig_text.replace(needle, arguments["replacement"], 1)
            err_msg = self._validate_file_content(target, updated)
            if err_msg:
                atomic_write_text(target, orig_text)
                return {"ok": False, "reverted": True, "error": err_msg}
            atomic_write_text(target, updated)
            return {"ok": True, "path": target.relative_to(self.mutable_dir).as_posix()}

        if name == "patch_config":
            cfg_path = self.mutable_dir / "config.json"
            current = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
            merged = _clamp_config_dict(_deep_merge(current, arguments["updates"]))
            atomic_write_text(cfg_path, json.dumps(merged, indent=2, ensure_ascii=False) + "\n")
            return {"ok": True, "config": merged}

        if name == "revert_file":
            try:
                target = self._resolve_mutable_path(arguments["path"])
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            if (self.worktree_dir / ".git").exists():
                rel = target.relative_to(self.worktree_dir).as_posix()
                proc = subprocess.run(["git", "checkout", "--", rel], cwd=self.worktree_dir, capture_output=True, text=True)
                if proc.returncode == 0:
                    return {"ok": True, "path": rel}
            resolved = target.resolve()
            if resolved in self._snapshots:
                atomic_write_text(target, self._snapshots[resolved])
                return {"ok": True, "path": target.relative_to(self.mutable_dir).as_posix()}
            return {"ok": False, "error": f"No snapshot available for {arguments['path']}"}

        if name == "run_worktree_checks":
            scan = scan_mutable_tree(self.mutable_dir, run_py_compile=True)
            diff_violations: list[str] = []
            diff_ok = True
            if (self.worktree_dir / ".git").exists():
                diff_rep = check_git_diff_boundary(self.worktree_dir, base_ref=self.base_ref)
                diff_ok = diff_rep.ok
                diff_violations = diff_rep.violations
            return {
                "ok": scan.ok and diff_ok,
                "scanned_files": scan.scanned_files,
                "violations": [{"file": v.file, "line": v.line, "rule": v.rule, "detail": v.detail} for v in scan.violations],
                "diff_violations": diff_violations,
            }

        if name == "curate_experience":
            mode = arguments.get("mode", "rft")
            count = 0
            if self.trajectories_path and self.trajectories_path.is_file():
                count = sum(1 for line in self.trajectories_path.read_text(encoding="utf-8").splitlines() if line.strip())
            return {"ok": True, "mode": mode, "trajectories_seen": count}

        if name == "submit_candidate":
            self.submitted = True
            self.hypothesis = arguments["hypothesis"]
            return {"ok": True, "submitted": True, "hypothesis": self.hypothesis}

        if name == "rollback_to_ancestor":
            node_id = arguments["node_id"]
            self.rolled_back_to = node_id
            if self.rollback_handler is not None:
                res = self.rollback_handler(node_id)
                return {"ok": bool(res.get("ok", True)), **res}
            return {"ok": True, "rolled_back_to": node_id}

        return {"ok": False, "error": f"Unhandled tool: {name}"}
