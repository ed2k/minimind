from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Callable

from selfrewrite.config import atomic_write_bytes, atomic_write_text
from selfrewrite.helpers import all_finite, flatten_floats, tensor_shape, to_nested_list


class EditValidationError(ValueError):
    """Raised when a proposed weight edit violates Algorithm 4 constraints."""


_FORBIDDEN_SUBSTRINGS = (
    "embed_tokens",
    "word_embeddings",
    "wte.",
    ".wte",
    "wpe.",
    ".wpe",
    "lm_head",
)
_FINAL_NORM_RE = re.compile(
    r"^(?:(?:language_model\.)?model\.|transformer\.)?(?:norm|ln_f)\.(?:weight|bias)$"
)
_TOP_OUTPUT_RE = re.compile(r"^(?:output|embed_out)\.(?:weight|bias)$")


def is_forbidden_tensor_name(name: str) -> bool:
    """Return True if tensor belongs to embeddings, final normalization, or LM output head."""
    normed = str(name or "").strip().lower()
    if not normed:
        return True
    if any(tok in normed for tok in _FORBIDDEN_SUBSTRINGS):
        return True
    if normed in ("wte", "wpe"):
        return True
    if _FINAL_NORM_RE.match(normed) or _TOP_OUTPUT_RE.match(normed):
        return True
    return False


def _frobenius_metrics(original: Any, candidate: Any) -> dict[str, float]:
    orig_mod = getattr(type(original), "__module__", "")
    cand_mod = getattr(type(candidate), "__module__", "")
    if orig_mod.startswith("torch") and cand_mod.startswith("torch"):
        import torch  # type: ignore[import-not-found]

        o = original.detach().float().reshape(-1)
        w = candidate.detach().float().reshape(-1)
        orig_sq = float(torch.dot(o, o).item())
        cand_sq = float(torch.dot(w, w).item())
        diff_sq = float(torch.dot(w - o, w - o).item())
        dot = float(torch.dot(w, o).item())
    else:
        o_vals = flatten_floats(original)
        w_vals = flatten_floats(candidate)
        orig_sq = sum(a * a for a in o_vals)
        cand_sq = sum(b * b for b in w_vals)
        diff_sq = sum((b - a) ** 2 for a, b in zip(o_vals, w_vals))
        dot = sum(a * b for a, b in zip(o_vals, w_vals))

    orig_norm = math.sqrt(max(0.0, orig_sq))
    cand_norm = math.sqrt(max(0.0, cand_sq))
    diff_norm = math.sqrt(max(0.0, diff_sq))
    rel_diff = (diff_norm / orig_norm) if orig_norm > 1e-12 else diff_norm

    if orig_sq > 1e-12:
        c = dot / orig_sq
        resid_sq = max(0.0, cand_sq - (dot * dot) / orig_sq)
        rel_scalar_resid = math.sqrt(resid_sq / orig_sq)
    else:
        c = 0.0
        rel_scalar_resid = 1.0 if cand_sq > 1e-12 else 0.0

    return {
        "orig_norm": round(orig_norm, 6),
        "edit_norm": round(cand_norm, 6),
        "diff_norm": round(diff_norm, 6),
        "rel_diff": round(rel_diff, 8),
        "best_scalar": round(c, 8),
        "rel_scalar_resid": round(rel_scalar_resid, 10),
    }


def validate_tensor_edit(
    name: str,
    original: Any,
    candidate: Any,
    expected_shape: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Enforce Algorithm 4 validity checks on a proposed replacement tensor."""
    if is_forbidden_tensor_name(name):
        raise EditValidationError(
            f"Refused edit to forbidden layer {name!r} (embedding, final norm, or LM head)"
        )

    orig_shape = tensor_shape(original)
    cand_shape = tensor_shape(candidate)
    if expected_shape is not None and tuple(expected_shape) != cand_shape:
        raise EditValidationError(
            f"Shape mismatch for {name!r}: expected {tuple(expected_shape)}, got {cand_shape}"
        )
    if orig_shape != cand_shape:
        raise EditValidationError(
            f"Shape mismatch for {name!r}: original is {orig_shape}, candidate is {cand_shape}"
        )
    if not all_finite(candidate):
        raise EditValidationError(f"Non-finite (NaN or Inf) entry in candidate edit for {name!r}")

    metrics = _frobenius_metrics(original, candidate)
    if metrics["diff_norm"] <= 1e-12:
        raise EditValidationError(f"Refused no-op edit for {name!r} (identical to original weights)")
    if metrics["rel_scalar_resid"] < 1e-5:
        raise EditValidationError(
            f"Refused global scalar rescaling (W' = c*W) or zeroing on {name!r} "
            f"(c={metrics['best_scalar']}, rel_resid={metrics['rel_scalar_resid']})"
        )

    return {"shape": list(cand_shape), **metrics}


def _safe_filename(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")[:64]
    return f"{cleaned}_{digest}"


def _serialize_tensor(tensor: Any, base_path: Path) -> tuple[Path, str]:
    mod = getattr(type(tensor), "__module__", "")
    if mod.startswith("torch"):
        try:
            from safetensors.torch import save as st_save  # type: ignore[import-not-found]

            out = base_path.with_suffix(".safetensors")
            target_t = tensor.detach().cpu()
            raw = st_save({"tensor": target_t})
            atomic_write_bytes(out, raw)
            return out, "safetensors"
        except Exception:
            pass
    out = base_path.with_suffix(".json")
    payload = {"shape": list(tensor_shape(tensor)), "data": to_nested_list(tensor)}
    atomic_write_text(out, json.dumps(payload))
    return out, "json"


def _deserialize_tensor(path: Path, fmt: str, target_torch: bool = False) -> Any:
    if fmt == "safetensors":
        from safetensors.torch import load_file  # type: ignore[import-not-found]

        return load_file(str(path), device="cpu")["tensor"]
    data = json.loads(path.read_text(encoding="utf-8"))["data"]
    if target_torch:
        import torch  # type: ignore[import-not-found]

        return torch.tensor(data, dtype=torch.bfloat16)
    return data


class EditStore:
    """Manages WORK/edited and Protocol 2 named snapshots (WORK/saved_edits/<tag>)."""

    def __init__(
        self,
        work_dir: Path,
        original_loader: Callable[[str], Any],
        tensor_shapes: dict[str, tuple[int, ...]],
        raw_open: Callable[..., Any] | None = None,
    ):
        self.work_dir = Path(work_dir)
        self.edited_dir = self.work_dir / "edited"
        self.saved_dir = self.work_dir / "saved_edits"
        self.index_path = self.edited_dir / "index.json"
        self._original_loader = original_loader
        self.tensor_shapes = {k: tuple(v) for k, v in tensor_shapes.items()}
        self._raw_open = raw_open
        self.edited_dir.mkdir(parents=True, exist_ok=True)
        self.saved_dir.mkdir(parents=True, exist_ok=True)

    def _read_index(self) -> dict[str, dict[str, Any]]:
        if not self.index_path.is_file():
            return {}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_index(self, index: dict[str, dict[str, Any]]) -> None:
        atomic_write_text(self.index_path, json.dumps(index, indent=2, sort_keys=True) + "\n")

    def save_edit(self, name: str, candidate: Any) -> dict[str, Any]:
        """Validate candidate tensor against original and persist atomically into WORK/edited."""
        if name not in self.tensor_shapes:
            raise EditValidationError(f"Unknown weight tensor {name!r}")
        expected_shape = self.tensor_shapes[name]
        original = self._original_loader(name)
        meta = validate_tensor_edit(name, original, candidate, expected_shape=expected_shape)

        base = self.edited_dir / _safe_filename(name)
        out_path, fmt = _serialize_tensor(candidate, base)
        index = self._read_index()
        old_entry = index.get(name)
        if old_entry and old_entry.get("filename") and old_entry["filename"] != out_path.name:
            old_file = self.edited_dir / old_entry["filename"]
            if old_file.exists():
                old_file.unlink()
        index[name] = {
            **meta,
            "name": name,
            "filename": out_path.name,
            "format": fmt,
        }
        self._write_index(index)
        return dict(index[name])

    def drop_edit(self, name: str | None = None) -> list[str]:
        """Remove one tensor edit or all active tensor edits."""
        index = self._read_index()
        if name is None:
            removed = sorted(index.keys())
            for entry in index.values():
                f = self.edited_dir / entry.get("filename", "")
                if f.is_file():
                    f.unlink()
            self._write_index({})
            return removed

        if name not in index:
            return []
        entry = index.pop(name)
        f = self.edited_dir / entry.get("filename", "")
        if f.is_file():
            f.unlink()
        self._write_index(index)
        return [name]

    def list_edits(self) -> list[dict[str, Any]]:
        """List metadata for currently active edited tensors."""
        index = self._read_index()
        return [dict(index[k]) for k in sorted(index.keys())]

    def load_edit_tensors(self, target_torch: bool = False) -> dict[str, Any]:
        """Load all active edited tensor values keyed by parameter name."""
        index = self._read_index()
        out: dict[str, Any] = {}
        for name, entry in index.items():
            fpath = self.edited_dir / entry["filename"]
            if fpath.is_file():
                out[name] = _deserialize_tensor(fpath, entry["format"], target_torch=target_torch)
        return out

    def save_named_edit(self, tag: str) -> dict[str, Any]:
        """Snapshot current WORK/edited tensors as a named edit for Protocol 2 comparison."""
        clean_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", str(tag or "").strip())
        if not clean_tag or clean_tag in (".", ".."):
            raise EditValidationError(f"Invalid saved edit tag {tag!r}")
        index = self._read_index()
        if not index:
            raise EditValidationError("Cannot snapshot an empty active edit; call save_edit() first")
        dst_dir = self.saved_dir / clean_tag
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        for entry in index.values():
            src = self.edited_dir / entry["filename"]
            if src.is_file():
                shutil.copy2(src, dst_dir / entry["filename"])
        atomic_write_text(dst_dir / "index.json", json.dumps(index, indent=2, sort_keys=True) + "\n")
        return {"tag": clean_tag, "num_tensors": len(index), "tensors": sorted(index.keys())}

    def load_named_edit(self, tag: str) -> dict[str, Any]:
        """Restore a Protocol 2 named edit snapshot back into active WORK/edited."""
        clean_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", str(tag or "").strip())
        src_dir = self.saved_dir / clean_tag
        idx_file = src_dir / "index.json"
        if not idx_file.is_file():
            raise EditValidationError(f"Saved edit {tag!r} does not exist")
        self.drop_edit(None)
        index = json.loads(idx_file.read_text(encoding="utf-8"))
        for entry in index.values():
            src = src_dir / entry["filename"]
            if src.is_file():
                shutil.copy2(src, self.edited_dir / entry["filename"])
        self._write_index(index)
        return {"tag": clean_tag, "num_tensors": len(index), "tensors": sorted(index.keys())}

    def list_named_edits(self) -> list[str]:
        if not self.saved_dir.is_dir():
            return []
        return sorted(p.name for p in self.saved_dir.iterdir() if p.is_dir() and (p / "index.json").is_file())

    def get_named_edit_tensors(self, tag: str, target_torch: bool = False) -> dict[str, Any]:
        clean_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", str(tag or "").strip())
        src_dir = self.saved_dir / clean_tag
        idx_file = src_dir / "index.json"
        if not idx_file.is_file():
            raise EditValidationError(f"Saved edit {tag!r} does not exist")
        index = json.loads(idx_file.read_text(encoding="utf-8"))
        out: dict[str, Any] = {}
        for name, entry in index.items():
            fpath = src_dir / entry["filename"]
            if fpath.is_file():
                out[name] = _deserialize_tensor(fpath, entry["format"], target_torch=target_torch)
        return out

    def diff_summary(self) -> dict[str, Any]:
        entries = self.list_edits()
        return {
            "num_tensors": len(entries),
            "tensors": [e["name"] for e in entries],
            "total_diff_norm": round(math.sqrt(sum(float(e["diff_norm"]) ** 2 for e in entries)), 6),
            "entries": entries,
        }
