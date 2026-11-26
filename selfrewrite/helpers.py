from __future__ import annotations

import math
import re
import statistics
from typing import Any, Iterable

_LAYER_IDX_RE = re.compile(r"(?:^|\.)(?:model\.)?layers\.(\d+)\.")


def _container_kind(obj: Any) -> str:
    mod = getattr(type(obj), "__module__", "")
    if mod.startswith("torch"):
        return "torch"
    if mod.startswith("numpy"):
        return "numpy"
    return "list"


def tensor_shape(t: Any) -> tuple[int, ...]:
    """Return shape tuple for a torch.Tensor, numpy.ndarray, or nested list."""
    kind = _container_kind(t)
    if kind in ("torch", "numpy"):
        return tuple(int(d) for d in t.shape)
    dims: list[int] = []
    cur = t
    while isinstance(cur, (list, tuple)):
        dims.append(len(cur))
        if not cur:
            break
        cur = cur[0]
    return tuple(dims)


def to_nested_list(t: Any) -> Any:
    """Convert tensor/array/nested iterable into pure-Python nested floats."""
    kind = _container_kind(t)
    if kind == "torch":
        return t.detach().cpu().float().tolist()
    if kind == "numpy":
        return t.astype("float64").tolist()
    if isinstance(t, (list, tuple)):
        return [to_nested_list(x) for x in t]
    return float(t)


def from_nested_like(data: Any, template: Any) -> Any:
    """Convert nested float lists back into the container type/dtype/device of template."""
    kind = _container_kind(template)
    if kind == "torch":
        import torch  # type: ignore[import-not-found]

        return torch.tensor(data, dtype=template.dtype, device=template.device)
    if kind == "numpy":
        import numpy as np  # type: ignore[import-not-found]

        return np.array(data, dtype=template.dtype)
    return data


def flatten_floats(t: Any) -> list[float]:
    """Flatten any tensor/array/nested list into a flat list of Python floats."""
    raw = to_nested_list(t)
    if not isinstance(raw, list):
        return [float(raw)]
    out: list[float] = []
    stack: list[Any] = [raw]
    while stack:
        cur = stack.pop()
        if isinstance(cur, list):
            for item in reversed(cur):
                stack.append(item)
        else:
            out.append(float(cur))
    return out


def all_finite(t: Any) -> bool:
    """Check that all entries in a tensor/array/nested list are finite."""
    kind = _container_kind(t)
    if kind == "torch":
        import torch  # type: ignore[import-not-found]

        return bool(torch.isfinite(t).all().item())
    if kind == "numpy":
        import numpy as np  # type: ignore[import-not-found]

        return bool(np.isfinite(t).all())
    return all(math.isfinite(v) for v in flatten_floats(t))


def layer_output_matrices(layer_idx: int, tensor_names: Iterable[str]) -> list[str]:
    """Return residual-writing output matrices for a transformer layer index."""
    idx_int = int(layer_idx)
    layer_names: list[str] = []
    for name in tensor_names:
        m = _LAYER_IDX_RE.search(name)
        if m and int(m.group(1)) == idx_int:
            layer_names.append(name)

    selected: list[str] = []
    for name in sorted(layer_names):
        if name.endswith((".o_proj.weight", ".out_proj.weight", ".down_proj.weight")):
            selected.append(name)
    return selected


def layer_state_tensors(layer_idx: int, tensor_names: Iterable[str]) -> list[str]:
    """Return Gated DeltaNet / recurrent state parameters (dt_bias, A_log, conv1d, in_proj) for a layer."""
    idx_int = int(layer_idx)
    suffixes = (
        ".dt_bias",
        ".A_log",
        ".conv1d.weight",
        ".conv1d.bias",
        ".in_proj_a.weight",
        ".in_proj_b.weight",
        ".in_proj_z.weight",
    )
    selected: list[str] = []
    for name in sorted(tensor_names):
        m = _LAYER_IDX_RE.search(name)
        if m and int(m.group(1)) == idx_int and name.endswith(suffixes):
            selected.append(name)
    return selected


def unit_vector(vec: Any) -> Any:
    """Normalize a 1D vector to unit L2 norm."""
    kind = _container_kind(vec)
    if kind == "torch":
        import torch  # type: ignore[import-not-found]

        vf = vec.detach().float()
        norm = torch.linalg.vector_norm(vf)
        if float(norm.item()) <= 1e-12:
            raise ValueError("Cannot normalize zero-norm vector")
        return (vf / norm).to(dtype=vec.dtype, device=vec.device)
    if kind == "numpy":
        import numpy as np  # type: ignore[import-not-found]

        vf = vec.astype("float64")
        norm = float(np.linalg.norm(vf))
        if norm <= 1e-12:
            raise ValueError("Cannot normalize zero-norm vector")
        return (vf / norm).astype(vec.dtype)

    vals = [float(x) for x in vec]
    if not vals:
        raise ValueError("Cannot normalize empty vector")
    norm = math.sqrt(sum(x * x for x in vals))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize zero-norm vector")
    return [x / norm for x in vals]


def mean_diff_direction(pos_vectors: Any, neg_vectors: Any) -> Any:
    """Compute unit direction from mean(neg_vectors) to mean(pos_vectors)."""
    kind = _container_kind(pos_vectors)
    if kind == "torch":
        import torch  # type: ignore[import-not-found]

        p = pos_vectors.detach().float()
        n = neg_vectors.detach().float() if _container_kind(neg_vectors) == "torch" else torch.tensor(to_nested_list(neg_vectors), dtype=torch.float32, device=p.device)
        if p.ndim != 2 or n.ndim != 2 or p.shape[0] == 0 or n.shape[0] == 0:
            raise ValueError("pos_vectors and neg_vectors must be non-empty 2D arrays")
        diff = p.mean(dim=0) - n.mean(dim=0)
        return unit_vector(diff).to(dtype=pos_vectors.dtype, device=pos_vectors.device)
    if kind == "numpy":
        import numpy as np  # type: ignore[import-not-found]

        p = np.asarray(pos_vectors, dtype="float64")
        n = np.asarray(neg_vectors, dtype="float64")
        if p.ndim != 2 or n.ndim != 2 or p.shape[0] == 0 or n.shape[0] == 0:
            raise ValueError("pos_vectors and neg_vectors must be non-empty 2D arrays")
        return unit_vector(p.mean(axis=0) - n.mean(axis=0)).astype(pos_vectors.dtype)

    p_rows = to_nested_list(pos_vectors)
    n_rows = to_nested_list(neg_vectors)
    if not p_rows or not n_rows:
        raise ValueError("pos_vectors and neg_vectors must be non-empty")
    dim = len(p_rows[0])
    if any(len(r) != dim for r in p_rows) or any(len(r) != dim for r in n_rows):
        raise ValueError("Mismatched vector dimensions")
    p_mean = [sum(r[j] for r in p_rows) / len(p_rows) for j in range(dim)]
    n_mean = [sum(r[j] for r in n_rows) / len(n_rows) for j in range(dim)]
    return unit_vector([p_mean[j] - n_mean[j] for j in range(dim)])


def remove_direction(W: Any, u: Any, alpha: float) -> Any:
    """Return W - alpha * u @ (u^T @ W) shrinking output component along unit vector u."""
    w_shape = tensor_shape(W)
    if len(w_shape) != 2:
        raise ValueError(f"Expected 2D matrix W, got shape {w_shape}")
    out_dim, in_dim = w_shape
    kind = _container_kind(W)

    if kind == "torch":
        import torch  # type: ignore[import-not-found]

        wf = W.detach().float()
        uf = u.detach().float() if _container_kind(u) == "torch" else torch.tensor(to_nested_list(u), dtype=torch.float32, device=wf.device)
        uf = uf.reshape(-1)
        if uf.numel() != out_dim:
            raise ValueError(f"u length {uf.numel()} does not match W output dim {out_dim}")
        u_hat = unit_vector(uf)
        proj = torch.matmul(u_hat, wf)  # [in_dim]
        edited = wf - float(alpha) * torch.outer(u_hat, proj)
        return edited.to(dtype=W.dtype, device=W.device)

    if kind == "numpy":
        import numpy as np  # type: ignore[import-not-found]

        wf = np.asarray(W, dtype="float64")
        uf = np.asarray(u, dtype="float64").reshape(-1)
        if uf.size != out_dim:
            raise ValueError(f"u length {uf.size} does not match W output dim {out_dim}")
        u_hat = unit_vector(uf)
        proj = u_hat @ wf
        edited = wf - float(alpha) * np.outer(u_hat, proj)
        return edited.astype(W.dtype)

    w_rows = to_nested_list(W)
    u_hat = unit_vector(u)
    if len(u_hat) != out_dim:
        raise ValueError(f"u length {len(u_hat)} does not match W output dim {out_dim}")
    # proj[j] = sum_i u_hat[i] * w_rows[i][j]
    proj = [sum(u_hat[i] * w_rows[i][j] for i in range(out_dim)) for j in range(in_dim)]
    scale = float(alpha)
    return [
        [w_rows[i][j] - scale * u_hat[i] * proj[j] for j in range(in_dim)]
        for i in range(out_dim)
    ]


def outlier_ratios(tensors_by_name: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Compute max(|W|), median(|W|), std, and max/median ratio per weight tensor."""
    report: dict[str, dict[str, float]] = {}
    for name, tensor in tensors_by_name.items():
        if isinstance(tensor, tuple):
            raise TypeError(
                f"outlier_ratios expects weight tensors (e.g. {{nm: load_original(nm)}}), got shape tuple for {name!r}"
            )
        kind = _container_kind(tensor)
        if kind == "torch":
            import torch  # type: ignore[import-not-found]

            t = tensor.detach()
            n = int(t.numel())
            max_abs = float(t.abs().max().item()) if n else 0.0
            if n == 0:
                med_abs = 0.0
                std_val = 0.0
            elif n <= 262_144:
                tf = t.float()
                med_abs = float(tf.abs().median().item())
                std_val = float(tf.std(unbiased=False).item()) if n > 1 else 0.0
            else:
                step = max(1, n // 262_144)
                sub = t.reshape(-1)[::step].float()
                med_abs = float(sub.abs().median().item())
                std_val = float(sub.std(unbiased=False).item()) if sub.numel() > 1 else 0.0
        elif kind == "numpy":
            import numpy as np  # type: ignore[import-not-found]

            arr = np.asarray(tensor)
            n = int(arr.size)
            max_abs = float(np.abs(arr).max()) if n else 0.0
            if n == 0:
                med_abs = 0.0
                std_val = 0.0
            elif n <= 262_144:
                af = np.asarray(arr, dtype="float64")
                med_abs = float(np.median(np.abs(af)))
                std_val = float(af.std()) if n > 1 else 0.0
            else:
                step = max(1, n // 262_144)
                sub = np.asarray(arr.reshape(-1)[::step], dtype="float64")
                med_abs = float(np.median(np.abs(sub)))
                std_val = float(sub.std()) if sub.size > 1 else 0.0
        else:
            vals = flatten_floats(tensor)
            abs_vals = [abs(v) for v in vals]
            max_abs = max(abs_vals) if abs_vals else 0.0
            med_abs = float(statistics.median(abs_vals)) if abs_vals else 0.0
            std_val = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0

        ratio = (max_abs / med_abs) if med_abs > 1e-12 else (float("inf") if max_abs > 0 else 0.0)
        report[name] = {
            "max_abs": round(max_abs, 8),
            "median_abs": round(med_abs, 8),
            "std": round(std_val, 8),
            "ratio_max_to_median": round(ratio, 4) if math.isfinite(ratio) else ratio,
        }
    return report


def clip_outliers_median(W: Any, factor: float = 15.0) -> Any:
    """Clamp entries exceeding factor * median(|W|) into [-bound, +bound]."""
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")
    kind = _container_kind(W)
    if kind == "torch":
        import torch  # type: ignore[import-not-found]

        t = W.detach()
        n = int(t.numel())
        if n == 0:
            return W.clone()
        if n <= 262_144:
            wf = t.float()
            med = float(wf.abs().median().item())
            if med <= 1e-12:
                return W.clone()
            bound = float(factor) * med
            return torch.clamp(wf, min=-bound, max=bound).to(dtype=W.dtype, device=W.device)
        step = max(1, n // 262_144)
        med = float(t.reshape(-1)[::step].float().abs().median().item())
        if med <= 1e-12:
            return W.clone()
        bound = float(factor) * med
        return torch.clamp(t.float(), min=-bound, max=bound).to(dtype=W.dtype, device=W.device)
    if kind == "numpy":
        import numpy as np  # type: ignore[import-not-found]

        arr = np.asarray(W)
        n = int(arr.size)
        if n == 0:
            return W.copy()
        if n <= 262_144:
            af = np.asarray(arr, dtype="float64")
            med = float(np.median(np.abs(af)))
            if med <= 1e-12:
                return W.copy()
            bound = float(factor) * med
            return np.clip(af, -bound, bound).astype(W.dtype)
        step = max(1, n // 262_144)
        med = float(np.median(np.abs(np.asarray(arr.reshape(-1)[::step], dtype="float64"))))
        if med <= 1e-12:
            return W.copy()
        bound = float(factor) * med
        return np.clip(np.asarray(arr, dtype="float64"), -bound, bound).astype(W.dtype)

    vals = flatten_floats(W)
    med = float(statistics.median(abs(v) for v in vals)) if vals else 0.0
    if med <= 1e-12:
        return to_nested_list(W)
    bound = float(factor) * med

    def _clip(obj: Any) -> Any:
        if isinstance(obj, list):
            return [_clip(x) for x in obj]
        v = float(obj)
        if v > bound:
            return bound
        if v < -bound:
            return -bound
        return v

    return _clip(to_nested_list(W))
