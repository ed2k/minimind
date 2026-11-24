from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from selfrewrite.config import atomic_write_text
from selfrewrite.helpers import _container_kind, flatten_floats, from_nested_like, tensor_shape, to_nested_list


@dataclass(frozen=True)
class CorruptionSpec:
    layer_idx: int
    tensor_name: str
    row_start: int
    num_rows: int
    sigma_mult: float = 200.0
    mode: str = "gaussian_noise"  # "gaussian_noise" | "saturation" | "zero_rows"
    seed: int = 0
    orig_std: float = 0.0
    damaged_max_abs: float = 0.0
    damaged_median_abs: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def split_train_eval_layers(
    num_layers: int,
    eval_band: tuple[float, float] = (0.42, 0.52),
    edge_skip: int = 1,
) -> tuple[list[int], list[int]]:
    """Partition transformer layers cleanly so middle repair eval_layers are never seen in training."""
    n = int(num_layers)
    if n < 3:
        raise ValueError(f"num_layers must be >= 3, got {num_layers}")
    lo = max(1, int(math.floor(eval_band[0] * n)))
    hi = min(n - 2, max(lo, int(math.ceil(eval_band[1] * n)) - 1))
    # Guarantee paper-like layer 30 of 64 sits inside eval_layers.
    if n == 64 and 30 not in range(lo, hi + 1):
        lo = min(lo, 28)
        hi = max(hi, 32)
    eval_layers = list(range(lo, hi + 1))
    eval_set = set(eval_layers)
    min_train = max(0, edge_skip)
    max_train = min(n - 1, n - 1 - max(0, edge_skip))
    train_layers = [idx for idx in range(min_train, max_train + 1) if idx not in eval_set]
    if not train_layers:
        train_layers = [idx for idx in range(n) if idx not in eval_set]
    return train_layers, eval_layers


def apply_block_corruption(weight_tensor: Any, spec: CorruptionSpec) -> tuple[Any, CorruptionSpec]:
    """Overwrite rows [row_start : row_start + num_rows] of a 2D weight matrix (Eq. 4)."""
    shape = tensor_shape(weight_tensor)
    if len(shape) != 2:
        raise ValueError(f"Expected 2D matrix for corruption, got {shape}")
    out_dim, in_dim = shape
    eff_rows = max(0, min(out_dim, int(spec.num_rows)))
    r0 = max(0, min(out_dim - eff_rows, int(spec.row_start)))
    r1 = r0 + eff_rows

    kind = _container_kind(weight_tensor)
    if kind == "torch":
        import torch  # type: ignore[import-not-found]

        wf = weight_tensor.detach().clone()
        orig_std = float(wf.float().std(unbiased=False).item()) if wf.numel() > 1 else 0.01
        sigma = max(1e-6, orig_std)
        noise_scale = float(spec.sigma_mult) * sigma
        with torch.no_grad():
            if eff_rows > 0:
                if spec.mode == "gaussian_noise":
                    gen = torch.Generator().manual_seed(int(spec.seed))
                    block = torch.randn((eff_rows, in_dim), generator=gen, dtype=torch.float32) * noise_scale
                    wf[r0:r1].copy_(block.to(device=wf.device, dtype=wf.dtype))
                elif spec.mode == "saturation":
                    pattern = torch.where(
                        (torch.arange(in_dim, device=wf.device) % 2) == 0,
                        torch.tensor(noise_scale, device=wf.device, dtype=torch.float32),
                        torch.tensor(-noise_scale, device=wf.device, dtype=torch.float32),
                    ).to(dtype=wf.dtype)
                    wf[r0:r1].copy_(pattern.unsqueeze(0).expand(eff_rows, in_dim))
                elif spec.mode == "zero_rows":
                    wf[r0:r1].zero_()
                else:
                    raise ValueError(f"Unknown corruption mode {spec.mode!r}")
        abs_after = wf.float().abs()
        damaged_max = float(abs_after.max().item()) if wf.numel() else 0.0
        damaged_med = float(abs_after.median().item()) if wf.numel() else 0.0
        updated = wf
    elif kind == "numpy":
        import numpy as np  # type: ignore[import-not-found]

        arr = np.array(weight_tensor, copy=True)
        orig_std = float(arr.astype("float64").std()) if arr.size > 1 else 0.01
        sigma = max(1e-6, orig_std)
        noise_scale = float(spec.sigma_mult) * sigma
        if eff_rows > 0:
            if spec.mode == "gaussian_noise":
                rng_np = np.random.default_rng(int(spec.seed))
                arr[r0:r1] = rng_np.normal(0.0, noise_scale, size=(eff_rows, in_dim)).astype(arr.dtype)
            elif spec.mode == "saturation":
                pat = np.where((np.arange(in_dim) % 2) == 0, noise_scale, -noise_scale).astype(arr.dtype)
                arr[r0:r1] = pat[None, :]
            elif spec.mode == "zero_rows":
                arr[r0:r1] = 0.0
            else:
                raise ValueError(f"Unknown corruption mode {spec.mode!r}")
        abs_after = np.abs(arr.astype("float64"))
        damaged_max = float(abs_after.max()) if arr.size else 0.0
        damaged_med = float(np.median(abs_after)) if arr.size else 0.0
        updated = arr
    else:
        vals = flatten_floats(weight_tensor)
        orig_std = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.01
        sigma = max(1e-6, orig_std)
        noise_scale = float(spec.sigma_mult) * sigma

        rng = random.Random(int(spec.seed))
        rows = to_nested_list(weight_tensor)
        for r in range(r0, r1):
            if spec.mode == "gaussian_noise":
                rows[r] = [rng.gauss(0.0, noise_scale) for _ in range(in_dim)]
            elif spec.mode == "saturation":
                rows[r] = [(noise_scale if (j % 2 == 0) else -noise_scale) for j in range(in_dim)]
            elif spec.mode == "zero_rows":
                rows[r] = [0.0 for _ in range(in_dim)]
            else:
                raise ValueError(f"Unknown corruption mode {spec.mode!r}")

        flat_after = [v for row in rows for v in row]
        abs_after = [abs(v) for v in flat_after]
        damaged_max = max(abs_after) if abs_after else 0.0
        damaged_med = float(statistics.median(abs_after)) if abs_after else 0.0
        updated = rows

    new_spec = CorruptionSpec(
        layer_idx=spec.layer_idx,
        tensor_name=spec.tensor_name,
        row_start=r0,
        num_rows=eff_rows,
        sigma_mult=spec.sigma_mult,
        mode=spec.mode,
        seed=spec.seed,
        orig_std=round(orig_std, 8),
        damaged_max_abs=round(damaged_max, 8),
        damaged_median_abs=round(damaged_med, 8),
    )
    return updated, new_spec


def create_damaged_workspace(
    clean_weights: dict[str, Any],
    hidden_dir: Path,
    layer_idx: int,
    tensor_name: str,
    num_rows: int = 8,
    sigma_mult: float = 200.0,
    mode: str = "gaussian_noise",
    seed: int = 0,
    row_start: int | None = None,
) -> tuple[dict[str, Any], CorruptionSpec]:
    """Create a corrupted weight dictionary while keeping damage ground truth outside WORK/."""
    if tensor_name not in clean_weights:
        raise KeyError(f"Tensor {tensor_name!r} not in clean_weights")
    target = clean_weights[tensor_name]
    out_dim = tensor_shape(target)[0]
    eff_rows = min(int(num_rows), out_dim)
    rng = random.Random(int(seed))
    eff_r0 = rng.randint(0, max(0, out_dim - eff_rows)) if row_start is None else int(row_start)

    init_spec = CorruptionSpec(
        layer_idx=int(layer_idx),
        tensor_name=tensor_name,
        row_start=eff_r0,
        num_rows=eff_rows,
        sigma_mult=float(sigma_mult),
        mode=mode,
        seed=int(seed),
    )
    corrupted_tensor, final_spec = apply_block_corruption(target, init_spec)

    damaged_weights = dict(clean_weights)
    damaged_weights[tensor_name] = corrupted_tensor

    hidden_root = Path(hidden_dir)
    hidden_root.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        hidden_root / "damage_manifest.json",
        json.dumps(final_spec.to_dict(), indent=2) + "\n",
    )
    return damaged_weights, final_spec
