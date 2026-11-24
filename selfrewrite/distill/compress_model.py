from __future__ import annotations

import math
from typing import Any


def estimate_compressed_params(
    hidden_size: int,
    num_layers: int,
    intermediate_size: int,
    vocab_size: int = 151936,
    num_heads: int = 16,
    num_kv_heads: int = 4,
    tie_word_embeddings: bool = False,
) -> dict[str, Any]:
    """Compute exact parameter count & memory footprint for a SwiGLU transformer config."""
    head_dim = max(1, hidden_size // max(1, num_heads))
    embed = vocab_size * hidden_size
    lm_head = 0 if tie_word_embeddings else vocab_size * hidden_size
    attn_per_layer = (
        hidden_size * (num_heads * head_dim)
        + 2 * hidden_size * (num_kv_heads * head_dim)
        + (num_heads * head_dim) * hidden_size
    )
    mlp_per_layer = 3 * hidden_size * intermediate_size
    norms = (2 * num_layers + 1) * hidden_size
    total_params = embed + lm_head + num_layers * (attn_per_layer + mlp_per_layer) + norms
    return {
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "intermediate_size": int(intermediate_size),
        "total_params": int(total_params),
        "params_millions": round(total_params / 1e6, 2),
        "bf16_mb": round(total_params * 2 / 1e6, 2),
        "int4_mb": round(total_params * 0.55 / 1e6, 2),
    }


def plan_compression(
    base_arch: dict[str, Any],
    target_mlp_ratio: float = 0.75,
    keep_layer_ratio: float = 1.0,
) -> dict[str, Any]:
    """Plan aligned SwiGLU MLP slicing & layer dropping while preserving endpoints."""
    h = int(base_arch["hidden_size"])
    nl = int(base_arch["num_layers"])
    inter = int(base_arch["intermediate_size"])
    vocab = int(base_arch.get("vocab_size", 151936))
    heads = int(base_arch.get("num_heads", 16))
    kv_heads = int(base_arch.get("num_kv_heads", 4))
    tied = bool(base_arch.get("tie_word_embeddings", False))

    raw_inter = max(64, int(round(inter * float(target_mlp_ratio))))
    target_inter = max(64, int(round(raw_inter / 64.0)) * 64)
    target_inter = min(inter, target_inter)

    keep_count = max(2, min(nl, int(round(nl * float(keep_layer_ratio)))))
    if keep_count == nl:
        keep_indices = list(range(nl))
    else:
        # Uniformly sample interior layers while always keeping layer 0 and layer nl-1.
        interior = keep_count - 2
        keep_indices = [0]
        if interior > 0:
            step = (nl - 2) / interior
            for i in range(interior):
                cand = 1 + int(round(i * step + 0.5 * step - 0.5))
                cand = max(keep_indices[-1] + 1, min(nl - 2 - (interior - 1 - i), cand))
                keep_indices.append(cand)
        keep_indices.append(nl - 1)

    before = estimate_compressed_params(h, nl, inter, vocab, heads, kv_heads, tied)
    after = estimate_compressed_params(h, len(keep_indices), target_inter, vocab, heads, kv_heads, tied)
    reduction_pct = round(100.0 * (1.0 - after["total_params"] / max(1, before["total_params"])), 2)

    return {
        "before": before,
        "after": after,
        "target_intermediate_size": target_inter,
        "keep_layer_indices": keep_indices,
        "param_reduction_pct": reduction_pct,
    }


def _slice_linear_in(lin: Any, idx_tensor: Any) -> Any:
    import torch  # type: ignore[import-not-found]
    from torch import nn  # type: ignore[import-not-found]

    new_lin = nn.Linear(int(idx_tensor.numel()), lin.out_features, bias=(lin.bias is not None), dtype=lin.weight.dtype, device=lin.weight.device)
    with torch.no_grad():
        new_lin.weight.copy_(lin.weight.index_select(1, idx_tensor.to(lin.weight.device)))
        if lin.bias is not None and new_lin.bias is not None:
            new_lin.bias.copy_(lin.bias)
    return new_lin


def _slice_linear_out(lin: Any, idx_tensor: Any) -> Any:
    import torch  # type: ignore[import-not-found]
    from torch import nn  # type: ignore[import-not-found]

    new_lin = nn.Linear(lin.in_features, int(idx_tensor.numel()), bias=(lin.bias is not None), dtype=lin.weight.dtype, device=lin.weight.device)
    with torch.no_grad():
        new_lin.weight.copy_(lin.weight.index_select(0, idx_tensor.to(lin.weight.device)))
        if lin.bias is not None and new_lin.bias is not None:
            new_lin.bias.copy_(lin.bias.index_select(0, idx_tensor.to(lin.bias.device)))
    return new_lin


def prune_mlp_width(model: Any, target_intermediate_size: int) -> dict[str, Any]:
    """Slice SwiGLU MLP channels across decoder layers by weight norm importance."""
    import torch  # type: ignore[import-not-found]
    from selfrewrite.backends.hf_gpu_backend import resolve_decoder_layers

    layers = resolve_decoder_layers(model)
    k = int(target_intermediate_size)
    if k < 1:
        raise ValueError(f"target_intermediate_size must be >= 1, got {k}")

    pruned_layers = 0
    for layer in layers:
        mlp = getattr(layer, "mlp", None)
        if mlp is None or not all(hasattr(mlp, attr) for attr in ("gate_proj", "up_proj", "down_proj")):
            continue
        cur_inter = int(mlp.down_proj.in_features)
        if k >= cur_inter:
            continue
        with torch.no_grad():
            down_norm = torch.linalg.vector_norm(mlp.down_proj.weight.detach().float(), dim=0)
            up_norm = torch.linalg.vector_norm(mlp.up_proj.weight.detach().float(), dim=1)
            scores = down_norm * up_norm
            _, top_idx = torch.topk(scores, k=k, largest=True, sorted=False)
            keep_idx, _ = torch.sort(top_idx)
        mlp.gate_proj = _slice_linear_out(mlp.gate_proj, keep_idx)
        mlp.up_proj = _slice_linear_out(mlp.up_proj, keep_idx)
        mlp.down_proj = _slice_linear_in(mlp.down_proj, keep_idx)
        pruned_layers += 1

    if hasattr(model, "config"):
        if hasattr(model.config, "intermediate_size"):
            model.config.intermediate_size = k
        text_cfg = getattr(model.config, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "intermediate_size"):
            text_cfg.intermediate_size = k
    return {"pruned_layers": pruned_layers, "intermediate_size": k}


def drop_decoder_layers(model: Any, keep_indices: list[int]) -> dict[str, Any]:
    """Drop intermediate decoder blocks while keeping module indexing & config consistent."""
    from torch import nn  # type: ignore[import-not-found]

    keep = sorted({int(i) for i in keep_indices})
    if not keep:
        raise ValueError("keep_indices must be non-empty")

    owner = None
    attr_name = None
    for chain in (
        ("model", "language_model", "model"),
        ("language_model", "model"),
        ("model", "language_model"),
        ("language_model",),
        ("model",),
        (),
        ("transformer",),
    ):
        cur = model
        ok = True
        for part in chain:
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                ok = False
                break
        if ok:
            if hasattr(cur, "layers"):
                owner, attr_name = cur, "layers"
                break
            if hasattr(cur, "h"):
                owner, attr_name = cur, "h"
                break
    if owner is None or attr_name is None:
        raise RuntimeError(f"Could not locate decoder ModuleList owner on {type(model).__name__}")

    orig_layers = list(getattr(owner, attr_name))
    kept_list = nn.ModuleList([orig_layers[i] for i in keep])
    for new_idx, blk in enumerate(kept_list):
        if hasattr(blk, "layer_idx"):
            blk.layer_idx = new_idx
        for sub_name in ("self_attn", "linear_attn"):
            sub_mod = getattr(blk, sub_name, None)
            if sub_mod is not None and hasattr(sub_mod, "layer_idx"):
                sub_mod.layer_idx = new_idx
    setattr(owner, attr_name, kept_list)

    if hasattr(model, "config"):
        if hasattr(model.config, "num_hidden_layers"):
            model.config.num_hidden_layers = len(kept_list)
        text_cfg = getattr(model.config, "text_config", None)
        if text_cfg is not None:
            if hasattr(text_cfg, "num_hidden_layers"):
                text_cfg.num_hidden_layers = len(kept_list)
            layer_types = getattr(text_cfg, "layer_types", None)
            if isinstance(layer_types, list) and layer_types:
                text_cfg.layer_types = [layer_types[i] for i in keep if i < len(layer_types)]
    return {
        "original_layers": len(orig_layers),
        "kept_layers": len(kept_list),
        "keep_indices": keep,
    }


def distill_kl_ce_step(
    student_logits: Any,
    teacher_logits: Any,
    labels: Any,
    temperature: float = 2.0,
    alpha_kl: float = 0.5,
) -> Any:
    """Compute shifted causal KL + CE logit distillation loss for compressed students."""
    import torch.nn.functional as F  # type: ignore[import-not-found]

    s_shift = student_logits[:, :-1, :].float()
    t_shift = teacher_logits[:, :-1, :].detach().float()
    y_shift = labels[:, 1:]

    valid_mask = (y_shift != -100)
    valid_count = valid_mask.sum()
    if int(valid_count.item()) == 0:
        return s_shift.sum() * 0.0

    ce_loss = F.cross_entropy(s_shift.reshape(-1, s_shift.size(-1)), y_shift.reshape(-1), ignore_index=-100)
    T = float(temperature)
    log_p_s = F.log_softmax(s_shift / T, dim=-1)
    p_t = F.softmax(t_shift / T, dim=-1)
    per_tok_kl = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1)
    mask_f = valid_mask.to(dtype=per_tok_kl.dtype)
    kl_loss = (per_tok_kl * mask_f).sum() / mask_f.sum().clamp(min=1.0) * (T * T)
    a = float(alpha_kl)
    return a * kl_loss + (1.0 - a) * ce_loss
