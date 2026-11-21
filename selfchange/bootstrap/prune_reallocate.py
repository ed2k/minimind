from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from selfchange.kernel.budgets import get_profile
from selfchange.kernel.integrity import atomic_write_text
from selfchange.training.model_loader import estimate_minimind_params, load_minimind_model, save_model_config
from selfchange.training.self_train_runner import format_sft_record, write_sft_jsonl

_DOMAIN_TERMS = (
    "def ",
    "class ",
    "return ",
    "import ",
    "assert ",
    "traceback",
    "syntaxerror",
    "memoryerror",
    "eoferror",
    "runtimeerror",
    "pytorch",
    "torch",
    "tensor",
    "lora",
    "dpo",
    "grpo",
    "cispo",
    "rlvr",
    "cuda",
    "grad_clip",
    "grad_norm",
    "learning_rate",
    "batch_size",
    "max_seq_len",
    "perplexity",
    "ppl",
    "ast.parse",
    "worktree",
    "replace_line_range",
    "patch_config",
    "read_file_lines",
    "read_log_tail",
    "submit_candidate",
    "rollback_to_ancestor",
    "<tool_call>",
    "<think>",
    "algorithm",
    "complexity",
    "gradient",
    "optimizer",
    "tokenizer",
    "json",
    "python",
    "sha-256",
    "kv-cache",
    "gqa",
    "pin_memory",
    "梯度",
    "微调",
    "显存",
    "学习率",
    "余弦退火",
    "重复惩罚",
    "断点续训",
    "自修改",
    "工作树",
    "代码",
    "算法",
    "矩阵",
    "向量",
    "模型",
    "参数",
    "函数",
    "语法",
    "调试",
    "锚点",
)

_TRIVIA_PATTERNS = [
    re.compile(r"\bcapital of\b", re.I),
    re.compile(r"\b(who invented|who discovered|who wrote|who won)\b", re.I),
    re.compile(r"\b(tallest mountain|largest ocean|longest river|population of)\b", re.I),
    re.compile(r"\b(recipe|ingredients|tablespoon|teaspoon|preheat oven|cook for)\b", re.I),
    re.compile(r"\b(horoscope|astrology|zodiac|lucky number|tell me a joke)\b", re.I),
    re.compile(r"\b(box office|grammy|oscar|celebrity|movie star|album)\b", re.I),
    re.compile(r"(首都|哪一年成立|哪个朝代|创始人是谁|菜谱|怎么炒|怎么做菜|星座运势|讲个笑话|写一首诗|作一首诗|演员是谁|票房)"),
]

DEFAULT_DOMAIN_PROBES = [
    "def compute_step(batch_size: int, div: int = 2) -> int:\n    return max(1, batch_size // div)\n",
    "<think>\nObserve: batch_halve.py line 2 uses min instead of max. Cause: clamp shrinks batch size to 1. Plan: call replace_line_range.\n</think>",
    "RuntimeError: CUDA out of memory in train_lora.py at batch_size=8. Patch batch_size=4 and accumulation_steps=2.",
    "<think>\nInspect train.log for nan loss and grad_norm inf before halving learning_rate.\n</think>",
]

DEFAULT_TRIVIA_PROBES = [
    "What is the capital city of France and what famous tower stands there?",
    "请给我讲一个关于夏天的搞笑笑话，再写一首七言绝句。",
    "Who won the FIFA World Cup in 1998 and which stadium hosted the final?",
    "今天晚饭想做红烧肉，需要准备哪些家常食材和调味料？",
]


def classify_trivia_vs_domain(text: str) -> dict:
    """Distinguish irrelevant general trivia/chit-chat from useful coding/debugging/technical text."""
    raw = text or ""
    lower = raw.lower()
    domain_hits = sum(1 for tok in _DOMAIN_TERMS if tok in lower)
    trivia_hits = sum(1 for pat in _TRIVIA_PATTERNS if pat.search(raw))
    is_trivia = bool(trivia_hits > 0 and (domain_hits == 0 or trivia_hits >= domain_hits + 2))
    return {"is_trivia": is_trivia, "domain_hits": domain_hits, "trivia_hits": trivia_hits}


def filter_trivia_from_anchor_jsonl(
    source_jsonl: Path,
    clean_out: Path,
    quarantine_out: Path | None = None,
    max_samples: int | None = None,
) -> dict:
    """Filter general trivia out of anchor JSONL while keeping tools, code, math, and technical QA."""
    source_jsonl = Path(source_jsonl)
    clean_rows: list[dict] = []
    quarantined_lines: list[str] = []
    total = 0

    for raw_ln in source_jsonl.read_text(encoding="utf-8").splitlines():
        if not raw_ln.strip():
            continue
        if max_samples is not None and total >= max_samples:
            break
        total += 1
        try:
            row = json.loads(raw_ln)
        except Exception:
            quarantined_lines.append(json.dumps({"reason": "invalid_json", "raw": raw_ln}, ensure_ascii=False) + "\n")
            continue

        rec = format_sft_record(row)
        convs = rec["conversations"]
        if not convs:
            quarantined_lines.append(json.dumps({"reason": "empty_conversations", "row": row}, ensure_ascii=False) + "\n")
            continue

        has_tools = any(bool(m.get("tools") or m.get("tool_calls")) or "<tool_call>" in str(m.get("content", "")) for m in convs)
        if has_tools:
            clean_rows.append(rec)
            continue

        full_text = "\n".join(str(m.get("content", "")) for m in convs)
        verdict = classify_trivia_vs_domain(full_text)
        if verdict["is_trivia"]:
            quarantined_lines.append(json.dumps({"reason": "off_domain_trivia", **verdict, "row": rec}, ensure_ascii=False) + "\n")
        else:
            clean_rows.append(rec)

    written = write_sft_jsonl(clean_out, clean_rows)
    if quarantine_out is not None:
        atomic_write_text(quarantine_out, "".join(quarantined_lines))
    return {
        "source_path": str(source_jsonl),
        "clean_path": str(clean_out),
        "quarantine_path": str(quarantine_out) if quarantine_out else None,
        "total_read": total,
        "kept": written,
        "quarantined_trivia": len(quarantined_lines),
    }


def rank_mlp_channels(
    domain_acts: list[float],
    trivia_acts: list[float],
    down_norms: list[float] | None = None,
    trivia_penalty: float = 0.5,
) -> list[int]:
    """Rank SwiGLU intermediate channels from highest domain utility to lowest (trivia-only/dead)."""
    n = len(domain_acts)
    scores: list[tuple[float, int]] = []
    for idx in range(n):
        dom = max(0.0, float(domain_acts[idx]))
        trv = max(0.0, float(trivia_acts[idx])) if idx < len(trivia_acts) else 0.0
        w_norm = max(1e-6, float(down_norms[idx])) if down_norms and idx < len(down_norms) else 1.0
        score = (dom * w_norm) / (1.0 + trivia_penalty * trv)
        scores.append((score, -idx))
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    return ranked


def profile_mlp_channel_activations(
    model,
    tokenizer,
    domain_texts: list[str],
    trivia_texts: list[str],
    device: str,
    max_length: int = 256,
) -> list[dict]:
    """Collect mean intermediate SwiGLU activations per layer on domain vs. trivia probe batches."""
    import torch  # type: ignore[import-not-found]

    model.eval()
    layers = getattr(getattr(model, "model", model), "layers", [])
    layer_stats: list[dict] = []

    def _run_probe(texts: list[str]) -> list[list[float]]:
        accum: list[torch.Tensor | None] = [None] * len(layers)
        counts = [0] * len(layers)
        hooks = []

        for l_idx, block in enumerate(layers):
            mlp = getattr(block, "mlp", None)
            down = getattr(mlp, "down_proj", None)
            if down is None:
                continue

            def _hook(_mod, inputs, _out, idx=l_idx):
                x = inputs[0].detach().float().abs().mean(dim=(0, 1))
                accum[idx] = x if accum[idx] is None else (accum[idx] + x)
                counts[idx] += 1

            hooks.append(down.register_forward_hook(_hook))

        try:
            with torch.no_grad():
                for txt in texts:
                    enc = tokenizer(txt, return_tensors="pt", truncation=True, max_length=max_length).to(device)
                    model(enc["input_ids"])
        finally:
            for h in hooks:
                h.remove()

        result: list[list[float]] = []
        for idx, acc in enumerate(accum):
            if acc is None or counts[idx] == 0:
                result.append([])
            else:
                result.append((acc / counts[idx]).cpu().tolist())
        return result

    dom_list = _run_probe(domain_texts)
    trv_list = _run_probe(trivia_texts)
    for l_idx, block in enumerate(layers):
        mlp = getattr(block, "mlp", None)
        down = getattr(mlp, "down_proj", None)
        norms: list[float] = []
        if down is not None:
            norms = down.weight.detach().float().norm(dim=0).cpu().tolist()
        dom = dom_list[l_idx]
        trv = trv_list[l_idx]
        layer_stats.append(
            {
                "layer": l_idx,
                "intermediate_size": len(dom) or len(norms),
                "domain_acts": dom,
                "trivia_acts": trv,
                "down_norms": norms,
                "ranked_indices": rank_mlp_channels(dom, trv, norms),
            }
        )
    return layer_stats


def reallocate_trivia_mlp_channels(
    weight_path: Path,
    tokenizer_path: Path,
    output_path: Path,
    profile_name: str | None = None,
    model_config: dict | None = None,
    reinit_ratio: float = 0.20,
    damp_scale: float = 0.02,
    domain_texts: list[str] | None = None,
    trivia_texts: list[str] | None = None,
) -> dict:
    """Re-initialize bottom trivia-dominated SwiGLU MLP channels while keeping all other layers intact."""
    import torch  # type: ignore[import-not-found]

    profile = get_profile(profile_name)
    model, tokenizer = load_minimind_model(Path(weight_path), Path(tokenizer_path), profile, model_config=model_config)
    stats = profile_mlp_channel_activations(
        model,
        tokenizer,
        domain_texts or DEFAULT_DOMAIN_PROBES,
        trivia_texts or DEFAULT_TRIVIA_PROBES,
        device=profile.device,
    )
    layers = model.model.layers
    reinit_counts: list[int] = []
    with torch.no_grad():
        for l_info in stats:
            ranked = l_info["ranked_indices"]
            if not ranked:
                continue
            cut_n = max(1, int(len(ranked) * min(0.5, max(0.01, reinit_ratio))))
            target_idx = torch.tensor(ranked[-cut_n:], dtype=torch.long, device=profile.device)
            mlp = layers[l_info["layer"]].mlp
            if not hasattr(mlp, "gate_proj"):
                continue
            mlp.gate_proj.weight[target_idx] = torch.randn_like(mlp.gate_proj.weight[target_idx]) * damp_scale
            mlp.up_proj.weight[target_idx] = torch.randn_like(mlp.up_proj.weight[target_idx]) * damp_scale
            mlp.down_proj.weight[:, target_idx] = 0.0
            reinit_counts.append(cut_n)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.half().cpu() if "cuda" in profile.device else v.cpu() for k, v in model.state_dict().items()}, output_path)
    arch = estimate_minimind_params(
        hidden_size=model.config.hidden_size,
        num_hidden_layers=model.config.num_hidden_layers,
        intermediate_size=model.config.intermediate_size,
        num_attention_heads=model.config.num_attention_heads,
        num_key_value_heads=model.config.num_key_value_heads,
        use_moe=bool(model.config.use_moe),
    )
    save_model_config(output_path.with_suffix(".config.json"), arch)
    return {
        "action": "reallocate_trivia_mlp_channels",
        "output_path": str(output_path),
        "reinit_ratio": reinit_ratio,
        "channels_reinitialized_per_layer": reinit_counts,
        "arch": arch,
    }


def prune_mlp_width(
    weight_path: Path,
    tokenizer_path: Path,
    output_path: Path,
    target_intermediate_size: int = 1216,
    profile_name: str | None = None,
    model_config: dict | None = None,
    domain_texts: list[str] | None = None,
    trivia_texts: list[str] | None = None,
) -> dict:
    """Physically slice SwiGLU MLP intermediate_size down using domain-vs-trivia channel rankings."""
    import torch  # type: ignore[import-not-found]
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # type: ignore[import-not-found]

    profile = get_profile(profile_name)
    src_model, tokenizer = load_minimind_model(Path(weight_path), Path(tokenizer_path), profile, model_config=model_config)
    aligned_inter = max(64, (int(target_intermediate_size) // 64) * 64)
    aligned_inter = min(aligned_inter, src_model.config.intermediate_size)

    stats = profile_mlp_channel_activations(
        src_model,
        tokenizer,
        domain_texts or DEFAULT_DOMAIN_PROBES,
        trivia_texts or DEFAULT_TRIVIA_PROBES,
        device=profile.device,
    )

    pruned_cfg = MiniMindConfig(
        hidden_size=src_model.config.hidden_size,
        num_hidden_layers=src_model.config.num_hidden_layers,
        intermediate_size=aligned_inter,
        num_attention_heads=src_model.config.num_attention_heads,
        num_key_value_heads=src_model.config.num_key_value_heads,
        use_moe=bool(src_model.config.use_moe),
    )
    dst_model = MiniMindForCausalLM(pruned_cfg).to(device=profile.device, dtype=src_model.lm_head.weight.dtype)
    src_sd = src_model.state_dict()
    dst_sd = dst_model.state_dict()
    for k, v in dst_sd.items():
        if k in src_sd and src_sd[k].shape == v.shape:
            dst_sd[k] = src_sd[k].clone()

    for l_info in stats:
        l_idx = l_info["layer"]
        ranked = l_info["ranked_indices"]
        if len(ranked) < aligned_inter:
            continue
        keep = torch.tensor(sorted(ranked[:aligned_inter]), dtype=torch.long, device=src_model.device)
        for proj in ("gate_proj", "up_proj"):
            gk = f"model.layers.{l_idx}.mlp.{proj}.weight"
            if gk in src_sd and gk in dst_sd:
                dst_sd[gk] = src_sd[gk][keep].clone()
        dk = f"model.layers.{l_idx}.mlp.down_proj.weight"
        if dk in src_sd and dk in dst_sd:
            dst_sd[dk] = src_sd[dk][:, keep].clone()

    dst_model.load_state_dict(dst_sd)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.half().cpu() if "cuda" in profile.device else v.cpu() for k, v in dst_model.state_dict().items()}, output_path)

    before_arch = estimate_minimind_params(
        hidden_size=src_model.config.hidden_size,
        num_hidden_layers=src_model.config.num_hidden_layers,
        intermediate_size=src_model.config.intermediate_size,
        num_attention_heads=src_model.config.num_attention_heads,
        num_key_value_heads=src_model.config.num_key_value_heads,
    )
    after_arch = estimate_minimind_params(
        hidden_size=pruned_cfg.hidden_size,
        num_hidden_layers=pruned_cfg.num_hidden_layers,
        intermediate_size=aligned_inter,
        num_attention_heads=pruned_cfg.num_attention_heads,
        num_key_value_heads=pruned_cfg.num_key_value_heads,
    )
    save_model_config(output_path.with_suffix(".config.json"), after_arch)
    return {
        "action": "prune_mlp_width",
        "output_path": str(output_path),
        "before": before_arch,
        "after": after_arch,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prune/reallocate general-trivia MLP capacity or filter trivia anchors")
    sub = parser.add_subparsers(dest="command", required=True)

    p_filt = sub.add_parser("filter-anchors", help="Filter off-domain trivia from anchor JSONL")
    p_filt.add_argument("--source", type=Path, required=True)
    p_filt.add_argument("--clean-out", type=Path, required=True)
    p_filt.add_argument("--quarantine-out", type=Path, default=None)
    p_filt.add_argument("--max-samples", type=int, default=None)

    p_re = sub.add_parser("reinit", help="Re-initialize bottom trivia-dominated MLP channels")
    p_re.add_argument("--weight-path", type=Path, required=True)
    p_re.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    p_re.add_argument("--output-path", type=Path, required=True)
    p_re.add_argument("--profile", type=str, default=None)
    p_re.add_argument("--reinit-ratio", type=float, default=0.20)

    p_pr = sub.add_parser("prune-mlp", help="Slice SwiGLU MLP intermediate_size physically")
    p_pr.add_argument("--weight-path", type=Path, required=True)
    p_pr.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    p_pr.add_argument("--output-path", type=Path, required=True)
    p_pr.add_argument("--target-intermediate-size", type=int, default=1216)
    p_pr.add_argument("--profile", type=str, default=None)

    args = parser.parse_args()
    if args.command == "filter-anchors":
        rep = filter_trivia_from_anchor_jsonl(args.source, args.clean_out, args.quarantine_out, args.max_samples)
    elif args.command == "reinit":
        rep = reallocate_trivia_mlp_channels(args.weight_path, args.tokenizer_path, args.output_path, args.profile, reinit_ratio=args.reinit_ratio)
    else:
        rep = prune_mlp_width(args.weight_path, args.tokenizer_path, args.output_path, args.target_intermediate_size, args.profile)
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
