from __future__ import annotations

import json
import math
from pathlib import Path

from selfchange.kernel.budgets import ComputeProfile
from selfchange.kernel.integrity import atomic_write_text

_ALL_TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def resolve_torch_dtype(profile: ComputeProfile):
    """Map ComputeProfile dtype string to a PyTorch dtype lazily."""
    import torch  # type: ignore[import-not-found]

    if profile.dtype == "float16" and "cuda" in profile.device:
        return torch.float16
    if profile.dtype == "bfloat16" and "cuda" in profile.device:
        return torch.bfloat16
    return torch.float32


def estimate_minimind_params(
    hidden_size: int = 768,
    num_hidden_layers: int = 8,
    intermediate_size: int | None = None,
    num_attention_heads: int = 8,
    num_key_value_heads: int = 4,
    vocab_size: int = 6400,
    use_moe: bool = False,
    num_experts: int = 4,
    tie_word_embeddings: bool = True,
) -> dict:
    """Compute exact parameter count and memory footprint for any MiniMindConfig shape without PyTorch."""
    eff_inter = intermediate_size if intermediate_size is not None else math.ceil(hidden_size * math.pi / 64) * 64
    head_dim = hidden_size // num_attention_heads
    embed = vocab_size * hidden_size
    lm_head = 0 if tie_word_embeddings else vocab_size * hidden_size
    attn_layer = (
        hidden_size * (num_attention_heads * head_dim)
        + 2 * hidden_size * (num_key_value_heads * head_dim)
        + (num_attention_heads * head_dim) * hidden_size
        + 2 * head_dim
    )
    block_norms = 2 * hidden_size
    mlp_single = 3 * hidden_size * eff_inter
    if use_moe:
        router = hidden_size * num_experts
        mlp_layer = router + num_experts * mlp_single
        mlp_active = router + mlp_single
    else:
        mlp_layer = mlp_single
        mlp_active = mlp_single
    total = embed + lm_head + num_hidden_layers * (attn_layer + block_norms + mlp_layer) + hidden_size
    active = embed + lm_head + num_hidden_layers * (attn_layer + block_norms + mlp_active) + hidden_size
    attn_total = num_hidden_layers * attn_layer
    mlp_total = num_hidden_layers * mlp_layer
    return {
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "intermediate_size": eff_inter,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "vocab_size": vocab_size,
        "use_moe": bool(use_moe),
        "total_params": total,
        "active_params": active,
        "fp32_mb": round(total * 4 / 1e6, 2),
        "fp16_mb": round(total * 2 / 1e6, 2),
        "int8_mb": round(total / 1e6, 2),
        "int4_mb": round(total * 0.55 / 1e6, 2),
        "embed_ratio": round(embed / max(1, total), 4),
        "attn_ratio": round(attn_total / max(1, total), 4),
        "mlp_ratio": round(mlp_total / max(1, total), 4),
    }


def save_model_config(path: Path, arch: dict) -> Path:
    """Persist compact MiniMind architecture config atomically alongside checkpoints."""
    target = Path(path)
    atomic_write_text(target, json.dumps(arch, indent=2, ensure_ascii=False) + "\n")
    return target


def load_model_config(path_or_weight: Path | None) -> dict | None:
    """Resolve model architecture metadata from an explicit JSON, sibling .config.json, or model_config.json."""
    if path_or_weight is None:
        return None
    p = Path(path_or_weight)
    candidates = [p] if p.suffix == ".json" else [p.with_suffix(".config.json"), p.parent / "model_config.json"]
    for cand in candidates:
        if cand.is_file():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "hidden_size" in data:
                    return data
            except Exception:
                pass
    return None


def apply_extended_lora(
    model,
    rank: int = 16,
    lora_alpha: float | None = None,
    target_all_linear: bool = False,
):
    """Attach LoRA adapters to square layers or all attention+MLP linear projections."""
    from torch import nn  # type: ignore[import-not-found]
    from model.model_lora import LoRA, apply_lora  # type: ignore[import-not-found]

    if not target_all_linear:
        apply_lora(model, rank=rank, lora_alpha=lora_alpha)
        return

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(sfx) for sfx in _ALL_TARGET_SUFFIXES):
            lora = LoRA(module.in_features, module.out_features, rank=rank, lora_alpha=lora_alpha).to(model.device)
            setattr(module, "lora", lora)
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            module.forward = forward_with_lora


def load_minimind_model(
    weight_path: Path | None,
    tokenizer_path: Path,
    profile: ComputeProfile,
    hidden_size: int | None = None,
    num_hidden_layers: int | None = None,
    use_moe: bool | None = None,
    lora_path: Path | None = None,
    lora_rank: int = 16,
    target_all_linear: bool = False,
    intermediate_size: int | None = None,
    num_attention_heads: int | None = None,
    num_key_value_heads: int | None = None,
    model_config: dict | None = None,
    quantize_int8: bool = False,
):
    """Load MiniMindForCausalLM and AutoTokenizer with profile-safe device, custom widths, and optional INT8."""
    import torch  # type: ignore[import-not-found]
    from transformers import AutoTokenizer  # type: ignore[import-not-found]
    from model.model_lora import load_lora  # type: ignore[import-not-found]
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # type: ignore[import-not-found]

    eff_cfg = dict(model_config) if model_config else (load_model_config(weight_path) or {})
    hs = hidden_size if hidden_size is not None else int(eff_cfg.get("hidden_size", profile.hidden_size))
    nl = num_hidden_layers if num_hidden_layers is not None else int(eff_cfg.get("num_hidden_layers", profile.num_hidden_layers))
    um = bool(use_moe) if use_moe is not None else bool(eff_cfg.get("use_moe", profile.use_moe))
    inter = intermediate_size if intermediate_size is not None else eff_cfg.get("intermediate_size")
    nh = num_attention_heads if num_attention_heads is not None else eff_cfg.get("num_attention_heads")
    nkv = num_key_value_heads if num_key_value_heads is not None else eff_cfg.get("num_key_value_heads")

    extra_kwargs = {
        k: int(v)
        for k, v in [("intermediate_size", inter), ("num_attention_heads", nh), ("num_key_value_heads", nkv)]
        if v is not None
    }
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    cfg = MiniMindConfig(
        hidden_size=hs,
        num_hidden_layers=nl,
        use_moe=um,
        max_seq_len=profile.max_total_len,
        **extra_kwargs,
    )
    model = MiniMindForCausalLM(cfg)
    if weight_path is not None and Path(weight_path).is_file():
        state_dict = torch.load(str(weight_path), map_location=profile.device)
        model.load_state_dict(state_dict, strict=False)
    if lora_path is not None and Path(lora_path).is_file():
        apply_extended_lora(model, rank=lora_rank, target_all_linear=target_all_linear)
        load_lora(model, str(lora_path))

    dtype = resolve_torch_dtype(profile)
    model = model.to(device=profile.device, dtype=dtype).eval()
    if quantize_int8 and "cpu" in profile.device:
        try:
            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        except Exception:
            pass
    return model, tokenizer


def normalize_messages_for_template(conversations: list[dict]) -> tuple[list[dict], list[dict] | None]:
    """Normalize SFTDataset JSON-string tools/tool_calls into Jinja-ready structures."""
    messages: list[dict] = []
    tools: list[dict] | None = None
    for raw in conversations:
        msg = dict(raw)
        if msg.get("role") == "system" and msg.get("tools"):
            raw_tools = msg.pop("tools")
            tools = json.loads(raw_tools) if isinstance(raw_tools, str) else raw_tools
        elif "tools" in msg and msg["tools"] is None:
            msg.pop("tools")
        if msg.get("tool_calls") and isinstance(msg["tool_calls"], str):
            msg["tool_calls"] = json.loads(msg["tool_calls"])
        elif "tool_calls" in msg and msg["tool_calls"] is None:
            msg.pop("tool_calls")
        if "reasoning_content" in msg and msg["reasoning_content"] is None:
            msg.pop("reasoning_content")
        messages.append(msg)
    return messages, tools


def restore_open_thinking_prefix(decoded: str, open_thinking: bool) -> str:
    """Restore the prompt-prefilled <think> opening tag so assistant outputs remain tag-balanced."""
    text = decoded or ""
    if open_thinking and "<think>" not in text and ("</think>" in text or text.strip()):
        return "<think>\n" + text
    return text


def generate_agent_reply(
    model,
    tokenizer,
    messages: list[dict],
    tools: list[dict] | None,
    max_new_tokens: int = 192,
    temperature: float = 0.2,
    top_p: float = 0.9,
    open_thinking: bool = True,
) -> str:
    """Render chat template and decode only newly generated response tokens."""
    import torch  # type: ignore[import-not-found]

    norm_messages, extracted_tools = normalize_messages_for_template(messages)
    effective_tools = tools if tools is not None else extracted_tools
    prompt_text = tokenizer.apply_chat_template(
        norm_messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=effective_tools or None,
        open_thinking=open_thinking,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True).to(model.device)
    do_sample = temperature > 0.0
    gen_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    with torch.no_grad():
        generated_ids = model.generate(**gen_kwargs)
    prompt_len = inputs["input_ids"].size(1)
    decoded = tokenizer.decode(generated_ids[0][prompt_len:], skip_special_tokens=False)
    return restore_open_thinking_prefix(decoded, open_thinking)
