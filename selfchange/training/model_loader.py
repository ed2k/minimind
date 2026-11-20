from __future__ import annotations

import json
from pathlib import Path

from selfchange.kernel.budgets import ComputeProfile

_ALL_TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def resolve_torch_dtype(profile: ComputeProfile):
    """Map ComputeProfile dtype string to a PyTorch dtype lazily."""
    import torch  # type: ignore[import-not-found]

    if profile.dtype == "float16" and "cuda" in profile.device:
        return torch.float16
    if profile.dtype == "bfloat16" and "cuda" in profile.device:
        return torch.bfloat16
    return torch.float32


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
):
    """Load MiniMindForCausalLM and AutoTokenizer with profile-safe device and precision."""
    import torch  # type: ignore[import-not-found]
    from transformers import AutoTokenizer  # type: ignore[import-not-found]
    from model.model_lora import load_lora  # type: ignore[import-not-found]
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # type: ignore[import-not-found]

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    cfg = MiniMindConfig(
        hidden_size=profile.hidden_size if hidden_size is None else hidden_size,
        num_hidden_layers=profile.num_hidden_layers if num_hidden_layers is None else num_hidden_layers,
        use_moe=profile.use_moe if use_moe is None else bool(use_moe),
        max_seq_len=profile.max_total_len,
    )
    model = MiniMindForCausalLM(cfg)
    if weight_path is not None and Path(weight_path).is_file():
        state_dict = torch.load(str(weight_path), map_location=profile.device)
        model.load_state_dict(state_dict, strict=False)
    if lora_path is not None and Path(lora_path).is_file():
        apply_extended_lora(model, rank=lora_rank, target_all_linear=target_all_linear)
        load_lora(model, str(lora_path))

    dtype = resolve_torch_dtype(profile)
    return model.to(device=profile.device, dtype=dtype).eval(), tokenizer


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
