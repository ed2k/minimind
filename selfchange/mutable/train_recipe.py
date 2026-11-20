from __future__ import annotations

SAFE_BOUNDS: dict[str, tuple[float, float, bool]] = {
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


def clamp_training_config(training_cfg: dict) -> dict:
    """Clamp numeric training hyperparameters strictly inside SAFE_BOUNDS."""
    out = dict(training_cfg)
    for key, (low, high, is_int) in SAFE_BOUNDS.items():
        if key not in out:
            continue
        val = max(low, min(high, float(out[key])))
        out[key] = int(round(val)) if is_int else val
    return out


def select_training_stage(
    metrics: dict,
    buffer_stats: dict,
    config: dict,
    safe_mode: bool = False,
) -> dict:
    """Choose next self-training stage and safe hyperparameters from buffer and metrics."""
    training = clamp_training_config(config.get("training", {}))
    if safe_mode:
        training["batch_size"] = max(1, int(training.get("batch_size", 2)) // 2)
        training["learning_rate"] = max(1e-8, float(training.get("learning_rate", 5e-5)) * 0.5)
        training["dpo_learning_rate"] = max(1e-9, float(training.get("dpo_learning_rate", 4e-8)) * 0.5)
        training["anchor_replay_ratio"] = max(0.35, float(training.get("anchor_replay_ratio", 0.25)))

    dpo_pairs = int(buffer_stats.get("dpo_pairs", 0))
    rft_samples = int(buffer_stats.get("rft_samples", 0))
    rl_ready = bool(metrics.get("rl_ready", False))

    if safe_mode and dpo_pairs >= 4:
        return {"stage": "dpo", "training": clamp_training_config(training), "reason": "safe_mode_dpo_recovery"}
    if dpo_pairs >= 8 and float(metrics.get("pass_at_1", 1.0)) < float(metrics.get("pass_at_4", 1.0)):
        return {"stage": "dpo", "training": clamp_training_config(training), "reason": "dpo_contrastive_sharpening"}
    if rl_ready and rft_samples >= 4 and not safe_mode:
        return {"stage": "rlvr", "training": clamp_training_config(training), "reason": "online_rlvr_enabled"}
    if rft_samples >= 8:
        return {"stage": "lora", "training": clamp_training_config(training), "reason": "rft_lora_update"}
    return {"stage": "none", "training": clamp_training_config(training), "reason": "insufficient_verified_buffer"}
