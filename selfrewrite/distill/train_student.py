from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from selfrewrite.config import DOWNSIZE_RUNGS, atomic_write_text, compute_file_sha256, get_profile
from selfrewrite.tasks.datasets import load_jsonl_problems


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_NOISE_PATTERNS = ("np.random.randn", "np.random.normal", "torch.randn", "rng.gauss")


@dataclass
class StudentTrainConfig:
    """Configuration for offline downsized student specialization on harvested A100 bundles."""

    rung_id: str = "R2"
    base_model_id: str = ""
    bundle_dir: Path = Path("selfrewrite/workspace/distill_bundle/v1")
    output_dir: Path = Path("selfrewrite/workspace/students/R2")
    splits_dir: Path | None = None
    backend: str = "hf"  # "hf" | "mock"
    profile_name: str = "auto"
    max_seq_tokens: int = 4096
    primitive_ratio: float = 0.45
    trajectory_ratio: float = 0.35
    anchor_ratio: float = 0.20
    lora_rank: int = 32
    lora_alpha: int = 64
    sft_epochs: int = 2
    sft_lr: float = 1e-4
    dpo_epochs: int = 1
    dpo_lr: float = 5e-6
    dpo_beta: float = 0.12
    run_rlvr: bool = False
    rlvr_steps: int = 20
    rlvr_lr: float = 2e-6
    rlvr_group_size: int = 4
    batch_size: int = 1
    grad_accum_steps: int = 4
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.base_model_id and self.rung_id in DOWNSIZE_RUNGS:
            self.base_model_id = DOWNSIZE_RUNGS[self.rung_id].model_id
        self.bundle_dir = Path(self.bundle_dir)
        self.output_dir = Path(self.output_dir)
        if self.splits_dir is not None:
            self.splits_dir = Path(self.splits_dir)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["bundle_dir"] = str(self.bundle_dir)
        d["output_dir"] = str(self.output_dir)
        d["splits_dir"] = str(self.splits_dir) if self.splits_dir is not None else None
        return d


def verify_bundle_audit_gate(bundle_dir: Path) -> dict[str, Any]:
    """Fail loudly if quality_audit.json or manifest.json is missing or audit failed."""
    root = Path(bundle_dir).resolve()
    audit_path = root / "quality_audit.json"
    manifest_path = root / "manifest.json"
    if not audit_path.is_file():
        raise ValueError(f"Refused training: missing {audit_path}")
    if not manifest_path.is_file():
        raise ValueError(f"Refused training: missing {manifest_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if audit.get("passed") is not True:
        raise ValueError(
            f"Refused training: distillation bundle failed audit ({audit.get('num_violations')} violations): "
            f"{audit.get('violations', [])[:2]}"
        )
    return {"audit": audit, "manifest": manifest}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _approx_msg_tokens(messages: list[dict[str, str]]) -> int:
    chars = sum(len(str(m.get("content", ""))) + 24 for m in messages)
    return max(1, chars // 3)


def build_sft_curriculum(
    bundle_dir: Path,
    distill_train_problems: list[Any],
    config: StudentTrainConfig,
) -> list[dict[str, Any]]:
    """Build balanced primitive + gold trajectory + distill_train anchor SFT curriculum."""
    verify_bundle_audit_gate(bundle_dir)
    root = Path(bundle_dir).resolve()

    raw_prims = _read_jsonl(root / "sft_turn_primitives.jsonl")
    raw_trajs = _read_jsonl(root / "sft_gold_trajectories.jsonl")

    prims = [
        {"source": "primitive", "skill": r.get("skill", "unknown"), "task": r.get("task", "improve"), "messages": r["messages"]}
        for r in raw_prims
        if _approx_msg_tokens(r.get("messages", [])) <= config.max_seq_tokens
    ]
    trajs = [
        {"source": "gold_trajectory", "task": r.get("task", "improve"), "messages": r["messages"]}
        for r in raw_trajs
        if _approx_msg_tokens(r.get("messages", [])) <= config.max_seq_tokens
    ]

    anchors: list[dict[str, Any]] = []
    for p in distill_train_problems:
        q = p["question"] if isinstance(p, dict) else getattr(p, "question")
        a = p["answer"] if isinstance(p, dict) else getattr(p, "answer")
        msgs = [
            {"role": "user", "content": str(q)},
            {
                "role": "assistant",
                "content": f"<think>\nWorking through the problem carefully.\n</think>\n\\boxed{{{a}}}",
            },
        ]
        if _approx_msg_tokens(msgs) <= config.max_seq_tokens:
            anchors.append({"source": "distill_anchor", "task": "math_anchor", "messages": msgs})

    rng = random.Random(int(config.seed))
    rng.shuffle(prims)
    rng.shuffle(trajs)
    rng.shuffle(anchors)

    # Compute target counts preserving ratios cleanly when all pools are available.
    base_total = max(len(prims) + len(trajs), 1)
    target_total = max(base_total, int(round(base_total / max(0.1, config.primitive_ratio + config.trajectory_ratio))))
    n_prim = int(round(target_total * config.primitive_ratio)) if prims else 0
    n_traj = int(round(target_total * config.trajectory_ratio)) if trajs else 0
    if not prims and not trajs and anchors:
        n_anch = min(len(anchors), max(4, int(round(len(anchors) * config.anchor_ratio))))
    else:
        n_anch = int(round(target_total * config.anchor_ratio)) if anchors else 0

    def _take(pool: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        if not pool or count <= 0:
            return []
        return [dict(pool[i % len(pool)]) for i in range(count)]

    curriculum = _take(prims, max(n_prim, len(prims))) + _take(trajs, max(n_traj, len(trajs))) + _take(anchors, n_anch)
    rng.shuffle(curriculum)

    prep_dir = config.output_dir / "prepared"
    prep_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        prep_dir / "prepared_sft.jsonl",
        "\n".join(json.dumps(r, ensure_ascii=False) for r in curriculum) + ("\n" if curriculum else ""),
    )
    atomic_write_text(
        prep_dir / "train_curriculum_manifest.json",
        json.dumps(
            {
                "total": len(curriculum),
                "primitives": sum(1 for r in curriculum if r["source"] == "primitive"),
                "gold_trajectories": sum(1 for r in curriculum if r["source"] == "gold_trajectory"),
                "distill_anchors": sum(1 for r in curriculum if r["source"] == "distill_anchor"),
                "max_seq_tokens": config.max_seq_tokens,
            },
            indent=2,
        )
        + "\n",
    )
    return curriculum


def build_dpo_dataset(
    bundle_dir: Path,
    max_seq_tokens: int = 4096,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Load, validate, and filter DPO contrastive pairs from an audited bundle."""
    verify_bundle_audit_gate(bundle_dir)
    raw = _read_jsonl(Path(bundle_dir).resolve() / "dpo_contrastive_pairs.jsonl")
    valid: list[dict[str, Any]] = []
    for r in raw:
        prompt = r.get("prompt")
        chosen = str(r.get("chosen", "")).strip()
        rejected = str(r.get("rejected", "")).strip()
        if not prompt or not chosen or not rejected or chosen == rejected:
            continue
        approx_tok = _approx_msg_tokens(list(prompt) + [{"role": "assistant", "content": chosen}])
        if approx_tok > int(max_seq_tokens):
            continue
        valid.append(
            {
                "skill": str(r.get("skill", "unknown")),
                "task": str(r.get("task", "improve")),
                "prompt": list(prompt),
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    if output_dir is not None:
        prep_dir = Path(output_dir) / "prepared"
        prep_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            prep_dir / "prepared_dpo.jsonl",
            "\n".join(json.dumps(r, ensure_ascii=False) for r in valid) + ("\n" if valid else ""),
        )
    return valid


def compute_rlvr_turn_reward(
    reply: str,
    action_kind: str,
    sandbox_result: dict[str, Any] | None = None,
    paired_stats: dict[str, Any] | None = None,
    decision: str | None = None,
    reserved_stats: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Deterministic verifiable process + outcome reward for optional student RLVR stage."""
    if action_kind == "invalid":
        syntax_r = -0.5
    elif sandbox_result is not None:
        syntax_r = 0.2 if sandbox_result.get("ok") else -0.4
    else:
        syntax_r = 0.2

    has_noise = any(pat in (reply or "") for pat in _NOISE_PATTERNS)
    noise_r = -0.8 if has_noise else 0.15

    lower = (reply or "").lower()
    inspect_r = 0.25 if (
        action_kind == "forward" and '"mode": "read"' in lower
        or any(fn in lower for fn in ("outlier_ratios", "load_census", "load_pass", "mean_diff_direction"))
    ) else 0.0

    paired_r = 0.0
    if isinstance(paired_stats, dict) and "delta_hat" in paired_stats:
        paired_r = max(-1.0, min(1.5, 2.0 * float(paired_stats["delta_hat"])))

    decision_r = 0.0
    if decision is not None and isinstance(reserved_stats, dict):
        res_d = float(reserved_stats.get("delta_hat", 0.0))
        dec = decision.upper()
        if (dec == "SUBMIT" and res_d > 0.0) or (dec == "WITHDRAW" and res_d <= 0.0):
            decision_r = 0.6
        elif dec == "SUBMIT" and res_d < 0.0:
            decision_r = -0.6

    total = round(syntax_r + noise_r + inspect_r + paired_r + decision_r, 6)
    return {
        "valid_syntax": syntax_r,
        "no_random_noise": noise_r,
        "read_or_inspect": inspect_r,
        "paired_delta": paired_r,
        "calibrated_decision": decision_r,
        "total_reward": total,
    }


def _render_chat(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    return "\n".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages)


def _set_use_cache(model: Any, enabled: bool) -> None:
    cfg = getattr(model, "config", None)
    if cfg is not None:
        if hasattr(cfg, "use_cache"):
            cfg.use_cache = bool(enabled)
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None and hasattr(text_cfg, "use_cache"):
            text_cfg.use_cache = bool(enabled)


def _enable_cuda_grad_checkpointing(model: Any) -> None:
    is_cuda = any(getattr(p, "is_cuda", False) for p in model.parameters())
    if not is_cuda:
        return
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass


def _attach_lora_for_training(model: Any, config: StudentTrainConfig) -> tuple[Any, str]:
    """Attach PEFT LoRA adapters across attention+MLP/linear_attn layers, with PyTorch fallback."""
    _set_use_cache(model, False)
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "in_proj", "out_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b",
    ]
    try:
        from peft import LoraConfig, TaskType, get_peft_model  # type: ignore[import-not-found]

        lcfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(config.lora_rank),
            lora_alpha=int(config.lora_alpha),
            lora_dropout=0.05,
            target_modules=target_modules,
        )
        peft_model = get_peft_model(model, lcfg)
        _enable_cuda_grad_checkpointing(peft_model)
        return peft_model, "peft"
    except ImportError:
        import torch  # type: ignore[import-not-found]
        from torch import nn  # type: ignore[import-not-found]

        class _SimpleLoRALinear(nn.Module):
            def __init__(self, base: nn.Linear, r: int, alpha: int):
                super().__init__()
                self.base = base
                self.r = r
                self.scaling = float(alpha) / float(max(1, r))
                dev = base.weight.device
                dt = torch.float32
                self.lora_A = nn.Parameter(torch.randn(r, base.in_features, device=dev, dtype=dt) * (1.0 / math.sqrt(max(1, r))))
                self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=dt))
                self.disabled = False
                for p in self.base.parameters():
                    p.requires_grad_(False)

            def forward(self, x):
                out = self.base(x)
                if self.disabled:
                    return out
                delta = (x.float() @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
                return out + delta.to(dtype=out.dtype)

        for p in model.parameters():
            p.requires_grad_(False)
        for name, mod in list(model.named_modules()):
            if isinstance(mod, nn.Linear) and any(name.endswith(sfx) for sfx in target_modules) and "." in name:
                parent_path, attr = name.rsplit(".", 1)
                parent = model.get_submodule(parent_path)
                setattr(parent, attr, _SimpleLoRALinear(mod, int(config.lora_rank), int(config.lora_alpha)))

        from contextlib import contextmanager

        def _disable_adapter():
            @contextmanager
            def _ctx():
                wrapped = [m for m in model.modules() if isinstance(m, _SimpleLoRALinear)]
                for w in wrapped:
                    w.disabled = True
                try:
                    yield
                finally:
                    for w in wrapped:
                        w.disabled = False

            return _ctx()

        model.disable_adapter = _disable_adapter  # type: ignore[attr-defined]
        _enable_cuda_grad_checkpointing(model)
        return model, "builtin_lora"


def run_hf_sft_stage(model: Any, tokenizer: Any, sft_rows: list[dict[str, Any]], config: StudentTrainConfig) -> dict[str, Any]:
    """Run assistant-masked SFT training on GPU/CPU with cosine decay."""
    import torch  # type: ignore[import-not-found]
    import torch.nn.functional as F  # type: ignore[import-not-found]

    if not sft_rows:
        return {"stage": "sft", "steps": 0, "final_loss": None}

    dev = next(model.parameters()).device
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return {"stage": "sft", "steps": 0, "final_loss": None}
    optimizer = torch.optim.AdamW(trainable, lr=float(config.sft_lr), weight_decay=0.01)

    total_steps = max(1, len(sft_rows) * int(config.sft_epochs))
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    step = 0

    for _ in range(int(config.sft_epochs)):
        for row in sft_rows:
            step += 1
            msgs = list(row["messages"])
            full_txt = _render_chat(tokenizer, msgs)
            prefix_txt = _render_chat(tokenizer, msgs[:-1]) if len(msgs) > 1 else ""
            full_ids = tokenizer(full_txt, return_tensors="pt", truncation=True, max_length=int(config.max_seq_tokens))["input_ids"].to(dev)
            if full_ids.shape[1] < 2:
                continue
            pref_len = len(tokenizer(prefix_txt, truncation=True, max_length=int(config.max_seq_tokens))["input_ids"]) if prefix_txt else 0
            labels = full_ids.clone()
            labels[:, : min(pref_len, labels.shape[1] - 1)] = -100
            tgt = labels[:, 1:]
            if int((tgt != -100).sum().item()) == 0:
                continue

            out = model(input_ids=full_ids, use_cache=False)
            logits = out.logits[:, :-1, :].float()
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)
            loss_scaled = loss / max(1, int(config.grad_accum_steps))
            loss_scaled.backward()
            losses.append(float(loss.item()))

            if step % max(1, int(config.grad_accum_steps)) == 0 or step == total_steps:
                lr_mult = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * step / total_steps))
                for pg in optimizer.param_groups:
                    pg["lr"] = float(config.sft_lr) * lr_mult
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    return {"stage": "sft", "steps": len(losses), "final_loss": round(sum(losses[-8:]) / max(1, len(losses[-8:])), 6) if losses else None}


def _seq_logprob(model: Any, tokenizer: Any, prompt_msgs: list[dict[str, str]], completion: str, max_len: int, dev: Any):
    import torch  # type: ignore[import-not-found]
    import torch.nn.functional as F  # type: ignore[import-not-found]

    full_msgs = list(prompt_msgs) + [{"role": "assistant", "content": completion}]
    pref_txt = _render_chat(tokenizer, prompt_msgs)
    full_txt = _render_chat(tokenizer, full_msgs)
    full_ids = tokenizer(full_txt, return_tensors="pt", truncation=True, max_length=max_len)["input_ids"].to(dev)
    if full_ids.shape[1] < 2:
        return None
    pref_len = min(len(tokenizer(pref_txt, truncation=True, max_length=max_len)["input_ids"]), full_ids.shape[1] - 1)
    logits = model(input_ids=full_ids, use_cache=False).logits[:, :-1, :].float()
    targets = full_ids[:, 1:].unsqueeze(-1)
    tok_lp = F.log_softmax(logits, dim=-1).gather(-1, targets).squeeze(-1)
    mask = torch.zeros_like(tok_lp)
    mask[:, max(0, pref_len - 1) :] = 1.0
    return (tok_lp * mask).sum(dim=-1).squeeze(0)


def run_hf_dpo_stage(model: Any, tokenizer: Any, dpo_rows: list[dict[str, Any]], config: StudentTrainConfig) -> dict[str, Any]:
    """Run adapter-based DPO optimization without loading a duplicate reference model."""
    import torch  # type: ignore[import-not-found]
    import torch.nn.functional as F  # type: ignore[import-not-found]

    if not dpo_rows:
        return {"stage": "dpo", "steps": 0, "final_loss": None}

    dev = next(model.parameters()).device
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return {"stage": "dpo", "steps": 0, "final_loss": None}
    optimizer = torch.optim.AdamW(trainable, lr=float(config.dpo_lr), weight_decay=0.0)
    losses: list[float] = []
    beta = float(config.dpo_beta)

    for _ in range(int(config.dpo_epochs)):
        for idx, row in enumerate(dpo_rows, start=1):
            with torch.no_grad(), model.disable_adapter():
                ref_c = _seq_logprob(model, tokenizer, row["prompt"], row["chosen"], int(config.max_seq_tokens), dev)
                ref_r = _seq_logprob(model, tokenizer, row["prompt"], row["rejected"], int(config.max_seq_tokens), dev)
            if ref_c is None or ref_r is None:
                continue
            pi_c = _seq_logprob(model, tokenizer, row["prompt"], row["chosen"], int(config.max_seq_tokens), dev)
            pi_r = _seq_logprob(model, tokenizer, row["prompt"], row["rejected"], int(config.max_seq_tokens), dev)
            if pi_c is None or pi_r is None:
                continue
            adv = beta * ((pi_c - ref_c) - (pi_r - ref_r))
            loss = -F.logsigmoid(adv)
            (loss / max(1, int(config.grad_accum_steps))).backward()
            losses.append(float(loss.item()))
            if idx % max(1, int(config.grad_accum_steps)) == 0 or idx == len(dpo_rows):
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    return {"stage": "dpo", "steps": len(losses), "final_loss": round(sum(losses[-8:]) / max(1, len(losses[-8:])), 6) if losses else None}


def _prepare_rlvr_episode_sandbox(tmp_root: Path, ep: dict[str, Any]) -> Any:
    import shutil
    from selfrewrite.config import atomic_write_text
    from selfrewrite.sandbox import SealedPythonSandbox

    work_dir = tmp_root / "WORK"
    work_dir.mkdir(parents=True, exist_ok=True)
    snap_str = str(ep.get("work_snapshot_dir") or "").strip()
    if snap_str:
        snap_dir = Path(snap_str)
        if snap_dir.is_dir():
            for fname in ("census.jsonl", "census_summary.json", "notes.md", "practice_problems.jsonl"):
                src = snap_dir / fname
                if src.is_file():
                    shutil.copy2(src, work_dir / fname)

    weights_dir = tmp_root / "_sandbox_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    data_path = weights_dir / "weights_data.json"
    spec_path = weights_dir / "weights_spec.json"
    c_layer = ep.get("corruption_layer")
    layer_ids = sorted({0, 1, *( [int(c_layer)] if c_layer is not None else [] )})
    fake_tensors: dict[str, Any] = {}
    for l_idx in layer_ids:
        w = [[0.01 * ((r + c + 1) % 5) for c in range(4)] for r in range(4)]
        if ep.get("task") == "repair" and c_layer is not None and l_idx == int(c_layer):
            w[0] = [12.5, -12.5, 11.0, -11.0]
        fake_tensors[f"model.layers.{l_idx}.mlp.down_proj.weight"] = w
        fake_tensors[f"model.layers.{l_idx}.self_attn.o_proj.weight"] = [[0.02 for _ in range(4)] for _ in range(4)]
    atomic_write_text(data_path, json.dumps(fake_tensors))
    atomic_write_text(
        spec_path,
        json.dumps(
            {
                "mode": "json_tensors",
                "data_path": str(data_path.resolve()),
                "shapes": {k: [len(v), len(v[0])] for k, v in fake_tensors.items()},
            },
            indent=2,
        ),
    )
    return SealedPythonSandbox(work_dir=work_dir, weights_spec_path=spec_path, timeout_sec=5.0, max_output_chars=4000)


def run_hf_rlvr_stage(
    model: Any,
    tokenizer: Any,
    rlvr_rows: list[dict[str, Any]],
    config: StudentTrainConfig,
    kl_coef: float = 0.05,
) -> dict[str, Any]:
    """Run group-normalized (G>=2) sandbox-executed GRPO-lite refinement over rlvr_episodes."""
    import shutil
    import tempfile
    import torch  # type: ignore[import-not-found]
    from selfrewrite.config import SessionConfig
    from selfrewrite.protocol import parse_editor_reply, parse_submit_decision

    if not rlvr_rows:
        return {"stage": "rlvr", "steps": 0, "mean_reward": 0.0, "final_loss": None}

    dev = next(model.parameters()).device
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return {"stage": "rlvr", "steps": 0, "mean_reward": 0.0, "final_loss": None}

    optimizer = torch.optim.AdamW(trainable, lr=float(config.rlvr_lr), weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    max_steps = max(1, int(config.rlvr_steps))
    group_size = max(2, int(config.rlvr_group_size))
    pad_tok = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_new = min(96, max(16, int(config.max_seq_tokens) // 8))

    rewards: list[float] = []
    losses: list[float] = []
    for step_idx in range(min(max_steps, len(rlvr_rows))):
        ep = rlvr_rows[step_idx % len(rlvr_rows)]
        task = str(ep.get("task", "improve"))
        s_cfg = SessionConfig(task=task if task in ("improve", "repair") else "improve", protocol=2)
        if ep.get("prompt"):
            prompt_msgs = list(ep["prompt"])
        else:
            prompt_msgs = [
                {
                    "role": "system",
                    "content": "Respond cleanly with your reasoning in <think>...</think> and one python or forward block.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Task: {task}. Census summary: {json.dumps(ep.get('census_summary', {}))}. "
                        "Inspect or propose one safe verification step."
                    ),
                },
            ]

        pref_txt = _render_chat(tokenizer, prompt_msgs)
        enc = tokenizer(pref_txt, return_tensors="pt", truncation=True, max_length=max(16, int(config.max_seq_tokens) - max_new))
        input_ids = enc["input_ids"].to(dev)
        prompt_len = int(input_ids.shape[1])
        batch_ids = input_ids.expand(group_size, -1)

        model.eval()
        with torch.no_grad():
            gen_kwargs: dict[str, Any] = {
                "input_ids": batch_ids,
                "max_new_tokens": max_new,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.95,
            }
            if pad_tok is not None:
                gen_kwargs["pad_token_id"] = pad_tok
            gen = model.generate(**gen_kwargs)

        candidates: list[tuple[str, int, float]] = []
        tmp_dir = Path(tempfile.mkdtemp(prefix="sr_rlvr_"))
        sb = None
        try:
            sb = _prepare_rlvr_episode_sandbox(tmp_dir, ep)
            c_layer = ep.get("corruption_layer")
            res_delta = ep.get("reserved_delta")
            for g_idx in range(group_size):
                comp_ids = gen[g_idx][prompt_len:]
                comp_len = int(comp_ids.numel())
                if comp_len == 0:
                    continue
                completion = tokenizer.decode(comp_ids, skip_special_tokens=False)
                action = parse_editor_reply(completion, s_cfg)
                sb_res = None
                loc_bonus = 0.0
                if action.kind == "python":
                    sb_res = sb.run(action.payload)
                    sb.run("drop_edit()")
                    if task == "repair" and c_layer is not None:
                        tag_l = f"layers.{int(c_layer)}."
                        if tag_l in str((sb_res or {}).get("output", "")) or tag_l in completion:
                            loc_bonus = 0.3

                dec = None
                res_stats = None
                if action.kind == "final":
                    parsed_dec = parse_submit_decision(completion)
                    if parsed_dec in ("SUBMIT", "WITHDRAW"):
                        dec = parsed_dec
                        if res_delta is not None:
                            res_stats = {"delta_hat": float(res_delta)}
                        else:
                            expected = str(ep.get("expected_decision", "SUBMIT")).upper()
                            res_stats = {"delta_hat": 0.1 if expected == "SUBMIT" else -0.1}

                rew_dict = compute_rlvr_turn_reward(
                    reply=completion,
                    action_kind=action.kind,
                    sandbox_result=sb_res,
                    decision=dec,
                    reserved_stats=res_stats,
                )
                r_val = float(rew_dict["total_reward"]) + loc_bonus
                candidates.append((completion, comp_len, r_val))
        finally:
            if sb is not None:
                sb.close()
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if not candidates:
            continue

        cand_rewards = [c[2] for c in candidates]
        rewards.extend(cand_rewards)
        mean_r = sum(cand_rewards) / len(cand_rewards)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in cand_rewards) / len(cand_rewards))
        advs = [((r - mean_r) / (std_r + 1e-4)) if std_r > 1e-6 else r for r in cand_rewards]

        term_losses: list[Any] = []
        for (completion, comp_len, _), adv_val in zip(candidates, advs):
            with torch.no_grad(), model.disable_adapter():
                ref_lp = _seq_logprob(model, tokenizer, prompt_msgs, completion, int(config.max_seq_tokens), dev)
            if ref_lp is None:
                continue
            model.train()
            pi_lp = _seq_logprob(model, tokenizer, prompt_msgs, completion, int(config.max_seq_tokens), dev)
            if pi_lp is None:
                continue
            d_lp = (pi_lp - ref_lp.detach()) / float(max(1, comp_len))
            term_losses.append(-(adv_val * d_lp) + float(kl_coef) * (d_lp ** 2))

        if not term_losses:
            continue
        loss = sum(term_losses) / float(len(term_losses))
        loss.backward()
        losses.append(float(loss.item()))
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return {
        "stage": "rlvr",
        "steps": len(losses),
        "group_size": group_size,
        "mean_reward": round(sum(rewards) / max(1, len(rewards)), 6) if rewards else 0.0,
        "final_loss": round(sum(losses[-8:]) / max(1, len(losses[-8:])), 6) if losses else None,
    }


def merge_and_freeze_student(
    model: Any,
    tokenizer: Any,
    output_dir: Path,
    provenance: dict[str, Any],
    adapter_kind: str = "peft",
) -> Path:
    """Merge LoRA adapters back into standalone weights, freeze all parameters, and save."""
    import torch  # type: ignore[import-not-found]
    from torch import nn  # type: ignore[import-not-found]

    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if adapter_kind == "peft" and hasattr(model, "merge_and_unload"):
        merged = model.merge_and_unload()
    else:
        for mod_name, mod in list(model.named_modules()):
            if hasattr(mod, "lora_A") and hasattr(mod, "lora_B") and hasattr(mod, "base") and "." in mod_name:
                with torch.no_grad():
                    delta = (mod.lora_B @ mod.lora_A) * mod.scaling
                    mod.base.weight.data.add_(delta.to(dtype=mod.base.weight.dtype, device=mod.base.weight.device))
                parent_p, attr = mod_name.rsplit(".", 1)
                setattr(model.get_submodule(parent_p), attr, mod.base)
        merged = model

    _set_use_cache(merged, True)
    merged.eval()
    for p in merged.parameters():
        p.requires_grad_(False)
    merged.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    prov = {**provenance, "frozen_for_eval": True, "saved_at": _utc_now()}
    atomic_write_text(out / "student_provenance.json", json.dumps(prov, indent=2) + "\n")
    return out


def train_downsize_student(config: StudentTrainConfig) -> dict[str, Any]:
    """Run audited student SFT -> DPO (-> optional RLVR) -> merge & freeze pipeline."""
    bundle_meta = verify_bundle_audit_gate(config.bundle_dir)
    distill_probs: list[Any] = []
    if config.splits_dir is not None and (config.splits_dir / "distill_train.jsonl").is_file():
        distill_probs = load_jsonl_problems(config.splits_dir / "distill_train.jsonl")

    sft_rows = build_sft_curriculum(config.bundle_dir, distill_probs, config)
    dpo_rows = build_dpo_dataset(config.bundle_dir, max_seq_tokens=config.max_seq_tokens, output_dir=config.output_dir)
    rlvr_rows = _read_jsonl(config.bundle_dir / "rlvr_episodes.jsonl")

    provenance = {
        "rung_id": config.rung_id,
        "base_model_id": config.base_model_id,
        "bundle_dir": str(config.bundle_dir.resolve()),
        "bundle_sha256": bundle_meta["manifest"].get("sha256", {}),
        "curriculum_counts": {
            "sft_rows": len(sft_rows),
            "dpo_rows": len(dpo_rows),
            "rlvr_episodes": len(rlvr_rows),
        },
        "config": config.to_dict(),
    }

    if config.backend == "mock":
        config.output_dir.mkdir(parents=True, exist_ok=True)
        prov = {
            **provenance,
            "frozen_for_eval": True,
            "stages": {
                "sft": {"stage": "sft", "steps": len(sft_rows) * config.sft_epochs, "final_loss": 0.42},
                "dpo": {"stage": "dpo", "steps": len(dpo_rows) * config.dpo_epochs, "final_loss": 0.19},
                "rlvr": {"stage": "rlvr", "steps": config.rlvr_steps if config.run_rlvr else 0, "skipped": not config.run_rlvr},
            },
            "saved_at": _utc_now(),
        }
        atomic_write_text(config.output_dir / "student_provenance.json", json.dumps(prov, indent=2) + "\n")
        atomic_write_text(
            config.output_dir / "mock_student.json",
            json.dumps({"rung_id": config.rung_id, "specialized": True, "frozen_for_eval": True}, indent=2) + "\n",
        )
        return prov

    # Real HF PyTorch training on Colab A100/L4/T4
    import torch  # type: ignore[import-not-found]
    from transformers import AutoTokenizer  # type: ignore[import-not-found]
    from selfrewrite.backends.hf_gpu_backend import load_hf_model

    profile = get_profile(config.profile_name)
    dtype = torch.bfloat16 if profile.dtype == "bfloat16" else torch.float16
    tok = AutoTokenizer.from_pretrained(config.base_model_id, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    model = load_hf_model(
        config.base_model_id,
        torch_dtype=dtype,
        device_map="auto" if "cuda" in profile.device else "cpu",
        trust_remote_code=True,
    )
    lora_model, adapter_kind = _attach_lora_for_training(model, config)
    sft_res = run_hf_sft_stage(lora_model, tok, sft_rows, config)
    dpo_res = run_hf_dpo_stage(lora_model, tok, dpo_rows, config)
    rlvr_res = (
        run_hf_rlvr_stage(lora_model, tok, rlvr_rows, config)
        if config.run_rlvr
        else {"stage": "rlvr", "steps": 0, "skipped": True}
    )
    provenance["stages"] = {
        "sft": sft_res,
        "dpo": dpo_res,
        "rlvr": rlvr_res,
    }
    merge_and_freeze_student(lora_model, tok, config.output_dir, provenance, adapter_kind=adapter_kind)
    return provenance
