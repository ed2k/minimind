from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
from pathlib import Path

from selfchange.bootstrap.generate_traces import _CLEAN_FIXTURES, inject_ast_bug
from selfchange.kernel.budgets import get_profile
from selfchange.kernel.tool_runtime import SELFCHANGE_TOOLS, WorktreeToolExecutor
from selfchange.kernel.verifier import compute_rlvr_reward
from selfchange.mutable.agent_loop import parse_tool_calls
from selfchange.mutable.prompts import SYSTEM_PROMPT, format_fault_card, format_observation


def compute_group_advantages(rewards_list: list[float], num_generations: int) -> tuple[list[float], int]:
    """Group-normalize rewards; explicitly zero advantages when group std < 1e-6."""
    if num_generations <= 0 or not rewards_list:
        return [], 0
    advantages: list[float] = []
    zero_std_groups = 0
    for start in range(0, len(rewards_list), num_generations):
        group = rewards_list[start : start + num_generations]
        mean_r = sum(group) / len(group)
        var = sum((r - mean_r) ** 2 for r in group) / len(group)
        std_r = math.sqrt(var)
        if std_r < 1e-6:
            zero_std_groups += 1
            advantages.extend([0.0] * len(group))
        else:
            advantages.extend((r - mean_r) / (std_r + 1e-4) for r in group)
    return advantages, zero_std_groups


def build_procedural_rlvr_tasks(num_tasks: int = 16, seed: int = 42) -> list[dict]:
    """Generate procedural fault-drill and AST-repair tasks for online RLVR."""
    rng = random.Random(seed)
    tasks: list[dict] = []
    for idx in range(num_tasks):
        kind = idx % 3
        if kind == 0:
            bs = rng.choice([4, 8])
            tasks.append({"task_id": f"rl_oom_{idx}", "kind": "oom", "batch_size": bs, "state": {"last_fault": "oom", "training": {"batch_size": bs}, "log_tail": f"CUDA out of memory batch_size={bs}"}})
        elif kind == 1:
            lr = rng.choice([1e-4, 2e-4])
            tasks.append({"task_id": f"rl_nan_{idx}", "kind": "nan", "learning_rate": lr, "state": {"last_fault": "nan_loss", "training": {"learning_rate": lr}, "log_tail": f"loss: nan lr={lr}"}})
        else:
            fname, clean_src, _ = _CLEAN_FIXTURES[idx % len(_CLEAN_FIXTURES)]
            variant = idx % 5
            broken_src, _fix = inject_ast_bug(clean_src, variant)
            tasks.append({"task_id": f"rl_ast_{idx}", "kind": "ast", "fname": fname, "broken_src": broken_src, "clean_src": clean_src, "state": {"last_fault": "syntax_or_test_error", "target_file": fname, "log_tail": f"Error in {fname}"}})
    return tasks


def score_episode_rlvr(episode_meta: dict) -> float:
    """Compute deterministic RLVR reward without any external reward model."""
    return compute_rlvr_reward(
        turn_outputs=episode_meta.get("turn_outputs", []),
        tool_calls=episode_meta.get("tool_calls", []),
        observed_before_edit=bool(episode_meta.get("observed_before_edit", False)),
        first_edit_valid=bool(episode_meta.get("first_edit_valid", False)),
        task_passed=bool(episode_meta.get("task_passed", False)),
        boundary_violated=bool(episode_meta.get("boundary_violated", False)),
        unfinished=bool(episode_meta.get("unfinished", False)),
        over_budget=bool(episode_meta.get("over_budget", False)),
        open_thinking=bool(episode_meta.get("open_thinking", True)),
    )


def rollout_selfchange_episode(
    rollout_engine,
    tokenizer,
    task: dict,
    max_turns: int = 4,
    max_new_tokens: int = 256,
    thinking_ratio: float = 0.5,
    device: str = "cpu",
) -> dict:
    """Run one multi-turn rollout against an isolated temp WorktreeToolExecutor."""
    import torch  # type: ignore[import-not-found]

    open_thinking = random.random() < thinking_ratio
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_fault_card(task["state"], task["state"].get("log_tail", ""))},
    ]
    tools = SELFCHANGE_TOOLS[:8] + SELFCHANGE_TOOLS[-2:]

    with tempfile.TemporaryDirectory() as tmp:
        wt = Path(tmp)
        m_dir = wt / "selfchange" / "mutable"
        m_dir.mkdir(parents=True, exist_ok=True)
        if task["kind"] == "ast":
            (m_dir / task["fname"]).write_text(task["broken_src"], encoding="utf-8")
        else:
            (m_dir / "config.json").write_text(json.dumps({"training": task["state"].get("training", {})}), encoding="utf-8")
        executor = WorktreeToolExecutor(wt, autonomy_level=1)

        prompt_ids: list[int] | None = None
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_old_logps: list[float] = []
        turn_outputs: list[str] = []
        tool_calls_made: list[dict] = []
        observed_before_edit = False
        first_edit_valid: bool | None = None
        unfinished = True

        for _ in range(max_turns):
            context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=open_thinking)
            if prompt_ids is None:
                prompt_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
            input_ids = torch.tensor([prompt_ids + response_ids], device=device)
            rollout_res = rollout_engine.rollout(
                prompt_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                num_generations=1,
                max_new_tokens=max_new_tokens,
                temperature=0.8,
            )
            valid_len = int(rollout_res.completion_mask[0].sum().item())
            new_ids = rollout_res.completion_ids[0, :valid_len].tolist()
            new_logps = rollout_res.per_token_logps[0, :valid_len].tolist()
            new_text = rollout_res.completions[0]
            turn_outputs.append(new_text)
            response_ids.extend(new_ids)
            response_mask.extend([int(t != tokenizer.eos_token_id) for t in new_ids])
            response_old_logps.extend(new_logps)

            calls, _ = parse_tool_calls(new_text)
            if not calls:
                unfinished = False
                break

            assistant_msg = {"role": "assistant", "content": new_text}
            messages.append(assistant_msg)
            for call in calls:
                name, args = call["name"], call["arguments"]
                if name in ("get_system_state", "read_log_tail", "read_file_lines") and first_edit_valid is None:
                    observed_before_edit = True
                res = executor.dispatch(name, args)
                if first_edit_valid is None and name in ("patch_config", "replace_line_range", "patch_file_replace"):
                    first_edit_valid = bool(res.get("ok"))
                tool_calls_made.append(call)
                messages.append({"role": "tool", "content": format_observation(name, res)})
                if name in ("submit_candidate", "rollback_to_ancestor") and res.get("ok"):
                    unfinished = False

            marker = f"<|obs_{len(response_ids)}|>"
            assistant_msg["content"] += marker
            marked = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=unfinished, tools=tools, open_thinking=open_thinking)
            assistant_msg["content"] = new_text
            _, found, obs = marked.partition(marker)
            obs_ids = tokenizer(obs, add_special_tokens=False)["input_ids"] if found else []
            if new_ids and new_ids[-1] == tokenizer.eos_token_id and obs_ids[:1] == [tokenizer.eos_token_id]:
                obs_ids = obs_ids[1:]
            response_ids.extend(obs_ids)
            response_mask.extend([0] * len(obs_ids))
            response_old_logps.extend([0.0] * len(obs_ids))
            if not unfinished:
                break

        if task["kind"] == "oom":
            cfg = json.loads((m_dir / "config.json").read_text(encoding="utf-8"))
            task_passed = int(cfg.get("training", {}).get("batch_size", task["batch_size"])) < task["batch_size"]
        elif task["kind"] == "nan":
            cfg = json.loads((m_dir / "config.json").read_text(encoding="utf-8"))
            task_passed = float(cfg.get("training", {}).get("learning_rate", task["learning_rate"])) < task["learning_rate"]
        else:
            task_passed = (m_dir / task["fname"]).read_text(encoding="utf-8").strip() == task["clean_src"].strip()

    reward = score_episode_rlvr(
        {
            "turn_outputs": turn_outputs,
            "tool_calls": tool_calls_made,
            "observed_before_edit": observed_before_edit,
            "first_edit_valid": bool(first_edit_valid),
            "task_passed": task_passed,
            "boundary_violated": False,
            "unfinished": unfinished,
            "over_budget": False,
            "open_thinking": open_thinking,
        }
    )
    return {
        "prompt_ids": prompt_ids or [],
        "response_ids": response_ids,
        "response_mask": response_mask,
        "response_old_logps": response_old_logps,
        "reward": reward,
        "task_passed": task_passed,
    }


def run_self_agent_training(args) -> dict:
    """Execute self-contained RLVR (GRPO/CISPO) training without an external reward model."""
    import torch  # type: ignore[import-not-found]
    import torch.nn.functional as F  # type: ignore[import-not-found]
    from torch import optim  # type: ignore[import-not-found]
    from contextlib import nullcontext
    from model.model_minimind import MiniMindConfig  # type: ignore[import-not-found]
    from trainer.rollout_engine import compute_per_token_logps, create_rollout_engine  # type: ignore[import-not-found]
    from trainer.trainer_utils import init_model, lm_checkpoint, setup_seed  # type: ignore[import-not-found]
    from selfchange.training.model_loader import resolve_torch_dtype

    profile = get_profile(args.profile)
    setup_seed(42)
    device = args.device or profile.device
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(ln) for ln in Path(args.data_path).read_text(encoding="utf-8").splitlines() if ln.strip()] if args.data_path and Path(args.data_path).is_file() else build_procedural_rlvr_tasks(args.num_tasks)

    if args.dry_run:
        rewards = [1.0 if i % 2 == 0 else -0.5 for i in range(len(tasks) * args.num_generations)]
        advs, zero_groups = compute_group_advantages(rewards, args.num_generations)
        return {"ok": True, "dry_run": True, "tasks": len(tasks), "zero_std_groups": zero_groups, "mean_adv": round(sum(advs) / max(1, len(advs)), 6)}

    lm_config = MiniMindConfig(hidden_size=profile.hidden_size, num_hidden_layers=profile.num_hidden_layers, max_seq_len=profile.max_total_len, use_moe=profile.use_moe)
    model, tokenizer = init_model(lm_config, args.from_weight, tokenizer_path=str(args.tokenizer_path), save_dir=str(save_dir), device=device)
    ref_model, _ = init_model(lm_config, args.from_weight, tokenizer_path=str(args.tokenizer_path), save_dir=str(save_dir), device=device)
    ref_model = ref_model.eval().requires_grad_(False)

    dtype = resolve_torch_dtype(profile)
    autocast_ctx = nullcontext() if "cpu" in device else torch.cuda.amp.autocast(dtype=dtype)
    rollout_engine = create_rollout_engine(engine_type="torch", policy_model=model, tokenizer=tokenizer, device=device, autocast_ctx=autocast_ctx)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir=str(args.checkpoints_dir)) if args.from_resume == 1 else None
    if ckp_data:
        model.load_state_dict(ckp_data["model"])
        optimizer.load_state_dict(ckp_data["optimizer"])

    total_zero_groups = 0
    mean_reward = 0.0
    for step, task in enumerate(tasks, start=1):
        episodes = [rollout_selfchange_episode(rollout_engine, tokenizer, task, max_turns=4, max_new_tokens=profile.max_gen_len, thinking_ratio=args.thinking_ratio, device=device) for _ in range(args.num_generations)]
        rewards_list = [ep["reward"] for ep in episodes]
        mean_reward = sum(rewards_list) / max(1, len(rewards_list))
        adv_list, zero_groups = compute_group_advantages(rewards_list, args.num_generations)
        total_zero_groups += zero_groups

        max_len = max((len(ep["prompt_ids"]) + len(ep["response_ids"]) for ep in episodes), default=2)
        max_len = min(max_len, profile.max_total_len)
        packed_ids, packed_masks, packed_old = [], [], []
        for ep in episodes:
            ids = (ep["prompt_ids"] + ep["response_ids"])[-max_len:]
            mask = ([0] * len(ep["prompt_ids"]) + ep["response_mask"])[-max_len:]
            old_lp = ([0.0] * max(len(ep["prompt_ids"]) - 1, 0) + ep["response_old_logps"])[-(len(ids) - 1) :]
            pad = max_len - len(ids)
            packed_ids.append(ids + [tokenizer.pad_token_id] * pad)
            packed_masks.append(mask + [0] * pad)
            packed_old.append(old_lp + [0.0] * ((max_len - 1) - len(old_lp)))

        input_ids = torch.tensor(packed_ids, device=device)
        full_mask = (input_ids != tokenizer.pad_token_id).long()
        comp_mask = torch.tensor(packed_masks, device=device, dtype=torch.float32)[:, 1:]
        old_lps = torch.tensor(packed_old, device=device, dtype=torch.float32)
        advs = torch.tensor(adv_list, device=device, dtype=torch.float32).unsqueeze(1)

        with autocast_ctx:
            logits = model(input_ids, attention_mask=full_mask).logits[:, :-1, :]
            per_tok_lp = F.log_softmax(logits, dim=-1).gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            ref_lp = compute_per_token_logps(ref_model, input_ids, input_ids.size(1) - 1, attention_mask=full_mask)

        kl_div = ref_lp - per_tok_lp
        per_tok_kl = torch.exp(kl_div) - kl_div - 1
        ratio = torch.exp(per_tok_lp - old_lps)
        if args.loss_type == "cispo":
            per_tok_loss = -(torch.clamp(ratio, max=5.0).detach() * advs * per_tok_lp - args.beta * per_tok_kl)
        else:
            per_tok_loss = -(torch.min(ratio * advs, torch.clamp(ratio, 0.8, 1.2) * advs) - args.beta * per_tok_kl)

        counts = comp_mask.sum(dim=1).clamp(min=1)
        loss = ((per_tok_loss * comp_mask).sum(dim=1) / counts).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        if step % max(1, args.save_interval) == 0 or step == len(tasks):
            out_path = save_dir / f"{args.save_weight}_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
            torch.save({k: v.half().cpu() if "cuda" in device else v.cpu() for k, v in model.state_dict().items()}, out_path)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, epoch=0, step=step, save_dir=str(args.checkpoints_dir))
            rollout_engine.update_policy(model)

    return {"ok": True, "tasks": len(tasks), "mean_reward": round(mean_reward, 4), "zero_std_groups": total_zero_groups}


def main() -> int:
    parser = argparse.ArgumentParser(description="Reward-Model-Optional SelfChange Agentic RLVR (GRPO/CISPO)")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_dir", type=Path, default=Path("selfchange/workspace/models/candidates"))
    parser.add_argument("--checkpoints_dir", type=Path, default=Path("selfchange/workspace/checkpoints"))
    parser.add_argument("--tokenizer_path", type=Path, default=Path("model"))
    parser.add_argument("--save_weight", type=str, default="self_agent")
    parser.add_argument("--from_weight", type=str, default="full_sft")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--num_tasks", type=int, default=8)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-7)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss_type", choices=["cispo", "grpo"], default="cispo")
    parser.add_argument("--thinking_ratio", type=float, default=0.5)
    parser.add_argument("--save_interval", type=int, default=4)
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1])
    parser.add_argument("--reward_model_path", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    res = run_self_agent_training(args)
    print(json.dumps(res, indent=2))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
