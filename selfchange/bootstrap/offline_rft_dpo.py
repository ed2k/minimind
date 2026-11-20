from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path

from selfchange.bootstrap.generate_traces import _CLEAN_FIXTURES, inject_ast_bug
from selfchange.kernel.budgets import ComputeProfile, get_profile
from selfchange.kernel.integrity import atomic_write_text
from selfchange.kernel.tool_runtime import SELFCHANGE_TOOLS, WorktreeToolExecutor
from selfchange.kernel.verifier import compute_rlvr_reward
from selfchange.mutable.agent_loop import run_agent_episode
from selfchange.mutable.data_curator import build_dpo_pairs, curate_rft_samples, mix_with_anchor
from selfchange.mutable.recovery_policy import propose_line_fix
from selfchange.training.self_train_runner import (
    promote_lora_adapter,
    run_training_stage,
    stage_from_weight_for_trainer,
)


def _scripted_rollout_model_factory(fname: str, broken_src: str, err_type: str, target_line: int, succeed: bool):
    turn = {"i": 0}

    def _model(_messages: list[dict], _tools: list[dict]) -> str:
        idx = turn["i"]
        turn["i"] += 1
        if not succeed:
            if idx == 0:
                return '<tool_call>{"name": "replace_line_range", "arguments": {"path": "' + fname + '", "start_line": 99, "end_line": 99, "lines": ["pass"]}}</tool_call>'
            return "<think>Failed to locate bug.</think>\nAborting."
        if idx == 0:
            return '<think>Read file lines first.</think>\n<tool_call>{"name": "read_file_lines", "arguments": {"path": "' + fname + '", "start": 1, "end": 20}}</tool_call>'
        if idx == 1:
            fix = propose_line_fix(broken_src.splitlines(), {"line": target_line, "error_type": err_type}) or {
                "start_line": 1,
                "end_line": 1,
                "lines": [broken_src.splitlines()[0]],
            }
            payload = {"name": "replace_line_range", "arguments": {"path": fname, "start_line": fix["start_line"], "end_line": fix["end_line"], "lines": fix["lines"]}}
            return f"<think>Apply single-line fix.</think>\n<tool_call>{json.dumps(payload)}</tool_call>"
        return '<think>Submit repaired candidate.</think>\n<tool_call>{"name": "submit_candidate", "arguments": {"hypothesis": "fixed ast bug"}}</tool_call>'

    return _model


def collect_offline_rollouts(
    worktree_dir: Path,
    model_fn_factory=None,
    num_tasks: int = 20,
    num_rollouts: int = 4,
    seed: int = 42,
) -> list[dict]:
    """Collect pass@K rollouts on procedural AST bug-repair tasks for offline RFT + DPO."""
    rng = random.Random(seed)
    trajectories: list[dict] = []
    cfg_path = Path(worktree_dir) / "selfchange" / "mutable" / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {"max_turns": 5}

    for task_idx in range(num_tasks):
        fname, clean_src, _ = _CLEAN_FIXTURES[task_idx % len(_CLEAN_FIXTURES)]
        variant = task_idx % 5
        broken_src, _fix = inject_ast_bug(clean_src, variant)
        target_line = 1 if variant == 0 else 2
        err_type = "syntaxerror" if variant == 0 else "assertionerror"

        for r_idx in range(num_rollouts):
            with tempfile.TemporaryDirectory() as tmp:
                wt = Path(tmp)
                m_dir = wt / "selfchange" / "mutable"
                m_dir.mkdir(parents=True, exist_ok=True)
                (m_dir / fname).write_text(broken_src, encoding="utf-8")
                executor = WorktreeToolExecutor(wt, autonomy_level=1)
                state = {"last_fault": err_type, "target_file": fname, "target_line": target_line, "log_tail": f"{err_type} in {fname}"}
                if model_fn_factory is None:
                    succeed = True if r_idx == 0 else (False if r_idx == 1 else rng.random() < 0.5)
                    mfn = _scripted_rollout_model_factory(fname, broken_src, err_type, target_line, succeed)
                else:
                    mfn = model_fn_factory(task_idx, r_idx, fname, broken_src)

                ep = run_agent_episode(state, SELFCHANGE_TOOLS, mfn, executor.dispatch, cfg)
                repaired = (m_dir / fname).read_text(encoding="utf-8").strip() == clean_src.strip()
                observed = any(c["name"] in ("read_file_lines", "read_log_tail", "get_system_state") for c in ep["tool_calls_made"])
                first_edit_ok = any(res.get("ok") for c, res in zip(ep["tool_calls_made"], ep["tool_results"]) if c["name"] in ("replace_line_range", "patch_config"))
                reward = compute_rlvr_reward(
                    ep["turn_outputs"],
                    ep["tool_calls_made"],
                    observed_before_edit=observed,
                    first_edit_valid=first_edit_ok,
                    task_passed=repaired,
                    boundary_violated=False,
                    unfinished=ep["unfinished"],
                    over_budget=False,
                    open_thinking=True,
                )
                trajectories.append(
                    {
                        "task_id": f"ast_{task_idx}",
                        "task_family": f"ast_v{variant}",
                        "passed": repaired,
                        "reward": reward,
                        "messages": ep["messages"],
                        "tool_calls": ep["tool_calls_made"],
                        "final_text": ep["final_text"],
                    }
                )
    return trajectories


def run_offline_rft_dpo_round(
    trajectories: list[dict],
    repo_root: Path,
    workspace_dir: Path,
    work_dir: Path,
    anchor_samples: list[dict],
    profile: ComputeProfile,
    from_weight: str = "warmstart_rung3",
    round_idx: int = 1,
    dry_run: bool = False,
) -> dict:
    """Curate offline RFT + DPO datasets from collected rollouts and run Stage 1.5 training."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    rft_samples = curate_rft_samples(trajectories, max_rep=0.15, max_per_family=50)
    mixed_rft = mix_with_anchor(rft_samples, anchor_samples, anchor_ratio=0.25)
    dpo_pairs = build_dpo_pairs(trajectories)

    rft_path = work_dir / f"rft_round_{round_idx}.jsonl"
    dpo_path = work_dir / f"dpo_round_{round_idx}.jsonl"
    atomic_write_text(rft_path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in mixed_rft))
    atomic_write_text(dpo_path, "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in dpo_pairs))

    rft_res = None
    dpo_res = None
    curr_weight = from_weight
    if mixed_rft:
        rft_save = f"offline_rft_r{round_idx}"
        rft_res = run_training_stage(
            stage="lora",
            repo_root=repo_root,
            workspace_dir=workspace_dir,
            data_path=rft_path,
            profile=profile,
            training_cfg={"epochs": 1, "learning_rate": 5e-5, "batch_size": profile.batch_size, "max_seq_len": profile.max_seq_len},
            from_weight=curr_weight,
            save_weight=rft_save,
            dry_run=dry_run,
        )
        if rft_res.ok:
            if not dry_run and rft_res.output_weight_path:
                base_pth = stage_from_weight_for_trainer(curr_weight, repo_root, workspace_dir, profile)
                if base_pth is not None:
                    moe_suffix = "_moe" if profile.use_moe else ""
                    merged_name = f"offline_rft_r{round_idx}_merged"
                    merged_path = (
                        workspace_dir
                        / "models"
                        / "candidates"
                        / f"{merged_name}_{profile.hidden_size}{moe_suffix}.pth"
                    )
                    promote_lora_adapter(
                        base_weight=base_pth,
                        lora_weight=Path(rft_res.output_weight_path),
                        merged_out=merged_path,
                        hidden_size=profile.hidden_size,
                        num_hidden_layers=profile.num_hidden_layers,
                        use_moe=profile.use_moe,
                    )
                    curr_weight = merged_name
                else:
                    curr_weight = rft_save
            else:
                curr_weight = rft_save

    if dpo_pairs:
        dpo_save = f"offline_dpo_r{round_idx}"
        dpo_res = run_training_stage(
            stage="dpo",
            repo_root=repo_root,
            workspace_dir=workspace_dir,
            data_path=dpo_path,
            profile=profile,
            training_cfg={"epochs": 1, "learning_rate": 4e-8, "dpo_beta": 0.15, "batch_size": profile.batch_size, "max_seq_len": profile.max_seq_len},
            from_weight=curr_weight,
            save_weight=dpo_save,
            dry_run=dry_run,
        )
        if dpo_res.ok:
            curr_weight = dpo_save

    return {
        "round": round_idx,
        "rft_count": len(rft_samples),
        "mixed_rft_count": len(mixed_rft),
        "dpo_pairs_count": len(dpo_pairs),
        "rft_path": str(rft_path),
        "dpo_path": str(dpo_path),
        "rft_ok": rft_res.ok if rft_res else False,
        "dpo_ok": dpo_res.ok if dpo_res else False,
        "final_weight_name": curr_weight,
    }


def _make_minimind_rollout_factory(model_path: Path, tokenizer_path: Path, profile: ComputeProfile):
    from selfchange.training.model_loader import generate_agent_reply, load_minimind_model

    model, tokenizer = load_minimind_model(model_path, tokenizer_path, profile)
    eos_tok = tokenizer.eos_token or "<|im_end|>"
    pad_tok = tokenizer.pad_token or "<|endoftext|>"

    def _factory(_task_idx: int, _r_idx: int, _fname: str, _broken_src: str):
        def _model(messages: list[dict], tools: list[dict]) -> str:
            raw = generate_agent_reply(
                model,
                tokenizer,
                messages,
                tools,
                max_new_tokens=min(160, profile.max_gen_len),
                temperature=0.7,
                top_p=0.9,
                open_thinking=True,
            )
            return raw.replace(eos_tok, "").replace(pad_tok, "").strip()

        return _model

    return _factory


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1.5 offline rejection sampling + contrastive DPO")
    parser.add_argument("--worktree", type=Path, default=Path("."))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--workspace-dir", type=Path, default=Path("selfchange/workspace"))
    parser.add_argument("--work-dir", type=Path, default=Path("selfchange/workspace/offline_rft_dpo"))
    parser.add_argument("--backend", choices=["scripted", "minimind"], default="scripted")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    parser.add_argument("--from-weight", type=str, default="warmstart_rung3")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--num-tasks", type=int, default=16)
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    profile = get_profile(args.profile)
    factory = None
    if args.backend == "minimind":
        if args.model_path is None:
            raise ValueError("--model-path is required when --backend minimind")
        factory = _make_minimind_rollout_factory(args.model_path, args.tokenizer_path, profile)

    outputs: list[dict] = []
    curr = args.from_weight
    for r in range(1, args.rounds + 1):
        trajs = collect_offline_rollouts(
            args.worktree,
            model_fn_factory=factory,
            num_tasks=args.num_tasks,
            num_rollouts=args.num_rollouts,
            seed=42 + r,
        )
        res = run_offline_rft_dpo_round(
            trajs,
            repo_root=args.repo_root,
            workspace_dir=args.workspace_dir,
            work_dir=args.work_dir,
            anchor_samples=[],
            profile=profile,
            from_weight=curr,
            round_idx=r,
            dry_run=args.dry_run,
        )
        curr = res["final_weight_name"]
        outputs.append(res)
    print(json.dumps({"rounds": outputs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
