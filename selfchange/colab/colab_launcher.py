from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from selfchange.kernel.autonomous_cycle import CycleConfig
from selfchange.kernel.budgets import ComputeProfile, get_profile
from selfchange.kernel.colab_sync import ColabDriveSync
from selfchange.kernel.integrity import KERNEL_MANIFEST_FILENAME, atomic_write_text
from selfchange.kernel.registry import LineageNode
from selfchange.kernel.supervisor import CycleContext, Supervisor
from selfchange.kernel.tool_runtime import SELFCHANGE_TOOLS, WorktreeToolExecutor
from selfchange.kernel.worktree import WorktreeHandle
from selfchange.mutable.agent_loop import run_agent_episode
from selfchange.colab.autonomous_engine import (
    SelfImprovingCycleRunner,
    build_exploration_tasks,
    diagnose_weakness_state,
    evaluate_model_strict_suite,
    resolve_local_train_base_pth,
    run_multi_rollout_exploration,
)
from selfchange.training.model_loader import load_model_config

_TOOLS_BY_NAME = {t["function"]["name"]: t for t in SELFCHANGE_TOOLS}


def _compact_tools(names: list[str]) -> list[dict]:
    selected: list[dict] = []
    for name in names:
        tool = _TOOLS_BY_NAME.get(name)
        if not tool:
            continue
        fn = tool["function"]
        selected.append({"type": "function", "function": {"name": fn["name"], "parameters": fn["parameters"]}})
    return selected


def _is_real_weight_file(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with path.open("rb") as f:
            return f.read(1) != b"{"
    except OSError:
        return False


def _torch_available() -> bool:
    try:
        __import__("torch")
        return True
    except ImportError:
        return False


def _resolve_gguf_candidate(workspace_dir: Path) -> tuple[Path, str] | None:
    gguf_dir = workspace_dir / "promoted_models" / "gguf"
    meta_file = gguf_dir / "latest_gguf.json"
    if meta_file.is_file():
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            raw = meta.get("champion_gguf") or (meta.get("selected_artifact") or {}).get("path")
            if raw:
                p = Path(raw)
                if not p.is_file():
                    p = gguf_dir / p.name
                if _is_real_weight_file(p):
                    return p.resolve(), "gguf_latest"
        except Exception:
            pass
    champ_gguf = gguf_dir / "champion.gguf"
    if _is_real_weight_file(champ_gguf):
        return champ_gguf.resolve(), "gguf_champion"
    if gguf_dir.is_dir():
        for g in sorted(gguf_dir.glob("*.gguf")):
            if _is_real_weight_file(g):
                return g.resolve(), f"gguf_{g.stem}"
    return None


def resolve_model_checkpoint(
    repo_root: Path,
    workspace_dir: Path,
    profile: ComputeProfile,
    explicit_model_path: Path | None = None,
) -> tuple[Path | None, str]:
    """Resolve the best available MiniMind checkpoint (supporting PyTorch .pth and zero-PyTorch .gguf)."""
    repo_root = Path(repo_root).resolve()
    workspace_dir = Path(workspace_dir).resolve()
    moe_suffix = "_moe" if profile.use_moe else ""
    hs = profile.hidden_size
    has_torch = _torch_available()

    champ_file = workspace_dir / "champion.json"
    if champ_file.is_file():
        try:
            champ_w = json.loads(champ_file.read_text(encoding="utf-8")).get("champion", {}).get("weight_path")
            if champ_w:
                cand_champ = Path(champ_w)
                if not cand_champ.is_absolute():
                    cand_champ = workspace_dir / cand_champ
                if _is_real_weight_file(cand_champ):
                    return cand_champ.resolve(), "champion"
        except Exception:
            pass

    if explicit_model_path is not None and str(explicit_model_path) != "auto":
        cand_exp = Path(explicit_model_path)
        if not cand_exp.is_absolute():
            cand_exp = repo_root / cand_exp
        if _is_real_weight_file(cand_exp):
            return cand_exp.resolve(), "explicit"

    if not has_torch:
        gguf_hit = _resolve_gguf_candidate(workspace_dir)
        if gguf_hit is not None:
            return gguf_hit

    candidates = [
        ("offline_dpo_r1", workspace_dir / "models" / "candidates" / f"offline_dpo_r1_{hs}{moe_suffix}.pth"),
        ("offline_rft_r1_merged", workspace_dir / "models" / "candidates" / f"offline_rft_r1_merged_{hs}{moe_suffix}.pth"),
        ("warmstart_rung3", workspace_dir / "models" / "candidates" / f"warmstart_rung3_{hs}{moe_suffix}.pth"),
        ("warmstart_rung2", workspace_dir / "models" / "candidates" / f"warmstart_rung2_{hs}{moe_suffix}.pth"),
        ("warmstart_rung1", workspace_dir / "models" / "candidates" / f"warmstart_rung1_{hs}{moe_suffix}.pth"),
        ("base_minimind3", workspace_dir / "models" / "base" / f"minimind3_{hs}{moe_suffix}.pth"),
        ("out_full_sft", repo_root / "out" / f"full_sft_{hs}{moe_suffix}.pth"),
    ]
    for label, path in candidates:
        if _is_real_weight_file(path):
            return path.resolve(), label

    gguf_hit = _resolve_gguf_candidate(workspace_dir)
    if gguf_hit is not None:
        return gguf_hit
    return None, "none"


class MiniMindProposerSession:
    """Reusable MiniMind-3 session supporting PyTorch .pth and zero-PyTorch llama.cpp .gguf checkpoints."""

    def __init__(
        self,
        initial_model_path: Path,
        tokenizer_path: Path,
        profile: ComputeProfile,
        autonomy_level: int = 1,
        workspace_dir: Path | None = None,
        enable_exploration: bool = True,
        llama_bin_dir: Path | None = None,
        llama_base_url: str | None = None,
    ):
        self.initial_model_path = Path(initial_model_path).resolve()
        self.tokenizer_path = Path(tokenizer_path).resolve()
        self.profile = profile
        self.autonomy_level = autonomy_level
        self.workspace_dir = Path(workspace_dir).resolve() if workspace_dir is not None else None
        self.enable_exploration = enable_exploration
        self.llama_bin_dir = Path(llama_bin_dir).resolve() if llama_bin_dir is not None else None
        self.llama_base_url = llama_base_url
        self.loaded_path: Path | None = None
        self.model = None
        self.tokenizer = None
        self._llama_server = None
        self.model_config: dict | None = load_model_config(self.initial_model_path)
        self.last_scorecard: dict | None = None
        self.last_exploration: dict | None = None
        self._scorecard_cache: dict[tuple[str, int], dict] = {}

    @property
    def is_gguf_mode(self) -> bool:
        if self.loaded_path is not None:
            return self.loaded_path.suffix.lower() == ".gguf"
        return self.initial_model_path.suffix.lower() == ".gguf"

    def close(self) -> None:
        if self._llama_server is not None:
            self._llama_server.close()
            self._llama_server = None
            if self.loaded_path is not None and self.loaded_path.suffix.lower() == ".gguf":
                self.model = None

    def _ensure_loaded(self, target_path: Path | None = None) -> None:
        desired = (
            Path(target_path).resolve()
            if (target_path is not None and _is_real_weight_file(Path(target_path)))
            else self.initial_model_path
        )
        if desired.suffix.lower() == ".gguf":
            if self._llama_server is not None and self.loaded_path == desired:
                return
            from selfchange.training.llamacpp_backend import LlamaCppServerSession

            if self._llama_server is None:
                self._llama_server = LlamaCppServerSession(
                    desired,
                    ctx_size=max(1024, self.profile.max_total_len),
                    bin_dir=self.llama_bin_dir,
                    base_url=self.llama_base_url,
                )
            else:
                self._llama_server.reload(desired)
            self.model = self._llama_server
            self.tokenizer = "llamacpp_gguf"
            self.loaded_path = desired
            self.model_config = load_model_config(desired) or self.model_config
            return

        if self.model is not None and self.loaded_path == desired:
            return
        from selfchange.training.model_loader import load_minimind_model

        if self._llama_server is not None:
            self._llama_server.close()
            self._llama_server = None
        if self.model is not None:
            self.model = None
            try:
                import torch  # type: ignore[import-not-found]

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        cfg = load_model_config(desired) or self.model_config
        self.model, self.tokenizer = load_minimind_model(desired, self.tokenizer_path, self.profile, model_config=cfg)
        self.loaded_path = desired
        self.model_config = cfg

    def eval_strict_scorecard(self, weight_path: Path | None = None) -> dict:
        self._ensure_loaded(weight_path)
        assert self.model is not None and self.tokenizer is not None and self.loaded_path is not None
        mtime = self.loaded_path.stat().st_mtime_ns if self.loaded_path.is_file() else 0
        key = (str(self.loaded_path), mtime)
        if key in self._scorecard_cache:
            card = self._scorecard_cache[key]
        elif self.is_gguf_mode:
            assert self._llama_server is not None
            max_new = min(160, self.profile.max_gen_len)
            card = evaluate_model_strict_suite(
                None,
                None,
                self.profile,
                max_tasks=20,
                k_rollouts=1,
                reply_fn=lambda msgs, tls, temp: self._llama_server.generate_reply(
                    msgs,
                    tools=tls,
                    temperature=temp,
                    max_new_tokens=max_new,
                    open_thinking=True,
                ),
            )
            card["mean_tok_per_sec"] = self._llama_server.mean_tok_per_sec()
            self._scorecard_cache[key] = card
        else:
            card = evaluate_model_strict_suite(self.model, self.tokenizer, self.profile, max_tasks=20, k_rollouts=1)
            self._scorecard_cache[key] = card
        self.last_scorecard = card
        if self.workspace_dir is not None:
            start_f = self.workspace_dir / "live_scorecard_start.json"
            if not start_f.is_file():
                atomic_write_text(start_f, json.dumps(card, indent=2, ensure_ascii=False) + "\n")
            atomic_write_text(self.workspace_dir / "live_scorecard_latest.json", json.dumps(card, indent=2, ensure_ascii=False) + "\n")
        return card

    def propose(self, ctx: CycleContext, handle: WorktreeHandle, executor: WorktreeToolExecutor) -> dict:
        parent_w = Path(ctx.parent.weight_path) if (ctx.parent and ctx.parent.weight_path) else None
        self._ensure_loaded(parent_w)
        assert self.model is not None and self.tokenizer is not None
        if self.last_scorecard is None:
            self.eval_strict_scorecard(parent_w)

        if self.enable_exploration and self.workspace_dir is not None:
            exp_tasks = build_exploration_tasks(self.last_scorecard, ctx.cycle_index, max_tasks=3)
            if self.is_gguf_mode:
                assert self._llama_server is not None
                max_new = min(160, self.profile.max_gen_len)
                exp_res = run_multi_rollout_exploration(
                    None,
                    None,
                    self.profile,
                    exp_tasks,
                    k_rollouts=4,
                    reply_fn=lambda msgs, tls, temp: self._llama_server.generate_reply(
                        msgs,
                        tools=tls,
                        temperature=temp,
                        max_new_tokens=max_new,
                        open_thinking=True,
                    ),
                )
            else:
                exp_res = run_multi_rollout_exploration(self.model, self.tokenizer, self.profile, exp_tasks, k_rollouts=4)
            self.last_exploration = exp_res
            traj_f = self.workspace_dir / "trajectories.jsonl"
            traj_f.parent.mkdir(parents=True, exist_ok=True)
            with traj_f.open("a", encoding="utf-8") as tf:
                for row in exp_res["trajectories"]:
                    tf.write(json.dumps({"cycle_index": ctx.cycle_index, **row}, ensure_ascii=False) + "\n")

        cfg_path = handle.path / "selfchange" / "mutable" / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {"max_turns": 4}
        if ctx.safe_mode or ctx.config_only:
            cfg["max_turns"] = min(int(cfg.get("max_turns", 4)), 4)

        effective_level = 0 if (ctx.safe_mode or ctx.config_only) else self.autonomy_level
        tool_names = (
            ["get_system_state", "patch_config", "submit_candidate"]
            if effective_level == 0
            else ["get_system_state", "read_file_lines", "patch_config", "replace_line_range", "submit_candidate"]
        )
        compact_tools = _compact_tools(tool_names)

        if self.is_gguf_mode:
            assert self._llama_server is not None

            def _model_fn(messages: list[dict], tools: list[dict]) -> str:
                txt, _ = self._llama_server.generate_reply(
                    messages,
                    tools=tools,
                    temperature=float(cfg.get("temperature", 0.2)),
                    max_new_tokens=min(160, self.profile.max_gen_len),
                    top_p=float(cfg.get("top_p", 0.9)),
                    open_thinking=True,
                )
                return txt
        else:
            from selfchange.training.model_loader import generate_agent_reply

            eos_tok = self.tokenizer.eos_token or "<|im_end|>"
            pad_tok = self.tokenizer.pad_token or "<|endoftext|>"

            def _model_fn(messages: list[dict], tools: list[dict]) -> str:
                raw = generate_agent_reply(
                    self.model,
                    self.tokenizer,
                    messages,
                    tools,
                    max_new_tokens=min(160, self.profile.max_gen_len),
                    temperature=float(cfg.get("temperature", 0.2)),
                    top_p=float(cfg.get("top_p", 0.9)),
                    open_thinking=True,
                )
                return raw.replace(eos_tok, "").replace(pad_tok, "").strip()

        initial_state = diagnose_weakness_state(
            self.last_scorecard,
            ctx.cycle_index,
            ctx.parent.node_id if ctx.parent else None,
            {"profile": ctx.profile.name, "training": cfg.get("training", {})},
            ctx.safe_mode,
            ctx.config_only,
        )
        ep = run_agent_episode(initial_state, compact_tools, _model_fn, executor.dispatch, cfg)
        if not executor.submitted:
            executor.dispatch("submit_candidate", {"hypothesis": f"minimind cycle {ctx.cycle_index}"})
        calls = ep["tool_calls_made"]
        results = ep["tool_results"]
        observed = any(c["name"] in ("get_system_state", "read_log_tail", "read_file_lines") for c in calls)
        first_edit_ok = any(
            res.get("ok")
            for c, res in zip(calls, results)
            if c["name"] in ("patch_config", "replace_line_range", "patch_file_replace")
        )
        return {
            "task_id": f"cycle_{ctx.cycle_index}",
            "task_family": "minimind_autonomous",
            "model_weight": str(self.loaded_path) if self.loaded_path else None,
            "observed_before_edit": observed,
            "first_edit_valid": first_edit_ok,
            "tool_calls": calls,
            "turn_outputs": ep["turn_outputs"],
            "messages": ep["messages"],
            "final_text": ep["final_text"],
            "unfinished": ep["unfinished"],
        }

    def eval_candidate(self, worktree_dir: Path, weight_path: Path | None = None) -> float:
        _ = worktree_dir
        card = self.eval_strict_scorecard(weight_path)
        return float(card["overall_score"])


def make_minimind_proposer(
    model_path: Path,
    tokenizer_path: Path,
    profile: ComputeProfile,
):
    """Build a proposer callback powered by a loaded MiniMindForCausalLM checkpoint."""
    return MiniMindProposerSession(model_path, tokenizer_path, profile).propose


def compute_baseline_quality(card: dict) -> float:
    """Compute weighted [0, 1] quality score across all 4 capability tiers."""
    t1 = card.get("tier1_language", {})
    t2 = card.get("tier2_toolcall", {})
    t3 = card.get("tier3_code_edit", {})
    t4 = card.get("tier4_drills", {})
    t1_s = 0.5 * float(t1.get("eos_rate", 0.0)) + 0.5 * max(0.0, 1.0 - 2.0 * float(t1.get("mean_rep_penalty", 0.5)))
    t2_s = (
        float(t2.get("json_parse_rate", 0.0))
        + float(t2.get("schema_validity_rate", 0.0))
        + float(t2.get("tag_balance_rate", 0.0))
    ) / 3.0
    t3_s = (
        0.2 * float(t3.get("ast_valid_rate", 0.0))
        + 0.4 * float(t3.get("pass_at_1", 0.0))
        + 0.4 * float(t3.get("pass_at_4", 0.0))
    )
    t4_s = float(t4.get("pass_rate", 0.0))
    return round(0.15 * t1_s + 0.25 * t2_s + 0.45 * t3_s + 0.15 * t4_s, 4)


def _read_json_safe(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def assess_run_improvement(
    workspace_dir: Path,
    start_champion: LineageNode | None,
    end_champion: LineageNode | None,
    min_delta: float = 1e-4,
    resumed: bool = False,
) -> dict:
    """Decide whether fine-tuned weights should be persisted based on verified improvement."""
    workspace_dir = Path(workspace_dir).resolve()
    untouched = _read_json_safe(workspace_dir / "baseline_untouched.json")
    prepared = _read_json_safe(workspace_dir / "baseline_prepared.json")
    untouched_q = compute_baseline_quality(untouched) if untouched else None
    prepared_q = compute_baseline_quality(prepared) if prepared else None

    prepared_improved = False
    if untouched is not None and prepared is not None and untouched_q is not None and prepared_q is not None:
        eos_ok = float(prepared.get("tier1_language", {}).get("eos_rate", 0.0)) >= (
            float(untouched.get("tier1_language", {}).get("eos_rate", 0.0)) - 0.05
        )
        schema_ok = float(prepared.get("tier2_toolcall", {}).get("schema_validity_rate", 0.0)) >= (
            float(untouched.get("tier2_toolcall", {}).get("schema_validity_rate", 0.0)) - 0.05
        )
        prepared_improved = (prepared_q > untouched_q + min_delta) and eos_ok and schema_ok
    if resumed and start_champion is not None:
        prepared_improved = False

    if end_champion is None:
        cycle_improved = False
    elif start_champion is None:
        cycle_improved = True
    else:
        cap_improved = (
            end_champion.eval_score > start_champion.eval_score + min_delta
            or end_champion.survival_score > start_champion.survival_score + min_delta
            or end_champion.self_repair_score > start_champion.self_repair_score + min_delta
        )
        cap_non_regressed = (
            end_champion.eval_score >= start_champion.eval_score - min_delta
            and end_champion.survival_score >= start_champion.survival_score - min_delta
            and end_champion.self_repair_score >= start_champion.self_repair_score - min_delta
        )
        artifact_changed = (
            end_champion.weight_path != start_champion.weight_path
            or end_champion.commit_sha != start_champion.commit_sha
        )
        cycle_improved = bool(
            cap_improved
            or (artifact_changed and cap_non_regressed and end_champion.total_score > start_champion.total_score + min_delta)
        )

    raw_weight = end_champion.weight_path if end_champion else None
    resolved_weight: Path | None = None
    if raw_weight:
        cand_w = Path(raw_weight)
        resolved_weight = cand_w if cand_w.is_absolute() else (workspace_dir / cand_w)
    has_weight = bool(resolved_weight is not None and resolved_weight.is_file())

    improved = bool(cycle_improved or prepared_improved)
    persist_weights = bool(improved and has_weight)
    if persist_weights:
        reason = "autonomous_cycle_improved" if cycle_improved else "bootstrap_prepared_improved"
    elif improved:
        reason = "improved_but_no_weight_file"
    else:
        reason = "no_improvement_over_baseline"

    return {
        "improved": improved,
        "persist_weights": persist_weights,
        "reason": reason,
        "resumed": resumed,
        "cycle_improved": cycle_improved,
        "prepared_improved": prepared_improved,
        "champion_weight_path": str(resolved_weight) if resolved_weight else None,
        "start_eval_score": start_champion.eval_score if start_champion else None,
        "end_eval_score": end_champion.eval_score if end_champion else None,
        "start_total_score": start_champion.total_score if start_champion else None,
        "end_total_score": end_champion.total_score if end_champion else None,
        "untouched_quality": untouched_q,
        "prepared_quality": prepared_q,
    }


def generate_run_report(workspace_dir: Path) -> Path:
    """Write a concise Markdown report summarizing session config, baselines, lineage, and training."""
    workspace_dir = Path(workspace_dir).resolve()
    archive_path = workspace_dir / "archive.jsonl"
    champ_path = workspace_dir / "champion.json"
    traj_path = workspace_dir / "trajectories.jsonl"
    state_path = workspace_dir / "supervisor_state.json"
    fatal_path = workspace_dir / "logs" / "fatal_report.json"
    session_path = workspace_dir / "session_info.json"
    history_path = workspace_dir / "logs" / "train_history.jsonl"

    nodes = [json.loads(ln) for ln in archive_path.read_text(encoding="utf-8").splitlines() if ln.strip()] if archive_path.is_file() else []
    champ = json.loads(champ_path.read_text(encoding="utf-8")).get("champion") if champ_path.is_file() else None
    traj_cnt = sum(1 for ln in traj_path.read_text(encoding="utf-8").splitlines() if ln.strip()) if traj_path.is_file() else 0
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    fatal = json.loads(fatal_path.read_text(encoding="utf-8")) if fatal_path.is_file() else None
    session_info = json.loads(session_path.read_text(encoding="utf-8")) if session_path.is_file() else None
    train_rows = [json.loads(ln) for ln in history_path.read_text(encoding="utf-8").splitlines() if ln.strip()] if history_path.is_file() else []

    lines = [
        "# MiniMind SelfChange Autonomous Run Report",
        "",
        f"- **Cycle Index**: `{state.get('cycle_index', 0)}`",
        f"- **Safe Mode**: `{state.get('safe_mode', False)}` | **Halted Reason**: `{state.get('halted_reason')}`",
        f"- **Trajectories Logged**: `{traj_cnt}` | **Archive Nodes**: `{len(nodes)}`",
    ]

    if session_info:
        lines.extend(
            [
                "",
                "## Session Configuration",
                f"- **Proposer Mode**: `{session_info.get('proposer_mode')}` | **Model Source**: `{session_info.get('model_source')}`",
                f"- **Active Model**: `{session_info.get('active_model_path')}`",
                f"- **Eval With Model**: `{session_info.get('eval_with_model')}` | **Anchor Path**: `{session_info.get('anchor_path')}`",
            ]
        )
        dec = session_info.get("persistence_decision")
        if isinstance(dec, dict):
            lines.append(
                f"- **Weight Persistence**: `persist_weights={dec.get('persist_weights')}` | "
                f"`improved={dec.get('improved')}` | reason=`{dec.get('reason')}`"
            )

    baseline_entries = [
        ("Untouched", workspace_dir / "baseline_untouched.json"),
        ("Prepared", workspace_dir / "baseline_prepared.json"),
    ]
    baseline_rows: list[str] = []
    for label, bpath in baseline_entries:
        if bpath.is_file():
            try:
                card = json.loads(bpath.read_text(encoding="utf-8"))
                t1 = card.get("tier1_language", {})
                t2 = card.get("tier2_toolcall", {})
                t3 = card.get("tier3_code_edit", {})
                baseline_rows.append(
                    f"| {label} | `{t1.get('dialogue_ppl')}` | `{t2.get('json_parse_rate')}` | "
                    f"`{t2.get('schema_validity_rate')}` | `{t3.get('pass_at_1')}` / `{t3.get('pass_at_4')}` | `{card.get('bootstrap_ready')}` |"
                )
            except Exception:
                pass
    if baseline_rows:
        lines.extend(
            [
                "",
                "## Baseline Scorecard Summary",
                "| Baseline | PPL | JSON Parse | Schema Valid | AST pass@1 / pass@4 | Bootstrap Ready |",
                "|---|---|---|---|---|---|",
                *baseline_rows,
            ]
        )

    lines.extend(["", "## Active Champion"])
    if champ:
        champ_w = Path(champ["weight_path"]).name if champ.get("weight_path") else "none"
        lines.append(
            f"- **ID**: `{champ['node_id']}` | **Total Score**: `{champ['total_score']}` "
            f"(survival=`{champ['survival_score']}`, eval=`{champ['eval_score']}`, self_repair=`{champ['self_repair_score']}`) | **Weight**: `{champ_w}`"
        )
    else:
        lines.append("- None")

    lines.extend(["", "## Lineage History", "| Node | Status | Score | Eval | Weight | Hypothesis |", "|---|---|---|---|---|---|"])
    for n in nodes[-15:]:
        wname = Path(n["weight_path"]).name if n.get("weight_path") else "-"
        lines.append(f"| `{n['node_id']}` | `{n['status']}` | `{n['total_score']}` | `{n.get('eval_score')}` | `{wname}` | {n['hypothesis']} |")

    if train_rows:
        lines.extend(
            [
                "",
                "## Periodic Self-Training History",
                "| Cycle | Stage / Event | OK | Exit | RFT / DPO | Loss | Weight / Note |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for tr in train_rows[-15:]:
            kind = tr.get("event") or tr.get("stage", "-")
            ok_val = tr["accepted"] if "accepted" in tr else tr.get("ok")
            exit_c = tr.get("exit_code", "-")
            counts = f"{tr.get('rft_count', '-')}/{tr.get('dpo_count', '-')}"
            loss_val = tr.get("final_loss", tr.get("total_score", "-"))
            note = tr.get("output_weight_path") or tr.get("weight_path") or tr.get("rejected_weight_path") or tr.get("merge_error") or tr.get("skipped_reason") or "-"
            if isinstance(note, str) and "/" in note and not tr.get("merge_error"):
                note = Path(note).name
            lines.append(f"| `{tr.get('cycle_index', '-')}` | `{kind}` | `{ok_val}` | `{exit_c}` | `{counts}` | `{loss_val}` | `{note}` |")

    if fatal:
        lines.extend(["", "## Last Fatal Report", f"- Reason: `{fatal.get('reason')}` at `{fatal.get('timestamp')}`"])

    out_path = workspace_dir / "report.md"
    atomic_write_text(out_path, "\n".join(lines) + "\n")
    return out_path


def launch_colab_session(
    repo_root: Path,
    workspace_dir: Path,
    drive_dir: Path | None = None,
    profile_name: str | None = None,
    max_cycles: int = 5,
    autonomy_level: int = 1,
    train_every_n_cycles: int = 3,
    dry_run: bool = False,
    kernel_dir: Path | None = None,
    manifest_path: Path | None = None,
    proposer_fn=None,
    model_path: Path | None = None,
    tokenizer_path: Path = Path("model"),
    require_model: bool = False,
    anchor_path: Path | None = None,
    eval_with_model: bool = False,
    extra_eval_fn=None,
    persist_only_if_improved: bool = True,
    llama_bin_dir: Path | None = None,
    llama_base_url: str | None = None,
    enable_local_cpu_train: bool = False,
    local_train_base_pth: Path | None = None,
    local_cpu_sft_count: int = 48,
    local_cpu_dpo_count: int = 16,
    local_cpu_timeout_sec: int = 1200,
    local_cpu_threads: int = 2,
) -> dict:
    """Re-entrant Colab/local session bootstrap: restore Drive -> run cycles -> report -> sync Drive."""
    repo_root = Path(repo_root).resolve()
    workspace_dir = Path(workspace_dir).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    profile = get_profile(profile_name)

    kdir = Path(kernel_dir).resolve() if kernel_dir is not None else repo_root / "selfchange" / "kernel"
    mpath = Path(manifest_path).resolve() if manifest_path is not None else kdir / KERNEL_MANIFEST_FILENAME

    drive_sync = ColabDriveSync(workspace_dir, drive_dir) if drive_dir is not None else None
    restore_info = {"restored": False, "reason": "no_drive_dir"}
    if drive_sync is not None:
        restore_info = drive_sync.restore_from_drive(
            verify_checksums=True,
            mutable_restore_dir=repo_root / "selfchange" / "mutable",
        )
        if restore_info.get("reason") == "checksum_mismatch":
            raise RuntimeError(f"Drive restore aborted due to checksum mismatch: {restore_info}")

    resolved_model: Path | None = None
    model_source = "dry_run" if dry_run else ("custom" if proposer_fn is not None else "none")
    session: MiniMindProposerSession | None = None

    if proposer_fn is None and not dry_run:
        resolved_model, model_source = resolve_model_checkpoint(
            repo_root,
            workspace_dir,
            profile,
            explicit_model_path=model_path,
        )
        if resolved_model is None:
            if require_model:
                raise FileNotFoundError(
                    f"MiniMind checkpoint required (--require-model) but none found "
                    f"(requested={model_path}, workspace={workspace_dir})."
                )
            print(
                f"[WARN] No MiniMind checkpoint found (requested={model_path}); falling back to scripted _default_proposer.",
                file=sys.stderr,
            )
        else:
            resolved_tok = Path(tokenizer_path) if Path(tokenizer_path).is_absolute() else (repo_root / tokenizer_path)
            session = MiniMindProposerSession(
                resolved_model,
                resolved_tok,
                profile,
                autonomy_level=autonomy_level,
                workspace_dir=workspace_dir,
                llama_bin_dir=llama_bin_dir,
                llama_base_url=llama_base_url,
            )
            proposer_fn = session.propose
            if eval_with_model and extra_eval_fn is None:
                extra_eval_fn = session.eval_candidate

    gguf_mode = bool(session is not None and session.is_gguf_mode)
    resolved_train_base_pth: Path | None = None
    if enable_local_cpu_train and gguf_mode and not dry_run:
        resolved_train_base_pth = resolve_local_train_base_pth(
            repo_root,
            workspace_dir,
            profile,
            explicit_pth=local_train_base_pth,
            model_config=session.model_config if session is not None else None,
        )
    local_train_ready = bool(gguf_mode and enable_local_cpu_train and not dry_run)
    backend_name = (
        "llamacpp_gguf+local_cpu_train"
        if local_train_ready
        else ("llamacpp_gguf" if gguf_mode else ("pytorch" if session is not None else "none"))
    )
    effective_dry_run = bool(dry_run or (gguf_mode and not enable_local_cpu_train))
    proposer_mode = "minimind" if session is not None else ("custom" if proposer_fn is not None else ("dry_run" if dry_run else "scripted_fallback"))

    resolved_anchor: Path | None = None
    if anchor_path is not None:
        cand_anchor = Path(anchor_path) if Path(anchor_path).is_absolute() else (repo_root / anchor_path)
        if cand_anchor.is_file():
            resolved_anchor = cand_anchor.resolve()
    else:
        default_anchor = workspace_dir / "models" / "base" / "data" / "anchor_sft.jsonl"
        if default_anchor.is_file():
            resolved_anchor = default_anchor.resolve()

    session_info = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile.name,
        "backend": backend_name,
        "proposer_mode": proposer_mode,
        "model_source": model_source,
        "active_model_path": str(resolved_model) if resolved_model else None,
        "enable_local_cpu_train": bool(enable_local_cpu_train),
        "local_train_base_pth": str(resolved_train_base_pth) if resolved_train_base_pth else None,
        "eval_with_model": bool(extra_eval_fn is not None),
        "anchor_path": str(resolved_anchor) if resolved_anchor else None,
        "require_model": require_model,
        "dry_run": effective_dry_run,
        "persist_only_if_improved": persist_only_if_improved,
    }
    atomic_write_text(workspace_dir / "session_info.json", json.dumps(session_info, indent=2, ensure_ascii=False) + "\n")

    runner = SelfImprovingCycleRunner(
        repo_root=repo_root,
        workspace_dir=workspace_dir,
        profile=profile,
        proposer_fn=proposer_fn,
        kernel_dir=kdir,
        manifest_path=mpath,
        cycle_config=CycleConfig(
            autonomy_level=autonomy_level,
            train_every_n_cycles=train_every_n_cycles,
            dry_run_training=effective_dry_run,
            anchor_path=resolved_anchor,
            base_weight_path=resolved_model,
        ),
        extra_eval_fn=extra_eval_fn,
        session=session,
        live_self_improve=(session is not None and not dry_run and (not gguf_mode or bool(enable_local_cpu_train))),
        model_config=session.model_config if session is not None else None,
        enable_local_cpu_train=bool(enable_local_cpu_train),
        local_train_base_pth=resolved_train_base_pth,
        local_cpu_sft_count=local_cpu_sft_count,
        local_cpu_dpo_count=local_cpu_dpo_count,
        local_cpu_timeout_sec=local_cpu_timeout_sec,
        local_cpu_threads=local_cpu_threads,
        llama_bin_dir=llama_bin_dir,
    )

    try:
        if not runner.registry.load_archive():
            head_proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True)
            head_sha = head_proc.stdout.strip() if head_proc.returncode == 0 else "0" * 40
            surv_s, eval_s, rep_s, eff_s, tot_s = 1.0, 1.0, 1.0, 1.0, 0.95
            if extra_eval_fn is not None:
                cal = runner.verifier.verify_candidate(
                    repo_root,
                    check_git=False,
                    extra_eval_fn=lambda wt: runner._invoke_extra_eval(wt, resolved_model),
                    min_extra_score=0.0,
                )
                surv_s, eval_s, rep_s, eff_s, tot_s = (
                    cal.survival_score,
                    cal.eval_score,
                    cal.self_repair_score,
                    cal.efficiency_score,
                    cal.total_score,
                )
                if "gate3" in cal.gates:
                    runner._best_extra_score = float(cal.gates["gate3"].details.get("extra_score", 0.0))
            runner.registry.register_node(
                LineageNode(
                    node_id="bootstrap_0",
                    parent_id=None,
                    commit_sha=head_sha,
                    weight_path=str(resolved_model) if resolved_model else None,
                    survival_score=surv_s,
                    eval_score=eval_s,
                    self_repair_score=rep_s,
                    efficiency_score=eff_s,
                    total_score=tot_s,
                    status="bootstrap",
                    hypothesis="initial verified baseline",
                    created_at=datetime.now(timezone.utc).isoformat(),
                )
            )

        start_champion = runner.registry.get_champion()

        def _persist_and_sync() -> dict:
            decision = assess_run_improvement(
                workspace_dir,
                start_champion=start_champion,
                end_champion=runner.registry.get_champion(),
                resumed=bool(restore_info.get("restored")),
            )
            session_info["updated_at"] = datetime.now(timezone.utc).isoformat()
            session_info["persistence_decision"] = decision
            atomic_write_text(
                workspace_dir / "session_info.json",
                json.dumps(session_info, indent=2, ensure_ascii=False) + "\n",
            )
            generate_run_report(workspace_dir)
            if drive_sync is not None:
                sync_weights = True if not persist_only_if_improved else bool(decision["persist_weights"])
                drive_sync.sync_to_drive(repo_root / "selfchange" / "mutable", include_weights=sync_weights)
            return decision

        sup = Supervisor(
            repo_root=repo_root,
            workspace_dir=workspace_dir,
            profile=profile,
            cycle_runner=runner,
            kernel_dir=kdir,
            manifest_path=mpath,
            register_signals=False,
            on_cycle_end=lambda _st, _out: _persist_and_sync(),
        )
        sup.load_state()
        final_state = sup.run(max_cycles=max_cycles)

        persistence_decision = _persist_and_sync()
        report_path = workspace_dir / "report.md"
        champ = runner.registry.get_champion()
        return {
            "restore": restore_info,
            "backend": backend_name,
            "proposer_mode": proposer_mode,
            "active_model_path": str(resolved_model) if resolved_model else None,
            "model_source": model_source,
            "eval_with_model": bool(extra_eval_fn is not None),
            "persistence_decision": persistence_decision,
            "state": asdict(final_state),
            "champion": champ.to_dict() if champ else None,
            "report_path": str(report_path),
        }
    finally:
        if session is not None:
            session.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-entrant Colab launcher for MiniMind selfchange")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--workspace-dir", type=Path, default=Path(__file__).resolve().parents[1] / "workspace")
    parser.add_argument("--drive-dir", type=Path, default=None)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("model"))
    parser.add_argument("--anchor-path", type=Path, default=None)
    parser.add_argument("--max-cycles", type=int, default=5)
    parser.add_argument("--autonomy-level", type=int, default=1)
    parser.add_argument("--train-every-n-cycles", type=int, default=3)
    parser.add_argument("--require-model", action="store_true")
    parser.add_argument("--eval-with-model", action="store_true")
    parser.add_argument("--always-persist-weights", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--enable-local-cpu-train", action="store_true")
    parser.add_argument("--train-base-pth", type=Path, default=None)
    parser.add_argument("--local-sft-count", type=int, default=48)
    parser.add_argument("--local-dpo-count", type=int, default=16)
    parser.add_argument("--local-train-timeout", type=int, default=1200)
    parser.add_argument("--local-cpu-threads", type=int, default=2)
    args = parser.parse_args()

    res = launch_colab_session(
        repo_root=args.repo_root,
        workspace_dir=args.workspace_dir,
        drive_dir=args.drive_dir,
        profile_name=args.profile,
        max_cycles=args.max_cycles,
        autonomy_level=args.autonomy_level,
        train_every_n_cycles=args.train_every_n_cycles,
        dry_run=args.dry_run,
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        require_model=args.require_model,
        anchor_path=args.anchor_path,
        eval_with_model=args.eval_with_model,
        persist_only_if_improved=not args.always_persist_weights,
        enable_local_cpu_train=args.enable_local_cpu_train,
        local_train_base_pth=args.train_base_pth,
        local_cpu_sft_count=args.local_sft_count,
        local_cpu_dpo_count=args.local_dpo_count,
        local_cpu_timeout_sec=args.local_train_timeout,
        local_cpu_threads=args.local_cpu_threads,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
