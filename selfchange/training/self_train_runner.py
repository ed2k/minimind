from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from selfchange.kernel.budgets import ComputeProfile
from selfchange.kernel.integrity import atomic_write_text
from selfchange.training.model_loader import estimate_minimind_params, load_model_config, save_model_config


@dataclass
class TrainingRunResult:
    """Telemetry returned by a single bounded self-training execution."""

    stage: str
    ok: bool
    exit_code: int
    output_weight_path: str | None
    resume_ckpt_path: str | None
    final_loss: float | None
    oom_detected: bool
    nan_detected: bool
    timed_out: bool
    duration_sec: float
    log_tail: str
    dry_run: bool = False


def detect_log_anomalies(log_text: str) -> tuple[bool, bool]:
    """Detect OOM and NaN/Inf loss signatures from training log output."""
    lower = (log_text or "").lower()
    oom = "out of memory" in lower or "memoryerror" in lower
    nan = bool(re.search(r"\bloss:\s*(nan|inf)\b", lower)) or "grad_norm: inf" in lower
    return oom, nan


def _extract_last_loss(log_text: str) -> float | None:
    matches = re.findall(r"\bloss:\s*([-+]?\d+(?:\.\d+)?)", log_text or "", flags=re.IGNORECASE)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def format_sft_record(row: dict) -> dict:
    """Normalize a trace or sample strictly into dataset/lm_dataset.py SFTDataset Features schema."""
    raw_convs = row.get("conversations") or row.get("messages") or []
    normalized: list[dict] = []
    for raw_msg in raw_convs:
        if not isinstance(raw_msg, dict):
            continue
        raw_rc = raw_msg.get("reasoning_content")
        raw_tools = raw_msg.get("tools")
        raw_tcalls = raw_msg.get("tool_calls")
        normalized.append(
            {
                "role": str(raw_msg.get("role", "user")),
                "content": str(raw_msg.get("content") or ""),
                "reasoning_content": None if raw_rc is None else str(raw_rc),
                "tools": (
                    None
                    if raw_tools is None
                    else (raw_tools if isinstance(raw_tools, str) else json.dumps(raw_tools, ensure_ascii=False))
                ),
                "tool_calls": (
                    None
                    if raw_tcalls is None
                    else (raw_tcalls if isinstance(raw_tcalls, str) else json.dumps(raw_tcalls, ensure_ascii=False))
                ),
            }
        )
    return {"conversations": normalized}


def format_dpo_record(row: dict) -> dict:
    """Normalize a preference pair strictly into dataset/lm_dataset.py DPODataset schema."""
    def _norm_side(msgs: list) -> list[dict]:
        out: list[dict] = []
        for raw_msg in msgs or []:
            if not isinstance(raw_msg, dict):
                continue
            out.append({"role": str(raw_msg.get("role", "user")), "content": str(raw_msg.get("content") or "")})
        return out

    return {
        "chosen": _norm_side(row.get("chosen") or []),
        "rejected": _norm_side(row.get("rejected") or []),
    }


def write_sft_jsonl(path: Path, rows: list[dict]) -> int:
    """Atomically write strictly formatted SFTDataset JSONL rows, returning count written."""
    formatted: list[str] = []
    for row in rows:
        rec = format_sft_record(row)
        if rec["conversations"]:
            formatted.append(json.dumps(rec, ensure_ascii=False) + "\n")
    atomic_write_text(Path(path), "".join(formatted))
    return len(formatted)


def write_dpo_jsonl(path: Path, rows: list[dict]) -> int:
    """Atomically write strictly formatted DPODataset JSONL rows, returning count written."""
    formatted: list[str] = []
    for row in rows:
        rec = format_dpo_record(row)
        if rec["chosen"] and rec["rejected"]:
            formatted.append(json.dumps(rec, ensure_ascii=False) + "\n")
    atomic_write_text(Path(path), "".join(formatted))
    return len(formatted)


def _resolve_arch(profile: ComputeProfile, model_config: dict | None = None) -> dict:
    cfg = dict(model_config) if model_config else {}
    return estimate_minimind_params(
        hidden_size=int(cfg.get("hidden_size", profile.hidden_size)),
        num_hidden_layers=int(cfg.get("num_hidden_layers", profile.num_hidden_layers)),
        intermediate_size=int(cfg["intermediate_size"]) if cfg.get("intermediate_size") is not None else None,
        num_attention_heads=int(cfg.get("num_attention_heads", 8)),
        num_key_value_heads=int(cfg.get("num_key_value_heads", 4)),
        use_moe=bool(cfg.get("use_moe", profile.use_moe)),
    )


def stage_from_weight_for_trainer(
    from_weight: str,
    repo_root: Path,
    workspace_dir: Path,
    profile: ComputeProfile,
    from_weight_path: Path | None = None,
    model_config: dict | None = None,
) -> Path | None:
    """Locate from_weight across explicit path/workspace/out and stage into repo_root/out for trainer/*.py."""
    if not from_weight or from_weight == "none":
        return None
    repo_root = Path(repo_root).resolve()
    workspace_dir = Path(workspace_dir).resolve()
    eff_cfg = model_config or (load_model_config(from_weight_path) if from_weight_path else None) or {}
    hs = int(eff_cfg.get("hidden_size", profile.hidden_size))
    um = bool(eff_cfg.get("use_moe", profile.use_moe))
    moe_suffix = "_moe" if um else ""
    filename = f"{from_weight}_{hs}{moe_suffix}.pth"

    search_candidates: list[Path] = []
    if from_weight_path is not None:
        search_candidates.append(Path(from_weight_path))
    search_candidates.extend(
        [
            repo_root / "out" / filename,
            workspace_dir / "models" / "candidates" / filename,
            workspace_dir / "models" / "base" / filename,
        ]
    )
    if from_weight == "full_sft":
        search_candidates.append(workspace_dir / "models" / "base" / f"minimind3_{hs}{moe_suffix}.pth")

    found: Path | None = None
    for cand in search_candidates:
        if cand.is_file():
            found = cand
            break
    if found is None:
        return None

    target_out = repo_root / "out" / filename
    if found.resolve() != target_out.resolve():
        target_out.parent.mkdir(parents=True, exist_ok=True)
        tmp = target_out.with_name(f".{target_out.name}.{os.getpid()}.tmp")
        try:
            shutil.copy2(found, tmp)
            os.replace(tmp, target_out)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
    return target_out


def promote_lora_adapter(
    base_weight: Path,
    lora_weight: Path,
    merged_out: Path,
    hidden_size: int,
    num_hidden_layers: int,
    use_moe: bool = False,
    lora_rank: int = 16,
    target_all_linear: bool = False,
    intermediate_size: int | None = None,
    num_attention_heads: int | None = None,
    num_key_value_heads: int | None = None,
    model_config: dict | None = None,
) -> Path:
    """Atomically merge a verified LoRA adapter into base weights using model/model_lora.py."""
    import torch  # type: ignore[import-not-found]
    from model.model_lora import merge_lora  # type: ignore[import-not-found]
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # type: ignore[import-not-found]
    from selfchange.training.model_loader import apply_extended_lora

    eff_cfg = dict(model_config) if model_config else (load_model_config(base_weight) or {})
    hs = int(eff_cfg.get("hidden_size", hidden_size))
    nl = int(eff_cfg.get("num_hidden_layers", num_hidden_layers))
    um = bool(eff_cfg.get("use_moe", use_moe))
    inter = intermediate_size if intermediate_size is not None else eff_cfg.get("intermediate_size")
    nh = num_attention_heads if num_attention_heads is not None else eff_cfg.get("num_attention_heads")
    nkv = num_key_value_heads if num_key_value_heads is not None else eff_cfg.get("num_key_value_heads")
    extra_kwargs = {
        k: int(v)
        for k, v in [("intermediate_size", inter), ("num_attention_heads", nh), ("num_key_value_heads", nkv)]
        if v is not None
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MiniMindConfig(hidden_size=hs, num_hidden_layers=nl, use_moe=um, **extra_kwargs)
    model = MiniMindForCausalLM(cfg).to(device)
    model.load_state_dict(torch.load(str(base_weight), map_location=device), strict=False)
    apply_extended_lora(model, rank=lora_rank, target_all_linear=target_all_linear)

    merged_out = Path(merged_out)
    merged_out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = merged_out.with_name(f".{merged_out.name}.{os.getpid()}.tmp")
    try:
        merge_lora(model, str(lora_weight), str(tmp_out))
        os.replace(tmp_out, merged_out)
    finally:
        if tmp_out.exists():
            tmp_out.unlink(missing_ok=True)
    save_model_config(
        merged_out.with_suffix(".config.json"),
        estimate_minimind_params(
            hidden_size=hs,
            num_hidden_layers=nl,
            intermediate_size=cfg.intermediate_size,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            use_moe=um,
        ),
    )
    return merged_out


def build_trainer_command(
    stage: str,
    repo_root: Path,
    data_path: Path,
    save_dir: Path,
    save_weight: str,
    from_weight: str,
    profile: ComputeProfile,
    training_cfg: dict,
    from_resume: bool = False,
    model_config: dict | None = None,
) -> tuple[list[str], Path]:
    """Construct profile-aware trainer command and expected output checkpoint path."""
    repo_root = Path(repo_root).resolve()
    save_dir = Path(save_dir).resolve()
    data_path = Path(data_path).resolve()

    raw_cfg = model_config if model_config is not None else (training_cfg.get("model_config") if isinstance(training_cfg.get("model_config"), dict) else None)
    hs = int(raw_cfg["hidden_size"]) if (raw_cfg and raw_cfg.get("hidden_size") is not None) else profile.hidden_size
    nl = int(raw_cfg["num_hidden_layers"]) if (raw_cfg and raw_cfg.get("num_hidden_layers") is not None) else profile.num_hidden_layers
    um = bool(raw_cfg.get("use_moe", profile.use_moe)) if raw_cfg else profile.use_moe
    moe_suffix = "_moe" if um else ""
    out_weight = save_dir / f"{save_weight}_{hs}{moe_suffix}.pth"

    lr = str(training_cfg.get("learning_rate", 5e-5))
    bs = str(training_cfg.get("batch_size", profile.batch_size))
    seq = str(min(profile.max_seq_len, int(training_cfg.get("max_seq_len", profile.max_seq_len))))
    epochs = str(training_cfg.get("epochs", 1))
    grad_clip = str(training_cfg.get("grad_clip", 1.0))
    acc = str(training_cfg.get("accumulation_steps", 1))
    resume_flag = "1" if from_resume else "0"

    if stage == "distill":
        t_cfg = training_cfg.get("teacher_config") if isinstance(training_cfg.get("teacher_config"), dict) else {}
        t_hs = str(int(t_cfg.get("hidden_size", profile.hidden_size)))
        t_nl = str(int(t_cfg.get("num_hidden_layers", profile.num_hidden_layers)))
        t_um = "1" if bool(t_cfg.get("use_moe", profile.use_moe)) else "0"
        from_student = str(training_cfg.get("from_student_weight", from_weight if from_weight else "none"))
        from_teacher = str(training_cfg.get("from_teacher_weight", "full_sft"))
        alpha = str(training_cfg.get("distill_alpha", 0.5))
        temp = str(training_cfg.get("distill_temperature", 1.5))
        cmd = [
            sys.executable,
            str(repo_root / "trainer" / "train_distillation.py"),
            "--save_dir", str(save_dir),
            "--save_weight", save_weight,
            "--epochs", epochs,
            "--batch_size", bs,
            "--learning_rate", lr,
            "--device", profile.device,
            "--dtype", profile.dtype,
            "--num_workers", str(profile.num_workers),
            "--accumulation_steps", acc,
            "--grad_clip", grad_clip,
            "--max_seq_len", seq,
            "--data_path", str(data_path),
            "--student_hidden_size", str(hs),
            "--student_num_layers", str(nl),
            "--student_use_moe", "1" if um else "0",
            "--teacher_hidden_size", t_hs,
            "--teacher_num_layers", t_nl,
            "--teacher_use_moe", t_um,
            "--from_student_weight", from_student,
            "--from_teacher_weight", from_teacher,
            "--from_resume", resume_flag,
            "--alpha", alpha,
            "--temperature", temp,
        ]
        if raw_cfg:
            if raw_cfg.get("intermediate_size") is not None:
                cmd.extend(["--student_intermediate_size", str(int(raw_cfg["intermediate_size"]))])
            if raw_cfg.get("num_attention_heads") is not None:
                cmd.extend(["--student_num_attention_heads", str(int(raw_cfg["num_attention_heads"]))])
            if raw_cfg.get("num_key_value_heads") is not None:
                cmd.extend(["--student_num_key_value_heads", str(int(raw_cfg["num_key_value_heads"]))])
        if t_cfg.get("intermediate_size") is not None:
            cmd.extend(["--teacher_intermediate_size", str(int(t_cfg["intermediate_size"]))])
        if t_cfg.get("num_attention_heads") is not None:
            cmd.extend(["--teacher_num_attention_heads", str(int(t_cfg["num_attention_heads"]))])
        if t_cfg.get("num_key_value_heads") is not None:
            cmd.extend(["--teacher_num_key_value_heads", str(int(t_cfg["num_key_value_heads"]))])
        return cmd, out_weight

    common = [
        "--save_dir", str(save_dir),
        "--epochs", epochs,
        "--batch_size", bs,
        "--device", profile.device,
        "--dtype", profile.dtype,
        "--num_workers", str(profile.num_workers),
        "--accumulation_steps", acc,
        "--grad_clip", grad_clip,
        "--hidden_size", str(hs),
        "--num_hidden_layers", str(nl),
        "--max_seq_len", seq,
        "--use_moe", "1" if um else "0",
        "--data_path", str(data_path),
        "--from_weight", from_weight,
        "--from_resume", resume_flag,
    ]
    if raw_cfg:
        if raw_cfg.get("intermediate_size") is not None:
            common.extend(["--intermediate_size", str(int(raw_cfg["intermediate_size"]))])
        if raw_cfg.get("num_attention_heads") is not None:
            common.extend(["--num_attention_heads", str(int(raw_cfg["num_attention_heads"]))])
        if raw_cfg.get("num_key_value_heads") is not None:
            common.extend(["--num_key_value_heads", str(int(raw_cfg["num_key_value_heads"]))])

    if stage == "pretrain":
        cmd = [sys.executable, str(repo_root / "trainer" / "train_pretrain.py"), "--save_weight", save_weight, "--learning_rate", lr, *common]
    elif stage == "full_sft":
        cmd = [sys.executable, str(repo_root / "trainer" / "train_full_sft.py"), "--save_weight", save_weight, "--learning_rate", lr, *common]
    elif stage == "lora":
        rank = str(training_cfg.get("lora_rank", 16))
        cmd = [sys.executable, str(repo_root / "trainer" / "train_lora.py"), "--lora_name", save_weight, "--lora_rank", rank, "--learning_rate", lr, *common]
    elif stage == "dpo":
        dpo_lr = str(training_cfg.get("dpo_learning_rate", training_cfg.get("learning_rate", 4e-8)))
        beta = str(training_cfg.get("dpo_beta", 0.15))
        cmd = [sys.executable, str(repo_root / "trainer" / "train_dpo.py"), "--save_weight", save_weight, "--learning_rate", dpo_lr, "--beta", beta, *common]
    elif stage == "rlvr":
        rl_lr = str(training_cfg.get("rl_learning_rate", 3e-7))
        kl_beta = str(training_cfg.get("kl_beta", 0.1))
        cmd = [
            sys.executable,
            str(repo_root / "selfchange" / "training" / "train_self_agent.py"),
            "--profile", profile.name,
            "--save_dir", str(save_dir),
            "--save_weight", save_weight,
            "--from_weight", from_weight,
            "--learning_rate", rl_lr,
            "--beta", kl_beta,
            "--data_path", str(data_path),
            "--from_resume", resume_flag,
        ]
    else:
        raise ValueError(f"Unsupported training stage {stage!r}")
    return cmd, out_weight


def run_training_stage(
    stage: str,
    repo_root: Path,
    workspace_dir: Path,
    data_path: Path,
    profile: ComputeProfile,
    training_cfg: dict,
    from_weight: str = "full_sft",
    save_weight: str = "cand",
    from_resume: bool = False,
    dry_run: bool = False,
    from_weight_path: Path | None = None,
    model_config: dict | None = None,
) -> TrainingRunResult:
    """Execute one bounded training stage (gated by continuous data review) or dry-run validation."""
    t0 = time.monotonic()
    repo_root = Path(repo_root).resolve()
    workspace_dir = Path(workspace_dir).resolve()
    data_path = Path(data_path).resolve()
    candidates_dir = workspace_dir / "models" / "candidates"
    logs_dir = workspace_dir / "logs"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    eff_arch = dict(model_config) if model_config is not None else (
        dict(training_cfg["model_config"]) if isinstance(training_cfg.get("model_config"), dict) else None
    )
    resolved_arch = _resolve_arch(profile, eff_arch)

    cmd, expected_out = build_trainer_command(
        stage=stage,
        repo_root=repo_root,
        data_path=data_path,
        save_dir=candidates_dir,
        save_weight=save_weight,
        from_weight=from_weight,
        profile=profile,
        training_cfg=training_cfg,
        from_resume=from_resume,
        model_config=eff_arch,
    )
    train_log = logs_dir / "train.log"

    if not data_path.is_file() or data_path.stat().st_size == 0:
        msg = f"Training data missing or empty: {data_path}\n"
        atomic_write_text(train_log, msg)
        return TrainingRunResult(stage, False, 2, None, None, None, False, False, False, 0.0, msg, dry_run=dry_run)

    enforce_data_gate = bool(training_cfg.get("enforce_data_gate", not dry_run))
    if enforce_data_gate and stage in ("pretrain", "full_sft", "lora", "dpo", "distill"):
        from selfchange.bootstrap.data_quality import review_and_record_dataset

        seq_int = min(profile.max_seq_len, int(training_cfg.get("max_seq_len", profile.max_seq_len)))
        fmt = "dpo" if stage == "dpo" else ("pretrain" if stage == "pretrain" else "sft")
        heldout_raw = training_cfg.get("heldout_path")
        review = review_and_record_dataset(
            data_path=data_path,
            workspace_dir=workspace_dir,
            dataset_format=fmt,
            max_seq_len=seq_int,
            heldout_path=Path(heldout_raw) if heldout_raw else None,
            cycle_tag=f"{stage}_{save_weight}",
        )
        if not review.get("passed", False):
            failed_str = ", ".join(review.get("failed_checks", [])) or "unknown_data_quality_failure"
            msg = f"[DATA-GATE-BLOCKED] stage={stage} data={data_path.name} failed={failed_str}\n"
            atomic_write_text(train_log, msg)
            return TrainingRunResult(
                stage=stage,
                ok=False,
                exit_code=3,
                output_weight_path=None,
                resume_ckpt_path=None,
                final_loss=None,
                oom_detected=False,
                nan_detected=False,
                timed_out=False,
                duration_sec=round(time.monotonic() - t0, 4),
                log_tail=msg,
                dry_run=dry_run,
            )

    if dry_run:
        atomic_write_text(
            expected_out,
            json.dumps(
                {"dry_run": True, "stage": stage, "from_weight": from_weight, "data_path": str(data_path), "model_config": resolved_arch},
                indent=2,
            )
            + "\n",
        )
        save_model_config(expected_out.with_suffix(".config.json"), resolved_arch)
        log_line = f"[DRY-RUN] stage={stage} loss: 1.2500 saved={expected_out.name}\n"
        atomic_write_text(train_log, log_line)
        return TrainingRunResult(
            stage=stage,
            ok=True,
            exit_code=0,
            output_weight_path=str(expected_out),
            resume_ckpt_path=None,
            final_loss=1.25,
            oom_detected=False,
            nan_detected=False,
            timed_out=False,
            duration_sec=round(time.monotonic() - t0, 4),
            log_tail=log_line,
            dry_run=True,
        )

    if stage == "distill":
        t_cfg = training_cfg.get("teacher_config") if isinstance(training_cfg.get("teacher_config"), dict) else None
        t_path = Path(training_cfg["teacher_weight_path"]) if training_cfg.get("teacher_weight_path") else None
        from_teacher = str(training_cfg.get("from_teacher_weight", "full_sft"))
        stage_from_weight_for_trainer(from_teacher, repo_root, workspace_dir, profile, from_weight_path=t_path, model_config=t_cfg)
        from_student = str(training_cfg.get("from_student_weight", from_weight if from_weight else "none"))
        if from_student != "none":
            stage_from_weight_for_trainer(from_student, repo_root, workspace_dir, profile, from_weight_path=from_weight_path, model_config=eff_arch)
    else:
        stage_from_weight_for_trainer(from_weight, repo_root, workspace_dir, profile, from_weight_path=from_weight_path, model_config=eff_arch)

    cwd = repo_root / "trainer" if stage in ("pretrain", "full_sft", "lora", "dpo", "distill") else repo_root
    trainer_env = os.environ.copy()
    trainer_env["PYTHONPATH"] = str(repo_root) + (os.pathsep + trainer_env["PYTHONPATH"] if trainer_env.get("PYTHONPATH") else "")
    trainer_env["PYTHONUNBUFFERED"] = "1"

    t_proc0 = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=trainer_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=float(profile.subprocess_timeout_sec))
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        returncode = proc.returncode if proc.returncode is not None else -9

    duration = round(time.monotonic() - t_proc0, 4)
    combined_log = ((stdout or "") + "\n" + (stderr or "")).strip()
    tail_lines = "\n".join(combined_log.splitlines()[-40:])
    atomic_write_text(train_log, tail_lines + "\n")

    oom, nan = detect_log_anomalies(combined_log)
    final_loss = _extract_last_loss(combined_log)
    ok = (not timed_out) and returncode == 0 and (not oom) and (not nan)
    if expected_out.is_file():
        save_model_config(expected_out.with_suffix(".config.json"), resolved_arch)
    hs = int(resolved_arch["hidden_size"])
    moe_s = "_moe" if resolved_arch["use_moe"] else ""
    resume_guess = repo_root / "checkpoints" / f"{save_weight}_{hs}{moe_s}_resume.pth"
    return TrainingRunResult(
        stage=stage,
        ok=ok,
        exit_code=returncode,
        output_weight_path=str(expected_out) if expected_out.is_file() else None,
        resume_ckpt_path=str(resume_guess) if resume_guess.is_file() else None,
        final_loss=final_loss,
        oom_detected=oom,
        nan_detected=nan,
        timed_out=timed_out,
        duration_sec=duration,
        log_tail=tail_lines,
        dry_run=False,
    )
