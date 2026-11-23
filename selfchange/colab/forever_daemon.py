from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from selfchange.colab.autonomous_engine import prune_forever_workspace
from selfchange.kernel.integrity import atomic_write_text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rotate_log_if_needed(log_path: Path, max_bytes: int) -> bool:
    if max_bytes <= 0 or not log_path.is_file():
        return False
    try:
        if log_path.stat().st_size <= max_bytes:
            return False
        rotated = log_path.with_name(f"{log_path.name}.1")
        shutil.copy2(log_path, rotated)
        log_path.write_text("", encoding="utf-8")
        return True
    except OSError:
        return False


def _free_disk_mb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return round(shutil.disk_usage(path).free / (1024.0 * 1024.0), 2)


def _read_json_safe(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _recent_recovery_count(events_path: Path, hours: int = 24) -> int:
    if not events_path.is_file():
        return 0
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    count = 0
    for ln in events_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        try:
            row = json.loads(ln)
            if row.get("event") != "safe_mode_recovery":
                continue
            ts = datetime.fromisoformat(str(row.get("timestamp", "")))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                count += 1
        except Exception:
            continue
    return count


def _append_event(events_path: Path, row: dict) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": _utc_now(), **row}, ensure_ascii=False) + "\n")


def _interruptible_sleep(seconds: float, stop_flag: dict, stop_file: Path) -> bool:
    deadline = time.monotonic() + max(0.0, float(seconds))
    while time.monotonic() < deadline:
        if stop_flag["stop"] or stop_file.exists():
            return False
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    return not (stop_flag["stop"] or stop_file.exists())


def _write_daemon_status(
    workspace_dir: Path,
    status: str,
    chunk_index: int,
    last_rc: int | None,
    child_pid: int | None = None,
) -> None:
    state = _read_json_safe(workspace_dir / "supervisor_state.json") or {}
    champ = (_read_json_safe(workspace_dir / "champion.json") or {}).get("champion")
    hist_path = workspace_dir / "logs" / "train_history.jsonl"
    latest_train = None
    if hist_path.is_file():
        try:
            rows = [json.loads(ln) for ln in hist_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            train_rows = [r for r in rows if r.get("stage") in ("lora", "dpo")]
            if train_rows:
                latest_train = {k: train_rows[-1].get(k) for k in ("cycle_index", "stage", "ok", "accepted", "final_loss", "before_score", "after_score", "effective_lr", "duration_sec")}
        except Exception:
            latest_train = None
    payload = {
        "updated_at": _utc_now(),
        "daemon_pid": os.getpid(),
        "child_pid": child_pid,
        "status": status,
        "chunk_index": chunk_index,
        "total_cycles": int(state.get("cycle_index", 0)),
        "consecutive_failures": int(state.get("consecutive_failures", 0)),
        "safe_mode": bool(state.get("safe_mode", False)),
        "halted_reason": state.get("halted_reason"),
        "last_rc": last_rc,
        "free_disk_mb": _free_disk_mb(workspace_dir),
        "active_champion": {k: champ.get(k) for k in ("node_id", "total_score", "eval_score", "weight_path")} if isinstance(champ, dict) else None,
        "latest_train_row": latest_train,
    }
    atomic_write_text(workspace_dir / "DAEMON_STATUS.json", json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def run_forever_daemon(
    repo_root: Path,
    workspace_dir: Path,
    profile_name: str = "local_intel_cpu",
    model_path: Path | None = None,
    train_base_pth: Path | None = None,
    anchor_path: Path | None = None,
    autonomy_level: int = 1,
    train_every_n_cycles: int = 2,
    chunk_cycles: int = 4,
    local_sft_count: int = 48,
    local_dpo_count: int = 16,
    local_train_timeout: int = 1200,
    local_cpu_threads: int = 2,
    cooldown_sec: float = 20.0,
    max_chunks: int = 0,
    max_safe_mode_recoveries: int = 3,
    recovery_cooldown_sec: float = 300.0,
    max_log_mb: float = 10.0,
    child_runner=None,
) -> dict:
    """Run bounded colab_launcher chunks sequentially forever so RAM, caches, disk, and logs stay healthy."""
    repo_root = Path(repo_root).resolve()
    workspace_dir = Path(workspace_dir).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)
    stop_file = workspace_dir / "STOP"
    events_path = workspace_dir / "logs" / "daemon_events.jsonl"
    daemon_log = workspace_dir / "daemon.log"
    launchd_log = workspace_dir / "launchd.log"
    state_path = workspace_dir / "supervisor_state.json"

    stop_flag = {"stop": False}
    active_proc: dict[str, subprocess.Popen | None] = {"proc": None}

    def _on_signal(_signum: int, _frame: object) -> None:
        stop_flag["stop"] = True
        proc = active_proc["proc"]
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, _on_signal)
        signal.signal(signal.SIGTERM, _on_signal)

    chunk_index = 0
    last_rc: int | None = None
    max_log_bytes = int(max(1.0, float(max_log_mb)) * 1024 * 1024)

    while max_chunks <= 0 or chunk_index < max_chunks:
        if stop_flag["stop"] or stop_file.exists():
            _write_daemon_status(workspace_dir, "stopped", chunk_index, last_rc)
            break

        _rotate_log_if_needed(daemon_log, max_log_bytes)
        _rotate_log_if_needed(launchd_log, max_log_bytes)
        prune_forever_workspace(workspace_dir)

        st = _read_json_safe(state_path)
        halted = st.get("halted_reason") if st else None
        if halted in ("kernel_integrity_violation", "stop_file", "signal"):
            _write_daemon_status(workspace_dir, "halted", chunk_index, last_rc)
            return {"chunks_completed": chunk_index, "status": "halted", "halted_reason": halted}

        if halted == "resource_exhausted":
            prune_forever_workspace(
                workspace_dir,
                keep_gguf_candidates=1,
                keep_synth_dirs=1,
                keep_scorecards=2,
                keep_reviews=4,
                max_trajectories=200,
            )
            free_mb = _free_disk_mb(workspace_dir)
            if free_mb <= 1024.0 or st is None:
                _write_daemon_status(workspace_dir, "halted_disk", chunk_index, last_rc)
                return {"chunks_completed": chunk_index, "status": "halted_disk", "halted_reason": halted, "free_disk_mb": free_mb}
            st["halted_reason"] = None
            atomic_write_text(state_path, json.dumps(st, indent=2, ensure_ascii=False) + "\n")
            _append_event(events_path, {"event": "resource_recovery", "free_disk_mb": free_mb})

        elif halted == "safe_mode_exhausted":
            rec_cnt = _recent_recovery_count(events_path, hours=24)
            if rec_cnt >= max(0, int(max_safe_mode_recoveries)) or st is None:
                _write_daemon_status(workspace_dir, "halted_recovery_exhausted", chunk_index, last_rc)
                return {"chunks_completed": chunk_index, "status": "halted_recovery_exhausted", "halted_reason": halted}
            _write_daemon_status(workspace_dir, "cooldown_safe_mode_recovery", chunk_index, last_rc)
            if not _interruptible_sleep(recovery_cooldown_sec, stop_flag, stop_file):
                _write_daemon_status(workspace_dir, "stopped", chunk_index, last_rc)
                break
            st["halted_reason"] = None
            st["safe_mode"] = True
            st["config_only"] = True
            st["safe_mode_failures"] = 0
            st["consecutive_failures"] = 0
            atomic_write_text(state_path, json.dumps(st, indent=2, ensure_ascii=False) + "\n")
            _append_event(events_path, {"event": "safe_mode_recovery", "recovery_index_24h": rec_cnt + 1})

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "selfchange.colab.colab_launcher",
            "--repo-root",
            str(repo_root),
            "--workspace-dir",
            str(workspace_dir),
            "--profile",
            profile_name,
            "--max-cycles",
            str(max(1, int(chunk_cycles))),
            "--autonomy-level",
            str(int(autonomy_level)),
            "--train-every-n-cycles",
            str(int(train_every_n_cycles)),
            "--enable-local-cpu-train",
            "--local-sft-count",
            str(int(local_sft_count)),
            "--local-dpo-count",
            str(int(local_dpo_count)),
            "--local-train-timeout",
            str(int(local_train_timeout)),
            "--local-cpu-threads",
            str(int(local_cpu_threads)),
            "--eval-with-model",
            "--require-model",
        ]
        if model_path is not None:
            cmd.extend(["--model-path", str(model_path)])
        if train_base_pth is not None:
            cmd.extend(["--train-base-pth", str(train_base_pth)])
        if anchor_path is not None:
            cmd.extend(["--anchor-path", str(anchor_path)])

        chunk_index += 1
        _append_event(events_path, {"event": "chunk_start", "chunk_index": chunk_index})
        _write_daemon_status(workspace_dir, "running_chunk", chunk_index, last_rc)

        if child_runner is not None:
            last_rc = int(child_runner(cmd, repo_root, daemon_log))
        else:
            with daemon_log.open("a", encoding="utf-8") as lf:
                lf.write(f"\n=== [{_utc_now()}] CHUNK {chunk_index} START ===\n")
                lf.flush()
                proc = subprocess.Popen(cmd, cwd=str(repo_root), stdout=lf, stderr=subprocess.STDOUT, text=True)
                active_proc["proc"] = proc
                try:
                    last_rc = proc.wait()
                finally:
                    active_proc["proc"] = None
                    lf.write(f"\n=== [{_utc_now()}] CHUNK {chunk_index} END rc={last_rc} ===\n")
                    lf.flush()

        prune_forever_workspace(workspace_dir)
        _append_event(events_path, {"event": "chunk_end", "chunk_index": chunk_index, "rc": last_rc})
        _write_daemon_status(workspace_dir, "idle", chunk_index, last_rc)

        if max_chunks > 0 and chunk_index >= max_chunks:
            break
        wait_sec = float(cooldown_sec) if last_rc == 0 else max(60.0, float(cooldown_sec))
        if not _interruptible_sleep(wait_sec, stop_flag, stop_file):
            _write_daemon_status(workspace_dir, "stopped", chunk_index, last_rc)
            break

    final_st = _read_json_safe(state_path) or {}
    return {
        "chunks_completed": chunk_index,
        "last_rc": last_rc,
        "total_cycles": int(final_st.get("cycle_index", 0)),
        "halted_reason": final_st.get("halted_reason"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Self-recycling endless daemon for MiniMind selfchange")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--workspace-dir", type=Path, default=Path(__file__).resolve().parents[1] / "workspace" / "local_train")
    parser.add_argument("--profile", type=str, default="local_intel_cpu")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--train-base-pth", type=Path, default=None)
    parser.add_argument("--anchor-path", type=Path, default=None)
    parser.add_argument("--autonomy-level", type=int, default=1)
    parser.add_argument("--train-every-n-cycles", type=int, default=2)
    parser.add_argument("--chunk-cycles", type=int, default=4)
    parser.add_argument("--local-sft-count", type=int, default=48)
    parser.add_argument("--local-dpo-count", type=int, default=16)
    parser.add_argument("--local-train-timeout", type=int, default=1200)
    parser.add_argument("--local-cpu-threads", type=int, default=2)
    parser.add_argument("--cooldown-sec", type=float, default=20.0)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--max-safe-mode-recoveries", type=int, default=3)
    parser.add_argument("--recovery-cooldown-sec", type=float, default=300.0)
    parser.add_argument("--max-log-mb", type=float, default=10.0)
    args = parser.parse_args()

    out = run_forever_daemon(
        repo_root=args.repo_root,
        workspace_dir=args.workspace_dir,
        profile_name=args.profile,
        model_path=args.model_path,
        train_base_pth=args.train_base_pth,
        anchor_path=args.anchor_path,
        autonomy_level=args.autonomy_level,
        train_every_n_cycles=args.train_every_n_cycles,
        chunk_cycles=args.chunk_cycles,
        local_sft_count=args.local_sft_count,
        local_dpo_count=args.local_dpo_count,
        local_train_timeout=args.local_train_timeout,
        local_cpu_threads=args.local_cpu_threads,
        cooldown_sec=args.cooldown_sec,
        max_chunks=args.max_chunks,
        max_safe_mode_recoveries=args.max_safe_mode_recoveries,
        recovery_cooldown_sec=args.recovery_cooldown_sec,
        max_log_mb=args.max_log_mb,
    )
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
