from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from selfchange.kernel.integrity import atomic_write_text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_remote_phase_scripts(
    repo_url: str = "https://github.com/ed2k/minimind.git",
    branch: str = "mychanges",
    total_traces: int = 3000,
    warmstart_stage: str = "full_sft",
    max_cycles: int = 10,
    autonomy_level: int = 1,
    always_persist_weights: bool = False,
    lightweight_setup: bool = False,
    explicit_model_path: str | None = None,
) -> dict[str, str]:
    """Build the 3 remote Colab Python driver scripts matching run_selfchange.ipynb."""
    persist_flag = '        "--always-persist-weights",\n' if always_persist_weights else ""
    model_flag = (
        f'        "--model-path",\n        {explicit_model_path!r},\n'
        if explicit_model_path
        else ""
    )

    if lightweight_setup:
        setup_tail = """    run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from selfchange.bootstrap.setup_base import build_synthetic_anchor_datasets; "
                "build_synthetic_anchor_datasets(Path('selfchange/workspace/models/base/data'), train_count=400, heldout_count=60, seed=42)"
            ),
        ],
        cwd=str(repo),
    )"""
    else:
        setup_tail = """    run(
        [
            sys.executable,
            "-m",
            "selfchange.bootstrap.setup_base",
            "--mode",
            "colab",
            "--repo-root",
            ".",
            "--output-dir",
            "selfchange/workspace/models/base",
        ],
        cwd=str(repo),
    )
    run(
        [
            sys.executable,
            "-m",
            "selfchange.bootstrap.eval_baseline",
            "--worktree",
            ".",
            "--backend",
            "minimind",
            "--model-path",
            "selfchange/workspace/models/base/minimind3_768.pth",
            "--tokenizer-path",
            "model",
            "--heldout-path",
            "selfchange/workspace/models/base/data/heldout_eval.jsonl",
            "--output",
            "selfchange/workspace/baseline_untouched.json",
        ],
        cwd=str(repo),
        check=False,
    )"""

    step1 = f"""import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: str | None = None, check: bool = True) -> int:
    print(f"+ {{' '.join(cmd)}}", flush=True)
    proc = subprocess.run(cmd, cwd=cwd)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.returncode


def main() -> None:
    run(["nvidia-smi"], check=False)
    repo = Path("/content/minimind")
    branch = {branch!r}
    repo_url = {repo_url!r}
    if (repo / ".git").is_dir():
        run(["git", "fetch", "origin", branch], cwd=str(repo))
        run(["git", "checkout", branch], cwd=str(repo))
        run(["git", "pull", "--ff-only", "origin", branch], cwd=str(repo))
    else:
        run(["git", "clone", "--depth", "1", "-b", branch, repo_url, str(repo)], cwd="/content")

    run(
        [
            sys.executable,
            "-m",
            "pip",
            "-q",
            "install",
            "transformers==4.57.6",
            "datasets==3.6.0",
            "huggingface_hub",
            "safetensors",
            "accelerate",
        ],
        cwd=str(repo),
    )
    os.chdir(repo)
    run([sys.executable, "-m", "selfchange.kernel.supervisor", "--check-only"], cwd=str(repo))
{setup_tail}


if __name__ == "__main__":
    main()
"""

    step2 = f"""import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], cwd: Path, check: bool = True) -> int:
    print(f"+ {{' '.join(cmd)}}", flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd))
    if check and proc.returncode != 0:
        log_path = cwd / "selfchange" / "workspace" / "logs" / "train.log"
        if log_path.is_file():
            print(log_path.read_text(encoding="utf-8"), flush=True)
        raise SystemExit(proc.returncode)
    return proc.returncode


def main() -> None:
    repo = Path("/content/minimind")
    os.chdir(repo)
    run(
        [
            sys.executable,
            "-m",
            "selfchange.bootstrap.warmstart_sft",
            "--work-dir",
            "selfchange/workspace/bootstrap_data",
            "--anchor-jsonl",
            "selfchange/workspace/models/base/data/anchor_sft.jsonl",
            "--total-traces",
            {str(int(total_traces))!r},
            "--stage",
            {warmstart_stage!r},
        ],
        cwd=repo,
    )
    run(
        [
            sys.executable,
            "-m",
            "selfchange.bootstrap.offline_rft_dpo",
            "--backend",
            "minimind",
            "--model-path",
            "auto",
            "--tokenizer-path",
            "model",
            "--anchor-jsonl",
            "selfchange/workspace/models/base/data/anchor_sft.jsonl",
            "--from-weight",
            "warmstart_rung3",
            "--rounds",
            "1",
            "--num-tasks",
            "24",
            "--num-rollouts",
            "4",
        ],
        cwd=repo,
    )
    run(
        [
            sys.executable,
            "-m",
            "selfchange.bootstrap.eval_baseline",
            "--worktree",
            ".",
            "--backend",
            "minimind",
            "--model-path",
            "auto",
            "--tokenizer-path",
            "model",
            "--heldout-path",
            "selfchange/workspace/models/base/data/heldout_eval.jsonl",
            "--output",
            "selfchange/workspace/baseline_prepared.json",
        ],
        cwd=repo,
        check=False,
    )


if __name__ == "__main__":
    main()
"""

    step3 = f"""import os
import subprocess
import sys
import tarfile
from pathlib import Path


def _pack_diagnostics(repo: Path, drive_mirror: Path, tar_out: Path) -> None:
    ws = repo / "selfchange" / "workspace"
    added: set[str] = set()
    with tarfile.open(tar_out, "w:gz") as tf:
        if drive_mirror.is_dir():
            for item in sorted(drive_mirror.rglob("*")):
                if not item.is_file() or item.suffix == ".pth":
                    continue
                rel = item.relative_to(drive_mirror).as_posix()
                tf.add(item, arcname=rel)
                added.add(rel)

        fallbacks = [
            "report.md",
            "session_info.json",
            "baseline_untouched.json",
            "baseline_prepared.json",
            "live_scorecard_start.json",
            "live_scorecard_end.json",
            "live_scorecard_latest.json",
            "curated_data/verified_selfchange_episodes.jsonl",
            "curated_data/verified_selfchange_dpo.jsonl",
            "logs/train.log",
            "logs/train_history.jsonl",
            "logs/fatal_report.json",
            "archive.jsonl",
            "champion.json",
            "supervisor_state.json",
            "HEARTBEAT.json",
            "trajectories.jsonl",
        ]
        for rel in fallbacks:
            if rel in added:
                continue
            src = ws / rel
            if src.is_file():
                tf.add(src, arcname=rel)
                added.add(rel)
        champ_dir = ws / "promoted_models"
        for extra_name in ("model_config.json",):
            p = champ_dir / extra_name
            if p.is_file() and extra_name not in added:
                tf.add(p, arcname=extra_name)
                added.add(extra_name)


def main() -> None:
    repo = Path("/content/minimind")
    ws = repo / "selfchange" / "workspace"
    drive_mirror = Path("/content/drive_mirror")
    tar_out = Path("/content/run_diagnostics.tar.gz")
    os.chdir(repo)
    cmd = [
        sys.executable,
        "-m",
        "selfchange.colab.colab_launcher",
        "--drive-dir",
        str(drive_mirror),
        "--require-model",
        "--eval-with-model",
        "--max-cycles",
        {str(int(max_cycles))!r},
        "--autonomy-level",
        {str(int(autonomy_level))!r},
{model_flag}{persist_flag}    ]
    proc = subprocess.run(cmd, cwd=str(repo))
    report = ws / "report.md"
    if report.is_file():
        print(report.read_text(encoding="utf-8"), flush=True)
    _pack_diagnostics(repo, drive_mirror, tar_out)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
"""
    return {"step1_setup.py": step1, "step2_bootstrap.py": step2, "step3_autonomous.py": step3}


def build_minimal_cpu_search_script(
    repo_url: str = "https://github.com/ed2k/minimind.git",
    branch: str = "mychanges",
    rungs: str | None = None,
    max_rungs: int = 4,
    total_traces: int = 600,
    always_persist_weights: bool = True,
) -> dict[str, str]:
    """Build remote scripts that prepare the 64M teacher and bracket-search for the minimal CPU model."""
    base_scripts = build_remote_phase_scripts(
        repo_url=repo_url,
        branch=branch,
        total_traces=total_traces,
        always_persist_weights=always_persist_weights,
        lightweight_setup=False,
    )
    rungs_flag = f'        "--rungs",\n        {rungs!r},\n' if rungs else ""
    step2_search = f"""import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def run(cmd: list[str], cwd: str | None = None, check: bool = True) -> int:
    print(f"+ {{' '.join(cmd)}}", flush=True)
    proc = subprocess.run(cmd, cwd=cwd)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.returncode


def main() -> None:
    repo = Path("/content/minimind")
    ws = repo / "selfchange" / "workspace"
    drive_mirror = Path("/content/drive_mirror")
    os.chdir(repo)

    # 1) Specialize 64M teacher via Stage 1 warm-start before distillation search
    run(
        [
            sys.executable,
            "-m",
            "selfchange.bootstrap.warmstart_sft",
            "--repo-root",
            ".",
            "--workspace-dir",
            "selfchange/workspace",
            "--anchor-jsonl",
            "selfchange/workspace/models/base/data/anchor_sft.jsonl",
            "--total-traces",
            {str(int(total_traces))!r},
            "--stage",
            "full_sft",
        ],
        cwd=str(repo),
    )

    teacher_w = ws / "models" / "candidates" / "warmstart_rung3_768.pth"
    if not teacher_w.is_file():
        teacher_w = ws / "models" / "base" / "minimind3_768.pth"

    cmd = [
        sys.executable,
        "-m",
        "selfchange.bootstrap.search_minimal_cpu",
        "--repo-root",
        ".",
        "--workspace-dir",
        "selfchange/workspace",
        "--teacher-weight",
        "warmstart_rung3",
        "--teacher-weight-path",
        str(teacher_w),
        "--sft-count",
        {str(max(120, int(total_traces)))!r},
        "--max-rungs",
        {str(int(max_rungs))!r},
{rungs_flag}    ]
    proc = subprocess.run(cmd, cwd=str(repo))

    rep_file = ws / "search_frontier_report.json"
    rep = json.loads(rep_file.read_text(encoding="utf-8")) if rep_file.is_file() else {{}}
    champ = rep.get("pareto_champion") or {{}}
    latest_meta = ws / "promoted_models" / "latest_champion.json"
    latest = json.loads(latest_meta.read_text(encoding="utf-8")) if latest_meta.is_file() else {{}}

    drive_mirror.mkdir(parents=True, exist_ok=True)
    w_dir = drive_mirror / "weights"
    w_dir.mkdir(parents=True, exist_ok=True)
    weight_rel_map = {{}}
    cw = latest.get("champion_weight_path")
    if cw and Path(cw).is_file():
        dst_w = w_dir / Path(cw).name
        shutil.copy2(cw, dst_w)
        weight_rel_map[f"weights/{{Path(cw).name}}"] = str(cw)
        for extra in (Path(cw).with_suffix(".config.json"), ws / "promoted_models" / "model_config.json"):
            if extra.is_file():
                shutil.copy2(extra, drive_mirror / extra.name)

    session_info = {{
        "persistence_decision": {{
            "persist_weights": bool(champ) or {str(bool(always_persist_weights))},
            "reason": f"pareto_champion_{{champ.get('rung_id', 'none')}}",
            "champion_weight_path": cw,
        }},
        "pareto_champion": champ,
    }}
    (drive_mirror / "session_info.json").write_text(json.dumps(session_info, indent=2), encoding="utf-8")
    (drive_mirror / "manifest.json").write_text(json.dumps({{"weight_rel_map": weight_rel_map, "files": {{}}}}, indent=2), encoding="utf-8")
    for fn in ("search_frontier_report.json", "search_frontier_report.md"):
        if (ws / fn).is_file():
            shutil.copy2(ws / fn, drive_mirror / fn)

    with tarfile.open("/content/run_diagnostics.tar.gz", "w:gz") as tf:
        for p in sorted(drive_mirror.iterdir()):
            if p.is_file():
                tf.add(p, arcname=p.name)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
"""
    return {"step1_setup.py": base_scripts["step1_setup.py"], "step2_search_minimal_cpu.py": step2_search}


def _resolve_export_pths(workspace_dir: Path, promoted_dir: Path, export_pth: str | None) -> list[Path]:
    paths: list[Path] = []
    if export_pth:
        for piece in str(export_pth).split(","):
            s = piece.strip()
            if not s:
                continue
            p = Path(s).resolve()
            if p.is_file() and p not in paths:
                paths.append(p)
        return paths

    defaults = [
        promoted_dir / "minimal_cpu_search" / "minimal_cpu_champion.pth",
        promoted_dir / "limitation_fix_v2" / "fixed_champion_768.pth",
    ]
    latest_json = promoted_dir / "latest_champion.json"
    if latest_json.is_file():
        try:
            pw = json.loads(latest_json.read_text(encoding="utf-8")).get("primary_weight")
            if pw:
                defaults.append(Path(pw))
        except Exception:
            pass
    for d in defaults:
        if d.is_file() and d.resolve() not in paths:
            paths.append(d.resolve())
    return paths


def _extract_gguf_artifacts(tar_local: Path, gguf_dir: Path) -> dict:
    gguf_dir = Path(gguf_dir).resolve()
    gguf_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_local, "r:gz") as tf:
        tf.extractall(gguf_dir, filter="data")

    latest_file = gguf_dir / "latest_gguf.json"
    info: dict = {"gguf_dir": str(gguf_dir), "files": sorted(p.name for p in gguf_dir.iterdir() if p.is_file())}
    if latest_file.is_file():
        try:
            meta = json.loads(latest_file.read_text(encoding="utf-8"))
            if meta.get("champion_gguf"):
                meta["champion_gguf"] = str((gguf_dir / Path(meta["champion_gguf"]).name).resolve())
            if isinstance(meta.get("selected_artifact"), dict) and meta["selected_artifact"].get("path"):
                meta["selected_artifact"]["path"] = str((gguf_dir / Path(meta["selected_artifact"]["path"]).name).resolve())
            for rep in meta.get("reports", []):
                if isinstance(rep.get("best_gguf"), dict) and rep["best_gguf"].get("path"):
                    rep["best_gguf"]["path"] = str((gguf_dir / Path(rep["best_gguf"]["path"]).name).resolve())
                for art in rep.get("artifacts", []):
                    if isinstance(art, dict) and art.get("path"):
                        art["path"] = str((gguf_dir / Path(art["path"]).name).resolve())
            atomic_write_text(latest_file, json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
            info["latest_gguf"] = meta
        except Exception:
            pass
    return info


def build_gguf_export_script(
    remote_pths: list[str],
    quants: str = "Q8_0,Q4_K_M",
    verify: bool = True,
    max_tasks: int = 20,
    fallback_remote_pth: str | None = None,
) -> str:
    """Build remote Colab script that exports .pth champion(s) -> verified GGUF and packs /content/run_gguf_artifacts.tar.gz."""
    pth_list = list(remote_pths)
    verify_flag = "--verify" if verify else "--no-verify"
    return f"""import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

repo = Path("/content/minimind")
ws = repo / "selfchange" / "workspace"
gguf_dir = ws / "promoted_models" / "gguf"
gguf_dir.mkdir(parents=True, exist_ok=True)
os.chdir(repo)

pths = [p for p in {pth_list!r} if Path(p).is_file()]
if not pths:
    champ_json = ws / "champion.json"
    if champ_json.is_file():
        try:
            cw = json.loads(champ_json.read_text(encoding="utf-8")).get("champion", {{}}).get("weight_path")
            if cw:
                cp = Path(cw) if Path(cw).is_absolute() else (ws / cw)
                if cp.is_file():
                    pths = [str(cp.resolve())]
        except Exception:
            pass
if not pths and {fallback_remote_pth!r} and Path({fallback_remote_pth!r}).is_file():
    pths = [{fallback_remote_pth!r}]
if not pths:
    raise FileNotFoundError("No .pth checkpoint available on Colab VM to export to GGUF.")

cmd = [sys.executable, "-m", "selfchange.colab.export_gguf", "--out-dir", str(gguf_dir), "--quants", {quants!r}, {verify_flag!r}, "--max-tasks", {str(int(max_tasks))!r}]
for p in pths:
    cmd.extend(["--pth", p])
proc = subprocess.run(cmd, cwd=str(repo), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
if proc.stdout:
    print(proc.stdout[-8000:], flush=True)
if proc.returncode != 0:
    raise RuntimeError(f"export_gguf failed with exit={{proc.returncode}}")

tar_out = Path("/content/run_gguf_artifacts.tar.gz")
with tarfile.open(tar_out, "w:gz") as tf:
    for p in sorted(gguf_dir.iterdir()):
        if p.is_file() and (p.suffix == ".gguf" or p.suffix == ".json"):
            tf.add(p, arcname=p.name)
print(f"Packed {{tar_out}} ({{tar_out.stat().st_size}} bytes)", flush=True)
"""


def _ensure_gdrive_folder(folder_name: str, run_cmd: Callable) -> str | None:
    search = run_cmd(
        ["gdrive", "readonly", "search", "--json", "--name-exact", folder_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if getattr(search, "returncode", 1) == 0 and getattr(search, "stdout", ""):
        try:
            data = json.loads(search.stdout)
            items = data if isinstance(data, list) else data.get("files", [])
            for item in items:
                if item.get("mimeType") == "application/vnd.google-apps.folder" and item.get("id"):
                    return str(item["id"])
        except json.JSONDecodeError:
            pass

    created = run_cmd(
        ["gdrive", "mutate", "mkdir", "--json", folder_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if getattr(created, "returncode", 1) == 0 and getattr(created, "stdout", ""):
        try:
            obj = json.loads(created.stdout)
            if isinstance(obj, dict) and obj.get("id"):
                return str(obj["id"])
        except json.JSONDecodeError:
            pass
    return None


def persist_improved_weights_from_colab(
    session: str,
    diagnostics_dir: Path,
    promoted_dir: Path,
    run_id: str,
    always_persist: bool = False,
    upload_drive: bool = False,
    run_cmd: Callable = subprocess.run,
) -> dict:
    """Persist promoted .pth weights from the Colab VM locally/Drive iff the run improved."""
    diagnostics_dir = Path(diagnostics_dir).resolve()
    promoted_dir = Path(promoted_dir).resolve()
    session_file = diagnostics_dir / "session_info.json"
    manifest_file = diagnostics_dir / "manifest.json"

    session_info = json.loads(session_file.read_text(encoding="utf-8")) if session_file.is_file() else {}
    manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.is_file() else {}
    decision = session_info.get("persistence_decision") or {}

    should_persist = bool(always_persist or decision.get("persist_weights"))
    if not should_persist:
        return {
            "persisted": False,
            "reason": decision.get("reason", "no_improvement_over_baseline"),
            "local_weights": [],
            "drive_uploaded": [],
        }

    remote_weights: list[str] = []
    for rel_drive in manifest.get("weight_rel_map", {}):
        rpath = f"/content/drive_mirror/{rel_drive}"
        if rpath not in remote_weights:
            remote_weights.append(rpath)
    champ_w = decision.get("champion_weight_path")
    if not remote_weights and champ_w:
        remote_weights.append(str(champ_w))

    if not remote_weights:
        return {
            "persisted": False,
            "reason": "no_weight_paths_in_manifest",
            "local_weights": [],
            "drive_uploaded": [],
        }

    run_weights_dir = promoted_dir / run_id
    run_weights_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    for rpath in remote_weights:
        local_dst = run_weights_dir / Path(rpath).name
        run_cmd(["colab", "download", "-s", session, rpath, str(local_dst)], check=True)
        downloaded.append(str(local_dst))
    for cfg_name in ("model_config.json", f"{Path(downloaded[0]).stem}.config.json"):
        diag_cfg = diagnostics_dir / cfg_name
        if diag_cfg.is_file():
            shutil.copy2(diag_cfg, run_weights_dir / cfg_name)
            shutil.copy2(diag_cfg, promoted_dir / cfg_name)

    champ_meta = {
        "run_id": run_id,
        "persisted_at": _utc_now(),
        "reason": decision.get("reason") or ("always_persist" if always_persist else "improved"),
        "decision": decision,
        "diagnostics_dir": str(diagnostics_dir),
        "primary_weight": downloaded[0],
        "weights": downloaded,
    }
    atomic_write_text(promoted_dir / "latest_champion.json", json.dumps(champ_meta, indent=2, ensure_ascii=False) + "\n")

    drive_uploaded: list[str] = []
    if upload_drive:
        folder_id = _ensure_gdrive_folder("minimind_selfchange", run_cmd)
        if folder_id:
            for item in [*downloaded, str(diagnostics_dir / "report.md"), str(diagnostics_dir / "manifest.json")]:
                if Path(item).is_file():
                    up = run_cmd(["gdrive", "mutate", "upload", item, "--parent", folder_id], check=False)
                    if getattr(up, "returncode", 1) == 0:
                        drive_uploaded.append(item)

    return {
        "persisted": True,
        "reason": champ_meta["reason"],
        "local_weights": downloaded,
        "latest_champion_json": str(promoted_dir / "latest_champion.json"),
        "drive_uploaded": drive_uploaded,
    }


def inspect_resume_seed(promoted_dir: Path) -> dict:
    """Inspect promoted_models/latest_champion.json and decide how to seed the next Colab VM."""
    promoted_dir = Path(promoted_dir).resolve()
    latest_file = promoted_dir / "latest_champion.json"
    if not latest_file.is_file():
        return {"mode": "none", "reason": "no_latest_champion"}
    try:
        data = json.loads(latest_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"mode": "none", "reason": "invalid_latest_champion"}

    cand_paths: list[Path] = []
    for raw in [*(data.get("weights") or []), data.get("primary_weight")]:
        if raw:
            p = Path(raw)
            if p.is_file() and p not in cand_paths:
                cand_paths.append(p)
    run_id = data.get("run_id")
    if run_id and (promoted_dir / str(run_id)).is_dir():
        for p in sorted((promoted_dir / str(run_id)).glob("*.pth")):
            if p not in cand_paths:
                cand_paths.append(p)

    if not cand_paths:
        return {"mode": "none", "reason": "missing_weight_files"}

    local_by_name = {p.name: p for p in cand_paths}
    primary_pth = cand_paths[0]
    ws_dir = promoted_dir.parent
    cur_dir = ws_dir / "curated_data"
    extra_carry: list[tuple[str, str]] = []
    sib_cfg = primary_pth.with_suffix(".config.json")
    if sib_cfg.is_file():
        extra_carry.append((str(sib_cfg), f"/content/minimind/selfchange/workspace/models/candidates/{primary_pth.stem}.config.json"))
    elif (promoted_dir / "model_config.json").is_file():
        extra_carry.append((str(promoted_dir / "model_config.json"), f"/content/minimind/selfchange/workspace/models/candidates/{primary_pth.stem}.config.json"))
    for ep_name in ("verified_selfchange_episodes.jsonl", "verified_selfchange_dpo.jsonl"):
        ep_f = cur_dir / ep_name
        if ep_f.is_file() and ep_f.stat().st_size > 0:
            extra_carry.append((str(ep_f), f"/content/minimind/selfchange/workspace/curated_data/{ep_name}"))

    diag_raw = data.get("diagnostics_dir")
    diag_dir = Path(diag_raw) if diag_raw else None
    manifest_file = (diag_dir / "manifest.json") if diag_dir else None

    if diag_dir is not None and manifest_file is not None and manifest_file.is_file():
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            files_map = manifest.get("files", {})
            weight_rel_map = manifest.get("weight_rel_map", {})
            non_weight_rels = [rel for rel in files_map if rel not in weight_rel_map]
            non_weights_ok = all((diag_dir / rel).is_file() for rel in non_weight_rels)
            weight_uploads: list[tuple[str, str]] = []
            weights_ok = bool(weight_rel_map)
            for rel_drive in weight_rel_map:
                matched = local_by_name.get(Path(rel_drive).name)
                if matched is None:
                    weights_ok = False
                    break
                weight_uploads.append((str(matched), f"/content/drive_mirror/{rel_drive}"))
            if non_weights_ok and weights_ok:
                return {
                    "mode": "full_mirror",
                    "run_id": run_id,
                    "diagnostics_dir": str(diag_dir),
                    "non_weight_files": non_weight_rels,
                    "weight_uploads": weight_uploads,
                    "primary_weight": str(primary_pth),
                    "extra_uploads": extra_carry,
                }
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "mode": "weight_only",
        "run_id": run_id,
        "primary_weight": str(primary_pth),
        "remote_model_path": f"/content/minimind/selfchange/workspace/models/candidates/{primary_pth.name}",
        "extra_uploads": extra_carry,
    }


def _upload_file_chunked(
    session: str,
    local_path: Path,
    remote_path: str,
    work_dir: Path,
    run_cmd: Callable = subprocess.run,
    chunk_bytes: int = 25 * 1024 * 1024,
) -> list[str]:
    """Upload file directly if <= chunk_bytes, else shard and reassemble remotely to avoid proxy 500s."""
    local_path = Path(local_path)
    if local_path.stat().st_size <= chunk_bytes:
        run_cmd(["colab", "upload", "-s", session, str(local_path), remote_path], check=True)
        return [remote_path]

    stem_tag = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in local_path.stem)
    chunk_dir = Path(work_dir) / "upload_chunks" / stem_tag
    chunk_dir.mkdir(parents=True, exist_ok=True)
    remote_parts: list[str] = []
    idx = 0
    with local_path.open("rb") as rf:
        while True:
            buf = rf.read(chunk_bytes)
            if not buf:
                break
            part_name = f"part_{idx:03d}"
            local_part = chunk_dir / part_name
            local_part.write_bytes(buf)
            remote_part = f"/content/_seed_part_{stem_tag}_{idx:03d}"
            run_cmd(["colab", "upload", "-s", session, str(local_part), remote_part], check=True)
            remote_parts.append(remote_part)
            idx += 1

    reassemble_script = Path(work_dir) / f"step_reassemble_{stem_tag}.py"
    reassemble_src = f"""from pathlib import Path

dst = Path({remote_path!r})
dst.parent.mkdir(parents=True, exist_ok=True)
parts = {remote_parts!r}
with dst.open("wb") as wf:
    for p_str in parts:
        p = Path(p_str)
        wf.write(p.read_bytes())
        p.unlink(missing_ok=True)
print(f"Reassembled {{dst}} ({{dst.stat().st_size}} bytes) from {{len(parts)}} parts", flush=True)
"""
    atomic_write_text(reassemble_script, reassemble_src)
    run_cmd(["colab", "exec", "-s", session, "--timeout", "180", "-f", str(reassemble_script)], check=True)
    return [remote_path]


def seed_colab_vm_from_promoted(
    session: str,
    seed_plan: dict,
    work_dir: Path,
    run_cmd: Callable = subprocess.run,
) -> dict:
    """Upload persisted local mirror state and/or .pth weights to the newly provisioned Colab VM."""
    mode = seed_plan.get("mode", "none")
    if mode == "none":
        return {"seeded": False, "mode": "none", "uploaded_files": []}

    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    uploaded: list[str] = []
    extra_uploads = [(str(lp), str(rp)) for lp, rp in seed_plan.get("extra_uploads", [])]
    extra_dirs = sorted({str(Path(rp).parent) for _, rp in extra_uploads})

    if mode == "full_mirror":
        diag_dir = Path(seed_plan["diagnostics_dir"])
        tar_path = work_dir / "seed_mirror_state.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(diag_dir / "manifest.json", arcname="manifest.json")
            for rel in seed_plan["non_weight_files"]:
                tf.add(diag_dir / rel, arcname=rel)

        remote_tar = "/content/seed_mirror_state.tar.gz"
        run_cmd(["colab", "upload", "-s", session, str(tar_path), remote_tar], check=True)
        uploaded.append(remote_tar)

        unpack_script = work_dir / "step_unpack_seed.py"
        unpack_src = f"""import shutil
import tarfile
from pathlib import Path

mirror = Path("/content/drive_mirror")
if mirror.exists():
    shutil.rmtree(mirror)
mirror.mkdir(parents=True, exist_ok=True)
with tarfile.open("/content/seed_mirror_state.tar.gz", "r:gz") as tf:
    tf.extractall(mirror, filter="data")
(mirror / "weights").mkdir(parents=True, exist_ok=True)
for extra_dir in {extra_dirs!r}:
    Path(extra_dir).mkdir(parents=True, exist_ok=True)
print(f"Seeded {{mirror}} with {{len(list(mirror.rglob('*')))}} entries", flush=True)
"""
        atomic_write_text(unpack_script, unpack_src)
        run_cmd(["colab", "exec", "-s", session, "--timeout", "120", "-f", str(unpack_script)], check=True)

        for local_w, remote_w in seed_plan["weight_uploads"]:
            uploaded.extend(_upload_file_chunked(session, Path(local_w), remote_w, work_dir, run_cmd=run_cmd))
        for local_extra, remote_extra in extra_uploads:
            uploaded.extend(_upload_file_chunked(session, Path(local_extra), remote_extra, work_dir, run_cmd=run_cmd))
        return {"seeded": True, "mode": "full_mirror", "uploaded_files": uploaded}

    # weight_only fallback
    remote_w = str(seed_plan["remote_model_path"])
    prep_script = work_dir / "step_prep_weight_dir.py"
    all_dirs = sorted({str(Path(remote_w).parent), *extra_dirs})
    prep_src = f"""from pathlib import Path

for target_str in {all_dirs!r}:
    target = Path(target_str)
    target.mkdir(parents=True, exist_ok=True)
    print(f"Prepared dir {{target}}", flush=True)
"""
    atomic_write_text(prep_script, prep_src)
    run_cmd(["colab", "exec", "-s", session, "--timeout", "60", "-f", str(prep_script)], check=True)
    uploaded.extend(_upload_file_chunked(session, Path(seed_plan["primary_weight"]), remote_w, work_dir, run_cmd=run_cmd))
    for local_extra, remote_extra in extra_uploads:
        uploaded.extend(_upload_file_chunked(session, Path(local_extra), remote_extra, work_dir, run_cmd=run_cmd))
    return {"seeded": True, "mode": "weight_only", "uploaded_files": uploaded}


def _overlay_local_code_to_colab(
    session: str,
    repo_root: Path,
    work_dir: Path,
    run_cmd: Callable = subprocess.run,
) -> list[str]:
    """Pack local selfchange/trainer Python updates (excluding workspace/kernel/caches) and apply on Colab."""
    repo_root = Path(repo_root).resolve()
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    tar_path = work_dir / "local_code_overlay.tar.gz"

    include_roots = [
        repo_root / "selfchange" / "bootstrap",
        repo_root / "selfchange" / "colab",
        repo_root / "selfchange" / "training",
        repo_root / "selfchange" / "tests",
        repo_root / "trainer",
    ]
    packed_files: list[Path] = []
    for root in include_roots:
        if root.is_dir():
            for p in sorted(root.rglob("*.py")):
                if "__pycache__" in p.parts:
                    continue
                packed_files.append(p)
    top_selfchange = repo_root / "selfchange"
    if top_selfchange.is_dir():
        for top_py in sorted(top_selfchange.glob("*.py")):
            packed_files.append(top_py)

    with tarfile.open(tar_path, "w:gz") as tf:
        for p in packed_files:
            tf.add(p, arcname=p.relative_to(repo_root).as_posix())

    remote_tar = "/content/local_code_overlay.tar.gz"
    uploaded = _upload_file_chunked(session, tar_path, remote_tar, work_dir, run_cmd=run_cmd)
    apply_script = work_dir / "step_apply_overlay.py"
    apply_src = """import subprocess
import sys
import tarfile
from pathlib import Path

repo = Path("/content/minimind")
repo.mkdir(parents=True, exist_ok=True)
with tarfile.open("/content/local_code_overlay.tar.gz", "r:gz") as tf:
    tf.extractall(repo, filter="data")
subprocess.run([sys.executable, "-m", "selfchange.kernel.supervisor", "--check-only"], cwd=repo, check=True)
print("Applied local code overlay and verified kernel integrity", flush=True)
"""
    atomic_write_text(apply_script, apply_src)
    run_cmd(["colab", "exec", "-s", session, "--timeout", "120", "-f", str(apply_script)], check=True)
    return uploaded


def run_colab_cli_pipeline(
    session: str = "minimind-selfchange",
    gpu: str = "T4",
    branch: str = "mychanges",
    repo_url: str = "https://github.com/ed2k/minimind.git",
    workspace_dir: Path = Path("selfchange/workspace"),
    max_cycles: int = 10,
    autonomy_level: int = 1,
    total_traces: int = 3000,
    warmstart_stage: str = "full_sft",
    always_persist_weights: bool = False,
    upload_drive: bool = False,
    keep_vm: bool = False,
    resume: bool = True,
    skip_bootstrap_on_resume: bool = True,
    run_cmd: Callable = subprocess.run,
    mode: str = "autonomous",
    search_rungs: str | None = None,
    search_max_rungs: int = 4,
    overlay_local_code: bool = False,
    repo_root: Path | None = None,
    export_pth: str | None = None,
    gguf_quants: str = "Q8_0,Q4_K_M",
) -> dict:
    """Provision a Colab GPU VM, run autonomous loop, minimal-CPU search, or GGUF export/sync, and stop VM."""
    workspace_dir = Path(workspace_dir).resolve()
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = workspace_dir / "colab_runs" / run_id
    scripts_dir = run_dir / "scripts"
    diag_dir = run_dir / "diagnostics"
    promoted_dir = workspace_dir / "promoted_models"
    gguf_dir = promoted_dir / "gguf"

    export_uploads: list[tuple[Path, str]] = []
    if mode == "search_minimal_cpu":
        will_resume = False
        skip_bootstrap = False
        scripts = build_minimal_cpu_search_script(
            repo_url=repo_url,
            branch=branch,
            rungs=search_rungs,
            max_rungs=search_max_rungs,
            total_traces=total_traces,
            always_persist_weights=always_persist_weights or True,
        )
        seed_plan = {"mode": "none", "reason": "minimal_cpu_search"}
    elif mode == "export_gguf":
        will_resume = False
        skip_bootstrap = True
        local_pths = _resolve_export_pths(workspace_dir, promoted_dir, export_pth)
        if not local_pths:
            raise FileNotFoundError(
                f"No .pth checkpoints found to export to GGUF (export_pth={export_pth!r}, promoted_dir={promoted_dir})."
            )
        remote_pths: list[str] = []
        for idx, pth in enumerate(local_pths):
            remote_p = f"/content/minimind/selfchange/workspace/export_inputs/{idx}_{pth.name}"
            remote_pths.append(remote_p)
            export_uploads.append((pth, remote_p))
            sib_cfg = pth.with_suffix(".config.json")
            if sib_cfg.is_file():
                export_uploads.append((sib_cfg, str(Path(remote_p).with_suffix(".config.json"))))
            elif (promoted_dir / "model_config.json").is_file():
                export_uploads.append((promoted_dir / "model_config.json", str(Path(remote_p).with_suffix(".config.json"))))

        base_scripts = build_remote_phase_scripts(
            repo_url=repo_url,
            branch=branch,
            total_traces=total_traces,
            lightweight_setup=True,
        )
        scripts = {
            "step1_setup.py": base_scripts["step1_setup.py"],
            "step_prep_export.py": f"""from pathlib import Path
Path('/content/minimind/selfchange/workspace/export_inputs').mkdir(parents=True, exist_ok=True)
print('Prepared export_inputs', flush=True)
""",
            "step2_export_gguf.py": build_gguf_export_script(remote_pths, quants=gguf_quants, verify=True, max_tasks=20),
        }
        seed_plan = {"mode": "none", "reason": "export_gguf"}
    else:
        seed_plan = inspect_resume_seed(promoted_dir) if resume else {"mode": "none", "reason": "resume_disabled"}
        will_resume = seed_plan["mode"] != "none"
        skip_bootstrap = bool(will_resume and skip_bootstrap_on_resume)
        lightweight_setup = bool(will_resume and skip_bootstrap)
        explicit_model_path = seed_plan.get("remote_model_path") if seed_plan["mode"] == "weight_only" else None
        if mode == "sync_train_gguf":
            traj_local = workspace_dir / "trajectories.jsonl"
            if traj_local.is_file() and traj_local.stat().st_size > 0:
                seed_plan.setdefault("extra_uploads", []).append(
                    (str(traj_local), "/content/minimind/selfchange/workspace/trajectories.jsonl")
                )

        scripts = build_remote_phase_scripts(
            repo_url=repo_url,
            branch=branch,
            total_traces=total_traces,
            warmstart_stage=warmstart_stage,
            max_cycles=max_cycles,
            autonomy_level=autonomy_level,
            always_persist_weights=always_persist_weights,
            lightweight_setup=lightweight_setup,
            explicit_model_path=explicit_model_path,
        )
        if mode == "sync_train_gguf":
            fallback_p = seed_plan.get("remote_model_path") or (
                f"/content/drive_mirror/{seed_plan['weight_uploads'][0][1].split('/content/drive_mirror/', 1)[-1]}"
                if seed_plan.get("weight_uploads")
                else None
            )
            scripts["step4_export_gguf.py"] = build_gguf_export_script(
                [],
                quants=gguf_quants,
                verify=True,
                max_tasks=20,
                fallback_remote_pth=fallback_p,
            )
    for name, content in scripts.items():
        atomic_write_text(scripts_dir / name, content)

    run_cmd(["colab", "new", "-s", session, "--gpu", gpu], check=True)
    run_error: str | None = None
    seed_result: dict = {"seeded": False, "mode": seed_plan["mode"], "uploaded_files": []}
    overlay_files: list[str] = []
    persistence: dict = {"persisted": False, "reason": "not_evaluated", "local_weights": []}
    gguf_result: dict = {}

    try:
        run_cmd(["colab", "exec", "-s", session, "--timeout", "900", "-f", str(scripts_dir / "step1_setup.py")], check=True)
        if overlay_local_code:
            eff_repo = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
            overlay_files = _overlay_local_code_to_colab(session, eff_repo, run_dir, run_cmd=run_cmd)
        if mode == "search_minimal_cpu":
            proc = run_cmd(["colab", "exec", "-s", session, "--timeout", "2400", "-f", str(scripts_dir / "step2_search_minimal_cpu.py")], check=False)
            if getattr(proc, "returncode", 0) != 0:
                run_error = f"step2_search_minimal_cpu exited with {proc.returncode}"
        elif mode == "export_gguf":
            run_cmd(["colab", "exec", "-s", session, "--timeout", "120", "-f", str(scripts_dir / "step_prep_export.py")], check=True)
            uploaded_exp: list[str] = []
            for local_p, remote_p in export_uploads:
                uploaded_exp.extend(_upload_file_chunked(session, local_p, remote_p, run_dir, run_cmd=run_cmd))
            seed_result = {"seeded": True, "mode": "export_gguf", "uploaded_files": uploaded_exp}
            exp_proc = run_cmd(["colab", "exec", "-s", session, "--timeout", "1800", "-f", str(scripts_dir / "step2_export_gguf.py")], check=False)
            if getattr(exp_proc, "returncode", 0) != 0:
                run_error = f"step2_export_gguf exited with {exp_proc.returncode}"
        else:
            if will_resume:
                seed_result = seed_colab_vm_from_promoted(session, seed_plan, run_dir, run_cmd=run_cmd)
            if not skip_bootstrap:
                run_cmd(["colab", "exec", "-s", session, "--timeout", "1800", "-f", str(scripts_dir / "step2_bootstrap.py")], check=True)
            step3_proc = run_cmd(
                ["colab", "exec", "-s", session, "--timeout", "1800", "-f", str(scripts_dir / "step3_autonomous.py")],
                check=False,
            )
            if getattr(step3_proc, "returncode", 0) != 0:
                run_error = f"step3_autonomous exited with {step3_proc.returncode}"
            if mode == "sync_train_gguf" and run_error is None:
                step4_proc = run_cmd(["colab", "exec", "-s", session, "--timeout", "1800", "-f", str(scripts_dir / "step4_export_gguf.py")], check=False)
                if getattr(step4_proc, "returncode", 0) != 0:
                    run_error = f"step4_export_gguf exited with {step4_proc.returncode}"

        if mode in ("export_gguf", "sync_train_gguf"):
            gguf_tar = run_dir / "run_gguf_artifacts.tar.gz"
            dl_gguf = run_cmd(["colab", "download", "-s", session, "/content/run_gguf_artifacts.tar.gz", str(gguf_tar)], check=False)
            if getattr(dl_gguf, "returncode", 1) == 0 and gguf_tar.is_file():
                gguf_result = _extract_gguf_artifacts(gguf_tar, gguf_dir)
            elif run_error is None:
                run_error = "missing_gguf_artifacts_from_colab"

        if mode != "export_gguf":
            tar_local = run_dir / "run_diagnostics.tar.gz"
            dl = run_cmd(["colab", "download", "-s", session, "/content/run_diagnostics.tar.gz", str(tar_local)], check=False)
            if getattr(dl, "returncode", 1) == 0 and tar_local.is_file():
                diag_dir.mkdir(parents=True, exist_ok=True)
                with tarfile.open(tar_local, "r:gz") as tf:
                    tf.extractall(diag_dir, filter="data")
                curated_dst = workspace_dir / "curated_data"
                for ep_name in ("verified_selfchange_episodes.jsonl", "verified_selfchange_dpo.jsonl"):
                    ep_src = diag_dir / "curated_data" / ep_name
                    if not ep_src.is_file():
                        continue
                    ep_dst = curated_dst / ep_name
                    src_lines = len(ep_src.read_text(encoding="utf-8").splitlines())
                    dst_lines = len(ep_dst.read_text(encoding="utf-8").splitlines()) if ep_dst.is_file() else 0
                    if src_lines > dst_lines:
                        curated_dst.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(ep_src, ep_dst)
                persistence = persist_improved_weights_from_colab(
                    session=session,
                    diagnostics_dir=diag_dir,
                    promoted_dir=promoted_dir,
                    run_id=run_id,
                    always_persist=always_persist_weights,
                    upload_drive=upload_drive,
                    run_cmd=run_cmd,
                )
    except Exception as exc:
        run_error = repr(exc)
        raise
    finally:
        if not keep_vm:
            run_cmd(["colab", "stop", "-s", session], check=False)

    out_summary = {
        "run_id": run_id,
        "session": session,
        "gpu": gpu,
        "mode": mode,
        "resume_seed": seed_result,
        "overlay_files": overlay_files,
        "bootstrap_skipped": skip_bootstrap,
        "diagnostics_dir": str(diag_dir),
        "persistence": persistence,
        "error": run_error,
    }
    if gguf_result:
        out_summary["gguf"] = gguf_result
    return out_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end Colab CLI runner with auto-resume, minimal-CPU search, and GGUF export")
    parser.add_argument("--session", type=str, default="minimind-selfchange")
    parser.add_argument("--gpu", type=str, default="T4")
    parser.add_argument("--branch", type=str, default="mychanges")
    parser.add_argument("--repo-url", type=str, default="https://github.com/ed2k/minimind.git")
    parser.add_argument("--workspace-dir", type=Path, default=Path("selfchange/workspace"))
    parser.add_argument("--mode", choices=["autonomous", "search_minimal_cpu", "export_gguf", "sync_train_gguf"], default="autonomous")
    parser.add_argument("--search-rungs", type=str, default=None)
    parser.add_argument("--search-max-rungs", type=int, default=4)
    parser.add_argument("--export-pth", type=str, default=None)
    parser.add_argument("--gguf-quants", type=str, default="Q8_0,Q4_K_M")
    parser.add_argument("--max-cycles", type=int, default=10)
    parser.add_argument("--autonomy-level", type=int, default=1)
    parser.add_argument("--total-traces", type=int, default=3000)
    parser.add_argument("--warmstart-stage", choices=["full_sft", "lora"], default="full_sft")
    parser.add_argument("--always-persist-weights", action="store_true")
    parser.add_argument("--upload-drive", action="store_true")
    parser.add_argument("--keep-vm", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-bootstrap-on-resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlay-local-code", action="store_true")
    args = parser.parse_args()

    out = run_colab_cli_pipeline(
        session=args.session,
        gpu=args.gpu,
        branch=args.branch,
        repo_url=args.repo_url,
        workspace_dir=args.workspace_dir,
        max_cycles=args.max_cycles,
        autonomy_level=args.autonomy_level,
        total_traces=args.total_traces,
        warmstart_stage=args.warmstart_stage,
        always_persist_weights=args.always_persist_weights,
        upload_drive=args.upload_drive,
        keep_vm=args.keep_vm,
        resume=args.resume,
        skip_bootstrap_on_resume=args.skip_bootstrap_on_resume,
        mode=args.mode,
        search_rungs=args.search_rungs,
        search_max_rungs=args.search_max_rungs,
        overlay_local_code=args.overlay_local_code,
        export_pth=args.export_pth,
        gguf_quants=args.gguf_quants,
    )
    print(json.dumps(out, indent=2))
    return 0 if out.get("error") is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
