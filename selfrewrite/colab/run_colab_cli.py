from __future__ import annotations

import argparse
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from selfrewrite.config import atomic_write_text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_remote_a100_scripts(
    repo_url: str = "https://github.com/ed2k/minimind.git",
    branch: str = "mychanges",
    model_id: str = "Qwen/Qwen3.5-27B",
    mode: str = "reproduce-and-harvest",
    rungs: str = "R1,R2,R3,R4",
    drive_dir: str = "/content/drive/MyDrive/minimind_selfrewrite",
    num_reproduce_improve: int = 8,
    num_harvest_improve: int = 8,
    num_harvest_repair: int = 12,
    programmatic_teacher: bool = False,
    fast_harvest_profile: bool = False,
) -> dict[str, str]:
    """Generate remote Colab A100 setup, reproduce/harvest/bracket, and detached runner scripts."""
    prog_flag = '        "--programmatic-teacher",\n' if programmatic_teacher else ""
    fast_flag = '        "--fast-harvest-profile",\n' if fast_harvest_profile else ""
    rungs_flag = f'        "--rungs",\n        {rungs!r},\n' if mode == "bracket" else ""

    step1_setup = f"""import os
import subprocess
import sys
from pathlib import Path

def run(cmd, cwd=None, check=True):
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, cwd=cwd)
    if check and p.returncode != 0:
        raise SystemExit(p.returncode)
    return p.returncode

def main():
    run(["nvidia-smi"], check=False)
    repo = Path("/content/minimind")
    if (repo / ".git").is_dir():
        run(["git", "fetch", "origin", {branch!r}], cwd=str(repo))
        run(["git", "checkout", {branch!r}], cwd=str(repo))
        run(["git", "pull", "--ff-only", "origin", {branch!r}], cwd=str(repo))
    else:
        run(["git", "clone", "--depth", "1", "-b", {branch!r}, {repo_url!r}, str(repo)], cwd="/content")
    run(
        [
            sys.executable, "-m", "pip", "-q", "install",
            "transformers>=5.0.0", "peft>=0.14.0", "datasets>=3.6.0", "huggingface_hub",
            "safetensors", "accelerate", "bitsandbytes", "sympy", "torchvision", "pillow",
        ],
        cwd=str(repo),
    )
    run(
        [sys.executable, "-m", "pip", "-q", "install", "flash-linear-attention", "causal-conv1d"],
        cwd=str(repo),
        check=False,
    )

if __name__ == "__main__":
    main()
"""

    step1c_test = """import subprocess
import sys
from pathlib import Path

def main():
    repo = Path("/content/minimind")
    proc = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "selfrewrite/tests", "-v"], cwd=str(repo))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

if __name__ == "__main__":
    main()
"""

    step2_run = f"""import os
import subprocess
import sys
import tarfile
from pathlib import Path

def main():
    repo = Path("/content/minimind")
    ws = repo / "selfrewrite" / "workspace"
    cmd = [
        sys.executable,
        "-m",
        "selfrewrite.colab.colab_launcher",
        "--workspace-dir",
        str(ws),
        "--mode",
        {mode!r},
        "--backend",
        "hf",
        "--model-id",
        {model_id!r},
        "--profile",
        "auto",
        "--drive-dir",
        {drive_dir!r},
        "--num-reproduce-improve",
        str({int(num_reproduce_improve)}),
        "--num-harvest-improve",
        str({int(num_harvest_improve)}),
        "--num-harvest-repair",
        str({int(num_harvest_repair)}),
{rungs_flag}{fast_flag}{prog_flag}    ]
    print("+", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(repo))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    bundle_tar = Path("/content/selfrewrite_artifacts.tar.gz")
    with tarfile.open(bundle_tar, "w:gz") as tf:
        for rel in ("distill_bundle", "report.md", "leaderboard.json", "downsize_leaderboard.json", "downsize_report.md"):
            p = ws / rel
            if p.exists():
                tf.add(p, arcname=rel)
    print("WROTE", bundle_tar, flush=True)

if __name__ == "__main__":
    main()
"""

    step2_detached_launch = """import json
import subprocess
import sys
from pathlib import Path

def main():
    repo = Path("/content/minimind")
    runner = Path("/content/run_detached_wrapper.py")
    wrapper_code = (
        "import json, subprocess, sys\\n"
        "from pathlib import Path\\n"
        "logf = open('/content/selfrewrite_detached.log', 'w')\\n"
        "p = subprocess.run([sys.executable, '/content/step2_reproduce_harvest.py'], cwd='/content/minimind', stdout=logf, stderr=subprocess.STDOUT)\\n"
        "logf.close()\\n"
        "Path('/content/selfrewrite_done.json').write_text(json.dumps({'returncode': p.returncode}))\\n"
    )
    runner.write_text(wrapper_code, encoding="utf-8")
    Path("/content/selfrewrite_done.json").unlink(missing_ok=True)
    proc = subprocess.Popen([sys.executable, "-u", str(runner)], cwd=str(repo), start_new_session=True)
    print(json.dumps({"launched_pid": proc.pid}), flush=True)

if __name__ == "__main__":
    main()
"""

    step2_detached_poll = """import json
from pathlib import Path

def main():
    done_file = Path("/content/selfrewrite_done.json")
    hb_file = Path("/content/minimind/selfrewrite/workspace/HEARTBEAT.json")
    log_file = Path("/content/selfrewrite_detached.log")
    hb = json.loads(hb_file.read_text(encoding="utf-8")) if hb_file.is_file() else {}
    tail = log_file.read_text(encoding="utf-8").splitlines()[-25:] if log_file.is_file() else []
    status = {"done": done_file.is_file(), "heartbeat": hb, "log_tail": tail}
    if done_file.is_file():
        status["result"] = json.loads(done_file.read_text(encoding="utf-8"))
        print("SELFREWRITE_DETACHED_STATUS:" + json.dumps(status), flush=True)
        if int(status["result"].get("returncode", 1)) != 0:
            raise SystemExit(int(status["result"].get("returncode", 1)))
        return
    print("SELFREWRITE_DETACHED_STATUS:" + json.dumps(status), flush=True)

if __name__ == "__main__":
    main()
"""

    return {
        "step1_setup.py": step1_setup,
        "step1c_test.py": step1c_test,
        "step2_reproduce_harvest.py": step2_run,
        "step2_detached_launch.py": step2_detached_launch,
        "step2_detached_poll.py": step2_detached_poll,
    }


def _pack_selfrewrite_overlay(repo_root: Path, tar_out: Path) -> Path:
    resolved_root = Path(repo_root).resolve()
    sr_dir = resolved_root / "selfrewrite"
    tar_out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_out, "w:gz") as tf:
        for item in sorted(sr_dir.rglob("*")):
            if "__pycache__" in item.parts or "workspace" in item.parts or item.suffix == ".pyc":
                continue
            tf.add(item, arcname=item.relative_to(resolved_root).as_posix(), recursive=False)
    return tar_out


def run_colab_cli_pipeline(
    session: str = "minimind-selfrewrite-a100",
    gpu: str = "A100",
    high_mem: bool = True,
    repo_url: str = "https://github.com/ed2k/minimind.git",
    branch: str = "mychanges",
    workspace_dir: Path = Path("selfrewrite/workspace"),
    model_id: str = "Qwen/Qwen3.5-27B",
    mode: str = "reproduce-and-harvest",
    rungs: str = "R1,R2,R3,R4",
    drive_dir: str = "/content/drive/MyDrive/minimind_selfrewrite",
    overlay_local_code: bool = True,
    repo_root: Path = Path("."),
    keep_vm: bool = False,
    dry_run: bool = False,
    fast_harvest_profile: bool = False,
    detached: bool = False,
    run_cmd: Callable = subprocess.run,
) -> dict[str, Any]:
    """Provision Colab A100 runtime via colab CLI, run reproduction & harvesting, and pull distill bundle."""
    ws = Path(workspace_dir).resolve()
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    run_dir = ws / "colab_runs" / run_id
    scripts_dir = run_dir / "scripts"
    scripts = build_remote_a100_scripts(
        repo_url=repo_url,
        branch=branch,
        model_id=model_id,
        mode=mode,
        drive_dir=drive_dir,
        rungs=rungs,
        fast_harvest_profile=fast_harvest_profile,
    )
    for name, content in scripts.items():
        atomic_write_text(scripts_dir / name, content)

    planned_cmds: list[list[str]] = []
    new_cmd = ["colab", "new", "-s", session, "--gpu", gpu]
    if high_mem:
        new_cmd.append("--high-mem")
    planned_cmds.append(new_cmd)
    planned_cmds.append(["colab", "drivemount", "-s", session])
    planned_cmds.append(["colab", "exec", "-s", session, "--timeout", "1800", "-f", str(scripts_dir / "step1_setup.py")])

    overlay_tar: Path | None = None
    if overlay_local_code:
        overlay_tar = run_dir / "selfrewrite_overlay.tar.gz"
        _pack_selfrewrite_overlay(Path(repo_root), overlay_tar)
        planned_cmds.append(["colab", "upload", "-s", session, str(overlay_tar), "/content/selfrewrite_overlay.tar.gz"])
        extract_script = run_dir / "scripts" / "step1b_extract_overlay.py"
        atomic_write_text(
            extract_script,
            "import tarfile; tarfile.open('/content/selfrewrite_overlay.tar.gz').extractall('/content/minimind')\n",
        )
        planned_cmds.append(["colab", "exec", "-s", session, "-f", str(extract_script)])

    planned_cmds.append(["colab", "exec", "-s", session, "--timeout", "600", "-f", str(scripts_dir / "step1c_test.py")])
    if detached:
        planned_cmds.append(["colab", "upload", "-s", session, str(scripts_dir / "step2_reproduce_harvest.py"), "/content/step2_reproduce_harvest.py"])
        planned_cmds.append(["colab", "exec", "-s", session, "--timeout", "60", "-f", str(scripts_dir / "step2_detached_launch.py")])
        planned_cmds.append(["colab", "exec", "-s", session, "--timeout", "60", "-f", str(scripts_dir / "step2_detached_poll.py")])
    else:
        planned_cmds.append(["colab", "exec", "-s", session, "--timeout", "14400", "-f", str(scripts_dir / "step2_reproduce_harvest.py")])
    planned_cmds.append(["colab", "download", "-s", session, "/content/selfrewrite_artifacts.tar.gz", str(run_dir / "selfrewrite_artifacts.tar.gz")])
    if not keep_vm:
        planned_cmds.append(["colab", "stop", "-s", session])

    if dry_run:
        return {
            "dry_run": True,
            "detached": bool(detached),
            "fast_harvest_profile": bool(fast_harvest_profile),
            "run_dir": str(run_dir),
            "planned_commands": planned_cmds,
            "scripts": sorted(scripts.keys()),
        }

    import time as _time

    poll_cmd = ["colab", "exec", "-s", session, "--timeout", "60", "-f", str(scripts_dir / "step2_detached_poll.py")]
    download_cmd = ["colab", "download", "-s", session, "/content/selfrewrite_artifacts.tar.gz", str(run_dir / "selfrewrite_artifacts.tar.gz")]
    try:
        if detached:
            setup_cmds = [c for c in planned_cmds if c != poll_cmd and c != download_cmd and (not c or c[:2] != ["colab", "stop"])]
            for cmd in setup_cmds:
                run_cmd(cmd, check=True)
            while True:
                p_res = run_cmd(poll_cmd, capture_output=True, text=True, check=True)
                out_s = (getattr(p_res, "stdout", "") or "") + (getattr(p_res, "stderr", "") or "")
                done_flag = False
                for line in out_s.splitlines():
                    if line.startswith("SELFREWRITE_DETACHED_STATUS:"):
                        payload = json.loads(line.split("SELFREWRITE_DETACHED_STATUS:", 1)[1])
                        done_flag = bool(payload.get("done"))
                if done_flag:
                    break
                _time.sleep(30)
            run_cmd(download_cmd, check=True)
        else:
            for cmd in planned_cmds[:-1] if not keep_vm else planned_cmds:
                run_cmd(cmd, check=True)
    finally:
        if not keep_vm:
            run_cmd(["colab", "stop", "-s", session], check=False)

    art_tar = run_dir / "selfrewrite_artifacts.tar.gz"
    if art_tar.is_file():
        with tarfile.open(art_tar, "r:gz") as tf:
            tf.extractall(ws)
    return {
        "dry_run": False,
        "run_dir": str(run_dir),
        "artifacts_tar": str(art_tar),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Headless Colab A100 driver for selfrewrite")
    parser.add_argument("--session", default="minimind-selfrewrite-a100")
    parser.add_argument("--gpu", default="A100")
    parser.add_argument("--no-high-mem", action="store_true")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-27B")
    parser.add_argument("--mode", default="reproduce-and-harvest")
    parser.add_argument("--rungs", default="R1,R2,R3,R4")
    parser.add_argument("--workspace-dir", default="selfrewrite/workspace")
    parser.add_argument("--drive-dir", default="/content/drive/MyDrive/minimind_selfrewrite")
    parser.add_argument("--fast-harvest-profile", action="store_true")
    parser.add_argument("--detached", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-vm", action="store_true")
    args = parser.parse_args(argv)

    res = run_colab_cli_pipeline(
        session=args.session,
        gpu=args.gpu,
        high_mem=not args.no_high_mem,
        model_id=args.model_id,
        mode=args.mode,
        rungs=args.rungs,
        workspace_dir=Path(args.workspace_dir),
        drive_dir=args.drive_dir,
        fast_harvest_profile=bool(args.fast_harvest_profile),
        detached=bool(args.detached),
        dry_run=args.dry_run,
        keep_vm=args.keep_vm,
    )
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
