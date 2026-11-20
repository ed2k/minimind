from __future__ import annotations

import argparse
import json
from pathlib import Path

from selfchange.bootstrap.generate_traces import generate_traces
from selfchange.kernel.budgets import get_profile
from selfchange.kernel.integrity import atomic_write_text
from selfchange.mutable.data_curator import mix_with_anchor
from selfchange.training.self_train_runner import run_training_stage, write_sft_jsonl


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def prepare_curriculum_datasets(
    work_dir: Path,
    anchor_jsonl: Path | None = None,
    total_traces: int = 500,
    max_seq_len: int = 768,
    anchor_ratio: float = 0.30,
    seed: int = 42,
) -> dict:
    """Generate progressive Rung 1 -> Rung 2 -> Rung 3 datasets blended with anchor SFT."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    anchors = _load_jsonl(anchor_jsonl) if anchor_jsonl is not None else []

    counts = {
        "1": max(20, int(total_traces * 0.25)),
        "2": max(20, int(total_traces * 0.35)),
        "3": max(20, total_traces - int(total_traces * 0.25) - int(total_traces * 0.35)),
    }
    outputs: dict[str, dict] = {}
    for rung_id, cnt in counts.items():
        raw_path = work_dir / f"traces_rung{rung_id}.jsonl"
        gen_stats = generate_traces(
            raw_path,
            count=cnt,
            seed=seed + int(rung_id) * 100,
            max_seq_len=max_seq_len,
            rung=rung_id,
            real_ast_subprocesses=4 if rung_id == "3" else 0,
        )
        op_rows = _load_jsonl(raw_path)
        mixed_rows = mix_with_anchor(op_rows, anchors, anchor_ratio=anchor_ratio)
        mixed_path = work_dir / f"curriculum_rung{rung_id}.jsonl"
        written = write_sft_jsonl(mixed_path, mixed_rows)
        outputs[f"rung{rung_id}"] = {
            "raw_path": str(raw_path),
            "mixed_path": str(mixed_path),
            "op_count": len(op_rows),
            "mixed_count": written,
            **gen_stats,
        }
    return outputs


def run_warmstart_curriculum(
    repo_root: Path,
    workspace_dir: Path,
    work_dir: Path,
    profile_name: str | None = None,
    anchor_jsonl: Path | None = None,
    total_traces: int = 500,
    stage: str = "full_sft",
    from_weight: str = "full_sft",
    dry_run: bool = False,
    seed: int = 42,
) -> dict:
    """Run progressive Rung 1 -> Rung 2 -> Rung 3 curriculum SFT."""
    profile = get_profile(profile_name)
    curriculum = prepare_curriculum_datasets(
        work_dir,
        anchor_jsonl=anchor_jsonl,
        total_traces=total_traces,
        max_seq_len=max(profile.max_seq_len, profile.max_total_len),
        seed=seed,
    )
    results: list[dict] = []
    current_from = from_weight
    for idx in (1, 2, 3):
        rung_key = f"rung{idx}"
        save_name = f"warmstart_{rung_key}"
        res = run_training_stage(
            stage=stage,
            repo_root=repo_root,
            workspace_dir=workspace_dir,
            data_path=Path(curriculum[rung_key]["mixed_path"]),
            profile=profile,
            training_cfg={"epochs": 1, "learning_rate": 2e-5 if stage == "full_sft" else 1e-4, "batch_size": profile.batch_size, "max_seq_len": profile.max_seq_len},
            from_weight=current_from,
            save_weight=save_name,
            dry_run=dry_run,
        )
        results.append(
            {
                "rung": idx,
                "ok": res.ok,
                "exit_code": res.exit_code,
                "output_weight_path": res.output_weight_path,
                "final_loss": res.final_loss,
                "log_tail": res.log_tail,
            }
        )
        if not res.ok:
            break
        current_from = save_name
    return {"curriculum": curriculum, "stages": results, "final_weight_name": current_from}


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1 progressive 3-rung curriculum warm-start")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--workspace-dir", type=Path, default=Path("selfchange/workspace"))
    parser.add_argument("--work-dir", type=Path, default=Path("selfchange/workspace/bootstrap_data"))
    parser.add_argument("--anchor-jsonl", type=Path, default=None)
    parser.add_argument("--total-traces", type=int, default=500)
    parser.add_argument("--stage", choices=["full_sft", "lora"], default="full_sft")
    parser.add_argument("--from-weight", type=str, default="full_sft")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = run_warmstart_curriculum(
        repo_root=args.repo_root,
        workspace_dir=args.workspace_dir,
        work_dir=args.work_dir,
        profile_name=args.profile,
        anchor_jsonl=args.anchor_jsonl,
        total_traces=args.total_traces,
        stage=args.stage,
        from_weight=args.from_weight,
        dry_run=args.dry_run,
    )
    print(json.dumps(out, indent=2))
    all_ok = bool(out.get("stages")) and all(s.get("ok") for s in out["stages"])
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
