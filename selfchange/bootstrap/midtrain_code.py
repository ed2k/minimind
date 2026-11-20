from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from selfchange.kernel.budgets import get_profile
from selfchange.kernel.integrity import atomic_write_text
from selfchange.training.self_train_runner import run_training_stage

_SYNTHETIC_TRACEBACKS = [
    'Traceback (most recent call last):\n  File "selfchange/mutable/recovery_policy.py", line 14, in plan_recovery_action\nRuntimeError: CUDA out of memory. Tried to allocate 512.00 MiB',
    'Epoch:[1/2](40/200), loss: nan, grad_norm: inf, lr: 0.00020000',
    'Traceback (most recent call last):\n  File "selfchange/mutable/oom_helper.py", line 1\n    def compute_step(x, div=2)\nSyntaxError: expected \':\'',
    'EOFError: Ran out of input while loading checkpoints/full_sft_768_resume.pth',
]

_TOOL_JSON_SNIPPETS = [
    '<tool_call>\n{"name": "read_log_tail", "arguments": {"log_name": "train.log", "lines": 30}}\n</tool_call>',
    '<tool_call>\n{"name": "patch_config", "arguments": {"updates": {"training": {"batch_size": 1, "accumulation_steps": 2}}}}\n</tool_call>',
    '<tool_call>\n{"name": "replace_line_range", "arguments": {"path": "oom_helper.py", "start_line": 2, "end_line": 2, "lines": ["    return max(1, x // div)"]}}\n</tool_call>',
]


def build_midtrain_corpus(
    repo_root: Path,
    output_jsonl: Path,
    anchor_pretrain_jsonl: Path | None = None,
    max_samples: int = 4000,
    seed: int = 42,
) -> dict:
    """Build 50/50 Python/traceback/JSON and general pretrain corpus in PretrainDataset format."""
    repo_root = Path(repo_root).resolve()
    rng = random.Random(seed)
    code_samples: list[dict] = []

    for sub in ("selfchange/mutable", "model", "trainer"):
        folder = repo_root / sub
        if not folder.is_dir():
            continue
        for py_file in sorted(folder.glob("*.py")):
            lines = py_file.read_text(encoding="utf-8").splitlines()
            for start in range(0, len(lines), 35):
                chunk = "\n".join(lines[start : start + 40]).strip()
                if len(chunk) >= 30:
                    code_samples.append({"text": f"# {py_file.name}\n{chunk}"})

    for tb in _SYNTHETIC_TRACEBACKS:
        code_samples.append({"text": tb})
    for tj in _TOOL_JSON_SNIPPETS:
        code_samples.append({"text": tj})

    rng.shuffle(code_samples)
    half_limit = max(1, max_samples // 2) if anchor_pretrain_jsonl and anchor_pretrain_jsonl.is_file() else max_samples
    code_selected = code_samples[:half_limit]

    anchor_selected: list[dict] = []
    if anchor_pretrain_jsonl is not None and anchor_pretrain_jsonl.is_file():
        for raw in anchor_pretrain_jsonl.read_text(encoding="utf-8").splitlines():
            if len(anchor_selected) >= len(code_selected):
                break
            if raw.strip():
                anchor_selected.append(json.loads(raw))

    combined = code_selected + anchor_selected
    rng.shuffle(combined)
    atomic_write_text(output_jsonl, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in combined))
    return {
        "output_path": str(output_jsonl),
        "code_samples": len(code_selected),
        "anchor_samples": len(anchor_selected),
        "total_samples": len(combined),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 0.5 Python/logs/JSON continued mid-training")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--workspace-dir", type=Path, default=Path("selfchange/workspace"))
    parser.add_argument("--output", type=Path, default=Path("selfchange/workspace/midtrain_corpus.jsonl"))
    parser.add_argument("--anchor-pretrain", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--from-weight", type=str, default="full_sft")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    stats = build_midtrain_corpus(
        args.repo_root,
        args.output,
        anchor_pretrain_jsonl=args.anchor_pretrain,
        max_samples=args.max_samples,
    )
    if args.run or args.dry_run:
        profile = get_profile(args.profile)
        train_res = run_training_stage(
            stage="pretrain",
            repo_root=args.repo_root,
            workspace_dir=args.workspace_dir,
            data_path=args.output,
            profile=profile,
            training_cfg={"learning_rate": 2e-5, "epochs": 1, "batch_size": profile.batch_size, "max_seq_len": profile.max_seq_len},
            from_weight=args.from_weight,
            save_weight="midtrain_code",
            dry_run=args.dry_run,
        )
        stats["training_ok"] = train_res.ok
        stats["output_weight_path"] = train_res.output_weight_path
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
