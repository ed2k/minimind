from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

from selfchange.kernel.integrity import atomic_write_text
from selfchange.training.self_train_runner import format_sft_record

_DIALOGUE_QA = [
    ("解释一下梯度累积的作用。", "梯度累积通过多次前向和反向传播累加梯度后再更新参数，可在显存受限时模拟更大的 batch_size。"),
    ("什么是学习率衰减？", "学习率衰减在训练后期逐步降低步长，有助于损失函数更稳定地收敛到较优局部区域。"),
    ("What does LoRA freeze during fine-tuning?", "LoRA freezes the original base model weights and trains only low-rank A and B adapter matrices."),
    ("Why use a git worktree for candidate edits?", "A git worktree isolates candidate edits in a separate checkout so failed experiments never corrupt the main branch."),
]

_TOOL_EXAMPLES = [
    (
        "计算 18 乘以 25 等于多少",
        [{"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}}],
        {"name": "calculate_math", "arguments": {"expression": "18 * 25"}},
        '{"result": "450"}',
        "18 乘以 25 的结果是 450。",
    ),
    (
        "Convert 10 km to miles",
        [{"type": "function", "function": {"name": "unit_converter", "description": "单位换算", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}}],
        {"name": "unit_converter", "arguments": {"value": 10, "from_unit": "km", "to_unit": "miles"}},
        '{"result": 6.2137}',
        "10 km equals approximately 6.2137 miles.",
    ),
]


def _make_anchor_record(idx: int, rng: random.Random) -> dict:
    if idx % 3 == 0:
        prompt, tools, call, obs, final_ans = _TOOL_EXAMPLES[idx % len(_TOOL_EXAMPLES)]
        return format_sft_record(
            {
                "conversations": [
                    {"role": "system", "content": "You are MiniMind.", "tools": json.dumps(tools, ensure_ascii=False)},
                    {"role": "user", "content": f"{prompt} (case {idx})"},
                    {"role": "assistant", "content": "", "tool_calls": json.dumps([call], ensure_ascii=False)},
                    {"role": "tool", "content": obs},
                    {"role": "assistant", "content": final_ans},
                ]
            }
        )
    q, a = _DIALOGUE_QA[(idx + rng.randint(0, 3)) % len(_DIALOGUE_QA)]
    return format_sft_record(
        {
            "conversations": [
                {"role": "user", "content": f"[{idx}] {q}"},
                {"role": "assistant", "content": a},
            ]
        }
    )


def build_synthetic_anchor_datasets(
    output_dir: Path,
    train_count: int = 200,
    heldout_count: int = 50,
    seed: int = 42,
) -> dict:
    """Create disjoint synthetic dialogue + tool-call anchor and held-out JSONL datasets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    train_path = output_dir / "anchor_sft.jsonl"
    heldout_path = output_dir / "heldout_eval.jsonl"

    train_lines = [json.dumps(_make_anchor_record(i, rng), ensure_ascii=False) + "\n" for i in range(train_count)]
    heldout_lines = [
        json.dumps(_make_anchor_record(train_count + 1000 + j, rng), ensure_ascii=False) + "\n"
        for j in range(heldout_count)
    ]

    atomic_write_text(train_path, "".join(train_lines))
    atomic_write_text(heldout_path, "".join(heldout_lines))
    return {
        "anchor_path": str(train_path),
        "heldout_path": str(heldout_path),
        "train_count": train_count,
        "heldout_count": heldout_count,
    }


def build_anchor_split(
    source_jsonl: Path,
    train_anchor_out: Path,
    heldout_out: Path,
    train_limit: int = 10000,
    heldout_limit: int = 500,
    seed: int = 42,
) -> dict:
    """Deterministically partition a real MiniMind JSONL dataset into disjoint anchor and held-out splits."""
    source_jsonl = Path(source_jsonl)
    raw_lines = [ln for ln in source_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    indices = list(range(len(raw_lines)))
    random.Random(seed).shuffle(indices)

    actual_heldout = min(heldout_limit, max(1, len(indices) // 10)) if indices else 0
    heldout_idx = indices[:actual_heldout]
    train_idx = indices[actual_heldout : actual_heldout + train_limit]

    atomic_write_text(heldout_out, "".join(raw_lines[i] + "\n" for i in heldout_idx))
    atomic_write_text(train_anchor_out, "".join(raw_lines[i] + "\n" for i in train_idx))
    return {
        "source": str(source_jsonl),
        "total_lines": len(raw_lines),
        "heldout_count": len(heldout_idx),
        "train_anchor_count": len(train_idx),
        "heldout_path": str(heldout_out),
        "anchor_path": str(train_anchor_out),
    }


def setup_micro_model(
    output_dir: Path,
    hidden_size: int = 128,
    num_hidden_layers: int = 4,
    seed: int = 42,
) -> dict:
    """Initialize a 2M micro-model (or dry-run metadata stub if torch is absent) plus anchor splits."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = build_synthetic_anchor_datasets(output_dir / "data", seed=seed)

    torch_available = False
    weight_file = output_dir / f"micro_base_{hidden_size}.pth"
    param_count = 0

    try:
        import torch  # type: ignore[import-not-found]
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

        torch.manual_seed(seed)
        cfg = MiniMindConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, use_moe=False)
        model = MiniMindForCausalLM(cfg)
        param_count = sum(p.numel() for p in model.parameters())
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, weight_file)
        torch_available = True
    except ImportError:
        atomic_write_text(
            output_dir / f"micro_base_{hidden_size}.meta.json",
            json.dumps({"stub": True, "hidden_size": hidden_size, "num_hidden_layers": num_hidden_layers, "seed": seed}, indent=2) + "\n",
        )

    meta = {
        "mode": "micro",
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "use_moe": False,
        "torch_available": torch_available,
        "param_count": param_count,
        "weight_path": str(weight_file) if torch_available else None,
        **splits,
    }
    atomic_write_text(output_dir / "model_meta.json", json.dumps(meta, indent=2) + "\n")
    return meta


def download_minimind3(
    output_dir: Path,
    source: str = "huggingface",
    model_id: str = "jingyaogong/minimind-3",
    convert_to_pth: bool = True,
    repo_root: Path | None = None,
) -> dict:
    """Download pretrained MiniMind-3 weights for Colab and optionally export native PyTorch .pth."""
    output_dir = Path(output_dir)
    hf_dir = output_dir / "hf_minimind3"
    hf_dir.mkdir(parents=True, exist_ok=True)

    if source == "huggingface":
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

        snapshot_download(repo_id=model_id, local_dir=str(hf_dir))
    elif source == "modelscope":
        from modelscope import snapshot_download  # type: ignore[import-not-found]

        snapshot_download(model_id=model_id, local_dir=str(hf_dir))
    else:
        raise ValueError(f"Unsupported download source {source!r}")

    pth_path: str | None = None
    out_full_sft_path: str | None = None
    if convert_to_pth:
        import torch  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM  # type: ignore[import-not-found]

        model = AutoModelForCausalLM.from_pretrained(str(hf_dir), trust_remote_code=True)
        out_pth = output_dir / "minimind3_768.pth"
        torch.save({k: v.cpu().half() for k, v in model.state_dict().items()}, out_pth)
        pth_path = str(out_pth)

        if repo_root is not None:
            out_dir = Path(repo_root).resolve() / "out"
            out_dir.mkdir(parents=True, exist_ok=True)
            staged = out_dir / "full_sft_768.pth"
            tmp = staged.with_name(f".{staged.name}.{os.getpid()}.tmp")
            try:
                shutil.copy2(out_pth, tmp)
                os.replace(tmp, staged)
            finally:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
            out_full_sft_path = str(staged)

    meta = {
        "mode": "colab",
        "source": source,
        "model_id": model_id,
        "hf_dir": str(hf_dir),
        "weight_path": pth_path,
        "staged_out_path": out_full_sft_path,
    }
    atomic_write_text(output_dir / "model_meta.json", json.dumps(meta, indent=2) + "\n")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Setup base model and anchor datasets for selfchange")
    parser.add_argument("--mode", choices=["micro", "colab"], default="micro")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("selfchange/workspace/models/base"))
    parser.add_argument("--source", choices=["huggingface", "modelscope"], default="huggingface")
    parser.add_argument("--model-id", type=str, default="jingyaogong/minimind-3")
    parser.add_argument("--source-sft", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "micro":
        res = setup_micro_model(args.output_dir, seed=args.seed)
    else:
        res = download_minimind3(
            args.output_dir,
            source=args.source,
            model_id=args.model_id,
            repo_root=args.repo_root,
        )
        data_dir = args.output_dir / "data"
        if args.source_sft is not None and args.source_sft.is_file():
            split_res = build_anchor_split(
                args.source_sft,
                data_dir / "anchor_sft.jsonl",
                data_dir / "heldout_eval.jsonl",
                seed=args.seed,
            )
            res.update(split_res)
        else:
            res.update(build_synthetic_anchor_datasets(data_dir, train_count=400, heldout_count=60, seed=args.seed))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
