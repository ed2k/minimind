from __future__ import annotations

import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path
from typing import Any

from selfrewrite.cli import main as cli_main
from selfrewrite.distill.export_minimind_data import adapt_text_to_minimind, export_qwen27b_dataset
from selfrewrite.harvest.audit import audit_distill_bundle
from selfrewrite.tasks.corruption import split_train_eval_layers
from selfrewrite.tasks.datasets import load_jsonl_problems

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "selfrewrite" / "data" / "qwen35_27b_v1"
WORKSPACE_DIR = REPO_ROOT / "selfrewrite" / "workspace"
_SFT_MSG_KEYS = {"role", "content", "reasoning_content", "tools", "tool_calls"}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _simulate_minimind_chat_template(messages: list[dict[str, Any]], open_thinking: bool | None = None, add_generation_prompt: bool = False) -> str:
    """Mirror `model/tokenizer_config.json` chat_template semantics for offline verification."""
    parts: list[str] = []
    if messages and messages[0].get("role") == "system":
        parts.append(f"<|im_start|>system\n{messages[0]['content']}<|im_end|>\n")
    for idx, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content") if isinstance(msg.get("content"), str) else ""
        if role == "user" or (role == "system" and idx != 0):
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        elif role == "assistant":
            reasoning = msg.get("reasoning_content")
            if isinstance(reasoning, str):
                rc = reasoning
                body = content
            elif "</think>" in content:
                rc = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                body = content.split("</think>")[-1].lstrip("\n")
            else:
                rc = ""
                body = content
            parts.append(f"<|im_start|>assistant\n<think>\n{rc.strip(chr(10))}\n</think>\n\n{body.lstrip(chr(10))}<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
        parts.append("<think>\n" if open_thinking is True else "<think>\n\n</think>\n\n")
    return "".join(parts)


class TestExportMiniMindData(unittest.TestCase):
    def test_tracked_qwen27b_dataset_and_minimind_schemas(self) -> None:
        canonical = DATASET_DIR / "canonical_27b"
        splits = DATASET_DIR / "splits"
        mm_dir = DATASET_DIR / "minimind_ready"

        self.assertTrue((canonical / "sft_gold_trajectories.jsonl").is_file())
        self.assertTrue((canonical / "sft_turn_primitives.jsonl").is_file())
        self.assertTrue((canonical / "dpo_contrastive_pairs.jsonl").is_file())
        self.assertTrue((canonical / "rlvr_episodes.jsonl").is_file())
        self.assertTrue((DATASET_DIR / "README.md").is_file())
        self.assertTrue((REPO_ROOT / "selfrewrite" / "MINIMIND_SELFREWRITE.md").is_file())

        # 1. Anti-leakage audit must pass with zero violations directly from tracked splits
        eval_texts = [
            item.question
            for fn in ("heldout_eval.jsonl", "reserved_eval.jsonl", "practice_eval.jsonl")
            for item in load_jsonl_problems(splits / fn)
        ]
        _, eval_layers = split_train_eval_layers(64)
        report = audit_distill_bundle(
            bundle_dir=canonical,
            forbidden_eval_texts=eval_texts,
            forbidden_eval_layers=eval_layers,
        )
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["num_violations"], 0)

        # 2. Zero hardcoded Colab '/content/' paths anywhere in tracked dataset dir
        for fpath in DATASET_DIR.rglob("*"):
            if fpath.is_file():
                txt = fpath.read_text(encoding="utf-8", errors="ignore")
                self.assertNotIn("/content/", txt, f"Found '/content/' path leak in {fpath}")

        # 3. Verify SFTDataset schema alignment and absence of duplicate <think> tags
        raw_sft = _load_jsonl(mm_dir / "sft_27b_steps_raw_arch.jsonl")
        adapted_sft = _load_jsonl(mm_dir / "sft_minimind_adapted.jsonl")
        self.assertEqual(len(raw_sft), 15)
        self.assertEqual(len(adapted_sft), 31)

        for idx, row in enumerate(adapted_sft):
            self.assertEqual(set(row.keys()), {"conversations"}, f"Extra keys break SFTDataset Features at row {idx}")
            convs = row["conversations"]
            self.assertGreaterEqual(len(convs), 2)
            for msg in convs:
                self.assertEqual(set(msg.keys()), _SFT_MSG_KEYS)
                self.assertIsInstance(msg["role"], str)
                self.assertIsInstance(msg["content"], str)
                self.assertNotIn("<|im_end|>", msg["content"])
                if msg["reasoning_content"] is not None:
                    self.assertNotIn("<think>", msg["reasoning_content"])
                    self.assertNotIn("</think>", msg["reasoning_content"])
            rendered = _simulate_minimind_chat_template(convs)
            self.assertNotIn("<think>\n<think>", rendered)
            self.assertIn("<|im_start|>assistant\n<think>", rendered)
            # All adapted primitives must fit <= 2048 tokens (~5200 chars), and full trajectories <= 4096 tokens (~8000 chars)
            total_chars = sum(len(m["content"]) + len(m["reasoning_content"] or "") for m in convs)
            first_prompt_chars = sum(len(m["content"]) + len(m["reasoning_content"] or "") for m in convs[:2])
            self.assertLessEqual(first_prompt_chars, 4000, f"Initial prompt too long at row {idx}: {first_prompt_chars}")
            if idx < 13:
                self.assertLessEqual(total_chars, 5200, f"Adapted primitive {idx} too long: {total_chars} chars")
            else:
                self.assertLessEqual(total_chars, 8000, f"Adapted trajectory {idx} too long: {total_chars} chars")

        # 4. Verify DPODataset schema & prompt budget
        dpo_rows = _load_jsonl(mm_dir / "dpo_minimind_pairs.jsonl")
        self.assertEqual(len(dpo_rows), 12)
        for idx, row in enumerate(dpo_rows):
            self.assertEqual(set(row.keys()), {"chosen", "rejected"})
            for branch in ("chosen", "rejected"):
                msgs = row[branch]
                self.assertGreaterEqual(len(msgs), 3)
                for msg in msgs:
                    self.assertEqual(set(msg.keys()), {"role", "content"})
                    self.assertNotIn("<|im_end|>", msg["content"])
                rendered = _simulate_minimind_chat_template(msgs)
                self.assertNotIn("<think>\n<think>", rendered)
            prompt_chars = sum(len(m["content"]) for m in row["chosen"][:-1])
            self.assertLess(prompt_chars, 4000, f"DPO row {idx} prompt too long ({prompt_chars} chars) for max_length=2048..4096")

        # 5. Verify RLAIFDataset schema (`conversations[:-1]` must end with a user turn)
        rlaif_rows = _load_jsonl(mm_dir / "rlaif_minimind_prompts.jsonl")
        self.assertEqual(len(rlaif_rows), 15)
        for row in rlaif_rows:
            self.assertEqual(set(row.keys()), {"conversations"})
            convs = row["conversations"]
            self.assertEqual(convs[-1]["role"], "assistant")
            self.assertEqual(convs[-2]["role"], "user")
            prompt = _simulate_minimind_chat_template(convs[:-1], open_thinking=True, add_generation_prompt=True)
            self.assertTrue(prompt.endswith("<|im_start|>assistant\n<think>\n"))

        # 6. Verify architecture tensor remapping works cleanly
        sample = (
            "Top matrix: model.language_model.layers.2.mlp.down_proj.weight and "
            "model.language_model.layers.62.linear_attn.out_proj.weight"
        )
        remapped = adapt_text_to_minimind(sample)
        self.assertIn("model.layers.1.mlp.down_proj.weight", remapped)
        self.assertIn("model.layers.6.self_attn.o_proj.weight", remapped)
        self.assertIsNone(re.search(r"model\.layers\.([89]|\d{2,})\.", remapped))

        # 7. Live dataset instantiation check when optional torch/datasets/transformers are present
        if all(importlib.util.find_spec(m) is not None for m in ("torch", "datasets", "transformers")):
            from transformers import AutoTokenizer
            from dataset.lm_dataset import DPODataset, RLAIFDataset, SFTDataset

            tok = AutoTokenizer.from_pretrained(str(REPO_ROOT / "model"))
            sft_ds = SFTDataset(str(mm_dir / "sft_minimind_adapted.jsonl"), tok, max_length=4096)
            self.assertEqual(len(sft_ds), 31)
            for i in range(len(sft_ds)):
                _, labels = sft_ds[i]
                self.assertGreater(int((labels != -100).sum().item()), 0)

            dpo_ds = DPODataset(str(mm_dir / "dpo_minimind_pairs.jsonl"), tok, max_length=2048)
            self.assertEqual(len(dpo_ds), 12)
            for i in range(len(dpo_ds)):
                item = dpo_ds[i]
                self.assertGreater(int(item["mask_chosen"].sum().item()), 0)
                self.assertGreater(int(item["mask_rejected"].sum().item()), 0)

            rlaif_ds = RLAIFDataset(str(mm_dir / "rlaif_minimind_prompts.jsonl"), tok, max_length=2048)
            self.assertEqual(len(rlaif_ds), 15)

    def test_export_reproducible_if_workspace_present(self) -> None:
        if not (WORKSPACE_DIR / "distill_bundle" / "v1" / "manifest.json").is_file():
            self.skipTest("Local workspace bundle not present")
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "export"
            res = export_qwen27b_dataset(workspace_dir=WORKSPACE_DIR, output_dir=out_dir, repo_root=REPO_ROOT)
            self.assertTrue(res["audit"]["passed"])
            self.assertEqual(cli_main(["audit-bundle", "--bundle-dir", str(out_dir / "canonical_27b"), "--splits-dir", str(out_dir / "splits"), "--num-layers", "64"]), 0)


if __name__ == "__main__":
    unittest.main()
