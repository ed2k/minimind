# Qwen/Qwen3.5-27B Self-Rewrite Dataset (`qwen35_27b_v1`)

Verified mechanistic weight self-rewriting steps extracted from live Colab L4 runs on `Qwen/Qwen3.5-27B`,
packaged with leak-audited splits and `dataset/lm_dataset.py`-compatible MiniMind training exports.

## Directory Layout

- `canonical_27b/` — Verified 27B bundle (`sft_gold_trajectories.jsonl`, `sft_turn_primitives.jsonl`, `dpo_contrastive_pairs.jsonl`, normalized `rlvr_episodes.jsonl`, `manifest.json`, `quality_audit.json`, plus `stage1_27b_leaderboard.json` and `stage2_downsize_leaderboard.json`).
- `splits/` — Disjoint problem splits (`distill_train.jsonl`, `practice_eval.jsonl`, `reserved_eval.jsonl`, `heldout_eval.jsonl`, `transfer_eval.jsonl`).
- `work_snapshots/` — Lightweight verified `census_summary.json` and `notes.md` state for RLVR episodes.
- `minimind_ready/` — Exact schemas consumed by `dataset/lm_dataset.py` (`SFTDataset`, `DPODataset`, `RLAIFDataset`) and MiniMind's Jinja chat template (`model/tokenizer_config.json`).

## Dataset Counts & Integrity

- Canonical 27B counts: `{"sft_gold_trajectories": 2, "sft_turn_primitives": 13, "dpo_contrastive_pairs": 12, "rlvr_episodes": 2}`
- Audit status: `passed=True`, `violations=0` (checked against `48` eval problems and `len=8` held-out layers)
- MiniMind-ready counts: `{"sft_27b_steps_raw_arch": 15, "sft_minimind_adapted": 31, "dpo_minimind_pairs": 12, "rlaif_minimind_prompts": 15}`

## Canonical SHA-256 Checksums

- `dpo_contrastive_pairs.jsonl`: `4195a291126401471ea0e9bf50fad8ec8f606b704fe416df41b4eb92b763746d`
- `rlvr_episodes.jsonl`: `89c7b1962d5632fd5f93e711baf7d7c6927f2e8533cf4f1a58a31847b8030c83`
- `sft_gold_trajectories.jsonl`: `333f3560b0ed2781407bc20e32d027954e3306758d938e1251c998f7bd259ab5`
- `sft_turn_primitives.jsonl`: `af3b951ae27f01aa3adef81305c26429967ae666a87d1fe3cc194bd9cedc55c7`

## Re-Verify Anti-Leakage Audit & Regenerate

```bash
python3 -m selfrewrite.cli audit-bundle \
  --bundle-dir selfrewrite/data/qwen35_27b_v1/canonical_27b \
  --splits-dir selfrewrite/data/qwen35_27b_v1/splits \
  --num-layers 64
```
