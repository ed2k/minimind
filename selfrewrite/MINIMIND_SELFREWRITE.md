# Weight Self-Rewriting in `minimind`: Achievements, Extracted 27B Data, and Training Blueprint

This document summarizes what has been built and empirically measured in `selfrewrite/`, documents the verified `Qwen/Qwen3.5-27B` self-rewriting training traces extracted into [`selfrewrite/data/qwen35_27b_v1/`](data/qwen35_27b_v1/README.md), and lays out the exact architectural, data, tokenizer, and multi-stage training requirements needed to teach `MiniMindForCausalLM` genuine weight self-rewriting capability.

---

## 1. Summary of Achievements

### 1.1 Gradient-Free Self-Rewriting Framework (`selfrewrite/`)
We built a standalone, leak-audited gradient-free self-rewriting runtime inside `selfrewrite/` (leaving `selfchange/` untouched):
- **Algorithms 1–4 Orchestrator & Protocol 2** ([`session.py`](session.py), [`protocol.py`](protocol.py)): Multi-turn closed-loop self-diagnosis (`census.jsonl`), forward activation reads (`read`), multi-candidate side-by-side generation (`edits`), context-window compaction, and a final out-of-sample `Framework Reserved-Practice Check` forcing an explicit `SUBMIT` / `WITHDRAW` commitment before held-out evaluation.
- **Sealed CPU Python Sandbox & Spectral Helpers** ([`sandbox.py`](sandbox.py), [`helpers.py`](helpers.py)): Persistent isolated worker inside `WORK/` with `torch.no_grad()`, AST safety checks, bounded runtime (`600s`) and stdout (`10,000` chars), canonical float32 tensor inspection (`load_original`, `save_edit`, `save_named_edit`), and mechanistic linear-algebra primitives (`outlier_ratios`, `clip_outliers_median`, `mean_diff_direction`, `remove_direction`, `layer_output_matrices`).
- **Live Weight-Overlay Backends** ([`hf_gpu_backend.py`](backends/hf_gpu_backend.py), [`minimind_backend.py`](backends/minimind_backend.py)): Zero-copy `HybridWeightOverlay` supporting both quantized `nf4_bf16_hybrid` weights (via LoRA / dequantize-patch-requantize for HuggingFace models) and native PyTorch `MiniMindForCausalLM` weights (`R4m`), with forbidden-tensor guards blocking edits to embeddings, final norms, `lm_head`, non-finite values, or global scalar rescaling (`edit_store.py`).
- **Paired Statistics & Anti-Leakage Auditing** ([`stats.py`](stats.py), [`build_bundle.py`](harvest/build_bundle.py), [`audit.py`](harvest/audit.py)): Exact McNemar sign test, Monte Carlo paired permutation test, `6`-step emergence ladder (`score_emergence_ladder`), and hard leakage auditing ensuring training bundles have zero overlap with `practice_eval`, `reserved_eval`, `heldout_eval`, or held-out corruption layers.

### 1.2 Live Colab L4 Empirical Results (`Qwen/Qwen3.5-27B`, Stage 1)
All 8 Stage 1 sessions completed cleanly (`stage1_rc = 0`) on a `24 GB` NVIDIA L4 GPU (`nf4_bf16_hybrid` profile), producing verified trajectories and negative contrast pairs recorded in [`stage1_27b_leaderboard.json`](data/qwen35_27b_v1/canonical_27b/stage1_27b_leaderboard.json):

| Session | Task | Editor Mode | Census Acc | Decision | Ladder | Key Observation |
|---|---|---|---:|---|---:|---|
| `reproduce_improve_s00` | `improve` | Autonomous 27B | `1.000` | `WITHDRAW` | `2/6` | 5 code runs, `0` code errors; withdrew cleanly; held-out `delta_hat = 0.0` (`m=16`). |
| `reproduce_repair_eval` | `repair` | Autonomous 27B (layer `30`, `sigma=200`, `max_abs=10.875`) | `0.000` | `WITHDRAW` | `1/6` | Damage crushed census to `0.0` (`cut=0.125`); autonomous 27B had `2` code errors and did not isolate layer `30`. |
| `harvest_improve_00` | `improve` | Autonomous 27B | `1.000` | `WITHDRAW` | `2/6` | Verbose census prints; no paired edit test; harvested as DPO rejected baseline. |
| `harvest_improve_01` | `improve` | Mechanistic teacher on live 27B | `1.000` | `WITHDRAW` | **`5/6`** | Read all `64` hidden layers (`read_1.npz`), selected `TARGET_LAYER 62` (`linear_attn.out_proj` + `mlp.down_proj`), saved `dir_a` (`alpha=0.25`) & `dir_b` (`alpha=0.35`), ran paired pass (`original: 1.0`, `dir_a: 0.9`, `dir_b: 0.9`), and withdrew via `drop_edit()`. |
| `harvest_repair_00` | `repair` | Autonomous 27B (layer `1`, `sigma=200`) | `0.625` | `WITHDRAW` | `1/6` | Missed true `layer 1` outlier (`9.5` max abs), had `1` code error; harvested as DPO rejected baseline. |
| `harvest_repair_01` | `repair` | Mechanistic teacher on live damaged 27B (layer `2`, `sigma=150`) | `1.000` | `WITHDRAW` | **`5/6`** | Scanned `128` output matrices via `outlier_ratios`, pinpointed `model.language_model.layers.2.mlp.down_proj.weight` (`ratio_max_to_median=1019.6793`, `max_abs=7.375`, `median_abs=0.00723267`), verified `64`-layer activations, tested `clip_15x` (`0.8`) vs `clip_25x` (`1.0`) vs `original` (`1.0`), calibrated withdrawal (`DROPPED_NO_GAIN`). |
| `harvest_fail_random_noise_4b` | `improve` | `MockFailureEditor` (random noise) | `1.000` | `WITHDRAW` | `3/6` | Random Gaussian edit collapsed paired accuracy `1.0 -> 0.0` (`delta_hat=-1.0`). |
| `harvest_fail_bad_submit_9b` | `improve` | `MockFailureEditor` (reckless `SUBMIT`) | `1.000` | `SUBMIT` | `4/6` | Added random noise to `layers.5`, ignored `delta_hat=-1.0` & reserved `delta_hat=-0.875`, recklessly submitted. |

### 1.3 Live Colab L4 Downsizing Bracket (`R3: 2B` & `R2: 4B`, Stage 2)
We distilled the audited `distill_bundle/v1` (`23` SFT rows / `12` DPO pairs / `2` RLVR episodes) into `R3` (`Qwen/Qwen3.5-2B`) and `R2` (`Qwen/Qwen3.5-4B`) in native `bfloat16` on L4 (`stage2_rc = 0`, peak `13.702 GiB` VRAM via `use_reentrant=False` gradient checkpointing and sequential single-graph DPO/RLVR backward). Results in [`stage2_downsize_leaderboard.json`](data/qwen35_27b_v1/canonical_27b/stage2_downsize_leaderboard.json):

| Rung | Model | Base Unmod Acc | Base Ladder | SFT Loss (`46` steps) | DPO Loss (`12` steps) | RLVR Reward (`2` steps, `G=4`) | Student Unedited Acc | Base Forgetting | Frozen Ladder | Promoted |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `R3` | `Qwen/Qwen3.5-2B` | `0.5625` | `0/6` | `0.266676` | `0.413365` | `0.350` | `0.0000` | `0.5625` | `0/6` | NO |
| `R2` | `Qwen/Qwen3.5-4B` | `0.8750` | `2/6` | `0.563796` | `0.436182` | `0.350` | `0.0000` | `0.8750` | `0/6` | NO |

**Key empirical findings from Stage 2 failure analysis:**
1. **Catastrophic base forgetting from tiny unanchored SFT**: Training rank-`32` (`lora_alpha=64`) LoRA across attention and MLP projections for `46` gradient steps at `sft_lr=1e-4` over only `23` curriculum rows (`13` multi-KB protocol primitives + `6` trajectory oversamples + only `4` short math anchors) shifted the chat/EOS formatting distribution enough to erase unmodified math accuracy (`0.5625 -> 0.0` on `2B`, `0.8750 -> 0.0` on `4B`).
2. **All-`WITHDRAW` signal imbalance**: Both `5/6`-ladder teacher sessions (`harvest_improve_01` and `harvest_repair_01`) started at `census accuracy=1.0` on 3-digit arithmetic and correctly withdrew when edits yielded `<= 0` gain. Consequently, the bundle taught the diagnostic Python steps (`1019.68x` outlier localization, hidden-state reads, paired testing, withdrawal) but contained **zero positive `SUBMIT` demonstrations** where a real corruption/bottleneck dropped baseline accuracy and a mechanistic edit measurably raised held-out accuracy (`delta_hat > 0`, `perm_p <= 0.05`).

---

## 2. Extracted Working 27B Training Data in Repo (`selfrewrite/data/qwen35_27b_v1/`)

To make every working `Qwen/Qwen3.5-27B` mechanistic step reproducible and directly consumable by `minimind` without touching `.gitignore`d `workspace/`, we implemented [`export_minimind_data.py`](distill/export_minimind_data.py) (`python3 -m selfrewrite.cli export-minimind`) and committed [`selfrewrite/data/qwen35_27b_v1/`](data/qwen35_27b_v1/README.md):

1. **[`canonical_27b/`](data/qwen35_27b_v1/canonical_27b/manifest.json)** (exact verified 27B bundle, zero held-out leaks across `48` eval problems and `8` eval layers `26..33`, `passed=true`):
   - [`sft_gold_trajectories.jsonl`](data/qwen35_27b_v1/canonical_27b/sft_gold_trajectories.jsonl) (`2` full 6-turn trajectories, `~12k` chars max): End-to-end `harvest_improve_01` and `harvest_repair_01` runs (`ladder=5/6`, `code_errors=0`).
   - [`sft_turn_primitives.jsonl`](data/qwen35_27b_v1/canonical_27b/sft_turn_primitives.jsonl) (`13` turn primitives): Covers all 6 mechanistic skills (`census_diagnosis`, `hidden_state_contrast`, `structured_edit`, `paired_validation`, `python_analysis`, `evidence_gate`, `trajectory_window`).
   - [`dpo_contrastive_pairs.jsonl`](data/qwen35_27b_v1/canonical_27b/dpo_contrastive_pairs.jsonl) (`12` pairs): Pairs `harvest_improve_01` / `harvest_repair_01` mechanistic steps (`chosen`) against real autonomous `Qwen3.5-27B` mistakes (`harvest_improve_00` verbose printing, `harvest_repair_00` failing to run `outlier_ratios` and missing `layers.1`) and `MockFailureEditor` noise/bad-submit turns (`rejected`).
   - [`rlvr_episodes.jsonl`](data/qwen35_27b_v1/canonical_27b/rlvr_episodes.jsonl) (`2` episodes): Path-normalized to `selfrewrite/data/qwen35_27b_v1/work_snapshots/`.

2. **[`minimind_ready/`](data/qwen35_27b_v1/minimind_ready/export_metadata.json)** (strictly aligned with [`dataset/lm_dataset.py`](../../dataset/lm_dataset.py) and [`model/tokenizer_config.json`](../../model/tokenizer_config.json)):
   - **`SFTDataset` PyArrow & Jinja rules**: `dataset/lm_dataset.py` enforces exact 5-field message structs (`role`, `content`, `reasoning_content`, `tools`, `tool_calls`) and rejects extra top-level JSON keys, while `model/tokenizer_config.json` prioritizes `message.reasoning_content` over splitting `<think>...</think>` inside `content`. Our exporter splits `<think>...</think>` into `reasoning_content` cleanly and strips trailing `<|im_end|>` markers to avoid double EOS / `<think>\n<think>` bugs.
   - **[`sft_27b_steps_raw_arch.jsonl`](data/qwen35_27b_v1/minimind_ready/sft_27b_steps_raw_arch.jsonl)** (`15` rows, max `11,904` chars): All `13` turn primitives + `2` full trajectories in exact `SFTDataset` format preserving raw `Qwen/Qwen3.5-27B` tensor names (`layers=64`).
   - **[`sft_minimind_adapted.jsonl`](data/qwen35_27b_v1/minimind_ready/sft_minimind_adapted.jsonl)** (`31` rows: `13` adapted primitives + `2` adapted trajectories + `16` CoT math anchors from `distill_train.jsonl`, max `7,613` chars total and `<= 5,115` chars across all `13` primitives): Replaces the `~3.5 KB` system prompt with a compact MiniMind Protocol 2 prompt (`~1.2 KB`), remaps `Qwen3.5-27B` `layers.62.linear_attn.out_proj.weight` / `layers.2.mlp.down_proj.weight` -> `MiniMindForCausalLM` `model.layers.6.self_attn.o_proj.weight` / `model.layers.1.mlp.down_proj.weight`, and condenses `64`-layer `read_1` JSON metadata to `8` layers (`layer_0_hidden..layer_7_hidden`).
   - **[`dpo_minimind_pairs.jsonl`](data/qwen35_27b_v1/minimind_ready/dpo_minimind_pairs.jsonl)** (`12` rows, max `4,550` chosen / `4,860` rejected chars): Exact `DPODataset` (`chosen`/`rejected` list of `{role, content}`) schema.
   - **[`rlaif_minimind_prompts.jsonl`](data/qwen35_27b_v1/minimind_ready/rlaif_minimind_prompts.jsonl)** (`15` rows): Exact `RLAIFDataset` schema where `conversations[:-1]` ends on a `user` observation and `conversations[-1]` holds the reference `assistant` step.

---

## 3. What It Takes to Train MiniMind for Self-Rewriting Capability

The `31` extracted MiniMind rows above are **high-signal seed examples**, not a standalone training corpus. Giving `MiniMindForCausalLM` reliable, generalizable self-rewriting ability requires closing four concrete gaps revealed by inspecting `model/`, `trainer/`, `dataset/`, and our Stage 2 L4 failure modes.

### Gap 1: Model Capacity & Architecture (`model/model_minimind.py`)
Inspecting [`model/model_minimind.py`](../../model/model_minimind.py) (`MiniMindConfig` defaults: `hidden_size=768`, `num_hidden_layers=8`, `vocab_size=6400`, `intermediate_size=2432`, `num_attention_heads=8`, `num_key_value_heads=4`), standard MiniMind checkpoints (`26M` at `H=512`, `~104M` at `H=768`) are far below even the `Qwen/Qwen3.5-2B` / `4B` rungs where general Python reasoning is fluent.

| Target Tier | Recommended `MiniMindConfig` | Approx Params | Achievable Scope |
|---|---|---:|---|
| **Tier 1: Constrained Mechanistic Floor (`R4m`)** | `hidden_size=768`, `num_hidden_layers=12` (or `use_moe=True, num_experts=4, num_experts_per_tok=1`) | `~104M–190M` | Single-primitive execution under the **Compact System Prompt** calling built-in helpers (`outlier_ratios`, `clip_outliers_median`, `remove_direction`, paired `forward` check, calibrated `WITHDRAW`/`SUBMIT`). Will NOT reliably invent novel linear-algebra algorithms from scratch. |
| **Tier 2: Autonomous Multi-Turn Self-Rewriter** | `hidden_size=1536`, `num_hidden_layers=24`, `num_attention_heads=16`, `num_key_value_heads=4`, `intermediate_size=4096` (or `use_moe=True, num_experts=8, num_experts_per_tok=2`) | `~0.65B–0.85B` dense (`~1.5B` MoE) | Minimum viable capacity to hold simultaneously: (a) clean math/reasoning accuracy (zero regression), (b) bug-free multi-line Python + NumPy REPL syntax over arbitrary layer counts `L`, and (c) multi-turn evidence synthesis (`read` -> hypothesize -> side-by-side `generate` -> `SUBMIT`/`WITHDRAW`). |

*Implementation advantage in this repo*: `MiniMindModel` already uses clean module names (`model.layers.{i}.self_attn.{q,k,v,o}_proj.weight` and `model.layers.{i}.mlp.{gate,up,down}_proj.weight`), which `MiniMindBackend` and `CanonicalWeightReader` natively inspect and edit without any quantization hurdles.

### Gap 2: Tokenizer Efficiency & Context Length (`model/tokenizer_config.json`, `trainer/`)
1. **Tokenizer Vocabulary (`6,400` -> `32,768` code-aware BPE for Tier 2)**:
   [`model/tokenizer.json`](../../model/tokenizer.json) uses a `6,400`-token bilingual vocab. Python identifiers (`model.language_model.layers`, `clip_outliers_median`, JSON keys) fragment into ~`1.8x–2.2x` more tokens than Qwen's `151,936`-vocab tokenizer. For Tier 2 (`>= 0.6B`), train a `32,768`-token BPE tokenizer balanced over Chinese/English prose + Python code while preserving MiniMind's special token IDs (`1=<|im_start|>`, `2=<|im_end|>`, `25=<think>`, `26=</think>`).
2. **Training Sequence Length (`768` -> `4,096` tokens)**:
   [`trainer/train_full_sft.py`](../../trainer/train_full_sft.py) defaults to `--max_seq_len 768` and [`trainer/train_lora.py`](../../trainer/train_lora.py) defaults to `--max_seq_len 340`. Because our prompt + JSON observation alone is `~500–1,400` tokens even in compacted form, truncating at `768` leaves zero assistant tokens unmasked (`labels == -100` across the assistant turn). Although `MiniMindConfig` sets `max_position_embeddings=32768`, RoPE must be exposed cleanly during pretraining/SFT (`--max_seq_len 4096`, optionally with YaRN `inference_rope_scaling=True` for `8,192+` sessions).

### Gap 3: Training Corpus Scale, Architecture Randomization & Replay Mix
To eliminate the Stage 2 `0.5625 / 0.8750 -> 0.0` forgetting failure AND prevent layer-index memorization (`layers.1` or `layers.6`):

| Corpus Slice | Target Mix | Minimum Count | Purpose & Design Rules |
|---|---:|---:|---|
| **1. General + Math CoT Replay** | **50%** | `25k–50k` rows | Multi-digit arithmetic, word math (`GSM8K`-style), general chat with `<think>...</think>` + `\boxed{}`. Mandatory anchor so LoRA/SFT never forgets how to produce `\boxed{}` answers within token budgets. |
| **2. Python REPL + NumPy/Tensor Coding** | **30%** | `15k–30k` rows | Single-turn and multi-turn Python scripts parsing `.jsonl`/`.npz`, computing norms, SVD/PCA, outlier statistics, JSON serialization — driving `code_errors` toward `0`. |
| **3. Randomized-Architecture Self-Rewrite Traces** | **20%** | `3k–10k` rows | Synthesized via [`programmatic_teacher.py`](distill/programmatic_teacher.py) + 27B/teacher rollouts across **randomized layer counts `L in {8, 12, 16, 24, 32, 64}`**, randomized corrupted layer indices (train layers only — never eval layers), and randomized tensor schemas (`self_attn.o_proj` / `linear_attn.out_proj` / `mlp.down_proj`). |
| **4. Contrastive DPO Pairs** | — | `3k–5k` pairs | Pair concise diagnostic Python vs verbose prose without code blocks (`harvest_improve_00`/`repair_00`), spectral edits vs `torch.randn` noise (`harvest_fail_random_noise_4b`), and calibrated `WITHDRAW` when `delta_hat <= 0` vs reckless `SUBMIT` (`harvest_fail_bad_submit_9b`). |
| **5. Positive `SUBMIT` + Calibrated `WITHDRAW` Balance** | **50 / 50** | Inside Slice 3 & 4 | **Critical fix over `v1`**: `v1` had `0` `SUBMIT` wins because clean 27B scored `1.0` on 3-digit math and `sigma_mult=150` on one layer of 27B also left math at `1.0`. Teacher harvesting must run on tasks/corruptions where base accuracy sits at `0.10–0.65` and valid clipping / direction damping yields `delta_hat >= +0.25` with `perm_p <= 0.05`. |

### Gap 4: End-to-End 5-Stage Training Pipeline Mapped to `trainer/` & `selfrewrite/`

Using [`selfrewrite/data/qwen35_27b_v1/minimind_ready/`](data/qwen35_27b_v1/minimind_ready/export_metadata.json) as the seed schema and scaling up Slices 1–5 above:

1. **Stage A — Base Pretraining + Code/Math Mid-Training (`train_pretrain.py`)**:
   - Pretrain `MiniMindForCausalLM` (`H=768, L=12` for Tier 1 floor probe, or `H=1536, L=24` for Tier 2) with Python + mathematical derivation mix up to context `2048 -> 4096`.
2. **Stage B — Reasoning & Python Tool-Use Full SFT (`trainer/train_full_sft.py`)**:
   - Establish strong `<think>...</think>` + `\boxed{}` reasoning and zero-error Python generation **before** touching self-rewriting (`--max_seq_len 4096 --learning_rate 2e-5`). Measure baseline `MiniMindBackend` census accuracy (`target >= 0.70` on Tier 2 before advancing).
3. **Stage C — Replay-Mixed Self-Rewrite SFT (`trainer/train_full_sft.py` or `trainer/train_lora.py`)**:
   - Mix scaled `selfrewrite` primitives & full trajectories (`20%`) with Math CoT (`50%`) + Python REPL (`30%`) replay. Use **lower LR and gentle adaptation** (`--learning_rate 2e-5` full SFT or `train_lora.py --lora_rank 16 --learning_rate 3e-5 --epochs 2 --max_seq_len 4096`) instead of our aggressive Stage 2 setting (`lr=1e-4, rank=32` on 80% protocol traces):
   ```bash
   python trainer/train_lora.py \
     --hidden_size 768 --num_hidden_layers 12 \
     --lora_rank 16 --lora_alpha 32 --lora_name lora_selfrewrite \
     --max_seq_len 4096 --batch_size 4 --learning_rate 3e-5 --epochs 2 \
     --from_weight full_sft \
     --data_path selfrewrite/data/qwen35_27b_v1/minimind_ready/sft_minimind_adapted.jsonl
   ```
   *(Merge LoRA weights into the standalone checkpoint before DPO or use `train_full_sft.py` directly on the full replay mix.)*
4. **Stage D — Contrastive DPO Alignment (`trainer/train_dpo.py`)**:
   - Train against [`dpo_minimind_pairs.jsonl`](data/qwen35_27b_v1/minimind_ready/dpo_minimind_pairs.jsonl) (expanded with Slice 4 pairs) keeping `trainer/train_dpo.py`'s conservative LR (`4e-8`–`5e-7`) and KL penalty so formatting/noise discipline sharpens without erasing math accuracy:
   ```bash
   python trainer/train_dpo.py \
     --hidden_size 768 --num_hidden_layers 12 \
     --max_seq_len 4096 --batch_size 2 --learning_rate 4e-8 --beta 0.15 --epochs 1 \
     --from_weight full_sft_selfrewrite \
     --data_path selfrewrite/data/qwen35_27b_v1/minimind_ready/dpo_minimind_pairs.jsonl
   ```
5. **Stage E — Executable-Sandbox RLVR (Extending `trainer/train_grpo.py` with `compute_rlvr_turn_reward`)**:
   - **Important incompatibility in stock `trainer/train_grpo.py`**: [`trainer/train_grpo.py`](../../trainer/train_grpo.py) (`calculate_rewards`, lines 37–68) scores replies using heuristic character-length bonuses (`20 <= len(response) <= 800`, `20 <= len(thinking) <= 300`) plus an external chat preference model (`LMForRewardModel`, default `internlm2-1_8b-reward`), which actively penalizes long diagnostic Python scripts and cannot verify weight edits.
   - Instead, replace `calculate_rewards` with [`compute_rlvr_turn_reward`](distill/train_student.py#L218-L264) running rollouts inside `SealedPythonSandbox` against real target snapshots: `+0.2` sandbox pass (`-0.4/-0.5` syntax error), `-0.8` random noise penalty, `+0.25` spectral/state read bonus, `+2.0 * delta_hat` paired outcome gain (clamped `[-1.0, +1.5]`), and `+-0.6` calibrated `SUBMIT`/`WITHDRAW` decision bonus, with group size `G >= 4` and KL coefficient `beta = 0.05–0.10`.
6. **Stage F — Frozen Zero-Gradient Evaluation Gate (`MiniMindBackend`, rung `R4m`)**:
   - Freeze all model weights (`requires_grad=False`, `eval()`), load into [`MiniMindBackend`](backends/minimind_backend.py), and evaluate on leak-disjoint `heldout_eval.jsonl` and held-out corruption layers (`run_downsize_bracket` promotion gates: `best_ladder >= 5/6`, `base_forgetting <= 0.02`, `repair_recovery_ratio >= 0.65` with `perm_p <= 0.05` on independent transfer tasks).
