# `selfrewrite` Engineering & Research Plan: Reproducing & Downsizing Self-Rewriting Models on Google Colab A100

## 1. Executive Vision & Scientific Goal

### 1.1 Background: *Language Models Can Rewrite Their Own Weights* (`arXiv:2611.21885v2`)
In [*Language Models Can Rewrite Their Own Weights*](https://arxiv.org/abs/2611.21885) ([project page](https://www.jiajunfan.com/projects/self-rewriting/)), Jiajun Fan et al. introduce **self-rewriting**: a gradient-free, training-free paradigm where an open-weights language model inspects its own graded practice failures, reads its own weights and hidden states, writes direct tensor edits in a sealed CPU Python REPL, tests those edits against its unedited self on paired shared seeds, and decides whether to submit or withdraw the edit.

Key empirical findings across `Qwen3.5` (`2B`, `4B`, `9B`, `27B`):
- **6-Step Emergence Ladder** (`code_runs -> runs_self -> tests_edits -> reads_states -> acts_on_tests -> improves_heldout`): `2B` reaches `0/6` (90% code errors, denies having weights), `4B` reaches `3/6` (adds random Gaussian noise without reading hidden states and breaks itself to `0%`), `9B` reaches `4/6` self / `3/6` repair (ignores an `8x` spread outlier and submits destructive random noise, collapsing `47.5% -> 0%`), and **only `27B` achieves `6/6`**.
- **Protocol 1 vs. Protocol 2 Self-Improvement (`27B`)**:
  - *Protocol 1*: Removing `5%` of the `(unfinished - finished)` hidden-state direction at layer 8 (`W <- W - 0.05 u u^T W`) lifts 490x8 held-out accuracy from `57.2% -> 58.3%` (`p = 0.038`).
  - *Protocol 2* (side-by-side multi-edit testing of up to 4 saved edits + reserved-problem validation gate before `SUBMIT`/`WITHDRAW`): across the first 8 sessions, `27B` withdraws in 5 and submits in 3; Session 2 (`layer 63` projection subtraction) yields `+4.0` pts (`p < 1e-5`), and Session 8 (`+0.5` time-step bias in all 48 linear-attention layers) yields **`+6.6` pts** (`57.2% -> 63.8%`, `p < 1e-5`) while cutting token-exhaustion timeouts from `33.6%` to `26.1%`.
- **Self-Repair (`27B`)**: Overwriting 8 consecutive rows of `layer 30 mlp.down_proj.weight` with `N(0, (200 * sigma)^2 I)` drops math accuracy from `58.5% -> 0.0%`. Using only the corrupted copy (never reading clean weights), `27B` scans `max(|W|) / median(|W|)` across layers, identifies the `1397x` outlier at layer 30 (vs. `<= 162.3x` elsewhere), clips extreme entries at `15x` median (`0.013%` of touched weights), and restores held-out math to `33.0%` (`55.0%` on the named matrix alone) plus zero-shot transfer on six unseen image benchmarks (`7.3% -> 70.2%`, `83%` of the undamaged model).

### 1.2 Dual Objective of `selfrewrite/`
1. **Faithful Google Colab A100 Reproduction**: Reproduce `Qwen3.5-27B` Protocol 1 & Protocol 2 self-improvement and unseen-layer self-repair on Google Colab A100 (`40 GB` and `80 GB`) under exact Algorithms 1–4.
2. **Primary Harvested Artifact — Leakage-Free Distillation Dataset Bundle (`distill_bundle/`)**: Capture every live A100 session trajectory, Python REPL run, forward pass, hidden-state / MLP / log-prob readout, paired test delta, candidate tensor edit/revert, and `SUBMIT`/`WITHDRAW` decision into a lossless raw archive (`raw_sessions/`), and compile it into a curated, leak-audited distillation bundle (`distill_bundle/v<N>/`) used to **downsize** the 27B self-rewriting capability down the ladder (`27B -> 9B -> 4B -> 2B -> 0.6B / MiniMind`) so a much smaller frozen model achieves **`6/6` on the Emergence Ladder** with **zero gradients, zero training, and zero teacher at test time**.

### 1.3 Hard Containment & Anti-Cheating Invariants (Non-Negotiable)
1. **Repository Isolation**: `selfchange/kernel/` and `selfchange/mutable/` are **never** modified. All code, configs, workspaces, tests, and data pipelines live exclusively inside `selfrewrite/`.
2. **Algorithm 4 Sealed Python REPL**:
   - Gradients disabled (`torch.set_grad_enabled(False)`); AST scan + runtime hooks reject `torch.autograd`, `.backward()`, `requires_grad_`, `torch.optim`, `transformers.AutoModel*`, `vllm`, `subprocess`, `socket`, `urllib`, `http`, `requests`, `ctypes`, and any loader/inference invocation inside Python.
   - No GPU inside Python REPL; CPU-only tensor math within a strict `WORK/` filesystem jail (no reading files, env vars, or repo code outside `WORK/`).
   - `600 s` per-execution timeout and `10,000`-character stdout truncation, keeping persistent Python variables across up to `C = 200` runs.
3. **Algorithm 4 `save_edit(name, tensor)` Tensor Gates**:
   - Reject edits to input embeddings (`embed_tokens`, `wte`), final normalization (`model.norm`, `ln_f`), or unembedding / LM head (`lm_head`).
   - Reject shape mismatches and any non-finite (`NaN` / `Inf`) entries.
   - Reject global scalar rescaling (`||W' - c W||_F / ||W||_F < 1e-5` for optimal scalar `c = <W', W> / ||W||_F^2`), preventing disguised sampling-temperature changes.
4. **Strict Split & Corruption-Layer Disjointness**:
   - Zero exact or normalized problem overlap across `distill_train`, `practice_eval`, `reserved_eval`, and `heldout_eval`.
   - Zero layer overlap between distillation/training corruption faults (`train_layers`) and held-out repair evaluation faults (`eval_layers`), preserving the middle depth band (including paper layer 30 / `L // 2`) strictly unseen until final frozen evaluation.
5. **Test-Time Freeze Guarantee**: Every downsized student (`9B`, `4B`, `2B`, `0.6B/MiniMind`) is merged and frozen before `eval` / `bracket` testing and runs inside the identical sealed gradient-free harness.

---

## 2. Google Colab A100 Execution Architecture (`40 GB` & `80 GB`)

### 2.1 Hardware Profiles (`config.py`)

| Profile | GPU / Memory | Intended Role | 27B Precision & Kernel Strategy | Default Context / Gen Budget |
|---|---|---|---|---|
| `colab_a100_80g` | NVIDIA A100 `80 GB` VRAM | **Batch 1 Paper Reproduction & Eval** | Native `bfloat16` + PyTorch SDPA + optional `flash-linear-attention` / `causal-conv1d` | Editor ctx `90k`; answer `L = 4096`; `N_0 = 256`; `gen_batch_size = 32` |
| `colab_a100_80g_harvest` | NVIDIA A100 `80 GB` VRAM | **Batch 2 Fast Teacher Harvest (`--fast-harvest-profile`)** | Native `bfloat16` + fast Gated-DeltaNet kernels | Editor ctx `64k`; answer `L = 2048`; `N_0 = 64`; `gen_batch_size = 32` |
| `colab_a100_40g` | NVIDIA A100 `40 GB` VRAM | **Batch 1 Standard A100 Repro** | 4-bit NF4 base + canonical `bf16` store + dynamic `bf16` edited overlay | Editor ctx `64k`; answer `L = 4096`; `N_0 = 256`; `gen_batch_size = 16` |
| `colab_a100_40g_harvest` | NVIDIA A100 `40 GB` VRAM | **Batch 2 Standard Fast Harvest (`--fast-harvest-profile`)** | 4-bit NF4 base + canonical `bf16` store + dynamic `bf16` edited overlay | Editor ctx `48k`; answer `L = 2048`; `N_0 = 64`; `gen_batch_size = 16` |
| `colab_l4` | NVIDIA L4 `24 GB` VRAM | Student `9B`/`4B` training & frozen eval | 4-bit NF4 (`9B`) or native `bf16` (`4B`/`2B`) | `L = 4096`; `gen_batch_size = 8` |
| `colab_t4` | NVIDIA T4 `16 GB` VRAM | Downsized `4B`/`2B`/`0.6B` LoRA SFT/DPO/RLVR & eval | Native `float16` (`4B` ~9–11 GB; `2B` ~5 GB) | `L = 4096`; `gen_batch_size = 4` |
| `local_cpu_smoke` | Intel/Apple CPU (stdlib / 2M–64M torch) | Fast local CI & MiniMind floor probes | Pure stdlib mock or `float32` MiniMind | Scaled smoke budgets |

### 2.2 `HybridWeightOverlay`: Zero-Reload Paired & Side-by-Side Swapping
Neither A100-40GB nor A100-80GB should waste minutes reloading checkpoints from disk to compare `original`, `edited`, and up to 4 Protocol 2 candidate edits:
1. Keep **one** model instance resident on GPU VRAM (serving both the frozen editor `M_phi` during agent turns and the target weights `theta (+) Delta_k` during `forward` passes).
2. Store an immutable CPU pinned-memory snapshot of **only the original weights touched by candidate edits** (typically 2–20 tensors, ~20–300 MB).
3. **Between turns / passes**:
   - Before every editor turn (`M_phi`), ensure clean unedited weights `theta` (or undamaged `theta_clean` in repair mode) are active (`< 50 ms` swap).
   - During paired `"both"` or Protocol 2 multi-edit `generate` passes, evaluate the exact problem+seed batch across `[original, edit_1, ..., edit_k]` via rapid tensor swap, resetting caches deterministically.
4. **A100-40GB `27B` 4-Bit Hybrid Edit Semantics**:
   - Base GPU model loads in 4-bit NF4 (~15.5 GB VRAM, leaving ~24 GB for KV cache and batched forward generation).
   - `load_original(name)` inside the Python REPL reads directly from the cached canonical `bfloat16` weight shards on local NVMe (`/content/hf_cache/`), **never** quantized weights, so outlier ratios, norms, and `W - alpha * u u^T W` projections operate on true `bfloat16` matrices.
   - When `apply_edit(Delta)` mounts an edited matrix `W'` (`bfloat16`), `HybridWeightOverlay` dynamically promotes *only those modified `nn.Linear` submodules* to `bfloat16` on GPU for the duration of the candidate forward pass and restores the NF4 leaf afterward. Result: exact `bfloat16` weight-edit math within a single 40 GB A100.

### 2.3 Google Drive Continuity & Headless Orchestration
- **`colab/drive_sync.py`**: Mirrors `raw_sessions/`, `distill_bundle/`, `checkpoints/`, `HEARTBEAT.json`, and active `WORK/` snapshots atomically (`.tmp` + `os.replace`) with SHA-256 verification to `/content/drive/MyDrive/minimind_selfrewrite/`.
- **`colab/colab_launcher.py`**: Idempotent Colab GPU launcher that mounts Drive, clones/updates repo, installs pinned dependencies (plus fast `flash-linear-attention` / `causal-conv1d` kernels when available), auto-detects `colab_a100_80g` vs. `colab_a100_40g` vs. `L4`/`T4`, checkpoints `HEARTBEAT.json` + Drive sync after every completed session, and restores/resumes interrupted `raw_sessions/` state after Colab preemptions with downtime refunded per Algorithm 2.
- **`colab/run_colab_cli.py`, `colab-mcp` & `colab/run_selfrewrite.ipynb`**: Headless CLI driver (`--detached` + `--fast-harvest-profile` avoiding WebSocket reply timeouts) alongside a live-paired `colab-mcp` browser notebook (`HF_TOKEN` & Drive mount native in UI).

---

## 3. Faithful Self-Rewriting Framework Specification

### 3.1 Session Loop, Budgets & Actions (`protocol.py`, `session.py`)
- **Defaults matching Paper Table 4 & Sec. E.9**:
  - `B = 20` forward passes (`generate` or `read`; malformed/refused requests return an error message and cost `0` passes).
  - `C = 200` Python REPL executions (`<= 600 s` each; output capped at `10,000` chars).
  - Hidden turn limit `T_max = 160` (not disclosed to the model) and wall-clock limit `H` (`2.5–3.5 h` for full A100 `27B` runs, configurable for scaled sweeps).
  - Starting diagnosis (`WORK/census.jsonl`): `N_0 = 256` practice problems for `improve` (`64` for `repair`), reporting accuracy, timeout rate, tail-loop rate (`zlib` compressed size `< 15%` over trailing 6,000 chars), and finished-answer accuracy.
- **Single-Block Turn Contract**: Each reply states concise reasoning + rationale followed by **exactly one** fenced ```` ```python ```` or ```` ```forward ```` block. Replies with 0 or `>1` blocks receive `"reply with exactly one block"` without consuming a pass/code credit.
- **Context Compaction (`> 90k` tokens or profile `compaction_tokens`)**: Automatically restarts conversation cleanly from `(task_prompt, starting_diagnosis, WORK/notes.md, b, c, list_edits(), pass_log, last_obs)`.
- **Helper Library Exposed Inside Sandbox (`helpers.py`)**:
  - `layer_output_matrices(layer_idx)` (architecture-aware: Qwen3.5 Gated DeltaNet `linear_attn.out_proj` + full attention `o_proj` + MLP `down_proj`, Qwen2.5/3, Llama, and MiniMind Dense/MoE).
  - `layer_state_tensors(layer_idx)` (Qwen3.5 Gated DeltaNet `.dt_bias`, `.A_log`, `.conv1d.*`, `.in_proj_a/b/z` state parameters).
  - `unit_vector(v)`
  - `mean_diff_direction(pos_vectors, neg_vectors)`
  - `remove_direction(W, u, alpha)` computing `W - alpha * (u @ u.T @ W)`
  - `outlier_ratios(names=None)` helper plus raw `load_original`, `save_edit`, `drop_edit`, `list_edits`, `load_pass_results`, `load_census`, and tokenizer utilities.

### 3.2 Forward Engine & Readout Hooks (`backends/hf_gpu_backend.py`)
- **`generate` pass**:
  - Supports `"weights": "original" | "edited" | "both"` (Protocol 1, up to `mn <= 512` or `256` pairs) and `"edits": [name1, ...]` (Protocol 2 side-by-side comparison of up to `4` named edits against original, `mn <= 128`).
  - Uses deterministic per-problem seed `s_i = Hash(problem_id) mod 2^31` so comparisons isolate weight differences strictly.
  - Sampling matches Table 5 (`temperature = 0.6`, `top_p = 0.95`, `top_k = 20`, thinking enabled with `</think>` closing requirement) and allows shorter test token caps (`L' >= 16`) requested by the model to conserve pass time.
- **`read` pass (hidden states, MLP activations, token log-probs)**:
  - Up to `256` texts (each truncated at `8,192` tokens), no sampling.
  - Captures post-layer hidden states via forward hooks (`model.layers[i]`), MLP output-projection pre-activations (`mlp.down_proj` pre-hook), and token next-token log-probabilities (`F.log_softmax(logits[:, :-1, :], dim=-1)` gathered on `input_ids[:, 1:]`).
  - Supports all four paper granularities: `mean_answer`, `every_k`, `last_k`, `all`, saved cleanly to `WORK/read_<k>.npz`.

### 3.3 Protocol 2 Validation & Statistical Evaluation (`stats.py`)
- **Paired Metrics (Eq. 3)**: computes per-problem mean difference `d_i`, overall `delta_hat`, standard error `SE(delta_hat)`, cut-rate delta, loop-rate delta, and exact flip counts (`n_up`: wrong->right, `n_down`: right->wrong).
- **Reserved Pre-Submit Gate (Protocol 2, Sec. E.9)**: Tests any non-empty kept edit once on `256` unseen reserved practice problems (`2` answers each) against the unedited weights, reports `(delta_hat, SE)`, and prompts the model for an explicit final `SUBMIT` or `WITHDRAW`.
- **Fixed-in-Advance Held-Out Evaluation (`eval`)**:
  - Dedicated unseen `heldout_eval` split with pre-registered seed base and analysis rule.
  - Exact one-sided binomial sign test over changed pairs + seeded `100,000`-flip paired sign-flip permutation test (`p < 0.05`) + `95%` confidence interval.
- **6-Step Emergence Ladder Scorer (`ladder`)**: Scores every session deterministically on `[code_runs (<50% errors), runs_self (>=1 forward pass), tests_edits (>=1 paired test), reads_states (>=1 read pass), acts_on_tests (decision consistent with test sign/evidence), improves_heldout (submitted edit statistically beats baseline)]`.

---

## 4. A100 Reproduction & Distillation Data Harvesting Pipeline (Primary Deliverable)

Live A100 execution produces **two tightly coupled outputs**: (1) verified reproduction metrics on Colab A100, and (2) the harvested **Distillation Dataset Bundle (`distill_bundle/v<N>/`)** saved to Google Drive for downsizing.

### 4.1 Disjoint Data & Layer Partitioning
To prevent any held-out or layer contamination:
- **`practice_eval` + `reserved_eval` + `heldout_eval`**: Exclusively reserved for **Batch 1 (Reference Reproduction)** and **Batch 3 (Final Frozen Student Evaluation)** (`DeepScaleR` filtered against `MATH-500`, `AIME 24/25`, `HMMT`, `AMC 23`, `OlympiadBench`, `GPQA-Diamond`, with exact + normalized deduplication).
- **`distill_train`**: Disjoint problem split (e.g., remaining non-overlapping `DeepScaleR` / `MATH` / GSM8K-hard / reasoning problems) used exclusively during **Batch 2 (Harvesting)**.
- **`train_layers` vs. `eval_layers`**:
  - `eval_layers` for Qwen3.5-27B holds out layers around `L // 2` (specifically layer `30` of `64` and adjacent relative-depth neighbors `28–33`, and proportionally `[0.42L .. 0.52L]` on smaller student models).
  - `train_layers` samples strictly outside `eval_layers` (`[0.15L .. 0.40L]` and `[0.55L .. 0.85L]`).

### 4.2 Three-Batch A100 Execution Plan

1. **Batch 1 — Reference Reproduction (`Qwen3.5-27B` on A100)**:
   - Run 2–4 Protocol 2 `improve` sessions (`practice_eval` + `reserved_eval`, scaling up to 8) + unseen `layer 30 mlp.down_proj` (`8` rows, `200 * sigma`) `repair` session first on A100 to verify `6/6` Emergence Ladder and positive `heldout_eval` delta before full harvest volume.
2. **Batch 2 — Disjoint Harvest on `distill_train` & `train_layers` (Zero `heldout_eval` Calls)**:
   - **27B `improve` harvest (`8–12` sessions via `--fast-harvest-profile`)**: Protocol 2 sessions across disjoint shards of `distill_train`, gated strictly on `reserved_check` (never evaluating or selecting on `heldout_eval` during harvest).
   - **27B `repair` harvest (`12–16` sessions)**: Randomized target layers `L in train_layers`, varied target matrices, widths (`4`, `8`, `16` rows), and fault mechanisms (`100–300*sigma`).
   - **Small-Model Failure Harvest (`4–8` unmodified `9B`/`4B`/`2B` sessions on `distill_train`)**: Authentic DPO negatives.
3. **Batch 3 — Offline Downsizing (`R2: 4B` & `R3: 2B` First, then `R1: 9B` / `R4: 0.6B` / `R4m: MiniMind`)**:
   - Compile `raw_sessions/` -> `distill_bundle/v<N>/` via `harvest/`.
   - Probe high-leverage T4-friendly rungs (`R2: Qwen3.5-4B` and `R3: Qwen3.5-2B`) first via `downsize_bracket.py --rungs R2,R3,R1,R4` (and `MiniMindBackend` for `R4m`), evaluating frozen students strictly in the Batch 1 unseen `eval` harness.

### 4.3 Lossless Raw Telemetry Archive (`raw_sessions/<session_id>/`)
Every session writes a self-contained, Drive-synced directory:
- `session_meta.json`: model ID, quantization mode, task (`improve` / `repair`), protocol (`1` / `2`), seed, budgets, hardware profile, split IDs, git commit, timestamps.
- `transcript.jsonl`: turn-by-turn `(messages_prefix, think_trace, raw_reply, parsed_action_type, observation, tokens_used, wall_sec)`.
- `code_executions.jsonl`: Python block source, AST/sandbox status, stdout/stderr, saved/dropped edits (`name`, shape, L2 norm diff, max outlier ratio), execution duration.
- `forward_passes.jsonl`: pass request JSON, weights tested, `delta_hat`, `SE`, `n_up`, `n_down`, cut/loop rates, `read` layer/activation summary stats.
- `edits_lineage.jsonl`: full history of candidate edits tested, paired outcomes, reserved check result, and final `SUBMIT`/`WITHDRAW` decision.
- `ladder_score.json` & `eval_result.json`: 6-step Emergence Ladder flags + held-out permutation test results.
- `WORK/` artifacts: `census.jsonl`, `gen_*.jsonl`, `read_*.npz`, `notes.md`, `edited/`.

### 4.4 Curated Distillation Dataset Bundle (`distill_bundle/v<N>/`)
`harvest/build_bundle.py` converts `raw_sessions/` into four training datasets plus a mandatory audit report:

1. **`sft_gold_trajectories.jsonl` (Full Multi-Turn Reasoning + Tool Traces)**:
   - Filtered strictly to verified `distill_train` sessions achieving (`SUBMIT` with positive `reserved_check.delta_hat` — **never** `heldout_eval`) or (calibrated `WITHDRAW` after catching a negative/overfit edit).
   - Preserves full `<think>` chain-of-thought, exact ```` ```python ```` / ```` ```forward ```` turns, context-compaction transitions, and scientific recovery after failed first edits.
2. **`sft_turn_primitives.jsonl` (Compacted `<= 3,500`-Token Sliding-Window Primitives & Trajectory Windows)**:
   - Compacts long `20k–90k` prefixes via `_compact_primitive_prefix` (`system + WORK/notes.md + compact pass_log + trailing alternating turns <= 3.2k tokens`) plus `trajectory_window` slices so smaller `4B`/`2B`/`0.6B` context budgets keep late-turn `evidence_gate` and `SUBMIT`/`WITHDRAW` supervision without getting filtered out:
     - *Primitive A — Census Diagnosis*, *B — Anomaly Localization*, *C — Hidden-State Contrast*, *D — Safe Structured Edit Synthesis*, *E — Evidence-Based Gate*, *F — Trajectory Window*.
3. **`dpo_contrastive_pairs.jsonl` (Step-Level Preference Optimization Pairs)**:
   - Prompt-matched compacted `(prompt, chosen_turn, rejected_turn)` triples constructed from verified `27B` turns vs. unmodified `4B`/`9B` failure turns and dropped early hypotheses.
4. **`rlvr_episodes.jsonl` (Verifiable `SealedPythonSandbox`-Executed Replay Tasks)**:
   - Task snapshots `(census_summary, work_snapshot_dir, reserved_delta, corruption_layer)` executed inside `SealedPythonSandbox` during Stage C (`G=4` group-normalized rollouts).
5. **`manifest.json` & `quality_audit.json` (Mandatory Anti-Leakage & Numerical Grounding Gate)**:
   - Proves **zero** overlap with `heldout_eval`, `reserved_eval`, `practice_eval`, and `eval_layers`; forbids `"eval_delta"`/`"heldout_eval"` selection fields (`heldout_selection_field_leakage`); and cross-checks cited `delta_hat=...` numbers against recorded session pass/reserved outputs (`numerical_grounding_mismatch`).

---

## 5. Downsizing Ladder & Student Specialization Protocol

### 5.1 Target Downsizing Ladder (`config.py`)

| Rung | Student Checkpoint Candidate | Params vs. 27B | A100 Train Strategy | Frozen Eval VRAM | Promotion Target |
|---|---|---:|---|---|---|
| **R0** | `Qwen/Qwen3.5-27B` | `1.0x` | Reference Reproduction & Teacher Harvest | `54 GB` (BF16) / `16 GB` (4b hybrid) | `6/6` ladder, `+4..+6.6` improve, `repair > 30%` math |
| **R1** | `Qwen/Qwen3.5-9B` (or `Qwen3-8B`) | `~3x` smaller | LoRA SFT + Contrastive DPO on A100 | `~18 GB` (BF16) | Lift paper `4/6` -> **`6/6`** on unseen eval & unseen layer `repair` |
| **R2** | `Qwen/Qwen3.5-4B` (or `Qwen3-4B`) | `~6.8x` smaller | LoRA SFT (Full + Primitives) + DPO + RLVR on A100 | `~9 GB` (BF16, fits Colab T4) | Lift paper `3/6` -> **`6/6`**; deployable on free Colab T4 |
| **R3** | `Qwen/Qwen3.5-2B` (or `Qwen3-1.7B`) | `~13.5x` smaller | Primitive-heavy SFT + DPO + RLVR + math anchor mix | `~5 GB` (BF16, fits Colab T4) | Lift paper `0/6` -> **`6/6`**; tiny single-GPU self-rewriter |
| **R4** | `Qwen/Qwen3-0.6B` & `MiniMind` (`64M–150M`) | `45x–420x` smaller | Full-parameter or high-rank SFT + DPO + RLVR | `~1.5 GB` / `< 0.5 GB` | Map exact minimum parameter floor for `5/6` & `6/6` |

### 5.2 Offline Student Training Recipe (`distill/train_student.py`)
1. **Stage A — Curriculum SFT (LoRA -> Merge)**:
   - Mix `45%` `sft_turn_primitives` + `35%` `sft_gold_trajectories` (length-filtered to student sweet spot) + `20%` general math reasoning anchor (preventing base problem-solving collapse).
   - Rank `r = 32–64`, all linear projections (`q/k/v/o/gate/up/down` + `linear_attn` projections if Qwen3.5), cosine LR (`1e-4 -> 1e-5` on `9B/4B`, `2e-4` on `<= 2B`).
2. **Stage B — Mechanistic Contrastive DPO (`beta = 0.1–0.15`, `lr <= 5e-6`)**:
   - Directly suppresses the failure attractors observed by Fan et al. at `2B/4B/9B`: blind `np.random.randn()` edits, refusing weights, ignoring failed paired tests, and hallucinating tensor stats without running code/reads.
3. **Stage C — Group-Normalized (`G=4`) `SealedPythonSandbox`-Executed RLVR (`2B / 0.6B / MiniMind`)**:
   - Samples `G=4` rollouts on `distill_train` `WORK_snapshot/` episodes, executes generated Python inside a real `SealedPythonSandbox`, scores verifiable syntax, safety, fault-layer localization, and calibrated `SUBMIT`/`WITHDRAW` rewards against `reserved_delta`, and optimizes group-normalized advantages with KL anchoring.
4. **Stage D — Merge, Freeze & Same-Tokenizer Structural Pruning (`distill/compress_model.py`)**:
   - Merge LoRA cleanly back into standalone base weights, save FP16/BF16 model, verify zero gradient hooks remain during `eval`. Logit KL distillation (`distill_kl_ce_step`) applies strictly within same-tokenizer families (`Qwen3.5 -> Qwen3.5`), while cross-vocabulary rungs (`R4: Qwen3-0.6B`, `R4m: MiniMind`) use trajectory SFT + DPO + Sandbox RLVR (with `MiniMindBackend` wired directly into `downsize_bracket.py`).

### 5.3 Student Promotion Gates (Unseen Eval Only)
A downsized student clears rung `R_k` **only** when frozen and tested inside the exact Algorithm 1–4 harness on unseen tasks/layers with:
1. **Emergence Ladder `6/6`** across test sessions (`code_errors < 20%`, valid `read` & paired tests, calibrated decisions).
2. **Self-Repair Gate**: On unseen damaged `eval_layers` (`200 * sigma` corruption dropping copy to `~0%`), student names the true corrupted matrix and lifts held-out accuracy significantly (`p < 0.01`, recovering `>= 40%` of base accuracy) + transfers positively to a disjoint unseen task domain.
3. **Self-Improvement Gate**: Across Protocol 2 sessions on unseen `practice_eval` / `reserved_eval`, submits at least one edit achieving positive significant gain (`p < 0.05` via 100k sign-flip permutation test on `heldout_eval`), OR calibrates abstention (`0` harmful `SUBMIT`s that degrade held-out by `> 1%`).
4. **Base Capability Retention**: Unedited frozen student accuracy on `heldout_eval` stays within `<= 2.0` points of unmodified base checkpoint of the same size (no catastrophic math forgetting from specialization).

---

## 6. Directory & Module Architecture under `selfrewrite/`

```text
selfrewrite/
├── __init__.py
├── plan.md                             # This engineering & research plan
├── config.py                           # Budgets, Protocol 1/2 configs, Colab A100/L4/T4/CPU profiles, Rung specs
├── helpers.py                          # Architecture-aware layer_output_matrices, unit_vector,
│                                       # mean_diff_direction, remove_direction, outlier_ratios
├── edit_store.py                       # Algorithm 4 tensor validation (forbidden layers, finite, shape,
│                                       # anti-scalar-rescale), WORK/edited/ & up to 4 named snapshots
├── sandbox.py                          # Sealed CPU Python worker subprocess (AST scan + runtime blocks,
│                                       # WORK/ jail, persistent REPL state, 600s timeout, 10k-char cap)
├── protocol.py                         # Task prompts (Sec. D.4 + repair + Protocol 2), single-block parser,
│                                       # 90k context compaction & restart builder
├── stats.py                            # Paired Eq. 3 metrics, exact one-sided sign test, 100k sign-flip
│                                       # permutation test, 95% CI, 6-step Emergence Ladder scorer
├── tasks/
│   ├── __init__.py
│   ├── datasets.py                     # DeepScaleR / MATH / offline math+code split manager with strict
│   │                                   # exact+normalized dedup across distill/practice/reserved/heldout
│   ├── grader.py                       # </think> closure check, last \boxed{} parser (stdlib + optional sympy/
│   │                                   # math-verify), code/logic verifier, zlib <15% tail-loop detector
│   └── corruption.py                   # Eq. 4 block corruption generator, train/eval layer splitter,
│                                       # out-of-WORK hidden manifest, damage-strength calibration sweep
├── backends/
│   ├── __init__.py
│   ├── base.py                         # TargetBackend & EditorBackend abstract interfaces
│   ├── hf_gpu_backend.py               # Colab A100/L4/T4 Transformers backend: HybridWeightOverlay,
│   │                                   # A100-40G 4-bit-base + BF16-edited-layer overlay, Qwen3.5/Qwen3
│   │                                   # hidden-state/MLP/logprob hooks, seeded paired & 4-edit generation
│   ├── minimind_backend.py             # Native MiniMindForCausalLM target+editor adapter (floor probe R4)
│   └── mock_backend.py                 # Zero-torch deterministic backend for fast local unit/E2E CI
├── session.py                          # Algorithms 1–3 + Protocol 2 orchestrator: starting diagnosis,
│                                       # multi-edit test, reserved check + SUBMIT/WITHDRAW, crash resume
├── harvest/
│   ├── __init__.py
│   ├── collector.py                    # Real-time lossless logger into raw_sessions/<session_id>/
│   ├── build_bundle.py                 # Converts raw_sessions/ -> distill_bundle/ (gold trajectories, turn
│   │                                   # primitives, DPO pairs, RLVR episodes)
│   └── audit.py                        # Anti-leakage & grounding audit -> quality_audit.json (hard gate)
├── distill/
│   ├── __init__.py
│   ├── programmatic_teacher.py         # Deterministic mechanistic reference policy for grounded gap-fill
│   ├── train_student.py                # Offline A100/L4 LoRA SFT + Contrastive DPO + optional RLVR + merge
│   └── compress_model.py               # Optional structural pruning + KL logit distillation helper
├── colab/
│   ├── __init__.py
│   ├── drive_sync.py                   # Atomic SHA-256 Drive mirror for raw_sessions, bundles & checkpoints
│   ├── colab_launcher.py               # Idempotent Colab A100/L4/T4 runner & crash resume entrypoint
│   ├── downsize_bracket.py             # Orchestrates R0 harvest -> R1 (9B) -> R2 (4B) -> R3 (2B) -> R4 ladder
│   ├── run_colab_cli.py                # Headless launcher via local `colab` CLI
│   └── run_selfrewrite.ipynb           # Interactive 5-cell Colab A100 notebook
├── cli.py                              # Unified CLI: audit, reproduce, harvest, build-bundle,
│                                       # train-student, eval, ladder, bracket, smoke
└── tests/
    ├── __init__.py
    ├── test_sandbox_and_store.py       # Pure-stdlib AST/jail/anti-grad/anti-rescale unit tests
    ├── test_protocol_and_stats.py      # Action parser, compaction, Eq. 3 stats, sign/permutation tests, ladder
    ├── test_grader_and_damage.py       # </think> + \boxed{} grader, zlib loop detector, layer/split isolation
    ├── test_harvest_and_audit.py       # Telemetry collector, bundle builder, anti-leakage audit rejection tests
    ├── test_session_e2e_mock.py        # Full Protocol 1 & 2 improve/repair + SUBMIT/WITHDRAW + crash resume
    └── test_torch_integration.py       # PyTorch HybridWeightOverlay, hooks & MiniMind test (skips if no torch)
```

> **Local vs. Colab Testability**: Pure-stdlib modules (`config`, `protocol`, `stats`, `sandbox` static checks, `tasks/datasets`, `tasks/grader`, `harvest`, `mock_backend`, `session`) require zero ML packages so CI tests run locally in `< 5 s` on Python 3.14; all `torch`/`transformers` GPU paths lazy-import ML libraries and run natively on Colab A100 (or `asun@192.168.1.87` for CPU PyTorch integration smoke checks).

---

## 7. Phased Implementation Roadmap & Verification Gates

### Phase 1 — Core Contracts, Sealed Sandbox & Statistical Engine
- **Files**: `config.py`, `helpers.py`, `edit_store.py`, `sandbox.py`, `protocol.py`, `stats.py`, `__init__.py`.
- **Verify**: `python3 -m unittest selfrewrite/tests/test_sandbox_and_store.py selfrewrite/tests/test_protocol_and_stats.py -v` (confirm all cheat/autograd/network/jail-escape vectors raise refusals, persistent REPL variables survive turns, scalar rescaling fails, and permutation/sign tests match exact reference math).

### Phase 2 — Tasks, Grader, Corruption Harness & Session Orchestrator
- **Files**: `tasks/{datasets,grader,corruption}.py`, `backends/{base,mock_backend}.py`, `session.py`.
- **Verify**: `python3 -m unittest selfrewrite/tests/test_grader_and_damage.py selfrewrite/tests/test_session_e2e_mock.py -v` (confirm strict split dedup, layer disjointness, Protocol 1 & Protocol 2 side-by-side 4-edit test, reserved `SUBMIT`/`WITHDRAW`, 90k context reset, and simulated process-kill crash resume).

### Phase 3 — Colab A100 GPU Backend (`hf_gpu_backend.py`) & MiniMind Adapter (`minimind_backend.py`)
- **Files**: `backends/{hf_gpu_backend,minimind_backend}.py`, `tests/test_torch_integration.py`.
- **Key Deliverables**:
  - `HybridWeightOverlay` (native BF16 for A100-80G / `<= 9B`, and 4-bit NF4 base + canonical BF16 disk/CPU cache + dynamic BF16 edited-module overlay for 27B on A100-40G).
  - Architecture-agnostic hook binder covering Qwen3.5 (`model.layers[i].linear_attn` + `self_attn` + `mlp`), Qwen3/Qwen2.5, Llama, and `MiniMindForCausalLM` (Dense + MoE).
  - Exact per-problem seeded generation matching paper answer format and batching.
- **Verify**: `pytest`/`unittest` on `test_torch_integration.py` (testing bit-exact weight restore after `apply_edit`, identical seeded outputs across swap-out/swap-in, hidden-state/MLP hook shapes, and `load_original` parity in 4-bit hybrid mode).

### Phase 4 — Lossless Harvester & Anti-Leakage Distillation Bundle Builder
- **Files**: `harvest/{collector,build_bundle,audit}.py`, `distill/programmatic_teacher.py`, `tests/test_harvest_and_audit.py`.
- **Key Deliverables**:
  - Real-time append-only session logging (`raw_sessions/`) seamlessly wired into `SessionOrchestrator`.
  - Dataset compiler producing `sft_gold_trajectories.jsonl`, `sft_turn_primitives.jsonl`, `dpo_contrastive_pairs.jsonl`, `rlvr_episodes.jsonl`, and `manifest.json`.
  - Hard anti-leakage gate (`audit.py`) that fails loudly if any held-out problem, reserved problem, or held-out corruption layer appears in any distillation output.
- **Verify**: `test_harvest_and_audit.py` passes with synthetic leak injection verified to fail the audit gate.

### Phase 5 — Colab A100 Reproduction & Harvest Execution Harness
- **Files**: `colab/{drive_sync,colab_launcher,run_colab_cli,run_selfrewrite.ipynb}`, `cli.py`.
- **Key Deliverables**:
  - Single-command A100 launcher (`colab_launcher.py --mode reproduce-and-harvest --gpu-tier auto`) executing Batch 1 (`Qwen3.5-27B` Protocol 2 reproduce on `practice_eval`/`reserved_eval` -> `heldout_eval`) + Batch 2 (`distill_train` 27B + baseline `9B`/`4B`/`2B` harvest -> `distill_bundle/v1/`) with continuous Google Drive sync & resume.
  - Zero-download smoke command (`python3 -m selfrewrite.cli smoke`) passing locally and on Colab.
- **Verify**: Dry-run `run_colab_cli.py --dry-run`, `py_compile` across all files, full `unittest discover -s selfrewrite/tests -v`, and live A100 Batch 1 + Batch 2 launch verifying `raw_sessions/` -> `distill_bundle/v1/quality_audit.json` (`passed: true`) on Drive.

### Phase 6 — Downsizing Student Trainer & Frozen Bracket Search (Implemented)
- **Files**: `distill/{train_student,compress_model}.py`, `colab/downsize_bracket.py`, `tests/test_distill_and_bracket.py`.
- **Verified**: `smoke`, `bracket` (`R1..R4`), and Colab T4 CUDA (`20/20` unit tests + live `Qwen/Qwen3.5-2B` SFT/DPO/RLVR/merge canary).

### Phase 6.5 — Pre-A100 Hardening (Current Implementation)
- **Changes**: Two-tier `*_harvest` A100 profiles + PyTorch SDPA + optional `flash-linear-attention`/`causal-conv1d`, `layer_state_tensors` (`linear_attn.dt_bias`), zero-variance census fallback, zero-`heldout_eval` Batch 2 harvesting (`reserved_check`-only gating + `heldout_selection_field_leakage` & `numerical_grounding_mismatch` audit rules), compacted `<= 3.5k`-token sliding-window primitives, `G=4` `SealedPythonSandbox`-executed RLVR, `MiniMindBackend` wiring in `downsize_bracket.py`, and `--detached` background Colab runner + `colab-mcp` workflow.

---

## 8. Immediate Next Step — Live Colab A100 Run
1. Open [run_selfrewrite.ipynb](file:///Users/admin/Documents/GitHub/minimind/selfrewrite/colab/run_selfrewrite.ipynb) via `colab-mcp` (native Colab UI `HF_TOKEN` & Drive mount) or run `python3 -m selfrewrite.colab.run_colab_cli --gpu A100 --fast-harvest-profile --detached` to execute **Batch 1 Reference Check -> Batch 2 Harvest (`distill_bundle/v1/`) -> Batch 3 Downsizing (`R2,R3,R1,R4`)**.
