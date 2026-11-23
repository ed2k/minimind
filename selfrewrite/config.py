from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> Path:
    """Atomically write UTF-8 text via a temporary sibling file and rename."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(text, encoding=encoding)
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return target


def atomic_write_bytes(path: Path, data: bytes) -> Path:
    """Atomically write raw bytes via a temporary sibling file and rename."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        tmp.write_bytes(data)
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return target


def compute_file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 hex digest of a file cleanly."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ComputeProfile:
    """Hardware and memory settings for local smoke tests and Colab GPUs."""

    name: str
    device: str
    dtype: str
    quantization: str
    vram_gb: int
    editor_max_context: int
    answer_max_tokens: int
    starting_diagnosis_n: int
    gen_batch_size: int
    max_wall_time_sec: int
    tensor_storage: str


PROFILES: dict[str, ComputeProfile] = {
    "colab_a100_80g": ComputeProfile(
        name="colab_a100_80g",
        device="cuda:0",
        dtype="bfloat16",
        quantization="none",
        vram_gb=80,
        editor_max_context=90000,
        answer_max_tokens=4096,
        starting_diagnosis_n=256,
        gen_batch_size=8,
        max_wall_time_sec=10800,
        tensor_storage="safetensors",
    ),
    "colab_a100_40g": ComputeProfile(
        name="colab_a100_40g",
        device="cuda:0",
        dtype="bfloat16",
        quantization="nf4_bf16_hybrid",
        vram_gb=40,
        editor_max_context=65536,
        answer_max_tokens=4096,
        starting_diagnosis_n=256,
        gen_batch_size=4,
        max_wall_time_sec=10800,
        tensor_storage="safetensors",
    ),
    "colab_l4": ComputeProfile(
        name="colab_l4",
        device="cuda:0",
        dtype="bfloat16",
        quantization="nf4_bf16_hybrid",
        vram_gb=24,
        editor_max_context=49152,
        answer_max_tokens=4096,
        starting_diagnosis_n=128,
        gen_batch_size=4,
        max_wall_time_sec=7200,
        tensor_storage="safetensors",
    ),
    "colab_t4": ComputeProfile(
        name="colab_t4",
        device="cuda:0",
        dtype="float16",
        quantization="none",
        vram_gb=16,
        editor_max_context=32768,
        answer_max_tokens=4096,
        starting_diagnosis_n=128,
        gen_batch_size=2,
        max_wall_time_sec=7200,
        tensor_storage="safetensors",
    ),
    "local_cpu_smoke": ComputeProfile(
        name="local_cpu_smoke",
        device="cpu",
        dtype="float32",
        quantization="none",
        vram_gb=0,
        editor_max_context=16384,
        answer_max_tokens=512,
        starting_diagnosis_n=16,
        gen_batch_size=4,
        max_wall_time_sec=600,
        tensor_storage="json",
    ),
}


@dataclass(frozen=True)
class DownsizeRung:
    """One model rung along the 27B -> small self-rewriter downsizing ladder."""

    rung_id: str
    model_id: str
    params_b: float
    recommended_profile: str
    train_strategy: str
    eval_quantization: str
    expected_baseline_ladder: int
    target_ladder: int
    notes: str = ""


DOWNSIZE_RUNGS: dict[str, DownsizeRung] = {
    "R0": DownsizeRung(
        rung_id="R0",
        model_id="Qwen/Qwen3.5-27B",
        params_b=27.0,
        recommended_profile="colab_a100_80g",
        train_strategy="none_reference_teacher",
        eval_quantization="none",
        expected_baseline_ladder=6,
        target_ladder=6,
        notes="Reference teacher; also runs on colab_a100_40g via nf4_bf16_hybrid.",
    ),
    "R1": DownsizeRung(
        rung_id="R1",
        model_id="Qwen/Qwen3.5-9B",
        params_b=9.0,
        recommended_profile="colab_a100_40g",
        train_strategy="lora_sft_dpo",
        eval_quantization="none",
        expected_baseline_ladder=4,
        target_ladder=6,
        notes="First downsized target; lift paper 4/6 baseline to 6/6.",
    ),
    "R2": DownsizeRung(
        rung_id="R2",
        model_id="Qwen/Qwen3.5-4B",
        params_b=4.0,
        recommended_profile="colab_t4",
        train_strategy="lora_sft_dpo_rlvr",
        eval_quantization="none",
        expected_baseline_ladder=3,
        target_ladder=6,
        notes="Primary T4-friendly native BF16/FP16 student rung.",
    ),
    "R3": DownsizeRung(
        rung_id="R3",
        model_id="Qwen/Qwen3.5-2B",
        params_b=2.0,
        recommended_profile="colab_t4",
        train_strategy="primitive_sft_dpo_rlvr",
        eval_quantization="none",
        expected_baseline_ladder=0,
        target_ladder=6,
        notes="Stretch 2B rung overcoming 90% code errors and weight denial.",
    ),
    "R4": DownsizeRung(
        rung_id="R4",
        model_id="Qwen/Qwen3-0.6B",
        params_b=0.6,
        recommended_profile="colab_t4",
        train_strategy="full_sft_dpo_rlvr",
        eval_quantization="none",
        expected_baseline_ladder=0,
        target_ladder=5,
        notes="Sub-1B minimum parameter floor probe.",
    ),
    "R4m": DownsizeRung(
        rung_id="R4m",
        model_id="minimind-64m",
        params_b=0.064,
        recommended_profile="local_cpu_smoke",
        train_strategy="full_sft_dpo_rlvr",
        eval_quantization="none",
        expected_baseline_ladder=0,
        target_ladder=5,
        notes="Extreme MiniMind-3 64M parameter floor probe.",
    ),
}


@dataclass
class SessionConfig:
    """Configuration for one gradient-free self-rewriting session (Algorithms 1-4)."""

    task: str = "improve"  # "improve" | "repair"
    protocol: int = 2  # 1 | 2
    max_forward_passes: int = 20
    max_code_runs: int = 200
    max_turns: int = 160
    wall_time_limit_sec: float = 10800.0
    code_timeout_sec: float = 600.0
    code_output_max_chars: int = 10000
    compaction_tokens: int = 90000
    answer_max_tokens: int = 4096
    editor_max_new_tokens: int = 12000
    read_max_tokens: int = 8192
    read_max_texts: int = 256
    starting_diagnosis_n: int = 256
    max_answers_per_pass: int | None = None
    max_named_edits: int = 4
    reserved_check_n: int = 256
    reserved_check_samples: int = 2
    sampling_temperature: float = 0.6
    sampling_top_p: float = 0.95
    sampling_top_k: int = 20
    require_thinking: bool = True
    loop_tail_chars: int = 6000
    loop_compress_ratio: float = 0.15
    model_arch_summary: str = "Transformer language model"
    session_id: str = ""
    seed: int = 0
    stop_file_name: str = "STOP"
    damage_info: dict | None = None

    def __post_init__(self) -> None:
        if self.task not in ("improve", "repair"):
            raise ValueError(f"Unsupported task {self.task!r}; expected 'improve' or 'repair'")
        if self.protocol not in (1, 2):
            raise ValueError(f"Unsupported protocol {self.protocol!r}; expected 1 or 2")
        if self.max_answers_per_pass is None:
            self.max_answers_per_pass = 512 if self.protocol == 1 else 128

    def to_dict(self) -> dict:
        return asdict(self)


def detect_profile_name() -> str:
    """Detect optimal compute profile from CUDA GPU VRAM when available."""
    override = os.environ.get("SELFREWRITE_PROFILE")
    if override and override in PROFILES:
        return override
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if total_gb >= 70.0:
                return "colab_a100_80g"
            if total_gb >= 35.0:
                return "colab_a100_40g"
            if total_gb >= 20.0:
                return "colab_l4"
            return "colab_t4"
    except Exception:
        pass
    return "local_cpu_smoke"


def get_profile(name: str | None = None) -> ComputeProfile:
    """Resolve explicit or hardware-detected ComputeProfile."""
    resolved = detect_profile_name() if (name is None or name == "auto") else name
    if resolved not in PROFILES:
        raise KeyError(f"Unknown compute profile {resolved!r}; expected one of {sorted(PROFILES)}")
    return PROFILES[resolved]
