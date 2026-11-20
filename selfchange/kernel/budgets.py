from __future__ import annotations

import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ComputeProfile:
    """Hardware and resource limits for an execution environment."""

    name: str
    device: str
    dtype: str
    use_grad_scaler: bool
    hidden_size: int
    num_hidden_layers: int
    use_moe: bool
    max_seq_len: int
    max_gen_len: int
    max_total_len: int
    batch_size: int
    num_generations: int
    num_workers: int
    cycle_timeout_sec: int
    subprocess_timeout_sec: int
    max_subprocess_memory_mb: int
    max_file_size_mb: int
    min_free_disk_mb: int
    max_cycles_per_run: int
    max_turns_per_cycle: int
    checkpoint_retention_count: int


PROFILES: dict[str, ComputeProfile] = {
    "local_intel_cpu": ComputeProfile(
        name="local_intel_cpu",
        device="cpu",
        dtype="float32",
        use_grad_scaler=False,
        hidden_size=128,
        num_hidden_layers=4,
        use_moe=False,
        max_seq_len=384,
        max_gen_len=192,
        max_total_len=640,
        batch_size=2,
        num_generations=2,
        num_workers=0,
        cycle_timeout_sec=180,
        subprocess_timeout_sec=60,
        max_subprocess_memory_mb=4096,
        max_file_size_mb=64,
        min_free_disk_mb=512,
        max_cycles_per_run=5,
        max_turns_per_cycle=6,
        checkpoint_retention_count=4,
    ),
    "colab_t4": ComputeProfile(
        name="colab_t4",
        device="cuda:0",
        dtype="float16",
        use_grad_scaler=True,
        hidden_size=768,
        num_hidden_layers=8,
        use_moe=False,
        max_seq_len=768,
        max_gen_len=512,
        max_total_len=1536,
        batch_size=2,
        num_generations=4,
        num_workers=2,
        cycle_timeout_sec=900,
        subprocess_timeout_sec=300,
        max_subprocess_memory_mb=10240,
        max_file_size_mb=256,
        min_free_disk_mb=2048,
        max_cycles_per_run=20,
        max_turns_per_cycle=6,
        checkpoint_retention_count=4,
    ),
    "colab_l4_a100": ComputeProfile(
        name="colab_l4_a100",
        device="cuda:0",
        dtype="bfloat16",
        use_grad_scaler=False,
        hidden_size=768,
        num_hidden_layers=8,
        use_moe=False,
        max_seq_len=1024,
        max_gen_len=768,
        max_total_len=2048,
        batch_size=4,
        num_generations=4,
        num_workers=4,
        cycle_timeout_sec=480,
        subprocess_timeout_sec=240,
        max_subprocess_memory_mb=16384,
        max_file_size_mb=512,
        min_free_disk_mb=4096,
        max_cycles_per_run=40,
        max_turns_per_cycle=8,
        checkpoint_retention_count=4,
    ),
}


def detect_profile_name() -> str:
    """Detect whether execution runs on Intel CPU or a Colab GPU."""
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            total_bytes = torch.cuda.get_device_properties(0).total_memory
            if total_bytes >= 20 * (1024**3):
                return "colab_l4_a100"
            return "colab_t4"
    except ImportError:
        pass
    return "local_intel_cpu"


def get_profile(name: str | None = None) -> ComputeProfile:
    """Return a ComputeProfile by explicit name or hardware auto-detection."""
    resolved = detect_profile_name() if name is None else name
    if resolved not in PROFILES:
        raise KeyError(f"Unknown compute profile {resolved!r}; expected one of {sorted(PROFILES)}")
    return PROFILES[resolved]


def check_resource_headroom(workspace_dir: Path, profile: ComputeProfile) -> tuple[bool, dict]:
    """Verify disk headroom (and optional CUDA memory) against profile minimums."""
    probe = Path(workspace_dir).resolve()
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    free_disk_mb = round(usage.free / (1024 * 1024), 2)
    ok = free_disk_mb >= profile.min_free_disk_mb

    info: dict = {
        "profile": profile.name,
        "platform": platform.system(),
        "machine": platform.machine(),
        "free_disk_mb": free_disk_mb,
        "min_free_disk_mb": profile.min_free_disk_mb,
        "disk_ok": ok,
        "profile_limits": asdict(profile),
    }

    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info(0)
            info["cuda_free_mb"] = round(free_b / (1024 * 1024), 2)
            info["cuda_total_mb"] = round(total_b / (1024 * 1024), 2)
    except Exception:
        pass

    return ok, info
