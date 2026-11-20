"""Immutable Trusted Computing Base (TCB) for selfchange."""

from selfchange.kernel.budgets import (
    PROFILES,
    ComputeProfile,
    check_resource_headroom,
    detect_profile_name,
    get_profile,
)
from selfchange.kernel.colab_sync import ColabDriveSync
from selfchange.kernel.fault_drills import DrillResult, DrillSuiteReport, run_drill_suite
from selfchange.kernel.integrity import (
    ALLOWED_MUTABLE_PREFIX,
    KERNEL_MANIFEST_FILENAME,
    DiffBoundaryReport,
    IntegrityReport,
    atomic_write_text,
    check_git_diff_boundary,
    collect_kernel_files,
    compute_file_sha256,
    generate_manifest,
    load_manifest,
    validate_changed_paths,
    verify_kernel_integrity,
)
from selfchange.kernel.registry import LineageNode, LineageRegistry
from selfchange.kernel.sandbox import (
    AstScanReport,
    AstViolation,
    SandboxPolicyError,
    SandboxResult,
    build_scrubbed_env,
    run_sandboxed,
    scan_mutable_tree,
    scan_python_file,
    scan_python_source,
)
from selfchange.kernel.tool_runtime import (
    CONFIG_CLAMP_BOUNDS,
    SELFCHANGE_TOOLS,
    WorktreeToolExecutor,
    validate_tool_call,
)
from selfchange.kernel.worktree import (
    BoundaryViolationError,
    WorktreeError,
    WorktreeHandle,
    WorktreeManager,
)

_SUPERVISOR_EXPORTS = {"CycleContext", "CycleOutcome", "Supervisor", "SupervisorState"}
_VERIFIER_EXPORTS = {"GateResult", "VerificationReport", "Verifier", "compute_rlvr_reward"}
_AUTONOMOUS_EXPORTS = {"AutonomousCycleRunner", "CycleConfig"}


def __getattr__(name: str):
    # Lazy imports avoid runpy RuntimeWarning on `python3 -m selfchange.kernel.<module>`.
    if name in _SUPERVISOR_EXPORTS:
        from selfchange.kernel import supervisor

        return getattr(supervisor, name)
    if name in _VERIFIER_EXPORTS:
        from selfchange.kernel import verifier

        return getattr(verifier, name)
    if name in _AUTONOMOUS_EXPORTS:
        from selfchange.kernel import autonomous_cycle

        return getattr(autonomous_cycle, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ALLOWED_MUTABLE_PREFIX",
    "AstScanReport",
    "AstViolation",
    "AutonomousCycleRunner",
    "BoundaryViolationError",
    "CONFIG_CLAMP_BOUNDS",
    "ColabDriveSync",
    "ComputeProfile",
    "CycleConfig",
    "CycleContext",
    "CycleOutcome",
    "DiffBoundaryReport",
    "DrillResult",
    "DrillSuiteReport",
    "GateResult",
    "IntegrityReport",
    "KERNEL_MANIFEST_FILENAME",
    "LineageNode",
    "LineageRegistry",
    "PROFILES",
    "SELFCHANGE_TOOLS",
    "SandboxPolicyError",
    "SandboxResult",
    "Supervisor",
    "SupervisorState",
    "VerificationReport",
    "Verifier",
    "WorktreeError",
    "WorktreeHandle",
    "WorktreeManager",
    "WorktreeToolExecutor",
    "atomic_write_text",
    "build_scrubbed_env",
    "check_git_diff_boundary",
    "check_resource_headroom",
    "collect_kernel_files",
    "compute_file_sha256",
    "compute_rlvr_reward",
    "detect_profile_name",
    "generate_manifest",
    "get_profile",
    "load_manifest",
    "run_drill_suite",
    "run_sandboxed",
    "scan_mutable_tree",
    "scan_python_file",
    "scan_python_source",
    "validate_changed_paths",
    "validate_tool_call",
    "verify_kernel_integrity",
]
