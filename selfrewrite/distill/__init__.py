"""Offline student downsizing, structural compression, & deterministic mechanistic reference policies."""

from selfrewrite.distill.compress_model import (
    distill_kl_ce_step,
    drop_decoder_layers,
    estimate_compressed_params,
    plan_compression,
    prune_mlp_width,
)
from selfrewrite.distill.programmatic_teacher import (
    ProgrammaticImproveEditor,
    ProgrammaticRepairEditor,
)
from selfrewrite.distill.train_student import (
    StudentTrainConfig,
    build_dpo_dataset,
    build_sft_curriculum,
    compute_rlvr_turn_reward,
    merge_and_freeze_student,
    run_hf_dpo_stage,
    run_hf_rlvr_stage,
    run_hf_sft_stage,
    train_downsize_student,
    verify_bundle_audit_gate,
)

__all__ = [
    "ProgrammaticImproveEditor",
    "ProgrammaticRepairEditor",
    "StudentTrainConfig",
    "verify_bundle_audit_gate",
    "build_sft_curriculum",
    "build_dpo_dataset",
    "compute_rlvr_turn_reward",
    "run_hf_sft_stage",
    "run_hf_dpo_stage",
    "run_hf_rlvr_stage",
    "merge_and_freeze_student",
    "train_downsize_student",
    "estimate_compressed_params",
    "plan_compression",
    "prune_mlp_width",
    "drop_decoder_layers",
    "distill_kl_ce_step",
]
