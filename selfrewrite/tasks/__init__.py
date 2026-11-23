"""Benchmark splits, answer grading, and synthetic block-damage harness."""

from selfrewrite.tasks.corruption import (
    CorruptionSpec,
    apply_block_corruption,
    create_damaged_workspace,
    split_train_eval_layers,
)
from selfrewrite.tasks.datasets import (
    Problem,
    build_builtin_suites,
    deduplicate_splits,
    load_jsonl_problems,
    normalize_problem_text,
    problem_seed,
    save_jsonl_problems,
)
from selfrewrite.tasks.grader import (
    answers_equivalent,
    extract_last_boxed,
    grade_response,
    has_closed_thinking,
    is_looping_answer,
)

__all__ = [
    "Problem",
    "CorruptionSpec",
    "build_builtin_suites",
    "deduplicate_splits",
    "load_jsonl_problems",
    "save_jsonl_problems",
    "normalize_problem_text",
    "problem_seed",
    "split_train_eval_layers",
    "apply_block_corruption",
    "create_damaged_workspace",
    "has_closed_thinking",
    "extract_last_boxed",
    "answers_equivalent",
    "is_looping_answer",
    "grade_response",
]
