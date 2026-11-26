from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from selfrewrite.config import atomic_write_text

_BENCHMARK_EXCLUDE_TOKENS = (
    "math-500",
    "math500",
    "aime",
    "hmmt",
    "amc",
    "olympiadbench",
    "gpqa",
)


@dataclass(frozen=True)
class Problem:
    problem_id: str
    question: str
    answer: str
    domain: str = "math"
    problem_type: str = "math"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_problem_text(text: str) -> str:
    """Canonicalize problem prompt for exact and near-duplicate cross-split checks."""
    s = str(text or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]+", "", s)
    return s.strip()


def problem_seed(problem_id: str, seed_base: int = 0) -> int:
    """Deterministic seed s_i = Hash(problem_id, seed_base) mod 2^31 (Algorithm 3)."""
    digest = hashlib.sha256(f"{seed_base}:{problem_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def deduplicate_splits(splits: dict[str, list[Problem]]) -> dict[str, list[Problem]]:
    """Remove any cross-split problem overlap while protecting eval splits first."""
    precedence = ["heldout_eval", "reserved_eval", "practice_eval", "transfer_eval", "distill_train"]
    ordered_keys = [k for k in precedence if k in splits] + [k for k in splits if k not in precedence]

    seen_norms: set[str] = set()
    cleaned: dict[str, list[Problem]] = {}
    for split_name in ordered_keys:
        unique_list: list[Problem] = []
        seen_ids: set[str] = set()
        for prob in splits[split_name]:
            norm = normalize_problem_text(prob.question)
            if not norm or norm in seen_norms or prob.problem_id in seen_ids:
                continue
            seen_norms.add(norm)
            seen_ids.add(prob.problem_id)
            unique_list.append(prob)
        cleaned[split_name] = unique_list
    return cleaned


def _make_disjoint_math_problem(split_tag: str, idx: int, rng: random.Random) -> Problem:
    # Disjoint offset bands per split guarantee zero algebraic collision even under normalization.
    band = {
        "heldout_eval": 700,
        "reserved_eval": 500,
        "practice_eval": 300,
        "distill_train": 100,
    }.get(split_tag, 900)

    kind = idx % 3
    pid = f"{split_tag}_{idx:04d}"
    if kind == 0:
        a = band + idx * 3 + rng.randint(1, 3)
        b = 6 + (idx % 9)
        c = 5 + (idx % 13)
        ans = a * b + c
        q = (
            f"Compute the exact value of {a} * {b} + {c}. "
            "Do not show steps in your final reply; output only the final number inside \\boxed{}."
        )
    elif kind == 1:
        x = band + idx * 2 + rng.randint(1, 2)
        k = 3 + (idx % 6)
        b_term = 11 + (idx % 13)
        rhs = k * x + b_term
        ans = x
        q = (
            f"Solve for x: {k}*x + {b_term} = {rhs}. "
            "Do not show steps in your final reply; output only the final number inside \\boxed{}."
        )
    else:
        n = 4 + (idx % 7)
        base_m = band + idx * 3
        total = n * (2 * base_m + (n - 1)) // 2
        ans = total
        q = (
            f"Find the sum of the {n} consecutive integers starting at {base_m}. "
            "Do not show steps in your final reply; output only the final number inside \\boxed{}."
        )

    return Problem(
        problem_id=pid,
        question=q,
        answer=str(ans),
        domain="math",
        problem_type="math",
        metadata={"split": split_tag, "kind": kind},
    )


def _make_transfer_problem(idx: int) -> Problem:
    pid = f"transfer_{idx:04d}"
    kind = idx % 2
    if kind == 0:
        vals = [idx + 3, idx + 7, idx + 11, idx + 19]
        ans = sum(v * v for v in vals if v % 2 == 1)
        q = (
            f"Given the Python list `nums = {vals}`, compute `sum(v*v for v in nums if v % 2 == 1)`. "
            "Do not show steps in your final reply; output only the final number inside \\boxed{}."
        )
    else:
        a = 100 + idx * 5
        b = 4 + (idx % 9)
        q = (
            f"What does the Python expression `({a} // {b}) + ({a} % {b})` evaluate to? "
            "Do not show steps in your final reply; output only the final number inside \\boxed{}."
        )
        ans = (a // b) + (a % b)
    return Problem(
        problem_id=pid,
        question=q,
        answer=str(ans),
        domain="python_eval",
        problem_type="math",
        metadata={"split": "transfer_eval"},
    )


def build_builtin_suites(
    seed: int = 42,
    practice_n: int = 32,
    reserved_n: int = 16,
    heldout_n: int = 32,
    distill_n: int = 32,
    transfer_n: int = 16,
) -> dict[str, list[Problem]]:
    """Build clean offline multi-split math + transfer benchmarks with zero overlap."""
    rng = random.Random(seed)
    raw = {
        "heldout_eval": [_make_disjoint_math_problem("heldout_eval", i, rng) for i in range(heldout_n)],
        "reserved_eval": [_make_disjoint_math_problem("reserved_eval", i, rng) for i in range(reserved_n)],
        "practice_eval": [_make_disjoint_math_problem("practice_eval", i, rng) for i in range(practice_n)],
        "distill_train": [_make_disjoint_math_problem("distill_train", i, rng) for i in range(distill_n)],
        "transfer_eval": [_make_transfer_problem(i) for i in range(transfer_n)],
    }
    return deduplicate_splits(raw)


def save_jsonl_problems(path: Path, problems: list[Problem]) -> Path:
    lines = [json.dumps(p.to_dict(), ensure_ascii=False) for p in problems]
    return atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def load_jsonl_problems(path: Path) -> list[Problem]:
    out: list[Problem] = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        row = json.loads(ln)
        out.append(
            Problem(
                problem_id=str(row["problem_id"]),
                question=str(row["question"]),
                answer=str(row["answer"]),
                domain=str(row.get("domain", "math")),
                problem_type=str(row.get("problem_type", "math")),
                metadata=dict(row.get("metadata") or {}),
            )
        )
    return out


def load_deepscaler_splits(
    dataset_name: str = "agentica-org/DeepScaleR-Preview-Dataset",
    split: str = "train",
    seed: int = 2026,
    practice_n: int = 3000,
    reserved_n: int = 300,
    heldout_n: int = 1000,
    distill_n: int = 2000,
) -> dict[str, list[Problem]]:
    """Load DeepScaleR splits lazily on Colab, excluding evaluation benchmarks & cross-split duplicates."""
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset(dataset_name, split=split)
    pool: list[Problem] = []
    for idx, row in enumerate(ds):
        q = str(row.get("problem") or row.get("question") or "").strip()
        a = str(row.get("answer") or row.get("solution") or "").strip()
        src = str(row.get("source") or row.get("dataset") or "").lower()
        if not q or not a:
            continue
        if any(tok in src for tok in _BENCHMARK_EXCLUDE_TOKENS):
            continue
        if "Please reason step by step" not in q:
            q_prompt = f"{q}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        else:
            q_prompt = q
        pool.append(
            Problem(
                problem_id=f"ds_{idx:06d}",
                question=q_prompt,
                answer=a,
                domain="math",
                problem_type="math",
                metadata={"source": src or dataset_name},
            )
        )

    rng = random.Random(seed)
    rng.shuffle(pool)
    splits: dict[str, list[Problem]] = {
        "heldout_eval": pool[:heldout_n],
        "reserved_eval": pool[heldout_n : heldout_n + reserved_n],
        "practice_eval": pool[heldout_n + reserved_n : heldout_n + reserved_n + practice_n],
        "distill_train": pool[
            heldout_n + reserved_n + practice_n : heldout_n + reserved_n + practice_n + distill_n
        ],
    }
    return deduplicate_splits(splits)
