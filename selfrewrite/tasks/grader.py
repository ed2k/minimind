from __future__ import annotations

import math
import re
import zlib
from fractions import Fraction
from typing import Any

_BOX_MARKER_RE = re.compile(r"\\(?:boxed|fbox)\s*\{")
_FRAC_RE = re.compile(r"^\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}$")


def has_closed_thinking(text: str) -> bool:
    """Return True when thinking closes with </think> before answer output."""
    raw = text or ""
    if "</think>" not in raw:
        return False
    _, after = raw.rsplit("</think>", 1)
    return bool(after.strip())


def extract_last_boxed(text: str, after_think: bool = True) -> str | None:
    """Extract the content of the last balanced \\boxed{...} expression."""
    raw = text or ""
    if after_think:
        if "</think>" not in raw:
            return None
        raw = raw.rsplit("</think>", 1)[1]

    last_content: str | None = None
    pos = 0
    while True:
        m = _BOX_MARKER_RE.search(raw, pos)
        if not m:
            break
        start = m.end()
        depth = 1
        i = start
        while i < len(raw) and depth > 0:
            ch = raw[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        if depth == 0:
            last_content = raw[start : i - 1].strip()
        pos = i
    return last_content


def _normalize_expr_text(expr: str) -> str:
    s = str(expr or "").strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    s = s.replace("\\left", "").replace("\\right", "").replace("\\,", "")
    s = s.replace(",", "").replace("$", "").strip()
    s = re.sub(r"^\((.+)\)$", r"\1", s).strip()
    m = _FRAC_RE.match(s.replace(" ", ""))
    if m:
        s = f"{m.group(1)}/{m.group(2)}"
    return s


def _to_number(expr: str) -> Fraction | float | None:
    s = _normalize_expr_text(expr)
    if not s:
        return None
    if s.endswith("%"):
        inner = _to_number(s[:-1])
        if inner is not None:
            return inner / 100
    try:
        if "/" in s and s.count("/") == 1:
            a_str, b_str = s.split("/", 1)
            return Fraction(a_str.strip()) / Fraction(b_str.strip())
        if re.fullmatch(r"[+-]?\d+", s):
            return Fraction(int(s), 1)
        return float(s)
    except Exception:
        return None


def answers_equivalent(pred: str | None, ref: str | None, problem_type: str = "math") -> bool:
    """Compare predicted final answer against ground-truth reference."""
    if pred is None or ref is None:
        return False
    p_norm = _normalize_expr_text(pred)
    r_norm = _normalize_expr_text(ref)
    if p_norm.lower() == r_norm.lower():
        return True

    if problem_type != "math":
        return re.sub(r"\s+", " ", p_norm).strip().lower() == re.sub(r"\s+", " ", r_norm).strip().lower()

    p_num = _to_number(p_norm)
    r_num = _to_number(r_norm)
    if p_num is not None and r_num is not None:
        if isinstance(p_num, Fraction) and isinstance(r_num, Fraction):
            return p_num == r_num
        return math.isclose(float(p_num), float(r_num), rel_tol=1e-6, abs_tol=1e-9)

    try:
        import sympy  # type: ignore[import-not-found]
        from sympy.parsing.latex import parse_latex  # type: ignore[import-not-found]

        diff = sympy.simplify(parse_latex(p_norm) - parse_latex(r_norm))
        if diff == 0:
            return True
    except Exception:
        pass
    return False


def is_looping_answer(
    text: str,
    cut_at_limit: bool,
    tail_chars: int = 6000,
    ratio_threshold: float = 0.15,
) -> bool:
    """Return True when a truncated answer compresses below 15% on its trailing characters."""
    if not cut_at_limit or not text:
        return False
    tail = text[-max(64, int(tail_chars)) :]
    raw_bytes = tail.encode("utf-8", errors="ignore")
    if len(raw_bytes) < 64:
        return False
    comp_bytes = zlib.compress(raw_bytes)
    return (len(comp_bytes) / len(raw_bytes)) < float(ratio_threshold)


def grade_response(
    problem: dict[str, Any] | Any,
    response_text: str,
    num_tokens: int,
    token_budget: int,
    require_thinking: bool = True,
    loop_tail_chars: int = 6000,
    loop_compress_ratio: float = 0.15,
) -> dict[str, Any]:
    """Grade one generated response per Algorithm 3."""
    p_answer = problem["answer"] if isinstance(problem, dict) else getattr(problem, "answer")
    p_type = (
        problem.get("problem_type", "math")
        if isinstance(problem, dict)
        else getattr(problem, "problem_type", "math")
    )
    cut_at_limit = int(num_tokens) >= int(token_budget)
    closed_think = has_closed_thinking(response_text) if require_thinking else True
    extracted = extract_last_boxed(response_text, after_think=require_thinking)
    if extracted is None and not require_thinking:
        lines = [ln.strip() for ln in (response_text or "").splitlines() if ln.strip()]
        extracted = lines[-1] if lines else None
    correct = bool(closed_think and not cut_at_limit and answers_equivalent(extracted, str(p_answer), problem_type=p_type))
    looping = is_looping_answer(
        response_text,
        cut_at_limit=cut_at_limit,
        tail_chars=loop_tail_chars,
        ratio_threshold=loop_compress_ratio,
    )
    return {
        "correct": correct,
        "cut_at_limit": cut_at_limit,
        "closed_think": closed_think,
        "looping": looping,
        "extracted_answer": extracted,
        "num_tokens": int(num_tokens),
    }
