from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from selfrewrite.colab.colab_launcher import _evaluate_on_split, run_reproduce_and_harvest_pipeline
from selfrewrite.config import SessionConfig, get_profile
from selfrewrite.edit_store import EditStore
from selfrewrite.harvest.audit import audit_distill_bundle
from selfrewrite.harvest.build_bundle import build_distill_bundle
from selfrewrite.stats import score_emergence_ladder
from selfrewrite.tasks.corruption import split_train_eval_layers
from selfrewrite.tasks.datasets import build_builtin_suites, load_jsonl_problems


def _cmd_smoke(args: argparse.Namespace) -> int:
    out = run_reproduce_and_harvest_pipeline(
        workspace_dir=Path(args.work_root),
        mode="smoke",
        backend_kind="mock",
        model_id="mock-transformer",
        profile_name="local_cpu_smoke",
        drive_dir=Path(args.work_root) / "mock_drive_mirror",
        num_reproduce_improve=1,
        num_harvest_improve=2,
        num_harvest_repair=2,
        seed=args.seed,
    )
    print(json.dumps(out, indent=2))
    audit_ok = bool(out.get("bundle", {}).get("audit", {}).get("passed", False))
    return 0 if audit_ok else 2


def _cmd_ladder(args: argparse.Namespace) -> int:
    raw_root = Path(args.raw_root)
    rows = []
    for sdir in sorted(raw_root.iterdir()):
        rfile = sdir / "session_result.json"
        if not rfile.is_file():
            continue
        res = json.loads(rfile.read_text(encoding="utf-8"))
        eval_file = sdir / "eval_result.json"
        eval_res = json.loads(eval_file.read_text(encoding="utf-8")) if eval_file.is_file() else None
        ladder = score_emergence_ladder(res, heldout_eval=eval_res)
        rows.append({"session_id": res.get("session_id", sdir.name), **ladder})
    print(json.dumps(rows, indent=2))
    return 0


def _cmd_audit_bundle(args: argparse.Namespace) -> int:
    splits_dir = Path(args.splits_dir)
    eval_texts: list[str] = []
    for sname in ("heldout_eval.jsonl", "reserved_eval.jsonl", "practice_eval.jsonl"):
        p = splits_dir / sname
        if p.is_file():
            eval_texts.extend(item.question for item in load_jsonl_problems(p))
    if not eval_texts:
        suites = build_builtin_suites()
        eval_texts = [item.question for k in ("heldout_eval", "reserved_eval", "practice_eval") for item in suites[k]]
    _, eval_layers = split_train_eval_layers(int(args.num_layers))
    report = audit_distill_bundle(
        bundle_dir=Path(args.bundle_dir),
        forbidden_eval_texts=eval_texts,
        forbidden_eval_layers=eval_layers,
    )
    print(json.dumps(report, indent=2))
    return 0 if report.get("passed") else 2


def _cmd_build_bundle(args: argparse.Namespace) -> int:
    splits_dir = Path(args.splits_dir)
    eval_texts: list[str] = []
    for sname in ("heldout_eval.jsonl", "reserved_eval.jsonl", "practice_eval.jsonl"):
        p = splits_dir / sname
        if p.is_file():
            eval_texts.extend(item.question for item in load_jsonl_problems(p))
    _, eval_layers = split_train_eval_layers(int(args.num_layers))
    res = build_distill_bundle(
        raw_sessions_dir=Path(args.raw_root),
        output_bundle_dir=Path(args.bundle_dir),
        forbidden_eval_texts=eval_texts,
        forbidden_eval_layers=eval_layers,
    )
    print(json.dumps(res, indent=2))
    return 0 if res.get("audit", {}).get("passed") else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="selfrewrite.cli", description="Self-Rewriting CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_smoke = sub.add_parser("smoke", help="Run zero-download E2E smoke test + harvest + audit")
    p_smoke.add_argument("--work-root", default="/tmp/selfrewrite_smoke")
    p_smoke.add_argument("--seed", type=int, default=2026)
    p_smoke.set_defaults(func=_cmd_smoke)

    for sub_name, default_mode in (
        ("reproduce", "reproduce"),
        ("harvest", "harvest"),
        ("reproduce-and-harvest", "reproduce-and-harvest"),
    ):
        p = sub.add_parser(sub_name, help=f"Run Colab/local {default_mode} pipeline")
        p.add_argument("--workspace-dir", default="selfrewrite/workspace")
        p.add_argument("--backend", choices=("hf", "mock"), default="hf")
        p.add_argument("--model-id", default="Qwen/Qwen3.5-27B")
        p.add_argument("--profile", default="auto")
        p.add_argument("--drive-dir", default=None)
        p.add_argument("--num-reproduce-improve", type=int, default=2)
        p.add_argument("--num-harvest-improve", type=int, default=4)
        p.add_argument("--num-harvest-repair", type=int, default=4)
        p.add_argument("--programmatic-teacher", action="store_true")
        p.add_argument("--seed", type=int, default=2026)

        def _make_runner(m: str):
            def _fn(a: argparse.Namespace) -> int:
                out = run_reproduce_and_harvest_pipeline(
                    workspace_dir=Path(a.workspace_dir),
                    mode=m,
                    backend_kind=a.backend,
                    model_id=a.model_id,
                    profile_name=None if a.profile == "auto" else a.profile,
                    drive_dir=Path(a.drive_dir) if a.drive_dir else None,
                    num_reproduce_improve=a.num_reproduce_improve,
                    num_harvest_improve=a.num_harvest_improve,
                    num_harvest_repair=a.num_harvest_repair,
                    use_programmatic_teacher=bool(a.programmatic_teacher),
                    seed=a.seed,
                )
                print(json.dumps(out, indent=2))
                return 0
            return _fn

        p.set_defaults(func=_make_runner(default_mode))

    p_lad = sub.add_parser("ladder", help="Score 6-step emergence ladder across raw_sessions")
    p_lad.add_argument("--raw-root", default="selfrewrite/workspace/raw_sessions")
    p_lad.set_defaults(func=_cmd_ladder)

    p_aud = sub.add_parser("audit-bundle", help="Run anti-leakage audit on a distillation bundle")
    p_aud.add_argument("--bundle-dir", required=True)
    p_aud.add_argument("--splits-dir", default="selfrewrite/workspace/splits")
    p_aud.add_argument("--num-layers", type=int, default=64)
    p_aud.set_defaults(func=_cmd_audit_bundle)

    p_bld = sub.add_parser("build-bundle", help="Compile raw_sessions into a distillation bundle")
    p_bld.add_argument("--raw-root", default="selfrewrite/workspace/raw_sessions")
    p_bld.add_argument("--bundle-dir", default="selfrewrite/workspace/distill_bundle/v1")
    p_bld.add_argument("--splits-dir", default="selfrewrite/workspace/splits")
    p_bld.add_argument("--num-layers", type=int, default=64)
    p_bld.set_defaults(func=_cmd_build_bundle)

    def _cmd_eval(a: argparse.Namespace) -> int:
        from selfrewrite.backends.mock_backend import MockTargetBackend, MockTransformerModel

        suites = build_builtin_suites(heldout_n=24)
        heldout = load_jsonl_problems(Path(a.heldout_path)) if a.heldout_path and Path(a.heldout_path).is_file() else suites["heldout_eval"]
        profile = get_profile(None if a.profile == "auto" else a.profile)
        if a.backend == "mock":
            target = MockTargetBackend(MockTransformerModel())
        else:
            from selfrewrite.backends.hf_gpu_backend import HFGPUBackend

            target = HFGPUBackend.from_pretrained(a.model_id, profile=profile)
        store = EditStore(Path(a.work_dir), lambda _: None, target.tensor_shapes())
        tensors = store.load_edit_tensors(target_torch=(a.backend != "mock"))
        cfg = SessionConfig(answer_max_tokens=profile.answer_max_tokens)
        res = _evaluate_on_split(target, tensors or None, heldout, cfg, seed_base=a.seed_base, n_samples=a.n_samples)
        print(json.dumps(res, indent=2))
        return 0

    p_ev = sub.add_parser("eval", help="Run fixed-in-advance held-out evaluation on WORK/edited")
    p_ev.add_argument("--work-dir", required=True)
    p_ev.add_argument("--backend", choices=("hf", "mock"), default="hf")
    p_ev.add_argument("--model-id", default="Qwen/Qwen3.5-27B")
    p_ev.add_argument("--profile", default="auto")
    p_ev.add_argument("--heldout-path", default="")
    p_ev.add_argument("--seed-base", type=int, default=9000)
    p_ev.add_argument("--n-samples", type=int, default=1)
    p_ev.set_defaults(func=_cmd_eval)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
