from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from selfchange.colab.autonomous_engine import evaluate_model_strict_suite
from selfchange.kernel.budgets import get_profile
from selfchange.kernel.integrity import atomic_write_text
from selfchange.training.llamacpp_backend import LlamaCppServerSession, download_llamacpp_prebuilt, find_llama_binary
from selfchange.training.model_loader import estimate_minimind_params, load_model_config, save_model_config


def export_pth_to_hf_qwen3(
    pth_path: Path,
    hf_out_dir: Path,
    tokenizer_dir: Path = Path("model"),
    model_config: dict | None = None,
) -> Path:
    """Convert a native MiniMind .pth checkpoint (including pruned widths) into HuggingFace Qwen3ForCausalLM."""
    import torch  # type: ignore[import-not-found]
    import transformers  # type: ignore[import-not-found]
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM  # type: ignore[import-not-found]

    pth_path = Path(pth_path).resolve()
    hf_out_dir = Path(hf_out_dir).resolve()
    tokenizer_dir = Path(tokenizer_dir).resolve()
    hf_out_dir.mkdir(parents=True, exist_ok=True)

    arch = dict(model_config) if model_config else (load_model_config(pth_path) or estimate_minimind_params(768, 8))
    hs = int(arch.get("hidden_size", 768))
    nl = int(arch.get("num_hidden_layers", 8))
    inter = int(arch.get("intermediate_size", 2432))
    nh = int(arch.get("num_attention_heads", 8))
    nkv = int(arch.get("num_key_value_heads", 4))
    vocab = int(arch.get("vocab_size", 6400))
    if bool(arch.get("use_moe", False)):
        raise ValueError("MoE GGUF export is not supported cleanly for low-end Intel CPU targets; use dense pruned checkpoints.")

    qwen_cfg = Qwen3Config(
        vocab_size=vocab,
        hidden_size=hs,
        intermediate_size=inter,
        num_hidden_layers=nl,
        num_attention_heads=nh,
        num_key_value_heads=nkv,
        head_dim=hs // nh,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        rope_theta=1e6,
        tie_word_embeddings=True,
        use_sliding_window=False,
        sliding_window=None,
    )
    model = Qwen3ForCausalLM(qwen_cfg)
    raw_sd = torch.load(str(pth_path), map_location="cpu")
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}
    model.load_state_dict(clean_sd, strict=True)
    model = model.to(torch.float16)
    model.save_pretrained(str(hf_out_dir), safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    tok.save_pretrained(str(hf_out_dir))

    if int(transformers.__version__.split(".")[0]) >= 5:
        tok_cfg_p = hf_out_dir / "tokenizer_config.json"
        cfg_p = hf_out_dir / "config.json"
        tok_cfg = json.loads(tok_cfg_p.read_text(encoding="utf-8"))
        tok_cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        tok_cfg["extra_special_tokens"] = {}
        atomic_write_text(tok_cfg_p, json.dumps(tok_cfg, indent=2, ensure_ascii=False) + "\n")
        m_cfg = json.loads(cfg_p.read_text(encoding="utf-8"))
        m_cfg["rope_theta"] = 1e6
        m_cfg["rope_scaling"] = None
        m_cfg.pop("rope_parameters", None)
        atomic_write_text(cfg_p, json.dumps(m_cfg, indent=2, ensure_ascii=False) + "\n")

    return hf_out_dir


def _resolve_llamacpp_converter_repo(llama_cpp_repo: Path | None) -> Path:
    if llama_cpp_repo is not None:
        cand = Path(llama_cpp_repo).resolve()
        if (cand / "convert_hf_to_gguf.py").is_file():
            return cand
    repo_root = Path(__file__).resolve().parents[2]
    offline_conv = (repo_root / "selfchange" / "workspace" / "bin" / "llama.cpp" / "converter").resolve()
    if (offline_conv / "convert_hf_to_gguf.py").is_file():
        return offline_conv
    return Path(llama_cpp_repo or Path("/content/llama.cpp")).resolve()


def _resolve_llamacpp_bin_dir(llama_bin_dir: Path | None, fallback_dir: Path) -> Path:
    if llama_bin_dir is not None:
        return Path(llama_bin_dir).resolve()
    found_q = find_llama_binary("llama-quantize")
    if found_q is not None:
        return found_q.parent
    return Path(fallback_dir).resolve()


def _ensure_llamacpp_converter_and_bins(llama_cpp_repo: Path, bin_dir: Path) -> tuple[Path, Path]:
    llama_cpp_repo = _resolve_llamacpp_converter_repo(llama_cpp_repo)
    converter = llama_cpp_repo / "convert_hf_to_gguf.py"
    if not converter.is_file():
        llama_cpp_repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", str(llama_cpp_repo)],
            check=True,
        )
    try:
        __import__("gguf")
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gguf", "pyyaml", "sentencepiece"], check=False)

    quant_bin = find_llama_binary("llama-quantize", search_dirs=[bin_dir]) or find_llama_binary("llama-quantize")
    if quant_bin is None:
        info = download_llamacpp_prebuilt(bin_dir)
        if not info.get("llama_quantize"):
            raise RuntimeError(f"Failed to acquire llama-quantize: {info}")
        quant_bin = Path(info["llama_quantize"])
    return converter, quant_bin.parent


def convert_hf_to_quantized_ggufs(
    hf_dir: Path,
    out_dir: Path,
    stem: str,
    quants: tuple[str, ...] = ("Q8_0", "Q4_K_M"),
    llama_cpp_repo: Path = Path("/content/llama.cpp"),
    llama_bin_dir: Path | None = None,
    keep_f16: bool = False,
) -> dict:
    """Convert HuggingFace Qwen3 dir -> f16 GGUF -> quantized GGUFs (falling back Q4_K_M -> Q4_0 if needed)."""
    hf_dir = Path(hf_dir).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    llama_cpp_repo = _resolve_llamacpp_converter_repo(llama_cpp_repo)
    eff_bin_dir = _resolve_llamacpp_bin_dir(llama_bin_dir, out_dir / "_llama_bin")

    converter, actual_bin_dir = _ensure_llamacpp_converter_and_bins(llama_cpp_repo, eff_bin_dir)
    quant_bin = find_llama_binary("llama-quantize", search_dirs=[actual_bin_dir])
    assert quant_bin is not None

    env = os.environ.copy()
    gguf_py = Path(llama_cpp_repo).resolve() / "gguf-py"
    if gguf_py.is_dir():
        env["PYTHONPATH"] = str(gguf_py) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    lib_dir = str(quant_bin.parent)
    env["LD_LIBRARY_PATH"] = lib_dir + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["DYLD_LIBRARY_PATH"] = lib_dir + (os.pathsep + env["DYLD_LIBRARY_PATH"] if env.get("DYLD_LIBRARY_PATH") else "")

    f16_path = out_dir / f"{stem}-f16.gguf"
    wrapper_code = f"""
import runpy, sys
from pathlib import Path
repo_dir = Path({str(Path(llama_cpp_repo).resolve())!r})
sys.path.insert(0, str(repo_dir))
gguf_py_dir = repo_dir / "gguf-py"
if gguf_py_dir.is_dir():
    sys.path.insert(0, str(gguf_py_dir))
import conversion.base as conv_base
for cls_name in ("TextModel", "ModelBase"):
    cls = getattr(conv_base, cls_name, None)
    if cls is not None and hasattr(cls, "get_vocab_base_pre"):
        _orig = cls.get_vocab_base_pre
        def _patched(self, tokenizer, __orig=_orig):
            try:
                return __orig(self, tokenizer)
            except NotImplementedError:
                return "gpt-2"
        cls.get_vocab_base_pre = _patched
sys.argv = [{str(converter)!r}, {str(hf_dir)!r}, "--outfile", {str(f16_path)!r}, "--outtype", "f16"]
runpy.run_path({str(converter)!r}, run_name="__main__")
"""
    conv_proc = subprocess.run(
        [sys.executable, "-c", wrapper_code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if conv_proc.returncode != 0:
        raise RuntimeError(f"convert_hf_to_gguf.py failed (rc={conv_proc.returncode}):\n{(conv_proc.stdout or '')[-2500:]}")

    artifacts: list[dict] = []
    for quant in quants:
        q_norm = quant.strip().upper()
        q_out = out_dir / f"{stem}-{q_norm.lower()}.gguf"
        proc = subprocess.run([str(quant_bin), str(f16_path), str(q_out), q_norm], env=env, capture_output=True, text=True)
        used_quant = q_norm
        fallback_from: str | None = None
        if proc.returncode != 0 and q_norm == "Q4_K_M":
            # Pruned hidden/intermediate widths (e.g. 1216) may fail K-quant 256-superblock alignment; Q4_0 uses 32-element blocks.
            fallback_from = q_norm
            used_quant = "Q4_0"
            q_out = out_dir / f"{stem}-q4_0.gguf"
            proc = subprocess.run([str(quant_bin), str(f16_path), str(q_out), "Q4_0"], env=env, capture_output=True, text=True)
        if proc.returncode != 0 or not q_out.is_file():
            raise RuntimeError(f"llama-quantize failed for {q_norm}: {proc.stderr[-1000:]}")
        artifacts.append(
            {
                "quant": used_quant,
                "fallback_from": fallback_from,
                "path": str(q_out),
                "size_mb": round(q_out.stat().st_size / 1e6, 2),
            }
        )

    if not keep_f16:
        f16_path.unlink(missing_ok=True)
    return {"bin_dir": str(actual_bin_dir), "artifacts": artifacts}


def _check_tokenizer_parity(sess: LlamaCppServerSession, tokenizer_dir: Path) -> dict:
    sample = (
        "<|im_start|>system\nOperate safely.<|im_end|>\n"
        "<|im_start|>user\nfault=nan_loss lr=0.0002<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n"
    )
    try:
        from transformers import AutoTokenizer  # type: ignore[import-not-found]

        hf_tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        hf_ids = [int(x) for x in hf_tok.encode(sample, add_special_tokens=False)]
        req = urllib.request.Request(
            f"{sess.base_url}/tokenize",
            data=json.dumps({"content": sample, "add_special": False, "parse_special": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            tok_data = json.loads(resp.read().decode("utf-8"))
        gguf_ids = [int(x["id"] if isinstance(x, dict) else x) for x in tok_data.get("tokens", [])]
        return {
            "exact_match": hf_ids == gguf_ids,
            "hf_len": len(hf_ids),
            "gguf_len": len(gguf_ids),
            "hf_head": hf_ids[:12],
            "gguf_head": gguf_ids[:12],
        }
    except Exception as exc:
        return {"exact_match": False, "error": repr(exc)}


def export_and_verify_gguf(
    pth_path: Path,
    out_dir: Path,
    tokenizer_dir: Path = Path("model"),
    quants: tuple[str, ...] = ("Q8_0", "Q4_K_M"),
    verify: bool = True,
    max_tasks: int = 20,
    llama_cpp_repo: Path = Path("/content/llama.cpp"),
    llama_bin_dir: Path | None = None,
) -> dict:
    """Export MiniMind .pth -> HF Qwen3 -> quantized GGUFs and verify each GGUF on STRICT_EVAL_SUITE."""
    pth_path = Path(pth_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pth_path.stem
    hf_dir = out_dir / f"_hf_{stem}"
    if hf_dir.exists():
        shutil.rmtree(hf_dir)

    export_pth_to_hf_qwen3(pth_path, hf_dir, tokenizer_dir=tokenizer_dir)
    q_info = convert_hf_to_quantized_ggufs(
        hf_dir=hf_dir,
        out_dir=out_dir,
        stem=stem,
        quants=quants,
        llama_cpp_repo=llama_cpp_repo,
        llama_bin_dir=llama_bin_dir,
    )
    shutil.rmtree(hf_dir, ignore_errors=True)

    prof = get_profile("local_intel_cpu")
    bin_dir = Path(q_info["bin_dir"])
    entries: list[dict] = []
    for item in q_info["artifacts"]:
        rec = dict(item)
        if verify:
            with LlamaCppServerSession(Path(item["path"]), ctx_size=1024, bin_dir=bin_dir) as sess:
                rec["tokenizer_parity"] = _check_tokenizer_parity(sess, Path(tokenizer_dir).resolve())
                card = evaluate_model_strict_suite(
                    None,
                    None,
                    prof,
                    max_tasks=max_tasks,
                    k_rollouts=1,
                    reply_fn=lambda msgs, tls, temp: sess.generate_reply(msgs, tools=tls, temperature=temp, max_new_tokens=160),
                )
            rec["strict_eval"] = {
                "passed_count": card["passed_count"],
                "total_count": card["total_count"],
                "pass_at_1": card["pass_at_1"],
                "eos_rate": card["eos_rate"],
                "overall_score": card["overall_score"],
                "tool_pass_at_1": card["tool_pass_at_1"],
                "code_pass_at_1": card["code_pass_at_1"],
                "math_pass_at_1": card["math_pass_at_1"],
                "noop_pass_rate": card["noop_pass_rate"],
                "failing_task_ids": card["failing_task_ids"],
                "mean_tok_per_sec": sess.mean_tok_per_sec(),
            }
        entries.append(rec)

    best_entry = entries[0] if entries else None
    if verify and entries:
        ref = max(entries, key=lambda e: (e["strict_eval"]["overall_score"], e["strict_eval"]["passed_count"]))
        ref_score = float(ref["strict_eval"]["overall_score"])
        ref_pass = int(ref["strict_eval"]["passed_count"])
        passing = [
            e
            for e in entries
            if float(e["strict_eval"]["overall_score"]) >= ref_score - 0.03
            and int(e["strict_eval"]["passed_count"]) >= ref_pass - 1
        ]
        best_entry = min(passing or [ref], key=lambda e: float(e["size_mb"]))

    report = {
        "source_pth": str(pth_path),
        "model_config": load_model_config(pth_path),
        "artifacts": entries,
        "best_gguf": best_entry,
    }
    atomic_write_text(out_dir / f"{stem}_gguf_report.json", json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def export_pth_to_single_gguf(
    pth_path: Path,
    out_gguf_path: Path,
    tokenizer_dir: Path = Path("model"),
    quant: str = "Q4_K_M",
    model_config: dict | None = None,
    llama_cpp_repo: Path | None = None,
    llama_bin_dir: Path | None = None,
) -> Path:
    """Export one MiniMind .pth checkpoint to a single quantized GGUF with sibling .config.json."""
    pth_path = Path(pth_path).resolve()
    out_gguf_path = Path(out_gguf_path).resolve()
    out_gguf_path.parent.mkdir(parents=True, exist_ok=True)
    arch = dict(model_config) if model_config else (load_model_config(pth_path) or None)

    tmp_root = Path(tempfile.mkdtemp(prefix=f"gguf_{out_gguf_path.stem}_", dir=str(out_gguf_path.parent)))
    try:
        hf_dir = tmp_root / "hf"
        gguf_dir = tmp_root / "gguf"
        export_pth_to_hf_qwen3(pth_path, hf_dir, tokenizer_dir=tokenizer_dir, model_config=arch)
        conv_repo = _resolve_llamacpp_converter_repo(llama_cpp_repo)
        eff_bin_dir = _resolve_llamacpp_bin_dir(llama_bin_dir, out_gguf_path.parent / "_llama_bin")
        q_info = convert_hf_to_quantized_ggufs(
            hf_dir=hf_dir,
            out_dir=gguf_dir,
            stem=out_gguf_path.stem,
            quants=(quant,),
            llama_cpp_repo=conv_repo,
            llama_bin_dir=eff_bin_dir,
            keep_f16=False,
        )
        artifacts = q_info.get("artifacts") or []
        if not artifacts:
            raise RuntimeError(f"No GGUF artifacts generated from {pth_path}")
        produced = Path(artifacts[0]["path"]).resolve()
        tmp_gguf = out_gguf_path.with_name(f".{out_gguf_path.name}.{os.getpid()}.tmp")
        try:
            shutil.copy2(produced, tmp_gguf)
            os.replace(tmp_gguf, out_gguf_path)
        finally:
            if tmp_gguf.exists():
                tmp_gguf.unlink(missing_ok=True)
        if arch:
            save_model_config(out_gguf_path.with_suffix(".config.json"), arch)
        return out_gguf_path
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export MiniMind .pth weights to verified GGUF via Qwen3 + llama.cpp")
    parser.add_argument("--pth", action="append", required=True, help="Checkpoint .pth path (repeat or comma-separate)")
    parser.add_argument("--out-dir", type=Path, default=Path("selfchange/workspace/promoted_models/gguf"))
    parser.add_argument("--tokenizer-dir", type=Path, default=Path("model"))
    parser.add_argument("--quants", type=str, default="Q8_0,Q4_K_M")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-tasks", type=int, default=20)
    parser.add_argument("--llama-cpp-repo", type=Path, default=Path("/content/llama.cpp"))
    parser.add_argument("--llama-bin-dir", type=Path, default=None)
    args = parser.parse_args()

    pth_list: list[Path] = []
    for raw in args.pth:
        for piece in str(raw).split(","):
            if piece.strip():
                pth_list.append(Path(piece.strip()))

    quants = tuple(q.strip() for q in args.quants.split(",") if q.strip())
    reports: list[dict] = []
    for p in pth_list:
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        reports.append(
            export_and_verify_gguf(
                pth_path=p,
                out_dir=args.out_dir,
                tokenizer_dir=args.tokenizer_dir,
                quants=quants,
                verify=args.verify,
                max_tasks=args.max_tasks,
                llama_cpp_repo=args.llama_cpp_repo,
                llama_bin_dir=args.llama_bin_dir,
            )
        )

    best_overall = None
    candidates = [r["best_gguf"] for r in reports if r.get("best_gguf")]
    if candidates:
        if args.verify and all("strict_eval" in c for c in candidates):
            max_sc = max(float(c["strict_eval"]["overall_score"]) for c in candidates)
            viable = [c for c in candidates if float(c["strict_eval"]["overall_score"]) >= max_sc - 0.04]
            best_overall = min(viable or candidates, key=lambda c: float(c["size_mb"]))
        else:
            best_overall = min(candidates, key=lambda c: float(c["size_mb"]))

        champ_link = Path(args.out_dir).resolve() / "champion.gguf"
        if Path(best_overall["path"]).resolve() != champ_link:
            shutil.copy2(best_overall["path"], champ_link)
        summary = {
            "champion_gguf": str(champ_link),
            "selected_artifact": best_overall,
            "reports": reports,
        }
        atomic_write_text(
            Path(args.out_dir).resolve() / "latest_gguf.json",
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        )
    else:
        summary = {"champion_gguf": None, "reports": reports}

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
