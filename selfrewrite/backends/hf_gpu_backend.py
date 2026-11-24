from __future__ import annotations

import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from selfrewrite.backends.base import EditorBackend, GenerationRecord, ReadResult, TargetBackend
from selfrewrite.config import ComputeProfile, SessionConfig, atomic_write_text
from selfrewrite.tasks.datasets import problem_seed


class CanonicalWeightReader:
    """Reads canonical full-precision/bf16 tensors lazily without duplicating 27B models in VRAM."""

    def __init__(
        self,
        weight_dir: Path | None = None,
        state_dict: dict[str, Any] | None = None,
        model: Any | None = None,
    ):
        self.weight_dir = Path(weight_dir).resolve() if weight_dir else None
        self._state_dict = state_dict
        self._model = model
        self._index_map: dict[str, str] | None = None
        self._shapes: dict[str, tuple[int, ...]] | None = None
        self._init_index()

    def _init_index(self) -> None:
        if self._state_dict is not None:
            self._shapes = {k: tuple(int(d) for d in v.shape) for k, v in self._state_dict.items()}
            return
        if self.weight_dir is not None:
            idx_file = self.weight_dir / "model.safetensors.index.json"
            single_file = self.weight_dir / "model.safetensors"
            if idx_file.is_file():
                meta = json.loads(idx_file.read_text(encoding="utf-8"))
                self._index_map = dict(meta.get("weight_map", {}))
            elif single_file.is_file():
                from safetensors import safe_open  # type: ignore[import-not-found]

                with safe_open(str(single_file), framework="pt", device="cpu") as sf:
                    self._index_map = {k: single_file.name for k in sf.keys()}
            if self._index_map:
                from safetensors import safe_open  # type: ignore[import-not-found]

                shapes: dict[str, tuple[int, ...]] = {}
                by_shard: dict[str, list[str]] = {}
                for k, rel in self._index_map.items():
                    by_shard.setdefault(rel, []).append(k)
                for rel, keys in by_shard.items():
                    with safe_open(str(self.weight_dir / rel), framework="pt", device="cpu") as sf:
                        for k in keys:
                            shapes[k] = tuple(int(d) for d in sf.get_slice(k).get_shape())
                self._shapes = shapes
                return
        if self._model is not None:
            self._shapes = {
                name: tuple(int(d) for d in param.shape)
                for name, param in self._model.named_parameters()
            }

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        return dict(self._shapes or {})

    def load_tensor(self, name: str) -> Any:
        if self._state_dict is not None:
            return self._state_dict[name].detach().cpu().clone()
        if self.weight_dir is not None and self._index_map is not None and name in self._index_map:
            from safetensors import safe_open  # type: ignore[import-not-found]

            shard = self.weight_dir / self._index_map[name]
            with safe_open(str(shard), framework="pt", device="cpu") as sf:
                return sf.get_tensor(name).clone()
        if self._model is not None:
            params = dict(self._model.named_parameters())
            return params[name].detach().cpu().clone()
        raise KeyError(f"Canonical tensor {name!r} unavailable")

    def export_weights_spec(self, dest_dir: Path) -> Path:
        dest_dir = Path(dest_dir).resolve()
        dest_dir.mkdir(parents=True, exist_ok=True)
        spec_path = dest_dir / "weights_spec.json"
        if self.weight_dir is not None and self._index_map is not None:
            spec = {
                "mode": "safetensors_dir",
                "base_dir": str(self.weight_dir),
                "weight_map": self._index_map,
                "shapes": {k: list(v) for k, v in self.tensor_shapes().items()},
            }
            atomic_write_text(spec_path, json.dumps(spec, indent=2))
            return spec_path

        # Materialize in-memory state dict to safetensors shard on disk cleanly for sandbox worker.
        from safetensors.torch import save_file  # type: ignore[import-not-found]

        tensors: dict[str, Any] = {}
        for name in self.tensor_shapes():
            tensors[name] = self.load_tensor(name)
        shard = dest_dir / "canonical_weights.safetensors"
        tmp = shard.with_name(f".{shard.name}.tmp")
        save_file(tensors, str(tmp))
        tmp.replace(shard)
        spec = {
            "mode": "safetensors_dir",
            "base_dir": str(dest_dir),
            "weight_map": {k: shard.name for k in tensors},
            "shapes": {k: list(v) for k, v in self.tensor_shapes().items()},
        }
        atomic_write_text(spec_path, json.dumps(spec, indent=2))
        return spec_path


def _resolve_parent_and_attr(root: Any, dotted_path: str) -> tuple[Any, str]:
    parts = dotted_path.split(".")
    cur = root
    for p in parts[:-1]:
        if p.isdigit() and hasattr(cur, "__getitem__"):
            cur = cur[int(p)]
        else:
            cur = getattr(cur, p)
    return cur, parts[-1]


def _is_quantized_linear(mod: Any) -> bool:
    cls_name = type(mod).__name__.lower()
    if "4bit" in cls_name or "8bit" in cls_name:
        return True
    w = getattr(mod, "weight", None)
    return w is not None and hasattr(w, "quant_state")


class HybridWeightOverlay:
    """Zero-reload pinned-CPU backup & GPU tensor/module swapper supporting BF16 & 4-bit hybrid."""

    def __init__(self, model: Any):
        self.model = model
        self._param_backups: dict[str, Any] = {}
        self._module_backups: dict[tuple[Any, str], Any] = {}

    def apply(self, edit_tensors: dict[str, Any] | None) -> None:
        if not edit_tensors:
            return
        import torch  # type: ignore[import-not-found]
        from torch import nn  # type: ignore[import-not-found]

        named_params = dict(self.model.named_parameters())
        named_mods = dict(self.model.named_modules())

        with torch.no_grad():
            for name, candidate in edit_tensors.items():
                if name.endswith(".weight"):
                    mod_path = name[: -len(".weight")]
                    mod = named_mods.get(mod_path)
                    if mod is not None and _is_quantized_linear(mod) and "." in mod_path:
                        parent_path, child_attr = mod_path.rsplit(".", 1)
                        parent_mod = named_mods[parent_path]
                        key = (parent_mod, child_attr)
                        if key not in self._module_backups:
                            self._module_backups[key] = mod
                        dev = mod.weight.device
                        if dev.type == "cuda":
                            promo_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                        else:
                            promo_dtype = torch.float32
                        bias = getattr(mod, "bias", None)
                        cand_t = candidate if isinstance(candidate, torch.Tensor) else torch.tensor(candidate)
                        new_lin = nn.Linear(
                            mod.in_features,
                            mod.out_features,
                            bias=(bias is not None),
                            dtype=promo_dtype,
                            device=dev,
                        )
                        new_lin.weight.copy_(cand_t.to(device=dev, dtype=promo_dtype))
                        if bias is not None and new_lin.bias is not None:
                            new_lin.bias.copy_(bias.detach().to(device=dev, dtype=promo_dtype))
                        setattr(parent_mod, child_attr, new_lin)
                        continue

                if name not in named_params:
                    raise KeyError(f"Edited parameter {name!r} not found in target model")
                param = named_params[name]
                if name not in self._param_backups:
                    cpu_copy = param.detach().cpu().clone()
                    if torch.cuda.is_available():
                        try:
                            cpu_copy = cpu_copy.pin_memory()
                        except Exception:
                            pass
                    self._param_backups[name] = cpu_copy
                cand_t = candidate if isinstance(candidate, torch.Tensor) else torch.tensor(candidate)
                param.data.copy_(cand_t.to(device=param.device, dtype=param.dtype))

    def restore(self) -> None:
        if not self._param_backups and not self._module_backups:
            return
        import torch  # type: ignore[import-not-found]

        had_module_backups = bool(self._module_backups)
        with torch.no_grad():
            for (parent_mod, child_attr), orig_mod in list(self._module_backups.items()):
                setattr(parent_mod, child_attr, orig_mod)
            self._module_backups.clear()

            if self._param_backups:
                named_params = dict(self.model.named_parameters())
                for name, orig_cpu in list(self._param_backups.items()):
                    if name in named_params:
                        p = named_params[name]
                        p.data.copy_(orig_cpu.to(device=p.device, dtype=p.dtype, non_blocking=False))
                self._param_backups.clear()

        if had_module_backups and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @contextmanager
    def applied(self, edit_tensors: dict[str, Any] | None) -> Iterator[None]:
        self.apply(edit_tensors)
        try:
            yield
        finally:
            self.restore()


def load_hf_model(model_id_or_dir: str, **load_kwargs: Any) -> Any:
    """Load causal or conditional-generation HuggingFace model (including Qwen3.5) cleanly."""
    from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore[import-not-found]

    trust_remote = bool(load_kwargs.get("trust_remote_code", True))
    cfg = AutoConfig.from_pretrained(str(model_id_or_dir), trust_remote_code=trust_remote)
    model_type = str(getattr(cfg, "model_type", "") or "").lower()
    archs = [str(a) for a in (getattr(cfg, "architectures", None) or [])]
    loader_cls: Any = AutoModelForCausalLM
    if model_type.startswith("qwen3_5") or any(a.endswith("ForConditionalGeneration") for a in archs):
        try:
            from transformers import AutoModelForImageTextToText  # type: ignore[import-not-found]

            loader_cls = AutoModelForImageTextToText
        except ImportError:
            loader_cls = AutoModelForCausalLM

    use_cuda = "cuda" in str(load_kwargs.get("device_map", "")).lower()
    if not use_cuda:
        try:
            import torch  # type: ignore[import-not-found]

            use_cuda = bool(torch.cuda.is_available())
        except Exception:
            use_cuda = False

    def _call_loader(cls: Any, kwargs: dict[str, Any]) -> Any:
        if use_cuda and "attn_implementation" not in kwargs:
            try:
                return cls.from_pretrained(str(model_id_or_dir), **{**kwargs, "attn_implementation": "sdpa"})
            except (TypeError, ValueError):
                pass
        return cls.from_pretrained(str(model_id_or_dir), **kwargs)

    if loader_cls is not AutoModelForCausalLM:
        try:
            return _call_loader(loader_cls, dict(load_kwargs))
        except (ImportError, ValueError):
            pass
    return _call_loader(AutoModelForCausalLM, dict(load_kwargs))


def resolve_decoder_layers(model: Any) -> list[Any]:
    """Find transformer decoder ModuleList across Qwen3.5, Qwen3, Llama, GPT, and MiniMind."""
    for attr_chain in (
        ("model", "language_model", "model", "layers"),
        ("language_model", "model", "layers"),
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
        ("layers",),
        ("transformer", "h"),
    ):
        cur = model
        ok = True
        for attr in attr_chain:
            if hasattr(cur, attr):
                cur = getattr(cur, attr)
            else:
                ok = False
                break
        if ok and hasattr(cur, "__len__") and len(cur) > 0:
            return list(cur)
    raise RuntimeError(f"Could not locate decoder layers on model {type(model).__name__}")


def _pool_sequence_tensor(seq_tensor: Any, start_idx: int, pooling: str, k: int) -> Any:
    """Slice/pool [T, D] activations along sequence positions per Algorithm 2 read pass."""
    import torch  # type: ignore[import-not-found]

    sl = seq_tensor[start_idx:] if seq_tensor.shape[0] > start_idx else seq_tensor
    if sl.shape[0] == 0:
        return seq_tensor[:1].mean(dim=0)
    if pooling == "mean_answer":
        return sl.mean(dim=0)
    if pooling == "every_k":
        step = max(1, int(k))
        return sl[::step]
    if pooling == "last_k":
        take = max(1, int(k))
        return sl[-take:] if take > 1 else sl[-1]
    return sl  # "all"


class HFGPUBackend(TargetBackend, EditorBackend):
    """Colab A100/L4/T4 Transformers backend with HybridWeightOverlay and readout hooks."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        profile: ComputeProfile,
        canonical_reader: CanonicalWeightReader | None = None,
        editor_model: Any | None = None,
        editor_tokenizer: Any | None = None,
        model_id: str = "unknown",
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"
        self.profile = profile
        self.canonical_reader = canonical_reader or CanonicalWeightReader(model=self.model)
        self.overlay = HybridWeightOverlay(self.model)
        self.editor_model = (editor_model or self.model).eval()
        self.editor_tokenizer = editor_tokenizer or self.tokenizer
        if hasattr(self.editor_tokenizer, "padding_side"):
            self.editor_tokenizer.padding_side = "left"
        self.model_id = model_id
        self._layers = resolve_decoder_layers(self.model)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        profile: ComputeProfile,
        canonical_dir: Path | None = None,
        cache_dir: Path | None = None,
        trust_remote_code: bool = True,
    ) -> "HFGPUBackend":
        """Load HF open-weights model on Colab A100 (BF16 on 80G; 4-bit hybrid on 40G/L4)."""
        import torch  # type: ignore[import-not-found]
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
        from transformers import AutoTokenizer  # type: ignore[import-not-found]

        local_dir = Path(canonical_dir) if canonical_dir else Path(
            snapshot_download(repo_id=model_id, cache_dir=str(cache_dir) if cache_dir else None)
        )
        tokenizer = AutoTokenizer.from_pretrained(str(local_dir), trust_remote_code=trust_remote_code)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        dtype = torch.bfloat16 if profile.dtype == "bfloat16" else torch.float16
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "device_map": "auto" if "cuda" in profile.device else "cpu",
        }
        if profile.quantization == "nf4_bf16_hybrid":
            from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs["torch_dtype"] = dtype
        else:
            load_kwargs["torch_dtype"] = dtype

        model = load_hf_model(str(local_dir), **load_kwargs)
        reader = CanonicalWeightReader(weight_dir=local_dir, model=model)
        return cls(model=model, tokenizer=tokenizer, profile=profile, canonical_reader=reader, model_id=model_id)

    @property
    def arch_summary(self) -> str:
        cfg = getattr(self.model, "config", None)
        text_cfg = getattr(cfg, "text_config", cfg)
        hs = getattr(text_cfg, "hidden_size", "?")
        nl = len(self._layers)
        return f"{self.model_id} (layers={nl}, hidden_size={hs}, profile={self.profile.name})"

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        return self.canonical_reader.tensor_shapes()

    def export_sandbox_weights_spec(self, dest_dir: Path) -> Path:
        return self.canonical_reader.export_weights_spec(dest_dir)

    def _model_device(self, mod: Any) -> Any:
        for p in mod.parameters():
            return p.device
        return "cpu"

    def _format_user_prompt(self, question: str, require_thinking: bool = True) -> str:
        messages = [{"role": "user", "content": question}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=bool(require_thinking),
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        return f"User: {question}\nAssistant:\n"

    def generate_batch(
        self,
        problems: list[Any],
        edit_tensors: dict[str, Any] | None,
        n_samples: int,
        max_tokens: int,
        config: SessionConfig,
        seed_base: int = 0,
    ) -> list[GenerationRecord]:
        import torch  # type: ignore[import-not-found]

        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"
        pad_tok = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        eos_tok = self.tokenizer.eos_token_id
        batch_size = max(1, int(self.profile.gen_batch_size))

        dev = self._model_device(self.model)
        out: list[GenerationRecord] = []
        with self.overlay.applied(edit_tensors), torch.no_grad():
            for start in range(0, len(problems), batch_size):
                chunk = problems[start : start + batch_size]
                pids = [
                    str(p["problem_id"] if isinstance(p, dict) else getattr(p, "problem_id"))
                    for p in chunk
                ]
                questions = [
                    str(p["question"] if isinstance(p, dict) else getattr(p, "question"))
                    for p in chunk
                ]
                prompts = [
                    self._format_user_prompt(q, require_thinking=bool(config.require_thinking))
                    for q in questions
                ]
                if len(prompts) > 1:
                    enc = self.tokenizer(prompts, return_tensors="pt", padding=True)
                else:
                    enc = self.tokenizer(prompts[0], return_tensors="pt")
                input_ids = enc["input_ids"].to(dev)
                attn_mask = enc.get("attention_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.to(dev)
                prompt_width = int(input_ids.shape[1])

                for j in range(int(n_samples)):
                    batch_seed = problem_seed("|".join(pids), seed_base=seed_base) + j * 1009
                    torch.manual_seed(batch_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(batch_seed)
                    gen_kwargs: dict[str, Any] = {
                        "input_ids": input_ids,
                        "max_new_tokens": int(max_tokens),
                        "do_sample": True,
                        "temperature": float(config.sampling_temperature),
                        "top_p": float(config.sampling_top_p),
                        "top_k": int(config.sampling_top_k),
                    }
                    if pad_tok is not None:
                        gen_kwargs["pad_token_id"] = pad_tok
                    if attn_mask is not None:
                        gen_kwargs["attention_mask"] = attn_mask
                    gen_ids = self.model.generate(**gen_kwargs)

                    for b_idx, pid in enumerate(pids):
                        new_ids = gen_ids[b_idx][prompt_width:]
                        if pad_tok is not None and new_ids.numel() > 0:
                            end = int(new_ids.shape[0])
                            if pad_tok != eos_tok:
                                while end > 0 and int(new_ids[end - 1].item()) == int(pad_tok):
                                    end -= 1
                            else:
                                while (
                                    end > 1
                                    and int(new_ids[end - 1].item()) == int(pad_tok)
                                    and int(new_ids[end - 2].item()) == int(pad_tok)
                                ):
                                    end -= 1
                            new_ids = new_ids[:end]
                        text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
                        out.append(
                            GenerationRecord(
                                problem_id=pid,
                                sample_idx=j,
                                seed=batch_seed,
                                response_text=text,
                                num_tokens=int(new_ids.shape[0]),
                            )
                        )
        return out

    def read_activations(
        self,
        texts: list[str],
        layers: list[int] | None,
        target: str,
        pooling: str,
        k: int,
        edit_tensors: dict[str, Any] | None,
        max_tokens: int,
        pass_id: int,
    ) -> ReadResult:
        import torch  # type: ignore[import-not-found]
        import torch.nn.functional as F  # type: ignore[import-not-found]

        req_layers = layers if layers is not None else list(range(len(self._layers)))
        dev = self._model_device(self.model)
        collected: dict[str, list[Any]] = {f"layer_{l}_{target}": [] for l in req_layers}
        if target == "logprobs":
            collected["token_logprobs"] = []

        with self.overlay.applied(edit_tensors), torch.no_grad():
            for text in texts:
                enc = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=int(max_tokens),
                )
                input_ids = enc["input_ids"].to(dev)

                hooks = []
                cur_captures: dict[int, Any] = {}

                if target in ("hidden", "mlp"):
                    for l_idx in req_layers:
                        layer_mod = self._layers[l_idx]
                        if target == "hidden":
                            def _mk_hook(idx: int):
                                def _hook(_mod, _inp, out):
                                    t = out[0] if isinstance(out, (tuple, list)) else out
                                    cur_captures[idx] = (t[0] if t.ndim == 3 else t).detach().cpu().float()
                                return _hook

                            hooks.append(layer_mod.register_forward_hook(_mk_hook(l_idx)))
                        else:
                            mlp_down = getattr(getattr(layer_mod, "mlp", None), "down_proj", layer_mod)

                            def _mk_pre(idx: int):
                                def _pre(_mod, inp):
                                    t = inp[0]
                                    cur_captures[idx] = (t[0] if t.ndim == 3 else t).detach().cpu().float()
                                return _pre

                            hooks.append(mlp_down.register_forward_pre_hook(_mk_pre(l_idx)))

                try:
                    outputs = self.model(input_ids=input_ids, use_cache=False)
                finally:
                    for h in hooks:
                        h.remove()

                if target == "logprobs":
                    logits = outputs.logits[:, :-1, :].float()
                    targets = input_ids[:, 1:].unsqueeze(-1)
                    lp = F.log_softmax(logits, dim=-1).gather(-1, targets).squeeze(-1)[0].detach().cpu()
                    collected["token_logprobs"].append(_pool_sequence_tensor(lp.unsqueeze(-1), 0, pooling, k).squeeze(-1))
                else:
                    start_pos = 0
                    for l_idx in req_layers:
                        if l_idx in cur_captures:
                            pooled = _pool_sequence_tensor(cur_captures[l_idx], start_pos, pooling, k)
                            collected[f"layer_{l_idx}_{target}"].append(pooled)

        arrays: dict[str, Any] = {}
        for key, items in collected.items():
            if not items:
                continue
            if len({tuple(x.shape) for x in items}) == 1:
                arrays[key] = torch.stack(items, dim=0).numpy()
            else:
                lengths = [int(x.shape[0]) for x in items]
                max_len = max(lengths)
                tail_shape = tuple(items[0].shape[1:])
                padded = torch.zeros((len(items), max_len, *tail_shape), dtype=items[0].dtype)
                for idx_item, item in enumerate(items):
                    padded[idx_item, : int(item.shape[0])] = item
                arrays[key] = padded.numpy()
                arrays[f"{key}_lengths"] = torch.tensor(lengths, dtype=torch.int64).numpy()

        return ReadResult(
            pass_id=pass_id,
            arrays=arrays,
            summary={"layers": req_layers, "num_texts": len(texts), "target": target, "pooling": pooling},
        )

    def complete_turn(
        self,
        messages: list[dict[str, str]],
        config: SessionConfig,
        turn_index: int = 0,
    ) -> str:
        import torch  # type: ignore[import-not-found]

        # Editor ALWAYS runs on clean frozen weights (restore any transient overlay).
        self.overlay.restore()
        ed_tok = self.editor_tokenizer
        ed_mod = self.editor_model
        dev = self._model_device(ed_mod)
        if hasattr(ed_tok, "apply_chat_template"):
            try:
                prompt_text = ed_tok.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=bool(config.require_thinking),
                )
            except TypeError:
                prompt_text = ed_tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        else:
            prompt_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:\n"

        enc = ed_tok(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=int(self.profile.editor_max_context),
        )
        input_ids = enc["input_ids"].to(dev)
        prompt_len = int(input_ids.shape[1])
        pad_tok = ed_tok.pad_token_id if ed_tok.pad_token_id is not None else ed_tok.eos_token_id
        with torch.no_grad():
            gen_kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "max_new_tokens": int(config.editor_max_new_tokens),
                "do_sample": True,
                "temperature": float(config.sampling_temperature),
                "top_p": float(config.sampling_top_p),
                "top_k": int(config.sampling_top_k),
            }
            if "attention_mask" in enc:
                gen_kwargs["attention_mask"] = enc["attention_mask"].to(dev)
            if pad_tok is not None:
                gen_kwargs["pad_token_id"] = pad_tok
            gen = ed_mod.generate(**gen_kwargs)
        return ed_tok.decode(gen[0][prompt_len:], skip_special_tokens=False)
