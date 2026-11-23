from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from selfrewrite.backends.base import EditorBackend, GenerationRecord, ReadResult, TargetBackend
from selfrewrite.backends.hf_gpu_backend import CanonicalWeightReader, HybridWeightOverlay, _pool_sequence_tensor
from selfrewrite.config import ComputeProfile, SessionConfig
from selfrewrite.tasks.datasets import problem_seed


def _load_sibling_config(weight_path: Path | None) -> dict[str, Any]:
    if weight_path is None:
        return {}
    p = Path(weight_path)
    for cand in (p.with_suffix(".config.json"), p.parent / "model_config.json"):
        if cand.is_file():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "hidden_size" in data:
                    return data
            except Exception:
                pass
    return {}


class MiniMindBackend(TargetBackend, EditorBackend):
    """Native PyTorch MiniMindForCausalLM target+editor adapter for R4m floor probes & smoke tests."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        profile: ComputeProfile,
        canonical_reader: CanonicalWeightReader | None = None,
        label: str = "minimind",
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.profile = profile
        self.label = label
        self.canonical_reader = canonical_reader or CanonicalWeightReader(model=self.model)
        self.overlay = HybridWeightOverlay(self.model)

    @classmethod
    def from_checkpoint(
        cls,
        weight_path: Path | None,
        tokenizer_dir: Path,
        profile: ComputeProfile,
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        intermediate_size: int | None = None,
        use_moe: bool = False,
        canonical_state_dict: dict[str, Any] | None = None,
    ) -> "MiniMindBackend":
        import torch  # type: ignore[import-not-found]
        from transformers import AutoTokenizer  # type: ignore[import-not-found]
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # type: ignore[import-not-found]

        file_cfg = _load_sibling_config(weight_path)
        hs = int(file_cfg.get("hidden_size", hidden_size))
        nl = int(file_cfg.get("num_hidden_layers", num_hidden_layers))
        um = bool(file_cfg.get("use_moe", use_moe))
        inter = file_cfg.get("intermediate_size", intermediate_size)
        kwargs: dict[str, Any] = {}
        if inter is not None:
            kwargs["intermediate_size"] = int(inter)
        if "num_attention_heads" in file_cfg:
            kwargs["num_attention_heads"] = int(file_cfg["num_attention_heads"])
        if "num_key_value_heads" in file_cfg:
            kwargs["num_key_value_heads"] = int(file_cfg["num_key_value_heads"])

        tok = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        cfg = MiniMindConfig(hidden_size=hs, num_hidden_layers=nl, use_moe=um, **kwargs)
        model = MiniMindForCausalLM(cfg)
        if canonical_state_dict is not None:
            model.load_state_dict(canonical_state_dict, strict=False)
        elif weight_path is not None and Path(weight_path).is_file():
            sd = torch.load(str(weight_path), map_location="cpu")
            model.load_state_dict(sd, strict=False)

        dtype = torch.bfloat16 if profile.dtype == "bfloat16" and "cuda" in profile.device else torch.float32
        model = model.to(device=profile.device, dtype=dtype).eval()
        reader = CanonicalWeightReader(model=model)
        return cls(
            model=model,
            tokenizer=tok,
            profile=profile,
            canonical_reader=reader,
            label=f"minimind-H{hs}-L{nl}",
        )

    @property
    def arch_summary(self) -> str:
        cfg = self.model.config
        return f"{self.label} (layers={cfg.num_hidden_layers}, hidden_size={cfg.hidden_size}, use_moe={cfg.use_moe})"

    @property
    def num_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        return self.canonical_reader.tensor_shapes()

    def export_sandbox_weights_spec(self, dest_dir: Path) -> Path:
        return self.canonical_reader.export_weights_spec(dest_dir)

    def _device(self):
        return next(self.model.parameters()).device

    def _prompt_text(self, question: str) -> str:
        messages = [{"role": "user", "content": question}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, open_thinking=True
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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

        dev = self._device()
        out: list[GenerationRecord] = []
        with self.overlay.applied(edit_tensors), torch.no_grad():
            for prob in problems:
                pid = str(prob["problem_id"] if isinstance(prob, dict) else getattr(prob, "problem_id"))
                q = str(prob["question"] if isinstance(prob, dict) else getattr(prob, "question"))
                enc = self.tokenizer(self._prompt_text(q), return_tensors="pt", truncation=True)
                input_ids = enc["input_ids"].to(dev)
                prompt_len = int(input_ids.shape[1])
                for j in range(int(n_samples)):
                    s = problem_seed(pid, seed_base=seed_base)
                    torch.manual_seed(s + j * 1009)
                    gen_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=enc.get("attention_mask", None).to(dev) if enc.get("attention_mask") is not None else None,
                        max_new_tokens=int(max_tokens),
                        do_sample=True,
                        temperature=float(config.sampling_temperature),
                        top_p=float(config.sampling_top_p),
                        top_k=int(config.sampling_top_k),
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    new_ids = gen_ids[0][prompt_len:]
                    decoded = self.tokenizer.decode(new_ids, skip_special_tokens=False)
                    if "<think>" not in decoded and ("</think>" in decoded or decoded.strip()):
                        decoded = "<think>\n" + decoded
                    out.append(GenerationRecord(pid, j, s, decoded, int(new_ids.shape[0])))
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

        req_layers = layers if layers is not None else list(range(len(self.model.model.layers)))
        dev = self._device()
        collected: dict[str, list[Any]] = {f"layer_{l}_{target}": [] for l in req_layers}
        if target == "logprobs":
            collected["token_logprobs"] = []

        with self.overlay.applied(edit_tensors), torch.no_grad():
            for text in texts:
                enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=int(max_tokens))
                input_ids = enc["input_ids"].to(dev)
                hooks = []
                cur_caps: dict[int, Any] = {}
                if target in ("hidden", "mlp"):
                    for l_idx in req_layers:
                        blk = self.model.model.layers[l_idx]
                        if target == "hidden":
                            def _mk(idx: int):
                                def _hk(_m, _i, out):
                                    t = out[0] if isinstance(out, (tuple, list)) else out
                                    cur_caps[idx] = t[0].detach().cpu().float()
                                return _hk

                            hooks.append(blk.register_forward_hook(_mk(l_idx)))
                        else:
                            down = getattr(blk.mlp, "down_proj", blk.mlp)

                            def _mk_pre(idx: int):
                                def _pre(_m, inp):
                                    cur_caps[idx] = inp[0][0].detach().cpu().float()
                                return _pre

                            hooks.append(down.register_forward_pre_hook(_mk_pre(l_idx)))
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
                    for l_idx in req_layers:
                        if l_idx in cur_caps:
                            collected[f"layer_{l_idx}_{target}"].append(_pool_sequence_tensor(cur_caps[l_idx], 0, pooling, k))

        arrays = {
            key: torch.stack(items, dim=0).numpy()
            for key, items in collected.items()
            if items
        }
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

        self.overlay.restore()
        dev = self._device()
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, open_thinking=True
            )
        except TypeError:
            prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(prompt_text, return_tensors="pt", truncation=True)
        input_ids = enc["input_ids"].to(dev)
        prompt_len = int(input_ids.shape[1])
        with torch.no_grad():
            gen = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=min(int(config.editor_max_new_tokens), 512),
                do_sample=True,
                temperature=float(config.sampling_temperature),
                top_p=float(config.sampling_top_p),
                top_k=int(config.sampling_top_k),
                eos_token_id=self.tokenizer.eos_token_id,
            )
        decoded = self.tokenizer.decode(gen[0][prompt_len:], skip_special_tokens=False)
        if "<think>" not in decoded and ("</think>" in decoded or decoded.strip()):
            decoded = "<think>\n" + decoded
        return decoded
