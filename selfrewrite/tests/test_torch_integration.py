from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch  # type: ignore[import-not-found]
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np  # type: ignore[import-not-found]
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import safetensors  # type: ignore[import-not-found]
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


if HAS_TORCH:
    class _DummyTokenizer:
        eos_token_id = 2
        pad_token_id = 0
        padding_side = "left"

        def apply_chat_template(
            self,
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        ):
            tag = "<think>\n" if enable_thinking else ""
            body = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return body + (f"\nassistant: {tag}" if add_generation_prompt else "")

        def _encode_one(self, text: str, max_length: int | None = None) -> list[int]:
            ids = [3 + (ord(ch) % 27) for ch in text]
            if not ids:
                ids = [3]
            if max_length is not None:
                ids = ids[:max_length]
            return ids

        def __call__(
            self,
            texts,
            return_tensors=None,
            padding=False,
            truncation=False,
            max_length=None,
        ):
            single = isinstance(texts, str)
            seq_inputs = [texts] if single else list(texts)
            encoded = [self._encode_one(t, max_length if truncation else None) for t in seq_inputs]
            max_len = max(len(s) for s in encoded) if (padding and len(encoded) > 1) else len(encoded[0])
            padded = []
            masks = []
            for s in encoded:
                pad_n = max(0, max_len - len(s))
                padded.append(([self.pad_token_id] * pad_n) + s)
                masks.append(([0] * pad_n) + ([1] * len(s)))
            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor(padded, dtype=torch.long),
                    "attention_mask": torch.tensor(masks, dtype=torch.long),
                }
            return {
                "input_ids": padded[0] if single else padded,
                "attention_mask": masks[0] if single else masks,
            }

        def decode(self, ids, skip_special_tokens=False):
            vals = ids.tolist() if hasattr(ids, "tolist") else list(ids)
            return f"<think>\ndecoded len={len(vals)}\n</think>\n\\boxed{{42}}"

        def save_pretrained(self, out_dir: str) -> None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / "dummy_tokenizer.json").write_text('{"dummy": true}\n', encoding="utf-8")

    class _DummyLM(torch.nn.Module):
        def __init__(self, vocab_size: int = 32, hidden_size: int = 16, inter_size: int = 24, num_layers: int = 3):
            super().__init__()
            self.config = SimpleNamespace(
                hidden_size=hidden_size,
                num_hidden_layers=num_layers,
                use_cache=True,
                model_type="qwen3",
                tie_word_embeddings=False,
                to_dict=lambda: {
                    "hidden_size": hidden_size,
                    "num_hidden_layers": num_layers,
                    "use_cache": True,
                    "model_type": "qwen3",
                },
                get=lambda k, default=None: getattr(self.config, k, default),
            )
            try:
                from transformers import GenerationConfig  # type: ignore[import-not-found]

                self.generation_config = GenerationConfig(eos_token_id=2, pad_token_id=0)
            except Exception:
                self.generation_config = SimpleNamespace(eos_token_id=2, pad_token_id=0, is_from_model_config=True)
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)

            class _Blk(torch.nn.Module):
                def __init__(self, idx: int):
                    super().__init__()
                    self.layer_idx = idx
                    self.self_attn = torch.nn.Module()
                    self.self_attn.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
                    self.self_attn.o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
                    self.self_attn.layer_idx = idx
                    self.mlp = torch.nn.Module()
                    self.mlp.gate_proj = torch.nn.Linear(hidden_size, inter_size, bias=False)
                    self.mlp.up_proj = torch.nn.Linear(hidden_size, inter_size, bias=False)
                    self.mlp.down_proj = torch.nn.Linear(inter_size, hidden_size, bias=False)

                def forward(self, x):
                    h = x + self.self_attn.o_proj(torch.tanh(self.self_attn.q_proj(x)))
                    m = self.mlp.down_proj(torch.nn.functional.silu(self.mlp.gate_proj(h)) * self.mlp.up_proj(h))
                    return h + m

            self.model.layers = torch.nn.ModuleList([_Blk(i) for i in range(num_layers)])
            self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

        def forward(self, input_ids=None, attention_mask=None, use_cache=False, inputs_embeds=None, **kwargs):
            x = inputs_embeds if inputs_embeds is not None else self.model.embed_tokens(input_ids)
            for layer in self.model.layers:
                x = layer(x)
            return SimpleNamespace(logits=self.lm_head(x))

        def get_input_embeddings(self):
            return self.model.embed_tokens

        def prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids, **kwargs}

        def generate(self, input_ids, attention_mask=None, max_new_tokens=8, do_sample=True, temperature=0.6, top_p=0.95, top_k=20, pad_token_id=0, **kwargs):
            cur = input_ids
            steps = min(4, max(1, int(max_new_tokens)))
            for _ in range(steps):
                logits = self.forward(cur).logits[:, -1, :]
                probs = torch.softmax(logits.float() / max(0.1, float(temperature)), dim=-1)
                nxt = torch.multinomial(probs, num_samples=1) if do_sample else torch.argmax(probs, dim=-1, keepdim=True)
                cur = torch.cat([cur, nxt], dim=1)
            return cur

        def save_pretrained(self, out_dir: str) -> None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), Path(out_dir) / "pytorch_model.bin")


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed in current Python environment")
class TestTorchIntegration(unittest.TestCase):
    def test_qwen35_nested_layers_and_structural_compression(self) -> None:
        from selfrewrite.backends.hf_gpu_backend import resolve_decoder_layers
        from selfrewrite.distill.compress_model import distill_kl_ce_step, drop_decoder_layers, prune_mlp_width

        class SwiMLP(torch.nn.Module):
            def __init__(self, h: int, inter: int):
                super().__init__()
                self.gate_proj = torch.nn.Linear(h, inter, bias=False)
                self.up_proj = torch.nn.Linear(h, inter, bias=False)
                self.down_proj = torch.nn.Linear(inter, h, bias=False)

            def forward(self, x):
                return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

        class Q35Block(torch.nn.Module):
            def __init__(self, layer_idx: int, h: int, inter: int, is_full: bool):
                super().__init__()
                self.layer_idx = layer_idx
                if is_full:
                    self.self_attn = torch.nn.Module()
                    self.self_attn.layer_idx = layer_idx
                    self.self_attn.o_proj = torch.nn.Linear(h, h, bias=False)
                else:
                    self.linear_attn = torch.nn.Module()
                    self.linear_attn.layer_idx = layer_idx
                    self.linear_attn.out_proj = torch.nn.Linear(h, h, bias=False)
                self.mlp = SwiMLP(h, inter)

            def forward(self, x):
                attn = self.self_attn.o_proj(x) if hasattr(self, "self_attn") else self.linear_attn.out_proj(x)
                return x + attn + self.mlp(x)

        class Qwen35NestedStub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(
                    model_type="qwen3_5",
                    text_config=SimpleNamespace(
                        hidden_size=8,
                        intermediate_size=16,
                        num_hidden_layers=4,
                        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                    ),
                )
                self.model = torch.nn.Module()
                self.model.language_model = torch.nn.Module()
                self.model.language_model.layers = torch.nn.ModuleList(
                    [Q35Block(i, 8, 16, i == 3) for i in range(4)]
                )

            def forward(self, x):
                for l in self.model.language_model.layers:
                    x = l(x)
                return x

        q35 = Qwen35NestedStub()
        self.assertEqual(len(resolve_decoder_layers(q35)), 4)

        prune_info = prune_mlp_width(q35, 8)
        self.assertEqual(prune_info["intermediate_size"], 8)
        self.assertEqual(q35.config.text_config.intermediate_size, 8)
        self.assertEqual(q35.model.language_model.layers[0].mlp.down_proj.in_features, 8)

        drop_info = drop_decoder_layers(q35, [0, 3])
        self.assertEqual(drop_info["kept_layers"], 2)
        self.assertEqual(q35.config.text_config.num_hidden_layers, 2)
        self.assertEqual(q35.config.text_config.layer_types, ["linear_attention", "full_attention"])
        self.assertEqual(q35.model.language_model.layers[1].self_attn.layer_idx, 1)
        out = q35(torch.randn(2, 4, 8))
        self.assertEqual(tuple(out.shape), (2, 4, 8))

        s_log = torch.randn(2, 5, 10, requires_grad=True)
        t_log = torch.randn(2, 5, 10)
        labels = torch.tensor([[-100, -100, 1, 2, 3], [-100, 4, 5, 6, 7]])
        loss = distill_kl_ce_step(s_log, t_log, labels)
        loss.backward()
        self.assertIsNotNone(s_log.grad)
        self.assertGreater(float(loss.item()), 0.0)

        # Edge case: all masked tokens must return zero finite loss instead of NaN
        all_masked = torch.full((2, 5), -100, dtype=torch.long)
        zero_loss = distill_kl_ce_step(s_log, t_log, all_masked)
        self.assertTrue(torch.isfinite(zero_loss))
        self.assertEqual(float(zero_loss.item()), 0.0)

    def test_hybrid_weight_overlay_standard_and_quantized_restore(self) -> None:
        from selfrewrite.backends.hf_gpu_backend import CanonicalWeightReader, HybridWeightOverlay, resolve_decoder_layers
        from selfrewrite.helpers import remove_direction, unit_vector

        class Fake4BitLinear(torch.nn.Module):
            def __init__(self, in_f: int, out_f: int):
                super().__init__()
                self.in_features = in_f
                self.out_features = out_f
                w = torch.randn(out_f, in_f)
                w.quant_state = {"fake_nf4": True}  # type: ignore[attr-defined]
                self.weight = torch.nn.Parameter(w, requires_grad=False)
                self.bias = None

        class FakeLayer(torch.nn.Module):
            def __init__(self, h: int, inter: int):
                super().__init__()
                self.self_attn = torch.nn.Module()
                self.self_attn.o_proj = torch.nn.Linear(h, h, bias=False)
                self.mlp = torch.nn.Module()
                self.mlp.down_proj = Fake4BitLinear(inter, h)

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([FakeLayer(8, 12) for _ in range(3)])

        m = FakeModel()
        orig_o = m.model.layers[1].self_attn.o_proj.weight.detach().clone()
        orig_mod = m.model.layers[2].mlp.down_proj
        u = unit_vector(torch.ones(8))
        new_o = remove_direction(orig_o, u, 0.2)
        new_down = torch.randn(8, 12)

        overlay = HybridWeightOverlay(m)
        with overlay.applied(
            {
                "model.layers.1.self_attn.o_proj.weight": new_o,
                "model.layers.2.mlp.down_proj.weight": new_down,
            }
        ):
            self.assertGreater(float((m.model.layers[1].self_attn.o_proj.weight.detach() - orig_o).abs().max().item()), 1e-4)
            self.assertIsInstance(m.model.layers[2].mlp.down_proj, torch.nn.Linear)
            self.assertIsNot(m.model.layers[2].mlp.down_proj, orig_mod)

        self.assertTrue(torch.allclose(orig_o, m.model.layers[1].self_attn.o_proj.weight.detach()))
        self.assertIs(m.model.layers[2].mlp.down_proj, orig_mod)
        self.assertEqual(len(resolve_decoder_layers(m)), 3)

        if HAS_SAFETENSORS:
            reader = CanonicalWeightReader(model=m)
            with tempfile.TemporaryDirectory() as td:
                spec = reader.export_weights_spec(Path(td))
                self.assertTrue(spec.is_file())

    def test_fast_torch_corruption_mixed_edit_store_and_sandbox_safetensors(self) -> None:
        from selfrewrite.edit_store import EditStore
        from selfrewrite.helpers import outlier_ratios, remove_direction, unit_vector
        from selfrewrite.sandbox import SealedPythonSandbox
        from selfrewrite.tasks.corruption import CorruptionSpec, apply_block_corruption

        w = torch.randn(32, 24) * 0.02
        corr, cspec = apply_block_corruption(
            w,
            CorruptionSpec(layer_idx=1, tensor_name="model.layers.1.mlp.down_proj.weight", row_start=4, num_rows=6, sigma_mult=200.0, mode="gaussian_noise", seed=11),
        )
        self.assertIsInstance(corr, torch.Tensor)
        self.assertEqual(tuple(corr.shape), (32, 24))
        ratios = outlier_ratios({"corr": corr})
        self.assertGreater(ratios["corr"]["ratio_max_to_median"], 80.0)

        _, zero_spec = apply_block_corruption(
            w,
            CorruptionSpec(layer_idx=1, tensor_name="t", row_start=0, num_rows=4, mode="zero_rows", seed=1),
        )
        self.assertEqual(zero_spec.num_rows, 4)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            work_dir = root / "WORK"
            work_dir.mkdir(parents=True, exist_ok=True)
            tensors = {"model.layers.1.mlp.down_proj.weight": corr, "model.layers.1.self_attn.o_proj.weight": torch.randn(16, 16) * 0.02}

            # Mixed torch original + numpy candidate in EditStore
            if HAS_NUMPY:
                store = EditStore(work_dir, lambda nm: tensors[nm], {k: tuple(v.shape) for k, v in tensors.items()})
                u = unit_vector(torch.ones(16)).numpy()
                cand_np = remove_direction(tensors["model.layers.1.self_attn.o_proj.weight"].numpy(), u, 0.25)
                meta = store.save_edit("model.layers.1.self_attn.o_proj.weight", cand_np)
                self.assertGreater(meta["rel_scalar_resid"], 1e-4)

            if HAS_SAFETENSORS:
                from selfrewrite.backends.hf_gpu_backend import CanonicalWeightReader

                reader = CanonicalWeightReader(state_dict=tensors)
                spec_path = reader.export_weights_spec(root / "_weights")
                sb = SealedPythonSandbox(work_dir, spec_path, timeout_sec=10.0)
                try:
                    res = sb.run(
                        "W = load_original('model.layers.1.self_attn.o_proj.weight')\n"
                        "u = unit_vector(torch.ones(W.shape[0]))\n"
                        "save_edit('model.layers.1.self_attn.o_proj.weight', remove_direction(W, u, 0.2))\n"
                        "print('SANDBOX_EDIT_OK', len(list_edits()))"
                    )
                    self.assertTrue(res["ok"], res)
                    self.assertIn("SANDBOX_EDIT_OK 1", res["output"])
                finally:
                    sb.close()

    def test_hf_gpu_backend_batched_generation_reads_and_complete_turn(self) -> None:
        from selfrewrite.backends.hf_gpu_backend import HFGPUBackend
        from selfrewrite.config import ComputeProfile, SessionConfig
        from selfrewrite.tasks.datasets import Problem

        model = _DummyLM()
        tok = _DummyTokenizer()
        prof = ComputeProfile(
            name="test_cpu",
            device="cpu",
            dtype="float32",
            quantization="none",
            vram_gb=0,
            editor_max_context=512,
            answer_max_tokens=16,
            starting_diagnosis_n=4,
            gen_batch_size=2,
            max_wall_time_sec=60,
            tensor_storage="json",
        )
        backend = HFGPUBackend(model=model, tokenizer=tok, profile=prof, model_id="dummy-lm")
        probs = [
            Problem("p1", "Short q?", "42"),
            Problem("p2", "Much longer question about algebra x+2=5?", "42"),
            Problem("p3", "Third problem?", "42"),
        ]
        cfg = SessionConfig(task="improve", protocol=2, answer_max_tokens=16, editor_max_new_tokens=16)

        recs_a = backend.generate_batch(probs, None, n_samples=1, max_tokens=8, config=cfg, seed_base=10)
        recs_b = backend.generate_batch(probs, None, n_samples=1, max_tokens=8, config=cfg, seed_base=10)
        self.assertEqual(len(recs_a), 3)
        self.assertEqual([r.response_text for r in recs_a], [r.response_text for r in recs_b])

        for tgt_mode in ("hidden", "mlp", "logprobs"):
            rr = backend.read_activations(
                texts=["short", "much longer sequence of tokens"],
                layers=[0, 2],
                target=tgt_mode,
                pooling="all",
                k=1,
                edit_tensors=None,
                max_tokens=64,
                pass_id=1,
            )
            self.assertTrue(any(k.endswith("_lengths") for k in rr.arrays), rr.arrays.keys())

        turn_out = backend.complete_turn([{"role": "user", "content": "Diagnose."}], cfg, turn_index=0)
        self.assertIn("</think>", turn_out)

    def test_student_sft_dpo_rlvr_and_merge_freeze(self) -> None:
        from selfrewrite.distill.train_student import (
            StudentTrainConfig,
            _attach_lora_for_training,
            merge_and_freeze_student,
            run_hf_dpo_stage,
            run_hf_rlvr_stage,
            run_hf_sft_stage,
        )

        model = _DummyLM()
        tok = _DummyTokenizer()
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "frozen"
            cfg = StudentTrainConfig(
                rung_id="R3",
                base_model_id="dummy-lm",
                bundle_dir=Path(td),
                output_dir=out_dir,
                backend="hf",
                lora_rank=4,
                lora_alpha=8,
                sft_epochs=1,
                dpo_epochs=1,
                rlvr_steps=2,
                grad_accum_steps=1,
                max_seq_tokens=128,
            )
            lora_model, kind = _attach_lora_for_training(model, cfg)
            self.assertIn(kind, ("peft", "builtin_lora"))

            sft_rows = [
                {
                    "messages": [
                        {"role": "user", "content": "Inspect weights."},
                        {"role": "assistant", "content": "<think>\nChecking.\n</think>\n```python\nprint(len(list_tensors()))\n```"},
                    ]
                }
            ]
            sft_res = run_hf_sft_stage(lora_model, tok, sft_rows, cfg)
            self.assertGreater(sft_res["steps"], 0)
            self.assertTrue(math.isfinite(sft_res["final_loss"]))

            dpo_rows = [
                {
                    "prompt": [{"role": "user", "content": "Repair damaged matrix."}],
                    "chosen": "<think>\nScan outlier_ratios first.\n</think>\n```python\nprint(1)\n```",
                    "rejected": "<think>\nAdd random noise.\n</think>\n```python\nw = w + np.random.randn(2)\n```",
                }
            ]
            dpo_res = run_hf_dpo_stage(lora_model, tok, dpo_rows, cfg)
            self.assertGreater(dpo_res["steps"], 0)
            self.assertTrue(math.isfinite(dpo_res["final_loss"]))

            rlvr_rows = [{"session_id": "ep1", "task": "repair", "corruption_layer": 1, "reserved_delta": 0.25, "census_summary": {"accuracy": 0.0}, "expected_decision": "SUBMIT"}]
            rlvr_res = run_hf_rlvr_stage(lora_model, tok, rlvr_rows, cfg)
            self.assertGreater(rlvr_res["steps"], 0)
            self.assertEqual(rlvr_res["group_size"], 4)

            saved = merge_and_freeze_student(lora_model, tok, out_dir, {"rung_id": "R3"}, adapter_kind=kind)
            self.assertTrue((saved / "student_provenance.json").is_file())
            prov = json.loads((saved / "student_provenance.json").read_text(encoding="utf-8"))
            self.assertTrue(prov["frozen_for_eval"])
            self.assertFalse(any(p.requires_grad for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
