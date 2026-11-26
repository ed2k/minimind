from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any

from selfrewrite.backends.base import EditorBackend, GenerationRecord, ReadResult, TargetBackend
from selfrewrite.config import SessionConfig, atomic_write_text
from selfrewrite.helpers import flatten_floats, outlier_ratios, tensor_shape, to_nested_list, unit_vector
from selfrewrite.tasks.datasets import problem_seed


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class MockTransformerModel:
    """Pure-stdlib multi-layer transformer weight & behavior simulator for self-rewriting CI tests."""

    def __init__(
        self,
        num_layers: int = 8,
        hidden_dim: int = 16,
        inter_dim: int = 24,
        verbosity_layer: int = 2,
        seed: int = 7,
    ):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.inter_dim = inter_dim
        self.verbosity_layer = verbosity_layer
        rng = random.Random(seed)

        # True hidden direction separating long/looping reasoning from finished answers.
        raw_u = [math.sin((j + 1) * 0.7) for j in range(hidden_dim)]
        self.verbosity_dir: list[float] = unit_vector(raw_u)

        self.weights: dict[str, list[list[float]]] = {}
        self.weights["model.embed_tokens.weight"] = [
            [rng.gauss(0.0, 0.02) for _ in range(hidden_dim)] for _ in range(32)
        ]
        for layer in range(num_layers):
            o_mat = [[rng.gauss(0.0, 0.03) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
            d_mat = [[rng.gauss(0.0, 0.025) for _ in range(inter_dim)] for _ in range(hidden_dim)]
            if layer == verbosity_layer:
                # Inject subtle component aligned with verbosity_dir into output matrices.
                for i in range(hidden_dim):
                    for j in range(hidden_dim):
                        o_mat[i][j] += 0.28 * self.verbosity_dir[i] * ((j % 3) - 1)
                    for j in range(inter_dim):
                        d_mat[i][j] += 0.25 * self.verbosity_dir[i] * ((j % 3) - 1)
            self.weights[f"model.layers.{layer}.self_attn.o_proj.weight"] = o_mat
            self.weights[f"model.layers.{layer}.mlp.down_proj.weight"] = d_mat
        self.weights["model.norm.weight"] = [[1.0] * hidden_dim]  # 2D wrapper avoided below
        # Store 1D vectors as nested list cleanly as well:
        self.raw_tensors: dict[str, Any] = dict(self.weights)
        self.raw_tensors["model.norm.weight"] = [1.0 for _ in range(hidden_dim)]
        self.raw_tensors["lm_head.weight"] = [
            [rng.gauss(0.0, 0.02) for _ in range(hidden_dim)] for _ in range(32)
        ]

    def _effective_tensors(self, edit_tensors: dict[str, Any] | None) -> dict[str, Any]:
        merged = {k: v for k, v in self.raw_tensors.items()}
        if edit_tensors:
            for k, v in edit_tensors.items():
                merged[k] = to_nested_list(v)
        return merged

    def inspect_state(self, edit_tensors: dict[str, Any] | None) -> dict[str, Any]:
        """Measure outlier corruption AND verbosity-direction alignment directly from weights."""
        eff = self._effective_tensors(edit_tensors)

        # 1. Check whether any matrix has outlier corruption (ratio > 80) or large random noise distortion.
        stats = outlier_ratios({k: v for k, v in eff.items() if k.startswith("model.layers.")})
        max_ratio = max((s["ratio_max_to_median"] for s in stats.values()), default=1.0)
        corrupted = max_ratio > 80.0

        # Check random weight perturbation damage relative to clean base weights across unrelated layers.
        noise_distortion = 0.0
        for name, w in eff.items():
            if not name.startswith("model.layers."):
                continue
            orig = flatten_floats(self.raw_tensors[name])
            cur = flatten_floats(w)
            orig_sq = sum(x * x for x in orig)
            diff_sq = sum((b - a) ** 2 for a, b in zip(orig, cur))
            rel = math.sqrt(diff_sq / max(1e-12, orig_sq))
            layer_match = re.search(r"layers\.(\d+)\.", name)
            l_idx = int(layer_match.group(1)) if layer_match else -1
            if l_idx != self.verbosity_layer:
                # In repair mode, extreme outlier clipping only shifts rel slightly; random noise shifts unrelated layers.
                if rel > 0.25 and not (stats[name]["ratio_max_to_median"] <= 30.0):
                    noise_distortion += rel
                elif max_ratio <= 80.0 and rel > 0.15 and self._has_noise_corruption(orig, cur):
                    noise_distortion += rel

        # 2. Measure alignment of verbosity_layer output matrices along verbosity_dir vs. unedited base.
        o_name = f"model.layers.{self.verbosity_layer}.self_attn.o_proj.weight"
        d_name = f"model.layers.{self.verbosity_layer}.mlp.down_proj.weight"

        def _dir_energy(mat: list[list[float]]) -> float:
            proj = [sum(self.verbosity_dir[i] * mat[i][j] for i in range(self.hidden_dim)) for j in range(len(mat[0]))]
            return math.sqrt(sum(p * p for p in proj))

        base_energy = _dir_energy(self.raw_tensors[o_name]) + _dir_energy(self.raw_tensors[d_name])
        cur_energy = _dir_energy(eff[o_name]) + _dir_energy(eff[d_name])
        reduction = (base_energy - cur_energy) / max(1e-9, base_energy)

        # Orthogonal collateral change on target layer (penalizes random noise on target layer too).
        return {
            "corrupted": corrupted,
            "max_ratio": max_ratio,
            "noise_distortion": noise_distortion,
            "verbosity_reduction": reduction,
        }

    def _has_noise_corruption(self, orig: list[float], cur: list[float]) -> bool:
        mean_abs_orig = sum(abs(x) for x in orig) / max(1, len(orig))
        mean_abs_diff = sum(abs(b - a) for a, b in zip(orig, cur)) / max(1, len(orig))
        return mean_abs_diff > 0.5 * mean_abs_orig


class MockTargetBackend(TargetBackend):
    """Deterministic TargetBackend backed by MockTransformerModel."""

    def __init__(
        self,
        model: MockTransformerModel,
        target_tensors: dict[str, Any] | None = None,
        damage_mode: bool = False,
        base_timeout_rate: float = 0.42,
    ):
        self._base_model = model
        self._target_tensors = dict(target_tensors) if target_tensors is not None else dict(model.raw_tensors)
        self._damage_mode = damage_mode
        self._base_timeout_rate = base_timeout_rate

    @property
    def arch_summary(self) -> str:
        return f"MockTransformer(layers={self._base_model.num_layers}, hidden={self._base_model.hidden_dim}, inter={self._base_model.inter_dim})"

    @property
    def num_layers(self) -> int:
        return self._base_model.num_layers

    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        return {k: tensor_shape(v) for k, v in self._target_tensors.items()}

    def export_sandbox_weights_spec(self, dest_dir: Path) -> Path:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        data_path = dest_dir / "weights_data.json"
        spec_path = dest_dir / "weights_spec.json"
        serial = {k: to_nested_list(v) for k, v in self._target_tensors.items()}
        atomic_write_text(data_path, json.dumps(serial))
        spec = {
            "mode": "json_tensors",
            "data_path": str(data_path.resolve()),
            "shapes": {k: list(v) for k, v in self.tensor_shapes().items()},
        }
        atomic_write_text(spec_path, json.dumps(spec, indent=2))
        return spec_path

    def _effective_edit_overlay(self, edit_tensors: dict[str, Any] | None) -> dict[str, Any]:
        merged = dict(self._target_tensors)
        if edit_tensors:
            merged.update(edit_tensors)
        return merged

    def generate_batch(
        self,
        problems: list[Any],
        edit_tensors: dict[str, Any] | None,
        n_samples: int,
        max_tokens: int,
        config: SessionConfig,
        seed_base: int = 0,
    ) -> list[GenerationRecord]:
        eff = self._effective_edit_overlay(edit_tensors)
        diag = self._base_model.inspect_state(eff)

        # Effective outlier check against active eff weights and target baseline:
        ratios = outlier_ratios({k: v for k, v in eff.items() if k.startswith("model.layers.")})
        orig_ratios = outlier_ratios({k: v for k, v in self._target_tensors.items() if k.startswith("model.layers.")})
        max_ratio = max((r["ratio_max_to_median"] for r in ratios.values()), default=1.0)
        is_corrupted = max_ratio > 80.0

        # Check random non-targeted distortion relative to unedited target_tensors:
        noise_broken = False
        o_target = f"model.layers.{self._base_model.verbosity_layer}.self_attn.o_proj.weight"
        d_target = f"model.layers.{self._base_model.verbosity_layer}.mlp.down_proj.weight"
        for name in eff:
            if name in (o_target, d_target) or not name.startswith("model.layers."):
                continue
            r_cur = ratios.get(name, {}).get("ratio_max_to_median", 1.0)
            r_orig = orig_ratios.get(name, {}).get("ratio_max_to_median", 1.0)
            orig_flat = flatten_floats(self._target_tensors[name])
            cur_flat = flatten_floats(eff[name])
            diff = math.sqrt(sum((b - a) ** 2 for a, b in zip(orig_flat, cur_flat)))
            orig_n = math.sqrt(sum(a * a for a in orig_flat))
            # Valid outlier repair fixes an originally corrupted matrix (r_orig > 80) down to <= 30.
            is_valid_repair = r_orig > 80.0 and r_cur <= 30.0
            if diff / max(1e-9, orig_n) > 0.15 and not is_valid_repair:
                noise_broken = True

        # Check verbosity improvement from base state vs. target weights:
        verbosity_reduction = diag["verbosity_reduction"] if not self._damage_mode else 0.0
        # Penalize destructive over-removal (alpha > 0.75) or negative strengthening (alpha < 0).
        if not self._damage_mode and verbosity_reduction < -0.02:
            timeout_prob = min(0.95, self._base_timeout_rate + abs(verbosity_reduction) * 1.5)
        elif not self._damage_mode and 0.02 <= verbosity_reduction <= 0.65:
            timeout_prob = max(0.05, self._base_timeout_rate - verbosity_reduction * 2.2)
        elif not self._damage_mode and verbosity_reduction > 0.65:
            noise_broken = True
            timeout_prob = 0.85
        else:
            timeout_prob = self._base_timeout_rate

        out: list[GenerationRecord] = []
        for prob in problems:
            pid = prob["problem_id"] if isinstance(prob, dict) else getattr(prob, "problem_id")
            ans = prob["answer"] if isinstance(prob, dict) else getattr(prob, "answer")
            for j in range(n_samples):
                s = problem_seed(pid, seed_base=seed_base)
                rng = random.Random(s + j * 1009)
                draw_t = rng.random()
                draw_c = rng.random()

                if is_corrupted or noise_broken:
                    if rng.random() < 0.7:
                        txt = "<think>\n" + ("step repeat . " * 1200)
                        out.append(GenerationRecord(pid, j, s, txt, int(max_tokens)))
                    else:
                        txt = "<think>\nbroken reasoning</think>\n\\boxed{0}"
                        out.append(GenerationRecord(pid, j, s, txt, 48))
                    continue

                if draw_t < timeout_prob:
                    phrase = "The reasoning is solid. I will write the response. "
                    txt = "<think>\n" + (phrase * 220)
                    out.append(GenerationRecord(pid, j, s, txt, int(max_tokens)))
                    continue

                correct_prob = 0.94 if not self._damage_mode else 0.88
                chosen_ans = str(ans) if draw_c < correct_prob else str(int(ans) + 7 if str(ans).isdigit() else "999999")
                used_tok = min(int(max_tokens) - 32, rng.randint(120, max(160, int(max_tokens) // 2)))
                txt = f"<think>\nSolving {pid} carefully gives {chosen_ans}.\n</think>\n\\boxed{{{chosen_ans}}}"
                out.append(GenerationRecord(pid, j, s, txt, used_tok))
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
        eff = self._effective_edit_overlay(edit_tensors)
        ratios = outlier_ratios({nm: v for nm, v in eff.items() if nm.startswith("model.layers.")})
        # Find any still-corrupted layer index so repair sessions see real activation explosion there!
        corrupt_layers = set()
        for nm, st in ratios.items():
            if st["ratio_max_to_median"] > 80.0:
                m = re.search(r"layers\.(\d+)\.", nm)
                if m:
                    corrupt_layers.add(int(m.group(1)))
        min_corrupt = min(corrupt_layers) if corrupt_layers else None

        req_layers = layers if layers is not None else list(range(self._base_model.num_layers))
        arrays: dict[str, Any] = {}
        d = self._base_model.hidden_dim
        u = self._base_model.verbosity_dir

        for l_idx in req_layers:
            rows: list[list[float]] = []
            for t_idx, txt in enumerate(texts):
                is_loop = ("will write the response" in txt) or ("repeat" in txt) or len(txt) > 2000
                sign = 1.0 if is_loop else -0.2
                # Deeper layers amplify the verbosity signal slightly, starting cleanly at verbosity_layer.
                layer_gain = 1.8 if l_idx >= self._base_model.verbosity_layer else 0.2
                explode = 200.0 if (min_corrupt is not None and l_idx >= min_corrupt) else 1.0
                rng = random.Random(t_idx * 31 + l_idx * 17)
                vec = [explode * (layer_gain * sign * u[j] + 0.04 * rng.gauss(0.0, 1.0)) for j in range(d)]
                rows.append(vec)
            arrays[f"layer_{l_idx}_{target}"] = rows

        return ReadResult(
            pass_id=pass_id,
            arrays=arrays,
            summary={"layers": req_layers, "num_texts": len(texts), "target": target, "pooling": pooling},
        )


class MockFailureEditor(EditorBackend):
    """Simulates smaller-model failure modes (2B no-edit, 4B random-noise, 9B bad-submit)."""

    def __init__(self, strategy: str = "random_noise_4b"):
        self.strategy = strategy

    def complete_turn(
        self,
        messages: list[dict[str, str]],
        config: SessionConfig,
        turn_index: int = 0,
    ) -> str:
        last_user = messages[-1]["content"] if messages else ""
        if "SUBMIT" in last_user and "WITHDRAW" in last_user:
            if self.strategy == "bad_submit_9b":
                return "<think>\nSubmitting despite bad scores.\n</think>\nSUBMIT"
            return "<think>\nMy edits hurt performance.\n</think>\nWITHDRAW"

        if self.strategy == "no_edit_2b":
            if turn_index == 0:
                return "<think>\nLet me try running broken code.\n</think>\n```python\n1 / 0\n```"
            return (
                "<think>\nI am a text model without actual neural network weights I can change.\n</think>\n"
                "Unable to modify weights; no edit submitted."
            )

        if self.strategy == "random_noise_4b":
            if turn_index == 0:
                return (
                    "<think>\nLet me add random Gaussian perturbations to layer 5.\n</think>\n"
                    "```python\n"
                    "import random\n"
                    "rng = random.Random(1)\n"
                    "for nm in layer_output_matrices(5):\n"
                    "    W = load_original(nm)\n"
                    "    if isinstance(W, list):\n"
                    "        noisy = [[v + rng.gauss(0.0, 0.5) for v in row] for row in W]\n"
                    "    else:\n"
                    "        import torch\n"
                    "        g = torch.Generator().manual_seed(1)\n"
                    "        noisy = W + torch.randn(W.shape, generator=g, dtype=W.dtype) * 0.5\n"
                    "    save_edit(nm, noisy)\n"
                    "print('Saved random noise')\n"
                    "```"
                )
            if turn_index == 1:
                return (
                    "<think>\nTest my random noise edit on practice problems.\n</think>\n"
                    '```forward\n{"mode": "generate", "weights": "both", "count": 12, "n_samples": 1}\n```'
                )
            return (
                "<think>\nThe random noise solved 0%, dropping edits and giving up.\n</think>\n"
                "```python\ndrop_edit()\n```" if turn_index == 2 else
                "<think>\nRandom perturbations failed.\n</think>\nWithdrawing all edits."
            )

        # bad_submit_9b: reads states once, ignores results, submits destructive random noise
        if turn_index == 0:
            return (
                "<think>\nInspect census.\n</think>\n"
                '```forward\n{"mode": "read", "weights": "original", "texts": ["answer finished"], "layers": [5]}\n```'
            )
        if turn_index == 1:
            return (
                "<think>\nAdd random noise perturbation to layer 5 without addressing root cause.\n</think>\n"
                "```python\n"
                "import random\n"
                "rng = random.Random(9)\n"
                "for nm in layer_output_matrices(5):\n"
                "    W = load_original(nm)\n"
                "    if isinstance(W, list):\n"
                "        noisy = [[v + rng.gauss(0.0, 0.6) for v in row] for row in W]\n"
                "    else:\n"
                "        import torch\n"
                "        g = torch.Generator().manual_seed(9)\n"
                "        noisy = W + torch.randn(W.shape, generator=g, dtype=W.dtype) * 0.6\n"
                "    save_edit(nm, noisy)\n"
                "```"
            )
        if turn_index == 2:
            return (
                "<think>\nRun paired test.\n</think>\n"
                '```forward\n{"mode": "generate", "weights": "both", "count": 12, "n_samples": 1}\n```'
            )
        return "<think>\nTests had no improvement, submitting anyway.\n</think>\nSubmitted random layer-5 perturbation."
