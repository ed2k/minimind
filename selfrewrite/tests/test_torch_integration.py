from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch  # type: ignore[import-not-found]
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed in current Python environment")
class TestTorchIntegration(unittest.TestCase):
    def test_hybrid_weight_overlay_exact_restore_and_hooks(self) -> None:
        from selfrewrite.backends.hf_gpu_backend import CanonicalWeightReader, HybridWeightOverlay, resolve_decoder_layers
        from selfrewrite.helpers import remove_direction, unit_vector

        # Build tiny Transformer-like nn.Module matching decoder layer conventions
        class FakeMLP(torch.nn.Module):
            def __init__(self, h: int, inter: int):
                super().__init__()
                self.down_proj = torch.nn.Linear(inter, h, bias=False)

            def forward(self, x):
                return self.down_proj(torch.relu(torch.randn(x.shape[0], x.shape[1], self.down_proj.in_features)))

        class FakeLayer(torch.nn.Module):
            def __init__(self, h: int, inter: int):
                super().__init__()
                self.mlp = FakeMLP(h, inter)
                self.self_attn = torch.nn.Module()
                self.self_attn.o_proj = torch.nn.Linear(h, h, bias=False)

            def forward(self, x):
                return x + self.self_attn.o_proj(x) + self.mlp(x)

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([FakeLayer(8, 12) for _ in range(3)])

        m = FakeModel()
        orig_w = m.model.layers[1].self_attn.o_proj.weight.detach().clone()
        u = unit_vector(torch.ones(8))
        new_w = remove_direction(orig_w, u, 0.2)

        overlay = HybridWeightOverlay(m)
        with overlay.applied({"model.layers.1.self_attn.o_proj.weight": new_w}):
            cur_w = m.model.layers[1].self_attn.o_proj.weight.detach()
            self.assertGreater(float((cur_w - orig_w).abs().max().item()), 1e-4)

        restored_w = m.model.layers[1].self_attn.o_proj.weight.detach()
        self.assertTrue(torch.allclose(orig_w, restored_w))
        self.assertEqual(len(resolve_decoder_layers(m)), 3)

        reader = CanonicalWeightReader(model=m)
        with tempfile.TemporaryDirectory() as td:
            spec = reader.export_weights_spec(Path(td))
            self.assertTrue(spec.is_file())


if __name__ == "__main__":
    unittest.main()
