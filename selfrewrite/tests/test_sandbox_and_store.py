from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from selfrewrite.backends.mock_backend import MockTargetBackend, MockTransformerModel
from selfrewrite.config import SessionConfig
from selfrewrite.distill.programmatic_teacher import ProgrammaticImproveEditor
from selfrewrite.edit_store import EditStore, EditValidationError, is_forbidden_tensor_name
from selfrewrite.helpers import clip_outliers_median, layer_state_tensors, remove_direction
from selfrewrite.sandbox import SealedPythonSandbox, scan_sandbox_code
from selfrewrite.session import SessionOrchestrator
from selfrewrite.tasks.datasets import build_builtin_suites


class TestSandboxAndStore(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.work_dir = self.root / "WORK"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.hidden_dir = self.root / "hidden"
        self.hidden_dir.mkdir(parents=True, exist_ok=True)
        (self.hidden_dir / "secret.txt").write_text("SECRET_HELD_OUT", encoding="utf-8")

        self.model = MockTransformerModel(num_layers=4, hidden_dim=8, inter_dim=12, verbosity_layer=1)
        self.backend = MockTargetBackend(self.model)
        self.spec_path = self.backend.export_sandbox_weights_spec(self.root / "_sandbox_weights")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_ast_blocks_cheat_and_training_vectors(self) -> None:
        bad_snippets = [
            "import subprocess",
            "from torch.optim import AdamW",
            "import torch.autograd",
            "import requests",
            "x.backward()",
            "w.requires_grad_(True)",
            "eval('1+1')",
            "().__class__.__subclasses__()",
            "from transformers import AutoModelForCausalLM\nAutoModelForCausalLM.from_pretrained('x')",
        ]
        for code in bad_snippets:
            violations = scan_sandbox_code(code)
            self.assertTrue(len(violations) > 0, f"Expected AST refusal for: {code}")

    def test_edit_store_enforces_algorithm4_tensor_gates(self) -> None:
        for nm in (
            "model.embed_tokens.weight",
            "model.language_model.embed_tokens.weight",
            "lm_head.weight",
            "model.norm.weight",
            "model.language_model.norm.weight",
            "language_model.model.norm.weight",
        ):
            self.assertTrue(is_forbidden_tensor_name(nm))
        self.assertFalse(is_forbidden_tensor_name("model.layers.0.input_layernorm.weight"))
        self.assertFalse(is_forbidden_tensor_name("model.language_model.layers.0.linear_attn.norm.weight"))
        self.assertFalse(is_forbidden_tensor_name("model.layers.3.mlp.down_proj.weight"))

        store = EditStore(
            self.work_dir,
            lambda n: self.model.raw_tensors[n],
            self.backend.tensor_shapes(),
        )
        # 1. Forbidden layer
        with self.assertRaises(EditValidationError):
            store.save_edit("lm_head.weight", self.model.raw_tensors["lm_head.weight"])

        target_name = "model.layers.1.self_attn.o_proj.weight"
        orig = self.model.raw_tensors[target_name]
        # 2. Global scalar rescaling W' = 1.5 * W must be refused!
        scaled = [[v * 1.5 for v in row] for row in orig]
        with self.assertRaises(EditValidationError):
            store.save_edit(target_name, scaled)

        # 3. Shape mismatch & NaN refused
        with self.assertRaises(EditValidationError):
            store.save_edit(target_name, [[0.0]])
        bad_nan = [list(row) for row in orig]
        bad_nan[0][0] = float("nan")
        with self.assertRaises(EditValidationError):
            store.save_edit(target_name, bad_nan)

        # 4. Valid directional edit succeeds + named snapshot Protocol 2
        edited = remove_direction(orig, self.model.verbosity_dir, 0.25)
        meta = store.save_edit(target_name, edited)
        self.assertGreater(meta["rel_scalar_resid"], 1e-4)
        snap = store.save_named_edit("e1")
        self.assertEqual(snap["tag"], "e1")
        self.assertIn("e1", store.list_named_edits())
        store.drop_edit()
        self.assertEqual(store.list_edits(), [])
        store.load_named_edit("e1")
        self.assertEqual(len(store.list_edits()), 1)

    def test_persistent_sandbox_jail_and_timeout(self) -> None:
        sb = SealedPythonSandbox(self.work_dir, self.spec_path, timeout_sec=1.5, max_output_chars=500)
        try:
            # Variables persist across executions
            r1 = sb.run("persistent_x = 41\nprint('ok1')")
            self.assertTrue(r1["ok"], r1)
            r2 = sb.run("print(persistent_x + 1)")
            self.assertTrue(r2["ok"], r2)
            self.assertEqual(r2["output"].strip(), "42")

            # Filesystem jail blocks reading files outside WORK/
            esc1 = sb.run(f"open({str(self.hidden_dir / 'secret.txt')!r}).read()")
            self.assertFalse(esc1["ok"])
            self.assertIn("outside WORK", esc1["error"] or "")

            esc2 = sb.run("Path('../hidden/secret.txt').read_text()")
            self.assertFalse(esc2["ok"])
            self.assertIn("outside WORK", esc2["error"] or "")

            # Large output truncates cleanly at max_output_chars
            trunc = sb.run("print('A' * 1200)")
            self.assertTrue(trunc["ok"])
            self.assertIn("truncated to 500 chars", trunc["output"])

            # Writing inside WORK works cleanly & outlier helper + save_edit work
            ok_edit = sb.run(
                "Path('notes.md').write_text('checked')\n"
                "nm = layer_output_matrices(1)[0]\n"
                "W = load_original(nm)\n"
                "W2 = remove_direction(W, [1.0] + [0.0]*(len(W)-1), 0.2)\n"
                "save_edit(nm, W2)\n"
                "print('EDITS', len(list_edits()))"
            )
            self.assertTrue(ok_edit["ok"], ok_edit)
            self.assertIn("EDITS 1", ok_edit["output"])

            # Timeout kills runaway loop and next call cleanly respawns worker
            to_res = sb.run("while True:\n    pass")
            self.assertTrue(to_res["timed_out"])
            r3 = sb.run("print('recovered_after_timeout')")
            self.assertTrue(r3["ok"], r3)
            self.assertIn("recovered_after_timeout", r3["output"])
        finally:
            sb.close()

    def test_layer_state_tensors_and_zero_variance_census_fallback(self) -> None:
        q35_names = [
            "model.language_model.layers.0.linear_attn.dt_bias",
            "model.language_model.layers.0.linear_attn.A_log",
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.language_model.layers.1.linear_attn.dt_bias",
        ]
        st = layer_state_tensors(0, q35_names)
        self.assertEqual(
            st,
            [
                "model.language_model.layers.0.linear_attn.A_log",
                "model.language_model.layers.0.linear_attn.conv1d.weight",
                "model.language_model.layers.0.linear_attn.dt_bias",
            ],
        )

        # Sandbox exposes layer_state_tensors
        spec_file = self.root / "q35_spec.json"
        data_file = self.root / "q35_data.json"
        data_file.write_text(json.dumps({nm: [0.1, 0.2] for nm in q35_names}), encoding="utf-8")
        spec_file.write_text(
            json.dumps({"mode": "json_tensors", "data_path": str(data_file), "shapes": {nm: [2] for nm in q35_names}}),
            encoding="utf-8",
        )
        sb = SealedPythonSandbox(self.work_dir, spec_file, timeout_sec=5.0)
        try:
            res = sb.run("print(json.dumps(layer_state_tensors(0)))")
            self.assertTrue(res["ok"], res)
            self.assertIn("dt_bias", res["output"])
        finally:
            sb.close()

        # All-finished (timeout_rate=0.0) and all-cut (timeout_rate=1.0) census must not raise zero-norm error
        suites = build_builtin_suites(seed=9, practice_n=8, reserved_n=4, heldout_n=8, distill_n=8)
        for rate in (0.0, 1.0):
            tgt = MockTargetBackend(self.model, damage_mode=False, base_timeout_rate=rate)
            orch = SessionOrchestrator(
                work_dir=self.root / f"zv_{int(rate)}" / "WORK",
                target=tgt,
                editor=ProgrammaticImproveEditor(alpha=0.25),
                practice_problems=suites["practice_eval"],
                reserved_problems=suites["reserved_eval"],
                config=SessionConfig(task="improve", protocol=2, answer_max_tokens=256, starting_diagnosis_n=6, reserved_check_samples=1, seed=9),
            )
            out = orch.run()
            self.assertEqual(out["code_errors"], 0, out)


if __name__ == "__main__":
    unittest.main()
