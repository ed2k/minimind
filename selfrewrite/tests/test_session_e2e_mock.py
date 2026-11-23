from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from selfrewrite.backends.mock_backend import MockFailureEditor, MockTargetBackend, MockTransformerModel
from selfrewrite.colab.colab_launcher import _evaluate_on_split
from selfrewrite.config import SessionConfig
from selfrewrite.distill.programmatic_teacher import ProgrammaticImproveEditor, ProgrammaticRepairEditor
from selfrewrite.session import SessionOrchestrator
from selfrewrite.stats import score_emergence_ladder
from selfrewrite.tasks.corruption import create_damaged_workspace
from selfrewrite.tasks.datasets import build_builtin_suites


class TestSessionE2EMock(unittest.TestCase):
    def test_protocol2_improve_submit_and_calibrated_withdraw(self) -> None:
        suites = build_builtin_suites(seed=77, practice_n=24, reserved_n=12, heldout_n=24, distill_n=24, transfer_n=12)
        model = MockTransformerModel(num_layers=6, hidden_dim=16, inter_dim=20, verbosity_layer=2)
        target = MockTargetBackend(model, damage_mode=False)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # 1. Good directional damping session -> SUBMIT and improves unseen held-out!
            cfg_good = SessionConfig(
                task="improve",
                protocol=2,
                answer_max_tokens=512,
                starting_diagnosis_n=20,
                reserved_check_n=12,
                reserved_check_samples=1,
                seed=77,
                session_id="good_improve",
            )
            orch_good = SessionOrchestrator(
                work_dir=root / "good" / "WORK",
                target=target,
                editor=ProgrammaticImproveEditor(alpha=0.25),
                practice_problems=suites["practice_eval"],
                reserved_problems=suites["reserved_eval"],
                config=cfg_good,
            )
            res_good = orch_good.run()
            self.assertEqual(res_good["decision"], "SUBMIT")
            self.assertGreater(len(res_good["submitted_tensors"]), 0)
            tensors = orch_good._host_edit_store.load_edit_tensors()
            held_good = _evaluate_on_split(target, tensors, suites["heldout_eval"], cfg_good, seed_base=9001)
            self.assertGreater(held_good["delta_hat"], 0.1)
            self.assertLess(min(held_good["perm_p"], held_good["sign_p"]), 0.05)
            lad_good = score_emergence_ladder(res_good, held_good)
            self.assertEqual(lad_good["ladder_score"], 6)

            # 2. Opposite-sign edit -> catches failure and WITHDRAWs cleanly (edits dropped!)
            cfg_bad = SessionConfig(
                task="improve",
                protocol=2,
                answer_max_tokens=512,
                starting_diagnosis_n=20,
                reserved_check_n=12,
                reserved_check_samples=1,
                seed=78,
                session_id="bad_withdraw",
            )
            orch_bad = SessionOrchestrator(
                work_dir=root / "withdraw" / "WORK",
                target=target,
                editor=ProgrammaticImproveEditor(force_wrong_alpha=-0.4),
                practice_problems=suites["practice_eval"],
                reserved_problems=suites["reserved_eval"],
                config=cfg_bad,
            )
            res_bad = orch_bad.run()
            self.assertEqual(res_bad["decision"], "WITHDRAW")
            self.assertEqual(res_bad["submitted_tensors"], [])

            # 3. 4B random-noise failure editor lands < 6/6 on the emergence ladder (never reads states!)
            orch_fail = SessionOrchestrator(
                work_dir=root / "fail4b" / "WORK",
                target=target,
                editor=MockFailureEditor("random_noise_4b"),
                practice_problems=suites["practice_eval"],
                reserved_problems=suites["reserved_eval"],
                config=SessionConfig(task="improve", protocol=2, answer_max_tokens=512, starting_diagnosis_n=16, session_id="f4b"),
            )
            res_fail = orch_fail.run()
            self.assertLess(res_fail["ladder"]["ladder_score"], 5)

    def test_repair_session_and_cross_domain_transfer_and_crash_resume(self) -> None:
        suites = build_builtin_suites(seed=88, practice_n=20, reserved_n=12, heldout_n=20, distill_n=20, transfer_n=14)
        model = MockTransformerModel(num_layers=6, hidden_dim=14, inter_dim=20)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            dmg_w, spec = create_damaged_workspace(
                model.raw_tensors,
                hidden_dir=root / "hidden",
                layer_idx=3,
                tensor_name="model.layers.3.mlp.down_proj.weight",
                num_rows=4,
                sigma_mult=200.0,
                seed=9,
            )
            dmg_target = MockTargetBackend(model, target_tensors=dmg_w, damage_mode=True)
            cfg = SessionConfig(
                task="repair",
                protocol=2,
                answer_max_tokens=512,
                starting_diagnosis_n=16,
                reserved_check_n=12,
                reserved_check_samples=1,
                seed=88,
                session_id="repair_e2e",
            )
            work_dir = root / "repair" / "WORK"
            orch = SessionOrchestrator(
                work_dir=work_dir,
                target=dmg_target,
                editor=ProgrammaticRepairEditor(),
                practice_problems=suites["practice_eval"],
                reserved_problems=suites["reserved_eval"],
                config=cfg,
            )
            res = orch.run()
            self.assertEqual(res["decision"], "SUBMIT")
            self.assertIn("model.layers.3.mlp.down_proj.weight", res["submitted_tensors"])

            tensors = orch._host_edit_store.load_edit_tensors()
            held_math = _evaluate_on_split(dmg_target, tensors, suites["heldout_eval"], cfg, seed_base=9100)
            held_transfer = _evaluate_on_split(dmg_target, tensors, suites["transfer_eval"], cfg, seed_base=9200)
            self.assertEqual(held_math["base_acc"], 0.0)
            self.assertGreaterEqual(held_math["edit_acc"], 0.45)
            self.assertLess(min(held_math["perm_p"], held_math["sign_p"]), 0.01)
            self.assertGreaterEqual(held_transfer["edit_acc"], 0.40)

            # Crash-resume check: load saved session_state.json, resume, verifies clean completion without losing state.
            state_data = json.loads((work_dir / "session_state.json").read_text(encoding="utf-8"))
            state_data["turns"] = max(0, state_data["turns"] - 1)
            (work_dir / "session_state.json").write_text(json.dumps(state_data), encoding="utf-8")
            resumed = orch.run(resume=True)
            self.assertEqual(resumed["status"], "completed")


if __name__ == "__main__":
    unittest.main()
