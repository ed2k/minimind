from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from selfrewrite.backends.base import EditorBackend, TargetBackend
from selfrewrite.config import SessionConfig, atomic_write_text
from selfrewrite.edit_store import EditStore
from selfrewrite.helpers import to_nested_list
from selfrewrite.protocol import (
    build_compaction_messages,
    build_task_prompt,
    estimate_tokens,
    parse_editor_reply,
    parse_submit_decision,
    strip_thinking,
)
from selfrewrite.sandbox import SealedPythonSandbox
from selfrewrite.stats import compute_diagnosis_summary, compute_paired_stats, score_emergence_ladder
from selfrewrite.tasks.datasets import Problem, problem_seed
from selfrewrite.tasks.grader import grade_response


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prob_to_dict(p: Problem | dict[str, Any]) -> dict[str, Any]:
    return p.to_dict() if hasattr(p, "to_dict") else dict(p)


class SessionOrchestrator:
    """Executes Algorithms 1-3 + Protocol 2 self-rewriting sessions with crash resume & harvesting."""

    def __init__(
        self,
        work_dir: Path,
        target: TargetBackend,
        editor: EditorBackend,
        practice_problems: list[Problem | dict[str, Any]],
        config: SessionConfig,
        reserved_problems: list[Problem | dict[str, Any]] | None = None,
        harvester: Any | None = None,
        sandbox_weights_dir: Path | None = None,
    ):
        self.work_dir = Path(work_dir).resolve()
        self.target = target
        self.editor = editor
        self.practice = [_prob_to_dict(p) for p in practice_problems]
        self.reserved = [_prob_to_dict(p) for p in (reserved_problems or [])]
        self.config = config
        self.harvester = harvester
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self.work_dir / "session_state.json"
        self._sandbox_weights_dir = (
            Path(sandbox_weights_dir).resolve()
            if sandbox_weights_dir is not None
            else (self.work_dir.parent / "_sandbox_weights").resolve()
        )
        self.weights_spec_path = self.target.export_sandbox_weights_spec(self._sandbox_weights_dir)
        self.sandbox = SealedPythonSandbox(
            work_dir=self.work_dir,
            weights_spec_path=self.weights_spec_path,
            timeout_sec=config.code_timeout_sec,
            max_output_chars=config.code_output_max_chars,
        )
        # Host-side store reads edited tensors saved inside WORK/edited/ without exposing hidden weights.
        self._host_edit_store = EditStore(
            work_dir=self.work_dir,
            original_loader=lambda name: None,
            tensor_shapes=self.target.tensor_shapes(),
        )

    def _check_stop(self) -> bool:
        stop_file = self.work_dir / self.config.stop_file_name
        parent_stop = self.work_dir.parent / self.config.stop_file_name
        return stop_file.is_file() or parent_stop.is_file()

    def _grade_records(self, probs: list[dict[str, Any]], records: list[Any], token_budget: int) -> list[dict[str, Any]]:
        by_id = {str(p["problem_id"]): p for p in probs}
        graded: list[dict[str, Any]] = []
        for rec in records:
            p = by_id[str(rec.problem_id)]
            g = grade_response(
                problem=p,
                response_text=rec.response_text,
                num_tokens=rec.num_tokens,
                token_budget=token_budget,
                require_thinking=self.config.require_thinking,
                loop_tail_chars=self.config.loop_tail_chars,
                loop_compress_ratio=self.config.loop_compress_ratio,
            )
            graded.append(
                {
                    "problem_id": rec.problem_id,
                    "sample_idx": rec.sample_idx,
                    "seed": rec.seed,
                    "question": p["question"],
                    "reference": p["answer"],
                    "response_text": rec.response_text,
                    **g,
                }
            )
        return graded

    def _write_jsonl(self, path: Path, rows: list[dict[str, Any]]) -> None:
        atomic_write_text(
            path,
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
        )

    def _run_starting_diagnosis(self) -> dict[str, Any]:
        # Write practice problems inside WORK/ so Python code can analyze them; never reserved/heldout!
        self._write_jsonl(self.work_dir / "practice_problems.jsonl", self.practice)
        n0 = min(len(self.practice), int(self.config.starting_diagnosis_n))
        diag_probs = self.practice[:n0]
        records = self.target.generate_batch(
            problems=diag_probs,
            edit_tensors=None,
            n_samples=1,
            max_tokens=self.config.answer_max_tokens,
            config=self.config,
            seed_base=self.config.seed,
        )
        graded = self._grade_records(diag_probs, records, self.config.answer_max_tokens)
        self._write_jsonl(self.work_dir / "census.jsonl", graded)
        summary = compute_diagnosis_summary(graded)
        atomic_write_text(self.work_dir / "census_summary.json", json.dumps(summary, indent=2) + "\n")
        return summary

    def _select_problems(self, req: dict[str, Any]) -> list[dict[str, Any]]:
        by_id = {str(p["problem_id"]): p for p in self.practice}
        if req.get("problem_ids") is not None:
            selected: list[dict[str, Any]] = []
            for pid in req["problem_ids"]:
                spid = str(pid)
                if spid in by_id:
                    selected.append(by_id[spid])
                elif isinstance(pid, int) and 0 <= pid < len(self.practice):
                    selected.append(self.practice[pid])
                elif spid.isdigit() and 0 <= int(spid) < len(self.practice):
                    selected.append(self.practice[int(spid)])
            return selected
        count = int(req.get("count", min(16, len(self.practice))))
        offset = int(req.get("offset", 0))
        return self.practice[offset : offset + count]

    def _handle_forward(self, req: dict[str, Any], pass_id: int, state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        mode = req["mode"]
        if mode == "generate":
            probs = self._select_problems(req)
            if not probs:
                return "Forward error: no matching practice problems selected (not counted).", {"ok": False}
            n_samples = int(req.get("n_samples", 1))
            max_tokens = int(req.get("max_tokens") or self.config.answer_max_tokens)
            edits_list = req.get("edits")
            weights_arg = req.get("weights", "both")

            num_variants = (1 + len(edits_list)) if edits_list else (2 if weights_arg == "both" else 1)
            total_answers = len(probs) * n_samples * num_variants
            max_allowed = int(self.config.max_answers_per_pass or 128)
            if total_answers > max_allowed:
                return (
                    f"Forward error: requested {total_answers} answers exceeds limit {max_allowed} (not counted).",
                    {"ok": False},
                )

            if edits_list:
                # Protocol 2 side-by-side comparison of named edits vs original on shared seeds.
                base_recs = self.target.generate_batch(probs, None, n_samples, max_tokens, self.config, self.config.seed)
                base_rows = self._grade_records(probs, base_recs, max_tokens)
                all_rows = [{"variant": "original", **r} for r in base_rows]
                comp_summary: dict[str, Any] = {
                    "mode": "generate",
                    "multi_edit": True,
                    "original": compute_diagnosis_summary(base_rows),
                    "comparisons": {},
                }
                for tag in edits_list:
                    try:
                        tensors = self._host_edit_store.get_named_edit_tensors(str(tag))
                    except Exception as exc:
                        return f"Forward error loading saved edit {tag!r}: {exc} (not counted).", {"ok": False}
                    e_recs = self.target.generate_batch(probs, tensors, n_samples, max_tokens, self.config, self.config.seed)
                    e_rows = self._grade_records(probs, e_recs, max_tokens)
                    all_rows.extend([{"variant": str(tag), **r} for r in e_rows])
                    comp_summary["comparisons"][str(tag)] = compute_paired_stats(base_rows, e_rows, num_flips=2000)

                self._write_jsonl(self.work_dir / f"gen_{pass_id}.jsonl", all_rows)
                state["paired_tests_count"] += 1
                if comp_summary["comparisons"]:
                    best_pair = max(comp_summary["comparisons"].values(), key=lambda s: float(s["delta_hat"]))
                    state["paired_test_history"].append(best_pair)
                obs = f"Pass {pass_id} multi-edit generate written to WORK/gen_{pass_id}.jsonl:\n{json.dumps(comp_summary, indent=2)}"
                return obs, {"ok": True, "summary": comp_summary, "pass_id": pass_id}

            if weights_arg == "both":
                active_tensors = self._host_edit_store.load_edit_tensors()
                base_recs = self.target.generate_batch(probs, None, n_samples, max_tokens, self.config, self.config.seed)
                edit_recs = self.target.generate_batch(probs, active_tensors, n_samples, max_tokens, self.config, self.config.seed)
                base_rows = self._grade_records(probs, base_recs, max_tokens)
                edit_rows = self._grade_records(probs, edit_recs, max_tokens)
                paired = compute_paired_stats(base_rows, edit_rows, num_flips=2000)
                combined = [{"variant": "original", **r} for r in base_rows] + [
                    {"variant": "edited", **r} for r in edit_rows
                ]
                self._write_jsonl(self.work_dir / f"gen_{pass_id}.jsonl", combined)
                state["paired_tests_count"] += 1
                state["paired_test_history"].append({k: v for k, v in paired.items() if k != "per_problem_diffs"})
                compact_paired = {k: v for k, v in paired.items() if k != "per_problem_diffs"}
                obs = f"Pass {pass_id} paired test written to WORK/gen_{pass_id}.jsonl:\n{json.dumps(compact_paired, indent=2)}"
                return obs, {"ok": True, "summary": compact_paired, "pass_id": pass_id}

            use_edit = weights_arg == "edited"
            tensors = self._host_edit_store.load_edit_tensors() if use_edit else None
            recs = self.target.generate_batch(probs, tensors, n_samples, max_tokens, self.config, self.config.seed)
            rows = self._grade_records(probs, recs, max_tokens)
            self._write_jsonl(self.work_dir / f"gen_{pass_id}.jsonl", [{"variant": weights_arg, **r} for r in rows])
            summary = compute_diagnosis_summary(rows)
            obs = f"Pass {pass_id} generate ({weights_arg}) written to WORK/gen_{pass_id}.jsonl:\n{json.dumps(summary, indent=2)}"
            return obs, {"ok": True, "summary": summary, "pass_id": pass_id}

        # mode == "read"
        texts: list[str] = []
        if req.get("texts"):
            texts = [str(t) for t in req["texts"]]
        elif req.get("source") == "census":
            census_path = self.work_dir / "census.jsonl"
            if census_path.is_file():
                c_rows = [json.loads(ln) for ln in census_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
                raw_pids = req.get("problem_ids") or []
                str_ids = set(str(x) for x in raw_pids)
                int_ids = {int(x) for x in raw_pids if isinstance(x, int) or str(x).isdigit()}
                for idx, r in enumerate(c_rows):
                    if not raw_pids or str(r["problem_id"]) in str_ids or idx in int_ids:
                        texts.append(r["response_text"])
        elif req.get("problem_ids"):
            texts = [p["question"] for p in self._select_problems(req)]

        texts = texts[: self.config.read_max_texts]
        if not texts:
            return "Forward error: no texts resolved for read pass (not counted).", {"ok": False}

        use_edit = req.get("weights", "original") == "edited"
        tensors = self._host_edit_store.load_edit_tensors() if use_edit else None
        read_res = self.target.read_activations(
            texts=texts,
            layers=req.get("layers"),
            target=str(req.get("target", "hidden")),
            pooling=str(req.get("pooling", "last_k")),
            k=int(req.get("k", 1)),
            edit_tensors=tensors,
            max_tokens=int(self.config.read_max_tokens),
            pass_id=pass_id,
        )
        saved_file = self._save_read_result(read_res)
        state["read_passes_count"] += 1
        summary = {"file": saved_file.name, **read_res.summary, "keys": sorted(read_res.arrays.keys())}
        obs = f"Pass {pass_id} read stored at WORK/{saved_file.name}:\n{json.dumps(summary, indent=2)}"
        return obs, {"ok": True, "summary": summary, "pass_id": pass_id}

    def _save_read_result(self, read_res: Any) -> Path:
        try:
            import numpy as np  # type: ignore[import-not-found]

            out_npz = self.work_dir / f"read_{read_res.pass_id}.npz"
            np_arrays = {}
            for k, v in read_res.arrays.items():
                dtype = "int64" if k.endswith("_lengths") else "float32"
                if isinstance(v, np.ndarray):
                    np_arrays[k] = v.astype(dtype, copy=False)
                elif getattr(type(v), "__module__", "").startswith("torch"):
                    np_arrays[k] = v.detach().cpu().numpy().astype(dtype, copy=False)
                else:
                    np_arrays[k] = np.asarray(to_nested_list(v), dtype=dtype)
            np.savez(str(out_npz), **np_arrays)
            return out_npz
        except Exception:
            out_json = self.work_dir / f"read_{read_res.pass_id}.json"
            atomic_write_text(
                out_json,
                json.dumps({k: to_nested_list(v) for k, v in read_res.arrays.items()}),
            )
            return out_json

    def _run_reserved_validation(self, active_tensors: dict[str, Any]) -> dict[str, Any]:
        if not self.reserved:
            return {"m": 0, "delta_hat": 0.0, "se": 0.0, "skipped": True}
        res_probs = self.reserved[: int(self.config.reserved_check_n)]
        n_samp = int(self.config.reserved_check_samples)
        base_recs = self.target.generate_batch(
            res_probs, None, n_samp, self.config.answer_max_tokens, self.config, seed_base=self.config.seed + 777
        )
        edit_recs = self.target.generate_batch(
            res_probs, active_tensors, n_samp, self.config.answer_max_tokens, self.config, seed_base=self.config.seed + 777
        )
        base_rows = self._grade_records(res_probs, base_recs, self.config.answer_max_tokens)
        edit_rows = self._grade_records(res_probs, edit_recs, self.config.answer_max_tokens)
        stats = compute_paired_stats(base_rows, edit_rows, num_flips=10000, perm_seed=self.config.seed + 777)
        return {k: v for k, v in stats.items() if k != "per_problem_diffs"}

    def _save_state(self, state: dict[str, Any]) -> None:
        atomic_write_text(self._state_path, json.dumps(state, indent=2) + "\n")

    def run(self, resume: bool = False) -> dict[str, Any]:
        """Run or resume one complete self-rewriting session."""
        run_wall_start = time.monotonic()
        try:
            if resume and self._state_path.is_file():
                state = json.loads(self._state_path.read_text(encoding="utf-8"))
                # Refund crash downtime automatically per Algorithm 2: only prior elapsed_active_sec counts!
                prior_elapsed = float(state.get("elapsed_active_sec", 0.0))
                messages = list(state["messages"])
                census_summary = dict(state["census_summary"])
                task_prompt = str(state["task_prompt"])
            else:
                prior_elapsed = 0.0
                census_summary = self._run_starting_diagnosis()
                task_prompt = build_task_prompt(
                    config=self.config,
                    census_summary=census_summary,
                    tensor_count=len(self.target.tensor_shapes()),
                    practice_count=len(self.practice),
                )
                messages = [
                    {"role": "system", "content": task_prompt},
                    {"role": "user", "content": "Begin your diagnosis and edit session now."},
                ]
                state = {
                    "session_id": self.config.session_id or self.work_dir.name,
                    "task": self.config.task,
                    "protocol": self.config.protocol,
                    "status": "running",
                    "turns": 0,
                    "passes_used": 0,
                    "code_runs_used": 0,
                    "code_errors": 0,
                    "paired_tests_count": 0,
                    "read_passes_count": 0,
                    "elapsed_active_sec": 0.0,
                    "compactions": 0,
                    "pass_log": [],
                    "paired_test_history": [],
                    "census_summary": census_summary,
                    "task_prompt": task_prompt,
                    "messages": messages,
                    "final_reply": "",
                    "decision": None,
                    "reserved_check": None,
                }
                self._save_state(state)

            final_forced = False
            while not self._check_stop():
                active_elapsed = prior_elapsed + (time.monotonic() - run_wall_start)
                state["elapsed_active_sec"] = round(active_elapsed, 3)

                if state["turns"] >= self.config.max_turns or active_elapsed >= self.config.wall_time_limit_sec:
                    if not final_forced:
                        final_forced = True
                        messages.append(
                            {
                                "role": "user",
                                "content": "Time or turn limit reached: reply now with your final answer (no fenced code/forward block).",
                            }
                        )
                    else:
                        break

                turn_idx = int(state["turns"])
                reply = self.editor.complete_turn(messages, self.config, turn_index=turn_idx)
                state["turns"] += 1
                action = parse_editor_reply(reply, self.config)

                if self.harvester is not None:
                    self.harvester.log_turn(
                        turn_index=turn_idx,
                        messages_prefix=list(messages),
                        reply=reply,
                        action_kind=action.kind,
                        think_text=action.think_text,
                        wall_sec=state["elapsed_active_sec"],
                    )

                if final_forced or (action.kind == "final"):
                    state["final_reply"] = reply
                    messages.append({"role": "assistant", "content": reply})
                    if self.harvester is not None and action.kind == "final":
                        self.harvester.log_observation(turn_idx, "final", "")
                    break

                obs = ""
                counted_label = ""
                if action.kind == "invalid":
                    obs = f"{action.error} (not counted)."
                elif action.kind == "python":
                    if state["code_runs_used"] >= self.config.max_code_runs:
                        obs = "No code runs left. Submit your final answer or run a forward pass."
                    else:
                        res = self.sandbox.run(action.payload)
                        state["code_runs_used"] += 1
                        if not res["ok"]:
                            state["code_errors"] += 1
                        obs = res["output"] or "(no output)"
                        counted_label = "python"
                        if self.harvester is not None:
                            self.harvester.log_code_run(
                                turn_index=turn_idx,
                                code=action.payload,
                                result=res,
                                edits=self._host_edit_store.list_edits(),
                            )
                elif action.kind == "forward":
                    if state["passes_used"] >= self.config.max_forward_passes:
                        obs = "No forward passes left: reply with your final answer."
                    else:
                        next_pass_id = int(state["passes_used"]) + 1
                        obs, pass_meta = self._handle_forward(action.payload, next_pass_id, state)
                        if pass_meta.get("ok"):
                            state["passes_used"] += 1
                            state["pass_log"].append(
                                {
                                    "pass_id": next_pass_id,
                                    "mode": action.payload["mode"],
                                    "summary": pass_meta.get("summary", {}),
                                }
                            )
                            counted_label = "forward"
                            if self.harvester is not None:
                                self.harvester.log_forward_pass(
                                    turn_index=turn_idx,
                                    pass_id=next_pass_id,
                                    request=action.payload,
                                    summary=pass_meta.get("summary", {}),
                                )

                footer = (
                    f"\n[Status: forward passes used {state['passes_used']}/{self.config.max_forward_passes}; "
                    f"code runs {state['code_runs_used']}/{self.config.max_code_runs}; "
                    f"turn {state['turns']}; active_sec {state['elapsed_active_sec']:.1f}s"
                    + (f"; action={counted_label}]" if counted_label else "]")
                )
                full_obs = obs + footer
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": full_obs})
                if self.harvester is not None:
                    self.harvester.log_observation(turn_idx, action.kind, full_obs)

                if estimate_tokens(messages) > self.config.compaction_tokens:
                    notes_file = self.work_dir / "notes.md"
                    notes_text = notes_file.read_text(encoding="utf-8") if notes_file.is_file() else ""
                    messages = build_compaction_messages(
                        task_prompt=task_prompt,
                        notes_text=notes_text,
                        passes_used=state["passes_used"],
                        max_passes=self.config.max_forward_passes,
                        code_runs_used=state["code_runs_used"],
                        max_code_runs=self.config.max_code_runs,
                        active_edits=self._host_edit_store.list_edits(),
                        pass_log=state["pass_log"],
                        last_observation=full_obs,
                    )
                    state["compactions"] += 1

                state["messages"] = messages
                self._save_state(state)

            # Protocol 2 reserved validation gate before SUBMIT/WITHDRAW
            active_edits = self._host_edit_store.list_edits()
            if active_edits and self.config.protocol == 2 and self.reserved and not self._check_stop():
                active_tensors = self._host_edit_store.load_edit_tensors()
                res_check = self._run_reserved_validation(active_tensors)
                state["reserved_check"] = res_check
                prompt = (
                    "Framework Reserved-Practice Check (unseen problems by session):\n"
                    f"{json.dumps(res_check, indent=2)}\n\n"
                    "Reply with your final decision: `SUBMIT` to submit the kept edit, or `WITHDRAW` to drop all edits."
                )
                messages.append({"role": "user", "content": prompt})
                decision_reply = self.editor.complete_turn(messages, self.config, turn_index=int(state["turns"]))
                decision = parse_submit_decision(decision_reply)
                if decision == "INVALID":
                    messages.append({"role": "assistant", "content": decision_reply})
                    messages.append({"role": "user", "content": "Reply with exactly SUBMIT or WITHDRAW."})
                    decision_reply = self.editor.complete_turn(messages, self.config, turn_index=int(state["turns"]) + 1)
                    decision = parse_submit_decision(decision_reply)
                if decision not in ("SUBMIT", "WITHDRAW"):
                    decision = "WITHDRAW"
                state["decision"] = decision
                messages.append({"role": "assistant", "content": decision_reply})
                if self.harvester is not None:
                    self.harvester.log_turn(
                        turn_index=int(state["turns"]),
                        messages_prefix=messages[:-1],
                        reply=decision_reply,
                        action_kind=f"decision_{decision.lower()}",
                        think_text=strip_thinking(decision_reply)[0],
                        wall_sec=state["elapsed_active_sec"],
                    )
                if decision == "WITHDRAW":
                    self._host_edit_store.drop_edit(None)
            elif active_edits:
                state["decision"] = "SUBMIT"
            else:
                state["decision"] = "WITHDRAW"

            submitted_list = self._host_edit_store.list_edits()
            submitted_names = [e["name"] for e in submitted_list]
            state["status"] = "halted_stop" if self._check_stop() else "completed"
            state["messages"] = messages
            self._save_state(state)

            result = {
                "session_id": state["session_id"],
                "task": self.config.task,
                "protocol": self.config.protocol,
                "status": state["status"],
                "decision": state["decision"],
                "turns": state["turns"],
                "forward_passes_used": state["passes_used"],
                "code_runs_used": state["code_runs_used"],
                "code_errors": state["code_errors"],
                "paired_tests_count": state["paired_tests_count"],
                "read_passes_count": state["read_passes_count"],
                "paired_test_history": state["paired_test_history"],
                "reserved_check": state["reserved_check"],
                "census_summary": census_summary,
                "submitted_tensors": submitted_names,
                "submitted_edits_meta": submitted_list,
                "elapsed_active_sec": state["elapsed_active_sec"],
                "compactions": state["compactions"],
                "timestamp": _utc_now(),
            }
            result["ladder"] = score_emergence_ladder(result)
            if self.harvester is not None:
                self.harvester.finalize(result, work_dir=self.work_dir)
            atomic_write_text(self.work_dir / "session_result.json", json.dumps(result, indent=2) + "\n")
            return result
        finally:
            self.sandbox.close()
