"""Prompt compiler: selection, instruction refinement, validation,
versioned store, NPC/character suppliers, CLI, registry, /prompt.

The compiler is DSPy-shaped (signatures, BootstrapFewShot demo
selection, MIPRO-style instruction refinement, held-out metric
evaluation) over the journal -> grade -> retrain loop that already
exists — it consumes the same graded rows the learner jobs consume.
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nomorals.core.config import Settings, load_settings
from nomorals.llm.prompt_compiler import (
    DEFAULT_MAX_COUNTER_EXAMPLES,
    DEFAULT_MAX_DEMOS,
    DEFAULT_MIN_EXAMPLES,
    CompiledPrompt,
    GradedExample,
    PromptSignature,
    PromptStore,
    compile_prompt,
    control_prompt,
    imitation_metric,
    render_base_prompt,
    render_prompt,
)

TASK_SIG = PromptSignature(
    task="test_task",
    instruction="Speak as the character: one line, in character. "
                "Reply with the line only.",
    input_fields=("situation",),
    output_fields=("line",),
)


def _example(i: int, grade: float, *, short: bool = True,
             situation: str | None = None) -> GradedExample:
    """One fixture example. Short outputs are terse; long outputs
    ramble — the trait gap the instruction refinement measures."""
    if short:
        output = f"Brave words {i} stand tall."
    else:
        output = (f"Terrible ramble {i} that goes on and on and on with "
                  f"no end in sight whatsoever, truly endless and "
                  f"wandering.")
    return GradedExample(
        inputs={"situation": situation or f"scene {i} in the tavern"},
        output=output, grade=grade, id=f"ex{i:02d}", source="owner")


def _graded_set(n: int = 12) -> list[GradedExample]:
    """12 examples: every 3rd grades +0.9 (short), every 3rd -0.8
    (long), the rest +0.2 (short). Base mean is +0.10."""
    out: list[GradedExample] = []
    for i in range(n):
        if i % 3 == 0:
            out.append(_example(i, 0.9, short=True))
        elif i % 3 == 1:
            out.append(_example(i, -0.8, short=False))
        else:
            out.append(_example(i, 0.2, short=True))
    return out


class ExampleFromRowTests(unittest.TestCase):
    def test_journal_row_shape(self) -> None:
        row = {"features": {"a": 1.0}, "label": 0.7,
               "line": "Brave words stand tall.",
               "situation": "a tavern brawl", "entry_id": "e1"}
        ex = GradedExample.from_row(row)
        self.assertIsNotNone(ex)
        assert ex is not None
        self.assertEqual(ex.inputs, {"situation": "a tavern brawl"})
        self.assertEqual(ex.output, "Brave words stand tall.")
        self.assertEqual(ex.grade, 0.7)
        self.assertEqual(ex.id, "e1")

    def test_context_row_shape(self) -> None:
        row = {"features": {}, "label": -0.4, "line": "A quiet line.",
               "context": "the market at dusk", "entry_id": "e2"}
        ex = GradedExample.from_row(row)
        self.assertIsNotNone(ex)
        assert ex is not None
        self.assertEqual(ex.inputs, {"context": "the market at dusk"})
        self.assertEqual(ex.grade, -0.4)

    def test_example_shape_passthrough(self) -> None:
        row = {"inputs": {"situation": "s"}, "output": "o", "grade": 0.5,
               "id": "x"}
        ex = GradedExample.from_row(row)
        self.assertIsNotNone(ex)
        assert ex is not None
        self.assertEqual(ex.grade, 0.5)

    def test_rejects_ungraded_and_bad_rows(self) -> None:
        self.assertIsNone(GradedExample.from_row(
            {"features": {}, "line": "x"}))
        self.assertIsNone(GradedExample.from_row(
            {"features": {}, "label": 0.5}))
        self.assertIsNone(GradedExample.from_row(
            {"features": {}, "label": "high", "line": "x"}))
        self.assertIsNone(GradedExample.from_row("not a row"))

    def test_grade_clamped_to_journal_scale(self) -> None:
        ex = GradedExample.from_row(
            {"features": {}, "label": 5.0, "line": "x"})
        self.assertIsNotNone(ex)
        assert ex is not None
        self.assertEqual(ex.grade, 1.0)


class CompileGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = PromptStore(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_compiles_and_promotes(self) -> None:
        out = compile_prompt(TASK_SIG, _graded_set(), store=self.store)
        self.assertEqual(out.status, "compiled", out.reason)
        self.assertIsNotNone(out.compiled)
        assert out.compiled is not None
        self.assertEqual(out.compiled.version, 1)
        self.assertEqual(len(out.compiled.demos), DEFAULT_MAX_DEMOS)
        self.assertEqual(len(out.compiled.counter_examples),
                         DEFAULT_MAX_COUNTER_EXAMPLES)
        assert out.report is not None
        self.assertAlmostEqual(out.report.baseline_grade, 0.10, places=2)
        self.assertGreater(out.report.demo_grade or 0, 0.7)
        self.assertFalse(out.report.validated)
        self.assertTrue(out.report.beats_baseline)

    def test_demos_are_the_best_graded(self) -> None:
        # The train split can hold out 0.9 examples, so the demos are
        # the best of TRAIN — the top two are always 0.9 (four exist,
        # at most two land in the test slice), sorted best-first, and
        # the counter-examples are always the -0.8 rows.
        out = compile_prompt(TASK_SIG, _graded_set(), store=self.store)
        assert out.compiled is not None
        grades = [d.grade for d in out.compiled.demos]
        self.assertEqual(grades, sorted(grades, reverse=True))
        self.assertGreaterEqual(len(grades), 2)
        self.assertTrue(all(g == 0.9 for g in grades[:2]))
        self.assertTrue(all(c.grade < -0.7
                            for c in out.compiled.counter_examples))

    def test_insufficient_below_floor(self) -> None:
        out = compile_prompt(TASK_SIG, _graded_set(3), store=self.store)
        self.assertEqual(out.status, "insufficient")
        self.assertIn("3/8", out.reason)
        self.assertIsNone(out.compiled)
        self.assertEqual(self.store.latest_version("test_task"), 0)

    def test_per_task_floor_wins(self) -> None:
        sig = PromptSignature(task="strict", instruction="Say it.",
                              min_examples=12)
        out = compile_prompt(sig, _graded_set(10), store=self.store)
        self.assertEqual(out.status, "insufficient")
        self.assertIn("10/12", out.reason)

    def test_held_when_no_positive_example(self) -> None:
        examples = [_example(i, -0.5) for i in range(10)]
        out = compile_prompt(TASK_SIG, examples, store=self.store)
        self.assertEqual(out.status, "held")
        self.assertIn("nothing good to imitate", out.reason)
        self.assertIsNone(out.compiled)

    def test_held_when_demos_match_average(self) -> None:
        examples = [_example(i, 0.5) for i in range(10)]
        out = compile_prompt(TASK_SIG, examples, store=self.store)
        self.assertEqual(out.status, "held")
        self.assertIsNone(out.compiled)

    def test_deterministic_same_seed(self) -> None:
        first = compile_prompt(TASK_SIG, _graded_set(), store=self.store)
        second = compile_prompt(TASK_SIG, _graded_set(),
                                store=PromptStore(self._tmp.name))
        assert first.compiled is not None
        assert second.compiled is not None
        self.assertEqual([d.id for d in first.compiled.demos],
                         [d.id for d in second.compiled.demos])
        self.assertEqual(first.compiled.prompt_text,
                         second.compiled.prompt_text)

    def test_dedup_keeps_highest_grade(self) -> None:
        examples = _graded_set()
        dup = GradedExample(inputs=dict(examples[0].inputs),
                            output=examples[0].output, grade=-0.9,
                            id="dup")
        out = compile_prompt(TASK_SIG, examples + [dup], store=self.store)
        assert out.report is not None
        self.assertEqual(out.report.n_examples, 12)

    def test_skips_rows_without_output(self) -> None:
        rows = [{"features": {}, "label": 0.5, "line": ""},
                {"features": {}, "label": 0.5}]
        examples = [GradedExample.from_row(r) for r in rows]
        out = compile_prompt(TASK_SIG, examples, store=self.store)  # type: ignore[arg-type]
        self.assertEqual(out.status, "insufficient")

    def test_disabled_settings(self) -> None:
        settings = SimpleNamespace(enabled=False)
        out = compile_prompt(TASK_SIG, _graded_set(), store=self.store,
                             settings=settings)
        self.assertEqual(out.status, "disabled")
        self.assertIsNone(out.compiled)


class InstructionRefinementTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = PromptStore(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_note_fires_on_trait_gap(self) -> None:
        out = compile_prompt(TASK_SIG, _graded_set(), store=self.store)
        assert out.compiled is not None
        note = out.compiled.instruction_note
        self.assertIn("shorter", note)
        self.assertIn("characters", note)
        self.assertIn(note, out.compiled.prompt_text)

    def test_note_empty_when_no_gap(self) -> None:
        examples = [_example(i, 0.9 if i % 2 else -0.8, short=True)
                    for i in range(12)]
        out = compile_prompt(TASK_SIG, examples, store=self.store)
        assert out.compiled is not None
        self.assertEqual(out.compiled.instruction_note, "")

    def test_note_names_measured_numbers(self) -> None:
        out = compile_prompt(TASK_SIG, _graded_set(), store=self.store)
        assert out.compiled is not None
        self.assertRegex(out.compiled.instruction_note, r"\d+ (characters|words)")


class RenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = PromptStore(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _compiled(self) -> CompiledPrompt:
        out = compile_prompt(TASK_SIG, _graded_set(), store=self.store)
        assert out.compiled is not None
        return out.compiled

    def test_template_placeholder_when_no_inputs(self) -> None:
        compiled = self._compiled()
        self.assertIn("{situation}", compiled.prompt_text)
        self.assertTrue(compiled.prompt_text.rstrip().endswith("Line:"))

    def test_render_fills_inputs(self) -> None:
        compiled = self._compiled()
        text = render_prompt(TASK_SIG, compiled.demos,
                             compiled.counter_examples,
                             compiled.instruction_note,
                             {"situation": "a duel at dawn"})
        self.assertIn("a duel at dawn", text)
        self.assertNotIn("{situation}", text)

    def test_base_prompt_has_no_demos(self) -> None:
        text = render_base_prompt(TASK_SIG, {"situation": "x"})
        self.assertNotIn("Examples", text)
        self.assertNotIn("Avoid", text)
        self.assertIn("x", text)

    def test_fit_trims_to_cap(self) -> None:
        import nomorals.llm.prompt_compiler as pc

        huge = [GradedExample(inputs={"situation": f"s{i}"},
                              output="word " * 500, grade=0.9,
                              id=f"h{i}") for i in range(6)]
        sig = PromptSignature(task="big", instruction="Say it.",
                              input_fields=("situation",),
                              output_fields=("line",))
        old_cap = pc.MAX_PROMPT_CHARS
        try:
            pc.MAX_PROMPT_CHARS = 600
            text = render_prompt(sig, huge, [], "", {"situation": "x"})
        finally:
            pc.MAX_PROMPT_CHARS = old_cap
        self.assertLessEqual(len(text), 600)
        self.assertIn("Say it.", text)


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = PromptStore(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_validated_beats_base(self) -> None:
        def parrot(prompt: str) -> str:
            if "Brave words 0 stand tall." in prompt:
                return "Brave words 0 stand tall."
            return "something entirely different here"

        out = compile_prompt(TASK_SIG, _graded_set(), suggest=parrot,
                             metric=imitation_metric, store=self.store)
        self.assertEqual(out.status, "compiled", out.reason)
        assert out.report is not None
        self.assertTrue(out.report.validated)
        self.assertTrue(out.report.beats_baseline)
        self.assertGreater(out.report.heldout_compiled_score or 0,
                           out.report.heldout_base_score or 0)
        self.assertIsNotNone(out.compiled)

    def test_validated_held_on_tie(self) -> None:
        def constant(prompt: str) -> str:
            return "the same line every time"

        out = compile_prompt(TASK_SIG, _graded_set(), suggest=constant,
                             metric=imitation_metric, store=self.store)
        self.assertEqual(out.status, "held")
        self.assertIn("does not beat the base prompt", out.reason)
        self.assertIsNone(out.compiled)
        self.assertEqual(self.store.latest_version("test_task"), 0)

    def test_runner_failure_falls_back_unvalidated(self) -> None:
        def broken(prompt: str) -> str:
            raise RuntimeError("model down")

        out = compile_prompt(TASK_SIG, _graded_set(), suggest=broken,
                             metric=imitation_metric, store=self.store)
        self.assertEqual(out.status, "compiled", out.reason)
        assert out.report is not None
        self.assertFalse(out.report.validated)
        self.assertEqual(out.report.detail.get("validation_failed"),
                         out.report.n_test)

    def test_imitation_metric_scales_by_grade(self) -> None:
        good = GradedExample(inputs={}, output="brave words stand tall",
                             grade=0.9)
        self.assertAlmostEqual(
            imitation_metric("brave words stand tall", good), 0.9)
        self.assertEqual(imitation_metric("nothing in common here", good),
                         0.0)
        bad = GradedExample(inputs={}, output="brave words stand tall",
                            grade=-0.9)
        self.assertAlmostEqual(
            imitation_metric("brave words stand tall", bad), -0.9)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = PromptStore(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _compiled(self, task: str = "test_task") -> CompiledPrompt:
        out = compile_prompt(
            PromptSignature(task=task, instruction="Say it."),
            _graded_set(), store=self.store)
        assert out.compiled is not None
        return out.compiled

    def test_save_load_roundtrip(self) -> None:
        first = self._compiled()
        loaded = self.store.load("test_task")
        assert loaded is not None
        self.assertEqual(loaded.version, 1)
        self.assertEqual(loaded.prompt_text, first.prompt_text)
        self.assertEqual(loaded.example_ids, first.example_ids)
        self.assertEqual(loaded.grades, first.grades)

    def test_versions_advance(self) -> None:
        self._compiled()
        second = self._compiled()
        self.assertEqual(second.version, 2)
        self.assertEqual(self.store.latest_version("test_task"), 2)
        self.assertEqual(len(self.store.history("test_task")), 2)

    def test_load_missing_is_none(self) -> None:
        self.assertIsNone(self.store.load("never_compiled"))
        self.assertIsNone(self.store.load_version("test_task", 99))

    def test_list_tasks(self) -> None:
        self._compiled("task_a")
        self._compiled("task_b")
        self.assertEqual(self.store.list_tasks(), ["task_a", "task_b"])

    def test_corrupt_manifest_is_empty_not_a_crash(self) -> None:
        self._compiled()
        manifest = Path(self._tmp.name) / "test_task" / "manifest.json"
        manifest.write_text("{not json", encoding="utf-8")
        self.assertEqual(self.store.latest_version("test_task"), 0)
        self.assertIsNone(self.store.load("test_task"))

    def test_history_carries_numbers(self) -> None:
        self._compiled()
        entry = self.store.history("test_task")[0]
        self.assertEqual(entry["version"], 1)
        self.assertEqual(entry["n_examples"], 12)
        self.assertTrue(entry["beats_baseline"])
        self.assertFalse(entry["validated"])


class JournalEventTests(unittest.TestCase):
    def test_compile_publishes_event_without_text(self) -> None:
        from nomorals.core.events import EventBus

        bus = EventBus()
        seen: list[Any] = []
        bus.subscribe("prompt.*", lambda e: seen.append(e), sync=True)
        with tempfile.TemporaryDirectory() as d:
            compile_prompt(TASK_SIG, _graded_set(),
                           store=PromptStore(d), bus=bus)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].topic, "prompt.compiled")
        self.assertEqual(seen[0].data["task"], "test_task")
        self.assertEqual(seen[0].data["n_examples"], 12)
        blob = json.dumps(seen[0].data)
        self.assertNotIn("Brave words", blob)
        self.assertNotIn("Terrible ramble", blob)


class SupplierTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.npc_dir = Path(self._tmp.name) / "npc"
        self.prompt_dir = Path(self._tmp.name) / "prompts"
        self._old_npc = os.environ.get("NM_NPC_LEARN_DIR")
        self._old_prompts = os.environ.get("NM_PROMPTS_DIR")
        os.environ["NM_NPC_LEARN_DIR"] = str(self.npc_dir)
        os.environ["NM_PROMPTS_DIR"] = str(self.prompt_dir)

    def tearDown(self) -> None:
        if self._old_npc is None:
            os.environ.pop("NM_NPC_LEARN_DIR", None)
        else:
            os.environ["NM_NPC_LEARN_DIR"] = self._old_npc
        if self._old_prompts is None:
            os.environ.pop("NM_PROMPTS_DIR", None)
        else:
            os.environ["NM_PROMPTS_DIR"] = self._old_prompts
        self._tmp.cleanup()

    def _journal_npc(self, game_id: str = "g1",
                     npc_id: str = "npc1", n: int = 12) -> None:
        from nomorals.games.npc_learn import NPCJournal

        journal = NPCJournal()
        for i in range(n):
            line = (f"Brave words {i} stand tall." if i % 3 == 0 else
                    (f"Terrible ramble {i} that goes on and on and on "
                     f"with no end in sight whatsoever, truly endless."
                     if i % 3 == 1 else f"A middling line {i} here."))
            journal.journal(game_id, npc_id, line=line,
                            candidates=[{"text": line, "score": 0.5,
                                         "features": {"a": 1.0}}],
                            chosen=0, situation=f"scene {i}")
            grade = 0.9 if i % 3 == 0 else (-0.8 if i % 3 == 1 else 0.2)
            self.assertTrue(journal.grade(game_id, npc_id, line, grade))

    def test_npc_rows_carry_situation(self) -> None:
        from nomorals.games.npc_learn import NPCJournal

        self._journal_npc()
        rows = NPCJournal().rows("g1", "npc1")
        self.assertEqual(len(rows), 12)
        self.assertEqual(rows[0]["situation"], "scene 0")
        self.assertIn("line", rows[0])
        self.assertIn("label", rows[0])

    def test_compile_npc_prompt_end_to_end(self) -> None:
        from nomorals.games.npc_learn import compile_npc_prompt

        self._journal_npc()
        out = compile_npc_prompt("g1", "npc1")
        self.assertEqual(out.status, "compiled", out.reason)
        assert out.compiled is not None
        self.assertEqual(out.compiled.version, 1)
        assert out.report is not None
        self.assertGreater(out.report.demo_grade or 0,
                           out.report.baseline_grade)
        self.assertEqual(PromptStore().list_tasks(), ["npc_dialogue"])

    def test_compile_npc_prompt_insufficient(self) -> None:
        from nomorals.games.npc_learn import compile_npc_prompt

        self._journal_npc(n=2)
        out = compile_npc_prompt("g1", "npc1")
        self.assertEqual(out.status, "insufficient")

    def test_character_rows_carry_context(self) -> None:
        from nomorals.characters.learn import LearnStore

        store = LearnStore(data_dir=Path(self._tmp.name) / "chars")
        for i in range(3):
            entry_id = store.journal("c1", {
                "ts": 0.0, "kind": "utterance",
                "context": f"context {i}", "line": f"line {i}",
                "candidates": [{"text": f"line {i}",
                                "features": {"a": 1.0}, "score": 0.5}],
                "chosen": 0, "grade": None, "grades": []})
            store.grade("c1", entry_id, 0.5, "owner")
        rows = store.export_training_data("c1")
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["context"], "context 0")

    def test_compile_character_prompt_end_to_end(self) -> None:
        from nomorals.characters.learn import (
            LearnStore,
            compile_character_prompt,
        )

        store = LearnStore(data_dir=Path(self._tmp.name) / "chars")
        for i in range(12):
            line = (f"Brave words {i} stand tall." if i % 3 == 0 else
                    (f"Terrible ramble {i} that goes on and on and on "
                     f"with no end in sight whatsoever, truly endless."
                     if i % 3 == 1 else f"A middling line {i} here."))
            entry_id = store.journal("c1", {
                "ts": 0.0, "kind": "utterance",
                "context": f"context {i}", "line": line,
                "candidates": [{"text": line,
                                "features": {"a": 1.0}, "score": 0.5}],
                "chosen": 0, "grade": None, "grades": []})
            grade = 0.9 if i % 3 == 0 else (-0.8 if i % 3 == 1 else 0.2)
            store.grade("c1", entry_id, grade, "owner")
        out = compile_character_prompt("c1", learn_store=store)
        self.assertEqual(out.status, "compiled", out.reason)
        assert out.compiled is not None
        self.assertEqual(out.compiled.task, "character_dialogue")


class ConfigTests(unittest.TestCase):
    def test_settings_defaults(self) -> None:
        settings = load_settings(env={}, use_env_file=False)
        pc = settings.prompt_compiler
        self.assertTrue(pc.enabled)
        self.assertEqual(pc.default_min_examples, DEFAULT_MIN_EXAMPLES)
        self.assertEqual(pc.default_max_demos, DEFAULT_MAX_DEMOS)
        self.assertEqual(pc.default_max_counter_examples,
                         DEFAULT_MAX_COUNTER_EXAMPLES)

    def test_settings_override_floor(self) -> None:
        settings = load_settings(
            env={}, use_env_file=False)
        settings.prompt_compiler.default_min_examples = 12
        with tempfile.TemporaryDirectory() as d:
            out = compile_prompt(TASK_SIG, _graded_set(10),
                                 store=PromptStore(d),
                                 settings=settings.prompt_compiler)
        self.assertEqual(out.status, "insufficient")

    def test_signature_floor_beats_settings(self) -> None:
        settings = load_settings(env={}, use_env_file=False)
        settings.prompt_compiler.default_min_examples = 12
        sig = PromptSignature(task="t", instruction="Say it.",
                              min_examples=8)
        with tempfile.TemporaryDirectory() as d:
            out = compile_prompt(sig, _graded_set(10),
                                 store=PromptStore(d),
                                 settings=settings.prompt_compiler)
        self.assertEqual(out.status, "compiled")


class ControlCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.store = PromptStore(self._tmp.name)
        compile_prompt(TASK_SIG, _graded_set(), store=self.store)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_control_status_names_task_and_version(self) -> None:
        text = control_prompt("status", store=self.store)
        self.assertIn("test_task", text)
        self.assertIn("v1", text)
        self.assertIn("base grade", text)

    def test_control_status_one_task(self) -> None:
        text = control_prompt("status test_task", store=self.store)
        self.assertIn("test_task", text)

    def test_control_list(self) -> None:
        text = control_prompt("list", store=self.store)
        self.assertIn("test_task", text)

    def test_control_empty_store(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            empty = PromptStore(d)
            self.assertIn("nothing compiled yet",
                          control_prompt("status", store=empty))

    def test_control_unknown_task(self) -> None:
        text = control_prompt("status nope", store=self.store)
        self.assertIn("no compiled prompt for 'nope'", text)

    def test_control_compile_refused_from_chat(self) -> None:
        text = control_prompt("compile test_task", store=self.store)
        self.assertIn("not from chat", text)

    def test_command_registered_in_control(self) -> None:
        from nomorals.social.chat.control import (
            _HELP_GROUPS,
            COMMAND_DETAILS,
            CONTROL_COMMANDS,
        )

        self.assertIn("prompt", CONTROL_COMMANDS)
        self.assertEqual(CONTROL_COMMANDS["prompt"], (0, None))
        self.assertIn("prompt", COMMAND_DETAILS)
        self.assertIn("compile", COMMAND_DETAILS["prompt"]["what"])
        groups = dict(_HELP_GROUPS)
        self.assertIn("prompt", groups["tools & automation"])


class CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.npc_dir = Path(self._tmp.name) / "npc"
        self.prompt_dir = Path(self._tmp.name) / "prompts"
        self._old_npc = os.environ.get("NM_NPC_LEARN_DIR")
        self._old_prompts = os.environ.get("NM_PROMPTS_DIR")
        os.environ["NM_NPC_LEARN_DIR"] = str(self.npc_dir)
        os.environ["NM_PROMPTS_DIR"] = str(self.prompt_dir)
        from nomorals.cmdline.commands.prompt import _cmd_prompt

        self._cmd_prompt = _cmd_prompt
        self.context = SimpleNamespace(settings=None)

    def tearDown(self) -> None:
        if self._old_npc is None:
            os.environ.pop("NM_NPC_LEARN_DIR", None)
        else:
            os.environ["NM_NPC_LEARN_DIR"] = self._old_npc
        if self._old_prompts is None:
            os.environ.pop("NM_PROMPTS_DIR", None)
        else:
            os.environ["NM_PROMPTS_DIR"] = self._old_prompts
        self._tmp.cleanup()

    def _run(self, args: SimpleNamespace) -> tuple[int, str]:
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = self._cmd_prompt(args, self.context)
        return rc, buf.getvalue()

    def _args(self, action: str, **kw: Any) -> SimpleNamespace:
        base = {"prompt_action": action, "task": "", "game_id": "",
                "npc_id": "", "char_id": "", "validate": False,
                "timeout": 30.0, "version": 0, "json": False}
        base.update(kw)
        return SimpleNamespace(**base)

    def _journal_npc(self, n: int = 12) -> None:
        from nomorals.games.npc_learn import NPCJournal

        journal = NPCJournal()
        for i in range(n):
            line = (f"Brave words {i} stand tall." if i % 3 == 0 else
                    (f"Terrible ramble {i} that goes on and on."
                     if i % 3 == 1 else f"A middling line {i} here."))
            journal.journal("g1", "npc1", line=line,
                            candidates=[{"text": line, "score": 0.5,
                                         "features": {"a": 1.0}}],
                            chosen=0, situation=f"scene {i}")
            grade = 0.9 if i % 3 == 0 else (-0.8 if i % 3 == 1 else 0.2)
            journal.grade("g1", "npc1", line, grade)

    def test_compile_then_status_then_show(self) -> None:
        self._journal_npc()
        rc, out = self._run(self._args(
            "compile", task="npc_dialogue", game_id="g1", npc_id="npc1"))
        self.assertEqual(rc, 0, out)
        self.assertIn("compiled", out)
        rc, out = self._run(self._args("status", task="npc_dialogue"))
        self.assertEqual(rc, 0, out)
        self.assertIn("npc_dialogue", out)
        self.assertIn("v1", out)
        rc, out = self._run(self._args("show", task="npc_dialogue"))
        self.assertEqual(rc, 0, out)
        self.assertIn("Examples", out)

    def test_compile_unknown_task(self) -> None:
        rc, _ = self._run(self._args("compile", task="nope"))
        self.assertEqual(rc, 2)

    def test_compile_missing_ids(self) -> None:
        rc, _ = self._run(self._args(
            "compile", task="npc_dialogue", game_id="g1"))
        self.assertEqual(rc, 2)

    def test_status_empty(self) -> None:
        rc, out = self._run(self._args("status"))
        self.assertEqual(rc, 0)
        self.assertIn("nothing compiled yet", out)

    def test_show_before_compile(self) -> None:
        rc, _ = self._run(self._args("show", task="npc_dialogue"))
        self.assertEqual(rc, 1)


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from nomorals.core.policy import CapabilitySet
        from nomorals.tools.registry import ToolRegistry

        context = SimpleNamespace(settings=Settings())
        cls.registry = ToolRegistry(context=context)
        cls.registry.register_builtins()
        cls.CapabilitySet = CapabilitySet

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.npc_dir = Path(self._tmp.name) / "npc"
        self.prompt_dir = Path(self._tmp.name) / "prompts"
        self._old_npc = os.environ.get("NM_NPC_LEARN_DIR")
        self._old_prompts = os.environ.get("NM_PROMPTS_DIR")
        os.environ["NM_NPC_LEARN_DIR"] = str(self.npc_dir)
        os.environ["NM_PROMPTS_DIR"] = str(self.prompt_dir)

    def tearDown(self) -> None:
        if self._old_npc is None:
            os.environ.pop("NM_NPC_LEARN_DIR", None)
        else:
            os.environ["NM_NPC_LEARN_DIR"] = self._old_npc
        if self._old_prompts is None:
            os.environ.pop("NM_PROMPTS_DIR", None)
        else:
            os.environ["NM_PROMPTS_DIR"] = self._old_prompts
        self._tmp.cleanup()

    def _journal_npc(self, n: int = 12) -> None:
        from nomorals.games.npc_learn import NPCJournal

        journal = NPCJournal()
        for i in range(n):
            line = f"Brave words {i} stand tall."
            journal.journal("g1", "npc1", line=line,
                            candidates=[{"text": line, "score": 0.5,
                                         "features": {"a": 1.0}}],
                            chosen=0, situation=f"scene {i}")
            journal.grade("g1", "npc1", line,
                          0.9 if i % 3 == 0 else -0.8)

    def test_tools_registered(self) -> None:
        for name, capability in (("prompt_compile", "model.call"),
                                 ("prompt_status", "fs.read")):
            spec = self.registry.get(name)
            self.assertIsNotNone(spec, name)
            assert spec is not None
            self.assertEqual(str(spec.capability), capability)
        failed = [m for m in self.registry.failed_modules()
                  if m.get("module") == "prompt_compiler"]
        self.assertEqual(failed, [])

    def test_compile_through_registry(self) -> None:
        self._journal_npc()
        outcome = self.registry.call(
            "prompt_compile", actor="test",
            capabilities=self.CapabilitySet.all(),
            task="npc_dialogue", game_id="g1", npc_id="npc1")
        self.assertTrue(outcome.ok, outcome)
        value = outcome.unwrap()
        self.assertEqual(value["status"], "compiled")
        self.assertEqual(value["version"], 1)

    def test_status_through_registry(self) -> None:
        self._journal_npc()
        self.registry.call(
            "prompt_compile", actor="test",
            capabilities=self.CapabilitySet.all(),
            task="npc_dialogue", game_id="g1", npc_id="npc1")
        outcome = self.registry.call(
            "prompt_status", actor="test",
            capabilities=self.CapabilitySet.all(),
            task="npc_dialogue")
        self.assertTrue(outcome.ok, outcome)
        value = outcome.unwrap()
        self.assertIn("npc_dialogue", value["status_text"])
        self.assertEqual(value["latest_version"], 1)

    def test_compile_unknown_task_is_error(self) -> None:
        outcome = self.registry.call(
            "prompt_compile", actor="test",
            capabilities=self.CapabilitySet.all(), task="nope")
        self.assertFalse(outcome.ok)


if __name__ == "__main__":
    unittest.main()
