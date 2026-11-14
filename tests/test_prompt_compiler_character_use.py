"""Prompt compiler at runtime: the compiled character_dialogue prompt
in Character.speak, gated by config, attributed in the journal.

The compiler itself (selection, validation, store) is covered by
tests/test_prompt_compiler.py, and the npc_dialogue use side by
tests/test_prompt_compiler_use.py. This file covers the character
USE side: the resolver gate (validated-only by default, unvalidated
only under the flag, disabled means base), the prompt the character
actually sends (persona/context prefix kept, base byte-for-byte when
no usable compile exists, compiled render ending on the same
"Situation: ...\\n{name}:" tail the base ends on), the journal entry
carrying the prompt version that produced the line ("base",
"character_dialogue v<N>", or "fallback" for a model-free line) so
grades attribute to their prompt, and validation rendering with the
runtime labels (the measured gain is the gain of the prompt that
actually runs).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.characters import Character
from nomorals.characters.context import CharacterContextBuilder
from nomorals.characters.learn import (
    LearnStore,
    _character_signature,
    attach,
    compile_character_prompt,
    resolve_character_prompt,
)
from nomorals.llm.prompt_compiler import (
    CompiledPrompt,
    CompileReport,
    GradedExample,
    PromptStore,
    imitation_metric,
    render_prompt,
)

TASK_SIG = _character_signature("Zara")


def _example(i: int, grade: float) -> GradedExample:
    return GradedExample(
        inputs={"context": f"scene {i} in the tavern"},
        output=f"Brave words {i} stand tall.", grade=grade,
        id=f"ex{i:02d}", source="owner")


def _save_compiled(store: PromptStore, *, validated: bool,
                   version_note: str = "") -> CompiledPrompt:
    """A compiled character_dialogue prompt in the store, built the
    way compile_prompt builds one (rendered text + report)."""
    demos = [_example(1, 0.9), _example(2, 0.8)]
    counters = [_example(3, -0.7)]
    note = "- Highest-graded outputs average 22 characters."
    report = CompileReport(
        n_examples=10, n_train=8, n_test=2, baseline_grade=0.10,
        demo_grade=0.85, selection_lift=0.75, validated=validated,
        beats_baseline=True, status="compiled",
        reason="fixture compile" + version_note)
    compiled = CompiledPrompt(
        task="character_dialogue", version=0,
        instruction=TASK_SIG.instruction, instruction_note=note,
        prompt_text=render_prompt(TASK_SIG, demos, counters, note, {}),
        demos=demos, counter_examples=counters, report=report,
        compiled_at=1.0, example_ids=["ex01", "ex02"],
        grades=[0.9, 0.8])
    store.save("character_dialogue", compiled)
    return compiled


def _settings(**overrides: object) -> SimpleNamespace:
    base = {"enabled": True, "use_unvalidated": False}
    base.update(overrides)
    return SimpleNamespace(**base)


def _char(tmp: str) -> Character:
    """A fixture character with a learn handle, wired to a temp
    prompt store via the _prompt_* attributes Character.speak reads."""
    char = Character(
        id="zara-1", name="Zara",
        persona={"witty": 0.9, "energetic": 0.8},
        core_motive="To ask the question nobody else will ask.",
        expression={"catchphrases": ["say less"]})
    handle = attach(char, store=LearnStore(Path(tmp) / "learn"))
    return char, handle


class ResolveTests(unittest.TestCase):
    """The gate: what counts as usable at runtime."""

    def test_no_compile_means_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PromptStore(tmp)
            self.assertIsNone(resolve_character_prompt(
                "zara-1", "the tavern door opens",
                character_name="Zara",
                settings=_settings(), store=store))

    def test_validated_compile_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PromptStore(tmp)
            _save_compiled(store, validated=True)
            resolved = resolve_character_prompt(
                "zara-1", "the tavern door opens",
                character_name="Zara",
                settings=_settings(), store=store)
            self.assertIsNotNone(resolved)
            text, version = resolved
            self.assertEqual(version, "character_dialogue v1")
            # The compiled render: instruction + note + demos + input.
            self.assertIn("Brave words 1 stand tall.", text)
            self.assertIn("Highest-graded outputs average", text)
            # The runtime labels: "Situation:" input, "{name}:" tail.
            self.assertIn("Situation: the tavern door opens", text)
            self.assertTrue(text.rstrip().endswith("Zara:"))
            self.assertNotIn("Context: the tavern door opens", text)
            self.assertNotIn("\nLine:", text)

    def test_validated_compile_used_without_settings(self):
        # Settings None reads the config file (defaults: enabled,
        # unvalidated off) — a validated compile is still usable.
        with tempfile.TemporaryDirectory() as tmp:
            store = PromptStore(tmp)
            _save_compiled(store, validated=True)
            resolved = resolve_character_prompt(
                "zara-1", "the tavern door opens",
                character_name="Zara", store=store)
            self.assertIsNotNone(resolved)
            self.assertEqual(resolved[1], "character_dialogue v1")

    def test_unvalidated_refused_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PromptStore(tmp)
            _save_compiled(store, validated=False)
            self.assertIsNone(resolve_character_prompt(
                "zara-1", "the tavern door opens",
                character_name="Zara",
                settings=_settings(), store=store))

    def test_unvalidated_used_when_flag_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PromptStore(tmp)
            _save_compiled(store, validated=False)
            resolved = resolve_character_prompt(
                "zara-1", "the tavern door opens",
                character_name="Zara",
                settings=_settings(use_unvalidated=True), store=store)
            self.assertIsNotNone(resolved)
            self.assertEqual(resolved[1], "character_dialogue v1")

    def test_disabled_compiling_means_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PromptStore(tmp)
            _save_compiled(store, validated=True)
            self.assertIsNone(resolve_character_prompt(
                "zara-1", "the tavern door opens",
                character_name="Zara",
                settings=_settings(enabled=False), store=store))

    def test_corrupt_store_means_base_never_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "character_dialogue"
            bad.mkdir()
            (bad / "manifest.json").write_text(
                "{not json", encoding="utf-8")
            store = PromptStore(tmp)
            self.assertIsNone(resolve_character_prompt(
                "zara-1", "the tavern door opens",
                character_name="Zara",
                settings=_settings(), store=store))

    def test_report_missing_means_base(self):
        # A stored prompt with no report cannot prove it is
        # validated — it is not usable.
        with tempfile.TemporaryDirectory() as tmp:
            store = PromptStore(tmp)
            _save_compiled(store, validated=True)
            task_dir = store._task_dir("character_dialogue")
            target = task_dir / "character_dialogue-v1.json"
            import json

            data = json.loads(target.read_text(encoding="utf-8"))
            data["report"] = {}
            target.write_text(json.dumps(data), encoding="utf-8")
            self.assertIsNone(resolve_character_prompt(
                "zara-1", "the tavern door opens",
                character_name="Zara",
                settings=_settings(), store=store))

    def test_no_name_renders_generic_label(self):
        # No character name anywhere: the resolver falls back to the
        # generic "Line:" label rather than failing.
        with tempfile.TemporaryDirectory() as tmp:
            store = PromptStore(tmp)
            _save_compiled(store, validated=True)
            resolved = resolve_character_prompt(
                "nobody-here", "the tavern door opens",
                settings=_settings(), store=store)
            self.assertIsNotNone(resolved)
            text, version = resolved
            self.assertEqual(version, "character_dialogue v1")
            self.assertTrue(text.rstrip().endswith("Line:"))


class CharacterPromptTests(unittest.TestCase):
    """The prompt the character actually sends + the journal entry."""

    def _wire(self, tmp, *, validated=None, settings=None):
        store = PromptStore(Path(tmp) / "prompts")
        if validated is not None:
            _save_compiled(store, validated=validated)
        char, handle = _char(tmp)
        char._prompt_settings = settings if settings is not None \
            else _settings()
        char._prompt_store = store
        captured: list[str] = []

        def suggest(prompt: str) -> str:
            captured.append(prompt)
            return "Say less, the tavern is warm tonight."

        return char, handle, captured, suggest

    def test_base_prompt_unchanged_without_compile(self):
        with tempfile.TemporaryDirectory() as tmp:
            char, handle, captured, suggest = self._wire(tmp)
            line = char.speak("the tavern door opens", suggest)
            self.assertEqual(line, "Say less, the tavern is warm tonight.")
            self.assertEqual(len(captured), 1)
            prompt = captured[0]
            # Persona/context prefix kept, base tail exact.
            self.assertIn("You are Zara.", prompt)
            self.assertIn("Situation: the tavern door opens", prompt)
            self.assertTrue(prompt.rstrip().endswith("Zara:"))
            self.assertNotIn("Brave words", prompt)
            # Byte-for-byte the v1 prompt: the base branch must not
            # drift from what speak built before this lane.
            expected = (
                CharacterContextBuilder().build(char, memories=[])
                + "\n\nSituation: the tavern door opens\nZara:")
            self.assertEqual(prompt, expected)
            # Journaled as base.
            entries = handle.store.doc(char.id)["entries"]
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].get("prompt_version"), "base")
            self.assertEqual(entries[0]["line"], line)

    def test_base_prompt_fallback_path_unchanged(self):
        # The context builder fails: speak falls back to the simple
        # persona block, and that prompt is byte-for-byte the old one.
        with tempfile.TemporaryDirectory() as tmp:
            char, handle, captured, suggest = self._wire(tmp)
            with mock.patch.object(
                    CharacterContextBuilder, "build",
                    side_effect=RuntimeError("boom")):
                line = char.speak("the tavern door opens", suggest)
            self.assertEqual(line, "Say less, the tavern is warm tonight.")
            prompt = captured[0]
            expected = (char.persona_block() + "\n"
                        + "Situation: the tavern door opens\nZara:")
            self.assertEqual(prompt, expected)
            entries = handle.store.doc(char.id)["entries"]
            self.assertEqual(entries[0].get("prompt_version"), "base")

    def test_compiled_prompt_used_when_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            char, handle, captured, suggest = self._wire(
                tmp, validated=True)
            line = char.speak("the tavern door opens", suggest)
            self.assertEqual(line, "Say less, the tavern is warm tonight.")
            prompt = captured[0]
            # Persona/context prefix STILL present; the compiled
            # instruction + demos replace the base tail's instruction
            # slot, not the persona.
            self.assertIn("You are Zara.", prompt)
            self.assertIn("Brave words 1 stand tall.", prompt)
            self.assertIn("Highest-graded outputs average", prompt)
            # Same tail the base ends on — the model completes after
            # the character's own name, not a generic label.
            self.assertIn("Situation: the tavern door opens", prompt)
            self.assertTrue(prompt.rstrip().endswith("Zara:"))
            entries = handle.store.doc(char.id)["entries"]
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].get("prompt_version"),
                             "character_dialogue v1")

    def test_unvalidated_compile_not_used_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            char, handle, captured, suggest = self._wire(
                tmp, validated=False)
            char.speak("the tavern door opens", suggest)
            self.assertNotIn("Brave words", captured[0])
            entries = handle.store.doc(char.id)["entries"]
            self.assertEqual(entries[0].get("prompt_version"), "base")

    def test_unvalidated_compile_used_when_flag_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            char, handle, captured, suggest = self._wire(
                tmp, validated=False,
                settings=_settings(use_unvalidated=True))
            char.speak("the tavern door opens", suggest)
            self.assertIn("Brave words 1 stand tall.", captured[0])
            entries = handle.store.doc(char.id)["entries"]
            self.assertEqual(entries[0].get("prompt_version"),
                             "character_dialogue v1")

    def test_fallback_line_records_fallback_not_a_prompt(self):
        # No model at all: the deterministic fallback speaks, and the
        # journal says so — the grade must not attribute to a prompt.
        with tempfile.TemporaryDirectory() as tmp:
            char, handle, captured, suggest = self._wire(
                tmp, validated=True)
            line = char.speak("the tavern door opens", None)
            self.assertTrue(line)
            self.assertEqual(captured, [])
            entries = handle.store.doc(char.id)["entries"]
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].get("prompt_version"), "fallback")
            self.assertEqual(entries[0]["kind"], "fallback")

    def test_candidates_path_uses_compiled_and_journals_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            char, handle, captured, suggest = self._wire(
                tmp, validated=True)
            line = char.speak("the tavern door opens", suggest,
                              candidates=3)
            self.assertTrue(line)
            self.assertEqual(len(captured), 3)
            self.assertTrue(all("Brave words 1 stand tall." in p
                                for p in captured))
            entries = handle.store.doc(char.id)["entries"]
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].get("prompt_version"),
                             "character_dialogue v1")


class CompileValidationLabelsTests(unittest.TestCase):
    """Validation renders with the runtime labels: what is measured
    is the prompt that actually runs."""

    def test_validation_renders_with_runtime_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = LearnStore(Path(tmp) / "chars")
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
            captured: list[str] = []

            def suggest(prompt: str) -> str:
                captured.append(prompt)
                return "Brave words 0 stand tall."

            out = compile_character_prompt(
                "c1", learn_store=store, suggest=suggest,
                metric=imitation_metric, character_name="Zara")
            # Validation ran on the held-out slice: base + compiled
            # prompts, both rendered with the runtime labels.
            self.assertTrue(captured, "validation made no model calls")
            self.assertTrue(all("Situation:" in p for p in captured))
            self.assertTrue(all(p.rstrip().endswith("Zara:")
                                for p in captured))
            self.assertTrue(all("Context:" not in p for p in captured))
            self.assertIn(out.status, ("compiled", "held"))


if __name__ == "__main__":
    unittest.main()
