"""Prompt compiler at runtime: the compiled npc_dialogue prompt in
GameMaster.npc_speak, gated by config, attributed in the journal.

The compiler itself (selection, validation, store) is covered by
tests/test_prompt_compiler.py. This file covers the USE side: the
resolver gate (validated-only by default, unvalidated only under the
flag, disabled means base), the prompt the GameMaster actually sends
(persona prepended, memories kept, base byte-for-byte when no usable
compile exists), and the journal entry carrying the prompt version
that produced the line ("base", "npc_dialogue v<N>", or "forge" for
a template line) so grades attribute to their prompt.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.games.gamemaster import GameMaster
from nomorals.games.npc import NPCProfile, NPCStore
from nomorals.games.npc_learn import (
    NPC_DIALOGUE_SIGNATURE,
    NPCJournal,
    resolve_npc_prompt,
)
from nomorals.llm.prompt_compiler import (
    CompiledPrompt,
    CompileReport,
    GradedExample,
    PromptSignature,
    PromptStore,
    render_prompt,
)

TASK_SIG = PromptSignature(**NPC_DIALOGUE_SIGNATURE)


def _example(i: int, grade: float) -> GradedExample:
    return GradedExample(
        inputs={"situation": f"scene {i} in the tavern"},
        output=f"Brave words {i} stand tall.", grade=grade,
        id=f"ex{i:02d}", source="owner")


def _save_compiled(store: PromptStore, *, validated: bool,
                   version_note: str = "") -> CompiledPrompt:
    """A compiled npc_dialogue prompt in the store, built the way
    compile_prompt builds one (rendered text + report)."""
    demos = [_example(1, 0.9), _example(2, 0.8)]
    counters = [_example(3, -0.7)]
    note = "- Highest-graded outputs average 22 characters."
    report = CompileReport(
        n_examples=10, n_train=8, n_test=2, baseline_grade=0.10,
        demo_grade=0.85, selection_lift=0.75, validated=validated,
        beats_baseline=True, status="compiled",
        reason="fixture compile" + version_note)
    compiled = CompiledPrompt(
        task="npc_dialogue", version=0,
        instruction=TASK_SIG.instruction, instruction_note=note,
        prompt_text=render_prompt(TASK_SIG, demos, counters, note, {}),
        demos=demos, counter_examples=counters, report=report,
        compiled_at=1.0, example_ids=["ex01", "ex02"],
        grades=[0.9, 0.8])
    store.save("npc_dialogue", compiled)
    return compiled


def _settings(**overrides: object) -> SimpleNamespace:
    base = {"enabled": True, "use_unvalidated": False}
    base.update(overrides)
    return SimpleNamespace(**base)


def _npc() -> NPCProfile:
    return NPCProfile(
        id="brin", name="Brin", game_id="tavern",
        personality={"brave": 0.8, "warm": 0.6}, voice_style="warm",
        mood={"valence": 0.2, "arousal": 0.3, "trust": 0.5})


class ResolveTests(unittest.TestCase):
    """The gate: what counts as usable at runtime."""

    def test_no_compile_means_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PromptStore(tmp)
            self.assertIsNone(resolve_npc_prompt(
                "brin", "the tavern door opens",
                settings=_settings(), store=store))

    def test_validated_compile_is_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PromptStore(tmp)
            _save_compiled(store, validated=True)
            resolved = resolve_npc_prompt(
                "brin", "the tavern door opens",
                settings=_settings(), store=store)
            self.assertIsNotNone(resolved)
            text, version = resolved
            self.assertEqual(version, "npc_dialogue v1")
            # The compiled render: instruction + note + demos + input.
            self.assertIn("Brave words 1 stand tall.", text)
            self.assertIn("Highest-graded outputs average", text)
            self.assertIn("Situation: the tavern door opens", text)
            self.assertTrue(text.rstrip().endswith("Line:"))

    def test_validated_compile_used_without_settings(self):
        # Settings None reads the config file (defaults: enabled,
        # unvalidated off) — a validated compile is still usable.
        with tempfile.TemporaryDirectory() as tmp:
            store = PromptStore(tmp)
            _save_compiled(store, validated=True)
            resolved = resolve_npc_prompt(
                "brin", "the tavern door opens", store=store)
            self.assertIsNotNone(resolved)
            self.assertEqual(resolved[1], "npc_dialogue v1")

    def test_unvalidated_refused_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PromptStore(tmp)
            _save_compiled(store, validated=False)
            self.assertIsNone(resolve_npc_prompt(
                "brin", "the tavern door opens",
                settings=_settings(), store=store))

    def test_unvalidated_used_when_flag_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PromptStore(tmp)
            _save_compiled(store, validated=False)
            resolved = resolve_npc_prompt(
                "brin", "the tavern door opens",
                settings=_settings(use_unvalidated=True), store=store)
            self.assertIsNotNone(resolved)
            self.assertEqual(resolved[1], "npc_dialogue v1")

    def test_disabled_compiling_means_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PromptStore(tmp)
            _save_compiled(store, validated=True)
            self.assertIsNone(resolve_npc_prompt(
                "brin", "the tavern door opens",
                settings=_settings(enabled=False), store=store))

    def test_corrupt_store_means_base_never_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "npc_dialogue"
            bad.mkdir()
            (bad / "manifest.json").write_text(
                "{not json", encoding="utf-8")
            store = PromptStore(tmp)
            self.assertIsNone(resolve_npc_prompt(
                "brin", "the tavern door opens",
                settings=_settings(), store=store))

    def test_report_missing_means_base(self):
        # A stored prompt with no report cannot prove it is
        # validated — it is not usable.
        with tempfile.TemporaryDirectory() as tmp:
            store = PromptStore(tmp)
            _save_compiled(store, validated=True)
            task_dir = store._task_dir("npc_dialogue")
            target = task_dir / "npc_dialogue-v1.json"
            import json

            data = json.loads(target.read_text(encoding="utf-8"))
            data["report"] = {}
            target.write_text(json.dumps(data), encoding="utf-8")
            self.assertIsNone(resolve_npc_prompt(
                "brin", "the tavern door opens",
                settings=_settings(), store=store))


class GameMasterPromptTests(unittest.TestCase):
    """The prompt the GameMaster actually sends + the journal entry."""

    def _gm(self, store, settings, tmp):
        captured: list[str] = []

        def suggest(prompt: str) -> str:
            captured.append(prompt)
            return "Aye, the tavern is warm tonight."

        env = {"NM_NPC_LEARN_DIR": tmp}
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        gm = GameMaster(NPCStore(), suggest=suggest, seed=7,
                        prompt_settings=settings, prompt_store=store)
        return gm, captured

    def test_base_prompt_unchanged_without_compile(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PromptStore(Path(tmp) / "prompts")
            gm, captured = self._gm(
                store, _settings(), str(Path(tmp) / "journal"))
            line = gm.npc_speak(_npc(), "the tavern door opens")
            self.assertEqual(line, "Aye, the tavern is warm tonight.")
            self.assertEqual(len(captured), 1)
            prompt = captured[0]
            # Persona prepended, memories kept, base instruction exact.
            self.assertIn("You are Brin, a character", prompt)
            self.assertIn("Situation: the tavern door opens", prompt)
            self.assertIn("Relevant memories:", prompt)
            self.assertIn(
                "Speak as Brin: one or two sentences, in your voice.",
                prompt)
            self.assertNotIn("Brave words", prompt)
            # Byte-for-byte the v1 prompt: the base branch must not
            # drift from what npc_speak built before this lane.
            npc = _npc()
            expected = (
                f"{npc.to_prompt()}\n\n"
                "Situation: the tavern door opens\n"
                "Relevant memories:\n"
                + "\n".join(f"- {m['text']}" for m in npc.recall(
                    "the tavern door opens", limit=3))
                + "\n\nSpeak as Brin: one or two sentences, in your "
                "voice. Stay in character. Reply with the line only — "
                "no analysis, no preamble, no stage directions.")
            self.assertEqual(prompt, expected)
            # Journaled as base.
            entries = NPCJournal(
                Path(tmp) / "journal").entries("tavern", "brin")
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].get("prompt_version"), "base")
            self.assertEqual(entries[0]["line"], line)

    def test_compiled_prompt_used_when_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PromptStore(Path(tmp) / "prompts")
            _save_compiled(store, validated=True)
            gm, captured = self._gm(
                store, _settings(), str(Path(tmp) / "journal"))
            line = gm.npc_speak(_npc(), "the tavern door opens")
            self.assertEqual(line, "Aye, the tavern is warm tonight.")
            prompt = captured[0]
            # Persona STILL prepended; the compiled instruction + demos
            # replace the base instruction, not the persona.
            self.assertIn("You are Brin, a character", prompt)
            self.assertIn("Relevant memories:", prompt)
            self.assertIn("Brave words 1 stand tall.", prompt)
            self.assertIn("Highest-graded outputs average", prompt)
            self.assertNotIn("Speak as Brin: one or two sentences",
                             prompt)
            self.assertTrue(prompt.rstrip().endswith("Line:"))
            entries = NPCJournal(
                Path(tmp) / "journal").entries("tavern", "brin")
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].get("prompt_version"),
                             "npc_dialogue v1")

    def test_unvalidated_compile_not_used_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PromptStore(Path(tmp) / "prompts")
            _save_compiled(store, validated=False)
            gm, captured = self._gm(
                store, _settings(), str(Path(tmp) / "journal"))
            gm.npc_speak(_npc(), "the tavern door opens")
            self.assertNotIn("Brave words", captured[0])
            entries = NPCJournal(
                Path(tmp) / "journal").entries("tavern", "brin")
            self.assertEqual(entries[0].get("prompt_version"), "base")

    def test_unvalidated_compile_used_when_flag_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PromptStore(Path(tmp) / "prompts")
            _save_compiled(store, validated=False)
            gm, captured = self._gm(
                store, _settings(use_unvalidated=True),
                str(Path(tmp) / "journal"))
            gm.npc_speak(_npc(), "the tavern door opens")
            self.assertIn("Brave words 1 stand tall.", captured[0])
            entries = NPCJournal(
                Path(tmp) / "journal").entries("tavern", "brin")
            self.assertEqual(entries[0].get("prompt_version"),
                             "npc_dialogue v1")

    def test_forge_line_records_forge_not_a_prompt(self):
        # No model at all: the template forge speaks, and the journal
        # says so — the grade must not attribute to a prompt.
        with tempfile.TemporaryDirectory() as tmp:
            store = PromptStore(Path(tmp) / "prompts")
            _save_compiled(store, validated=True)
            env = {"NM_NPC_LEARN_DIR": str(Path(tmp) / "journal")}
            with mock.patch.dict(os.environ, env):
                gm = GameMaster(NPCStore(), suggest=None, seed=7,
                                prompt_settings=_settings(),
                                prompt_store=store)
                line = gm.npc_speak(_npc(), "the tavern door opens")
            self.assertTrue(line)
            entries = NPCJournal(
                Path(tmp) / "journal").entries("tavern", "brin")
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].get("prompt_version"), "forge")

    def test_candidates_path_uses_compiled_and_journals_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = PromptStore(Path(tmp) / "prompts")
            _save_compiled(store, validated=True)
            gm, captured = self._gm(
                store, _settings(), str(Path(tmp) / "journal"))
            line = gm.npc_speak(_npc(), "the tavern door opens",
                                candidates=3)
            self.assertTrue(line)
            self.assertEqual(len(captured), 3)
            self.assertTrue(all("Brave words 1 stand tall." in p
                                for p in captured))
            entries = NPCJournal(
                Path(tmp) / "journal").entries("tavern", "brin")
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0].get("prompt_version"),
                             "npc_dialogue v1")


if __name__ == "__main__":
    unittest.main()
