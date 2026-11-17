"""Gapfix R3-53: the audio expose facade is deleted, not wired.

The verdict (survey: ~/workspace/gapfix53-verify/GAPFIX53_SURVEY.md) is
DELETE: `nomorals/audio/expose.py` has zero production importers and
every one of its ten functions is a pass-through to an engine that
already has a direct live caller (or no callers at all). The deletion
tests below fail until the file, its test section, and its ROADMAP
bullet are gone; the guard tests pin the engines that survive it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / "nomorals"


def _tree_hits(needle: str) -> list[str]:
    """Files under nomorals/ + tests/ + ROADMAP.md mentioning `needle`.

    This file itself is excluded: it names the deleted module in its
    own docstring and needle arguments.
    """
    hits: list[str] = []
    for base in (ROOT / "nomorals", ROOT / "tests"):
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".md"}:
                if path.name == "test_gapfix_audio_expose.py":
                    continue
                try:
                    text = path.read_text(errors="ignore")
                except OSError:
                    continue
                if needle in text:
                    hits.append(str(path.relative_to(ROOT)))
    roadmap = ROOT / "ROADMAP.md"
    if roadmap.exists() and needle in roadmap.read_text(errors="ignore"):
        hits.append("ROADMAP.md")
    return sorted(hits)


class TestAudioExposeDeleted:
    def test_no_audio_expose_symbol_anywhere(self):
        hits = _tree_hits("audio.expose")
        assert hits == [], f"audio.expose still referenced in: {hits}"

    def test_no_audio_expose_path_anywhere(self):
        hits = _tree_hits("audio/expose")
        assert hits == [], f"audio/expose still referenced in: {hits}"

    def test_expose_file_is_gone(self):
        assert not (TREE / "audio" / "expose.py").exists()

    def test_expose_test_section_is_gone(self):
        test_file = ROOT / "tests" / "test_ten_packages_expose.py"
        text = test_file.read_text()
        assert "TestAudioExpose" not in text
        assert "from nomorals.audio.expose import" not in text

    def test_roadmap_no_longer_lists_three_facades(self):
        text = (ROOT / "ROADMAP.md").read_text()
        assert "nomorals/audio/expose.py" not in text
        assert "all three" not in text


class TestEnginesSurviveDeletion:
    """Every engine the facade wrapped stays directly reachable."""

    def test_fingerprint_engines_reachable(self):
        from nomorals.audio.fingerprint import (
            analyze,
            fingerprint,
            identify,
            match_batch,
        )

        assert identify("/nonexistent/nope.wav")["ok"] is False
        assert match_batch(["/nonexistent/nope.wav"])[0]["ok"] is False
        assert analyze("/nonexistent/nope.wav").ok is False
        # The raw fingerprint raises the named error (the facade's
        # envelope caught it); the direct path is honest, not silent.
        from nomorals.audio.fingerprint import AudioReadError

        with pytest.raises(AudioReadError):
            fingerprint("/nonexistent/nope.wav")

    def test_edit_engines_reachable(self):
        from nomorals.audio.edit import remove_fillers, transcript_edit

        assert remove_fillers("/nonexistent/nope.wav")["ok"] is False
        assert transcript_edit("/nonexistent/nope.wav") is None

    def test_character_engine_reachable_with_current_chapter(self):
        from nomorals.audio.characters import Character, talk_to

        ch = Character(
            name="Ada",
            book_title="Test Book",
            knowledge_cutoff=5,
            chapter_memories={
                1: ["Ada lives in Lagos."],
                5: ["SECRET-FIVE Ada betrays everyone."],
            },
        )
        # No dialogue engine: honest refusal, never a fake voice.
        got = talk_to(ch, "hello?", llm_fn=None)
        assert got.ok is False
        # The engine itself keeps the chapter cap the facade computed.
        prompt = ch.to_system_prompt(current_chapter=2)
        assert "chapter 2" in prompt
        assert "SECRET-FIVE" not in prompt

    def test_backend_registry_reachable(self):
        from nomorals.audio.pipeline import EngineRegistry

        assert EngineRegistry("bogus-profile").profile == "pc"
        assert EngineRegistry("termux").profile == "termux"

    def test_live_paths_still_import(self):
        # /sham's engine imports and the character control entry.
        from nomorals.audio.characters import control_character
        from nomorals.audio.fingerprint import (
            describe_audio,
            match_local_db,
        )

        assert callable(control_character)
        assert callable(describe_audio)
        assert callable(match_local_db)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
