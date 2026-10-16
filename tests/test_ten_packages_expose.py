"""Tests for the TEN_PACKAGES expose layer (build-map recovery task).

Invariants under test:

* Expose wrappers delegate to the real modules or report honestly
  unavailable — they never fake a result.
* ROADMAP.md numbers match live counts (package count at least).
* BUILD_MAP.md either cites sources (file:line) or honestly says the
  full map is not recoverable.

All offline. No network, no real devices, no real money, no HA.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ROADMAP = ROOT / "ROADMAP.md"
BUILD_MAP = ROOT / "BUILD_MAP.md"


# ── helpers ─────────────────────────────────────────────────────────────────


def live_package_count() -> int:
    """Sub-packages under nomorals/ (dirs, excluding __pycache__)."""
    nomorals = ROOT / "nomorals"
    return sum(
        1 for p in nomorals.iterdir()
        if p.is_dir() and p.name != "__pycache__"
    )


# ── integrations/expose ─────────────────────────────────────────────────────


class TestIntegrationsExpose:
    def test_email_backends_honest(self):
        from nomorals.integrations.expose import describe_email_backends

        by_name = {b["name"]: b for b in describe_email_backends()}
        assert by_name["gmail_api"]["available"] is True
        assert by_name["imap"]["available"] is True
        # Browser is named in _detect_backend's docstring but no code
        # path returns it — the expose layer must say so, not fake it.
        assert by_name["browser"]["available"] is False
        assert "no code path returns it" in by_name["browser"]["note"]

    def test_calendar_backends_honest(self):
        from nomorals.integrations.expose import describe_calendar_backends

        by_name = {b["name"]: b for b in describe_calendar_backends()}
        assert by_name["google_api"]["available"] is True
        assert by_name["caldav"]["available"] is False
        assert by_name["browser"]["available"] is False

    def test_diff_snapshots_pure(self):
        from nomorals.integrations.expose import diff_snapshots

        out = diff_snapshots(
            {"light.kitchen": "off", "lock.front": "locked", "sensor.temp": "21"},
            {"light.kitchen": "on", "lock.front": "locked", "switch.coffee": "on"},
        )
        assert out["added"] == ["switch.coffee"]
        assert out["removed"] == ["sensor.temp"]
        assert out["changed"] == {
            "light.kitchen": {"before": "off", "after": "on"}}
        assert out["unchanged"] == ["lock.front"]

    def test_steal_score_matches_engine_formula(self):
        from nomorals.integrations.expose import steal_score_from_history

        # 30-day history, median 100_000; current 80_000; trust 0.8;
        # cross-site best 85_000.
        got = steal_score_from_history(
            [100_000, 100_000, 100_000, 100_000], 80_000,
            trust=0.8, cross_site_best=85_000)
        assert got is not None
        assert got["median_30d"] == 100_000
        assert got["savings_vs_median"] == pytest.approx(20.0)
        assert got["savings_vs_cross_site"] == pytest.approx(
            (85_000 - 80_000) / 85_000 * 100)
        # Engine formula: median*0.5 + cross*0.3 + trust*20.
        assert got["steal_score"] == pytest.approx(
            20.0 * 0.5 + got["savings_vs_cross_site"] * 0.3 + 0.8 * 20)

    def test_steal_score_none_on_empty_history(self):
        from nomorals.integrations.expose import steal_score_from_history

        assert steal_score_from_history([], 80_000) is None
        assert steal_score_from_history([0, 0], 80_000) is None

    def test_parse_automation_never_executes(self):
        from nomorals.integrations.expose import parse_automation

        devices = [
            {"entity_id": "light.kitchen", "name": "kitchen lights"},
            {"entity_id": "switch.coffee_maker", "name": "coffee"},
        ]
        desc = parse_automation(
            "every morning at 7, kitchen lights + coffee", devices=devices)
        assert desc["starter"].kind == "time"
        assert desc["starter"].at == "07:00"
        assert [a.entity_id for a in desc["actions"]] == [
            "light.kitchen", "switch.coffee_maker"]
        assert desc["requires_confirmation"] is True
        assert desc["confirmed"] is False
        assert desc["ha_automation"] is None
        assert desc["errors"] == []

    def test_parse_automation_unresolved_is_honest(self):
        from nomorals.integrations.expose import parse_automation

        desc = parse_automation("every morning at 7, kitchen lights + coffee")
        assert desc["unresolved"] == ["kitchen lights", "coffee"]
        assert any(e["code"] == "unknown_device" for e in desc["errors"])
        assert desc["confirmed"] is False

    def test_confirm_then_activate_gating(self):
        from nomorals.integrations import expose

        devices = [
            {"entity_id": "light.kitchen", "name": "kitchen lights"},
            {"entity_id": "switch.coffee_maker", "name": "coffee"},
        ]
        desc = expose.parse_automation(
            "every morning at 7, kitchen lights + coffee", devices=devices)
        confirmed = expose.confirm_automation(desc)
        assert confirmed["confirmed"] is True
        assert confirmed["activated"] is False
        assert confirmed["ha_automation_id"] == ""
        assert confirmed["ha_automation"]["trigger"] == {
            "platform": "time", "at": "07:00:00"}

        # Unconfirmed descriptors can never reach activation.
        with pytest.raises(RuntimeError, match="unconfirmed"):
            asyncio.run(expose.activate_automation(desc, integration=None))

    def test_confirm_refuses_invalid_draft(self):
        from nomorals.integrations import expose
        from nomorals.integrations.routines import RoutineBuildError

        desc = expose.parse_automation(
            "every morning at 7, kitchen lights + coffee")
        with pytest.raises(RoutineBuildError):
            expose.confirm_automation(desc)

    def test_what_changed_since_delegates(self):
        from nomorals.integrations import expose

        twin = expose.make_home_twin(":memory:")
        try:
            twin.ingest("light.kitchen", "on", ts=1_700_000_000.0)
            twin.ingest("light.kitchen", "off", ts=1_700_000_600.0)
            text = expose.what_changed_since(twin, 1_700_000_000.0)
            assert "kitchen" in text
            assert expose.what_changed_since(twin, 1_800_000_000.0) == (
                "nothing changed since then.")
        finally:
            twin.close()


# ── audio/expose ────────────────────────────────────────────────────────────


class TestAudioExpose:
    def test_fingerprint_missing_file_honest(self):
        from nomorals.audio.expose import fingerprint_audio

        got = fingerprint_audio("/nonexistent/nope.wav")
        assert got["ok"] is False
        assert got["reason"]

    def test_identify_missing_file_honest(self):
        from nomorals.audio.expose import identify_audio

        got = identify_audio("/nonexistent/nope.wav")
        assert got["ok"] is False

    def test_talk_to_character_needs_engine(self):
        from nomorals.audio.characters import Character
        from nomorals.audio.expose import talk_to_character

        ch = Character(name="Ada", book_title="Test Book", knowledge_cutoff=3)
        got = talk_to_character(ch, "hello?", llm_fn=None)
        assert got.ok is False
        assert "no dialogue engine" in got.reason

    def test_talk_to_character_rejects_bad_spoiler_level(self):
        from nomorals.audio.characters import Character
        from nomorals.audio.expose import talk_to_character

        ch = Character(name="Ada", book_title="Test Book", knowledge_cutoff=3)
        got = talk_to_character(ch, "hello?", spoiler_level=0,
                                llm_fn=lambda s, q: "hi")
        assert got.ok is False

    def test_spoiler_level_caps_prompt(self):
        from nomorals.audio.characters import Character
        from nomorals.audio.expose import character_prompt, talk_to_character

        ch = Character(name="Ada", book_title="Test Book", knowledge_cutoff=5,
                       chapter_memories={
                           1: ["Ada lives in Lagos."],
                           5: ["SECRET-FIVE Ada betrays everyone."],
                       })
        prompt = character_prompt(ch, spoiler_level=2)
        assert "chapter 2" in prompt
        assert "SECRET-FIVE" not in prompt
        # And talk_to with the same cap never sees the ch-5 memory.
        seen: dict[str, str] = {}

        def fake_llm(system: str, question: str) -> str:
            seen["system"] = system
            return "I know nothing past chapter 2."

        got = talk_to_character(ch, "what happens later?", spoiler_level=2,
                                llm_fn=fake_llm)
        assert got.ok is True
        assert "SECRET-FIVE" not in seen["system"]

    def test_spoiler_level_cannot_extend_cutoff(self):
        from nomorals.audio.characters import Character
        from nomorals.audio.expose import character_prompt

        ch = Character(name="Ada", book_title="Test Book", knowledge_cutoff=2,
                       chapter_memories={2: ["Ada lives in Lagos."]})
        # Reader claims ch 99 — the character's own cutoff (2) still wins.
        assert "chapter 2" in character_prompt(ch, spoiler_level=99)

    def test_pick_backends_by_profile(self):
        from nomorals.audio.expose import pick_backends

        termux = pick_backends("termux")
        assert termux["profile"] == "termux"
        assert termux["stt_candidates"][0] == "parakeet-onnx"
        # Unknown profile falls back to pc (registry's own rule).
        assert pick_backends("bogus-profile")["profile"] == "pc"
        # Public audience structurally drops XTTS (non-commercial).
        laptop_public = pick_backends("laptop", audience="public")
        assert "xtts" not in laptop_public["tts_candidates"]
        laptop_private = pick_backends("laptop", audience="private")
        assert "xtts" in laptop_private["tts_candidates"]

    def test_remove_fillers_missing_file_honest(self):
        from nomorals.audio.expose import remove_fillers_from

        got = remove_fillers_from("/nonexistent/nope.wav")
        assert got["ok"] is False


# ── documents/expose ────────────────────────────────────────────────────────


class TestDocumentsExpose:
    def test_sixteen_formats_including_epub_odt_ods(self):
        from nomorals.documents.expose import SUPPORTED_FORMATS, list_formats

        assert len(SUPPORTED_FORMATS) == 16
        for fmt in ("epub", "odt", "ods", "pdf", "docx", "xlsx", "pptx",
                    "rtf", "json", "xml", "csv", "tsv", "txt", "html",
                    "markdown", "xls"):
            assert fmt in SUPPORTED_FORMATS
        assert len(list_formats()) == 16
        assert all(f["stdlib_only"] for f in list_formats())

    def test_read_and_convert_markdown_round_trip(self):
        from nomorals.documents.expose import (
            convert_document, read_document)

        doc = read_document(data=b"# Hello\n\nSome text.\n", filename="x.md")
        assert doc.sections
        md = convert_document(doc, "markdown")
        assert "# Hello" in md
        assert "Some text." in md

    def test_read_document_requires_exactly_one_source(self):
        from nomorals.documents.expose import read_document

        with pytest.raises(ValueError):
            read_document()
        with pytest.raises(ValueError):
            read_document(path="x.md", data=b"hi")

    def test_convert_unknown_target_raises(self):
        from nomorals.documents.expose import convert_document, read_document

        doc = read_document(data=b"hello\n", filename="x.txt")
        with pytest.raises(ValueError, match="unknown convert target"):
            convert_document(doc, "carrier-pigeon")

    def test_convert_csv_needs_table(self):
        from nomorals.documents import DocumentError
        from nomorals.documents.expose import convert_document, read_document

        doc = read_document(data=b"hello\n", filename="x.txt")
        with pytest.raises(DocumentError):
            convert_document(doc, "csv")

    def test_document_capabilities_honest(self):
        from nomorals.documents.expose import document_capabilities

        caps = document_capabilities()
        assert caps["format_count"] == 16
        assert caps["parse_stdlib_only"] is True
        assert caps["parse_offline"] is True
        assert "python-docx" in caps["optional_deps"]["docx_output"]


# ── ROADMAP numbers verified live ───────────────────────────────────────────


class TestRoadmapNumbers:
    def test_package_count_matches_live(self):
        text = ROADMAP.read_text()
        m = re.search(r"(?m)^packages:\s*(\d+)\s*$", text)
        assert m, "ROADMAP must carry a 'packages: N' verified-count line"
        assert int(m.group(1)) == live_package_count()

    def test_schema_version_matches_migrations(self):
        from nomorals.storage.migrations import MIGRATIONS, latest_version

        text = ROADMAP.read_text()
        m = re.search(r"(?m)^schema version:\s*(\d+)\s*$", text)
        assert m, "ROADMAP must carry a 'schema version: N' line"
        assert int(m.group(1)) == latest_version() == len(MIGRATIONS)

    def test_table_count_matches_live_db_when_present(self):
        db = ROOT / "data" / "nomorals.db"
        if not db.exists():
            pytest.skip("no live data/nomorals.db in this checkout")
        text = ROADMAP.read_text()
        m = re.search(r"(?m)^tables:\s*(\d+) real", text)
        assert m, "ROADMAP must carry a 'tables: N real …' line"
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            tables = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")]
        finally:
            con.close()
        real = [t for t in tables
                if not (t.endswith(("_fts", "_fts_config", "_fts_content",
                                    "_fts_data", "_fts_docsize", "_fts_idx")))
                and t not in ("sqlite_sequence", "schema_migrations",
                              "schema_migrations_repeatable")]
        assert int(m.group(1)) == len(real)

    def test_stale_v7_claim_is_gone(self):
        text = ROADMAP.read_text()
        assert "7 forward-only checksummed migrations to 52" not in text
        assert "schema v7" not in text.lower().replace(
            "stale v7/52 claim corrected", "")


# ── BUILD_MAP honesty ───────────────────────────────────────────────────────


class TestBuildMap:
    def test_six_numbers_cited(self):
        text = BUILD_MAP.read_text()
        for number, anchor in (
                ("#46", "nomorals/learn/tutor.py"),
                ("#53", "nomorals/health/coach.py"),
                ("#76", "tests/test_contracts.py"),
                ("#80", "nomorals/legal/regulatory.py"),
                ("#83", "nomorals/matching/__init__.py"),
                ("#87", "nomorals/health/challenges.py")):
            assert number in text, f"BUILD_MAP missing {number}"
            assert anchor in text, f"BUILD_MAP missing citation {anchor}"

    def test_honest_about_unrecoverable_full_map(self):
        text = BUILD_MAP.read_text()
        assert "NOT recoverable" in text
        assert "undefined" in text.lower()
        # No invented canonical plan: the file must say no plan doc exists.
        assert "no canonical plan document exists" in text.lower()
