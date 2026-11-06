"""Tail-86 pins: nomorals/books/collab.py census sites.

359 — review(): a crashing model suggest falls back to
_heuristic_review, the module's documented no-model floor (the
same path taken openly when suggest is None). Designed ladder.
429 — _parse_critique: CONVERTED by targeted-fix Pass 4. An
unparseable SCORE used to leave the pre-try 70.0 default
un-divided, which the Critique clamp took to 1.0 — a perfect
score for garbage — while a missing SCORE parsed the string
"70" -> 0.7. The default is now 0.7, so unparseable and
missing take the same honest value; parseable scores are
honoured exactly as before.
532 — owner voiceprint build crash -> None: advisory only, the
genuine no-owner-texts value; never a fabricated voiceprint.
547 — voice_check compare crash -> None: advisory drift report;
None is the genuine no-voiceprint value consumers already take
as "no report".
563 — cast_character: a dict whose CharacterSheet.from_dict
crashes is still cast (name-keyed in _characters); only the
sheet is missing, and character_sheet() returns the genuine
no-sheet None (plain named objects are cast sheetless by design).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import nomorals.books.collab as collab_mod
import nomorals.books.voiceprint as vp_mod
from nomorals.books.collab import CollaborativeSession, CriticAgent


def _raise(exc):
    raise exc


TEXT = "The rain fell hard on the harbour road. " * 40


# ── 359: model -> heuristic ladder ──────────────────────────────────

def test_359_review_model_crash_falls_back_to_heuristic_floor():
    critic = CriticAgent(None)

    def bad_suggest(prompt, max_tokens=800):
        raise RuntimeError("model down")

    got = critic.review(TEXT, {}, None, bad_suggest)
    floor = critic.review(TEXT, {}, None, None)
    assert (got.score, got.verdict, got.issues, got.suggestions) == \
        (floor.score, floor.verdict, floor.issues, floor.suggestions)


# ── 429: score parse default ────────────────────────────────────────

def test_429_parse_critique_unparseable_score_defaults():
    critic = CriticAgent(None)
    c = critic._parse_critique("SCORE: not-a-number\nVERDICT: ship\n")
    assert c.score == 0.7  # CONVERTED (Pass 4): unparseable == missing
    assert c.verdict == "ship"
    # controls: missing SCORE -> "70"/100; parseable SCORE honoured
    assert critic._parse_critique("VERDICT: revise\n").score == 0.7
    assert critic._parse_critique("SCORE: 85\n").score == 0.85


# ── 532: voiceprint build ───────────────────────────────────────────

def test_532_voiceprint_build_crash_leaves_none(monkeypatch):
    monkeypatch.setattr(vp_mod, "build_owner_voiceprint",
                        lambda texts: _raise(RuntimeError("boom")))
    s = CollaborativeSession("slug", None, owner_texts=["my own prose"])
    assert s.owner_voiceprint is None
    # control: the genuine no-texts value is the same None
    assert CollaborativeSession("slug", None).owner_voiceprint is None


# ── 547: voice check ────────────────────────────────────────────────

def test_547_voice_check_compare_crash_returns_none(monkeypatch):
    s = CollaborativeSession("slug", None)
    s.owner_voiceprint = SimpleNamespace(words=10)
    monkeypatch.setattr(vp_mod, "compare",
                        lambda a, b: _raise(RuntimeError("boom")))
    assert s.voice_check("some text") is None
    # control: no voiceprint at all returns the same None, unpatched
    assert CollaborativeSession("slug", None).voice_check("x") is None


# ── 563: cast without sheet ─────────────────────────────────────────

def test_563_cast_character_bad_sheet_dict_casts_without_sheet(monkeypatch):
    s = CollaborativeSession("slug", None)
    monkeypatch.setattr(
        collab_mod.CharacterSheet, "from_dict",
        classmethod(lambda cls, d: _raise(RuntimeError("bad sheet"))))
    s.cast_character({"name": "Ada"})
    assert len(s._characters) == 1
    assert len(s._sheets) == 0
    # control: a plain named object is cast sheetless by design
    s.cast_character(SimpleNamespace(name="Bala"))
    assert s.character_sheet("Bala") is None
    assert "Bala" in s._characters
