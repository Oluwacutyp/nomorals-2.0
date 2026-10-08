"""Tests for taste memory + conversational production (nomorals/media/taste.py)."""
import json
import os
from pathlib import Path

import pytest

from nomorals.media.taste import (
    TasteProfile, TasteStore, load_taste, modify_profile,
    detect_produce_intent, detect_feedback, mood_hint,
    last_production_profile,
)


@pytest.fixture()
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


class Ctx:
    settings = None


def _prof(bpm=140.0, key="F", energy=("dark", "driving")):
    from nomorals.media.producer import ReferenceProfile
    return ReferenceProfile(title="t", bpm=bpm, key=key, mode="minor",
                            energy_words=energy, genre="electronic",
                            mood=",".join(energy), ok=True)


# ── persistence ────────────────────────────────────────────────────────

def test_profile_persists_and_reloads(tmp_home):
    s = load_taste(Ctx())
    s.record_production(_prof(), source="link", ref="x")
    assert s.profile.production_count == 1
    s2 = load_taste(Ctx())
    assert s2.profile.production_count == 1
    assert s2.profile.last_production["bpm"] == 140.0
    assert "dark" in s2.profile.energy_weights


def test_corrupt_json_starts_fresh(tmp_home):
    path = Path(os.path.expanduser("~")) / ".nomorals" / "taste.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json[[[", encoding="utf-8")
    s = load_taste(Ctx())
    assert s.profile.production_count == 0
    assert s.profile.last_production == {}


def test_missing_file_starts_fresh(tmp_home):
    s = load_taste(Ctx())
    assert s.profile.production_count == 0
    assert s.suggest_profile() is None


def test_suggest_profile_needs_history(tmp_home):
    s = load_taste(Ctx())
    assert s.suggest_profile() is None
    # one production is not enough either (production_count < 2, no weights)
    s.record_production(_prof(), source="link", ref="x")
    # production records energy weights now, so suggest works
    prof = s.suggest_profile()
    assert prof is not None
    assert 70 <= prof.bpm <= 180


# ── feedback ───────────────────────────────────────────────────────────

def test_feedback_liked_updates_weights(tmp_home):
    s = load_taste(Ctx())
    s.record_production(_prof(), source="link", ref="x")
    before = dict(s.profile.energy_weights)
    msg = s.record_feedback(True, "the drop goes hard")
    assert "taste updated" in msg
    assert s.profile.energy_weights["dark"] > before.get("dark", 0)
    assert len(s.profile.tracks) == 1
    assert s.profile.tracks[0].liked is True


def test_feedback_notes_parse_tempo_hints(tmp_home):
    s = load_taste(Ctx())
    s.record_production(_prof(), source="link", ref="x")
    s.record_feedback(False, "too slow, needs more energy")
    assert s.profile.bpm_bias > 0  # "too slow" -> prefer faster
    assert s.profile.energy_weights.get("driving", 0) > 0


def test_feedback_without_production_is_graceful(tmp_home):
    s = load_taste(Ctx())
    msg = s.record_feedback(True, "love it")
    assert "nothing to rate" in msg


def test_disliked_weights_decay(tmp_home):
    s = load_taste(Ctx())
    s.record_production(_prof(energy=("chill",)), source="link", ref="x")
    s.record_feedback(False, "too chill")
    # disliked characteristic weights get pushed down
    assert s.profile.energy_weights.get("chill", 0) <= 0.05 or \
        "chill" not in s.profile.energy_weights


# ── suggest from taste ─────────────────────────────────────────────────

def test_suggest_profile_favors_liked(tmp_home):
    s = load_taste(Ctx())
    s.record_production(_prof(bpm=150.0, energy=("dark", "aggressive")),
                        source="link", ref="a")
    s.record_feedback(True, "yes")
    s.record_production(_prof(bpm=95.0, energy=("chill", "dreamy")),
                        source="link", ref="b")
    s.record_feedback(False, "too slow")
    prof = s.suggest_profile()
    assert prof is not None
    assert prof.bpm > 120  # liked the fast one, disliked the slow one
    assert "dark" in prof.energy_words or "aggressive" in prof.energy_words


# ── iterative modification ─────────────────────────────────────────────

def test_modify_faster_raises_bpm():
    out = modify_profile(_prof(bpm=130.0), "faster")
    assert out is not _prof  # new object (identity not needed; value check:)
    assert out.bpm == 144.0


def test_modify_slower_lowers_bpm():
    out = modify_profile(_prof(bpm=130.0), "slower please")
    assert out.bpm == 116.0


def test_modify_harder_adds_energy():
    out = modify_profile(_prof(energy=("chill",)), "make it harder")
    assert "aggressive" in out.energy_words
    assert "chill" not in out.energy_words
    assert out.bpm == 148.0


def test_modify_darker():
    out = modify_profile(_prof(energy=("uplifting",)), "darker")
    assert "dark" in out.energy_words
    assert "uplifting" not in out.energy_words


def test_modify_unknown_returns_same():
    p = _prof()
    assert modify_profile(p, "something totally unrelated") is p


def test_modify_never_mutates_input():
    p = _prof(bpm=130.0, energy=("chill",))
    out = modify_profile(p, "faster")
    assert p.bpm == 130.0
    assert p.energy_words == ("chill",)
    assert out.bpm == 144.0


def test_modify_bpm_clamped():
    out = modify_profile(_prof(bpm=175.0), "faster faster faster")
    assert out.bpm <= 180.0
    out2 = modify_profile(_prof(bpm=75.0), "slower")
    assert out2.bpm >= 70.0


# ── NL detection ───────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "make me something",
    "make me a track",
    "make me something for the gym",
    "produce a track for tonight",
    "compose me something dark",
    "something dark for tonight",
    "something for the gym",
    "cook me up a beat",
    "https://open.spotify.com/track/6r7b1UHvO3fBZe7wBXWTaZ",
])
def test_detect_produce_intent_fires(text):
    assert detect_produce_intent(text) is not None


@pytest.mark.parametrize("text", [
    "what's the weather like",
    "I'm in a good mood",          # bare mood: no fire (ambiguous)
    "make me a sandwich",          # not music
    "/produce dark edm",           # slash commands are not NL
    "",
    "hey how are you",
])
def test_detect_produce_intent_quiet(text):
    assert detect_produce_intent(text) is None


def test_detect_produce_mood_with_music_ctx():
    assert detect_produce_intent("I'm in a good mood, make me something") is not None


def test_mood_hint_extracts_energy():
    assert "aggressive" in mood_hint("make me something for the gym")
    assert "dark" in mood_hint("something dark for tonight")
    assert mood_hint("hello world") == ()


@pytest.mark.parametrize("text,liked", [
    ("I like this", True),
    ("i love this", True),
    ("this is fire", True),
    ("not feeling this track", False),
    ("too slow", False),
    ("don't like this", False),
])
def test_detect_feedback(text, liked):
    fb = detect_feedback(text)
    assert fb is not None
    assert fb[0] is liked


@pytest.mark.parametrize("text", [
    "nah",               # too ambiguous alone
    "hello",
    "that's cool",
    "/like",
    "",
])
def test_detect_feedback_quiet(text):
    assert detect_feedback(text) is None


def test_detect_feedback_forgiving():
    assert detect_feedback(None) is None
    assert detect_produce_intent(None) is None
    assert mood_hint(None) == ()
    p = _prof()
    assert modify_profile(p, None) is p


# ── last production round-trip ────────────────────────────────────────

def test_last_production_profile_roundtrip(tmp_home):
    s = load_taste(Ctx())
    assert last_production_profile(s) is None
    s.record_production(_prof(bpm=152.0, key="G"), source="link", ref="x")
    lp = last_production_profile(s)
    assert lp is not None
    assert lp.bpm == 152.0
    assert lp.key == "G"
