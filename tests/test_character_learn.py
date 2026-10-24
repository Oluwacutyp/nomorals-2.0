"""Tests for the character learn loop (journal + grade + adjust).

Covers: journal round-trip, cap prune, corrupt recovery, grading
(explicit/session/errors), candidate selection (ties, dedupe, collapse),
weight bounds, persona flag-off + leash, prefs activation, durability
(atomic save, failed save keeps memory), store auto-attach, speak()
default unchanged, tool `learned` action, NM_HOME independence, and
to_dict shape stability.
"""
import json
import os

import pytest

from nomorals.characters import Character, CharacterStore
from nomorals.characters.learn import (
    LearnStore, attach, detach, get_handle,
    score_candidates, pick_best, extract_features,
    DEFAULT_WEIGHTS, MAX_ENTRIES, SESSION_GRADES,
)


def make_char(name="Zara", **kw):
    base = dict(persona={"witty": 0.9, "energetic": 0.8},
                core_motive="To ask the question nobody else will ask.",
                expression={"catchphrases": ["say less"]})
    base.update(kw)
    return Character(name=name, **base)


def _store(tmp_path):
    return LearnStore(tmp_path / "data")


def _journal_entry(line, features=None):
    feats = features or {k: 0.5 for k in
                         ("consistency", "catchphrase", "length_fit",
                          "punct_fit", "novelty")}
    return {"ts": 0.0, "kind": "utterance", "context": "ctx",
            "line": line,
            "candidates": [{"text": line, "features": dict(feats),
                            "score": 0.5}],
            "chosen": 0, "grade": None, "grades": []}


# ── journal ────────────────────────────────────────────────────────────

class TestJournal:
    def test_round_trip_and_reload(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path))
        eid = h.journal_line("hello there, friend.")
        assert eid == "L-000001"
        fresh = LearnStore(tmp_path / "data")
        d = fresh.doc(c.id)
        assert len(d["entries"]) == 1
        assert d["entries"][0]["line"] == "hello there, friend."
        assert d["entries"][0]["id"] == eid

    def test_cap_prunes_oldest_but_total_counts_all(self, tmp_path):
        c = make_char()
        st = _store(tmp_path)
        for i in range(MAX_ENTRIES + 5):
            st.journal(c.id, _journal_entry(f"line {i}"))
        d = st.doc(c.id)
        assert len(d["entries"]) == MAX_ENTRIES
        assert d["total"] == MAX_ENTRIES + 5
        assert d["entries"][0]["line"] == f"line 5"

    def test_corrupt_file_recovers_to_empty_and_rewrites(self, tmp_path):
        c = make_char()
        st = _store(tmp_path)
        st.journal(c.id, _journal_entry("before corruption"))
        p = st._path(c.id)
        assert p is not None
        p.write_text("{not json", encoding="utf-8")
        fresh = LearnStore(tmp_path / "data")
        d = fresh.doc(c.id)
        assert d["entries"] == []
        fresh.journal(c.id, _journal_entry("after recovery"))
        d2 = LearnStore(tmp_path / "data").doc(c.id)
        assert len(d2["entries"]) == 1
        assert d2["entries"][0]["line"] == "after recovery"

    def test_to_dict_shape_unchanged_by_handle(self, tmp_path):
        c = make_char()
        attach(c, store=_store(tmp_path))
        d = c.to_dict()
        assert "_learn" not in d
        assert "learn" not in d
        c2 = Character.from_dict(d)
        assert c2.name == c.name
        assert c2.persona == c.persona


# ── grading ────────────────────────────────────────────────────────────

class TestGrading:
    def test_explicit_grade_updates_entry_and_stats(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path))
        eid = h.journal_line("a fine line.")
        e = h.grade(eid, 0.8, "owner", note="landed well")
        assert e["grade"] == 0.8
        assert e["grades"][0]["source"] == "owner"
        st = h.store.doc(c.id)["stats"]
        assert st["graded"] == 1
        assert st["mean_grade"] == 0.8

    def test_grade_rejects_bad_input(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path))
        eid = h.journal_line("a line.")
        with pytest.raises(ValueError):
            h.grade(eid, 0.5, "telepathy")
        with pytest.raises(ValueError):
            h.grade(eid, 2.0, "owner")
        with pytest.raises(ValueError):
            h.grade(eid, float("nan"), "owner")
        with pytest.raises(ValueError):
            h.grade("L-999999", 0.5, "owner")

    def test_session_grade_moves_weights_a_quarter_step(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path))
        h.begin_session()
        h.journal_line("short line.")
        before = dict(h.store.doc(c.id)["weights"])
        graded = h.grade_session("good_conversation")
        assert len(graded) == 1
        after = h.store.doc(c.id)["weights"]
        # session value 0.25 at lr 0.0125, features <= 1: delta <= 0.003125
        # (zero features legitimately move zero — e.g. no catchphrase)
        for k in before:
            assert 0.0 <= after[k] - before[k] <= 0.003125 + 1e-9
        assert after["consistency"] - before["consistency"] > 0.0
        e = h.store.doc(c.id)["entries"][0]
        assert e["grade"] == 0.25

    def test_session_grade_unknown_kind_is_noop(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path))
        h.begin_session()
        h.journal_line("a line.")
        assert h.grade_session("not_a_kind") == []
        assert h.store.doc(c.id)["entries"][0]["grade"] is None

    def test_export_training_data_only_graded(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path))
        e1 = h.journal_line("graded line.")
        h.journal_line("ungraded line.")
        h.grade(e1, 1.0, "owner")
        rows = h.export_training_data()
        assert len(rows) == 1
        assert rows[0]["label"] == 1.0
        assert set(rows[0]["features"]) == set(DEFAULT_WEIGHTS)


# ── candidate scoring ──────────────────────────────────────────────────

class TestCandidates:
    def test_catchphrase_candidate_wins(self, tmp_path):
        c = make_char()
        texts = ["plain line here.", "say less, plain line here.",
                 "another plain line."]
        scored = score_candidates(texts, c)
        assert scored[1].features["catchphrase"] == 1.0
        assert scored[0].features["catchphrase"] == 0.0
        assert pick_best(scored) == 1

    def test_tie_goes_to_earliest(self):
        c = make_char()
        scored = score_candidates(["hello world this.",
                                   "hello world this!"], c)
        assert scored[0].score == scored[1].score
        assert pick_best(scored) == 0

    def test_speak_picks_best_and_journals_candidates(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path))
        lines = iter(["plain line here.", "say less, plain line here.",
                      "another plain line."])
        out = c.speak("greet me", lambda p: next(lines), candidates=3)
        assert out == "say less, plain line here."
        d = h.store.doc(c.id)
        assert len(d["entries"]) == 1
        assert len(d["entries"][0]["candidates"]) == 3
        assert d["entries"][0]["chosen"] == 1

    def test_speak_dedupes_and_collapses_deterministic(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path))
        lines = iter(["same line.", "same line.", "other line."])
        c.speak("hi", lambda p: next(lines), candidates=3)
        d = h.store.doc(c.id)
        assert len(d["entries"][0]["candidates"]) == 2

        c2 = make_char(name="Milo")
        h2 = attach(c2, store=_store(tmp_path))
        out = c2.speak("hi", lambda p: "always this.", candidates=3)
        assert out == "always this."
        d2 = h2.store.doc(c2.id)
        assert len(d2["entries"][0]["candidates"]) == 1

    def test_speak_default_unchanged(self):
        c = make_char()
        assert c.speak("hi", lambda p: "Hello there.") == "Hello there."
        fb = c.speak("hi", None)
        assert fb and not fb.startswith("[")
        # empty model output falls back to the same deterministic line
        assert c.speak("hi", lambda p: "") == fb


# ── adjustment ─────────────────────────────────────────────────────────

class TestAdjustment:
    def _grade_many(self, st, char_id, eid, value, n):
        for _ in range(n):
            st.grade(char_id, eid, value, "owner")

    def test_weights_clamped_both_directions(self, tmp_path):
        c = make_char()
        st = _store(tmp_path)
        feats = {k: 1.0 for k in DEFAULT_WEIGHTS}
        eid = st.journal(c.id, _journal_entry("x", feats))
        self._grade_many(st, c.id, eid, 1.0, 40)
        assert st.doc(c.id)["weights"]["consistency"] == 2.0
        c2 = make_char(name="Milo")
        eid2 = st.journal(c2.id, _journal_entry("x", feats))
        self._grade_many(st, c2.id, eid2, -1.0, 40)
        assert st.doc(c2.id)["weights"]["consistency"] == 0.0

    def test_persona_flag_off_means_no_movement(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path), adjust_persona=False)
        eid = h.journal_line("a line.")
        h.grade(eid, 1.0, "owner")
        assert c.persona == {"witty": 0.9, "energetic": 0.8}
        assert h.store.doc(c.id)["trait_offsets"] == {}

    def test_persona_flag_on_leashed_at_tenth(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path), adjust_persona=True)
        eid = h.journal_line("a line.")
        for _ in range(10):
            h.grade(eid, 1.0, "owner")
        assert c.persona["witty"] == 1.0  # 0.9 + 0.10 leash
        # energetic is strong (0.8 >= 0.7) so it moves too: 0.8 + 0.10
        assert c.persona["energetic"] == 0.9
        d = h.store.doc(c.id)
        assert d["seed_persona"]["witty"] == 0.9
        for _ in range(15):
            h.grade(eid, -1.0, "owner")
        assert c.persona["witty"] == 0.8  # 0.9 - 0.10 leash
        assert abs(d["trait_offsets"]["witty"] + 0.1) < 1e-9

    def test_prefs_activate_after_ten_positives(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path))
        for _ in range(10):
            eid = h.journal_line("one two three four five.")
            h.grade(eid, 1.0, "owner")
        prefs = h.store.doc(c.id)["prefs"]
        assert prefs["n_graded_positive"] == 10
        assert prefs["mean_len"] == 5.0
        feats = extract_features("one two three.", c, prefs=prefs)
        assert feats["length_fit"] > 0.5


# ── durability ─────────────────────────────────────────────────────────

class TestDurability:
    def test_failed_save_keeps_memory_and_returns_false(self, tmp_path):
        blocker = tmp_path / "blocker"
        blocker.write_text("a file, not a dir", encoding="utf-8")
        st = LearnStore(blocker)
        c = make_char()
        eid = st.journal(c.id, _journal_entry("kept in memory"))
        assert eid
        assert st.save(c.id) is False
        assert len(st.doc(c.id)["entries"]) == 1

    def test_nm_home_bogus_does_not_redirect_journal(self, tmp_path,
                                                    monkeypatch):
        monkeypatch.setenv("NM_HOME", str(tmp_path / "bogus"))
        c = make_char()
        h = attach(c, store=_store(tmp_path))
        h.journal_line("still here.")
        p = tmp_path / "data" / "learn" / f"{c.id}.json"
        assert p.exists()
        assert json.loads(p.read_text("utf-8"))["entries"][0]["line"] \
            == "still here."


# ── wiring ─────────────────────────────────────────────────────────────

class TestWiring:
    def test_store_auto_attach_and_speak_journals(self, tmp_path):
        store = CharacterStore(data_dir=tmp_path / "chars")
        store.save(make_char())
        c = store.get_by_name("Zara")
        assert c is not None
        assert get_handle(c) is not None
        c.speak("hi", lambda p: "store-backed line.")
        p = tmp_path / "chars" / "learn" / f"{c.id}.json"
        assert p.exists()
        assert json.loads(p.read_text("utf-8"))["entries"][0]["line"] \
            == "store-backed line."
        assert len(store.all()) == 1  # learn/ subdir does not pollute all()

    def test_converse_grades_session(self, tmp_path):
        from nomorals.characters.dialogue import converse
        c = make_char()
        attach(c, store=_store(tmp_path))
        dlg = converse("You are Devon.", c, "hey",
                       lambda p: "short reply.", rounds=1)
        assert dlg.turns
        d = get_handle(c).store.doc(c.id)
        assert len(d["entries"]) == 1
        # short lines -> good_conversation, session value 0.25
        assert d["entries"][0]["grade"] == 0.25

    def test_tool_learned_action(self, tmp_path, monkeypatch):
        from nomorals.tools import characters as tool_mod

        class FakeRegistry:
            def __init__(self):
                from types import SimpleNamespace
                self.context = SimpleNamespace(suggest=lambda p: "hi")
            def register(self, name, **kwargs):
                def deco(fn):
                    self.fn = fn
                    return fn
                return deco

        root = tmp_path / "home"
        monkeypatch.setenv("NOMORALS_DATA_DIR", str(root))
        store = CharacterStore(data_dir=root / "characters")
        store.save(make_char())
        c = store.get_by_name("Zara")
        get_handle(c).journal_line("a learned line.")
        reg = FakeRegistry()
        tool_mod.register(reg)
        out = reg.fn(action="learned", target="Zara")
        assert "learn journal" in out
        assert "1 entries" in out
        assert reg.fn(action="learned", target="Nobody") \
            == "no character named 'Nobody'."
        detach(c)
