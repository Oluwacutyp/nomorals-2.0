"""Tests for decide() journaling (arc phase 3).

Covers: decide journal round-trip (options/chosen/game/state/kind),
return-value identity with and without a handle, no journaling when
the handle is absent or options are empty, journal failure never
changing the move, grade_decide mapping + errors + per-option
counters, the _adjust skip (voice weights/prefs untouched by moves),
export_training_data exclusion, the report decide section, backward
compatibility with pre-decide v1 docs, session-grade separation
(grade_session never touches moves), and run_agent_match end-of-
match grading (win/loss/draw, brain win, unknown/ambiguous winner
skips, no-handle match unchanged).
"""
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.characters import (
    Character, CharacterStore, AgentSeat, run_agent_match,
)
from nomorals.characters.learn import (
    LearnStore, attach, get_handle,
    DEFAULT_WEIGHTS, DECIDE_OUTCOMES,
)
from nomorals.characters.match import _grade_match_decides


def make_char(name="Zara", **kw):
    base = dict(persona={"witty": 0.9, "energetic": 0.8},
                core_motive="To ask the question nobody else will ask.",
                expression={"catchphrases": ["say less"]})
    base.update(kw)
    return Character(name=name, **base)


def _store(tmp_path):
    return LearnStore(tmp_path / "data")


def _suggest_first(prompt: str) -> str:
    low = prompt.lower()
    if "legal options" in low:
        for line in prompt.splitlines():
            s = line.strip()
            if s.startswith("- "):
                return s[2:].strip()
    return "stub line, staying in character."


def _decide_entry(store, char_id):
    d = store.doc(char_id)
    assert len(d["entries"]) == 1
    return d["entries"][0]


# ── journal ────────────────────────────────────────────────────────────

class TestDecideJournal:
    def test_round_trip(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path))
        move = c.decide("auction", "bid rules", "lot 3, price 50",
                        ["bid 50", "bid 60", "pass"], _suggest_first)
        assert move == "bid 50"
        e = _decide_entry(h.store, c.id)
        assert e["kind"] == "decide"
        assert e["game"] == "auction"
        assert e["options"] == ["bid 50", "bid 60", "pass"]
        assert e["chosen"] == 0
        assert e["line"] == ""
        assert e["context"] == "lot 3, price 50"
        assert e["grade"] is None and e["grades"] == []
        assert e["outcome"] is None
        assert "candidates" not in e
        # reload from disk: same entry
        fresh = LearnStore(tmp_path / "data")
        assert fresh.doc(c.id)["entries"][0]["options"] == e["options"]

    def test_state_truncated_to_200(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path))
        c.decide("auction", "rules", "x" * 500,
                 ["a", "b"], _suggest_first)
        assert len(_decide_entry(h.store, c.id)["context"]) == 200

    def test_fallback_pick_journaled(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path))
        rng = random.Random(7)
        move = c.decide("auction", "rules", "state",
                        ["a", "b", "c"], lambda p: "nonsense",
                        rng=rng)
        assert move in ("a", "b", "c")
        e = _decide_entry(h.store, c.id)
        assert e["options"][e["chosen"]] == move

    def test_return_identical_with_and_without_handle(self):
        c1, c2 = make_char(), make_char()
        attach(c1, store=LearnStore(Path(tempfile.mkdtemp())))
        m1 = c1.decide("auction", "rules", "state",
                       ["a", "b", "c"], _suggest_first,
                       rng=random.Random(11))
        m2 = c2.decide("auction", "rules", "state",
                       ["a", "b", "c"], _suggest_first,
                       rng=random.Random(11))
        assert m1 == m2 == "a"

    def test_no_handle_no_journal(self, tmp_path):
        c = make_char()
        c.decide("auction", "rules", "state", ["a", "b"], _suggest_first)
        assert not (tmp_path / "data" / "learn").exists()

    def test_empty_options_journals_nothing(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path))
        assert c.decide("auction", "rules", "state", [], _suggest_first) == ""
        assert h.store.doc(c.id)["entries"] == []

    def test_journal_failure_never_changes_move(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path))

        def _boom(char_id, entry):
            raise RuntimeError("disk gone")

        h.store.journal = _boom
        move = c.decide("auction", "rules", "state",
                        ["a", "b"], _suggest_first)
        assert move == "a"
        # journal_decide itself is best-effort too
        assert h.journal_decide("auction", ["a", "b"], 1) == ""

    def test_drain_decide_ids(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path))
        h.journal_decide("auction", ["a", "b"], 0)
        h.journal_decide("auction", ["a", "b"], 1)
        ids = h.drain_decide_ids()
        assert len(ids) == 2 and ids[0].startswith("L-")
        assert h.drain_decide_ids() == []


# ── grading ────────────────────────────────────────────────────────────

class TestDecideGrading:
    def _graded(self, tmp_path, outcome):
        c = make_char()
        h = attach(c, store=_store(tmp_path))
        eid = h.journal_decide("auction", ["bid 50", "pass"], 0)
        entry = h.grade_decide(eid, outcome)
        return c, h, entry

    def test_outcome_mapping(self, tmp_path):
        for outcome, value in DECIDE_OUTCOMES.items():
            c, h, entry = self._graded(tmp_path / outcome, outcome)
            assert entry["grade"] == value
            assert entry["outcome"] == outcome
            assert entry["grades"][0]["source"] == "system"

    def test_counters_accumulate_per_game_option(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path))
        for outcome in ("won", "won", "lost"):
            eid = h.journal_decide("auction", ["bid 50", "pass"], 0)
            h.grade_decide(eid, outcome)
        eid = h.journal_decide("auction", ["bid 50", "pass"], 1)
        h.grade_decide(eid, "draw")
        eid = h.journal_decide("chess", ["e4", "d4"], 1)
        h.grade_decide(eid, "won")
        stats = h.store.doc(c.id)["decide_stats"]
        assert stats["auction"]["bid 50"] == {"n": 3, "won": 2,
                                              "lost": 1, "draw": 0}
        assert stats["auction"]["pass"] == {"n": 1, "won": 0,
                                            "lost": 0, "draw": 1}
        assert stats["chess"]["d4"] == {"n": 1, "won": 1,
                                        "lost": 0, "draw": 0}

    def test_bad_outcome_unknown_entry_line_entry_fail(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path))
        eid = h.journal_decide("auction", ["a", "b"], 0)
        with pytest.raises(ValueError):
            h.grade_decide(eid, "maybe")
        with pytest.raises(ValueError):
            h.grade_decide("L-999999", "won")
        line_id = h.journal_line("hello there, friend.")
        with pytest.raises(ValueError):
            h.grade_decide(line_id, "won")
        # the failed line-grade left no side effect
        entry = next(e for e in h.store.doc(c.id)["entries"]
                     if e["id"] == line_id)
        assert entry["grade"] is None and entry["grades"] == []
        # and the decide entry is still ungraded
        d_entry = next(e for e in h.store.doc(c.id)["entries"]
                       if e["id"] == eid)
        assert d_entry["grade"] is None

    def test_adjust_skip_weights_prefs_persona(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path), adjust_persona=True)
        before_w = dict(h.store.doc(c.id)["weights"])
        eid = h.journal_decide("auction", ["a", "b"], 0)
        h.grade_decide(eid, "won")
        d = h.store.doc(c.id)
        assert d["weights"] == before_w
        assert d["prefs"]["n_graded_positive"] == 0
        assert d["prefs"]["mean_len"] is None
        assert d["trait_offsets"] == {}
        assert c.persona == {"witty": 0.9, "energetic": 0.8}

    def test_export_excludes_decide(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path))
        line_id = h.journal_line("hello there, friend.")
        h.grade(line_id, 0.5, "owner")
        eid = h.journal_decide("auction", ["a", "b"], 0)
        h.grade_decide(eid, "won")
        rows = h.export_training_data()
        assert len(rows) == 1 and rows[0]["line"] == "hello there, friend."

    def test_report_decide_section_and_no_line_pollution(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path))
        line_id = h.journal_line("hello there, friend.")
        h.grade(line_id, 0.5, "owner")
        eid = h.journal_decide("auction", ["bid 50", "pass"], 0)
        h.grade_decide(eid, "won")
        rep = h.report()
        assert "decide moves: 1 journaled, 1 graded" in rep
        assert "decide auction: bid 50 1/1 won" in rep
        assert "    + hello there, friend." in rep
        # the option text appears ONLY in the decide line, never as a
        # top/bottom line
        assert rep.count("bid 50") == 1

    def test_report_without_lines_has_no_line_listings(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path))
        eid = h.journal_decide("auction", ["a", "b"], 0)
        h.grade_decide(eid, "lost")
        rep = h.report()
        assert "top lines:" not in rep and "bottom lines:" not in rep
        assert "decide moves: 1 journaled, 1 graded" in rep

    def test_session_grades_never_touch_moves(self, tmp_path):
        c = make_char()
        h = attach(c, store=_store(tmp_path))
        h.begin_session()
        h.journal_line("hello there, friend.")
        eid = h.journal_decide("auction", ["a", "b"], 0)
        graded = h.grade_session("deep_conversation")
        assert len(graded) == 1  # only the line
        d_entry = next(e for e in h.store.doc(c.id)["entries"]
                       if e["id"] == eid)
        assert d_entry["grade"] is None
        assert h.drain_decide_ids() == [eid]


# ── backward compatibility ─────────────────────────────────────────────

class TestBackwardCompat:
    def test_old_v1_doc_loads_and_still_adjusts(self, tmp_path):
        import json
        c = make_char()
        st = _store(tmp_path)
        # hand-crafted pre-decide doc: no decide_stats key at all
        doc = {"version": 1, "char_id": c.id, "total": 1,
               "entries": [{"ts": 0.0, "id": "L-000001",
                            "kind": "utterance",
                            "context": "ctx", "line": "hello there.",
                            "candidates": [{"text": "hello there.",
                                            "features": {k: 0.5 for k in
                                                         ("consistency",
                                                          "catchphrase",
                                                          "length_fit",
                                                          "punct_fit",
                                                          "novelty")},
                                            "score": 0.5}],
                            "chosen": 0, "grade": None, "grades": []}],
               "weights": dict(DEFAULT_WEIGHTS),
               "prefs": {"mean_len": None, "emoji_rate": None,
                         "question_rate": None, "exclaim_rate": None,
                         "n_graded_positive": 0},
               "trait_offsets": {}, "seed_persona": {},
               "stats": {"graded": 0, "mean_grade": None}}
        p = st._path(c.id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc), encoding="utf-8")
        fresh = LearnStore(tmp_path / "data")
        d = fresh.doc(c.id)
        assert d["decide_stats"] == {}
        # old line entries still adjust weights exactly as before
        h = attach(c, store=fresh)
        before = dict(d["weights"])
        h.grade(d["entries"][0]["id"], 1.0, "owner")
        assert fresh.doc(c.id)["weights"] != before
        # and a decide entry journals into the old doc fine
        eid = h.journal_decide("auction", ["a", "b"], 1)
        h.grade_decide(eid, "won")
        assert fresh.doc(c.id)["decide_stats"]["auction"]["b"]["won"] == 1


# ── match-end grading ──────────────────────────────────────────────────

class TestMatchGrading:
    def _seats_with_handles(self, tmp_path):
        zara, kilo = make_char("Zara"), make_char("Kilo")
        hz = attach(zara, store=LearnStore(tmp_path / "zdata"))
        hk = attach(kilo, store=LearnStore(tmp_path / "kdata"))
        seats = [AgentSeat(kind="character", character=zara),
                 AgentSeat(kind="character", character=kilo)]
        return zara, kilo, hz, hk, seats

    def test_grade_match_decides_by_key(self, tmp_path):
        zara, kilo, hz, hk, seats = self._seats_with_handles(tmp_path)
        hz.journal_decide("auction", ["a", "b"], 0)
        hk.journal_decide("auction", ["a", "b"], 1)
        _grade_match_decides(seats, SimpleNamespace(key=f"char:{zara.id}"),
                             "auction")
        ze = hz.store.doc(zara.id)["entries"][0]
        ke = hk.store.doc(kilo.id)["entries"][0]
        assert ze["outcome"] == "won" and ke["outcome"] == "lost"
        assert hz.drain_decide_ids() == []  # already drained

    def test_grade_match_decides_draw_and_brain_win(self, tmp_path):
        zara, kilo, hz, hk, seats = self._seats_with_handles(tmp_path)
        hz.journal_decide("auction", ["a", "b"], 0)
        hk.journal_decide("auction", ["a", "b"], 1)
        _grade_match_decides(seats, "draw", "auction")
        assert hz.store.doc(zara.id)["entries"][0]["outcome"] == "draw"
        assert hk.store.doc(kilo.id)["entries"][0]["outcome"] == "draw"
        zara2, kilo2, hz2, hk2, seats2 = self._seats_with_handles(
            Path(tempfile.mkdtemp()))
        hz2.journal_decide("auction", ["a", "b"], 0)
        hk2.journal_decide("auction", ["a", "b"], 1)
        _grade_match_decides(seats2, SimpleNamespace(key="brain:devon"),
                             "auction")
        assert hz2.store.doc(zara2.id)["entries"][0]["outcome"] == "lost"
        assert hk2.store.doc(kilo2.id)["entries"][0]["outcome"] == "lost"

    def test_grade_match_decides_skips_unknown(self, tmp_path):
        zara, kilo, hz, hk, seats = self._seats_with_handles(tmp_path)
        hz.journal_decide("auction", ["a", "b"], 0)
        hk.journal_decide("auction", ["a", "b"], 1)
        for winner in (None, "Zara", SimpleNamespace(key="char:nope"),
                       SimpleNamespace(key="other:1"),
                       SimpleNamespace()):
            _grade_match_decides(seats, winner, "auction")
        assert hz.store.doc(zara.id)["entries"][0]["grade"] is None
        assert hk.store.doc(kilo.id)["entries"][0]["grade"] is None
        # ids still pending for a later, real hook
        assert len(hz.drain_decide_ids()) == 1

    def test_no_handle_seats_skipped_not_crashed(self, tmp_path):
        zara, kilo = make_char("Zara"), make_char("Kilo")
        seats = [AgentSeat(kind="character", character=zara),
                 AgentSeat(kind="character", character=kilo)]
        _grade_match_decides(seats, "draw", "auction")  # no handles: no-op

    def test_full_auction_match_grades_moves(self, tmp_path):
        store = CharacterStore(data_dir=Path(tempfile.mkdtemp()))
        zara, kilo = make_char("Zara"), make_char("Kilo")
        hz = attach(zara, store=LearnStore(tmp_path / "zdata"))
        hk = attach(kilo, store=LearnStore(tmp_path / "kdata"))
        store.save(zara)
        store.save(kilo)
        seats = [AgentSeat(kind="brain", name="Devon"),
                 AgentSeat(kind="character", character=zara),
                 AgentSeat(kind="character", character=kilo)]
        res = run_agent_match("auction", seats, _suggest_first, seed=42,
                              store=store, max_turns=60)
        assert res.moves > 5
        assert res.winner in ("Devon", "Zara", "Kilo", "draw")
        for char, h in ((zara, hz), (kilo, hk)):
            moves = [e for e in h.store.doc(char.id)["entries"]
                     if e["kind"] == "decide"]
            assert len(moves) > 0
            assert all(e["grade"] is not None for e in moves)
            if res.winner == "draw":
                assert all(e["outcome"] == "draw" for e in moves)
            elif res.winner == char.name:
                assert all(e["outcome"] == "won" for e in moves)
            else:
                assert all(e["outcome"] == "lost" for e in moves)
            total = sum(c["n"] for g in
                        h.store.doc(char.id)["decide_stats"].values()
                        for c in g.values())
            assert total == len(moves)
            assert h.drain_decide_ids() == []

    def test_match_moves_identical_with_and_without_handles(self, tmp_path):
        def _run(with_handles):
            store = CharacterStore(data_dir=Path(tempfile.mkdtemp()))
            zara, kilo = make_char("Zara"), make_char("Kilo")
            if with_handles:
                attach(zara, store=LearnStore(tmp_path / "zdata"))
                attach(kilo, store=LearnStore(tmp_path / "kdata"))
            store.save(zara)
            store.save(kilo)
            seats = [AgentSeat(kind="brain", name="Devon"),
                     AgentSeat(kind="character", character=zara),
                     AgentSeat(kind="character", character=kilo)]
            return run_agent_match("auction", seats, _suggest_first,
                                   seed=42, store=store, max_turns=60)

        a, b = _run(True), _run(False)
        assert a.moves == b.moves and a.winner == b.winner
        assert a.transcript == b.transcript
