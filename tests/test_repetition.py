"""Knowledge-state-aware resurfacing + spaced repetition (build-map #40).

All offline: SQLite in tmp dirs / :memory:, no network, no LLM.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

from nomorals.memory.delivery import (
    DeliveryScorer,
    KnowledgeState,
    score_delivery,
)
from nomorals.memory.repetition import (
    GRADE_AGAIN,
    GRADE_EASY,
    GRADE_GOOD,
    GRADE_HARD,
    RepetitionScheduler,
)


def _sched(tmp_path):
    return RepetitionScheduler(db=str(tmp_path / "rep.db"))


def _ks():
    return KnowledgeState(db=":memory:")


def _mem(mid, content, importance=0.5):
    return SimpleNamespace(id=mid, content=content, importance=importance)


# ── SM-2 scheduling ───────────────────────────────────────────────────────

class TestScheduling:
    def test_schedule_creates_due_card(self, tmp_path):
        s = _sched(tmp_path)
        assert s.schedule("m1", "my girlfriend's name is Ada")
        card = s.get("m1")
        assert card is not None
        assert card.ease == 2.5
        assert card.reps == 0
        assert card.due_at <= time.time() + 1  # due immediately

    def test_schedule_is_idempotent(self, tmp_path):
        s = _sched(tmp_path)
        s.schedule("m1", "first text")
        s.schedule("m1", "updated text")
        assert s.count() == 1
        assert s.get("m1").text == "updated text"

    def test_good_grows_intervals(self, tmp_path):
        s = _sched(tmp_path)
        s.schedule("m1", "fact one")
        c = s.review("m1", GRADE_GOOD)
        assert c.reps == 1 and c.interval_days == 1.0
        c = s.review("m1", GRADE_GOOD)
        assert c.reps == 2 and c.interval_days == 6.0
        c = s.review("m1", GRADE_GOOD)
        assert c.reps == 3
        assert abs(c.interval_days - 6.0 * 2.5) < 0.01  # × ease

    def test_again_shrinks_and_counts_lapse(self, tmp_path):
        s = _sched(tmp_path)
        s.schedule("m1", "fact one")
        s.review("m1", GRADE_GOOD)
        s.review("m1", GRADE_GOOD)
        c = s.review("m1", GRADE_AGAIN)
        assert c.lapses == 1
        assert c.reps == 0
        assert c.interval_days == 1.0  # relearn tomorrow
        assert c.ease < 2.5  # ease dropped

    def test_hard_grows_slowly(self, tmp_path):
        s = _sched(tmp_path)
        s.schedule("m1", "fact one")
        s.review("m1", GRADE_GOOD)
        s.review("m1", GRADE_GOOD)  # interval 6
        c = s.review("m1", GRADE_HARD)
        assert abs(c.interval_days - 6.0 * 1.2) < 0.01

    def test_easy_boosts(self, tmp_path):
        s = _sched(tmp_path)
        s.schedule("m1", "fact one")
        c = s.review("m1", GRADE_EASY)
        assert c.ease > 2.5
        assert c.interval_days >= 4.0

    def test_invalid_grade_rejected(self, tmp_path):
        s = _sched(tmp_path)
        s.schedule("m1", "fact one")
        assert s.review("m1", 99) is None
        assert s.review("m1", 0) is None

    def test_review_unknown_id(self, tmp_path):
        s = _sched(tmp_path)
        assert s.review("nope", GRADE_GOOD) is None

    def test_due_ordering_oldest_first(self, tmp_path):
        s = _sched(tmp_path)
        s.schedule("m1", "first")
        time.sleep(0.02)
        s.schedule("m2", "second")
        due = s.due(limit=5)
        assert [c.memory_id for c in due] == ["m1", "m2"]

    def test_due_respects_limit(self, tmp_path):
        s = _sched(tmp_path)
        for i in range(5):
            s.schedule(f"m{i}", f"fact {i}")
        assert len(s.due(limit=2)) == 2

    def test_bury_pushes_down_curve(self, tmp_path):
        s = _sched(tmp_path)
        s.schedule("m1", "I know this already")
        assert s.bury("m1", days=30) is True
        assert s.due(limit=5) == []  # not due for 30 days
        card = s.get("m1")
        assert card.buried_until > time.time() + 29 * 86400

    def test_remove(self, tmp_path):
        s = _sched(tmp_path)
        s.schedule("m1", "fact one")
        assert s.remove("m1") is True
        assert s.get("m1") is None
        assert s.count() == 0

    def test_never_raises_on_broken_db(self):
        s = RepetitionScheduler(db="/nonexistent-dir-xyz/rep.db")
        assert s.schedule("m1", "x") is False
        assert s.review("m1", GRADE_GOOD) is None
        assert s.due(limit=5) == []
        assert s.bury("m1") is False
        assert s.get("m1") is None
        assert s.count() == 0
        assert s.remove("m1") is False


# ── knowledge state ───────────────────────────────────────────────────────

class TestKnowledgeState:
    def test_mark_known_then_knows(self):
        ks = _ks()
        assert ks.mark_known("m1", "the sky is blue") is True
        assert ks.knows_id("m1") is True
        assert ks.knows("the sky is blue") is True

    def test_knows_is_fuzzy(self):
        ks = _ks()
        ks.mark_known("m1", "my girlfriend had an operation last week")
        # same fact, different words → still known
        assert ks.knows("my girlfriend had an operation last week!") is True
        assert ks.knows("the weather is nice today") is False

    def test_knows_empty_is_false(self):
        ks = _ks()
        assert ks.knows("") is False
        assert ks.knows("   ") is False

    def test_mark_surfaced_counts(self):
        ks = _ks()
        ks.mark_surfaced("m1", "some fact")
        ks.mark_surfaced("m1", "some fact")
        assert ks.surfaced_count("m1") == 2
        assert ks.surfaced_count("m2") == 0

    def test_recent_briefings_rule(self):
        ks = _ks()
        ks.mark_briefing("b1", ["m1", "m2"])
        ks.mark_briefing("b2", ["m3"])
        ks.mark_briefing("b3", ["m4"])
        assert ks.was_in_recent_briefings("m1") is True
        assert ks.was_in_recent_briefings("m3") is True
        assert ks.was_in_recent_briefings("m9") is False
        # 4th briefing pushes b1 out of the last-3 window
        ks.mark_briefing("b4", ["m5"])
        assert ks.was_in_recent_briefings("m1") is False
        assert ks.was_in_recent_briefings("m3") is True

    def test_never_raises_on_broken_db(self):
        ks = KnowledgeState(db="/nonexistent-dir-xyz/ks.db")
        assert ks.mark_known("m1", "x") is False
        assert ks.mark_surfaced("m1") is False
        assert ks.mark_briefing("b1", ["m1"]) is False
        assert ks.knows_id("m1") is False
        assert ks.knows("anything at all here") is False
        assert ks.surfaced_count("m1") == 0
        assert ks.was_in_recent_briefings("m1") is False


# ── novelty in the scorer ─────────────────────────────────────────────────

class TestNovelty:
    def test_novelty_score_none_state_is_fully_novel(self):
        scorer = DeliveryScorer()
        assert scorer.novelty_score(_mem("m1", "anything"), None) == 1.0

    def test_novelty_zero_when_known(self):
        scorer = DeliveryScorer()
        ks = _ks()
        ks.mark_known("m1", "the sky is blue")
        assert scorer.novelty_score(_mem("m1", "the sky is blue"), ks) == 0.0

    def test_novelty_zero_when_in_recent_briefing(self):
        scorer = DeliveryScorer()
        ks = _ks()
        ks.mark_briefing("b1", ["m1"])
        assert scorer.novelty_score(_mem("m1", "brand new wording here"), ks) == 0.0

    def test_novelty_one_when_new(self):
        scorer = DeliveryScorer()
        ks = _ks()
        ks.mark_known("other", "something else entirely")
        assert scorer.novelty_score(_mem("m1", "completely fresh fact"), ks) == 1.0

    def test_score_backward_compat_no_state(self):
        # Without a knowledge state the formula must match the old one:
        # 0.4*topic + 0.25*freshness + 0.2*tone + 0.15*importance
        scorer = DeliveryScorer()
        mem = _mem("m1", "happy birthday party fun", importance=0.8)
        (res,) = scorer.score([mem], current_text="what a great day")
        expected = (0.4 * res.topic_fit + 0.25 * res.freshness
                    + 0.2 * res.tone_fit + 0.15 * res.importance_boost)
        assert abs(res.score - expected) < 1e-9
        assert res.novelty == 1.0
        assert "already-known" not in res.reasons

    def test_score_with_state_suppresses_known(self):
        scorer = DeliveryScorer()
        ks = _ks()
        mem = _mem("m1", "happy birthday party fun", importance=1.0)
        plain = scorer.score([mem], current_text="what a great day")[0]
        ks.mark_known("m1", "happy birthday party fun")
        known = scorer.score([mem], current_text="what a great day",
                             knowledge_state=ks)[0]
        assert known.novelty == 0.0
        assert "already-known" in known.reasons
        assert known.score < plain.score

    def test_score_with_state_keeps_novel(self):
        scorer = DeliveryScorer()
        ks = _ks()
        ks.mark_known("other", "unrelated old fact")
        mem = _mem("m1", "happy birthday party fun", importance=0.8)
        res = scorer.score([mem], current_text="what a great day",
                           knowledge_state=ks)[0]
        assert res.novelty == 1.0
        assert "already-known" not in res.reasons

    def test_score_delivery_convenience_passes_state(self):
        ks = _ks()
        ks.mark_known("m1", "old news here")
        res = score_delivery([_mem("m1", "old news here")], knowledge_state=ks)
        assert res[0].novelty == 0.0


# ── briefing section ──────────────────────────────────────────────────────

class TestFromYourPast:
    def test_no_cards_no_section(self):
        from nomorals.agents.morning_briefing import FromYourPastProvider
        # fresh home-path scheduler has no due cards (or is broken → None)
        sec = FromYourPastProvider().collect(None, 0.0)
        # Either None (no cards) — never raises, never fabricates
        assert sec is None or sec.name == "from_past"

    def test_section_format(self, tmp_path, monkeypatch):
        from nomorals.agents import morning_briefing as mb
        from nomorals.memory import repetition as rep_mod
        sched = RepetitionScheduler(db=str(tmp_path / "rep.db"))
        sched.schedule("fact-1", "Ada's birthday is March 3rd")
        sched.schedule("fact-2", "prefers morning briefings")
        monkeypatch.setattr(rep_mod, "RepetitionScheduler",
                            lambda *a, **k: sched)
        provider = mb.FromYourPastProvider.__new__(mb.FromYourPastProvider)
        sec = provider._collect(None, 0.0)
        assert sec is not None
        assert sec.name == "from_past"
        assert "🕰️" in sec.title
        text = sec.render_text()
        assert "Ada's birthday is March 3rd" in text
        assert "prefers morning briefings" in text
        assert "first noted" in text
        assert sec.priority == 90  # yields to urgent sections
        # items carry memory ids for knowledge-state bookkeeping
        assert {i["memory_id"] for i in sec.items} == {"fact-1", "fact-2"}

    def test_section_respects_limit(self, tmp_path, monkeypatch):
        from nomorals.agents import morning_briefing as mb
        from nomorals.memory import repetition as rep_mod
        sched = RepetitionScheduler(db=str(tmp_path / "rep.db"))
        for i in range(10):
            sched.schedule(f"f{i}", f"memory number {i}")
        monkeypatch.setattr(rep_mod, "RepetitionScheduler",
                            lambda *a, **k: sched)
        provider = mb.FromYourPastProvider.__new__(mb.FromYourPastProvider)
        sec = provider._collect(None, 0.0)
        assert sec is not None
        assert len(sec.items) == 3  # max 3 resurfaced

    def test_never_raises(self):
        from nomorals.agents.morning_briefing import FromYourPastProvider
        # even with a hostile scheduler it must not raise
        assert FromYourPastProvider().collect(object(), 0.0) is None or True
