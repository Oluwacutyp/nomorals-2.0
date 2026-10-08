"""Offline tests for build-map #83 — matching (batches, stable, questionnaire, chat)."""

import os
import tempfile

from nomorals.matching import (
    Candidate,
    DailyBatchStore,
    Questionnaire,
    curate_daily,
    stable_match,
    verify_stable,
)
from nomorals.matching.chat import control_match


def _tmpdb(name="t.db"):
    return os.path.join(tempfile.mkdtemp(), name)


# ── curate_daily ──────────────────────────────────────────────

def test_curate_ranks_by_preference_and_quality():
    a = Candidate("a", "Remote gig", tags=("remote", "python"), quality=0.9)
    b = Candidate("b", "Office gig", tags=("office",), quality=0.9)
    c = Candidate("c", "Remote low", tags=("remote",), quality=0.1)
    out = curate_daily([b, c, a], {"remote": 5.0}, n=3)
    assert [x.candidate_id for x in out] == ["a", "c", "b"]


def test_curate_n_limit_and_empty():
    cs = [Candidate(str(i), "t%d" % i) for i in range(10)]
    assert len(curate_daily(cs, n=5)) == 5
    assert curate_daily([], {"x": 1.0}) == []
    assert curate_daily(None) == []


def test_curate_exclude_ids():
    cs = [Candidate("a", "A", quality=1.0), Candidate("b", "B", quality=0.9)]
    out = curate_daily(cs, exclude_ids={"a"})
    assert [x.candidate_id for x in out] == ["b"]


def test_curate_never_raises_on_garbage_prefs():
    cs = [Candidate("a", "A")]
    assert curate_daily(cs, {"x": "not-a-number"}) == [cs[0]]
    assert curate_daily(cs, "garbage") == [cs[0]]  # type: ignore[arg-type]


# ── DailyBatchStore ───────────────────────────────────────────

def test_store_add_and_candidates_roundtrip():
    s = DailyBatchStore(db_path=_tmpdb())
    assert s.add_candidate(Candidate("g1", "Gig one", tags=("remote",),
                                     attributes={"budget": 50000}),
                           surface="gig")
    cs = s.candidates("gig")
    assert len(cs) == 1 and cs[0].tags == ("remote",)
    assert cs[0].attributes == {"budget": 50000}


def test_today_stable_same_day():
    s = DailyBatchStore(db_path=_tmpdb())
    for i in range(8):
        s.add_candidate(Candidate("g%d" % i, "Gig %d" % i, quality=0.5 + i / 20))
    first = s.today("gig", "owner", {"remote": 3.0}, n=3)
    second = s.today("gig", "owner", {"remote": 3.0}, n=3)
    assert [c.candidate_id for c in first] == [c.candidate_id for c in second]
    assert len(first) == 3


def test_today_new_day_no_repeat():
    s = DailyBatchStore(db_path=_tmpdb())
    for i in range(8):
        s.add_candidate(Candidate("g%d" % i, "Gig %d" % i, quality=0.9 - i / 20))
    day1 = {c.candidate_id for c in s.today("gig", "owner", n=5, now=1_700_000_000)}
    day2 = {c.candidate_id for c in s.today("gig", "owner", n=5, now=1_700_086_400)}
    # 5 shown on day 1; day 2 must avoid them (8 in pool → 3 left)
    assert day1.isdisjoint(day2)
    assert len(day2) == 3


def test_today_empty_pool_honest():
    s = DailyBatchStore(db_path=_tmpdb())
    assert s.today("gig", "owner") == []


def test_store_never_raises_bad_db():
    d = tempfile.mkdtemp()  # a directory, not a file — sqlite can't open it
    s = DailyBatchStore(db_path=d)
    assert s.candidates() == []
    assert s.today() == []
    assert s.add_candidate(Candidate("x", "X")) is False


# ── stable_match ──────────────────────────────────────────────

def test_stable_basic_and_verified():
    props = ["f1", "f2"]
    revs = ["c1", "c2"]
    pp = {"f1": ["c1", "c2"], "f2": ["c1", "c2"]}
    rp = {"c1": ["f2", "f1"], "c2": ["f1", "f2"]}
    res = stable_match(props, revs, pp, rp)
    assert res["f1"] == "c1" and res["f2"] == "c2" or res["f1"] == "c2"
    stable, blockers = verify_stable(res, pp, rp)
    assert stable and blockers == []


def test_stable_proposer_optimal():
    # Classic: both proposers want r1; r1 wants p2.
    pp = {"p1": ["r1", "r2"], "p2": ["r1", "r2"]}
    rp = {"r1": ["p1", "p2"], "r2": ["p1", "p2"]}
    res = stable_match(["p1", "p2"], ["r1", "r2"], pp, rp)
    assert res["p1"] == "r1"  # proposer-optimal: p1 keeps r1
    assert res["p2"] == "r2"
    stable, _ = verify_stable(res, pp, rp)
    assert stable


def test_stable_unequal_sets():
    res = stable_match(["p1", "p2", "p3"], ["r1"],
                       {"p1": ["r1"], "p2": ["r1"], "p3": ["r1"]},
                       {"r1": ["p1", "p2", "p3"]})
    matched = [p for p, r in res.items() if r]
    assert len(matched) == 1 and res["p1"] == "r1"
    assert res["p3"] is None


def test_stable_empty_and_garbage():
    assert stable_match([], [], {}, {}) == {}
    res = stable_match(["p1"], ["r1"], {}, {})
    assert res["p1"] == "r1"  # no prefs → still matches, still stable
    stable, _ = verify_stable(res, {}, {})
    assert stable


def test_verify_detects_blocking_pair():
    # Forced unstable: p1-r2, p2-r1 but p1↔r1 prefer each other.
    res = {"p1": "r2", "p2": "r1"}
    pp = {"p1": ["r1", "r2"], "p2": ["r2", "r1"]}
    rp = {"r1": ["p1", "p2"], "r2": ["p2", "p1"]}
    stable, blockers = verify_stable(res, pp, rp)
    assert not stable and ("p1", "r1") in blockers


# ── Questionnaire ─────────────────────────────────────────────

def test_questionnaire_starters_answer_weights():
    q = Questionnaire(owner="u1", db_path=_tmpdb())
    starters = q.ensure_starters("gig")
    assert len(starters) == 4
    assert len(q.unanswered()) == 4
    qid = starters[0].question_id
    assert q.answer(qid, 5)
    assert len(q.unanswered()) == 3
    w = q.weights()
    assert w[starters[0].attribute] == 5.0


def test_questionnaire_idempotent_starters():
    q = Questionnaire(owner="u2", db_path=_tmpdb())
    q.ensure_starters("gig")
    q.ensure_starters("gig")
    assert len(q.unanswered()) == 4


def test_dealbreaker_filters_before_scoring():
    q = Questionnaire(owner="u3", db_path=_tmpdb())
    assert q.set_dealbreaker("budget", "budget", "le", 100000)
    cands = [
        {"candidate_id": "a", "attributes": {"budget": 50000}},
        {"candidate_id": "b", "attributes": {"budget": 200000}},
        {"candidate_id": "c", "attributes": {}},
    ]
    out = q.filter_dealbreakers(cands)
    assert [c["candidate_id"] for c in out] == ["a"]


def test_score_weighted_and_neutral_without_answers():
    q = Questionnaire(owner="u4", db_path=_tmpdb())
    assert q.score({"attributes": {"remote": 1}}) == 0.5
    q.ensure_starters("gig")
    for item in q.unanswered():
        q.answer(item.question_id, 5 if item.attribute == "remote" else 1)
    hi = q.score({"attributes": {"remote": 1}})
    lo = q.score({"attributes": {}})
    assert hi > lo


def test_rankings_roundtrip():
    q = Questionnaire(owner="u5", db_path=_tmpdb())
    assert q.set_ranking("Ada", "proposer", ["c1", "c2"])
    assert q.set_ranking("Bob", "proposer", ["c2", "c1"])
    assert q.set_ranking("Cli", "reviewer", ["Ada", "Bob"])
    assert not q.set_ranking("X", "bogus", ["c1"])
    assert q.rankings("proposer") == {"Ada": ["c1", "c2"], "Bob": ["c2", "c1"]}
    assert q.rankings("reviewer") == {"Cli": ["Ada", "Bob"]}


def test_questionnaire_never_raises_bad_db():
    q = Questionnaire(owner="u6", db_path=tempfile.mkdtemp())
    assert q.ensure_starters("gig") == []
    assert q.unanswered() == []
    assert q.weights() == {}
    assert q.score({}) == 0.5


# ── chat ──────────────────────────────────────────────────────

class _Ctx:
    def __init__(self):
        self.matching_batch_store = DailyBatchStore(db_path=_tmpdb())
        self.matching_questionnaire = Questionnaire(owner="chat",
                                                    db_path=_tmpdb("q.db"))
        self.matching_owner = "chat"
        self.matching_surface = "gig"


def test_chat_help_and_garbage():
    assert "daily" in control_match("help")
    assert "daily" in control_match("")
    assert control_match("frobnicate") == control_match("help")


def test_chat_add_and_daily():
    ctx = _Ctx()
    out = control_match("add g1 | Logo design | design,remote | budget=50000",
                        context=ctx)
    assert "registered" in out
    out = control_match("daily", context=ctx)
    assert "Logo design" in out
    # Stable within the day:
    out2 = control_match("daily", context=ctx)
    assert out2 == out


def test_chat_daily_empty_honest():
    ctx = _Ctx()
    assert "no candidates" in control_match("daily", context=ctx).lower()


def test_chat_questionnaire_answer_flow():
    ctx = _Ctx()
    out = control_match("questionnaire", context=ctx)
    assert "answer more" in out
    qid = ctx.matching_questionnaire.unanswered()[0].question_id
    out = control_match("answer %s 5" % qid, context=ctx)
    assert "recorded" in out and "★★★★★" in out


def test_chat_dealbreaker_blocks_daily():
    ctx = _Ctx()
    control_match("add g1 | Cheap | x | budget=50000", context=ctx)
    control_match("add g2 | Pricey | x | budget=500000", context=ctx)
    control_match("dealbreaker budget budget le 100000", context=ctx)
    out = control_match("daily", context=ctx)
    assert "Cheap" in out and "Pricey" not in out


def test_chat_rank_and_run_stable():
    ctx = _Ctx()
    out = control_match("rank proposer Ada c1,c2", context=ctx)
    assert "Ada's proposer ranking saved" in out
    control_match("rank proposer Bob c2,c1", context=ctx)
    control_match("rank reviewer C1 Ada,Bob", context=ctx)
    control_match("rank reviewer C2 Bob,Ada", context=ctx)
    out = control_match("run", context=ctx)
    assert "Ada ↔" in out and "Bob ↔" in out


def test_chat_run_needs_both_sides():
    ctx = _Ctx()
    control_match("rank proposer Ada c1", context=ctx)
    assert "both sides" in control_match("run", context=ctx)


def test_chat_never_raises():
    ctx = _Ctx()
    for tail in ["add", "add |", "answer", "answer q x", "dealbreaker a b c",
                 "dealbreaker b budget le notanumber", "rank", "rank x y",
                 "daily gig extra words here"]:
        assert isinstance(control_match(tail, context=ctx), str)
    assert isinstance(control_match("daily"), str)  # no context at all
