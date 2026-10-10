"""Sweep tests for nomorals/matching/ — mined-then-built upgrade.

Covers: Gale-Shapley + verify, Hospital-Resident capacities, Irving
stable roommates (brute-force cross-checked), TTC, serial dictatorship,
Hungarian assignment, MMR diversity, Thompson exploration, feedback loop,
OkCupid-pattern questionnaire (acceptable sets, 1/10/50/250 ladder,
two-sided match %), Feeld sections, deal-breaker predicates, and the
/match chat surface. Everything uses temp DBs; the module never raises.
"""

import itertools
import os
import random
import sqlite3
import tempfile
import time
import unittest
from types import SimpleNamespace

from nomorals.matching import (
    IMPORTANCE_WEIGHTS,
    Candidate,
    DailyBatchStore,
    Questionnaire,
    curate_daily,
    explain_pick,
    mmr_rerank,
    optimal_assignment,
    roommate_match,
    serial_dictatorship,
    stable_match,
    stable_match_capacities,
    top_trading_cycles,
    verify_roommate_stable,
    verify_stable,
)
from nomorals.matching.chat import control_match


def _tmpdb():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)
    return path


def _ctx(**kw):
    base = {
        "matching_batch_store": DailyBatchStore(_tmpdb()),
        "matching_questionnaire": Questionnaire("owner", _tmpdb()),
        "matching_owner": "owner",
        "matching_surface": "gig",
    }
    base.update(kw)
    return SimpleNamespace(**base)


# ── brute-force oracles (ground truth for the tricky algorithms) ──

def _brute_roommates(people, prefs):
    def pairings(idxs):
        if not idxs:
            yield []
            return
        a = idxs[0]
        for i in range(1, len(idxs)):
            b = idxs[i]
            for rest in pairings(idxs[1:i] + idxs[i + 1:]):
                yield [(a, b)] + rest

    out = []
    for pm in pairings(list(range(len(people)))):
        d = {}
        for a, b in pm:
            d[people[a]] = people[b]
            d[people[b]] = people[a]
        ok, _ = verify_roommate_stable(d, prefs)
        if ok:
            out.append(d)
    return out


def _brute_assignment(scores):
    rows = list(scores)
    cols = []
    for r in rows:
        for c in scores[r]:
            if c not in cols:
                cols.append(c)
    best = 0.0
    if len(rows) <= len(cols):
        for pc in itertools.permutations(cols, len(rows)):
            best = max(best, sum(scores[rows[i]][pc[i]]
                                for i in range(len(rows))))
    else:
        for pr in itertools.permutations(rows, len(cols)):
            best = max(best, sum(scores[pr[j]][cols[j]]
                                for j in range(len(cols))))
    return best


# ── stable_match ──────────────────────────────────────────────────

class TestStableMatch(unittest.TestCase):
    def test_daffidwilde_documented_example(self):
        # daffidwilde/matching README: game.solve() == {A: E, B: D, C: F}
        pp = {"A": ["D", "E", "F"], "B": ["D", "F", "E"], "C": ["F", "D", "E"]}
        rp = {"D": ["B", "C", "A"], "E": ["A", "C", "B"], "F": ["C", "B", "A"]}
        r = stable_match(["A", "B", "C"], ["D", "E", "F"], pp, rp)
        self.assertEqual(dict(r), {"A": "E", "B": "D", "C": "F"})
        ok, blockers = verify_stable(r, pp, rp)
        self.assertTrue(ok)
        self.assertEqual(blockers, [])

    def test_proposer_optimal_and_unequal(self):
        pp = {"a": ["x", "y"], "b": ["x", "y"], "c": ["y"]}
        rp = {"x": ["a", "b"], "y": ["b", "a", "c"]}
        r = stable_match(["a", "b", "c"], ["x", "y"], pp, rp)
        ok, _ = verify_stable(r, pp, rp)
        self.assertTrue(ok)
        # a proposes first, gets his top choice (proposer-optimal)
        self.assertEqual(r["a"], "x")

    def test_unknown_ids_ignored_never_raises(self):
        r = stable_match(["a"], ["x"], {"a": ["zzz", "x"]}, {"x": ["qqq"]})
        self.assertEqual(r["a"], "x")
        r = stable_match([], [], {}, {})
        self.assertEqual(dict(r), {})
        r = stable_match(None, None, None, None)
        self.assertEqual(dict(r), {})

    def test_capacities_mentor_takes_two(self):
        pp = {"a": ["M", "N"], "b": ["M", "N"], "c": ["N", "M"]}
        rp = {"M": ["a", "b", "c"], "N": ["c", "a", "b"]}
        r = stable_match_capacities(["a", "b", "c"], ["M", "N"],
                                    pp, rp, {"M": 2, "N": 1})
        self.assertEqual(r.rosters(), {"M": ["a", "b"], "N": ["c"]})
        self.assertIn("2/2", r.summary())
        # default capacity is 1 == plain stable_match
        r1 = stable_match_capacities(["a", "b"], ["x", "y"],
                                     {"a": ["x"], "b": ["y"]},
                                     {"x": ["a"], "y": ["b"]}, {})
        self.assertEqual(dict(r1), {"a": "x", "b": "y"})


# ── roommates (Irving) ────────────────────────────────────────────

class TestRoommates(unittest.TestCase):
    def test_solvable_instance_verified(self):
        people = ["a", "b", "c", "d"]
        prefs = {"a": ["b", "c", "d"], "b": ["a", "d", "c"],
                 "c": ["d", "a", "b"], "d": ["c", "b", "a"]}
        r = roommate_match(people, prefs)
        self.assertTrue(r.stable)
        ok, _ = verify_roommate_stable(dict(r), prefs)
        self.assertTrue(ok)

    def test_unsolvable_reported_honestly(self):
        # Gale & Shapley's classic 4-person counterexample.
        prefs = {"1": ["2", "3", "4"], "2": ["3", "1", "4"],
                 "3": ["1", "2", "4"], "4": ["1", "2", "3"]}
        self.assertEqual(_brute_roommates(["1", "2", "3", "4"], prefs), [])
        r = roommate_match(["1", "2", "3", "4"], prefs)
        self.assertFalse(r.stable)
        self.assertIn("no stable pairing", r.summary())

    def test_random_instances_match_brute_force(self):
        rng = random.Random(20261010)
        for _ in range(30):
            people = ["p%d" % i for i in range(6)]
            prefs = {}
            for p in people:
                others = [q for q in people if q != p]
                rng.shuffle(others)
                prefs[p] = others
            r = roommate_match(people, prefs)
            d = dict(r)
            bf = _brute_roommates(people, prefs)
            if r.stable:
                self.assertTrue(bf, "claimed stable but none exists")
                ok, _ = verify_roommate_stable(d, prefs)
                self.assertTrue(ok, "claimed stable but blocking pairs exist")
                self.assertEqual(sorted(d.values()), sorted(people))
            else:
                self.assertEqual(bf, [], "missed an existing stable pairing")

    def test_incomplete_lists(self):
        prefs = {"a": ["b"], "b": ["a"], "c": ["d"], "d": ["c"]}
        r = roommate_match(["a", "b", "c", "d"], prefs)
        self.assertTrue(r.stable)
        self.assertEqual(dict(r),
                         {"a": "b", "b": "a", "c": "d", "d": "c"})

    def test_odd_pool_sits_one_out(self):
        people = ["a", "b", "c"]
        prefs = {"a": ["b", "c"], "b": ["a", "c"], "c": ["a", "b"]}
        r = roommate_match(people, prefs)
        self.assertTrue(r.stable)
        self.assertEqual(len(r.sat_out), 1)
        paired = [p for p in people if p not in r.sat_out]
        self.assertEqual(len(paired), 2)
        self.assertEqual(r[paired[0]], paired[1])


# ── allocation mechanisms ─────────────────────────────────────────

class TestAllocation(unittest.TestCase):
    def test_ttc_swap_and_cycle(self):
        t = top_trading_cycles(["a", "b"], {"a": ["y", "x"], "b": ["x", "y"]},
                               {"a": "x", "b": "y"})
        self.assertEqual(t, {"a": "y", "b": "x"})
        t = top_trading_cycles(
            ["a", "b", "c"],
            {"a": ["y", "x", "z"], "b": ["z", "y", "x"], "c": ["x", "z", "y"]},
            {"a": "x", "b": "y", "c": "z"})
        self.assertEqual(t, {"a": "y", "b": "z", "c": "x"})

    def test_ttc_no_trade_keeps_endowment(self):
        t = top_trading_cycles(["a", "b"], {"a": ["zzz"], "b": ["y", "x"]},
                               {"a": "x", "b": "y"})
        self.assertEqual(t["a"], "x")

    def test_serial_dictatorship(self):
        s = serial_dictatorship(["a", "b"], {"a": ["y", "x"], "b": ["x", "y"]},
                                ["x", "y"], order=["a", "b"])
        self.assertEqual(s, {"a": "y", "b": "x"})
        s1 = serial_dictatorship(["a", "b"], {"a": ["y"], "b": ["x"]},
                                 ["x", "y"], seed=42)
        s2 = serial_dictatorship(["a", "b"], {"a": ["y"], "b": ["x"]},
                                 ["x", "y"], seed=42)
        self.assertEqual(s1, s2)  # seeded RSD is deterministic

    def test_optimal_assignment_matches_brute_force(self):
        rng = random.Random(99)
        for _ in range(60):
            n, m = rng.randint(1, 4), rng.randint(1, 4)
            rows = ["r%d" % i for i in range(n)]
            cols = ["c%d" % j for j in range(m)]
            scores = {r: {c: rng.random() * 10 for c in cols} for r in rows}
            got = optimal_assignment(scores)
            want = _brute_assignment(scores)
            got_tot = sum(scores[r][c] for r, c in got.items() if c)
            self.assertAlmostEqual(got_tot, want, places=9)
        # rectangular: more rows than cols -> some unmatched
        got = optimal_assignment({"a": {"x": 5}, "b": {"x": 9}, "c": {"x": 1}})
        self.assertEqual(got["b"], "x")
        self.assertIsNone(got["a"])
        self.assertIsNone(got["c"])


# ── batches ───────────────────────────────────────────────────────

def _cand(cid, tags, quality=0.5, **attrs):
    return Candidate(candidate_id=cid, title=cid, tags=tuple(tags),
                     quality=quality, attributes=attrs)


class TestBatches(unittest.TestCase):
    def test_curate_daily_scoring_order(self):
        pool = [_cand("a", ["remote"], 0.9), _cand("b", ["onsite"], 0.9),
                _cand("c", ["remote"], 0.1)]
        out = curate_daily(pool, {"remote": 5}, 3, diversify=False)
        self.assertEqual([c.candidate_id for c in out], ["a", "c", "b"])

    def test_mmr_diversity_beats_pure_relevance(self):
        dupes = [_cand("d%d" % i, ["python", "django"], 0.95)
                 for i in range(5)]
        odd = _cand("odd", ["rust"], 0.9)
        pool = dupes + [odd]
        prefs = {"python": 5, "django": 5, "rust": 4}
        plain = curate_daily(pool, prefs, 3, diversify=False)
        self.assertNotIn("odd", [c.candidate_id for c in plain])
        diverse = curate_daily(pool, prefs, 3, diversify=True, mmr_lambda=0.5)
        self.assertIn("odd", [c.candidate_id for c in diverse])

    def test_mmr_lambda1_is_pure_relevance(self):
        pool = [_cand("a", ["x"], 0.9), _cand("b", ["x"], 0.8)]
        scores = {"a": 0.9, "b": 0.8}
        out = mmr_rerank(pool, scores, 2, lambda_=1.0)
        self.assertEqual([c.candidate_id for c in out], ["a", "b"])

    def test_explore_is_seeded_deterministic(self):
        pool = [_cand("c%d" % i, ["t%d" % i], 0.5 + i * 0.05)
                for i in range(8)]
        a = curate_daily(pool, {}, 4, explore=0.5, seed=7)
        b = curate_daily(pool, {}, 4, explore=0.5, seed=7)
        self.assertEqual([c.candidate_id for c in a],
                         [c.candidate_id for c in b])

    def test_explore_samples_proven_winner(self):
        pool = [_cand("w%d" % i, ["t"], 0.5) for i in range(6)]
        stats = {"w0": (10, 10)}  # 10/10 likes: Thompson should find it
        out = curate_daily(pool, {}, 3, explore=0.34, stats=stats, seed=3,
                           diversify=False)
        ids = [c.candidate_id for c in out]
        self.assertIn("w0", ids)

    def test_explain_pick_names_contributors(self):
        c = _cand("g1", ["remote", "python"], 0.8)
        text = explain_pick(c, {"remote": 5, "python": 3})
        self.assertIn("remote", text)
        self.assertIn("score", text)

    def test_today_stable_all_day_fresh_tomorrow(self):
        store = DailyBatchStore(_tmpdb())
        for i in range(8):
            store.add_candidate(_cand("g%d" % i, ["t%d" % (i % 3)], 0.5))
        now = time.time()
        day1 = [c.candidate_id for c in store.today("gig", "o", {}, 3,
                                                   now=now, explore=0,
                                                   diversify=False)]
        again = [c.candidate_id for c in store.today("gig", "o", {}, 3,
                                                    now=now + 3600, explore=0,
                                                    diversify=False)]
        self.assertEqual(day1, again)  # stable all day
        day2 = [c.candidate_id for c in store.today("gig", "o", {}, 3,
                                                   now=now + 86400, explore=0,
                                                   diversify=False)]
        self.assertTrue(set(day1).isdisjoint(day2))  # no repeats
        hist = store.batch_history("gig", "o")
        self.assertEqual(len(hist), 2)

    def test_feedback_trains_batches(self):
        store = DailyBatchStore(_tmpdb())
        store.add_candidate(_cand("g1", ["x"], 0.5))
        self.assertTrue(store.record_feedback("gig", "o", "g1", True))
        self.assertTrue(store.record_feedback("gig", "o", "g1", False))
        self.assertEqual(store.feedback_stats("gig", "o")["g1"], (1, 2))
        # a shown batch counts impressions
        store.today("gig", "o", {}, 1, explore=0, diversify=False)
        self.assertEqual(store.feedback_stats("gig", "o")["g1"], (1, 3))

    def test_remove_candidate(self):
        store = DailyBatchStore(_tmpdb())
        store.add_candidate(_cand("g1", ["x"], 0.5))
        self.assertTrue(store.remove_candidate("g1"))
        self.assertFalse(store.remove_candidate("g1"))
        self.assertEqual(store.candidates(), [])


# ── questionnaire ─────────────────────────────────────────────────

class TestQuestionnaire(unittest.TestCase):
    def test_importance_ladder(self):
        self.assertEqual(IMPORTANCE_WEIGHTS,
                         {1: 1, 2: 10, 3: 50, 4: 250, 5: 250})

    def test_acceptable_and_satisfaction(self):
        q = Questionnaire("o", _tmpdb())
        q.add_question("gig:remote", "remote?", "remote")
        q.answer("gig:remote", 3, "yes")
        q.set_acceptable("gig:remote", ["yes", "hybrid"])
        self.assertAlmostEqual(q.satisfaction({"remote": "yes"}), 1.0)
        self.assertAlmostEqual(q.satisfaction({"remote": "no"}), 0.0)
        self.assertAlmostEqual(q.satisfaction({"remote": "hybrid"}), 1.0)
        # breakdown names the weight and the verdict
        rows = q.match_breakdown({"remote": "no"})
        self.assertEqual(rows[0]["weight"], 50)
        self.assertFalse(rows[0]["matched"])

    def test_answer_unknown_question_is_false(self):
        q = Questionnaire("o", _tmpdb())
        self.assertFalse(q.answer("nope:missing", 3, "x"))
        self.assertFalse(q.set_acceptable("nope:missing", ["x"]))

    def test_match_percent_two_sided(self):
        qa = Questionnaire("a", _tmpdb())
        qb = Questionnaire("b", _tmpdb())
        for qq in (qa, qb):
            qq.add_question("q1", "do you like X?", "x")
        qa.answer("q1", 4, "yes")
        qa.set_acceptable("q1", ["yes"])
        qb.answer("q1", 2, "yes")
        qb.set_acceptable("q1", ["yes"])
        self.assertAlmostEqual(qa.match_percent(qb), 100.0)
        self.assertAlmostEqual(qb.match_percent(qa), 100.0)  # symmetric
        # one side rejects -> geometric mean punishes: sqrt(1*0) = 0
        qb.set_acceptable("q1", ["no"])
        self.assertAlmostEqual(qa.match_percent(qb), 0.0)
        # no common questions -> 0, never raises
        qc = Questionnaire("c", _tmpdb())
        self.assertEqual(qa.match_percent(qc), 0.0)

    def test_dealbreaker_between(self):
        q = Questionnaire("o", _tmpdb())
        q.set_dealbreaker("budget", "price", "between", [100, 500])
        ok = q.filter_dealbreakers([{"attributes": {"price": 250}},
                                    {"attributes": {"price": 999}}])
        self.assertEqual(len(ok), 1)
        self.assertFalse(q.set_dealbreaker("bad", "x", "nope", 1))

    def test_sections_and_boundaries(self):
        q = Questionnaire("o", _tmpdb())
        started = q.ensure_starters("gig")
        self.assertTrue(started)
        by_section = {}
        for item in q.answered() + q.unanswered():
            by_section.setdefault(item.section, []).append(item.question_id)
        self.assertIn("gig:budget_fit",
                      [i.question_id for i in q.boundary_questions()])
        # unanswered filter by section
        self.assertEqual(q.unanswered("desires"),
                         [i for i in q.unanswered()
                          if i.section == "desires"])

    def test_capacities(self):
        q = Questionnaire("o", _tmpdb())
        self.assertTrue(q.set_capacity("mentor_ada", 3))
        self.assertEqual(q.capacities(), {"mentor_ada": 3})

    def test_migration_from_legacy_schema(self):
        path = _tmpdb()
        db = sqlite3.connect(path)
        db.execute("""CREATE TABLE answers (
                        owner TEXT, question_id TEXT, text TEXT,
                        attribute TEXT, importance INTEGER, answer TEXT,
                        updated_at REAL,
                        PRIMARY KEY (owner, question_id))""")
        db.execute("""INSERT INTO answers VALUES
                      ('o', 'q1', 't?', 'x', 2, 'yes', 1.0)""")
        db.commit()
        db.close()
        q = Questionnaire("o", path)  # migration adds columns
        got = q.get("q1")
        self.assertIsNotNone(got)
        self.assertEqual(got.answer, "yes")
        self.assertTrue(q.set_acceptable("q1", ["yes"]))
        self.assertAlmostEqual(q.satisfaction({"x": "yes"}), 1.0)


# ── chat ──────────────────────────────────────────────────────────

class TestChat(unittest.TestCase):
    def test_daily_add_like_pass_why(self):
        ctx = _ctx()
        out = control_match("add g1 | Fix login bug | python,remote | "
                            "budget=500,remote=yes", ctx)
        self.assertIn("registered", out)
        out = control_match("daily", ctx)
        self.assertIn("Fix login bug", out)
        self.assertIn("like", out)  # footer hints
        out = control_match("like g1", ctx)
        self.assertIn("👍", out)
        out = control_match("pass g1", ctx)
        self.assertIn("👎", out)
        out = control_match("why g1", ctx)
        self.assertIn("score", out)
        out = control_match("remove g1", ctx)
        self.assertIn("removed", out)
        out = control_match("daily", ctx)
        self.assertIn("no candidates", out)

    def test_questionnaire_flow(self):
        ctx = _ctx()
        out = control_match("questionnaire", ctx)
        self.assertIn("answer more", out)
        out = control_match("answer gig:remote 5 yes", ctx)
        self.assertIn("★★★★★", out)
        self.assertIn("acceptable", out)  # nudge toward full pattern
        out = control_match("acceptable gig:remote yes,hybrid", ctx)
        self.assertIn("acceptable answers set", out)
        out = control_match("dealbreaker budget budget le 1000", ctx)
        self.assertIn("deal-breaker", out)
        out = control_match("stats", ctx)
        self.assertIn("deal-breakers: budget", out)

    def test_run_capacity_aware(self):
        ctx = _ctx()
        control_match("rank proposer amy m1,m2", ctx)
        control_match("rank proposer bob m1,m2", ctx)
        control_match("rank proposer cid m2,m1", ctx)
        control_match("rank reviewer m1 amy,bob,cid", ctx)
        control_match("rank reviewer m2 cid,amy,bob", ctx)
        control_match("capacity m1 2", ctx)
        out = control_match("run", ctx)
        self.assertIn("2/2", out)  # capacity-aware roster
        self.assertIn("verified: no blocking pairs", out)

    def test_run_plain_stable(self):
        ctx = _ctx()
        control_match("rank proposer a x,y", ctx)
        control_match("rank reviewer x a", ctx)
        control_match("rank reviewer y a", ctx)
        out = control_match("run", ctx)
        self.assertIn("a ↔ x", out)
        self.assertIn("no blocking pairs", out)

    def test_roommates_command(self):
        ctx = _ctx()
        control_match("rank proposer a b,c,d", ctx)
        control_match("rank proposer b a,d,c", ctx)
        control_match("rank proposer c d,a,b", ctx)
        control_match("rank proposer d c,b,a", ctx)
        out = control_match("roommates a,b,c,d", ctx)
        self.assertIn("↔", out)
        self.assertIn("verified", out)

    def test_assign_command(self):
        ctx = _ctx()
        control_match("rank proposer a x,y", ctx)
        control_match("rank proposer b y,x", ctx)
        control_match("rank reviewer x a,b", ctx)
        control_match("rank reviewer y b,a", ctx)
        out = control_match("assign", ctx)
        self.assertIn("max-total", out)
        self.assertIn("a → x", out)
        self.assertIn("b → y", out)

    def test_ttc_command(self):
        ctx = _ctx()
        out = control_match("ttc a=x b=y -- a:y,x b:x,y", ctx)
        self.assertIn("a → y", out)
        self.assertIn("b → x", out)

    def test_draft_command(self):
        ctx = _ctx()
        control_match("rank proposer amy g1,g2", ctx)
        control_match("rank proposer bob g2,g1", ctx)
        out = control_match("draft amy bob seed=1", ctx)
        self.assertIn("amy →", out)
        self.assertIn("bob →", out)

    def test_never_raises_on_garbage(self):
        ctx = _ctx()
        for tail in ["", "frobnicate", "daily", "add", "add |", "like",
                     "why", "answer", "answer q notanint", "dealbreaker",
                     "dealbreaker n a bogus v", "rank", "rank x y",
                     "capacity", "run", "roommates", "roommates a",
                     "assign", "ttc", "ttc a=x", "stats", None]:
            out = control_match(tail, ctx)
            self.assertIsInstance(out, str)
            self.assertTrue(out)


if __name__ == "__main__":
    unittest.main()
