"""Games-engine pass (DESIGN §13/§34 follow-ups, 2026-11-03):

* state-held matching — math duel answers, teach and upvote resolve
  through the member's held state; quoted ids that are not the held
  one score/teach/vote nothing, and no id is needed at all;
* cross-platform duel matchmaking for math duels — the global queue
  pairs the first compatible waiters across platforms;
* short shareable join codes — open duels (math + relay invites)
  carry a 6-character human code: single-use, expiring,
  rate-limited against guessing;
* §34 timed rounds — a round settles at the first graded answer or
  at expiry; the close carries the answer with the generator's OWN
  working; late answers never score;
* §34 turn clocks — engine games pass the turn on a stall, count
  consecutive stalls per player, and forfeit at the limit.
"""
from __future__ import annotations

import unittest

from nomorals.games.engine import GameEngine
from nomorals.games.join_codes import JOIN_CODE_ALPHABET
from nomorals.games.math_class import MathClass
from nomorals.games.math_commands import MathCommandHandler
from nomorals.games.math_economy import MathEconomy
from nomorals.games.players import Player
from nomorals.storage.db import Database

T0 = 1_700_000_000.0

ADA = "telegram:ada"
BOB = "discord:bob"
CAROL = "whatsapp:carol"
DAVE = "telegram:dave"


def _handler() -> MathCommandHandler:
    eco = MathEconomy()
    for p in (ADA, BOB, CAROL, DAVE):
        eco.verify_member(p, fingerprint=f"dev-{p}")
    return MathCommandHandler(MathClass(), eco)


def _duel_answers(h: MathCommandHandler, duel_id: str) -> list[str]:
    duel = h.math.store["duels"][duel_id]
    return [p.answer_text for p in duel.problems]


# ── state-held matching ───────────────────────────────────────────────────

class HeldDuelAnswerTests(unittest.TestCase):
    def test_answer_scores_with_no_id_at_all(self):
        h = _handler()
        duel = h.handle("/math duel discord:bob", ADA, now=T0)
        answers = _duel_answers(h, duel["duel_id"])
        res = h.handle(f"/math duel_answer {answers[0]}", ADA, now=T0 + 5)
        self.assertTrue(res["ok"])
        self.assertTrue(res["correct"])
        self.assertEqual(res["round_winner"], ADA)
        # the opponent's held duel is the same duel — also id-free
        res2 = h.handle(f"/math duel_answer {answers[1]}", BOB, now=T0 + 6)
        self.assertTrue(res2["correct"])
        self.assertEqual(res2["round"], 1)

    def test_foreign_or_fabricated_id_scores_nothing(self):
        h = _handler()
        mine = h.handle("/math duel discord:bob", ADA, now=T0)
        # a second, real duel belonging to other people
        foreign = h.handle("/math duel telegram:dave", CAROL, now=T0)
        answers = _duel_answers(h, mine["duel_id"])
        for bad_id in (foreign["duel_id"], "duel_deadbeef"):
            res = h.handle(
                f"/math duel_answer {bad_id} 0 {answers[0]}", ADA, now=T0 + 5)
            self.assertFalse(res["ok"])
            self.assertEqual(res["error"], "stale_duel")
        duel = h.math.store["duels"][mine["duel_id"]]
        self.assertEqual(duel.round_winners, [])
        self.assertEqual(h.economy.balance(ADA)["coins"], 0)

    def test_legacy_grammar_with_own_id_still_resolves(self):
        h = _handler()
        duel = h.handle("/math duel discord:bob", ADA, now=T0)
        answers = _duel_answers(h, duel["duel_id"])
        res = h.handle(
            f"/math duel_answer {duel['duel_id']} 0 {answers[0]}",
            ADA, now=T0 + 5)
        self.assertTrue(res["ok"])
        self.assertTrue(res["correct"])

    def test_no_open_duel_is_an_honest_error(self):
        h = _handler()
        res = h.handle("/math duel_answer 42", ADA, now=T0)
        self.assertFalse(res["ok"])
        self.assertEqual(res["error"], "no_open_duel")


class HeldSoloRoundTests(unittest.TestCase):
    def test_first_answer_settles_the_round_one_shot(self):
        h = _handler()
        h.handle("/math easy", ADA, now=T0)
        pid = h._active[ADA]
        problem = h.math.store["active"][pid]
        res = h.handle(f"/math answer {problem.answer_text}", ADA, now=T0 + 5)
        self.assertTrue(res["correct"])
        paid = h.economy.balance(ADA)["coins"]
        self.assertGreater(paid, 0)
        # a repeat answer gets the scoreless close, never a 2nd grade
        res2 = h.handle(f"/math answer {problem.answer_text}", ADA, now=T0 + 6)
        self.assertTrue(res2.get("closed"))
        self.assertFalse(res2.get("correct"))
        self.assertEqual(h.economy.balance(ADA)["coins"], paid)
        wins = h.math.store["scores"][ADA]["wins"]
        self.assertEqual(wins, 1)


class HeldTeachUpvoteTests(unittest.TestCase):
    def _daily_taught(self, h: MathCommandHandler) -> str:
        h.handle("/math daily", ADA, now=T0)
        pid = h._active[ADA]
        problem = h.math.store["active"][pid]
        h.handle(f"/math answer {problem.answer_text}", ADA, now=T0 + 5)
        taught = h.handle("/math teach Count the parts first.", ADA,
                          now=T0 + 6)
        assert taught["ok"], taught
        return pid

    def test_teach_bare_and_upvote_bare(self):
        h = _handler()
        self._daily_taught(h)
        h.handle("/math daily", BOB, now=T0 + 7)
        pid = h._active[BOB]
        problem = h.math.store["active"][pid]
        answered = h.handle(f"/math answer {problem.answer_text}", BOB,
                            now=T0 + 8)
        self.assertIn("held_explanation", answered)
        self.assertEqual(answered["held_explanation"]["author"], ADA)
        voted = h.handle("/math upvote", BOB, now=T0 + 9)
        self.assertTrue(voted["ok"])
        self.assertTrue(voted["paid"])
        self.assertGreater(h.economy.balance(ADA)["coins"], 0)
        # one-shot: the hold is consumed
        again = h.handle("/math upvote", BOB, now=T0 + 10)
        self.assertFalse(again["ok"])
        self.assertEqual(again["error"], "no_held_explanation")

    def test_upvote_with_foreign_id_pays_nothing(self):
        h = _handler()
        self._daily_taught(h)
        h.handle("/math daily", BOB, now=T0 + 7)
        pid = h._active[BOB]
        problem = h.math.store["active"][pid]
        h.handle(f"/math answer {problem.answer_text}", BOB, now=T0 + 8)
        before = h.economy.balance(ADA)["coins"]
        res = h.handle("/math upvote expl_forged999", BOB, now=T0 + 9)
        self.assertFalse(res["ok"])
        self.assertEqual(res["error"], "stale_explanation")
        self.assertEqual(h.economy.balance(ADA)["coins"], before)
        # the held upvote still works afterwards (the forgery didn't
        # consume the hold)
        voted = h.handle("/math upvote", BOB, now=T0 + 10)
        self.assertTrue(voted.get("paid"))

    def test_teach_with_stale_problem_id_teaches_nothing(self):
        h = _handler()
        pid = self._daily_taught(h)
        # a fresh problem moves the held target on
        h.handle("/math easy", ADA, now=T0 + 20)
        res = h.handle(f"/math teach {pid} Stale words.", ADA, now=T0 + 21)
        self.assertFalse(res["ok"])
        self.assertEqual(res["error"], "stale_problem")
        texts = [e["text"] for e in h.economy.explanations_for(pid)]
        self.assertEqual(texts, ["Count the parts first."])


# ── matchmaking (§13) ─────────────────────────────────────────────────────

class MathMatchmakingTests(unittest.TestCase):
    def test_queue_pairs_strangers_across_platforms(self):
        h = _handler()
        queued = h.handle("/math duel", ADA, now=T0)
        self.assertEqual(queued["kind"], "duel_queue")
        matched = h.handle("/math duel", BOB, now=T0 + 30)
        self.assertEqual(matched["kind"], "duel")
        self.assertTrue(matched["matched"])
        self.assertEqual(matched["players"], [ADA, BOB])
        # identical problems for both seats (fair contest)
        duel = h.math.store["duels"][matched["duel_id"]]
        questions = [p.question for p in duel.problems]
        self.assertEqual(len(questions), 5)
        # the waiter learns on their next interaction, problems in hand
        nxt = h.handle("/math streak", ADA, now=T0 + 31)
        notes = nxt.get("notes") or []
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["kind"], "duel_filled")
        self.assertEqual(notes[0]["opponent"], BOB)
        self.assertEqual([p["question"] for p in notes[0]["problems"]],
                         questions)
        # and both play from their own side, id-free
        first = duel.problems[0]
        res = h.handle(f"/math duel_answer {first.answer_text}", ADA,
                       now=T0 + 40)
        self.assertTrue(res["correct"])

    def test_first_compatible_waiter_pairs_first(self):
        h = _handler()
        h.handle("/math duel queue hard", ADA, now=T0)
        h.handle("/math duel queue easy", BOB, now=T0 + 1)
        matched = h.handle("/math duel queue hard", CAROL, now=T0 + 2)
        self.assertEqual(matched["players"], [ADA, CAROL])
        # BOB is still waiting (easy ≠ hard, nobody widened yet)
        self.assertEqual(len(h._duel_queue), 1)
        self.assertEqual(h._duel_queue[0]["player"], BOB)

    def test_queue_spot_expires_with_a_note(self):
        h = _handler()
        h.handle("/math duel", ADA, now=T0)
        closes = h.sweep_rounds(now=T0 + 601)
        self.assertEqual(closes, [])
        self.assertEqual(h._duel_queue, [])
        nxt = h.handle("/math streak", ADA, now=T0 + 602)
        kinds = [n["kind"] for n in nxt.get("notes") or []]
        self.assertIn("queue_expired", kinds)


# ── join codes ────────────────────────────────────────────────────────────

class JoinCodeTests(unittest.TestCase):
    def test_open_duel_code_shape_join_and_play(self):
        h = _handler()
        opened = h.handle("/math duel open", ADA, now=T0)
        self.assertEqual(opened["kind"], "duel_open")
        code = opened["join_code"]
        self.assertEqual(len(code), 6)
        self.assertTrue(all(c in JOIN_CODE_ALPHABET for c in code))
        # the internal id is a different thing entirely
        self.assertNotEqual(code, opened["duel_id"])
        joined = h.handle(f"/math join {code.lower()}", CAROL, now=T0 + 60)
        self.assertTrue(joined["ok"])
        self.assertEqual(joined["players"], [ADA, CAROL])
        # the creator hears about it on their next interaction
        nxt = h.handle("/math streak", ADA, now=T0 + 61)
        kinds = [n["kind"] for n in nxt.get("notes") or []]
        self.assertIn("duel_filled", kinds)
        # and the duel plays from both sides
        duel = h.math.store["duels"][joined["duel_id"]]
        res = h.handle(f"/math duel_answer {duel.problems[0].answer_text}",
                       CAROL, now=T0 + 70)
        self.assertTrue(res["correct"])

    def test_code_is_single_use(self):
        h = _handler()
        opened = h.handle("/math duel open", ADA, now=T0)
        code = opened["join_code"]
        self.assertTrue(h.handle(f"/math join {code}", BOB, now=T0 + 1)["ok"])
        again = h.handle(f"/math join {code}", CAROL, now=T0 + 2)
        self.assertFalse(again["ok"])
        self.assertEqual(again["error"], "bad_code")

    def test_code_expires_with_the_invite(self):
        h = _handler()
        opened = h.handle("/math duel open", ADA, now=T0)
        late = h.handle(f"/math join {opened['join_code']}", BOB,
                        now=T0 + 3601)
        self.assertFalse(late["ok"])
        self.assertEqual(late["error"], "code_expired")

    def test_failed_joins_are_rate_limited(self):
        h = _handler()
        opened = h.handle("/math duel open", ADA, now=T0)
        for i in range(5):
            res = h.handle("/math join ZZZZZZ", CAROL, now=T0 + i)
            self.assertEqual(res["error"], "bad_code")
        blocked = h.handle(f"/math join {opened['join_code']}", CAROL,
                           now=T0 + 10)
        self.assertEqual(blocked["error"], "too_many_attempts")
        # someone else's budget is untouched
        ok = h.handle(f"/math join {opened['join_code']}", BOB, now=T0 + 11)
        self.assertTrue(ok["ok"])


# ── §34 timed rounds ──────────────────────────────────────────────────────

class TimedRoundTests(unittest.TestCase):
    def test_expiry_closes_scoreless_with_the_generators_working(self):
        mc = MathClass()
        problem = mc.generate(ADA, "easy", "quickfire", now=T0)
        res = mc.submit(ADA, problem.problem_id, problem.answer_text,
                        now=T0 + 61)
        self.assertTrue(res["closed"])
        self.assertFalse(res["scored"])
        self.assertFalse(res["correct"])
        # the close carries the generator's OWN derivation, verbatim
        self.assertEqual(res["answer"], problem.answer_text)
        self.assertEqual(res["solution"], problem.solution)
        self.assertEqual(res["solution_steps"], problem.solution_steps)
        scores = mc.store["scores"].get(ADA, {})
        self.assertEqual(scores.get("wins", 0), 0)
        self.assertEqual(scores.get("losses", 0), 0)

    def test_in_time_answer_still_scores(self):
        mc = MathClass()
        problem = mc.generate(ADA, "easy", "quickfire", now=T0)
        res = mc.submit(ADA, problem.problem_id, problem.answer_text,
                        now=T0 + 59)
        self.assertTrue(res["correct"])
        self.assertTrue(res["scored"])

    def test_sweep_posts_the_close_and_late_answers_never_score(self):
        h = _handler()
        h.handle("/math easy", ADA, now=T0)
        pid = h._active[ADA]
        problem = h.math.store["active"][pid]
        closes = h.sweep_rounds(now=T0 + 61)
        self.assertEqual(len(closes), 1)
        self.assertEqual(closes[0]["kind"], "round_close")
        self.assertEqual(closes[0]["problem"]["answer"],
                         problem.answer_text)
        self.assertEqual(closes[0]["problem"]["solution_steps"],
                         problem.solution_steps)
        nxt = h.handle("/math streak", ADA, now=T0 + 62)
        kinds = [n["kind"] for n in nxt.get("notes") or []]
        self.assertIn("round_close", kinds)
        late = h.handle(f"/math answer {problem.answer_text}", ADA,
                        now=T0 + 63)
        self.assertTrue(late.get("closed"))
        self.assertEqual(h.economy.balance(ADA)["coins"], 0)

    def test_timed_set_closes_as_one_record(self):
        h = _handler()
        h.handle("/math set", ADA, now=T0)
        closes = h.sweep_rounds(now=T0 + 301)
        self.assertEqual(len(closes), 1)
        self.assertEqual(closes[0]["kind"], "set_close")
        self.assertEqual(len(closes[0]["problems"]), 10)
        for reveal in closes[0]["problems"]:
            self.assertIn("answer", reveal)
            self.assertIn("solution_steps", reveal)

    def test_duel_clock_settles_on_the_score(self):
        h = _handler()
        duel = h.handle("/math duel discord:bob", ADA, now=T0)
        answers = _duel_answers(h, duel["duel_id"])
        won = h.handle(f"/math duel_answer {answers[0]}", ADA, now=T0 + 10)
        self.assertTrue(won["correct"])
        late = h.handle(f"/math duel_answer {answers[1]}", BOB, now=T0 + 301)
        self.assertFalse(late["ok"])
        self.assertEqual(late["error"], "duel_expired")
        self.assertEqual(late["duel_winner"], ADA)
        self.assertEqual(late["score"], {ADA: 1, BOB: 0})
        # the close carries the live round's answer + working
        self.assertEqual(late["reveal"]["answer"],
                         h.math.store["duels"][duel["duel_id"]]
                         .problems[1].answer_text)
        # holds are cleared — nobody is stuck in a settled duel
        nxt = h.handle("/math duel_answer 1", BOB, now=T0 + 302)
        self.assertEqual(nxt["error"], "no_open_duel")


# ── §34 turn clocks (engine) ──────────────────────────────────────────────

class Ctx:
    def __init__(self, db):
        self.db = db


def _engine():
    db = Database(":memory:")
    db.migrate()
    sent: list[tuple[str, str]] = []
    engine = GameEngine(Ctx(db),
                        send=lambda chat, text: sent.append((chat, text)))
    return engine, db, sent


TG_ADA = Player.from_sender("telegram", "111", "Ada")
TG_BOB = Player.from_sender("telegram", "222", "Bob")
TG_CAROL = Player.from_sender("telegram", "333", "Carol")


def _stall(engine: GameEngine, room) -> None:
    """Run out the current player's clock and tick the scheduler.

    ``turn_started`` goes to the epoch — ancient under any timeout,
    and no wall-clock read in the test itself (clock ratchet)."""
    room.turn_started = 0.0
    engine._handle_timeout(room)


class TurnClockTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = _engine()

    def tearDown(self):
        self.engine.shutdown()

    def _group_connect4(self):
        room, _ = self.engine.start("telegram:g1", "connect4", TG_ADA,
                                    kind="group")
        self.engine.join("telegram:g1", TG_BOB)
        return room

    def test_stall_passes_turn_and_journals(self):
        room = self._group_connect4()
        self.assertEqual(room.current.key, TG_ADA.key)
        _stall(self.engine, room)
        self.assertEqual(room.current.key, TG_BOB.key)
        self.assertEqual(room.state["_stalls"][TG_ADA.key], 1)
        events = [e["event"] for e in room.state["_clock_events"]]
        self.assertIn("stall", events)
        self.assertIn("pass", events)
        texts = " ".join(t for _, t in self.sent)
        self.assertIn("turn passes", texts)
        self.assertIn("stall 1/3", texts)

    def test_real_move_resets_the_stall_count(self):
        room = self._group_connect4()
        _stall(self.engine, room)                    # Ada stall 1
        self.engine.move("telegram:g1", "4", TG_BOB)  # Bob plays
        self.engine.move("telegram:g1", "4", TG_ADA)  # Ada plays → reset
        self.assertNotIn(TG_ADA.key, room.state.get("_stalls", {}))
        self.engine.move("telegram:g1", "5", TG_BOB)
        _stall(self.engine, room)                    # Ada stall 1 again
        self.engine.move("telegram:g1", "5", TG_BOB)
        _stall(self.engine, room)                    # Ada stall 2
        self.assertEqual(room.state["_stalls"][TG_ADA.key], 2)
        self.assertIsNotNone(self.engine.live("telegram:g1"))

    def test_third_consecutive_stall_forfeits_to_the_opponent(self):
        room = self._group_connect4()
        for col in ("4", "5", "6"):
            _stall(self.engine, room)                 # Ada stalls
            if self.engine.live("telegram:g1") is None:
                break
            self.engine.move("telegram:g1", col, TG_BOB)
        self.assertIsNone(self.engine.live("telegram:g1"))
        texts = " ".join(t for _, t in self.sent)
        self.assertIn("forfeit", texts)
        bob = self.engine.store.get_for(TG_BOB)
        ada = self.engine.store.get_for(TG_ADA)
        self.assertGreaterEqual(bob.wins, 1)
        self.assertGreaterEqual(ada.losses, 1)
        # ranked too: the forfeit is a real result on the ELO board
        from nomorals.games.matchmaking import get_rating
        self.assertGreater(get_rating(self.db, TG_BOB.key, "connect4"),
                           1000)
        events = [e["event"] for e in room.state["_clock_events"]]
        self.assertIn("forfeit", events)

    def test_stall_forfeit_against_the_house(self):
        room, _ = self.engine.start("telegram:dm1", "connect4", TG_ADA,
                                    kind="dm")
        for _ in range(3):
            if self.engine.live("telegram:dm1") is None:
                break
            _stall(self.engine, room)
        self.assertIsNone(self.engine.live("telegram:dm1"))
        ada = self.engine.store.get_for(TG_ADA)
        self.assertGreaterEqual(ada.losses, 1)
        texts = " ".join(t for _, t in self.sent)
        self.assertIn("forfeit", texts)

    def test_games_with_own_timeout_keep_their_semantics(self):
        room, _ = self.engine.start("telegram:g2", "pvp", TG_ADA,
                                    kind="group")
        self.engine.join("telegram:g2", TG_BOB)
        _stall(self.engine, room)
        # pvp counts its own misses; the engine stall law stays out
        self.assertEqual(room.state["misses"][TG_ADA.key], 1)
        self.assertNotIn("_stalls", room.state)


# ── relay join codes ──────────────────────────────────────────────────────

class RelayJoinCodeTests(unittest.TestCase):
    def setUp(self):
        self.engine, self.db, self.sent = _engine()
        self.relay = self.engine.relay

    def tearDown(self):
        self.engine.shutdown()

    def test_invite_code_is_short_human_and_works_lowercase(self):
        inv = self.relay.create_invite("telegram:111", TG_ADA, "connect4")
        self.assertEqual(len(inv.code), 6)
        self.assertTrue(all(c in JOIN_CODE_ALPHABET for c in inv.code))
        room = self.relay.accept_invite(inv.code.lower(), "discord:222",
                                        Player.from_sender(
                                            "discord", "222", "Bob"))
        self.assertEqual(room.game_name, "connect4")

    def test_accept_guessing_is_rate_limited(self):
        for _ in range(5):
            with self.assertRaises(ValueError):
                self.relay.accept_invite("ZZZZZZ", "telegram:999", TG_ADA)
        with self.assertRaises(ValueError) as ctx:
            self.relay.accept_invite("ZZZZZZ", "telegram:999", TG_ADA)
        self.assertIn("too many wrong codes", str(ctx.exception))
        # a different chat keeps its own budget
        inv = self.relay.create_invite("telegram:111", TG_ADA, "connect4")
        room = self.relay.accept_invite(inv.code, "discord:222",
                                        Player.from_sender(
                                            "discord", "222", "Bob"))
        self.assertIsNotNone(room)


if __name__ == "__main__":
    unittest.main()
