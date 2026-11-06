"""AA-10 / AA-16 / AA-52 — dormant games organs, wired.

Production-path tests: every test drives the surface a member or
the system actually uses — the engine's trivia forge and a live
trivia game (AA-10), a real engine game finishing in a group room,
the durable season tick and the ``/game season`` surface (AA-16),
and the live tool registry (AA-52) — never the dormant organ alone.
"""

from __future__ import annotations

import calendar
import random
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.games.engine import GameEngine
from nomorals.games.players import Player
from nomorals.games.trivia_forge import TriviaForge
from nomorals.storage.db import Database


class Ctx:
    """Engine/tool context: a db plus settings pointing every
    community store at the test's temp dir (never the real home)."""

    def __init__(self, db, community_dir: Path, workspace_dir: Path):
        self.db = db
        self.settings = SimpleNamespace(
            community_dir=str(community_dir),
            workspace_dir=str(workspace_dir))


def _ts(year: int, month: int, day: int) -> float:
    return float(calendar.timegm((year, month, day, 0, 0, 0, 0, 0, 0)))


Q1_2026 = _ts(2026, 2, 1)   # inside 2026-Q1
Q2_2026 = _ts(2026, 5, 1)   # inside 2026-Q2


# ── AA-52: gameplayer joins the live registry ───────────────────────────────

class GameplayerRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nomorals.tools.registry import ToolRegistry

        cls.registry = ToolRegistry()
        cls.registry.register_builtins()

    def test_autopilot_play_registered(self):
        self.assertIn("autopilot_play", self.registry.names())
        spec = self.registry.get("autopilot_play")
        self.assertIsNotNone(spec)
        failed = [f for f in self.registry.failed_modules()
                  if f["module"] == "gameplayer"]
        self.assertEqual(failed, [])

    def test_autopilot_play_callable_through_registry(self):
        # A dry-run session against a private address must reach the
        # game's own SSRF guard — proof the registry dispatched into
        # the real tool (a missing tool answers ToolNotFound instead),
        # with zero network traffic.
        from nomorals.core.policy import CapabilitySet

        out = self.registry.call(
            "autopilot_play", game="generic", url="http://127.0.0.1:9/",
            max_moves=1, actor="test", capabilities=CapabilitySet.all())
        self.assertIn("SSRF", str(getattr(out, "error", "")))


# ── AA-10: packs + riddles served by the engine's trivia path ───────────────

class PackServingTests(unittest.TestCase):
    def test_engine_forge_deals_pack_questions(self):
        # The engine's own forge (no model, no bank stock — the
        # degraded case) must serve the hand-written packs.
        from nomorals.games.extra_content import TRIVIA_PACKS

        db = Database(":memory:")
        db.migrate()
        engine = GameEngine(Ctx(db, Path("."), Path(".")),
                            send=lambda chat, text: None)
        try:
            recs = engine.forge.deal_detailed(10, "normal")
        finally:
            engine.shutdown()
        packs = [r for r in recs if r["kind"] == "pack"]
        self.assertTrue(packs, f"no pack questions in {[r['kind'] for r in recs]}")
        for r in packs:
            pack = TRIVIA_PACKS[r["source"]]
            row = next(q for q in pack["questions"] if q["q"] == r["q"])
            self.assertEqual(r["answer"], row["options"][row["answer"]])
            self.assertEqual(
                sorted(r["distractors"]),
                sorted(o for i, o in enumerate(row["options"])
                       if i != row["answer"]))

    def test_riddles_served_by_deal(self):
        from nomorals.games.extra_content import RIDDLES

        forge = TriviaForge(seed=11)
        recs = forge.deal_detailed(51, "normal")
        riddles = [r for r in recs if r["kind"] == "riddle"]
        self.assertTrue(riddles, "no riddle questions dealt")
        known = {r["q"]: r["a"] for r in RIDDLES}
        for r in riddles:
            self.assertIn(r["answer"], known[r["q"]])

    def test_live_trivia_game_serves_pack_content(self):
        # TriviaRoyale is a live catalog game; its new_state draws
        # from the forge — pack/riddle questions must be playable
        # at a real table.
        from nomorals.games.extra_content import RIDDLES, TRIVIA_PACKS
        from nomorals.games.games.easy import TriviaRoyaleGame

        known = {q["q"] for p in TRIVIA_PACKS.values()
                 for q in p["questions"]} | {r["q"] for r in RIDDLES}
        game = TriviaRoyaleGame()
        state = game.new_state(random.Random(5), trivia_forge=TriviaForge(seed=5))
        served = [q for q, _a in state["questions"]]
        self.assertEqual(len(served), 8)
        self.assertTrue(any(q in known for q in served),
                        f"no pack/riddle content at the table: {served}")


# ── AA-16: seasons — engine feed, tick, surface ─────────────────────────────

class _FakeScheduler:
    def __init__(self):
        self.jobs: list[dict] = []
        self._seq = 0

    def list_jobs(self):
        return list(self.jobs)

    def add(self, name, spec, kind, payload, routine=False):
        self._seq += 1
        job = {"id": f"j{self._seq}", "name": name, "spec": spec,
               "kind": kind, "payload": payload, "routine": routine}
        self.jobs.append(job)
        return job

    def remove(self, job_id):
        self.jobs = [j for j in self.jobs if j["id"] != job_id]


class SeasonJobTests(unittest.TestCase):
    def test_ensure_job_idempotent_and_routine(self):
        from nomorals.games.tournaments_seasons import (
            SEASON_JOB,
            SEASON_SPEC,
            ensure_season_tick_job,
        )

        sched = _FakeScheduler()
        job = ensure_season_tick_job(sched)
        self.assertIsNotNone(job)
        self.assertEqual(job["name"], SEASON_JOB)
        self.assertEqual(job["spec"], SEASON_SPEC)
        self.assertEqual(job["payload"], {"tool": "__season_tick__", "args": {}})
        self.assertTrue(job["routine"])
        again = ensure_season_tick_job(sched)
        self.assertEqual(again["id"], job["id"])
        self.assertEqual(len(sched.jobs), 1)

    def test_ensure_job_replaces_stale_spec(self):
        from nomorals.games.tournaments_seasons import ensure_season_tick_job

        sched = _FakeScheduler()
        sched.add("season tick", "every 5m", "tool",
                  {"tool": "__season_tick__", "args": {}}, routine=True)
        job = ensure_season_tick_job(sched)
        self.assertEqual(job["spec"], "every 1h")
        self.assertEqual(len(sched.jobs), 1)

    def test_ensure_job_no_scheduler(self):
        from nomorals.games.tournaments_seasons import ensure_season_tick_job

        self.assertIsNone(ensure_season_tick_job(None))


class SeasonEngineFeedTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._td = tempfile.TemporaryDirectory(dir=_verify_tmp())
        self.tmp = Path(self._td.name)
        self.community = self.tmp / "community"
        self.db = Database(":memory:")
        self.db.migrate()
        self.sent: list[tuple[str, str]] = []
        ctx = Ctx(self.db, self.community, self.tmp)
        self.engine = GameEngine(
            ctx, send=lambda chat, text: self.sent.append((chat, text)))

    def tearDown(self):
        self.engine.shutdown()
        self._td.cleanup()

    def _play_rps(self, chat_key: str, kind: str, player: Player) -> None:
        room, _ = self.engine.start(chat_key, "rps", player, kind=kind)
        for _ in range(100):
            if room.status == "finished":
                break
            self.engine.move(chat_key, "rock", player)
        self.assertEqual(room.status, "finished")

    def test_group_game_folds_into_season_board(self):
        from nomorals.games.tournaments_seasons import SeasonStore, season_id_for

        ada = Player.from_sender("telegram", "111", "Ada")
        self._play_rps("telegram:g1", "group", ada)
        store = SeasonStore(self.community / "seasons")
        data = store.load("telegram:g1")
        self.assertIsNotNone(data["season"])
        self.assertEqual(data["season"]["id"], season_id_for())
        board = data["board"]
        self.assertIn("Ada", board["points"])
        self.assertGreaterEqual(board["games"]["Ada"], 1)
        # member id captured for prize routing — and it is the
        # sender id, never rendered on the board itself.
        self.assertEqual(data["members"].get("Ada"), "111")

    def test_dm_game_does_not_feed_season(self):
        from nomorals.games.tournaments_seasons import SeasonStore

        ada = Player.from_sender("telegram", "111", "Ada")
        self._play_rps("telegram:dm1", "dm", ada)
        store = SeasonStore(self.community / "seasons")
        data = store.load("telegram:dm1")
        self.assertIsNone(data["season"])


class SeasonTickTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._td = tempfile.TemporaryDirectory(dir=_verify_tmp())
        self.tmp = Path(self._td.name)
        self.root = self.tmp / "seasons"

    def tearDown(self):
        self._td.cleanup()

    def _seed_q1(self):
        from nomorals.games.tournaments_seasons import record_season_results

        gid = "telegram:g9"
        record_season_results(
            gid, "trivia",
            [("Ada", "111", 300, True), ("Bob", "222", 200, False),
             ("Carol", "333", 100, False)],
            root=self.root, now=Q1_2026)
        record_season_results(
            gid, "rps", [("Ada", "111", 50, True)],
            root=self.root, now=Q1_2026)
        return gid

    def test_tick_closes_season_and_pays_prizes(self):
        from nomorals.games.tournaments_seasons import SeasonStore, run_season_tick_once

        gid = self._seed_q1()
        paid: list[tuple[str, int, str]] = []

        def ledger_for(group_id, members):
            def ledger(player, amount, reason):
                paid.append((player, amount, members.get(player, "")))
                return True
            return ledger

        summary = run_season_tick_once(root=self.root, now=Q2_2026,
                                       ledger_for=ledger_for)
        self.assertTrue(summary["ok"])
        self.assertEqual([c["season"] for c in summary["closed"]], ["2026-Q1"])
        # default prize table: 500 / 250 / 100 by rank
        self.assertEqual(sorted(paid), [("Ada", 500, "111"),
                                        ("Bob", 250, "222"),
                                        ("Carol", 100, "333")])
        self.assertEqual(len(summary["paid"]), 3)
        self.assertEqual(summary["unpaid"], [])
        store = SeasonStore(self.root)
        data = store.load(gid)
        self.assertEqual(data["season"]["id"], "2026-Q2")
        self.assertTrue(data["archive"][0]["settled"])
        # a second tick must not pay twice
        summary2 = run_season_tick_once(root=self.root, now=Q2_2026,
                                        ledger_for=ledger_for)
        self.assertEqual(summary2["paid"], [])
        self.assertEqual(len(paid), 3)

    def test_tick_unpaid_prize_stays_open_for_retry(self):
        from nomorals.games.tournaments_seasons import SeasonStore, run_season_tick_once

        gid = self._seed_q1()
        summary = run_season_tick_once(
            root=self.root, now=Q2_2026,
            ledger_for=lambda g, m: (lambda p, a, r: False))
        self.assertEqual(len(summary["unpaid"]), 3)
        store = SeasonStore(self.root)
        self.assertFalse(store.load(gid)["archive"][0]["settled"])

    def test_tick_pays_through_real_group_economy(self):
        from nomorals.community.group_economy import GroupEconomy
        from nomorals.games.tournaments_seasons import run_season_tick_once

        gid = self._seed_q1()
        ctx = Ctx(None, self.tmp / "community", self.tmp)
        summary = run_season_tick_once(context=ctx, root=self.root,
                                       now=Q2_2026)
        self.assertEqual(len(summary["paid"]), 3)
        econ = GroupEconomy(self.tmp / "data" / "social" / "group_economy")
        self.assertEqual(econ.balance(gid, "111"), 500)
        self.assertEqual(econ.balance(gid, "222"), 250)
        self.assertEqual(econ.balance(gid, "333"), 100)

    def test_current_season_surface_data(self):
        from nomorals.games.tournaments_seasons import current_season

        gid = self._seed_q1()
        season, board = current_season(gid, root=self.root, now=Q1_2026)
        self.assertEqual(season["id"], "2026-Q1")
        rows = board.standings()
        self.assertEqual([r["player"] for r in rows], ["Ada", "Bob", "Carol"])
        self.assertEqual(rows[0]["points"], 350)


class SeasonSurfaceTests(unittest.TestCase):
    def test_game_season_verb_renders_board(self):
        import tempfile

        from nomorals.agents.partner.runtime_games import RuntimeGamesMixin
        from nomorals.games.tournaments_seasons import record_season_results

        td = tempfile.TemporaryDirectory(dir=_verify_tmp())
        tmp = Path(td.name)
        community = tmp / "community"
        db = Database(":memory:")
        db.migrate()
        record_season_results(
            "telegram:g7", "trivia", [("Ada", "111", 120, True)],
            root=community / "seasons", now=time.time())
        ctx = Ctx(db, community, tmp)
        engine = GameEngine(ctx, send=lambda chat, text: None)
        mixin = RuntimeGamesMixin.__new__(RuntimeGamesMixin)
        mixin.context = ctx
        mixin._game_engine_obj = engine
        try:
            out = mixin._control_game("season", "telegram:g7",
                                      player=None, kind="dm")
        finally:
            engine.shutdown()
            td.cleanup()
        self.assertIn("Season", out)
        self.assertIn("Ada", out)
        self.assertIn("120", out)


def _verify_tmp() -> str:
    import os

    return os.environ.get("NM_TEST_TMPDIR", "/tmp")


if __name__ == "__main__":
    unittest.main()
