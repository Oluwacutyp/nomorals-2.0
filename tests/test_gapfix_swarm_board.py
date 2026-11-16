"""Swarm board — failing-first tests (Antigravity /swarm, Devon v2).

The board is the missing piece from the surface mining: Devon's
/swarm runs parallel agents + fusion, but the team never talks to
each other mid-run. This file pins the board before it exists:

* agents post to ONE shared board and see each other's posts
* @name mentions route to the named agent (@all / @workers groups)
* briefs are public — every member sees what every worker owns,
  must not touch, and what "done" looks like
* a verifier is attached per 2–4 workers and checks their output
* a stalled team (no board activity inside the window) gets woken
  by the monitor, once per stall episode
* a swarm's board + briefs are durable (survive a restart via the
  store, like workflows)

Imports are lazy (inside the test) so that, before the build,
each test FAILS with ModuleNotFoundError instead of erroring at
collection.
"""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace


def _board_module():
    from nomorals.agents import swarm_board

    return swarm_board


class _FakeScheduler:
    def __init__(self):
        self.jobs = []

    def list_jobs(self):
        return list(self.jobs)

    def add(self, name, spec, kind, payload, routine=False):
        job = {"id": f"job-{len(self.jobs)}", "name": name,
               "spec": spec, "kind": kind, "payload": payload,
               "routine": routine}
        self.jobs.append(job)
        return job

    def remove(self, job_id):
        self.jobs = [j for j in self.jobs if j["id"] != job_id]


class _FakeNotifier:
    def __init__(self):
        self.sent: list[tuple[str, str, str]] = []

    def publish(self, kind, title="", body="", **kw):
        self.sent.append((kind, title, body))


def _board(run_id="swarm-test-1", goal="test the board"):
    mod = _board_module()
    return mod.SwarmBoard(run_id, goal=goal)


def _board_with_workers(n=3):
    board = _board()
    for i in range(n):
        board.add_member(f"worker-{i + 1}", role="worker")
    return board


class MentionTests(unittest.TestCase):
    def test_posts_visible_to_every_member(self):
        board = _board_with_workers(2)
        board.post("worker-1", "found the bug in the parser")
        self.assertEqual(len(board.feed), 1)
        # Another member's unread feed carries the post.
        self.assertEqual(len(board.unread("worker-2")), 1)
        self.assertEqual(board.unread("worker-2")[0]["author"],
                         "worker-1")

    def test_mention_routes_to_named_member_only(self):
        board = _board_with_workers(3)
        board.post("worker-1", "@worker-2 check the parser fix")
        self.assertEqual(len(board.mentions_for("worker-2")), 1)
        self.assertEqual(board.mentions_for("worker-3"), [])
        self.assertEqual(board.mentions_for("worker-1"), [])

    def test_all_mention_reaches_everyone(self):
        board = _board_with_workers(2)
        board.add_member("verifier-1", role="verifier")
        board.post("worker-1", "@all the parser is fixed")
        self.assertEqual(len(board.mentions_for("worker-2")), 1)
        self.assertEqual(len(board.mentions_for("verifier-1")), 1)

    def test_workers_group_excludes_verifiers(self):
        board = _board_with_workers(2)
        board.add_member("verifier-1", role="verifier")
        board.post("worker-1", "@workers re-run your checks")
        self.assertEqual(len(board.mentions_for("worker-2")), 1)
        self.assertEqual(board.mentions_for("verifier-1"), [])

    def test_unknown_mention_is_plain_text_not_a_crash(self):
        board = _board_with_workers(1)
        post = board.post("worker-1", "@nobody hello?")
        self.assertIn("nobody", post["mentions"])
        self.assertEqual(board.mentions_for("nobody"), [])

    def test_parse_mentions(self):
        mod = _board_module()
        self.assertEqual(mod.parse_mentions("@parser see @all and @workers"),
                         {"parser", "all", "workers"})
        self.assertEqual(mod.parse_mentions("no mentions here"), set())
        # An email-ish token is not a mention.
        self.assertEqual(mod.parse_mentions("mail me@home today"), set())


class BriefTests(unittest.TestCase):
    def test_briefs_are_public_to_every_member(self):
        board = _board_with_workers(2)
        board.brief("worker-1", owns="audit the parser",
                    must_not_touch="the api tests",
                    done="a digest posted to the board")
        briefs = board.briefs()
        self.assertEqual(len(briefs), 1)
        self.assertIn("audit the parser", briefs[0]["text"])
        # The other worker reads it too.
        self.assertEqual(len(board.unread("worker-2")), 1)
        self.assertEqual(board.unread("worker-2")[0]["kind"], "brief")

    def test_brief_names_owns_and_done(self):
        board = _board_with_workers(1)
        post = board.brief("worker-1", owns="audit the parser",
                           must_not_touch="the api",
                           done="digest posted")
        self.assertIn("audit the parser", post["text"])
        self.assertIn("the api", post["text"])
        self.assertIn("digest posted", post["text"])


class VerifierTests(unittest.TestCase):
    def test_one_verifier_for_one_to_four_workers(self):
        for n in (1, 2, 4):
            board = _board_with_workers(n)
            verifiers = board.assign_verifiers()
            self.assertEqual(len(verifiers), 1, f"n={n}")
            for i in range(n):
                self.assertEqual(
                    board.verifier_for(f"worker-{i + 1}"), verifiers[0])

    def test_two_verifiers_for_five_workers(self):
        board = _board_with_workers(5)
        verifiers = board.assign_verifiers()
        self.assertEqual(len(verifiers), 2)
        covered = [board.verifier_for(f"worker-{i + 1}")
                   for i in range(5)]
        self.assertEqual(sorted(set(covered)), sorted(verifiers))
        # No verifier covers more than 4 workers.
        for verifier in verifiers:
            members = [f"worker-{i + 1}" for i in range(5)
                       if board.verifier_for(f"worker-{i + 1}") == verifier]
            self.assertLessEqual(len(members), 4)

    def test_parse_verifier_verdict(self):
        mod = _board_module()
        digest = ("OK @worker-1: parser clean\n"
                  "FLAG @worker-2: missing edge case on empty input\n"
                  "FLAG @worker-2: no test for the retry path\n")
        flags = mod.parse_verifier_verdict(digest)
        self.assertEqual(set(flags), {"worker-2"})
        self.assertEqual(len(flags["worker-2"]), 2)
        self.assertIn("missing edge case", flags["worker-2"][0])


class StallTests(unittest.TestCase):
    def _stale_board(self, quiet_for=3600.0):
        mod = _board_module()
        store = mod.SwarmBoardStore()
        board = mod.SwarmBoard("swarm-stale-1", goal="stale work")
        board.add_member("worker-1", role="worker")
        board.add_member("worker-2", role="worker")
        board.post("worker-1", "started")
        # Backdate the board's last activity.
        board._last_activity_at = time.time() - quiet_for
        store.save_board(board)
        return store, board

    def test_stalled_team_gets_woken_once(self):
        mod = _board_module()
        store, board = self._stale_board()
        notifier = _FakeNotifier()
        out = mod.run_swarm_board_tick_once(
            SimpleNamespace(settings=SimpleNamespace(home=None)),
            notifier=notifier, store=store)
        self.assertTrue(out["ok"])
        self.assertEqual(out["woken"], 1)
        fresh = store.get_board("swarm-stale-1")
        wake_posts = [p for p in fresh.feed if p["author"] == "monitor"]
        self.assertEqual(len(wake_posts), 1)
        # The wake-up names the quiet worker.
        self.assertIn("worker-1", wake_posts[0]["mentions"])
        self.assertEqual(len(notifier.sent), 1)
        # A second tick inside the window does NOT re-wake.
        out2 = mod.run_swarm_board_tick_once(
            SimpleNamespace(settings=SimpleNamespace(home=None)),
            notifier=notifier, store=store)
        self.assertEqual(out2["woken"], 0)
        self.assertEqual(len(store.get_board("swarm-stale-1").feed),
                         len(fresh.feed))

    def test_fresh_board_is_not_woken(self):
        mod = _board_module()
        store = mod.SwarmBoardStore()
        board = mod.SwarmBoard("swarm-fresh-1", goal="fresh work")
        board.add_member("worker-1", role="worker")
        board.post("worker-1", "started")
        store.save_board(board)
        out = mod.run_swarm_board_tick_once(
            SimpleNamespace(settings=SimpleNamespace(home=None)),
            notifier=_FakeNotifier(), store=store)
        self.assertTrue(out["ok"])
        self.assertEqual(out["woken"], 0)

    def test_tick_never_raises_on_a_broken_store(self):
        mod = _board_module()

        class _BrokenStore:
            def list_boards(self):
                raise RuntimeError("store boom")

        out = mod.run_swarm_board_tick_once(
            SimpleNamespace(settings=SimpleNamespace(home=None)),
            store=_BrokenStore())
        self.assertFalse(out["ok"])
        self.assertIn("error", out)


class DurabilityTests(unittest.TestCase):
    def test_board_survives_a_store_round_trip(self):
        mod = _board_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "swarm_boards.json"
            store = mod.SwarmBoardStore(path)
            board = mod.SwarmBoard("swarm-durable-1", goal="keep me")
            board.add_member("worker-1", role="worker")
            board.add_member("worker-2", role="worker")
            board.brief("worker-1", owns="audit the parser",
                        must_not_touch="the api",
                        done="digest posted")
            board.post("worker-1", "found one bug")
            board.set_state("worker-1", "waiting")
            store.save_board(board)
            fresh = mod.SwarmBoardStore(path)
            restored = fresh.get_board("swarm-durable-1")
            self.assertIsNotNone(restored)
            self.assertEqual(restored.goal, "keep me")
            self.assertEqual(len(restored.feed), 2)  # brief + post
            self.assertEqual(restored.briefs()[0]["owns"],
                             "audit the parser")
            states = {m["name"]: m["state"] for m in restored.members}
            self.assertEqual(states["worker-1"], "waiting")
            self.assertEqual(states["worker-2"], "active")

    def test_corrupt_store_quarantines_and_degrades(self):
        mod = _board_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "swarm_boards.json"
            path.write_text("{not json", encoding="utf-8")
            store = mod.SwarmBoardStore(path)
            self.assertTrue(store.degraded_reason)
            self.assertEqual(store.list_boards(), [])
            quarantined = list(Path(tmp).glob("swarm_boards.json.*"))
            self.assertEqual(len(quarantined), 1)
            # The store still works after degrading.
            board = mod.SwarmBoard("swarm-after-1", goal="after")
            store.save_board(board)
            self.assertIsNotNone(store.get_board("swarm-after-1"))

    def test_resolve_by_unique_prefix(self):
        mod = _board_module()
        store = mod.SwarmBoardStore()
        board = mod.SwarmBoard("swarm-abc123", goal="prefix me")
        store.save_board(board)
        self.assertIsNotNone(store.resolve("swarm-abc"))
        self.assertIsNone(store.resolve("swarm-zzz"))


class JobTests(unittest.TestCase):
    def test_ensure_swarm_board_job_is_idempotent(self):
        mod = _board_module()
        sched = _FakeScheduler()
        first = mod.ensure_swarm_board_job(sched)
        second = mod.ensure_swarm_board_job(sched)
        self.assertEqual(first, second)
        self.assertEqual(len(sched.jobs), 1)
        self.assertEqual(first["name"], mod.SWARM_BOARD_JOB)
        self.assertEqual(first["payload"]["tool"], mod.SWARM_BOARD_TOOL)
        stale = mod.ensure_swarm_board_job(sched, spec="every 1m")
        self.assertEqual(len(sched.jobs), 1)
        self.assertEqual(stale["spec"], "every 1m")

    def test_ensure_returns_none_without_a_scheduler(self):
        mod = _board_module()
        self.assertIsNone(mod.ensure_swarm_board_job(None))


class SwarmWiringTests(unittest.TestCase):
    """The board wired into the real /swarm path (stubbed legs)."""

    def _stub_agent(self, script):
        from nomorals.agents import swarm as sw

        calls: list[dict] = []
        lock = threading.Lock()

        class Stubbed(sw.SwarmAgent):
            def _run_leg(self, subtask, deadline, *, index=0, run_id="",
                         on_progress=None, total_legs=0, **kw):
                with lock:
                    calls.append({"subtask": subtask, "index": index,
                                  "verifier": bool(kw.get("verifier")),
                                  "verifier_for": kw.get("verifier_for", ())})
                return script(self, subtask, index, kw)

        agent = Stubbed(SimpleNamespace())
        return agent, calls

    def test_run_posts_briefs_and_leg_posts_to_the_board(self):
        def _ok(agent, subtask, index, kw):
            return {"subtask": subtask, "ok": True,
                    "digest": f"done {index}", "planned_by": "test",
                    "steps": 1, "seconds": 0.1}

        agent, calls = self._stub_agent(_ok)
        result = agent.run("audit the parser and the api", workers=2)
        self.assertTrue(result.ok)
        board = agent.board
        self.assertIsNotNone(board)
        kinds = [p["kind"] for p in board.feed]
        self.assertIn("brief", kinds)
        self.assertGreaterEqual(kinds.count("result"), 2)
        briefs = board.briefs()
        self.assertEqual(len(briefs), 2)
        # Every worker can read every brief.
        self.assertEqual(len(board.unread(briefs[1]["name"])), len(board.feed))
        states = {m["name"]: m["state"] for m in board.members}
        self.assertTrue(all(s == "waiting" for s in states.values()))

    def test_verifier_flag_triggers_exactly_one_rerun(self):
        state = {"verifier_seen": False}

        def _script(agent, subtask, index, kw):
            if kw.get("verifier"):
                state["verifier_seen"] = True
                target = agent.board.members[0]["name"]
                return {"subtask": subtask, "ok": True,
                        "digest": f"FLAG @{target}: missing edge case",
                        "planned_by": "test", "steps": 1, "seconds": 0.1}
            return {"subtask": subtask, "ok": True,
                    "digest": f"done {index}", "planned_by": "test",
                    "steps": 1, "seconds": 0.1}

        agent, calls = self._stub_agent(_script)
        result = agent.run("audit the parser and the api", workers=2)
        self.assertTrue(result.ok)
        self.assertTrue(state["verifier_seen"])
        # Worker legs only (verifier legs excluded): the flagged
        # worker ran twice (once + one re-run), the other once.
        by_index: dict[int, int] = {}
        for call in calls:
            if call["verifier"]:
                continue
            by_index[call["index"]] = by_index.get(call["index"], 0) + 1
        self.assertEqual(sorted(by_index.values()), [1, 2])
        # The re-run brief carried the verifier's note.
        reruns = [c for c in calls
                  if c["index"] == 0 and not c["verifier"]]
        self.assertEqual(len(reruns), 2)
        self.assertTrue(any("missing edge case" in c["subtask"]
                            for c in reruns))

    def test_crashed_leg_is_killed_not_revived(self):
        def _script(agent, subtask, index, kw):
            if index == 0 and not kw.get("verifier"):
                return {"subtask": subtask, "ok": False,
                        "digest": "leg crashed", "planned_by": "test",
                        "steps": 0, "seconds": 0.1}
            return {"subtask": subtask, "ok": True,
                    "digest": "done", "planned_by": "test",
                    "steps": 1, "seconds": 0.1}

        agent, calls = self._stub_agent(_script)
        agent.run("audit the parser and the api", workers=2)
        board = agent.board
        states = {m["name"]: m["state"] for m in board.members}
        # Worker 0 failed, was re-run once, failed again, and is
        # recorded killed — not revived.
        self.assertIn("killed", states.values())
        runs_of_zero = [c for c in calls
                        if c["index"] == 0 and not c["verifier"]]
        self.assertEqual(len(runs_of_zero), 2)


class RenderTests(unittest.TestCase):
    def test_render_names_members_briefs_and_mentions(self):
        board = _board_with_workers(2)
        board.brief("worker-1", owns="audit the parser",
                    must_not_touch="the api", done="digest posted")
        board.post("worker-1", "@worker-2 the parser is clean")
        out = board.render_board()
        self.assertIn("worker-1", out)
        self.assertIn("worker-2", out)
        self.assertIn("audit the parser", out)
        self.assertIn("@worker-2", out)


if __name__ == "__main__":
    unittest.main()
