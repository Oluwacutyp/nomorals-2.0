"""Wave H2 Worker 3 — persistent event timeline tests.

Covers: subscribing to the H2 topic contract on a real EventBus, the
query filters (mission/session/topic glob/since/until/limit), newest-first
ordering, and malformed-event resilience.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

from nomorals.core.events import Event, EventBus
from nomorals.os.timeline import TIMELINE_PATTERNS, Timeline


def _publish(bus: EventBus, topic: str, ts: float, **data) -> Event:
    return bus.publish(Event(topic=topic, ts=ts, source="test", data=data))


class TimelineRecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-timeline-")
        self.addCleanup(self.tmp.cleanup)
        self.timeline = Timeline(Path(self.tmp.name) / "timeline.db")
        self.addCleanup(self.timeline.close)
        self.bus = EventBus()
        self.addCleanup(self.bus.stop)

    def _attach_and_seed(self) -> float:
        self.timeline.attach(self.bus)
        base = 1_700_000_000.0
        with self.bus:  # exiting drains the async dispatcher queue
            _publish(self.bus, "mission.transition", base + 1,
                     mission_id="m1", from_state="running", to_state="done",
                     session_id="s1")
            _publish(self.bus, "mission.verify", base + 2,
                     mission_id="m1", verdict="pass", session_id="s1")
            _publish(self.bus, "artifact.created", base + 3,
                     artifact_id="a1", mission_id="m1", session_id="s1",
                     uri="artifact://a1")
            _publish(self.bus, "session.created", base + 4,
                     session_id="s2", frontend="cli", principal="owner")
            _publish(self.bus, "session.ended", base + 5, session_id="s2")
            _publish(self.bus, "task.started", base + 6,
                     task_id="t1", mission_id="m2", session_id="s1")
            _publish(self.bus, "task.done", base + 7,
                     task_id="t1", mission_id="m2", session_id="s1")
            # Not in the contract — must NOT be recorded.
            _publish(self.bus, "memory.consolidated", base + 8, session_id="s1")
        return base

    def test_contract_topics_recorded_unrelated_ignored(self):
        self._attach_and_seed()
        rows = self.timeline.query(limit=100)
        topics = {r["topic"] for r in rows}
        self.assertEqual(len(rows), 7)
        self.assertNotIn("memory.consolidated", topics)
        for pattern in ("mission.*", "session.*", "task.*", "artifact.*"):
            self.assertTrue(any(t.startswith(pattern[:-1]) for t in topics),
                            f"expected a topic matching {pattern}")

    def test_subscribed_patterns(self):
        self.assertEqual(
            set(TIMELINE_PATTERNS),
            {"mission.*", "artifact.*", "session.*", "task.*",
             "document.*", "browser.*", "codews.*",
             "wisdom.*", "connector.*", "datasci.*", "plugin.*", "mesh.*",
             "sync.*", "power.*", "stream.*", "search.*", "trigger.*"})

    def test_newest_first_ordering(self):
        self._attach_and_seed()
        rows = self.timeline.query(limit=100)
        self.assertEqual(rows[0]["topic"], "task.done")
        self.assertEqual(rows[-1]["topic"], "mission.transition")
        tss = [r["ts"] for r in rows]
        self.assertEqual(tss, sorted(tss, reverse=True))

    def test_filter_by_mission(self):
        self._attach_and_seed()
        rows = self.timeline.query(mission_id="m1", limit=100)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["mission_id"] == "m1" for r in rows))

    def test_filter_by_session(self):
        self._attach_and_seed()
        rows = self.timeline.query(session_id="s2", limit=100)
        self.assertEqual({r["topic"] for r in rows},
                         {"session.created", "session.ended"})

    def test_filter_by_artifact(self):
        self._attach_and_seed()
        rows = self.timeline.query(artifact_id="a1", limit=100)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["topic"], "artifact.created")
        self.assertEqual(rows[0]["data"]["uri"], "artifact://a1")

    def test_topic_glob(self):
        self._attach_and_seed()
        rows = self.timeline.query(topic="mission.*", limit=100)
        self.assertEqual({r["topic"] for r in rows},
                         {"mission.transition", "mission.verify"})
        rows = self.timeline.query(topic="session.ended", limit=100)
        self.assertEqual(len(rows), 1)

    def test_since_until_limit(self):
        base = self._attach_and_seed()
        rows = self.timeline.query(since=base + 4, limit=100)
        self.assertTrue(all(r["ts"] >= base + 4 for r in rows))
        self.assertEqual(len(rows), 4)
        rows = self.timeline.query(until=base + 2, limit=100)
        self.assertEqual(len(rows), 2)
        rows = self.timeline.query(limit=3)
        self.assertEqual(len(rows), 3)

    def test_recent(self):
        self._attach_and_seed()
        rows = self.timeline.recent(2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["topic"], "task.done")

    def test_data_survives_round_trip(self):
        self._attach_and_seed()
        rows = self.timeline.query(topic="mission.transition", limit=10)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["data"]["from_state"], "running")
        self.assertEqual(row["data"]["to_state"], "done")
        self.assertEqual(row["source"], "test")
        self.assertTrue(row["event_id"].startswith("evt_"))

    def test_malformed_event_does_not_break_recording(self):
        self.timeline.attach(self.bus)
        bad = Event(topic="task.broken", ts="not-a-timestamp", source="test")
        bad.data = "this is not a dict"  # type: ignore[assignment]
        event_id = self.timeline.record(bad)
        self.assertIsNotNone(event_id)
        rows = self.timeline.query(topic="task.broken", limit=10)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["data"]["_raw"], "this is not a dict")

        # A malformed event arriving through the bus must not kill the bus.
        bad2 = Event(topic="task.broken2", source="test")
        bad2.data = 12345  # type: ignore[assignment]
        with self.bus:
            self.bus.publish(bad2)
            self.bus.publish(Event(topic="task.ok", source="test",
                                   data={"task_id": "t9"}))
        rows = self.timeline.query(limit=100)
        topics = {r["topic"] for r in rows}
        self.assertIn("task.broken2", topics)
        self.assertIn("task.ok", topics)

    def test_detach_stops_recording(self):
        self.timeline.attach(self.bus)
        with self.bus:
            self.bus.publish(Event(topic="task.one", source="test"))
        self.assertEqual(len(self.timeline.query(limit=100)), 1)
        self.timeline.detach(self.bus)
        with self.bus:
            self.bus.publish(Event(topic="task.two", source="test"))
        self.assertEqual(len(self.timeline.query(limit=100)), 1)

    def test_in_memory_timeline(self):
        with Timeline(None) as tl:
            tl.record(Event(topic="task.x", source="test", data={"a": 1}))
            rows = tl.query(limit=10)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["data"]["a"], 1)


class TimelineCliTests(unittest.TestCase):
    """``nm timeline`` reads the persisted log from the default storage DB."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-timeline-cli-")
        self.addCleanup(self.tmp.cleanup)
        from nomorals.agents.context import build_context
        from nomorals.core.config import Settings
        self._ctx_mgr = build_context(Settings(home=self.tmp.name))
        self.ctx = self._ctx_mgr.__enter__()
        self.addCleanup(self._ctx_mgr.__exit__, None, None, None)

    def _seed(self) -> None:
        tl = Timeline(self.ctx.db.path)
        try:
            tl.record(Event(topic="mission.transition", ts=time.time(),
                            source="test",
                            data={"mission_id": "m9", "from_state": "a",
                                  "to_state": "b"}))
            tl.record(Event(topic="session.created", ts=time.time(),
                            source="test",
                            data={"session_id": "s9", "frontend": "cli"}))
        finally:
            tl.close()

    def _run(self, *argv: str) -> str:
        import io
        from contextlib import redirect_stdout
        from nomorals.cli import _parser, _cmd_timeline
        args = _parser().parse_args(["timeline", *argv])
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_timeline(args, self.ctx)
        self.assertEqual(rc, 0)
        return buf.getvalue()

    def test_timeline_json_output(self):
        self._seed()
        out = self._run("--json")
        rows = json.loads(out)
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["topic"] for r in rows},
                         {"mission.transition", "session.created"})

    def test_timeline_filters(self):
        self._seed()
        out = self._run("--mission", "m9", "--json")
        rows = json.loads(out)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["topic"], "mission.transition")

    def test_timeline_human_readable(self):
        self._seed()
        out = self._run("--limit", "5")
        self.assertIn("mission.transition", out)
        self.assertIn("session.created", out)

    def test_timeline_empty_db(self):
        out = self._run("--json")
        self.assertEqual(json.loads(out), [])


class CliSessionAttachTests(unittest.TestCase):
    """The CLI session stash: create-or-reuse, never breaks dispatch."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-cli-sess-")
        self.addCleanup(self.tmp.cleanup)

    def _ctx(self):
        from nomorals.storage.db import Database
        db = Database(str(Path(self.tmp.name) / "devon.db"))
        ctx = type("Ctx", (), {"extras": {}, "db": db})()
        self.addCleanup(db.close)
        return ctx

    def test_attach_creates_cli_owner_session(self):
        from nomorals.cli import _attach_cli_session
        ctx = self._ctx()
        _attach_cli_session(ctx)
        session = ctx.extras.get("os_session")
        self.assertIsNotNone(session)
        self.assertEqual(session.frontend, "cli")
        self.assertEqual(session.principal, "owner")

    def test_attach_reuses_active_cli_session(self):
        from nomorals.cli import _attach_cli_session
        ctx = self._ctx()
        _attach_cli_session(ctx)
        first = ctx.extras["os_session"]
        ctx2 = self._ctx()  # same underlying db file
        _attach_cli_session(ctx2)
        second = ctx2.extras["os_session"]
        self.assertEqual(first.id, second.id)

    def test_attach_never_raises_on_broken_context(self):
        from nomorals.cli import _attach_cli_session
        ctx = type("Ctx", (), {"extras": {}, "db": None})()
        try:
            _attach_cli_session(ctx)  # in-memory store; must not raise
        except Exception as exc:  # noqa: BLE001 - documents the contract
            self.fail(f"_attach_cli_session raised: {exc!r}")


if __name__ == "__main__":
    unittest.main()
