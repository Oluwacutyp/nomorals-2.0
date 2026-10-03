"""Timeline coverage for the newer organs (wisdom, connectors, datasci,
plugins, mesh, sync, power, stream, search, triggers).

Each organ publishes fail-open lifecycle events on the process bus; the
:class:`~nomorals.os.timeline.Timeline` subscribes via
``TIMELINE_PATTERNS`` and persists them. These tests pin both halves:

* every organ's topics are covered by ``TIMELINE_PATTERNS``;
* running the organ's lifecycle functions with a live attached timeline
  actually persists the expected events;
* a broken bus never breaks the organ function (fail-open telemetry).

Conventions follow the sibling suites (``test_documents_timeline.py``,
``test_browser_timeline.py``, ``test_codews_timeline.py``).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nomorals.core.events import global_bus
from nomorals.os.timeline import TIMELINE_PATTERNS, Timeline

EXPECTED_PATTERNS = (
    "wisdom.*", "connector.*", "datasci.*", "plugin.*", "mesh.*",
    "sync.*", "power.*", "stream.*", "search.*", "trigger.*",
)


class BusTimelineCase(unittest.TestCase):
    """Attach an in-memory timeline to the real bus (sync delivery)."""

    def setUp(self) -> None:
        self.timeline = Timeline(None)
        self._sub_ids = self.timeline.attach(global_bus, sync=True)

    def tearDown(self) -> None:
        try:
            self.timeline.detach(global_bus)
        finally:
            self.timeline.close()

    def topics(self, prefix: str) -> list[str]:
        return [row["topic"] for row in
                self.timeline.query(topic=prefix + "*", limit=500)]

    def rows(self, prefix: str) -> list[dict]:
        return self.timeline.query(topic=prefix + "*", limit=500)


class TimelinePatternsTests(unittest.TestCase):
    def test_all_organ_patterns_present(self) -> None:
        for pattern in EXPECTED_PATTERNS:
            self.assertIn(pattern, TIMELINE_PATTERNS,
                          f"TIMELINE_PATTERNS missing {pattern!r}")

    def test_legacy_patterns_untouched(self) -> None:
        for pattern in ("mission.*", "artifact.*", "session.*", "task.*",
                        "document.*", "browser.*", "codews.*"):
            self.assertIn(pattern, TIMELINE_PATTERNS)


class WisdomTimelineTests(BusTimelineCase):
    def _guide(self):
        from nomorals.wisdom import PracticeGuide
        tmp = tempfile.mkdtemp(prefix="wisdom-timeline-test-")
        ctx = SimpleNamespace(settings=SimpleNamespace(workspace_dir=tmp))
        return PracticeGuide(ctx)

    def test_practice_run_emits_started_and_completed(self) -> None:
        guide = self._guide()
        lines: list[str] = []

        class FakeClock:
            def sleep(self, seconds: float) -> None:  # noqa: D102
                pass

            def now(self) -> float:  # noqa: D102
                return 1700000000.0

        result = guide.run("box-breathing", clock=FakeClock(),
                           out=lines.append)
        self.assertTrue(result["completed"])

        topics = self.topics("wisdom.")
        self.assertIn("wisdom.practice.started", topics)
        self.assertIn("wisdom.practice.completed", topics)

        started = self.rows("wisdom.practice.started")[0]
        self.assertEqual(started["data"]["session_id"], "box-breathing")
        completed = self.rows("wisdom.practice.completed")[0]
        self.assertEqual(completed["data"]["session_id"], "box-breathing")
        self.assertEqual(completed["data"]["phases_done"],
                         result["phases_done"])

    def test_practice_run_survives_broken_bus(self) -> None:
        guide = self._guide()
        lines: list[str] = []

        class FakeClock:
            def sleep(self, seconds: float) -> None:  # noqa: D102
                pass

            def now(self) -> float:  # noqa: D102
                return 1700000000.0

        with patch.object(global_bus, "publish",
                          side_effect=RuntimeError("bus down")):
            result = guide.run("box-breathing", clock=FakeClock(),
                               out=lines.append)
        self.assertTrue(result["completed"])
        self.assertTrue(lines)  # the run itself still produced output


class ConnectorTimelineTests(BusTimelineCase):
    def _fake_cls(self, ok: bool = True):
        from nomorals.connectors.base import Connector, ConnectResult

        class FakeConnector(Connector):
            id = "fake-tl"
            name = "FakeTL"
            description = "timeline test connector"
            auth_methods = ()

            def connect(self, **kwargs):  # noqa: D102
                return ConnectResult(ok=ok, account="tl-user")

            def disconnect(self):  # noqa: D102
                return None

            def status(self):  # noqa: D102
                raise NotImplementedError

            def test_connection(self):  # noqa: D102
                return True

        return FakeConnector

    def test_connect_and_disconnect_emit(self) -> None:
        conn = self._fake_cls()(MagicMock())
        result = conn.connect()
        self.assertTrue(result.ok)
        conn.disconnect()

        topics = self.topics("connector.")
        self.assertIn("connector.connected", topics)
        self.assertIn("connector.disconnected", topics)
        row = self.rows("connector.connected")[0]
        self.assertEqual(row["data"]["connector_id"], "fake-tl")
        self.assertEqual(row["data"]["account"], "tl-user")

    def test_failed_connect_emits_connect_failed_and_raises(self) -> None:
        from nomorals.connectors.base import Connector, ConnectResult

        class BadConnector(Connector):
            id = "bad-tl"
            name = "BadTL"
            description = "x"
            auth_methods = ()

            def connect(self, **kwargs):  # noqa: D102
                return ConnectResult(ok=False)

            def disconnect(self):  # noqa: D102
                return None

            def status(self):  # noqa: D102
                raise NotImplementedError

            def test_connection(self):  # noqa: D102
                return False

        class BoomConnector(BadConnector):
            id = "boom-tl"
            name = "BoomTL"

            def connect(self, **kwargs):  # noqa: D102
                raise RuntimeError("auth exploded")

        bad = BadConnector(MagicMock())
        self.assertFalse(bad.connect().ok)
        boom = BoomConnector(MagicMock())
        with self.assertRaises(RuntimeError):
            boom.connect()
        topics = self.topics("connector.")
        self.assertIn("connector.connect_failed", topics)

    def test_connect_survives_broken_bus(self) -> None:
        conn = self._fake_cls()(MagicMock())
        with patch.object(global_bus, "publish",
                          side_effect=RuntimeError("bus down")):
            result = conn.connect()
            conn.disconnect()  # must not raise
        self.assertTrue(result.ok)


class PluginTimelineTests(BusTimelineCase):
    def setUp(self) -> None:
        super().setUp()
        from nomorals.plugins.registry import PluginRegistry
        from nomorals.storage.db import Database
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Database(str(Path(self._tmp.name) / "t.db"))
        self.reg = PluginRegistry(self.db, Path(self._tmp.name) / "plugins")

    def tearDown(self) -> None:
        self._tmp.cleanup()
        super().tearDown()

    def _plugin_dir(self) -> Path:
        d = Path(self._tmp.name) / "demo-1.0.0-src"
        d.mkdir(parents=True, exist_ok=True)
        manifest = {
            "name": "demo", "version": "1.0.0",
            "description": "demo plugin", "author": "test",
            "entry_points": {"main": "demo:run"},
            "permissions": [],
        }
        (d / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
        (d / "demo.py").write_text("def run(caps):\n    return {'ok': True}\n",
                                   encoding="utf-8")
        return d

    def test_install_enable_disable_remove_emit(self) -> None:
        self.reg.install(self._plugin_dir())
        self.reg.disable("demo", "1.0.0")
        self.reg.enable("demo", "1.0.0")
        self.reg.remove("demo", "1.0.0")

        topics = self.topics("plugin.")
        for expected in ("plugin.installed", "plugin.disabled",
                         "plugin.enabled", "plugin.removed"):
            self.assertIn(expected, topics)

        row = self.rows("plugin.installed")[0]
        self.assertEqual(row["data"]["name"], "demo")
        self.assertEqual(row["data"]["version"], "1.0.0")

    def test_install_survives_broken_bus(self) -> None:
        with patch.object(global_bus, "publish",
                          side_effect=RuntimeError("bus down")):
            plugin = self.reg.install(self._plugin_dir())
        self.assertEqual(plugin.name, "demo")


class TriggerTimelineTests(BusTimelineCase):
    def _engine(self):
        from nomorals.storage.db import Database
        from nomorals.triggers.engine import TriggerEngine
        return TriggerEngine(
            Database(":memory:"),
            notify_fn=lambda trig, title, body, eng: {"ok": True},
        )

    def test_add_fire_disable_remove_emit(self) -> None:
        eng = self._engine()
        trigger = eng.add("tl-trigger", "webhook", {}, "notify",
                          {"title": "T", "body": "B"})
        result = eng.manual_fire(trigger.id)
        self.assertTrue(result["fired"])
        eng.set_enabled(trigger.id, False)
        eng.set_enabled(trigger.id, True)
        self.assertTrue(eng.remove(trigger.id))

        topics = self.topics("trigger.")
        for expected in ("trigger.added", "trigger.fired",
                         "trigger.disabled", "trigger.enabled",
                         "trigger.removed"):
            self.assertIn(expected, topics)

        row = self.rows("trigger.fired")[0]
        self.assertEqual(row["data"]["trigger_id"], trigger.id)
        self.assertEqual(row["data"]["action"], "notify")

    def test_fire_survives_broken_bus(self) -> None:
        eng = self._engine()
        trigger = eng.add("tl-trigger-2", "webhook", {}, "notify",
                          {"title": "T", "body": "B"})
        with patch.object(global_bus, "publish",
                          side_effect=RuntimeError("bus down")):
            result = eng.manual_fire(trigger.id)
        self.assertTrue(result["fired"])


class SearchTimelineTests(BusTimelineCase):
    def test_search_performed_emits(self) -> None:
        from nomorals.search.federated import federated_search
        from nomorals.search.model import SearchResult
        from nomorals.search.sources import SourceAdapter, valid_source_names

        class FakeAdapter(SourceAdapter):
            name = "memory"
            result_type = "memory"
            description = "fake"

            def probe(self):  # noqa: D102
                return None

            def search(self, query, *, limit, since=None,  # noqa: D102
                       before=None):
                return [SearchResult(
                    query=query, title="m1", snippet="s",
                    source="memory", type="memory", raw_score=1.0,
                    source_id="memory:m1")]

        adapters = {name: FakeAdapter() for name in valid_source_names()}
        resp = federated_search("timeline probe", adapters=adapters)
        self.assertGreaterEqual(len(resp.hits), 1)

        topics = self.topics("search.")
        self.assertIn("search.performed", topics)
        row = self.rows("search.performed")[0]
        self.assertEqual(row["data"]["query"], "timeline probe")
        self.assertEqual(row["data"]["result_count"], len(resp.hits))

    def test_search_survives_broken_bus(self) -> None:
        from nomorals.search.federated import federated_search
        from nomorals.search.sources import SourceAdapter, valid_source_names

        class FakeAdapter(SourceAdapter):
            name = "memory"
            result_type = "memory"
            description = "fake"

            def probe(self):  # noqa: D102
                return None

            def search(self, query, *, limit, since=None,  # noqa: D102
                       before=None):
                return []

        adapters = {name: FakeAdapter() for name in valid_source_names()}
        with patch.object(global_bus, "publish",
                          side_effect=RuntimeError("bus down")):
            resp = federated_search("q", adapters=adapters)
        self.assertEqual(resp.hits, [])


class DataSciTimelineTests(BusTimelineCase):
    def setUp(self) -> None:
        super().setUp()
        from nomorals.datasci.workspace import DataWorkspace
        self._tmp = tempfile.TemporaryDirectory()
        self.ws = DataWorkspace(Path(self._tmp.name) / "ws")
        self.csv = Path(self._tmp.name) / "data.csv"
        self.csv.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")

    def tearDown(self) -> None:
        self._tmp.cleanup()
        super().tearDown()

    def test_load_and_drop_emit(self) -> None:
        ds = self.ws.load("tl-data", self.csv)
        self.assertEqual(ds.rows, 2)
        self.ws.drop("tl-data")

        topics = self.topics("datasci.")
        self.assertIn("datasci.dataset.loaded", topics)
        self.assertIn("datasci.dataset.dropped", topics)
        row = self.rows("datasci.dataset.loaded")[0]
        self.assertEqual(row["data"]["name"], "tl-data")
        self.assertEqual(row["data"]["rows"], 2)
        self.assertEqual(row["data"]["columns"], ["a", "b"])

    def test_load_survives_broken_bus(self) -> None:
        with patch.object(global_bus, "publish",
                          side_effect=RuntimeError("bus down")):
            ds = self.ws.load("tl-data", self.csv)
        self.assertEqual(ds.rows, 2)


class MeshTimelineTests(BusTimelineCase):
    def _db(self):
        from nomorals.storage.db import Database
        from nomorals.storage.migrations import MIGRATIONS
        from nomorals.storage.schema import MigrationRunner
        db = Database(":memory:")
        MigrationRunner(db).apply_all(MIGRATIONS)
        return db

    def test_dispatch_complete_fail_emit(self) -> None:
        from nomorals.mesh.tasks import MeshTasks
        tasks = MeshTasks(self._db())
        job_id = tasks.dispatch("backup", {"x": 1}, origin_node="cloud",
                                target_node="phone")
        tasks.complete(job_id)
        job_id2 = tasks.dispatch("scan", {}, origin_node="cloud",
                                 target_node="phone")
        tasks.fail(job_id2, "boom", retry=False)

        topics = self.topics("mesh.")
        for expected in ("mesh.task.dispatched", "mesh.task.completed",
                         "mesh.task.failed"):
            self.assertIn(expected, topics)
        row = self.rows("mesh.task.dispatched")[0]
        dispatched_types = {r["data"]["task_type"]
                            for r in self.rows("mesh.task.dispatched")}
        self.assertEqual(dispatched_types, {"backup", "scan"})
        self.assertEqual(row["data"]["origin_node"], "cloud")

    def test_dispatch_survives_broken_bus(self) -> None:
        from nomorals.mesh.tasks import MeshTasks
        tasks = MeshTasks(self._db())
        with patch.object(global_bus, "publish",
                          side_effect=RuntimeError("bus down")):
            job_id = tasks.dispatch("backup", {}, origin_node="cloud")
        self.assertTrue(job_id)


class SyncTimelineTests(BusTimelineCase):
    def test_sync_completed_emits(self) -> None:
        from nomorals.storage.db import Database
        from nomorals.sync.engine import LocalPeer, SyncEngine
        from nomorals.sync.store import SyncStore
        db_a, db_b = Database(":memory:"), Database(":memory:")
        store_a, store_b = (SyncStore(db_a, device_id="a"),
                            SyncStore(db_b, device_id="b"))
        eng = SyncEngine(db_a, store_a)
        store_a.put("from_a", {"v": 1})
        store_b.put("from_b", {"v": 2})
        result = eng.sync(LocalPeer(store_b))
        self.assertEqual(result.pushed, 1)
        self.assertEqual(result.pulled, 1)

        topics = self.topics("sync.")
        self.assertIn("sync.completed", topics)
        row = self.rows("sync.completed")[0]
        self.assertEqual(row["data"]["peer_id"], "hub")
        self.assertEqual(row["data"]["pushed"], 1)
        self.assertEqual(row["data"]["pulled"], 1)

    def test_sync_survives_broken_bus(self) -> None:
        from nomorals.storage.db import Database
        from nomorals.sync.engine import LocalPeer, SyncEngine
        from nomorals.sync.store import SyncStore
        db_a, db_b = Database(":memory:"), Database(":memory:")
        store_a, store_b = (SyncStore(db_a, device_id="a"),
                            SyncStore(db_b, device_id="b"))
        eng = SyncEngine(db_a, store_a)
        store_a.put("k", {"v": 1})
        with patch.object(global_bus, "publish",
                          side_effect=RuntimeError("bus down")):
            result = eng.sync(LocalPeer(store_b))
        self.assertEqual(result.pushed, 1)


class PowerTimelineTests(BusTimelineCase):
    def _sched(self):
        from nomorals.power.monitor import PowerMonitor
        from nomorals.power.scheduler import PowerAwareScheduler
        from nomorals.storage.db import Database
        from nomorals.storage.migrations import MIGRATIONS
        from nomorals.storage.schema import MigrationRunner

        class Manager:
            def sample(self):  # noqa: D102
                return self

            def consult(self, mission=None):  # noqa: D102
                return {"ok": True}

        db = Database(":memory:")
        MigrationRunner(db).apply_all(MIGRATIONS)
        return PowerAwareScheduler(
            db, monitor=PowerMonitor(sampler=lambda: Manager()))

    def test_dispatch_complete_fail_emit(self) -> None:
        sched = self._sched()
        job_id = sched.dispatch("ping", {}, power_class="light")
        sched.complete(job_id)
        job_id2 = sched.dispatch("ping2", {}, power_class="light")
        sched.fail(job_id2, "boom", retry=False)

        topics = self.topics("power.")
        for expected in ("power.task.dispatched", "power.task.completed",
                         "power.task.failed"):
            self.assertIn(expected, topics)
        row = self.rows("power.task.dispatched")[0]
        dispatched_types = {r["data"]["task_type"]
                            for r in self.rows("power.task.dispatched")}
        self.assertEqual(dispatched_types, {"ping", "ping2"})
        self.assertEqual(row["data"]["power_class"], "light")

    def test_dispatch_survives_broken_bus(self) -> None:
        sched = self._sched()
        with patch.object(global_bus, "publish",
                          side_effect=RuntimeError("bus down")):
            job_id = sched.dispatch("ping", {}, power_class="light")
        self.assertTrue(job_id)


class StreamTimelineTests(BusTimelineCase):
    def test_start_and_stop_emit(self) -> None:
        from nomorals.stream.server import StreamServer
        server = StreamServer(lambda: Timeline(None),
                              host="127.0.0.1", port=0)
        try:
            server.start(background=True)
            self.assertTrue(server.url.startswith("http://127.0.0.1:"))
        finally:
            server.stop()

        topics = self.topics("stream.")
        self.assertIn("stream.started", topics)
        self.assertIn("stream.stopped", topics)
        row = self.rows("stream.started")[0]
        self.assertEqual(row["data"]["host"], "127.0.0.1")
        self.assertTrue(row["data"]["url"].startswith("http://"))

    def test_start_survives_broken_bus(self) -> None:
        from nomorals.stream.server import StreamServer
        server = StreamServer(lambda: Timeline(None),
                              host="127.0.0.1", port=0)
        try:
            with patch.object(global_bus, "publish",
                              side_effect=RuntimeError("bus down")):
                server.start(background=True)  # must not raise
        finally:
            server.stop()


if __name__ == "__main__":
    unittest.main()
