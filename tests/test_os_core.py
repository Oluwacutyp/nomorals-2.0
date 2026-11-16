"""Tests for the Wave H2 OS core: kernel, service registry, health monitor,
session/project stores, and read-only adapters.

All stores use in-memory SQLite; the kernel test uses its own EventBus so
it never disturbs the global bus dispatcher.
"""

from __future__ import annotations

import unittest

from nomorals.core.events import Event, EventBus
from nomorals.os.adapters import AccountSessionAdapter, VoiceSessionAdapter
from nomorals.os.health import (
    HealthMonitor,
    HealthStatus,
    as_health_check,
)
from nomorals.os.kernel import OSKernel
from nomorals.os.project import ProjectStore, artifact_scope
from nomorals.os.services import ServiceNotFound, ServiceRegistry
from nomorals.os.session import SessionStore


class TestOSKernel(unittest.TestCase):
    def test_start_stop_idempotent(self) -> None:
        kernel = OSKernel(bus=EventBus())
        self.assertFalse(kernel.started)
        kernel.start()
        kernel.start()  # second start is a no-op
        self.assertTrue(kernel.started)
        kernel.stop()
        kernel.stop()  # second stop is a no-op
        self.assertFalse(kernel.started)

    def test_start_wires_service_health_checks_and_runs_them(self) -> None:
        kernel = OSKernel(bus=EventBus())
        kernel.start()
        try:
            self.assertGreater(len(kernel.health.check_names()), 0)
            for name in kernel.health.check_names():
                status = kernel.health.last_status(name)
                self.assertIsNotNone(status)
                self.assertTrue(status.ok, f"{name}: {status.detail}")
        finally:
            kernel.stop()

    def test_context_manager(self) -> None:
        with OSKernel(bus=EventBus()) as kernel:
            self.assertTrue(kernel.started)
        self.assertFalse(kernel.started)

    def test_status_snapshot(self) -> None:
        kernel = OSKernel(bus=EventBus())
        kernel.start()
        try:
            snapshot = kernel.status()
            self.assertTrue(snapshot["started"])
            self.assertIn("browser", snapshot["services"])
            self.assertIn("service:browser", snapshot["health"])
        finally:
            kernel.stop()


class TestServiceRegistry(unittest.TestCase):
    def test_register_lookup_list_unregister(self) -> None:
        registry = ServiceRegistry()
        registry.register("thing", lambda: object(), description="test thing")
        self.assertIn("thing", registry.list_services())
        first = registry.lookup("thing")
        second = registry.lookup("thing")
        self.assertIs(first, second)  # cached
        self.assertTrue(registry.unregister("thing"))
        self.assertNotIn("thing", registry.list_services())
        self.assertFalse(registry.unregister("thing"))

    def test_lazy_factory_called_once(self) -> None:
        calls: list[int] = []

        def factory() -> object:
            calls.append(1)
            return object()

        registry = ServiceRegistry()
        registry.register("lazy", factory)
        self.assertFalse(registry.is_constructed("lazy"))
        registry.lookup("lazy")
        registry.lookup("lazy")
        registry.lookup("lazy")
        self.assertEqual(len(calls), 1)
        self.assertTrue(registry.is_constructed("lazy"))

    def test_lookup_unknown_raises(self) -> None:
        registry = ServiceRegistry()
        with self.assertRaises(ServiceNotFound):
            registry.lookup("nope")

    def test_double_register_without_replace_raises(self) -> None:
        registry = ServiceRegistry()
        registry.register("dup", lambda: 1)
        with self.assertRaises(ValueError):
            registry.register("dup", lambda: 2)
        registry.register("dup", lambda: 2, replace=True)
        self.assertEqual(registry.lookup("dup"), 2)

    def test_builtin_descriptors_present(self) -> None:
        registry = ServiceRegistry()
        for name in ("browser", "model", "bridge", "queue"):
            self.assertIn(name, registry.list_services())
            self.assertIsNotNone(registry.health_check_for(name))


class TestHealthMonitor(unittest.TestCase):
    def test_check_all_reports_statuses(self) -> None:
        monitor = HealthMonitor()
        monitor.register_check(as_health_check("good", lambda: True))
        monitor.register_check(as_health_check("bad", lambda: (False, "kaput")))
        results = monitor.check_all()
        self.assertTrue(results["good"].ok)
        self.assertFalse(results["bad"].ok)
        self.assertEqual(results["bad"].detail, "kaput")
        self.assertIsNotNone(monitor.last_status("good"))

    def test_recovery_hook_fires_on_failure(self) -> None:
        fired: list[HealthStatus] = []
        monitor = HealthMonitor()
        monitor.register_check(as_health_check("flaky", lambda: False))
        monitor.on_failure("flaky", fired.append)
        monitor.check_all()
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].name, "flaky")
        self.assertFalse(fired[0].ok)

    def test_recovery_hook_not_fired_on_success(self) -> None:
        fired: list[HealthStatus] = []
        monitor = HealthMonitor()
        monitor.register_check(as_health_check("fine", lambda: True))
        monitor.on_failure("fine", fired.append)
        monitor.check_all()
        self.assertEqual(fired, [])

    def test_wildcard_hook_catches_any_failure(self) -> None:
        fired: list[str] = []
        monitor = HealthMonitor()
        monitor.register_check(as_health_check("a", lambda: False))
        monitor.register_check(as_health_check("b", lambda: True))
        monitor.on_failure("*", lambda status: fired.append(status.name))
        monitor.check_all()
        self.assertEqual(fired, ["a"])

    def test_raising_check_becomes_failed_status(self) -> None:
        def boom() -> bool:
            raise RuntimeError("kablam")

        monitor = HealthMonitor()
        monitor.register_check(as_health_check("boom", boom))
        results = monitor.check_all()
        self.assertFalse(results["boom"].ok)
        self.assertIn("kablam", results["boom"].detail)


class TestSessionStore(unittest.TestCase):
    def test_create_get_end_round_trip(self) -> None:
        store = SessionStore()
        session = store.create(frontend="telegram", principal="owner",
                               project_id="proj_1",
                               state={"chat_id": 123})
        self.assertTrue(session.id.startswith("sess_"))
        self.assertTrue(session.active)

        fetched = store.get(session.id)
        self.assertIsNotNone(fetched)
        assert fetched is not None
        self.assertEqual(fetched.frontend, "telegram")
        self.assertEqual(fetched.state, {"chat_id": 123})

        self.assertEqual(len(store.list_active()), 1)
        self.assertTrue(store.end(session.id))
        self.assertEqual(store.list_active(), [])
        ended = store.get(session.id)
        assert ended is not None
        self.assertFalse(ended.active)
        self.assertFalse(store.end("sess_missing"))

    def test_project_scoping_round_trip(self) -> None:
        projects = ProjectStore()
        project = projects.create("demo")
        sessions = SessionStore()
        s1 = sessions.create(frontend="cli", project_id=project.id)
        sessions.create(frontend="api", project_id="other")
        scoped = sessions.list_for_project(project.id)
        self.assertEqual([s.id for s in scoped], [s1.id])

    def test_events_published_on_bus(self) -> None:
        seen: list[Event] = []
        bus = EventBus()
        bus.subscribe("session.*", seen.append, sync=True)
        import nomorals.os.session as session_module
        real_bus = session_module.global_bus
        session_module.global_bus = bus
        try:
            store = SessionStore()
            session = store.create(frontend="tui", principal="owner")
            store.end(session.id)
        finally:
            session_module.global_bus = real_bus
        topics = [e.topic for e in seen]
        self.assertIn("session.created", topics)
        self.assertIn("session.ended", topics)
        created = next(e for e in seen if e.topic == "session.created")
        self.assertEqual(created.data["session_id"], session.id)
        self.assertEqual(created.data["frontend"], "tui")
        self.assertEqual(created.data["principal"], "owner")

    def test_update_persists_state(self) -> None:
        store = SessionStore()
        session = store.create(frontend="cli")
        session.state["note"] = "hello"
        session.artifact_scope.append("art_1")
        store.update(session)
        fetched = store.get(session.id)
        assert fetched is not None
        self.assertEqual(fetched.state["note"], "hello")
        self.assertEqual(fetched.artifact_scope, ["art_1"])


class TestProjectStore(unittest.TestCase):
    def test_create_get_list_missions(self) -> None:
        store = ProjectStore()
        project = store.create("wave-h2", description="os control plane",
                               mission_ids=["m1"])
        self.assertTrue(project.id.startswith("proj_"))
        self.assertEqual(project.mission_ids, ["m1"])

        self.assertTrue(store.add_mission(project.id, "m2"))
        self.assertTrue(store.add_mission(project.id, "m2"))  # idempotent
        fetched = store.get(project.id)
        assert fetched is not None
        self.assertEqual(fetched.mission_ids, ["m1", "m2"])

        self.assertTrue(store.remove_mission(project.id, "m1"))
        self.assertEqual(store.get(project.id).mission_ids, ["m2"])  # type: ignore[union-attr]

        self.assertFalse(store.add_mission("proj_missing", "m9"))
        self.assertFalse(store.remove_mission("proj_missing", "m9"))
        self.assertIsNone(store.get("proj_missing"))

        names = [p.name for p in store.list()]
        self.assertIn("wave-h2", names)

    def test_set_state(self) -> None:
        store = ProjectStore()
        project = store.create("p")
        self.assertTrue(store.set_state(project.id, "paused"))
        fetched = store.get(project.id)
        assert fetched is not None
        self.assertEqual(fetched.state, "paused")
        self.assertFalse(store.set_state("proj_missing", "paused"))

    def test_artifact_scope_composes_for_mission(self) -> None:
        class FakeArtifact:
            def __init__(self, artifact_id: str) -> None:
                self.id = artifact_id

        class FakeStore:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def for_mission(self, mission_id: str) -> list[FakeArtifact]:
                self.calls.append(mission_id)
                return [FakeArtifact(f"{mission_id}-a"),
                        FakeArtifact(f"{mission_id}-b")]

        store = ProjectStore()
        project = store.create("p", mission_ids=["m1", "m2"])
        fake = FakeStore()
        artifacts = artifact_scope(fake, project)
        self.assertEqual([a.id for a in artifacts],
                         ["m1-a", "m1-b", "m2-a", "m2-b"])
        self.assertEqual(fake.calls, ["m1", "m2"])

    def test_artifact_scope_dedupes_and_survives_bad_missions(self) -> None:
        class FakeArtifact:
            def __init__(self, artifact_id: str) -> None:
                self.id = artifact_id

        class FlakyStore:
            def for_mission(self, mission_id: str) -> list[FakeArtifact]:
                if mission_id == "bad":
                    raise RuntimeError("db gone")
                return [FakeArtifact("shared")]

        store = ProjectStore()
        project = store.create("p", mission_ids=["m1", "bad", "m2"])
        artifacts = artifact_scope(FlakyStore(), project)
        self.assertEqual([a.id for a in artifacts], ["shared"])

    def test_artifact_scope_without_for_mission_returns_empty(self) -> None:
        store = ProjectStore()
        project = store.create("p", mission_ids=["m1"])
        self.assertEqual(artifact_scope(object(), project), [])


class _FakeAccount:
    def __init__(self) -> None:
        self.service = "gmail"
        self.username = "user@example.com"
        self.oauth_token = None
        self.scope = "read"
        self.metadata = {"label": "main"}
        self.created_at = 1000.0
        self.last_used = 2000.0

    def is_valid(self) -> bool:
        return True


class _FakeAccountManager:
    def __init__(self, account: _FakeAccount | None = None) -> None:
        self._account = account or _FakeAccount()

    def get_session(self, service: str, username: str) -> _FakeAccount:
        assert service == "gmail"
        assert username == "user@example.com"
        return self._account


class _FakeVoiceSession:
    def __init__(self) -> None:
        self.session_id = "vs_1"
        self.state = "listening"
        self.device_id = "pixel"
        self.profile = "warm"
        self.stats = None
        self.consent = object()


class TestAdapters(unittest.TestCase):
    def test_account_adapter_projects_view(self) -> None:
        adapter = AccountSessionAdapter(_FakeAccountManager())
        view = adapter.to_os_session("gmail", "user@example.com",
                                     frontend="api", principal="owner",
                                     project_id="proj_1")
        self.assertEqual(view.frontend, "api")
        self.assertEqual(view.project_id, "proj_1")
        self.assertEqual(view.state["kind"], "account")
        self.assertEqual(view.state["service"], "gmail")
        self.assertTrue(view.state["valid"])
        self.assertFalse(view.state["has_oauth"])
        self.assertEqual(view.conversation_id, "gmail:user@example.com")

    def test_account_adapter_never_raises(self) -> None:
        # No manager at all.
        view = AccountSessionAdapter(None).to_os_session("gmail", "u@x.com")
        self.assertTrue(view.state["degraded"])
        # Manager missing the expected method.
        view2 = AccountSessionAdapter(object()).to_os_session("x", "y")
        self.assertTrue(view2.state["degraded"])
        # get_session itself raises.
        class Broken:
            def get_session(self, service: str, username: str) -> None:
                raise RuntimeError("vault locked")
        view3 = AccountSessionAdapter(Broken()).to_os_session("x", "y")
        self.assertTrue(view3.state["degraded"])

    def test_voice_adapter_projects_view(self) -> None:
        adapter = VoiceSessionAdapter(_FakeVoiceSession())
        view = adapter.to_os_session(principal="owner")
        self.assertEqual(view.frontend, "voice")
        self.assertEqual(view.state["kind"], "voice")
        self.assertEqual(view.state["voice_state"], "listening")
        self.assertEqual(view.state["device_id"], "pixel")
        self.assertEqual(view.conversation_id, "vs_1")

    def test_voice_adapter_never_raises(self) -> None:
        view = VoiceSessionAdapter().to_os_session()
        self.assertTrue(view.state["degraded"])
        # Half-initialized object: missing attributes degrade gracefully.
        view2 = VoiceSessionAdapter(object()).to_os_session()
        self.assertEqual(view2.state["kind"], "voice")
        self.assertEqual(view2.state["voice_state"], "")


if __name__ == "__main__":
    unittest.main()
