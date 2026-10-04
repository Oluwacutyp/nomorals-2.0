"""Round 10 — startup path, shutdown path, health checks.

Covers the R10 fixes:
* ``install_sigterm_as_interrupt`` maps SIGTERM onto KeyboardInterrupt so
  every long-running entry point (CLI, ``nm serve``, partner runtime)
  shuts down gracefully on a supervisor's SIGTERM.
* ``_build_router`` builds the ModelBroker BEFORE ``attach_learning`` so
  the learning trajectory store is injected into the broker that stays
  attached (the reverse order silently discarded it).
* ``GET /health`` carries local-only operational facts (pid, uptime,
  provider names, bot runtime state) without network or model calls.
* ``nm status`` gains a ``runtime`` section reading the partner bot's
  status beacon (alive / stale / stopped / not_running).
"""

from __future__ import annotations

import json
import signal
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class SigtermHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous = signal.getsignal(signal.SIGTERM)

    def tearDown(self) -> None:
        signal.signal(signal.SIGTERM, self._previous)

    def test_installs_handler_that_raises_keyboard_interrupt(self) -> None:
        from nomorals.core.shutdown import install_sigterm_as_interrupt

        self.assertTrue(install_sigterm_as_interrupt())
        handler = signal.getsignal(signal.SIGTERM)
        self.assertTrue(callable(handler))
        with self.assertRaises(KeyboardInterrupt):
            handler(signal.SIGTERM, None)  # type: ignore[operator]

    def test_idempotent_second_install_keeps_handler(self) -> None:
        from nomorals.core.shutdown import install_sigterm_as_interrupt

        self.assertTrue(install_sigterm_as_interrupt())
        first = signal.getsignal(signal.SIGTERM)
        self.assertTrue(install_sigterm_as_interrupt())
        second = signal.getsignal(signal.SIGTERM)
        self.assertIs(first, second)

    def test_noop_off_main_thread(self) -> None:
        import threading

        from nomorals.core.shutdown import install_sigterm_as_interrupt

        # Isolate from whatever other tests left installed: pin a known
        # pre-state on the main thread first.  Full-suite runs used to
        # flake here when an earlier test left the devon handler
        # installed — the worker then short-circuited on the
        # "already installed" check and returned True instead of False.
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        results: list[bool] = []
        t = threading.Thread(
            target=lambda: results.append(install_sigterm_as_interrupt()),
            daemon=True,
        )
        t.start()
        # Generous join: under a loaded full-suite run a short timeout can
        # expire from CPU starvation, not from the helper misbehaving.
        t.join(timeout=60)
        self.assertFalse(t.is_alive(), "worker thread hung")
        self.assertEqual(results, [False])
        # the worker must not touch the process-global handler
        self.assertIs(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)

    def test_off_main_thread_reports_already_installed(self) -> None:
        # Companion isolation case: when the devon handler IS already
        # installed on the main thread, a worker correctly reports the
        # mapping in effect (True) without reinstalling anything.
        import threading

        from nomorals.core.shutdown import install_sigterm_as_interrupt

        self.assertTrue(install_sigterm_as_interrupt())
        installed = signal.getsignal(signal.SIGTERM)
        results: list[bool] = []
        t = threading.Thread(
            target=lambda: results.append(install_sigterm_as_interrupt()),
            daemon=True,
        )
        t.start()
        t.join(timeout=60)
        self.assertFalse(t.is_alive(), "worker thread hung")
        self.assertEqual(results, [True])
        self.assertIs(signal.getsignal(signal.SIGTERM), installed)


class BrokerLearningOrderTests(unittest.TestCase):
    def test_learning_store_lands_in_attached_broker(self) -> None:
        """attach_learning must inject the trajectory store into the broker
        that stays attached to the router — not a throwaway it replaces."""
        from nomorals.agents.context import _build_router
        from nomorals.core.config import Settings
        from nomorals.core.events import EventBus
        from nomorals.storage.db import Database

        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings()
            settings.home = tmp
            db = Database(Path(tmp) / "t.db")
            db.migrate()
            bus = EventBus().start()
            try:
                router = _build_router(settings, bus, db=db)
                broker = router.broker
                self.assertIsNotNone(broker)
                # The attached broker is the one learning enriched.
                self.assertIsNotNone(
                    getattr(broker, "trajectories", None),
                    "trajectory store missing from attached broker: "
                    "learning attach was overwritten",
                )
                self.assertGreaterEqual(len(router.providers()), 1)
            finally:
                bus.stop()
                db.close()


class HealthEndpointTests(unittest.TestCase):
    def _server(self, home: str | None = None):
        from nomorals.api.server import APIServer

        settings = SimpleNamespace(home=home or "")
        context = SimpleNamespace(db=None, router=None, settings=settings,
                                  started_at=time.time() - 42.0)
        return APIServer(context)

    def test_health_carries_local_facts(self) -> None:
        server = self._server()
        status, payload = server.dispatch("GET", "/health", {}, {})
        self.assertEqual(status, 200)
        self.assertTrue(payload["ok"])
        # Back-compat fields unchanged.
        self.assertIn("version", payload)
        self.assertIn("schema_version", payload)
        # New local-only facts: no network, no model calls.
        self.assertIsInstance(payload["pid"], int)
        self.assertGreaterEqual(payload["uptime_s"], 40.0)
        self.assertEqual(payload["providers"], [])
        self.assertEqual(payload["runtime"]["state"], "not_running")

    def test_health_reports_alive_bot(self) -> None:
        from nomorals.agents.beacon import BEACON_INTERVAL_S

        with tempfile.TemporaryDirectory() as tmp:
            beacon = Path(tmp) / "state" / "status.json"
            beacon.parent.mkdir(parents=True, exist_ok=True)
            beacon.write_text(json.dumps({
                "ts": time.time(), "pid": 999, "uptime_s": 123.4,
                "stats": {}, "last_reply": {}, "last_error": "boom",
                "stopped": False,
            }))
            server = self._server(home=tmp)
            status, payload = server.dispatch("GET", "/health", {}, {})
            self.assertEqual(status, 200)
            self.assertEqual(payload["runtime"]["state"], "alive")
            self.assertLessEqual(payload["runtime"]["beacon_age_s"],
                                 BEACON_INTERVAL_S + 5)
            self.assertEqual(payload["runtime"]["last_error"], "boom")

    def test_health_with_provider_names(self) -> None:
        from nomorals.api.server import APIServer

        router = SimpleNamespace(providers=lambda: ["mock", "ocr"])
        context = SimpleNamespace(db=None, router=router,
                                  settings=SimpleNamespace(home=""),
                                  started_at=time.time())
        server = APIServer(context)
        status, payload = server.dispatch("GET", "/health", {}, {})
        self.assertEqual(status, 200)
        self.assertEqual(payload["providers"], ["mock", "ocr"])


class StatusRuntimeSectionTests(unittest.TestCase):
    def _context(self, tmp: str):
        return SimpleNamespace(settings=SimpleNamespace(home=tmp))

    def _write_beacon(self, tmp: str, **over: object) -> None:
        state = {"ts": time.time(), "pid": 1, "uptime_s": 60.0,
                 "stats": {}, "last_reply": {}, "last_error": "",
                 "stopped": False}
        state.update(over)
        p = Path(tmp) / "state" / "status.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(state))

    def test_not_running_without_beacon(self) -> None:
        from nomorals.cmdline.commands.status import _runtime_section

        with tempfile.TemporaryDirectory() as tmp:
            data, text = _runtime_section(self._context(tmp))
            self.assertEqual(data["state"], "not_running")
            self.assertIn("not running", text[0])

    def test_alive_beacon(self) -> None:
        from nomorals.cmdline.commands.status import _runtime_section

        with tempfile.TemporaryDirectory() as tmp:
            self._write_beacon(
                tmp, platforms={"telegram": {"running": True}},
                last_error="send failed once",
            )
            data, text = _runtime_section(self._context(tmp))
            self.assertEqual(data["state"], "alive")
            self.assertIn("ALIVE", text[0])
            self.assertIn("telegram", text[0])
            self.assertIn("last error", text[1])
            self.assertEqual(data["adapters"], ["telegram"])

    def test_clean_stop_beacon(self) -> None:
        from nomorals.cmdline.commands.status import _runtime_section

        with tempfile.TemporaryDirectory() as tmp:
            self._write_beacon(tmp, stopped=True,
                               ts=time.time() - 120)
            data, text = _runtime_section(self._context(tmp))
            self.assertEqual(data["state"], "stopped")
            self.assertIn("stopped cleanly", text[0])

    def test_stale_beacon(self) -> None:
        from nomorals.cmdline.commands.status import _runtime_section

        with tempfile.TemporaryDirectory() as tmp:
            self._write_beacon(tmp, ts=time.time() - 3600)
            data, text = _runtime_section(self._context(tmp))
            self.assertEqual(data["state"], "stale")
            self.assertIn("STALE", text[0])


class ServeSigtermTests(unittest.TestCase):
    def test_serve_installs_sigterm_handler_in_foreground(self) -> None:
        """serve() foreground maps SIGTERM before blocking in serve_forever."""
        import threading

        from nomorals.api import server as server_mod

        previous = signal.getsignal(signal.SIGTERM)
        installed: list[bool] = []
        try:
            # serve() imports the helper lazily; patch the source module.
            from nomorals.core import shutdown as shutdown_mod
            real = shutdown_mod.install_sigterm_as_interrupt

            def spy() -> bool:
                installed.append(True)
                return real()

            with mock.patch.object(shutdown_mod,
                                   "install_sigterm_as_interrupt", spy):
                context = SimpleNamespace(
                    settings=SimpleNamespace(api=SimpleNamespace(token="")))
                t = threading.Thread(
                    target=server_mod.serve, kwargs={
                        "context": context, "host": "127.0.0.1", "port": 0,
                    }, daemon=True)
                t.start()
                deadline = time.time() + 10
                while not installed and time.time() < deadline:
                    time.sleep(0.05)
                self.assertTrue(installed, "SIGTERM handler not installed")
        finally:
            signal.signal(signal.SIGTERM, previous)


class ContextCloseTests(unittest.TestCase):
    def _context(self, executor):
        from unittest.mock import MagicMock

        from nomorals.agents.context import AgentContext

        return AgentContext(
            settings=MagicMock(), db=MagicMock(), bus=MagicMock(),
            executor=executor)

    def test_close_uses_graceful_executor_shutdown(self) -> None:
        from unittest.mock import MagicMock

        executor = MagicMock()
        ctx = self._context(executor)
        ctx.close()
        executor.shutdown.assert_called_once_with(wait=True, timeout=10.0)
        # Idempotent: second close must not touch the executor again.
        ctx.close()
        self.assertEqual(executor.shutdown.call_count, 1)

    def test_close_falls_back_for_kwarg_less_executors(self) -> None:
        from unittest.mock import MagicMock

        class Plain:
            def __init__(self):
                self.calls = 0

            def shutdown(self):
                self.calls += 1

        executor = Plain()
        ctx = self._context(executor)
        ctx.close()  # must not raise on the TypeError path
        self.assertEqual(executor.calls, 1)
        ctx.db.close.assert_called_once()
        ctx.bus.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
