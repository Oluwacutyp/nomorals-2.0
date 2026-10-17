"""Regression: Ctrl+C did not stop ``scripts/run_chat_bot.py`` on Windows.

Root cause: ``main()`` parked the main thread in an indefinite
``threading.Event.wait()``. On Windows that blocks in
``WaitForSingleObject`` and the CPython SIGINT handler never gets to run,
so Ctrl+C was deferred forever and the bot could only be killed from a
second window. The fix: install SIGINT/SIGTERM (+SIGBREAK on Windows)
handlers that only set a stop event, and wait on that event with a short
timeout (``_SHUTDOWN_POLL_S``) so the main thread wakes regularly and
processes pending signals.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import signal
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_CHAT_BOT = REPO_ROOT / "scripts" / "run_chat_bot.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_chat_bot", RUN_CHAT_BOT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InstallHandlersTest(unittest.TestCase):
    def test_registers_sigint_sigterm_and_sigbreak_when_present(self):
        mod = _load_script()
        stop = threading.Event()
        log = logging.getLogger("test-shutdown")
        installed: dict = {}
        real_signal = signal.signal

        def _fake_signal(sig, handler):
            installed[sig] = handler
            return real_signal(signal.SIG_DFL, signal.SIG_DFL) if False else None

        with mock.patch.object(signal, "signal", side_effect=_fake_signal):
            mod._install_signal_handlers(stop, log)
        self.assertIn(signal.SIGINT, installed)
        self.assertIn(signal.SIGTERM, installed)
        if getattr(signal, "SIGBREAK", None) is not None:
            self.assertIn(signal.SIGBREAK, installed)

    def test_installed_handler_sets_stop(self):
        mod = _load_script()
        stop = threading.Event()
        log = logging.getLogger("test-shutdown")
        installed: dict = {}
        with mock.patch.object(
            signal, "signal", side_effect=lambda sig, h: installed.__setitem__(sig, h)
        ):
            mod._install_signal_handlers(stop, log)
        handler = installed[signal.SIGINT]
        handler(signal.SIGINT, None)
        self.assertTrue(stop.is_set())

    def test_install_is_safe_off_main_thread(self):
        # signal.signal raises ValueError off the main thread; the helper
        # must swallow it, not crash the boot.
        mod = _load_script()
        stop = threading.Event()
        log = logging.getLogger("test-shutdown")
        errors: list = []

        def _run():
            try:
                mod._install_signal_handlers(stop, log)
            except Exception as exc:  # noqa: BLE001 - test asserts none escapes
                errors.append(exc)

        t = threading.Thread(target=_run)
        t.start()
        t.join(timeout=5)
        self.assertFalse(t.is_alive())
        self.assertEqual(errors, [])


class WaitForShutdownTest(unittest.TestCase):
    def test_returns_immediately_when_already_set(self):
        mod = _load_script()
        stop = threading.Event()
        stop.set()
        started = time.monotonic()
        mod._wait_for_shutdown(stop)
        self.assertLess(time.monotonic() - started, 1.0)

    def test_returns_promptly_when_set_from_another_thread(self):
        # The core regression: the old indefinite stop.wait() never
        # returned to the interpreter on Windows, so a pending SIGINT
        # handler never ran. The polling wait must exit within ~2s of
        # the stop flag being set.
        mod = _load_script()
        stop = threading.Event()

        def _setter():
            time.sleep(0.2)
            stop.set()

        threading.Thread(target=_setter, daemon=True).start()
        started = time.monotonic()
        mod._wait_for_shutdown(stop)
        self.assertLess(time.monotonic() - started, 2.0)

    def test_real_sigint_stops_the_wait(self):
        # End-to-end signal path: a real SIGINT delivered to this process
        # must make the wait return (the installed handler sets the event).
        if threading.current_thread() is not threading.main_thread():
            self.skipTest("real-signal test needs the main thread")
        mod = _load_script()
        stop = threading.Event()
        log = logging.getLogger("test-shutdown")
        previous = signal.getsignal(signal.SIGINT)
        try:
            mod._install_signal_handlers(stop, log)

            def _killer():
                time.sleep(0.2)
                os.kill(os.getpid(), signal.SIGINT)

            threading.Thread(target=_killer, daemon=True).start()
            started = time.monotonic()
            mod._wait_for_shutdown(stop)
            self.assertLess(time.monotonic() - started, 2.0)
            self.assertTrue(stop.is_set())
        finally:
            try:
                signal.signal(signal.SIGINT, previous)
            except (ValueError, OSError, RuntimeError):
                pass


class MainWiringTest(unittest.TestCase):
    def test_main_stops_runtime_and_prints_devon_stopped(self):
        mod = _load_script()
        fake_runtime = mock.MagicMock()
        fake_runtime.start.return_value = ["telegram-bot"]
        fake_runtime.gateway.adapters.get.return_value = None
        fake_context = mock.MagicMock()
        fake_context.__enter__.return_value = fake_context
        fake_context.__exit__.return_value = False
        calls = {"wait": 0}

        def _fake_wait(stop):
            calls["wait"] += 1
            stop.set()

        with (
            mock.patch("nomorals.core.config.load_settings", return_value=mock.MagicMock()),
            mock.patch("nomorals.core.logging_setup.setup_logging"),
            mock.patch("nomorals.agents.context.build_context", return_value=fake_context),
            mock.patch("nomorals.agents.partner_runtime.PartnerRuntime", return_value=fake_runtime),
            mock.patch.object(mod, "_wait_for_shutdown", side_effect=_fake_wait),
        ):
            with mock.patch("builtins.print") as mock_print:
                rc = mod.main()
        self.assertEqual(rc, 0)
        self.assertEqual(calls["wait"], 1)
        fake_runtime.stop.assert_called_once()
        printed = " ".join(str(c.args[0]) for c in mock_print.call_args_list if c.args)
        self.assertIn("Devon stopped.", printed)


if __name__ == "__main__":
    unittest.main()
