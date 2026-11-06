"""Tail-100 pins: nomorals/builders/app_builder.py census sites.

The app runtime helpers are best-effort ladders around OS process
control: every rung that fails hands over to the next rung, and the
final rung reports failure honestly (False / None) instead of
claiming success. All four sites are held as exclusions:

2252 — ``AppBuilder.remove``: CONVERTED by the Pass 3 targeted
fix. remove() now attempts a stop only for an app actually
registered as running (the served/deployed registries are the
"is it running" truth, so the stops' "not running" ToolError
vocabulary is never even invoked for a stopped app), and the
result dict carries the stop outcome: ``stopped`` plus
``stop_errors`` naming any genuine failure (a ToolError from a
stop that had something to stop, or a process still alive after
the kill). ``removed: True`` remains earned by the shutil.rmtree
that actually runs, and the delete still proceeds when a stop
fails — but it no longer claims a clean teardown. The pin below
now asserts the nothing-running outcome values; the genuine-
failure halves are pinned in test_plumbing_pass3.py.

2456 — ``_signal_tree``: process-group signal fails (Windows has no
killpg, the group is gone, permissions) -> pass and fall back to
signalling the single process. The docstring's own contract:
"Returns True when a signal was delivered, False when the process
was already gone or the signal could not be delivered." PARENT:
flagged-accepted remainder (no noqa in code).

2547 — ``_find_child_pid``: the /proc scan itself raises OSError
-> pass and fall through to the ps(1) fallback rung below. A scan
that cannot run finds nothing; it never invents a pid.

2624 — ``_quick_http``: the health probe request crashes -> return
None. None is the probe's "no answer" value (the same value a
missing port returns); it is never dressed up as a healthy status.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.builders import app_builder as AB
from nomorals.core.errors import ToolError


def _builder(tmp: Path) -> AB.AppBuilder:
    settings = SimpleNamespace(workspace_dir=str(tmp))
    ctx = SimpleNamespace(settings=settings)
    return AB.AppBuilder(ctx)


class TestRemoveStopToolError(unittest.TestCase):
    """2252: stop failures are best-effort; the delete still lands."""

    def test_stop_toolerror_still_removes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            builder = _builder(Path(td))
            app_dir = builder._apps_dir / "demo-app"
            app_dir.mkdir(parents=True)
            (app_dir / "manifest.json").write_text("{}", encoding="utf-8")

            def stop_boom(name):
                raise ToolError("not running")

            builder.stop = stop_boom  # type: ignore[method-assign]
            builder.stop_deploy = stop_boom  # type: ignore[method-assign]
            result = builder.remove("demo-app")
            self.assertTrue(result["removed"])
            self.assertEqual(result["app"], "demo-app")
            self.assertFalse(app_dir.exists())
            # Pass 3: the app was never registered as running, so no
            # stop was attempted and the outcome is honestly clean.
            self.assertTrue(result["stopped"])
            self.assertEqual(result["stop_errors"], [])

    def test_remove_missing_app_raises(self):
        # Control: the swallow is scoped to the stop calls — a missing
        # app still fails loudly before any delete is attempted.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            builder = _builder(Path(td))
            builder._apps_dir.mkdir(parents=True, exist_ok=True)
            with self.assertRaises(ToolError):
                builder.remove("ghost-app")


class TestSignalTreeFallback(unittest.TestCase):
    """2456: group signal failure -> single-process signal rung."""

    def test_group_failure_falls_back_to_single_kill(self):
        killed: list[tuple[int, object]] = []

        def fake_kill(pid, sig):
            killed.append((pid, sig))

        with mock.patch.object(AB.os, "killpg",
                               side_effect=PermissionError("no group")), \
                mock.patch.object(AB.os, "getpgid", return_value=4242), \
                mock.patch.object(AB.os, "kill", side_effect=fake_kill):
            self.assertTrue(AB._signal_tree(4242, "SIGTERM"))
        self.assertEqual([pid for pid, _ in killed], [4242])

    def test_both_rungs_fail_returns_false(self):
        with mock.patch.object(AB.os, "killpg",
                               side_effect=ProcessLookupError("gone")), \
                mock.patch.object(AB.os, "getpgid", return_value=4242), \
                mock.patch.object(AB.os, "kill",
                                  side_effect=ProcessLookupError("gone")):
            self.assertFalse(AB._signal_tree(4242, "SIGTERM"))

    def test_group_success_never_touches_single_kill(self):
        # Control: a delivered group signal is the whole story.
        with mock.patch.object(AB.os, "killpg") as killpg, \
                mock.patch.object(AB.os, "getpgid", return_value=4242), \
                mock.patch.object(AB.os, "kill") as kill:
            self.assertTrue(AB._signal_tree(4242, "SIGTERM"))
        killpg.assert_called_once()
        kill.assert_not_called()


class TestFindChildPidProcCrash(unittest.TestCase):
    """2547: /proc listing crash -> ps fallback rung, no invented pid."""

    def test_listdir_crash_returns_none_for_unknown_dir(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            app_dir = Path(td) / "no-such-app"
            with mock.patch.object(
                    AB.os, "listdir",
                    side_effect=OSError("/proc vanished")):
                self.assertIsNone(AB._find_child_pid(app_dir, ["run"]))

    def test_normal_scan_returns_none_for_unknown_dir(self):
        # Control: the healthy /proc scan also finds nothing for a
        # directory no process lives in — the crash rung degrades to
        # the same honest answer.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            app_dir = Path(td) / "no-such-app"
            self.assertIsNone(AB._find_child_pid(app_dir, ["run"]))


class TestQuickHttpCrash(unittest.TestCase):
    """2624: probe request crash -> None, never a fake status."""

    def test_urlopen_crash_returns_none(self):
        with mock.patch("urllib.request.urlopen",
                        side_effect=OSError("connection refused")):
            self.assertIsNone(AB._quick_http(9, "/health"))

    def test_no_port_returns_none_without_probing(self):
        with mock.patch("urllib.request.urlopen") as urlopen:
            self.assertIsNone(AB._quick_http(0, "/health"))
        urlopen.assert_not_called()

    def test_healthy_probe_reports_real_status(self):
        # Control: a real response is reported as itself.
        class _Resp:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        with mock.patch("urllib.request.urlopen", return_value=_Resp()):
            self.assertEqual(AB._quick_http(8080, "/health"),
                             {"status": 200})


if __name__ == "__main__":
    unittest.main()
