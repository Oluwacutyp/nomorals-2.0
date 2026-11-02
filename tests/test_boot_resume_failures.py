"""Tail-113 pins: tests/test_boot_resume.py census sites (174, 202, 206).

The subject is itself a test module; its three swallows are test
scaffolding by design, held as exclusions:

174 — ``_crashed_mission`` swallows the KeyboardInterrupt raised by
      the patched step executor (E103 + E106 on one statement). The
      swallow IS the simulated kill: the fixture models a process
      dying mid-step, then asserts the exact shape a kill leaves
      behind (row RUNNING, heartbeat stale, step a completed).
202 — ``_BootBase.tearDown``: a runtime.stop() failure is swallowed
      ("teardown must not fail", per the in-code noqa note) so the
      remaining cleanup still runs.
206 — ``tearDown``: a context.close() failure is swallowed so
      tmp.cleanup() still runs — proven below by the temp dir being
      gone after both teardowns fail.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import tests.test_boot_resume as boot
from nomorals.missions import MissionStatus, MissionStore


def test_crashed_mission_swallow_leaves_kill_shape() -> None:
    context, tmp = boot._make_context()
    try:
        store = MissionStore(context.db)
        mission_id, executed = boot._crashed_mission(context, store)
        row = store.get(mission_id)
        assert executed == ["a"]
        assert row.status == MissionStatus.RUNNING
        assert row.state.get("completed_steps") == ["a"]
    finally:
        context.close()
        tmp.cleanup()


class _StopBoom:
    def stop(self) -> None:
        raise RuntimeError("stop boom")


class _CloseBoom:
    def close(self) -> None:
        raise RuntimeError("close boom")


def test_teardown_swallows_and_still_cleans_up() -> None:
    case = boot._BootBase("run")
    case.runtime = _StopBoom()
    case.context = _CloseBoom()
    case.tmp = tempfile.TemporaryDirectory()
    tmp_dir = case.tmp.name
    case.tearDown()  # must not raise despite both failures
    assert not Path(tmp_dir).exists()
