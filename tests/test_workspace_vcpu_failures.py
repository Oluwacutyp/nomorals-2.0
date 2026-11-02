"""Tail-110 pins for nomorals/workspace/vcpu.py — 4 exclusions, 0 converts.

Three of the four sites are ``except BaseException`` (E102) BY DESIGN:
a virtual CPU's worker machinery must contain even KeyboardInterrupt /
SystemExit raised inside task code, or one bad task kills the core for
every other task. None of them swallows: each records (stats counters,
_last_error) and delivers the exception to the task's own Future, which
is the module's recording substrate — the pins read both back. The
fourth site is the error logger's own guard, whose non-recording is
structural (the recorder of last resort has nowhere else to report).

Sites:
* 387 E102 — worker loop: a crash in _execute itself (the task's
  future already carries task-level exceptions, so reaching here
  means the worker machinery broke) increments stats["restarts"],
  records _last_error, restarts the core, and gives up to ERROR
  status only after MAX_RESTARTS. Pinned: one forced crash is
  recorded and the core recovers for the next task.
* 439 E102 — _execute_inline: a task raising (Exception OR
  BaseException) has the exception set on its own Future, counts
  stats["tasks_failed"], and is logged via _log_task_error; the
  core keeps serving later tasks.
* 463 E102 — _execute_with_timeout helper thread: the same capture,
  staged through box["error"] and delivered by the supervisor path
  to the Future + stats + log (delivery half read at lines 488-497).
* 532 E103 — _log_task_error: if the logging setup itself raises,
  the helper swallows it — it is called only from error paths and
  must never raise into them. The healthy path is pinned as control.
"""

from __future__ import annotations

import time
from concurrent.futures import Future
from unittest import mock

import pytest

from nomorals.workspace.vcpu import (
    VirtualCPU,
    _WorkItem,
    _log_task_error,
)


@pytest.fixture()
def cpu():
    c = VirtualCPU(0, name="pin").start()
    yield c
    c.stop()


def test_control_task_result_and_run_count(cpu):
    fut = cpu.submit(lambda: 42)
    assert fut.result(timeout=5) == 42
    assert cpu.stats["tasks_run"] == 1


def test_439_task_exception_delivered_and_counted(cpu):
    def boom():
        raise ValueError("task broke")

    fut = cpu.submit(boom)
    with pytest.raises(ValueError):
        fut.result(timeout=5)
    assert cpu.stats["tasks_failed"] == 1


def test_439_baseexception_delivered_and_core_survives(cpu):
    def kiboom():
        raise KeyboardInterrupt()

    fut = cpu.submit(kiboom)
    with pytest.raises(KeyboardInterrupt):
        fut.result(timeout=5)
    assert cpu.stats["tasks_failed"] == 1
    assert cpu.submit(lambda: "alive").result(timeout=5) == "alive"


def test_463_timeout_path_exception_delivered_and_counted(cpu):
    def tboom():
        raise RuntimeError("timed broke")

    fut = cpu.submit(tboom, timeout=5.0)
    with pytest.raises(RuntimeError):
        fut.result(timeout=10)
    assert cpu.stats["tasks_failed"] == 1


def test_387_worker_crash_recorded_and_core_recovers():
    c = VirtualCPU(1, name="pin387").start()
    try:
        original = VirtualCPU._execute

        def broken(self, item):
            raise RuntimeError("worker fault")

        VirtualCPU._execute = broken
        c.submit(lambda: None)  # future never resolves; not awaited
        deadline = time.time() + 5
        while time.time() < deadline and c.stats["restarts"] < 1:
            time.sleep(0.05)
        assert c.stats["restarts"] >= 1
        assert c._last_error.startswith("RuntimeError")
        VirtualCPU._execute = original
        assert c.submit(lambda: "recovered").result(timeout=5) == "recovered"
    finally:
        VirtualCPU._execute = original
        c.stop()


def _item() -> _WorkItem:
    return _WorkItem(seq=1, priority=0, fn=lambda: None, args=(),
                     kwargs={}, future=Future(), name="t", timeout=0.0)


def test_532_logging_failure_swallowed():
    with mock.patch("nomorals.core.logging_setup.get_logger",
                    side_effect=RuntimeError("logging down")):
        _log_task_error("pin", _item(), ValueError("x"))  # must not raise


def test_532_control_healthy_logging_does_not_raise():
    _log_task_error("pin", _item(), ValueError("x"))
