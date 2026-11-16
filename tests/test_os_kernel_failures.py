"""Tail-80 failure pins — nomorals/os/kernel.py.

All six sites are EXCLUSIONS.  They are the shutdown path's
teardown guards: four step guards inside ``stop()`` (102 the
shutdown hooks, 106 ``services.stop_all``, 110
``health.stop_background``, 116 ``bus.stop``), the
``arm_signals`` platform guard (166), and the ``_atexit_stop``
guard (175).  ``stop()`` returns None and makes no success
claim — the started/ready flags are cleared BEFORE teardown
runs, so ``status()`` never reports "running" after a failed
teardown — and OSKernel has no production importer outside
``nomorals/os/`` (it is exercised by test_os_core /
test_os_sweep).  Each guard exists so the remaining teardown
steps still run; the pins assert exactly that continuation
behavior.  Probed live first (nm-tail80-verify/probes.py).
"""
import threading

import pytest

from nomorals.core.events import EventBus
from nomorals.os.kernel import OSKernel


def _started_kernel():
    kernel = OSKernel(bus=EventBus())
    kernel.start()
    return kernel


def _boom(*a, **k):
    raise RuntimeError("teardown boom")


# ── 102: a raising shutdown hook does not stop teardown ──
def test_102_raising_hook_swallowed_and_teardown_continues():
    kernel = _started_kernel()
    ran = []
    kernel.on_shutdown(lambda: ran.append("recorder"), name="recorder")
    kernel.on_shutdown(_boom, name="bad")  # LIFO: bad runs first
    kernel.stop()  # must not raise
    assert ran == ["recorder"]
    assert kernel.started is False


# ── 106: services.stop_all failure does not skip later steps ──
def test_106_stop_all_failure_still_stops_health_and_bus():
    kernel = _started_kernel()
    calls = []
    orig_health_stop = kernel.health.stop_background
    orig_bus_stop = kernel.bus.stop
    kernel.services.stop_all = _boom
    kernel.health.stop_background = lambda: (
        calls.append("health"), orig_health_stop())[0]
    kernel.bus.stop = lambda: (calls.append("bus"), orig_bus_stop())[0]
    kernel.stop()  # must not raise
    assert calls == ["health", "bus"]


# ── 110: health.stop_background failure does not skip the bus ──
def test_110_health_failure_still_drains_bus():
    kernel = _started_kernel()
    calls = []
    orig_bus_stop = kernel.bus.stop
    kernel.health.stop_background = _boom
    kernel.bus.stop = lambda: (calls.append("bus"), orig_bus_stop())[0]
    kernel.stop()  # must not raise
    assert calls == ["bus"]


# ── 116: bus.stop failure does not escape stop() ──
def test_116_bus_failure_swallowed_by_stop():
    kernel = _started_kernel()
    kernel.bus.stop = _boom
    assert kernel.stop() is None
    assert kernel.started is False


# ── 166: signal arming off the main thread is absorbed ──
def test_166_arm_signals_off_main_thread_absorbed():
    kernel = OSKernel(bus=EventBus())
    box = {}

    def arm():
        try:
            box["ret"] = kernel.arm_signals()
        except Exception as exc:  # noqa: BLE001 - captured for assertion
            box["err"] = exc

    t = threading.Thread(target=arm)
    t.start()
    t.join()
    assert "err" not in box
    assert box["ret"] is kernel
    assert kernel._signals_armed is True


# ── 175: _atexit_stop never propagates ──
def test_175_atexit_stop_never_raises():
    kernel = _started_kernel()
    real_stop = kernel.stop
    kernel.stop = _boom
    try:
        kernel._atexit_stop()  # must not raise
    finally:
        kernel.stop = real_stop
        real_stop()
