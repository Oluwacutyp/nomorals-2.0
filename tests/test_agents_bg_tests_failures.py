"""Tail-98 pins — nomorals/agents/bg_tests.py silent-failure sites.

115 E103 — ``_drain()``: an OSError/ValueError from a closing child
stream stops the drain; the tallies keep whatever was already fed.
Progress reporting degrades, it never invents counts. EXCLUSION.

177 E104 — ``_reap()``: the final ``communicate()`` mop-up crashing
leaves ``rest`` as "" — the run still records its real exit code and
marks done, and output tallied incrementally by ``_drain()`` during
the run is kept. Under-reporting at worst, never fabrication: the
summary-regex cross-check in ``result()`` only sees output that was
actually read. EXCLUSION.

251 E103 — ``wait()``: a crashing ``on_progress`` callback is
swallowed ("progress never sinks") — the display hook's failure must
not kill the wait loop or lose the run's result. EXCLUSION.
"""
import codecs
import sys
import time

import nomorals.agents.bg_tests as B


# ── 115: _drain stream crash -> no raise, tallies untouched ───────

class _BoomStream:
    def fileno(self):
        raise OSError("stream closed")


def test_115_drain_stream_crash_is_swallowed(tmp_path):
    run = B.BackgroundTestRun([sys.executable, "-c", "pass"],
                              cwd=tmp_path)
    try:
        run._proc.stdout = _BoomStream()
        run._drain()  # must not raise
        assert run._passed == 0 and run._failed == 0
    finally:
        run._proc.kill()
        run._proc.wait()


# ── 177: _reap communicate crash -> done with real exit code ──────

class _FakeProc:
    def __init__(self, boom):
        self._boom = boom

    def poll(self):
        return 0

    def communicate(self, timeout=None):
        if self._boom:
            raise RuntimeError("communicate boom")
        return ("PASSED test_x\n", None)


def _bare_run(boom):
    run = B.BackgroundTestRun.__new__(B.BackgroundTestRun)
    run._proc = _FakeProc(boom)
    run._done = False
    run._exit_code = None
    run._output = []
    run._passed = run._failed = run._errors = run._skipped = 0
    run._pending = ""
    run._decoder = codecs.getincrementaldecoder("utf-8")(
        errors="replace")
    run._t0 = time.perf_counter()
    run._timed_out = False
    run._killed = False
    return run


def test_177_reap_communicate_crash_still_reaps():
    run = _bare_run(boom=True)
    run._reap()  # must not raise
    assert run._done is True
    assert run._exit_code == 0
    # control: a healthy communicate feeds the final output chunk
    run2 = _bare_run(boom=False)
    run2._reap()
    assert run2._done is True and run2._passed == 1


# ── 251: on_progress crash -> wait completes with the result ──────

def test_251_progress_callback_crash_does_not_sink_wait(tmp_path):
    run = B.BackgroundTestRun(
        [sys.executable, "-c",
         "print('PASSED test_a'); print('1 passed in 0.01s')"],
        cwd=tmp_path)

    def _boom_cb(status):
        raise RuntimeError("display boom")

    result = run.wait(poll_interval=0.05, on_progress=_boom_cb)
    assert result["ok"] is True
    assert result["passed"] == 1
