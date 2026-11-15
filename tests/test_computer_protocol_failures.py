"""Pass #217 pins: computer/protocol.py — 1 exclusion, 0 converts.

- 101: ``DriverClient.close`` shutdown write to a dead process.
  ``close()`` is a best-effort shutdown: it first asks the driver
  politely (a ``shutdown`` JSON line on stdin), then waits, then
  terminates. When the stdin write raises BrokenPipeError/OSError
  the process is already gone — which is the goal of the close —
  and the terminate path below agrees (poll shows it dead, wait
  returns at once). The swallow is the correct behavior: the
  process being gone is the outcome close() wants, and the
  function's contract already proceeds to "closed" on a
  non-error path.

computer/protocol.py DOES record failures elsewhere (the outer
``close`` except records "protocol close"), so NOT-recording is
pinned against the computer ledger: computer recent_failures()
before/after each trigger must be unchanged.

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Triggers use a fake Popen whose stdin
write raises, never a swallow.
"""

from nomorals.computer import failures as comp_fail
from nomorals.computer.protocol import DriverClient, DriverDied


def _snap():
    return list(comp_fail.recent_failures(limit=200))


class _FakeStdin:
    def __init__(self):
        self.writes = []

    def write(self, data):
        raise BrokenPipeError("driver already gone")

    def flush(self):
        pass


class _FakeProc:
    """A driver that is alive at poll() time but whose stdin is
    already dead — the exact race the :101 handler covers."""

    def __init__(self):
        self.stdin = _FakeStdin()
        self.terminated = False
        self.killed = False

    def poll(self):
        return None  # alive when close() looks

    def wait(self, timeout=None):
        return 0  # ...but already reaped when close() waits

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def _client_with_dead_stdin():
    client = DriverClient(["/bin/true"])
    client._proc = _FakeProc()
    return client


# ── L101: the shutdown write to a dead process is not a failure ────

def test_101_close_on_dead_stdin_does_not_raise():
    client = _client_with_dead_stdin()
    before = _snap()
    client.close()  # must not raise
    assert client._proc is None
    assert isinstance(client._dead, DriverDied)
    assert _snap() == before


def test_101_close_on_dead_stdin_needs_no_terminate():
    # The process was already gone: close() never reaches the
    # terminate/kill rungs.
    client = _client_with_dead_stdin()
    proc = client._proc
    client.close()
    assert proc.terminated is False
    assert proc.killed is False


def test_101_close_on_healthy_stdin_writes_shutdown():
    # The control: when the stdin write works, close() sends the
    # polite shutdown line and waits for the exit.
    sent = []

    class _GoodStdin:
        def write(self, data):
            sent.append(data)

        def flush(self):
            pass

    class _GoodProc(_FakeProc):
        def __init__(self):
            super().__init__()
            self.stdin = _GoodStdin()

    client = DriverClient(["/bin/true"])
    client._proc = _GoodProc()
    client.close()
    assert sent and b"shutdown" in sent[0]
    assert isinstance(client._dead, DriverDied)
