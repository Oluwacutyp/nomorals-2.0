"""Process isolation for forcible termination.

Thread timeouts are cooperative only: ``Future.cancel()`` is a no-op
once the callable is running, so a hung tool leaves an UNKNOWN
outcome. Where forcible termination is appropriate (isolated
processes for dangerous / long-running tools), run the callable in
a CHILD PROCESS and ``SIGKILL`` it on timeout — a dead process has a
KNOWN outcome (dead), never unknown.

Mechanism: ``os.fork()`` + a result pipe. Fork inherits memory, so
the callable may be a bound method or closure — nothing is pickled
except the RESULT (written back through the pipe). The parent waits
with a deadline; on expiry it ``SIGKILL``\\ s the child and reaps it.
Child mutations never leak into the parent (separate address space).

POSIX-only (fork). On platforms without fork the call reports
``not-supported`` and the caller falls back to the cooperative
thread path. Never raises.
"""
from __future__ import annotations

import os
import pickle
import signal
import time
from dataclasses import dataclass
from typing import Any, Callable

__all__ = ["IsolatedResult", "run_isolated", "fork_supported"]


def fork_supported() -> bool:
    return hasattr(os, "fork") and hasattr(os, "kill")


@dataclass
class IsolatedResult:
    """Outcome of :func:`run_isolated`.

    ``status``: ``"ok"`` (value holds the return), ``"error"``
    (the callable raised — ``error`` holds the repr), ``"crashed"``
    (child died abnormally — ``error`` holds the exit signal/code),
    ``"timeout-killed"`` (child was SIGKILLed at the deadline),
    ``"not-supported"`` (no fork on this platform).
    """

    status: str
    value: Any = None
    error: str = ""
    elapsed_s: float = 0.0


def run_isolated(fn: Callable[..., Any], *args: Any,
                 timeout_s: float = 60.0, **kwargs: Any) -> IsolatedResult:
    """Run ``fn(*args, **kwargs)`` in a forked child process.

    The child writes its pickled ``(ok, payload)`` through a pipe and
    ``os._exit``\\ s. The parent reads with a deadline: on timeout the
    child is SIGKILLed and reaped (``timeout-killed``); a child that
    dies on its own reports ``crashed`` with its exit status. An
    unpicklable result is reported as ``error`` (the child already
    exited cleanly — the CALL completed, only the return trip failed).
    """
    started = time.time()
    if not fork_supported():
        return IsolatedResult(status="not-supported",
                              error="os.fork unavailable on this platform",
                              elapsed_s=0.0)
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        # ── child ──────────────────────────────────────────────
        try:
            os.close(read_fd)
            try:
                result = fn(*args, **kwargs)
                payload = pickle.dumps((True, result))
            except Exception as exc:  # noqa: BLE001 - report the raise
                try:
                    payload = pickle.dumps((False, repr(exc)))
                except Exception:  # noqa: BLE001 - unpicklable repr?!
                    payload = pickle.dumps((False, "call raised"))
            try:
                os.write(write_fd, payload)
            except OSError:  # noqa: BLE001 - parent gone; nothing to do
                pass
            os.close(write_fd)
        finally:
            # Never return into the parent's code: _exit skips atexit
            # handlers and stdio flushes that could double-run.
            os._exit(0)
    # ── parent ─────────────────────────────────────────────────
    os.close(write_fd)
    deadline = started + max(0.05, float(timeout_s))
    chunks: list[bytes] = []
    killed = False
    wait_status: int | None = None
    try:
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                # Deadline: SIGKILL + reap. The outcome is KNOWN
                # (dead), never unknown.
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                killed = True
                break
            import select

            r, _, _ = select.select([read_fd], [], [], min(remaining, 0.2))
            if not r:
                # No data yet: is the child already dead?
                done_pid, reaped = os.waitpid(pid, os.WNOHANG)
                if done_pid == pid:
                    wait_status = reaped
                    break
                continue
            try:
                chunk = os.read(read_fd, 65536)
            except OSError:
                chunk = b""
            if not chunk:
                break
            chunks.append(chunk)
            # The child wrote its whole payload then exited; keep
            # draining until EOF so a large result is not truncated.
    finally:
        os.close(read_fd)
        # Always reap (including the killed child) — never a zombie.
        if wait_status is None:
            try:
                _, wait_status = os.waitpid(pid, 0)
            except ChildProcessError:
                wait_status = 0
    elapsed = time.time() - started
    if killed:
        return IsolatedResult(status="timeout-killed",
                              error=f"process killed after {timeout_s}s "
                                    f"timeout",
                              elapsed_s=elapsed)
    if chunks:
        try:
            ok, payload = pickle.loads(b"".join(chunks))
        except Exception as exc:  # noqa: BLE001 - corrupt result pipe
            return IsolatedResult(status="crashed",
                                  error=f"result pipe unreadable: {exc}",
                                  elapsed_s=elapsed)
        if ok:
            return IsolatedResult(status="ok", value=payload,
                                  elapsed_s=elapsed)
        return IsolatedResult(status="error", error=str(payload),
                              elapsed_s=elapsed)
    # No payload: the child died without reporting.
    if os.WIFEXITED(wait_status):
        code = os.WEXITSTATUS(wait_status)
        if code == 0:
            return IsolatedResult(status="error",
                                  error="child exited without a result",
                                  elapsed_s=elapsed)
        return IsolatedResult(status="crashed",
                              error=f"child exited with code {code}",
                              elapsed_s=elapsed)
    if os.WIFSIGNALED(wait_status):
        return IsolatedResult(status="crashed",
                              error=f"child killed by signal "
                                    f"{os.WTERMSIG(wait_status)}",
                              elapsed_s=elapsed)
    return IsolatedResult(status="crashed",
                          error="child died without a result",
                          elapsed_s=elapsed)
