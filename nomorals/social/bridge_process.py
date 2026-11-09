"""Process-table inspection for bridges Devon did not spawn.

Ownership of a bridge process is an IDENTITY question, not a
spawn question (orphanfix): a bridge that survives a Devon
restart is still Devon's — but only if it can be identified as
hers.  Identifying it takes two kinds of evidence:

* what the process on the account's port SAYS it is (the bridge
  protocol's status payload — program marker, account JID, port;
  see :mod:`nomorals.social.pairing`), and
* what the OPERATING SYSTEM says runs there: the pid listening
  on the port, that pid's program image and (where the platform
  exposes it) its command line — a bridge is launched with
  ``--port <n> --creds <account dir>``, so the command line is
  itself account-binding evidence.

This module is the OS half.  It is stdlib-only (the repo carries
no hard psutil dependency — see immune/baseline.py for the same
posture): Windows resolves a listening port's owner through
``GetExtendedTcpTable`` and images through
``QueryFullProcessImageNameW`` (ctypes); Linux reads ``/proc``
(the net tables for the socket inode, then the fd scan for its
owner, ``exe``/``cmdline`` for the image).  Every function is
total: any failure — unsupported platform, missing permission,
a process that vanished mid-look — answers None/False, never
raises.  A platform that cannot answer gives the caller NO
process evidence; identity then rests on the protocol payload
alone, and stop authority follows the caller's evidence rule.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

__all__ = [
    "AdoptedProcess",
    "ProcessInspector",
    "default_inspector",
]

_IS_WINDOWS = sys.platform.startswith("win")
_IS_LINUX = sys.platform.startswith("linux")


class ProcessInspector:
    """The OS process table, in the shapes bridge supervision
    needs.  One instance is stateless; tests inject a fake with
    the same five methods to stage processes that are not real.
    """

    # ── port → pid ──────────────────────────────────────────────
    def listener_pid(self, port: int) -> int | None:
        """The pid LISTENING on ``port`` (TCP, either stack), or
        None when nothing listens there or the platform cannot
        say."""
        try:
            if _IS_WINDOWS:
                return _win_listener_pid(int(port))
            if _IS_LINUX:
                return _linux_listener_pid(int(port))
        except Exception:  # noqa: BLE001 - no evidence, never a crash
            return None
        return None

    # ── pid → image ─────────────────────────────────────────────
    def process_image(self, pid: int) -> tuple[str, str] | None:
        """(exe path, command line) for ``pid``, or None.

        The command line is "" where the platform does not
        expose it (Windows: only the image path is read here —
        reading another process's PEB command line is not worth
        the fragility when the image + protocol payload already
        bind the Go bridge, the production engine)."""
        try:
            if _IS_WINDOWS:
                exe = _win_process_image(int(pid))
                return (exe, "") if exe else None
            if _IS_LINUX:
                return _linux_process_image(int(pid))
        except Exception:  # noqa: BLE001
            return None
        return None

    # ── liveness + stop verbs ───────────────────────────────────
    def pid_alive(self, pid: int) -> bool:
        try:
            if _IS_WINDOWS:
                return _win_pid_alive(int(pid))
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists; another owner's process is alive
        except Exception:  # noqa: BLE001
            return False
        # kill(pid, 0) also answers for a ZOMBIE — a process that
        # already exited but whose parent has not reaped it.  A
        # zombie holds no port and no program file, so for every
        # purpose here it is dead; without this check a killed
        # orphan whose parent lingers would read "still running"
        # forever and block the quiesce it was killed for.
        try:
            stat = Path(f"/proc/{int(pid)}/stat").read_text(
                encoding="utf-8", errors="replace")
            state = stat.rsplit(")", 1)[-1].split()[0]
            if state in ("Z", "X", "x"):
                return False
        except Exception:  # noqa: BLE001 - no /proc read, the
            pass              # kill() verdict stands
        return True

    def terminate_pid(self, pid: int) -> bool:
        """The graceful stop: SIGTERM on POSIX.  On Windows the
        only process-stop verb is TerminateProcess — exactly
        what subprocess.Popen.terminate() does there too, so an
        adopted bridge is stopped no harder than a spawned one."""
        try:
            if _IS_WINDOWS:
                return _win_terminate_pid(int(pid))
            import signal

            os.kill(int(pid), signal.SIGTERM)
            return True
        except Exception:  # noqa: BLE001
            return False

    def kill_pid(self, pid: int) -> bool:
        """The escalation: SIGKILL on POSIX, TerminateProcess on
        Windows (the same verb terminate uses there)."""
        try:
            if _IS_WINDOWS:
                return _win_terminate_pid(int(pid))
            import signal

            os.kill(int(pid), signal.SIGKILL)
            return True
        except Exception:  # noqa: BLE001
            return False


class AdoptedProcess:
    """A Popen-shaped handle over a process Devon did not spawn.

    Carries exactly the surface the bridge supervisor uses on a
    spawned process — ``pid``, ``poll()``, ``terminate()``,
    ``kill()`` — backed by a :class:`ProcessInspector` and a pid.
    ``poll()`` returns None while the process lives and -1 once
    it is gone (the real exit code of a process Devon did not
    spawn is not hers to know; every consumer in the supervisor
    tests only None-ness).
    """

    def __init__(self, inspector: ProcessInspector, pid: int) -> None:
        self._inspector = inspector
        self.pid = int(pid)

    def poll(self) -> int | None:
        return None if self._inspector.pid_alive(self.pid) else -1

    def terminate(self) -> None:
        self._inspector.terminate_pid(self.pid)

    def kill(self) -> None:
        self._inspector.kill_pid(self.pid)


_DEFAULT: ProcessInspector | None = None


def default_inspector() -> ProcessInspector:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ProcessInspector()
    return _DEFAULT


# ── Windows (ctypes, stdlib only) ────────────────────────────────

def _win_listener_pid(port: int) -> int | None:
    import ctypes
    from ctypes import wintypes

    iphlpapi = ctypes.WinDLL("iphlpapi")
    # TCP_TABLE_OWNER_PID_LISTENER = 3; AF_INET = 2, AF_INET6 = 23.
    for af in (2, 23):
        pid = _win_listener_pid_af(iphlpapi, ctypes, wintypes, af, port)
        if pid is not None:
            return pid
    return None


def _win_listener_pid_af(iphlpapi: Any, ctypes: Any, wintypes: Any,
                         af: int, port: int) -> int | None:
    size = wintypes.DWORD(0)
    # First call sizes the table (ERROR_INSUFFICIENT_BUFFER=122).
    iphlpapi.GetExtendedTcpTable(None, ctypes.byref(size), False,
                                 af, 3, 0)
    if size.value == 0:
        return None
    buf = ctypes.create_string_buffer(size.value)
    rc = iphlpapi.GetExtendedTcpTable(buf, ctypes.byref(size),
                                       False, af, 3, 0)
    if rc != 0:
        return None
    count = ctypes.cast(buf, ctypes.POINTER(wintypes.DWORD))[0]
    if af == 2:
        # MIB_TCPROW_OWNER_PID: 6 DWORDs, port in the low word in
        # network byte order.
        row_dwords = 6
        port_off, pid_off = 2, 5
    else:
        # MIB_TCP6ROW_OWNER_PID: 16B addr + scope + port + 16B
        # addr + scope + port + state + pid = 13 DWORDs.
        row_dwords = 13
        port_off, pid_off = 5, 12
    words = ctypes.cast(
        buf, ctypes.POINTER(wintypes.DWORD * (1 + count * row_dwords)))
    import socket as _socket

    for i in range(count):
        base = 1 + i * row_dwords
        raw_port = words.contents[base + port_off] & 0xFFFF
        if _socket.ntohs(raw_port) == port:
            return int(words.contents[base + pid_off])
    return None


def _win_open(pid: int, access: int) -> Any:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = kernel32.OpenProcess(access, False, pid)
    return handle or None


def _win_process_image(pid: int) -> str | None:
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = _win_open(pid, PROCESS_QUERY_LIMITED_INFORMATION)
    if not handle:
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        ok = kernel32.QueryFullProcessImageNameW(
            handle, 0, buf, ctypes.byref(size))
        return buf.value if ok else None
    finally:
        kernel32.CloseHandle(handle)


def _win_pid_alive(pid: int) -> bool:
    import ctypes
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    handle = _win_open(pid, PROCESS_QUERY_LIMITED_INFORMATION)
    if not handle:
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    try:
        code = wintypes.DWORD(0)
        if not kernel32.GetExitCodeProcess(handle,
                                           ctypes.byref(code)):
            return False
        return code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _win_terminate_pid(pid: int) -> bool:
    import ctypes

    PROCESS_TERMINATE = 0x0001
    PROCESS_SYNCHRONIZE = 0x00100000
    handle = _win_open(pid, PROCESS_TERMINATE | PROCESS_SYNCHRONIZE)
    if not handle:
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    try:
        return bool(kernel32.TerminateProcess(handle, 1))
    finally:
        kernel32.CloseHandle(handle)


# ── Linux (/proc) ────────────────────────────────────────────────

def _linux_listener_pid(port: int) -> int | None:
    inodes: set[str] = set()
    hexport = f"{port:04X}"
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(table, encoding="ascii") as fh:
                lines = fh.read().splitlines()
        except OSError:
            continue
        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 10:
                continue
            local = parts[1]
            state = parts[3]
            if state != "0A":  # LISTEN
                continue
            if local.rsplit(":", 1)[-1].upper() == hexport:
                inodes.add(parts[9])
    if not inodes:
        return None
    # One fd scan: socket inode → owning pid.
    try:
        pids = [int(n) for n in os.listdir("/proc") if n.isdigit()]
    except OSError:
        return None
    for pid in pids:
        fd_dir = f"/proc/{pid}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(os.path.join(fd_dir, fd))
            except OSError:
                continue
            if target.startswith("socket:[") \
                    and target[8:-1] in inodes:
                return pid
    return None


def _linux_process_image(pid: int) -> tuple[str, str] | None:
    try:
        exe = os.readlink(f"/proc/{pid}/exe")
    except OSError:
        exe = ""
    cmdline = ""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            raw = fh.read()
        cmdline = raw.replace(b"\0", b" ").decode(
            "utf-8", "replace").strip()
    except OSError:
        pass
    if not exe and not cmdline:
        return None
    return exe, cmdline
