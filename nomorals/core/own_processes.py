"""Own processes — Devon's record of the processes SHE spawned,
and the fenced kill the immune police arrest through (doc 20 P7).

SCOPE LAW (the whole point of this module): the police may
terminate ONLY a process Devon herself spawned and registered
here, identified by her own records. A pid that merely exists on
the host — GoldenSwing, a bridge she adopted but did not spawn,
the owner's own work, anything — is outside the fence and is
refused IN WORDS, untouched. Host-population signals never select
a target (the baseline monitor's standing refusal); a target is
named by registry identity only.

IDENTITY = pid + start token. A pid alone proves nothing: pids
are recycled. At registration the process's start token is read
from the OS (Linux: ``/proc/<pid>/stat`` starttime; Windows: the
creation FILETIME via ``GetProcessTimes``) and stored on the row.
At kill time the token is re-read and must match exactly — a
mismatch means the pid now belongs to a different process than
the one Devon spawned, and the kill refuses. Where no token can
be read (a platform that does not expose one, a process that
vanished mid-look), the identity is UNVERIFIABLE and the kill
refuses: an unprovable identity is not an identity.

The kill itself is confirm-or-report: SIGTERM first (the process
group when the recorded pid leads its own group — sandboxed
children are spawned in their own session; otherwise the single
pid, never Devon's own group), a bounded grace, SIGKILL
escalation, and a liveness re-check. A kill that cannot be
confirmed raises instead of claiming success — the police receipt
machinery turns the raise into a failed effect, in words.

Persistence mirrors the grant registry: one path-backed JSON
store (atomic tmp + fsync + ``os.replace``), one row per spawned
process plus an event journal. Rows for ended processes are
pruned on save (newest 300 kept); running rows are never pruned.

Layer: core (L1), stdlib-only process inspection in the
``social/bridge_process.py`` idiom (total functions, never raise
on a look) — duplicated rather than imported because that module
sits in a higher layer.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .failures import record_failure
from .ids import new_short_id
from .logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "OwnProcessRefusal",
    "OwnProcessRegistry",
    "default_own_process_registry",
    "default_processes_path",
    "mark_spawned_exited",
    "police_kill_fn",
    "process_start_token",
    "register_spawned",
    "reset_shared_process_registries",
    "shared_process_registry",
]

_FORMAT = "core.own_processes.v1"
_MAX_EVENTS = 1000
_MAX_ENDED_ROWS = 300
_TERM_GRACE_S = 2.0
_KILL_GRACE_S = 2.0

_IS_WINDOWS = sys.platform.startswith("win")
_IS_LINUX = sys.platform.startswith("linux")


class OwnProcessRefusal(Exception):
    """The fenced kill refused — the message says why, in words."""


# ── OS process identity (total: failures answer None) ────────────


def process_start_token(pid: int) -> str | None:
    """The process's start token, or None when it cannot be read
    (process gone, platform without one, permission). Two reads of
    the same live process always agree; a recycled pid never
    matches the original's token."""
    try:
        pid = int(pid)
        if _IS_LINUX:
            stat = Path(f"/proc/{pid}/stat").read_text(
                encoding="utf-8", errors="replace")
            # Fields after the comm: index 0 is state (field 3);
            # starttime is field 22 → index 19 of the remainder.
            fields = stat.rsplit(")", 1)[-1].split()
            return str(int(fields[19]))
        if _IS_WINDOWS:
            return _win_start_token(pid)
    except Exception:  # noqa: BLE001 - no evidence, never a crash
        return None
    return None


def _win_start_token(pid: int) -> str | None:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED
    if not handle:
        return None
    try:
        creation = wintypes.FILETIME()
        exit_t = wintypes.FILETIME()
        kernel_t = wintypes.FILETIME()
        user_t = wintypes.FILETIME()
        if not kernel32.GetProcessTimes(
                handle, ctypes.byref(creation), ctypes.byref(exit_t),
                ctypes.byref(kernel_t), ctypes.byref(user_t)):
            return None
        return str((creation.dwHighDateTime << 32)
                   | creation.dwLowDateTime)
    finally:
        kernel32.CloseHandle(handle)


def pid_alive(pid: int) -> bool:
    """Liveness in the bridge idiom: a zombie is dead (it holds
    nothing and can do nothing; its parent just hasn't reaped it)."""
    try:
        pid = int(pid)
        if _IS_WINDOWS:
            return _win_pid_alive(pid)
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists; another owner's process is alive
    except Exception:  # noqa: BLE001
        return False
    if _IS_LINUX:
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(
                encoding="utf-8", errors="replace")
            state = stat.rsplit(")", 1)[-1].split()[0]
            if state in ("Z", "X", "x"):
                return False
        except Exception:  # noqa: BLE001 - the kill() verdict stands
            pass
    return True


def _win_pid_alive(pid: int) -> bool:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    kernel32.CloseHandle(handle)
    return True


def _terminate(pid: int, *, hard: bool) -> str:
    """Signal one process (its group when it leads its own).
    Returns the verb used. Windows has one verb for both."""
    if _IS_WINDOWS:
        _win_terminate(pid)
        return "TerminateProcess"
    import signal

    sig = signal.SIGKILL if hard else signal.SIGTERM
    try:
        if os.getpgid(pid) == pid:
            os.killpg(pid, sig)
            return f"{sig.name} (process group)"
    except (ProcessLookupError, PermissionError):
        pass
    os.kill(pid, sig)
    return sig.name


def _win_terminate(pid: int) -> None:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(0x0001 | 0x1000, False, pid)  # TERMINATE
    if not handle:
        raise OwnProcessRefusal(
            f"could not open process {pid} to terminate it")
    try:
        if not kernel32.TerminateProcess(handle, 1):
            raise OwnProcessRefusal(
                f"TerminateProcess failed for process {pid}")
    finally:
        kernel32.CloseHandle(handle)


def _wait_dead(pid: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not pid_alive(pid):
            return True
        time.sleep(0.05)
    return not pid_alive(pid)


# ── the registry ─────────────────────────────────────────────────


class OwnProcessRegistry:
    """Durable record of the processes Devon spawned, by proc id."""

    def __init__(self, path: Any = None) -> None:
        self.path: Path | None = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._processes: dict[str, dict[str, Any]] = {}
        self._events: list[dict[str, Any]] = []
        if self.path is not None:
            self._load()

    # ── persistence ──

    def _load(self) -> None:
        assert self.path is not None
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as fh:
            payload = json.load(fh)
        self._processes = {
            str(pid_): dict(row)
            for pid_, row in (payload.get("processes") or {}).items()
        }
        self._events = [dict(e) for e in payload.get("events") or []]

    def save(self) -> None:
        if self.path is None:
            return
        with self._lock:
            rows = self._processes
            if len(rows) > _MAX_ENDED_ROWS + 50:
                running = {k: v for k, v in rows.items()
                           if v.get("state") == "running"}
                ended = sorted(
                    (v for v in rows.values()
                     if v.get("state") != "running"),
                    key=lambda v: float(v.get("ended_at")
                                        or v.get("spawned_at") or 0))
                kept = {v["proc_id"]: v
                        for v in ended[-_MAX_ENDED_ROWS:]}
                self._processes = {**kept, **running}
            payload = {
                "format": _FORMAT,
                "saved_ts": time.time(),
                "processes": self._processes,
                "events": self._events[-_MAX_EVENTS:],
            }
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)

    # ── registration ──

    def register(self, pid: int, *, label: str = "",
                 spawned_by: str = "", argv0: str = "") -> dict[str, Any]:
        """Record one just-spawned process, capturing its start
        token NOW (the fencing evidence the kill will re-check)."""
        pid = int(pid)
        token = process_start_token(pid)
        row = {
            "proc_id": new_short_id("proc_"),
            "pid": pid,
            "start_token": token,
            "label": str(label or ""),
            "spawned_by": str(spawned_by or ""),
            "argv0": str(argv0 or ""),
            "spawned_at": round(time.time(), 3),
            "state": "running",
            "ended_at": None,
            "end_note": "",
            "killed_by": "",
        }
        with self._lock:
            self._processes[row["proc_id"]] = row
            self._events.append({
                "ts": time.time(), "kind": "spawned",
                "proc_id": row["proc_id"], "pid": pid,
                "actor": row["spawned_by"],
                "detail": (f"{row['label'] or 'process'} "
                           f"({row['argv0'] or '?'}) spawned"),
            })
            self.save()
        return dict(row)

    def spawn(self, argv: list, *, label: str = "", spawned_by: str = "",
              **popen_kwargs: Any) -> tuple[subprocess.Popen, dict[str, Any]]:
        """Spawn a child AND register it — the one-call seam for
        work Devon starts and may later need to arrest. The child
        leads its own session by default (group-killable without
        ever touching Devon's own group)."""
        popen_kwargs.setdefault("start_new_session", True)
        proc = subprocess.Popen(argv, **popen_kwargs)
        row = self.register(proc.pid, label=label,
                            spawned_by=spawned_by,
                            argv0=str(argv[0]) if argv else "")
        return proc, row

    def mark_exited(self, proc_id: str, note: str = "") -> bool:
        """Record that a registered process ended on its own (the
        spawn seams call this when their child completes)."""
        with self._lock:
            row = self._processes.get(str(proc_id))
            if row is None or row.get("state") != "running":
                return False
            row["state"] = "exited"
            row["ended_at"] = round(time.time(), 3)
            row["end_note"] = str(note or "")
            self.save()
            return True

    # ── reads ──

    def get(self, proc_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._processes.get(str(proc_id))
            return dict(row) if row is not None else None

    def all(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._processes.values()]

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(e) for e in self._events]

    def resolve(self, target: str) -> dict[str, Any] | None:
        """A registry row by proc id, or — for a bare numeric
        target — the newest RUNNING row carrying that pid. A pid
        with no row resolves to None: outside the fence."""
        target = str(target or "")
        with self._lock:
            row = self._processes.get(target)
            if row is not None:
                return dict(row)
            if target.isdigit():
                pid = int(target)
                candidates = [r for r in self._processes.values()
                              if r.get("pid") == pid
                              and r.get("state") == "running"]
                if candidates:
                    return dict(max(
                        candidates,
                        key=lambda r: float(r.get("spawned_at") or 0)))
            return None

    # ── the fenced kill ──

    def kill(self, target: str, *, actor: str = "") -> dict[str, Any]:
        """Terminate a registered own process, fenced by identity.

        Every refusal raises :class:`OwnProcessRefusal` with the
        reason in words and touches NOTHING: unknown target,
        already-ended record, unverifiable identity, recycled pid.
        A kill that is signalled but cannot be confirmed also
        raises — success is claimed only for a confirmed death.
        """
        with self._lock:
            resolved = self.resolve(target)
            if resolved is None:
                raise OwnProcessRefusal(
                    f"no record of process {target!r} — the police "
                    "kill is fenced to processes Devon herself "
                    "spawned and registered; nothing was touched")
            proc_id = resolved["proc_id"]
            # Mutate the STORED row (resolve hands back a copy):
            # the state flips below are the record, not a draft.
            row = self._processes[proc_id]
            pid = int(row["pid"])
            if row.get("state") != "running":
                raise OwnProcessRefusal(
                    f"process {proc_id} is already recorded "
                    f"{row.get('state')} — no kill performed")
            if not pid_alive(pid):
                row["state"] = "exited"
                row["ended_at"] = round(time.time(), 3)
                row["end_note"] = "found already exited at kill time"
                self.save()
                raise OwnProcessRefusal(
                    f"process {proc_id} (pid {pid}) had already "
                    "exited before the kill — no kill performed")
            registered_token = row.get("start_token")
            if not registered_token:
                raise OwnProcessRefusal(
                    f"process {proc_id} (pid {pid}) has no start "
                    "token on record — its identity is unverifiable "
                    "on this platform, and the kill never acts on a "
                    "bare pid; nothing was touched")
            current_token = process_start_token(pid)
            if current_token is None:
                raise OwnProcessRefusal(
                    f"process {proc_id} (pid {pid}): the start "
                    "token cannot be re-read, so its identity is "
                    "unverifiable now; nothing was touched")
            if str(current_token) != str(registered_token):
                raise OwnProcessRefusal(
                    f"pid {pid} has been recycled — its current "
                    "start token does not match the registered "
                    f"process {proc_id}; refusing to kill a "
                    "process Devon did not spawn")
            verb = _terminate(pid, hard=False)
            if not _wait_dead(pid, _TERM_GRACE_S):
                verb = _terminate(pid, hard=True)
                if not _wait_dead(pid, _KILL_GRACE_S):
                    raise OwnProcessRefusal(
                        f"process {proc_id} (pid {pid}) was "
                        "signalled but is still alive — the kill "
                        "is NOT confirmed and is not claimed")
            row["state"] = "killed"
            row["ended_at"] = round(time.time(), 3)
            row["end_note"] = f"terminated via {verb}"
            row["killed_by"] = str(actor or "")
            self._events.append({
                "ts": time.time(), "kind": "killed",
                "proc_id": proc_id, "pid": pid,
                "actor": str(actor or ""),
                "detail": (f"{row.get('label') or 'process'} "
                           f"terminated via {verb}"),
            })
            self.save()
            return {
                "killed": True,
                "proc_id": proc_id,
                "pid": pid,
                "label": row.get("label") or "",
                "signal": verb,
            }


# ── shared instance (one per path, the wiring pattern) ───────────

_SHARED: dict[str, OwnProcessRegistry] = {}
_SHARED_LOCK = threading.Lock()


def default_processes_path() -> Path:
    """``<home>/data/own-processes.json`` — home is ``$NM_HOME``
    when set, else ``~/.nomorals``."""
    home = os.environ.get("NM_HOME", "").strip()
    base = Path(home) if home else Path.home() / ".nomorals"
    return base / "data" / "own-processes.json"


def shared_process_registry(path: Any = None) -> OwnProcessRegistry:
    resolved = Path(path) if path is not None else default_processes_path()
    key = str(resolved)
    with _SHARED_LOCK:
        registry = _SHARED.get(key)
        if registry is None:
            try:
                registry = OwnProcessRegistry(resolved)
            except Exception as exc:  # noqa: BLE001 - degrade, loudly
                record_failure("own_processes.shared_load", exc)
                registry = OwnProcessRegistry()
            _SHARED[key] = registry
        return registry


def default_own_process_registry() -> OwnProcessRegistry:
    return shared_process_registry()


def reset_shared_process_registries() -> None:
    """Drop all shared registries (tests / restart simulation)."""
    with _SHARED_LOCK:
        _SHARED.clear()


# ── spawn-seam helpers (fail-open: recording never breaks a run) ──


def register_spawned(pid: int, *, label: str = "", spawned_by: str = "",
                     argv0: str = "") -> dict[str, Any] | None:
    """Register a just-spawned process on the default registry.
    Returns the row, or None when recording failed — the spawn
    itself is never failed by its record."""
    try:
        return default_own_process_registry().register(
            pid, label=label, spawned_by=spawned_by, argv0=argv0)
    except Exception as exc:  # noqa: BLE001 - fail-open by design
        record_failure("own_processes.register_spawned", exc)
        return None


def mark_spawned_exited(row: dict[str, Any] | None, note: str = "") -> None:
    """Mark a row from :func:`register_spawned` as ended."""
    if not row:
        return
    try:
        default_own_process_registry().mark_exited(
            str(row.get("proc_id") or ""), note=note)
    except Exception as exc:  # noqa: BLE001 - fail-open by design
        record_failure("own_processes.mark_spawned_exited", exc)


def police_kill_fn(registry: OwnProcessRegistry | None = None
                   ) -> Callable[[str], Any]:
    """The police's ``kill_fn``: the fenced own-process kill. The
    registry resolves at call time — building the closure cannot
    fail, so a surface can wire it unconditionally."""

    def _kill(process_id: str) -> dict[str, Any]:
        reg = registry or default_own_process_registry()
        return reg.kill(str(process_id), actor="immune.police")

    return _kill
