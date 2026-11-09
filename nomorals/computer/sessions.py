"""Desktop sessions: persistent desktops with a journaled life.

A session has an identity, a display, and a lifetime independent
of any one task: tasks attach, drive, and detach; the desktop and
its open applications survive between them. Session state lives in
a JSON store under ``$NM_HOME/computer/`` (sessions.json + an
append-only journal.jsonl + captured frames), so a Devon restart
re-reads the truth: on first touch each open session is reconciled
against reality — the driver is respawned and pinged, the virtual
display's liveness is checked — and a session whose display is
gone is marked *lost*, honestly, never silently revived.

The control gate lives here too, and it is the personal-surface
law in code:

1. **Owner stop is sticky.** ``owner_stop`` persists in the store;
   no reconcile, restart, or reconcile-pass clears it. Only an
   explicit owner resume does.
2. **Owner takeover pauses the agent.** While ``takeover`` is set,
   actions are refused; perception (watching) stays available.
   Handback clears the takeover — never the stop.
3. **A degraded brain is observe-only.** The gate asks an injected
   brain probe (the wiring layers pass llm.power's state); when
   the brain is not fully healthy, actions are refused and only
   perception flows.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core import failure_store as _failure_store
from . import failures, provision
from .protocol import DriverClient, DriverError

#: Frames kept per session; the watch surface reads the newest.
MAX_FRAMES_PER_SESSION = 60

#: Virtual desktop size for sessions Devon creates herself.
VIRTUAL_WIDTH = 1280
VIRTUAL_HEIGHT = 1024


class SessionError(Exception):
    """A session cannot be opened, found, or driven; message says why."""


class GateRefused(SessionError):
    """The control gate refused an action; ``reasons`` says why."""

    def __init__(self, reasons: list[str]) -> None:
        self.reasons = reasons
        super().__init__("; ".join(reasons))


@dataclass
class SessionRecord:
    id: str
    mode: str  # "virtual" | "host" | "windows"
    display: str
    state: str = "open"  # "open" | "closed" | "lost"
    created: float = 0.0
    last_activity: float = 0.0
    mission: str = ""
    width: int = 0
    height: int = 0
    xvfb_pid: int = 0
    auth_path: str = ""
    owner_stop: bool = False
    takeover: bool = False
    takeover_by: str = ""
    note: str = ""

    def describe(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SessionManager:
    """Owns session records, driver clients, and the control gate."""

    brain_probe: Callable[[], bool] | None = None
    home: Path | None = None
    _clients: dict[str, DriverClient] = field(
        default_factory=dict, repr=False, compare=False)
    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False)
    _xvfb_procs: dict[str, Any] = field(
        default_factory=dict, repr=False, compare=False)

    # -- storage ---------------------------------------------------------

    @property
    def root(self) -> Path:
        base = self.home or _failure_store.home_dir()
        root = Path(base) / "computer"
        root.mkdir(parents=True, exist_ok=True)
        return root

    @property
    def _store_path(self) -> Path:
        return self.root / "sessions.json"

    def _load(self) -> dict[str, SessionRecord]:
        try:
            doc = json.loads(self._store_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        out: dict[str, SessionRecord] = {}
        for sid, row in (doc.get("sessions") or {}).items():
            if isinstance(row, dict):
                try:
                    out[sid] = SessionRecord(**{
                        k: row[k] for k in SessionRecord.__dataclass_fields__
                        if k in row})
                except (TypeError, ValueError) as exc:
                    failures.record_failure("sessions load row", exc)
        return out

    def _save(self, records: dict[str, SessionRecord]) -> None:
        doc = {"sessions": {sid: asdict(rec) for sid, rec in records.items()}}
        tmp = self._store_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        os.replace(tmp, self._store_path)

    def journal(self, event: str, session_id: str = "",
                **detail: Any) -> None:
        """Append one journal line. Never raises."""
        row = {"ts": time.time(), "session": session_id, "event": event}
        for key, value in detail.items():
            text = value if isinstance(value, (int, float, bool)) \
                else str(value)
            row[key] = text[:300] if isinstance(text, str) else text
        try:
            with open(self.root / "journal.jsonl", "a",
                      encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except OSError as exc:
            failures.record_failure("sessions journal", exc)

    def journal_tail(self, limit: int = 20,
                     session_id: str = "") -> list[dict[str, Any]]:
        try:
            lines = (self.root / "journal.jsonl").read_text(
                encoding="utf-8").splitlines()
        except OSError:
            return []
        out: list[dict[str, Any]] = []
        for line in reversed(lines):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if session_id and row.get("session") != session_id:
                continue
            out.append(row)
            if len(out) >= limit:
                break
        return out

    # -- records ----------------------------------------------------------

    def sessions(self) -> list[SessionRecord]:
        with self._lock:
            records = self._load()
        return sorted(records.values(), key=lambda r: r.created)

    def get(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            return self._load().get(session_id)

    def resolve(self, session_id: str = "") -> SessionRecord | None:
        """The named session, or the most recently active open one."""
        records = self.sessions()
        if session_id:
            for rec in records:
                if rec.id == session_id:
                    return rec
            return None
        open_recs = [r for r in records if r.state == "open"]
        if not open_recs:
            return records[-1] if records else None
        return max(open_recs, key=lambda r: r.last_activity or r.created)

    def _mutate(self, session_id: str, **changes: Any) -> SessionRecord:
        with self._lock:
            records = self._load()
            rec = records.get(session_id)
            if rec is None:
                raise SessionError(f"no session {session_id!r}")
            for key, value in changes.items():
                setattr(rec, key, value)
            self._save(records)
            return rec

    def _touch(self, rec: SessionRecord) -> None:
        with suppress(SessionError):
            self._mutate(rec.id, last_activity=time.time())

    # -- lifecycle ---------------------------------------------------------

    def open_session(self, mission: str = "") -> SessionRecord:
        """Open a new desktop session on this host.

        Linux with a display in the environment attaches to that
        display (mode "host" — Devon never kills a server she did
        not start). Linux without one gets a virtual display she
        provisions and owns (mode "virtual"). Windows attaches to
        the console desktop (mode "windows").
        """
        sid = "desk-" + uuid.uuid4().hex[:10]
        now = time.time()
        rec = SessionRecord(id=sid, mode="", display="",
                            created=now, last_activity=now,
                            mission=mission)
        xvfb: provision.XvfbProvision | None = None
        try:
            bridge = provision.ensure_bridge()
            if sys.platform.startswith("win"):
                rec.mode = "windows"
                rec.display = "console"
                rec.note = "the host's console desktop (shared)"
            elif os.environ.get("NM_COMPUTER_DISPLAY") or \
                    os.environ.get("DISPLAY"):
                rec.mode = "host"
                rec.display = (os.environ.get("NM_COMPUTER_DISPLAY")
                               or os.environ["DISPLAY"])
            else:
                rec.mode = "virtual"
                number = provision.find_free_display()
                auth = self.root / f"xauth-{sid}"
                provision.write_xauthority(auth, number)
                xvfb = provision.start_xvfb(
                    number, VIRTUAL_WIDTH, VIRTUAL_HEIGHT, auth)
                rec.display = xvfb.display
                rec.xvfb_pid = xvfb.pid
                rec.auth_path = str(auth)
                self._xvfb_procs[sid] = xvfb.proc
            client = self._spawn_client(rec, bridge.path)
            probe = client.probe()
            if not probe.get("available"):
                client.close()
                raise SessionError(
                    "the desktop driver reports itself unavailable: "
                    + str(probe.get("reason") or "no reason given"))
            rec.width = int(probe.get("width") or 0)
            rec.height = int(probe.get("height") or 0)
            self._clients[sid] = client
        except Exception as exc:
            if xvfb is not None:
                provision.stop_process(xvfb.pid)
            if isinstance(exc, SessionError):
                raise
            failures.record_failure("sessions open", exc)
            raise SessionError(f"the session would not open: {exc}") from exc
        with self._lock:
            records = self._load()
            records[sid] = rec
            self._save(records)
        self.journal("session_opened", sid, mode=rec.mode,
                     display=rec.display, mission=mission,
                     width=rec.width, height=rec.height)
        return rec

    def _spawn_client(self, rec: SessionRecord,
                      bridge_path: Path | None = None) -> DriverClient:
        path = bridge_path or provision.ensure_bridge().path
        env = dict(os.environ)
        if rec.mode in ("virtual", "host"):
            env["DISPLAY"] = rec.display
        if rec.mode == "virtual" and rec.auth_path:
            env["XAUTHORITY"] = rec.auth_path
        client = DriverClient([str(path)], env=env,
                              name=f"desktop-bridge[{rec.id}]")
        client.start()
        return client

    def reconcile(self, rec: SessionRecord) -> SessionRecord:
        """Re-attach reality to an open record, or mark it lost.

        A record that says "open" is a claim; this checks it: the
        virtual display must still live, and a fresh driver must
        answer a ping. Anything else is *lost* — journaled and
        surfaced, never papered over.
        """
        stored = self.get(rec.id)
        if stored is not None:
            rec = stored
        if rec.state != "open":
            return rec
        if rec.mode == "virtual":
            socket_gone = not provision.display_socket_path(
                _display_number(rec.display)).exists()
            if not provision.process_alive(rec.xvfb_pid) or socket_gone:
                self._drop_client(rec.id)
                lost = self._mutate(
                    rec.id, state="lost",
                    note="the virtual display is gone (the Xvfb "
                         "process Devon started has exited)")
                self.journal("session_lost", rec.id, reason=lost.note)
                return lost
        client = self._clients.get(rec.id)
        fresh = client is None
        if fresh:
            try:
                client = self._spawn_client(rec)
                self._clients[rec.id] = client
            except Exception as exc:  # noqa: BLE001 — reported as lost
                failures.record_failure("sessions reconcile spawn", exc)
                lost = self._mutate(
                    rec.id, state="lost",
                    note=f"the desktop driver would not start: {exc}")
                self.journal("session_lost", rec.id, reason=lost.note)
                return lost
        try:
            if client.ping():
                if fresh:
                    self.journal("session_reattached", rec.id,
                                 display=rec.display)
                return rec
            raise DriverError("ping returned no pong")
        except Exception:  # noqa: BLE001 — one respawn, then honest
            self._drop_client(rec.id)
            try:
                client = self._spawn_client(rec)
                self._clients[rec.id] = client
                if client.ping():
                    self.journal("session_reattached", rec.id,
                                 display=rec.display,
                                 after="driver restart")
                    return rec
            except Exception as exc2:  # noqa: BLE001
                exc = exc2
            failures.record_failure("sessions reconcile ping", exc)
            lost = self._mutate(
                rec.id, state="lost",
                note=f"the display no longer answers: {exc}")
            self.journal("session_lost", rec.id, reason=lost.note)
            return lost

    def _drop_client(self, session_id: str) -> None:
        client = self._clients.pop(session_id, None)
        if client is not None:
            client.close()

    def client_for(self, rec: SessionRecord) -> DriverClient:
        rec = self.reconcile(rec)
        if rec.state != "open":
            raise SessionError(
                f"session {rec.id} is {rec.state}"
                + (f" — {rec.note}" if rec.note else ""))
        client = self._clients.get(rec.id)
        if client is None:  # reconcile spawned it; be safe anyway
            client = self._spawn_client(rec)
            self._clients[rec.id] = client
        return client

    def close_session(self, session_id: str) -> SessionRecord:
        rec = self.get(session_id)
        if rec is None:
            raise SessionError(f"no session {session_id!r}")
        self._drop_client(session_id)
        if rec.mode == "virtual" and rec.xvfb_pid:
            provision.stop_process(rec.xvfb_pid)
            self._xvfb_procs.pop(session_id, None)
        closed = self._mutate(session_id, state="closed",
                              note="closed by request")
        self.journal("session_closed", session_id)
        return closed

    # -- the control gate ---------------------------------------------------

    def set_owner_stop(self, session_id: str, on: bool,
                       by: str = "owner") -> SessionRecord:
        rec = self._mutate(session_id, owner_stop=bool(on))
        self.journal("owner_stop_set" if on else "owner_stop_cleared",
                     session_id, by=by)
        return rec

    def set_takeover(self, session_id: str, on: bool,
                     by: str = "owner") -> SessionRecord:
        rec = self._mutate(session_id, takeover=bool(on),
                           takeover_by=by if on else "")
        self.journal("takeover_started" if on else "takeover_handed_back",
                     session_id, by=by)
        return rec

    def gate_reasons(self, rec: SessionRecord) -> list[str]:
        """Why actions are refused on this session (empty = allowed).

        The flags are read from the STORE, never from the caller's
        possibly-stale in-memory copy: a stop or takeover recorded
        by any surface binds the very next gate check, whoever
        holds which record object.
        """
        stored = self.get(rec.id)
        if stored is not None:
            rec = stored
        reasons: list[str] = []
        if rec.state != "open":
            reasons.append(f"the session is {rec.state}")
        if rec.owner_stop:
            reasons.append(
                "the owner stopped this computer — the stop is sticky "
                "and only the owner's resume clears it")
        if rec.takeover:
            reasons.append(
                "the owner has taken over this desktop"
                + (f" ({rec.takeover_by})" if rec.takeover_by else "")
                + " — the agent's hands are off until handback")
        if self.brain_probe is not None:
            try:
                healthy = bool(self.brain_probe())
            except Exception as exc:  # noqa: BLE001 — probe failure = not healthy
                failures.record_failure("sessions brain_probe", exc)
                healthy = False
            if not healthy:
                reasons.append(
                    "the brain is degraded or offline — observe-only: "
                    "watching is allowed, acting is not")
        return reasons

    def can_act(self, rec: SessionRecord) -> bool:
        return not self.gate_reasons(rec)

    # -- driving ---------------------------------------------------------

    def act(self, rec: SessionRecord, method: str,
            params: dict[str, Any] | None = None) -> dict[str, Any]:
        """One gated driver action, journaled with its outcome."""
        rec = self.reconcile(rec)
        reasons = self.gate_reasons(rec)
        if reasons:
            self.journal("action_refused", rec.id, method=method,
                         reasons="; ".join(reasons))
            raise GateRefused(reasons)
        client = self.client_for(rec)
        try:
            result = client.call(method, params or {})
        except Exception as exc:
            self.journal("action_failed", rec.id, method=method,
                         error=f"{type(exc).__name__}: {exc}")
            failures.record_failure(f"sessions act {method}", exc)
            raise
        self._touch(rec)
        self.journal("action", rec.id, method=method,
                     params=_params_summary(params or {}))
        return result

    def call(self, rec: SessionRecord, method: str,
             params: dict[str, Any] | None = None) -> dict[str, Any]:
        """One UNGATED driver call — perception only (watch paths).

        Watching is always allowed on an open session: the owner
        stop, a takeover, and a degraded brain all bind the hands,
        never the eyes.
        """
        rec = self.reconcile(rec)
        if rec.state != "open":
            raise SessionError(
                f"session {rec.id} is {rec.state}"
                + (f" — {rec.note}" if rec.note else ""))
        client = self.client_for(rec)
        result = client.call(method, params or {})
        self._touch(rec)
        return result

    # -- frames ------------------------------------------------------------

    def save_frame(self, session_id: str, png: bytes) -> Path:
        folder = self.root / "frames" / session_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{time.time_ns()}.png"
        path.write_bytes(png)
        frames = sorted(folder.glob("*.png"))
        for stale in frames[:-MAX_FRAMES_PER_SESSION]:
            with suppress(OSError):
                stale.unlink()
        return path

    def latest_frame(self, session_id: str) -> Path | None:
        folder = self.root / "frames" / session_id
        frames = sorted(folder.glob("*.png"))
        return frames[-1] if frames else None


def _display_number(display: str) -> int:
    text = display.rsplit(":", 1)[-1].split(".", 1)[0]
    try:
        return int(text)
    except ValueError:
        return -1


def _params_summary(params: dict[str, Any]) -> str:
    parts = []
    for key, value in sorted(params.items()):
        text = str(value)
        parts.append(f"{key}={text[:80]}")
    return " ".join(parts)[:280]
