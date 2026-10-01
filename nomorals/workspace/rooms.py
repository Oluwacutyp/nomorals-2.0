"""Project rooms — persistent per-goal workspaces (Prompt 05).

Every goal/project gets a **room**: a directory under
``<workspace>/rooms/<slug>/`` plus a DB row, where ALL of its context
lives — plan, progress, files, scratch work, logs, decisions.  Entering a
room rehydrates from ``ROOM.md`` alone, so Devon never loses its place
across sessions.

Layout::

    rooms/<slug>/
      ROOM.md    # auto-maintained: title, status, step, blockers, decisions
      plan.md    # the decomposed plan
      files/     # deliverables and working documents
      scratch/   # throwaway experiments (safe to clean)
      logs/      # per-step execution logs
      inbox/     # drop-ins for this room (Prompt 06 inbox, room-scoped)

``ROOM.md`` carries a fenced ``state`` JSON block — that is the rehydration
contract: a fresh ``RoomManager`` on the same root can resume the room from
the file alone.  The ``rooms`` DB table is an index (list/search/linked
lookups); on ``enter()`` a missing row is rebuilt from ``ROOM.md``.

Security: :class:`RoomContext.path` is a sandbox boundary.  Every resolved
path is verified to stay inside the room root (symlinks resolved first);
any escape raises :class:`RoomEscapeError`.  All text written to logs and
``ROOM.md`` passes through :func:`redact_secrets` first.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..core.ids import new_short_id
from .inbox import redact_secrets

_log = get_logger(__name__)

ROOM_KINDS = ("goal", "project", "ad_hoc")
ROOM_STATUSES = ("active", "paused", "archived")

_MAX_SLUG = 60
_MAX_DECISIONS = 100
_STATE_FENCE = "```json"


class RoomEscapeError(Exception):
    """Raised when a room path resolves outside the room root."""


def slugify(title: str) -> str:
    """Filesystem-safe slug: lowercase, dashes, max 60 chars."""
    s = (title or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    s = re.sub(r"-{2,}", "-", s)
    return (s[:_MAX_SLUG].rstrip("-") or f"room-{new_short_id()[:8]}")


def _utcnow() -> float:
    return time.time()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


# ── Room record ────────────────────────────────────────────────────────────


@dataclass
class Room:
    id: str
    slug: str
    kind: str = "ad_hoc"
    linked_id: str = ""
    title: str = ""
    status: str = "active"
    created_at: float = 0.0
    last_entered_at: float = 0.0
    updated_at: float = 0.0
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def current_step(self) -> str:
        return str(self.state.get("current_step") or "")

    @property
    def blockers(self) -> list[str]:
        b = self.state.get("blockers") or []
        return [str(x) for x in b] if isinstance(b, list) else []

    @property
    def dirty(self) -> bool:
        return bool(self.state.get("dirty"))

    @property
    def decisions(self) -> list[dict[str, Any]]:
        d = self.state.get("decisions") or []
        return list(d) if isinstance(d, list) else []

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "slug": self.slug, "kind": self.kind,
            "linked_id": self.linked_id, "title": self.title,
            "status": self.status, "created_at": self.created_at,
            "last_entered_at": self.last_entered_at,
            "updated_at": self.updated_at,
            "current_step": self.current_step,
            "blockers": self.blockers, "dirty": self.dirty,
            "decisions": self.decisions,
        }


# ── ROOM.md render / parse ────────────────────────────────────────────────


def _render_room_md(room: Room, links: list[str]) -> str:
    st = room.state
    linked = f"{room.kind}:{room.linked_id}" if room.linked_id else "none"
    decisions = room.decisions
    if decisions:
        dec_lines = "\n".join(
            f"- [{d.get('at', '?')}] {d.get('decision', '')}"
            + (f" — {d.get('rationale', '')}" if d.get("rationale") else "")
            for d in decisions[-20:]
        )
    else:
        dec_lines = "(none)"
    blockers = room.blockers
    blk_lines = "\n".join(f"- {b}" for b in blockers) if blockers else "(none)"
    link_lines = "\n".join(f"- {s}" for s in links) if links else "(none)"
    # state goes through redaction before it ever touches disk
    state_json = redact_secrets(json.dumps(st, indent=2, default=str))
    body = f"""# Room: {redact_secrets(room.title)}

slug: {room.slug}
kind: {room.kind}
linked: {linked}
status: {room.status}
created: {_iso(room.created_at)}
last_entered: {_iso(room.last_entered_at) if room.last_entered_at else "(never)"}

## Current step
{redact_secrets(room.current_step) or "(none)"}

## Blockers
{redact_secrets(blk_lines)}

## Decisions
{redact_secrets(dec_lines)}

## Links
{link_lines}

## State
{_STATE_FENCE}
{state_json}
```
"""
    return body


_STATE_RE = re.compile(r"## State\s+```json\s*\n(.*?)\n```", re.DOTALL)


def _parse_room_md(text: str) -> dict[str, Any]:
    """Extract the state dict from a ROOM.md file.  {} when absent."""
    m = _STATE_RE.search(text or "")
    if not m:
        return {}
    try:
        st = json.loads(m.group(1))
        return st if isinstance(st, dict) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _parse_room_header(text: str) -> dict[str, str]:
    """Best-effort parse of the ROOM.md header fields (for rehydration)."""
    out: dict[str, str] = {}
    for key in ("slug", "kind", "linked", "status", "title"):
        m = re.search(rf"^{key}:\s*(.+)$", text or "", re.MULTILINE)
        if m:
            out[key] = m.group(1).strip()
    m = re.search(r"^# Room:\s*(.+)$", text or "", re.MULTILINE)
    if m:
        out["title"] = m.group(1).strip()
    return out


# ── RoomContext: the agent's handle inside a room ───────────────────────────


class RoomContext:
    """What an agent works with while inside a room.

    Usable as a context manager::

        with manager.enter(slug) as ctx:
            out = ctx.path("files", "report.md")
            out.write_text("...")
            ctx.log("step-1", "done", {"words": 1200})
            ctx.decide("use sqlite", "simpler than postgres for this")
            ctx.checkpoint("step 1 complete")

    Exiting the ``with`` block checkpoints automatically.
    """

    def __init__(self, manager: "RoomManager", room: Room) -> None:
        self._manager = manager
        self._room = room
        # resolved once: the sandbox root every path is checked against
        self._dir = (manager.rooms_dir / room.slug).resolve()
        self._closed = False

    @property
    def room(self) -> Room:
        return self._room

    @property
    def slug(self) -> str:
        return self._room.slug

    # -- sandbox -----------------------------------------------------------
    def path(self, *parts: str) -> Path:
        """Resolve ``parts`` INSIDE the room.

        Raises :class:`RoomEscapeError` on any escape attempt: ``..``,
        absolute paths, or symlinks (inside or outside the room) that
        resolve outside the room root.
        """
        for p in parts:
            if not p:
                continue
            if os.path.isabs(p):
                raise RoomEscapeError(
                    f"absolute path rejected in room {self._room.slug}: {p!r}")
            # a part that is exactly ".." or starts with "../" is rejected
            # early for a clear error; the resolve() check below is the
            # real boundary (it also catches symlink tricks)
            segs = Path(p).parts
            if any(s == ".." for s in segs):
                raise RoomEscapeError(
                    f"parent traversal rejected in room {self._room.slug}: "
                    f"{p!r}")
        target = (self._dir.joinpath(*[p for p in parts if p])).resolve()
        if target != self._dir and self._dir not in target.parents:
            raise RoomEscapeError(
                f"path escapes room {self._room.slug}: "
                f"{os.path.join(*parts)!r}")
        return target

    def mkdir(self, *parts: str) -> Path:
        """``path()`` + create the directory (still sandboxed)."""
        p = self.path(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # -- logging / decisions ------------------------------------------------
    def log(self, step: str, event: str, data: Any = None) -> None:
        """Append a JSON-lines entry to the room's activity log."""
        entry = {
            "ts": _utcnow(), "at": _iso(_utcnow()),
            "step": redact_secrets(str(step))[:120],
            "event": redact_secrets(str(event))[:120],
            "data": redact_secrets(
                json.dumps(data, default=str)[:4000]
                if data is not None else ""),
        }
        log_file = self._dir / "logs" / "activity.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def decide(self, decision: str, rationale: str = "") -> None:
        """Record a decision in ROOM.md and in state (redacted, capped)."""
        now = _utcnow()
        self._room.state.setdefault("decisions", []).append({
            "at": _iso(now), "ts": now,
            "decision": redact_secrets(decision)[:500],
            "rationale": redact_secrets(rationale)[:500],
        })
        # cap the log; ROOM.md renders the tail
        self._room.state["decisions"] = self._room.state["decisions"][
            -_MAX_DECISIONS:]
        self.log("room", "decision", {"decision": decision})
        self._manager._write_room_md(self._room)  # noqa: SLF001

    def set_step(self, step: str) -> None:
        """Update the room's current step (and ROOM.md)."""
        self._room.state["current_step"] = redact_secrets(step)[:500]
        self._manager._write_room_md(self._room)  # noqa: SLF001

    def add_blocker(self, blocker: str) -> None:
        """Flag a blocker — rooms with blockers are skipped by tick()."""
        blockers = self._room.state.setdefault("blockers", [])
        b = redact_secrets(blocker)[:300]
        if b not in blockers:
            blockers.append(b)
        self.log("room", "blocker", {"blocker": blocker})
        self._manager._write_room_md(self._room)  # noqa: SLF001

    def clear_blockers(self) -> None:
        self._room.state["blockers"] = []
        self.log("room", "blockers_cleared", {})
        self._manager._write_room_md(self._room)  # noqa: SLF001

    # -- persistence ---------------------------------------------------------
    def checkpoint(self, summary: str = "") -> None:
        """Persist full state NOW: clear the dirty flag, write ROOM.md+DB."""
        if summary:
            self._room.state["last_checkpoint"] = redact_secrets(summary)[:300]
            self._room.state["last_checkpoint_at"] = _utcnow()
        self._room.state["dirty"] = False
        self._manager._persist(self._room)  # noqa: SLF001

    # -- context manager ------------------------------------------------------
    def __enter__(self) -> "RoomContext":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # exiting checkpoints — even on exception, so the room is never
        # left dirty-without-a-trace; the exception still propagates
        try:
            if exc_type is not None:
                self.log("room", "exit_with_error",
                         {"error": f"{exc_type.__name__}: {exc}"})
                # an errored session stays dirty: it needs reconciliation,
                # not a clean bill of health
                self._room.state["dirty"] = True
                self._manager._persist(self._room)  # noqa: SLF001
            else:
                self.checkpoint("session exit")
        finally:
            self._closed = True

    def close(self) -> None:
        """Explicit close outside a ``with`` block (checkpoints)."""
        if not self._closed:
            self.__exit__(None, None, None)


# ── RoomManager ─────────────────────────────────────────────────────────────


class RoomManager:
    """Owns rooms: creation, entry/exit, persistence, tick, search.

    ``root`` is the workspace root (``context.settings.workspace_dir``);
    rooms live at ``<root>/rooms/<slug>/``.  ``db`` is Devon's Database
    (migration 56 creates the tables; the manager also creates them
    ``IF NOT EXISTS`` defensively).
    """

    def __init__(self, root: str | os.PathLike[str], db: Any = None) -> None:
        self.root = Path(root).resolve()
        self.rooms_dir = self.root / "rooms"
        self.rooms_dir.mkdir(parents=True, exist_ok=True)
        self.db = db
        self._lock = threading.RLock()
        if db is not None:
            self._ensure_tables()

    # -- schema ----------------------------------------------------------------
    def _ensure_tables(self) -> None:
        assert self.db is not None
        self.db.execute_statements(
            """
            CREATE TABLE IF NOT EXISTS rooms (
                id              TEXT PRIMARY KEY,
                slug            TEXT NOT NULL UNIQUE,
                kind            TEXT NOT NULL DEFAULT 'ad_hoc',
                linked_id       TEXT NOT NULL DEFAULT '',
                title           TEXT NOT NULL DEFAULT '',
                status          TEXT NOT NULL DEFAULT 'active',
                created_at      REAL NOT NULL DEFAULT 0,
                last_entered_at REAL NOT NULL DEFAULT 0,
                updated_at      REAL NOT NULL DEFAULT 0,
                state_json      TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_rooms_linked
                ON rooms(kind, linked_id);
            CREATE INDEX IF NOT EXISTS idx_rooms_status ON rooms(status);
            CREATE TABLE IF NOT EXISTS room_links (
                slug_a     TEXT NOT NULL,
                slug_b     TEXT NOT NULL,
                created_at REAL NOT NULL DEFAULT 0,
                PRIMARY KEY (slug_a, slug_b)
            );
            """)

    # -- RoomProvider protocol (lets this replace FilesystemRooms) ------------
    def inbox_path(self, slug: str) -> Path | None:
        room = self.get(slug)
        if room is None:
            return None
        p = self.rooms_dir / room.slug / "inbox"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def files_path(self, slug: str) -> Path | None:
        room = self.get(slug)
        if room is None:
            return None
        p = self.rooms_dir / room.slug / "files"
        p.mkdir(parents=True, exist_ok=True)
        return p

    # -- create ------------------------------------------------------------------
    def create(self, title: str, kind: str = "ad_hoc",
               linked_id: str = "", plan: list[str] | None = None,
               state: dict[str, Any] | None = None) -> Room:
        """Create a room: DB row + directory layout + ROOM.md + plan.md."""
        if kind not in ROOM_KINDS:
            raise ValueError(f"bad room kind: {kind!r}")
        base = slugify(title)
        slug = base
        with self._lock:
            n = 2
            while self._slug_taken(slug):
                suffix = f"-{n}"
                slug = (base[:_MAX_SLUG - len(suffix)] + suffix)
                n += 1
            now = _utcnow()
            room = Room(
                id=new_short_id("room"), slug=slug, kind=kind,
                linked_id=linked_id or "", title=(title or "").strip()[:200],
                status="active", created_at=now, last_entered_at=0.0,
                updated_at=now,
                state={"current_step": "", "blockers": [],
                       "decisions": [], "dirty": False,
                       **(state or {})},
            )
            room_dir = self.rooms_dir / slug
            for sub in ("files", "scratch", "logs", "inbox"):
                (room_dir / sub).mkdir(parents=True, exist_ok=True)
            plan_text = "\n".join(
                f"{i + 1}. {redact_secrets(str(s))[:400]}"
                for i, s in enumerate(plan or [])) or "(no plan yet)"
            (room_dir / "plan.md").write_text(
                f"# Plan: {redact_secrets(room.title)}\n\n{plan_text}\n",
                encoding="utf-8")
            self._write_room_md(room)
            self._persist(room)
            _log.info("room created: %s (%s)", slug, kind)
            return room

    def _slug_taken(self, slug: str) -> bool:
        if (self.rooms_dir / slug).exists():
            return True
        if self.db is None:
            return False
        try:
            rows = self.db.query_rows(
                "SELECT 1 FROM rooms WHERE slug=? LIMIT 1", (slug,))
            return bool(rows)
        except Exception:  # noqa: BLE001 — table may not exist yet
            return False

    # -- read --------------------------------------------------------------------
    def _row_to_room(self, row: Any) -> Room:
        d = dict(row) if not isinstance(row, dict) else row
        try:
            state = json.loads(d.get("state_json") or "{}")
        except (json.JSONDecodeError, ValueError):
            state = {}
        if not isinstance(state, dict):
            state = {}
        return Room(
            id=str(d.get("id") or ""), slug=str(d.get("slug") or ""),
            kind=str(d.get("kind") or "ad_hoc"),
            linked_id=str(d.get("linked_id") or ""),
            title=str(d.get("title") or ""),
            status=str(d.get("status") or "active"),
            created_at=float(d.get("created_at") or 0),
            last_entered_at=float(d.get("last_entered_at") or 0),
            updated_at=float(d.get("updated_at") or 0),
            state=state,
        )

    def get(self, slug: str) -> Room | None:
        if self.db is None:
            return None
        try:
            rows = self.db.query_rows(
                "SELECT * FROM rooms WHERE slug=? LIMIT 1", (slug,))
        except Exception:  # noqa: BLE001
            return None
        return self._row_to_room(rows[0]) if rows else None

    def get_by_linked(self, kind: str, linked_id: str) -> Room | None:
        if self.db is None or not linked_id:
            return None
        try:
            rows = self.db.query_rows(
                "SELECT * FROM rooms WHERE kind=? AND linked_id=? LIMIT 1",
                (kind, linked_id))
        except Exception:  # noqa: BLE001
            return None
        return self._row_to_room(rows[0]) if rows else None

    def list(self, status: str = "") -> list[Room]:
        if self.db is None:
            return []
        try:
            if status:
                rows = self.db.query_rows(
                    "SELECT * FROM rooms WHERE status=? ORDER BY updated_at DESC",
                    (status,))
            else:
                rows = self.db.query_rows(
                    "SELECT * FROM rooms ORDER BY updated_at DESC")
        except Exception:  # noqa: BLE001
            return []
        return [self._row_to_room(r) for r in rows]

    # -- enter / exit --------------------------------------------------------------
    def enter(self, slug: str) -> RoomContext:
        """Enter a room, returning its context.

        Rehydration: if the DB row is missing but ``ROOM.md`` exists on
        disk, the room is rebuilt from the file alone (the rehydration
        contract).  A room whose state is ``dirty`` is reconciled from
        its logs and flagged — never silently resumed mid-step.
        """
        with self._lock:
            room = self.get(slug)
            if room is None:
                room = self._rehydrate_from_disk(slug)
            if room is None:
                raise KeyError(f"no room {slug!r}")
            if room.dirty:
                self._reconcile_dirty(room)
            room.state["dirty"] = True
            room.last_entered_at = _utcnow()
            room.updated_at = room.last_entered_at
            self._persist(room)
            return RoomContext(self, room)

    def _rehydrate_from_disk(self, slug: str) -> Room | None:
        room_md = self.rooms_dir / slug / "ROOM.md"
        if not room_md.is_file():
            return None
        try:
            text = room_md.read_text(encoding="utf-8")
        except OSError:
            return None
        header = _parse_room_header(text)
        state = _parse_room_md(text)
        linked = header.get("linked", "none")
        kind, _, linked_id = linked.partition(":")
        if kind not in ROOM_KINDS:
            kind = "ad_hoc"
            linked_id = ""
        room = Room(
            id=new_short_id("room"), slug=slug, kind=kind,
            linked_id=linked_id if linked_id != "none" else "",
            title=header.get("title", slug),
            status=header.get("status", "active")
            if header.get("status") in ROOM_STATUSES else "active",
            created_at=_utcnow(), state=state or {
                "current_step": "", "blockers": [], "decisions": [],
                "dirty": False},
        )
        self._persist(room)
        _log.warning("room %s rehydrated from ROOM.md (DB row was missing)",
                     slug)
        return room

    def _reconcile_dirty(self, room: Room) -> None:
        """A dirty room exited without checkpoint — reconcile, don't resume.

        Reads the tail of the activity log to report where it stopped,
        keeps ``dirty`` set until an explicit checkpoint, and records the
        event so the morning briefing can flag it.
        """
        tail: list[str] = []
        log_file = self.rooms_dir / room.slug / "logs" / "activity.log"
        try:
            if log_file.is_file():
                lines = log_file.read_text(
                    encoding="utf-8").strip().split("\n")
                for line in lines[-5:]:
                    try:
                        e = json.loads(line)
                        tail.append(
                            f"{e.get('at', '?')} {e.get('step')}: "
                            f"{e.get('event')}")
                    except (json.JSONDecodeError, ValueError):
                        continue
        except OSError:  # noqa: E103 - dirty-room recovery is best-effort; missing log is expected
            pass
        room.state["dirty_reconciled_at"] = _utcnow()
        room.state["dirty_tail"] = tail
        room.state.setdefault("blockers", [])
        note = ("recovered from unclean exit — review before continuing")
        if note not in room.state["blockers"]:
            room.state["blockers"].append(note)
        _log.warning("room %s was dirty; reconciled from logs (%d entries)",
                     room.slug, len(tail))

    # -- persistence -----------------------------------------------------------------
    def _room_links(self, slug: str) -> list[str]:
        if self.db is None:
            return []
        try:
            rows = self.db.query_rows(
                "SELECT slug_b AS other FROM room_links WHERE slug_a=? "
                "UNION SELECT slug_a AS other FROM room_links WHERE slug_b=?",
                (slug, slug))
            return sorted({str(r["other"]) for r in rows})
        except Exception:  # noqa: BLE001
            return []

    def _write_room_md(self, room: Room) -> None:
        links = self._room_links(room.slug)
        text = _render_room_md(room, links)
        room_dir = self.rooms_dir / room.slug
        room_dir.mkdir(parents=True, exist_ok=True)
        # atomic: tmp + rename so a crash never leaves half a ROOM.md
        tmp = room_dir / "ROOM.md.tmp"
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, room_dir / "ROOM.md")

    def _persist(self, room: Room) -> None:
        room.updated_at = _utcnow()
        self._write_room_md(room)
        if self.db is None:
            return
        try:
            self.db.execute(
                "INSERT INTO rooms (id, slug, kind, linked_id, title, status,"
                " created_at, last_entered_at, updated_at, state_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(slug) DO UPDATE SET"
                " kind=excluded.kind, linked_id=excluded.linked_id,"
                " title=excluded.title, status=excluded.status,"
                " last_entered_at=excluded.last_entered_at,"
                " updated_at=excluded.updated_at,"
                " state_json=excluded.state_json",
                (room.id, room.slug, room.kind, room.linked_id, room.title,
                 room.status, room.created_at, room.last_entered_at,
                 room.updated_at,
                 redact_secrets(json.dumps(room.state, default=str))))
        except Exception as exc:  # noqa: BLE001 — disk is source of truth
            _log.warning("room %s DB persist failed: %s", room.slug, exc)

    # -- lifecycle ----------------------------------------------------------------------
    def archive(self, slug: str) -> Room:
        return self._set_status(slug, "archived")

    def pause(self, slug: str) -> Room:
        return self._set_status(slug, "paused")

    def resume(self, slug: str) -> Room:
        return self._set_status(slug, "active")

    def _set_status(self, slug: str, status: str) -> Room:
        with self._lock:
            room = self.get(slug) or self._rehydrate_from_disk(slug)
            if room is None:
                raise KeyError(f"no room {slug!r}")
            room.status = status
            self._persist(room)
            return room

    def link(self, slug_a: str, slug_b: str) -> dict[str, Any]:
        """Record a read-only cross reference between two rooms."""
        if slug_a == slug_b:
            raise ValueError("cannot link a room to itself")
        with self._lock:
            for s in (slug_a, slug_b):
                if self.get(s) is None and \
                        self._rehydrate_from_disk(s) is None:
                    raise KeyError(f"no room {s!r}")
            if self.db is not None:
                a, b = sorted((slug_a, slug_b))
                try:
                    self.db.execute(
                        "INSERT OR IGNORE INTO room_links "
                        "(slug_a, slug_b, created_at) VALUES (?,?,?)",
                        (a, b, _utcnow()))
                except Exception as exc:  # noqa: BLE001
                    _log.warning("room link persist failed: %s", exc)
            # record in both rooms' ROOM.md
            for s in (slug_a, slug_b):
                room = self.get(s)
                if room is not None:
                    self._write_room_md(room)
            return {"linked": [slug_a, slug_b]}

    # -- search ------------------------------------------------------------------------------
    def search(self, query: str, *, deep: bool = False,
               limit: int = 20) -> list[dict[str, Any]]:
        """Search ROOM.md + decisions across rooms.

        Default searches titles, current step, blockers and decisions —
        NOT file contents (``deep=True`` also greps ``files/``).
        """
        q = (query or "").lower().strip()
        if not q:
            return []
        hits: list[dict[str, Any]] = []
        for room in self.list():
            hay = "\n".join([
                room.title, room.slug, room.current_step,
                *room.blockers,
                *(str(d.get("decision", "")) for d in room.decisions),
                *(str(d.get("rationale", "")) for d in room.decisions),
            ]).lower()
            if q in hay:
                hits.append({"slug": room.slug, "title": room.title,
                             "kind": room.kind, "status": room.status,
                             "current_step": room.current_step[:200],
                             "where": "room"})
            if deep and len(hits) < limit:
                files_dir = self.rooms_dir / room.slug / "files"
                if files_dir.is_dir():
                    for f in sorted(files_dir.rglob("*")):
                        if len(hits) >= limit:
                            break
                        if not f.is_file() or f.stat().st_size > 200_000:
                            continue
                        try:
                            text = f.read_text(
                                encoding="utf-8", errors="replace")
                        except OSError:
                            continue
                        if q in text.lower():
                            hits.append({
                                "slug": room.slug, "title": room.title,
                                "kind": room.kind, "status": room.status,
                                "where": f"files/{f.name}"})
                            break  # one hit per room in deep mode
            if len(hits) >= limit:
                break
        return hits[:limit]

    def stale_rooms(self, days: float = 30.0) -> list[Room]:
        """Rooms idle longer than ``days`` — reported, never auto-archived."""
        cutoff = _utcnow() - days * 86400
        return [r for r in self.list(status="active")
                if (r.last_entered_at or r.created_at) < cutoff]


    # -- tick: between-chat continuity ---------------------------------------------------
    def tick(self, *, executor: Any = None,
             goal_system: Any = None, project_manager: Any = None,
             max_steps_per_tick: int = 3) -> dict[str, Any]:
        """Advance each active room's linked goal/project by one tick.

        Respects per-room ``max_steps_per_tick`` (default 3).  Rooms with
        blockers are SKIPPED (not retried) until the blocker clears.  Dirty
        rooms are reconciled, not advanced.  Idempotent: a crash mid-step
        leaves the room dirty, which the next tick reconciles.
        """
        summary: dict[str, Any] = {"advanced": [], "skipped": [],
                                   "reconciled": [], "errors": []}
        for room in self.list(status="active"):
            slug = room.slug
            try:
                if room.blockers:
                    summary["skipped"].append(
                        {"slug": slug,
                         "reason": f"blocked: {room.blockers[0][:120]}"})
                    continue
                if room.dirty:
                    with self._lock:
                        fresh = self.get(slug) or room
                        self._reconcile_dirty(fresh)
                        self._persist(fresh)
                    summary["reconciled"].append({"slug": slug})
                    continue
                steps = self._advance_linked(
                    room, executor=executor, goal_system=goal_system,
                    project_manager=project_manager,
                    max_steps=max_steps_per_tick)
                summary["advanced"].append({"slug": slug, "steps": steps})
            except Exception as exc:  # noqa: BLE001 — one room never kills tick
                _log.warning("room tick failed for %s: %s", slug, exc)
                summary["errors"].append({"slug": slug,
                                          "error": f"{type(exc).__name__}: "
                                                   f"{exc}"[:200]})
        return summary

    def _advance_linked(self, room: Room, *, executor: Any,
                        goal_system: Any, project_manager: Any,
                        max_steps: int) -> int:
        """Drive a room's linked goal/project forward.  Returns steps done."""
        done = 0
        if room.kind == "goal" and room.linked_id and goal_system is not None:
            for _ in range(max(1, max_steps)):
                g = goal_system.get(room.linked_id)
                if g is None or g.status != "active":
                    break
                if not any(s.status == "pending" for s in g.steps):
                    break
                goal_system.advance(room.linked_id, executor=executor)
                done += 1
        elif (room.kind == "project" and room.linked_id
                and project_manager is not None):
            for _ in range(max(1, max_steps)):
                p = project_manager._load(room.linked_id) \
                    if hasattr(project_manager, "_load") else None
                if p is None or getattr(p, "status", "") not in {
                        "running", "planning"}:
                    break
                if p.next_step is None:
                    break
                project_manager.advance(room.linked_id, executor=executor)
                done += 1
        return done


# ── scheduler ─────────────────────────────────────────────────────────────

ROOMS_TICK_JOB_NAME = "rooms tick"


def ensure_rooms_tick_job(context: Any) -> dict[str, Any]:
    """Register the single rooms tick job on the agent scheduler.

    Idempotent by name; safe to call on every boot.  Follows the
    watchers ``ensure_sweeper_job`` pattern.  Every 5m the job calls the
    ``room`` tool's ``tick`` action, which advances each active room's
    linked goal/project (≤3 steps, blockers skipped).
    """
    from ..agents.scheduler import Scheduler

    sched = Scheduler(context)
    try:
        have = [j for j in sched.list_jobs()
                if j.get("name") == ROOMS_TICK_JOB_NAME]
    except Exception:  # noqa: BLE001 — scheduler table may not exist yet
        have = []
    if have:
        return {"name": ROOMS_TICK_JOB_NAME, "already_scheduled": True,
                "job_id": have[0].get("id")}
    job = sched.add(ROOMS_TICK_JOB_NAME, "every 5m", "tool",
                    {"tool": "room", "args": {"action": "tick"}})
    _log.info("scheduled rooms tick job: %s", ROOMS_TICK_JOB_NAME)
    return {"name": ROOMS_TICK_JOB_NAME, "scheduled": True,
            "job_id": job.get("id")}


# ── tool registration ─────────────────────────────────────────────────────


def _manager_from_context(context: Any) -> RoomManager:
    from pathlib import Path as _Path
    root = _Path(context.settings.workspace_dir)
    return RoomManager(root=root, db=getattr(context, "db", None))


def register(registry: Any) -> None:
    """Register the ``room`` tool (agent-callable room management)."""
    from ..core.policy import Capability

    context = registry.context

    @registry.register(
        "room",
        description=(
            "Project rooms: persistent per-goal workspaces. "
            "action=new (title, kind=goal|project|ad_hoc, linked_id, plan) | "
            "list (status) | enter (slug → ROOM.md summary) | status (slug) | "
            "archive | pause | resume | link (slug_a, slug_b) | "
            "search (query, deep) | tick (advance active rooms) | "
            "stale (days). Rooms isolate work: files in files/, logs in "
            "logs/, decisions in ROOM.md."
        ),
        capability=Capability.MEM_WRITE,
    )
    def room(action: str = "list", slug: str = "", title: str = "",
             kind: str = "ad_hoc", linked_id: str = "",
             plan: str = "", query: str = "", deep: bool = False,
             status: str = "", slug_a: str = "", slug_b: str = "",
             days: float = 30.0) -> dict[str, Any]:
        mgr = _manager_from_context(context)
        if action == "new":
            plan_list = [l.strip() for l in plan.split("\n") if l.strip()] \
                if plan else None
            r = mgr.create(title or "untitled", kind=kind,
                           linked_id=linked_id, plan=plan_list)
            return {"room": r.to_dict()}
        if action == "enter":
            with mgr.enter(slug) as ctx:
                md = (mgr.rooms_dir / slug / "ROOM.md").read_text(
                    encoding="utf-8")
            return {"slug": slug, "summary": md[:3000]}
        if action == "status":
            r = mgr.get(slug)
            if r is None:
                return {"found": False}
            return {"found": True, "room": r.to_dict()}
        if action == "archive":
            return {"room": mgr.archive(slug).to_dict()}
        if action == "pause":
            return {"room": mgr.pause(slug).to_dict()}
        if action == "resume":
            return {"room": mgr.resume(slug).to_dict()}
        if action == "link":
            return mgr.link(slug_a, slug_b)
        if action == "search":
            return {"hits": mgr.search(query, deep=bool(deep))}
        if action == "stale":
            return {"stale": [r.to_dict()
                              for r in mgr.stale_rooms(days=days)]}
        if action == "tick":
            from ..agents.goals import GoalSystem
            from ..agents.projects import ProjectManager
            return mgr.tick(goal_system=GoalSystem(context),
                            project_manager=ProjectManager(context))
        return {"rooms": [r.to_dict() for r in mgr.list(status=status)]}
