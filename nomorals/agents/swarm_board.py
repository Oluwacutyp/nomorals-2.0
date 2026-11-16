"""The swarm board: one shared message board per swarm run.

Antigravity's /swarm coordinates named workers on ONE shared board:
public briefs, @mentions, a verifier per 2–4 workers, a monitor that
wakes a stalled team. Devon's /swarm already runs parallel agents +
fusion (`agents/swarm.py`), and a key/value `Blackboard` already
exists — but a blackboard is a *state* store (``post`` overwrites
the same key), not a *conversation*: no append-only feed, no
mentions, no briefs, no verifier, no monitor, no member states,
no durable board store.

This module is the conversation layer on top of the blackboard:

* :class:`SwarmBoard` — append-only feed (one unique blackboard
  key per post, so the feed survives on the board's own terms),
  public briefs, member roster with active / waiting / killed
  states, one verifier per 2–4 workers, @name / @all / @workers /
  @verifiers mention routing with per-member unread tracking,
  stall detection, and an owner-readable render.
* :class:`SwarmBoardStore` — durable boards, WorkflowStore shape:
  one atomic JSON index (``<home>/data/swarm_boards.json``),
  tmp+replace writes, corrupt → quarantine + degrade to empty
  with a named reason, bad rows skipped, reads never raise.
* The monitor tick — ``ensure_swarm_board_job`` +
  :func:`run_swarm_board_tick_once` + the ``__swarm_board_tick__``
  routine tool job (every 5m, the delegation-manager tick's
  cadence): orphaned / cross-restart boards that went quiet with
  unfinished work get one `@monitor` wake-up per stall episode.

Layering: this module is L5 (`agents`), same package as
``swarm.py`` / ``blackboard.py``. It never imports the notifier —
the tick takes an injected one (the digest lane's lesson).
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .blackboard import Blackboard
from .failures import record_failure

__all__ = [
    "STALL_WINDOW_S",
    "SWARM_BOARD_JOB",
    "SWARM_BOARD_SPEC",
    "SWARM_BOARD_TOOL",
    "SwarmBoard",
    "SwarmBoardStore",
    "default_swarm_boards_path",
    "ensure_swarm_board_job",
    "parse_mentions",
    "parse_verifier_verdict",
    "parse_breaker_verdict",
    "run_swarm_board_tick_once",
    "store_for_context",
    "worker_name_for",
]

_log = get_logger(__name__)

SWARM_BOARD_JOB = "swarm board tick"
SWARM_BOARD_SPEC = "every 5m"
SWARM_BOARD_TOOL = "__swarm_board_tick__"

#: A board quiet longer than this with unfinished work is stalled.
#: A leg's wall budget is 420s, so quiet past 600s is a stall,
#: not a slow leg.
STALL_WINDOW_S = 600.0
#: No second wake-up inside this window after one was posted.
WAKE_COOLDOWN_S = 600.0
#: One verifier covers at most this many workers.
MAX_VERIFIER_WORKERS = 4

MEMBER_STATES = ("active", "waiting", "killed")
BOARD_STATES = ("active", "done", "failed")

_MENTION_RE = re.compile(r"@([A-Za-z0-9][A-Za-z0-9_-]*)")
_FLAG_RE = re.compile(
    r"^\s*FLAG\s+@([A-Za-z0-9][A-Za-z0-9_-]*)\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE)
#: A breaker's attack line: ``BREAK @name: <concrete attack>``.
#: ``HOLD @name: ...`` lines are not attacks.
_BREAK_RE = re.compile(
    r"^\s*BREAK\s+@([A-Za-z0-9][A-Za-z0-9_-]*)\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE)
_WORD_RE = re.compile(r"[a-z0-9]+")
#: Words that carry no identity when naming a worker from its subtask.
_NAME_STOPWORDS = frozenset({
    "the", "a", "an", "and", "for", "of", "to", "in", "on", "at",
    "investigate", "is", "are", "do", "does", "with", "from",
})


def parse_mentions(text: str) -> set[str]:
    """Names mentioned in ``text`` (``@name``), without the @.

    ``mail me@home`` is not a mention — the @ must start the token
    (start of string or preceded by whitespace / punctuation that
    is not a word character).
    """
    found = set()
    for match in _MENTION_RE.finditer(text or ""):
        start = match.start()
        if start > 0:
            prev = text[start - 1]
            if prev.isalnum() or prev in "._-":
                continue
        found.add(match.group(1))
    return found


def parse_verifier_verdict(digest: str) -> dict[str, list[str]]:
    """Flags from a verifier leg's digest.

    One line per problem: ``FLAG @name: <note>``. ``OK @name: ...``
    lines are not flags. Returns {worker name: [notes]}.
    """
    flags: dict[str, list[str]] = {}
    for match in _FLAG_RE.finditer(digest or ""):
        flags.setdefault(match.group(1), []).append(match.group(2).strip())
    return flags


def parse_breaker_verdict(digest: str) -> dict[str, list[str]]:
    """Attacks from a breaker leg's digest.

    One line per concrete attack: ``BREAK @name: <attack>``.
    ``HOLD @name: ...`` lines are not attacks. Returns
    {worker name: [attacks]}.
    """
    attacks: dict[str, list[str]] = {}
    for match in _BREAK_RE.finditer(digest or ""):
        attacks.setdefault(match.group(1), []).append(
            match.group(2).strip())
    return attacks


def worker_name_for(subtask: str, index: int,
                    taken: set[str] | None = None) -> str:
    """A deterministic worker name from a subtask.

    First two content words of the subtask, slugified
    (``audit the parser`` → ``audit-parser``); falls back to
    ``worker-<n>`` when nothing is left. ``taken`` names get a
    numeric suffix so a run's names are unique.
    """
    taken = set(taken or ())
    words = [w for w in _WORD_RE.findall((subtask or "").lower())
             if w not in _NAME_STOPWORDS][:2]
    base = "-".join(words) if words else f"worker-{index + 1}"
    name = base
    suffix = 2
    while name in taken:
        name = f"{base}-{suffix}"
        suffix += 1
    return name


def default_swarm_boards_path(home: Path | str | None = None) -> Path:
    """The durable swarm-boards index under a home."""
    base = Path(home) if home is not None else Path.home()
    return base / "data" / "swarm_boards.json"


def store_for_context(context: Any) -> SwarmBoardStore:
    """The durable board store under a runtime's home (in-memory
    when the runtime has no home). Never raises."""
    try:
        home = getattr(getattr(context, "settings", None), "home", None)
        if home is None:
            home = getattr(context, "home", None)
        if home is None:
            return SwarmBoardStore()
        return SwarmBoardStore(path=default_swarm_boards_path(home))
    except Exception:  # noqa: BLE001 - durability is a bonus
        return SwarmBoardStore()


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (tmp file + os.replace)."""
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class SwarmBoard:
    """One shared message board for one swarm run.

    The feed is append-only: every post gets a sequence number and
    its own blackboard key (``swarm.<run>.msg.<seq>``), so the
    blackboard's overwrite-a-key semantics never collapse two
    posts. The in-memory feed list is the render / replay source;
    the blackboard is the live cross-leg store.
    """

    def __init__(self, run_id: str, *, goal: str = "",
                 board: Blackboard | None = None) -> None:
        self.run_id = str(run_id)
        self.goal = str(goal or "")
        self.state = "active"
        self.chat_key = ""
        self._board = board if board is not None else Blackboard()
        self._lock = threading.RLock()
        self._feed: list[dict[str, Any]] = []
        self._members: dict[str, dict[str, Any]] = {}
        self._verifiers: dict[str, list[str]] = {}
        self._breakers: dict[str, list[str]] = {}
        self._seq = 0
        now = time.time()
        self.created_at = now
        self.updated_at = now
        self._last_activity_at = now
        self.last_wake_at = 0.0

    # ── feed ───────────────────────────────────────────────────────
    @property
    def feed(self) -> list[dict[str, Any]]:
        """The ordered feed (oldest first). A copy — never mutate it."""
        with self._lock:
            return [dict(p) for p in self._feed]

    @property
    def members(self) -> list[dict[str, Any]]:
        """The roster (join order). A copy."""
        with self._lock:
            return [dict(m) for m in self._members.values()]

    @property
    def last_activity_at(self) -> float:
        with self._lock:
            return self._last_activity_at

    def _append(self, author: str, text: str, kind: str,
                extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """One post onto the feed and the blackboard. The single
        place a post is built, so the feed and the blackboard copy
        never diverge."""
        author = str(author or "swarm")
        text = str(text or "")
        with self._lock:
            self._seq += 1
            post: dict[str, Any] = {
                "seq": self._seq,
                "author": author,
                "text": text,
                "mentions": sorted(parse_mentions(text)),
                "kind": str(kind or "message"),
                "ts": time.time(),
            }
            if extra:
                post.update(extra)
            self._feed.append(post)
            self._last_activity_at = post["ts"]
            self.updated_at = post["ts"]
            member = self._members.get(author)
            if member is not None:
                member["last_post_at"] = post["ts"]
        try:
            self._board.post(
                f"swarm.{self.run_id}.msg.{post['seq']}",
                dict(post), author=author, topic=self.run_id,
                metadata={"seq": post["seq"], "kind": post["kind"]})
        except Exception as exc:  # noqa: BLE001 - posting never breaks a leg
            record_failure(f"swarm_board:{self.run_id}:post", exc)
        return dict(post)

    def post(self, author: str, text: str, *,
             kind: str = "message") -> dict[str, Any]:
        """Append a post. Never raises — a post that cannot land is
        recorded as a failure and dropped, never fatal to the leg."""
        return self._append(author, text, kind)

    def brief(self, name: str, *, owns: str, must_not_touch: str,
              done: str) -> dict[str, Any]:
        """Post a PUBLIC brief for a member.

        Every member reads every brief — that is the point: who owns
        what, what they must not touch, and what "done" looks like
        is visible to the whole team, not whispered to one worker.
        """
        name = str(name or "").strip()
        if not name:
            raise ValueError("a brief needs a member name")
        self.add_member(name, "worker")
        text = (
            f"📋 brief for @{name}\n"
            f"owns: {owns}\n"
            f"must not touch: {must_not_touch}\n"
            f"done looks like: {done}")
        return self._append("coordinator", text, "brief", extra={
            "name": name,
            "owns": str(owns),
            "must_not_touch": str(must_not_touch),
            "done": str(done),
        })

    def _worker_briefs_locked(self) -> list[dict[str, Any]]:
        """Worker briefs. Call with the lock held. The coordinator's
        own brief is the run's header, not a worker's orders, so it
        is excluded."""
        workers = {m["name"] for m in self._members.values()
                   if m["role"] == "worker"}
        return [dict(p) for p in self._feed
                if p.get("kind") == "brief" and p.get("name") in workers]

    def briefs(self) -> list[dict[str, Any]]:
        """Worker briefs (public: every member can read them all)."""
        with self._lock:
            return self._worker_briefs_locked()

    # ── roster ─────────────────────────────────────────────────────
    def add_member(self, name: str, role: str,
                   group: str | None = None) -> dict[str, Any]:
        """Add a member (idempotent — re-adding returns the member)."""
        name = str(name or "").strip()
        if not name:
            raise ValueError("a member needs a name")
        role = str(role or "worker").strip().lower()
        with self._lock:
            member = self._members.get(name)
            if member is not None:
                return dict(member)
            member = {
                "name": name,
                "role": role,
                "group": group or "",
                "state": "active",
                # A member joins having read nothing: the whole
                # board — briefs included — is unread to them.
                "last_seen_seq": 0,
                "last_post_at": 0.0,
                "joined_at": time.time(),
            }
            self._members[name] = member
            return dict(member)

    def set_state(self, name: str, state: str) -> dict[str, Any]:
        """Move a member between active / waiting / killed.

        States follow real leg events only (leg starts → active,
        leg finishes → waiting, leg exhausts its re-run → killed).
        Unknown member or unknown state raises — a caller that
        moves a ghost is a bug, and the board fails closed.
        """
        if state not in MEMBER_STATES:
            raise ValueError(
                f"unknown member state {state!r} — {MEMBER_STATES}")
        with self._lock:
            member = self._members.get(name)
            if member is None:
                raise ValueError(f"no swarm member {name!r} on this board")
            member["state"] = state
            self.updated_at = time.time()
            return dict(member)

    # ── verifiers ──────────────────────────────────────────────────
    def assign_verifiers(self) -> list[str]:
        """One verifier per 2–4 workers (1–4 workers → 1 verifier,
        5 workers → 2 verifiers, …). Idempotent."""
        with self._lock:
            if self._verifiers:
                return sorted(self._verifiers)
            workers = [m["name"] for m in self._members.values()
                       if m["role"] == "worker"]
            verifiers: list[str] = []
            for start in range(0, len(workers), MAX_VERIFIER_WORKERS):
                group = workers[start:start + MAX_VERIFIER_WORKERS]
                name = f"verifier-{len(verifiers) + 1}"
                self._verifiers[name] = list(group)
                self.add_member(name, "verifier")
                verifiers.append(name)
            return verifiers

    def verifier_for(self, worker: str) -> str | None:
        """The verifier covering ``worker``, or None."""
        with self._lock:
            for verifier, group in self._verifiers.items():
                if worker in group:
                    return verifier
        return None

    def verifier_group(self, verifier: str) -> list[str]:
        """The workers one verifier covers."""
        with self._lock:
            return list(self._verifiers.get(verifier, []))

    # ── breakers ───────────────────────────────────────────────────
    def assign_breakers(self) -> list[str]:
        """One breaker per 2–4 workers (1–4 workers → 1 breaker,
        5 workers → 2 breakers, …). Idempotent. Mirrors
        :meth:`assign_verifiers`: the breaker is a distinct role
        whose job is to ATTACK the attempt, not review it."""
        with self._lock:
            if self._breakers:
                return sorted(self._breakers)
            workers = [m["name"] for m in self._members.values()
                       if m["role"] == "worker"]
            breakers: list[str] = []
            for start in range(0, len(workers), MAX_VERIFIER_WORKERS):
                group = workers[start:start + MAX_VERIFIER_WORKERS]
                name = f"breaker-{len(breakers) + 1}"
                self._breakers[name] = list(group)
                self.add_member(name, "breaker")
                breakers.append(name)
            return breakers

    def breaker_for(self, worker: str) -> str | None:
        """The breaker covering ``worker``, or None."""
        with self._lock:
            for breaker, group in self._breakers.items():
                if worker in group:
                    return breaker
        return None

    def breaker_group(self, breaker: str) -> list[str]:
        """The workers one breaker covers."""
        with self._lock:
            return list(self._breakers.get(breaker, []))

    # ── mentions / unread ──────────────────────────────────────────
    def mentions_for(self, name: str, *, since: int = 0) -> list[dict[str, Any]]:
        """Posts mentioning ``name`` (``@name``, ``@all``, or the
        member's group — ``@workers`` / ``@verifiers``).

        A name that is not a member gets nothing: an unknown
        mention is plain text, never a routed message.
        """
        with self._lock:
            member = self._members.get(name)
            if member is None:
                return []
            role = member["role"]
            hits = []
            for post in self._feed:
                if post["seq"] <= since:
                    continue
                mentions = set(post.get("mentions") or ())
                if (name in mentions or "all" in mentions
                        or (role == "worker" and "workers" in mentions)
                        or (role == "verifier"
                            and "verifiers" in mentions)
                        or (role == "breaker"
                            and "breakers" in mentions)):
                    hits.append(dict(post))
            return hits

    def unread(self, name: str) -> list[dict[str, Any]]:
        """Posts since the member last read (own posts excluded)."""
        with self._lock:
            member = self._members.get(name)
            if member is None:
                return []
            seen = int(member.get("last_seen_seq") or 0)
            return [dict(p) for p in self._feed
                    if p["seq"] > seen and p["author"] != name]

    def mark_read(self, name: str, seq: int) -> None:
        """Record that ``name`` read up to ``seq``. Never raises."""
        with self._lock:
            member = self._members.get(name)
            if member is not None:
                member["last_seen_seq"] = max(
                    int(member.get("last_seen_seq") or 0), int(seq))

    # ── stalls ─────────────────────────────────────────────────────
    def stall_check(self, *, window_s: float = STALL_WINDOW_S,
                    now: float | None = None) -> dict[str, Any]:
        """Is this board stalled? A board is stalled when it has
        been quiet for ``window_s`` with unfinished work (a worker
        still active or waiting — killed workers are finished)."""
        at = float(now) if now is not None else time.time()
        with self._lock:
            quiet_for = max(0.0, at - self._last_activity_at)
            unfinished = [m["name"] for m in self._members.values()
                          if m["role"] == "worker"
                          and m["state"] in ("active", "waiting")]
            quietest: str | None = None
            quietest_at = 0.0
            for member in self._members.values():
                if member["role"] != "worker":
                    continue
                if member["state"] not in ("active", "waiting"):
                    continue
                last = float(member.get("last_post_at") or 0.0) or float(
                    member.get("joined_at") or self.created_at)
                if quietest is None or last < quietest_at:
                    quietest = member["name"]
                    quietest_at = last
            return {
                "stalled": bool(unfinished) and quiet_for >= float(window_s),
                "quiet_for_s": round(quiet_for, 1),
                "last_seq": self._seq,
                "last_at": self._last_activity_at,
                "unfinished": unfinished,
                "quietest": quietest,
            }

    # ── render / persistence ───────────────────────────────────────
    def render_board(self, limit: int = 30) -> str:
        """Owner-readable board: briefs, members + states, feed.
        Never raises."""
        try:
            with self._lock:
                members = [dict(m) for m in self._members.values()]
                feed = [dict(p) for p in self._feed[-max(1, int(limit)):]]
                briefs = self._worker_briefs_locked()
            lines = [f"🐝 swarm board — {self.run_id}",
                     f"goal: {self.goal[:120]}",
                     f"state: {self.state} · {len(feed)} post(s) shown"]
            if members:
                lines.append("")
                lines.append("members:")
                for member in members:
                    lines.append(
                        f"  @{member['name']} [{member['role']}] — "
                        f"{member['state']}")
            for brief in briefs:
                lines.append("")
                lines.append(f"📋 brief for @{brief.get('name', '?')}: "
                             f"{brief.get('owns', '')[:100]}")
            if feed:
                lines.append("")
                lines.append("board:")
                for post in feed:
                    mentions = (" " + " ".join(f"@{m}" for m in
                                                post.get("mentions") or ())
                                if post.get("mentions") else "")
                    first = (post.get("text") or "").split("\n")[0][:160]
                    lines.append(f"  #{post['seq']} @{post['author']}"
                                 f" [{post['kind']}] {first}{mentions}")
            else:
                lines.append("(empty)")
            return "\n".join(lines)
        except Exception:  # noqa: BLE001 — rendering never breaks callers
            return f"swarm board {self.run_id} (render failed)"

    def to_record(self) -> dict[str, Any]:
        """Durable record (plain data)."""
        with self._lock:
            return {
                "id": self.run_id,
                "goal": self.goal,
                "state": self.state,
                "chat_key": self.chat_key,
                "members": [dict(m) for m in self._members.values()],
                "briefs": self._worker_briefs_locked(),
                "feed": [dict(p) for p in self._feed],
                "verifiers": {k: list(v)
                              for k, v in self._verifiers.items()},
                "breakers": {k: list(v)
                             for k, v in self._breakers.items()},
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "last_activity_at": self._last_activity_at,
                "last_wake_at": self.last_wake_at,
            }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> SwarmBoard:
        """Rebuild a board from :meth:`to_record`."""
        board = cls(str(record.get("id") or ""),
                    goal=str(record.get("goal") or ""))
        board.state = str(record.get("state") or "active")
        board.chat_key = str(record.get("chat_key") or "")
        board.created_at = float(record.get("created_at") or board.created_at)
        board.updated_at = float(record.get("updated_at") or board.updated_at)
        board.last_wake_at = float(record.get("last_wake_at") or 0.0)
        members = record.get("members") or []
        if isinstance(members, list):
            for member in members:
                if not isinstance(member, dict) or not member.get("name"):
                    continue
                board._members[str(member["name"])] = {
                    "name": str(member["name"]),
                    "role": str(member.get("role") or "worker"),
                    "group": str(member.get("group") or ""),
                    "state": (str(member.get("state"))
                              if str(member.get("state")) in MEMBER_STATES
                              else "active"),
                    "last_seen_seq": int(member.get("last_seen_seq") or 0),
                    "last_post_at": float(member.get("last_post_at") or 0.0),
                    "joined_at": float(
                        member.get("joined_at") or board.created_at),
                }
        feed = record.get("feed") or []
        if isinstance(feed, list):
            from .blackboard import BlackboardEntry

            for post in feed:
                if not isinstance(post, dict):
                    continue
                entry = {
                    "seq": int(post.get("seq") or 0),
                    "author": str(post.get("author") or ""),
                    "text": str(post.get("text") or ""),
                    "mentions": [str(m) for m in
                                 (post.get("mentions") or [])],
                    "kind": str(post.get("kind") or "message"),
                    "ts": float(post.get("ts") or 0.0),
                }
                for key in ("name", "owns", "must_not_touch", "done"):
                    if post.get(key) is not None:
                        entry[key] = post[key]
                board._feed.append(entry)
                board._seq = max(board._seq, entry["seq"])
                # Re-seat the blackboard key directly: a restore is
                # state recovery, not new work, so no watcher fires.
                stored = BlackboardEntry(
                    key=f"swarm.{board.run_id}.msg.{entry['seq']}",
                    value=dict(entry), author=entry["author"],
                    topic=board.run_id,
                    metadata={"seq": entry["seq"],
                              "kind": entry["kind"]})
                stored.created_at = entry["ts"]
                stored.updated_at = entry["ts"]
                board._board._entries[stored.key] = stored
        verifiers = record.get("verifiers") or {}
        if isinstance(verifiers, dict):
            for name, group in verifiers.items():
                if isinstance(group, list):
                    board._verifiers[str(name)] = [str(g) for g in group]
        breakers = record.get("breakers") or {}
        if isinstance(breakers, dict):
            for name, group in breakers.items():
                if isinstance(group, list):
                    board._breakers[str(name)] = [str(g) for g in group]
        board._last_activity_at = float(
            record.get("last_activity_at") or board._last_activity_at)
        return board


class SwarmBoardStore:
    """Durable swarm boards, one atomic JSON index.

    Shape follows WorkflowStore: tmp+replace writes; a corrupt
    store is quarantined and degrades to empty with a named
    reason; bad rows are skipped; reads never raise.
    ``path=None`` means in-memory only (tests, and runtimes with
    no home); the record shape is identical either way.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._lock = threading.RLock()
        self._boards: dict[str, dict[str, Any]] = {}
        self._degraded = ""
        if self.path is not None:
            self._load()

    @property
    def degraded_reason(self) -> str:
        """Why this store degraded, or "" when it is healthy."""
        return self._degraded

    def _load(self) -> None:
        assert self.path is not None
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            self._quarantine(exc)
            return
        if not isinstance(raw, dict):
            self._quarantine(ValueError(
                f"swarm boards store must hold an object, found "
                f"{type(raw).__name__}"))
            return
        for row in raw.get("boards", []) or []:
            if not isinstance(row, dict):
                continue
            if not isinstance(row.get("id"), str) or not row["id"]:
                continue
            record = {
                "id": row["id"],
                "goal": str(row.get("goal", "")),
                "state": (str(row.get("state"))
                          if str(row.get("state")) in BOARD_STATES
                          else "active"),
                "chat_key": str(row.get("chat_key", "")),
                "members": row.get("members")
                if isinstance(row.get("members"), list) else [],
                "briefs": row.get("briefs")
                if isinstance(row.get("briefs"), list) else [],
                "feed": row.get("feed")
                if isinstance(row.get("feed"), list) else [],
                "verifiers": row.get("verifiers")
                if isinstance(row.get("verifiers"), dict) else {},
                "created_at": float(row.get("created_at") or 0.0),
                "updated_at": float(row.get("updated_at") or 0.0),
                "last_activity_at": float(
                    row.get("last_activity_at") or 0.0),
                "last_wake_at": float(row.get("last_wake_at") or 0.0),
            }
            self._boards[record["id"]] = record

    def _quarantine(self, exc: Exception) -> None:
        assert self.path is not None
        bad = self.path.with_name(
            f"{self.path.name}.corrupt-{int(time.time())}")
        try:
            os.replace(self.path, bad)
        except OSError:
            bad = self.path
        self._degraded = (
            f"swarm boards store is corrupt — quarantined to {bad}: {exc}")

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            self.path,
            json.dumps({"boards": list(self._boards.values())},
                       ensure_ascii=False, indent=1))

    def save_board(self, board: SwarmBoard) -> dict[str, Any]:
        """Upsert a board's record. Returns a copy of the record."""
        record = board.to_record()
        with self._lock:
            self._boards[record["id"]] = copy.deepcopy(record)
            with contextlib.suppress(OSError):
                self._save()
        return copy.deepcopy(record)

    def get_board(self, run_id: str) -> SwarmBoard | None:
        """The board for a run id, rebuilt from its record."""
        with self._lock:
            record = self._boards.get(str(run_id))
            if record is None:
                return None
            return SwarmBoard.from_record(copy.deepcopy(record))

    def resolve(self, ref: str) -> SwarmBoard | None:
        """Resolve by exact id or unique id prefix.

        An ambiguous prefix resolves to nothing — the caller names
        the board it means instead of guessing.
        """
        ref = (ref or "").strip()
        if not ref:
            return None
        with self._lock:
            if ref in self._boards:
                return SwarmBoard.from_record(
                    copy.deepcopy(self._boards[ref]))
            matches = [r for r in self._boards.values()
                       if r["id"].startswith(ref)]
            if len(matches) == 1:
                return SwarmBoard.from_record(copy.deepcopy(matches[0]))
        return None

    def list_boards(self) -> list[dict[str, Any]]:
        """All records, oldest first."""
        with self._lock:
            records = sorted(self._boards.values(),
                             key=lambda r: r.get("created_at", 0.0))
        return copy.deepcopy(records)

    def mark_done(self, run_id: str) -> dict[str, Any] | None:
        """Mark a board done. Returns the record, or None."""
        with self._lock:
            record = self._boards.get(str(run_id))
            if record is None:
                return None
            record["state"] = "done"
            record["updated_at"] = time.time()
            with contextlib.suppress(OSError):
                self._save()
            return copy.deepcopy(record)

    def mark_failed(self, run_id: str) -> dict[str, Any] | None:
        """Mark a board failed. Returns the record, or None."""
        with self._lock:
            record = self._boards.get(str(run_id))
            if record is None:
                return None
            record["state"] = "failed"
            record["updated_at"] = time.time()
            with contextlib.suppress(OSError):
                self._save()
            return copy.deepcopy(record)


# ── the monitor tick ─────────────────────────────────────────────

def ensure_swarm_board_job(scheduler: Any, *,
                           spec: str = SWARM_BOARD_SPEC) -> Any:
    """Register the durable swarm-board tick on the live scheduler.

    Mirror of the digest / watch-loop / drift / workflow ensures:
    a routine tool job, one per scheduler, replaced when its spec
    goes stale. Never raises — scheduling is best-effort; a
    missing or broken scheduler yields None.
    """
    if scheduler is None:
        return None
    try:
        for job in (scheduler.list_jobs() or []):
            name = (job.get("name") if isinstance(job, dict)
                    else getattr(job, "name", None))
            if name != SWARM_BOARD_JOB:
                continue
            job_spec = (job.get("spec") if isinstance(job, dict)
                        else getattr(job, "spec", None))
            if job_spec == spec:
                return job
            with contextlib.suppress(Exception):
                job_id = (job.get("id") if isinstance(job, dict)
                          else getattr(job, "id", None))
                scheduler.remove(job_id)
            break
        return scheduler.add(SWARM_BOARD_JOB, spec, "tool",
                             {"tool": SWARM_BOARD_TOOL, "args": {}},
                             routine=True)
    except Exception:  # noqa: BLE001 - scheduling is best-effort
        return None


def run_swarm_board_tick_once(context: Any, *,
                              notifier: Any = None,
                              store: SwarmBoardStore | None = None
                              ) -> dict[str, Any]:
    """Run one swarm-board tick. Never raises.

    For every board still ``active``: if it went quiet past the
    stall window with unfinished work, post one `@monitor`
    wake-up naming the quietest unfinished worker, and tell the
    owner once per stall episode (``last_wake_at`` rate-limits
    it — a second tick inside the window does not re-wake).
    """
    try:
        store = store if store is not None else store_for_context(context)
    except Exception as exc:  # noqa: BLE001 - the tick never dies
        record_failure("swarm_board.tick.setup", exc)
        return {"ok": False, "error": f"swarm board tick failed: {exc}"}
    checked = 0
    woken = 0
    try:
        for record in store.list_boards():
            if record.get("state") != "active":
                continue
            checked += 1
            board = store.get_board(record["id"])
            if board is None:
                continue
            now = time.time()
            stall = board.stall_check(now=now)
            if not stall["stalled"]:
                continue
            if board.last_wake_at and \
                    now - board.last_wake_at < WAKE_COOLDOWN_S:
                continue
            # Name every unfinished worker: the wake-up is addressed
            # to the whole stalled team, not one scapegoat.
            named = ", ".join("@" + n for n in stall["unfinished"])
            text = (
                f"@all stalled — no board activity for "
                f"{stall['quiet_for_s']:.0f}s with work unfinished. "
                f"{named}, can you get this moving again?")
            board.post("monitor", text, kind="wake")
            board.last_wake_at = now
            store.save_board(board)
            woken += 1
            if notifier is not None:
                try:
                    notifier.publish(
                        "swarm-board",
                        title="Swarm stalled",
                        body=text)
                except Exception as exc:  # noqa: BLE001 - delivery is best-effort
                    record_failure(
                        f"swarm_board:{board.run_id}:wake", exc)
        return {"ok": True, "checked": checked, "woken": woken}
    except Exception as exc:  # noqa: BLE001 - the tick never dies
        record_failure("swarm_board.tick", exc)
        return {"ok": False, "error": f"swarm board tick failed: {exc}"}
