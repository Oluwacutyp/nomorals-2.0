"""Themed community groups: chat rooms with topics, posts, and membership.

#84 — the Bumble BFF → Geneva pivot signal: 1:1 matching didn't
retain; groups and events do. A themed group is a named room
("Lagos Devs", "Weekend Hikers") with a topic, a member roster, and a
post feed. Groups are community-scoped (not chat-scoped): the same
"Lagos Devs" group is visible from any chat.

Geneva-grade upgrades (mined 2026-10-10):
* **Roles** — owner / admin / member. The creator is owner; roles gate
  announcements, pins, promotions. (Discord/Geneva pattern.)
* **Announcements** — owner/admin posts rendered first, always visible.
* **Pinned posts** — pin/unpin any post; pins render in the show card.
* **Search** — full-text over name/topic, so groups are discoverable.
* **Welcome message + join gate** — a welcome shown on join; an optional
  join question whose answer is recorded (Meetup screening-question pattern).
* **Event linkage** — ``show`` surfaces the group's upcoming events
  (lazy import of events.py: no cycles, no import at module level).
* **Reputation** — per-member reliability from event history (+1 showed,
  −1 no-showed; the real Meetup attendance-guideline pattern).
* **Icebreakers** — quiet-group nudges suggest a random icebreaker prompt.

Isolation: this module imports stdlib + ``nomorals.core`` only.
No memory, no accounts, no vaults, no connectors — the package
docstring is the law, ``tests/test_community_isolation.py`` the gate.

Chat: ``/cgroup new <name> | <topic>``, ``/cgroup join <group_id>``,
``/cgroup list``, ``/cgroup post <group_id> <text>``,
``/cgroup show <group_id>``, plus ``announce``, ``pin``, ``role``,
``search``, ``events``, ``rep``.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger

_log = get_logger("nomorals.community.groups")

_DATA_DIR = Path.home() / ".devon" / "community" / "themed_groups"
_POSTS_PER_GROUP = 200
_QUIET_DAYS = 7.0

ROLE_OWNER = "owner"
ROLE_ADMIN = "admin"
ROLE_MEMBER = "member"
_ROLES = (ROLE_OWNER, ROLE_ADMIN, ROLE_MEMBER)

_ICEBREAKERS = (
    "What's everyone working on this week? 🚀",
    "Drop one song that's been on repeat for you 🎧",
    "What's the best thing you ate this week? 🍲",
    "Share a photo of your view right now 📸",
    "What's one thing you're looking forward to this month?",
    "Recommend something — book, show, tool, anything 📚",
    "What did you learn this week that surprised you?",
    "Weekend plans? Anyone doing something fun?",
    "Hot take time: what's an unpopular opinion you stand by? 🌶️",
    "What's a small win you had recently? Let's celebrate 🎉",
)


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "group"


def _now() -> float:
    return time.time()


# ── data model ─────────────────────────────────────────────────────────────


@dataclass
class GroupPost:
    """One post in a themed group's feed."""

    id: str
    group_id: str
    author: str
    text: str
    created_at: float = 0.0
    announcement: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GroupPost":
        return cls(
            id=str(data.get("id", "")),
            group_id=str(data.get("group_id", "")),
            author=str(data.get("author", "")),
            text=str(data.get("text", "")),
            created_at=float(data.get("created_at") or 0.0),
            announcement=bool(data.get("announcement", False)),
        )


@dataclass
class ThemedGroup:
    """A themed room: name, topic, members, roles, and a post feed."""

    id: str
    name: str
    topic: str
    members: list[str] = field(default_factory=list)
    created_by: str = ""
    created_at: float = 0.0
    # Geneva-grade fields — all defaulted, old JSON keeps loading fine.
    roles: dict[str, str] = field(default_factory=dict)  # member → owner/admin/member
    welcome: str = ""  # shown to new joiners
    join_question: str = ""  # optional gate question; answer recorded
    join_answers: dict[str, str] = field(default_factory=dict)
    pinned: list[str] = field(default_factory=list)  # post ids

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThemedGroup":
        roles = {str(k): str(v) for k, v in (data.get("roles") or {}).items()
                 if str(v) in _ROLES}
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            topic=str(data.get("topic", "")),
            members=[str(m) for m in (data.get("members") or [])],
            created_by=str(data.get("created_by", "")),
            created_at=float(data.get("created_at") or 0.0),
            roles=roles,
            welcome=str(data.get("welcome", "")),
            join_question=str(data.get("join_question", "")),
            join_answers={str(k): str(v) for k, v in (data.get("join_answers") or {}).items()},
            pinned=[str(p) for p in (data.get("pinned") or [])],
        )

    def role_of(self, member: str) -> str:
        """Effective role; the creator is owner even if roles weren't stored."""
        if member in self.roles:
            return self.roles[member]
        if member and member == self.created_by:
            return ROLE_OWNER
        return ROLE_MEMBER if member in self.members else ""


class GroupStore:
    """Community-wide themed-group registry. JSON files, never raises."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else _DATA_DIR

    def _groups_path(self) -> Path:
        return self.data_dir / "groups.json"

    def _posts_path(self, group_id: str) -> Path:
        return self.data_dir / f"posts_{_sanitize(group_id)}.json"

    def _load_groups(self) -> list[ThemedGroup]:
        try:
            raw = json.loads(self._groups_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        groups: list[ThemedGroup] = []
        for item in raw if isinstance(raw, list) else []:
            try:
                g = ThemedGroup.from_dict(item)
                if g.id and g.name:
                    groups.append(g)
            except (TypeError, ValueError, AttributeError):
                continue
        return groups

    def _save_groups(self, groups: list[ThemedGroup]) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._groups_path().with_suffix(".tmp")
            tmp.write_text(
                json.dumps([g.to_dict() for g in groups],
                           ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            tmp.replace(self._groups_path())
        except OSError as exc:
            _log.warning("themed-group save failed: %s", exc)

    def _load_posts(self, group_id: str) -> list[GroupPost]:
        try:
            raw = json.loads(self._posts_path(group_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        posts: list[GroupPost] = []
        for item in raw if isinstance(raw, list) else []:
            try:
                p = GroupPost.from_dict(item)
                if p.id and p.group_id:
                    posts.append(p)
            except (TypeError, ValueError, AttributeError):
                continue
        return posts

    def _save_posts(self, group_id: str, posts: list[GroupPost]) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._posts_path(group_id).with_suffix(".tmp")
            tmp.write_text(
                json.dumps([p.to_dict() for p in posts],
                           ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            tmp.replace(self._posts_path(group_id))
        except OSError as exc:
            _log.warning("group-posts save failed: %s", exc)

    def _mutate(self, group_id: str, fn) -> ThemedGroup | None:
        """Load → apply fn(group) → save → return group, or None."""
        groups = self._load_groups()
        for g in groups:
            if g.id == group_id:
                try:
                    fn(g)
                except Exception as exc:  # noqa: BLE001 — mutation must not blow up
                    _log.warning("group mutation failed: %s", exc)
                    return None
                self._save_groups(groups)
                return g
        return None

    # ── groups ─────────────────────────────────────────────────────────

    def create_group(self, name: str, topic: str, created_by: str = "") -> ThemedGroup | None:
        """Create a themed room. Returns None when the name is empty."""
        name = (name or "").strip()
        if not name:
            return None
        now = _now()
        group = ThemedGroup(
            id="grp_" + new_short_id(),
            name=name[:80],
            topic=(topic or "").strip()[:200],
            members=[created_by] if created_by else [],
            created_by=created_by,
            created_at=now,
            roles={created_by: ROLE_OWNER} if created_by else {},
        )
        groups = self._load_groups()
        groups.append(group)
        self._save_groups(groups)
        return group

    def get(self, group_id: str) -> ThemedGroup | None:
        for g in self._load_groups():
            if g.id == group_id:
                return g
        return None

    def list_groups(self) -> list[ThemedGroup]:
        return sorted(self._load_groups(), key=lambda g: g.created_at)

    def search(self, query: str) -> list[ThemedGroup]:
        """Full-text search over group names and topics."""
        q = (query or "").strip().lower()
        if not q:
            return self.list_groups()
        return [g for g in self._load_groups()
                if q in g.name.lower() or q in g.topic.lower()]

    def join_group(self, group_id: str, member: str, answer: str = "") -> str | None:
        """Add a member. Returns the welcome text (or gate question), or None.

        Returning None means the group doesn't exist / member empty.
        When the group has a join question, the member is added and the
        question is returned so the chat layer can ask it; the answer is
        recorded via :meth:`answer_join_question`.
        """
        if not member:
            return None
        result: dict[str, Any] = {}

        def _add(g: ThemedGroup) -> None:
            if member not in g.members:
                g.members.append(member)
            if member not in g.roles:
                g.roles[member] = ROLE_MEMBER
            if answer and g.join_question:
                g.join_answers[member] = answer[:500]
            result["welcome"] = g.welcome
            result["question"] = g.join_question

        g = self._mutate(group_id, _add)
        if g is None:
            return None
        welcome = result.get("welcome") or ""
        question = result.get("question") or ""
        if question and not answer:
            return f"Welcome! 🎉\n{question}\n(answer with: /cgroup answer {group_id} <your answer>)"
        return f"Welcome! 🎉\n{welcome}" if welcome else "Welcome! 🎉"

    def answer_join_question(self, group_id: str, member: str, answer: str) -> bool:
        if not member or not (answer or "").strip():
            return False

        def _ans(g: ThemedGroup) -> None:
            if member in g.members and g.join_question:
                g.join_answers[member] = answer.strip()[:500]

        return self._mutate(group_id, _ans) is not None

    def leave_group(self, group_id: str, member: str) -> bool:
        outcome = {"left": False}

        def _leave(g: ThemedGroup) -> None:
            if member in g.members:
                g.members.remove(member)
                g.roles.pop(member, None)
                g.join_answers.pop(member, None)
                outcome["left"] = True

        g = self._mutate(group_id, _leave)
        return g is not None and outcome["left"]

    def member_groups(self, member: str) -> list[ThemedGroup]:
        """All groups a member belongs to — the warm pool for discovery."""
        return [g for g in self._load_groups() if member in g.members]

    # ── roles ──────────────────────────────────────────────────────────

    def set_role(self, group_id: str, actor: str, member: str,
                 role: str) -> tuple[bool, str]:
        """Promote/demote. Owner-only; the owner can't be demoted."""
        role = (role or "").lower()
        if role not in _ROLES:
            return False, f"role must be one of: {', '.join(_ROLES)}"

        outcome: dict[str, Any] = {"ok": False, "msg": ""}

        def _set(g: ThemedGroup) -> None:
            if g.role_of(actor) != ROLE_OWNER:
                outcome["msg"] = "only the group owner can change roles"
                return
            if member not in g.members:
                outcome["msg"] = f"{member} isn't in this group"
                return
            if member == g.created_by and role != ROLE_OWNER:
                outcome["msg"] = "the founder keeps the owner role"
                return
            g.roles[member] = role
            outcome["ok"] = True
            outcome["msg"] = f"{member} is now {role}"

        if self._mutate(group_id, _set) is None:
            return False, "no such group"
        return outcome["ok"], outcome["msg"] or "role updated"

    def set_welcome(self, group_id: str, actor: str, text: str) -> bool:
        def _w(g: ThemedGroup) -> None:
            if g.role_of(actor) in (ROLE_OWNER, ROLE_ADMIN):
                g.welcome = (text or "").strip()[:500]

        g = self._mutate(group_id, _w)
        return g is not None and g.role_of(actor) in (ROLE_OWNER, ROLE_ADMIN)

    def set_join_question(self, group_id: str, actor: str, text: str) -> bool:
        def _q(g: ThemedGroup) -> None:
            if g.role_of(actor) in (ROLE_OWNER, ROLE_ADMIN):
                g.join_question = (text or "").strip()[:300]

        g = self._mutate(group_id, _q)
        return g is not None and g.role_of(actor) in (ROLE_OWNER, ROLE_ADMIN)

    # ── posts ──────────────────────────────────────────────────────────

    def post(self, group_id: str, author: str, text: str,
             announcement: bool = False) -> GroupPost | None:
        """Post to a group feed. Author must be a member. Never raises."""
        text = (text or "").strip()
        if not text or not author:
            return None
        group = self.get(group_id)
        if group is None or author not in group.members:
            return None
        if announcement and group.role_of(author) not in (ROLE_OWNER, ROLE_ADMIN):
            return None  # announcements are role-gated
        p = GroupPost(
            id="post_" + new_short_id(),
            group_id=group_id,
            author=author,
            text=text[:2000],
            created_at=_now(),
            announcement=announcement,
        )
        posts = self._load_posts(group_id)
        posts.append(p)
        self._save_posts(group_id, posts[-_POSTS_PER_GROUP:])
        return p

    def announce(self, group_id: str, author: str, text: str) -> GroupPost | None:
        """Owner/admin announcement — rendered first in the feed."""
        return self.post(group_id, author, text, announcement=True)

    def pin(self, group_id: str, actor: str, post_id: str) -> bool:
        def _pin(g: ThemedGroup) -> None:
            if g.role_of(actor) not in (ROLE_OWNER, ROLE_ADMIN):
                raise PermissionError("role-gated")
            posts = {p.id for p in self._load_posts(group_id)}
            if post_id in posts and post_id not in g.pinned:
                g.pinned.append(post_id)

        try:
            return self._mutate(group_id, _pin) is not None
        except PermissionError:
            return False

    def unpin(self, group_id: str, actor: str, post_id: str) -> bool:
        def _unpin(g: ThemedGroup) -> None:
            if g.role_of(actor) not in (ROLE_OWNER, ROLE_ADMIN):
                raise PermissionError("role-gated")
            if post_id in g.pinned:
                g.pinned.remove(post_id)

        try:
            return self._mutate(group_id, _unpin) is not None
        except PermissionError:
            return False

    def pinned_posts(self, group_id: str) -> list[GroupPost]:
        group = self.get(group_id)
        if group is None:
            return []
        by_id = {p.id: p for p in self._load_posts(group_id)}
        return [by_id[pid] for pid in group.pinned if pid in by_id]

    def feed(self, group_id: str, limit: int = 20) -> list[GroupPost]:
        return self._load_posts(group_id)[-max(1, limit):]

    def last_post_at(self, group_id: str) -> float:
        posts = self._load_posts(group_id)
        return max((p.created_at for p in posts), default=0.0)

    def quiet_groups(self, days: float = _QUIET_DAYS,
                     now: float | None = None) -> list[ThemedGroup]:
        """Groups with no posts in ``days`` — candidates for a nudge."""
        now = now if now is not None else _now()
        cutoff = now - days * 86400.0
        return [g for g in self._load_groups()
                if g.members and self.last_post_at(g.id) < cutoff]

    def member_activity(self, group_id: str) -> list[tuple[str, int]]:
        """Post counts per member, descending — the group's loudest voices."""
        counts: dict[str, int] = {}
        for p in self._load_posts(group_id):
            counts[p.author] = counts.get(p.author, 0) + 1
        return sorted(counts.items(), key=lambda kv: -kv[1])

    # ── cross-module (lazy, no import cycles) ──────────────────────────

    def upcoming_events(self, group_id: str, limit: int = 5,
                        events_store=None, now: float | None = None) -> list:
        """This group's upcoming events (events.py is the source of truth)."""
        try:
            if events_store is None:
                from .events import EventStore
                events_store = EventStore()
            return events_store.list_events(group_id, upcoming_only=True,
                                            now=now)[:max(1, limit)]
        except Exception:  # noqa: BLE001 — cross-module, never breaks groups
            return []

    def member_reputation(self, member: str, events_store=None,
                          meetups_store=None) -> dict[str, Any]:
        """Reliability score from event history: +1 showed, −1 no-showed.

        The real Meetup attendance-guideline pattern, computed from this
        module's own stores (never owner data).
        """
        shows = 0
        no_shows = 0
        try:
            if events_store is None:
                from .events import EventStore
                events_store = EventStore()
            for ev in events_store._load():
                if ev.rsvps.get(member) != "yes":
                    continue
                if meetups_store is None:
                    from .meetups import MeetupStore
                    meetups_store = MeetupStore()
                arrived = meetups_store.checked_in(ev.id)
                if member in arrived:
                    shows += 1
                elif ev.is_past():
                    no_shows += 1
        except Exception:  # noqa: BLE001
            pass
        score = shows - no_shows
        label = ("🌟 reliable" if score >= 3 else "✅ solid" if score >= 1
                 else "⚠️ flaky" if score <= -1 else "🆕 new")
        return {"member": member, "score": score, "shows": shows,
                "no_shows": no_shows, "label": label}


# ── warm-start discovery ───────────────────────────────────────────────────


def warm_matches(store: GroupStore, member: str,
                 top_n: int = 5) -> list[tuple[str, int]]:
    """People sharing context with ``member`` via common groups.

    Sorted by number of shared groups, descending. The warm pool for
    introductions — meet people through groups you're already in.
    """
    mine = {g.id for g in store.member_groups(member)}
    counts: dict[str, int] = {}
    for g in store.list_groups():
        if g.id in mine:
            for m in g.members:
                if m != member:
                    counts[m] = counts.get(m, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])[:max(1, top_n)]


def icebreaker(seed: int | None = None) -> str:
    """A random conversation starter for quiet groups."""
    rng = random.Random(seed)
    return rng.choice(_ICEBREAKERS)


# ── rendering ──────────────────────────────────────────────────────────────


def format_group_card(store: GroupStore, group: ThemedGroup,
                      post_limit: int = 5, events_store=None) -> str:
    """God-tier group card: header, roles, pins, announcements, events."""
    g = group
    lines = [f"🏠 **{g.name}**", f"   _{g.topic or 'no topic yet'}_",
             f"   👥 {len(g.members)} members"]
    owners = [m for m in g.members if g.role_of(m) == ROLE_OWNER]
    admins = [m for m in g.members if g.role_of(m) == ROLE_ADMIN]
    if owners:
        lines.append(f"   👑 {', '.join(owners)}")
    if admins:
        lines.append(f"   🛡️ {', '.join(admins[:5])}")
    pins = store.pinned_posts(g.id)
    if pins:
        lines.append("   📌 Pinned:")
        for p in pins[:3]:
            lines.append(f"      {p.author}: {p.text[:100]}")
    events = store.upcoming_events(g.id, events_store=events_store)
    if events:
        lines.append("   📍 Upcoming:")
        for ev in events[:3]:
            when = time.strftime("%a %m-%d %H:%M", time.localtime(ev.starts_at))
            lines.append(f"      {when} — {ev.title} (`{ev.id}`)")
    lines.append("   ── recent ──")
    for p in store.feed(g.id, post_limit):
        mark = "📢" if p.announcement else "·"
        lines.append(f"   {mark} {p.author}: {p.text[:120]}")
    lines.append(f"   `{g.id}`")
    return "\n".join(lines)


# ── chat ─────────────────────────────────────────────────────────────────


def _usage() -> str:
    return (
        "/cgroup new <name> | <topic> · join <group_id> · list · search <q>\n"
        "/cgroup post <group_id> <text> · announce <group_id> <text>\n"
        "/cgroup show <group_id> · pin <group_id> <post_id> · unpin …\n"
        "/cgroup role <group_id> <member> <owner|admin|member>\n"
        "/cgroup welcome <group_id> <text> · events <group_id>\n"
        "/cgroup rep <member> · nudge · discover <member>\n"
        "Also: /cgroup event … · /cgroup meetup … — see events.py / meetups.py."
    )


def control_cgroup(tail: str, context=None, chat=None,
                   sender: str = "", sender_id: str = "") -> str:
    """Chat entry point for groups + events + meetups. Never raises."""
    try:
        return _control_cgroup(tail, context, chat, sender, sender_id)
    except Exception as exc:  # noqa: BLE001 — chat must never blow up
        _log.warning("cgroup control failed: %s", exc)
        return "Community groups hit a snag — try again."


def _control_cgroup(tail: str, context, chat,
                    sender: str, sender_id: str) -> str:
    from . import events as _events
    from . import meetups as _meetups

    store = GroupStore()
    who = (sender or sender_id or "anon").strip() or "anon"
    parts = (tail or "").strip().split(None, 1)
    if not parts:
        return _usage()
    cmd, rest = parts[0].lower(), (parts[1] if len(parts) > 1 else "")

    if cmd in ("event", "meetup"):
        module = _events if cmd == "event" else _meetups
        return module.control(tail, context, chat, sender, sender_id)

    if cmd == "new":
        name, _, topic = rest.partition("|")
        g = store.create_group(name.strip(), topic.strip(), created_by=who)
        if g is None:
            return "Give the group a name: /cgroup new <name> | <topic>"
        return (f"🏠 Group created: **{g.name}** (`{g.id}`)\n"
                f"Topic: {g.topic or '—'}\n"
                f"You're the 👑 owner. Set a welcome: /cgroup welcome {g.id} <text>")

    if cmd == "join":
        segs = rest.split(None, 1)
        gid = segs[0] if segs else ""
        g = store.get(gid)
        if g is None:
            return f"No group `{gid}`. /cgroup list to see them."
        msg = store.join_group(gid, who)
        n = len(store.get(gid).members) if store.get(gid) else 0
        note = f"\n{msg}" if msg else ""
        return f"Welcome to **{g.name}**! ({n} member{'s' if n != 1 else ''}){note}"

    if cmd == "answer":
        segs = rest.split(None, 2)
        if len(segs) < 3:
            return "Usage: /cgroup answer <group_id> <your answer>"
        ok = store.answer_join_question(segs[0], who, segs[2])
        return "Answer recorded. 👍" if ok else "Couldn't record that answer."

    if cmd == "leave":
        if store.leave_group(rest.strip(), who):
            return "Left the group."
        return "Couldn't leave that group."

    if cmd == "list":
        groups = store.list_groups()
        if not groups:
            return "No groups yet. Create one: /cgroup new Lagos Devs | build things together"
        lines = ["🏠 **Community groups:**"]
        for g in groups:
            mark = "✅" if who in g.members else "·"
            role = g.role_of(who)
            rg = f" { {'owner': '👑', 'admin': '🛡️'}.get(role, '')}" if who in g.members else ""
            lines.append(f"{mark} **{g.name}** `{g.id}` — {g.topic or 'no topic'} "
                         f"({len(g.members)} members){rg}")
        return "\n".join(lines)

    if cmd == "search":
        groups = store.search(rest)
        if not groups:
            return f"No groups matching {rest!r}."
        lines = [f"🔍 **Groups matching {rest!r}:**"]
        for g in groups[:10]:
            lines.append(f"· **{g.name}** `{g.id}` — {g.topic or 'no topic'} "
                         f"({len(g.members)})")
        return "\n".join(lines)

    if cmd == "post":
        gid, _, text = rest.partition(" ")
        p = store.post(gid.strip(), who, text)
        if p is None:
            return "Post failed — check the group id and that you've joined: /cgroup join <group_id>"
        return f"📝 Posted to `{gid.strip()}` (`{p.id}`)."

    if cmd == "announce":
        gid, _, text = rest.partition(" ")
        p = store.announce(gid.strip(), who, text)
        if p is None:
            return "Announcements are for owners/admins — and you need to be a member."
        return "📢 Announcement posted."

    if cmd in ("pin", "unpin"):
        segs = rest.split(None, 1)
        if len(segs) < 2:
            return f"Usage: /cgroup {cmd} <group_id> <post_id>"
        ok = store.pin(segs[0], who, segs[1]) if cmd == "pin" else store.unpin(segs[0], who, segs[1])
        return ("📌 Pinned." if cmd == "pin" else "📌 Unpinned.") if ok else \
            f"Couldn't {cmd} — owners/admins only, check the ids."

    if cmd == "role":
        segs = rest.split()
        if len(segs) < 3:
            return "Usage: /cgroup role <group_id> <member> <owner|admin|member>"
        ok, msg = store.set_role(segs[0], who, segs[1], segs[2])
        return ("✅ " if ok else "⚠️ ") + msg

    if cmd == "welcome":
        gid, _, text = rest.partition(" ")
        if not gid or not text.strip():
            return "Usage: /cgroup welcome <group_id> <text>"
        ok = store.set_welcome(gid.strip(), who, text)
        return "Welcome message set. 👋" if ok else "Only owners/admins can set the welcome."

    if cmd == "gate":
        gid, _, text = rest.partition(" ")
        if not gid:
            return "Usage: /cgroup gate <group_id> <question>  (empty question clears)"
        ok = store.set_join_question(gid.strip(), who, text)
        return ("Join question set. 🚪" if text.strip() else "Join question cleared.") \
            if ok else "Only owners/admins can set the join question."

    if cmd == "show":
        g = store.get(rest.strip())
        if g is None:
            return "No such group."
        return format_group_card(store, g)

    if cmd == "events":
        g = store.get(rest.strip())
        if g is None:
            return "No such group."
        evs = store.upcoming_events(g.id)
        if not evs:
            return f"No upcoming events in **{g.name}**. Create one: /cgroup event new …"
        from .events import EventStore
        lines = [f"📍 **Upcoming in {g.name}:**"]
        lines.extend(EventStore.format_event(e) for e in evs)
        return "\n".join(lines)

    if cmd == "rep":
        member = rest.strip() or who
        r = store.member_reputation(member)
        return (f"{r['label']} **{member}** — reliability {r['score']:+d} "
                f"({r['shows']} showed · {r['no_shows']} no-showed)")

    if cmd == "activity":
        g = store.get(rest.strip())
        if g is None:
            return "No such group."
        top = store.member_activity(g.id)[:10]
        if not top:
            return "No posts yet."
        lines = [f"📊 **Most active in {g.name}:**"]
        for m, c in top:
            lines.append(f"  {m} — {c} post{'s' if c != 1 else ''}")
        return "\n".join(lines)

    if cmd == "nudge":
        quiet = store.quiet_groups()
        if not quiet:
            return "Every group is active. No nudges needed. 💪"
        lines = ["😴 **Quiet groups** (no posts in 7 days) — your turn to say hi:"]
        for g in quiet[:5]:
            lines.append(f"  **{g.name}** `{g.id}`\n"
                         f"  💡 Try: “{icebreaker()}”")
        return "\n".join(lines)

    if cmd == "discover":
        member = rest.strip() or who
        top = warm_matches(store, member)
        if not top:
            return f"No shared-context matches for {member} yet."
        lines = [f"🤝 **Warm matches for {member}** (shared groups):"]
        for m, c in top:
            lines.append(f"  {m} — {c} shared group{'s' if c != 1 else ''}")
        return "\n".join(lines)

    return _usage()
