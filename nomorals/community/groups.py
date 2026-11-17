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

Isolation: this module imports stdlib + ``nomorals.core`` only —
plus ``nomorals.i18n`` (Wave 7), whose catalogs are static string
data, plus the in-package ``.digest`` organ (itself stdlib + core
only): no owner data crosses that seam in either direction.
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
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from .digest import (
    DEFAULT_RITUALS,
    Ritual,
    RitualScheduler,
    build_digest,
    next_occurrences,
    render_digest,
    render_ritual,
)

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


def _default_rituals(group_id: str) -> list[Ritual]:
    """The five default rituals, stamped with the group's id."""
    return [replace(r, group_id=group_id) for r in DEFAULT_RITUALS]


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
    language: str = ""  # i18n locale for membership messages; "" = default
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
            language=str(data.get("language", "")),
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

    def _rituals_path(self, group_id: str) -> Path:
        return self.data_dir / f"rituals_{_sanitize(group_id)}.json"

    def load_rituals(self, group_id: str) -> list[Ritual]:
        """Rituals for a group — the saved set, or the defaults when
        none was saved yet. A corrupt file (or all-bad rows) degrades
        to the defaults; never raises."""
        try:
            raw = json.loads(
                self._rituals_path(group_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return _default_rituals(group_id)
        rituals: list[Ritual] = []
        for item in raw if isinstance(raw, list) else []:
            try:
                r = Ritual.from_dict(item)
                if r.name:
                    rituals.append(r)
            except (TypeError, ValueError, AttributeError):
                continue
        return rituals or _default_rituals(group_id)

    def save_rituals(self, group_id: str, rituals: list[Ritual]) -> None:
        """Atomic save (tmp+replace, same shape as _save_posts).
        Never raises."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._rituals_path(group_id).with_suffix(".tmp")
            tmp.write_text(
                json.dumps([r.to_dict() for r in rituals],
                           ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            tmp.replace(self._rituals_path(group_id))
        except OSError as exc:
            _log.warning("group-rituals save failed: %s", exc)

    def system_post(self, group_id: str, text: str) -> GroupPost | None:
        """Post as Devon herself — the ritual delivery path. Bypasses
        the membership check (Devon is not a member of themed groups);
        same atomic save and 200-post cap as post(). The member-post
        guard is untouched. Never raises."""
        text = (text or "").strip()
        if not text:
            return None
        group = self.get(group_id)
        if group is None:
            return None
        p = GroupPost(
            id="post_" + new_short_id(),
            group_id=group_id,
            author="Devon",
            text=text[:2000],
            created_at=_now(),
        )
        posts = self._load_posts(group_id)
        posts.append(p)
        self._save_posts(group_id, posts[-_POSTS_PER_GROUP:])
        return p

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

    def set_language(self, group_id: str, language: str) -> bool:
        """Set the group's i18n locale for membership messages.

        The tag is stored normalized (``pt_br`` → ``pt-BR``); an
        unusable tag clears the setting back to the default.
        Returns False when the group does not exist.
        """
        from ..i18n.locale import normalize_locale

        def _lang(g: ThemedGroup) -> None:
            g.language = normalize_locale(language)

        return self._mutate(group_id, _lang) is not None

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
        if announcement:
            # Announcements carry the group's semantic marker + lead
            # emoji (group_extras, AA-40): the stored post renders
            # marked in every feed.  Lazy import, same pattern as the
            # events linkage below — no module-level cycle.
            from .group_extras import mark_semantics

            text = mark_semantics(text, "announcement")
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
        "/cgroup welcome <group_id> <text> · lang <group_id> <locale> · events <group_id>\n"
        "/cgroup rep <member> · nudge · discover <member>\n"
        "/cgroup memory <group_id> remember <key> = <value> · recall <key> · "
        "search <query> · list · forget <key> · persona\n"
        "/cgroup digest <group_id> · rituals <group_id> [list|run <name>|enable|disable|add <name> | <HH:MM|DAY HH:MM> | <message>]\n"
        "/cgroup schedule <minutes> <text> · posts · devices (AA-18 orchestration)\n"
        "Also: /cgroup event … · /cgroup meetup … — see events.py / meetups.py."
    )


def _group_message(key: str, grp: ThemedGroup | None, *,
                   count: float | None = None, **params: Any) -> str:
    """Render a membership message from the i18n catalogs (Wave 7).

    The locale is the group's stored ``language`` (default when
    unset).  When a fallback serves — the group's locale has no
    catalog row and a chain relative or the source string served —
    that fact is recorded in the log with both locales named; the
    chat text itself is just the message, as members should see it.
    """
    from ..i18n import translate_detailed

    locale = grp.language if grp is not None else ""
    rendered = translate_detailed(key, locale or None,
                                  count=count, **params)
    if rendered.fell_back or rendered.source != "catalog":
        _log.info("cgroup i18n: %r requested=%s served=%s source=%s",
                  key, rendered.requested_locale,
                  rendered.served_locale, rendered.source)
    return rendered.text


def control_cgroup(tail: str, context=None, chat=None,
                   sender: str = "", sender_id: str = "") -> str:
    """Chat entry point for groups + events + meetups. Never raises."""
    try:
        return _control_cgroup(tail, context, chat, sender, sender_id)
    except Exception as exc:  # noqa: BLE001 — chat must never blow up
        _log.warning("cgroup control failed: %s", exc)
        return "Community groups hit a snag — try again."


def _control_memory(store: GroupStore, rest: str, who: str,
                    context: Any) -> str:
    """Group memory + persona (audit gap R2-26): the audit's first
    option — a group-scoped command calling remember_group /
    recall_group and build_persona, inside the existing isolation
    boundary (memory lives under the group's own id, never shared
    across groups). Membership-gated; ``forget`` is owner/admin.
    Never raises."""
    from . import mirroring as _mirroring

    usage = ("Usage: /cgroup memory <group_id> "
             "remember <key> = <value> | recall <key> | "
             "search <query> | list | forget <key> | persona")
    segs = rest.split(None, 1)
    if not segs:
        return usage
    gid = segs[0]
    g = store.get(gid)
    if g is None:
        return f"No group `{gid}`. /cgroup list to see them."
    if who not in g.members:
        return (f"only members of **{g.name}** can use its memory — "
                f"join it first: /cgroup join {gid}")
    tail = segs[1] if len(segs) > 1 else ""
    parts = tail.split(None, 1)
    verb = parts[0].lower() if parts else ""
    arg = parts[1] if len(parts) > 1 else ""
    if verb == "remember":
        key, sep, value = arg.partition("=")
        if not sep or not value.strip():
            # `remember <key> <value>` without `=`.
            fields = arg.split(None, 1)
            key = fields[0] if fields else ""
            value = fields[1] if len(fields) > 1 else ""
        key = key.strip()
        value = value.strip()
        if not key or not value:
            return "Usage: /cgroup memory <group_id> remember <key> = <value>"
        _mirroring.remember_group(gid, key=key, value=value,
                                  author_id=who, author_name=who)
        return f"🧠 Remembered `{key}` for **{g.name}**."
    if verb == "recall":
        key = arg.strip()
        if not key:
            return "Usage: /cgroup memory <group_id> recall <key>"
        found = _mirroring.recall_group(gid, key=key)
        if not found:
            return f"This group never stored `{key}`."
        return f"`{key}`: {found[0]['value']}"
    if verb == "search":
        query = arg.strip()
        if not query:
            return "Usage: /cgroup memory <group_id> search <query>"
        found = _mirroring.recall_group(gid, query=query)
        if not found:
            return f"Nothing in **{g.name}** matches {query!r}."
        lines = [f"🔍 **Matches in {g.name}:**"]
        for row in found[:10]:
            lines.append(f"· `{row['key']}`: {row['value']}")
        return "\n".join(lines)
    if verb == "list":
        found = _mirroring.recall_group(gid)
        if not found:
            return f"**{g.name}** has no stored memory yet."
        lines = [f"🧠 **Memory in {g.name}:**"]
        for row in found[:20]:
            lines.append(f"· `{row['key']}`: {row['value']}")
        return "\n".join(lines)
    if verb == "forget":
        key = arg.strip()
        if not key:
            return "Usage: /cgroup memory <group_id> forget <key>"
        if g.role_of(who) not in ("owner", "admin"):
            return "forgetting a memory is for the group's owner/admin."
        from .group_memory import GroupMemoryStore

        if GroupMemoryStore().forget(gid, key):
            return f"🧠 Forgot `{key}`."
        return f"No such fact `{key}` in **{g.name}**."
    if verb == "persona":
        from .group_memory import GroupMemoryStore
        from .persona import build_persona

        persona = build_persona(GroupMemoryStore(), gid)
        if not persona.has_history:
            return ("This group has no stored history yet — "
                    "remember something first: /cgroup memory "
                    f"{gid} remember <key> = <value>")
        lines = [f"🎭 **{persona.name}** — the assistant for this group."]
        if persona.traits:
            lines.append("How it comes across: "
                         + "; ".join(persona.traits))
        if persona.greeting:
            lines.append(f"Greeting style: {persona.greeting}")
        lines.append(f"Facts ({len(persona.facts)}):")
        for fact in persona.facts[:10]:
            lines.append(f"· `{fact['key']}`: {fact['value']}")
        return "\n".join(lines)
    return usage


def _control_digest(store: GroupStore, rest: str, who: str,
                    context: Any) -> str:
    """Group digest (audit gap R2-29): build_digest over the group's
    own feed — decisions, unanswered questions, mentions of the
    caller. Membership-gated (a digest quotes other members' posts).
    Never raises."""
    usage = "Usage: /cgroup digest <group_id>"
    segs = rest.split(None, 1)
    if not segs:
        return usage
    gid = segs[0]
    g = store.get(gid)
    if g is None:
        return f"No group `{gid}`. /cgroup list to see them."
    if who not in g.members:
        return (f"only members of **{g.name}** can read its digest — "
                f"join it first: /cgroup join {gid}")
    posts = store.feed(gid, limit=_POSTS_PER_GROUP)
    messages = [{"id": p.id, "sender": p.author, "text": p.text,
                 "ts": p.created_at, "reply_to": ""}
                for p in posts]
    digest = build_digest(messages, group_id=gid, group_name=g.name,
                          me=who)
    return render_digest(digest, me=who)


def _control_rituals(store: GroupStore, rest: str, who: str,
                     context: Any) -> str:
    """Group rituals (audit gap R2-29): list / run / enable / disable /
    add over the group's persisted rituals. List is membership-gated;
    run / enable / disable / add post to the whole group, so they are
    owner/admin only. A manual run marks the ritual delivered, so the
    tick does not immediately re-fire it. An added ritual is saved
    through the same store the tick reads, so it fires on its own
    clock. Never raises."""
    from datetime import UTC, datetime

    usage = ("Usage: /cgroup rituals <group_id> [list | run <name> | "
             "enable <name> | disable <name> | add <name> | "
             "<HH:MM | DAY HH:MM> | <message>]")
    segs = rest.split(None, 1)
    if not segs:
        return usage
    gid = segs[0]
    g = store.get(gid)
    if g is None:
        return f"No group `{gid}`. /cgroup list to see them."
    if who not in g.members:
        return (f"only members of **{g.name}** can see its rituals — "
                f"join it first: /cgroup join {gid}")
    tail = segs[1] if len(segs) > 1 else ""
    parts = tail.split(None, 1)
    verb = parts[0].lower() if parts else "list"
    arg = parts[1].strip() if len(parts) > 1 else ""
    rituals = store.load_rituals(gid)
    if verb == "list":
        now = datetime.now(tz=UTC)
        lines = [f"🕯️ Rituals for **{g.name}** —"]
        for r in rituals:
            nxt = r.next_due(now)
            when = nxt.strftime("%a %H:%M UTC") if nxt else "never"
            state = "enabled" if r.enabled else "disabled"
            lines.append(f"• {r.name} ({r.kind}) — {state}, next {when}")
        return "\n".join(lines)
    if verb not in ("run", "enable", "disable", "add"):
        return usage
    if g.role_of(who) not in (ROLE_OWNER, ROLE_ADMIN):
        return (f"Only the group owner or an admin can {verb} a ritual "
                f"in **{g.name}**.")
    if verb == "add":
        return _add_ritual(store, g, arg, rituals, usage)
    match = [r for r in rituals if r.name.lower() == arg.lower()]
    if not match:
        names = ", ".join(r.name for r in rituals)
        return f"No ritual named `{arg}`. Rituals: {names}"
    ritual = match[0]
    if verb == "run":
        text = render_ritual(ritual, datetime.now(tz=UTC))
        if store.system_post(gid, text) is None:
            return f"Could not post `{ritual.name}` to **{g.name}**."
        RitualScheduler(rituals).mark_delivered(ritual)
        store.save_rituals(gid, rituals)
        return f"🕯️ Delivered `{ritual.name}` to **{g.name}**."
    if verb == "enable":
        ritual.enabled = True
        store.save_rituals(gid, rituals)
        return f"`{ritual.name}` enabled for **{g.name}**."
    ritual.enabled = False
    store.save_rituals(gid, rituals)
    return f"`{ritual.name}` disabled for **{g.name}**."


def _add_ritual(store: GroupStore, g: ThemedGroup, arg: str,
                rituals: list[Ritual], usage: str) -> str:
    """Create a custom ritual: ``add <name> | <HH:MM | DAY HH:MM> |
    <message>``. The ``|`` shape is the house style (``/cgroup new
    <name> | <topic>``). A bare time is daily; a leading weekday
    (MO..SU) makes it weekly. The schedule is validated through the
    same ``parse_rrule`` the tick uses, so an added ritual can only
    ever be one that fires. Caller is already admin-gated."""
    from datetime import UTC, datetime

    _WEEKDAYS = ("MO", "TU", "WE", "TH", "FR", "SA", "SU")
    segs = [p.strip() for p in arg.split("|")]
    if len(segs) != 3 or not all(segs):
        return usage
    name, when, template = segs
    existing = [r for r in rituals if r.name.lower() == name.lower()]
    if existing:
        return (f"There's already a ritual named `{existing[0].name}` "
                f"in **{g.name}** — pick another name, or disable the "
                f"old one first.")
    toks = when.split()
    if len(toks) == 1:
        day, clock = "", toks[0]
    elif len(toks) == 2:
        day, clock = toks[0].upper(), toks[1]
    else:
        return (f"couldn't read the time {when!r} — use HH:MM for "
                f"daily, or DAY HH:MM for weekly, e.g. 09:30 or "
                f"FR 16:00.")
    if day and day not in _WEEKDAYS:
        return (f"couldn't read the weekday {toks[0]!r} — use MO, TU, "
                f"WE, TH, FR, SA or SU, e.g. FR 16:00.")
    try:
        hour_s, _, minute_s = clock.partition(":")
        hour, minute = int(hour_s), int(minute_s)
        if not minute_s or not 0 <= hour <= 23 or not 0 <= minute <= 59:
            raise ValueError("hour/minute out of range")
    except ValueError:
        return (f"couldn't read the time {clock!r} — use 24-hour "
                f"HH:MM, e.g. 09:30.")
    if day:
        rrule = (f"FREQ=WEEKLY;BYDAY={day};BYHOUR={hour};"
                 f"BYMINUTE={minute}")
        when_text = f"weekly on {day} at {hour:02d}:{minute:02d} UTC"
    else:
        rrule = f"FREQ=DAILY;BYHOUR={hour};BYMINUTE={minute}"
        when_text = f"daily at {hour:02d}:{minute:02d} UTC"
    ritual = Ritual(name=name, kind="custom", rrule=rrule,
                    template=template, group_id=g.id, enabled=True)
    now = datetime.now(tz=UTC)
    try:
        nxt = next_occurrences(ritual.recurrence, now, 1, start=now)
    except ValueError as exc:
        return f"that schedule never fires: {exc}"
    if not nxt:
        return "that schedule never fires — pick another time."
    rituals.append(ritual)
    store.save_rituals(g.id, rituals)
    when_due = nxt[0].strftime("%a %H:%M UTC")
    return (f"🕯️ Added `{name}` to **{g.name}** — {when_text}, "
            f"next {when_due}.")


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
        g = store.get(gid) or g
        n = len(g.members)
        note = f"\n{msg}" if msg else ""
        welcome = _group_message("Welcome to **{group}**!", g, group=g.name)
        members = _group_message("{count} member", g, count=n)
        return f"{welcome} ({members}){note}"

    if cmd == "lang":
        segs = rest.split(None, 1)
        gid = segs[0] if segs else ""
        g = store.get(gid)
        if g is None:
            return f"No group `{gid}`. /cgroup list to see them."
        if len(segs) < 2:
            current = g.language or "en (default)"
            return (f"Group language: {current}. "
                    f"Set it: /cgroup lang {gid} <locale, e.g. es, pt-BR, yo>")
        if store.set_language(gid, segs[1]):
            g = store.get(gid) or g
            return f"Group language set to {g.language or 'en (default)'}."
        return f"No group `{gid}`. /cgroup list to see them."

    if cmd == "answer":
        segs = rest.split(None, 2)
        if len(segs) < 3:
            return "Usage: /cgroup answer <group_id> <your answer>"
        ok = store.answer_join_question(segs[0], who, segs[2])
        g = store.get(segs[0])
        if ok:
            return _group_message("Answer recorded. 👍", g)
        return _group_message("Couldn't record that answer.", g)

    if cmd == "leave":
        gid = rest.strip()
        g = store.get(gid)
        if store.leave_group(gid, who):
            return _group_message("Left the group.", g)
        return _group_message("Couldn't leave that group.", g)

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

    if cmd == "memory":
        return _control_memory(store, rest, who, context)

    if cmd == "digest":
        return _control_digest(store, rest, who, context)

    if cmd == "rituals":
        return _control_rituals(store, rest, who, context)

    # ── AA-18: device orchestration — scheduled posts + fleet view ──
    if cmd in ("schedule", "posts", "devices"):
        from .orchestration import runtime_orchestrator

        orchestrator, device_id = runtime_orchestrator(context)
        if cmd == "devices":
            health = orchestrator.router.registry.health()
            if not health:
                return "No devices registered yet."
            lines = [f"🛰️ **Devices** (you are routed by: {device_id}):"]
            for dev, info in sorted(health.items()):
                mark = "🟢" if info["healthy"] else "🔴"
                lines.append(f"  {mark} {dev} — {info['kind']} · "
                             f"heartbeat {info['age_s']}s ago")
            recent = orchestrator.router.log.recent(3)
            for decision in recent:
                lines.append(f"  ↪ {decision.task}: → "
                             f"{decision.chosen_device or 'nobody'} "
                             f"({decision.reason})")
            return "\n".join(lines)
        if chat is None or getattr(chat, "kind", "dm") != "group":
            return "Scheduled posts belong to a group chat — run it there."
        from .miniapps import _group_key

        group_key = _group_key(chat)
        if cmd == "posts":
            mine = [p for p in orchestrator.posts if p.group_id == group_key]
            if not mine:
                return "No scheduled posts for this group."
            lines = ["🗓️ **Scheduled posts for this group:**"]
            for post in mine[-8:]:
                when = time.strftime("%H:%M", time.localtime(post.due_ts))
                state = "sent" if post.sent else "pending"
                lines.append(f"  • {when} [{state}] {post.text[:80]}")
            return "\n".join(lines)
        segs = rest.split(None, 1)
        if len(segs) < 2 or not segs[0].isdigit():
            return "Schedule with: /cgroup schedule <minutes> <text>"
        minutes = int(segs[0])
        orchestrator.schedule_post(
            group_key, str(getattr(chat, "platform", "") or ""),
            f"{getattr(chat, 'platform', '')}:{getattr(chat, 'chat_id', '')}",
            segs[1], time.time() + minutes * 60)
        return (f"🗓️ Scheduled for this group in {minutes} min — "
                f"a healthy device will post it ({device_id} is on duty "
                f"if the router picks it).")

    return _usage()
