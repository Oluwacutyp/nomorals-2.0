"""Themed community groups: chat rooms with topics, posts, and membership.

#84 — the Bumble BFF → Geneva pivot signal: 1:1 matching didn't
retain; groups and events do. A themed group is a named room
("Lagos Devs", "Weekend Hikers") with a topic, a member roster, and a
post feed. Groups are community-scoped (not chat-scoped): the same
"Lagos Devs" group is visible from any chat.

Isolation: this module imports stdlib + ``nomorals.core`` only.
No memory, no accounts, no vaults, no connectors — the package
docstring is the law, ``tests/test_community_isolation.py`` the gate.

Chat: ``/cgroup new <name> | <topic>``, ``/cgroup join <group_id>``,
``/cgroup list``, ``/cgroup post <group_id> <text>``,
``/cgroup show <group_id>``.
"""

from __future__ import annotations

import json
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


def _sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "group"


# ── data model ─────────────────────────────────────────────────────────────


@dataclass
class GroupPost:
    """One post in a themed group's feed."""

    id: str
    group_id: str
    author: str
    text: str
    created_at: float = 0.0

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
        )


@dataclass
class ThemedGroup:
    """A themed room: name, topic, members, and a post feed."""

    id: str
    name: str
    topic: str
    members: list[str] = field(default_factory=list)
    created_by: str = ""
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThemedGroup":
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            topic=str(data.get("topic", "")),
            members=[str(m) for m in (data.get("members") or [])],
            created_by=str(data.get("created_by", "")),
            created_at=float(data.get("created_at") or 0.0),
        )


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

    # ── groups ─────────────────────────────────────────────────────────

    def create_group(self, name: str, topic: str, created_by: str = "") -> ThemedGroup | None:
        """Create a themed room. Returns None when the name is empty."""
        name = (name or "").strip()
        if not name:
            return None
        now = time.time()
        group = ThemedGroup(
            id="grp_" + new_short_id(),
            name=name[:80],
            topic=(topic or "").strip()[:200],
            members=[created_by] if created_by else [],
            created_by=created_by,
            created_at=now,
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

    def join_group(self, group_id: str, member: str) -> bool:
        """Add a member. True on success (or already a member)."""
        if not member:
            return False
        groups = self._load_groups()
        for g in groups:
            if g.id == group_id:
                if member not in g.members:
                    g.members.append(member)
                    self._save_groups(groups)
                return True
        return False

    def leave_group(self, group_id: str, member: str) -> bool:
        groups = self._load_groups()
        for g in groups:
            if g.id == group_id and member in g.members:
                g.members.remove(member)
                self._save_groups(groups)
                return True
        return False

    def member_groups(self, member: str) -> list[ThemedGroup]:
        """All groups a member belongs to — the warm pool for discovery."""
        return [g for g in self._load_groups() if member in g.members]

    # ── posts ──────────────────────────────────────────────────────────

    def post(self, group_id: str, author: str, text: str) -> GroupPost | None:
        """Post to a group feed. Author must be a member. Never raises."""
        text = (text or "").strip()
        if not text or not author:
            return None
        group = self.get(group_id)
        if group is None or author not in group.members:
            return None
        p = GroupPost(
            id="post_" + new_short_id(),
            group_id=group_id,
            author=author,
            text=text[:2000],
            created_at=time.time(),
        )
        posts = self._load_posts(group_id)
        posts.append(p)
        self._save_posts(group_id, posts[-_POSTS_PER_GROUP:])
        return p

    def feed(self, group_id: str, limit: int = 20) -> list[GroupPost]:
        return self._load_posts(group_id)[-max(1, limit):]

    def last_post_at(self, group_id: str) -> float:
        posts = self._load_posts(group_id)
        return max((p.created_at for p in posts), default=0.0)

    def quiet_groups(self, days: float = _QUIET_DAYS,
                     now: float | None = None) -> list[ThemedGroup]:
        """Groups with no posts in ``days`` — candidates for a nudge."""
        now = now if now is not None else time.time()
        cutoff = now - days * 86400.0
        return [g for g in self._load_groups()
                if g.members and self.last_post_at(g.id) < cutoff]


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


# ── chat ─────────────────────────────────────────────────────────────────


def _usage() -> str:
    return (
        "/cgroup new <name> | <topic> · join <group_id> · list · "
        "post <group_id> <text> · show <group_id> · nudge · discover <member>\n"
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
                f"You're in. Invite people with /cgroup join {g.id}")

    if cmd == "join":
        gid = rest.strip()
        g = store.get(gid)
        if g is None:
            return f"No group `{gid}`. /cgroup list to see them."
        store.join_group(gid, who)
        fresh = store.get(gid)
        n = len(fresh.members) if fresh else len(g.members) + 1
        return f"Welcome to **{g.name}**! 🎉 ({n} member{'s' if n != 1 else ''})"

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
            lines.append(f"{mark} **{g.name}** `{g.id}` — {g.topic or 'no topic'} "
                         f"({len(g.members)} members)")
        return "\n".join(lines)

    if cmd == "post":
        gid, _, text = rest.partition(" ")
        p = store.post(gid.strip(), who, text)
        if p is None:
            return "Post failed — check the group id and that you've joined: /cgroup join <group_id>"
        return f"📝 Posted to `{gid.strip()}`."

    if cmd == "show":
        g = store.get(rest.strip())
        if g is None:
            return "No such group."
        lines = [f"🏠 **{g.name}** — {g.topic or 'no topic'}",
                 f"Members ({len(g.members)}): {', '.join(g.members[:20]) or '—'}",
                 "Recent posts:"]
        for p in store.feed(g.id, 5):
            lines.append(f"  {p.author}: {p.text[:120]}")
        return "\n".join(lines)

    if cmd == "nudge":
        # Anti-ghosting: surface quiet groups to the requester.
        quiet = store.quiet_groups()
        if not quiet:
            return "Every group is active. No nudges needed. 💪"
        lines = ["😴 **Quiet groups** (no posts in 7 days) — your turn to say hi:"]
        for g in quiet[:5]:
            lines.append(f"  **{g.name}** `{g.id}` — post something: /cgroup post {g.id} <text>")
        return "\n".join(lines)

    if cmd == "discover":
        # Warm-start discovery: people sharing context with this member.
        member = rest.strip() or who
        top = warm_matches(store, member)
        if not top:
            return f"No shared-context matches for {member} yet."
        lines = [f"🤝 **Warm matches for {member}** (shared groups):"]
        for m, c in top:
            lines.append(f"  {m} — {c} shared group{'s' if c != 1 else ''}")
        return "\n".join(lines)

    return _usage()
