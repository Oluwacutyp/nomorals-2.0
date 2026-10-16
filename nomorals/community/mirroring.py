"""Cross-platform mirroring + group memory (inside isolation).

One roster (Telegram + WhatsApp + Discord) via the prior wave's
``bridges.UnifiedRoster``; messages arriving on one platform are
mirrored to the others through *injected* senders — this module never
sends by itself and never invents a target.

Isolation — the hard rules, enforced by construction:

* Only whitelisted fields cross the mirror: ``group_id``,
  ``author_name``, ``origin_platform``, ``text``, ``message_id``.
  Anything else in the input dict (owner fields, contacts, account
  ids, credentials, ...) is dropped on the floor, never forwarded.
  :func:`mirrored_payload` is the single choke point and it is
  unit-tested against an input stuffed with owner-looking fields.
* Group memory is scoped inside isolation: :func:`remember_group`
  writes through ``group_memory.GroupMemoryStore`` (per-group file,
  group_id-verified reads). It physically cannot read owner data.
* Dedup: a ``message_id`` mirrored once is never mirrored again in
  this process (and callers can persist the seen set).
* Loop protection: a message already marked ``bridged`` is not
  re-mirrored (no ping-pong between platforms).
* Attribution is never hidden: mirrored text is prefixed with
  ``[name @ platform]`` via ``bridges.format_bridged``.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger
from .bridges import UnifiedRoster, format_bridged
from .group_memory import GroupMemoryStore

_log = get_logger(__name__)

__all__ = [
    "MIRRORED_FIELDS", "mirrored_payload", "MirrorDelivery",
    "MirrorEngine", "GroupMirror", "remember_group", "recall_group",
]

#: The ONLY fields allowed to cross the mirror. Everything else is dropped.
MIRRORED_FIELDS = ("group_id", "author_name", "origin_platform", "text", "message_id")


def mirrored_payload(message: dict[str, Any]) -> dict[str, Any]:
    """Whitelist-filter a message for mirroring. Drops everything else.

    This is the choke point: even if a caller passes a dict stuffed
    with owner data, only the five whitelisted fields survive.
    """
    out: dict[str, Any] = {}
    for key in MIRRORED_FIELDS:
        if key in message:
            out[key] = message[key]
    out.setdefault("group_id", "")
    out.setdefault("author_name", "")
    out.setdefault("origin_platform", "")
    out.setdefault("text", "")
    out.setdefault("message_id", "")
    return out


@dataclass
class MirrorDelivery:
    to_platform: str
    to_target: str
    text: str
    sent: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MirrorEngine:
    """Mirror one message across platforms via injected senders.

    ``senders`` maps platform → ``callable(target, text) -> bool``.
    ``targets`` maps platform → chat/channel id (caller-supplied).
    No sender or no target for a platform → skipped, never guessed.
    """

    def __init__(self, senders: dict[str, Callable[[str, str], bool]] | None = None,
                 targets: dict[str, str] | None = None) -> None:
        self.senders = dict(senders or {})
        self.targets = dict(targets or {})
        self.seen_ids: set[str] = set()

    def mirror(self, message: dict[str, Any]) -> list[MirrorDelivery]:
        payload = mirrored_payload(message)
        if message.get("bridged"):
            return []  # loop protection: already bridged, don't re-mirror
        msg_id = str(payload.get("message_id") or "")
        if msg_id and msg_id in self.seen_ids:
            return []  # dedup
        if msg_id:
            self.seen_ids.add(msg_id)
        origin = str(payload.get("origin_platform", ""))
        text = format_bridged(str(payload.get("author_name", "")),
                              origin, str(payload.get("text", "")))
        deliveries: list[MirrorDelivery] = []
        for platform, target in self.targets.items():
            if platform == origin:
                continue  # never echo back to the origin platform
            sender = self.senders.get(platform)
            if sender is None:
                deliveries.append(MirrorDelivery(
                    to_platform=platform, to_target=target, text=text,
                    sent=False, error="no sender injected for platform"))
                continue
            try:
                sent = bool(sender(target, text))
            except Exception as exc:  # noqa: BLE001 — sender is caller code
                _log.warning("mirror sender failed for %s: %s", platform, exc)
                deliveries.append(MirrorDelivery(
                    to_platform=platform, to_target=target, text=text,
                    sent=False, error=str(exc)))
                continue
            deliveries.append(MirrorDelivery(
                to_platform=platform, to_target=target, text=text, sent=sent))
        return deliveries


@dataclass
class GroupMirror:
    """One group mirrored across platforms, with one roster + one memory.

    ``roster`` resolves ``(platform, platform_user_id)`` → one member
    (see ``bridges.UnifiedRoster``). Members link their own accounts;
    nothing is pulled from owner data.
    """

    group_id: str
    roster: UnifiedRoster = field(default_factory=UnifiedRoster)
    engine: MirrorEngine = field(default_factory=MirrorEngine)
    memory_root: Path | str | None = None

    def handle_incoming(self, platform: str, platform_user_id: str,
                        display_name: str, text: str,
                        message_id: str = "") -> dict[str, Any]:
        """Attribute → mirror → remember. Returns deliveries + memory id."""
        member = self.roster.resolve(platform, platform_user_id)
        name = member.display_name if member else (display_name or platform_user_id)
        member_id = member.id if member else ""
        deliveries = self.engine.mirror({
            "group_id": self.group_id,
            "author_name": name,
            "origin_platform": platform,
            "text": text,
            "message_id": message_id or f"{platform}:{time.time_ns()}",
        })
        mem_id = remember_group(
            self.group_id, key=f"msg:{message_id or time.time_ns()}",
            value=f"{name}: {text}",
            author_id=member_id, author_name=name,
            root=self.memory_root)
        return {"member_id": member_id, "deliveries": deliveries,
                "memory_id": mem_id}


def remember_group(group_id: str, *, key: str, value: str,
                   author_id: str = "", author_name: str = "",
                   tags: list[str] | None = None,
                   root: Path | str | None = None) -> str:
    """Store one group fact. Group-scoped, never owner data."""
    store = GroupMemoryStore(root=root)
    entry = store.remember(group_id, key=key, value=value,
                           author_id=author_id, author_name=author_name,
                           tags=tags or [])
    return entry.id


def recall_group(group_id: str, *, key: str = "", query: str = "",
                 root: Path | str | None = None) -> list[dict[str, Any]]:
    """Recall group facts. Cross-group reads raise inside the store."""
    store = GroupMemoryStore(root=root)
    if key:
        entry = store.recall(group_id, key)
        return [entry.to_dict()] if entry else []
    if query:
        return [e.to_dict() for e in store.search(group_id, query)]
    return [e.to_dict() for e in store.list_entries(group_id)]
