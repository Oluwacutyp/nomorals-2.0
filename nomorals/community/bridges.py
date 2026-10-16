"""Cross-platform bridges: one roster across Telegram + WhatsApp + Discord.

A community may live on several platforms at once. A bridge keeps ONE
roster: the same human is one member even when they show up as
``@ada`` on Telegram, ``+234...`` on WhatsApp, and ``ada#1234`` on
Discord. ``UnifiedRoster`` maps ``(platform, platform_user_id)`` →
one member id, so a message can be attributed to the right person no
matter where it arrived from.

Sending discipline — the hard rule of this module
-------------------------------------------------
This module NEVER sends anything by itself. ``bridge_message()`` only
calls sender callables the caller injects, and only for platforms the
caller supplied both a sender AND a target (chat/channel id) for. No
senders injected → nothing is sent, and the call still returns the
routing plan so the caller can inspect what *would* have gone out.
No target for a platform → that platform is skipped, never guessed:
channel ids are never invented, and a message is never broadcast to
an assumed destination.

Isolation: stdlib + ``nomorals.core`` only. No owner-data imports.
Platforms are plain strings ("telegram" / "whatsapp" / "discord");
this module knows no platform SDKs and holds no credentials.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.ids import new_short_id

__all__ = [
    "PLATFORMS",
    "BridgeMember",
    "UnifiedRoster",
    "Delivery",
    "plan_bridge",
    "bridge_message",
    "BridgeStore",
    "default_bridge_dir",
    "format_bridged",
]

PLATFORMS = ("telegram", "whatsapp", "discord")


def default_bridge_dir() -> Path:
    """Default on-disk home: inside the community dir."""
    return Path.home() / ".devon" / "community" / "bridges"


def _normalize_platform(platform: str) -> str:
    normalized = platform.strip().lower()
    if normalized not in PLATFORMS:
        raise ValueError(
            f"unknown platform {platform!r} (expected one of {PLATFORMS})"
        )
    return normalized


def format_bridged(member_name: str, origin_platform: str, text: str) -> str:
    """Attribution format for a bridged message. No identity is ever
    hidden: the receiving side always sees who said it and where."""
    return f"[{member_name} @ {origin_platform}] {text}"


@dataclass
class BridgeMember:
    """One human across platforms. ``platforms`` maps platform →
    platform user id. Identity comes from the group itself (members link
    their own accounts); nothing is pulled from owner data."""

    id: str
    display_name: str
    platforms: dict[str, str] = field(default_factory=dict)
    created_ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BridgeMember":
        return cls(
            id=str(data["id"]),
            display_name=str(data["display_name"]),
            platforms={str(k): str(v) for k, v in data.get("platforms", {}).items()},
            created_ts=float(data.get("created_ts", 0.0)),
        )


@dataclass
class Delivery:
    """One planned (or executed) delivery to a single platform."""

    to_platform: str
    to_target: str  # chat/channel id supplied by the caller
    member_id: str
    member_name: str
    origin_platform: str
    text: str
    sent: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UnifiedRoster:
    """(platform, platform_user_id) → one member. Pure in-memory logic;
    persistence lives in BridgeStore."""

    def __init__(self, members: list[BridgeMember] | None = None) -> None:
        self.members: list[BridgeMember] = list(members or [])

    def add_member(self, display_name: str) -> BridgeMember:
        member = BridgeMember(
            id=new_short_id("member"),
            display_name=display_name,
            created_ts=time.time(),
        )
        self.members.append(member)
        return member

    def get(self, member_id: str) -> BridgeMember | None:
        for m in self.members:
            if m.id == member_id:
                return m
        return None

    def link(self, member_id: str, platform: str, platform_user_id: str) -> None:
        """Link a platform identity to a member. One platform identity
        belongs to exactly one member: linking it elsewhere unlinks it
        from the previous member first (no silent duplicates)."""
        platform = _normalize_platform(platform)
        if not platform_user_id.strip():
            raise ValueError("platform_user_id must not be empty")
        member = self.get(member_id)
        if member is None:
            raise ValueError(f"unknown member {member_id!r}")
        for other in self.members:
            if (
                other.id != member_id
                and other.platforms.get(platform) == platform_user_id
            ):
                del other.platforms[platform]
        member.platforms[platform] = platform_user_id

    def unlink(self, member_id: str, platform: str) -> bool:
        member = self.get(member_id)
        if member is None:
            return False
        return member.platforms.pop(_normalize_platform(platform), None) is not None

    def resolve(self, platform: str, platform_user_id: str) -> BridgeMember | None:
        """The member behind a platform identity, or None if unlinked."""
        platform = _normalize_platform(platform)
        for m in self.members:
            if m.platforms.get(platform) == platform_user_id:
                return m
        return None

    def member_platforms(self, member_id: str) -> dict[str, str]:
        member = self.get(member_id)
        return dict(member.platforms) if member else {}

    def to_dicts(self) -> list[dict[str, Any]]:
        return [m.to_dict() for m in self.members]

    @classmethod
    def from_dicts(cls, data: list[dict[str, Any]]) -> "UnifiedRoster":
        return cls([BridgeMember.from_dict(d) for d in data])


def plan_bridge(
    roster: UnifiedRoster,
    origin_platform: str,
    origin_user_id: str,
    text: str,
    targets: dict[str, str],
) -> list[Delivery]:
    """Pure routing plan: which platforms a message goes to.

    * The origin platform is NEVER a target (no echo back to sender's side).
    * Only platforms with an explicit target in ``targets`` are included.
    * Unknown origin users still bridge — attributed as their raw
      platform id — because a bridge must not drop a message it cannot
      attribute; attribution falls back to the platform id, never to
      owner data.
    * No sending happens here: this is the plan ``bridge_message``
      executes against injected senders.
    """
    origin = _normalize_platform(origin_platform)
    if not text.strip():
        raise ValueError("cannot bridge an empty message")
    member = roster.resolve(origin, origin_user_id)
    name = member.display_name if member else origin_user_id
    member_id = member.id if member else ""
    deliveries: list[Delivery] = []
    for platform in PLATFORMS:
        if platform == origin:
            continue
        target = targets.get(platform, "")
        if not target.strip():
            continue  # no target supplied → skip, never guess one
        deliveries.append(
            Delivery(
                to_platform=platform,
                to_target=target,
                member_id=member_id,
                member_name=name,
                origin_platform=origin,
                text=format_bridged(name, origin, text),
            )
        )
    return deliveries


def bridge_message(
    roster: UnifiedRoster,
    origin_platform: str,
    origin_user_id: str,
    text: str,
    targets: dict[str, str],
    senders: dict[str, Callable[[str, str], Any]] | None = None,
) -> list[Delivery]:
    """Route a message to the other platforms via injected senders.

    ``senders`` maps platform → callable ``(target, text)``. A platform
    is only sent to when it has BOTH a target and an injected sender.
    With no ``senders`` (or none matching the plan) nothing is sent and
    every delivery comes back with ``sent=False`` — the plan is still
    returned so the caller can log/inspect it. Sender exceptions are
    captured per-delivery (``error``) instead of aborting the other
    platforms; the caller decides whether to retry.
    """
    deliveries = plan_bridge(roster, origin_platform, origin_user_id, text, targets)
    senders = senders or {}
    for delivery in deliveries:
        sender = senders.get(delivery.to_platform)
        if sender is None:
            continue  # no injected sender → never auto-send
        try:
            sender(delivery.to_target, delivery.text)
            delivery.sent = True
        except Exception as exc:  # noqa: BLE001 — captured per delivery
            delivery.sent = False
            delivery.error = str(exc)
    return deliveries


class BridgeStore:
    """JSON persistence for one group's roster."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir else default_bridge_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, group_key: str) -> Path:
        safe = "".join(c for c in group_key if c.isalnum() or c in "-_")
        folder = self.data_dir / (safe or "default")
        folder.mkdir(parents=True, exist_ok=True)
        return folder / "roster.json"

    def save(self, group_key: str, roster: UnifiedRoster) -> None:
        self._path(group_key).write_text(
            json.dumps(roster.to_dicts(), indent=2), encoding="utf-8"
        )

    def load(self, group_key: str) -> UnifiedRoster:
        path = self._path(group_key)
        if not path.exists():
            return UnifiedRoster()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return UnifiedRoster()
        return UnifiedRoster.from_dicts(data if isinstance(data, list) else [])
