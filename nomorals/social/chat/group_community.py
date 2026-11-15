"""Telegram Communities (DESIGN §16): the community-of-groups surface.

A Telegram Community is several supergroups, channels and bots linked
together around a shared topic or audience (Bot API 10.2, July 2026;
``community_chat_joined`` added in 10.3).  The platform exposes exactly
three things a bot can read:

* ``Community`` — just ``id`` + ``name``.
* ``ChatFullInfo.community`` — a chat's membership, read via ``getChat``.
* Three service messages: ``community_chat_added`` (``{community}``),
  ``community_chat_joined`` (``{community}``) and ``community_chat_removed``
  (holds NO information — the organ clears only what the group currently
  carries and audits the prior tag).

What Devon had before this organ: nothing on the Telegram side, and a
Devon-side ``community_id`` tag in the group settings that was READ by
the §14 community boards and the audio rooms but WRITTEN NOWHERE — the
boards could never match a group.  This organ is the tag's first writer.

The organ never builds a parallel system: the tag lives in the group
settings (so backups carry it and the boards read it unchanged), the
cross-chat registry is derived by scanning the store (no new durable
shape), service events ride the existing ``meta["service"]`` switch in
``_group_surface_turn``, and detection rides the existing adapter
verbs (Bot API ``getChat``; WhatsApp ``group_info`` + ``communities``,
both already exposed).  WhatsApp Communities are the parity reference:
create/link/unlink/list/broadcast already exist over there as bridge
verbs and registered tools — this organ gives Telegram the same
read/detect/sync surface and one shared ``/community`` command over both.

Honest limits, stated where they bite, never worked around:
* MTProto ``channelFull`` carries NO community field — the userbot
  surface cannot detect Telegram Communities; detection is Bot API only.
* ``community_chat_removed`` names no community.
* A platform sync never overwrites a manually-set tag
  (``community_source`` "manual" beats "platform"); a manual link is
  the admin's explicit act and wins until they unlink it.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "SOURCE_PLATFORM",
    "SOURCE_MANUAL",
    "parse_community",
    "community_of",
    "describe_community",
    "note_community_added",
    "note_community_removed",
    "note_community_joined",
    "set_community_tag",
    "clear_community_tag",
    "communities_index",
    "detect_community",
    "sync_group_community",
    "sync_all_communities",
]

#: The tag was written by a platform probe (getChat / linkedParent).
SOURCE_PLATFORM = "platform"
#: The tag was written by an admin's explicit /community link.
SOURCE_MANUAL = "manual"


def parse_community(raw: Any) -> dict[str, str] | None:
    """Normalize a Bot API ``Community`` dict to ``{"id", "name"}``.

    The id is the key (Telegram community ids are integers, possibly
    large — always carried as a string here); the name may be empty.
    Anything absent or malformed returns None — never a guessed id.
    """
    if not isinstance(raw, dict):
        return None
    cid = raw.get("id")
    if cid is None or str(cid).strip() == "":
        return None
    return {"id": str(cid).strip(),
            "name": str(raw.get("name") or "").strip()}


def community_of(surface: Any, key: str) -> dict[str, str] | None:
    """The community a group carries, or None when untagged."""
    settings = surface.store.settings(key)
    cid = str(settings.get("community_id") or "").strip()
    if not cid:
        return None
    return {"id": cid,
            "name": str(settings.get("community_name") or "").strip(),
            "source": str(settings.get("community_source") or "")}


def describe_community(surface: Any, key: str) -> str:
    """One line for boards and the console: the community or "none"."""
    found = community_of(surface, key)
    if found is None:
        return "no community"
    name = found["name"] or found["id"]
    source = found.get("source") or ""
    suffix = f" ({source})" if source else ""
    return f"community {name}{suffix}"


def _write_tag(surface: Any, key: str, community: dict[str, str] | None,
               source: str) -> None:
    """Persist the tag triple through the surface's own writer."""
    surface.update_settings(
        key,
        community_id=(community or {}).get("id", "") if community else "",
        community_name=(community or {}).get("name", "") if community
        else "",
        community_source=source if community else "")


def note_community_added(surface: Any, key: str,
                         community: dict[str, str] | None) -> dict[str, Any]:
    """Observe a ``community_chat_added`` service message.

    Sets the tag (and name) on the group — unless the group already
    carries a MANUALLY-set tag pointing elsewhere: the admin's explicit
    link wins, the conflict is audited, and nothing is overwritten.
    Returns a small report; never raises.
    """
    parsed = parse_community(community)
    if parsed is None:
        return {"ok": False, "key": key, "reason": "no community payload"}
    current = community_of(surface, key)
    if current is not None and current.get("source") == SOURCE_MANUAL \
            and current["id"] != parsed["id"]:
        surface._audit(key, "community_added_conflict",
                       community_id=parsed["id"],
                       community_name=parsed["name"],
                       kept_id=current["id"],
                       kept_name=current["name"])
        surface.store.save()
        return {"ok": False, "key": key, "reason": "manual tag kept",
                "community": parsed, "kept": current}
    if current is not None and current["id"] == parsed["id"]:
        if current["name"] != parsed["name"]:
            _write_tag(surface, key, parsed, SOURCE_PLATFORM)
        surface._audit(key, "community_added",
                       community_id=parsed["id"],
                       community_name=parsed["name"],
                       already=True)
        surface.store.save()
        return {"ok": True, "key": key, "community": parsed,
                "changed": False}
    _write_tag(surface, key, parsed, SOURCE_PLATFORM)
    surface._audit(key, "community_added", community_id=parsed["id"],
                   community_name=parsed["name"])
    surface.store.save()
    return {"ok": True, "key": key, "community": parsed, "changed": True}


def note_community_removed(surface: Any, key: str) -> dict[str, Any]:
    """Observe a ``community_chat_removed`` service message.

    The message holds NO information — it names no community — so the
    organ clears only what the group currently carries (and only when
    the platform put it there; a manual tag is the admin's explicit
    act and survives a bare removal notice).  The prior tag is
    audited, so the removal is never a silent loss.
    """
    current = community_of(surface, key)
    if current is None:
        return {"ok": True, "key": key, "changed": False,
                "reason": "no community carried"}
    if current.get("source") == SOURCE_MANUAL:
        surface._audit(key, "community_removed",
                       community_id=current["id"],
                       community_name=current["name"],
                       kept=True, reason="manual tag kept")
        surface.store.save()
        return {"ok": True, "key": key, "changed": False,
                "kept": current}
    _write_tag(surface, key, None, "")
    surface._audit(key, "community_removed",
                   community_id=current["id"],
                   community_name=current["name"])
    surface.store.save()
    return {"ok": True, "key": key, "changed": True,
            "removed": current}


def note_community_joined(surface: Any, key: str,
                          community: dict[str, str] | None, *,
                          user_id: str = "",
                          username: str = "") -> dict[str, Any]:
    """Observe a ``community_chat_joined`` service message.

    Attribution only: the chat was joined by a user FROM a community,
    which does not make the chat a member of it — the tag is never
    touched.  The community and the joining user are journaled so the
    roster-diff backbone can attribute the arrival to community
    discovery.
    """
    parsed = parse_community(community)
    if parsed is None:
        return {"ok": False, "key": key, "reason": "no community payload"}
    surface._audit(key, "community_join", community_id=parsed["id"],
                   community_name=parsed["name"],
                   user_id=str(user_id or ""),
                   username=str(username or ""))
    surface.store.save()
    return {"ok": True, "key": key, "community": parsed}


def set_community_tag(surface: Any, key: str, community_id: str,
                      name: str = "") -> dict[str, Any]:
    """Manually link a group to a community (the Devon tag).

    For groups whose platform has no Community object, or to override
    a platform reading.  The admin's explicit act: source "manual",
    which platform syncs never overwrite.  Audited.
    """
    cid = str(community_id or "").strip()
    if not cid:
        return {"ok": False, "key": key, "reason": "empty community id"}
    tag = {"id": cid, "name": str(name or "").strip()}
    current = community_of(surface, key)
    _write_tag(surface, key, tag, SOURCE_MANUAL)
    surface._audit(key, "community_linked", community_id=tag["id"],
                   community_name=tag["name"],
                   previous_id=(current or {}).get("id", ""))
    surface.store.save()
    return {"ok": True, "key": key, "community": tag,
            "changed": current is None or current["id"] != cid}


def clear_community_tag(surface: Any, key: str) -> dict[str, Any]:
    """Manually unlink a group from its community.  Audited."""
    current = community_of(surface, key)
    if current is None:
        return {"ok": True, "key": key, "changed": False,
                "reason": "no community carried"}
    _write_tag(surface, key, None, "")
    surface._audit(key, "community_unlinked",
                   community_id=current["id"],
                   community_name=current["name"])
    surface.store.save()
    return {"ok": True, "key": key, "changed": True,
            "removed": current}


def communities_index(surface: Any) -> dict[str, dict[str, Any]]:
    """The cross-chat registry, derived by scanning the store.

    ``{community_id: {"name", "platforms", "groups": [keys]}}`` —
    groups AND channels (a channel can belong to a community too).
    No new durable shape: the tag lives in the group settings, so
    backups carry it and this index is always a fresh read.
    """
    index: dict[str, dict[str, Any]] = {}
    for key in sorted(surface.store.data.keys()):
        found = community_of(surface, key)
        if found is None:
            continue
        platform = str(key).split(":", 1)[0]
        row = index.get(found["id"])
        if row is None:
            row = {"name": found["name"], "platforms": [],
                   "groups": []}
            index[found["id"]] = row
        if found["name"] and not row["name"]:
            row["name"] = found["name"]
        if platform not in row["platforms"]:
            row["platforms"].append(platform)
        row["groups"].append(key)
    for row in index.values():
        row["platforms"].sort()
        row["groups"].sort()
    return index


def detect_community(adapter: Any, platform: str,
                     chat: Any) -> tuple[dict[str, str] | None, str]:
    """Probe one chat's platform community.  Never raises.

    Telegram Bot API (``telegram-bot``): ``getChat`` →
    ``ChatFullInfo.community``.  WhatsApp: ``group_info`` →
    ``linked_parent`` + the parent's name from ``communities()``.
    The userbot (``telegram``) has no community field on MTProto —
    that is an honest platform limit, reported as an error, never
    faked.  Returns (community | None, error-string).
    """
    platform = str(platform or "").strip().lower()
    if adapter is None:
        return None, "no adapter for this platform"
    try:
        if platform == "telegram-bot":
            getter = getattr(adapter, "get_chat_community", None)
            if not callable(getter):
                return None, ("the Telegram bot adapter has no "
                              "community read")
            out = getter(chat) or {}
            if not isinstance(out, dict):
                return None, "unexpected community read shape"
            if not out.get("ok"):
                return None, str(out.get("error") or "getChat failed")
            return parse_community(out.get("community")), ""
        if platform.startswith("whatsapp"):
            info = adapter.group_info(chat) or {}
            if not isinstance(info, dict):
                return None, "unexpected group info shape"
            parent = str(info.get("linked_parent") or "").strip()
            if not parent:
                return None, ""
            name = ""
            try:
                for row in adapter.communities() or []:
                    if str(row.get("id") or "") == parent:
                        name = str(row.get("subject") or "")
                        break
            except Exception:  # noqa: BLE001 - name is garnish
                name = ""
            return {"id": parent, "name": name}, ""
        return None, ("communities cannot be read on this surface "
                      "(MTProto exposes no community field)")
    except Exception as exc:  # noqa: BLE001 - probe never raises
        return None, str(exc)[:200]


def sync_group_community(surface: Any, key: str,
                         adapter: Any) -> dict[str, Any]:
    """Re-probe one group's platform community and write on change.

    A manual tag is never overwritten by a platform reading; a
    platform-sourced tag is replaced when the platform disagrees, and
    cleared when the platform reports no community.  Returns a small
    report; never raises.
    """
    platform, _, chat_id = str(key).partition(":")
    current = community_of(surface, key)
    if current is not None and current.get("source") == SOURCE_MANUAL:
        return {"ok": True, "key": key, "changed": False,
                "community": current, "reason": "manual tag kept"}
    from .base import ChatRef

    chat = ChatRef(platform=platform, chat_id=chat_id)
    found, error = detect_community(adapter, platform, chat)
    if error:
        return {"ok": False, "key": key, "error": error,
                "community": current}
    if found is None:
        if current is None:
            return {"ok": True, "key": key, "changed": False,
                    "community": None}
        _write_tag(surface, key, None, "")
        surface._audit(key, "community_sync_cleared",
                       community_id=current["id"],
                       community_name=current["name"])
        surface.store.save()
        return {"ok": True, "key": key, "changed": True,
                "community": None, "removed": current}
    if current is not None and current["id"] == found["id"]:
        if current["name"] != found["name"]:
            _write_tag(surface, key, found, SOURCE_PLATFORM)
        return {"ok": True, "key": key, "changed": False,
                "community": found}
    _write_tag(surface, key, found, SOURCE_PLATFORM)
    surface._audit(key, "community_sync", community_id=found["id"],
                   community_name=found["name"],
                   previous_id=(current or {}).get("id", ""))
    surface.store.save()
    return {"ok": True, "key": key, "changed": True,
            "community": found, "previous": current}


def sync_all_communities(surface: Any,
                         adapters: dict[str, Any],
                         keys: list[str] | None = None) -> list[dict[str, Any]]:
    """Re-probe a set of groups/channels.  One bad group never kills
    the pass — its report carries the error.  ``keys`` defaults to
    every key in the store; callers that sweep on their own initiative
    pass claimed keys only (no state accrues in foreign rooms)."""
    reports: list[dict[str, Any]] = []
    for key in sorted(keys if keys is not None
                      else surface.store.data.keys()):
        platform = str(key).split(":", 1)[0]
        reports.append(sync_group_community(
            surface, key, (adapters or {}).get(platform)))
    return reports
