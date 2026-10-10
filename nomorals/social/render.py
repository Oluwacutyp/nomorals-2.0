"""God-tier dynamic outputs for groups, channels, and communities.

The problem: group/channel tools returned raw dicts and the brain improvised
formatting every time. This module renders systematic, platform-native output:
Telegram gets HTML, WhatsApp gets emoji-led text, Discord gets embed dicts.

Every renderer takes structured data and a platform string. Never raises —
worst case returns a plain-text fallback.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any

__all__ = [
    "render_group_card",
    "render_members",
    "render_community",
    "render_channel",
    "render_group_list",
]


def _plat(platform: str) -> str:
    p = (platform or "").strip().lower()
    if p.startswith("wa") or p == "whatsapp":
        return "whatsapp"
    if p.startswith("discord"):
        return "discord"
    return "telegram"


def _esc(text: str, platform: str) -> str:
    if platform == "telegram":
        return html.escape(text or "")
    return text or ""


def _fmt_date(ts: float) -> str:
    if not ts:
        return "unknown"
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%b %d, %Y")
    except (ValueError, OSError, OverflowError):
        return "unknown"


def _display_name(entry: dict[str, Any]) -> str:
    name = str(entry.get("name") or "").strip()
    if name:
        return name
    jid = str(entry.get("jid") or entry.get("id") or "")
    # +2348012345678@s.whatsapp.net -> +2348012345678
    return jid.split("@")[0] if "@" in jid else jid


def render_group_card(info: dict[str, Any],
                     members: list[dict[str, Any]] | None = None,
                     platform: str = "telegram") -> str | dict[str, Any]:
    """A rich group info card. ``info`` is the adapter's group_info dict."""
    p = _plat(platform)
    try:
        subject = str(info.get("subject") or "Unnamed group")
        desc = str(info.get("desc") or "")
        size = int(info.get("size") or 0)
        admins = info.get("admins") or []
        announce = bool(info.get("announce"))
        restrict = bool(info.get("restrict"))
        ephemeral = int(info.get("ephemeral_hours") or 0)
        created = _fmt_date(float(info.get("creation") or 0))
        invite = str(info.get("invite") or "")
        is_community = bool(info.get("is_community"))

        n_admins = len(admins)
        n_members = len(members) if members is not None else size

        if p == "discord":
            fields = [
                {"name": "Members", "value": str(size), "inline": True},
                {"name": "Admins", "value": str(n_admins), "inline": True},
                {"name": "Created", "value": created, "inline": True},
            ]
            if announce:
                fields.append({"name": "Mode", "value": "📢 Announcement only",
                               "inline": True})
            if restrict:
                fields.append({"name": "Settings", "value": "🔒 Admins edit info",
                               "inline": True})
            if ephemeral:
                fields.append({"name": "Disappearing", "value": f"{ephemeral}h",
                               "inline": True})
            embed: dict[str, Any] = {
                "title": f"💬 {subject}",
                "description": desc or None,
                "fields": fields,
            }
            if invite:
                embed["fields"].append({"name": "Invite", "value": invite,
                                       "inline": False})
            return embed

        if p == "whatsapp":
            lines = [f"💬 *{subject}*"]
            if desc:
                lines.append(f"_{desc}_")
            lines.append("")
            lines.append(f"👥 {size} members · 👑 {n_admins} admin"
                         f"{'s' if n_admins != 1 else ''}")
            lines.append(f"📅 created {created}")
            flags = []
            if announce:
                flags.append("📢 announcement-only")
            if restrict:
                flags.append("🔒 admins edit info")
            if ephemeral:
                flags.append(f"⏳ disappearing {ephemeral}h")
            if is_community:
                flags.append("🏘️ community group")
            if flags:
                lines.append(" · ".join(flags))
            if invite:
                lines.append(f"\n🔗 {invite}")
            return "\n".join(lines)

        # telegram: HTML
        lines = [f"💬 <b>{_esc(subject, p)}</b>"]
        if desc:
            lines.append(f"<i>{_esc(desc, p)}</i>")
        lines.append("")
        lines.append(f"👥 {size} members · 👑 {n_admins} admin"
                     f"{'s' if n_admins != 1 else ''}")
        lines.append(f"📅 created {created}")
        flags = []
        if announce:
            flags.append("📢 announcement-only")
        if restrict:
            flags.append("🔒 admins edit info")
        if ephemeral:
            flags.append(f"⏳ disappearing {ephemeral}h")
        if is_community:
            flags.append("🏘️ community group")
        if flags:
            lines.append(" · ".join(flags))
        if invite:
            lines.append(f"\n🔗 {html.escape(invite)}")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001 — never break the chat on rendering
        return str(info.get("subject") or "group")


def render_members(members: list[dict[str, Any]],
                   platform: str = "telegram",
                   limit: int = 50) -> str | dict[str, Any]:
    """A member roster with roles. Never invents names — falls back to JID."""
    p = _plat(platform)
    try:
        members = members or []
        admins = [m for m in members if (m.get("role") or "") in ("admin", "superadmin")]
        regulars = [m for m in members if (m.get("role") or "") not in ("admin", "superadmin")]
        shown = members[:limit]
        total = len(members)

        def _row(m: dict[str, Any]) -> str:
            name = _display_name(m)
            role = str(m.get("role") or "")
            crown = "👑 " if role in ("admin", "superadmin") else ""
            return f"{crown}{_esc(name, p) if p == 'telegram' else name}"

        if p == "discord":
            admin_names = ", ".join(_display_name(m) for m in admins[:10]) or "—"
            return {
                "title": f"👥 Members ({total})",
                "fields": [
                    {"name": f"👑 Admins ({len(admins)})", "value": admin_names,
                     "inline": False},
                    {"name": f"Members ({len(regulars)})",
                     "value": ", ".join(_display_name(m) for m in regulars[:20]) or "—",
                     "inline": False},
                ],
                "footer": {"text": f"showing {min(total, limit)} of {total}"} if total > limit else None,
            }

        if p == "whatsapp":
            lines = [f"👥 *Members ({total})*"]
            if admins:
                lines.append(f"\n👑 *Admins ({len(admins)})*")
                lines.extend(f"• { _display_name(m)}" for m in admins[:limit])
            if regulars:
                lines.append(f"\n*Members ({len(regulars)})*")
                lines.extend(f"• {_display_name(m)}" for m in regulars[:max(0, limit - len(admins))])
            if total > limit:
                lines.append(f"\n_…and {total - limit} more_")
            return "\n".join(lines)

        # telegram HTML
        lines = [f"👥 <b>Members ({total})</b>"]
        if admins:
            lines.append(f"\n👑 <b>Admins ({len(admins)})</b>")
            lines.extend(f"• {_row(m)}" for m in admins[:limit])
        if regulars:
            lines.append(f"\n<b>Members ({len(regulars)})</b>")
            lines.extend(f"• {_row(m)}" for m in regulars[:max(0, limit - len(admins))])
        if total > limit:
            lines.append(f"\n<i>…and {total - limit} more</i>")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return f"{len(members or [])} members"


def render_community(info: dict[str, Any],
                     subgroups: list[dict[str, Any]] | None = None,
                     platform: str = "telegram") -> str | dict[str, Any]:
    """A community overview: info + linked subgroups."""
    p = _plat(platform)
    try:
        card = render_group_card(info, platform=platform)
        subgroups = subgroups or []
        if not subgroups:
            return card
        sub_lines = [f"🏘️ {s.get('subject', 'unnamed')} ({s.get('size', '?')})"
                     for s in subgroups[:20]]
        suffix = f"\n\n*Subgroups ({len(subgroups)})*\n" + "\n".join(f"• {l}" for l in sub_lines) \
            if p == "whatsapp" else \
            f"\n\n<b>Subgroups ({len(subgroups)})</b>\n" + "\n".join(f"• {html.escape(l) if p == 'telegram' else l}" for l in sub_lines)
        if isinstance(card, dict):
            card = dict(card)
            fields = list(card.get("fields") or [])
            fields.append({"name": f"🏘️ Subgroups ({len(subgroups)})",
                           "value": "\n".join(sub_lines[:10]), "inline": False})
            card["fields"] = fields
            return card
        return str(card) + suffix
    except Exception:  # noqa: BLE001
        return render_group_card(info, platform=platform)


def render_channel(info: dict[str, Any],
                   platform: str = "telegram") -> str | dict[str, Any]:
    """A channel info card."""
    p = _plat(platform)
    try:
        name = str(info.get("name") or info.get("subject") or "Unnamed channel")
        desc = str(info.get("description") or info.get("desc") or "")
        followers = info.get("followers", info.get("size", 0))
        if p == "discord":
            return {
                "title": f"📢 {name}",
                "description": desc or None,
                "fields": [{"name": "Followers", "value": str(followers), "inline": True}],
            }
        if p == "whatsapp":
            lines = [f"📢 *{name}*"]
            if desc:
                lines.append(f"_{desc}_")
            lines.append(f"\n👥 {followers} followers")
            return "\n".join(lines)
        lines = [f"📢 <b>{html.escape(name)}</b>"]
        if desc:
            lines.append(f"<i>{html.escape(desc)}</i>")
        lines.append(f"\n👥 {followers} followers")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return str(info.get("name") or "channel")


def render_group_list(groups: list[dict[str, Any]],
                      platform: str = "telegram") -> str | dict[str, Any]:
    """A compact list of groups/channels the account is in."""
    p = _plat(platform)
    try:
        groups = groups or []
        if not groups:
            return "no groups found" if p != "discord" else {
                "title": "Groups", "description": "no groups found"}
        rows = []
        for g in groups[:30]:
            name = str(g.get("subject") or g.get("name") or "unnamed")
            size = g.get("size", "?")
            rows.append((name, size))
        if p == "discord":
            return {
                "title": f"💬 Groups ({len(groups)})",
                "description": "\n".join(f"• {n} ({s})" for n, s in rows),
            }
        if p == "whatsapp":
            lines = [f"💬 *Groups ({len(groups)})*"]
            lines.extend(f"• {n} ({s} members)" for n, s in rows)
            return "\n".join(lines)
        lines = [f"💬 <b>Groups ({len(groups)})</b>"]
        lines.extend(f"• {html.escape(n)} ({s} members)" for n, s in rows)
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return f"{len(groups or [])} groups"
