"""Group defense — the automatic-admin organ (DESIGN §17 +
defense brief §2).

Sensors on top of the anticrash organ
(:mod:`nomorals.security.payload_inspection`) and the
Moderator: antichannel, antitag, antidelete and antiedit as
evidentiary sensors, and ONE escalation spine — a per-member
strike ledger shared by every sensor and by manual warns, so
the ladder counts history instead of resetting per tool
(warn → mute at 3 → kick at 5 → ban at 7, 7-day window).

Platform honesty is structural here: availability is DATA the
settings surface renders (Telegram antidelete is impossible via
the Bot API — MTProto-only; a sensor without platform power
degrades to warn + audit, never a fake execution).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ...security.payload_inspection import MessageEnvelope

DEFAULTS: dict[str, Any] = {
    "autonomy": True,
    "anticrash": {"enabled": True, "action": "delete"},
    "antichannel": {"enabled": True, "action": "delete_warn",
                    "exempt_linked": True},
    "antitag": {"enabled": True, "max_mentions": 10,
                "max_roster_pct": 20, "action": "delete_warn"},
    "antidelete": {"enabled": True, "mode": "log",
                   "callout_to": "admins"},
    "antiedit": {"enabled": True, "mode": "log",
                 "callout_to": "admins"},
    "escalation": {"window_days": 7, "mute_minutes": 60,
                   "kick_at": 5, "ban_at": 7},
}


def _merge(base: dict[str, Any], stored: dict[str, Any]
           ) -> dict[str, Any]:
    out = dict(base)
    for key, value in (stored or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            inner = dict(out[key])
            inner.update(value)
            out[key] = inner
        else:
            out[key] = value
    return out


@dataclass
class DefenseConfig:
    autonomy: bool = True
    anticrash_enabled: bool = True
    anticrash_action: str = "delete"      # delete | flag
    antichannel_enabled: bool = True
    antichannel_action: str = "delete_warn"
    antichannel_exempt_linked: bool = True
    antitag_enabled: bool = True
    antitag_max_mentions: int = 10
    antitag_max_roster_pct: int = 20
    antitag_action: str = "delete_warn"
    antidelete_enabled: bool = True
    antidelete_mode: str = "log"         # log | callout
    antidelete_callout_to: str = "admins"  # admins | group
    antiedit_enabled: bool = True
    antiedit_mode: str = "log"
    antiedit_callout_to: str = "admins"
    window_days: int = 7
    mute_minutes: int = 60
    kick_at: int = 5
    ban_at: int = 7

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> "DefenseConfig":
        raw = _merge(DEFAULTS, (settings or {}).get("defense") or {})
        ac = raw["anticrash"]
        ch = raw["antichannel"]
        tg = raw["antitag"]
        dl = raw["antidelete"]
        ed = raw["antiedit"]
        esc = raw["escalation"]
        action = str(ac.get("action") or "delete").lower()
        return cls(
            autonomy=bool(raw.get("autonomy", True)),
            anticrash_enabled=bool(ac.get("enabled", True)),
            anticrash_action=action if action in ("delete", "flag")
            else "delete",
            antichannel_enabled=bool(ch.get("enabled", True)),
            antichannel_action=str(ch.get("action")
                                   or "delete_warn"),
            antichannel_exempt_linked=bool(
                ch.get("exempt_linked", True)),
            antitag_enabled=bool(tg.get("enabled", True)),
            antitag_max_mentions=max(
                1, int(tg.get("max_mentions") or 10)),
            antitag_max_roster_pct=max(
                1, int(tg.get("max_roster_pct") or 20)),
            antitag_action=str(tg.get("action") or "delete_warn"),
            antidelete_enabled=bool(dl.get("enabled", True)),
            antidelete_mode=str(dl.get("mode") or "log").lower(),
            antidelete_callout_to=str(
                dl.get("callout_to") or "admins").lower(),
            antiedit_enabled=bool(ed.get("enabled", True)),
            antiedit_mode=str(ed.get("mode") or "log").lower(),
            antiedit_callout_to=str(
                ed.get("callout_to") or "admins").lower(),
            window_days=max(1, int(esc.get("window_days") or 7)),
            mute_minutes=max(1, int(esc.get("mute_minutes") or 60)),
            kick_at=max(0, int(esc.get("kick_at") or 0)),
            ban_at=max(0, int(esc.get("ban_at") or 0)),
        )


# ── sensor hits ─────────────────────────────────────────────────

@dataclass
class SensorHit:
    rule: str                       # sensor id ("antichannel", …)
    reason: str                     # one human line (audit + warning)
    action: str                     # configured action string
    signal: dict[str, Any] = field(default_factory=dict)


def check_antichannel(env: MessageEnvelope, cfg: DefenseConfig, *,
                      linked_chat_id: str = "") -> SensorHit | None:
    """Channel content injected into a group: a channel speaking
    as itself, an automatic channel forward, or a channel
    membership link in the text.  The group's OWN linked channel
    is exempt by default — linked-channel posts are the feature,
    not the attack.  Membership links are matched STRUCTURALLY
    (URL entities + link shapes), never by channel name."""
    if not cfg.antichannel_enabled:
        return None
    fwd = env.forward or {}
    sender_chat = str(fwd.get("sender_chat_id") or "")

    def _exempt(chat_id: str) -> bool:
        return bool(cfg.antichannel_exempt_linked
                    and linked_chat_id and chat_id
                    and chat_id == str(linked_chat_id))

    if fwd.get("is_automatic") and cfg.antichannel_exempt_linked:
        # An automatic forward into a group comes ONLY from the
        # group's own linked channel (a platform invariant) —
        # that is the feature working, not an injection.
        return None
    if fwd.get("origin_kind") == "channel" \
            and not _exempt(sender_chat):
        return SensorHit(
            "antichannel",
            "message injected from a channel "
            f"({fwd.get('origin_name') or 'unnamed channel'})",
            cfg.antichannel_action,
            {"origin": "forward", "sender_chat_id": sender_chat})
    if sender_chat and not _exempt(sender_chat):
        return SensorHit(
            "antichannel",
            "message posted as a channel identity",
            cfg.antichannel_action,
            {"origin": "sender_chat", "sender_chat_id": sender_chat})
    if str(env.platform).startswith("whatsapp") \
            and "@newsletter" in f"{env.chat_key} {env.sender_id}":
        return SensorHit(
            "antichannel",
            "channel (newsletter) content in the group",
            cfg.antichannel_action, {"origin": "newsletter"})
    haystacks = [env.text.lower()]
    for ent in env.entities:
        if ent.get("type") not in ("url", "text_link"):
            continue
        start = int(ent.get("offset") or 0)
        length = int(ent.get("length") or 0)
        if length:
            haystacks.append(env.text[start:start + length].lower())
        if ent.get("url"):
            haystacks.append(str(ent["url"]).lower())
    if any("whatsapp.com/channel/" in h for h in haystacks):
        return SensorHit(
            "antichannel",
            "channel membership link posted in the group",
            cfg.antichannel_action, {"origin": "channel_link"})
    return None


def check_antitag(env: MessageEnvelope, cfg: DefenseConfig, *,
                  roster_size: int = 0,
                  sender_is_admin: bool = False) -> SensorHit | None:
    """Mass-mention protection: more than the configured
    absolute ceiling OR more than the configured share of the
    group's roster in one message — the smaller bound wins, so
    the rule tracks group size.  @everyone counts as the whole
    roster.  Admins are exempt (announcements are their job)."""
    if not cfg.antitag_enabled or sender_is_admin:
        return None
    mentioned = len(env.mentions)
    everyone = any(e.get("type") == "mention_everyone"
                   for e in env.entities)
    if everyone:
        mentioned = max(mentioned, roster_size or mentioned + 1)
    bound = cfg.antitag_max_mentions
    if roster_size > 0:
        pct_bound = max(1, int(roster_size
                               * cfg.antitag_max_roster_pct / 100))
        bound = min(bound, pct_bound)
    if mentioned > bound:
        return SensorHit(
            "antitag",
            f"mass tag — {mentioned} members mentioned in one "
            f"message (limit {bound})",
            cfg.antitag_action,
            {"mentions": mentioned, "bound": bound,
             "roster": roster_size, "everyone": everyone})
    return None


# ── the escalation spine ────────────────────────────────────────

def add_strike(member: dict[str, Any], *, rule: str, action: str,
               now: float, window_days: int) -> int:
    """Append one strike to the member ledger (pruning entries
    outside the window first) and return the live count."""
    strikes = [s for s in (member.get("strikes") or [])
               if isinstance(s, dict)
               and now - float(s.get("ts") or 0)
               <= window_days * 86400]
    strikes.append({"rule": rule, "action": action, "ts": now})
    member["strikes"] = strikes
    return len(strikes)


def strike_count(member: dict[str, Any], *, now: float,
                 window_days: int) -> int:
    return len([s for s in (member.get("strikes") or [])
                if isinstance(s, dict)
                and now - float(s.get("ts") or 0)
                <= window_days * 86400])


def recent_strikes(member: dict[str, Any], *, now: float,
                   window_days: int) -> list[dict[str, Any]]:
    return [s for s in (member.get("strikes") or [])
            if isinstance(s, dict)
            and now - float(s.get("ts") or 0)
            <= window_days * 86400]


def ladder_rung(count: int, cfg: DefenseConfig, *,
                warn_threshold: int) -> str:
    """The ladder action when a NEW strike lands exactly on a
    rung — "" between rungs.  A rung of 0 disables that step
    (an owner-set 0 is a choice, not a default)."""
    if cfg.ban_at and count == cfg.ban_at:
        return "ban"
    if cfg.kick_at and count == cfg.kick_at:
        return "kick"
    if warn_threshold and count == warn_threshold:
        return "mute"
    return ""


# ── public warnings (house style: one message, the record) ─────

RULE_PHRASES: dict[str, str] = {
    "anticrash": "that payload can crash members' phones",
    "antichannel": "channel posts don't belong in this group",
    "antitag": "mass-tagging the group isn't allowed",
    "flood": "flooding the group isn't allowed",
    "links": "that link isn't allowed here",
    "words": "that word is filtered here",
    "repeat": "repeating the same message isn't allowed",
    "banlist": "you're on this group's ban list",
    "manual_warn": "an admin warned you",
}


def rule_phrase(rule: str) -> str:
    return RULE_PHRASES.get(rule, f"rule broken: {rule}")


def render_warning(mention: str, rule: str, consequence: str) -> str:
    """⛔ {who} — {rule, plain}. {what happened, and what comes
    next if it repeats}.  One message; it IS the record (the
    no-meta-narration law)."""
    return (f"⛔ {mention} — {rule_phrase(rule)}. "
            f"{consequence}")


def consequence_text(*, executed: str, count: int,
                     cfg: DefenseConfig, warn_threshold: int
                     ) -> str:
    """What happened + the next rung, in the member's terms."""
    parts = []
    if executed == "delete":
        parts.append("Message deleted")
    elif executed == "mute":
        parts.append(f"Message deleted — muted for "
                     f"{cfg.mute_minutes} min")
    elif executed == "kick":
        parts.append("Removed from the group")
    elif executed == "ban":
        parts.append("Banned from the group")
    elif executed == "warn":
        parts.append("This is a warning")
    if executed not in ("kick", "ban"):
        parts.append(f"strike {count}")
        nxt = _next_rung(count, cfg, warn_threshold)
        if nxt:
            parts.append(nxt)
    return ". ".join(parts) + "."


def _next_rung(count: int, cfg: DefenseConfig,
               warn_threshold: int) -> str:
    rungs = []
    if warn_threshold and count < warn_threshold:
        rungs.append((warn_threshold,
                      f"mute at {warn_threshold}"))
    if cfg.kick_at and count < cfg.kick_at:
        rungs.append((cfg.kick_at, f"removal at {cfg.kick_at}"))
    if cfg.ban_at and count < cfg.ban_at:
        rungs.append((cfg.ban_at, f"ban at {cfg.ban_at}"))
    if not rungs:
        return ""
    rungs.sort()
    return f"next: {rungs[0][1]} strikes"


# ── platform honesty ────────────────────────────────────────────

#: (platform-prefix, adapter kind) → what the delete/edit
#: sensors can actually see.  Adapter kinds: "botapi" (Telegram
#: Bot API), "mtproto" (Telegram userbot), "baileys" (WhatsApp),
#: "user" (Discord user session).
AVAILABILITY: dict[tuple[str, str], dict[str, Any]] = {
    ("telegram-bot", "botapi"): {
        "antidelete": False,
        "antidelete_note": "antidelete unavailable on the bot "
        "surface — delete events need the userbot (MTProto) "
        "surface",
        "antiedit": True, "antiedit_note": ""},
    ("telegram", "mtproto"): {
        "antidelete": True, "antidelete_note": "",
        "antiedit": True, "antiedit_note": ""},
    ("whatsapp", "baileys"): {
        "antidelete": True, "antidelete_note": "",
        "antiedit": True, "antiedit_note": ""},
    ("discord", "user"): {
        "antidelete": True, "antidelete_note": "",
        "antiedit": True, "antiedit_note": ""},
}


def defense_availability(platform: str, adapter_kind: str
                         ) -> dict[str, Any]:
    """Honest capability data for one surface.  Unknown surfaces
    claim nothing (False + a plain note), never optimism."""
    key = (platform, adapter_kind)
    if key in AVAILABILITY:
        return dict(AVAILABILITY[key])
    if platform.startswith("telegram"):
        return dict(AVAILABILITY[("telegram-bot", "botapi")])
    return {
        "antidelete": False,
        "antidelete_note": f"delete events are not visible on the "
        f"{platform or 'this'} surface",
        "antiedit": False,
        "antiedit_note": f"edit events are not visible on the "
        f"{platform or 'this'} surface"}


def now_ts() -> float:
    return time.time()
