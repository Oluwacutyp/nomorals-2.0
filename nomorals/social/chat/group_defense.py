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

import re
import time
import unicodedata
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
    # Media locks (DESIGN §4 + Rose locks): one toggle per kind,
    # all OFF by default — an unconfigured group sees no change.
    # The sensor itself is on; with every lock off it never
    # fires.  `action` is the block action (the antichannel
    # vocabulary, extended with mute/kick/ban); `actions` holds
    # per-kind overrides — a kind's own action wins over the
    # block's.  Silent variants (sban/smute/skick/swarn) are
    # config-only until Silent Actions lands: the sensor maps
    # them to their loud equivalent and says so in the signal.
    "media_locks": {"enabled": True, "action": "delete_warn",
                    "stickers": False, "gifs": False,
                    "voice": False, "video_notes": False,
                    "photos": False, "videos": False,
                    "audio": False, "documents": False,
                    "forwards": False, "channel_posts": False,
                    "links": False, "games": False,
                    "polls": False, "rtl": False,
                    "emails": False, "phones": False,
                    "actions": {}},
    "scamlink": {"enabled": True, "action": "delete_warn",
                 "suspicious": "flag", "max_links": 3,
                 "young_days": 30},
    "escalation": {"window_days": 7, "mute_minutes": 60,
                   "kick_at": 5, "ban_at": 7},
}


#: The media kinds a lock can cover (DESIGN §4 names the first
#: five; the rest are the family the envelope expresses).  The
#: Rose extension adds the text/content kinds: links (a URL in
#: the text or a url/text_link entity), games (a Telegram game
#: or dice message), polls (a poll message), rtl (right-to-left
#: text), emails (an address in the text), phones (a phone
#: number in the text).  There is deliberately no `bot` lock:
#: the adapters drop bot senders at intake, so the surface
#: never sees a bot message for a lock to fire on.
MEDIA_LOCK_KINDS: tuple[str, ...] = (
    "stickers", "gifs", "voice", "video_notes", "photos",
    "videos", "audio", "documents", "forwards", "channel_posts",
    "links", "games", "polls", "rtl", "emails", "phones",
)

#: Display labels for the lock kinds (grammar + panels).
MEDIA_LOCK_LABELS: dict[str, str] = {
    "stickers": "stickers", "gifs": "GIFs", "voice": "voice",
    "video_notes": "video notes", "photos": "photos",
    "videos": "videos", "audio": "audio", "documents": "files",
    "forwards": "forwards", "channel_posts": "channel posts",
    "links": "links", "games": "games", "polls": "polls",
    "rtl": "RTL text", "emails": "email addresses",
    "phones": "phone numbers",
}

#: The actions a lock can carry.  The block action and every
#: per-kind override come from this vocabulary; anything else
#: degrades to delete_warn, never to a crash and never to
#: silently not locking.
LOCK_ACTIONS: tuple[str, ...] = (
    "delete", "warn", "delete_warn", "mute", "kick", "ban",
)

#: Silent-action variants (Rose Silent Actions): config-only
#: until that phase lands — a lock configured with one behaves
#: as its loud equivalent, and the hit's signal says so.
LOCK_SILENT_ACTIONS: tuple[str, ...] = (
    "sban", "smute", "skick", "swarn",
)
LOCK_ALL_ACTIONS: tuple[str, ...] = (
    LOCK_ACTIONS + LOCK_SILENT_ACTIONS)
LOCK_SILENT_MAP: dict[str, str] = {
    "sban": "ban", "smute": "mute", "skick": "kick",
    "swarn": "warn",
}

#: Raw envelope media kinds → canonical lock kinds.  Producers
#: spell the same media differently (Telegram `photo` vs the
#: WhatsApp bridge's `image`), so the sensor normalizes before
#: it compares — a lock fires on what the message IS, never on
#: one platform's spelling of it.
_RAW_KIND_MAP: dict[str, str] = {
    "photo": "photos", "image": "photos",
    "video": "videos",
    "animation": "gifs", "gif": "gifs",
    "audio": "audio",
    "voice": "voice", "ptt": "voice",
    "video_note": "video_notes",
    "document": "documents", "file": "documents",
    "sticker": "stickers",
}

#: MIME fallback when the envelope carries no kind at all (a
#: bridge too old to send one): the family only, never a
#: voice/sticker guess — those need the kind to tell them apart.
_MIME_KIND_MAP: dict[str, str] = {
    "image/gif": "gifs", "image": "photos",
    "video": "videos", "audio": "audio",
}


#: The per-group escalation ladder (DESIGN §30): the warn limit
#: is the final-action count; interval actions are DESIGNED steps
#: at stated counts (mute / restrict / fine / kick), each
#: optional; the window is when warns expire.  The defaults ARE
#: the shipped ladder — mute at 3, kick at 5, ban at 7, 7-day
#: window — so a group that never configures it sees no change.
ESCALATION_DEFAULTS: dict[str, Any] = {
    "warn_limit": 7,
    "final_action": "ban",              # kick | ban
    "window_days": 7,
    "intervals": [
        {"at": 3, "action": "mute", "minutes": 60},
        {"at": 5, "action": "kick"},
    ],
}

#: Interval actions the ladder can fire at a designed count.
INTERVAL_ACTIONS = ("mute", "restrict", "fine", "kick")


def _normalize_intervals(raw: Any) -> list[dict[str, Any]]:
    """Coerce a stored intervals list into clean steps: known
    actions only, at >= 1, sorted by count, one step per count
    (a later entry at the same count wins — it is the newer
    design)."""
    by_at: dict[int, dict[str, Any]] = {}
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "").lower()
        if action not in INTERVAL_ACTIONS:
            continue
        try:
            at = int(item.get("at") or 0)
        except (TypeError, ValueError):
            continue
        if at < 1:
            continue
        step: dict[str, Any] = {"at": at, "action": action}
        if action == "mute":
            step["minutes"] = max(
                1, int(item.get("minutes") or 60))
        elif action == "fine":
            step["amount"] = max(
                1, int(item.get("amount") or 0)) or 1
        by_at[at] = step
    return [by_at[at] for at in sorted(by_at)]


@dataclass
class EscalationConfig:
    """One group's ladder (§30), read by EVERY strike consumer:
    the defense sensors, manual warns, acceptance enforcement.
    ``from_settings`` reads the ``escalation`` settings block
    when present and otherwise derives the same ladder from the
    legacy defense escalation fields + warn_threshold, so raw
    settings dicts from before the block existed behave
    identically."""
    warn_limit: int = 7
    final_action: str = "ban"
    window_days: int = 7
    intervals: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_settings(cls, settings: dict[str, Any]
                      ) -> "EscalationConfig":
        settings = settings or {}
        block = settings.get("escalation")
        if isinstance(block, dict):
            merged = dict(ESCALATION_DEFAULTS)
            merged.update({k: v for k, v in block.items()
                           if k in merged})
            final = str(merged.get("final_action")
                        or "ban").lower()
            return cls(
                warn_limit=max(1, int(merged.get("warn_limit")
                                      or 7)),
                final_action=final if final in ("kick", "ban")
                else "ban",
                window_days=max(1, int(merged.get("window_days")
                                       or 7)),
                intervals=_normalize_intervals(
                    merged.get("intervals")))
        defense = _merge(DEFAULTS,
                         settings.get("defense") or {})
        legacy = defense["escalation"]
        threshold = int(settings.get("warn_threshold") or 0)
        mute_minutes = max(1, int(legacy.get("mute_minutes")
                                  or 60))
        kick_at = int(legacy.get("kick_at") or 0)
        ban_at = int(legacy.get("ban_at") or 0)
        intervals: list[dict[str, Any]] = []
        if threshold:
            intervals.append({"at": threshold, "action": "mute",
                              "minutes": mute_minutes})
        if kick_at and kick_at != ban_at:
            intervals.append({"at": kick_at, "action": "kick"})
        if ban_at:
            limit, final = ban_at, "ban"
        elif kick_at:
            limit, final = kick_at, "kick"
            intervals = [s for s in intervals
                         if s["at"] != kick_at]
        else:
            limit, final = 7, "ban"
        return cls(warn_limit=limit, final_action=final,
                   window_days=max(
                       1, int(legacy.get("window_days") or 7)),
                   intervals=_normalize_intervals(intervals))

    def interval_at(self, count: int) -> dict[str, Any] | None:
        """The designed step landing exactly on ``count``."""
        for step in self.intervals:
            if step["at"] == count:
                return step
        return None

    def action_at(self, count: int) -> dict[str, Any] | None:
        """What fires when a new strike lands on ``count``: the
        interval step at that count, else the final action when
        the limit is reached (counts past the limit keep firing
        the final action — a banned member re-offending after a
        rejoin does not fall off the ladder)."""
        step = self.interval_at(count)
        if step is not None:
            return step
        if count >= self.warn_limit:
            return {"at": self.warn_limit,
                    "action": self.final_action}
        return None

    def next_step(self, count: int) -> dict[str, Any] | None:
        """The next designed consequence after ``count`` strikes:
        the nearest interval above it, else the final action."""
        for step in self.intervals:
            if step["at"] > count:
                return step
        if count < self.warn_limit:
            return {"at": self.warn_limit,
                    "action": self.final_action}
        return None

    def mute_minutes(self) -> int:
        """Minutes of the first mute step (the legacy flat
        figure other organs quote)."""
        for step in self.intervals:
            if step["action"] == "mute":
                return int(step.get("minutes") or 60)
        return 60

    def describe(self) -> str:
        """The ladder in one plain line, for settings surfaces."""
        parts = []
        for step in self.intervals:
            label = step["action"]
            if label == "mute":
                label = f"mute {step.get('minutes', 60)}m"
            elif label == "fine":
                label = f"fine {step.get('amount', 0)}"
            parts.append(f"{label} at {step['at']}")
        parts.append(f"{self.final_action} at {self.warn_limit}")
        return " → ".join(parts) + \
            f" · warns expire after {self.window_days}d"


def ladder_label(step: dict[str, Any]) -> str:
    """One ladder step in the member's terms ("mute for 60 min",
    "a 250-spark fine", "removal", "a ban")."""
    action = str(step.get("action") or "")
    if action == "mute":
        return f"mute for {int(step.get('minutes') or 60)} min"
    if action == "restrict":
        return "restriction until an admin lifts it"
    if action == "fine":
        return f"a {int(step.get('amount') or 0)}-spark fine"
    if action == "kick":
        return "removal"
    if action == "ban":
        return "a ban"
    return action


def warn_expiry_text(oldest_ts: float, *, window_days: int,
                     now: float) -> str:
    """When the oldest active warn clears, in the member's
    terms ("oldest warn clears 10 Nov") — "" when there is no
    active warn to expire.  Dates render day-first like the
    reference notices the owner captured (§31)."""
    if not oldest_ts:
        return ""
    clears = time.localtime(
        float(oldest_ts) + window_days * 86400)
    return ("oldest warn clears "
            + time.strftime("%d %b", clears))


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
    # Media locks (DESIGN §4): the merged block, read by the
    # sensor — `enabled` + `action` + one bool per kind.
    media_locks: dict[str, Any] = field(default_factory=dict)
    scamlink_enabled: bool = True
    scamlink_action: str = "delete_warn"
    scamlink_suspicious: str = "flag"   # flag | off — what a
                                        # suspicious link does
                                        # (never a delete)
    scamlink_max_links: int = 3
    scamlink_young_days: int = 30
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
        ml = raw["media_locks"]
        sl = raw["scamlink"]
        esc = raw["escalation"]
        # The §30 ladder is the source of truth when the group
        # has an escalation block: the flat fields below are
        # DERIVED from it (window, first mute's minutes, the
        # kick interval, the final action) so ladder_rung and
        # every older consumer quote the group's own ladder.
        if isinstance((settings or {}).get("escalation"), dict):
            ladder = EscalationConfig.from_settings(settings)
            kick_step = next(
                (s for s in ladder.intervals
                 if s["action"] == "kick"), None)
            return cls(
                autonomy=bool(raw.get("autonomy", True)),
                anticrash_enabled=bool(ac.get("enabled", True)),
                anticrash_action=str(ac.get("action")
                                     or "delete"),
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
                antitag_action=str(tg.get("action")
                                   or "delete_warn"),
                antidelete_enabled=bool(dl.get("enabled", True)),
                antidelete_mode=str(dl.get("mode")
                                    or "log").lower(),
                antidelete_callout_to=str(
                    dl.get("callout_to") or "admins").lower(),
                antiedit_enabled=bool(ed.get("enabled", True)),
                antiedit_mode=str(ed.get("mode") or "log").lower(),
                antiedit_callout_to=str(
                    ed.get("callout_to") or "admins").lower(),
                media_locks=dict(ml),
                scamlink_enabled=bool(sl.get("enabled", True)),
                scamlink_action=str(sl.get("action")
                                    or "delete_warn"),
                scamlink_suspicious=str(
                    sl.get("suspicious") or "flag").lower(),
                scamlink_max_links=max(
                    1, int(sl.get("max_links") or 3)),
                scamlink_young_days=max(
                    1, int(sl.get("young_days") or 30)),
                window_days=ladder.window_days,
                mute_minutes=ladder.mute_minutes(),
                kick_at=int(kick_step["at"]) if kick_step else 0,
                ban_at=ladder.warn_limit
                if ladder.final_action == "ban" else 0)
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
            media_locks=dict(ml),
            scamlink_enabled=bool(sl.get("enabled", True)),
            scamlink_action=str(sl.get("action")
                                or "delete_warn"),
            scamlink_suspicious=str(
                sl.get("suspicious") or "flag").lower(),
            scamlink_max_links=max(
                1, int(sl.get("max_links") or 3)),
            scamlink_young_days=max(
                1, int(sl.get("young_days") or 30)),
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


def _media_lock_kind(env: MessageEnvelope) -> str:
    """The message's canonical media kind, or "" when it carries
    no media.  The kind is the primary signal; the MIME family
    is only a fallback for a producer that sends no kind."""
    media = env.media or {}
    raw = str(media.get("kind") or "").strip().lower()
    if raw in _RAW_KIND_MAP:
        return _RAW_KIND_MAP[raw]
    if env.sticker is not None:
        return "stickers"
    mime = str(media.get("mime") or "").strip().lower()
    if mime in _MIME_KIND_MAP:
        return _MIME_KIND_MAP[mime]
    family = mime.split("/", 1)[0] if "/" in mime else ""
    return _MIME_KIND_MAP.get(family, "")


def _is_channel_post(env: MessageEnvelope) -> bool:
    """Channel-originated traffic: a forward from a channel, a
    message posted as a channel identity (sender_chat), or a
    WhatsApp newsletter posting into the group (the antichannel
    signal, reused — the lock is the admin's explicit toggle
    over the same family)."""
    fwd = env.forward or {}
    if fwd.get("origin_kind") == "channel":
        return True
    if fwd.get("sender_chat_id"):
        return True
    return (str(env.platform).startswith("whatsapp")
            and "@newsletter"
            in f"{env.chat_key} {env.sender_id}")


#: Bare-URL pattern for the links lock (the scamlink
#: extractor's shape: url/text_link entities first, bare URLs
#: in the text second).
_URL_RE = re.compile(r"https?://[^\s<>\"')\]]+")
_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
#: A phone number in the text: an optional +, then digits with
#: separators — the 9-digit floor in the matcher keeps dates
#: and short numbers out.
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def _text_lock_kinds(env: MessageEnvelope) -> list[str]:
    """The text/content lock kinds this message carries, in
    MEDIA_LOCK_KINDS order.  Links come from the platform's own
    url/text_link entities first, then bare URLs in the text
    (the scamlink extractor's order); games/polls ride the
    envelope flags the adapters stamp; RTL is a strong
    right-to-left character (Hebrew/Arabic ranges via the
    Unicode bidi class); emails and phones are conservative
    text patterns — a phone needs 9+ digits, so dates and short
    numbers never trip it."""
    text = str(env.text or "")
    found: list[str] = []
    if any(str(e.get("type") or "").lower()
           in ("url", "text_link")
           for e in (env.entities or [])) \
            or _URL_RE.search(text):
        found.append("links")
    if getattr(env, "game", False):
        found.append("games")
    if getattr(env, "poll", False):
        found.append("polls")
    if any(unicodedata.bidirectional(ch) in ("R", "AL")
           for ch in text):
        found.append("rtl")
    if _EMAIL_RE.search(text):
        found.append("emails")
    for match in _PHONE_RE.finditer(text):
        if sum(c.isdigit() for c in match.group(0)) >= 9:
            found.append("phones")
            break
    order = {k: i for i, k in enumerate(MEDIA_LOCK_KINDS)}
    return sorted(set(found), key=lambda k: order[k])


def _lock_hit(block: dict[str, Any] | None, kind: str,
              extra: dict[str, Any] | None = None
              ) -> tuple[str, dict[str, Any]]:
    """(action, signal) for one lock hit: the kind's own action
    wins over the block's; a silent variant maps to its loud
    equivalent and the signal keeps the configured form, so the
    audit record shows what the admin asked for and what ran.
    An unknown action degrades to delete_warn, never to a crash
    and never to silently not locking."""
    merged = dict(DEFAULTS["media_locks"])
    merged.update(block or {})
    action = str(merged.get("action") or "delete_warn").lower()
    overrides = merged.get("actions") or {}
    if isinstance(overrides, dict):
        own = str(overrides.get(kind) or "").lower()
        if own:
            action = own
    signal: dict[str, Any] = {"kind": kind}
    if extra:
        signal.update(extra)
    if action in LOCK_SILENT_MAP:
        signal["silent_action"] = action
        action = LOCK_SILENT_MAP[action]
    if action not in LOCK_ACTIONS:
        action = "delete_warn"
    return action, signal


def locked_media_kinds(block: dict[str, Any] | None) -> list[str]:
    """The kinds this group's media-locks block has switched on,
    in MEDIA_LOCK_KINDS order.  Unknown keys are ignored — a
    stored typo never becomes a lock."""
    merged = dict(DEFAULTS["media_locks"])
    merged.update(block or {})
    return [k for k in MEDIA_LOCK_KINDS if merged.get(k)]


def describe_media_locks(block: dict[str, Any] | None) -> str:
    """One state line for the panels and the dashboard: the
    locked kinds + the action, or the open state.  A kind with
    its own action names it; a silent action names its loud
    equivalent, and the line says it runs loud until Silent
    Actions lands."""
    merged = dict(DEFAULTS["media_locks"])
    merged.update(block or {})
    locked = locked_media_kinds(merged)
    if not locked:
        return "none locked — everything open"
    action = str(merged.get("action") or "delete_warn").lower()
    silent_any = action in LOCK_SILENT_MAP
    if silent_any:
        action = LOCK_SILENT_MAP[action]
    if action not in LOCK_ACTIONS:
        action = "delete_warn"
    overrides = merged.get("actions") or {}
    labels: list[str] = []
    for k in locked:
        own = ""
        if isinstance(overrides, dict):
            own = str(overrides.get(k) or "").lower()
        loud = LOCK_SILENT_MAP.get(own, own)
        if own and own in LOCK_ALL_ACTIONS and loud != action:
            labels.append(f"{MEDIA_LOCK_LABELS[k]} ({loud})")
            if own in LOCK_SILENT_MAP:
                silent_any = True
        else:
            if own in LOCK_SILENT_MAP:
                silent_any = True
            labels.append(MEDIA_LOCK_LABELS[k])
    out = f"{', '.join(labels)} locked ({action})"
    if silent_any:
        out += (" — silent actions run as their loud equivalent "
                "until Silent Actions lands")
    return out


def check_media_locks(env: MessageEnvelope, cfg: DefenseConfig, *,
                      linked_chat_id: str = "",
                      sender_is_admin: bool = False,
                      ) -> SensorHit | None:
    """Media locks (DESIGN §4 + Rose locks): a locked kind in
    this group is one hit — the first locked kind wins, so the
    pipeline strikes once per message.  Admins are exempt
    (antitag shape); the group's OWN linked channel is exempt
    (the antichannel rule — automatic forwards and the explicit
    linked id are the feature, not the attack, and a member
    manually forwarding the linked channel's post is still a
    forward, so only the channel-posts kind exempts on the id).
    Each hit carries its kind's own action (or the block's),
    with a silent variant mapped to its loud equivalent — the
    mapping lives in _lock_hit, so no call site can forget it.
    An unknown action degrades to delete_warn, never to a
    crash and never to silently not locking."""
    if sender_is_admin:
        return None
    block = cfg.media_locks or {}
    if not block.get("enabled", True):
        return None
    locked = locked_media_kinds(block)
    if not locked:
        return None
    fwd = env.forward or {}
    if fwd.get("is_automatic"):
        # An automatic forward into a group comes ONLY from the
        # group's own linked channel (a platform invariant).
        return None
    sender_chat = str(fwd.get("sender_chat_id") or "")
    if linked_chat_id and sender_chat \
            and sender_chat == str(linked_chat_id) \
            and not fwd.get("is_forwarded"):
        # The message speaks AS the group's own linked channel
        # (sender_chat, no forward header) — the feature, not
        # an injection.  A member manually forwarding the
        # linked channel's post IS a forward (is_forwarded),
        # so it still meets the forwards lock below.
        return None
    kind = _media_lock_kind(env)
    if kind and kind in locked:
        action, signal = _lock_hit(block, kind)
        return SensorHit(
            "media_locks",
            f"{MEDIA_LOCK_LABELS[kind]} are locked in this group",
            action, signal)
    for tkind in _text_lock_kinds(env):
        if tkind in locked:
            action, signal = _lock_hit(block, tkind)
            return SensorHit(
                "media_locks",
                f"{MEDIA_LOCK_LABELS[tkind]} are locked in this "
                "group",
                action, signal)
    if fwd.get("is_forwarded") and "forwards" in locked:
        action, signal = _lock_hit(
            block, "forwards",
            {"origin_kind": str(fwd.get("origin_kind") or "")})
        return SensorHit(
            "media_locks", "forwards are locked in this group",
            action, signal)
    if "channel_posts" in locked and _is_channel_post(env) \
            and not (linked_chat_id and sender_chat
                     and sender_chat == str(linked_chat_id)):
        # A post FROM the group's own linked channel is the
        # feature, not a foreign channel post — the exemption
        # rides the sender id, not the forward header, so a
        # member manually forwarding it still meets the
        # forwards lock above.
        action, signal = _lock_hit(
            block, "channel_posts",
            {"sender_chat_id": sender_chat})
        return SensorHit(
            "media_locks",
            "channel posts are locked in this group",
            action, signal)
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
    "scamlink": "that link is a known scam",
    "flood": "flooding the group isn't allowed",
    "links": "that link isn't allowed here",
    "words": "that word is filtered here",
    "word_filter": "that word is filtered here",
    "repeat": "repeating the same message isn't allowed",
    # legacy alias: records and notices stored before the
    # Moderator's rule key was aligned to "repeat" carry
    # "repeated" — they must render the same house phrase.
    "repeated": "repeating the same message isn't allowed",
    "banlist": "you're on this group's ban list",
    "manual_warn": "an admin warned you",
    "acceptance": "your profile doesn't meet this group's "
                  "acceptance rules",
}


def rule_phrase(rule: str) -> str:
    return RULE_PHRASES.get(rule, f"rule broken: {rule}")


def render_warning(mention: str, rule: str, consequence: str) -> str:
    """⛔ {who} — {rule, plain}. {what happened, and what comes
    next if it repeats}.  One message; it IS the record (the
    no-meta-narration law).  Composed by the typography organ
    (DESIGN §27, moderation class): the name + rule lead plain,
    the consequence carries the emphasis."""
    from .typography import moderation_warning

    return moderation_warning(mention, rule_phrase(rule),
                              consequence)


def consequence_text(*, executed: str, count: int,
                     cfg: DefenseConfig, warn_threshold: int,
                     esc: EscalationConfig | None = None,
                     fine_amount: int = 0,
                     oldest_ts: float = 0.0,
                     now: float = 0.0) -> str:
    """What happened + the member's ladder position, in their
    terms: the count of the limit, the next consequence, and
    when the oldest active warn clears (§30/§31).  With ``esc``
    the position comes from the group's own ladder; without it
    the legacy flat wording stands byte-for-byte."""
    parts = []
    if executed == "delete":
        parts.append("Message deleted")
    elif executed == "mute":
        minutes = cfg.mute_minutes
        if esc is not None:
            step = esc.interval_at(count)
            if step and step.get("action") == "mute":
                minutes = int(step.get("minutes") or minutes)
        parts.append(f"Message deleted — muted for "
                     f"{minutes} min")
    elif executed == "kick":
        parts.append("Removed from the group")
    elif executed == "ban":
        parts.append("Banned from the group")
    elif executed == "warn":
        parts.append("This is a warning")
    elif executed == "fine":
        parts.append(f"Fined {fine_amount} sparks"
                     if fine_amount else "Fined")
    elif executed == "restrict":
        parts.append("Restricted until an admin lifts it")
    if fine_amount and executed != "fine":
        parts.append(f"fined {fine_amount} sparks")
    if esc is not None:
        if executed not in ("kick", "ban"):
            parts.append(f"strike {count} of {esc.warn_limit}")
            nxt = esc.next_step(count)
            if nxt is not None:
                parts.append(
                    f"next: {ladder_label(nxt)} "
                    f"at {nxt['at']}")
            expiry = warn_expiry_text(
                oldest_ts, window_days=esc.window_days,
                now=now)
            if expiry:
                parts.append(expiry)
    elif executed not in ("kick", "ban"):
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
