"""Group surface (DESIGN §1 — Devon Telegram group & community
system, P1 wiring): the layer that claims non-DM chats and routes
them into the community organs.

Before this module the organs existed but nothing in the live
event path called them: `community/welcome_moderation.py` (welcome
flows, CaptchaGate, Moderator, BanList, Roster) worked at its own
layer while real group traffic fell through to the public-chat
machinery.  The surface is the missing wiring:

- per-group SETTINGS, persisted — one config read by both the
  enforcer and the rules renderer, so published rules can never
  drift from enforcement (DESIGN §12);
- a per-group MEMBER LEDGER: identity snapshots (user id,
  username, display name) with temporal change events — a
  username or name change is DETECTED and recorded the first time
  the member is observed after changing it;
- the MODERATION PIPELINE: every group text runs the community
  Moderator (flood / link allowlist / repeats / cross-group
  BanList) and yields actions for the caller to execute;
- WELCOME / GOODBYE composition from the group's own settings
  through the community welcome flow;
- a per-group AUDIT LOG that is HELD, never posted (owner ruling
  2026-11-01): actions, change events and unexecuted actions are
  recorded with reasons and read on demand.

Persistence follows the house rule: unreadable state is NEVER
silently reset (that would orphan members and settings) — a
corrupt store file is quarantined aside and GroupStoreError
raised.  Writes are atomic (tmp + replace).
"""
from __future__ import annotations

import copy
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...community.group_real_estate import (
    ESTATE_DEFAULTS,
    GroupRealEstate,
    GroupRealEstateError,
)
from ...community.welcome_moderation import (
    BanList,
    ModerationConfig,
    Moderator,
    WelcomeConfig,
    render_goodbye,
    render_welcome,
)
from ...security.payload_inspection import (
    SignatureStore,
    inspect_payload,
    normalize_envelope,
)
from .failures import record_failure
from .group_acceptance import ACCEPTANCE_DEFAULTS
from .group_reputation import (
    CATEGORIES as REPUTATION_CATEGORIES,
)
from .group_reputation import (
    PROPOSAL_TTL_S as REPUTATION_PROPOSAL_TTL_S,
)
from .group_reputation import (
    REPUTATION_DEFAULTS,
    ReputationBook,
    normalize_person,
)
from .group_defense import (
    DEFAULTS as DEFENSE_DEFAULTS,
)
from .expression import EXPRESSION_DEFAULTS
from .typography import DEFAULT_PRESENTATION, notice_line, sparks
from .group_defense import (
    ESCALATION_DEFAULTS,
    DefenseConfig,
    EscalationConfig,
    add_strike,
    check_antichannel,
    check_antitag,
    consequence_text,
    ladder_label,
    recent_strikes,
    render_warning,
)

AUDIT_RING = 500
NOTICE_RING = 100
CHANGE_RING = 200
MSG_CACHE_RING = 200

#: Platforms that ARE Devon-the-bot — her own bot identities
#: (DESIGN §33).  A group key on one of these exists only
#: because her bot is in that room receiving its traffic.
#: Every other platform key (``telegram`` the userbot,
#: ``whatsapp`` Baileys, ``discord``) is one of the owner's
#: PERSONAL accounts: those sessions converse as him (never
#: gated) but carry no bot-system standing on their own.
BOT_PLATFORMS = ("telegram-bot", "telegram-personal")
_ACTION_RANK = {"allow": 0, "warn": 1, "delete": 2, "mute": 3,
                "acceptance_restrict": 3, "kick": 4, "ban": 5}


def _deep_merge(base: dict[str, Any], stored: dict[str, Any]
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


class GroupStoreError(RuntimeError):
    """The group store exists but cannot be read.  Raised, never
    swallowed: starting over an unreadable store would silently
    drop every group's settings, ledger and audit history."""


def _default_settings() -> dict[str, Any]:
    return {
        "welcome_enabled": True,
        "welcome_text": "",
        "goodbye_enabled": False,
        "clean_service": True,          # delete raw join/leave
                                        # service messages after
                                        # handling (hygiene, §P1)
        "rules": [],
        "captcha_mode": "off",          # off | button (P1 wiring
                                        # of CaptchaGate lands with
                                        # the button router)
        "moderation": {
            "flood_max": 6,
            "flood_window_s": 10.0,
            "repeat_threshold": 3,
            "link_allowlist": [],
            "word_filters": [],
            "filter_action": "delete",
        },
        "economy_enabled": True,        # consumed by P2
        "transfer_tax_pct": 2,          # /pay fee to the treasury
        "warn_threshold": 3,            # warns before an auto-mute
        "mute_default_s": 0,            # standing mute length when
                                        # a name is given without
                                        # one (0 = ask each time)
        "fine_amount": 0,               # sparks an auto-mute costs
        "governance_threshold": 500,    # treasury spends above
                                        # this need a 2nd admin
        "defense": copy.deepcopy(DEFENSE_DEFAULTS),
                                        # automatic admin (§17):
                                        # autonomy + sensors +
                                        # escalation spine
        "estate": copy.deepcopy(ESTATE_DEFAULTS),
                                        # real estate (§17/§20):
                                        # market cut, sell-back,
                                        # ownership cap, mortgage
                                        # LTV + rate levers
        "presentation": copy.deepcopy(DEFAULT_PRESENTATION),
                                        # typography (§27): theme
                                        # pack, per-class preset
                                        # overrides, the stamp
        "acceptance": copy.deepcopy(ACCEPTANCE_DEFAULTS),
                                        # join acceptance (§28):
                                        # mode, criteria, actions,
                                        # grace, the raid posture
        "escalation": copy.deepcopy(ESCALATION_DEFAULTS),
                                        # the warn ladder (§30):
                                        # limit, interval steps,
                                        # final action, warn
                                        # expiry window — defaults
                                        # ARE the shipped 3/5/7
        "games_enabled": True,
        "system_claim": {"designated": False,
                         "bot_admin": None},
                                        # the §33 claim: may
                                        # Devon's BOT SYSTEMS
                                        # speak / accrue in this
                                        # room?  designated =
                                        # the owner named this
                                        # room for system output;
                                        # bot_admin = the
                                        # runtime's last verified
                                        # standing of her bot
                                        # (None = never probed)
        "expression": copy.deepcopy(EXPRESSION_DEFAULTS),
                                        # media expression
                                        # (§21): GIF/sticker
                                        # moments on/off, gif
                                        # frequency, welcome
                                        # media, ban theatre —
                                        # warnings stay dry at
                                        # every setting
        "reputation": copy.deepcopy(REPUTATION_DEFAULTS),
                                        # cross-group reputation
                                        # (§32): what this group
                                        # does when an admitted
                                        # joiner carries a live
                                        # record — public notice
                                        # / admins-only trail /
                                        # restricted until an
                                        # admin clears / off.  The
                                        # record never decides
                                        # admission itself.
        "state": "discovered",          # discovered|configured|
                                        # active|dormant
    }


@dataclass
class GroupStore:
    """Durable per-group state: settings + member ledger + audit.

    Shape: ``{group_key: {"settings": {...}, "members":
    {user_id: {...}}, "changes": [...], "audit": [...]}}``.
    """

    path: Path
    data: dict[str, dict[str, Any]] = field(default_factory=dict)

    def load(self) -> "GroupStore":
        if not self.path.exists():
            self.data = {}
            return self
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("group store root is not an object")
        except (OSError, ValueError) as exc:
            quarantine = self.path.with_suffix(
                f".corrupt-{int(time.time())}")
            try:
                self.path.replace(quarantine)
            except OSError:
                pass
            raise GroupStoreError(
                f"group store at {self.path} is unreadable "
                f"({type(exc).__name__}: {exc}); quarantined to "
                f"{quarantine} — refusing to start from empty "
                f"state") from exc
        self.data = raw
        return self

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1),
                       encoding="utf-8")
        tmp.replace(self.path)

    def group(self, group_key: str) -> dict[str, Any]:
        entry = self.data.get(group_key)
        if entry is None:
            entry = {"settings": _default_settings(),
                     "members": {}, "changes": [], "audit": [],
                     "join_requests": {},
                     "acceptance_state": {"recent_joins": [],
                                          "raid_until": 0.0,
                                          "req_seq": 0}}
            self.data[group_key] = entry
        return entry

    def settings(self, group_key: str) -> dict[str, Any]:
        merged = _default_settings()
        stored = self.group(group_key).get("settings") or {}
        merged.update({k: v for k, v in stored.items()
                       if k in merged
                       and k not in ("defense", "presentation",
                                     "estate", "acceptance",
                                     "escalation", "system_claim",
                                     "expression", "reputation")})
        mod = dict(merged["moderation"])
        mod.update((stored.get("moderation") or {}))
        merged["moderation"] = mod
        merged["defense"] = _deep_merge(
            copy.deepcopy(DEFENSE_DEFAULTS),
            stored.get("defense") or {})
        merged["estate"] = _deep_merge(
            copy.deepcopy(ESTATE_DEFAULTS),
            stored.get("estate") or {})
        merged["presentation"] = _deep_merge(
            copy.deepcopy(DEFAULT_PRESENTATION),
            stored.get("presentation") or {})
        merged["acceptance"] = _deep_merge(
            copy.deepcopy(ACCEPTANCE_DEFAULTS),
            stored.get("acceptance") or {})
        merged["escalation"] = _deep_merge(
            copy.deepcopy(ESCALATION_DEFAULTS),
            stored.get("escalation") or {})
        merged["system_claim"] = _deep_merge(
            {"designated": False, "bot_admin": None},
            stored.get("system_claim") or {})
        merged["expression"] = _deep_merge(
            copy.deepcopy(EXPRESSION_DEFAULTS),
            stored.get("expression") or {})
        merged["reputation"] = _deep_merge(
            copy.deepcopy(REPUTATION_DEFAULTS),
            stored.get("reputation") or {})
        return merged


def _slug(group_key: str) -> str:
    return str(group_key).replace(":", "_").replace("/", "_")


class GroupSurface:
    """One surface for every group/community chat, all platforms.

    Keys are chat keys (``platform:chat_id``).  The surface is
    deliberately transport-free: the runtime feeds it messages and
    service events and executes the actions it returns, so the
    same pipeline serves Telegram (bot + userbot), WhatsApp and
    Discord groups.
    """

    def __init__(self, store: GroupStore,
                 ban_list: BanList | None = None) -> None:
        self.store = store
        self.bans = ban_list or BanList()
        self._moderators: dict[str, Moderator] = {}
        self._msg_cache: dict[str, deque] = {}
        self._signatures: SignatureStore | None = None
        self._reputation: ReputationBook | None = None

    @property
    def signatures(self) -> SignatureStore:
        """Learned anticrash shapes, persisted beside the group
        store (derived data — the store quarantines itself on
        corruption and relearns by observation)."""
        if self._signatures is None:
            self._signatures = SignatureStore(
                self.store.path.parent / "payload_signatures.json")
        return self._signatures

    @property
    def reputation(self) -> ReputationBook:
        """The cross-group reputation book (§32): ONE global
        ledger beside the group store — records are evidence, so
        the book quarantines and raises on corruption rather than
        resetting (ReputationBook owns that discipline)."""
        if self._reputation is None:
            self._reputation = ReputationBook(
                self.store.path.parent / "reputation.json")
        return self._reputation

    def defense_config(self, group_key: str) -> DefenseConfig:
        return DefenseConfig.from_settings(
            self.store.settings(group_key))

    # ---- the system-output gate (DESIGN §33) --------------------
    # Two hats on the owner's personal accounts: his sessions
    # converse AS HIM in any room he belongs to (never gated —
    # that path does not come through here), while Devon's BOT
    # SYSTEMS (economy, quests, games, welcomes, notices,
    # briefings) may only speak and accrue in CLAIMED rooms.
    # This helper is the one gate every system-output path
    # consults, evaluated from live settings at send time, so
    # a claim that lapses stops output on the very next send.

    def claim_state(self, group_key: str) -> dict[str, Any]:
        """The room's §33 facts, exactly as recorded: owner
        designation, the runtime's last verified bot standing,
        the derived claim and its source ("" when unclaimed)."""
        claim = self.store.settings(group_key).get(
            "system_claim") or {}
        designated = bool(claim.get("designated"))
        bot_admin = claim.get("bot_admin")
        platform = str(group_key).split(":", 1)[0]
        bot_platform = platform in BOT_PLATFORMS
        bot = bot_admin is True or (
            bot_platform and bot_admin is not False)
        claimed = designated or bot
        source = ("designated" if designated
                  else "bot" if bot else "")
        return {"claimed": claimed, "designated": designated,
                "bot_admin": bot_admin,
                "bot_platform": bot_platform, "source": source}

    def system_output_allowed(self, group_key: str, *,
                              via_platform: str = "") -> bool:
        """May Devon's bot systems post into / accrue in this
        room right now?  Claimed when the owner designated the
        room, or when her bot operates there — proved by its
        verified admin standing, by the sending session being
        one of her bot platforms (``via_platform``, when the
        caller knows it), or by the room's own key living on a
        bot platform.  A VERIFIED demotion (bot_admin False)
        lapses the bot claim on every session at once; a
        designation stands until the owner lifts it."""
        state = self.claim_state(group_key)
        if state["designated"]:
            return True
        if state["bot_admin"] is True:
            return True
        if state["bot_admin"] is False:
            return False
        if state["bot_platform"]:
            return True
        session = str(via_platform or "").split(":", 1)[0]
        return session in BOT_PLATFORMS

    def set_designated(self, group_key: str, on: bool, *,
                       by: str = "") -> dict[str, Any]:
        """The owner's designation (§33b): name this room for
        system output, or lift the name.  Audited either way —
        the flip is a standing decision about where his
        personal sessions may speak as the bot system."""
        claim = dict(self.store.settings(group_key).get(
            "system_claim") or {})
        claim["designated"] = bool(on)
        self.update_settings(group_key, system_claim=claim)
        self._audit(group_key, "system_claim",
                    designated=bool(on), by=by)
        self.store.save()
        return self.claim_state(group_key)

    def note_bot_admin(self, group_key: str, is_admin: bool
                       ) -> None:
        """Record the runtime's verified standing of Devon's
        bot in this room (its probe of the bot's own member
        status).  Only a CHANGE writes + audits: a demotion
        lapsing the claim is an event worth the /glog line; a
        steady state is not."""
        claim = dict(self.store.settings(group_key).get(
            "system_claim") or {})
        if claim.get("bot_admin") is bool(is_admin):
            return
        claim["bot_admin"] = bool(is_admin)
        self.update_settings(group_key, system_claim=claim)
        self._audit(group_key, "bot_claim",
                    bot_admin=bool(is_admin))
        self.store.save()

    def _card_render(self, group_key: str, card: Any) -> str:
        """Render a typography Card (§27) for this group's
        platform — derived from the group key's prefix, the one
        platform fact the transport-free surface owns — under
        the group's own presentation settings."""
        from .typography import render_card

        platform = str(group_key).split(":", 1)[0]
        return render_card(
            card, platform=platform,
            settings=self.store.settings(group_key)).text

    # ---- settings / moderation plumbing -------------------------
    def _moderator(self, group_key: str) -> Moderator:
        mod = self._moderators.get(group_key)
        if mod is None:
            cfg = self.store.settings(group_key)["moderation"]
            mod = Moderator(
                ban_list=self.bans,
                config=ModerationConfig(
                    flood_max=int(cfg.get("flood_max") or 6),
                    flood_window_s=float(
                        cfg.get("flood_window_s") or 10.0),
                    repeat_threshold=int(
                        cfg.get("repeat_threshold") or 3),
                    link_allowlist=list(
                        cfg.get("link_allowlist") or []),
                    word_filters=list(
                        cfg.get("word_filters") or []),
                    filter_action=str(
                        cfg.get("filter_action") or "delete"),
                ))
            self._moderators[group_key] = mod
        return mod

    def refresh_moderator(self, group_key: str) -> None:
        """Drop a group's cached Moderator (settings changed)."""
        self._moderators.pop(group_key, None)

    def update_settings(self, group_key: str, **changes: Any
                        ) -> dict[str, Any]:
        entry = self.store.group(group_key)
        settings = self.store.settings(group_key)
        for key, value in changes.items():
            if key in settings:
                settings[key] = value
        entry["settings"] = settings
        if settings.get("state") == "discovered":
            settings["state"] = "configured"
        self.store.save()
        self.refresh_moderator(group_key)
        return settings

    # ---- audit ---------------------------------------------------
    def _audit(self, group_key: str, kind: str, **details: Any
               ) -> None:
        entry = self.store.group(group_key)
        entry["audit"].append(
            {"ts": time.time(), "kind": kind, **details})
        del entry["audit"][:-AUDIT_RING]

    def audit(self, group_key: str, limit: int = 50
              ) -> list[dict[str, Any]]:
        return list(self.store.group(group_key)["audit"])[-limit:]

    # ---- member ledger -------------------------------------------
    def _observe_identity(self, group_key: str, user_id: str, *,
                          username: str = "", name: str = "",
                          ts: float) -> list[dict[str, Any]]:
        """Snapshot a member's identity; return CHANGE EVENTS for
        anything that differs from the stored snapshot."""
        entry = self.store.group(group_key)
        member = entry["members"].get(user_id)
        events: list[dict[str, Any]] = []
        if member is None:
            entry["members"][user_id] = {
                "user_id": user_id,
                "username": username or "",
                "name": name or "",
                "first_seen": ts,
                "joined_ts": None,
                "last_seen": ts,
                "message_count": 0,
                "left_ts": None,
            }
            return events
        for field_name, new in (("username", username),
                                ("name", name)):
            if new and member.get(field_name) != new:
                event = {
                    "kind": f"{field_name}_changed",
                    "user_id": user_id,
                    "old": member.get(field_name) or "",
                    "new": new, "ts": ts}
                events.append(event)
                member[field_name] = new
        member["last_seen"] = ts
        if events:
            entry["changes"].extend(events)
            del entry["changes"][:-CHANGE_RING]
            for event in events:
                self._audit(group_key, "identity_change",
                            change_kind=event["kind"],
                            user_id=user_id, old=event["old"],
                            new=event["new"])
        return events

    def note_join(self, group_key: str, user_id: str, *,
                  username: str = "", name: str = "",
                  last_name: str = "", is_bot: bool = False,
                  has_photo: bool | None = None,
                  ts: float | None = None) -> dict[str, Any]:
        """Record a join in the ledger WITHOUT rendering the
        welcome (acceptance may hold it until admission — the
        welcome fires when the member is actually in).  The
        extra identity facts (last name, bot flag, photo
        presence) are stored beside the name so a later
        §28 re-check judges the same facts the join was
        judged on.  Returns the member record."""
        now = ts if ts is not None else time.time()
        self._observe_identity(group_key, user_id,
                               username=username, name=name, ts=now)
        entry = self.store.group(group_key)
        member = entry["members"][user_id]
        member["joined_ts"] = now
        member["left_ts"] = None
        if last_name:
            member["last_name"] = last_name
        if is_bot:
            member["is_bot"] = True
        if has_photo is not None:
            member["has_photo"] = bool(has_photo)
        self._audit(group_key, "join", user_id=user_id,
                    username=username, name=name)
        self.store.save()
        return member

    def welcome_text_for(self, group_key: str, *,
                         name: str = "", username: str = "",
                         group_name: str = "") -> str:
        """The welcome text for one member ("" when welcome is
        off), rendered from the group's own settings through
        the community welcome flow."""
        settings = self.store.settings(group_key)
        if not settings.get("welcome_enabled"):
            return ""
        config = WelcomeConfig(
            group_name=group_name,
            rules=list(settings.get("rules") or []))
        custom = str(settings.get("welcome_text") or "")
        if custom:
            config.welcome_template = custom
        body = render_welcome(name or username or "there", config)
        # The welcome is a presented surface (§27, announcement
        # class): the group's own template supplies the words,
        # the typography organ supplies the form — its first
        # line becomes the card's title, the stamp closes it.
        from .typography import Card

        lines = [ln for ln in str(body).splitlines() if ln.strip()]
        if not lines:
            return ""
        card = Card("announcement", title=lines[0], icon="👋")
        card.lines(lines[1:])
        return self._card_render(group_key, card)

    def record_join(self, group_key: str, user_id: str, *,
                    username: str = "", name: str = "",
                    group_name: str = "",
                    ts: float | None = None) -> str:
        """Record a join; return the welcome text to send ("" when
        welcome is off)."""
        self.note_join(group_key, user_id, username=username,
                       name=name, ts=ts)
        return self.welcome_text_for(
            group_key, name=name, username=username,
            group_name=group_name)

    def record_leave(self, group_key: str, user_id: str, *,
                     group_name: str = "",
                     ts: float | None = None) -> str:
        now = ts if ts is not None else time.time()
        entry = self.store.group(group_key)
        member = entry["members"].get(user_id)
        if member is not None:
            member["left_ts"] = now
        self._audit(group_key, "leave", user_id=user_id)
        self.store.save()
        settings = self.store.settings(group_key)
        if not settings.get("goodbye_enabled") or member is None:
            return ""
        config = WelcomeConfig(
            group_name=group_name,
            rules=list(settings.get("rules") or []))
        body = render_goodbye(
            member.get("name") or member.get("username") or "there",
            config)
        from .typography import Card

        card = Card("announcement", title=str(body), icon="👋")
        return self._card_render(group_key, card)

    # ---- join acceptance (DESIGN §28) -----------------------------
    # The evaluation organ lives in group_acceptance.py; the
    # surface owns its state: restricted records on the member
    # ledger, the admin request queue, the raid posture, and the
    # /glog trail of every evaluation and decision.

    def _acceptance_state(self, entry: dict[str, Any]
                          ) -> dict[str, Any]:
        return entry.setdefault(
            "acceptance_state",
            {"recent_joins": [], "raid_until": 0.0, "req_seq": 0})

    def _acceptance_raid(self, group_key: str,
                         entry: dict[str, Any], *,
                         now: float, count_join: bool = False
                         ) -> bool:
        """Join-burst bookkeeping: returns True while the raid
        posture holds.  Every arrival (walk-in or request) is a
        beat in the burst window."""
        cfg = self.store.settings(
            group_key).get("acceptance") or {}
        state = self._acceptance_state(entry)
        window = float(cfg.get("raid_window_s") or 120)
        recent = [t for t in (state.get("recent_joins") or [])
                  if now - float(t) <= window]
        if count_join:
            recent.append(now)
            state["recent_joins"] = recent[-64:]
            if len(recent) >= int(cfg.get("raid_burst") or 8):
                state["raid_until"] = now + float(
                    cfg.get("raid_cooldown_s") or 600)
        else:
            state["recent_joins"] = recent
        return now < float(state.get("raid_until") or 0)

    def acceptance_cfg(self, group_key: str, *,
                       now: float | None = None
                       ) -> tuple[dict[str, Any], bool]:
        """(effective acceptance config, raid active?) — the
        raid-tightened view every decision reads."""
        from .group_acceptance import effective_cfg

        now = now if now is not None else time.time()
        entry = self.store.group(group_key)
        raid = self._acceptance_raid(group_key, entry, now=now)
        cfg = effective_cfg(
            self.store.settings(group_key).get("acceptance") or {},
            raid=raid)
        return cfg, raid

    def acceptance_previously_banned(self, group_key: str,
                                     user_id: str
                                     ) -> dict[str, Any] | None:
        """Was this account banned HERE and never unbanned?
        Sources, newest evidence wins: the shared BanList, then
        the group's own audit trail (manual /ban writes
        admin_action, the defense ladder writes
        moderation_executed; an unban of either kind clears)."""
        try:
            ban = self.bans.is_banned(user_id, group_key)
            if ban is not None:
                return {"source": "ban_list",
                        "reason": str(getattr(ban, "reason", "")
                                      or "")}
            for record in self.bans.bans_for(user_id):
                if getattr(record, "active", False) and \
                        getattr(record, "scope", "") == "global":
                    return {"source": "ban_list_global",
                            "reason": str(
                                getattr(record, "reason", "") or "")}
        except Exception as exc:  # noqa: BLE001 - evidence path
            record_failure(
                "group_surface.acceptance_previously_banned", exc)
        entry = self.store.group(group_key)
        for item in reversed(entry.get("audit") or []):
            if str(item.get("user_id") or "") != str(user_id):
                continue
            if item.get("kind") not in ("admin_action",
                                        "moderation_executed"):
                continue
            action = str(item.get("action") or "")
            if action == "ban":
                return {"source": str(item.get("kind")),
                        "reason": str(item.get("reason") or "")}
            if action == "unban":
                return None
        return None

    def _acceptance_facts(self, group_key: str,
                          facts: dict[str, Any]) -> dict[str, Any]:
        """Enrich caller-supplied identity facts with the facts
        only the store knows: ban history, the bot whitelist,
        the group's word-filter list (name hygiene rides it)."""
        out = dict(facts or {})
        settings = self.store.settings(group_key)
        cfg = settings.get("acceptance") or {}
        banned = self.acceptance_previously_banned(
            group_key, str(out.get("user_id") or ""))
        out["previously_banned"] = banned is not None
        out["ban_reason"] = (banned or {}).get("reason", "")
        whitelist = {str(w).strip().lower()
                     for w in (cfg.get("bot_whitelist") or [])}
        out["whitelisted"] = (
            str(out.get("user_id") or "").lower() in whitelist
            or ("@" + str(out.get("username") or "").lower())
            in whitelist
            or str(out.get("username") or "").lower() in whitelist)
        out["filter_words"] = list(
            (settings.get("moderation") or {}).get(
                "word_filters") or [])
        return out

    def _set_restricted(self, entry: dict[str, Any], user_id: str,
                        *, missing: list[str], reasons: list[str],
                        via: str, action: str,
                        grace_until: float | None,
                        now: float) -> None:
        member = entry["members"].get(user_id)
        if member is None:
            return
        member["acceptance"] = {
            "state": "restricted", "missing": list(missing),
            "reasons": list(reasons), "via": via, "action": action,
            "since": now, "grace_until": grace_until,
            "welcome_pending": True}

    def _clear_acceptance(self, entry: dict[str, Any],
                          user_id: str) -> dict[str, Any]:
        member = entry["members"].get(user_id) or {}
        acc = member.pop("acceptance", None) or {}
        return acc

    def acceptance_on_join(self, group_key: str,
                           facts: dict[str, Any], *,
                           has_photo: bool | None = None,
                           now: float | None = None
                           ) -> dict[str, Any]:
        """A walk-in arrival (open/invite mode, or any platform
        join event): ledger first, evaluation second, enforcement
        described as actions for the caller to execute.  Returns
        {verdict, missing, reasons, actions, welcome, raid,
        grace_minutes, unverifiable}."""
        from .group_acceptance import evaluate

        now = now if now is not None else time.time()
        entry = self.store.group(group_key)
        raid = self._acceptance_raid(group_key, entry, now=now,
                                     count_join=True)
        cfg, _ = self.acceptance_cfg(group_key, now=now)
        user_id = str(facts.get("user_id") or "")
        self.note_join(
            group_key, user_id,
            username=str(facts.get("username") or ""),
            name=str(facts.get("name") or ""),
            last_name=str(facts.get("last_name") or ""),
            is_bot=bool(facts.get("is_bot")),
            has_photo=has_photo, ts=now)
        judged = self._acceptance_facts(group_key, facts)
        judged["has_photo"] = has_photo
        ev = evaluate(cfg, judged)
        grace = int(cfg.get("grace_minutes") or 60)
        self._audit(group_key, "acceptance_eval", user_id=user_id,
                    via="join", verdict=ev.verdict,
                    missing=ev.missing, unknown=ev.unknown,
                    decided_by="rule", raid=raid)
        out: dict[str, Any] = {
            "verdict": ev.verdict, "missing": ev.missing,
            "reasons": ev.reasons, "actions": [], "welcome": False,
            "raid": raid, "grace_minutes": grace,
            "unverifiable": list(ev.unknown)}
        if not ev.missing:
            if ev.unknown:
                # Open mode cannot hold a walk-in for a human:
                # admit, and journal the unverifiable criterion.
                self._audit(group_key, "acceptance_unverifiable",
                            user_id=user_id, unknown=ev.unknown,
                            via="join")
                out["verdict"] = "admit"
                out["welcome"] = True
                self.store.save()
                return out
            if raid and str(cfg.get("mode") or "open") == "open":
                # Raid posture: open behaves as request — a clean
                # join is HELD (restricted, releasable from the
                # queue) instead of waved through.
                from .group_acceptance import FIX_LINES

                req = self._queue_record(
                    entry, kind="hold", user_id=user_id,
                    facts=facts, missing=[],
                    reasons=[FIX_LINES["raid_hold"]], now=now)
                self._set_restricted(
                    entry, user_id, missing=[],
                    reasons=[FIX_LINES["raid_hold"]], via="raid",
                    action="", grace_until=None, now=now)
                self._audit(group_key, "acceptance_raid_hold",
                            user_id=user_id, request=req["id"])
                out.update({
                    "verdict": "restricted", "missing": [],
                    "reasons": [FIX_LINES["raid_hold"]],
                    "request_id": req["id"],
                    "actions": [{"kind": "acceptance_restrict",
                                 "user_id": user_id,
                                 "rule": "acceptance"}]})
                self.store.save()
                return out
            out["verdict"] = "admit"
            out["welcome"] = True
            self._clear_acceptance(entry, user_id)
            self.store.save()
            return out
        if ev.verdict == "restricted":
            self._set_restricted(
                entry, user_id, missing=ev.missing,
                reasons=ev.reasons, via="join", action="restricted",
                grace_until=None, now=now)
            out["actions"] = [{"kind": "acceptance_restrict",
                               "user_id": user_id,
                               "rule": "acceptance"}]
        elif ev.verdict == "kick":
            # A kick-action failure is admitted restricted for
            # the grace window first (§28) — a real person
            # mid-setup isn't bounced instantly; the cycle
            # removes them when the grace runs out unfixed.
            self._set_restricted(
                entry, user_id, missing=ev.missing,
                reasons=ev.reasons, via="join", action="kick",
                grace_until=now + grace * 60, now=now)
            out["verdict"] = "restricted"
            out["pending_kick"] = True
            out["actions"] = [{"kind": "acceptance_restrict",
                               "user_id": user_id,
                               "rule": "acceptance"}]
        elif ev.verdict == "decline":
            # A returning banned member is banned again (the ban
            # is the point); anyone else declined at the door is
            # removed but may rejoin once fixed.
            kind = "ban" if "banned" in ev.missing else "kick"
            self._clear_acceptance(entry, user_id)
            out["actions"] = [{"kind": kind, "user_id": user_id,
                               "rule": "acceptance"}]
        self.store.save()
        return out

    def _queue_record(self, entry: dict[str, Any], *, kind: str,
                      user_id: str, facts: dict[str, Any],
                      missing: list[str], reasons: list[str],
                      now: float,
                      unknown: list[str] | None = None
                      ) -> dict[str, Any]:
        state = self._acceptance_state(entry)
        state["req_seq"] = int(state.get("req_seq") or 0) + 1
        req_id = f"jr-{state['req_seq']:04d}"
        record = {
            "id": req_id, "kind": kind, "user_id": user_id,
            "username": str(facts.get("username") or ""),
            "name": str(facts.get("name") or ""),
            "last_name": str(facts.get("last_name") or ""),
            "is_bot": bool(facts.get("is_bot")),
            "has_photo": facts.get("has_photo"),
            "missing": list(missing), "reasons": list(reasons),
            "unknown": list(unknown or []),
            "state": "pending", "ts": now, "decided_by": ""}
        entry.setdefault("join_requests", {})[req_id] = record
        return record

    def acceptance_on_request(self, group_key: str,
                              facts: dict[str, Any], *,
                              has_photo: bool | None = None,
                              now: float | None = None
                              ) -> dict[str, Any]:
        """A Telegram join REQUEST (request mode): the candidate
        is not in the group yet.  Returns {verdict, request_id,
        missing, reasons, raid} where verdict is approve |
        approve_restricted | queued | decline — execution
        (approveChatJoinRequest / declineChatJoinRequest) is the
        caller's, through the adapter."""
        from .group_acceptance import evaluate

        now = now if now is not None else time.time()
        entry = self.store.group(group_key)
        raid = self._acceptance_raid(group_key, entry, now=now,
                                     count_join=True)
        cfg, _ = self.acceptance_cfg(group_key, now=now)
        user_id = str(facts.get("user_id") or "")
        judged = self._acceptance_facts(group_key, facts)
        judged["has_photo"] = has_photo
        ev = evaluate(cfg, judged)
        self._audit(group_key, "acceptance_eval", user_id=user_id,
                    via="request", verdict=ev.verdict,
                    missing=ev.missing, unknown=ev.unknown,
                    decided_by="rule", raid=raid)
        record = self._queue_record(
            entry, kind="request", user_id=user_id,
            facts={**facts, "has_photo": has_photo},
            missing=ev.missing, reasons=ev.reasons,
            unknown=ev.unknown, now=now)
        out: dict[str, Any] = {
            "missing": ev.missing, "reasons": ev.reasons,
            "request_id": record["id"], "record": record,
            "raid": raid}
        if ev.verdict == "decline" or (
                ev.verdict == "kick"):
            # Kick at the request door IS a decline (they are
            # not in to be kicked); they may re-request fixed.
            record["state"] = "declined"
            record["decided_by"] = "rule"
            out["verdict"] = "decline"
        elif ev.verdict == "restricted":
            if cfg.get("auto_approve"):
                record["state"] = "approved"
                record["decided_by"] = "rule"
                out["verdict"] = "approve_restricted"
            else:
                out["verdict"] = "queued"
        elif ev.clean and not ev.unknown:
            if cfg.get("auto_approve"):
                record["state"] = "approved"
                record["decided_by"] = "rule"
                out["verdict"] = "approve"
            else:
                out["verdict"] = "queued"
        else:  # unverifiable photo → a human decides
            out["verdict"] = "queued"
        self.store.save()
        return out

    def acceptance_request_admitted(self, group_key: str,
                                    req_id: str, *,
                                    now: float | None = None
                                    ) -> dict[str, Any]:
        """Bookkeeping after the caller approved a request:
        ledger the join, and if the approval was restricted,
        engage the restricted state.  Returns {welcome,
        restricted, missing, reasons, grace_minutes}."""
        now = now if now is not None else time.time()
        entry = self.store.group(group_key)
        record = (entry.get("join_requests") or {}).get(req_id)
        if record is None:
            return {"welcome": False, "restricted": False}
        user_id = str(record.get("user_id") or "")
        self.note_join(
            group_key, user_id,
            username=str(record.get("username") or ""),
            name=str(record.get("name") or ""),
            last_name=str(record.get("last_name") or ""),
            is_bot=bool(record.get("is_bot")),
            has_photo=record.get("has_photo"), ts=now)
        cfg, _ = self.acceptance_cfg(group_key, now=now)
        restricted = bool(record.get("missing"))
        if restricted:
            self._set_restricted(
                entry, user_id, missing=record["missing"],
                reasons=record.get("reasons") or [], via="request",
                action="restricted", grace_until=None, now=now)
        self.store.save()
        return {"welcome": not restricted, "restricted": restricted,
                "missing": record.get("missing") or [],
                "reasons": record.get("reasons") or [],
                "grace_minutes": int(cfg.get("grace_minutes") or 60)}

    def acceptance_queue(self, group_key: str) -> list[dict[str, Any]]:
        """Pending queue records, oldest first (FIFO fairness)."""
        entry = self.store.group(group_key)
        records = [r for r in
                   (entry.get("join_requests") or {}).values()
                   if r.get("state") == "pending"]
        return sorted(records, key=lambda r: float(r.get("ts") or 0))

    def acceptance_decide(self, group_key: str, req_id: str, *,
                          approve: bool, by: str,
                          now: float | None = None
                          ) -> dict[str, Any]:
        """An admin's queue decision.  Returns {ok, execute,
        record} — execute tells the caller what to perform:
        join_request (adapter approve/decline), release
        (unrestrict a held member), kick (remove a held member),
        or nothing (already decided / unknown id)."""
        now = now if now is not None else time.time()
        entry = self.store.group(group_key)
        record = (entry.get("join_requests") or {}).get(req_id)
        if record is None or record.get("state") != "pending":
            return {"ok": False, "execute": "none",
                    "record": record}
        record["state"] = "approved" if approve else "declined"
        record["decided_by"] = by
        record["decided_ts"] = now
        user_id = str(record.get("user_id") or "")
        self._audit(group_key, "acceptance_decision",
                    user_id=user_id, request=req_id,
                    approve=approve, by=by,
                    record_kind=record.get("kind"))
        if record.get("kind") == "request":
            execute = "join_request"
        elif approve:
            execute = "release"
            acc = self._clear_acceptance(entry, user_id)
            record["welcome"] = bool(acc.get("welcome_pending"))
        else:
            execute = "kick"
            self._clear_acceptance(entry, user_id)
        self.store.save()
        return {"ok": True, "execute": execute, "record": record}

    def acceptance_recheck(self, group_key: str, user_id: str, *,
                           username: str = "", name: str = "",
                           has_photo: bool | None = None,
                           now: float | None = None
                           ) -> dict[str, Any]:
        """The “check me again” button: re-run the criteria live
        for a restricted/warned member.  Facts are the freshest
        each surface can supply — the tap-time username/name,
        the ledger's stored name parts, and a fresh photo lookup
        when the caller made one.  Returns {state: passed |
        still | not_restricted, missing, reasons, welcome}."""
        from .group_acceptance import evaluate

        now = now if now is not None else time.time()
        entry = self.store.group(group_key)
        member = entry["members"].get(user_id)
        if member is None or not member.get("acceptance"):
            return {"state": "not_restricted", "missing": [],
                    "reasons": [], "welcome": False}
        acc = member["acceptance"]
        cfg, raid = self.acceptance_cfg(group_key, now=now)
        if acc.get("via") == "raid" and raid:
            # A raid hold is an admin's call while the rush is
            # on — the button doesn't self-release mid-raid.
            from .group_acceptance import FIX_LINES

            return {"state": "still", "missing": [],
                    "reasons": [FIX_LINES["raid_hold"]],
                    "welcome": False, "was_restricted": True}
        facts = {
            "user_id": user_id,
            "username": username or member.get("username") or "",
            "first_name": str(member.get("name") or "").split(
                None, 1)[0] if member.get("name") else "",
            "last_name": member.get("last_name") or "",
            "name": name or member.get("name") or "",
            "is_bot": bool(member.get("is_bot")),
            "has_photo": (has_photo if has_photo is not None
                          else member.get("has_photo")),
        }
        judged = self._acceptance_facts(group_key, facts)
        judged["has_photo"] = facts["has_photo"]
        ev = evaluate(cfg, judged)
        if not ev.missing:
            # Unknown-only also passes: unverifiable is not a
            # failure (journaled, like open-mode joins).
            if ev.unknown:
                self._audit(group_key, "acceptance_unverifiable",
                            user_id=user_id, unknown=ev.unknown,
                            via="recheck")
            self._clear_acceptance(entry, user_id)
            self._audit(group_key, "acceptance_recheck",
                        user_id=user_id, passed=True, raid=raid)
            self.store.save()
            return {"state": "passed", "missing": [],
                    "reasons": [],
                    "welcome": bool(acc.get("welcome_pending")),
                    "was_restricted":
                        acc.get("state") == "restricted"}
        acc["missing"] = ev.missing
        acc["reasons"] = ev.reasons
        self._audit(group_key, "acceptance_recheck",
                    user_id=user_id, passed=False,
                    missing=ev.missing)
        self.store.save()
        return {"state": "still", "missing": ev.missing,
                "reasons": ev.reasons, "welcome": False,
                "was_restricted":
                    acc.get("state") == "restricted"}

    def acceptance_release(self, group_key: str, user_id: str, *,
                           by: str = "rule",
                           now: float | None = None
                           ) -> dict[str, Any]:
        """Clear a member's acceptance state (an admin's manual
        release, or a passed re-check the caller already
        journaled).  Returns {released, welcome}."""
        entry = self.store.group(group_key)
        member = entry["members"].get(user_id)
        if member is None or not member.get("acceptance"):
            return {"released": False, "welcome": False}
        acc = self._clear_acceptance(entry, user_id)
        self._audit(group_key, "acceptance_release",
                    user_id=user_id, by=by)
        self.store.save()
        return {"released": True,
                "welcome": bool(acc.get("welcome_pending"))}

    def acceptance_cycle(self, group_key: str, *,
                         now: float | None = None,
                         enforce: bool = True
                         ) -> list[dict[str, Any]]:
        """Grace expiries, riding the message path like the
        estate/quest cycles.  A kick-action restriction whose
        grace ran out unfixed removes the member; a profile
        warning whose grace ran out engages the restriction.
        With enforce=False (defense autonomy off) expiries are
        journaled once and held, never executed."""
        now = now if now is not None else time.time()
        entry = self.store.group(group_key)
        actions: list[dict[str, Any]] = []
        for user_id, member in list(entry["members"].items()):
            acc = member.get("acceptance") or {}
            if not acc:
                continue
            grace_until = acc.get("grace_until")
            if grace_until is None or now < float(grace_until):
                continue
            if acc.get("state") == "warned":
                if not enforce:
                    if not acc.get("escalation_flagged"):
                        acc["escalation_flagged"] = True
                        self._audit(
                            group_key, "acceptance_grace_expired",
                            user_id=user_id, state="warned",
                            suppressed="autonomy_off")
                    continue
                acc.update({
                    "state": "restricted", "action": "restricted",
                    "grace_until": None, "via": "profile"})
                self._audit(group_key, "acceptance_escalated",
                            user_id=user_id, from_state="warned")
                actions.append({"kind": "acceptance_restrict",
                                "user_id": user_id,
                                "rule": "acceptance"})
            elif acc.get("state") == "restricted" \
                    and acc.get("action") == "kick":
                if not enforce:
                    if not acc.get("escalation_flagged"):
                        acc["escalation_flagged"] = True
                        self._audit(
                            group_key, "acceptance_grace_expired",
                            user_id=user_id, state="restricted",
                            suppressed="autonomy_off")
                    continue
                self._clear_acceptance(entry, user_id)
                self._audit(group_key, "acceptance_grace_kick",
                            user_id=user_id,
                            missing=acc.get("missing"))
                actions.append({"kind": "kick",
                                "user_id": user_id,
                                "rule": "acceptance"})
        if actions or any(
                (m.get("acceptance") or {}).get(
                    "escalation_flagged")
                for m in entry["members"].values()):
            self.store.save()
        return actions

    def acceptance_profile_check(self, group_key: str,
                                 user_id: str, *,
                                 username: str = "",
                                 privileged: bool = False,
                                 now: float | None = None
                                 ) -> dict[str, Any] | None:
        """The standing-condition re-check (§28): a member who
        DROPS a required attribute (their stored username is
        gone from their live identity) is warned first, with a
        grace window to restore it; the restriction engages
        only when the grace runs out (acceptance_cycle).  A
        member who restores it inside the window clears
        silently.  Admins/operators are exempt — the door rules
        govern admission, not the staff.  Returns {warn} /
        {recovered} / None."""
        now = now if now is not None else time.time()
        cfg, _raid = self.acceptance_cfg(group_key, now=now)
        if not (cfg.get("criteria") or {}).get("username"):
            return None
        if privileged:
            return None
        entry = self.store.group(group_key)
        member = entry["members"].get(user_id)
        if member is None:
            return None
        acc = member.get("acceptance") or {}
        if username:
            if acc.get("state") == "warned":
                self._clear_acceptance(entry, user_id)
                self._audit(group_key, "acceptance_profile",
                            user_id=user_id, recovered=True)
                self.store.save()
                return {"recovered": True}
            return None
        if acc:
            return None       # already in the flow
        if not member.get("username"):
            return None       # never had one on record — the
                              # drop can't be told from absence
        from .group_acceptance import FIX_LINES

        grace = int(cfg.get("grace_minutes") or 60)
        member["acceptance"] = {
            "state": "warned", "missing": ["username"],
            "reasons": [FIX_LINES["username"]], "via": "profile",
            "action": "restricted", "since": now,
            "grace_until": now + grace * 60,
            "welcome_pending": False}
        self._audit(group_key, "acceptance_profile",
                    user_id=user_id, dropped="username",
                    grace_minutes=grace)
        self.store.save()
        return {"warn": True, "reasons": [FIX_LINES["username"]],
                "grace_minutes": grace}

    # ---- the message pipeline ------------------------------------
    def _cache_message(self, group_key: str, *, message_id: str,
                       user_id: str, text: str, sticker_key: str,
                       now: float) -> None:
        """Bounded per-group message cache — the evidentiary
        basis for antidelete/antiedit (in-memory by design: it
        covers the live session, it is not an archive)."""
        if not message_id:
            return
        cache = self._msg_cache.setdefault(
            group_key, deque(maxlen=MSG_CACHE_RING))
        cache.append({"message_id": str(message_id),
                      "user_id": user_id, "text": text or "",
                      "ts": now, "sticker_key": sticker_key or ""})

    def _cached(self, group_key: str, message_id: str
                ) -> dict[str, Any] | None:
        for item in reversed(self._msg_cache.get(group_key, ())):
            if item["message_id"] == str(message_id):
                return item
        return None

    def handle_message(self, group_key: str, user_id: str, *,
                       username: str = "", name: str = "",
                       text: str = "", message_id: str = "",
                       ts: float | None = None,
                       envelope: Any = None,
                       sender_is_admin: bool = False,
                       sender_is_operator: bool = False
                       ) -> dict[str, Any]:
        """One ordinary group message: ledger observation, then
        the DEFENSE pipeline in its fixed order (brief §1.4) —
        identity → anticrash inspection → Moderator → envelope
        sensors → escalation spine → fines / economy / notices.
        Returns ``{"changes", "actions", "notices"}``; actions
        are dicts for the caller to execute (delete / warn / mute
        / kick / ban).  A message any stage deletes earns NO
        economy credit and fires no quest/season notices."""
        now = ts if ts is not None else time.time()
        changes = self._observe_identity(
            group_key, user_id, username=username, name=name, ts=now)
        entry = self.store.group(group_key)
        member = entry["members"][user_id]
        member["message_count"] = int(
            member.get("message_count") or 0) + 1
        if message_id:
            member["last_message_id"] = str(message_id)
        settings = self.store.settings(group_key)
        cfg = DefenseConfig.from_settings(settings)
        esc = EscalationConfig.from_settings(settings)
        warn_threshold = int(settings.get("warn_threshold") or 3)
        suppressed = not cfg.autonomy
        privileged = sender_is_admin or sender_is_operator
        mention = name or (f"@{username}" if username else user_id)
        # The §31 notice identity: mention + the numeric id the
        # resolver guarantees — the same label on every surface.
        who = f"{mention} [{user_id}]"
        # Join acceptance is a STANDING condition (§28): a
        # member who dropped a required attribute is warned with
        # a grace window (the restriction itself engages in the
        # cycle below).  Enforcement sleeps with defense
        # autonomy — the same one switch.
        acc_event = None
        if not suppressed:
            acc_event = self.acceptance_profile_check(
                group_key, user_id, username=username,
                privileged=privileged, now=now)
        env = normalize_envelope(
            envelope if isinstance(envelope, dict) else None,
            platform=group_key.split(":", 1)[0], chat_key=group_key,
            sender_id=user_id, message_id=message_id, text=text)
        self._cache_message(
            group_key, message_id=message_id, user_id=user_id,
            text=text, sticker_key=(env.sticker or {}).get("key")
            or "", now=now)

        actions: list[dict[str, Any]] = []
        strike_counts: dict[str, int] = {}
        struck: set[str] = set()
        fines_due: list[int] = []

        def strike(stage: str, rule: str, action_kind: str) -> int:
            """One strike per stage per message on the shared
            spine; landing on one of the group's designed steps
            (§30) appends that step's action — a fine is money,
            collected in finish; mute/restrict/kick/ban are
            platform actions.  Under autonomy OFF the spine is
            observe-only too: the audits carry the verdict, the
            ledger does not move."""
            if privileged or suppressed or stage in struck:
                return strike_counts.get(rule, 0)
            struck.add(stage)
            count = add_strike(member, rule=rule,
                               action=action_kind, now=now,
                               window_days=esc.window_days)
            strike_counts[rule] = count
            self._audit(group_key, "strike", user_id=user_id,
                        rule=rule, count=count,
                        window_days=esc.window_days)
            step = esc.action_at(count)
            if step is None:
                return count
            if step["action"] == "fine":
                fines_due.append(int(step.get("amount") or 0))
                return count
            kind = {"mute": "mute", "kick": "kick", "ban": "ban",
                    "restrict": "acceptance_restrict"}[
                        step["action"]]
            if all(_ACTION_RANK.get(a.get("kind"), 0)
                   < _ACTION_RANK[kind] for a in actions):
                if kind == "mute":
                    actions.append({
                        "kind": "delete", "rule": "escalation",
                        "reason": f"strike {count} — {rule}",
                        "user_id": user_id, "group_id": group_key})
                actions.append({
                    "kind": kind, "rule": "escalation",
                    "reason": f"strike {count} — {rule}",
                    "user_id": user_id, "group_id": group_key,
                    "minutes": int(step.get("minutes")
                                   or cfg.mute_minutes),
                    "_strike_count": count, "_strike_rule": rule})
            return count

        def finish(blocked: bool) -> dict[str, Any]:
            # A designed fine step is collected BEFORE the
            # warning is composed, so the notice only ever claims
            # what was actually charged.  It is a punishment, not
            # activity earnings — the blocked flag (which gates
            # earning on surviving messages) does not apply; the
            # autonomy switch does.
            paid_fine = 0
            if fines_due and not suppressed:
                paid_fine = self.apply_fine(
                    group_key, user_id, amount=sum(fines_due),
                    now=now)
            # ONE public warning per offense, on the strongest
            # action, worded by the organ (never meta-narration)
            # — and recorded as a §31 notice, so an admin can
            # undo exactly this action from the message itself.
            if actions and not suppressed:
                top = max(actions, key=lambda a: _ACTION_RANK.get(
                    a.get("kind"), 0))
                rule = top.get("_strike_rule") or top.get(
                    "rule") or ""
                count = top.get("_strike_count") \
                    or strike_counts.get(rule, 0)
                if top.get("kind") in ("warn", "delete", "mute",
                                       "kick", "ban",
                                       "acceptance_restrict"):
                    executed = "restrict" if (
                        top["kind"] == "acceptance_restrict"
                    ) else top["kind"]
                    live = recent_strikes(
                        member, now=now,
                        window_days=esc.window_days)
                    oldest = min(
                        (float(s.get("ts") or 0) for s in live),
                        default=0.0)
                    top["warning"] = render_warning(
                        who, rule,
                        consequence_text(
                            executed=executed, count=count,
                            cfg=cfg,
                            warn_threshold=warn_threshold,
                            esc=esc, fine_amount=paid_fine,
                            oldest_ts=oldest, now=now))
                    strike_ts = 0.0
                    for s in reversed(member.get("strikes")
                                      or []):
                        if isinstance(s, dict) \
                                and s.get("rule") == rule:
                            strike_ts = float(s.get("ts") or 0)
                            break
                    top["notice_id"] = self.record_notice(
                        group_key, user_id=user_id,
                        mention=mention, rule=rule,
                        action=top["kind"], count=count,
                        limit=esc.warn_limit,
                        strike_ts=strike_ts, strike_rule=rule,
                        minutes=int(top.get("minutes") or 0)
                        if top["kind"] == "mute" else 0,
                        fine=paid_fine, text=top["warning"],
                        now=now)
            for action in actions:
                action.pop("_strike_count", None)
                action.pop("_strike_rule", None)
            out_actions = actions
            n_suppressed = 0
            if suppressed:
                n_suppressed = len(actions)
                out_actions = []
                blocked = False
            elif not blocked:
                # Moderation fines (DESIGN §5 sink): an auto-mute
                # costs sparks — paid to the treasury.  (A ladder
                # fine charged above already used this member's
                # fine slot; the spacing guard keeps it to one.)
                for action in actions:
                    if action.get("kind") == "mute":
                        self.apply_fine(group_key, user_id, now=now)
            notices: list[str] = []
            if not blocked:
                self._economy_activity(group_key, user_id, now)
                season_note = self.season_cycle(group_key, now=now)
                if season_note:
                    notices.append(season_note)
                for paid in self.group_quest_cycle(group_key,
                                                   now=now):
                    notices.append(notice_line(
                        "🎉",
                        f"group quest complete: "
                        f"<b>{paid['name']}</b> — "
                        f"+{sparks(paid['bonus'])} to the "
                        f"treasury!"))
                notices.extend(self.estate_cycle(group_key,
                                                 now=now))
                # Acceptance grace expiries ride the same cycle:
                # kick-action graces remove, profile warnings
                # engage their restriction.  (When autonomy is
                # off, out_actions is already emptied — the cycle
                # runs in audit-only mode and holds its kicks.)
                cycle_actions = self.acceptance_cycle(
                    group_key, now=now, enforce=not suppressed)
                actions.extend(cycle_actions)
                if not suppressed:
                    # Each acceptance execution is a §31 notice
                    # too: the member sees who/what/why, and an
                    # admin can undo the restriction from it.
                    for action in cycle_actions:
                        if action.get("kind") not in (
                                "acceptance_restrict", "kick"):
                            continue
                        target = str(action.get("user_id")
                                     or "")
                        trec = entry["members"].get(target) \
                            or {}
                        tmention = trec.get("name") or (
                            f"@{trec['username']}"
                            if trec.get("username") else target)
                        label = (
                            "Restricted — restore the required "
                            "profile details to lift it."
                            if action["kind"]
                            == "acceptance_restrict"
                            else "Removed from the group.")
                        text = render_warning(
                            f"{tmention} [{target}]",
                            "acceptance", label)
                        action["notice_id"] = \
                            self.record_notice(
                                group_key, user_id=target,
                                mention=tmention,
                                rule="acceptance",
                                action=action["kind"], text=text,
                                now=now)
                if acc_event and acc_event.get("warn"):
                    notices.append(notice_line(
                        "🚪",
                        f"{who} — this group requires a "
                        f"username. Restore yours within "
                        f"{acc_event.get('grace_minutes', 60)} "
                        f"minutes or you'll be muted until you "
                        f"do."))
            self.store.save()
            return {"changes": changes, "actions": out_actions,
                    "notices": notices, "suppressed": n_suppressed}

        # ── stage 1: anticrash inspection ──────────────────────
        if cfg.anticrash_enabled:
            cache = self._msg_cache.get(group_key, ())
            recent = [c for c in cache if c["user_id"] == user_id
                      and now - c["ts"] <= 60]
            context = {
                "roster_size": len(entry["members"]),
                "sticker_velocity": sum(
                    1 for c in recent if c["sticker_key"]),
                "sticker_repeats": sum(
                    1 for c in recent
                    if c["sticker_key"]
                    and c["sticker_key"]
                    == (env.sticker or {}).get("key")),
            }
            verdict = inspect_payload(
                env, context=context, signatures=self.signatures)
            if verdict.action == "quarantine":
                self._audit(
                    group_key, "anticrash", user_id=user_id,
                    action=("delete" if cfg.anticrash_action
                            == "delete" else "flag"),
                    rule="anticrash", reason=verdict.reason,
                    score=verdict.score, classes=verdict.classes,
                    signature=verdict.signature,
                    **({"suppressed": "autonomy_off"}
                       if suppressed else {}))
                self.signatures.observe(verdict.signature, user_id,
                                        now=now)
                if cfg.anticrash_action == "delete":
                    actions.append({
                        "kind": "delete", "rule": "anticrash",
                        "reason": verdict.reason,
                        "user_id": user_id, "group_id": group_key})
                    strike("anticrash", "anticrash", "delete")
                    return finish(blocked=True)
            elif verdict.action == "flag":
                self._audit(
                    group_key, "anticrash_flag", user_id=user_id,
                    rule="anticrash", reason=verdict.reason,
                    score=verdict.score, classes=verdict.classes,
                    signature=verdict.signature,
                    **({"suppressed": "autonomy_off"}
                       if suppressed else {}))
                self.signatures.observe(verdict.signature, user_id,
                                        now=now)

        # ── stage 2: the Moderator (unchanged organ) ───────────
        if text:
            for action in self._moderator(group_key).check_message(
                    group_key, user_id, text, ts=now):
                actions.append(action.to_dict())
                self._audit(group_key, "moderation",
                            action=action.kind, rule=action.rule,
                            reason=action.reason, user_id=user_id,
                            **({"suppressed": "autonomy_off"}
                               if suppressed else {}))
                if action.kind in ("warn", "delete", "mute", "ban"):
                    strike("moderator", action.rule, action.kind)

        # ── stage 3: envelope sensors ──────────────────────────
        hit = check_antichannel(env, cfg)
        if hit is not None:
            self._audit(group_key, hit.rule, user_id=user_id,
                        action=hit.action, reason=hit.reason,
                        signal=hit.signal,
                        **({"suppressed": "autonomy_off"}
                           if suppressed else {}))
            if "delete" in hit.action:
                actions.append({
                    "kind": "delete", "rule": hit.rule,
                    "reason": hit.reason, "user_id": user_id,
                    "group_id": group_key})
            else:
                actions.append({
                    "kind": "warn", "rule": hit.rule,
                    "reason": hit.reason, "user_id": user_id,
                    "group_id": group_key})
            strike("antichannel", hit.rule,
                   "delete" if "delete" in hit.action else "warn")
        hit = check_antitag(env, cfg,
                            roster_size=len(entry["members"]),
                            sender_is_admin=privileged)
        if hit is not None:
            self._audit(group_key, hit.rule, user_id=user_id,
                        action=hit.action, reason=hit.reason,
                        signal=hit.signal,
                        **({"suppressed": "autonomy_off"}
                           if suppressed else {}))
            if "delete" in hit.action:
                actions.append({
                    "kind": "delete", "rule": hit.rule,
                    "reason": hit.reason, "user_id": user_id,
                    "group_id": group_key})
            else:
                actions.append({
                    "kind": "warn", "rule": hit.rule,
                    "reason": hit.reason, "user_id": user_id,
                    "group_id": group_key})
            strike("antitag", hit.rule,
                   "delete" if "delete" in hit.action else "warn")

        blocked = any(a.get("kind") in ("delete", "kick", "ban")
                      for a in actions)
        return finish(blocked)

    # ---- evidentiary sensors: delete / edit ----------------------
    def _member_label(self, group_key: str, user_id: str) -> str:
        member = self.store.group(group_key)["members"].get(
            user_id) or {}
        return (member.get("name")
                or (f"@{member['username']}"
                    if member.get("username") else "")
                or user_id or "someone")

    def handle_delete(self, group_key: str, *,
                      message_id: str, by_user_id: str = "",
                      ts: float | None = None) -> dict[str, Any]:
        """A message was deleted in the group (fed by the
        surfaces that can SEE delete events — MTProto / Baileys /
        Discord; the Bot API never produces these).  Evidentiary
        only: the cached original is quoted into the held audit;
        a callout is composed when the group's mode asks for one.
        The message itself is gone — nothing here pretends
        otherwise."""
        now = ts if ts is not None else time.time()
        cfg = self.defense_config(group_key)
        cached = self._cached(group_key, message_id)
        author = str(cached["user_id"]) if cached else ""
        excerpt = (cached["text"][:200] if cached else "")
        if cached:
            cache = self._msg_cache.get(group_key)
            if cache is not None:
                self._msg_cache[group_key] = deque(
                    (c for c in cache
                     if c["message_id"] != str(message_id)),
                    maxlen=MSG_CACHE_RING)
        suppressed = not cfg.autonomy
        self._audit(group_key, "antidelete", user_id=author,
                    by=by_user_id, message_id=str(message_id),
                    cached=bool(cached), excerpt=excerpt,
                    mode=cfg.antidelete_mode,
                    enabled=cfg.antidelete_enabled,
                    **({"suppressed": "autonomy_off"}
                       if suppressed else {}))
        self.store.save()
        callouts: list[dict[str, Any]] = []
        if cfg.antidelete_enabled and not suppressed \
                and cfg.antidelete_mode == "callout":
            who = self._member_label(group_key, author) \
                if author else "Someone"
            shown = f" “{excerpt}”" if excerpt else \
                " (content not in the live cache)"
            callouts.append({
                "to": cfg.antidelete_callout_to,
                "text": f"🗑 {who} deleted a message:{shown}"})
        return {"callouts": callouts, "cached": bool(cached)}

    def handle_edit(self, group_key: str, *, message_id: str,
                    new_text: str, ts: float | None = None
                    ) -> dict[str, Any]:
        """A message was edited: both versions land in the held
        audit (the evidence is the point — a scam that edits
        itself innocent leaves the original on record)."""
        now = ts if ts is not None else time.time()
        cfg = self.defense_config(group_key)
        cached = self._cached(group_key, message_id)
        author = str(cached["user_id"]) if cached else ""
        old = cached["text"] if cached else ""
        if cached:
            cached["text"] = new_text or ""
        suppressed = not cfg.autonomy
        self._audit(group_key, "antiedit", user_id=author,
                    message_id=str(message_id),
                    cached=bool(cached), old=old[:200],
                    new=(new_text or "")[:200],
                    mode=cfg.antiedit_mode,
                    enabled=cfg.antiedit_enabled,
                    **({"suppressed": "autonomy_off"}
                       if suppressed else {}))
        self.store.save()
        callouts: list[dict[str, Any]] = []
        if cfg.antiedit_enabled and not suppressed \
                and cfg.antiedit_mode == "callout":
            who = self._member_label(group_key, author) \
                if author else "Someone"
            callouts.append({
                "to": cfg.antiedit_callout_to,
                "text": f"✏️ {who} edited a message: "
                        f"“{old[:120]}” → “{(new_text or '')[:120]}”"})
        return {"callouts": callouts, "cached": bool(cached)}

    def strikes_for(self, group_key: str, user_id: str, *,
                    now: float | None = None) -> dict[str, Any]:
        """The member's live strike ledger (windowed) — the
        read model behind /strikes and the ladder."""
        now = now if now is not None else time.time()
        cfg = self.defense_config(group_key)
        esc = self.escalation_config(group_key)
        member = self.store.group(group_key)["members"].get(
            user_id) or {}
        strikes = recent_strikes(member, now=now,
                                 window_days=esc.window_days)
        nxt = esc.next_step(len(strikes))
        return {"count": len(strikes), "strikes": strikes,
                "window_days": esc.window_days,
                "limit": esc.warn_limit,
                "final_action": esc.final_action,
                "next_at": int(nxt["at"]) if nxt else 0,
                "next_label": ladder_label(nxt) if nxt else "",
                "oldest_ts": min(
                    (float(s.get("ts") or 0) for s in strikes),
                    default=0.0)}

    # ---- economy (P2: the country, wired live) --------------------
    @property
    def economy(self) -> Any:
        econ = getattr(self, "_economy", None)
        if econ is None:
            from ...community.group_economy import GroupEconomy

            econ = GroupEconomy(
                data_dir=self.store.path.parent / "group_economy")
            self._economy = econ
        return econ

    def _extras(self, group_key: str) -> dict[str, Any]:
        return self.store.group(group_key).setdefault("extras", {})

    def economy_frozen(self, group_key: str) -> str | None:
        """Freeze reason when the group's ledger is unreadable,
        else None.  Probed once per group per day; a failed probe
        freezes economy WRITES (the money is never guessed at)."""
        from ...community.group_economy import GroupEconomyError

        day = time.strftime("%Y-%m-%d", time.gmtime())
        probe = getattr(self, "_econ_probe", None)
        if probe is None:
            probe = self._econ_probe = {}
        cached = probe.get(group_key)
        if cached and cached[0] == day:
            return cached[1]
        reason: str | None = None
        try:
            self.economy.verify(group_key)
        except GroupEconomyError as exc:
            reason = str(exc)
            self._audit(group_key, "economy_frozen", reason=reason)
        probe[group_key] = (day, reason)
        return reason

    def economy_on(self, group_key: str) -> bool:
        return bool(self.store.settings(
            group_key)["economy_enabled"]) \
            and self.economy_frozen(group_key) is None

    # ---- real estate (§17 slice 2 / §20: the whole class) ----
    @property
    def estate(self) -> GroupRealEstate:
        est = getattr(self, "_estate", None)
        if est is None:
            est = GroupRealEstate(self.economy)
            self._estate = est
        return est

    def estate_cfg(self, group_key: str) -> dict[str, Any]:
        return dict(self.store.settings(group_key)["estate"])

    def estate_frozen(self, group_key: str) -> str | None:
        """Freeze reason when the estate store is unreadable,
        else None (the economy_frozen discipline, probed once
        per group per day)."""
        day = time.strftime("%Y-%m-%d", time.gmtime())
        probe = getattr(self, "_estate_probe", None)
        if probe is None:
            probe = self._estate_probe = {}
        cached = probe.get(group_key)
        if cached and cached[0] == day:
            return cached[1]
        reason: str | None = None
        try:
            self.estate.health(group_key)
        except GroupRealEstateError as exc:
            reason = str(exc)
            self._audit(group_key, "estate_frozen", reason=reason)
        probe[group_key] = (day, reason)
        return reason

    def estate_on(self, group_key: str) -> bool:
        return self.economy_on(group_key) \
            and self.estate_frozen(group_key) is None

    def _estate_gate(self, group_key: str) -> dict[str, Any] | None:
        """None when the estate runs here, else a refusal dict
        in the facade's shape."""
        if not self.store.settings(group_key)["economy_enabled"]:
            return {"ok": False, "reason": "off"}
        if self.estate_frozen(group_key) is not None:
            return {"ok": False, "reason": "frozen"}
        return None

    def estate_summary(self, group_key: str, user_id: str, *,
                       now: float | None = None) -> dict[str, Any]:
        if self._estate_gate(group_key):
            return {"gems": 0, "properties": [],
                    "property_count": 0, "property_value": 0,
                    "rent_ready_gems": 0, "manager_level": 0,
                    "premium_owned": [], "mortgages": []}
        return self.estate.summary(group_key, user_id, now=now)

    def estate_catalog(self, group_key: str, *,
                       now: float | None = None) -> list[dict[str, Any]]:
        if self._estate_gate(group_key):
            return []
        return self.estate.catalog(group_key, now=now)

    def estate_buy(self, group_key: str, user_id: str,
                   district: str, tier: str, *,
                   now: float | None = None) -> dict[str, Any]:
        gate = self._estate_gate(group_key)
        if gate:
            return gate
        econ = self.economy
        if not econ.is_verified(group_key, user_id):
            econ.verify_member(group_key, user_id)
        return self.estate.buy(group_key, user_id, district, tier,
                               cfg=self.estate_cfg(group_key),
                               now=now)

    def estate_buy_public(self, group_key: str, district: str,
                          tier: str, *, by: str = "",
                          now: float | None = None) -> dict[str, Any]:
        """Public property (§20g): the treasury buys a building.
        Reservation-aware like every treasury spend — escrowed
        bids and open drops are not spendable."""
        gate = self._estate_gate(group_key)
        if gate:
            return gate
        price = self.estate.price(group_key, district, tier) \
            if district and tier else 0
        available = self.economy.treasury_balance(group_key) \
            - self.reserved_treasury(group_key)
        if price > available:
            return {"ok": False, "reason": "funds",
                    "available": available, "price": price}
        out = self.estate.buy_public(group_key, district, tier,
                                     by=by, now=now)
        if out.get("ok"):
            self._audit(group_key, "estate_public_buy", by=by,
                        district=district, tier=tier,
                        price=out["price"])
        return out

    def estate_sell(self, group_key: str, user_id: str, pid: str,
                    *, now: float | None = None) -> dict[str, Any]:
        gate = self._estate_gate(group_key)
        if gate:
            return gate
        return self.estate.sell(group_key, user_id, pid,
                                cfg=self.estate_cfg(group_key),
                                now=now)

    def estate_collect_rent(self, group_key: str, user_id: str, *,
                            now: float | None = None
                            ) -> dict[str, Any]:
        gate = self._estate_gate(group_key)
        if gate:
            return gate
        if self.jail_state(group_key, user_id, now=now):
            return {"ok": False, "reason": "jailed"}
        econ = self.economy
        if not econ.is_verified(group_key, user_id):
            econ.verify_member(group_key, user_id)
        return self.estate.collect_rent(group_key, user_id,
                                        now=now)

    def estate_list(self, group_key: str, user_id: str, pid: str,
                    ask: int, *, now: float | None = None
                    ) -> dict[str, Any]:
        gate = self._estate_gate(group_key)
        if gate:
            return gate
        return self.estate.list_property(group_key, user_id, pid,
                                         ask, now=now)

    def estate_unlist(self, group_key: str, user_id: str,
                      pid: str, *, now: float | None = None
                      ) -> dict[str, Any]:
        gate = self._estate_gate(group_key)
        if gate:
            return gate
        return self.estate.unlist(group_key, user_id, pid, now=now)

    def estate_market(self, group_key: str) -> list[dict[str, Any]]:
        if self._estate_gate(group_key):
            return []
        return self.estate.market(group_key)

    def estate_buy_listing(self, group_key: str, buyer: str,
                           lid: str, *,
                           now: float | None = None
                           ) -> dict[str, Any]:
        gate = self._estate_gate(group_key)
        if gate:
            return gate
        return self.estate.buy_listing(
            group_key, buyer, lid, cfg=self.estate_cfg(group_key),
            now=now)

    def estate_auction_start(self, group_key: str, user_id: str,
                             pid: str, start: int, hours: float, *,
                             now: float | None = None
                             ) -> dict[str, Any]:
        gate = self._estate_gate(group_key)
        if gate:
            return gate
        return self.estate.start_auction(group_key, user_id, pid,
                                         start, hours, now=now)

    def estate_bid(self, group_key: str, user_id: str, aid: str,
                   amount: int, *, now: float | None = None
                   ) -> dict[str, Any]:
        gate = self._estate_gate(group_key)
        if gate:
            return gate
        return self.estate.place_bid(group_key, user_id, aid,
                                     amount, now=now)

    def estate_auctions(self, group_key: str, *,
                        now: float | None = None
                        ) -> list[dict[str, Any]]:
        if self._estate_gate(group_key):
            return []
        return self.estate.open_auctions(
            group_key, cfg=self.estate_cfg(group_key), now=now)

    def estate_borrow(self, group_key: str, user_id: str,
                      pid: str, amount: int, *,
                      now: float | None = None) -> dict[str, Any]:
        gate = self._estate_gate(group_key)
        if gate:
            return gate
        if self.jail_state(group_key, user_id, now=now):
            return {"ok": False, "reason": "jailed"}
        # The treasury's lendable sparks respect ALL reservations
        # (open drops + auction escrow), like grants do.
        available = self.economy.treasury_balance(group_key) \
            - self.reserved_treasury(group_key)
        if amount > available:
            return {"ok": False, "reason": "funds",
                    "available": available}
        return self.estate.borrow(group_key, user_id, pid, amount,
                                  cfg=self.estate_cfg(group_key),
                                  now=now)

    def estate_repay(self, group_key: str, user_id: str, mid: str,
                     amount: int, *, now: float | None = None
                     ) -> dict[str, Any]:
        gate = self._estate_gate(group_key)
        if gate:
            return gate
        return self.estate.repay(group_key, user_id, mid, amount,
                                 now=now)

    def estate_mortgages(self, group_key: str, user_id: str = ""
                         ) -> list[dict[str, Any]]:
        if self._estate_gate(group_key):
            return []
        return self.estate.mortgages(group_key, user_id)

    def estate_insure(self, group_key: str, user_id: str,
                      pid: str, *, now: float | None = None
                      ) -> dict[str, Any]:
        gate = self._estate_gate(group_key)
        if gate:
            return gate
        return self.estate.insure(group_key, user_id, pid,
                                  now=now)

    def estate_pay_repair(self, group_key: str, user_id: str,
                          pid: str, *, now: float | None = None
                          ) -> dict[str, Any]:
        gate = self._estate_gate(group_key)
        if gate:
            return gate
        return self.estate.pay_repair_debt(group_key, user_id,
                                           pid, now=now)

    def estate_manager_buy(self, group_key: str, user_id: str, *,
                           now: float | None = None
                           ) -> dict[str, Any]:
        gate = self._estate_gate(group_key)
        if gate:
            return gate
        return self.estate.buy_manager(group_key, user_id,
                                       now=now)

    def estate_premium_buy(self, group_key: str, user_id: str,
                           item_id: str, *,
                           now: float | None = None
                           ) -> dict[str, Any] | None:
        if self._estate_gate(group_key):
            return None
        return self.estate.buy_premium(group_key, user_id,
                                       item_id, now=now)

    def estate_grant_gems(self, group_key: str, recipient: str,
                          amount: int, *, proposal_id: str,
                          by: str = "",
                          now: float | None = None
                          ) -> dict[str, Any]:
        """Execute a gems gift — called by the operator path and
        by approve_spend for ``gems_gift`` proposals.  There is
        no other road: the organ's only governance faucet."""
        gate = self._estate_gate(group_key)
        if gate:
            return gate
        econ = self.economy
        if not econ.is_verified(group_key, recipient):
            econ.verify_member(group_key, recipient)
        entry = self.estate.grant_gems_governance(
            group_key, recipient, amount,
            proposal_id=proposal_id, now=now)
        if entry is None:
            return {"ok": False, "reason": "gems"}
        self._audit(group_key, "estate_gems_gift", by=by,
                    recipient=recipient, amount=amount,
                    proposal=proposal_id)
        return {"ok": True, "amount": amount,
                "gems": self.estate.gems_balance(group_key,
                                                 recipient)}

    def estate_season_rewards(self, group_key: str,
                              champs: list[tuple[str, int]], *,
                              now: float | None = None
                              ) -> list[tuple[str, int]]:
        """Season podium gems (§17 premium source), granted at
        rollover.  Returns what was actually granted."""
        from ...community.group_real_estate import (
            SEASON_GEM_REWARDS)

        if self._estate_gate(group_key):
            return []
        granted: list[tuple[str, int]] = []
        for rank, (member, _earned) in enumerate(champs[:3], 1):
            gems = SEASON_GEM_REWARDS.get(rank) or 0
            if gems and self.estate.credit_gems(
                    group_key, member, gems, source="season",
                    reason="season reward", ref="season",
                    now=now):
                granted.append((member, gems))
        return granted

    def estate_top_landlord(self, group_key: str
                            ) -> tuple[str, int] | None:
        if self._estate_gate(group_key):
            return None
        return self.estate.top_landlord(group_key)

    def estate_cycle(self, group_key: str, *,
                     rng: Any = None,
                     now: float | None = None) -> list[str]:
        """The estate tick, riding the message path beside the
        quest/season cycles: auctions settle, mortgages run,
        public rent lands, the day's district event fires.
        Returns group-worthy announcement lines (personal
        outcomes — one member's auto-collected rent, a mortgage
        paid off — stay off the group feed)."""
        import random

        now = now if now is not None else time.time()
        if not self.system_output_allowed(group_key):
            return []  # §33: the estate never ticks abroad
        if not self.estate_on(group_key):
            return []
        from ...community.group_real_estate import DISTRICTS, TIERS

        notices: list[str] = []
        try:
            events = self.estate.tick(
                group_key, rng=rng or random.Random(),
                cfg=self.estate_cfg(group_key), now=now)
        except GroupRealEstateError:
            return []
        for event in events:
            kind = event.get("kind")
            if kind == "auction_settled":
                if event.get("winner"):
                    who = self._member_label(
                        group_key, str(event["winner"]))
                    notices.append(notice_line(
                        "🔨",
                        f"auction settled: "
                        f"<b>{TIERS[event['tier']]['name']}</b> "
                        f"went to {who} for "
                        f"{sparks(event['price'])}"))
                else:
                    notices.append(notice_line(
                        "🔨", "an auction closed with no bids — "
                              "the property is back with its "
                              "owner."))
            elif kind == "repossession":
                self._audit(group_key, "estate_repossession",
                            mortgage=event.get("mortgage"),
                            property=event.get("property"),
                            written_off=event.get("written_off"))
                notices.append(notice_line(
                    "🏛️",
                    f"mortgage default: a property was "
                    f"repossessed into public stock "
                    f"({sparks(event.get('written_off') or 0)} "
                    f"of debt written off, on the record)."))
            elif kind == "public_rent":
                notices.append(notice_line(
                    "🏛️",
                    f"public property paid "
                    f"{sparks(event['sparks'])} of rent into "
                    f"the treasury."))
            elif kind == "district_event":
                name = DISTRICTS[event["district"]]["name"]
                if event["event"] == "boom":
                    notices.append(notice_line(
                        "📈", f"<b>{name}</b> is booming — "
                              f"property prices there are up "
                              f"(index {event.get('index')})."))
                elif event["event"] == "bust":
                    notices.append(notice_line(
                        "📉", f"<b>{name}</b> is in a slump — "
                              f"property prices there are down "
                              f"(index {event.get('index')})."))
                elif event["event"] == "tenants":
                    notices.append(notice_line(
                        "🛋️", f"tenant trouble in <b>{name}</b> "
                              f"— rents there pay 25% less for "
                              f"3 days."))
                elif event["event"] == "repairs":
                    notices.append(notice_line(
                        "🔧", f"repair wave in <b>{name}</b> — "
                              f"{sparks(event.get('billed') or 0)} "
                              f"of repair bills paid, "
                              f"{sparks(event.get('covered') or 0)} "
                              f"covered by insurance."))
        return notices

    def _economy_activity(self, group_key: str, user_id: str,
                          now: float) -> None:
        """Every real group message feeds the economy: the member
        is verified by presence, the message counts as activity
        (quests / hashrate material), and the first message of a
        day pays the streak spark.  Best-effort: the economy never
        breaks the message path."""
        try:
            if not self.system_output_allowed(group_key):
                # §33: no foreign accrual — a room Devon-the-bot
                # doesn't operate in builds NO economy state,
                # whichever of the owner's accounts sits there.
                return
            if not self.economy_on(group_key):
                return
            econ = self.economy
            if not econ.is_verified(group_key, user_id):
                econ.verify_member(group_key, user_id)
            econ.record_activity(group_key, user_id, "message",
                                 now=now)
            if econ.activity_count(group_key, user_id, "streak_day",
                                   "daily", now) == 0:
                econ.record_activity(group_key, user_id, "streak_day",
                                     now=now)
        except Exception:  # noqa: BLE001 - faucet is best-effort
            pass

    def game_award(self, group_key: str, user_id: str, coins: int,
                   game: str, *, now: float | None = None) -> int:
        """One purse (DESIGN §13/§5): coins won in a game played
        IN this group land in the group's purse for that member.
        Returns what was credited (0 when off / frozen / capped —
        the games wallet keeps its own copy either way, this is
        the group country's share of group play)."""
        try:
            if not self.system_output_allowed(group_key):
                return 0  # §33: no game accrual in foreign rooms
            if coins <= 0 or not self.economy_on(group_key):
                return 0
            econ = self.economy
            if not econ.is_verified(group_key, user_id):
                econ.verify_member(group_key, user_id)
            econ.record_activity(group_key, user_id, "game_play",
                                 now=now)
            entry = econ.award(group_key, user_id, int(coins),
                               f"game: {game}", ref=f"game:{game}",
                               now=now)
            return int(entry.amount) if entry else 0
        except Exception:  # noqa: BLE001 - bridge is best-effort
            return 0

    def work_shift(self, group_key: str, user_id: str, *,
                   now: float | None = None) -> dict[str, Any]:
        """/work: a cooldown-bound shift.  Pay follows engagement
        (the organ's hashrate) and the member's shift level."""
        now = now if now is not None else time.time()
        if not self.economy_on(group_key):
            return {"ok": False, "reason": "off"}
        state = self._extras(group_key).setdefault("work", {})
        mine = state.get(user_id) or {}
        last = float(mine.get("last") or 0)
        if now - last < 3600:
            return {"ok": False, "reason": "cooldown",
                    "wait_s": int(3600 - (now - last))}
        econ = self.economy
        if not econ.is_verified(group_key, user_id):
            econ.verify_member(group_key, user_id)
        shifts = int(mine.get("shifts") or 0)
        level = shifts // 10
        rate = int(econ.hashrate(group_key, user_id, now)["hashrate"])
        pay = min(60, 10 + min(40, rate) + 2 * level)
        entry = econ.award(group_key, user_id, pay, "work shift",
                           ref="work", now=now)
        if entry is None:
            return {"ok": False, "reason": "cap"}
        state[user_id] = {"last": now, "shifts": shifts + 1}
        self.store.save()
        return {"ok": True, "pay": pay, "level": level,
                "balance": econ.balance(group_key, user_id)}

    def pay(self, group_key: str, sender: str, recipient: str,
            amount: int, *, now: float | None = None
            ) -> dict[str, Any]:
        """/pay with the treasury tax: the fee leg is a treasury
        contribution, the net leg a transfer — both ledgered."""
        now = now if now is not None else time.time()
        if not self.economy_on(group_key):
            return {"ok": False, "reason": "off"}
        econ = self.economy
        amount = int(amount)
        if amount <= 0 or sender == recipient:
            return {"ok": False, "reason": "shape"}
        if econ.balance(group_key, sender) < amount:
            return {"ok": False, "reason": "funds"}
        pct = int(self.store.settings(
            group_key).get("transfer_tax_pct") or 0)
        fee = (amount * pct) // 100
        if fee:
            if not econ.contribute_treasury(group_key, sender, fee,
                                            now=now):
                return {"ok": False, "reason": "funds"}
        if not econ.transfer(group_key, sender, recipient,
                             amount - fee, now=now):
            return {"ok": False, "reason": "funds"}
        return {"ok": True, "fee": fee, "net": amount - fee,
                "balance": econ.balance(group_key, sender),
                "treasury": econ.treasury_balance(group_key)}

    def open_drop(self, group_key: str, funder: str, total: int,
                  per_claim: int, *, now: float | None = None
                  ) -> dict[str, Any]:
        """A giveaway drop, treasury-escrowed: the funder pays the
        pool into the treasury up front; claims pay out of it.
        Anti-drain: one claim per member, fixed claim size."""
        now = now if now is not None else time.time()
        if not self.economy_on(group_key):
            return {"ok": False, "reason": "off"}
        total, per_claim = int(total), int(per_claim)
        if total <= 0 or per_claim <= 0 or per_claim > total:
            return {"ok": False, "reason": "shape"}
        if not self.economy.contribute_treasury(
                group_key, funder, total, now=now):
            return {"ok": False, "reason": "funds"}
        drops = self._extras(group_key).setdefault("drops", [])
        drop = {"id": f"drop-{int(now)}-{len(drops)}",
                "total": total, "per_claim": per_claim,
                "remaining": total, "funder": funder,
                "claims": {}, "opened_at": now, "open": True}
        drops.append(drop)
        self._audit(group_key, "drop_opened", funder=funder,
                    total=total, per_claim=per_claim)
        self.store.save()
        return {"ok": True, "drop": drop}

    def claim_drop(self, group_key: str, user_id: str, *,
                   now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        if not self.economy_on(group_key):
            return {"ok": False, "reason": "off"}
        drops = self._extras(group_key).setdefault("drops", [])
        drop = next((d for d in reversed(drops) if d.get("open")),
                    None)
        if drop is None:
            return {"ok": False, "reason": "no_drop"}
        if drop["remaining"] < drop["per_claim"]:
            drop["open"] = False
            self.store.save()
            return {"ok": False, "reason": "empty"}
        if user_id in (drop.get("claims") or {}):
            return {"ok": False, "reason": "already"}
        econ = self.economy
        if not econ.is_verified(group_key, user_id):
            econ.verify_member(group_key, user_id)
        if not econ.treasury_grant(group_key, user_id,
                                   int(drop["per_claim"]),
                                   reason=f"drop {drop['id']}",
                                   now=now):
            return {"ok": False, "reason": "treasury"}
        drop["remaining"] -= int(drop["per_claim"])
        drop.setdefault("claims", {})[user_id] = now
        if drop["remaining"] < drop["per_claim"]:
            drop["open"] = False
        self.store.save()
        return {"ok": True, "amount": int(drop["per_claim"]),
                "remaining": drop["remaining"],
                "closed": not drop["open"],
                "drop_total": int(drop["total"]),
                "drop_id": drop["id"],
                "balance": econ.balance(group_key, user_id)}

    def economy_health(self, group_key: str, *,
                       now: float | None = None) -> dict[str, Any]:
        """The briefing's economy line: holders, concentration,
        treasury, and how many members were active this week."""
        now = now if now is not None else time.time()
        econ = self.economy
        balances = {m: b for m, b in econ.balances(group_key).items()
                    if m and not econ.is_pseudo_member(m)}
        total = sum(balances.values())
        top = max(balances.values()) if balances else 0
        activities = econ._load_json(group_key, "activities", []) or []
        active = {a.get("member") for a in activities
                  if float(a.get("ts") or 0) >= now - 7 * 86400}
        return {"holders": len(balances), "circulating": total,
                "treasury": econ.treasury_balance(group_key),
                "jackpot": econ.jackpot_balance(group_key),
                "policy": econ.issuance_state(group_key, now=now),
                "top_share": (round(top / total, 3)
                              if total else 0.0),
                "active_7d": len(active - {None, ""})}

    # ---- the full loop: bank / daily / crime / gambling / boards
    # (the Septorch-class surface on this ledger — every verb is a
    # real state change, journaled, with honest refusals) ---------

    def reserved_treasury(self, group_key: str) -> int:
        """Treasury sparks already promised: open drops plus
        auction escrow (standing high bids are HELD in the
        treasury — §20a — and are not the treasury's to spend
        until their auction settles)."""
        drops = self._extras(group_key).setdefault("drops", [])
        total = sum(int(d.get("remaining") or 0)
                    for d in drops if d.get("open"))
        try:
            total += self.estate.auction_reserved(group_key)
        except GroupRealEstateError:
            # The estate store is unreadable: how much escrow
            # the treasury holds is UNKNOWN, so it is treated
            # as unspendable rather than guessed at (estate
            # commands surface the freeze itself).
            total += 10 ** 12
        return total

    def open_drop_from_treasury(self, group_key: str, total: int,
                                per_claim: int, *,
                                by: str = "",
                                now: float | None = None
                                ) -> dict[str, Any]:
        """A drop funded by the treasury itself: no contribution
        leg — the pool IS a reservation against the treasury, and
        grants respect reservations (see grant())."""
        now = now if now is not None else time.time()
        if not self.economy_on(group_key):
            return {"ok": False, "reason": "off"}
        total, per_claim = int(total), int(per_claim)
        if total <= 0 or per_claim <= 0 or per_claim > total:
            return {"ok": False, "reason": "shape"}
        available = self.economy.treasury_balance(group_key) \
            - self.reserved_treasury(group_key)
        if total > available:
            return {"ok": False, "reason": "funds",
                    "available": available}
        drops = self._extras(group_key).setdefault("drops", [])
        drop = {"id": f"drop-{int(now)}-{len(drops)}",
                "total": total, "per_claim": per_claim,
                "remaining": total, "funder": "treasury",
                "claims": {}, "opened_at": now, "open": True}
        drops.append(drop)
        self._audit(group_key, "drop_opened", funder="treasury",
                    by=by, total=total, per_claim=per_claim)
        self.store.save()
        return {"ok": True, "drop": drop}

    def grant(self, group_key: str, recipient: str, amount: int, *,
              by: str = "", reason: str = "",
              now: float | None = None) -> dict[str, Any]:
        """/grant: an admin pays a member OUT of the treasury.
        Reservation-aware: sparks promised to open drops are not
        grantable."""
        now = now if now is not None else time.time()
        if not self.economy_on(group_key):
            return {"ok": False, "reason": "off"}
        amount = int(amount)
        if amount <= 0:
            return {"ok": False, "reason": "shape"}
        econ = self.economy
        available = econ.treasury_balance(group_key) \
            - self.reserved_treasury(group_key)
        if amount > available:
            return {"ok": False, "reason": "funds",
                    "available": available}
        if not econ.is_verified(group_key, recipient):
            econ.verify_member(group_key, recipient)
        if not econ.treasury_grant(group_key, recipient, amount,
                                   reason=reason or "admin grant",
                                   now=now):
            return {"ok": False, "reason": "funds",
                    "available": available}
        self._audit(group_key, "treasury_grant", by=by,
                    recipient=recipient, amount=amount)
        self.store.save()
        return {"ok": True, "amount": amount,
                "treasury": econ.treasury_balance(group_key),
                "balance": econ.balance(group_key, recipient)}

    def propose_spend(self, group_key: str, *, kind: str,
                      payload: dict[str, Any], by: str,
                      now: float | None = None) -> dict[str, Any]:
        """Treasury governance (DESIGN §5): a spend above the
        group's threshold does not execute on one admin's word —
        it becomes a proposal a SECOND admin must approve.  The
        proposal is state, listed in /glog, never chat vapor."""
        now = now if now is not None else time.time()
        proposals = self._extras(group_key).setdefault(
            "proposals", {})
        pid = f"prop-{int(now)}-{len(proposals)}"
        proposals[pid] = {"kind": kind, "payload": payload,
                          "by": by, "at": now, "open": True}
        self._audit(group_key, "proposal_opened", proposal=pid,
                    spend_kind=kind, by=by, **{
                        k: v for k, v in payload.items()
                        if isinstance(v, (int, str))})
        self.store.save()
        return {"id": pid, **proposals[pid]}

    def approve_spend(self, group_key: str, proposal_id: str, *,
                      by: str,
                      now: float | None = None) -> dict[str, Any]:
        """Execute an open proposal — only by a DIFFERENT admin
        than the proposer (the handler checks admin rights)."""
        now = now if now is not None else time.time()
        proposals = self._extras(group_key).setdefault(
            "proposals", {})
        proposal = proposals.get(proposal_id)
        if not proposal or not proposal.get("open"):
            return {"ok": False, "reason": "no_proposal"}
        if proposal.get("by") == by:
            return {"ok": False, "reason": "self_approve"}
        payload = proposal.get("payload") or {}
        if proposal["kind"] == "grant":
            out = self.grant(group_key, str(payload["recipient"]),
                             int(payload["amount"]), by=by,
                             reason=str(payload.get("reason") or ""),
                             now=now)
        elif proposal["kind"] == "drop":
            out = self.open_drop_from_treasury(
                group_key, int(payload["total"]),
                int(payload["per_claim"]), by=by, now=now)
        elif proposal["kind"] == "estate_public":
            out = self.estate_buy_public(
                group_key, str(payload["district"]),
                str(payload["tier"]), by=by, now=now)
        elif proposal["kind"] == "gems_gift":
            out = self.estate_grant_gems(
                group_key, str(payload["recipient"]),
                int(payload["amount"]), proposal_id=proposal_id,
                by=by, now=now)
        else:
            return {"ok": False, "reason": "kind"}
        if out.get("ok"):
            proposal["open"] = False
            proposal["approved_by"] = by
            self._audit(group_key, "proposal_executed",
                        proposal=proposal_id, by=by)
            self.store.save()
        return out

    def open_proposals(self, group_key: str) -> dict[str, Any]:
        proposals = self._extras(group_key).setdefault(
            "proposals", {})
        return {pid: p for pid, p in proposals.items()
                if p.get("open")}

    def deposit(self, group_key: str, user_id: str,
                amount: int | None, *,
                now: float | None = None) -> dict[str, Any]:
        """/deposit: wallet → the member's bank vault (amount
        None = everything).  Banked sparks can't be robbed."""
        now = now if now is not None else time.time()
        if not self.economy_on(group_key):
            return {"ok": False, "reason": "off"}
        econ = self.economy
        wallet = econ.balance(group_key, user_id)
        amount = wallet if amount is None else int(amount)
        if amount <= 0:
            return {"ok": False, "reason": "empty"}
        if not econ.bank_deposit(group_key, user_id, amount,
                                 now=now):
            return {"ok": False, "reason": "funds"}
        return {"ok": True, "amount": amount,
                "wallet": econ.balance(group_key, user_id),
                "bank": econ.bank_balance(group_key, user_id)}

    def withdraw(self, group_key: str, user_id: str,
                 amount: int | None, *,
                 now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        if not self.economy_on(group_key):
            return {"ok": False, "reason": "off"}
        econ = self.economy
        bank = econ.bank_balance(group_key, user_id)
        amount = bank if amount is None else int(amount)
        if amount <= 0:
            return {"ok": False, "reason": "empty"}
        if not econ.bank_withdraw(group_key, user_id, amount,
                                  now=now):
            return {"ok": False, "reason": "funds"}
        return {"ok": True, "amount": amount,
                "wallet": econ.balance(group_key, user_id),
                "bank": econ.bank_balance(group_key, user_id)}

    def daily_claim(self, group_key: str, user_id: str, *,
                    now: float | None = None) -> dict[str, Any]:
        """/daily: the front-door faucet.  Once a day; a streak
        of consecutive days grows it (data in the organ)."""
        from ...community.group_economy import (
            DAILY_BASE, DAILY_STREAK_CAP, DAILY_STREAK_STEP)

        now = now if now is not None else time.time()
        if not self.economy_on(group_key):
            return {"ok": False, "reason": "off"}
        state = self._extras(group_key).setdefault("daily", {})
        mine = state.get(user_id) or {}
        today = time.strftime("%Y-%m-%d", time.gmtime(now))
        if mine.get("day") == today:
            return {"ok": False, "reason": "cooldown",
                    "streak": int(mine.get("streak") or 0)}
        yesterday = time.strftime(
            "%Y-%m-%d", time.gmtime(now - 86400))
        streak = (int(mine.get("streak") or 0) + 1
                  if mine.get("day") == yesterday else 1)
        amount = DAILY_BASE + min(
            DAILY_STREAK_CAP, DAILY_STREAK_STEP * (streak - 1))
        econ = self.economy
        if not econ.is_verified(group_key, user_id):
            econ.verify_member(group_key, user_id)
        entry = econ.award(group_key, user_id, amount,
                           "daily claim", ref="daily", now=now)
        if entry is None:
            return {"ok": False, "reason": "cap"}
        state[user_id] = {"day": today, "streak": streak}
        self.store.save()
        return {"ok": True, "amount": amount, "streak": streak,
                "balance": econ.balance(group_key, user_id)}

    def jail_state(self, group_key: str, user_id: str, *,
                   now: float | None = None) -> dict[str, Any] | None:
        """The member's live jail record, or None (expired jails
        clear themselves on read)."""
        now = now if now is not None else time.time()
        jail = self._extras(group_key).setdefault("jail", {})
        record = jail.get(user_id)
        if not record:
            return None
        if float(record.get("until") or 0) <= now:
            jail.pop(user_id, None)
            self.store.save()
            return None
        return record

    def rob(self, group_key: str, actor: str, target: str, *,
            rng: Any, now: float | None = None) -> dict[str, Any]:
        """/rob: steal from a member's WALLET.  Failure means
        jail (30 min) with a bail price — the consequence loop
        that makes the crime a system, not a coin toss."""
        now = now if now is not None else time.time()
        if not self.economy_on(group_key):
            return {"ok": False, "reason": "off"}
        if self.jail_state(group_key, actor, now=now):
            return {"ok": False, "reason": "jailed",
                    "jail": self.jail_state(group_key, actor,
                                            now=now)}
        state = self._extras(group_key).setdefault("rob", {})
        last = float((state.get(actor) or {}).get("last") or 0)
        if now - last < 600:
            return {"ok": False, "reason": "cooldown",
                    "wait_s": int(600 - (now - last))}
        econ = self.economy
        out = econ.rob_attempt(group_key, actor, target, rng,
                               now=now)
        if not out.get("ok"):
            return {"ok": False, "reason": out.get("reason",
                                                   "target")}
        state[actor] = {"last": now}
        if out.get("success"):
            self.store.save()
            return {"ok": True, "success": True,
                    "amount": out["amount"],
                    "balance": econ.balance(group_key, actor)}
        wallet = econ.balance(group_key, actor)
        bail = max(10, min(100, wallet // 4)) if wallet else 10
        jail = self._extras(group_key).setdefault("jail", {})
        jail[actor] = {"until": now + 1800, "bail": bail,
                       "at": now}
        self._audit(group_key, "jailed", user_id=actor,
                    reason="failed robbery", bail=bail)
        self.store.save()
        return {"ok": True, "success": False, "jailed": True,
                "bail": bail, "until": now + 1800}

    def bail(self, group_key: str, user_id: str, *,
             now: float | None = None) -> dict[str, Any]:
        """/bail: pay the treasury to walk out of jail now."""
        now = now if now is not None else time.time()
        if not self.economy_on(group_key):
            return {"ok": False, "reason": "off"}
        record = self.jail_state(group_key, user_id, now=now)
        if not record:
            return {"ok": False, "reason": "free"}
        cost = int(record.get("bail") or 10)
        econ = self.economy
        if not econ._move(group_key, user_id, "", cost, "fine",
                          "bail paid to treasury", now):
            return {"ok": False, "reason": "funds", "cost": cost}
        jail = self._extras(group_key).setdefault("jail", {})
        jail.pop(user_id, None)
        self._audit(group_key, "bailed", user_id=user_id, cost=cost)
        self.store.save()
        return {"ok": True, "cost": cost,
                "balance": econ.balance(group_key, user_id)}

    def slots(self, group_key: str, user_id: str, bet: int, *,
              rng: Any, now: float | None = None) -> dict[str, Any]:
        if self.jail_state(group_key, user_id, now=now):
            return {"ok": False, "reason": "jailed"}
        return self.economy.gamble_slots(group_key, user_id, bet,
                                         rng, now=now)

    def coinflip(self, group_key: str, user_id: str, bet: int,
                 choice: str, *, rng: Any,
                 now: float | None = None) -> dict[str, Any]:
        if self.jail_state(group_key, user_id, now=now):
            return {"ok": False, "reason": "jailed"}
        return self.economy.gamble_coinflip(
            group_key, user_id, bet, choice, rng, now=now)

    def top_holders(self, group_key: str, *,
                    limit: int = 10) -> list[dict[str, Any]]:
        """The board: real members by TOTAL wealth (wallet +
        bank + property value at market, §20f — property is
        wealth, and the board tells the truth about wealth).
        Pseudo-members (treasury, vaults, jackpot) are
        accounts, not people — never ranked."""
        econ = self.economy
        rows = []
        for member, wallet in econ.balances(group_key).items():
            if not member or econ.is_pseudo_member(member):
                continue
            bank = econ.bank_balance(group_key, member)
            rows.append({"member": member, "wallet": wallet,
                         "bank": bank, "total": wallet + bank})
        try:
            estate = self.estate
            for row in rows:
                row["property"] = estate.member_property_value(
                    group_key, row["member"])
        except GroupRealEstateError:
            for row in rows:
                row["property"] = 0
        for row in rows:
            row.setdefault("property", 0)
            row["worth"] = row["total"] + row["property"]
        rows.sort(key=lambda r: -r["worth"])
        return rows[:limit]

    def group_quest_cycle(self, group_key: str, *,
                          now: float | None = None
                          ) -> list[dict[str, Any]]:
        """Pay newly-completed group quests into the treasury,
        once per period each (hourly-throttled check; called from
        the message path).  Returns the quests paid THIS cycle."""
        from ...community.group_economy import _period_key

        if not self.system_output_allowed(group_key):
            # §33: an unclaimed room never completes a quest —
            # no treasury exists there to pay into (this also
            # freezes state accrued before the gate existed).
            return []
        now = now if now is not None else time.time()
        extras = self._extras(group_key)
        last = float(extras.get("quest_check_at") or 0)
        if now - last < 3600:
            return []
        extras["quest_check_at"] = now
        if not self.economy_on(group_key):
            return []
        progress = self.economy.group_quest_progress(
            group_key, now=now)
        paid_state = extras.setdefault("group_quests_paid", {})
        paid: list[dict[str, Any]] = []
        for quest_id, quest in progress.items():
            if not quest["complete"]:
                continue
            key = _period_key(quest["period"], now)
            if paid_state.get(quest_id) == key:
                continue
            if self.economy.treasury_bonus(
                    group_key, int(quest["bonus"]),
                    f"group quest: {quest['name']}", now=now):
                paid_state[quest_id] = key
                self._audit(group_key, "group_quest_paid",
                            quest=quest_id, bonus=quest["bonus"])
                paid.append(quest)
        if paid:
            self.store.save()
        return paid

    def season_cycle(self, group_key: str, *,
                     now: float | None = None) -> str | None:
        """Season rollover (DESIGN §5): when the season turns,
        archive the ended season's ledger, crown its champions,
        and return the announcement (None when nothing turned)."""
        from ...community.group_economy import season_key

        if not self.system_output_allowed(group_key):
            return None  # §33: no season turns in foreign rooms
        now = now if now is not None else time.time()
        extras = self._extras(group_key)
        current = season_key(now)
        seen = extras.get("season_seen")
        if seen is None:
            extras["season_seen"] = current
            return None
        if seen == current:
            return None
        econ = self.economy
        champs = econ.season_top(group_key, seen, limit=3)
        archived = econ.archive_season(group_key, seen)
        extras["season_seen"] = current
        self._audit(group_key, "season_rollover", season=seen,
                    archived=archived)
        self.store.save()
        if not champs:
            return None
        members = self.members(group_key)
        lines = [f"🏆 <b>season {seen} champions</b> — "
                 f"a new season starts now:"]
        medals = ["🥇", "🥈", "🥉"]
        for i, (uid, earned) in enumerate(champs):
            record = members.get(uid) or {}
            who = record.get("name") or record.get("username") \
                or uid
            lines.append(f"{medals[i]} {who} — {earned} earned")
        # Season rewards pay GEMS to the podium (§17: a stated
        # premium source), and property wealth gets its own
        # crown — assets persist across seasons; earnings reset.
        for uid, gems in self.estate_season_rewards(
                group_key, champs, now=now):
            record = members.get(uid) or {}
            who = record.get("name") or record.get("username") \
                or uid
            lines.append(f"💎 {who} — +{gems} gems "
                         f"(season reward)")
        landlord = self.estate_top_landlord(group_key)
        if landlord:
            record = members.get(landlord[0]) or {}
            who = record.get("name") or record.get("username") \
                or landlord[0]
            lines.append(f"🏘️ {who} — the group's landlord: "
                         f"{landlord[1]} sparks of property")
        return "\n".join(lines)

    def apply_fine(self, group_key: str, user_id: str, *,
                   amount: int | None = None,
                   now: float | None = None) -> int:
        """The moderation fine sink: settings-driven, paid to the
        treasury, capped at what the member actually holds.  One
        fine per punishment event — a member already muted does
        not get fined again for the same episode (10-minute
        spacing between fines for one member).  ``amount``
        overrides the settings figure — the §30 ladder's designed
        fine steps charge their own amount."""
        now = now if now is not None else time.time()
        if amount is None:
            try:
                amount = int(self.store.settings(
                    group_key).get("fine_amount") or 0)
            except Exception:  # noqa: BLE001 - settings never break chat
                return 0
        amount = int(amount)
        if amount <= 0 or not self.economy_on(group_key):
            return 0
        fines = self._extras(group_key).setdefault("fines", {})
        if now - float(fines.get(user_id) or 0) < 600:
            return 0
        fines[user_id] = now
        econ = self.economy
        paid = min(amount, econ.balance(group_key, user_id))
        if paid <= 0:
            return 0
        if econ._move(group_key, user_id, "", paid, "fine",
                      "moderation fine", now):
            self._audit(group_key, "fine", user_id=user_id,
                        amount=paid)
            return paid
        return 0

    def grant_pin(self, group_key: str, user_id: str) -> None:
        self._extras(group_key).setdefault(
            "pending_pins", {})[user_id] = True
        self.store.save()

    def take_pending_pin(self, group_key: str, user_id: str) -> bool:
        pins = self._extras(group_key).setdefault("pending_pins", {})
        if pins.pop(user_id, None):
            self.store.save()
            return True
        return False

    def record_warn(self, group_key: str, user_id: str, *,
                    by: str = "", reason: str = "") -> int:
        """One admin /warn on a member: lands on the SHARED
        strike spine (defense brief §2.2 — manual warns and
        sensor strikes are one ledger, one ladder), audited, and
        the WINDOWED strike count returned (the caller runs the
        escalation ladder against the warn threshold).  The
        legacy warn bookkeeping is kept alongside for the member
        record."""
        entry = self.store.group(group_key)
        member = entry["members"].setdefault(
            user_id, {"user_id": user_id, "message_count": 0})
        member["warn_count"] = int(member.get("warn_count") or 0) + 1
        member.setdefault("warns", []).append(
            {"ts": time.time(), "by": by, "reason": reason[:200]})
        member["warns"] = member["warns"][-20:]
        cfg = self.defense_config(group_key)
        count = add_strike(member, rule="manual_warn",
                           action="warn", now=time.time(),
                           window_days=cfg.window_days)
        self._audit(group_key, "warn", user_id=user_id, by=by,
                    count=count, reason=reason[:200])
        self.store.save()
        return count

    def clear_warns(self, group_key: str, user_id: str) -> None:
        member = self.store.group(group_key)["members"].get(user_id)
        if member:
            member["warn_count"] = 0
            member["warns"] = []
            member["strikes"] = []
            self.store.save()

    def reset_warn_bookkeeping(self, group_key: str,
                               user_id: str) -> None:
        """Zero the LEGACY warn counters only — the strike spine
        keeps its history.  Used when a manual-warn ladder step
        fires short of the final action: the admin-facing warn
        count restarts, the ladder does not (§30)."""
        member = self.store.group(group_key)["members"].get(user_id)
        if member:
            member["warn_count"] = 0
            member["warns"] = []
            self.store.save()

    # ---- the escalation ladder (§30) ------------------------------
    def escalation_config(self, group_key: str) -> EscalationConfig:
        return EscalationConfig.from_settings(
            self.store.settings(group_key))

    def warn_step(self, group_key: str, count: int
                  ) -> dict[str, Any] | None:
        """The ladder step a manual warn landing on ``count``
        fires (None between steps)."""
        return self.escalation_config(group_key).action_at(count)

    def remove_strike(self, group_key: str, user_id: str, *,
                      rule: str = "", ts: float = 0.0
                      ) -> dict[str, Any] | None:
        """Remove ONE strike — the newest matching ``rule`` (and
        ``ts`` when given).  The undo primitive behind the
        notice layer: reversing a warn removes exactly the
        strike that warn recorded, never the member's whole
        history.  A manual_warn removal also steps the legacy
        warn bookkeeping back one."""
        entry = self.store.group(group_key)
        member = entry["members"].get(user_id)
        if not member:
            return None
        strikes = member.get("strikes") or []
        for idx in range(len(strikes) - 1, -1, -1):
            strike = strikes[idx]
            if not isinstance(strike, dict):
                continue
            if rule and strike.get("rule") != rule:
                continue
            if ts and float(strike.get("ts") or 0) != float(ts):
                continue
            removed = strikes.pop(idx)
            member["strikes"] = strikes
            if removed.get("rule") == "manual_warn":
                member["warn_count"] = max(
                    0, int(member.get("warn_count") or 0) - 1)
                warns = member.get("warns") or []
                if warns:
                    warns.pop()
                    member["warns"] = warns
            self.store.save()
            return removed
        return None

    # ---- cross-group reputation (§32) -----------------------------
    # The book is global; this section is the surface's half of
    # it: identity assembly from EVERY ledger (the §29 stitch),
    # the per-group proposal queue (a record is created only by
    # an admin's confirmation), the join check, dismissals and
    # the reputation restriction state — kept apart from the
    # acceptance state on purpose: a door-rule recheck must never
    # clear a reputation hold, and vice versa.

    def reputation_person(self, group_key: str, user_id: str, *,
                          username: str = "",
                          name: str = "") -> dict[str, Any]:
        """Assemble the §29 person for one member: their id on
        this platform plus every ledger's word on them — all
        usernames they have carried here, and (the bridge) any
        other platform's row whose username matches one of those
        handles.  A changed handle therefore never sheds the
        person, and neither does a platform hop."""
        platform = group_key.split(":", 1)[0]
        user_id = str(user_id)
        ids: dict[str, str] = {platform: user_id}
        usernames: list[str] = []
        names: list[str] = []

        def _note(uname: Any, nm: Any) -> None:
            clean = str(uname or "").strip().lstrip("@").lower()
            if clean and clean not in usernames:
                usernames.append(clean)
            label = str(nm or "").strip()
            if label and label not in names:
                names.append(label)

        _note(username, name)
        for gkey, entry in self.store.data.items():
            if gkey.split(":", 1)[0] != platform:
                continue
            row = (entry.get("members") or {}).get(user_id)
            if row:
                _note(row.get("username"), row.get("name"))
        known = set(usernames)
        if known:
            for gkey, entry in self.store.data.items():
                plat = gkey.split(":", 1)[0]
                for uid, row in (entry.get("members")
                                 or {}).items():
                    handle = str(row.get("username") or "") \
                        .strip().lstrip("@").lower()
                    if handle and handle in known:
                        ids.setdefault(plat, str(uid))
                        _note(handle, row.get("name"))
        return normalize_person({"ids": ids,
                                 "usernames": usernames,
                                 "names": names})

    def reputation_join_check(
            self, group_key: str, user_id: str, *,
            username: str = "", name: str = "",
            now: float | None = None) -> dict[str, Any]:
        """What a join meets: the live records naming this person
        (worst first; aging applied inside the book), this group's
        warning mode, and whether the group already dismissed the
        worst record.  Admission itself is §28's business — this
        check never touches it."""
        now = now if now is not None else time.time()
        mode = str((self.store.settings(group_key)
                    .get("reputation") or {}).get("warning")
                   or "public")
        person = self.reputation_person(
            group_key, user_id, username=username, name=name)
        records = self.reputation.live_for(person, now=now)
        if not records:
            return {"found": False, "mode": mode, "records": [],
                    "person": person}
        for record in records:
            self.reputation.stitch(record["id"], person, now=now)
        record = records[0]
        dismissals = self.store.group(group_key) \
            .get("reputation_dismissals") or {}
        return {"found": True, "mode": mode, "record": record,
                "records": records, "more": len(records) - 1,
                "dismissed": record["id"] in dismissals,
                "person": person}

    def reputation_view(self, group_key: str, user_id: str, *,
                        username: str = "", name: str = "",
                        now: float | None = None) -> dict[str, Any]:
        """The admin /rep read: every record naming this person —
        live, retracted and aged — worst first."""
        person = self.reputation_person(
            group_key, user_id, username=username, name=name)
        return {"person": person,
                "records": self.reputation.all_for(
                    person, now=now)}

    def reputation_note_warned(self, group_key: str,
                               record_id: str, *,
                               user_id: str, mode: str,
                               now: float | None = None) -> None:
        """A warning actually landed (posted, or written to the
        admins-only trail): the book's journal and this group's
        /glog both carry it."""
        self.reputation.note_event(
            record_id, "warned",
            now=now if now is not None else time.time(),
            group=group_key, mode=mode)
        self._audit(group_key, "reputation_warned",
                    user_id=str(user_id), record=record_id,
                    mode=mode)
        self.store.save()

    def reputation_note_suppressed(self, group_key: str,
                                   record_id: str, *,
                                   user_id: str, reason: str,
                                   mode: str) -> None:
        """A warning that did NOT land is journaled too — the
        /glog reader can always see why (§33 gate, dismissal,
        quiet mode, mode off)."""
        self._audit(group_key, "reputation_suppressed",
                    user_id=str(user_id), record=record_id,
                    mode=mode, suppressed=reason)
        self.store.save()

    def reputation_dismiss(self, group_key: str, record_id: str, *,
                           by: str, by_label: str = "",
                           now: float | None = None) -> dict[str, Any]:
        """A local admin dismisses the warning FOR THIS GROUP:
        the record stands globally; the dismissal is journaled
        here and in the book."""
        now = now if now is not None else time.time()
        record = self.reputation.get(record_id)
        if record is None:
            return {"ok": False}
        entry = self.store.group(group_key)
        entry.setdefault("reputation_dismissals", {})[record_id] = {
            "by": str(by), "by_label": by_label, "ts": now}
        self.reputation.note_event(record_id, "dismissed",
                                   now=now, group=group_key,
                                   by=str(by))
        self._audit(group_key, "reputation_dismiss",
                    record=record_id, by=by, by_label=by_label)
        self.store.save()
        return {"ok": True, "record": record}

    def reputation_restricted(self, group_key: str,
                              user_id: str) -> dict[str, Any] | None:
        member = self.member(group_key, str(user_id)) or {}
        state = member.get("reputation") or {}
        return dict(state) if state.get("state") == "restricted" \
            else None

    def reputation_apply_restriction(
            self, group_key: str, user_id: str,
            record: dict[str, Any], *,
            now: float | None = None) -> bool:
        entry = self.store.group(group_key)
        member = (entry.get("members") or {}).get(str(user_id))
        if member is None:
            return False
        member["reputation"] = {
            "state": "restricted", "record_id": record["id"],
            "category": record.get("category") or "",
            "since": now if now is not None else time.time()}
        self._audit(group_key, "reputation_restrict",
                    user_id=str(user_id), record=record["id"],
                    category=record.get("category") or "")
        self.store.save()
        return True

    def reputation_clear(self, group_key: str, user_id: str, *,
                         by: str, by_label: str = "",
                         now: float | None = None) -> dict[str, Any]:
        """An admin taps Clear: the reputation hold on this member
        in this group lifts (the record itself stands)."""
        entry = self.store.group(group_key)
        member = (entry.get("members") or {}).get(str(user_id))
        state = (member or {}).get("reputation") or {}
        if not member or state.get("state") != "restricted":
            return {"cleared": False}
        record_id = str(state.get("record_id") or "")
        member.pop("reputation", None)
        if record_id:
            self.reputation.note_event(
                record_id, "cleared",
                now=now if now is not None else time.time(),
                group=group_key, by=str(by))
        self._audit(group_key, "reputation_clear",
                    user_id=str(user_id), record=record_id,
                    by=by, by_label=by_label)
        self.store.save()
        return {"cleared": True, "record_id": record_id}

    # ── flag proposals: propose → confirm, never propose → brand ──

    def _reputation_state(self, entry: dict[str, Any]) \
            -> dict[str, Any]:
        return entry.setdefault("reputation_state", {"prop_seq": 0})

    def stage_flag(self, group_key: str, *, user_id: str,
                   person: dict[str, Any], category: str,
                   note: str, by: str, by_label: str = "",
                   direct: bool = False, source: str = "",
                   now: float | None = None) -> dict[str, Any]:
        """Stage a flag PROPOSAL in the source group.  Nothing
        here brands anyone: only decide_flag(approve=True) creates
        a book record.  A duplicate pending proposal for the same
        person + category is returned as-is, not stacked."""
        now = now if now is not None else time.time()
        if category not in REPUTATION_CATEGORIES:
            category = "other"
        entry = self.store.group(group_key)
        for prop in self.flag_proposals(group_key, now=now):
            if str(prop.get("user_id")) == str(user_id) \
                    and prop.get("category") == category:
                return {**prop, "duplicate": True}
        state = self._reputation_state(entry)
        state["prop_seq"] = int(state.get("prop_seq") or 0) + 1
        fid = f"fr-{state['prop_seq']:04d}"
        prop = {"id": fid, "user_id": str(user_id),
                "person": normalize_person(person),
                "category": category,
                "note": str(note or "")[:400],
                "by": str(by), "by_label": by_label,
                "direct": bool(direct), "source": source,
                "proposed_ts": now}
        entry.setdefault("reputation_proposals", {})[fid] = prop
        self._audit(group_key, "reputation_proposed",
                    user_id=str(user_id), proposal=fid,
                    category=category, by=by, direct=bool(direct),
                    source=source)
        self.store.save()
        return dict(prop)

    def flag_proposals(self, group_key: str, *,
                       now: float | None = None) \
            -> list[dict[str, Any]]:
        """Pending proposals, oldest first.  Expired ones are
        pruned and journaled — an unconfirmed proposal dies
        quietly and brands nobody."""
        now = now if now is not None else time.time()
        entry = self.store.group(group_key)
        props = entry.get("reputation_proposals") or {}
        out: list[dict[str, Any]] = []
        changed = False
        for fid in list(props):
            prop = props[fid]
            if now - float(prop.get("proposed_ts") or 0) \
                    > REPUTATION_PROPOSAL_TTL_S:
                del props[fid]
                changed = True
                self._audit(group_key,
                            "reputation_proposal_expired",
                            user_id=prop.get("user_id") or "",
                            proposal=fid,
                            category=prop.get("category") or "")
            else:
                out.append(dict(prop))
        if changed:
            self.store.save()
        return out

    def decide_flag(self, group_key: str, fid: str, *,
                    approve: bool, by: str, by_label: str = "",
                    now: float | None = None) -> dict[str, Any]:
        """The human decision.  Approve creates the book record —
        attributed to the CONFIRMING admin, sourced to this group;
        discard leaves only the audit line."""
        now = now if now is not None else time.time()
        entry = self.store.group(group_key)
        self.flag_proposals(group_key, now=now)   # prune expired
        prop = (entry.get("reputation_proposals") or {}).pop(
            fid, None)
        if prop is None:
            return {"ok": False}
        if not approve:
            self._audit(group_key, "reputation_proposal_discarded",
                        user_id=prop.get("user_id") or "",
                        proposal=fid,
                        category=prop.get("category") or "", by=by)
            self.store.save()
            return {"ok": True, "record": None, "proposal": prop}
        record = self.reputation.add_record(
            person=prop.get("person") or {},
            category=prop.get("category") or "other",
            note=prop.get("note") or "",
            source_group=group_key,
            source_platform=group_key.split(":", 1)[0],
            confirmed_by=by, confirmed_by_label=by_label, now=now)
        self._audit(group_key, "reputation_flag",
                    user_id=prop.get("user_id") or "", proposal=fid,
                    record=record["id"], category=record["category"],
                    by=by, by_label=by_label)
        self.store.save()
        return {"ok": True, "record": record, "proposal": prop}

    def reputation_retract_for(self, group_key: str, user_id: str,
                               *, by: str, by_label: str = "",
                               any_source: bool = False,
                               now: float | None = None) \
            -> list[dict[str, Any]]:
        """Retraction clears the record everywhere.  By default a
        group retracts only records IT sourced; any_source is the
        operator's override.  Past warnings are not un-sent — the
        record simply stops warning."""
        now = now if now is not None else time.time()
        person = self.reputation_person(group_key, user_id)
        out: list[dict[str, Any]] = []
        for record in self.reputation.live_for(person, now=now):
            if not any_source \
                    and record.get("source_group") != group_key:
                continue
            gone = self.reputation.retract(
                record["id"], by=by, by_label=by_label, now=now)
            if gone:
                self._audit(group_key, "reputation_retract",
                            user_id=str(user_id),
                            record=record["id"],
                            category=record.get("category") or "",
                            by=by)
                out.append(gone)
        return out

    # ---- the notice layer (§31) -----------------------------------
    def record_notice(self, group_key: str, *, user_id: str,
                      mention: str, rule: str, action: str,
                      count: int = 0, limit: int = 0,
                      strike_ts: float = 0.0, strike_rule: str = "",
                      minutes: int = 0, fine: int = 0, text: str = "",
                      extra: dict[str, Any] | None = None,
                      now: float | None = None) -> str:
        """Record one automated action as an undoable notice.
        The record is the state the posted message's buttons are
        authored from: who, the trigger rule, the action taken,
        the ladder position, and exactly what undo must reverse
        (the strike it recorded, the platform action, any drop
        it opened).  Bounded ring — old notices age out of the
        registry, never out of the audit journal."""
        now = now if now is not None else time.time()
        entry = self.store.group(group_key)
        notices = entry.setdefault("notices", {})
        nid = f"n{int(now * 1000)}-{len(notices)}"
        while nid in notices:
            nid += "x"
        notices[nid] = {
            "id": nid, "user_id": str(user_id),
            "mention": mention, "rule": rule, "action": action,
            "count": int(count), "limit": int(limit),
            "strike_ts": float(strike_ts),
            "strike_rule": strike_rule or rule,
            "minutes": int(minutes), "fine": int(fine),
            "text": text, "extra": dict(extra or {}),
            "message_id": "", "undone": False, "undone_by": "",
            "undone_at": 0.0, "at": now,
        }
        while len(notices) > NOTICE_RING:
            notices.pop(next(iter(notices)))
        self._audit(group_key, "notice_posted", notice=nid,
                    user_id=str(user_id), action=action, rule=rule)
        return nid

    def get_notice(self, group_key: str, notice_id: str
                   ) -> dict[str, Any] | None:
        return self.store.group(group_key).get(
            "notices", {}).get(notice_id)

    def set_notice_message(self, group_key: str, notice_id: str,
                           message_id: str) -> None:
        """Remember the posted message's platform id, so an undo
        can EDIT the notice in place where the platform allows."""
        record = self.get_notice(group_key, notice_id)
        if record is not None and message_id:
            record["message_id"] = str(message_id)
            self.store.save()

    def mark_notice_undone(self, group_key: str, notice_id: str,
                           *, by: str) -> dict[str, Any] | None:
        """Journal an undo exactly like the action was journaled
        (§31): who undid it, when, on which notice."""
        record = self.get_notice(group_key, notice_id)
        if record is None:
            return None
        record["undone"] = True
        record["undone_by"] = str(by)
        record["undone_at"] = time.time()
        self._audit(group_key, "notice_undo", notice=notice_id,
                    by=str(by), user_id=record.get("user_id"),
                    action=record.get("action"),
                    rule=record.get("rule"))
        self.store.save()
        return record

    def clear_acceptance(self, group_key: str, user_id: str
                         ) -> None:
        """Lift a member's acceptance state entirely (the undo
        of an acceptance restriction): the next message re-runs
        the profile check fresh, with its full grace."""
        entry = self.store.group(group_key)
        self._clear_acceptance(entry, user_id)
        self.store.save()

    def refund_fine(self, group_key: str, user_id: str,
                    amount: int, *, by: str = "") -> bool:
        """Pay a notice's fine back out of the treasury (the
        money half of an undo).  Reservation-aware like every
        treasury grant."""
        amount = int(amount or 0)
        if amount <= 0 or not self.economy_on(group_key):
            return False
        if not self.economy.treasury_grant(
                group_key, user_id, amount,
                reason="fine refunded (notice undone)",
                now=time.time()):
            return False
        self._audit(group_key, "fine_refunded", user_id=user_id,
                    amount=amount, by=by)
        self.store.save()
        return True

    def cancel_drop(self, group_key: str, drop_id: str, *,
                    by: str = "") -> dict[str, Any]:
        """Close an open drop early — the undo for a drop
        announcement.  Unclaimed sparks refund to the funder (a
        treasury-funded drop simply releases its reservation);
        claims already paid are journaled history and stand."""
        drops = self._extras(group_key).setdefault("drops", [])
        drop = next((d for d in drops if d.get("id") == drop_id),
                    None)
        if drop is None or not drop.get("open"):
            return {"ok": False, "reason": "no_drop"}
        drop["open"] = False
        remaining = int(drop.get("remaining") or 0)
        funder = str(drop.get("funder") or "")
        refunded = 0
        if remaining > 0 and funder and funder != "treasury":
            if self.economy.treasury_grant(
                    group_key, funder, remaining,
                    reason=f"drop {drop_id} cancelled",
                    now=time.time()):
                refunded = remaining
        self._audit(group_key, "drop_cancelled", drop=drop_id,
                    by=by, refunded=refunded)
        self.store.save()
        return {"ok": True, "refunded": refunded, "drop": drop}

    # ---- read models ----------------------------------------------
    def member(self, group_key: str, user_id: str
               ) -> dict[str, Any] | None:
        return self.store.group(group_key)["members"].get(user_id)

    def members(self, group_key: str) -> dict[str, Any]:
        return dict(self.store.group(group_key)["members"])

    def member_by_message_id(self, group_key: str, message_id: str
                             ) -> dict[str, Any] | None:
        """The ledger member who sent a message Devon recorded
        (the last-message map) — reply targeting for messages
        the platform doesn't attribute in the command itself."""
        wanted = str(message_id or "")
        if not wanted:
            return None
        for member in self.store.group(group_key)["members"].values():
            if str(member.get("last_message_id") or "") == wanted:
                return member
        return None

    def note_resolution(self, group_key: str, user_id: str, *,
                        username: str = "", name: str = "",
                        source: str = "",
                        ts: float | None = None
                        ) -> dict[str, Any] | None:
        """Write a platform-proven identity back into the ledger
        (DESIGN §29, path 6): the member-resolution organ calls
        this whenever a platform surface — not a guess — names
        who a member is, so the local paths get stronger with
        use.  Identity CHANGES still flow through the same
        change-event machinery as a live observation."""
        user_id = str(user_id or "")
        if not user_id:
            return None
        now = ts if ts is not None else time.time()
        self._observe_identity(group_key, user_id,
                               username=username or "",
                               name=name or "", ts=now)
        member = self.store.group(group_key)["members"].get(user_id)
        if member is not None and source:
            member["resolved_via"] = source
        self.store.save()
        return member

    def changes(self, group_key: str, limit: int = 50
                ) -> list[dict[str, Any]]:
        return list(
            self.store.group(group_key)["changes"])[-limit:]

    def rules_text(self, group_key: str) -> str:
        """The published rules, rendered from the SAME settings
        the enforcer reads (DESIGN §12: one config, two readers)
        — through the typography organ (§27, announcement
        class): the rules list is a literal block, so a rule's
        own text can never become markup.
        """
        from .typography import Card

        settings = self.store.settings(group_key)
        card = Card("announcement", title="Group rules", icon="📜")
        rules = settings.get("rules") or []
        if rules:
            card.board("\n".join(
                f"{i}. {rule}"
                for i, rule in enumerate(rules, 1)))
        else:
            card.line("No written rules yet — be decent, no "
                      "spam, no flooding.")
        mod = settings["moderation"]
        card.line(
            f"Moderation: flood limit {mod['flood_max']} messages "
            f"per {int(mod['flood_window_s'])}s · repeats ×"
            f"{mod['repeat_threshold']} are removed")
        if mod.get("link_allowlist"):
            card.line("Links: only allowlisted domains are "
                      "permitted.")
        return self._card_render(group_key, card)


def default_surface(workspace_dir: str | Path) -> GroupSurface:
    """The runtime's surface: store under the workspace's data dir."""
    root = Path(workspace_dir) / "data" / "social"
    return GroupSurface(GroupStore(root / "groups.json").load())
