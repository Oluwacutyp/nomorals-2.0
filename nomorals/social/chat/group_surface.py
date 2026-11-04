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

from ...community.group_conflict import (
    CONFLICT_DEFAULTS,
    GroupConflict,
    GroupConflictError,
    WarBook,
)
from ...community.group_exchange import EXCHANGE_DEFAULTS, GroupExchange
from ...community.group_gangs import (
    GANGS36_DEFAULTS,
    GangRegistry,
    GangRegistryError,
    person_key as _person_key,
)
from ...community.group_real_estate import (
    DISTRICTS,
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
from .group_cleanup import CLEANUP_DEFAULTS
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
from .group_boards import (
    BOARD_LIMIT as _BOARD_LIMIT,
    BoardBook,
    board_title as _board_title,
    is_game_flavored as _board_game_flavored,
    metric_format as _board_metric_format,
    movement_marks as _board_movement,
    parse_board_id as _parse_board_id,
)
from .group_defense import (
    DEFAULTS as DEFENSE_DEFAULTS,
)
from .group_rhythm import (
    BORDERLINE_SCORE,
    CLEAR_SCORE,
    CONTRIBUTION_RING,
    GROUP_MESSAGE_MILESTONES,
    HELD_RING,
    MEMBER_MESSAGE_MILESTONES,
    RHYTHM_DEFAULTS,
    day_window as _rhythm_day_window,
    in_window as _rhythm_in_window,
    is_contribution_candidate,
    local_parts as _rhythm_local_parts,
    parse_hhmm as _rhythm_parse_hhmm,
    score_contribution,
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
        "cleanup": copy.deepcopy(CLEANUP_DEFAULTS),
                                        # member cleanup: the
                                        # deleted-account scan +
                                        # the inactivity sweep
                                        # (removals switch +
                                        # the inactivity window
                                        # in days) — the class
                                        # the DESIGN gap list
                                        # named since 2026-10-25
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
        "conflict": copy.deepcopy(CONFLICT_DEFAULTS),
        "gangs": copy.deepcopy(GANGS36_DEFAULTS),
                                        # §36 cross-group gangs:
                                        # the chapter capacity
                                        # table, the flagship's
                                        # identity, HQ prices,
                                        # raid + prestige tuning
        "exchange": copy.deepcopy(EXCHANGE_DEFAULTS),
                                        # the §25 exchange: the
                                        # weekly pool, the rate
                                        # band's base, the reverse
                                        # spread, the territory
                                        # skim, the allowance base
        "channel": {"cadence_min": 0,
                    "links": {},
                    "analytics": True},
                                        # a CHANNEL entry's own
                                        # block (§16/P4a): the
                                        # posting cadence floor,
                                        # the trigger links
                                        # {source group key:
                                        # [trigger kinds]} its
                                        # queue cross-posts from,
                                        # and whether subscriber
                                        # counts are sampled where
                                        # the platform exposes
                                        # them.  Groups carry the
                                        # block unused.
        "boards": {"daily": [],
                   "weekly": [],
                   "hour": 20},
                                        # §14 board publishing:
                                        # board specs this group
                                        # posts on the daily /
                                        # weekly beat at its
                                        # local hour (bare metric
                                        # names mean this group's
                                        # own board; full board
                                        # ids publish those)
        "community_id": "",             # the community this
                                        # group belongs to, when
                                        # it carries one (§14
                                        # community boards read
                                        # the tag; untagged
                                        # groups join none)
        "rhythm": copy.deepcopy(RHYTHM_DEFAULTS),
                                        # the room's clock (P3):
                                        # silent hours + UTC
                                        # offset, nightshield,
                                        # the group briefing,
                                        # reminders, poll
                                        # defaults, §26 review
                                        # the conflict layer
                                        # (§23/§24/§35): mode
                                        # off|crime|warfare, the
                                        # system-drop budget,
                                        # heist/crime cooldowns,
                                        # war escrow + round
                                        # clock, territory rates,
                                        # the police faction
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
                                     "expression", "reputation",
                                     "conflict", "rhythm",
                                     "channel", "boards",
                                     "exchange", "gangs",
                                     "cleanup")})
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
        merged["conflict"] = _deep_merge(
            copy.deepcopy(CONFLICT_DEFAULTS),
            stored.get("conflict") or {})
        merged["rhythm"] = _deep_merge(
            copy.deepcopy(RHYTHM_DEFAULTS),
            stored.get("rhythm") or {})
        merged["channel"] = _deep_merge(
            {"cadence_min": 0, "links": {}, "analytics": True},
            stored.get("channel") or {})
        merged["boards"] = _deep_merge(
            {"daily": [], "weekly": [], "hour": 20},
            stored.get("boards") or {})
        merged["exchange"] = _deep_merge(
            copy.deepcopy(EXCHANGE_DEFAULTS),
            stored.get("exchange") or {})
        merged["gangs"] = _deep_merge(
            copy.deepcopy(GANGS36_DEFAULTS),
            stored.get("gangs") or {})
        merged["cleanup"] = _deep_merge(
            copy.deepcopy(CLEANUP_DEFAULTS),
            stored.get("cleanup") or {})
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
                 ban_list: BanList | None = None,
                 owner_registry: Any = None) -> None:
        self.store = store
        self.bans = ban_list or BanList()
        # The §19 owner record (identities / sudo / deputies):
        # the pipeline reads roles from it; None = no record
        # wired (bare rigs), roles simply never resolve.
        self.owner_registry = owner_registry
        self._moderators: dict[str, Moderator] = {}
        self._msg_cache: dict[str, deque] = {}
        self._signatures: SignatureStore | None = None
        self._reputation: ReputationBook | None = None
        self._conflict: GroupConflict | None = None
        self._exchange: GroupExchange | None = None
        self._war_book: WarBook | None = None
        self._gangs: GangRegistry | None = None
        self._gangs_touched: set[str] = set()

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
        self._rhythm_count(group_key, "joins", now)
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
        self._rhythm_count(group_key, "leaves", now)
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

    # ---- member cleanup (deleted accounts + inactivity) -------
    def cleanup_cfg(self, group_key: str) -> dict[str, Any]:
        """The group's member-cleanup block, defaults-merged:
        whether the remove commands may act, and the inactivity
        window in days the sweep judges against."""
        return dict(self.store.settings(group_key).get(
            "cleanup") or {})

    def cleanup_auto_due(self, group_key: str, *,
                         now: float | None = None,
                         interval_s: float = 6 * 3600) -> bool:
        """Is this group's automatic deleted-account pass due?
        The stamp lives on the group entry (runtime state, not
        a setting): a full member enumeration is an upkeep cost,
        never a per-message one."""
        last = float((self.store.group(group_key).get(
            "cleanup_auto") or {}).get("last_run") or 0)
        now = now if now is not None else time.time()
        return now - last >= interval_s

    def note_cleanup_auto(self, group_key: str, *,
                          ts: float | None = None) -> None:
        """Stamp the automatic pass as attempted (stamped before
        the work, so a failing enumeration doesn't retry on
        every message)."""
        entry = self.store.group(group_key)
        entry["cleanup_auto"] = {
            "last_run": ts if ts is not None else time.time()}
        self.store.save()

    def mark_removed(self, group_key: str, user_id: str, *,
                     ts: float | None = None) -> None:
        """Ledger bookkeeping for a cleanup removal the runtime
        already executed + journaled through the admin seam:
        the member is out, so the ledger stops counting them as
        present (a later Bot-API-path scan must not re-remove a
        ghost).  No audit entry here — the caller's admin-action
        journal IS the record — and no goodbye: a removal is not
        a leave-taking."""
        entry = self.store.group(group_key)
        member = entry["members"].get(user_id)
        if member is not None:
            member["left_ts"] = ts if ts is not None \
                else time.time()
            self.store.save()

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
        if now < float(state.get("raid_until") or 0):
            return True
        # The nightshield (P3) rides this same posture: while
        # it is up, every acceptance decision reads raid — an
        # open door behaves as a request door and grace shrinks
        # (effective_cfg), with the CRITERIA unchanged.
        return self.rhythm_nightshield(group_key, now=now)

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
                       sender_is_operator: bool = False,
                       game_input: bool = False
                       ) -> dict[str, Any]:
        """One ordinary group message: ledger observation, then
        the DEFENSE pipeline in its fixed order (brief §1.4) —
        identity → anticrash inspection → Moderator → envelope
        sensors → escalation spine → fines / economy / notices.
        Returns ``{"changes", "actions", "notices"}``; actions
        are dicts for the caller to execute (delete / warn / mute
        / kick / ban).  A message any stage deletes earns NO
        economy credit and fires no quest/season notices.
        ``game_input`` is the caller's verdict (from real game
        room state) that this message is a seated player's input
        to a live game — it exempts the message from the
        Moderator's counting stages only."""
        now = ts if ts is not None else time.time()
        changes = self._observe_identity(
            group_key, user_id, username=username, name=name, ts=now)
        entry = self.store.group(group_key)
        member = entry["members"][user_id]
        member["message_count"] = int(
            member.get("message_count") or 0) + 1
        if message_id:
            member["last_message_id"] = str(message_id)
        # §19/§20: the owner ladder is marked on the ledger and
        # read by the spine below — owner, sudo and this group's
        # deputies stand on `privileged`, so AUTOMATIC actions
        # never target them (manual admin actions are separate
        # handlers, and still log).  A stranger wearing the
        # owner's face is flagged to the held audit (§20d),
        # once per flag change — never per message.
        role = ""
        if self.owner_registry is not None:
            platform = group_key.split(":", 1)[0]
            role = self.owner_registry.role_of(
                platform, user_id, group_key=group_key)
            if role:
                if member.get("role") != role:
                    member["role"] = role
            elif "role" in member:
                del member["role"]
            if not role:
                flag = self.owner_registry.owner_elsewhere(
                    platform, user_id, username=username,
                    name=name)
                if flag and member.get(
                        "impersonation_flag") != flag:
                    member["impersonation_flag"] = flag
                    self._audit(group_key, "impersonation_flag",
                                user_id=user_id, reason=flag)
                elif not flag and "impersonation_flag" in member:
                    del member["impersonation_flag"]
        # §19's hard line: automatic actions NEVER target the
        # ladder.  The stages below still AUDIT what they saw
        # (the record is evidence) — they simply land nothing
        # on an owner, a sudo, or this group's deputy.  Manual
        # admin actions are other handlers entirely, and log.
        role_exempt = bool(role)
        settings = self.store.settings(group_key)
        cfg = DefenseConfig.from_settings(settings)
        esc = EscalationConfig.from_settings(settings)
        warn_threshold = int(settings.get("warn_threshold") or 3)
        suppressed = not cfg.autonomy
        privileged = sender_is_admin or sender_is_operator \
            or bool(role)
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

        msg_text = text  # finish() assigns its own `text`
        # below (the acceptance notice) — the rhythm beat reads
        # the message through this alias.

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
                # The conflict beat (§23/§24/§35): system
                # drops, war rounds, territory income and the
                # police ride the same cycle — the organ
                # self-gates on mode, claim and freeze.
                notices.extend(self.conflict_cycle(group_key,
                                                   now=now))
                # The rhythm beat (P3): the day counter, the
                # §26 contribution read on this surviving
                # message, and the rhythm cycle itself — polls
                # closing, milestones, reminders, the briefing,
                # the morning digest.
                if self.system_output_allowed(group_key):
                    # §33: rhythm state (day shape, contribution
                    # events) is bot-system state — it accrues in
                    # claimed rooms only, like every other organ.
                    self._rhythm_count(group_key, "messages",
                                       now)
                    self._note_contribution(
                        group_key, user_id, msg_text,
                        has_media=bool(getattr(env, "media",
                                               None)),
                        has_link=("http" in msg_text
                                  or "t.me/" in msg_text
                                  or any(e.get("type") == "url"
                                         for e in (getattr(
                                             env, "entities",
                                             None) or []))),
                        now=now)
                notices.extend(self.rhythm_cycle(group_key,
                                                 now=now))
                # The §14 board beat: this group's scheduled
                # board publishes ride the same notices list, so
                # silent hours hold them into the digest exactly
                # like every other cycle post.
                notices.extend(self.boards_cycle(group_key,
                                                 now=now))
                if notices and self.rhythm_silent(group_key,
                                                   now=now):
                    # Silent hours: non-urgent SYSTEM output is
                    # held for the morning digest.  Moderation
                    # never travels this list — warnings ride the
                    # actions and their §31 notices — and the
                    # acceptance warning below is enforcement
                    # too: neither is ever held.
                    for held_text in notices:
                        self.rhythm_hold(group_key, "cycle",
                                         held_text, now=now)
                    notices = []
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
                if cfg.anticrash_action == "delete" \
                        and not role_exempt:
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
        # Two verdicts ride in from the caller and are forwarded
        # verbatim — the Moderator stays text-only and dumb:
        # * privileged (owner/operator/admin, resolved above from
        #   the adapter's own role answer): staff are NEVER
        #   auto-moderated — no warn/delete/mute out of the
        #   automated stages.  (Stage 1 anticrash keeps its own
        #   shipped rule for staff payloads: deleted for safety,
        #   never struck.)
        # * game_input (the runtime's games seam, judged from
        #   real room state — a live room, the sender seated, the
        #   text a move the game recognises): solicited play is
        #   not spam, so the counting stages do not count it.
        if text:
            for action in self._moderator(group_key).check_message(
                    group_key, user_id, text, ts=now,
                    exempt=privileged, uncounted=game_input):
                self._audit(group_key, "moderation",
                            action=action.kind, rule=action.rule,
                            reason=action.reason, user_id=user_id,
                            **({"suppressed": "autonomy_off"}
                               if suppressed else {}),
                            **({"role_exempt": role}
                               if role_exempt else {}))
                if role_exempt:
                    # §19: audited above, landed never — the
                    # Moderator's verdict on the owner's own
                    # message is a record, not an action.
                    continue
                actions.append(action.to_dict())
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
            if not role_exempt:
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

    # ── wager purse (games): the estate side of the table ──
    # A game played in this group stakes REAL group coins: the
    # runtime binds the room's purse to these methods.  Debits are
    # gated exactly like awards (§33 + economy on) — no wager state
    # accrues in a foreign room.  Credits are NOT gated: a credit
    # returns money a gated debit already took (stake back, push,
    # winnings), and refusing it would trap the member's coins.

    def game_wager_available(self, group_key: str) -> bool:
        """True when a wager table may bind this group's purse."""
        try:
            return bool(self.system_output_allowed(group_key)
                        and self.economy_on(group_key))
        except Exception:  # noqa: BLE001
            return False

    def game_wager_balance(self, group_key: str, user_id: str) -> int:
        try:
            return int(self.economy.balance(group_key, user_id))
        except Exception:  # noqa: BLE001
            return 0

    def game_wager_debit(self, group_key: str, user_id: str,
                         amount: int, reason: str, ref: str = "") -> bool:
        """Escrow a wager stake from the member's group wallet.
        False = not available or insufficient — nothing moved."""
        try:
            if amount <= 0 or not self.game_wager_available(group_key):
                return False
            econ = self.economy
            if not econ.is_verified(group_key, user_id):
                econ.verify_member(group_key, user_id)
            return econ.wager_debit(group_key, user_id, int(amount),
                                    reason, ref=ref) is not None
        except Exception:  # noqa: BLE001
            return False

    def game_wager_credit(self, group_key: str, user_id: str,
                          amount: int, reason: str,
                          ref: str = "") -> int:
        """Pay a settled wager into the member's group wallet
        (stake returned / winnings).  Returns what landed."""
        try:
            if amount <= 0:
                return 0
            econ = self.economy
            if not econ.is_verified(group_key, user_id):
                econ.verify_member(group_key, user_id)
            entry = econ.wager_credit(group_key, user_id, int(amount),
                                      reason, ref=ref)
            return int(entry.amount) if entry else 0
        except Exception:  # noqa: BLE001
            return 0

    # ── the exchange (§25): games coins ⇄ group sparks ──
    # One purse, two pockets.  The math lives in the exchange
    # organ; the surface resolves what the organ cannot see —
    # the territory skim (a power gang, through the conflict
    # organ) and the member's gang rank — and moves both
    # wallets: the estate through its convert legs, the games
    # wallet through the caller-bound coin wallet (the runtime
    # binds the real PlayerStore; the wager purse binding is
    # untouched by any of this).  Refusals are whole: every
    # check runs before the first movement, and no leg is
    # partial — either the full amount converts or nothing does.

    @property
    def exchange(self) -> GroupExchange:
        if self._exchange is None:
            self._exchange = GroupExchange(self.economy)
        return self._exchange

    def exchange_cfg(self, group_key: str) -> dict[str, Any]:
        return dict(self.store.settings(group_key).get(
            "exchange") or {})

    def _exchange_power(self, group_key: str,
                        now: float) -> dict[str, Any] | None:
        """The gang in power (§23 capture), or None — an
        unreadable conflict state means no skim, never a guess."""
        try:
            return self.conflict.power_gang(group_key, now=now)
        except GroupConflictError:
            return None

    def _gang_rank(self, group_key: str, member: str) -> int:
        """0 none · 1 member · 2 officer · 3 leader."""
        try:
            gang = self.conflict.gang_of(group_key, member)
        except GroupConflictError:
            return 0
        if not gang:
            return 0
        if member == gang.get("leader"):
            return 3
        if member in (gang.get("officers") or []):
            return 2
        return 1

    def exchange_quote(self, group_key: str, user_id: str, *,
                       level: int = 1, mastery_steps: int = 0,
                       now: float | None = None) -> dict[str, Any]:
        """The pre-conversion picture: live rate both ways,
        pool state, the member's allowances, the skim in force."""
        now = now if now is not None else time.time()
        cfg = self.exchange_cfg(group_key)
        power = self._exchange_power(group_key, now)
        skim_pct = int(cfg.get("skim_pct") or 0) if power else 0
        quote = self.exchange.quote(
            group_key, cfg, member=user_id, level=level,
            mastery_steps=mastery_steps,
            gang_rank=self._gang_rank(group_key, user_id),
            skim_pct=skim_pct, now=now)
        quote["skim_gang"] = (str(power.get("name") or "")
                              if power else "")
        quote["sparks"] = self.economy.balance(group_key, user_id)
        return quote

    def exchange_convert(self, group_key: str, user_id: str,
                         direction: str, amount: int | None, *,
                         wallet: Any, level: int = 1,
                         mastery_steps: int = 0,
                         now: float | None = None) -> dict[str, Any]:
        """Convert for one member.  ``direction``: "in" (coins
        → sparks) or "out" (sparks → coins).  ``amount`` is in
        the INPUT currency; None means the maximum the pool,
        the allowance and the balance jointly allow.  ``wallet``
        is the bound games-coin wallet: balance() / spend(n,
        reason) -> bool / credit(n, reason) -> int."""
        from ...core.ids import new_short_id

        from ...community.group_exchange import (
            forward_sparks,
            min_coins_for_spark,
            reverse_coins,
        )
        now = now if now is not None else time.time()
        cfg = self.exchange_cfg(group_key)
        if not cfg.get("enabled", True):
            return {"ok": False, "reason": "off"}
        econ = self.economy
        if not econ.is_verified(group_key, user_id):
            econ.verify_member(group_key, user_id)
        power = self._exchange_power(group_key, now)
        skim_pct = max(0, min(10, int(cfg.get("skim_pct") or 0))) \
            if power else 0
        quote = self.exchange_quote(
            group_key, user_id, level=level,
            mastery_steps=mastery_steps, now=now)
        rate = quote["rate"]
        cv = new_short_id("cv")
        if direction == "in":
            pool_left = quote["pool"]["remaining"]
            if quote["closed"]:
                return {"ok": False, "reason": "closed",
                        "quote": quote}
            allow_left = quote["allowance_left"]
            coins = self._exchange_max_coins(
                rate, pool_left, allow_left, skim_pct,
                int(wallet.balance())) if amount is None \
                else int(amount)
            if coins <= 0:
                if amount is not None:
                    return {"ok": False, "reason": "amount",
                            "quote": quote}
                # "max" fitted nothing: name the binding cap.
                if allow_left < 1:
                    return {"ok": False, "reason": "allowance",
                            "allowance_left": allow_left,
                            "quote": quote}
                if int(wallet.balance()) < min_coins_for_spark(rate):
                    return {"ok": False, "reason": "funds_coins",
                            "have": int(wallet.balance()),
                            "quote": quote}
                return {"ok": False, "reason": "pool",
                        "pool_left": pool_left, "quote": quote}
            gross = forward_sparks(coins, rate)
            if gross < 1:
                return {"ok": False, "reason": "too_small",
                        "min_coins": min_coins_for_spark(rate),
                        "quote": quote}
            if gross > pool_left:
                return {"ok": False, "reason": "pool",
                        "pool_left": pool_left, "quote": quote}
            skim = gross * skim_pct // 100 if skim_pct else 0
            net = gross - skim
            if net > allow_left:
                return {"ok": False, "reason": "allowance",
                        "allowance_left": allow_left, "quote": quote}
            if int(wallet.balance()) < coins:
                return {"ok": False, "reason": "funds_coins",
                        "have": int(wallet.balance()), "quote": quote}
            if not wallet.spend(coins, f"exchange: convert to "
                                f"{group_key} sparks"):
                return {"ok": False, "reason": "funds_coins",
                        "have": int(wallet.balance()), "quote": quote}
            econ.convert_credit(
                group_key, user_id, net,
                f"converted {coins} games coins → {net} sparks "
                f"at {rate}/100", ref=f"convert:{cv}:in", now=now)
            if skim:
                econ.convert_credit(
                    group_key, "", skim,
                    f"exchange territory skim to "
                    f"{power.get('name')} (conversion {cv})",
                    ref=f"convert:{cv}:skim", now=now)
                self.conflict.gang_skim_credit(
                    group_key, str(power.get("id") or ""), skim,
                    now=now)
            return {"ok": True, "direction": "in", "coins": coins,
                    "gross": gross, "skim": skim, "sparks": net,
                    "rate": rate, "gang": quote["skim_gang"],
                    "coins_left": int(wallet.balance()),
                    "sparks_left": econ.balance(group_key, user_id),
                    "quote": self.exchange_quote(
                        group_key, user_id, level=level,
                        mastery_steps=mastery_steps, now=now)}
        # ── the reverse door: sparks → coins ──
        allow_left = quote["reverse_left"]
        sparks = min(int(econ.balance(group_key, user_id)),
                     allow_left) if amount is None else int(amount)
        if sparks <= 0:
            return {"ok": False, "reason": "amount", "quote": quote}
        coins = reverse_coins(sparks, rate, cfg)
        if coins < 1:
            need = 1
            while reverse_coins(need, rate, cfg) < 1:
                need += 1
            return {"ok": False, "reason": "too_small",
                    "min_sparks": need, "quote": quote}
        if sparks > allow_left:
            return {"ok": False, "reason": "allowance",
                    "allowance_left": allow_left, "quote": quote}
        skim = sparks * skim_pct // 100 if skim_pct else 0
        if econ.convert_debit(
                group_key, user_id, sparks,
                f"converted {sparks} sparks → {coins} games coins "
                f"at {rate}/100", ref=f"convert:{cv}:out",
                now=now) is None:
            return {"ok": False, "reason": "funds_sparks",
                    "have": econ.balance(group_key, user_id),
                    "quote": quote}
        if skim:
            econ.convert_credit(
                group_key, "", skim,
                f"exchange territory skim to "
                f"{power.get('name')} (conversion {cv})",
                ref=f"convert:{cv}:skimout", now=now)
            self.conflict.gang_skim_credit(
                group_key, str(power.get("id") or ""), skim, now=now)
        wallet.credit(coins, f"exchange: convert from "
                      f"{group_key} sparks")
        return {"ok": True, "direction": "out", "sparks": sparks,
                "skim": skim, "coins": coins, "rate": rate,
                "gang": quote["skim_gang"],
                "coins_left": int(wallet.balance()),
                "sparks_left": econ.balance(group_key, user_id),
                "quote": self.exchange_quote(
                    group_key, user_id, level=level,
                    mastery_steps=mastery_steps, now=now)}

    @staticmethod
    def _exchange_max_coins(rate: int, pool_left: int,
                            allow_left: int, skim_pct: int,
                            coin_balance: int) -> int:
        """The largest coin amount whose conversion fits the
        pool AND the allowance AND the member's coins."""
        from ...community.group_exchange import forward_sparks

        gross_cap = pool_left
        if skim_pct > 0:
            approx = allow_left * 100 // max(1, 100 - skim_pct)
            gross_cap = min(gross_cap, approx + 1)
        else:
            gross_cap = min(gross_cap, allow_left)
        coins = min(int(coin_balance),
                    max(0, gross_cap) * 100 // max(1, rate))
        # floor dust can overshoot the caps by a hair — walk
        # down until the real (floored) figures fit.
        for _ in range(1200):
            if coins <= 0:
                break
            gross = forward_sparks(coins, rate)
            net = gross - (gross * skim_pct // 100 if skim_pct
                           else 0)
            if gross <= pool_left and net <= allow_left:
                break
            coins -= 1
        return max(0, coins)

    def work_shift(self, group_key: str, user_id: str, *,
                   now: float | None = None) -> dict[str, Any]:
        """/work: a cooldown-bound shift.  Pay follows engagement
        (the organ's hashrate) and the member's shift level."""
        now = now if now is not None else time.time()
        if not self.economy_on(group_key):
            return {"ok": False, "reason": "off"}
        if self.hospital_state(group_key, user_id, now=now):
            return {"ok": False, "reason": "hospital",
                    "hospital": self.hospital_state(
                        group_key, user_id, now=now)}
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
        # Governance capture (§23): while a gang holds power
        # and has SET its tax, the gang's rate — inside the
        # hard 0–10 range — is the group's transfer tax.  The
        # mechanism (this fee leg) never changes; who sets
        # the number does.
        try:
            power = self.conflict.power_gang(group_key, now=now)
        except GroupConflictError:
            power = None
        if power is not None and power.get("tax_pct") is not None:
            pct = int(power["tax_pct"])
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
        try:
            total += self.conflict.reservations(group_key)
        except GroupConflictError:
            # Gang treasuries, war pots and bounties live in
            # the conflict state: unreadable = unknown escrow,
            # unspendable rather than guessed at (the estate
            # precedent above).
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
        return self._execute_proposal(group_key, proposal_id,
                                      by=by, now=now)

    def _execute_proposal(self, group_key: str, proposal_id: str,
                          *, by: str,
                          now: float) -> dict[str, Any]:
        """The one execution path for governance proposals —
        an admin's approve and a passed treasury poll both land
        here, so a poll's execution journals exactly like any
        other governance act (``proposal_executed``, by = the
        approver or ``poll <ref>``)."""
        proposals = self._extras(group_key).setdefault(
            "proposals", {})
        proposal = proposals.get(proposal_id)
        if not proposal or not proposal.get("open"):
            return {"ok": False, "reason": "no_proposal"}
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

    def jail_member(self, group_key: str, user_id: str,
                    minutes: int, *, reason: str,
                    now: float | None = None) -> dict[str, Any]:
        """Put a member in the group's jail — THE jail, shared
        by rob failures, the conflict verbs, heist busts and
        the police (one consequence, every road leads here).
        Bail follows the rob shape: a quarter of the wallet,
        floored and capped."""
        now = now if now is not None else time.time()
        wallet = self.economy.balance(group_key, user_id)
        bail = max(10, min(150, wallet // 4)) if wallet else 10
        jail = self._extras(group_key).setdefault("jail", {})
        record = {"until": now + int(minutes) * 60, "bail": bail,
                  "at": now, "reason": str(reason)[:120]}
        jail[user_id] = record
        self._audit(group_key, "jailed", user_id=user_id,
                    reason=str(reason)[:120], bail=bail,
                    minutes=int(minutes))
        self.store.save()
        return record

    def jail_clear(self, group_key: str, user_id: str) -> bool:
        """Lift a jail record (the undo of a jail notice)."""
        jail = self._extras(group_key).setdefault("jail", {})
        if user_id not in jail:
            return False
        jail.pop(user_id, None)
        self.store.save()
        return True

    def hospital_state(self, group_key: str, user_id: str, *,
                       now: float | None = None
                       ) -> dict[str, Any] | None:
        """The member's live hospital record (§23 hit verbs put
        their targets here), or None — expired stays clear
        themselves on read, like jail."""
        now = now if now is not None else time.time()
        ward = self._extras(group_key).setdefault("hospital", {})
        record = ward.get(user_id)
        if not record:
            return None
        if float(record.get("until") or 0) <= now:
            ward.pop(user_id, None)
            self.store.save()
            return None
        return record

    def hospital_member(self, group_key: str, user_id: str,
                        minutes: int, *, bill: int,
                        by: str = "",
                        now: float | None = None
                        ) -> dict[str, Any]:
        """Put a member in hospital with a bill: /bail pays the
        bill to walk out early, a medkit discharges free, or
        the stay simply runs out."""
        now = now if now is not None else time.time()
        ward = self._extras(group_key).setdefault("hospital", {})
        record = {"until": now + int(minutes) * 60,
                  "bill": int(bill), "at": now, "by": by}
        ward[user_id] = record
        self._audit(group_key, "hospitalized", user_id=user_id,
                    bill=int(bill), minutes=int(minutes), by=by)
        self.store.save()
        return record

    def hospital_clear(self, group_key: str, user_id: str) -> bool:
        ward = self._extras(group_key).setdefault("hospital", {})
        if user_id not in ward:
            return False
        ward.pop(user_id, None)
        self.store.save()
        return True

    def _shielded(self, group_key: str, user_id: str, *,
                  now: float) -> bool:
        """Inside their first 24h here?  The bully index
        weighs hitting a newcomer hardest (§24) — the shield
        is read from the member ledger's own stamps: the
        join stamp when a join event was seen, else the
        first-seen stamp (a member already in the room when
        Devon arrived is new TO THE SYSTEM either way)."""
        record = self.member(group_key, user_id) or {}
        joined = float(record.get("joined_ts")
                       or record.get("first_seen") or 0)
        return bool(joined and 0 <= now - joined < 86400)

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
        if self.hospital_state(group_key, actor, now=now):
            return {"ok": False, "reason": "hospital",
                    "hospital": self.hospital_state(
                        group_key, actor, now=now)}
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
        if self._conflict_mode(group_key) != "off":
            try:
                self.conflict.note_crime(
                    group_key, actor=actor, target=target,
                    verb="rob", success=bool(out.get("success")),
                    take=int(out.get("amount") or 0),
                    shielded=self._shielded(group_key, target,
                                            now=now),
                    now=now)
            except GroupConflictError:
                pass  # a frozen conflict organ never blocks rob;
                # the freeze itself surfaces on conflict paths
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
            # Not jailed — but a hospital bill is also a door
            # /bail opens (§23): pay it and walk out early.
            ward = self.hospital_state(group_key, user_id,
                                       now=now)
            if not ward:
                return {"ok": False, "reason": "free"}
            bill = int(ward.get("bill") or 0)
            if bill <= 0:
                return {"ok": False, "reason": "hospital",
                        "hospital": ward}
            if not self.economy._move(
                    group_key, user_id, "", bill, "fine",
                    "hospital bill paid", now):
                return {"ok": False, "reason": "funds",
                        "cost": bill}
            self.hospital_clear(group_key, user_id)
            self._audit(group_key, "hospital_paid",
                        user_id=user_id, cost=bill)
            self.store.save()
            return {"ok": True, "cost": bill, "hospital": True,
                    "balance": self.economy.balance(
                        group_key, user_id)}
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
        text = "\n".join(lines)
        # A turned season is a channel trigger (§16): linked,
        # claimed channels get the podium cross-posted from
        # their posting queue.
        self._channel_trigger(group_key, "season_podium", text,
                              now=now)
        return text

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

    # ---- the conflict layer (§23/§24/§35) -------------------------
    @property
    def conflict(self) -> GroupConflict:
        if self._conflict is None:
            self._conflict = GroupConflict(self.economy)
        return self._conflict

    @property
    def war_book(self) -> WarBook:
        """The inter-group war book: ONE global ledger beside
        the group store (the reputation book's shape)."""
        if self._war_book is None:
            self._war_book = WarBook(
                self.store.path.parent / "conflict_wars.json")
        return self._war_book

    # ---- cross-group gangs (§36) --------------------------------

    @property
    def gangs(self) -> GangRegistry:
        """The §36 registry: ONE global book beside the group
        store (the war book's shape) — global gang records,
        chapters, the flagship, war declarations, claim codes
        and the owner's police focus."""
        if self._gangs is None:
            self._gangs = GangRegistry(
                self.conflict,
                self.store.path.parent / "gang_registry.json")
        return self._gangs

    def gangs_cfg(self, group_key: str) -> dict[str, Any]:
        return dict(self.store.settings(group_key).get(
            "gangs") or {})

    def gangs_frozen(self) -> str | None:
        try:
            self.gangs.all_gangs()
        except GangRegistryError as exc:
            return str(exc)
        return None

    def _owner_person(self, group_key: str
                      ) -> tuple[str | None, str | None]:
        """(person key, local id) for the owner on this group's
        platform, from the §19 owner record — (None, None)
        when the record doesn't know the owner there."""
        reg = self.owner_registry
        if reg is None:
            return None, None
        segment = str(group_key).split(":", 1)[0]
        for plat in (segment, "telegram"):
            try:
                ids = reg.identities(plat) or []
            except Exception:  # noqa: BLE001 - no record, no
                ids = []       # owner — seeding simply defers
            if ids:
                return _person_key(segment, str(ids[0])), str(
                    ids[0])
        return None, None

    def _person_of(self, group_key: str, user_id: str) -> str:
        """The §36 person key for a local identity — resolved
        through the registry's bindings first (a claim-code
        bound identity is one person on every platform)."""
        try:
            return self.gangs.person_for(group_key, user_id)
        except GangRegistryError:
            return _person_key(
                str(group_key).split(":", 1)[0], str(user_id))

    def gangs_touch(self, group_key: str, *,
                    now: float | None = None) -> None:
        """First contact with a claimed group (§36): the
        flagship chapter seeds lazily — once per group per
        process, on the cycle's claimed-room beat.  Unclaimed
        rooms are never touched (no state accrues there, §33).
        A frozen registry freezes the gangs layer only; the
        street keeps its own discipline."""
        if group_key in self._gangs_touched:
            return
        if not self.claim_state(group_key)["claimed"]:
            return
        self._gangs_touched.add(group_key)
        now = now if now is not None else time.time()
        person, local = self._owner_person(group_key)
        try:
            out = self.gangs.ensure_flagship(
                group_key, owner_person=person,
                owner_local_id=local, now=now,
                cfg=self.gangs_cfg(group_key))
        except (GangRegistryError, GroupConflictError):
            return
        if out.get("ok") and out.get("created"):
            self._audit(group_key, "flagship_seeded",
                        user_id=str(local or ""),
                        gang=str(out["gang"].get("name") or ""))

    def _gang_register(self, group_key: str, gang: dict[str, Any],
                       *, now: float) -> dict[str, Any] | None:
        """Register a chapter gang into the §36 registry on
        first touch (best-effort: a name held globally leaves
        the gang local-only, never blocks its street life)."""
        try:
            return self.gangs.ensure_registered(
                group_key, gang, now=now)
        except (GangRegistryError, GroupConflictError):
            return None

    def conflict_cfg(self, group_key: str) -> dict[str, Any]:
        return dict(self.store.settings(group_key)["conflict"])

    def _conflict_mode(self, group_key: str) -> str:
        mode = str(self.conflict_cfg(group_key).get("mode")
                   or "off")
        return mode if mode in ("off", "crime", "warfare") \
            else "off"

    def conflict_frozen(self, group_key: str) -> str | None:
        """Freeze reason when the conflict state is unreadable
        (the economy_frozen discipline, probed once per group
        per day): an unreadable gang/war state is never read
        as empty street."""
        day = time.strftime("%Y-%m-%d", time.gmtime())
        probe = getattr(self, "_conflict_probe", None)
        if probe is None:
            probe = self._conflict_probe = {}
        cached = probe.get(group_key)
        if cached and cached[0] == day:
            return cached[1]
        reason: str | None = None
        try:
            health = self.conflict.health(group_key)
            if not health.get("ok"):
                reason = str(health.get("error") or "unreadable")
        except GroupConflictError as exc:
            reason = str(exc)
        if reason:
            self._audit(group_key, "conflict_frozen",
                        reason=reason)
        probe[group_key] = (day, reason)
        return reason

    def _conflict_gate(self, group_key: str, need: str = "crime"
                       ) -> dict[str, Any] | None:
        """None when the conflict layer runs at the needed
        level here, else a refusal dict.  Levels: crime verbs,
        heists, gangs and drops need ``crime``; wars and
        territory need ``warfare`` (§23's per-group mode)."""
        if not self.store.settings(group_key)["economy_enabled"]:
            return {"ok": False, "reason": "off"}
        if self.conflict_frozen(group_key) is not None:
            return {"ok": False, "reason": "frozen"}
        rank = {"off": 0, "crime": 1, "warfare": 2}
        mode = self._conflict_mode(group_key)
        if rank[mode] < rank[need]:
            return {"ok": False, "reason": "mode", "mode": mode,
                    "need": need}
        return None

    # ── gear ──

    def gear_buy(self, group_key: str, user_id: str, item_id: str,
                 *, is_admin: bool = False,
                 now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "crime")
        if gate:
            return gate
        out = self.conflict.buy_gear(group_key, user_id, item_id,
                                     now=now, is_admin=is_admin)
        if out.get("ok"):
            self._audit(group_key, "gear_buy", user_id=user_id,
                        item=item_id)
            self.store.save()
        return out

    def gear_install(self, group_key: str, user_id: str,
                     item_id: str, scope: str, *,
                     can_shared: bool = False,
                     now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "crime")
        if gate:
            return gate
        if scope.startswith("vault:") and scope != (
                f"vault:{user_id}") and not can_shared:
            return {"ok": False, "reason": "scope"}
        if not scope.startswith("vault:") and not can_shared:
            return {"ok": False, "reason": "scope"}
        out = self.conflict.install_defense(
            group_key, user_id, item_id, scope, now=now)
        if out.get("ok"):
            self._audit(group_key, "defense_install",
                        user_id=user_id, item=item_id, scope=scope)
            self.store.save()
        return out

    def gear_use(self, group_key: str, user_id: str, item_id: str,
                 *, now: float | None = None) -> dict[str, Any]:
        """Use a held support item: a medkit discharges you
        from hospital; a repair kit restores your most-worn
        tool.  (Bribe envelopes are spent by /bribe itself.)"""
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "crime")
        if gate:
            return gate
        if item_id == "medkit":
            if not self.hospital_state(group_key, user_id,
                                       now=now):
                return {"ok": False, "reason": "not_hospital"}
            if not self.conflict.use_consumable(
                    group_key, user_id, "medkit", now=now):
                return {"ok": False, "reason": "not_owned"}
            self.hospital_clear(group_key, user_id)
            self._audit(group_key, "medkit_used",
                        user_id=user_id)
            self.store.save()
            return {"ok": True, "item": "medkit"}
        if item_id == "repair_kit":
            return self.conflict.repair_item(group_key, user_id,
                                             now=now)
        return {"ok": False, "reason": "not_usable"}

    def loadout_view(self, group_key: str, user_id: str, *,
                     now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        organ = self.conflict
        state = organ._load_state(group_key)
        stars, index, _ = organ.wanted_stars(
            group_key, user_id, state=state, now=now)
        return {"items": organ.loadout(group_key, user_id,
                                       state=state),
                "power": organ.member_power(
                    group_key, user_id, state=state, now=now),
                "rep": int((state.get("rep") or {}).get(
                    user_id, 0)),
                "stars": stars, "index": index,
                "defenses": {scope: items for scope, items in
                             (state.get("defenses") or {}).items()
                             if scope == f"vault:{user_id}"}}

    # ── crime verbs + heists ──

    def _crime_ready(self, group_key: str, actor: str, *,
                     now: float) -> dict[str, Any] | None:
        if self.jail_state(group_key, actor, now=now):
            return {"ok": False, "reason": "jailed",
                    "jail": self.jail_state(group_key, actor,
                                            now=now)}
        if self.hospital_state(group_key, actor, now=now):
            return {"ok": False, "reason": "hospital",
                    "hospital": self.hospital_state(
                        group_key, actor, now=now)}
        cfg = self.conflict_cfg(group_key)
        cd = self._extras(group_key).setdefault("crime_cd", {})
        last = float((cd.get(actor) or {}).get("last") or 0)
        wait = int(cfg.get("crime_cooldown_s") or 0) - (now - last)
        if wait > 0:
            return {"ok": False, "reason": "cooldown",
                    "wait_s": int(wait)}
        return None

    def crime_verb(self, group_key: str, actor: str, target: str,
                   verb: str, *, rng: Any,
                   now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "crime")
        if gate:
            return gate
        if actor == target:
            return {"ok": False, "reason": "self"}
        ready = self._crime_ready(group_key, actor, now=now)
        if ready:
            return ready
        if self.jail_state(group_key, target, now=now):
            return {"ok": False, "reason": "target_jailed"}
        organ = self.conflict
        out = organ.hit_verb(
            group_key, actor, target, verb, rng=rng, now=now,
            shielded=self._shielded(group_key, target, now=now))
        if not out.get("ok"):
            return out
        cd = self._extras(group_key).setdefault("crime_cd", {})
        cd[actor] = {"last": now}
        if out.get("success"):
            if int(out.get("hospital_min") or 0):
                bill = 40 if verb == "kill" else 70
                self.hospital_member(
                    group_key, target, int(out["hospital_min"]),
                    bill=bill, by=actor, now=now)
            claimed = organ.claim_bounties(
                group_key, actor, target, now=now, via=verb)
            out["bounty"] = int(claimed.get("amount") or 0)
        elif int(out.get("jail_min") or 0):
            self.jail_member(group_key, actor,
                             int(out["jail_min"]),
                             reason=f"failed {verb}", now=now)
            out["jailed"] = True
        self._audit(group_key, f"crime_{verb}", user_id=actor,
                    target=target,
                    success=bool(out.get("success")),
                    take=int(out.get("take") or 0))
        self.store.save()
        return out

    def crime_sabotage(self, group_key: str, actor: str,
                       target: str, *, rng: Any,
                       now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "crime")
        if gate:
            return gate
        if actor == target:
            return {"ok": False, "reason": "self"}
        ready = self._crime_ready(group_key, actor, now=now)
        if ready:
            return ready
        out = self.conflict.sabotage(
            group_key, actor, target, rng=rng, now=now,
            shielded=self._shielded(group_key, target, now=now))
        if not out.get("ok"):
            return out
        cd = self._extras(group_key).setdefault("crime_cd", {})
        cd[actor] = {"last": now}
        if not out.get("success"):
            self.jail_member(group_key, actor, 20,
                             reason="caught sabotaging", now=now)
            out["jailed"] = True
        self._audit(group_key, "crime_sabotage", user_id=actor,
                    target=target,
                    success=bool(out.get("success")),
                    bill=int(out.get("bill") or 0))
        self.store.save()
        return out

    def heist_start(self, group_key: str, leader: str,
                    target_kind: str, target_id: str, *,
                    now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "crime")
        if gate:
            return gate
        ready = self._crime_ready(group_key, leader, now=now)
        if ready:
            return ready
        cfg = self.conflict_cfg(group_key)
        cd = self._extras(group_key).setdefault("heist_cd", {})
        last = float((cd.get(leader) or {}).get("last") or 0)
        wait = int(cfg.get("heist_cooldown_s") or 0) - (now - last)
        if wait > 0:
            return {"ok": False, "reason": "cooldown",
                    "wait_s": int(wait)}
        if target_kind == "bank" and target_id not in DISTRICTS:
            return {"ok": False, "reason": "district"}
        if target_kind == "vault":
            if target_id == leader:
                return {"ok": False, "reason": "self"}
            if not self.member(group_key, target_id):
                return {"ok": False, "reason": "target"}
        gang = self.conflict.gang_of(group_key, leader)
        if target_kind in ("treasury", "bank") and (
                gang is None or len(gang.get("members") or [])
                < 5):
            return {"ok": False, "reason": "tier",
                    "need": "a gang of 5+ runs the big scores"}
        out = self.conflict.start_heist(
            group_key, leader, target_kind, target_id, now=now)
        if out.get("ok"):
            self._audit(group_key, "heist_start", user_id=leader,
                        target=f"{target_kind}:{target_id}")
            self.store.save()
        return out

    def heist_join(self, group_key: str, member: str, *,
                   now: float | None = None) -> dict[str, Any]:
        """Join the open heist lobby (one at a time per
        group — the lobby is found, never named by id)."""
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "crime")
        if gate:
            return gate
        ready = self._crime_ready(group_key, member, now=now)
        if ready:
            return ready
        organ = self.conflict
        state = organ._load_state(group_key)
        for lobby in (state.get("heists") or {}).values():
            if lobby.get("status") == "open":
                return organ.join_heist(group_key, member,
                                        str(lobby["id"]), now=now)
        return {"ok": False, "reason": "not_found"}

    def heist_go(self, group_key: str, leader: str, *, rng: Any,
                 now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "crime")
        if gate:
            return gate
        organ = self.conflict
        state = organ._load_state(group_key)
        lobby = next((h for h in (state.get("heists") or {})
                      .values()
                      if h.get("status") == "open"
                      and h.get("leader") == leader), None)
        if lobby is None:
            return {"ok": False, "reason": "not_found"}
        available = self.economy.treasury_balance(group_key) \
            - self.reserved_treasury(group_key)
        out = organ.resolve_heist(
            group_key, str(lobby["id"]), rng=rng, now=now,
            available_treasury=max(0, available))
        if not out.get("ok"):
            return out
        cd = self._extras(group_key).setdefault("heist_cd", {})
        for member in out.get("crew") or []:
            cd[member] = {"last": now}
        if out.get("outcome") == "bust":
            for member in out.get("crew") or []:
                self.jail_member(group_key, member, 45,
                                 reason="heist bust", now=now)
            out["jailed"] = list(out.get("crew") or [])
        self._audit(group_key, "heist_done", user_id=leader,
                    outcome=out.get("outcome"),
                    take=int(out.get("take") or 0))
        self.store.save()
        return out

    def heist_open(self, group_key: str) -> dict[str, Any] | None:
        state = self.conflict._load_state(group_key)
        for lobby in (state.get("heists") or {}).values():
            if lobby.get("status") == "open":
                return lobby
        return None

    # ── gangs ──

    def gang_create(self, group_key: str, user_id: str,
                    name: str, lifestyle: str, *,
                    now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "crime")
        if gate:
            return gate
        out = self.conflict.create_gang(group_key, user_id, name,
                                        lifestyle, now=now)
        if out.get("ok"):
            try:
                self.war_book.register_gang(
                    group_key, str(self.store.path), out["gang"])
            except GroupConflictError:
                pass  # index is a directory, not the gang —
                # a broken book never blocks the street forming
            # §36: every gang is born into the global registry
            # (a record with one chapter); a globally held name
            # leaves it local-only, never blocked.
            self._gang_register(group_key, out["gang"], now=now)
            self._audit(group_key, "gang_create", user_id=user_id,
                        gang=str(out["gang"].get("name") or ""))
            self.store.save()
        return out

    def gang_join(self, group_key: str, user_id: str, name: str,
                  *, now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "crime")
        if gate:
            return gate
        # §36: a registered gang's joins run the loyalty rules
        # (one cross-group + one local membership) in the
        # registry, which performs the local join itself.
        try:
            gang = self.conflict.gang_by_name(group_key, name)
        except GroupConflictError:
            gang = None
        if gang is not None:
            record = self._gang_register(group_key, gang,
                                         now=now)
            if record is not None:
                member = self.store.group(group_key)["members"] \
                    .get(user_id) or {}
                try:
                    out = self.gangs.join_chapter(
                        str(record["id"]), group_key,
                        person=self._person_of(group_key, user_id),
                        local_id=user_id,
                        name=str(member.get("name") or ""),
                        username=str(member.get("username")
                                     or ""),
                        now=now, cfg=self.gangs_cfg(group_key))
                except (GangRegistryError, GroupConflictError):
                    out = {"ok": False, "reason": "frozen"}
                if out.get("ok"):
                    self.store.save()
                    return {"ok": True, "gang": out["chapter"]}
                return out
        out = self.conflict.join_gang(group_key, user_id, name,
                                      now=now)
        if out.get("ok"):
            self.store.save()
        return out

    def gang_leave(self, group_key: str, user_id: str, *,
                   now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        gang = None
        try:
            gang = self.conflict.gang_of(group_key, user_id)
        except GroupConflictError:
            gang = None
        if gang is not None and gang.get("global"):
            try:
                out = self.gangs.leave_chapter(
                    str(gang["global"]), group_key,
                    person=self._person_of(group_key, user_id),
                    local_id=user_id, now=now)
            except (GangRegistryError, GroupConflictError):
                out = {"ok": False, "reason": "frozen"}
            if out.get("ok"):
                if out.get("dissolved"):
                    try:
                        self.war_book.unregister_gang(
                            group_key, str(gang.get("id") or ""))
                    except GroupConflictError:
                        pass
                self.store.save()
                return {"ok": True, "dissolved": bool(
                    out.get("dissolved")), "gang": gang}
            return out
        out = self.conflict.leave_gang(group_key, user_id,
                                       now=now)
        if out.get("ok"):
            if out.get("dissolved"):
                try:
                    self.war_book.unregister_gang(
                        group_key, str(out["gang"].get("id")
                                       or ""))
                except GroupConflictError:
                    pass
            self.store.save()
        return out

    def gang_card(self, group_key: str, user_id: str, *,
                  now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        organ = self.conflict
        state = organ._load_state(group_key)
        gang = organ.gang_of(group_key, user_id, state=state)
        if gang is None:
            return {"ok": False, "reason": "no_gang"}
        from ...community.group_conflict import gang_tier
        strength = organ.gang_strength(group_key, gang,
                                       state=state, now=now)
        held = [d for d, h in state["territory"].items()
                if h.get("gang") == gang.get("id")]
        view = {"ok": True, "gang": gang,
                "tier": gang_tier(len(gang.get("members") or [])),
                "strength": strength, "territory": held,
                "power": organ.power_gang(
                    group_key, state=state, now=now) is gang
                or (organ.power_gang(group_key, state=state,
                                     now=now) or {}).get(
                    "id") == gang.get("id")}
        # §36: the global standing rides the card when the
        # gang is registered — tier/prestige over the whole
        # roster, chapters, HQ, and the caller's global rank.
        record = self._gang_register(group_key, gang, now=now)
        if record is not None:
            gview = self.gangs.global_view(
                record, cfg=self.gangs_cfg(group_key))
            person = self._person_of(group_key, user_id)
            entry = (record.get("roster") or {}).get(person) \
                or {}
            gview["your_role"] = str(entry.get("role")
                                     or "member")
            gview["your_slot"] = str(gview.get("class") or "")
            view["global"] = gview
        return view

    def gang_list(self, group_key: str, *,
                  now: float | None = None) -> list[dict[str, Any]]:
        now = now if now is not None else time.time()
        organ = self.conflict
        state = organ._load_state(group_key)
        out = []
        for gang in (state.get("gangs") or {}).values():
            out.append({"gang": gang, "strength":
                        organ.gang_strength(
                            group_key, gang, state=state,
                            now=now)["total"]})
        out.sort(key=lambda r: -r["strength"])
        return out

    def gang_power_actor(self, group_key: str, user_id: str, *,
                         now: float | None = None
                         ) -> dict[str, Any] | None:
        """The power gang when this member may drive it
        (governance capture, §23) — leader/officers only."""
        now = now if now is not None else time.time()
        if self._conflict_mode(group_key) != "warfare":
            return None
        try:
            return self.conflict.power_actor(group_key, user_id,
                                             now=now)
        except GroupConflictError:
            return None

    def territory_map(self, group_key: str) -> dict[str, str]:
        try:
            return self.conflict.territory_map(group_key)
        except GroupConflictError:
            return {}

    # ── the police + bounties ──

    def police_board(self, group_key: str, user_id: str, *,
                     now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        organ = self.conflict
        state = organ._load_state(group_key)
        wanted = []
        members = set((state.get("rep") or {}).keys())
        for event in state.get("journal") or []:
            if event.get("kind") == "crime":
                members.add(str(event.get("actor") or ""))
        for member in sorted(m for m in members if m):
            stars, index, inputs = organ.wanted_stars(
                group_key, member, state=state, now=now)
            if stars > 0 or index >= 6.0:
                wanted.append({"member": member, "stars": stars,
                               "index": index,
                               "inputs": inputs[-3:]})
        wanted.sort(key=lambda r: -r["index"])
        stars, index, inputs = organ.wanted_stars(
            group_key, user_id, state=state, now=now)
        return {"wanted": wanted[:10],
                "bounties": organ.open_bounties(group_key,
                                                state=state),
                "me": {"stars": stars, "index": index,
                       "inputs": inputs}}

    def police_bribe(self, group_key: str, user_id: str, *,
                     rng: Any, now: float | None = None,
                     use_envelope: bool = False
                     ) -> dict[str, Any]:
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "crime")
        if gate:
            return gate
        out = self.conflict.bribe(group_key, user_id, rng=rng,
                                  now=now,
                                  use_envelope=use_envelope)
        if out.get("ok") and int(out.get("jail_min") or 0):
            self.jail_member(group_key, user_id,
                             int(out["jail_min"]),
                             reason="bribe went wrong", now=now)
            out["jailed"] = True
        if out.get("ok"):
            self._audit(group_key, "bribe", user_id=user_id,
                        success=bool(out.get("success")))
            self.store.save()
        return out

    def bounty_post(self, group_key: str, user_id: str,
                    target: str, amount: int, *,
                    now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "crime")
        if gate:
            return gate
        if user_id == target:
            return {"ok": False, "reason": "self"}
        out = self.conflict.post_bounty(group_key, user_id,
                                        target, amount, now=now)
        if out.get("ok"):
            self._audit(group_key, "bounty_post",
                        user_id=user_id, target=target,
                        amount=int(amount))
            self.store.save()
        return out

    def bounty_hunt(self, group_key: str, hunter: str,
                    target: str, *, rng: Any,
                    now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "crime")
        if gate:
            return gate
        if hunter == target:
            return {"ok": False, "reason": "self"}
        ready = self._crime_ready(group_key, hunter, now=now)
        if ready:
            return ready
        out = self.conflict.hunt(group_key, hunter, target,
                                 rng=rng, now=now)
        if not out.get("ok"):
            return out
        if out.get("success"):
            self.jail_member(group_key, target, 30,
                             reason="brought in by a bounty "
                                    "hunter", now=now)
            out["jailed_target"] = True
        else:
            fine = int(out.get("fine_due") or 0)
            econ = self.economy
            if fine and econ.balance(group_key, hunter) >= fine:
                econ._move(group_key, hunter, "", fine, "fine",
                           "failed bounty hunt", now)
                out["fine"] = fine
            else:
                self.jail_member(group_key, hunter, 10,
                                 reason="hunt went wrong",
                                 now=now)
                out["jailed"] = True
        self._audit(group_key, "bounty_hunt", user_id=hunter,
                    target=target,
                    success=bool(out.get("success")))
        self.store.save()
        return out

    # ── wars (in-group) ──

    def war_declare(self, group_key: str, user_id: str,
                    target_name: str, *,
                    now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "warfare")
        if gate:
            return gate
        out = self.conflict.declare_war(
            group_key, user_id, target_name, now=now,
            cfg=self.conflict_cfg(group_key))
        if out.get("ok"):
            self._audit(group_key, "war_declare",
                        user_id=user_id,
                        target=str(out["theirs"].get("name")
                                   or ""))
            self.store.save()
        return out

    def war_answer(self, group_key: str, user_id: str,
                   accept: bool, *,
                   now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "warfare")
        if gate:
            return gate
        organ = self.conflict
        gang = organ.gang_of(group_key, user_id)
        if gang is None:
            return {"ok": False, "reason": "no_gang"}
        state = organ._load_state(group_key)
        war = next((w for w in (state.get("wars") or {}).values()
                    if w.get("status") == "declared"
                    and w.get("b") == gang.get("id")), None)
        if war is None:
            return {"ok": False, "reason": "no_war"}
        out = organ.answer_war(group_key, user_id,
                               str(war["id"]), accept, now=now,
                               cfg=self.conflict_cfg(group_key))
        if out.get("ok"):
            self.store.save()
        return out

    def war_join(self, group_key: str, user_id: str, *,
                 now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "warfare")
        if gate:
            return gate
        out = self.conflict.war_join(group_key, user_id, now=now)
        if out.get("ok"):
            self.store.save()
        return out

    def war_status(self, group_key: str, user_id: str, *,
                   now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        organ = self.conflict
        state = organ._load_state(group_key)
        gang = organ.gang_of(group_key, user_id, state=state)
        if gang is None:
            return {"ok": False, "reason": "no_gang"}
        war = organ.war_for_gang(group_key, gang["id"],
                                 state=state)
        inter = [w for w in self.war_book.wars_for_group(
            group_key) if w.get("status") in (
                "declared", "active")
            and ((w.get("a") or {}).get("gang") == gang.get("id")
                 or (w.get("b") or {}).get("gang")
                 == gang.get("id"))]
        return {"ok": True, "war": war,
                "inter": inter[0] if inter else None,
                "gang": gang}

    def war_spoils(self, group_key: str, user_id: str,
                   district: str, *,
                   now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "warfare")
        if gate:
            return gate
        out = self.conflict.war_claim_spoils(
            group_key, user_id, district, now=now)
        if out.get("ok"):
            self._audit(group_key, "territory_spoils",
                        user_id=user_id, district=district)
            self.store.save()
        return out

    def territory_claim(self, group_key: str, user_id: str,
                        district: str, *,
                        now: float | None = None
                        ) -> dict[str, Any]:
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "warfare")
        if gate:
            return gate
        out = self.conflict.claim_territory(
            group_key, user_id, district, now=now)
        if out.get("ok"):
            self._audit(group_key, "territory_claim",
                        user_id=user_id, district=district)
            self.store.save()
        return out

    # ── wars (inter-group, through the book) ──

    def _conflict_for_store(self, store_path: str
                            ) -> GroupConflict:
        from ...community.group_economy import GroupEconomy
        econ = GroupEconomy(
            data_dir=Path(store_path).parent / "group_economy")
        return GroupConflict(econ)

    def _war_side(self, group_key: str, gang: dict[str, Any], *,
                  now: float) -> dict[str, Any]:
        from ...community.group_conflict import gear_power
        organ = self.conflict
        state = organ._load_state(group_key)
        return {"group": group_key,
                "store": str(self.store.path),
                "gang": str(gang.get("id") or ""),
                "gang_name": str(gang.get("name") or ""),
                "lifestyle": str(gang.get("lifestyle")
                                 or "street"),
                "strength": organ.gang_strength(
                    group_key, gang, state=state,
                    now=now)["total"],
                "armory_power": gear_power(
                    list(gang.get("armory") or []), "war",
                    now=now),
                "leader_power": organ.member_power(
                    group_key, str(gang.get("leader") or ""),
                    state=state, now=now)}

    def war_declare_inter(self, group_key: str, user_id: str,
                          target_name: str, *,
                          now: float | None = None
                          ) -> dict[str, Any]:
        """Declare on a gang in ANOTHER claimed group: both
        gangs must be gang-tier or better, the escrow leaves
        this gang's treasury up front, and the war lives in
        the shared book until the other side answers."""
        from ...community.group_conflict import gang_tier
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "warfare")
        if gate:
            return gate
        organ = self.conflict
        mine = organ.gang_of(group_key, user_id)
        if mine is None:
            return {"ok": False, "reason": "no_gang"}
        if gang_tier(len(mine.get("members") or [])) == "crew":
            return {"ok": False, "reason": "tier"}
        try:
            self.war_book.register_gang(
                group_key, str(self.store.path), mine)
            found = [f for f in self.war_book.find_gangs(
                target_name) if f["group"] != group_key]
        except GroupConflictError:
            return {"ok": False, "reason": "book"}
        if not found:
            return {"ok": False, "reason": "not_found"}
        target = found[0]
        try:
            other = self._conflict_for_store(
                str(target["store"]))
            theirs = other.gang_by_name(str(target["group"]),
                                        target_name)
        except GroupConflictError:
            return {"ok": False, "reason": "frozen"}
        if theirs is None:
            return {"ok": False, "reason": "not_found"}
        if gang_tier(len(theirs.get("members") or [])) == "crew":
            return {"ok": False, "reason": "tier_target"}
        for war in self.war_book.wars_for_group(group_key):
            if war.get("status") in ("declared", "active"):
                return {"ok": False, "reason": "busy"}
        cfg = self.conflict_cfg(group_key)
        escrow = int(cfg.get("war_escrow") or 0)
        side_a = self._war_side(group_key, mine, now=now)
        other_surface_note = str(target["group"])
        side_b = {"group": other_surface_note,
                  "store": str(target["store"]),
                  "gang": str(theirs.get("id") or ""),
                  "gang_name": str(theirs.get("name") or ""),
                  "lifestyle": str(theirs.get("lifestyle")
                                   or "street"),
                  "strength": 0, "armory_power": 0,
                  "leader_power": 10}
        try:
            other_state = other._load_state(other_surface_note)
            from ...community.group_conflict import gear_power
            side_b["strength"] = other.gang_strength(
                other_surface_note, theirs, state=other_state,
                now=now)["total"]
            side_b["armory_power"] = gear_power(
                list(theirs.get("armory") or []), "war", now=now)
            side_b["leader_power"] = other.member_power(
                other_surface_note,
                str(theirs.get("leader") or ""),
                state=other_state, now=now)
        except GroupConflictError:
            return {"ok": False, "reason": "frozen"}
        war = self.war_book.create_war(side_a, side_b,
                                       escrow=escrow, now=now,
                                       cfg=cfg)
        paid = organ.escrow_for_inter_war(
            group_key, user_id, str(war["id"]), escrow, now=now)
        if not paid.get("ok"):
            war["status"] = "void"
            self.war_book.update(war)
            return paid
        # §36: pair the two gangs in the registry so dual
        # holders can declare a side for this war.
        rec_mine = self._gang_register(group_key, mine, now=now)
        rec_theirs = None
        if theirs.get("global"):
            try:
                rec_theirs = self.gangs.gang(
                    str(theirs["global"]))
            except GangRegistryError:
                rec_theirs = None
        if rec_mine is not None and rec_theirs is not None:
            try:
                self.gangs.note_war(str(war["id"]),
                                    str(rec_mine["id"]),
                                    str(rec_theirs["id"]),
                                    now=now)
            except GangRegistryError:
                pass
        self._audit(group_key, "war_inter_declare",
                    user_id=user_id,
                    target=str(theirs.get("name") or ""))
        self.store.save()
        return {"ok": True, "war": war, "mine": mine,
                "theirs": theirs}

    def war_answer_inter(self, group_key: str, user_id: str,
                         accept: bool, *,
                         now: float | None = None
                         ) -> dict[str, Any]:
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "warfare")
        if gate:
            return gate
        organ = self.conflict
        gang = organ.gang_of(group_key, user_id)
        if gang is None:
            return {"ok": False, "reason": "no_gang"}
        war = next((w for w in self.war_book.wars_for_group(
            group_key) if w.get("status") == "declared"
            and (w.get("b") or {}).get("gang") == gang.get("id")),
            None)
        if war is None:
            return {"ok": False, "reason": "no_war"}
        cfg = self.conflict_cfg(group_key)
        if accept:
            paid = organ.escrow_for_inter_war(
                group_key, user_id, str(war["id"]),
                int(war.get("escrow") or 0), now=now)
            if not paid.get("ok"):
                return paid
            war["status"] = "active"
            war["next_round_at"] = now + float(
                cfg.get("war_round_s") or 3600)
        else:
            war["status"] = "declined"
        self.war_book.update(war)
        self.store.save()
        return {"ok": True, "war": war}

    def war_join_inter(self, group_key: str, user_id: str, *,
                       now: float | None = None
                       ) -> dict[str, Any]:
        """Opt into your gang's inter-group war: your CURRENT
        power is snapshotted into the book (the book is the
        only place both sides' strength can meet)."""
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "warfare")
        if gate:
            return gate
        organ = self.conflict
        gang = organ.gang_of(group_key, user_id)
        if gang is None:
            return {"ok": False, "reason": "no_gang"}
        war = next((w for w in self.war_book.wars_for_group(
            group_key) if w.get("status") == "active"
            and ((w.get("a") or {}).get("gang") == gang.get("id")
                 or (w.get("b") or {}).get("gang")
                 == gang.get("id"))), None)
        if war is None:
            return {"ok": False, "reason": "no_war"}
        side = "a" if (war.get("a") or {}).get(
            "gang") == gang.get("id") else "b"
        parts = dict((war.get("participants") or {}).get(
            side) or {})
        if user_id in parts:
            return {"ok": False, "reason": "already"}
        # §36 loyalty: a dual holder fights only for the side
        # they declared for this war — undeclared, they sit
        # it out; declared for the other side, they're refused.
        if gang.get("global"):
            try:
                check = self.gangs.war_join_check(
                    person=self._person_of(group_key, user_id),
                    war_id=str(war["id"]),
                    global_id=str(gang["global"]))
            except GangRegistryError:
                check = {"ok": True}
            if not check.get("ok"):
                return check
        parts[user_id] = organ.member_power(group_key, user_id,
                                            now=now)
        war.setdefault("participants", {})[side] = parts
        self.war_book.update(war)
        return {"ok": True, "war": war, "side": side}

    # ── §36: settled-war prestige + the HQ raid ──────────────

    def _gangs_note_settled(self, group_key: str,
                            war: dict[str, Any], *,
                            now: float) -> None:
        """Prestige for a settled inter-group war / HQ raid —
        the registry awards each war exactly once, whichever
        side's cycle settles it first."""
        try:
            gangs = self.gangs
            cfg = self.gangs_cfg(group_key)
            winner = str(war.get("winner") or "")
            if war.get("kind") == "hq_raid":
                a = war.get("a") or {}
                b = war.get("b") or {}
                rec_a = self.gangs.gang(str(
                    war.get("attacker_global") or ""))
                rec_b = self.gangs.gang(str(
                    war.get("defender_global") or ""))
                if rec_a is not None and rec_b is not None:
                    gangs.note_hq_raid_outcome(
                        str(war["id"]),
                        attacker_gid=str(rec_a["id"]),
                        defender_gid=str(rec_b["id"]),
                        attacker_won=winner == "a",
                        now=now, cfg=cfg)
                return
            side = (war.get(winner) or {}) if winner else {}
            other_group = str(side.get("group") or "")
            if not other_group:
                return
            organ = self.conflict if other_group == group_key \
                else self._conflict_for_store(str(
                    side.get("store") or self.store.path))
            gang = (organ._load_state(other_group).get(
                "gangs") or {}).get(str(side.get("gang") or ""))
            if gang is not None and gang.get("global"):
                gangs.note_inter_war_win(
                    str(war["id"]), str(gang["global"]),
                    now=now, cfg=cfg)
        except (GangRegistryError, GroupConflictError):
            pass  # prestige is garnish on a settled war; the
            # settlement itself already landed

    def war_raid_hq(self, group_key: str, user_id: str,
                    target_name: str, *,
                    now: float | None = None
                    ) -> dict[str, Any]:
        """Declare an HQ raid (§36): an inter-group war whose
        target is a gang's HQ chapter — the crown jewel.
        Bigger escrow, spoils drawn from the fortress's actual
        vault (blunted by its tier), prestige on top.  It runs
        through the SAME war book, rounds and settlement shape
        as any inter-group war — kind ``hq_raid``, settled by
        settle_hq_raid."""
        from ...community.group_conflict import gang_tier
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "warfare")
        if gate:
            return gate
        organ = self.conflict
        mine = organ.gang_of(group_key, user_id)
        if mine is None:
            return {"ok": False, "reason": "no_gang"}
        if gang_tier(len(mine.get("members") or [])) == "crew":
            return {"ok": False, "reason": "tier"}
        try:
            record = self.gangs.gang_by_name(target_name)
        except GangRegistryError:
            return {"ok": False, "reason": "frozen"}
        if record is None:
            return {"ok": False, "reason": "not_found"}
        rec_mine = self._gang_register(group_key, mine, now=now)
        if rec_mine is not None \
                and rec_mine["id"] == record["id"]:
            return {"ok": False, "reason": "self"}
        status = self.gangs.hq_status(str(record["id"]))
        hq = status.get("hq")
        if not status.get("ok") or hq is None:
            return {"ok": False, "reason": "no_hq"}
        hq_group = str(hq["group"])
        if hq_group == group_key:
            return {"ok": False, "reason": "local"}
        for war in self.war_book.wars_for_group(group_key):
            if war.get("status") in ("declared", "active"):
                return {"ok": False, "reason": "busy"}
        # the defender snapshot comes from the HQ chapter in
        # its own group (through that group's store, the
        # _war_side discipline), blunted by the fortress tier
        found = [f for f in self.war_book.find_gangs(
            str(record["name"])) if f["group"] == hq_group]
        store_path = str(found[0]["store"]) if found \
            else str(self.store.path)
        try:
            other = self._conflict_for_store(store_path)
            other_state = other._load_state(hq_group)
        except GroupConflictError:
            return {"ok": False, "reason": "frozen"}
        theirs = (other_state.get("gangs") or {}).get(str(
            (record.get("chapters") or {}).get(
                hq_group, {}).get("gang") or ""))
        if theirs is None:
            return {"ok": False, "reason": "gone"}
        cfg = self.conflict_cfg(group_key)
        gcfg = self.gangs_cfg(group_key)
        raid_cfg = dict(gcfg.get("hq_raid") or {})
        escrow = int(cfg.get("war_escrow") or 0) * int(
            raid_cfg.get("escrow_mult") or 2)
        tier = int(hq["tier"])
        from ...community.group_conflict import gear_power
        strength_b = other.gang_strength(
            hq_group, theirs, state=other_state,
            now=now)["total"]
        blunt = 1.0 + (int(raid_cfg.get(
            "strength_per_tier_pct") or 10) * tier) / 100.0
        side_a = self._war_side(group_key, mine, now=now)
        side_b = {"group": hq_group, "store": store_path,
                  "gang": str(theirs.get("id") or ""),
                  "gang_name": str(theirs.get("name") or ""),
                  "lifestyle": str(theirs.get("lifestyle")
                                   or "street"),
                  "strength": int(strength_b * blunt),
                  "armory_power": gear_power(
                      list(theirs.get("armory") or []), "war",
                      now=now),
                  "leader_power": other.member_power(
                      hq_group, str(theirs.get("leader") or ""),
                      state=other_state, now=now),
                  "hq_tier": tier}
        vault = int(hq["vault"])
        pool = min(vault, int(raid_cfg.get("pool_cap") or 0))
        pool = int(pool * max(0.0, 1.0 - (
            int(raid_cfg.get("pool_per_tier_pct") or 15)
            * tier) / 100.0))
        war = self.war_book.create_war(side_a, side_b,
                                       escrow=escrow, now=now,
                                       cfg=cfg)
        war["kind"] = "hq_raid"
        war["raid_pool"] = pool
        war["hq_tier"] = tier
        war["hq_group"] = hq_group
        if rec_mine is not None:
            war["attacker_global"] = str(rec_mine["id"])
        war["defender_global"] = str(record["id"])
        self.war_book.update(war)
        paid = organ.escrow_for_inter_war(
            group_key, user_id, str(war["id"]), escrow, now=now)
        if not paid.get("ok"):
            war["status"] = "void"
            self.war_book.update(war)
            return paid
        if rec_mine is not None:
            try:
                self.gangs.note_war(str(war["id"]),
                                    str(rec_mine["id"]),
                                    str(record["id"]), now=now)
            except GangRegistryError:
                pass
        self._audit(group_key, "hq_raid_declare",
                    user_id=user_id,
                    target=str(record["name"]))
        self.store.save()
        return {"ok": True, "war": war, "mine": mine,
                "theirs": theirs, "hq": hq, "pool": pool}

    # ── §36: chapter / HQ / loyalty surface verbs ────────────

    def gang_found_chapter(self, group_key: str, user_id: str,
                           gang_name: str, *,
                           now: float | None = None
                           ) -> dict[str, Any]:
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "crime")
        if gate:
            return gate
        try:
            record = self.gangs.gang_by_name(gang_name)
            if record is None:
                return {"ok": False, "reason": "no_gang"}
            out = self.gangs.found_chapter(
                str(record["id"]), group_key,
                by_person=self._person_of(group_key, user_id),
                by_local_id=user_id, now=now,
                cfg=self.gangs_cfg(group_key))
        except (GangRegistryError, GroupConflictError):
            return {"ok": False, "reason": "frozen"}
        if out.get("ok"):
            try:
                self.war_book.register_gang(
                    group_key, str(self.store.path),
                    out["chapter"])
            except GroupConflictError:
                pass
            self._audit(group_key, "gang_chapter_founded",
                        user_id=user_id,
                        gang=str(record["name"]))
            self.store.save()
        return out

    def gang_register_here(self, group_key: str, user_id: str, *,
                           now: float | None = None
                           ) -> dict[str, Any]:
        """Register the caller's existing gang into the §36
        registry — registration, not re-founding: history,
        treasury and roster carry over as they stand."""
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "crime")
        if gate:
            return gate
        gang = self.conflict.gang_of(group_key, user_id)
        if gang is None:
            return {"ok": False, "reason": "no_gang"}
        if gang.get("global"):
            record = None
            try:
                record = self.gangs.gang(str(gang["global"]))
            except GangRegistryError:
                record = None
            if record is not None:
                return {"ok": True, "gang": record,
                        "already": True}
        try:
            out = self.gangs.register_local_gang(
                group_key, str(gang.get("id") or ""), now=now)
        except (GangRegistryError, GroupConflictError):
            return {"ok": False, "reason": "frozen"}
        if out.get("ok"):
            self._audit(group_key, "gang_registered",
                        user_id=user_id,
                        gang=str(out["gang"].get("name") or ""))
            self.store.save()
        return out

    def _callers_gang_record(self, group_key: str, user_id: str,
                             *, now: float
                             ) -> tuple[dict[str, Any] | None,
                                        dict[str, Any] | None]:
        """(global record, chapter gang) for the caller's gang
        in this group — registering on touch; (None, gang)
        when the gang runs local-only."""
        gang = self.conflict.gang_of(group_key, user_id)
        if gang is None:
            return None, None
        record = self._gang_register(group_key, gang, now=now)
        return record, gang

    def gang_hq_status(self, group_key: str, user_id: str, *,
                       now: float | None = None
                       ) -> dict[str, Any]:
        now = now if now is not None else time.time()
        record, _gang = self._callers_gang_record(
            group_key, user_id, now=now)
        if record is None:
            return {"ok": False, "reason": "no_gang"}
        try:
            return self.gangs.hq_status(str(record["id"]))
        except (GangRegistryError, GroupConflictError):
            return {"ok": False, "reason": "frozen"}

    def gang_hq_designate(self, group_key: str, user_id: str, *,
                          now: float | None = None
                          ) -> dict[str, Any]:
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "warfare")
        if gate:
            return gate
        record, _gang = self._callers_gang_record(
            group_key, user_id, now=now)
        if record is None:
            return {"ok": False, "reason": "no_gang"}
        try:
            out = self.gangs.hq_designate(
                str(record["id"]), group_key,
                by_person=self._person_of(group_key, user_id),
                now=now)
        except (GangRegistryError, GroupConflictError):
            return {"ok": False, "reason": "frozen"}
        if out.get("ok"):
            self._audit(group_key, "hq_designate",
                        user_id=user_id,
                        gang=str(record["name"]))
            self.store.save()
        return out

    def gang_hq_buy(self, group_key: str, user_id: str, *,
                    now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "warfare")
        if gate:
            return gate
        record, _gang = self._callers_gang_record(
            group_key, user_id, now=now)
        if record is None:
            return {"ok": False, "reason": "no_gang"}
        try:
            out = self.gangs.hq_buy_tier(
                str(record["id"]), by_local_id=user_id,
                now=now, cfg=self.gangs_cfg(group_key))
        except (GangRegistryError, GroupConflictError):
            return {"ok": False, "reason": "frozen"}
        if out.get("ok"):
            self._audit(group_key, "hq_tier", user_id=user_id,
                        gang=str(record["name"]),
                        tier=str(out.get("tier_name") or ""))
            self.store.save()
        return out

    def gang_hq_move(self, group_key: str, user_id: str, *,
                     now: float | None = None) -> dict[str, Any]:
        """Relocate the gang's HQ to the chapter the caller is
        standing in (cost + tier drop, registry rules)."""
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "warfare")
        if gate:
            return gate
        record, _gang = self._callers_gang_record(
            group_key, user_id, now=now)
        if record is None:
            return {"ok": False, "reason": "no_gang"}
        try:
            out = self.gangs.hq_relocate(
                str(record["id"]), group_key,
                by_person=self._person_of(group_key, user_id),
                now=now, cfg=self.gangs_cfg(group_key))
        except (GangRegistryError, GroupConflictError):
            return {"ok": False, "reason": "frozen"}
        if out.get("ok"):
            self._audit(group_key, "hq_relocate",
                        user_id=user_id,
                        gang=str(record["name"]))
            self.store.save()
        return out

    def gang_hq_vault(self, group_key: str, user_id: str,
                      verb: str, amount: int, *,
                      now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "crime")
        if gate:
            return gate
        record, _gang = self._callers_gang_record(
            group_key, user_id, now=now)
        if record is None:
            return {"ok": False, "reason": "no_gang"}
        if record.get("hq_group") != group_key:
            # the vault is a partition of the HQ chapter's
            # treasury — you stand at the fortress to use it.
            return {"ok": False, "reason": "hq_elsewhere",
                    "hq_group": record.get("hq_group")}
        try:
            if verb == "deposit":
                out = self.gangs.hq_vault_deposit(
                    str(record["id"]), local_id=user_id,
                    amount=int(amount), now=now)
            else:
                out = self.gangs.hq_vault_withdraw(
                    str(record["id"]), local_id=user_id,
                    amount=int(amount), now=now)
        except (GangRegistryError, GroupConflictError):
            return {"ok": False, "reason": "frozen"}
        if out.get("ok"):
            self.store.save()
        return out

    def gang_hq_armory(self, group_key: str, user_id: str,
                       verb: str, item_id: str = "", *,
                       now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "crime")
        if gate:
            return gate
        record, _gang = self._callers_gang_record(
            group_key, user_id, now=now)
        if record is None:
            return {"ok": False, "reason": "no_gang"}
        person = self._person_of(group_key, user_id)
        try:
            if verb == "list":
                status = self.gangs.hq_status(str(record["id"]))
                hq = status.get("hq")
                if hq is None:
                    return {"ok": False, "reason": "no_hq"}
                return {"ok": True, "armory": hq["armory"],
                        "hq": hq}
            if verb == "store":
                out = self.gangs.hq_armory_store(
                    str(record["id"]), group_key, person=person,
                    local_id=user_id, item_id=item_id, now=now)
            else:
                out = self.gangs.hq_armory_draw(
                    str(record["id"]), group_key, person=person,
                    local_id=user_id, item_id=item_id, now=now)
        except (GangRegistryError, GroupConflictError):
            return {"ok": False, "reason": "frozen"}
        if out.get("ok"):
            self.store.save()
        return out

    def gang_claim_code(self, group_key: str, user_id: str, *,
                        now: float | None = None
                        ) -> dict[str, Any]:
        """Mint a chapter claim code for the caller's gang."""
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "crime")
        if gate:
            return gate
        record, _gang = self._callers_gang_record(
            group_key, user_id, now=now)
        if record is None:
            return {"ok": False, "reason": "no_gang"}
        try:
            return self.gangs.mint_claim_code(
                str(record["id"]),
                person=self._person_of(group_key, user_id),
                now=now, cfg=self.gangs_cfg(group_key))
        except (GangRegistryError, GroupConflictError):
            return {"ok": False, "reason": "frozen"}

    def gang_redeem_code(self, group_key: str, user_id: str,
                         code: str, *,
                         now: float | None = None
                         ) -> dict[str, Any]:
        """Redeem a claim code here: binds this platform
        identity into the minter's roster + takes the seat."""
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "crime")
        if gate:
            return gate
        member = self.store.group(group_key)["members"].get(
            user_id) or {}
        try:
            out = self.gangs.redeem_claim_code(
                code, group_key, local_id=user_id,
                name=str(member.get("name") or ""),
                username=str(member.get("username") or ""),
                now=now)
        except (GangRegistryError, GroupConflictError):
            return {"ok": False, "reason": "frozen"}
        if out.get("ok"):
            self._audit(group_key, "gang_claim_redeemed",
                        user_id=user_id,
                        gang=str(out["gang"].get("name") or ""))
            self.store.save()
        return out

    def gang_war_side(self, group_key: str, user_id: str,
                      gang_name: str, *,
                      now: float | None = None) -> dict[str, Any]:
        """Declare your side for the live inter-group war
        against the named gang — the dual holder's loyalty
        act (§36).  Finds the active/declared war whose other
        side is that gang and records the declaration."""
        now = now if now is not None else time.time()
        gate = self._conflict_gate(group_key, "warfare")
        if gate:
            return gate
        gang = self.conflict.gang_of(group_key, user_id)
        if gang is None:
            return {"ok": False, "reason": "no_gang"}
        record = self._gang_register(group_key, gang, now=now)
        if record is None:
            return {"ok": False, "reason": "no_gang"}
        try:
            chosen = self.gangs.gang_by_name(gang_name)
        except GangRegistryError:
            return {"ok": False, "reason": "frozen"}
        if chosen is None:
            return {"ok": False, "reason": "no_gang"}
        pair = {str(record["id"]), str(chosen["id"])}
        for war in self.war_book.wars_for_group(group_key):
            if war.get("status") not in ("declared", "active"):
                continue
            sides = set()
            for s in ("a", "b"):
                side = war.get(s) or {}
                s_group = str(side.get("group") or "")
                if not s_group:
                    continue
                organ = self.conflict if s_group == group_key \
                    else self._conflict_for_store(str(
                        side.get("store") or self.store.path))
                try:
                    s_state = organ._load_state(s_group)
                except GroupConflictError:
                    continue
                s_gang = (s_state.get("gangs") or {}).get(
                    str(side.get("gang") or ""))
                if s_gang is not None and s_gang.get("global"):
                    sides.add(str(s_gang["global"]))
            if sides == pair:
                try:
                    return self.gangs.declare_war_side(
                        person=self._person_of(group_key,
                                               user_id),
                        war_id=str(war["id"]),
                        global_id=str(chosen["id"]), now=now)
                except GangRegistryError:
                    return {"ok": False, "reason": "frozen"}
        return {"ok": False, "reason": "no_war"}

    # ── §36: the owner's police override (surface verbs) ─────

    def police_wanted_view(self, group_key: str, member: str, *,
                           now: float | None = None
                           ) -> dict[str, Any]:
        now = now if now is not None else time.time()
        try:
            return self.gangs.police_view(group_key, member,
                                          now=now)
        except (GangRegistryError, GroupConflictError) as exc:
            return {"ok": False, "reason": "frozen",
                    "error": str(exc)}

    def police_owner_set_stars(self, group_key: str,
                               member: str, stars: int, *,
                               by_person: str,
                               now: float | None = None
                               ) -> dict[str, Any]:
        now = now if now is not None else time.time()
        try:
            out = self.gangs.police_set_stars(
                group_key, member, stars, by_person=by_person,
                now=now)
        except (GangRegistryError, GroupConflictError):
            return {"ok": False, "reason": "frozen"}
        if out.get("ok"):
            self._audit(group_key, "police_override",
                        user_id=member, verb="set_stars",
                        stars=int(stars))
            self.store.save()
        return out

    def police_owner_pardon(self, group_key: str, member: str,
                            *, by_person: str,
                            now: float | None = None
                            ) -> dict[str, Any]:
        """Owner pardon: the registry forgives the slate; an
        active jail lifts HERE — the release is the one
        member-visible part of an override."""
        now = now if now is not None else time.time()
        try:
            out = self.gangs.police_pardon(
                group_key, member, by_person=by_person,
                now=now)
        except (GangRegistryError, GroupConflictError):
            return {"ok": False, "reason": "frozen"}
        if out.get("ok"):
            released = False
            if self.jail_state(group_key, member,
                               now=now) is not None:
                released = self.jail_clear(group_key, member)
            out["released"] = bool(released)
            self._audit(group_key, "police_override",
                        user_id=member, verb="pardon",
                        released=bool(released))
            self.store.save()
        return out

    def police_owner_call_off(self, group_key: str,
                              member: str, *, by_person: str,
                              now: float | None = None
                              ) -> dict[str, Any]:
        now = now if now is not None else time.time()
        try:
            out = self.gangs.police_call_off(
                group_key, member, by_person=by_person,
                now=now)
        except (GangRegistryError, GroupConflictError):
            return {"ok": False, "reason": "frozen"}
        if out.get("ok"):
            self._audit(group_key, "police_override",
                        user_id=member, verb="call_off")
            self.store.save()
        return out

    def police_owner_focus(self, group_key: str,
                           aggression: float, *,
                           by_person: str,
                           now: float | None = None
                           ) -> dict[str, Any]:
        now = now if now is not None else time.time()
        try:
            out = self.gangs.police_set_focus(
                group_key, aggression, by_person=by_person,
                now=now)
        except (GangRegistryError, GroupConflictError):
            return {"ok": False, "reason": "frozen"}
        if out.get("ok"):
            self._audit(group_key, "police_override",
                        user_id="", verb="focus",
                        aggression=float(aggression))
            self.store.save()
        return out

    def console_police_view(self, group_key: str, *,
                            now: float | None = None
                            ) -> dict[str, Any]:
        """The console's police card data: the group's focus
        plus every member the ladder currently sees (stars or
        a rising index), read through the registry's view."""
        now = now if now is not None else time.time()
        organ = self.conflict
        try:
            state = organ._load_state(group_key)
        except GroupConflictError:
            return {"ok": False, "reason": "frozen"}
        members = set((state.get("rep") or {}).keys())
        for event in state.get("journal") or []:
            if event.get("kind") == "crime":
                members.add(str(event.get("actor") or ""))
        rows = []
        try:
            focus = self.gangs.police_focus(group_key)
            for member in sorted(m for m in members if m):
                view = self.gangs.police_view(
                    group_key, member, now=now)
                if view["stars"] > 0 or view["index"] >= 6.0:
                    rows.append(view)
        except GangRegistryError:
            return {"ok": False, "reason": "frozen"}
        rows.sort(key=lambda r: -float(r["index"]))
        return {"ok": True, "focus": focus, "wanted": rows[:10]}

    def member_id_for_token(self, group_key: str, token: str
                            ) -> str | None:
        """Resolve a console member token to a ledger user id:
        the id itself, an @username, or a display name — the
        group's own ledger only (no guessing beyond it)."""
        wanted = str(token or "").strip()
        if not wanted:
            return None
        members = self.store.group(group_key)["members"]
        if wanted in members:
            return wanted
        bare = wanted.lstrip("@").lower()
        for uid, member in members.items():
            if str(member.get("username") or "").lower() == bare:
                return str(uid)
        for uid, member in members.items():
            if str(member.get("name") or "").lower() == \
                    wanted.lower():
                return str(uid)
        return None

    # ── system drops (§35) ──

    def sysdrop_claim(self, group_key: str, user_id: str,
                      drop_id: str, *,
                      now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.time()
        if self._conflict_mode(group_key) == "off":
            return {"ok": False, "reason": "mode"}
        if self.conflict_frozen(group_key) is not None:
            return {"ok": False, "reason": "frozen"}
        out = self.conflict.claim_sysdrop(group_key, user_id,
                                          drop_id, now=now)
        if out.get("ok"):
            self._audit(group_key, "sysdrop_claim",
                        user_id=user_id, drop=drop_id,
                        loot=str((out.get("got") or {}).get(
                            "kind") or ""))
            self.store.save()
        return out

    def conflict_feed_take(self, group_key: str
                           ) -> list[dict[str, Any]]:
        """Drain the queued button-carrying announcements
        (system-drop openings) for the runtime to post."""
        extras = self._extras(group_key)
        feed = list(extras.get("conflict_feed") or [])
        if feed:
            extras["conflict_feed"] = []
            self.store.save()
        return feed

    def _report_tallies(self, group_key: str, *, now: float
                        ) -> dict[str, int]:
        """Distinct reporters per target over 7 days, counted
        from the audit journal /report already writes."""
        since = now - 7 * 86400
        seen: dict[str, set] = {}
        for entry in self.audit(group_key, limit=AUDIT_RING):
            if entry.get("kind") != "report":
                continue
            if float(entry.get("ts") or 0) < since:
                continue
            target = str(entry.get("user_id") or "")
            reporter = str(entry.get("by") or "")
            if target and reporter and target != reporter:
                seen.setdefault(target, set()).add(reporter)
        return {t: len(r) for t, r in seen.items() if len(r) >= 3}

    def conflict_cycle(self, group_key: str, *, rng: Any = None,
                       now: float | None = None) -> list[str]:
        """The conflict beat on the message cycle (§23/§24/
        §35): drop openings roll against the admin's daily
        budget, war rounds fall due, territory pays, expired
        drops close quietly, and the police sweep the bully
        index — each action journaled with its evidence, the
        reversible one (jail) posted as a §31 notice with an
        Undo, the settlements (fines, raids, bounties) posted
        plainly as final.  Claimed rooms only (§33): in a
        foreign room the conflict layer neither speaks nor
        accrues."""
        import random as _random
        now = now if now is not None else time.time()
        rng = rng if rng is not None else _random
        notices: list[str] = []
        # §36: first contact with a claimed room seeds the
        # flagship chapter (lazy, once per process) — even
        # while the street itself sleeps.
        self.gangs_touch(group_key, now=now)
        if self._conflict_mode(group_key) == "off" \
                or not self.economy_on(group_key) \
                or not self.system_output_allowed(group_key):
            return notices
        if self.conflict_frozen(group_key) is not None:
            return notices
        cfg = self.conflict_cfg(group_key)
        # Nightshield (P3): while it is up, war rounds and drop
        # openings pause — the street sleeps; territory income and
        # the police sweep are bookkeeping, not events, and run.
        shield = self.rhythm_nightshield(group_key, now=now)
        organ = self.conflict
        extras = self._extras(group_key)
        state = organ._load_state(group_key)
        # drops expire quietly — audit only, never a post
        for drop in organ.expire_sysdrops(group_key, now=now):
            self._audit(group_key, "sysdrop_expired",
                        drop=str(drop.get("id") or ""))
        # hourly-throttled machinery: wars, territory, police
        last = float(extras.get("conflict_cycle_at") or 0)
        if now - last >= 900:
            extras["conflict_cycle_at"] = now
            free = self.economy.treasury_balance(group_key) \
                - self.reserved_treasury(group_key)
            for event in ([] if shield else organ.cycle_wars(
                    group_key, rng=rng, now=now, cfg=cfg)):
                war = event.get("war") or {}
                gangs = state.get("gangs") or {}
                name_a = (gangs.get(str(war.get("a"))) or {}).get(
                    "name", "a crew")
                name_b = (gangs.get(str(war.get("b"))) or {}).get(
                    "name", "a crew")
                if event.get("kind") == "war_expired":
                    notices.append(notice_line(
                        "🏳️", f"the war between <b>{name_a}</b> "
                        f"and <b>{name_b}</b> was never answered "
                        f"— escrow returned."))
                elif event.get("finished"):
                    war_text = notice_line(
                        "⚔️", f"<b>{event['winner_gang']['name']}"
                        f"</b> wins the war against "
                        f"<b>{event['loser_gang']['name']}</b> "
                        f"and takes the "
                        f"{sparks(int(event.get('pot') or 0))} "
                        f"pot!"
                        + (" A district can be claimed with "
                           "/war spoils <district>."
                           if event.get("spoils_pending")
                           else ""))
                    notices.append(war_text)
                    self._channel_trigger(group_key, "war_result",
                                          war_text, now=now)
                else:
                    notices.append(notice_line(
                        "⚔️", f"war round {event.get('round')}: "
                        f"<b>{event['winner_gang']['name']}</b> "
                        f"takes it "
                        f"({event.get('power_a', 0):.0f} vs "
                        f"{event.get('power_b', 0):.0f})."))
            state = organ._load_state(group_key)
            for event in organ.cycle_territory(
                    group_key, now=now, cfg=cfg,
                    free_treasury=max(0, free)):
                if event.get("kind") == "territory_lost":
                    notices.append(notice_line(
                        "🏚️", f"<b>{event['gang']}</b> lost "
                        f"<b>{DISTRICTS.get(event['district'], event['district'])}</b>"
                        f" — the upkeep went unpaid."))
                elif int(event.get("amount") or 0) > 0:
                    notices.append(notice_line(
                        "🏙️", f"<b>{event['gang']}</b> collects "
                        f"{sparks(int(event['amount']))} from "
                        f"<b>{DISTRICTS.get(event['district'], event['district'])}</b>"
                        f" today."))
            self._conflict_cycle_inter(group_key, notices,
                                       rng=rng, now=now, cfg=cfg)
            if cfg.get("police"):
                tallies = self._report_tallies(group_key, now=now)
                if tallies:
                    organ.note_report_attention(
                        group_key, reports=tallies, now=now)
                free = self.economy.treasury_balance(group_key) \
                    - self.reserved_treasury(group_key)
                try:
                    focus = self.gangs.police_focus(group_key)
                except GangRegistryError:
                    focus = 1.0
                for action in organ.police_sweep(
                        group_key, now=now,
                        free_treasury=max(0, free), focus=focus):
                    notices.append(self._police_notice(
                        group_key, action, now=now))
            state = organ._load_state(group_key)
        # system drops: the budget roll (§35)
        state = organ._load_state(group_key)
        budget = int(cfg.get("drops_per_day") or 0)
        open_now = sum(1 for d in (state.get("drops") or {})
                       .values() if d.get("status") == "open")
        if not shield and budget > 0 and open_now == 0 and float(
                rng.random()) < float(cfg.get("drop_chance")
                                      or 0.25):
            drop = organ.open_drop(group_key, rng=rng, now=now,
                                   cfg=cfg)
            if drop is not None:
                feed = list(extras.get("conflict_feed") or [])
                feed.append({"kind": "sysdrop", "drop": drop})
                extras["conflict_feed"] = feed[-10:]
                self._audit(group_key, "sysdrop_opened",
                            drop=str(drop.get("id") or ""),
                            loot=str(drop.get("kind") or ""))
        self.store.save()
        return notices

    def _police_notice(self, group_key: str,
                       action: dict[str, Any], *,
                       now: float) -> str:
        """Render one police action: apply its jail here (the
        surface owns jail), record the §31 notice — Undo
        reverses the JAIL only; fines, raids and bounties are
        settlements and the notice says so by carrying no
        reversible flag for them."""
        member = str(action.get("member") or "")
        record = self.member(group_key, member) or {}
        label = record.get("name") or (
            f"@{record['username']}" if record.get("username")
            else member)
        bits = []
        if action.get("fine"):
            bits.append(f"fined {sparks(int(action['fine']))}")
        if action.get("raid"):
            bits.append(f"vault raided for "
                        f"{sparks(int(action['raid']))}")
        if action.get("bounty"):
            bits.append(f"a {sparks(int(action['bounty']))} "
                        f"bounty posted")
        jailed = int(action.get("jail_min") or 0)
        if jailed:
            self.jail_member(group_key, member, jailed,
                             reason=f"police ({action['stars']}★ "
                                    "wanted)", now=now)
            bits.append(f"jailed for {jailed}m")
        if not bits:
            bits.append("is being watched")
        text = (f"🚔 <b>{label}</b> is {action['stars']}★ "
                f"wanted — " + ", ".join(bits) + ".")
        # NB: the fine is deliberately NOT the record's `fine`
        # field — that field is the notice layer's refundable
        # slot, and a police fine is a settlement.  It rides
        # in extra, where an undo can read it but never pay
        # it back.
        self.record_notice(
            group_key, user_id=member, mention=str(label),
            rule="police",
            action="police_jail" if jailed else "police",
            minutes=jailed, fine=0,
            text=text,
            extra={"reversible": bool(jailed),
                   "settlement": bool(action.get("fine")
                                      or action.get("raid")
                                      or action.get("bounty")),
                   "fine_paid": int(action.get("fine") or 0),
                   "raid_paid": int(action.get("raid") or 0),
                   "index": action.get("index"),
                   "inputs": action.get("inputs")},
            now=now)
        self._audit(group_key, "police_action", user_id=member,
                    stars=action.get("stars"),
                    action=action.get("action"),
                    index=action.get("index"))
        return notice_line("🚔", text)

    def _conflict_cycle_inter(self, group_key: str,
                              notices: list[str], *, rng: Any,
                              now: float,
                              cfg: dict[str, Any]) -> None:
        """Inter-group wars on this group's beat: rounds fall
        due in the book, declarations arrive as feed lines,
        finished wars settle this side exactly once."""
        from ...community.group_conflict import inter_war_round
        try:
            wars = self.war_book.wars_for_group(group_key)
        except GroupConflictError:
            return
        organ = self.conflict
        gang = None
        extras = self._extras(group_key)
        for war in wars:
            mine_a = (war.get("a") or {}).get("group") == group_key
            side = "a" if mine_a else "b"
            enemy = (war.get("b") if mine_a else war.get("a")
                     ) or {}
            if war.get("status") == "declared" and side == "b" \
                    and not war.get("notified_b"):
                war["notified_b"] = True
                self.war_book.update(war)
                feed = list(extras.get("conflict_feed") or [])
                feed.append({
                    "kind": "line",
                    "text": notice_line(
                        "⚔️", f"<b>{(war.get('a') or {}).get('gang_name', 'a gang')}</b>"
                        f" has declared war on your gang "
                        f"<b>{(war.get('b') or {}).get('gang_name', '')}</b>"
                        f" — leadership answers with /war "
                        f"accept or /war decline.")})
                extras["conflict_feed"] = feed[-10:]
            if war.get("status") == "declared" and now - float(
                    war.get("declared_at") or now) > 2 * 86400:
                war["status"] = "expired"
                self.war_book.update(war)
                if side == "a":
                    state = organ._load_state(group_key)
                    escrow = int((state.get("escrow_inter")
                                  or {}).pop(str(war["id"]), 0))
                    if escrow:
                        mine = (state.get("gangs") or {}).get(
                            str((war.get("a") or {}).get("gang")
                                or ""))
                        if mine is not None:
                            mine["treasury"] = int(
                                mine.get("treasury") or 0) + escrow
                        organ._save_state(group_key, state)
            if war.get("status") == "active" and float(
                    war.get("next_round_at") or 0) <= now:
                event = inter_war_round(war, rng=rng, now=now,
                                        cfg=cfg)
                self.war_book.update(war)
                winner = (war.get("a") if event["winner_side"]
                          == "a" else war.get("b") or {})
                notices.append(notice_line(
                    "⚔️", f"inter-group war round "
                    f"{event['round']}: <b>{winner.get('gang_name', 'a gang')}</b>"
                    f" takes it ({event['power_a']:.0f} vs "
                    f"{event['power_b']:.0f})."))
                if event.get("finished"):
                    war_text = notice_line(
                        "🏆", f"<b>{(war.get(war.get('winner') or 'a') or {}).get('gang_name', 'a gang')}</b>"
                        f" wins the inter-group war against "
                        f"<b>{enemy.get('gang_name', 'the enemy')}</b>!")
                    notices.append(war_text)
                    self._channel_trigger(group_key, "war_result",
                                          war_text, now=now)
            if war.get("status") == "done" and not (
                    war.get("settled") or {}).get(side):
                if war.get("kind") == "hq_raid":
                    settled = organ.settle_hq_raid(
                        group_key, war, now=now)
                else:
                    settled = organ.settle_inter_war(
                        group_key, war, now=now)
                flags = dict(war.get("settled") or {})
                flags[side] = True
                war["settled"] = flags
                self.war_book.update(war)
                if settled is not None:
                    self._gangs_note_settled(group_key, war,
                                             now=now)
                if settled and settled.get("won"):
                    notices.append(notice_line(
                        "💰", f"<b>{settled.get('gang', 'your gang')}</b>"
                        f" collects the war spoils: "
                        f"{sparks(int(settled.get('escrow') or 0))}"
                        f" escrow returned, and the same again "
                        f"paid to its leader as spoils."))

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


    # ---- the rhythm layer (P3: §30/§31/§26) --------------------
    # The room's clock.  Everything periodic rides the existing
    # message-driven cycle (handle_message → rhythm_cycle); there
    # is no new daemon.  State lives in the store with the group:
    # settings under settings["rhythm"], the working state under
    # extras (rhythm_held / rhythm_days / rhythm_state / polls /
    # contributions / memory).

    def rhythm_cfg(self, group_key: str) -> dict[str, Any]:
        return dict(self.store.settings(group_key)["rhythm"])

    def rhythm_local(self, group_key: str,
                     ts: float) -> dict[str, Any]:
        cfg = self.rhythm_cfg(group_key)
        return _rhythm_local_parts(
            ts, int(cfg.get("tz_offset_min") or 0))

    def _rhythm_window_open(self, group_key: str,
                            now: float) -> bool:
        cfg = self.rhythm_cfg(group_key)
        parts = self.rhythm_local(group_key, now)
        return _rhythm_in_window(
            parts["minute"],
            _rhythm_parse_hhmm(cfg.get("silent_start")),
            _rhythm_parse_hhmm(cfg.get("silent_end")))

    def rhythm_silent(self, group_key: str, *,
                      now: float | None = None) -> bool:
        """Silent hours are up: non-urgent system output holds
        for the morning digest.  (Moderation never consults this
        — enforcement is not scheduled.)"""
        now = now if now is not None else time.time()
        if not self.rhythm_cfg(group_key).get("silent_enabled"):
            return False
        return self._rhythm_window_open(group_key, now)

    def rhythm_nightshield(self, group_key: str, *,
                           now: float | None = None) -> bool:
        """The nightshield rides the silent-hours WINDOW (the
        group's night) but is its own switch: posture tightens
        and drop/war events pause while it is up."""
        now = now if now is not None else time.time()
        if not self.rhythm_cfg(group_key).get("nightshield"):
            return False
        return self._rhythm_window_open(group_key, now)

    def rhythm_hold(self, group_key: str, kind: str, text: str,
                    *, now: float | None = None) -> None:
        """Queue one system post for the morning digest."""
        now = now if now is not None else time.time()
        held = self._extras(group_key).setdefault(
            "rhythm_held", [])
        held.append({"ts": now, "kind": str(kind),
                     "text": str(text)})
        del held[:-HELD_RING]
        self._audit(group_key, "rhythm_held", held_kind=str(kind),
                    queued=len(held))
        self.store.save()

    def _rhythm_count(self, group_key: str, field: str,
                      ts: float) -> None:
        """One tick on the group's local-day shape counters —
        the briefing's raw material, straight from the ledger
        events (message survived / join / leave)."""
        day = self.rhythm_local(group_key, ts)["day"]
        days = self._extras(group_key).setdefault(
            "rhythm_days", {})
        row = days.setdefault(
            day, {"messages": 0, "joins": 0, "leaves": 0})
        row[field] = int(row.get(field) or 0) + 1
        for old in sorted(days)[:-10]:
            days.pop(old, None)

    def rhythm_cycle(self, group_key: str, *,
                     now: float | None = None) -> list[str]:
        """The rhythm beat on the message cycle: release the
        digest when the quiet ends, close due polls (treasury
        polls execute through governance), detect milestones,
        fire due reminders, post the briefing at its hour, and —
        weekly, when §26 review is on — surface the reviewed
        rankers as a gift proposal for admins to decide.
        Claimed rooms only (§33): abroad, rhythm neither speaks
        nor accrues."""
        now = now if now is not None else time.time()
        if not self.system_output_allowed(group_key):
            return []
        extras = self._extras(group_key)
        last = float(extras.get("rhythm_cycle_at") or 0)
        if now - last < 60:
            return []
        extras["rhythm_cycle_at"] = now
        cfg = self.rhythm_cfg(group_key)
        out: list[str] = []
        if not self.rhythm_silent(group_key, now=now):
            held = list(extras.get("rhythm_held") or [])
            if held:
                extras["rhythm_held"] = []
                out.append(self._rhythm_digest_text(held))
                self._audit(group_key, "rhythm_released",
                            items=len(held))
        for poll in list((extras.get("polls") or {}).values()):
            if not poll.get("closed") and float(
                    poll.get("closes_at") or 0) <= now:
                out.append(self._close_poll(group_key, poll,
                                            now=now))
        for entry in self._detect_milestones(group_key, now):
            out.append(str(entry["text"]))
            self._channel_trigger(group_key, "milestone",
                                  str(entry["text"]), now=now)
        if cfg.get("reminders_enabled"):
            out.extend(self._fire_reminders(group_key, cfg,
                                            now))
        if cfg.get("briefing_enabled"):
            text = self._briefing_if_due(group_key, cfg, now)
            if text:
                out.append(text)
        gift = self._gift_proposal_if_due(group_key, cfg, now)
        if gift:
            out.append(gift)
        self.store.save()
        return [text for text in out if text]

    def _rhythm_digest_text(self, held: list[dict[str, Any]]
                            ) -> str:
        lines = [f"🌅 <b>While the room was quiet</b> — "
                 f"{len(held)} post(s) held for the morning:"]
        for item in held:
            lines.append(str(item.get("text") or ""))
        return "\n\n".join(lines)

    # -- reminders (§31 cards on the rhythm tick) ----------------

    def _fire_reminders(self, group_key: str,
                        cfg: dict[str, Any],
                        now: float) -> list[str]:
        parts = self.rhythm_local(group_key, now)
        state = self._extras(group_key).setdefault(
            "rhythm_state", {})
        fired = state.setdefault("reminders_fired", {})
        out: list[str] = []
        for reminder in cfg.get("reminders") or []:
            rid = str(reminder.get("id") or "")
            if not rid:
                continue
            days = reminder.get("days") or []
            if days and parts["weekday"] not in days:
                continue
            due = _rhythm_parse_hhmm(reminder.get("time"))
            if due is None or parts["minute"] < due:
                continue
            if fired.get(rid) == parts["day"]:
                continue
            fired[rid] = parts["day"]
            self._audit(group_key, "reminder_fired", reminder=rid)
            if reminder.get("kind") == "rules":
                out.append(self.rules_text(group_key))
            else:
                out.append(notice_line(
                    "⏰", str(reminder.get("text") or "")))
        return out

    # -- the group briefing ---------------------------------------

    def _briefing_if_due(self, group_key: str,
                         cfg: dict[str, Any],
                         now: float) -> str:
        parts = self.rhythm_local(group_key, now)
        if parts["hour"] < int(cfg.get("briefing_hour") or 8):
            return ""
        state = self._extras(group_key).setdefault(
            "rhythm_state", {})
        if state.get("briefing_day") == parts["day"]:
            return ""
        state["briefing_day"] = parts["day"]
        self._audit(group_key, "briefing_posted",
                    day=parts["day"])
        return self.compose_briefing(group_key, now=now)

    def compose_briefing(self, group_key: str, *,
                         now: float | None = None) -> str:
        """Yesterday's shape, from the store's real ledgers
        only: day counters, counted contributions, the economy
        ledger, territory as it stands (with flips since the
        last briefing), yesterday's milestones, today's
        reminders.  No data → an honest short card."""
        from .typography import Card

        now = now if now is not None else time.time()
        cfg = self.rhythm_cfg(group_key)
        offset = int(cfg.get("tz_offset_min") or 0)
        parts = _rhythm_local_parts(now, offset)
        yday = _rhythm_local_parts(now - 86400, offset)["day"]
        start, end = _rhythm_day_window(yday, offset)
        extras = self._extras(group_key)
        counts = (extras.get("rhythm_days") or {}).get(
            yday) or {}
        card = Card("announcement", title=f"Yesterday — {yday}",
                    icon="🌅")
        messages = int(counts.get("messages") or 0)
        joins = int(counts.get("joins") or 0)
        leaves = int(counts.get("leaves") or 0)
        if messages or joins or leaves:
            card.stats([
                ("Messages", f"{messages:,}"),
                ("Joined", f"{joins:,}"),
                ("Left", f"{leaves:,}"),
            ], title="The room")
        # top contributors: counted contribution events in the
        # window (post-§26 review, whatever the group's mode).
        per_member: dict[str, int] = {}
        for event in extras.get("contributions") or []:
            if event.get("state") != "counted":
                continue
            if start <= float(event.get("ts") or 0) < end:
                uid = str(event.get("user_id") or "")
                per_member[uid] = per_member.get(uid, 0) + 1
        if per_member:
            top = sorted(per_member.items(),
                         key=lambda kv: -kv[1])[:3]
            card.block_lines(
                [f"{self._member_label(group_key, uid)} — "
                 f"{n} counted contribution(s)"
                 for uid, n in top],
                title="Top contributors")
        # economy movement from the ledger itself.
        econ = self.economy
        treasury_delta = 0
        earners: dict[str, int] = {}
        try:
            ledger = econ._ledger(group_key)
        except Exception:
            ledger = []  # a frozen ledger reads as no movement,
            # never as invented movement (the freeze itself is
            # surfaced by the economy's own paths)
        for entry in ledger:
            ts = float(getattr(entry, "ts", 0) or 0)
            if not (start <= ts < end):
                continue
            amount = int(getattr(entry, "amount", 0) or 0)
            member = str(getattr(entry, "member", "") or "")
            if member == "":
                treasury_delta += amount
            elif amount > 0 and not econ.is_pseudo_member(
                    member):
                earners[member] = earners.get(member, 0) + amount
        econ_lines = []
        if treasury_delta:
            econ_lines.append(
                f"Treasury {'+' if treasury_delta > 0 else ''}"
                f"{treasury_delta:,} sparks")
        if earners:
            uid, amount = max(earners.items(),
                              key=lambda kv: kv[1])
            econ_lines.append(
                f"Top earner: "
                f"{self._member_label(group_key, uid)} "
                f"(+{amount:,} sparks)")
        if econ_lines:
            card.block_lines(econ_lines, title="Economy")
        # territory as it stands + flips since the last briefing
        terr_lines = self._briefing_territory(group_key)
        if terr_lines:
            card.block_lines(terr_lines, title="Territory")
        # yesterday's milestones
        book = extras.get("memory") or {}
        yday_ms = [e for e in book.get("entries") or []
                   if start <= float(e.get("ts") or 0) < end]
        if yday_ms:
            card.block_lines(
                [str(e.get("text") or "") for e in yday_ms[-3:]],
                title="Milestones")
        # today's reminders
        today = [r for r in cfg.get("reminders") or []
                 if cfg.get("reminders_enabled")
                 and (not (r.get("days") or [])
                      or parts["weekday"] in (r.get("days")
                                              or []))]
        if today:
            card.block_lines(
                [f"{r.get('time')} — "
                 + ("the group rules card"
                    if r.get("kind") == "rules"
                    else str(r.get("text") or ""))
                 for r in today],
                title="Today's reminders")
        if not (messages or joins or leaves or per_member
                or econ_lines or terr_lines or yday_ms):
            card.line("A quiet day — nothing on the ledgers "
                      "yet. The room's story starts with its "
                      "first busy day.")
        return self._card_render(group_key, card)

    def _briefing_territory(self, group_key: str) -> list[str]:
        """Current district holders + flips since the last
        briefing's snapshot (the snapshot diff is the honest
        'changes' read — no invented history)."""
        if self._conflict_mode(group_key) == "off":
            return []
        try:
            state = self.conflict._load_state(group_key)
        except Exception:
            return []
        gangs = state.get("gangs") or {}
        current: dict[str, str] = {}
        for district, hold in (state.get("territory")
                               or {}).items():
            gid = str((hold or {}).get("gang") or "")
            name = (gangs.get(gid) or {}).get("name") or gid
            if name:
                current[str(district)] = str(name)
        rstate = self._extras(group_key).setdefault(
            "rhythm_state", {})
        seen = rstate.get("territory_seen")
        rstate["territory_seen"] = current
        if not current:
            return []
        lines = []
        for district, name in sorted(current.items()):
            label = DISTRICTS.get(district, district)
            label = label["name"] if isinstance(
                label, dict) else str(label)
            mark = ""
            if isinstance(seen, dict) and district in seen \
                    and seen[district] != name:
                mark = f" (new — was {seen[district]})"
            lines.append(f"{label} — {name}{mark}")
        return lines


    # -- executing polls ------------------------------------------
    # One counting core in the surface.  No adapter exposes a
    # native poll verb today (checked: telegram / whatsapp /
    # discord), so every platform renders the house button-card
    # poll — buttons where the platform has them, /vote numbers
    # where it does not — and this core counts either way.

    def lapse_proposal(self, group_key: str, proposal_id: str,
                       *, by: str,
                       now: float | None = None) -> None:
        """Close an open proposal WITHOUT executing it (a
        treasury poll that failed or missed quorum): the money
        never moved, and the journal says why it closed."""
        now = now if now is not None else time.time()
        proposals = self._extras(group_key).setdefault(
            "proposals", {})
        proposal = proposals.get(proposal_id)
        if proposal and proposal.get("open"):
            proposal["open"] = False
            proposal["lapsed_by"] = by
            self._audit(group_key, "proposal_lapsed",
                        proposal=proposal_id, by=by)
            self.store.save()

    def create_poll(self, group_key: str, *, question: str,
                    options: list[str], by: str,
                    kind: str = "opinion",
                    action: dict[str, Any] | None = None,
                    quorum: int | None = None,
                    duration_s: float | None = None,
                    changeable: bool = False,
                    now: float | None = None) -> dict[str, Any]:
        """Open a poll.  A TREASURY poll first opens its real
        governance proposal — the poll IS the vote on that
        proposal, and passing executes it through the same path
        an admin's approve takes."""
        now = now if now is not None else time.time()
        cfg = self.rhythm_cfg(group_key)
        options = [str(o).strip()[:80] for o in options
                   if str(o).strip()][:6]
        if len(options) < 2:
            return {"ok": False, "reason": "options"}
        extras = self._extras(group_key)
        seq = int(extras.get("poll_seq") or 0) + 1
        poll: dict[str, Any] = {
            "id": f"poll-{seq}", "question": str(question)[:300],
            "options": options, "by": by, "kind": kind,
            "quorum": int(quorum if quorum is not None
                          else cfg.get("poll_quorum") or 3),
            "changeable": bool(changeable),
            "created": now,
            "closes_at": now + float(
                duration_s or cfg.get("poll_duration_s")
                or 86400),
            "votes": {}, "closed": False, "result": None,
        }
        if kind == "treasury":
            action = action or {}
            proposal = self.propose_spend(
                group_key,
                kind=str(action.get("kind") or "grant"),
                payload=dict(action.get("payload") or {}),
                by=by, now=now)
            if not proposal.get("id"):
                return {"ok": False, "reason": "proposal"}
            poll["proposal_id"] = proposal["id"]
            poll["action"] = action
        extras["poll_seq"] = seq
        extras.setdefault("polls", {})[poll["id"]] = poll
        self._audit(group_key, "poll_created", ref=poll["id"],
                    poll_kind=kind, by=by)
        self.store.save()
        return {"ok": True, "poll": poll}

    def polls(self, group_key: str, *,
              open_only: bool = False) -> list[dict[str, Any]]:
        polls = list((self._extras(group_key).get("polls")
                      or {}).values())
        polls.sort(key=lambda p: float(p.get("created") or 0))
        if open_only:
            polls = [p for p in polls if not p.get("closed")]
        return polls

    def get_poll(self, group_key: str,
                 ref: str) -> dict[str, Any] | None:
        return (self._extras(group_key).get("polls")
                or {}).get(str(ref))

    def vote_poll(self, group_key: str, ref: str, user_id: str,
                  option_index: int, *,
                  now: float | None = None) -> dict[str, Any]:
        """One vote per member.  Changing a vote before close is
        allowed only when the poll was opened as changeable."""
        now = now if now is not None else time.time()
        poll = self.get_poll(group_key, ref) if ref else None
        if poll is None:
            open_polls = self.polls(group_key, open_only=True)
            poll = open_polls[-1] if open_polls else None
        if poll is None:
            return {"ok": False, "reason": "no_poll"}
        if poll.get("closed") or now >= float(
                poll.get("closes_at") or 0):
            return {"ok": False, "reason": "closed"}
        idx = int(option_index)
        if idx < 0 or idx >= len(poll.get("options") or []):
            return {"ok": False, "reason": "option"}
        votes = poll.setdefault("votes", {})
        existing = votes.get(user_id)
        if existing is not None and int(existing) != idx \
                and not poll.get("changeable"):
            return {"ok": False, "reason": "locked",
                    "poll": poll}
        votes[user_id] = idx
        self._audit(group_key, "poll_vote", ref=poll["id"],
                    user_id=user_id)
        self.store.save()
        return {"ok": True, "poll": poll,
                "changed": existing is not None}

    def close_poll(self, group_key: str, ref: str, *,
                   by: str,
                   now: float | None = None) -> dict[str, Any]:
        """Close a poll NOW (its creator or an admin — the
        handler checks) and settle it like a deadline close."""
        now = now if now is not None else time.time()
        poll = self.get_poll(group_key, ref)
        if poll is None:
            return {"ok": False, "reason": "no_poll"}
        if poll.get("closed"):
            return {"ok": False, "reason": "closed"}
        text = self._close_poll(group_key, poll, now=now)
        self.store.save()
        return {"ok": True, "poll": poll, "text": text}

    def _poll_tally(self, poll: dict[str, Any]) -> list[int]:
        counts = [0] * len(poll.get("options") or [])
        for idx in (poll.get("votes") or {}).values():
            idx = int(idx)
            if 0 <= idx < len(counts):
                counts[idx] += 1
        return counts

    def _close_poll(self, group_key: str, poll: dict[str, Any],
                    *, now: float) -> str:
        poll["closed"] = True
        poll["closed_at"] = now
        counts = self._poll_tally(poll)
        total = sum(counts)
        lines = [f"🗳️ <b>Poll closed</b> — {poll['question']}"]
        for i, option in enumerate(poll.get("options") or []):
            lines.append(f"• {option} — {counts[i]} vote(s)")
        if poll.get("kind") == "treasury":
            yes, no = counts[0], counts[1]
            quorum = int(poll.get("quorum") or 3)
            passed = yes > no and total >= quorum
            executed = False
            if passed:
                out = self._execute_proposal(
                    group_key, str(poll.get("proposal_id") or ""),
                    by=f"poll {poll['id']}", now=now)
                executed = bool(out.get("ok"))
                if not executed:
                    self._audit(
                        group_key, "poll_execution_failed",
                        ref=poll["id"],
                        reason=str(out.get("reason") or ""))
            else:
                self.lapse_proposal(
                    group_key, str(poll.get("proposal_id") or ""),
                    by=f"poll {poll['id']}", now=now)
            poll["result"] = {"counts": counts, "total": total,
                              "passed": passed,
                              "executed": executed}
            self._audit(group_key, "poll_closed",
                        ref=poll["id"], poll_kind="treasury",
                        passed=passed, executed=executed,
                        votes=total)
            action = poll.get("action") or {}
            payload = action.get("payload") or {}
            if executed:
                lines.append(
                    f"✅ Passed ({total} votes, quorum {quorum})"
                    f" — executed: "
                    f"{int(payload.get('amount') or 0):,} sparks "
                    f"to "
                    f"{self._member_label(group_key, str(payload.get('recipient') or ''))}.")
            elif passed:
                lines.append(
                    "⚠️ The vote passed but the execution "
                    "failed — nothing moved; the journal has "
                    "the reason and an admin can re-propose.")
            else:
                lines.append(
                    f"❌ Did not pass ({total} votes, quorum "
                    f"{quorum}) — nothing was spent.")
        else:
            best = max(counts) if counts else 0
            winners = [poll["options"][i]
                       for i, c in enumerate(counts) if c == best]
            poll["result"] = {"counts": counts, "total": total,
                              "winners": winners}
            self._audit(group_key, "poll_closed",
                        ref=poll["id"], poll_kind="opinion",
                        votes=total)
            if total == 0:
                lines.append("No votes were cast.")
            elif len(winners) == 1:
                lines.append(f"🏆 <b>{winners[0]}</b> wins.")
            else:
                lines.append(
                    "🤝 A tie between "
                    + " and ".join(f"<b>{w}</b>" for w in winners)
                    + ".")
        return "\n".join(lines)

    # -- group memory + milestones --------------------------------

    def memory_book(self, group_key: str) -> dict[str, Any]:
        return self._extras(group_key).setdefault(
            "memory", {"seen": {}, "entries": [], "records": {}})

    def milestones(self, group_key: str, *,
                   limit: int = 10) -> list[dict[str, Any]]:
        entries = self.memory_book(group_key).get(
            "entries") or []
        return list(reversed(entries))[:limit]

    def _milestone_truths(self, group_key: str, now: float,
                          book: dict[str, Any]
                          ) -> list[dict[str, Any]]:
        """Everything the ledgers can prove is true RIGHT NOW,
        as (key, kind, text, user_id) milestone candidates.  The
        seen-set decides which are news."""
        out: list[dict[str, Any]] = []
        members = self.members(group_key)
        total = sum(int(m.get("message_count") or 0)
                    for m in members.values())
        for th in reversed(GROUP_MESSAGE_MILESTONES):
            if total >= th:
                out.append({
                    "key": f"group_msgs:{th}", "kind": "messages",
                    "text": f"🎉 The group just passed "
                            f"{th:,} messages together.",
                    "user_id": ""})
                break
        for uid, member in members.items():
            count = int(member.get("message_count") or 0)
            for th in reversed(MEMBER_MESSAGE_MILESTONES):
                if count >= th:
                    out.append({
                        "key": f"member_msgs:{uid}:{th}",
                        "kind": "member_messages",
                        "text": f"🎉 "
                                f"{self._member_label(group_key, uid)}"
                                f" just passed {th:,} messages "
                                f"in this group.",
                        "user_id": uid})
                    break
            joined = float(member.get("joined_ts") or 0)
            if joined:
                years = int((now - joined) // (365.25 * 86400))
                if years >= 1:
                    out.append({
                        "key": f"anniv:{uid}:{years}",
                        "kind": "anniversary",
                        "text": f"🎂 "
                                f"{self._member_label(group_key, uid)}"
                                f" has been with this group for "
                                f"{years} year"
                                f"{'s' if years != 1 else ''}.",
                        "user_id": uid})
        # economy records: richest balance + biggest heist are
        # RECORDS — a truth only when the standing record falls.
        records = book.setdefault("records", {})
        econ = self.economy
        try:
            balances = econ.balances(group_key)
        except Exception:
            balances = {}
        best_uid, best_total = "", 0
        for uid, wallet in balances.items():
            if not uid or econ.is_pseudo_member(uid):
                continue
            total_bal = int(wallet) + int(
                econ.bank_balance(group_key, uid))
            if total_bal > best_total:
                best_uid, best_total = uid, total_bal
        standing = records.get("richest") or {}
        if best_total > int(standing.get("total") or 0):
            out.append({
                "key": f"richest:{best_uid}:{best_total}",
                "kind": "richest", "user_id": best_uid,
                "text": f"👑 "
                        f"{self._member_label(group_key, best_uid)}"
                        f" now holds the richest balance this "
                        f"group has seen: {best_total:,} sparks.",
                "record": {"richest": {"user_id": best_uid,
                                       "total": best_total}}})
        if self._conflict_mode(group_key) != "off":
            try:
                state = self.conflict._load_state(group_key)
            except Exception:
                state = {}
            if state.get("gangs"):
                out.append({
                    "key": "first_gang", "kind": "first_gang",
                    "user_id": "",
                    "text": "🏴 The group's first gang has "
                            "formed."})
            if state.get("wars"):
                out.append({
                    "key": "first_war", "kind": "first_war",
                    "user_id": "",
                    "text": "⚔️ The group's first war is on."})
            takes = [int(e.get("take") or 0)
                     for e in state.get("journal") or []
                     if e.get("kind") == "heist"]
            standing_heist = int(records.get("heist") or 0)
            if takes and max(takes) > standing_heist:
                top_take = max(takes)
                out.append({
                    "key": f"heist:{top_take}", "kind": "heist",
                    "user_id": "",
                    "text": f"💰 Biggest heist yet: "
                            f"{top_take:,} sparks taken in one "
                            f"run.",
                    "record": {"heist": top_take}})
        return out

    def _detect_milestones(self, group_key: str, now: float
                           ) -> list[dict[str, Any]]:
        """Record what the ledgers newly prove.  The FIRST pass
        for a group seeds silently: everything already true is
        marked seen and the records baselined, so the book never
        announces a past it didn't witness — milestones accrue
        from this build forward, seeded only from facts the
        ledger already proves."""
        book = self.memory_book(group_key)
        seen = book.setdefault("seen", {})
        truths = self._milestone_truths(group_key, now, book)
        if not book.get("seeded"):
            book["seeded"] = True
            for truth in truths:
                seen[truth["key"]] = True
                for rkey, rvalue in (
                        truth.get("record") or {}).items():
                    book["records"][rkey] = rvalue
            # a group whose richest/heist records predate the
            # book still baselines them (no announcement).
            if truths:
                self._audit(group_key, "memory_seeded",
                            facts=len(truths))
            return []
        new: list[dict[str, Any]] = []
        for truth in truths:
            if truth["key"] in seen:
                continue
            seen[truth["key"]] = True
            for rkey, rvalue in (truth.get("record")
                                 or {}).items():
                book["records"][rkey] = rvalue
            entry = {"key": truth["key"], "kind": truth["kind"],
                     "text": truth["text"], "ts": now,
                     "user_id": truth.get("user_id") or ""}
            book.setdefault("entries", []).append(entry)
            new.append(entry)
            self._audit(group_key, "milestone",
                        mkind=truth["kind"], key=truth["key"])
        return new

    # -- §26 contribution review -----------------------------------

    def review_mode(self, group_key: str) -> str:
        review = self.rhythm_cfg(group_key).get("review") or {}
        mode = str(review.get("mode") or "off")
        return mode if mode in ("off", "auto", "assisted",
                                "manual") else "off"

    def _note_contribution(self, group_key: str, user_id: str,
                           text: str, *, has_media: bool,
                           has_link: bool,
                           now: float) -> dict[str, Any] | None:
        """Score one surviving message as a §26 contribution
        candidate and file it under the group's review mode.
        Counted events feed the contribution rankers and the
        briefing; the member's durable counted total lives on
        their ledger record, so the ranker survives the event
        ring."""
        text = str(text or "")
        if not text or text.startswith("/"):
            return None
        if not is_contribution_candidate(
                text, has_media=has_media, has_link=has_link):
            return None
        member = self.member(group_key, user_id)
        if member is None:
            return None
        own = list(member.get("contribution_texts") or [])
        recent = [str(item.get("text") or "")
                  for item in list(self._msg_cache.get(
                      group_key, ()))[-30:]
                  if item.get("user_id") != user_id]
        score, reasons = score_contribution(
            text, own_texts=own, recent_texts=recent,
            has_media=has_media, has_link=has_link)
        mode = self.review_mode(group_key)
        if mode == "off":
            state = "counted"  # as today: no review at all
        elif mode == "manual":
            state = "pending"  # a human approves everything
        elif mode == "assisted":
            state = ("counted" if score >= CLEAR_SCORE
                     else "pending" if score == BORDERLINE_SCORE
                     else "rejected")
        else:  # auto: the ensemble decides, members can appeal
            state = "counted" if score >= BORDERLINE_SCORE \
                else "rejected"
        extras = self._extras(group_key)
        seq = int(extras.get("contribution_seq") or 0) + 1
        extras["contribution_seq"] = seq
        event = {"id": f"contrib-{seq}", "user_id": user_id,
                 "text": text[:500], "ts": now, "score": score,
                 "reasons": reasons, "state": state,
                 "appealed": False}
        events = extras.setdefault("contributions", [])
        events.append(event)
        del events[:-CONTRIBUTION_RING]
        member["contribution_texts"] = (own + [text])[-20:]
        if state == "counted":
            self._contribution_counted(group_key, user_id)
        if mode != "off":
            self._audit(group_key, "contribution",
                        user_id=user_id, state=state, score=score)
        return event

    def _contribution_counted(self, group_key: str,
                              user_id: str) -> None:
        member = self.member(group_key, user_id)
        if member is None:
            return
        tally = member.setdefault("contributions", {})
        tally["counted"] = int(tally.get("counted") or 0) + 1

    def contribution_ranking(self, group_key: str, *,
                             limit: int = 10
                             ) -> list[dict[str, Any]]:
        """The reviewed ranker (§26): members by COUNTED
        contributions — pending and rejected work never counts
        while review is on; with review off, everything counts
        as it did before §26 existed."""
        rows = []
        for uid, member in self.members(
                group_key).items():
            counted = int((member.get("contributions")
                           or {}).get("counted") or 0)
            if counted > 0:
                rows.append({"user_id": uid, "counted": counted,
                             "label": self._member_label(
                                 group_key, uid)})
        rows.sort(key=lambda r: (-r["counted"], r["user_id"]))
        return rows[:limit]

    def contributions_of(self, group_key: str, user_id: str
                         ) -> list[dict[str, Any]]:
        return [e for e in self._extras(group_key).get(
            "contributions") or []
            if e.get("user_id") == user_id]

    def contribution_queue(self, group_key: str
                           ) -> list[dict[str, Any]]:
        """Pending review work, oldest first — appeals ride the
        same queue, flagged."""
        return [e for e in self._extras(group_key).get(
            "contributions") or [] if e.get("state") == "pending"]

    def review_contribution(self, group_key: str, event_id: str,
                            *, approve: bool, by: str,
                            now: float | None = None
                            ) -> dict[str, Any]:
        """An admin's ruling on a queued contribution — final,
        journaled; approving is the moment it starts counting."""
        now = now if now is not None else time.time()
        for event in self._extras(group_key).get(
                "contributions") or []:
            if event.get("id") != event_id:
                continue
            if event.get("state") != "pending":
                return {"ok": False, "reason": "decided",
                        "event": event}
            event["state"] = "counted" if approve else "rejected"
            event["ruled_by"] = by
            if approve:
                self._contribution_counted(
                    group_key, str(event.get("user_id") or ""))
            self._audit(group_key, "contribution_reviewed",
                        user_id=str(event.get("user_id") or ""),
                        approve=approve, by=by,
                        appeal=bool(event.get("appealed")))
            self.store.save()
            return {"ok": True, "event": event}
        return {"ok": False, "reason": "unknown"}

    def appeal_contribution(self, group_key: str, event_id: str,
                            user_id: str, *,
                            now: float | None = None
                            ) -> dict[str, Any]:
        """One appeal per item (§26): a rejected contribution
        goes back to the queue, flagged as an appeal; the
        admin's ruling on it is final."""
        for event in self._extras(group_key).get(
                "contributions") or []:
            if event.get("id") != event_id:
                continue
            if event.get("user_id") != user_id:
                return {"ok": False, "reason": "not_yours"}
            if event.get("state") != "rejected":
                return {"ok": False, "reason": "not_rejected"}
            if event.get("appealed"):
                return {"ok": False, "reason": "appealed"}
            event["appealed"] = True
            event["state"] = "pending"
            self._audit(group_key, "contribution_appealed",
                        user_id=user_id)
            self.store.save()
            return {"ok": True, "event": event}
        return {"ok": False, "reason": "unknown"}

    # -- §26: the weekly gift proposal ------------------------------

    def _gift_proposal_if_due(self, group_key: str,
                              cfg: dict[str, Any],
                              now: float) -> str:
        """Once a week, when §26 review is ON, the reviewed
        rankers surface as a gift PROPOSAL through governance
        (DESIGN §26) — the system proposes, admins decide; the
        payment itself rides the treasury path like any other
        grant."""
        if self.review_mode(group_key) == "off":
            return ""
        if not self.economy_on(group_key):
            return ""
        import time as _time
        tm = _time.gmtime(now + int(
            cfg.get("tz_offset_min") or 0) * 60)
        week = _time.strftime("%Y-W%W", tm)
        state = self._extras(group_key).setdefault(
            "rhythm_state", {})
        if state.get("gift_week") == week:
            return ""
        since = now - 7 * 86400
        per_member: dict[str, int] = {}
        for event in self._extras(group_key).get(
                "contributions") or []:
            if event.get("state") == "counted" and float(
                    event.get("ts") or 0) >= since:
                uid = str(event.get("user_id") or "")
                per_member[uid] = per_member.get(uid, 0) + 1
        if not per_member:
            return ""
        uid, count = max(per_member.items(),
                         key=lambda kv: kv[1])
        if count < 3:
            return ""
        free = self.economy.treasury_balance(group_key) \
            - self.reserved_treasury(group_key)
        if free < 100:
            return ""
        amount = max(50, min(500, int(free * 0.05)))
        proposal = self.propose_spend(
            group_key, kind="grant",
            payload={"recipient": uid, "amount": amount,
                     "reason": "§26 reviewed-contribution gift "
                               "proposal (weekly)"},
            by="rhythm", now=now)
        if not proposal.get("id"):
            return ""
        state["gift_week"] = week
        self._audit(group_key, "gift_proposal",
                    proposal=proposal["id"], user_id=uid,
                    amount=amount)
        return (f"🎁 <b>Gift proposal open</b> — "
                f"{sparks(amount)} to "
                f"{self._member_label(group_key, uid)}, the "
                f"week's top reviewed contributor "
                f"({count} counted). Admins decide: "
                f"/approve {proposal['id']} or "
                f"/deny {proposal['id']}.")

    # -- channels (§16 / P4a): registry, queue, analytics --------
    #
    # A channel stays SILENT (§16 law, unchanged): no commands,
    # no catalog, no replies ever run in it.  What this build
    # adds is the OWNER side: channels Devon admins are claimed
    # like groups (the §33 bot-admin stamp / owner designation),
    # and claimed channels carry a POSTING QUEUE the owner runs
    # from the DM console.  The queue posts on its cadence when
    # the runtime drives the cycle (channel activity, group
    # turns, console actions — no new daemon); every post is
    # journaled with its source.  Channel state lives in the
    # channel's own store entry: extras["channel"] (registry
    # note), extras["channel_queue"], extras["channel_posts"],
    # extras["channel_counts"], settings["channel"] (cadence,
    # trigger links, analytics switch).

    #: The group events an owner can link a channel to.
    CHANNEL_TRIGGERS = ("season_podium", "war_result",
                        "milestone")
    _CHANNEL_CYCLE_GAP_S = 15.0
    _CHANNEL_SAMPLE_GAP_S = 6 * 3600.0
    _CHANNEL_QUEUE_CAP = 100
    _CHANNEL_POSTS_RING = 100
    _CHANNEL_COUNTS_RING = 120
    _CHANNEL_POSTS_PER_CYCLE = 5

    def note_channel_seen(self, key: str, *, title: str = "",
                          now: float | None = None) -> None:
        """Registry note: this chat is a channel Devon has seen
        (a silent read — the channel itself still gets no
        ingest, no replies, no state beyond this note and,
        once claimed, its queue)."""
        now = now if now is not None else time.time()
        extras = self._extras(key)
        note = extras.setdefault("channel", {})
        changed = False
        if not note.get("seen_at"):
            note["seen_at"] = now
            changed = True
        if title and note.get("title") != title:
            note["title"] = str(title)
            changed = True
        if changed:
            self.store.save()

    def is_channel(self, key: str) -> bool:
        entry = self.store.data.get(key) or {}
        return bool((entry.get("extras") or {}).get("channel"))

    def channel_keys(self) -> list[str]:
        return sorted(k for k in self.store.data
                      if self.is_channel(k))

    def channel_title(self, key: str) -> str:
        entry = self.store.data.get(key) or {}
        note = (entry.get("extras") or {}).get("channel") or {}
        return str(note.get("title") or "")

    def channel_cfg(self, key: str) -> dict[str, Any]:
        return dict(self.store.settings(key).get("channel")
                    or {})

    def channel_queue(self, key: str) -> list[dict[str, Any]]:
        extras = self._extras(key)
        return [dict(e) for e in
                (extras.get("channel_queue") or [])]

    def channel_enqueue(self, key: str, entry: dict[str, Any],
                        *, now: float | None = None
                        ) -> dict[str, Any] | None:
        """Queue one post.  Returns the stored entry (with its
        internal id), or None when the queue is full — the
        caller reports the refusal honestly."""
        now = now if now is not None else time.time()
        extras = self._extras(key)
        queue = list(extras.get("channel_queue") or [])
        if len(queue) >= self._CHANNEL_QUEUE_CAP:
            self._audit(key, "channel_queue_full",
                        entry_kind=str(entry.get("kind") or ""))
            return None
        seq = int(extras.get("channel_seq") or 0) + 1
        extras["channel_seq"] = seq
        stored = {
            "id": f"p{seq}",
            "kind": str(entry.get("kind") or "once"),
            "text": str(entry.get("text") or ""),
            "media": entry.get("media") or None,
            "due": float(entry.get("due") or now),
            "time": str(entry.get("time") or ""),
            "weekday": entry.get("weekday"),
            "trigger": str(entry.get("trigger") or ""),
            "source": str(entry.get("source") or "dm"),
            "source_group": str(entry.get("source_group") or ""),
            "created_by": str(entry.get("created_by") or ""),
            "created_at": now,
            "last_posted": 0.0,
        }
        queue.append(stored)
        extras["channel_queue"] = queue
        self._audit(key, "channel_queued", entry=stored["id"],
                    entry_kind=stored["kind"],
                    source=stored["source"],
                    trigger=stored["trigger"])
        self.store.save()
        return dict(stored)

    def channel_cancel(self, key: str, position: int
                       ) -> dict[str, Any] | None:
        """Cancel the queue entry at a 1-based position (the
        number the console shows — internal ids never print)."""
        extras = self._extras(key)
        queue = list(extras.get("channel_queue") or [])
        if position < 1 or position > len(queue):
            return None
        gone = queue.pop(position - 1)
        extras["channel_queue"] = queue
        self._audit(key, "channel_cancelled",
                    entry=str(gone.get("id") or ""),
                    entry_kind=str(gone.get("kind") or ""))
        self.store.save()
        return gone

    def channel_clear(self, key: str) -> int:
        extras = self._extras(key)
        count = len(extras.get("channel_queue") or [])
        if count:
            extras["channel_queue"] = []
            self._audit(key, "channel_queue_cleared", count=count)
            self.store.save()
        return count

    def _channel_next_due(self, entry: dict[str, Any],
                          now: float, tz_offset_min: int) -> float:
        """The next occurrence of a recurring entry strictly
        after ``now``, in the channel's own local time."""
        minutes = _rhythm_parse_hhmm(entry.get("time")) or 0
        shifted = now + int(tz_offset_min or 0) * 60
        day_floor = shifted - (shifted % 86400)
        target = day_floor + minutes * 60 - int(
            tz_offset_min or 0) * 60
        if entry.get("kind") == "weekly":
            parts = _rhythm_local_parts(now, tz_offset_min)
            want = int(entry.get("weekday") or 0)
            target += ((want - int(parts["weekday"])) % 7) \
                * 86400
            if target <= now:
                target += 7 * 86400
        elif target <= now:
            target += 86400
        return float(target)

    def _channel_post_text(self, key: str,
                           entry: dict[str, Any]) -> str:
        """Render one queue entry for posting.  Owner-composed
        posts go through the typography layer as announcements
        (a presented surface — the stamp rides by default under
        the channel's own presentation settings).  Trigger
        cross-posts carry text the source group already
        rendered (a season podium, a war result, a milestone
        line) — re-wrapping would double-render it, so the
        composed text posts as-is with the same stamp rule
        applied through the same style resolution."""
        from .typography import Card, resolve_style, \
            _append_stamp

        text = str(entry.get("text") or "")
        if entry.get("kind") == "trigger":
            style = resolve_style(self.store.settings(key),
                                  "announcement")
            if style.stamp:
                text = _append_stamp(text, style.stamp_text,
                                     "canonical")
            return text
        title = self.channel_title(key) or "announcement"
        card = Card("announcement", title=title, icon="📣")
        for line in text.split("\n"):
            card.line(line)
        return self._card_render(key, card)

    def channel_cycle(self, key: str, *,
                      now: float | None = None,
                      send: Any = None,
                      force: bool = False) -> list[dict[str, Any]]:
        """Post due queue entries for one channel.

        ``send(text, entry) -> (ok, message_id | error)`` is
        the runtime's transport; the surface decides due-ness,
        claim, cadence and rendering, advances the queue only
        on a confirmed send, and journals every post with its
        source.  An unclaimed channel posts NOTHING (§33: the
        claim is the license for system output).  A failed send
        leaves the entry due and stops the run — the next cycle
        retries, nothing is silently dropped."""
        now = now if now is not None else time.time()
        posts: list[dict[str, Any]] = []
        if send is None or not self.is_channel(key):
            return posts
        if not self.claim_state(key)["claimed"]:
            return posts
        extras = self._extras(key)
        last_run = float(extras.get("channel_cycle_at") or 0)
        if not force and now - last_run < self._CHANNEL_CYCLE_GAP_S:
            return posts
        extras["channel_cycle_at"] = now
        cfg = self.channel_cfg(key)
        gap = max(0, int(cfg.get("cadence_min") or 0)) * 60
        last_post = float(extras.get("channel_last_post") or 0)
        queue = list(extras.get("channel_queue") or [])
        due = sorted((e for e in queue
                      if float(e.get("due") or 0) <= now),
                     key=lambda e: float(e.get("due") or 0))
        changed = False
        for entry in due[:self._CHANNEL_POSTS_PER_CYCLE]:
            if gap and last_post and now - last_post < gap:
                break  # the cadence floor spaces posts out
            text = self._channel_post_text(key, entry)
            try:
                ok, info = send(text, dict(entry))
            except Exception as exc:  # transport honesty: a
                ok, info = False, f"{type(exc).__name__}: {exc}"
            if not ok:
                self._audit(key, "channel_post_failed",
                            entry=str(entry.get("id") or ""),
                            entry_kind=str(entry.get("kind") or ""),
                            error=str(info or "")[:120])
                changed = True
                break
            record = {"ts": now, "entry": str(entry.get("id")
                                             or ""),
                      "kind": str(entry.get("kind") or ""),
                      "source": str(entry.get("source") or ""),
                      "trigger": str(entry.get("trigger") or ""),
                      "message_id": str(info or "")}
            ring = list(extras.get("channel_posts") or [])
            ring.append(record)
            extras["channel_posts"] = \
                ring[-self._CHANNEL_POSTS_RING:]
            self._audit(key, "channel_post",
                        entry=record["entry"],
                        entry_kind=record["kind"],
                        source=record["source"],
                        trigger=record["trigger"],
                        message_id=record["message_id"])
            posts.append(record)
            last_post = now
            extras["channel_last_post"] = now
            changed = True
            if entry.get("kind") in ("daily", "weekly"):
                tz = int((self.rhythm_cfg(key) or {}).get(
                    "tz_offset_min") or 0)
                entry["due"] = self._channel_next_due(
                    entry, now, tz)
                entry["last_posted"] = now
            else:
                queue.remove(entry)
        if changed:
            extras["channel_queue"] = queue
            self.store.save()
        return posts

    def _channel_trigger(self, group_key: str, kind: str,
                         text: str, *,
                         now: float | None = None) -> None:
        """Cross-post fan-out: a group event (season podium,
        war result, milestone) is offered to every channel the
        owner linked to this group for this kind.  The SOURCE
        group's claim gates the trigger (§33); the channel's
        own claim is checked again at post time in
        channel_cycle.  The post rides the channel's queue, so
        it is journaled with its source like any other post."""
        now = now if now is not None else time.time()
        if not text or kind not in self.CHANNEL_TRIGGERS:
            return
        if not self.system_output_allowed(group_key):
            return
        for ckey in self.channel_keys():
            links = (self.channel_cfg(ckey).get("links") or {})
            kinds = links.get(group_key) or []
            if kind in kinds:
                self.channel_enqueue(ckey, {
                    "kind": "trigger", "trigger": kind,
                    "text": text, "due": now,
                    "source": f"trigger:{kind}",
                    "source_group": group_key,
                }, now=now)

    def note_channel_count(self, key: str, count: int, *,
                           now: float | None = None) -> bool:
        """Record one subscriber-count sample (taken by the
        runtime from an adapter that actually exposes counts).
        Sampling is throttled so the series is a series, not a
        smear; returns True when the sample landed."""
        now = now if now is not None else time.time()
        extras = self._extras(key)
        ring = list(extras.get("channel_counts") or [])
        if ring and now - float(ring[-1].get("ts") or 0) \
                < self._CHANNEL_SAMPLE_GAP_S:
            return False
        ring.append({"ts": now, "count": int(count)})
        extras["channel_counts"] = \
            ring[-self._CHANNEL_COUNTS_RING:]
        self.store.save()
        return True

    def note_channel_views(self, key: str, message_id: str,
                           views: int, *,
                           now: float | None = None) -> None:
        """Record a post view count where a platform actually
        returned one (a send receipt that carried the counter).
        Counts the platform never returned stay absent, and
        the analytics card says views are not exposed — the
        honest state.  The durable series is the re-read beat
        (:meth:`note_channel_views_read`); this is the
        send-time seed for it."""
        if not message_id:
            return
        extras = self._extras(key)
        seen = dict(extras.get("channel_views") or {})
        seen[str(message_id)] = int(views)
        extras["channel_views"] = seen
        self.store.save()

    def channel_published_ids(self, key: str) -> list[str]:
        """The platform message ids of posts this channel's
        queue has published (the posts ring) — the ids a views
        re-read asks the platform about."""
        extras = self._extras(key)
        return [str(p.get("message_id") or "")
                for p in (extras.get("channel_posts") or [])
                if p.get("message_id")]

    def channel_views_due(self, key: str, *,
                          now: float | None = None) -> bool:
        """Is a views re-read beat due?  Views ride the same
        sample discipline as counts (≤1 per sample gap), so a
        busy message cycle never turns into a platform poll."""
        now = now if now is not None else time.time()
        extras = self._extras(key)
        last = float(extras.get("channel_views_at") or 0)
        return now - last >= self._CHANNEL_SAMPLE_GAP_S

    def note_channel_views_read(self, key: str,
                                views_by_id: dict[str, int], *,
                                now: float | None = None) -> int:
        """Record one views re-read beat: the platform's CURRENT
        counters for published posts, fetched long after send
        time (views grow; a send receipt's number is stale the
        moment it lands).  Only ids the platform actually
        returned a count for are recorded — an id missing from
        the mapping keeps whatever it had, and a channel whose
        posts were never counted keeps views absent.  The beat
        stamps even when every count came back already-known:
        the read happened.  Returns how many counts landed."""
        now = now if now is not None else time.time()
        extras = self._extras(key)
        seen = dict(extras.get("channel_views") or {})
        landed = 0
        for message_id, views in (views_by_id or {}).items():
            if not message_id or views is None:
                continue
            seen[str(message_id)] = int(views)
            landed += 1
        extras["channel_views"] = seen
        extras["channel_views_at"] = now
        self.store.save()
        return landed

    def channel_stats_due(self, key: str, *,
                          now: float | None = None) -> bool:
        """Is a broadcast-stats sample due?  Same ≤1-per-gap
        discipline as counts — a refused sample (a channel too
        small for Telegram stats) stamps the attempt too, so
        refusal never becomes a per-turn retry storm."""
        now = now if now is not None else time.time()
        extras = self._extras(key)
        last = float(extras.get("channel_stats_at") or 0)
        return now - last >= self._CHANNEL_SAMPLE_GAP_S

    def note_channel_stats(self, key: str,
                           stats: dict[str, Any], *,
                           now: float | None = None) -> None:
        """Record one broadcast-stats sample the platform
        actually returned (Telegram stats.getBroadcastStats via
        the userbot): follower level + previous-period level
        and per-post averages.  A landed sample clears any
        recorded refusal — the latest attempt is the state."""
        now = now if now is not None else time.time()
        extras = self._extras(key)
        extras["channel_stats"] = {
            "ts": now,
            "followers": stats.get("followers"),
            "followers_prev": stats.get("followers_prev"),
            "views_per_post": stats.get("views_per_post"),
            "views_per_post_prev": stats.get(
                "views_per_post_prev"),
            "shares_per_post": stats.get("shares_per_post"),
            "reactions_per_post": stats.get(
                "reactions_per_post"),
            "enabled_notifications_pct": stats.get(
                "enabled_notifications_pct"),
        }
        extras["channel_stats_at"] = now
        extras.pop("channel_stats_state", None)
        self.store.save()

    def note_channel_stats_unavailable(self, key: str,
                                       reason: str, *,
                                       now: float | None = None
                                       ) -> None:
        """Record that the platform REFUSED a broadcast-stats
        sample (channel too small, account not admin — Telegram
        says why).  The refusal is a state with a reason and a
        stamp, rendered as absence-with-reason: not a zero,
        not a silent blank, and stamped so the next attempt
        waits for the next sample beat."""
        now = now if now is not None else time.time()
        extras = self._extras(key)
        extras["channel_stats_state"] = {
            "reason": str(reason or "")[:200], "ts": now}
        extras["channel_stats_at"] = now
        self.store.save()

    def channel_analytics(self, key: str) -> dict[str, Any]:
        """The analytics truth for one channel, from recorded
        samples only: subscriber level + growth deltas when
        samples exist, post counts from the journal, view
        totals only when a platform returned views, broadcast
        stats only when a platform granted them.  Absence
        is reported as absence (``subscribers: None``), never
        as zero — "cannot read" and "0 subscribers" are
        different facts and the card keeps them apart.  A
        stats refusal is reported as its reason
        (``stats_unavailable``), distinct from both a sample
        and from never having asked."""
        extras = self._extras(key)
        counts = list(extras.get("channel_counts") or [])
        posts = list(extras.get("channel_posts") or [])
        views = dict(extras.get("channel_views") or {})
        raw_stats = extras.get("channel_stats") or None
        stats = None
        if raw_stats:
            followers = raw_stats.get("followers")
            prev = raw_stats.get("followers_prev")
            stats = {
                "followers": followers,
                "followers_delta": (
                    int(followers) - int(prev)
                    if followers is not None
                    and prev is not None else None),
                "views_per_post": raw_stats.get(
                    "views_per_post"),
                "shares_per_post": raw_stats.get(
                    "shares_per_post"),
                "reactions_per_post": raw_stats.get(
                    "reactions_per_post"),
                "enabled_notifications_pct": raw_stats.get(
                    "enabled_notifications_pct"),
                "sampled_at": float(raw_stats.get("ts") or 0),
            }
        stats_state = extras.get("channel_stats_state") or {}

        def _delta(window: float) -> int | None:
            if len(counts) < 2:
                return None
            latest = counts[-1]
            cutoff = float(latest.get("ts") or 0) - window
            base = None
            for sample in counts[:-1]:
                if float(sample.get("ts") or 0) <= cutoff:
                    base = sample
            if base is None:
                base = counts[0]
            return int(latest.get("count") or 0) - int(
                base.get("count") or 0)

        return {
            "samples": len(counts),
            "subscribers": (int(counts[-1].get("count") or 0)
                            if counts else None),
            "delta_24h": _delta(86400),
            "delta_7d": _delta(7 * 86400),
            "posts": len(posts),
            "last_post_at": (float(posts[-1].get("ts") or 0)
                             if posts else 0.0),
            "views_total": (sum(int(v) for v in views.values())
                            if views else None),
            "views_sampled_at": float(
                extras.get("channel_views_at") or 0),
            "stats": stats,
            "stats_unavailable": (
                str(stats_state.get("reason"))
                if stats_state.get("reason") else None),
            "stats_checked_at": float(
                extras.get("channel_stats_at") or 0),
        }

    # -- leaderboards (§14): one organ, many boards --------------
    #
    # Rows assemble here (the surface owns the store and the
    # economy / estate / conflict organs); the snapshot book
    # and movement math live in group_boards.  Peeking renders
    # movement against the last PUBLISHED snapshot without
    # advancing it; publishing (the group cycle beat, below)
    # records the new baseline.  Reputation is never a metric.

    def _board_book(self) -> BoardBook:
        book = getattr(self, "_boards_book_cache", None)
        if book is None:
            book = BoardBook(self.store.path.parent
                             / "boards.json")
            self._boards_book_cache = book
        return book

    def _board_gang_rows(self, metric: str,
                         limit: int) -> list[dict[str, Any]]:
        """§36 gang-scope rows: the global registry ranked —
        prestige (or chapters / worldwide members), with the
        gang's tier and reach riding as the sub-line.  A frozen
        registry reads as an empty board, never a guess."""
        rows: list[dict[str, Any]] = []
        try:
            records = self.gangs.all_gangs()
        except GangRegistryError:
            return rows
        for record in records:
            view = self.gangs.global_view(record)
            if metric == "prestige":
                value = int(view["prestige"])
            elif metric == "chapters":
                value = int(view["chapter_count"])
            else:
                value = int(view["roster_size"])
            note = "flagship" if view["flagship"] else ""
            sub = (f"{view['tier']} · "
                   f"{view['chapter_count']} chapter(s)")
            if view["hq"]:
                sub += f" · HQ {view['hq']['tier_name']}"
            rows.append({"key": str(view["id"]),
                         "label": str(view["name"]),
                         "value": value, "note": note,
                         "sub": sub})
        rows.sort(key=lambda r: (-int(r["value"]),
                                 str(r["label"]).lower()))
        return rows[:limit]

    def _board_group_rows(self, group_key: str, metric: str,
                          limit: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        members = self.members(group_key)
        if metric == "sparks":
            for held in self.top_holders(group_key,
                                         limit=100000):
                uid = str(held["member"])
                sub = ""
                if held.get("bank"):
                    sub = f"🏦 {int(held['bank'])} banked"
                if held.get("property"):
                    sub = (sub + " · " if sub else "") + \
                        f"🏘️ {int(held['property'])} property"
                rows.append({
                    "key": uid,
                    "label": self._member_label(group_key, uid),
                    "value": int(held.get("worth")
                                 or held["total"]),
                    "note": "", "sub": sub})
        elif metric == "gems":
            for uid in members:
                if self.economy.is_pseudo_member(uid):
                    continue
                value = self.estate.gems_balance(group_key, uid)
                if value > 0:
                    rows.append({
                        "key": uid,
                        "label": self._member_label(group_key,
                                                    uid),
                        "value": int(value), "note": "",
                        "sub": ""})
        elif metric == "messages":
            for uid, record in members.items():
                if self.economy.is_pseudo_member(uid):
                    continue
                value = int(record.get("message_count") or 0)
                if value > 0:
                    rows.append({
                        "key": uid,
                        "label": self._member_label(group_key,
                                                    uid),
                        "value": value, "note": "", "sub": ""})
        elif metric == "contributions":
            for row in self.contribution_ranking(group_key,
                                                 limit=limit):
                rows.append({
                    "key": str(row["user_id"]),
                    "label": str(row.get("label") or ""),
                    "value": int(row.get("counted") or 0),
                    "note": "", "sub": ""})
        elif metric == "gangs":
            for held in self.gang_list(group_key):
                gang = held.get("gang") or {}
                rows.append({
                    "key": str(gang.get("id") or ""),
                    "label": str(gang.get("name") or "a crew"),
                    "value": int(held.get("strength") or 0),
                    "note": f"{len(gang.get('members') or [])} "
                            f"members",
                    "sub": ""})
        rows.sort(key=lambda r: (-int(r["value"]),
                                 str(r["label"]).lower()))
        return rows[:limit]

    def _board_people(self, group_keys: list[str]
                      ) -> dict[str, dict[str, Any]]:
        """Stitch the people across groups (§32 bridge): the
        same platform account in several groups is one person;
        a shared normalized username across platforms stitches
        those accounts into one person too — the way people are
        actually found cross-app.  Display names alone never
        stitch (they are not identities)."""
        people: dict[str, dict[str, Any]] = {}
        by_id: dict[tuple[str, str], str] = {}
        by_name: dict[str, str] = {}
        for gkey in group_keys:
            platform = str(gkey).split(":", 1)[0]
            for uid, record in self.members(gkey).items():
                if self.economy.is_pseudo_member(uid):
                    continue
                username = str(record.get("username") or "") \
                    .strip().lower().lstrip("@")
                pkey = by_id.get((platform, uid))
                if pkey is None and username:
                    pkey = by_name.get(username)
                if pkey is None:
                    pkey = (f"u:{username}" if username
                            else f"{platform}:{uid}")
                    people[pkey] = {
                        "label": "", "platforms": set(),
                        "ids": {}, "username": username}
                person = people[pkey]
                by_id[(platform, uid)] = pkey
                if username:
                    by_name[username] = pkey
                    person["username"] = username
                person["ids"][gkey] = uid
                person["platforms"].add(platform)
                name = str(record.get("name") or "").strip()
                if name and (not person["label"]
                             or person["label"].startswith("@")):
                    person["label"] = name
                elif not person["label"] and username:
                    person["label"] = f"@{username}"
                elif not person["label"]:
                    person["label"] = uid
        return people

    def _board_people_rows(self, group_keys: list[str],
                           metric: str, limit: int
                           ) -> list[dict[str, Any]]:
        people = self._board_people(group_keys)
        worths: dict[str, dict[str, int]] = {}
        if metric == "sparks":
            for gkey in group_keys:
                worths[gkey] = {
                    str(h["member"]): int(h.get("worth")
                                          or h["total"])
                    for h in self.top_holders(gkey,
                                              limit=100000)}
        rows: list[dict[str, Any]] = []
        for pkey, person in people.items():
            total = 0
            for gkey, uid in person["ids"].items():
                record = (self.members(gkey).get(uid) or {})
                if metric == "sparks":
                    total += worths.get(gkey, {}).get(uid, 0)
                elif metric == "messages":
                    total += int(record.get("message_count")
                                 or 0)
                elif metric == "contributions":
                    total += int((record.get("contributions")
                                  or {}).get("counted") or 0)
            if total > 0:
                rows.append({
                    "key": pkey, "label": person["label"],
                    "value": total,
                    "note": " · ".join(sorted(
                        person["platforms"])),
                    "sub": ""})
        rows.sort(key=lambda r: (-int(r["value"]),
                                 str(r["label"]).lower()))
        return rows[:limit]

    def _board_game_rows(self, parsed: dict[str, Any],
                         games_rows: list[dict[str, Any]] | None,
                         limit: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for profile in games_rows or []:
            name = str(profile.get("name")
                       or profile.get("player_key") or "")
            platform = str(profile.get("platform") or "")
            if parsed["scope"] == "games":
                metric = parsed["metric"]
                value = int(profile.get(metric) or 0)
                if value <= 0:
                    continue
                sub = ""
                if metric == "xp" and profile.get("level"):
                    sub = f"level {int(profile['level'])}"
                rows.append({"key": str(profile.get(
                    "player_key") or ""), "label": name,
                    "value": value, "note": platform,
                    "sub": sub})
                continue
            game = parsed["game"]
            stats = (profile.get("per_game") or {}).get(game) \
                or {}
            if int(stats.get("played") or 0) <= 0:
                continue
            metric = parsed["metric"]
            if metric == "elo":
                if int(stats.get("elo_games") or 0) <= 0:
                    continue  # unrated is not a standing
                value = int(stats.get("elo") or 0)
                sub = f"{int(stats['elo_games'])} rated games"
            else:
                value = int(stats.get(metric) or 0)
                sub = f"{int(stats.get('played') or 0)} played"
            if value <= 0 and metric != "elo":
                continue
            rows.append({"key": str(profile.get("player_key")
                                    or ""), "label": name,
                         "value": value, "note": platform,
                         "sub": sub})
        rows.sort(key=lambda r: (-int(r["value"]),
                                 str(r["label"]).lower()))
        return rows[:limit]

    def board_rows(self, board_id: str, *, group_key: str = "",
                   games_rows: list[dict[str, Any]] | None = None,
                   limit: int = _BOARD_LIMIT
                   ) -> list[dict[str, Any]] | None:
        """The ranked rows for a board, or None when the board
        id is invalid — or names games data this call doesn't
        carry (the cycle has no games provider; /boards does).
        """
        parsed = _parse_board_id(board_id)
        if parsed is None:
            return None
        scope = parsed["scope"]
        if scope == "group":
            if not group_key:
                return None
            return self._board_group_rows(group_key,
                                          parsed["metric"],
                                          limit)
        if scope in ("game", "games"):
            if games_rows is None:
                return None
            return self._board_game_rows(parsed, games_rows,
                                         limit)
        if scope == "gangs":
            return self._board_gang_rows(parsed["metric"],
                                         limit)
        group_keys = [k for k in sorted(self.store.data)
                      if not self.is_channel(k)]
        if scope == "platform":
            group_keys = [k for k in group_keys
                          if k.split(":", 1)[0]
                          == parsed["platform"]]
        elif scope == "community":
            group_keys = [
                k for k in group_keys
                if str(self.store.settings(k).get(
                    "community_id") or "")
                == parsed["community"]]
        return self._board_people_rows(group_keys,
                                       parsed["metric"], limit)

    def render_board(self, board_id: str, *, group_key: str = "",
                     games_rows: list[dict[str, Any]] | None = None,
                     publish: bool = False,
                     now: float | None = None,
                     limit: int = 10) -> str | None:
        """Render a board through typography (§27): the board
        preset for game-flavored boards, clean otherwise;
        movement markers read the last published snapshot.
        ``publish=True`` records the rendered ranking as the
        new baseline AFTER the text is composed — a peek never
        advances it."""
        from .typography import Card, render_card, sparks

        parsed = _parse_board_id(board_id)
        if parsed is None:
            return None
        rows = self.board_rows(board_id, group_key=group_key,
                               games_rows=games_rows,
                               limit=_BOARD_LIMIT)
        if rows is None:
            return None
        book = self._board_book()
        marks = _board_movement(book.latest(parsed["id"]), rows)
        meta_fmt = _board_metric_format(parsed)

        def _fmt(value: int) -> str:
            if meta_fmt == "sparks":
                return sparks(value)
            if meta_fmt == "gems":
                return f"💎 {value:,} gems"
            return f"{value:,}"

        card = Card("game" if _board_game_flavored(parsed)
                    else "leaderboard",
                    title=_board_title(parsed), icon="🏆")
        if not rows:
            card.line("no standings yet — nothing honest to "
                      "rank.")
        else:
            medals = ["🥇", "🥈", "🥉"]
            stats = []
            for i, row in enumerate(rows[:limit]):
                mark = medals[i] if i < 3 else f"{i + 1}."
                label = str(row.get("label") or "")
                move = marks.get(str(row.get("key") or ""))
                if move:
                    label = f"{move} {label}"
                if row.get("note"):
                    label = f"{label} ({row['note']})"
                value = _fmt(int(row.get("value") or 0))
                if row.get("sub"):
                    value = f"{value} · {row['sub']}"
                stats.append((f"{mark} {label}", value, True))
            card.stats(stats)
        if group_key:
            text = self._card_render(group_key, card)
        else:
            text = render_card(card, platform="telegram-bot",
                               settings=None).text
        if publish:
            book.publish(parsed["id"], rows, now=now)
            if group_key:
                self._audit(group_key, "board_published",
                            board=parsed["id"])
        return text

    def boards_cycle(self, group_key: str, *,
                     now: float | None = None) -> list[str]:
        """The board beat on the message cycle: at the group's
        local publish hour, post its configured daily boards
        (and weekly boards on the local Monday), each render
        publishing its snapshot.  Cycle posts travel the
        caller's notices list, so silent hours hold them like
        every other cycle post; §33 gates the beat itself."""
        now = now if now is not None else time.time()
        if not self.system_output_allowed(group_key):
            return []
        cfg = self.store.settings(group_key).get("boards") \
            or {}
        daily = [str(s) for s in (cfg.get("daily") or [])]
        weekly = [str(s) for s in (cfg.get("weekly") or [])]
        if not daily and not weekly:
            return []
        tz = int((self.rhythm_cfg(group_key) or {}).get(
            "tz_offset_min") or 0)
        parts = _rhythm_local_parts(now, tz)
        if int(parts["hour"]) < int(cfg.get("hour", 20)):
            return []
        extras = self._extras(group_key)
        state = extras.setdefault("boards_state", {})
        due: list[str] = []
        if daily and state.get("daily") != parts["day"]:
            state["daily"] = parts["day"]
            due.extend(daily)
        if weekly and int(parts["weekday"]) == 0:
            week = time.strftime("%G-W%V", time.gmtime(
                now + tz * 60))
            if state.get("weekly") != week:
                state["weekly"] = week
                due.extend(weekly)
        out: list[str] = []
        for spec in due:
            board_id = spec if ":" in spec else f"group:{spec}"
            text = self.render_board(board_id,
                                     group_key=group_key,
                                     publish=True, now=now)
            if text:
                out.append(text)
            else:
                # e.g. a games board: the cycle carries no
                # games provider, so the publish is skipped —
                # journaled, never faked.
                self._audit(group_key, "board_publish_skipped",
                            board=board_id)
        if due:
            self.store.save()
        return out


def default_surface(workspace_dir: str | Path) -> GroupSurface:
    """The runtime's surface: store under the workspace's data dir."""
    from .owner_system import default_registry

    root = Path(workspace_dir) / "data" / "social"
    return GroupSurface(
        GroupStore(root / "groups.json").load(),
        owner_registry=default_registry(workspace_dir))
