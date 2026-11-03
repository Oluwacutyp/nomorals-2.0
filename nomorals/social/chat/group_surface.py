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
from .group_defense import (
    DEFAULTS as DEFENSE_DEFAULTS,
)
from .typography import DEFAULT_PRESENTATION, notice_line, sparks
from .group_defense import (
    DefenseConfig,
    add_strike,
    check_antichannel,
    check_antitag,
    consequence_text,
    ladder_rung,
    recent_strikes,
    render_warning,
)

AUDIT_RING = 500
CHANGE_RING = 200
MSG_CACHE_RING = 200
_ACTION_RANK = {"allow": 0, "warn": 1, "delete": 2, "mute": 3,
                "kick": 4, "ban": 5}


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
        "games_enabled": True,
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
                     "members": {}, "changes": [], "audit": []}
            self.data[group_key] = entry
        return entry

    def settings(self, group_key: str) -> dict[str, Any]:
        merged = _default_settings()
        stored = self.group(group_key).get("settings") or {}
        merged.update({k: v for k, v in stored.items()
                       if k in merged
                       and k not in ("defense", "presentation",
                                     "estate")})
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

    @property
    def signatures(self) -> SignatureStore:
        """Learned anticrash shapes, persisted beside the group
        store (derived data — the store quarantines itself on
        corruption and relearns by observation)."""
        if self._signatures is None:
            self._signatures = SignatureStore(
                self.store.path.parent / "payload_signatures.json")
        return self._signatures

    def defense_config(self, group_key: str) -> DefenseConfig:
        return DefenseConfig.from_settings(
            self.store.settings(group_key))

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

    def record_join(self, group_key: str, user_id: str, *,
                    username: str = "", name: str = "",
                    group_name: str = "",
                    ts: float | None = None) -> str:
        """Record a join; return the welcome text to send ("" when
        welcome is off).  The text renders from the group's own
        settings through the community welcome flow."""
        now = ts if ts is not None else time.time()
        self._observe_identity(group_key, user_id,
                               username=username, name=name, ts=now)
        entry = self.store.group(group_key)
        member = entry["members"][user_id]
        member["joined_ts"] = now
        member["left_ts"] = None
        self._audit(group_key, "join", user_id=user_id,
                    username=username, name=name)
        self.store.save()
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
        warn_threshold = int(settings.get("warn_threshold") or 3)
        suppressed = not cfg.autonomy
        privileged = sender_is_admin or sender_is_operator
        mention = name or (f"@{username}" if username else user_id)
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

        def strike(stage: str, rule: str, action_kind: str) -> int:
            """One strike per stage per message on the shared
            spine; landing on a rung appends the ladder action.
            Under autonomy OFF the spine is observe-only too: the
            audits carry the verdict, the ledger does not move."""
            if privileged or suppressed or stage in struck:
                return strike_counts.get(rule, 0)
            struck.add(stage)
            count = add_strike(member, rule=rule,
                               action=action_kind, now=now,
                               window_days=cfg.window_days)
            strike_counts[rule] = count
            self._audit(group_key, "strike", user_id=user_id,
                        rule=rule, count=count,
                        window_days=cfg.window_days)
            rung = ladder_rung(count, cfg,
                               warn_threshold=warn_threshold)
            if rung and all(_ACTION_RANK.get(a.get("kind"), 0)
                            < _ACTION_RANK[rung] for a in actions):
                if rung == "mute":
                    actions.append({
                        "kind": "delete", "rule": "escalation",
                        "reason": f"strike {count} — {rule}",
                        "user_id": user_id, "group_id": group_key})
                actions.append({
                    "kind": rung, "rule": "escalation",
                    "reason": f"strike {count} — {rule}",
                    "user_id": user_id, "group_id": group_key,
                    "minutes": cfg.mute_minutes,
                    "_strike_count": count, "_strike_rule": rule})
            return count

        def finish(blocked: bool) -> dict[str, Any]:
            # ONE public warning per offense, on the strongest
            # action, worded by the organ (never meta-narration).
            if actions and not suppressed:
                top = max(actions, key=lambda a: _ACTION_RANK.get(
                    a.get("kind"), 0))
                rule = top.get("_strike_rule") or top.get(
                    "rule") or ""
                count = top.get("_strike_count") \
                    or strike_counts.get(rule, 0)
                if top.get("kind") in ("warn", "delete", "mute",
                                       "kick", "ban"):
                    top["warning"] = render_warning(
                        mention, rule,
                        consequence_text(
                            executed=top["kind"], count=count,
                            cfg=cfg,
                            warn_threshold=warn_threshold))
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
                # costs sparks — paid to the treasury.
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
        member = self.store.group(group_key)["members"].get(
            user_id) or {}
        strikes = recent_strikes(member, now=now,
                                 window_days=cfg.window_days)
        return {"count": len(strikes), "strikes": strikes,
                "window_days": cfg.window_days}

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
                   now: float | None = None) -> int:
        """The moderation fine sink: settings-driven, paid to the
        treasury, capped at what the member actually holds.  One
        fine per punishment event — a member already muted does
        not get fined again for the same episode (10-minute
        spacing between fines for one member)."""
        now = now if now is not None else time.time()
        try:
            amount = int(self.store.settings(
                group_key).get("fine_amount") or 0)
        except Exception:  # noqa: BLE001 - settings never break chat
            return 0
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
