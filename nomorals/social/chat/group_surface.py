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

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ...community.welcome_moderation import (
    BanList,
    ModerationConfig,
    Moderator,
    WelcomeConfig,
    render_goodbye,
    render_welcome,
)

AUDIT_RING = 500
CHANGE_RING = 200


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
                       if k in merged})
        mod = dict(merged["moderation"])
        mod.update((stored.get("moderation") or {}))
        merged["moderation"] = mod
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
        return render_welcome(name or username or "there", config)

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
        return render_goodbye(
            member.get("name") or member.get("username") or "there",
            config)

    # ---- the message pipeline ------------------------------------
    def handle_message(self, group_key: str, user_id: str, *,
                       username: str = "", name: str = "",
                       text: str = "", message_id: str = "",
                       ts: float | None = None) -> dict[str, Any]:
        """One ordinary group message: ledger observation +
        moderation.  Returns ``{"changes": [...], "actions":
        [...]}`` — actions are ModerationAction dicts for the
        caller to execute (delete / warn / mute / ban)."""
        now = ts if ts is not None else time.time()
        changes = self._observe_identity(
            group_key, user_id, username=username, name=name, ts=now)
        entry = self.store.group(group_key)
        member = entry["members"][user_id]
        member["message_count"] = int(
            member.get("message_count") or 0) + 1
        actions: list[dict[str, Any]] = []
        if text:
            for action in self._moderator(group_key).check_message(
                    group_key, user_id, text, ts=now):
                actions.append(action.to_dict())
                self._audit(group_key, "moderation",
                            action=action.kind, rule=action.rule,
                            reason=action.reason, user_id=user_id)
        if message_id:
            member["last_message_id"] = str(message_id)
        self._economy_activity(group_key, user_id, now)
        self.store.save()
        return {"changes": changes, "actions": actions}

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
                    if m}
        total = sum(balances.values())
        top = max(balances.values()) if balances else 0
        activities = econ._load_json(group_key, "activities", []) or []
        active = {a.get("member") for a in activities
                  if float(a.get("ts") or 0) >= now - 7 * 86400}
        return {"holders": len(balances), "circulating": total,
                "treasury": econ.treasury_balance(group_key),
                "top_share": (round(top / total, 3)
                              if total else 0.0),
                "active_7d": len(active - {None, ""})}

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
        """One admin /warn on a member: counted in the member
        record, audited, and the count returned (the caller runs
        the escalation ladder against the warn threshold)."""
        entry = self.store.group(group_key)
        member = entry["members"].setdefault(
            user_id, {"user_id": user_id, "message_count": 0})
        member["warn_count"] = int(member.get("warn_count") or 0) + 1
        member.setdefault("warns", []).append(
            {"ts": time.time(), "by": by, "reason": reason[:200]})
        member["warns"] = member["warns"][-20:]
        self._audit(group_key, "warn", user_id=user_id, by=by,
                    count=member["warn_count"], reason=reason[:200])
        self.store.save()
        return int(member["warn_count"])

    def clear_warns(self, group_key: str, user_id: str) -> None:
        member = self.store.group(group_key)["members"].get(user_id)
        if member:
            member["warn_count"] = 0
            member["warns"] = []
            self.store.save()

    # ---- read models ----------------------------------------------
    def member(self, group_key: str, user_id: str
               ) -> dict[str, Any] | None:
        return self.store.group(group_key)["members"].get(user_id)

    def members(self, group_key: str) -> dict[str, Any]:
        return dict(self.store.group(group_key)["members"])

    def changes(self, group_key: str, limit: int = 50
                ) -> list[dict[str, Any]]:
        return list(
            self.store.group(group_key)["changes"])[-limit:]

    def rules_text(self, group_key: str) -> str:
        """The published rules, rendered from the SAME settings
        the enforcer reads (DESIGN §12: one config, two readers).
        """
        import html as _html

        settings = self.store.settings(group_key)
        lines = ["📜 <b>Group rules</b>"]
        rules = settings.get("rules") or []
        if rules:
            lines.extend(f"{i}. {_html.escape(str(rule))}"
                         for i, rule in enumerate(rules, 1))
        else:
            lines.append("No written rules yet — be decent, no "
                         "spam, no flooding.")
        mod = settings["moderation"]
        lines.append(
            f"Moderation: flood limit {mod['flood_max']} messages "
            f"per {int(mod['flood_window_s'])}s · repeats ×"
            f"{mod['repeat_threshold']} are removed")
        if mod.get("link_allowlist"):
            lines.append("Links: only allowlisted domains are "
                         "permitted.")
        return "\n".join(lines)


def default_surface(workspace_dir: str | Path) -> GroupSurface:
    """The runtime's surface: store under the workspace's data dir."""
    root = Path(workspace_dir) / "data" / "social"
    return GroupSurface(GroupStore(root / "groups.json").load())
