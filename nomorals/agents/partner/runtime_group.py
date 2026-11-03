"""RuntimeGroupMixin: the group surface in the live event path
(DESIGN §1, P1 — Devon Telegram group & community system).

Group chats are claimed by the surface BEFORE conversation:
service events (join/leave) run the welcome/ledger organs and are
consumed; ordinary messages are observed into the member ledger
(identity snapshots + change detection) and run through the
moderation pipeline, whose actions this mixin executes through
the platform adapter's admin surface where it has one:

- ``delete`` → ``admin_delete``; ``ban`` → ``admin_ban``;
  ``mute`` → ``admin_restrict`` (timed, 1h by default posture);
- ``warn`` → one short public notice (moderation is an event
  Devon owns — the quiet-by-default law names it);
- an action the platform path cannot execute is recorded in the
  group's held audit log as unexecuted, with the reason — never
  faked, never narrated into the chat.

A message deleted or banned by moderation is consumed: Devon
does not also converse with it.  Everything else flows on to the
normal pipeline (claimed-once, games, conversation).
"""

from __future__ import annotations

import logging
import time
from typing import Any

_log = logging.getLogger(__name__)


class RuntimeGroupMixin:
    """Group-surface turn for PartnerRuntime._process_inner."""

    def _group_surface(self):  # type: ignore[no-untyped-def]
        surface = getattr(self, "_group_surface_obj", None)
        if surface is None:
            from ...social.chat.group_surface import default_surface

            ws_dir = getattr(self.context.settings, "workspace_dir", ".")
            surface = default_surface(ws_dir)
            self._group_surface_obj = surface
        return surface

    def _group_adapter(self, platform: str):  # type: ignore[no-untyped-def]
        gateway = getattr(self, "gateway", None)
        adapters = getattr(gateway, "adapters", {}) or {}
        return adapters.get(platform)

    def _group_surface_turn(self, message) -> bool:  # type: ignore[no-untyped-def]
        """Run one group message through the surface.  Returns
        True when the message is fully consumed (service event,
        or moderated away)."""
        surface = self._group_surface()
        key = message.chat.key
        meta = message.meta or {}
        service = str(meta.get("service") or "")

        if service == "join":
            welcomed = []
            for member in meta.get("new_members") or []:
                if not isinstance(member, dict):
                    continue
                text = surface.record_join(
                    key, str(member.get("id") or ""),
                    username=str(member.get("username") or ""),
                    name=str(member.get("name") or ""),
                    group_name=str(message.chat.title or ""))
                if text:
                    welcomed.append(text)
            for text in welcomed:
                self._send(message.chat, text)
            self._clean_service_message(surface, message)
            return True
        if service == "leave":
            left = meta.get("left_member") or {}
            if isinstance(left, dict) and left.get("id"):
                text = surface.record_leave(
                    key, str(left.get("id") or ""),
                    group_name=str(message.chat.title or ""))
                if text:
                    self._send(message.chat, text)
            self._clean_service_message(surface, message)
            return True

        user_id = str(message.sender_id or message.sender or "")
        if not user_id:
            return False
        result = surface.handle_message(
            key, user_id,
            username=str(message.sender_username or ""),
            name=str(message.sender or ""),
            text=str(message.text or ""))
        actions = result.get("actions") or []
        if not actions:
            return False
        adapter = self._group_adapter(message.chat.platform)
        consumed = False
        for action in actions:
            kind = str(action.get("kind") or "")
            executed = self._execute_group_action(
                surface, key, adapter, message, action)
            if kind == "warn" and executed:
                who = (message.sender_username
                       and f"@{message.sender_username}") \
                    or message.sender or "there"
                self._send(
                    message.chat,
                    f"⚠️ {who} — {action.get('reason') or 'warning'}")
            if kind in ("delete", "ban") and executed:
                consumed = True
        return consumed

    def _clean_service_message(self, surface, message) -> None:  # type: ignore[no-untyped-def]
        """Service-message hygiene (P1): the raw "X joined the
        group" service message is deleted after the surface has
        handled it — the welcome is the visible artifact, the
        platform's own notice is clutter.  Setting-gated,
        best-effort, audited on success."""
        try:
            settings = surface.store.settings(message.chat.key)
            if not settings.get("clean_service"):
                return
            if not message.message_id:
                return
            adapter = self._group_adapter(message.chat.platform)
            fn = getattr(adapter, "admin_delete", None) \
                if adapter else None
            if fn is None:
                return
            fn(message.chat, str(message.message_id))
            surface._audit(message.chat.key, "service_cleaned",
                           message_id=str(message.message_id))
            surface.store.save()
        except Exception:  # noqa: BLE001 - hygiene never breaks joins
            _log.debug("group service cleanup failed", exc_info=True)

    # ---- admin gate ------------------------------------------------
    def _group_admin_ok(self, message) -> bool:  # type: ignore[no-untyped-def]
        """Operator (on surfaces where operator power exists) OR
        the platform's own word that this sender admins the group.
        On the gated public bot nobody is an operator — surface law
        — so platform admin status is the authority there, exactly
        how group-helper bots work."""
        try:
            if self._is_operator(message):
                return True
        except Exception:  # noqa: BLE001 - fall to platform check
            pass
        adapter = self._group_adapter(message.chat.platform)
        if adapter is None:
            return False
        fn = getattr(adapter, "admin_member", None) or getattr(
            adapter, "admin_member_mtproto", None)
        if fn is None:
            return False
        try:
            info = fn(message.chat, str(message.sender_id or ""))
        except Exception:  # noqa: BLE001 - no proof, no admin
            return False
        if not isinstance(info, dict):
            return False
        status = info.get("status") or info.get("role") or ""
        if not status and isinstance(info.get("result"), dict):
            status = info["result"].get("status") or ""
        return str(status).lower() in (
            "creator", "administrator", "admin", "owner")

    # ---- group commands (P1 slice 2) --------------------------------
    def _control_group_help(self, message) -> str:  # type: ignore[no-untyped-def]
        """Bare /help INSIDE a group (owner order 2026-11-03):
        list what is actually available in this group — the member
        surface and the admin surface side by side, rendered from
        this group's live state, not the global owner catalog.
        Everyone sees both sections; the admin commands themselves
        stay gated at execution."""
        key = message.chat.key
        surface = self._group_surface()
        settings = surface.store.settings(key)
        members = surface.members(key)
        title = message.chat.title or "this group"
        lines = [
            f"🛠️ {title} — what I do here",
            "",
            "for everyone:",
            "  /rules — the group rules",
            "  /member [@name] — a member's card: tenure, activity, "
            "username/name changes",
            "  /ping — check I'm alive, and how fast I answered",
            "  🎮 games — /ttt /connect4 /reversi /checkers /mines "
            "and the rest of the board; /game lists them, moves are "
            "plain text (a1, b2 …). Some games queue you against "
            "players on other platforms.",
            "  /help <command> — the full page for any command",
            "",
            "for admins:",
            "  /gsettings — welcome & goodbye, the rules list, "
            "flood/link moderation, service-message cleanup",
            "  /glog — the group audit log (I keep it; I only show "
            "it when an admin asks)",
            "  I moderate on my own: floods, repeat spam and "
            "off-allowlist links get deleted; repeat offenders are "
            "muted, then banned — all tuned in /gsettings.",
            "",
            f"this group: {len(members)} members in my ledger · "
            f"{len(settings.get('rules') or [])} rules · welcome "
            f"{'on' if settings['welcome_enabled'] else 'off'} · "
            f"link moderation "
            f"{'allowlist' if (settings['moderation'].get('link_allowlist')) else 'open'}",
        ]
        return "\n".join(lines)

    def _group_key_or_none(self, message):  # type: ignore[no-untyped-def]
        if message is None:
            return None
        from ...social.chat.base import ChatKind

        if message.chat.kind != ChatKind.GROUP:
            return None
        return message.chat.key

    def _control_rules(self, message) -> str:  # type: ignore[no-untyped-def]
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        return self._group_surface().rules_text(key)

    def _control_member(self, arg, message) -> str:  # type: ignore[no-untyped-def]
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        surface = self._group_surface()
        members = surface.members(key)
        target = None
        wanted = str(arg or "").strip().lstrip("@").lower()
        if wanted:
            for uid, member in members.items():
                if uid == wanted or str(
                        member.get("username") or "").lower() == wanted \
                        or str(member.get("name") or "").lower() == wanted:
                    target = member
                    break
            if target is None:
                return (f"I haven't seen {arg} in this group yet — "
                        f"the ledger only knows people who've been "
                        f"active or joined since I started watching.")
        else:
            target = members.get(str(message.sender_id or ""))
            if target is None:
                return "I don't have a card for you in this group yet."
        return _member_card(surface, key, target)

    def _control_glog(self, arg, message) -> str:  # type: ignore[no-untyped-def]
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        if not self._group_admin_ok(message):
            return "the group log is for admins."
        try:
            n = int(str(arg or "10").strip() or 10)
        except ValueError:
            n = 10
        entries = self._group_surface().audit(key, limit=max(1, min(n, 50)))
        if not entries:
            return "nothing in the group log yet."
        lines = ["📋 group log (held by me, shown on demand):"]
        for entry in entries:
            lines.append(_audit_line(entry))
        return "\n".join(lines)

    def _control_gsettings(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        if not self._group_admin_ok(message):
            return "group settings are for admins."
        surface = self._group_surface()
        text = str(tail or "").strip()
        if not text:
            s = surface.store.settings(key)
            mod = s["moderation"]
            return (
                f"⚙️ group settings\n"
                f"welcome: {'on' if s['welcome_enabled'] else 'off'} · "
                f"goodbye: {'on' if s['goodbye_enabled'] else 'off'} · "
                f"service cleanup: {'on' if s['clean_service'] else 'off'}\n"
                f"rules: {len(s.get('rules') or [])} · "
                f"flood: {mod['flood_max']} per "
                f"{int(mod['flood_window_s'])}s · repeats ×"
                f"{mod['repeat_threshold']} · links: "
                f"{', '.join(mod.get('link_allowlist') or []) or 'open'}\n"
                f"change: /gsettings welcome on|off · rules add "
                f"<text> · rules clear · flood <n> · links "
                f"<domains>|off · clean on|off")
        parts = text.split(None, 2)
        head = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""
        rest2 = parts[2] if len(parts) > 2 else ""
        on = rest.lower() == "on"
        if head == "welcome" and rest.lower() in ("on", "off"):
            surface.update_settings(key, welcome_enabled=on)
            return f"welcome {'on' if on else 'off'}."
        if head == "goodbye" and rest.lower() in ("on", "off"):
            surface.update_settings(key, goodbye_enabled=on)
            return f"goodbye {'on' if on else 'off'}."
        if head == "clean" and rest.lower() in ("on", "off"):
            surface.update_settings(key, clean_service=on)
            return f"service cleanup {'on' if on else 'off'}."
        if head == "rules" and rest.lower() == "add" and rest2:
            current = surface.store.settings(key)
            rules = list(current.get("rules") or []) + [rest2]
            surface.update_settings(key, rules=rules)
            return f"rule added ({len(rules)} total)."
        if head == "rules" and rest.lower() == "clear":
            surface.update_settings(key, rules=[])
            return "rules cleared."
        if head == "flood" and rest.isdigit():
            current = surface.store.settings(key)
            mod = dict(current["moderation"])
            mod["flood_max"] = max(2, int(rest))
            surface.update_settings(key, moderation=mod)
            return f"flood limit {mod['flood_max']} per window."
        if head == "links":
            current = surface.store.settings(key)
            mod = dict(current["moderation"])
            if rest.lower() == "off" or not rest:
                mod["link_allowlist"] = []
                surface.update_settings(key, moderation=mod)
                return "links open (no allowlist)."
            domains = [d.strip().lower() for d in
                       (rest + " " + rest2).replace(",", " ").split()
                       if d.strip()]
            mod["link_allowlist"] = domains
            surface.update_settings(key, moderation=mod)
            return f"link allowlist: {', '.join(domains)}."
        return ("I didn't follow that — /gsettings alone shows the "
                "current settings and the change grammar.")

    def _execute_group_action(self, surface, key, adapter, message,
                              action) -> bool:  # type: ignore[no-untyped-def]
        """Execute one moderation action through the adapter's
        admin surface; audit the outcome either way.  Returns True
        only when the platform actually performed it."""
        kind = str(action.get("kind") or "")
        user_id = str(action.get("user_id") or "")
        method = {"delete": "admin_delete", "ban": "admin_ban",
                  "mute": "admin_restrict"}.get(kind)
        if method is None:
            return kind == "warn"  # warn executes as the notice
        fn = getattr(adapter, method, None) if adapter else None
        if fn is None:
            surface._audit(
                key, "moderation_unexecuted", action=kind,
                rule=action.get("rule"), user_id=user_id,
                reason=f"adapter has no {method}")
            surface.store.save()
            return False
        try:
            if kind == "delete":
                fn(message.chat, str(message.message_id or ""))
            elif kind == "ban":
                fn(message.chat, user_id)
            else:
                # A one-hour mute; the two Telegram adapters spell
                # it differently (minutes vs until) — speak each
                # one's own grammar.
                try:
                    fn(message.chat, user_id, minutes=60)
                except TypeError:
                    fn(message.chat, user_id,
                       until=int(time.time()) + 3600)
        except Exception as exc:  # noqa: BLE001 - audited, not raised
            surface._audit(
                key, "moderation_unexecuted", action=kind,
                rule=action.get("rule"), user_id=user_id,
                reason=f"{type(exc).__name__}: {exc}")
            surface.store.save()
            return False
        surface._audit(key, "moderation_executed", action=kind,
                       rule=action.get("rule"), user_id=user_id)
        surface.store.save()
        return True


def _fmt_ts(ts) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M UTC",
                             time.gmtime(float(ts)))
    except (TypeError, ValueError):
        return "?"


def _member_card(surface, key: str, member: dict) -> str:
    name = member.get("name") or member.get("username") or "member"
    handle = (f"@{member['username']}" if member.get("username")
              else "no username")
    lines = [f"👤 {name} ({handle})"]
    joined = member.get("joined_ts") or member.get("first_seen")
    lines.append(f"joined: {_fmt_ts(joined)} · messages: "
                 f"{member.get('message_count') or 0} · last seen: "
                 f"{_fmt_ts(member.get('last_seen'))}")
    if member.get("left_ts"):
        lines.append(f"left: {_fmt_ts(member['left_ts'])}")
    changes = [c for c in surface.changes(key, limit=200)
               if c.get("user_id") == member.get("user_id")]
    for change in changes[-3:]:
        lines.append(
            f"{change.get('kind', 'changed')}: "
            f"{change.get('old') or '—'} → {change.get('new') or '—'} "
            f"({_fmt_ts(change.get('ts'))})")
    return "\n".join(lines)


def _audit_line(entry: dict) -> str:
    kind = entry.get("kind", "?")
    bits = [_fmt_ts(entry.get("ts")), kind]
    for field_name in ("action", "rule", "user_id", "change_kind",
                       "reason"):
        if entry.get(field_name):
            bits.append(f"{field_name}={entry[field_name]}")
    if entry.get("old") or entry.get("new"):
        bits.append(f"{entry.get('old') or '—'} → "
                    f"{entry.get('new') or '—'}")
    return " · ".join(str(b) for b in bits)
