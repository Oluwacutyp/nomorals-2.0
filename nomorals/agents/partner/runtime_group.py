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
            return True
        if service == "leave":
            left = meta.get("left_member") or {}
            if isinstance(left, dict) and left.get("id"):
                text = surface.record_leave(
                    key, str(left.get("id") or ""),
                    group_name=str(message.chat.title or ""))
                if text:
                    self._send(message.chat, text)
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
