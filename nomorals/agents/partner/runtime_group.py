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

    def _send(self, chat, text: str, *, buttons=None,  # type: ignore[no-untyped-def]
              parse_mode: str = ""):
        """The mixin's own outbound channel (welcomes, goodbyes,
        notices).  This method was CALLED by every welcome path
        since P1 but defined NOWHERE on PartnerRuntime — the live
        bug found 2026-11-03: every join's welcome died on
        AttributeError, caught upstream and logged, never sent.
        Defined here, on the mixin that calls it, riding the same
        gateway contract (buttons + parse_mode) as every surface.
        Returns the gateway's SendResult (callers that need the
        posted message's id — the notice layer — read it there;
        everyone else ignores it, as before)."""
        return self.gateway.send(chat.platform, chat, str(text),
                                 buttons=buttons,
                                 parse_mode=parse_mode)

    def _send_best_effort(self, chat, text: str, **kw):  # type: ignore[no-untyped-def]
        """A side-send from a control handler (a notice for an
        action whose RECORD is already durable on the surface):
        the send is the announcement, never the state — a
        transport that can't take it here degrades to the debug
        log instead of failing the command that caused it."""
        try:
            return self._send(chat, text, **kw)
        except Exception:  # noqa: BLE001 - announcement, not state
            _log.debug("group side-send failed", exc_info=True)
            return None

    # ---- the system-output gate (DESIGN §33) --------------------
    def _system_send_ok(self, surface, key, message,  # type: ignore[no-untyped-def]
                        what: str) -> bool:
        """§33, evaluated at send time: may Devon's BOT SYSTEMS
        post this into this room, through THIS session?  The
        surface's one helper decides (owner designation, the
        bot's own standing, the sending session's platform).
        A refusal is a journaled non-event in the group's held
        audit — never a post, never a fake success.  The
        as-him conversational path never comes through here."""
        platform = str(getattr(message.chat, "platform", "")
                       or "")
        if surface.system_output_allowed(key,
                                         via_platform=platform):
            return True
        surface._audit(key, "system_output_suppressed",
                       what=what, via=platform)
        surface.store.save()
        return False

    _BOT_CLAIM_TTL_S = 600

    def _probe_bot_claim(self, surface, key, message) -> None:  # type: ignore[no-untyped-def]
        """Keep the room's bot standing (§33a) honest: when a
        BOT-platform session serves this group, verify the
        bot's own member status through the adapter and stamp
        the result on the surface — a demotion observed here
        lapses the claim for every session at once.  Probed at
        most once per room per TTL (the fact changes rarely);
        an adapter that cannot say leaves the stamp alone."""
        platform = str(getattr(message.chat, "platform", "")
                       or "")
        if platform not in ("telegram-bot", "telegram-personal"):
            return
        probes = getattr(self, "_bot_claim_probes", None)
        if probes is None:
            probes = self._bot_claim_probes = {}
        now = time.time()
        if now - float(probes.get(key) or 0) \
                < self._BOT_CLAIM_TTL_S:
            return
        adapter = self._group_adapter(platform)
        bot_id = getattr(adapter, "_bot_id", None) \
            if adapter is not None else None
        fn = getattr(adapter, "admin_member", None) \
            if adapter is not None else None
        if bot_id is None or fn is None:
            return
        probes[key] = now
        try:
            info = fn(message.chat, str(bot_id))
        except Exception:  # noqa: BLE001 - unproven, unstamped
            return
        if not isinstance(info, dict):
            return
        status = info.get("status") or info.get("role") or ""
        if not status and isinstance(info.get("result"), dict):
            status = info["result"].get("status") or ""
        if not status:
            return
        surface.note_bot_admin(
            key, str(status).lower() in (
                "creator", "administrator", "admin", "owner"))

    def _send_action_notice(self, surface, key, message,  # type: ignore[no-untyped-def]
                            action) -> None:
        """Post one §31 notice for an executed action.  The text
        and the Undo button are authored from the SAME record
        the surface stored (never re-derived at send time), and
        the posted message's id is written back to the record so
        an undo can edit the notice in place where the platform
        supports it."""
        from ...social.chat.group_notices import (
            notice_keyboard, notice_text)

        if not self._system_send_ok(surface, key, message,
                                    "notice"):
            return
        nid = str(action.get("notice_id") or "")
        record = surface.get_notice(key, nid) if nid else None
        text = str(action.get("warning") or "") or str(
            (record or {}).get("text") or "")
        if not text:
            return
        platform = str(message.chat.platform or "")
        if record is not None:
            body = notice_text({**record, "text": text}, platform)
            buttons = notice_keyboard(
                record, settings=surface.store.settings(key))
        else:
            body, buttons = text, None
        result = self._send(
            message.chat, body, buttons=buttons,
            parse_mode=self._rendered_mode(message, body))
        if record is not None:
            mid = getattr(result, "message_id", "") or ""
            if mid:
                surface.set_notice_message(key, nid, str(mid))
        if str(action.get("kind") or "") == "ban":
            # A ban lands with theatre (§21); a warning never
            # does. The expression mixin owns the firing — the
            # getattr guard keeps bare group rigs (tests) whole.
            fire = getattr(self, "_fire_expression", None)
            if fire is not None:
                fire(message, "ban")

    def _rendered_mode(self, message, text: str) -> str:  # type: ignore[no-untyped-def]
        """Parse mode for a surface-rendered text (welcome /
        goodbye): the typography organ renders Telegram-family
        surfaces as HTML, so the send must SAY so — otherwise
        the gateway's F1 presentation would convert the already-
        rendered HTML a second time.  Other platforms' renders
        carry their own native marks and send bare."""
        platform = str(getattr(message.chat, "platform", "") or "")
        if platform.startswith("telegram") and "<" in str(text):
            return "HTML"
        return ""

    def _card(self, message, card, buttons=None):  # type: ignore[no-untyped-def]
        """Render a typography Card (§27) for this message's
        platform under the group's presentation settings, and
        bundle the buttons the handler authored from the same
        state.  This is the ONE way group control replies are
        composed now: handlers name what renders, the group's
        theme decides how it looks."""
        from ...social.chat.tgbot_buttons import actionable
        from ...social.chat.typography import render_card

        surface = self._group_surface()
        key = message.chat.key
        rendered = render_card(
            card, platform=str(message.chat.platform or ""),
            settings=surface.store.settings(key))
        if buttons is not None:
            return actionable(rendered.text, buttons)
        return rendered.text

    def _welcome_buttons(self, surface: Any, key: str) -> list:
        """Buttons authored from the group's LIVE state: rules are
        always reachable; the sparks row appears only when the
        economy actually runs here. Labels come from the one
        button vocabulary (§21), under the group's own
        presentation settings."""
        from ...social.chat.expression import buttons_from_spec

        settings = surface.store.settings(key)
        spec: list = [[("rules", "Rules", "rules")]]
        try:
            if surface.economy_on(key):
                spec.append([("balance", "My sparks", "balance"),
                             ("quests", "Quests", "quests")])
        except Exception:  # noqa: BLE001 - buttons never block a welcome
            pass
        return buttons_from_spec(spec, settings=settings)

    # ---- join acceptance execution (DESIGN §28) -------------------
    # The surface judges; these helpers perform.  Every execution
    # speaks the adapter's own grammar and reports honestly —
    # a missing surface is an audited non-execution, never a
    # claimed success.

    def _acceptance_restrict(self, adapter, chat, user_id: str, *,
                             on: bool) -> bool:  # type: ignore[no-untyped-def]
        """Restrict (read-only) or release one member.  The two
        Telegram adapters spell restriction differently (MTProto
        minutes=0 = indefinite; the Bot API's until defaults to
        forever) — try each one's own grammar."""
        if adapter is None:
            return False
        if on:
            fn = getattr(adapter, "admin_restrict", None)
            if fn is None:
                return False
            try:
                out = fn(chat, user_id, minutes=0)
            except TypeError:
                try:
                    out = fn(chat, user_id)
                except Exception:  # noqa: BLE001 - reported below
                    return False
            except Exception:  # noqa: BLE001 - reported below
                return False
        else:
            fn = getattr(adapter, "admin_unrestrict", None)
            if fn is None:
                return False
            try:
                out = fn(chat, user_id)
            except Exception:  # noqa: BLE001 - reported below
                return False
        return not (isinstance(out, dict)
                    and out.get("ok") is False)

    def _acceptance_photo(self, adapter, chat, user_id: str):  # type: ignore[no-untyped-def]
        """Fresh profile-photo fact for the §28 photo criterion:
        True/False, or None when the platform can't say (an
        adapter without the surface, or a failed lookup) —
        None is UNKNOWN to the evaluator, never a failure."""
        fn = getattr(adapter, "admin_member_photo", None) \
            if adapter else None
        if fn is None:
            return None
        try:
            out = fn(chat, user_id)
        except Exception:  # noqa: BLE001 - unknown, not failed
            return None
        if isinstance(out, dict) and out.get("ok"):
            return bool(out.get("has_photo"))
        return None

    def _acceptance_join_request(self, surface, key, adapter,  # type: ignore[no-untyped-def]
                                 chat, user_id: str, *,
                                 approve: bool) -> bool:
        """Perform one join-request decision through the
        adapter; audit the outcome either way."""
        fn = getattr(adapter, "admin_join_request", None) \
            if adapter else None
        action = ("join_request_approve" if approve
                  else "join_request_decline")
        if fn is None:
            surface._audit(
                key, "moderation_unexecuted", action=action,
                rule="acceptance", user_id=user_id,
                reason="adapter has no admin_join_request")
            surface.store.save()
            return False
        try:
            out = fn(chat, user_id, approve)
            if isinstance(out, dict) and out.get("ok") is False:
                raise RuntimeError(str(out.get("error")
                                       or "platform refused"))
        except Exception as exc:  # noqa: BLE001 - audited
            surface._audit(
                key, "moderation_unexecuted", action=action,
                rule="acceptance", user_id=user_id,
                reason=f"{type(exc).__name__}: {exc}")
            surface.store.save()
            return False
        surface._audit(key, "moderation_executed", action=action,
                       rule="acceptance", user_id=user_id)
        surface.store.save()
        return True

    def _acceptance_facts(self, member: dict) -> dict:  # type: ignore[no-untyped-def]
        """Identity facts for one arrival, in the evaluator's
        shape, from a converter-supplied member/requester dict."""
        name = str(member.get("name") or "")
        first = str(member.get("first_name") or name)
        last = str(member.get("last_name") or "")
        if not last and name and first and name.startswith(first):
            rest = name[len(first):].strip()
            if rest:
                last = rest
        return {"user_id": str(member.get("id") or ""),
                "username": str(member.get("username") or ""),
                "name": name or first, "first_name": first,
                "last_name": last,
                "is_bot": bool(member.get("is_bot"))}

    def _send_acceptance_note(self, surface, key, message,  # type: ignore[no-untyped-def]
                              label: str, reasons: list, *,
                              grace_minutes: int | None = None,
                              pending_kick: bool = False) -> None:
        """The restricted member's one message (§28): what to
        fix + the “check me again” button, through the
        typography organ like every presented surface."""
        from ...social.chat.group_acceptance import fix_it_text
        from ...social.chat.typography import Card

        if not self._system_send_ok(surface, key, message,
                                    "acceptance_note"):
            return
        card = Card("moderation", title=f"{label} — one moment",
                    icon="🚪")
        body = fix_it_text(
            list(reasons or []),
            grace_minutes=grace_minutes if pending_kick else None)
        card.lines(body.splitlines())
        from ...social.chat.expression import buttons_from_spec

        text = surface._card_render(key, card)
        self._send(message.chat, text,
                   buttons=buttons_from_spec(
                       [[("check", "Check me again", "recheck")]],
                       settings=surface.store.settings(key)),
                   parse_mode=self._rendered_mode(message, text))

    def _send_acceptance_queue_note(self, surface, key, message,  # type: ignore[no-untyped-def]
                                    record: dict,
                                    reasons: list) -> None:
        """The admin queue notice for one held request (§28):
        who, what their evaluation said, and Approve / Decline
        buttons authored from the request itself.  The request
        id rides the buttons only — never the visible text
        (internal ids don't print)."""
        from ...social.chat.typography import Card

        if not self._system_send_ok(surface, key, message,
                                    "acceptance_queue"):
            return
        label = (record.get("name")
                 or (f"@{record['username']}"
                     if record.get("username") else "")
                 or record.get("user_id") or "someone")
        card = Card("moderation", title="Join request waiting",
                    icon="🚪")
        lines = [f"{label} wants in."]
        if reasons:
            lines.append("their check: " + "; ".join(reasons))
        elif record.get("unknown"):
            lines.append("one check couldn't be verified — "
                         "a human decides.")
        else:
            lines.append("they meet every rule — auto-approve "
                         "is off, so this one's yours.")
        lines.append("requests stay open until an admin decides "
                     "or the invite link is revoked.")
        from ...social.chat.expression import buttons_from_spec

        card.lines(lines)
        text = surface._card_render(key, card)
        rid = str(record.get("id") or "")
        self._send(message.chat, text, buttons=buttons_from_spec([
            [("approve", "Approve", f"joinq approve {rid}"),
             ("decline", "Decline", f"joinq decline {rid}")],
            [("queue", "The queue", "joinq")]],
            settings=surface.store.settings(key)),
            parse_mode=self._rendered_mode(message, text))

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
        self._probe_bot_claim(surface, key, message)
        meta = message.meta or {}
        service = str(meta.get("service") or "")

        if service == "reaction":
            # Reactions IN (§22): a member's reaction is room
            # signal — journaled to the held audit (the surface
            # engagement counts read from), never answered, never
            # narrated. Not system output: the §33 gate does not
            # apply to READING the room.
            surface._audit(
                key, "reaction",
                user_id=str(message.sender_id or ""),
                emojis=",".join(meta.get("reaction_new") or []),
                removed=",".join(meta.get("reaction_old") or []),
                message_id=str(meta.get("reacted_message_id")
                               or ""))
            surface.store.save()
            return True
        if service == "join":
            adapter = self._group_adapter(message.chat.platform)
            acc_cfg = (surface.store.settings(key)
                       .get("acceptance") or {})
            criteria = acc_cfg.get("criteria") or {}
            # Acceptance is INERT when no criterion is on and
            # the mode is open: the standing P1 path, byte for
            # byte.  The moment a group configures a rule, every
            # arrival is judged (§28).
            inert = (str(acc_cfg.get("mode") or "open") == "open"
                     and not any(criteria.values()))
            welcomed = []
            for member in meta.get("new_members") or []:
                if not isinstance(member, dict):
                    continue
                user_id = str(member.get("id") or "")
                if not user_id:
                    continue
                if inert:
                    text = surface.record_join(
                        key, user_id,
                        username=str(member.get("username") or ""),
                        name=str(member.get("name") or ""),
                        group_name=str(message.chat.title or ""))
                    if text:
                        welcomed.append(text)
                    continue
                facts = self._acceptance_facts(member)
                has_photo = member.get("has_photo")
                if criteria.get("photo") and has_photo is None:
                    has_photo = self._acceptance_photo(
                        adapter, message.chat, user_id)
                out = surface.acceptance_on_join(
                    key, facts, has_photo=has_photo)
                for action in out.get("actions") or []:
                    self._execute_group_action(
                        surface, key, adapter, message, action)
                if out.get("welcome"):
                    text = surface.welcome_text_for(
                        key, name=facts["name"],
                        username=facts["username"],
                        group_name=str(message.chat.title or ""))
                    if text:
                        welcomed.append(text)
                elif out.get("verdict") == "restricted":
                    label = (facts["name"]
                             or (f"@{facts['username']}"
                                 if facts["username"] else "")
                             or user_id)
                    self._send_acceptance_note(
                        surface, key, message, label,
                        out.get("reasons") or [],
                        grace_minutes=out.get("grace_minutes"),
                        pending_kick=bool(out.get("pending_kick")))
            if welcomed and self._system_send_ok(
                    surface, key, message, "welcome"):
                for text in welcomed:
                    self._send(
                        message.chat, text,
                        buttons=self._welcome_buttons(surface, key),
                        parse_mode=self._rendered_mode(message,
                                                       text))
                # The welcome moment (§21): a GIF/sticker rides
                # the welcome where the group's expression
                # settings allow — same gate, same room.
                fire = getattr(self, "_fire_expression", None)
                if fire is not None:
                    fire(message, "welcome")
            self._clean_service_message(surface, message)
            return True
        if service == "join_request":
            # Telegram's approval-mode invite flow (DESIGN §28):
            # the candidate is not in the group — the surface
            # judges, this branch performs the decision through
            # the adapter, and held requests surface ONE queue
            # notice with the decision on its buttons.
            adapter = self._group_adapter(message.chat.platform)
            requester = meta.get("requester") or {}
            if isinstance(requester, dict) and requester.get("id"):
                facts = self._acceptance_facts(requester)
                user_id = facts["user_id"]
                acc_cfg = (surface.store.settings(key)
                           .get("acceptance") or {})
                criteria = acc_cfg.get("criteria") or {}
                has_photo = requester.get("has_photo")
                if criteria.get("photo") and has_photo is None:
                    has_photo = self._acceptance_photo(
                        adapter, message.chat, user_id)
                out = surface.acceptance_on_request(
                    key, facts, has_photo=has_photo)
                verdict = out.get("verdict")
                if verdict in ("approve", "approve_restricted"):
                    done = self._acceptance_join_request(
                        surface, key, adapter, message.chat,
                        user_id, approve=True)
                    if done:
                        admitted = \
                            surface.acceptance_request_admitted(
                                key, str(out.get("request_id")
                                         or ""))
                        if admitted.get("restricted"):
                            self._execute_group_action(
                                surface, key, adapter, message,
                                {"kind": "acceptance_restrict",
                                 "user_id": user_id,
                                 "rule": "acceptance"})
                            label = (facts["name"]
                                     or (f"@{facts['username']}"
                                         if facts["username"]
                                         else "") or user_id)
                            self._send_acceptance_note(
                                surface, key, message, label,
                                admitted.get("reasons") or [],
                                grace_minutes=admitted.get(
                                    "grace_minutes"))
                        elif admitted.get("welcome"):
                            text = surface.welcome_text_for(
                                key, name=facts["name"],
                                username=facts["username"],
                                group_name=str(
                                    message.chat.title or ""))
                            if text and self._system_send_ok(
                                    surface, key, message,
                                    "welcome"):
                                self._send(
                                    message.chat, text,
                                    buttons=self._welcome_buttons(
                                        surface, key),
                                    parse_mode=self._rendered_mode(
                                        message, text))
                                fire = getattr(
                                    self, "_fire_expression", None)
                                if fire is not None:
                                    fire(message, "welcome")
                elif verdict == "decline":
                    self._acceptance_join_request(
                        surface, key, adapter, message.chat,
                        user_id, approve=False)
                else:  # queued — a human decides
                    record = out.get("record") or {}
                    if record:
                        self._send_acceptance_queue_note(
                            surface, key, message, record,
                            out.get("reasons") or [])
            return True
        if service == "leave":
            left = meta.get("left_member") or {}
            if isinstance(left, dict) and left.get("id"):
                text = surface.record_leave(
                    key, str(left.get("id") or ""),
                    group_name=str(message.chat.title or ""))
                if text and self._system_send_ok(
                        surface, key, message, "goodbye"):
                    self._send(
                        message.chat, text,
                        parse_mode=self._rendered_mode(message,
                                                       text))
            self._clean_service_message(surface, message)
            return True
        if service == "delete":
            # Evidentiary antidelete (defense brief §2.1): fed
            # only by surfaces that can see deletions (MTProto /
            # Baileys / Discord) — the Bot API never sends these.
            ids = meta.get("deleted_ids") or []
            if not ids and meta.get("deleted_id"):
                ids = [meta["deleted_id"]]
            for mid in ids:
                out = surface.handle_delete(
                    key, message_id=str(mid),
                    by_user_id=str(meta.get("by_user_id") or ""))
                for callout in out.get("callouts") or []:
                    if callout.get("to") == "group" \
                            and self._system_send_ok(
                                surface, key, message,
                                "antidelete_callout"):
                        self._send(message.chat,
                                   str(callout.get("text") or ""))
            return True

        user_id = str(message.sender_id or message.sender or "")
        if not user_id:
            return False
        if meta.get("edited"):
            # Evidentiary antiedit: record both versions, then
            # let the edited text flow through the normal path.
            surface.handle_edit(
                key, message_id=str(message.message_id or ""),
                new_text=str(message.text or ""))
        result = surface.handle_message(
            key, user_id,
            username=str(message.sender_username or ""),
            name=str(message.sender or ""),
            text=str(message.text or ""),
            message_id=str(message.message_id or ""),
            envelope=meta.get("envelope"),
            sender_is_admin=self._group_admin_ok(message),
            sender_is_operator=self._operator_or_false(message))
        notices = result.get("notices") or []
        if notices and self._system_send_ok(
                surface, key, message, "notice"):
            from ...social.chat.expression import (
                classify_notice_moment)

            for notice in notices:
                self._send(message.chat, str(notice))
                # Celebration notices carry a moment (§21): a
                # finished quest / a season podium gets her
                # media voice; every other notice stays text.
                moment = classify_notice_moment(str(notice))
                if moment:
                    fire = getattr(self, "_fire_expression", None)
                    if fire is not None:
                        fire(message, moment)
        self._execute_pending_pin(surface, message)
        actions = result.get("actions") or []
        if not actions:
            return False
        adapter = self._group_adapter(message.chat.platform)
        consumed = False
        for action in actions:
            kind = str(action.get("kind") or "")
            executed = self._execute_group_action(
                surface, key, adapter, message, action)
            warning = str(action.get("warning") or "")
            if executed and (warning or action.get("notice_id")):
                # The organ's one public warning IS the record
                # (no meta-narration): name + rule + consequence
                # — posted through the §31 notice layer, with
                # its Undo authored from the same record.
                self._send_action_notice(surface, key, message,
                                         action)
            elif kind == "warn" and executed \
                    and self._system_send_ok(
                        surface, key, message, "warning"):
                who = (message.sender_username
                       and f"@{message.sender_username}") \
                    or message.sender or "there"
                self._send(
                    message.chat,
                    f"⚠️ {who} — {action.get('reason') or 'warning'}")
            if kind in ("delete", "ban", "kick") and executed:
                consumed = True
        return consumed

    def _execute_pending_pin(self, surface, message) -> None:  # type: ignore[no-untyped-def]
        """A bought Pinned Message grant acts on the buyer's next
        message: pin it (best-effort, audited)."""
        try:
            user_id = str(message.sender_id or message.sender or "")
            if not user_id or not message.message_id:
                return
            if not surface.take_pending_pin(message.chat.key, user_id):
                return
            adapter = self._group_adapter(message.chat.platform)
            fn = getattr(adapter, "admin_pin", None) if adapter \
                else None
            if fn is None:
                return
            fn(message.chat, str(message.message_id))
            surface._audit(message.chat.key, "pin_grant_used",
                           user_id=user_id,
                           message_id=str(message.message_id))
            surface.store.save()
        except Exception:  # noqa: BLE001 - a grant never breaks chat
            _log.debug("group pin grant failed", exc_info=True)

    def _group_coin_sink(self, chat_key, player, coins, game) -> None:  # type: ignore[no-untyped-def]
        """GameEngine bridge (P2): a human's in-group game coins
        land in the group purse.  Player keys are
        ``platform:sender_id`` — the sender id IS the member id."""
        try:
            key = str(getattr(player, "key", "") or "")
            if ":" not in key:
                return
            member = key.split(":", 1)[1]
            self._group_surface().game_award(
                chat_key, member, int(coins), str(game))
        except Exception:  # noqa: BLE001 - bridge is best-effort
            _log.debug("group coin sink failed", exc_info=True)

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
    def _operator_or_false(self, message) -> bool:  # type: ignore[no-untyped-def]
        """Operator status, quietly: surfaces without operator
        power (and bare test rigs) read as non-operator."""
        try:
            return bool(self._is_operator(message))
        except Exception:  # noqa: BLE001 - no operator surface
            return False

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

    def _member_status(self, message):  # type: ignore[no-untyped-def]
        """The platform's word on this sender's standing in the
        group: (status string, raw info dict) — ("", {}) when the
        surface cannot say."""
        adapter = self._group_adapter(message.chat.platform)
        if adapter is None:
            return "", {}
        fn = getattr(adapter, "admin_member", None) or getattr(
            adapter, "admin_member_mtproto", None)
        if fn is None:
            return "", {}
        try:
            info = fn(message.chat, str(message.sender_id or ""))
        except Exception:  # noqa: BLE001 - no proof, no standing
            return "", {}
        if not isinstance(info, dict):
            return "", {}
        status = info.get("status") or info.get("role") or ""
        if not status and isinstance(info.get("result"), dict):
            status = info["result"].get("status") or ""
        return str(status).lower(), info

    def _group_owner_ok(self, message) -> bool:  # type: ignore[no-untyped-def]
        """The OWNER gate (defense brief §2.3): who may flip the
        autonomy master switch.  The operator always may.  On
        WhatsApp any group admin may (the platform hands group
        control to admins collectively).  On Telegram only the
        group's creator/owner — admin is a delegated role there,
        and the kill-switch outranks delegation.  Discord: the
        server owner."""
        if self._operator_or_false(message):
            return True
        platform = str(message.chat.platform or "")
        if platform.startswith("whatsapp"):
            return self._group_admin_ok(message)
        status, info = self._member_status(message)
        if platform.startswith("telegram"):
            return status in ("creator", "owner")
        if platform.startswith("discord"):
            return status in ("owner", "creator") \
                or info.get("is_owner") is True
        return False

    def _defense_surface_kind(self, message) -> str:  # type: ignore[no-untyped-def]
        """Which adapter family serves this chat — the key the
        platform-honesty matrix reads (Bot API vs MTProto is a
        capability cliff for delete events)."""
        platform = str(message.chat.platform or "")
        if platform.startswith("whatsapp"):
            return "baileys"
        if platform.startswith("discord"):
            return "user"
        if platform == "telegram":
            return "mtproto"
        adapter = self._group_adapter(platform)
        cls = type(adapter).__name__.lower() if adapter else ""
        return "mtproto" if "mtproto" in cls or "user" in cls \
            else "botapi"

    def _defense_update(self, surface, key, sensor, **fields):  # type: ignore[no-untyped-def]
        """Mutate one defense settings block through the store's
        own settings machinery (defaults-merged, persisted)."""
        import copy as _copy

        current = surface.store.settings(key)
        defense = _copy.deepcopy(current.get("defense") or {})
        if sensor == "_root":
            defense.update(fields)
        else:
            block = dict(defense.get(sensor) or {})
            block.update(fields)
            defense[sensor] = block
        surface.update_settings(key, defense=defense)
        return defense

    def _presentation_update(self, surface, key, **fields):  # type: ignore[no-untyped-def]
        """Mutate the presentation settings block (DESIGN §27:
        theme / per-class presets / stamp) through the store's
        own settings machinery (defaults-merged, persisted)."""
        import copy as _copy

        current = surface.store.settings(key)
        pres = _copy.deepcopy(current.get("presentation") or {})
        pres.update(fields)
        surface.update_settings(key, presentation=pres)
        return pres

    def _expression_update(self, surface, key, **fields):  # type: ignore[no-untyped-def]
        """Mutate the expression settings block (DESIGN §21)
        through the store's own settings machinery — the one
        write path behind both /gsettings and the console."""
        import copy as _copy

        current = surface.store.settings(key)
        expr = _copy.deepcopy(current.get("expression") or {})
        expr.update(fields)
        surface.update_settings(key, expression=expr)
        return expr

    def _apply_expression_grammar(self, surface, key, words):  # type: ignore[no-untyped-def]
        """The expression grammar, shared by /gsettings (words
        after the `expression` head) and the console's
        expression section (bare words). Returns the reply line,
        or None when the words aren't expression grammar.
        Forms: `on|off` · `gifs off|low|normal` ·
        `welcome on|off` · `ban on|off`."""
        w = [str(x).lower() for x in words or []]
        if w == ["on"] or w == ["off"]:
            on = w[0] == "on"
            self._expression_update(surface, key, enabled=on)
            return (f"expression {'on' if on else 'off'} — "
                    + ("moments get their media voice here."
                       if on else
                       "moments ship text-only here."))
        if len(w) == 2 and w[0] == "gifs" \
                and w[1] in ("off", "low", "normal"):
            self._expression_update(surface, key, gifs=w[1])
            return {
                "off": "gifs off — every moment ships text-only.",
                "low": "gifs low — welcomes and celebrations "
                       "only; bans land without theatre.",
                "normal": "gifs normal — welcomes, celebrations "
                          "and ban theatre.",
            }[w[1]]
        if len(w) == 2 and w[0] == "welcome" \
                and w[1] in ("on", "off"):
            on = w[1] == "on"
            self._expression_update(surface, key,
                                    welcome_media=on)
            return f"welcome media {'on' if on else 'off'}."
        if len(w) == 2 and w[0] == "ban" \
                and w[1] in ("on", "off"):
            on = w[1] == "on"
            self._expression_update(surface, key, ban_theatre=on)
            return (f"ban theatre {'on' if on else 'off'} — "
                    "warnings stay dry either way.")
        return None

    # ---- group commands (P1 slice 2) --------------------------------
    def _control_group_help(self, message):  # type: ignore[no-untyped-def]
        """Bare /help INSIDE a group (owner order 2026-11-03):
        list what is actually available in this group — the member
        surface and the admin surface side by side, rendered from
        this group's live state, not the global owner catalog.
        Everyone sees both sections; the admin commands themselves
        stay gated at execution.

        Presentation build: the reply is an ActionableMessage —
        text and buttons authored from the SAME state (the
        sparks row exists exactly when the economy runs here; the
        buttons mirror the sections on screen).  Typography
        (§27): composed as a Card, settings class — the group's
        theme decides the form."""
        from ...social.chat.typography import Card

        key = message.chat.key
        surface = self._group_surface()
        settings = surface.store.settings(key)
        members = surface.members(key)
        eco = surface.economy_on(key)
        card = Card("settings",
                    title=f"{message.chat.title or 'this group'} "
                          f"— what I do here",
                    icon="🛠️")
        everyone = [
            "/rules — the group rules",
            "/report — flag a message or member for the admins "
            "(reply to it, or /report @member why)",
            "/member [@name] — a member's card: tenure, activity, "
            "username/name changes",
            "/recheck — if the door rules let you in restricted "
            "(muted until you fix what's missing), tap this after "
            "fixing it and the checks re-run live",
            "/ping — check I'm alive, and how fast I answered",
            "🎮 games — /ttt /connect4 /reversi /checkers /mines "
            "and the rest of the board; /game lists them, moves are "
            "plain text (a1, b2 …). Some games queue you against "
            "players on other platforms.",
        ]
        if eco:
            everyone.append(
                "🪙 sparks (the group purse) — /balance · /daily · "
                "/mine · /work · /pay @member 50 · /deposit · "
                "/withdraw · /gshop · /quests · /treasury · /top · "
                "/tax · /claim when a drop is live. Chatting, "
                "streaks, quests and in-group game wins all earn.")
            everyone.append(
                "🥷 the risky side — /rob @member (fail and it's "
                "jail; /bail gets you out) · 🎰 /slots 10 · 🪙 "
                "/coinflip 10 — losses feed the treasury and the "
                "jackpot pot; triple 7️⃣ takes the pot.")
            everyone.append(
                "🏘️ property — /property (districts, the market "
                "index, your portfolio) · /buyproperty "
                "<district> <tier> · /rent collects daily rent "
                "as gems 💎 · /sellproperty · /listproperty "
                "(the member market) · /auctions (treasury "
                "escrow) · /mortgage (borrow against a "
                "building) · /insure (cover against repair "
                "waves). Gems buy the premium shelf in /gshop "
                "and the penthouse/tower tiers — they never "
                "come from sparks.")
        everyone.append(
            "/help /balance — the full page for any command")
        card.block_lines(everyone, title="for everyone:")
        card.block_lines([
            "/gsettings — welcome & goodbye, the rules list, "
            "flood/link moderation, service-message cleanup, the "
            "economy switch and transfer tax",
            "/drop 100 10 — fund a giveaway pool from your own "
            "sparks (or /drop treasury 100 10 from the treasury); "
            "members claim with /claim",
            "/grant @member 100 — pay a member from the treasury; "
            "above the governance threshold a second admin must "
            "/approve it first",
            "/grant gems @member 25 — gift gems 💎 (the premium "
            "purse moves only through governance)",
            "/buyproperty public <district> <tier> — the treasury "
            "buys a public building; its rent funds drops and "
            "prizes (large buys go through /approve like any "
            "treasury spend)",
            "moderation, by hand: reply to a member's message "
            "with /warn, /mute 10m|2h|1d, /ban or /kick — or name "
            "@member instead. /unmute /unban undo them; /pin and "
            "/purge work on the message you reply to; warns pile "
            "up to an automatic mute (limit in /gsettings).",
            "/glog — the group audit log (I keep it; I only show "
            "it when an admin asks)",
            "/strikes — the defense strike ledger: who is on "
            "the ladder, and how close the rungs are",
            "/joinq — the join queue: requests the acceptance "
            "rules held for a human, decided here or on the "
            "notice's buttons; the door rules themselves live "
            "in /gsettings (acceptance block)",
            "I moderate on my own: floods, repeat spam and "
            "off-allowlist links get deleted; crash payloads are "
            "quarantined, channel injections and mass tags are "
            "struck; repeat offenders are muted, kicked, then "
            "banned — all tuned in /gsettings.",
        ], title="for admins:")
        card.note(
            f"{len(members)} members in my ledger · "
            f"{len(settings.get('rules') or [])} rules · welcome "
            f"{'on' if settings['welcome_enabled'] else 'off'} · "
            f"link moderation "
            f"{'allowlist' if (settings['moderation'].get('link_allowlist')) else 'open'}")
        from ...social.chat.expression import buttons_from_spec

        spec = [[("rules", "Rules", "rules"),
                 ("games", "Games", "game")]]
        if eco:
            spec.append([("balance", "My sparks", "balance"),
                         ("quests", "Quests", "quests"),
                         ("treasury", "Treasury", "treasury")])
        spec.append([("settings", "Group settings", "gsettings"),
                     ("glog", "Audit log", "glog")])
        buttons = buttons_from_spec(spec, settings=settings)
        return self._card(message, card, buttons)

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
        wanted = str(arg or "").strip()
        if wanted:
            # §29: the card names a person by every path, not
            # only the ledger's memory.
            res = self._resolver(surface).resolve_token(
                key, message, wanted)
            if not res.found:
                if res.channel_post:
                    return (f"that was posted as "
                            f"{res.channel_label} — a channel, "
                            f"not a member with a card.")
                return res.exhaustion_text()
            target = surface.member(key, res.member.user_id)
            if target is None:
                # Resolved on a platform surface but never
                # recorded: the card is built from what the
                # resolution itself proved.
                target = {"user_id": res.member.user_id,
                          "username": res.member.username,
                          "name": res.member.name,
                          "first_seen": None, "joined_ts": None,
                          "message_count": 0, "last_seen": None}
            return _member_card(surface, key, target)
        target = surface.members(key).get(
            str(message.sender_id or ""))
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
        entries = self._group_surface().audit(
            key, limit=max(1, min(n, 50)))
        if not entries:
            return "nothing in the group log yet."
        from ...social.chat.typography import Card

        card = Card("settings", title="Group log", icon="📋")
        card.board("\n".join(_audit_line(entry)
                             for entry in entries))
        card.note("held by me, shown on demand")
        return self._card(message, card)

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

            from ...social.chat.expression import button_label

            def _toggle(action: str, label: str, kind: str,
                        on_now: bool) -> tuple:
                # The button SHOWS the current state and carries
                # the flip — authored from the live settings, so
                # it can never disagree with the summary above it.
                # Its emoji comes from the one button vocabulary
                # (§21), under this group's presentation settings.
                return (button_label(
                            action,
                            f"{label}: {'ON' if on_now else 'OFF'}",
                            settings=s),
                        f"gsettings {kind} "
                        f"{'off' if on_now else 'on'}")

            pres = s.get("presentation") or {}
            theme_cycle = {"house": "clean", "clean": "board",
                           "board": "house"}
            theme_now = str(pres.get("theme") or "house")
            buttons = [
                [_toggle("welcome", "Welcome", "welcome",
                         bool(s["welcome_enabled"])),
                 _toggle("goodbye", "Goodbye", "goodbye",
                         bool(s["goodbye_enabled"]))],
                [_toggle("cleanup", "Cleanup", "clean",
                         bool(s["clean_service"])),
                 _toggle("economy", "Economy", "economy",
                         bool(s["economy_enabled"]))],
                [(button_label("theme", f"Theme: {theme_now}",
                               settings=s),
                  f"gsettings theme "
                  f"{theme_cycle.get(theme_now, 'house')}"),
                 _toggle("stamp", "Stamp", "stamp",
                         bool(pres.get("stamp", True)))],
            ]
            # Defense block (DESIGN §17): sensor states from the
            # same settings the pipeline reads + the platform-
            # honesty notes + what has degraded on THIS surface.
            from ...social.chat.group_defense import (
                defense_availability)
            d = s.get("defense") or {}
            d_ac = d.get("anticrash") or {}
            d_ch = d.get("antichannel") or {}
            d_tg = d.get("antitag") or {}
            d_dl = d.get("antidelete") or {}
            d_ed = d.get("antiedit") or {}
            d_esc = d.get("escalation") or {}
            avail = defense_availability(
                str(message.chat.platform or ""),
                self._defense_surface_kind(message))
            honesty = []
            if not avail["antidelete"]:
                honesty.append(avail["antidelete_note"])
            if not avail["antiedit"]:
                honesty.append(avail["antiedit_note"])
            degraded = []
            for entry in surface.audit(key, limit=50):
                if entry.get("kind") == "moderation_unexecuted":
                    degraded.append(
                        f"{entry.get('action')} — "
                        f"{entry.get('reason')}")
            seen: set = set()
            degraded = [x for x in degraded
                        if not (x in seen or seen.add(x))][:3]
            # Typography (§27): the settings panel is a Card —
            # state blocks, then the change grammar as code
            # lines, each rendered per platform by the organ.
            from ...social.chat.typography import (
                DEFAULT_STAMP_TEXT, Card, resolve_style)

            card = Card("settings", title="Group settings",
                        icon="⚙️")
            card.block_lines([
                f"welcome: {'on' if s['welcome_enabled'] else 'off'} · "
                f"goodbye: {'on' if s['goodbye_enabled'] else 'off'} · "
                f"service cleanup: {'on' if s['clean_service'] else 'off'}",
                f"rules: {len(s.get('rules') or [])} · "
                f"flood: {mod['flood_max']} per "
                f"{int(mod['flood_window_s'])}s · repeats ×"
                f"{mod['repeat_threshold']} · links: "
                f"{', '.join(mod.get('link_allowlist') or []) or 'open'}"
                f" · filters: {len(mod.get('word_filters') or [])} "
                f"({mod.get('filter_action') or 'delete'})",
                f"economy: {'on' if s['economy_enabled'] else 'off'}"
                f" · transfer tax {s.get('transfer_tax_pct', 2)}%"
                f" · fine {s.get('fine_amount', 0)}"
                f" · governance ≥{s.get('governance_threshold', 500)}",
            ])
            defense_lines = [
                f"autonomy "
                f"{'ON' if d.get('autonomy', True) else 'OFF'}"
                f" (owner's switch — WhatsApp: any admin; "
                f"Telegram: group owner only)",
                f"anticrash: "
                f"{'on' if d_ac.get('enabled', True) else 'off'}"
                f" ({d_ac.get('action', 'delete')}) · antichannel: "
                f"{'on' if d_ch.get('enabled', True) else 'off'}"
                f" · antitag: "
                f"{'on' if d_tg.get('enabled', True) else 'off'}"
                f" (≤{d_tg.get('max_mentions', 10)} mentions or "
                f"{d_tg.get('max_roster_pct', 20)}% of the group)",
                f"antidelete: "
                f"{'on' if d_dl.get('enabled', True) else 'off'}"
                f" ({d_dl.get('mode', 'log')}) · antiedit: "
                f"{'on' if d_ed.get('enabled', True) else 'off'}"
                f" ({d_ed.get('mode', 'log')}) · "
                f"mute default: "
                f"{_duration_text(int(s.get('mute_default_s') or 0)) if s.get('mute_default_s') else 'ask'}",
            ]
            defense_lines += [f"⚠ {note}" for note in honesty]
            if degraded:
                defense_lines.append(
                    "degraded here: " + "; ".join(degraded))
            card.block_lines(defense_lines, title="Defense")
            game_style = resolve_style(s, "game")
            economy_style = resolve_style(s, "economy")
            card.block_lines([
                f"theme: {game_style.theme} · games render "
                f"{game_style.preset} · panels render "
                f"{economy_style.preset}",
                f"stamp: "
                f"{'ON' if pres.get('stamp', True) else 'OFF'}"
                f" — \"{pres.get('stamp_text') or DEFAULT_STAMP_TEXT}\""
                f" on presented panels"
                + (f" (classes: "
                   f"{', '.join(pres.get('stamp_classes') or [])})"
                   if pres.get("stamp_classes") else ""),
            ], title="Presentation")
            # Acceptance block (DESIGN §28): the door rules as
            # the evaluator reads them + the live queue depth +
            # the raid posture right now.
            acc = s.get("acceptance") or {}
            acc_crit = acc.get("criteria") or {}
            crit_on = [name for name in (
                "username", "photo", "firstname", "lastname",
                "hygiene", "banned", "bots") if acc_crit.get(name)]
            _acc_cfg, raid_now = surface.acceptance_cfg(key)
            queue_n = len(surface.acceptance_queue(key))
            card.block_lines([
                f"mode: {acc.get('mode', 'open')} · action on a "
                f"failed check: {acc.get('action', 'restricted')}"
                f" · grace {int(acc.get('grace_minutes') or 60)}m"
                f" · auto-approve "
                f"{'on' if acc.get('auto_approve', True) else 'off'}",
                "criteria on: "
                + (", ".join(crit_on) if crit_on
                   else "none — the door is open"),
                f"queue: {queue_n} waiting · raid posture: "
                f"{'ACTIVE (tightened)' if raid_now else 'calm'}",
            ], title="Acceptance")
            # Escalation block (DESIGN §30): the ladder EVERY
            # strike consumer reads — sensors, manual warns,
            # acceptance enforcement — as the group designed it.
            esc_now = surface.escalation_config(key)
            card.block_lines([
                esc_now.describe(),
                "every warning names the count of the limit, "
                "the next step, and when the oldest warn clears",
            ], title="Escalation")
            # Estate block (§17/§20 levers — their grammar lives
            # here and in the DM console).
            est = s.get("estate") or {}
            card.block_lines([
                f"market cut {est.get('market_cut_pct', 5)}% · "
                f"sell-back {est.get('sellback_pct', 70)}% · "
                f"ownership cap {est.get('ownership_cap', 12)} · "
                f"mortgage LTV {est.get('ltv_pct', 60)}% at "
                f"{est.get('mortgage_rate_pct', 5)}%",
            ], title="Estate")
            # System-output claim (§33): where Devon's bot
            # systems may speak and accrue in this room — the
            # two-hats law, read from the same state the gate
            # itself reads.
            claim_now = surface.claim_state(key)
            if claim_now["claimed"]:
                claim_line = (
                    f"system output: ON — claimed "
                    f"({claim_now['source']})")
            else:
                claim_line = (
                    "system output: OFF — my bot doesn't run "
                    "here; my sessions only watch and converse")
            card.block_lines([
                claim_line,
                "the owner designates a room with "
                "`/gsettings system output on`",
            ], title="System output")
            # Expression block (§21): her GIF/sticker voice in
            # this room — state read from the same settings the
            # moment service reads, library size included.
            expr = s.get("expression") or {}
            lib_n = ""
            try:
                svc = getattr(self, "_expression_service", None)
                if svc is not None:
                    lib_n = (f" · library "
                             f"{len(svc().library)} items")
            except Exception:  # noqa: BLE001 - display only
                lib_n = ""
            card.block_lines([
                f"expression: "
                f"{'on' if expr.get('enabled', True) else 'off'}"
                f" · gifs {expr.get('gifs', 'normal')}"
                f" · welcome media "
                f"{'on' if expr.get('welcome_media', True) else 'off'}"
                f" · ban theatre "
                f"{'on' if expr.get('ban_theatre', True) else 'off'}"
                f"{lib_n}",
                "warnings stay dry at every setting — a warning "
                "is a record, not a show",
            ], title="Expression")
            card.block_lines([
                "`/gsettings welcome on|off` · "
                "`/gsettings goodbye on|off` · "
                "`/gsettings clean on|off`",
                "`/gsettings rules add TEXT` · "
                "`/gsettings rules clear`",
                "`/gsettings flood N` · "
                "`/gsettings links DOMAINS|off`",
                "`/gsettings economy on|off` · "
                "`/gsettings tax PCT` · "
                "`/gsettings warns N` · "
                "`/gsettings mute default 2h|off`",
                "`/gsettings fine N` · "
                "`/gsettings governance N`",
                "`/gsettings filter add WORD` · "
                "`/gsettings filter action delete|warn|mute`",
                "`/gsettings autonomy on|off` · "
                "`/gsettings anticrash on|off|delete|flag`",
                "`/gsettings antichannel on|off` · "
                "`/gsettings antitag on|off` · "
                "`/gsettings antitag N`",
                "`/gsettings antidelete on|off` · "
                "`/gsettings antidelete mode log|callout`",
                "`/gsettings antiedit on|off` · "
                "`/gsettings antiedit mode log|callout`",
                "`/gsettings theme house|clean|board` · "
                "`/gsettings class game board|clean|default`",
                "`/gsettings stamp on|off` · "
                "`/gsettings stamp text TEXT|default` · "
                "`/gsettings stamp classes a,b|default`",
                "`/gsettings acceptance mode open|request|invite`",
                "`/gsettings acceptance username|photo|firstname"
                "|lastname|hygiene|banned|bots on|off`",
                "`/gsettings acceptance action decline|restricted"
                "|kick` · `/gsettings acceptance grace MIN`",
                "`/gsettings acceptance auto on|off` · "
                "`/gsettings acceptance override CRITERION "
                "decline|restricted|kick|default`",
                "`/gsettings acceptance whitelist add|remove "
                "@name|id`",
                "`/gsettings escalation limit N` · "
                "`/gsettings escalation final kick|ban` · "
                "`/gsettings escalation window DAYS`",
                "`/gsettings escalation at N mute [10m|2h]` · "
                "`… at N fine AMOUNT` · `… at N restrict|kick` · "
                "`… clear N` · `… reset`",
                "`/gsettings estate cut PCT` · "
                "`/gsettings estate sellback PCT` · "
                "`/gsettings estate cap N` · "
                "`/gsettings estate ltv PCT` · "
                "`/gsettings estate rate PCT`",
                "`/gsettings system output on|off` "
                "(owner — designates this room for my "
                "bot-system output, §33)",
                "`/gsettings expression on|off` · "
                "`/gsettings expression gifs off|low|normal`",
                "`/gsettings expression welcome on|off` · "
                "`/gsettings expression ban on|off`",
            ], title="Change it")
            card.note("tap a button to flip a switch, or type a "
                      "command")
            return self._card(message, card, buttons)
        parts = text.split(None, 2)
        head = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""
        rest2 = parts[2] if len(parts) > 2 else ""
        on = rest.lower() == "on"
        if head == "system" and rest.lower() == "output" \
                and rest2.lower() in ("on", "off"):
            # §33 designation is the OWNER's call: it decides
            # where his personal sessions may speak as the bot
            # system, so a group admin alone cannot flip it.
            if not self._operator_or_false(message):
                return ("designating a room for my system "
                        "output is the owner's call — it "
                        "decides where my bot systems may "
                        "speak through his accounts.")
            state = surface.set_designated(
                key, rest2.lower() == "on",
                by=str(message.sender_id or ""))
            if state["designated"]:
                return ("system output: this room is "
                        "designated — my bot systems (economy, "
                        "quests, games, notices) run here.")
            return ("system output: designation lifted — "
                    "this room is claimed only where my bot "
                    "itself operates.")
        if head == "expression":
            # Media expression (§21): on/off, GIF frequency,
            # welcome media, ban theatre — the same helper the
            # DM console's expression section applies.
            out = self._apply_expression_grammar(
                surface, key, text.split()[1:])
            if out is not None:
                return out
            return ("usage: /gsettings expression on|off · "
                    "expression gifs off|low|normal · "
                    "expression welcome on|off · "
                    "expression ban on|off")
        if head == "warns" and rest.isdigit():
            surface.update_settings(
                key, warn_threshold=max(1, int(rest)))
            return f"warn limit {max(1, int(rest))} before an auto-mute."
        if head == "mute" and rest.lower() == "default":
            if rest2.lower() in ("off", "none", "0", ""):
                surface.update_settings(key, mute_default_s=0)
                return ("no standing mute length — when a member "
                        "is named without one, I'll ask.")
            secs, _why = _parse_duration(rest2)
            if secs:
                surface.update_settings(key, mute_default_s=secs)
                return (f"standing mute length "
                        f"{_duration_text(secs)} when a named "
                        f"member is given none.")
            return "usage: /gsettings mute default 10m|2h|1d|off"
        if head == "fine" and rest.lstrip("-").isdigit():
            amount = max(0, int(rest))
            surface.update_settings(key, fine_amount=amount)
            return (f"moderation fine {amount} sparks per auto-mute "
                    f"(paid to the treasury)."
                    if amount else "moderation fines off.")
        if head == "governance" and rest.isdigit():
            threshold = max(0, int(rest))
            surface.update_settings(
                key, governance_threshold=threshold)
            return (f"governance threshold {threshold} sparks — "
                    "bigger treasury spends need a second admin.")
        if head == "economy" and rest.lower() in ("on", "off"):
            surface.update_settings(key, economy_enabled=on)
            return f"economy {'on' if on else 'off'}."
        if head == "tax" and rest.rstrip("%").isdigit():
            pct = max(0, min(25, int(rest.rstrip("%"))))
            surface.update_settings(key, transfer_tax_pct=pct)
            return f"transfer tax {pct}% to the treasury."
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
        if head == "filter" and rest.lower() == "add" and rest2:
            current = surface.store.settings(key)
            mod = dict(current["moderation"])
            filters = list(mod.get("word_filters") or []) + [rest2]
            mod["word_filters"] = filters
            surface.update_settings(key, moderation=mod)
            return f"filter added ({len(filters)} active)."
        if head == "filter" and rest.lower() == "remove" and rest2:
            current = surface.store.settings(key)
            mod = dict(current["moderation"])
            filters = [f for f in (mod.get("word_filters") or [])
                       if f != rest2]
            mod["word_filters"] = filters
            surface.update_settings(key, moderation=mod)
            return f"filter list now has {len(filters)} entries."
        if head == "filter" and rest.lower() == "clear":
            current = surface.store.settings(key)
            mod = dict(current["moderation"])
            mod["word_filters"] = []
            surface.update_settings(key, moderation=mod)
            return "filters cleared."
        if head == "filter" and rest.lower() == "action" \
                and rest2.lower() in ("delete", "warn", "mute"):
            current = surface.store.settings(key)
            mod = dict(current["moderation"])
            mod["filter_action"] = rest2.lower()
            surface.update_settings(key, moderation=mod)
            return f"filter action: {rest2.lower()}."
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
        # ── defense grammar (DESIGN §17) ────────────────────────
        if head == "autonomy" and rest.lower() in ("on", "off"):
            # The master switch outranks ordinary admin: WhatsApp
            # any group admin; Telegram the group OWNER only.
            if not self._group_owner_ok(message):
                return ("autonomy is the owner's switch — on "
                        "Telegram only the group owner can flip "
                        "it (WhatsApp: any group admin). Admins "
                        "keep every manual command either way.")
            self._defense_update(surface, key, "_root",
                                 autonomy=on)
            return (f"defense autonomy {'on' if on else 'off'}."
                    + ("" if on else
                       " Sensors still watch and log to /glog — "
                       "nothing is enforced or announced."))
        if head in ("anticrash", "antichannel", "antitag",
                    "antidelete", "antiedit"):
            if rest.lower() in ("on", "off"):
                self._defense_update(surface, key, head,
                                     enabled=on)
                return f"{head} {'on' if on else 'off'}."
            if head == "anticrash" and rest.lower() in (
                    "delete", "flag"):
                self._defense_update(surface, key, head,
                                     action=rest.lower())
                return (f"anticrash action: {rest.lower()} "
                        f"quarantined payloads.")
            if head == "antitag" and rest.isdigit():
                ceiling = max(1, int(rest))
                self._defense_update(surface, key, head,
                                     max_mentions=ceiling)
                return f"antitag ceiling: {ceiling} mentions."
            if head in ("antidelete", "antiedit") \
                    and rest.lower() == "mode" \
                    and rest2.lower() in ("log", "callout"):
                self._defense_update(surface, key, head,
                                     mode=rest2.lower())
                return f"{head} mode: {rest2.lower()}."
            if head in ("antidelete", "antiedit") \
                    and rest.lower() == "to" \
                    and rest2.lower() in ("group", "admins"):
                self._defense_update(surface, key, head,
                                     callout_to=rest2.lower())
                return f"{head} callouts go to: {rest2.lower()}."
        # ── presentation grammar (DESIGN §27) ───────────────────
        if head == "theme" and rest.lower() in (
                "house", "clean", "board"):
            self._presentation_update(surface, key,
                                      theme=rest.lower())
            return (f"presentation theme: {rest.lower()} — the "
                    "whole group's surfaces follow it.")
        if head == "class" and rest.lower() in (
                "economy", "moderation", "game", "briefing",
                "announcement", "leaderboard", "settings",
                "chat") and rest2.lower() in (
                "clean", "board", "default"):
            current = surface.store.settings(key)
            classes = dict(
                (current.get("presentation") or {}).get(
                    "classes") or {})
            if rest2.lower() == "default":
                classes.pop(rest.lower(), None)
                self._presentation_update(surface, key,
                                          classes=classes)
                return (f"{rest.lower()} surfaces follow the "
                        "theme again.")
            classes[rest.lower()] = rest2.lower()
            self._presentation_update(surface, key,
                                      classes=classes)
            return (f"{rest.lower()} surfaces render "
                    f"{rest2.lower()} in this group.")
        if head == "stamp" and rest.lower() in ("on", "off"):
            self._presentation_update(
                surface, key, stamp=(rest.lower() == "on"))
            return (f"footer stamp {'on' if rest.lower() == 'on' else 'off'}"
                    " for presented panels.")
        if head == "stamp" and rest.lower() == "text":
            if rest2.lower() in ("", "default", "clear"):
                self._presentation_update(surface, key,
                                          stamp_text="")
                return "stamp text back to the house default."
            self._presentation_update(surface, key,
                                      stamp_text=rest2)
            return f"stamp text: “{rest2}”."
        if head == "stamp" and rest.lower() == "classes":
            if rest2.lower() in ("", "default", "clear"):
                self._presentation_update(surface, key,
                                          stamp_classes=[])
                return ("stamp classes back to the presented "
                        "defaults.")
            wanted = [c.strip().lower() for c in
                      rest2.replace(" ", ",").split(",")
                      if c.strip()]
            from ...social.chat.typography import SURFACE_CLASSES

            wanted = [c for c in wanted if c in SURFACE_CLASSES]
            if not wanted:
                return ("name at least one surface class: "
                        "economy, game, briefing, announcement, "
                        "leaderboard, moderation, settings, chat.")
            self._presentation_update(surface, key,
                                      stamp_classes=wanted)
            return f"stamp shows on: {', '.join(wanted)}."
        # ── escalation grammar (DESIGN §30) ────────────────────
        if head == "escalation":
            reply = self._apply_escalation_grammar(
                surface, key, text.split()[1:])
            if reply is not None:
                return reply
            return ("escalation grammar: limit N · final "
                    "kick|ban · window DAYS · at N mute [10m] · "
                    "at N fine AMOUNT · at N restrict|kick · "
                    "clear N · reset — /gsettings shows the "
                    "ladder.")
        # ── estate grammar (§17/§20 levers) ─────────────────────
        if head == "estate":
            reply = self._apply_estate_grammar(
                surface, key, text.split()[1:])
            if reply is not None:
                return reply
            return ("estate grammar: cut PCT · sellback PCT · "
                    "cap N · ltv PCT · rate PCT — /gsettings "
                    "shows the levers.")
        # ── acceptance grammar (DESIGN §28) ─────────────────────
        if head == "acceptance":
            sub = rest.lower()
            if sub == "mode" and rest2.lower() in (
                    "open", "request", "invite"):
                self._acceptance_update(surface, key,
                                         mode=rest2.lower())
                return (f"join mode: {rest2.lower()} — "
                        + {"open": "anyone can walk in; the "
                                   "criteria judge them at the "
                                   "door.",
                           "request": "approval-mode invite "
                                      "links produce requests "
                                      "the rules judge first.",
                           "invite": "joins arrive by invite; "
                                     "the criteria still judge "
                                     "them on arrival."}
                        [rest2.lower()])
            if sub in ("username", "photo", "firstname",
                       "lastname", "hygiene", "banned", "bots") \
                    and rest2.lower() in ("on", "off"):
                current = surface.store.settings(key)
                crit = dict((current.get("acceptance") or {})
                            .get("criteria") or {})
                crit[sub] = (rest2.lower() == "on")
                self._acceptance_update(surface, key,
                                        criteria=crit)
                return (f"acceptance criterion {sub}: "
                        f"{rest2.lower()}.")
            if sub == "action" and rest2.lower() in (
                    "decline", "restricted", "kick"):
                self._acceptance_update(surface, key,
                                         action=rest2.lower())
                explain = {
                    "decline": "the request is refused / the "
                               "join is removed, reason on the "
                               "record",
                    "restricted": "they come in muted with a "
                                  "fix-it note and a "
                                  "check-me-again button",
                    "kick": "they come in muted for the grace "
                            "window, then are removed if still "
                            "unfixed"}[rest2.lower()]
                return (f"a failed check now means: "
                        f"{rest2.lower()} ({explain}).")
            if sub == "grace" and rest2.rstrip("m").isdigit():
                minutes = max(1, min(10080,
                                     int(rest2.rstrip("m"))))
                self._acceptance_update(surface, key,
                                         grace_minutes=minutes)
                return (f"acceptance grace: {minutes} minutes "
                        f"before a kick-action escalates.")
            if sub == "auto" and rest2.lower() in ("on", "off"):
                self._acceptance_update(
                    surface, key,
                    auto_approve=(rest2.lower() == "on"))
                return (f"auto-approve of clean requests: "
                        f"{rest2.lower()}.")
            if sub == "override" and rest2:
                bits = rest2.split(None, 1)
                crit_name = bits[0].lower()
                act = bits[1].lower() if len(bits) > 1 else ""
                if crit_name in ("username", "photo", "firstname",
                                 "lastname", "hygiene", "banned",
                                 "bots") and act in (
                        "decline", "restricted", "kick",
                        "default"):
                    current = surface.store.settings(key)
                    overrides = dict(
                        (current.get("acceptance") or {})
                        .get("overrides") or {})
                    if act == "default":
                        overrides.pop(crit_name, None)
                    else:
                        overrides[crit_name] = act
                    self._acceptance_update(surface, key,
                                            overrides=overrides)
                    return (f"{crit_name} failures now take: "
                            f"{act if act != 'default' else 'the group action'}.")
                return ("usage: /gsettings acceptance override "
                        "username|photo|firstname|lastname|hygiene"
                        "|banned|bots decline|restricted|kick"
                        "|default")
            if sub == "whitelist" and rest2:
                bits = rest2.split(None, 1)
                verb = bits[0].lower()
                token = bits[1].strip() if len(bits) > 1 else ""
                if verb in ("add", "remove") and token:
                    current = surface.store.settings(key)
                    wl = [str(w) for w in (
                        (current.get("acceptance") or {})
                        .get("bot_whitelist") or [])]
                    if verb == "add" and token not in wl:
                        wl.append(token)
                    if verb == "remove":
                        wl = [w for w in wl if w != token]
                    self._acceptance_update(surface, key,
                                            bot_whitelist=wl)
                    return (f"bot whitelist: "
                            f"{', '.join(wl) or 'empty'}.")
                return ("usage: /gsettings acceptance whitelist "
                        "add|remove @name|id")
            return ("acceptance grammar: mode, the criterion "
                    "names, action, grace, auto, override, "
                    "whitelist — /gsettings shows the block.")
        return ("I didn't follow that — /gsettings alone shows the "
                "current settings and the change grammar.")

    # ---- escalation + estate settings (DESIGN §30 / §20) -------
    def _escalation_update(self, surface, key, **fields):  # type: ignore[no-untyped-def]
        """Mutate the escalation settings block (the §30 ladder)
        through the store's own settings machinery."""
        import copy as _copy

        current = surface.store.settings(key)
        block = _copy.deepcopy(current.get("escalation") or {})
        block.update(fields)
        surface.update_settings(key, escalation=block)
        return block

    def _estate_update(self, surface, key, **fields):  # type: ignore[no-untyped-def]
        """Mutate the estate settings block (the §17/§20 levers)
        through the store's own settings machinery — the grammar
        the estate build left to this console slice."""
        import copy as _copy

        current = surface.store.settings(key)
        block = _copy.deepcopy(current.get("estate") or {})
        block.update(fields)
        surface.update_settings(key, estate=block)
        return block

    def _apply_escalation_grammar(self, surface, key, words):  # type: ignore[no-untyped-def]
        """The escalation change grammar, shared verbatim by
        /gsettings and the DM console: ``limit N`` · ``final
        kick|ban`` · ``window DAYS`` · ``at N mute [10m|2h]`` ·
        ``at N fine AMOUNT`` · ``at N restrict|kick`` ·
        ``clear N`` · ``reset``.  Returns the reply line, or
        None when the words don't parse."""
        from ...social.chat.group_defense import (
            ESCALATION_DEFAULTS, EscalationConfig)

        words = [str(w) for w in words]
        if not words:
            return None
        esc = surface.escalation_config(key)

        def ladder() -> str:
            return ("ladder now: "
                    f"{surface.escalation_config(key).describe()}.")

        head = words[0].lower()
        if head == "limit" and len(words) == 2 \
                and words[1].isdigit():
            self._escalation_update(
                surface, key,
                warn_limit=max(1, int(words[1])))
            return (f"warn limit {max(1, int(words[1]))} — the "
                    "final action lands there. " + ladder())
        if head == "final" and len(words) == 2 \
                and words[1].lower() in ("kick", "ban"):
            self._escalation_update(surface, key,
                                    final_action=words[1].lower())
            return (f"final action: {words[1].lower()} at the "
                    f"limit. {ladder()}")
        if head == "window" and len(words) == 2:
            digits = words[1].lower().rstrip("d")
            if digits.isdigit():
                self._escalation_update(
                    surface, key,
                    window_days=max(1, int(digits)))
                return (f"warns expire after "
                        f"{max(1, int(digits))} days. {ladder()}")
            return None
        if head == "reset" and len(words) == 1:
            import copy as _copy

            self._escalation_update(
                surface, key,
                **_copy.deepcopy(ESCALATION_DEFAULTS))
            return ("escalation reset to the house ladder. "
                    f"{ladder()}")
        if head == "clear" and len(words) == 2 \
                and words[1].isdigit():
            at = int(words[1])
            steps = [s for s in esc.intervals if s["at"] != at]
            if len(steps) == len(esc.intervals):
                return f"no designed step sits at {at}."
            self._escalation_update(surface, key, intervals=steps)
            return f"step at {at} removed. {ladder()}"
        if head == "at" and len(words) >= 3 and words[1].isdigit():
            at = int(words[1])
            action = words[2].lower()
            steps = [s for s in esc.intervals if s["at"] != at]
            if action == "mute":
                minutes = 60
                if len(words) >= 4:
                    secs, _rest = _parse_duration(words[3])
                    if not secs:
                        return None
                    minutes = max(1, secs // 60)
                steps.append({"at": at, "action": "mute",
                              "minutes": minutes})
            elif action == "fine":
                if len(words) < 4 or not words[3].isdigit():
                    return None
                steps.append({"at": at, "action": "fine",
                              "amount": max(1, int(words[3]))})
            elif action in ("restrict", "kick"):
                steps.append({"at": at, "action": action})
            else:
                return None
            steps.sort(key=lambda s: s["at"])
            self._escalation_update(surface, key, intervals=steps)
            return (f"step designed: {action} at {at}. "
                    f"{ladder()}")
        return None

    def _apply_estate_grammar(self, surface, key, words):  # type: ignore[no-untyped-def]
        """The estate lever grammar, shared by /gsettings and
        the DM console: ``cut PCT`` · ``sellback PCT`` · ``cap
        N`` · ``ltv PCT`` · ``rate PCT``."""
        words = [str(w) for w in words]
        if len(words) != 2:
            return None
        field = {"cut": "market_cut_pct",
                 "sellback": "sellback_pct",
                 "cap": "ownership_cap",
                 "ltv": "ltv_pct",
                 "rate": "mortgage_rate_pct"}.get(words[0].lower())
        digits = words[1].rstrip("%")
        if field is None or not digits.isdigit():
            return None
        value = int(digits)
        if field == "ownership_cap":
            value = max(1, value)
        else:
            value = max(0, min(100, value))
        self._estate_update(surface, key, **{field: value})
        return f"estate {words[0].lower()}: {value}."

    # ---- the DM admin console (DESIGN §30) ----------------------
    _CONSOLE_TTL_S = 1800
    _CONSOLE_SECTIONS = ("defense", "acceptance", "economy",
                         "estate", "presentation", "welcome",
                         "expression")

    def _console_session_map(self) -> dict:  # type: ignore[no-untyped-def]
        sessions = getattr(self, "_console_sessions", None)
        if sessions is None:
            sessions = self._console_sessions = {}
        return sessions

    def _console_member_status(self, key, user_id):  # type: ignore[no-untyped-def]
        """(status, info) for a user in a TARGET group — the
        _member_status logic aimed at the group being configured
        (the console itself runs in a DM; authority must be read
        where the authority lives)."""
        platform = str(key).split(":", 1)[0]
        adapter = self._group_adapter(platform)
        if adapter is None:
            return "", {}
        fn = getattr(adapter, "admin_member", None) or getattr(
            adapter, "admin_member_mtproto", None)
        if fn is None:
            return "", {}
        from ...social.chat.base import ChatRef

        try:
            info = fn(ChatRef.parse(str(key)), str(user_id))
        except Exception:  # noqa: BLE001 - no proof, no standing
            return "", {}
        if not isinstance(info, dict):
            return "", {}
        status = info.get("status") or info.get("role") or ""
        if not status and isinstance(info.get("result"), dict):
            status = info["result"].get("status") or ""
        return str(status).lower(), info

    def _console_is_admin(self, key, user_id) -> bool:  # type: ignore[no-untyped-def]
        status, _info = self._console_member_status(key, user_id)
        return status in ("creator", "administrator", "admin",
                          "owner")

    def _console_owner_ok(self, key, user_id, message) -> bool:  # type: ignore[no-untyped-def]
        """The owner gate against the TARGET group — the
        _group_owner_ok split, relocated: the operator always;
        WhatsApp any admin; Telegram the group's creator/owner
        only; Discord the server owner."""
        if self._operator_or_false(message):
            return True
        platform = str(key).split(":", 1)[0]
        if platform.startswith("whatsapp"):
            return self._console_is_admin(key, user_id)
        status, info = self._console_member_status(key, user_id)
        if platform.startswith("telegram"):
            return status in ("creator", "owner")
        if platform.startswith("discord"):
            return status in ("owner", "creator") \
                or info.get("is_owner") is True
        return False

    def _console_status_lines(self) -> list:  # type: ignore[no-untyped-def]
        """Runtime status for the console home: process uptime
        from the runtime's own boot marker (console_snapshot),
        per-platform connected duration from each adapter's
        run-loop start stamp, and the userbot session state.
        Runtime state only — a value the runtime doesn't track
        is omitted, never invented."""
        lines: list = []
        snap: dict = {}
        fn = getattr(self, "console_snapshot", None)
        if callable(fn):
            try:
                snap = fn() or {}
            except Exception:  # noqa: BLE001 - status is garnish
                snap = {}
        uptime = snap.get("uptime_s")
        if uptime is not None:
            lines.append(f"uptime {_duration_text(int(uptime))} "
                         "(this process)")
        gateway = getattr(self, "gateway", None)
        adapters = getattr(gateway, "adapters", {}) or {}
        now = time.time()
        for name in sorted(adapters):
            stats = getattr(adapters[name], "stats", {}) or {}
            started = float(stats.get("started_at") or 0)
            if started:
                lines.append(
                    f"{name}: connected "
                    f"{_duration_text(max(0, int(now - started)))}")
            else:
                lines.append(f"{name}: not running")
        if "telegram" in adapters:
            stats = getattr(adapters["telegram"], "stats", {}) \
                or {}
            live = bool(float(stats.get("started_at") or 0))
            lines.append("userbot session: "
                         + ("live" if live else "down"))
        return lines

    def _control_console(self, tail, message):  # type: ignore[no-untyped-def]
        """The DM admin console (§30): an admin opens Devon's DM,
        picks a group THEY admin (proved per group through the
        platform's own admin surface — never assumed from the
        DM), and configures its systems in tappable sections.
        Changes apply immediately and journal to the group's
        /glog naming the admin.  Authority is identical to
        in-group: owner-only items stay owner-only, evaluated
        against the target group."""
        from ...social.chat.base import ChatKind

        if message.chat.kind != ChatKind.DM:
            return ("the console lives in my DM — open a chat "
                    "with me there and send /console.")
        surface = self._group_surface()
        sender = str(message.sender_id or message.sender or "")
        words = str(tail or "").split()
        skey = f"{message.chat.platform}:{sender}"
        session = self._console_session_map().get(skey) or {}
        if time.time() - float(session.get("at") or 0) \
                > self._CONSOLE_TTL_S:
            session = {}
        if not words:
            return self._console_home(surface, message, sender,
                                      skey)
        token = words[0].lower()
        key = (session.get("groups") or {}).get(token)
        if key is None:
            return self._console_home(
                surface, message, sender, skey,
                note="that group handle expired — pick again:")
        if len(words) == 1:
            return self._console_group_menu(surface, key, message,
                                            token)
        section = words[1].lower()
        if section not in self._CONSOLE_SECTIONS:
            return self._console_group_menu(surface, key, message,
                                            token)
        if len(words) == 2:
            return self._console_section(surface, key, message,
                                         token, section)
        # The autonomy pre-gate: a refused flip must not journal
        # as a console change, so the owner check (against the
        # TARGET group) runs before anything is applied.
        if section == "defense" and len(words) > 2 \
                and words[2].lower() == "autonomy" \
                and not self._console_owner_ok(key, sender,
                                               message):
            return ("autonomy is the owner's switch — on "
                    "Telegram only the group owner can flip it "
                    "(WhatsApp: any group admin). Nothing was "
                    "changed.")
        reply = self._console_apply(surface, key, message, sender,
                                    section, words[2:])
        if reply is None:
            return (f"I didn't follow that for the {section} "
                    "section — its card shows the grammar.")
        surface._audit(key, "console_change", by=sender,
                       user_id=sender, section=section,
                       change=" ".join(words[2:]))
        surface.store.save()
        return self._console_section(surface, key, message, token,
                                     section, lead=f"✓ {reply}")

    def _console_home(self, surface, message, sender, skey,  # type: ignore[no-untyped-def]
                      note: str = ""):
        from ...social.chat.typography import Card

        keys = [k for k in sorted(surface.store.data.keys())
                if self._operator_or_false(message)
                or self._console_is_admin(k, sender)]
        if not keys:
            return ("you don't admin any group I serve — the "
                    "console opens for group admins.")
        groups: dict = {}
        rows: list = []
        buttons: list = []
        for i, key in enumerate(keys, 1):
            token = f"g{i}"
            groups[token] = key
            members = surface.store.group(key).get("members") \
                or {}
            claim = surface.claim_state(key)
            claim_txt = (f"system output on ({claim['source']})"
                         if claim["claimed"]
                         else "system output off")
            rows.append(f"{token} — {key} "
                        f"({len(members)} in the ledger) · "
                        f"{claim_txt}")
            buttons.append([(f"⚙️ {token} · {key}",
                             f"console {token}")])
        self._console_session_map()[skey] = {
            "groups": groups, "at": time.time()}
        card = Card("settings", title="Devon console", icon="🛠️")
        if note:
            card.line(note)
        status = self._console_status_lines()
        if status:
            card.block_lines(status, title="Status")
        card.block_lines(rows, title="Your groups")
        card.note("tap a group, or type /console g1 — then a "
                  "section: " + ", ".join(self._CONSOLE_SECTIONS))
        return self._card(message, card, buttons)

    def _console_group_menu(self, surface, key, message, token):  # type: ignore[no-untyped-def]
        from ...social.chat.typography import Card

        esc = surface.escalation_config(key)
        s = surface.store.settings(key)
        card = Card("settings", title=f"Console — {key}",
                    icon="🛠️")
        card.block_lines([
            f"defense autonomy "
            f"{'ON' if (s.get('defense') or {}).get('autonomy', True) else 'OFF'}"
            f" · ladder: {esc.describe()}",
            f"acceptance mode "
            f"{(s.get('acceptance') or {}).get('mode', 'open')}"
            f" · economy "
            f"{'on' if s.get('economy_enabled') else 'off'}"
            f" · governance "
            f"≥{s.get('governance_threshold', 500)}",
            f"welcome "
            f"{'on' if s.get('welcome_enabled') else 'off'}"
            f" · rules {len(s.get('rules') or [])} · presentation "
            f"theme {(s.get('presentation') or {}).get('theme', 'house')}",
            f"expression "
            f"{'on' if (s.get('expression') or {}).get('enabled', True) else 'off'}"
            f" · gifs {(s.get('expression') or {}).get('gifs', 'normal')}",
        ])
        from ...social.chat.expression import buttons_from_spec

        buttons = buttons_from_spec([
            [("defense", "Defense + ladder",
              f"console {token} defense"),
             ("acceptance", "Acceptance",
              f"console {token} acceptance")],
            [("economy", "Economy", f"console {token} economy"),
             ("estate", "Estate", f"console {token} estate")],
            [("presentation", "Presentation",
              f"console {token} presentation"),
             ("expression", "Expression",
              f"console {token} expression")],
            [("welcome", "Welcome + rules",
              f"console {token} welcome"),
             ("back", "All groups", "console")],
        ], settings=s)
        card.note("tap a section, or type its grammar — every "
                  "change lands in the group's /glog with your "
                  "name on it")
        return self._card(message, card, buttons)

    def _console_section(self, surface, key, message, token,  # type: ignore[no-untyped-def]
                         section, lead: str = ""):
        from ...social.chat.typography import (
            DEFAULT_STAMP_TEXT, Card)

        s = surface.store.settings(key)
        card = Card("settings",
                    title=f"{section.title()} — {key}", icon="🛠️")
        if lead:
            card.line(lead)
        buttons: list = []

        from ...social.chat.expression import (
            button_label, buttons_from_spec)

        def _toggle(action: str, label: str, grammar: str,
                    on_now: bool):
            return (button_label(
                        action,
                        f"{label}: {'ON' if on_now else 'OFF'}",
                        settings=s),
                    f"console {token} {section} {grammar} "
                    f"{'off' if on_now else 'on'}")

        if section == "defense":
            d = s.get("defense") or {}
            esc = surface.escalation_config(key)
            card.block_lines([
                f"autonomy "
                f"{'ON' if d.get('autonomy', True) else 'OFF'}"
                " (owner's switch — Telegram: group owner; "
                "WhatsApp: any admin)",
                f"sensors: anticrash "
                f"{'on' if (d.get('anticrash') or {}).get('enabled', True) else 'off'}"
                f" · antichannel "
                f"{'on' if (d.get('antichannel') or {}).get('enabled', True) else 'off'}"
                f" · antitag "
                f"{'on' if (d.get('antitag') or {}).get('enabled', True) else 'off'}"
                f" · antidelete "
                f"{'on' if (d.get('antidelete') or {}).get('enabled', True) else 'off'}"
                f" · antiedit "
                f"{'on' if (d.get('antiedit') or {}).get('enabled', True) else 'off'}",
                f"moderation: flood "
                f"{(s.get('moderation') or {}).get('flood_max', 6)}"
                f" per window · filters "
                f"{len((s.get('moderation') or {}).get('word_filters') or [])}"
                f" · standing mute "
                f"{_duration_text(int(s.get('mute_default_s') or 0)) if s.get('mute_default_s') else 'ask'}"
                f" · mute fine {s.get('fine_amount', 0)}",
                f"ladder: {esc.describe()}",
            ])
            buttons = [
                [_toggle("autonomy", "Autonomy", "autonomy",
                         bool(d.get("autonomy", True))),
                 _toggle("anticrash", "Anticrash", "anticrash",
                         bool((d.get('anticrash') or {})
                              .get('enabled', True)))],
                [_toggle("antichannel", "Antichannel", "antichannel",
                         bool((d.get('antichannel') or {})
                              .get('enabled', True))),
                 _toggle("antitag", "Antitag", "antitag",
                         bool((d.get('antitag') or {})
                              .get('enabled', True)))],
            ]
            card.block_lines([
                "`autonomy on|off` · `anticrash on|off|delete|flag`",
                "`antichannel|antitag|antidelete|antiedit on|off`"
                " · `antitag N` · `antidelete mode log|callout`",
                "`flood N` · `links DOMAINS|off` · "
                "`filter add|remove WORD` · `filter clear` · "
                "`filter action delete|warn|mute`",
                "`warns N` · `mute default 2h|off` · `fine N`",
                "`escalation limit N` · `escalation final "
                "kick|ban` · `escalation window DAYS`",
                "`escalation at N mute [10m]` · `escalation at N "
                "fine AMOUNT` · `escalation at N restrict|kick` "
                "· `escalation clear N` · `escalation reset`",
            ], title="Grammar")
        elif section == "acceptance":
            acc = s.get("acceptance") or {}
            crit = acc.get("criteria") or {}
            on = [n for n in ("username", "photo", "firstname",
                              "lastname", "hygiene", "banned",
                              "bots") if crit.get(n)]
            card.block_lines([
                f"mode {acc.get('mode', 'open')} · action "
                f"{acc.get('action', 'restricted')} · grace "
                f"{int(acc.get('grace_minutes') or 60)}m · "
                f"auto-approve "
                f"{'on' if acc.get('auto_approve', True) else 'off'}",
                "criteria on: "
                + (", ".join(on) if on else "none — door open"),
            ])
            card.block_lines([
                "`mode open|request|invite` · `action "
                "decline|restricted|kick` · `grace MIN`",
                "`username|photo|firstname|lastname|hygiene|"
                "banned|bots on|off` · `auto on|off`",
                "`override CRITERION decline|restricted|kick|"
                "default` · `whitelist add|remove @name|id`",
            ], title="Grammar")
        elif section == "economy":
            card.block_lines([
                f"economy "
                f"{'on' if s.get('economy_enabled') else 'off'}"
                f" · transfer tax "
                f"{s.get('transfer_tax_pct', 2)}% · governance "
                f"threshold {s.get('governance_threshold', 500)}"
                f" · moderation fine {s.get('fine_amount', 0)}",
            ])
            buttons = [[_toggle("economy", "Economy", "economy",
                                bool(s.get("economy_enabled")))]]
            card.block_lines([
                "`economy on|off` · `tax PCT` · `governance N`"
                " · `fine N`",
            ], title="Grammar")
        elif section == "estate":
            est = s.get("estate") or {}
            card.block_lines([
                f"market cut {est.get('market_cut_pct', 5)}% · "
                f"sell-back {est.get('sellback_pct', 70)}% · "
                f"ownership cap {est.get('ownership_cap', 12)}",
                f"mortgage LTV {est.get('ltv_pct', 60)}% · rate "
                f"{est.get('mortgage_rate_pct', 5)}%",
            ])
            card.block_lines([
                "`cut PCT` · `sellback PCT` · `cap N` · "
                "`ltv PCT` · `rate PCT`",
            ], title="Grammar")
        elif section == "presentation":
            pres = s.get("presentation") or {}
            card.block_lines([
                f"theme {pres.get('theme', 'house')} · stamp "
                f"{'ON' if pres.get('stamp', True) else 'OFF'}"
                f" — “{pres.get('stamp_text') or DEFAULT_STAMP_TEXT}”",
            ])
            theme_cycle = {"house": "clean", "clean": "board",
                           "board": "house"}
            theme_now = str(pres.get("theme") or "house")
            buttons = [
                [(button_label("theme", f"Theme: {theme_now}",
                               settings=s),
                  f"console {token} presentation theme "
                  f"{theme_cycle.get(theme_now, 'house')}"),
                 _toggle("stamp", "Stamp", "stamp",
                         bool(pres.get("stamp", True)))],
            ]
            card.block_lines([
                "`theme house|clean|board` · `class game|"
                "economy|… board|clean|default`",
                "`stamp on|off` · `stamp text TEXT|default` · "
                "`stamp classes a,b|default`",
            ], title="Grammar")
        elif section == "welcome":
            card.block_lines([
                f"welcome "
                f"{'on' if s.get('welcome_enabled') else 'off'}"
                f" · goodbye "
                f"{'on' if s.get('goodbye_enabled') else 'off'}"
                f" · service cleanup "
                f"{'on' if s.get('clean_service') else 'off'}"
                f" · rules {len(s.get('rules') or [])}",
            ])
            buttons = [
                [_toggle("welcome", "Welcome", "welcome",
                         bool(s.get("welcome_enabled"))),
                 _toggle("goodbye", "Goodbye", "goodbye",
                         bool(s.get("goodbye_enabled")))],
                [_toggle("cleanup", "Cleanup", "clean",
                         bool(s.get("clean_service")))],
            ]
            card.block_lines([
                "`welcome on|off` · `goodbye on|off` · "
                "`clean on|off`",
                "`rules add TEXT` · `rules clear`",
            ], title="Grammar")
        elif section == "expression":
            expr = s.get("expression") or {}
            lib_line = "library size unknown"
            try:
                svc = getattr(self, "_expression_service", None)
                if svc is not None:
                    lib_line = (f"library {len(svc().library)} "
                                f"items (steals, makes and "
                                f"provider finds — it grows)")
            except Exception:  # noqa: BLE001 - display only
                pass
            card.block_lines([
                f"expression "
                f"{'ON' if expr.get('enabled', True) else 'OFF'}"
                f" · gifs {expr.get('gifs', 'normal')}",
                f"welcome media "
                f"{'on' if expr.get('welcome_media', True) else 'off'}"
                f" · ban theatre "
                f"{'on' if expr.get('ban_theatre', True) else 'off'}"
                f" · {lib_line}",
                "moments: welcome · quest complete · jackpot · "
                "season podium · duel wins · estate milestones · "
                "bans (theatre) — warnings stay dry, always",
            ])
            gif_cycle = {"off": "low", "low": "normal",
                         "normal": "off"}
            gif_now = str(expr.get("gifs") or "normal")
            buttons = [
                [_toggle("expression", "Expression", "",
                         bool(expr.get("enabled", True))),
                 (button_label("gif", f"GIFs: {gif_now}",
                               settings=s),
                  f"console {token} expression gifs "
                  f"{gif_cycle.get(gif_now, 'normal')}")],
                [_toggle("welcome", "Welcome media", "welcome",
                         bool(expr.get("welcome_media", True))),
                 _toggle("ban", "Ban theatre", "ban",
                         bool(expr.get("ban_theatre", True)))],
            ]
            card.block_lines([
                "`on|off` · `gifs off|low|normal`",
                "`welcome on|off` · `ban on|off`",
            ], title="Grammar")
        buttons += buttons_from_spec(
            [[("back", "Sections", f"console {token}"),
              ("home", "Console home", "console")]], settings=s)
        return self._card(message, card, buttons)

    def _console_apply(self, surface, key, message, sender,  # type: ignore[no-untyped-def]
                       section, words):
        """Apply one console grammar line to the TARGET group.
        The words are the section's own grammar — the same words
        /gsettings takes in-group, minus the command head — and
        they land through the same update helpers.  Returns the
        reply line, or None when the words don't parse."""
        words = [str(w) for w in words]
        if not words:
            return None
        head = words[0].lower()
        rest = words[1].lower() if len(words) > 1 else ""
        on = rest == "on"
        if section == "defense":
            if head == "escalation":
                return self._apply_escalation_grammar(
                    surface, key, words[1:])
            if head == "autonomy" and rest in ("on", "off"):
                if not self._console_owner_ok(key, sender,
                                              message):
                    return ("autonomy is the owner's switch — "
                            "on Telegram only the group owner "
                            "can flip it (WhatsApp: any admin).")
                self._defense_update(surface, key, "_root",
                                     autonomy=on)
                return f"defense autonomy {'on' if on else 'off'}."
            if head in ("anticrash", "antichannel", "antitag",
                        "antidelete", "antiedit"):
                if rest in ("on", "off"):
                    self._defense_update(surface, key, head,
                                         enabled=on)
                    return f"{head} {'on' if on else 'off'}."
                if head == "anticrash" and rest in (
                        "delete", "flag"):
                    self._defense_update(surface, key, head,
                                         action=rest)
                    return f"anticrash action: {rest}."
                if head == "antitag" and rest.isdigit():
                    self._defense_update(
                        surface, key, head,
                        max_mentions=max(1, int(rest)))
                    return f"antitag ceiling: {rest} mentions."
                if head in ("antidelete", "antiedit") \
                        and rest == "mode" and len(words) > 2 \
                        and words[2].lower() in ("log", "callout"):
                    self._defense_update(surface, key, head,
                                         mode=words[2].lower())
                    return f"{head} mode: {words[2].lower()}."
                if head in ("antidelete", "antiedit") \
                        and rest == "to" and len(words) > 2 \
                        and words[2].lower() in ("group", "admins"):
                    self._defense_update(
                        surface, key, head,
                        callout_to=words[2].lower())
                    return f"{head} callouts: {words[2].lower()}."
                return None
            if head == "flood" and rest.isdigit():
                current = surface.store.settings(key)
                mod = dict(current["moderation"])
                mod["flood_max"] = max(2, int(rest))
                surface.update_settings(key, moderation=mod)
                return f"flood limit {mod['flood_max']} per window."
            if head == "links":
                current = surface.store.settings(key)
                mod = dict(current["moderation"])
                if rest in ("off", ""):
                    mod["link_allowlist"] = []
                    surface.update_settings(key, moderation=mod)
                    return "links open (no allowlist)."
                domains = [d.strip().lower() for d in
                           " ".join(words[1:]).replace(",", " ")
                           .split() if d.strip()]
                mod["link_allowlist"] = domains
                surface.update_settings(key, moderation=mod)
                return f"link allowlist: {', '.join(domains)}."
            if head == "filter" and len(words) >= 2:
                current = surface.store.settings(key)
                mod = dict(current["moderation"])
                if rest == "add" and len(words) > 2:
                    filters = list(
                        mod.get("word_filters") or []) + [words[2]]
                    mod["word_filters"] = filters
                    surface.update_settings(key, moderation=mod)
                    return f"filter added ({len(filters)} active)."
                if rest == "remove" and len(words) > 2:
                    filters = [f for f in
                               (mod.get("word_filters") or [])
                               if f != words[2]]
                    mod["word_filters"] = filters
                    surface.update_settings(key, moderation=mod)
                    return (f"filter list now has {len(filters)} "
                            "entries.")
                if rest == "clear":
                    mod["word_filters"] = []
                    surface.update_settings(key, moderation=mod)
                    return "filters cleared."
                if rest == "action" and len(words) > 2 \
                        and words[2].lower() in (
                        "delete", "warn", "mute"):
                    mod["filter_action"] = words[2].lower()
                    surface.update_settings(key, moderation=mod)
                    return f"filter action: {words[2].lower()}."
                return None
            if head == "warns" and rest.isdigit():
                surface.update_settings(
                    key, warn_threshold=max(1, int(rest)))
                return (f"warn limit {max(1, int(rest))} before "
                        "an auto-mute.")
            if head == "mute" and rest == "default":
                arg = words[2] if len(words) > 2 else ""
                if arg.lower() in ("off", "none", "0", ""):
                    surface.update_settings(key, mute_default_s=0)
                    return "no standing mute length."
                secs, _why = _parse_duration(arg)
                if secs:
                    surface.update_settings(key,
                                            mute_default_s=secs)
                    return (f"standing mute length "
                            f"{_duration_text(secs)}.")
                return None
            if head == "fine" and rest.lstrip("-").isdigit():
                amount = max(0, int(rest))
                surface.update_settings(key, fine_amount=amount)
                return f"moderation fine {amount} sparks."
            return None
        if section == "acceptance":
            sub = head
            arg = words[1] if len(words) > 1 else ""
            tail = " ".join(words[2:])
            if sub == "mode" and arg.lower() in (
                    "open", "request", "invite"):
                self._acceptance_update(surface, key,
                                        mode=arg.lower())
                return f"join mode: {arg.lower()}."
            if sub in ("username", "photo", "firstname",
                       "lastname", "hygiene", "banned", "bots") \
                    and arg.lower() in ("on", "off"):
                current = surface.store.settings(key)
                crit = dict((current.get("acceptance") or {})
                            .get("criteria") or {})
                crit[sub] = (arg.lower() == "on")
                self._acceptance_update(surface, key,
                                        criteria=crit)
                return f"acceptance criterion {sub}: {arg.lower()}."
            if sub == "action" and arg.lower() in (
                    "decline", "restricted", "kick"):
                self._acceptance_update(surface, key,
                                        action=arg.lower())
                return f"failed checks now mean: {arg.lower()}."
            if sub == "grace" and arg.rstrip("m").isdigit():
                minutes = max(1, min(10080, int(arg.rstrip("m"))))
                self._acceptance_update(surface, key,
                                        grace_minutes=minutes)
                return f"acceptance grace: {minutes} minutes."
            if sub == "auto" and arg.lower() in ("on", "off"):
                self._acceptance_update(
                    surface, key, auto_approve=(arg.lower() == "on"))
                return f"auto-approve: {arg.lower()}."
            if sub == "override" and tail:
                bits = tail.split(None, 1)
                crit_name = bits[0].lower()
                act = bits[1].lower() if len(bits) > 1 else ""
                if crit_name in ("username", "photo", "firstname",
                                 "lastname", "hygiene", "banned",
                                 "bots") and act in (
                        "decline", "restricted", "kick", "default"):
                    current = surface.store.settings(key)
                    overrides = dict(
                        (current.get("acceptance") or {})
                        .get("overrides") or {})
                    if act == "default":
                        overrides.pop(crit_name, None)
                    else:
                        overrides[crit_name] = act
                    self._acceptance_update(surface, key,
                                            overrides=overrides)
                    return f"{crit_name} failures now take: {act}."
                return None
            if sub == "whitelist" and tail:
                bits = tail.split(None, 1)
                verb = bits[0].lower()
                token = bits[1].strip() if len(bits) > 1 else ""
                if verb in ("add", "remove") and token:
                    current = surface.store.settings(key)
                    wl = [str(w) for w in (
                        (current.get("acceptance") or {})
                        .get("bot_whitelist") or [])]
                    if verb == "add" and token not in wl:
                        wl.append(token)
                    if verb == "remove":
                        wl = [w for w in wl if w != token]
                    self._acceptance_update(surface, key,
                                            bot_whitelist=wl)
                    return (f"bot whitelist: "
                            f"{', '.join(wl) or 'empty'}.")
                return None
            return None
        if section == "economy":
            if head == "economy" and rest in ("on", "off"):
                surface.update_settings(key, economy_enabled=on)
                return f"economy {'on' if on else 'off'}."
            if head == "tax" and rest.rstrip("%").isdigit():
                pct = max(0, min(25, int(rest.rstrip("%"))))
                surface.update_settings(key, transfer_tax_pct=pct)
                return f"transfer tax {pct}%."
            if head == "governance" and rest.isdigit():
                surface.update_settings(
                    key, governance_threshold=max(0, int(rest)))
                return f"governance threshold {int(rest)} sparks."
            if head == "fine" and rest.lstrip("-").isdigit():
                amount = max(0, int(rest))
                surface.update_settings(key, fine_amount=amount)
                return f"moderation fine {amount} sparks."
            return None
        if section == "estate":
            return self._apply_estate_grammar(surface, key, words)
        if section == "presentation":
            if head == "theme" and rest in (
                    "house", "clean", "board"):
                self._presentation_update(surface, key, theme=rest)
                return f"presentation theme: {rest}."
            if head == "class" and len(words) > 2:
                cls = rest
                preset = words[2].lower()
                if cls in ("economy", "moderation", "game",
                           "briefing", "announcement", "leaderboard",
                           "settings", "chat") and preset in (
                        "clean", "board", "default"):
                    current = surface.store.settings(key)
                    classes = dict(
                        (current.get("presentation") or {})
                        .get("classes") or {})
                    if preset == "default":
                        classes.pop(cls, None)
                    else:
                        classes[cls] = preset
                    self._presentation_update(surface, key,
                                              classes=classes)
                    return f"{cls} surfaces render {preset}."
                return None
            if head == "stamp" and rest in ("on", "off"):
                self._presentation_update(surface, key, stamp=on)
                return f"footer stamp {'on' if on else 'off'}."
            if head == "stamp" and rest == "text":
                text = " ".join(words[2:])
                if text.lower() in ("", "default", "clear"):
                    self._presentation_update(surface, key,
                                              stamp_text="")
                    return "stamp text back to the house default."
                self._presentation_update(surface, key,
                                          stamp_text=text)
                return f"stamp text: “{text}”."
            if head == "stamp" and rest == "classes":
                text = " ".join(words[2:])
                if text.lower() in ("", "default", "clear"):
                    self._presentation_update(surface, key,
                                              stamp_classes=[])
                    return "stamp classes back to the defaults."
                from ...social.chat.typography import \
                    SURFACE_CLASSES

                wanted = [c.strip().lower() for c in
                          text.replace(" ", ",").split(",")
                          if c.strip() in SURFACE_CLASSES]
                if not wanted:
                    return None
                self._presentation_update(surface, key,
                                          stamp_classes=wanted)
                return f"stamp shows on: {', '.join(wanted)}."
            return None
        if section == "welcome":
            if head == "welcome" and rest in ("on", "off"):
                surface.update_settings(key, welcome_enabled=on)
                return f"welcome {'on' if on else 'off'}."
            if head == "goodbye" and rest in ("on", "off"):
                surface.update_settings(key, goodbye_enabled=on)
                return f"goodbye {'on' if on else 'off'}."
            if head == "clean" and rest in ("on", "off"):
                surface.update_settings(key, clean_service=on)
                return f"service cleanup {'on' if on else 'off'}."
            if head == "rules" and rest == "add" \
                    and len(words) > 2:
                current = surface.store.settings(key)
                rules = list(current.get("rules") or []) + [
                    " ".join(words[2:])]
                surface.update_settings(key, rules=rules)
                return f"rule added ({len(rules)} total)."
            if head == "rules" and rest == "clear":
                surface.update_settings(key, rules=[])
                return "rules cleared."
            return None
        if section == "expression":
            # The console's words ARE the gsettings tail (minus
            # the head): one grammar, one write path (§21).
            return self._apply_expression_grammar(
                surface, key, list(words))
        return None

    # ---- the notice undo (DESIGN §31) ---------------------------
    def _control_undonotice(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        """Undo ONE automated action from its notice: the strike
        it recorded is removed, any fine is refunded, and the
        platform action is reversed as far as the platform
        allows (unmute / unban / unrestrict; a kick needs
        nothing — the member can already rejoin; a deleted
        message is stated plainly as unrestorable).  Journaled
        like the action (notice_undo in /glog, naming the
        admin).  Where the platform edits, the notice itself is
        edited in place and this handler answers with silence;
        elsewhere the outcome posts as one follow-up line."""
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        if not self._group_admin_ok(message):
            return "undoing an action is for admins."
        surface = self._group_surface()
        words = str(tail or "").strip().split()
        nid = words[0] if words else ""
        record = surface.get_notice(key, nid) if nid else None
        if record is None:
            return ("I don't have that notice on record — "
                    "notices age out of the registry (the /glog "
                    "journal keeps the history).")
        if record.get("undone"):
            return "already undone — the reversal stands."
        actor = str(message.sender_id or message.sender or "")
        user_id = str(record.get("user_id") or "")
        bits: list[str] = []
        if record.get("strike_ts"):
            removed = surface.remove_strike(
                key, user_id,
                rule=str(record.get("strike_rule")
                         or record.get("rule") or ""),
                ts=float(record["strike_ts"]))
            if removed is not None:
                bits.append("the strike is removed")
        fine = int(record.get("fine") or 0)
        if fine:
            if surface.refund_fine(key, user_id, fine, by=actor):
                bits.append(f"the {fine}-spark fine is refunded")
            else:
                bits.append("the fine could not be refunded "
                            "(see /glog)")
        action = str(record.get("action") or "")
        extra = record.get("extra") or {}
        if action == "mute":
            ok = self._mod_run(
                surface, key, message, "unmute", user_id,
                self._mod_unrestrict_call(message, user_id))
            bits.append("the mute is lifted" if ok else
                        "the mute could not be lifted (see /glog)")
        elif action == "ban":
            ok = self._mod_run(
                surface, key, message, "unban", user_id,
                self._mod_unban_call(message, user_id))
            bits.append("the ban is lifted" if ok else
                        "the ban could not be lifted (see /glog)")
        elif action == "acceptance_restrict":
            ok = self._mod_run(
                surface, key, message, "unmute", user_id,
                self._mod_unrestrict_call(message, user_id))
            surface.clear_acceptance(key, user_id)
            bits.append("the restriction is lifted" if ok else
                        "the restriction could not be lifted "
                        "(see /glog)")
        elif action == "kick":
            bits.append("a removal needs nothing lifted — they "
                        "can rejoin")
        elif action == "delete":
            bits.append("the deleted message itself can't be "
                        "restored")
        elif action == "drop":
            out = surface.cancel_drop(
                key, str(extra.get("drop_id") or ""), by=actor)
            if out.get("ok"):
                refund = int(out.get("refunded") or 0)
                bits.append("the drop is closed"
                            + (f" — {refund} unclaimed sparks "
                               "refunded" if refund else ""))
            else:
                bits.append("the drop was already closed")
        outcome = "; ".join(bits) if bits else "the record is reversed"
        surface.mark_notice_undone(key, nid, by=actor)
        if self._edit_notice_in_place(surface, key, message,
                                      record, outcome):
            return ""
        mention = str(record.get("mention") or user_id)
        return f"↩️ {mention} [{user_id}] — undone: {outcome}."

    def _mod_unrestrict_call(self, message, user_id):  # type: ignore[no-untyped-def]
        def call(adapter):  # type: ignore[no-untyped-def]
            fn = getattr(adapter, "admin_unrestrict", None)
            if fn is None:
                raise RuntimeError("this platform path cannot "
                                   "lift a restriction")
            return fn(message.chat, user_id)

        return call

    def _mod_unban_call(self, message, user_id):  # type: ignore[no-untyped-def]
        def call(adapter):  # type: ignore[no-untyped-def]
            fn = getattr(adapter, "admin_unban", None)
            if fn is None:
                raise RuntimeError("this platform path cannot "
                                   "unban")
            return fn(message.chat, user_id)

        return call

    def _edit_notice_in_place(self, surface, key, message,  # type: ignore[no-untyped-def]
                              record, outcome: str) -> bool:
        """Edit the posted notice to show its reversal, where
        the adapter supports editing and the posted message's id
        was captured.  Honest False everywhere else — the caller
        posts the follow-up line instead."""
        from ...social.chat.group_notices import undone_text

        message_id = str(record.get("message_id") or "")
        if not message_id:
            return False
        adapter = self._group_adapter(message.chat.platform)
        edit = getattr(adapter, "edit_message", None) \
            if adapter else None
        if edit is None:
            return False
        try:
            out = edit(message.chat, message_id,
                       undone_text(record, outcome), buttons=[])
        except Exception:  # noqa: BLE001 - fall back to a send
            return False
        return out is not None

    # ---- join acceptance commands (DESIGN §28) --------------------
    def _acceptance_update(self, surface, key, **fields):  # type: ignore[no-untyped-def]
        """Mutate the acceptance settings block through the
        store's own settings machinery (defaults-merged,
        persisted) — the _defense_update shape."""
        import copy as _copy

        current = surface.store.settings(key)
        block = _copy.deepcopy(current.get("acceptance") or {})
        block.update(fields)
        surface.update_settings(key, acceptance=block)
        return block

    def _control_recheck(self, message) -> str:  # type: ignore[no-untyped-def]
        """“Check me again” (§28): the restricted member's own
        button/command.  Re-runs the criteria live; a pass
        releases them on the spot and the welcome finally
        fires."""
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group thing — tap it in the group."
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        if not user_id:
            return "I couldn't tell who tapped that."
        adapter = self._group_adapter(message.chat.platform)
        acc_cfg = (surface.store.settings(key)
                   .get("acceptance") or {})
        has_photo = None
        if (acc_cfg.get("criteria") or {}).get("photo"):
            has_photo = self._acceptance_photo(
                adapter, message.chat, user_id)
        res = surface.acceptance_recheck(
            key, user_id,
            username=str(message.sender_username or ""),
            name=str(message.sender or ""), has_photo=has_photo)
        if res.get("state") == "not_restricted":
            return "you're not on the acceptance list — nothing to check."
        if res.get("state") == "passed":
            if res.get("was_restricted"):
                self._execute_group_action(
                    surface, key, adapter, message,
                    {"kind": "acceptance_unrestrict",
                     "user_id": user_id, "rule": "acceptance"})
            if res.get("welcome"):
                member = surface.store.group(key)["members"].get(
                    user_id) or {}
                text = surface.welcome_text_for(
                    key, name=str(member.get("name") or ""),
                    username=str(member.get("username") or ""),
                    group_name=str(message.chat.title or ""))
                if text:
                    return text
            return "✅ all checks pass — you're in. Welcome!"
        from ...social.chat.group_acceptance import fix_it_text
        from ...social.chat.typography import Card

        card = Card("moderation", title="Not quite yet", icon="🚪")
        card.lines(fix_it_text(res.get("reasons") or []).splitlines())
        return self._card(message, card,
                          [[("✅ Check me again", "recheck")]])

    def _joinq_label(self, surface, key, record) -> str:  # type: ignore[no-untyped-def]
        return (record.get("name")
                or (f"@{record['username']}"
                    if record.get("username") else "")
                or surface._member_label(
                    key, str(record.get("user_id") or ""))
                or "someone")

    def _joinq_execute(self, surface, key, adapter, message,  # type: ignore[no-untyped-def]
                       decided: dict, *, approve: bool) -> str:
        """Perform one decided queue record; returns the outcome
        line for the admin.  Execution failures are audited by
        the executors themselves and reported here plainly."""
        record = decided.get("record") or {}
        user_id = str(record.get("user_id") or "")
        label = self._joinq_label(surface, key, record)
        execute = decided.get("execute")
        if execute == "join_request":
            done = self._acceptance_join_request(
                surface, key, adapter, message.chat, user_id,
                approve=approve)
            if not done:
                return (f"{label}: Telegram wouldn't carry that "
                        f"out — the request may have expired or "
                        f"been decided already (see /glog).")
            if not approve:
                return f"🚫 {label}'s request declined."
            admitted = surface.acceptance_request_admitted(
                key, str(record.get("id") or ""))
            if admitted.get("restricted"):
                self._execute_group_action(
                    surface, key, adapter, message,
                    {"kind": "acceptance_restrict",
                     "user_id": user_id, "rule": "acceptance"})
                self._send_acceptance_note(
                    surface, key, message, label,
                    admitted.get("reasons") or [],
                    grace_minutes=admitted.get("grace_minutes"))
                return (f"✅ {label} approved — in, restricted "
                        f"until they fix: "
                        f"{'; '.join(admitted.get('reasons') or [])}")
            if admitted.get("welcome"):
                text = surface.welcome_text_for(
                    key, name=str(record.get("name") or ""),
                    username=str(record.get("username") or ""),
                    group_name=str(message.chat.title or ""))
                if text and self._system_send_ok(
                        surface, key, message, "welcome"):
                    self._send(
                        message.chat, text,
                        buttons=self._welcome_buttons(surface, key),
                        parse_mode=self._rendered_mode(message,
                                                       text))
            return f"✅ {label} approved — they're in."
        if execute == "release":
            done = self._execute_group_action(
                surface, key, adapter, message,
                {"kind": "acceptance_unrestrict",
                 "user_id": user_id, "rule": "acceptance"})
            if record.get("welcome"):
                text = surface.welcome_text_for(
                    key, name=str(record.get("name") or ""),
                    username=str(record.get("username") or ""),
                    group_name=str(message.chat.title or ""))
                if text and self._system_send_ok(
                        surface, key, message, "welcome"):
                    self._send(
                        message.chat, text,
                        buttons=self._welcome_buttons(surface, key),
                        parse_mode=self._rendered_mode(message,
                                                       text))
            return (f"✅ {label} released — full member."
                    if done else
                    f"{label} released on the books — the "
                    f"platform didn't lift the restriction "
                    f"(see /glog).")
        if execute == "kick":
            done = self._execute_group_action(
                surface, key, adapter, message,
                {"kind": "kick", "user_id": user_id,
                 "rule": "acceptance"})
            return (f"🚫 {label} removed from the queue."
                    if done else
                    f"{label}: the removal didn't go through "
                    f"(see /glog).")
        return f"{label}: nothing to do."

    def _control_joinq(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        """The admin join queue (§28): who's waiting, what their
        evaluation said, and the decisions — typed or tapped."""
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        adapter = self._group_adapter(message.chat.platform)
        text = str(tail or "").strip()
        parts = text.split()
        by = str(message.sender_id or message.sender or "")
        if parts and parts[0].lower() in ("approve", "decline") \
                and len(parts) >= 2:
            approve = parts[0].lower() == "approve"
            if parts[1].lower() == "all":
                lines = []
                for record in surface.acceptance_queue(key):
                    decided = surface.acceptance_decide(
                        key, str(record.get("id") or ""),
                        approve=approve, by=by)
                    if decided.get("ok"):
                        lines.append(self._joinq_execute(
                            surface, key, adapter, message,
                            decided, approve=approve))
                if not lines:
                    return "the join queue is empty."
                return "\n".join(lines)
            decided = surface.acceptance_decide(
                key, parts[1], approve=approve, by=by)
            if not decided.get("ok"):
                return ("that request isn't waiting anymore — "
                        "/joinq shows what's live.")
            return self._joinq_execute(
                surface, key, adapter, message, decided,
                approve=approve)
        pending = surface.acceptance_queue(key)
        if not pending:
            return ("the join queue is empty — nobody's waiting "
                    "on a decision.")
        from ...social.chat.typography import Card

        now = time.time()
        card = Card("moderation", title="Join queue", icon="🚪")
        lines = []
        buttons = []
        for record in pending:
            label = self._joinq_label(surface, key, record)
            waited = max(0, int((now - float(
                record.get("ts") or now)) // 60))
            detail = "; ".join(record.get("reasons") or [])
            if not detail and record.get("unknown"):
                detail = ("couldn't verify: "
                          + ", ".join(record["unknown"]))
            if not detail:
                detail = ("meets every rule"
                          if record.get("kind") == "request"
                          else "held by the raid posture")
            lines.append(f"{label} — {detail} "
                         f"· waiting {waited}m")
            rid = str(record.get("id") or "")
            buttons.append(
                [(f"✅ {label}"[:24], f"joinq approve {rid}"),
                 (f"🚫 {label}"[:24], f"joinq decline {rid}")])
        card.lines(lines)
        card.note("requests stay open until decided or the "
                  "invite link is revoked")
        return self._card(message, card, buttons)

    # ---- manual moderation (DESIGN §4: the admin command set) ------
    def _mod_gate(self, message):  # type: ignore[no-untyped-def]
        """(key, None) for an admin in a group, else (None, text)."""
        key = self._group_key_or_none(message)
        if key is None:
            return None, "that's a group command — run it inside the group."
        if not self._group_admin_ok(message):
            return None, "that's an admin call — I'm not taking it from a non-admin."
        return key, None

    def _resolver(self, surface):  # type: ignore[no-untyped-def]
        """The member-resolution organ (DESIGN §29): every
        path Devon has — reply, mention, ledger, bot admin
        surface, the userbot — behind one call."""
        from ...social.chat.member_resolution import MemberResolver

        return MemberResolver(surface, self._group_adapter)

    def _mod_target(self, surface, key, tail, message):  # type: ignore[no-untyped-def]
        """Resolve who a moderation command acts on (§29):
        the AUTHOR of the replied-to message first (platform
        report, ledger message map, then the userbot fetching
        the message itself — the path that reaches pre-
        deployment messages), else the named token by every
        path.  Returns the organ's Resolution."""
        return self._resolver(surface).resolve_command(
            key, message, str(tail or ""))

    def _mod_no_target(self, res, verb: str) -> str:  # type: ignore[no-untyped-def]
        """The honest failure text when a moderation command
        named someone the resolver could not land (§29): a
        channel post is its own fact with its own remedy;
        a named-but-missing member gets ONE line naming what
        was tried — never the generic who-prompt.  "" means
        nothing was named at all (caller shows usage)."""
        if res.channel_post:
            return (f"that message was posted as "
                    f"{res.channel_label} — there's no member "
                    f"to {verb}. Reply to it with /purge to "
                    f"delete it, and /gsettings antichannel "
                    f"keeps channel posts out.")
        if res.named_token:
            return res.exhaustion_text()
        return ""

    def _mod_run(self, surface, key, message, action, user_id,  # type: ignore[no-untyped-def]
                 call) -> bool:
        """Execute one adapter admin call; audit either way; a
        platform refusal (ok=False dict or exception) is a failed
        action, never a reported success."""
        adapter = self._group_adapter(message.chat.platform)
        actor = str(message.sender_id or message.sender or "")
        try:
            if adapter is None:
                raise RuntimeError("no adapter for this platform")
            result = call(adapter)
            if isinstance(result, dict) \
                    and result.get("ok") is False:
                raise RuntimeError(str(result.get("error")
                                       or "platform refused"))
        except Exception as exc:  # noqa: BLE001 - audited, not raised
            surface._audit(key, "admin_action_failed", action=action,
                           user_id=user_id, by=actor,
                           reason=f"{type(exc).__name__}: {exc}")
            surface.store.save()
            return False
        surface._audit(key, "admin_action", action=action,
                       user_id=user_id, by=actor)
        surface.store.save()
        return True

    def _mod_mute_call(self, message, user_id, seconds):  # type: ignore[no-untyped-def]
        def call(adapter):  # type: ignore[no-untyped-def]
            fn = getattr(adapter, "admin_restrict", None)
            if fn is None:
                raise RuntimeError("this platform path cannot mute")
            if seconds and seconds > 0:
                try:
                    return fn(message.chat, user_id,
                              minutes=max(1, seconds // 60))
                except TypeError:
                    return fn(message.chat, user_id,
                              until=int(time.time()) + seconds)
            try:
                return fn(message.chat, user_id, minutes=0)
            except TypeError:
                return fn(message.chat, user_id, until=0)
        return call

    def _control_mute(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        res = self._mod_target(surface, key, tail, message)
        if not res.found:
            note = self._mod_no_target(res, "mute")
            if note:
                return note
            return ("who should I mute? Reply to their message with "
                    "/mute, or /mute @member [10m|2h|1d].")
        user_id, label = res.member.user_id, res.member.label
        seconds, reason = _parse_duration(res.rest)
        if seconds is None:
            # Target first, then the rest (§29): the group's
            # standing default when one is set; the standing
            # hour for a reply-target (the grammar every group
            # bot uses); and for a NAMED member with no length
            # given, ask for the length — never "who?".
            default_s = int(surface.store.settings(
                key).get("mute_default_s") or 0)
            if default_s:
                seconds = default_s
            elif res.via_reply:
                seconds = 3600
            else:
                return (f"for how long should I mute {label}? "
                        f"/mute {res.named_token} 10m|2h|1d — or "
                        f"set a standing default: /gsettings "
                        f"mute default 2h.")
        ok = self._mod_run(
            surface, key, message, "mute", user_id,
            self._mod_mute_call(message, user_id, seconds))
        if not ok:
            return "the mute didn't go through — the platform refused it (it's in /glog)."
        span = "indefinitely" if seconds == 0 else \
            f"for {_duration_text(seconds)}"
        return f"🔇 {label} muted {span}" + \
            (f" — {reason}" if reason else "")

    def _control_unmute(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        res = self._mod_target(surface, key, tail, message)
        if not res.found:
            note = self._mod_no_target(res, "unmute")
            if note:
                return note
            return "who should I unmute? Reply to them or /unmute @member."
        user_id, label = res.member.user_id, res.member.label

        def call(adapter):  # type: ignore[no-untyped-def]
            fn = getattr(adapter, "admin_unrestrict", None)
            if fn is None:
                raise RuntimeError("this platform path cannot unmute")
            return fn(message.chat, user_id)

        if not self._mod_run(surface, key, message, "unmute",
                             user_id, call):
            return "the unmute didn't go through — the platform refused it (it's in /glog)."
        return f"🔊 {label} can speak again."

    def _control_ban(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        res = self._mod_target(surface, key, tail, message)
        if not res.found:
            note = self._mod_no_target(res, "ban")
            if note:
                return note
            return "who should I ban? Reply to their message or /ban @member."
        user_id, label, rest = (res.member.user_id,
                                res.member.label, res.rest)

        def call(adapter):  # type: ignore[no-untyped-def]
            fn = getattr(adapter, "admin_ban", None)
            if fn is None:
                raise RuntimeError("this platform path cannot ban")
            return fn(message.chat, user_id)

        if not self._mod_run(surface, key, message, "ban", user_id,
                             call):
            return "the ban didn't go through — the platform refused it (it's in /glog)."
        return f"🔨 {label} is banned" + \
            (f" — {rest}" if rest.strip() else "")

    def _control_unban(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        res = self._mod_target(surface, key, tail, message)
        if not res.found:
            note = self._mod_no_target(res, "unban")
            if note:
                return note
            return "who should I unban? /unban @member or their id."
        user_id, label = res.member.user_id, res.member.label

        def call(adapter):  # type: ignore[no-untyped-def]
            fn = getattr(adapter, "admin_unban", None)
            if fn is None:
                raise RuntimeError("this platform path cannot unban")
            return fn(message.chat, user_id)

        if not self._mod_run(surface, key, message, "unban",
                             user_id, call):
            return "the unban didn't go through — the platform refused it (it's in /glog)."
        return f"{label} is unbanned — they can rejoin."

    def _control_kick(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        res = self._mod_target(surface, key, tail, message)
        if not res.found:
            note = self._mod_no_target(res, "kick")
            if note:
                return note
            return "who should I kick? Reply to their message or /kick @member."
        user_id, label, rest = (res.member.user_id,
                                res.member.label, res.rest)

        def call(adapter):  # type: ignore[no-untyped-def]
            ban = getattr(adapter, "admin_ban", None)
            unban = getattr(adapter, "admin_unban", None)
            if ban is None or unban is None:
                raise RuntimeError("this platform path cannot kick")
            out = ban(message.chat, user_id)
            if isinstance(out, dict) and out.get("ok") is False:
                return out
            # A kick is a ban immediately lifted: out now, free
            # to rejoin later.
            return unban(message.chat, user_id)

        if not self._mod_run(surface, key, message, "kick",
                             user_id, call):
            return "the kick didn't go through — the platform refused it (it's in /glog)."
        return f"👢 {label} is out" + \
            (f" — {rest}" if rest.strip() else "") + \
            " (they can rejoin)."

    def _control_warn(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        res = self._mod_target(surface, key, tail, message)
        if not res.found:
            note = self._mod_no_target(res, "warn")
            if note:
                return note
            return "who should I warn? Reply to their message or /warn @member."
        user_id, label, rest = (res.member.user_id,
                                res.member.label, res.rest)
        actor = str(message.sender_id or message.sender or "")
        reason = rest.strip()
        count = surface.record_warn(key, user_id, by=actor,
                                    reason=reason)
        # The ladder is the group's own (§30): the count a
        # warning quotes is the NEXT designed step, and landing
        # on a step fires it — an interval step restarts the
        # admin-facing warn count only (the spine keeps its
        # history); the final action resets everything, as the
        # shipped ladder always did.
        esc = surface.escalation_config(key)
        step = esc.action_at(count)
        nxt = esc.next_step(count - 1)
        denom = int(nxt["at"]) if nxt else esc.warn_limit
        if step is None:
            return (f"⚠️ {label} — warning {count}/{denom}"
                    + (f": {reason}" if reason else ""))
        ok, phrase, paid = self._fire_warn_step(
            surface, key, message, user_id, step)
        is_final = count >= esc.warn_limit and \
            step["action"] == esc.final_action
        if is_final:
            surface.clear_warns(key, user_id)
        else:
            surface.reset_warn_bookkeeping(key, user_id)
        if not ok:
            return (f"⚠️ {label}: warning {count}/{denom} — "
                    f"that's the step ({phrase}), but it was "
                    "refused (see /glog).")
        if is_final:
            tail_note = f" — that's the limit: {phrase}, " \
                        "warns reset."
        else:
            tail_note = f" — that's the step: {phrase}."
        self._post_warn_notice(surface, key, message, user_id,
                               label, step, count, paid)
        return f"⚠️ {label}: warning {count}/{denom}{tail_note}"

    def _fire_warn_step(self, surface, key, message, user_id,  # type: ignore[no-untyped-def]
                        step) -> tuple:
        """Execute one ladder step a manual warn landed on.
        Returns (ok, phrase, paid_fine) — the phrase in the
        member's terms, from what ACTUALLY executed."""
        action = str(step.get("action") or "")
        if action == "fine":
            paid = surface.apply_fine(
                key, user_id, amount=int(step.get("amount") or 0))
            if paid:
                return True, f"fined {paid} sparks", paid
            return False, (f"a {int(step.get('amount') or 0)}-spark "
                           "fine (nothing to collect)"), 0
        if action == "mute":
            minutes = int(step.get("minutes") or 60)
            ok = self._mod_run(
                surface, key, message, "mute", user_id,
                self._mod_mute_call(message, user_id, minutes * 60))
            phrase = "muted for an hour" if minutes == 60 else \
                f"muted for {minutes} min"
            return ok, phrase, 0
        if action == "restrict":
            def call(adapter):  # type: ignore[no-untyped-def]
                if not self._acceptance_restrict(
                        adapter, message.chat, user_id, on=True):
                    raise RuntimeError("restrict refused")
                return {"ok": True}

            ok = self._mod_run(surface, key, message, "restrict",
                               user_id, call)
            return ok, "restricted until an admin lifts it", 0
        if action == "kick":
            def kick_call(adapter):  # type: ignore[no-untyped-def]
                ban = getattr(adapter, "admin_ban", None)
                unban = getattr(adapter, "admin_unban", None)
                if ban is None or unban is None:
                    raise RuntimeError("platform can't kick")
                out = ban(message.chat, user_id)
                unban(message.chat, user_id)
                return out

            ok = self._mod_run(surface, key, message, "kick",
                               user_id, kick_call)
            return ok, "removed from the group", 0
        if action == "ban":
            def ban_call(adapter):  # type: ignore[no-untyped-def]
                fn = getattr(adapter, "admin_ban", None)
                if fn is None:
                    raise RuntimeError("platform can't ban")
                return fn(message.chat, user_id)

            ok = self._mod_run(surface, key, message, "ban",
                               user_id, ban_call)
            return ok, "banned from the group", 0
        return False, action, 0

    def _post_warn_notice(self, surface, key, message, user_id,  # type: ignore[no-untyped-def]
                          label, step, count, paid) -> None:
        """The §31 notice for a ladder step a manual warn fired:
        recorded on the surface, posted with its Undo."""
        from ...social.chat.group_defense import (
            consequence_text, render_warning)

        esc = surface.escalation_config(key)
        info = surface.strikes_for(key, user_id)
        strike_ts = 0.0
        for strike in reversed(info["strikes"]):
            if strike.get("rule") == "manual_warn":
                strike_ts = float(strike.get("ts") or 0)
                break
        kind = {"restrict": "acceptance_restrict"}.get(
            str(step.get("action")), str(step.get("action")))
        text = render_warning(
            f"{label} [{user_id}]", "manual_warn",
            consequence_text(
                executed="restrict" if kind
                == "acceptance_restrict" else kind,
                count=count, cfg=surface.defense_config(key),
                warn_threshold=0, esc=esc,
                fine_amount=paid if kind == "fine" else 0,
                oldest_ts=info["oldest_ts"], now=time.time()))
        nid = surface.record_notice(
            key, user_id=user_id, mention=label,
            rule="manual_warn", action=kind, count=count,
            limit=esc.warn_limit, strike_ts=strike_ts,
            strike_rule="manual_warn",
            minutes=int(step.get("minutes") or 0)
            if kind == "mute" else 0,
            fine=paid if kind == "fine" else 0, text=text)
        # The notice RECORD is the state (undo works from it
        # whenever the tap lands); the post is best-effort.
        try:
            self._send_action_notice(surface, key, message,
                                     {"notice_id": nid})
        except Exception:  # noqa: BLE001 - the record is durable
            _log.debug("warn notice post failed", exc_info=True)

    def _control_strikes(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        """The strike ledger (defense brief §2.2): one member's
        record, or the group's current holders when no target is
        named.  Read from the same spine the sensors write."""
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        res = self._mod_target(surface, key, tail, message)
        if not res.found:
            note = self._mod_no_target(res, "look up")
            if note:
                return note
            user_id, label = None, ""
        else:
            user_id, label = res.member.user_id, res.member.label
        if not user_id:
            rows = []
            for uid, member in surface.members(key).items():
                info = surface.strikes_for(key, uid)
                if info["count"]:
                    rows.append((info["count"], uid, member))
            if not rows:
                return "no strikes on record right now."
            rows.sort(key=lambda r: -r[0])
            from ...social.chat.typography import Card

            card = Card("moderation", title="Strike ledger",
                        icon="⚖️")
            card.stats([
                (str(member.get("name") or (
                    f"@{member['username']}"
                    if member.get("username") else uid)),
                 f"{count} strike{'s' if count != 1 else ''}")
                for count, uid, member in rows[:10]])
            card.note("live window")
            return self._card(message, card)
        info = surface.strikes_for(key, user_id)
        window = info["window_days"]
        if not info["count"]:
            return (f"{label} — clean record (no strikes in the "
                    f"last {window} days).")
        from ...social.chat.typography import Card

        card = Card("moderation", title="Strike record", icon="⚖️")
        card.line(f"**{label}** — {info['count']} strike"
                  f"{'s' if info['count'] != 1 else ''} "
                  f"(last {window} days)")
        card.block_lines([
            f"· {strike.get('rule')} ({strike.get('action')})"
            f" — {_fmt_ts(strike.get('ts'))}"
            for strike in info["strikes"]])
        # The ledger reads against the group's OWN ladder (§30):
        # the next designed step, the limit it climbs to, and
        # when the oldest active warn clears.
        from ...social.chat.group_defense import warn_expiry_text

        notes = []
        if info["next_at"]:
            notes.append(f"next: {info['next_label']} "
                         f"at {info['next_at']}")
        notes.append(f"limit {info['limit']} "
                     f"({info['final_action']})")
        expiry = warn_expiry_text(info["oldest_ts"],
                                  window_days=window, now=time.time())
        if expiry:
            notes.append(expiry)
        card.note(" · ".join(notes))
        return self._card(message, card)

    def _control_pin(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        target_id = str(getattr(message, "reply_to", "") or "")
        if not target_id:
            return "reply to the message you want pinned, with /pin."

        def call(adapter):  # type: ignore[no-untyped-def]
            fn = getattr(adapter, "admin_pin", None)
            if fn is None:
                raise RuntimeError("this platform path cannot pin")
            return fn(message.chat, target_id)

        if not self._mod_run(surface, key, message, "pin",
                             target_id, call):
            return "the pin didn't go through — the platform refused it (it's in /glog)."
        return "📌 pinned."

    def _control_unpin(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()

        def call(adapter):  # type: ignore[no-untyped-def]
            fn = getattr(adapter, "admin_unpin", None)
            if fn is None:
                raise RuntimeError("this platform path cannot unpin")
            return fn(message.chat)

        if not self._mod_run(surface, key, message, "unpin", "",
                             call):
            return "the unpin didn't go through — the platform refused it (it's in /glog)."
        return "📌 unpinned."

    def _control_report(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        """Member-side /report: filed into the held audit log with
        the reported member + the reporter's words; admins pull it
        with /glog.  Not admin-gated — reporting is for everyone."""
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        surface = self._group_surface()
        res = self._mod_target(surface, key, tail, message)
        user_id = res.member.user_id if res.found else ""
        label = res.member.label if res.found else (
            res.channel_label if res.channel_post else "")
        reporter = str(message.sender_id or message.sender or "")
        surface._audit(key, "report", user_id=user_id or "",
                       by=reporter, target_label=label,
                       reason=str(tail or "")[:300],
                       replied_text=str(
                           (message.meta or {}).get(
                               "reply_to_text") or "")[:200])
        surface.store.save()
        if user_id:
            return (f"📣 reported {label} — it's in the group log "
                    f"for the admins (/glog).")
        if res.channel_post:
            return (f"📣 reported — that message was posted as "
                    f"{res.channel_label} (a channel post, not a "
                    f"member); it's in the group log for the "
                    f"admins (/glog).")
        if res.named_token:
            return (f"📣 reported — I couldn't pin "
                    f"{res.named_token} to a member, but your "
                    f"words are in the group log for the admins "
                    f"(/glog).")
        return ("📣 reported — it's in the group log for the "
                "admins (/glog). Reply to the message next time "
                "and I'll attach who wrote it.")

    def _control_purge(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        target_id = str(getattr(message, "reply_to", "") or "")
        if not target_id:
            return "reply to the message you want gone, with /purge."

        def call(adapter):  # type: ignore[no-untyped-def]
            fn = getattr(adapter, "admin_delete", None)
            if fn is None:
                raise RuntimeError("this platform path cannot delete")
            return fn(message.chat, target_id)

        if not self._mod_run(surface, key, message, "purge",
                             target_id, call):
            return "the delete didn't go through — the platform refused it (it's in /glog)."
        return "🧹 gone."

    # ---- economy commands (P2) --------------------------------------
    def _economy_gate(self, message):  # type: ignore[no-untyped-def]
        """Shared gate for economy commands: in-group + economy
        on + ledger healthy.  Returns (key, None) or (None, text)."""
        key = self._group_key_or_none(message)
        if key is None:
            return None, "that's a group command — run it inside the group."
        surface = self._group_surface()
        if not surface.system_output_allowed(
                key, via_platform=str(
                    getattr(message.chat, "platform", "") or "")):
            # §33: the economy is a bot system — it does not run
            # (or build state) in a room Devon-the-bot doesn't
            # operate in, whichever account is sitting there.
            return None, ("my group systems aren't running in "
                          "this group — they speak only where my "
                          "bot operates, or where the owner has "
                          "designated the room (/gsettings system "
                          "output on).")
        frozen = surface.economy_frozen(key)
        if frozen:
            return None, ("the group economy is frozen — its ledger "
                          "needs an admin's eyes before any sparks "
                          "move again.")
        if not surface.store.settings(key)["economy_enabled"]:
            return None, "the group economy is off here."
        return key, None

    def _member_ref(self, surface, key, token, message):  # type: ignore[no-untyped-def]
        """Resolve a /pay target to a member id — now the
        §29 organ's job (every path, write-back included);
        kept as the thin delegate the economy handlers call."""
        res = self._resolver(surface).resolve_token(
            key, message, str(token or ""))
        return res.member.user_id if res.found else None

    def _named_ref(self, surface, key, token, message):  # type: ignore[no-untyped-def]
        """(user_id, failure_text): resolve a named economy
        target; on a miss the failure text is the organ's
        honest exhaustion line, not a ledger-only shrug."""
        res = self._resolver(surface).resolve_token(
            key, message, str(token or ""))
        if res.found:
            return res.member.user_id, ""
        return None, res.exhaustion_text()

    def _control_balance(self, message):  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        from ...social.chat.typography import Card, sparks

        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        stmt = surface.economy.statement(key, user_id)
        member = surface.members(key).get(user_id) or {}
        name = str(member.get("name") or message.sender or "you")
        wallet = int(stmt.get("balance", 0))
        bank = surface.economy.bank_balance(key, user_id)
        card = Card("economy", title=name, icon="💰")
        rows = [("Wallet", sparks(wallet), True)]
        if bank:
            rows.append(("Banked", sparks(bank), True))
            rows.append(("Total", sparks(wallet + bank), True))
        # One purse (§25): gems + property stand on the same
        # card as the sparks — one wealth picture, three
        # denominations, each with its own faucets and sinks.
        estate = surface.estate_summary(key, user_id)
        if estate["gems"] or estate["property_count"]:
            rows.append(("Gems", f"{estate['gems']} 💎", True))
        if estate["property_count"]:
            rows.append(
                ("Property",
                 f"{estate['property_count']} · "
                 f"{sparks(estate['property_value'])}", True))
        card.stats(rows)
        for badge in stmt.get("badges", []):
            card.line(f"🏅 {badge.get('badge')} "
                      f"({badge.get('season')})")
        card.stats([
            ("Group treasury",
             sparks(surface.economy.treasury_balance(key)), True),
            ("Jackpot pot",
             sparks(surface.economy.jackpot_balance(key)), True),
        ], title="The group")
        if bank:
            card.note("banked sparks are safe from robbery")
        disclaimer = str(stmt.get("disclaimer") or "")
        if disclaimer:
            card.flavor(disclaimer)
        buttons = [[("⛏️ Mine", "mine"), ("🛠️ Work", "work")],
                   [("🗺️ Quests", "quests"), ("🛒 Shop", "gshop")],
                   [("🏘️ Property", "property"),
                    ("💎 Rent", "rent")]]
        vault_row = []
        if wallet > 0:
            vault_row.append(("🏦 Deposit", "deposit all"))
        if bank > 0:
            vault_row.append(("💵 Withdraw", "withdraw all"))
        if vault_row:
            buttons.append(vault_row)
        return self._card(message, card, buttons)

    def _control_mine(self, message):  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        from ...social.chat.tgbot_buttons import actionable
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        if surface.jail_state(key, user_id):
            return "🔒 no mining from jail — /bail first."
        buttons = [[("🛠️ Work", "work"), ("🪙 My balance", "balance")]]
        entry = surface.economy.mining_claim(key, user_id)
        if entry is None:
            rate = surface.economy.claim_rate(key, user_id)
            return actionable(
                f"⛏️ nothing to claim yet — claims cool down "
                f"between hits, and the daily cap is real. "
                f"Next claim pays <b>{rate}</b> sparks.", buttons)
        rate = surface.economy.hashrate(key, user_id)
        from ...social.chat.typography import Card, sparks

        card = Card("economy", title="Mining claim", icon="⛏️")
        card.line(f"**+{sparks(entry.amount)}**")
        card.stats([
            ("Balance",
             sparks(surface.economy.balance(key, user_id)), True),
            ("Hashrate", str(rate["hashrate"])),
        ])
        card.note("hashrate is engagement: real activity this "
                  "week")
        return self._card(message, card, buttons)

    def _control_work(self, message):  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        if surface.jail_state(key, user_id):
            return "🔒 no shifts from jail — /bail first."
        result = surface.work_shift(key, user_id)
        if not result["ok"]:
            if result["reason"] == "cooldown":
                return (f"🛠️ you're between shifts — next one opens "
                        f"in ~{max(1, result['wait_s'] // 60)} min.")
            return "🛠️ no shift paid — the daily earn cap is hit."
        from ...social.chat.typography import Card, sparks

        card = Card("economy", title="Shift done", icon="🛠️")
        card.line(f"**+{sparks(result['pay'])}**")
        card.stats([("Level", str(result["level"])),
                    ("Balance", sparks(result["balance"]), True)])
        return self._card(
            message, card,
            [[("⛏️ Mine", "mine"), ("🪙 My balance", "balance")]])

    def _control_pay(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        parts = str(tail or "").split()
        amount = None
        target_token = ""
        for part in parts:
            if part.isdigit() and amount is None:
                amount = int(part)
            elif not target_token:
                target_token = part
        if amount is None or not target_token:
            return "usage: /pay @member 50"
        recipient, miss = self._named_ref(surface, key,
                                          target_token, message)
        if not recipient:
            return miss
        sender = str(message.sender_id or message.sender or "")
        result = surface.pay(key, sender, recipient, amount)
        if not result["ok"]:
            if result["reason"] == "funds":
                return "not enough sparks for that."
            return "that payment doesn't work as written."
        from ...social.chat.typography import Card, sparks

        card = Card("economy", title="Payment sent", icon="💸")
        card.line(f"**{sparks(result['net'])}** paid.")
        rows = [("Your balance", sparks(result["balance"]), True)]
        if result["fee"]:
            rows.insert(0, ("Treasury fee",
                            sparks(result["fee"]), True))
        card.stats(rows)
        return self._card(message, card,
                          [[("🪙 My balance", "balance")]])

    def _control_gshop(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        from ...community.group_economy import STATUS_SHOP

        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        text = str(tail or "").strip()
        if text.lower().startswith("buy "):
            item_id = text[4:].strip().lower()
            from ...community.group_real_estate import PREMIUM_SHELF

            if item_id in PREMIUM_SHELF:
                grant = surface.estate_premium_buy(key, user_id,
                                                   item_id)
                if grant is None:
                    item = PREMIUM_SHELF[item_id]
                    gems = surface.estate.gems_balance(key,
                                                       user_id)
                    return (f"not enough gems — {item['name']} "
                            f"costs {item['cost']} 💎, you have "
                            f"{gems} 💎. Gems come from property "
                            f"rent, season rewards and "
                            f"governance gifts — never from "
                            f"sparks.")
                return (f"💎 {grant['name']} is yours — granted "
                        f"for gems and ledgered.")
            grant = surface.economy.buy_status(key, user_id, item_id)
            if grant is None:
                item = STATUS_SHOP.get(item_id)
                if item is None:
                    return "no such item — /gshop lists the shelf."
                return (f"not enough sparks — {item['name']} costs "
                        f"{item['cost']}, you have "
                        f"{surface.economy.balance(key, user_id)}.")
            if item_id == "pinned_message":
                surface.grant_pin(key, user_id)
                return ("📌 Pinned Message bought — your next "
                        "message here gets pinned for the group.")
            if item_id == "shout_out":
                who = (message.sender_username
                       and f"@{message.sender_username}") \
                    or message.sender or "a member"
                return (f"📣 SHOUT-OUT for {who}! bought with "
                        f"sparks, heard by the whole group.")
            return (f"✨ {grant['name']} is yours — granted and "
                    f"ledgered.")
        from ...social.chat.typography import Card, sparks

        balance = surface.economy.balance(key, user_id)
        card = Card("economy", title="Group shop", icon="🛒")
        gems = surface.estate.gems_balance(key, user_id)
        card.line(f"your sparks: **{sparks(balance)}** · "
                  f"your gems: **{gems} 💎**")
        shelf = []
        buy_row: list = []
        buttons: list = []
        for item_id, item in STATUS_SHOP.items():
            afford = balance >= int(item["cost"])
            shelf.append(
                f"`{item_id}` — **{item['name']}** · "
                f"{sparks(item['cost'])} · {item['blurb']}"
                + ("" if afford else " · *not yet*"))
            # Buttons are authored from the LIVE shelf + balance:
            # an item gets a buy button exactly when this member
            # can actually buy it — the buttons match the display.
            if afford:
                buy_row.append(
                    (f"Buy {item['name']} · {item['cost']}",
                     f"gshop buy {item_id}"))
                if len(buy_row) == 2:
                    buttons.append(buy_row)
                    buy_row = []
        if buy_row:
            buttons.append(buy_row)
        card.block_lines(shelf)
        # The premium shelf (§17): gems only — a second purse
        # with its own faucets, never a sparks price in disguise.
        from ...community.group_real_estate import PREMIUM_SHELF

        premium = []
        premium_row: list = []
        for item_id, item in PREMIUM_SHELF.items():
            afford = gems >= int(item["cost"])
            premium.append(
                f"`{item_id}` — **{item['name']}** · "
                f"{item['cost']} 💎 · {item['blurb']}"
                + ("" if afford else " · *not yet*"))
            if afford:
                premium_row.append(
                    (f"Buy {item['name']} · {item['cost']}💎",
                     f"gshop buy {item_id}"))
                if len(premium_row) == 2:
                    buttons.append(premium_row)
                    premium_row = []
        if premium_row:
            buttons.append(premium_row)
        card.block_lines(premium, title="Premium shelf — gems 💎")
        buttons.append([("🪙 My balance", "balance")])
        return self._card(message, card, buttons)

    def _control_quests(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        text = str(tail or "").strip()
        if text.lower().startswith("claim "):
            quest_id = text[6:].strip().lower()
            entry = surface.economy.claim_quest(key, user_id, quest_id)
            if entry is None:
                return ("that quest isn't claimable — either not "
                        "finished yet, or already claimed this "
                        "period.")
            from ...social.chat.typography import Card, sparks

            card = Card("economy", title="Quest claimed", icon="🏅")
            card.line(f"**+{sparks(entry.amount)}**")
            card.stats([("Balance",
                         sparks(surface.economy.balance(
                             key, user_id)), True)])
            return self._card(
                message, card,
                [[("🪙 My balance", "balance"),
                  ("🗺️ Quests", "quests")]])
        from ...social.chat.typography import Card

        progress = surface.economy.quest_progress(key, user_id)
        card = Card("economy", title="Your quests", icon="🗺️")
        mine = []
        buttons = []
        for quest_id, quest in progress.items():
            mark = "✅ claimed" if quest["claimed"] else (
                "🎁 ready — /quests claim " + quest_id
                if quest["complete"] else
                f"{quest['have']}/{quest['need']}")
            mine.append(f"`{quest_id}` — **{quest['name']}**: "
                        f"{quest['blurb']} · {mark} · "
                        f"reward {quest['reward']}")
            # A claim button exists exactly for quests this
            # member can claim RIGHT NOW — authored from the
            # same progress state the list renders.
            if quest["complete"] and not quest["claimed"]:
                buttons.append([(f"🎁 Claim: {quest['name']}",
                                  f"quests claim {quest_id}")])
        card.block_lines(mine)
        # Group quests: the COLLECTIVE board — everyone's
        # activity counts; completion pays the treasury.
        import time as _time

        from ...community.group_economy import _period_key
        group_progress = surface.economy.group_quest_progress(
            key, now=_time.time())
        paid_state = surface._extras(key).get(
            "group_quests_paid") or {}
        together = []
        for quest_id, quest in group_progress.items():
            paid = paid_state.get(quest_id) == _period_key(
                quest["period"], _time.time())
            mark = ("✅ paid to treasury" if paid else
                    "🎉 complete — bonus lands soon"
                    if quest["complete"] else
                    f"{quest['have']}/{quest['need']}")
            together.append(f"`{quest_id}` — **{quest['name']}**: "
                            f"{quest['blurb']} · {mark} · "
                            f"treasury bonus {quest['bonus']}")
        card.block_lines(together,
                         title="Group quests — everyone together")
        buttons.append([("🪙 My balance", "balance")])
        return self._card(message, card, buttons)

    def _control_treasury(self, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        from ...social.chat.typography import Card, sparks

        surface = self._group_surface()
        health = surface.economy_health(key)
        policy = health.get("policy") or {}
        reserved = surface.reserved_treasury(key)
        card = Card("economy", title="Treasury", icon="🏦")
        rows = [("Treasury", sparks(health["treasury"]), True)]
        if reserved:
            rows.append(("Reserved for open drops",
                         sparks(reserved), True))
        rows += [
            ("Circulating", sparks(health["circulating"]), True),
            ("Holders", str(health["holders"])),
            ("Biggest holder",
             f"{int(health['top_share'] * 100)}%"),
            ("Active this week", str(health["active_7d"])),
            ("Jackpot pot", sparks(health.get("jackpot", 0)), True),
        ]
        card.stats(rows)
        if policy.get("throttled"):
            card.line(
                "📉 **monetary policy: faucets throttled** — "
                f"issuance ({sparks(policy['earned'])}) is running "
                f"hot against sinks ({sparks(policy['sunk'])}) "
                "this week, so activity pays half until sinks "
                "catch up.")
        else:
            card.line(
                f"📈 monetary policy: normal — "
                f"{sparks(policy.get('earned', 0))} issued vs "
                f"{sparks(policy.get('sunk', 0))} sunk this week.")
        proposals = surface.open_proposals(key)
        if proposals:
            card.line(f"🗳️ {len(proposals)} treasury proposal(s) "
                      "waiting for a second admin (/approve).")
        card.note("triple 7️⃣ on /slots takes the jackpot pot")
        return self._card(message, card)

    def _control_drop(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        if not self._group_admin_ok(message):
            return "drops are started by admins."
        key, err = self._economy_gate(message)
        if err:
            return err
        words = str(tail or "").split()
        numbers = [int(p) for p in words if p.isdigit()]
        if len(numbers) < 2:
            return ("usage: /drop <total> <per claim> — or "
                    "/drop treasury <total> <per claim> to fund "
                    "it from the treasury.")
        surface = self._group_surface()
        funder = str(message.sender_id or message.sender or "")
        if any(w.lower() == "treasury" for w in words):
            threshold = int(surface.store.settings(
                key).get("governance_threshold") or 0)
            if threshold and numbers[0] > threshold \
                    and not self._operator_or_false(message):
                proposal = surface.propose_spend(
                    key, kind="drop",
                    payload={"total": numbers[0],
                             "per_claim": numbers[1]},
                    by=funder)
                from ...social.chat.tgbot_buttons import actionable
                return actionable(
                    f"🗳️ that treasury drop ({numbers[0]} sparks) "
                    f"is above the governance threshold "
                    f"({threshold}) — proposal "
                    f"<code>{proposal['id']}</code> is open. A "
                    "SECOND admin approves it:",
                    [[("✅ Approve drop", f"approve {proposal['id']}")]])
            result = surface.open_drop_from_treasury(
                key, numbers[0], numbers[1], by=funder)
            if not result["ok"]:
                if result["reason"] == "funds":
                    return (f"the treasury can't fund that — "
                            f"{result.get('available', 0)} available "
                            "after open-drop reservations.")
                return "that drop doesn't work as written."
            from ...social.chat.typography import Card, sparks

            drop = result["drop"]
            claims = drop["total"] // drop["per_claim"]
            card = Card("announcement", title="Drop", icon="🎁")
            card.line(f"**{sparks(drop['total'])}** from the "
                      f"treasury is up for grabs.")
            card.stats([("Per claim", sparks(drop["per_claim"]),
                         True),
                        ("Claims", str(claims))])
            card.note("tap Claim or send /claim")
            nid = self._record_drop_notice(
                surface, key, message, drop, treasury=True)
            return self._card(
                message, card,
                [[(f"🎁 Claim {drop['per_claim']} sparks", "claim"),
                  ("🪙 My balance", "balance")],
                 [("↩️ Cancel drop", f"undonotice {nid}")]])
        result = surface.open_drop(key, funder, numbers[0],
                                   numbers[1])
        if not result["ok"]:
            if result["reason"] == "funds":
                return "you don't hold that many sparks to fund a drop."
            return "that drop doesn't work as written."
        from ...social.chat.typography import Card, sparks

        drop = result["drop"]
        claims = drop["total"] // drop["per_claim"]
        # The button is authored from THIS drop's state: its label
        # carries the actual per-claim amount of the pool on
        # screen — never a generic claim button.
        card = Card("announcement", title="Drop", icon="🎁")
        card.line(f"**{sparks(drop['total'])}** in the pool.")
        card.stats([("Per claim", sparks(drop["per_claim"]), True),
                    ("Claims", str(claims))])
        card.note("tap Claim or send /claim")
        nid = self._record_drop_notice(
            surface, key, message, drop, treasury=False)
        return self._card(
            message, card,
            [[(f"🎁 Claim {drop['per_claim']} sparks", "claim"),
              ("🪙 My balance", "balance")],
             [("↩️ Cancel drop", f"undonotice {nid}")]])

    def _record_drop_notice(self, surface, key, message, drop,  # type: ignore[no-untyped-def]
                            *, treasury: bool) -> str:
        """Record a drop opening as a §31 notice: an admin can
        cancel the drop from its own announcement (unclaimed
        sparks refund; claims already paid stand)."""
        from ...social.chat.group_notices import drop_notice_text

        funder = str(drop.get("funder") or "")
        if treasury:
            # The notice's member is the admin who opened it
            # ("treasury" is the funding source, not a person).
            user_id = str(message.sender_id
                          or message.sender or "")
            mention = str(message.sender or user_id)
        else:
            user_id = funder
            mention = str(message.sender or funder)
        return surface.record_notice(
            key, user_id=user_id, mention=mention, rule="drop",
            action="drop",
            text=drop_notice_text(
                mention, funder, total=int(drop["total"]),
                per_claim=int(drop["per_claim"]),
                treasury=treasury),
            extra={"drop_id": str(drop["id"]),
                   "treasury": treasury})

    def _control_claim(self, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        result = surface.claim_drop(key, user_id)
        if not result["ok"]:
            return {"no_drop": "no drop is live right now.",
                    "empty": "the pool just ran dry — that was the "
                             "last of it.",
                    "already": "you already claimed from this drop.",
                    "treasury": "the pool couldn't pay out — an "
                                "admin needs to look."}.get(
                        result["reason"], "can't claim right now.")
        from ...social.chat.typography import Card, sparks

        if result.get("closed"):
            # The drop's closing record (§31): the pool is done,
            # said once, in the group where it ran.
            if self._system_send_ok(surface, key, message,
                                    "drop_close"):
                self._send_best_effort(message.chat, (
                    f"🎁 the drop is done — "
                    f"**{sparks(result.get('drop_total') or 0)}** "
                    "claimed in full."))
        card = Card("economy", title="Drop claimed", icon="🎁")
        card.line(f"**+{sparks(result['amount'])}** claimed!")
        card.stats([("Pool left", sparks(result["remaining"]),
                     True),
                    ("Your balance", sparks(result["balance"]),
                     True)])
        return self._card(message, card,
                          [[("🪙 My balance", "balance")]])

    # ---- the full economy loop (bank / daily / crime / casino /
    # boards / governance) --------------------------------------

    def _amount_arg(self, tail, wallet: int) -> int | None:  # type: ignore[no-untyped-def]
        """Parse "<n>|all" against a live balance; None = all of
        it (the Septorch grammar the owner reached for first)."""
        text = str(tail or "").strip().lower()
        if not text or text == "all":
            return None
        digits = text.split()[0]
        return int(digits) if digits.isdigit() else -1

    def _control_deposit(self, tail, message):  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        amount = self._amount_arg(
            tail, surface.economy.balance(key, user_id))
        if amount is not None and amount < 0:
            return "usage: /deposit 50 — or /deposit all"
        result = surface.deposit(key, user_id, amount)
        if not result["ok"]:
            if result["reason"] == "empty":
                return "your wallet is empty — nothing to deposit."
            return "that deposit doesn't work as written."
        from ...social.chat.typography import Card, sparks

        card = Card("economy", title="Deposit", icon="🏦")
        card.line(f"deposited **{sparks(result['amount'])}** — "
                  f"now in your vault.")
        card.stats([("Wallet", sparks(result["wallet"]), True),
                    ("Banked", sparks(result["bank"]), True)])
        card.note("robbers can't touch banked sparks")
        return self._card(
            message, card,
            [[("💵 Withdraw", "withdraw all"),
              ("🪙 My balance", "balance")]])

    def _control_withdraw(self, tail, message):  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        amount = self._amount_arg(
            tail, surface.economy.bank_balance(key, user_id))
        if amount is not None and amount < 0:
            return "usage: /withdraw 50 — or /withdraw all"
        result = surface.withdraw(key, user_id, amount)
        if not result["ok"]:
            if result["reason"] == "empty":
                return "your bank vault is empty."
            return "that withdrawal doesn't work as written."
        from ...social.chat.typography import Card, sparks

        card = Card("economy", title="Withdraw", icon="💵")
        card.line(f"withdrew **{sparks(result['amount'])}** — "
                  f"back in your wallet.")
        card.stats([("Wallet", sparks(result["wallet"]), True),
                    ("Banked", sparks(result["bank"]), True)])
        return self._card(message, card,
                          [[("🪙 My balance", "balance")]])

    def _control_group_daily(self, message):  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        result = surface.daily_claim(key, user_id)
        if not result["ok"]:
            if result["reason"] == "cooldown":
                return (f"🌅 already claimed today — streak at "
                        f"{result.get('streak', 0)}. Tomorrow it "
                        "grows.")
            return "🌅 no daily paid — the earn cap is hit."
        from ...social.chat.typography import Card, sparks

        card = Card("economy", title="Daily", icon="🌅")
        card.line(f"daily claimed: **+{sparks(result['amount'])}**")
        card.stats([("Streak", f"{result['streak']} days"),
                    ("Balance", sparks(result["balance"]), True)])
        return self._card(
            message, card,
            [[("🪙 My balance", "balance"), ("🛠️ Work", "work")]])

    def _rob_target(self, surface, key, tail, message):  # type: ignore[no-untyped-def]
        """Who gets robbed (§29): the replied-to author, or the
        named member — by every resolution path.  (The old
        code stringified the reply-sender DICT into a phantom
        member id; the organ returns the real ID.)"""
        res = self._resolver(surface).resolve_command(
            key, message, str(tail or ""))
        return res.member.user_id if res.found else None

    def _control_rob(self, tail, message):  # type: ignore[no-untyped-def]
        import random

        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        actor = str(message.sender_id or message.sender or "")
        target = self._rob_target(surface, key, tail, message)
        if not target:
            return ("usage: reply to a member with /rob, or "
                    "/rob @member")
        if target == actor:
            return "robbing yourself is just moving sparks around."
        result = surface.rob(key, actor, target,
                             rng=random.Random())
        if not result["ok"]:
            reason = result["reason"]
            if reason == "jailed":
                jail = result.get("jail") or {}
                from ...social.chat.tgbot_buttons import actionable
                return actionable(
                    f"🔒 you're in jail — bail is "
                    f"<b>{jail.get('bail', 0)} sparks</b> to the "
                    "treasury, or wait it out.",
                    [[("🔓 Pay bail", "bail")]])
            if reason == "cooldown":
                return (f"🥷 too soon — the heat dies down in "
                        f"~{max(1, result['wait_s'] // 60)} min.")
            if reason == "empty":
                return "their wallet is empty — nothing to take."
            return "that robbery doesn't work as written."
        from ...social.chat.typography import Card, sparks

        if result.get("success"):
            card = Card("economy", title="Got away", icon="🥷")
            card.line(f"**{sparks(result['amount'])}** lifted.")
            card.stats([("Balance", sparks(result["balance"]),
                         True)])
            card.note("bank it before someone returns the "
                      "favour: /deposit all")
            return self._card(
                message, card,
                [[("🏦 Deposit", "deposit all"),
                  ("🪙 My balance", "balance")]])
        card = Card("economy", title="Caught", icon="🚔")
        card.line("You're in jail for 30 minutes.")
        card.stats([("Bail", sparks(result["bail"]), True)])
        card.note("bail goes to the treasury")
        return self._card(message, card,
                          [[("🔓 Pay bail", "bail")]])

    def _control_bail(self, message):  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        result = surface.bail(key, user_id)
        if not result["ok"]:
            if result["reason"] == "free":
                return "you're not in jail."
            return (f"bail is {result.get('cost', 0)} sparks and "
                    "your wallet can't cover it.")
        from ...social.chat.typography import Card, sparks

        card = Card("economy", title="Bail paid", icon="🔓")
        card.line(f"**{sparks(result['cost'])}** to the treasury "
                  f"— you're out.")
        card.stats([("Balance", sparks(result["balance"]), True)])
        return self._card(message, card)

    def _bet_arg(self, tail, default: int = 10) -> int:  # type: ignore[no-untyped-def]
        for part in str(tail or "").split():
            if part.isdigit():
                return int(part)
        return default

    def _control_slots(self, tail, message):  # type: ignore[no-untyped-def]
        import random

        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        bet = self._bet_arg(tail)
        result = surface.slots(key, user_id, bet,
                               rng=random.Random())
        if not result["ok"]:
            if result.get("reason") == "jailed":
                return "🔒 no slots from jail — /bail first."
            return "not enough sparks in your wallet for that bet."
        reels = " ".join(result["symbols"])
        from ...social.chat.expression import buttons_from_spec
        from ...social.chat.typography import Card, sparks

        buttons = buttons_from_spec(
            [[("spin", f"Spin again ({bet})", f"slots {bet}"),
              ("balance", "My balance", "balance")]],
            settings=surface.store.settings(key))
        if result.get("jackpot"):
            card = Card("game", title="Slots", icon="🎰")
            card.board(reels)
            card.line(f"🎉 **JACKPOT — {sparks(result['jackpot'])}!** "
                      f"plus {sparks(result['payout'])} from the "
                      f"reels.")
            card.stats([("Balance", sparks(result["balance"]),
                         True)])
            # The jackpot moment (§21): the room celebrates in
            # media too, where expression is on and claimed.
            fire = getattr(self, "_fire_expression", None)
            if fire is not None:
                fire(message, "jackpot")
            return self._card(message, card, buttons)
        if result["payout"] > 0:
            card = Card("game", title="Slots", icon="🎰")
            card.board(reels)
            card.line(f"**+{sparks(result['payout'])}** "
                      f"(×{result['multiplier']})")
            card.stats([("Balance", sparks(result["balance"]),
                         True)])
            return self._card(message, card, buttons)
        card = Card("game", title="Slots", icon="🎰")
        card.board(reels)
        card.line("no win this spin.")
        card.stats([("Jackpot pot",
                     sparks(surface.economy.jackpot_balance(key)),
                     True)])
        card.note("the house share feeds the treasury and the "
                  "jackpot pot")
        return self._card(message, card, buttons)

    def _control_coinflip(self, tail, message):  # type: ignore[no-untyped-def]
        import random

        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        bet = self._bet_arg(tail)
        choice = "tails" if "tail" in str(tail or "").lower() \
            else "heads"
        result = surface.coinflip(key, user_id, bet, choice,
                                  rng=random.Random())
        if not result["ok"]:
            if result.get("reason") == "jailed":
                return "🔒 no coinflips from jail — /bail first."
            return "not enough sparks in your wallet for that bet."
        from ...social.chat.expression import buttons_from_spec
        from ...social.chat.typography import Card, sparks

        buttons = buttons_from_spec(
            [[("flip", f"Flip again ({bet})", f"coinflip {bet}"),
              ("balance", "My balance", "balance")]],
            settings=surface.store.settings(key))
        card = Card("game", title="Coinflip", icon="🪙")
        card.board(f"🪙 {result['flip']}")
        if result["won"]:
            card.line(f"you called it — "
                      f"**+{sparks(result['payout'])}**")
            card.stats([("Balance", sparks(result["balance"]),
                         True)])
        else:
            card.line("not your call this time.")
            card.note("the house share feeds the treasury and "
                      "the jackpot")
        return self._card(message, card, buttons)

    def _control_top(self, message):  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        from ...social.chat.typography import Card, sparks

        surface = self._group_surface()
        rows = surface.top_holders(key, limit=10)
        if not rows:
            return "no sparks held yet — /work and /mine start it."
        members = surface.members(key)
        card = Card("leaderboard", title="richest in the group",
                    icon="🏆")
        medals = ["🥇", "🥈", "🥉"]
        stats = []
        for i, row in enumerate(rows):
            record = members.get(row["member"]) or {}
            who = str(record.get("name") or record.get("username")
                      or row["member"])
            mark = medals[i] if i < 3 else f"{i + 1}."
            value = sparks(row.get("worth", row["total"]))
            if row["bank"]:
                value += f" (🏦 {sparks(row['bank'])} banked)"
            if row.get("property"):
                value += (f" (🏘️ {sparks(row['property'])} "
                          f"property)")
            stats.append((f"{mark} {who}", value, True))
        card.stats(stats)
        return self._card(message, card)

    def _control_tax(self, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        from ...social.chat.typography import Card, sparks

        surface = self._group_surface()
        s = surface.store.settings(key)
        policy = surface.economy.issuance_state(key)
        state = ("throttled (activity pays half)"
                 if policy["throttled"] else "normal")
        card = Card("economy", title="The group's monetary "
                    "policy", icon="🧾")
        card.stats([
            ("Transfer tax",
             f"{s.get('transfer_tax_pct', 2)}% to the treasury"),
            ("Moderation fine", sparks(s.get("fine_amount", 0)),
             True),
            ("Earn caps", "200/day · 800/week"),
            ("Governance",
             f"treasury spends above "
             f"{sparks(s.get('governance_threshold', 500))} need "
             f"a second admin"),
            ("Issuance this week",
             f"{sparks(policy['earned'])} issued / "
             f"{sparks(policy['sunk'])} sunk"),
        ])
        card.note(f"faucets {state}")
        return self._card(message, card)

    def _control_grant(self, tail, message):  # type: ignore[no-untyped-def]
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        if not self._group_admin_ok(message):
            return "treasury grants are for admins."
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        raw = str(tail or "").strip()
        if raw.lower().startswith("gems "):
            return self._control_grant_gems(raw[5:].strip(),
                                            message, key, surface)
        parts = str(tail or "").split()
        amount = None
        target_token = ""
        for part in parts:
            if part.isdigit() and amount is None:
                amount = int(part)
            elif not target_token:
                target_token = part
        if amount is None or not target_token:
            return "usage: /grant @member 100"
        recipient, miss = self._named_ref(surface, key,
                                          target_token, message)
        if not recipient:
            return miss
        admin = str(message.sender_id or message.sender or "")
        threshold = int(surface.store.settings(
            key).get("governance_threshold") or 0)
        # Governance binds ADMINS spending the country's money;
        # the operator is sovereign (Septorch's owner posture)
        # and never deadlocks against his own threshold.
        if threshold and amount > threshold \
                and not self._operator_or_false(message):
            proposal = surface.propose_spend(
                key, kind="grant",
                payload={"recipient": recipient, "amount": amount,
                         "reason": "admin grant"}, by=admin)
            from ...social.chat.tgbot_buttons import actionable
            return actionable(
                f"🗳️ that grant ({amount} sparks) is above the "
                f"governance threshold ({threshold}) — proposal "
                f"<code>{proposal['id']}</code> is open. A SECOND "
                "admin approves it:",
                [[("✅ Approve grant", f"approve {proposal['id']}")]])
        result = surface.grant(key, recipient, amount, by=admin)
        if not result["ok"]:
            return (f"the treasury can't cover that — "
                    f"{result.get('available', 0)} available after "
                    "open-drop reservations.")
        from ...social.chat.typography import Card, sparks

        card = Card("economy", title="Grant paid", icon="🏦")
        card.line(f"**{sparks(amount)}** granted from the "
                  f"treasury.")
        card.stats([("Treasury now", sparks(result["treasury"]),
                     True)])
        return self._card(message, card)

    def _control_grant_gems(self, tail, message, key, surface):  # type: ignore[no-untyped-def]
        """/grant gems @member 25: a gems gift — the premium
        purse's governance road (§17/§25).  Gems are never
        minted on one admin's word above the threshold: the
        gift becomes a proposal a second admin executes; the
        operator, sovereign as ever, may gift directly."""
        amount = None
        target_token = ""
        for part in str(tail or "").split():
            if part.isdigit() and amount is None:
                amount = int(part)
            elif not target_token:
                target_token = part
        if amount is None or not target_token:
            return "usage: /grant gems @member 25"
        recipient, miss = self._named_ref(surface, key,
                                          target_token, message)
        if not recipient:
            return miss
        admin = str(message.sender_id or message.sender or "")
        threshold = int(surface.store.settings(
            key).get("governance_threshold") or 0)
        if threshold and amount > threshold \
                and not self._operator_or_false(message):
            proposal = surface.propose_spend(
                key, kind="gems_gift",
                payload={"recipient": recipient,
                         "amount": amount,
                         "reason": "admin gems gift"}, by=admin)
            from ...social.chat.tgbot_buttons import actionable
            return actionable(
                f"🗳️ that gems gift ({amount} 💎) is above the "
                f"governance threshold ({threshold}) — "
                f"proposal <code>{proposal['id']}</code> is "
                f"open. A SECOND admin approves it:",
                [[("✅ Approve gift",
                   f"approve {proposal['id']}")]])
        result = surface.estate_grant_gems(
            key, recipient, amount,
            proposal_id="operator-direct", by=admin)
        if not result["ok"]:
            return "that gift can't be issued right now."
        from ...social.chat.typography import Card

        card = Card("economy", title="Gems gifted", icon="💎")
        card.line(f"**{amount} 💎** gifted from the premium "
                  f"purse — through governance, on the record.")
        card.stats([("Their gems now",
                     f"{result['gems']} 💎", True)])
        return self._card(message, card)

    def _control_approve(self, tail, message):  # type: ignore[no-untyped-def]
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        if not self._group_admin_ok(message):
            return "approvals are for admins."
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        pid = str(tail or "").strip().split()[0] \
            if str(tail or "").strip() else ""
        if not pid:
            open_now = surface.open_proposals(key)
            if not open_now:
                return "no treasury proposals are open."
            from ...social.chat.typography import Card

            card = Card("economy", title="Open treasury "
                        "proposals", icon="🗳️")
            card.block_lines([
                f"`{open_pid}` — {proposal.get('kind')}: "
                f"{proposal.get('payload') or {}}"
                for open_pid, proposal in open_now.items()])
            card.note("approve with /approve <id> (a different "
                      "admin than the proposer)")
            return self._card(message, card)
        admin = str(message.sender_id or message.sender or "")
        proposal = surface.open_proposals(key).get(pid) or {}
        result = surface.approve_spend(key, pid, by=admin)
        if not result["ok"]:
            return {"no_proposal": "no open proposal with that id.",
                    "self_approve": "the proposer can't approve "
                                    "their own spend — a second "
                                    "admin has to.",
                    "funds": "the treasury can't cover it any "
                             "more."}.get(result["reason"],
                                          "can't approve that.")
        # A governance execution is a §31 notice consumer: the
        # group sees what its treasury just did, on the record.
        kind = str(proposal.get("kind") or "")
        payload = proposal.get("payload") or {}
        if kind == "drop" and result.get("drop"):
            self._announce_governance_drop(surface, key, message,
                                           result["drop"])
        elif kind == "grant":
            recipient = str(payload.get("recipient") or "")
            label = self._estate_member_label(surface, key,
                                              recipient)
            text = (f"🏦 {label} [{recipient}] received "
                    f"**{int(payload.get('amount') or 0)} sparks** "
                    f"from the treasury — proposal approved by "
                    f"{message.sender or 'an admin'}.")
            surface.record_notice(
                key, user_id=recipient, mention=label,
                rule="governance", action="grant", text=text,
                extra={"reversible": False, "proposal": pid})
            if self._system_send_ok(surface, key, message,
                                    "governance"):
                self._send_best_effort(message.chat, text)
        return (f"✅ proposal <code>{pid}</code> approved and "
                "executed.")

    def _announce_governance_drop(self, surface, key, message,  # type: ignore[no-untyped-def]
                                  drop) -> None:
        """A drop a governance vote opened gets the same public
        announcement (and the same undoable notice) as an
        admin-opened one — before this slice it opened silently
        and members could not claim what they never saw."""
        from ...social.chat.typography import Card, sparks

        claims = int(drop["total"]) // int(drop["per_claim"])
        card = Card("announcement", title="Drop", icon="🎁")
        card.line(f"**{sparks(drop['total'])}** from the "
                  f"treasury is up for grabs.")
        card.stats([("Per claim", sparks(drop["per_claim"]), True),
                    ("Claims", str(claims))])
        card.note("tap Claim or send /claim")
        nid = self._record_drop_notice(surface, key, message,
                                       drop, treasury=True)
        rendered = self._card(
            message, card,
            [[(f"🎁 Claim {drop['per_claim']} sparks", "claim"),
              ("🪙 My balance", "balance")],
             [("↩️ Cancel drop", f"undonotice {nid}")]])
        if self._system_send_ok(surface, key, message, "drop"):
            self._send_best_effort(
                message.chat, rendered.text,
                buttons=rendered.buttons,
                parse_mode=self._rendered_mode(message,
                                               rendered.text))

    # ── real estate (§17 slice 2 / §20: the whole class) ──────
    def _estate_gate(self, message):  # type: ignore[no-untyped-def]
        """Economy gate + the estate freeze: (key, None) or
        (None, text)."""
        key, err = self._economy_gate(message)
        if err:
            return None, err
        if self._group_surface().estate_frozen(key):
            return None, ("the property market is frozen — its "
                          "records need an admin's eyes before "
                          "anything trades again.")
        return key, None

    def _estate_member_label(self, surface, key, user_id):  # type: ignore[no-untyped-def]
        record = surface.members(key).get(str(user_id)) or {}
        return str(record.get("name") or record.get("username")
                   or user_id)

    def _control_property(self, tail, message):  # type: ignore[no-untyped-def]
        key, err = self._estate_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        if str(tail or "").strip().lower() == "manager":
            return self._estate_manager(message, key, surface,
                                        user_id)
        from ...community.group_real_estate import (
            MANAGER_LEVELS, TIERS)
        from ...social.chat.typography import (
            Card, fmt_amount, sparks)

        summary = surface.estate_summary(key, user_id)
        card = Card("economy", title="Property", icon="🏘️")
        card.stats([
            ("Your gems", f"{summary['gems']} 💎", True),
            ("Property value",
             sparks(summary["property_value"]), True),
            ("Rent standing",
             f"{summary['rent_ready_gems']} 💎", True),
        ])
        wallet = surface.economy.balance(key, user_id)
        gems = summary["gems"]
        buy_buttons: list = []
        for row in surface.estate_catalog(key):
            prices = []
            for tier_id, spec in TIERS.items():
                price = row["tiers"][tier_id]
                if spec["currency"] == "gems":
                    prices.append(f"{spec['icon']} {price}💎")
                else:
                    prices.append(
                        f"{spec['icon']} {fmt_amount(price)}")
            trend = row["trend_pct"]
            arrow = "▲" if trend > 0 else (
                "▼" if trend < 0 else "▬")
            card.block_lines([
                f"**{row['name']}** · index {row['value']} "
                f"({arrow} {abs(trend)}% this week)",
                " · ".join(prices),
            ])
            afford = [(t, row["tiers"][t]) for t in TIERS
                      if (TIERS[t]["currency"] == "sparks"
                          and row["tiers"][t] <= wallet)
                      or (TIERS[t]["currency"] == "gems"
                          and row["tiers"][t] <= gems)]
            if afford:
                tier_id, price = afford[0]
                cur = "💎" if TIERS[tier_id]["currency"] == "gems" \
                    else " sparks"
                buy_buttons.append(
                    (f"Buy {TIERS[tier_id]['name']} · "
                     f"{row['name']} ({fmt_amount(price)}{cur})",
                     f"buyproperty {row['district']} {tier_id}"))
        if summary["properties"]:
            lines = []
            for prop in summary["properties"]:
                spec = TIERS[prop["tier"]]
                cur = "💎" if prop["currency"] == "gems" else ""
                flags = []
                if prop.get("listed"):
                    flags.append("on the market")
                if prop.get("auction"):
                    flags.append("at auction")
                if prop.get("mortgage"):
                    flags.append("mortgaged")
                if float(prop.get("insured_until") or 0) > 0:
                    flags.append("insured")
                if int(prop.get("repair_debt") or 0):
                    flags.append(
                        f"repair debt {prop['repair_debt']}")
                lines.append(
                    f"`{prop['id']}` {spec['icon']} "
                    f"**{spec['name']}** — worth "
                    f"{fmt_amount(prop['value'])}{cur} · rent "
                    f"{prop['rent_gems']}💎 standing"
                    + (f" · {' · '.join(flags)}" if flags else ""))
            card.block_lines(lines, title="Yours")
        level = summary["manager_level"]
        mgr = MANAGER_LEVELS.get(level)
        card.note(
            f"manager: {mgr['name'] if mgr else 'none'} — "
            f"/property manager hires the next rung "
            f"(auto-collect, more rent, longer accrual)")
        buttons = []
        if summary["rent_ready_gems"] > 0:
            buttons.append([("💎 Collect rent", "rent")])
        for i in range(0, len(buy_buttons), 2):
            buttons.append(buy_buttons[i:i + 2])
        buttons.append([("🏪 Market", "listproperty"),
                        ("🔨 Auctions", "auctions")])
        return self._card(message, card, buttons)

    def _estate_manager(self, message, key, surface, user_id):  # type: ignore[no-untyped-def]
        result = surface.estate_manager_buy(key, user_id)
        if not result["ok"]:
            if result["reason"] == "top":
                return ("🧑‍💼 you already run an Estate Agency — "
                        "the top rung. Your whole portfolio is "
                        "managed.")
            return (f"🧑‍💼 the next rung costs "
                    f"{result.get('cost', 0)} sparks — you have "
                    f"{surface.economy.balance(key, user_id)}.")
        from ...social.chat.typography import Card, sparks

        card = Card("economy", title="Management hired",
                    icon="🧑‍💼")
        card.line(f"**{result['name']}** now runs your "
                  f"properties.")
        card.stats([
            ("Rent", f"+{result['yield_pct']}%"),
            ("Accrual cap", f"{result['cap_days']} days"),
            ("Auto-collect", "yes — even while you're jailed "
                             "or away" if result["auto"] else "no"),
        ])
        card.note(f"paid {sparks(result['cost'])} to the "
                  f"treasury")
        return self._card(message, card,
                          [[("🏘️ My property", "property")]])

    def _control_buyproperty(self, tail, message):  # type: ignore[no-untyped-def]
        key, err = self._estate_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        parts = str(tail or "").split()
        if parts and parts[0].lower() == "public":
            return self._estate_buy_public(message, key, surface,
                                           parts[1:], user_id)
        if len(parts) < 2:
            return ("usage: /buyproperty <district> <tier> — "
                    "/property shows the districts and tiers.")
        district, tier = parts[0].lower(), parts[1].lower()
        result = surface.estate_buy(key, user_id, district, tier)
        if not result["ok"]:
            return {
                "catalog": "no such district or tier — "
                           "/property shows the map.",
                "cap": f"you hold the group's property cap "
                       f"({result.get('cap')}) — sell or trade "
                       f"before buying more.",
                "funds": f"not enough sparks — that costs "
                         f"{result.get('price')} and you have "
                         f"{surface.economy.balance(key, user_id)}.",
                "gems": f"that tier is priced in gems 💎 — it "
                        f"costs {result.get('price')}💎 and you "
                        f"have "
                        f"{surface.estate.gems_balance(key, user_id)}💎. "
                        f"Gems come from property rent, season "
                        f"rewards and governance gifts — never "
                        f"from sparks.",
            }.get(result["reason"], "can't buy that.")
        from ...community.group_real_estate import DISTRICTS, TIERS
        from ...social.chat.typography import Card, sparks

        prop = result["property"]
        spec = TIERS[prop["tier"]]
        card = Card("economy", title="Property bought",
                    icon="🏘️")
        card.line(f"**{spec['name']}** in "
                  f"{DISTRICTS[prop['district']]['name']} is "
                  f"yours.")
        price_text = (f"{result['price']} 💎"
                      if prop["currency"] == "gems"
                      else sparks(result["price"]))
        card.stats([
            ("Paid", price_text, True),
            ("Rent", f"{spec['rent']} 💎 / day", True),
            ("District index",
             str(result["index"]["value"])),
            ("Property id", prop["id"]),
        ])
        card.note("rent accrues daily — /rent collects it as "
                  "gems")
        return self._card(
            message, card,
            [[("🏘️ My property", "property"),
              ("💎 Rent", "rent")]])

    def _estate_buy_public(self, message, key, surface, parts,  # type: ignore[no-untyped-def]
                           user_id):
        if not self._group_admin_ok(message):
            return "public property is an admin call."
        if len(parts) < 2:
            return ("usage: /buyproperty public <district> "
                    "<tier>")
        district, tier = parts[0].lower(), parts[1].lower()
        from ...community.group_real_estate import DISTRICTS, TIERS

        if district not in DISTRICTS or tier not in TIERS:
            return "no such district or tier — /property shows it."
        if TIERS[tier]["currency"] == "gems":
            return ("public property is bought with treasury "
                    "sparks — gem tiers are members' premium "
                    "shelf only.")
        price = surface.estate.price(key, district, tier)
        threshold = int(surface.store.settings(
            key).get("governance_threshold") or 0)
        if threshold and price > threshold \
                and not self._operator_or_false(message):
            proposal = surface.propose_spend(
                key, kind="estate_public",
                payload={"district": district, "tier": tier},
                by=user_id)
            from ...social.chat.tgbot_buttons import actionable
            return actionable(
                f"🗳️ a public {TIERS[tier]['name']} "
                f"({price} sparks of treasury money) is above "
                f"the governance threshold ({threshold}) — "
                f"proposal <code>{proposal['id']}</code> is "
                f"open. A SECOND admin approves it:",
                [[("✅ Approve purchase",
                   f"approve {proposal['id']}")]])
        result = surface.estate_buy_public(key, district, tier,
                                           by=user_id)
        if not result["ok"]:
            return (f"the treasury can't cover that — "
                    f"{result.get('available', 0)} available "
                    f"after reservations.")
        from ...community.group_real_estate import DISTRICTS
        from ...social.chat.typography import Card, sparks

        card = Card("economy", title="Public property bought",
                    icon="🏛️")
        card.line(f"**{TIERS[tier]['name']}** in "
                  f"{DISTRICTS[district]['name']} now belongs "
                  f"to the group.")
        card.stats([("Paid from treasury", sparks(price), True),
                    ("Treasury now",
                     sparks(surface.economy.treasury_balance(key)),
                     True)])
        card.note("its rent lands in the treasury daily — "
                  "funding drops and prizes")
        return self._card(message, card)

    def _control_sellproperty(self, tail, message):  # type: ignore[no-untyped-def]
        key, err = self._estate_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        parts = str(tail or "").split()
        if not parts:
            return ("usage: /sellproperty <property id> — "
                    "/property lists yours with their ids.")
        pid = parts[0]
        result = surface.estate_sell(key, user_id, pid)
        if not result["ok"]:
            return {
                "not_found": "no property of yours with that "
                             "id — /property lists yours.",
                "listed": "that property is on the market — "
                          "/listproperty cancel <id> first.",
                "auction": "that property is at auction — the "
                           "auction owns it until it settles.",
                "mortgaged": "that property carries a mortgage "
                             "— repay it before selling.",
                "funds": "the treasury can't cover the "
                         "buy-back right now.",
            }.get(result["reason"], "can't sell that.")
        from ...social.chat.typography import Card, sparks

        proceeds = (f"{result['proceeds']} 💎"
                    if result["currency"] == "gems"
                    else sparks(result["proceeds"]))
        card = Card("economy", title="Property sold back",
                    icon="🏘️")
        card.line(f"the system bought it back at the "
                  f"depreciated rate.")
        card.stats([("Market value",
                     f"{result['value']}"
                     + (" 💎" if result["currency"] == "gems"
                        else " sparks"), True),
                    ("You received", proceeds, True)])
        return self._card(message, card,
                          [[("🏘️ Property", "property")]])

    def _control_rent(self, message):  # type: ignore[no-untyped-def]
        key, err = self._estate_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        result = surface.estate_collect_rent(key, user_id)
        if not result["ok"]:
            if result["reason"] == "jailed":
                return ("🔒 no rent collection from jail — /bail "
                        "first, or hire a manager (/property "
                        "manager) and it collects for you.")
            return "the property market is off here."
        import time as _time

        from ...community.group_real_estate import TIERS
        from ...social.chat.typography import Card

        if not result["lines"]:
            summary = surface.estate_summary(key, user_id)
            if not summary["properties"]:
                return ("you don't own property yet — /property "
                        "shows the map, /buyproperty starts you.")
            soonest = min(
                float(p.get("anchor") or 0) + 86400
                for p in summary["properties"])
            wait_h = max(1, int((soonest - _time.time()) // 3600))
            return (f"no rent standing yet — the next full day "
                    f"lands in ~{wait_h}h. Rent pays gems 💎.")
        card = Card("economy", title="Rent collected", icon="💎")
        card.line(f"**+{result['gems']} 💎**")
        lines = []
        for line in result["lines"]:
            spec = TIERS[line["tier"]]
            text = (f"{spec['icon']} **{spec['name']}** — "
                    f"{line['periods']} day(s) · "
                    f"+{line['gems']} 💎")
            if line["garnished"]:
                text += (f" ({line['garnished']} 💎 went to "
                         f"its repair debt)")
            lines.append(text)
        card.block_lines(lines)
        if result["offset"]:
            card.note(f"{result['offset']} 💎 of rent offset "
                      f"repair debt first")
        card.stats([("Gems now", f"{result['balance']} 💎", True)])
        return self._card(
            message, card,
            [[("🏘️ Property", "property"),
              ("🪙 My balance", "balance")]])

    def _control_listproperty(self, tail, message):  # type: ignore[no-untyped-def]
        key, err = self._estate_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        parts = str(tail or "").split()
        if parts and parts[0].lower() == "buy" and len(parts) >= 2:
            result = surface.estate_buy_listing(key, user_id,
                                                parts[1])
            if not result["ok"]:
                return {
                    "not_found": "that listing is gone — "
                                 "/listproperty shows the live "
                                 "market.",
                    "own": "that's your own listing.",
                    "cap": "you hold the group's property cap "
                           "— sell before buying more.",
                    "funds": f"not enough sparks — the ask is "
                             f"{result.get('ask')} and you have "
                             f"{surface.economy.balance(key, user_id)}.",
                }.get(result["reason"], "can't buy that.")
            from ...social.chat.typography import Card, sparks

            card = Card("economy", title="Market purchase",
                        icon="🏪")
            card.line("the property is yours — its rent clock "
                      "kept running through the sale.")
            card.stats([("Ask paid", sparks(result["ask"]), True),
                        ("Seller received",
                         sparks(result["net"]), True),
                        ("Treasury cut", sparks(result["cut"]),
                         True)])
            return self._card(message, card,
                              [[("🏘️ My property", "property")]])
        if parts and parts[0].lower() == "cancel" \
                and len(parts) >= 2:
            result = surface.estate_unlist(key, user_id, parts[1])
            if not result["ok"]:
                return ("no listed property of yours with that "
                        "id.")
            return "🏪 listing withdrawn — the property is back in your hands."
        if len(parts) >= 2 and parts[1].isdigit():
            result = surface.estate_list(key, user_id, parts[0],
                                         int(parts[1]))
            if not result["ok"]:
                return {
                    "not_found": "no property of yours with "
                                 "that id — /property lists "
                                 "yours.",
                    "currency": "gem-priced tiers trade for "
                                "gems at auction only — the "
                                "sparks market can't list them.",
                    "listed": "that property is already on "
                              "the market.",
                    "auction": "that property is at auction.",
                    "mortgaged": "a mortgaged property can't "
                                 "be listed.",
                    "shape": "the ask must be a positive "
                             "number of sparks.",
                }.get(result["reason"], "can't list that.")
            return (f"🏪 listed for {parts[1]} sparks as "
                    f"`{result['listing']}` — /listproperty "
                    f"shows the board; the treasury takes the "
                    f"group's market cut on a sale.")
        # bare: the market board
        from ...community.group_real_estate import DISTRICTS, TIERS
        from ...social.chat.typography import Card, fmt_amount

        listings = surface.estate_market(key)
        card = Card("economy", title="Property market", icon="🏪")
        if not listings:
            card.line("nothing listed right now — list yours: "
                      "/listproperty <property id> <ask>")
        buttons = []
        lines = []
        for listing in listings:
            spec = TIERS[listing["tier"]]
            seller = self._estate_member_label(
                surface, key, listing["seller"])
            mine = listing["seller"] == user_id
            lines.append(
                f"`{listing['id']}` {spec['icon']} "
                f"**{spec['name']}** · "
                f"{DISTRICTS[listing['district']]['name']} — "
                f"**{fmt_amount(listing['ask'])} sparks** "
                f"(value ≈ {fmt_amount(listing['value'])}) · "
                f"{'yours' if mine else seller}")
            if not mine and surface.economy.balance(
                    key, user_id) >= int(listing["ask"]):
                buttons.append(
                    [(f"Buy {spec['name']} · "
                      f"{fmt_amount(listing['ask'])}",
                      f"listproperty buy {listing['id']}")])
        card.block_lines(lines)
        cut = surface.estate_cfg(key)["market_cut_pct"]
        card.note(f"the treasury takes {cut}% of every sale · "
                  f"rent clocks follow the property")
        buttons.append([("🏘️ Property", "property")])
        return self._card(message, card, buttons)

    def _control_auctions(self, tail, message):  # type: ignore[no-untyped-def]
        key, err = self._estate_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        parts = str(tail or "").split()
        if parts and parts[0].lower() == "start":
            if len(parts) < 3 or not parts[2].isdigit():
                return ("usage: /auctions start <property id> "
                        "<start price> [hours]")
            hours = 24.0
            if len(parts) >= 4:
                try:
                    hours = max(0.05, float(parts[3]))
                except ValueError:
                    return "hours must be a number."
            result = surface.estate_auction_start(
                key, user_id, parts[1], int(parts[2]), hours)
            if not result["ok"]:
                return {
                    "not_found": "no property of yours with "
                                 "that id.",
                    "currency": "gem-priced tiers can't be "
                                "auctioned for sparks.",
                    "listed": "that property is on the market "
                              "— unlist it first.",
                    "auction": "that property is already at "
                               "auction.",
                    "mortgaged": "a mortgaged property can't "
                                 "be auctioned.",
                    "shape": "the start price must be "
                             "positive.",
                }.get(result["reason"], "can't start that.")
            return (f"🔨 auction `{result['auction']}` is open "
                    f"— /auctions shows it; bids are held in "
                    f"the treasury until it settles.")
        if parts and parts[0].lower() == "bid":
            if len(parts) < 3 or not parts[2].isdigit():
                return "usage: /auctions bid <auction id> <amount>"
            result = surface.estate_bid(key, user_id, parts[1],
                                        int(parts[2]))
            if not result["ok"]:
                return {
                    "not_found": "that auction is closed or "
                                 "gone — /auctions shows the "
                                 "live ones.",
                    "own": "you can't bid on your own auction.",
                    "leading": "you're already the high bidder "
                               "— your sparks are held; raise "
                               "only by being outbid first.",
                    "low": f"too low — the next bid is at "
                           f"least {result.get('floor')}.",
                    "funds": "not enough sparks in your wallet "
                             "for that bid.",
                }.get(result["reason"], "can't bid that.")
            from ...social.chat.typography import Card, sparks

            card = Card("economy", title="Bid placed", icon="🔨")
            card.line(f"you lead at **{sparks(result['high'])}** "
                      f"— held in the treasury until the "
                      f"auction settles.")
            if result["refunded"]:
                card.line("the outbid member was refunded in "
                          "full.")
            if result["extended"]:
                card.note("anti-snipe: the close moved out — "
                          "a bid inside the last 2 minutes "
                          "extends it")
            return self._card(message, card,
                              [[("🔨 Auctions", "auctions")]])
        # bare: the auction board
        import time as _time

        from ...community.group_real_estate import DISTRICTS, TIERS
        from ...social.chat.typography import Card, fmt_amount

        auctions = surface.estate_auctions(key)
        card = Card("economy", title="Auctions", icon="🔨")
        if not auctions:
            card.line("no auctions live — start one: /auctions "
                      "start <property id> <start price> [hours]")
        buttons = []
        lines = []
        for auction in auctions:
            spec = TIERS[auction["tier"]]
            left_s = max(0, int(float(auction["ends"])
                                - _time.time()))
            left = (f"{left_s // 3600}h"
                    if left_s >= 3600 else f"{left_s // 60}m")
            high = int(auction["high"] or 0)
            leader = ""
            if auction["high_bidder"]:
                leader = (" · you lead"
                          if auction["high_bidder"] == user_id
                          else " · led by " +
                          self._estate_member_label(
                              surface, key,
                              auction["high_bidder"]))
            lines.append(
                f"`{auction['id']}` {spec['icon']} "
                f"**{spec['name']}** · "
                f"{DISTRICTS[auction['district']]['name']} — "
                f"high **{fmt_amount(high or auction['start'])} "
                f"sparks** · {left} left{leader}")
            if auction["seller"] != user_id \
                    and auction["high_bidder"] != user_id:
                nxt = high + max(1, high // 10) if high \
                    else int(auction["start"])
                if surface.economy.balance(key, user_id) >= nxt:
                    buttons.append(
                        [(f"Bid {fmt_amount(nxt)} on "
                          f"{spec['name']}",
                          f"auctions bid {auction['id']} {nxt}")])
        card.block_lines(lines)
        card.note("bids are held by the treasury; outbids "
                  "refund in full; a late bid extends the close")
        buttons.append([("🏘️ Property", "property")])
        return self._card(message, card, buttons)

    def _control_mortgage(self, tail, message):  # type: ignore[no-untyped-def]
        key, err = self._estate_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        parts = str(tail or "").split()
        if parts and parts[0].lower() == "borrow":
            if len(parts) < 3 or not parts[2].isdigit():
                return ("usage: /mortgage borrow <property id> "
                        "<amount>")
            result = surface.estate_borrow(key, user_id,
                                           parts[1],
                                           int(parts[2]))
            if not result["ok"]:
                return {
                    "not_found": "no property of yours with "
                                 "that id.",
                    "currency": "gem-priced tiers can't "
                                "collateralise a sparks loan.",
                    "listed": "a listed property can't be "
                              "mortgaged.",
                    "auction": "a property at auction can't "
                               "be mortgaged.",
                    "mortgaged": "that property already "
                                 "carries a mortgage.",
                    "ltv": f"too much — against a value of "
                           f"{result.get('value')}, the most "
                           f"the treasury lends is "
                           f"{result.get('max')}.",
                    "funds": f"the treasury can't lend that "
                             f"right now — "
                             f"{result.get('available', 0)} "
                             f"available after reservations.",
                    "jailed": "🔒 no borrowing from jail.",
                    "shape": "the amount must be positive.",
                }.get(result["reason"], "can't borrow that.")
            from ...social.chat.typography import Card, sparks

            m = result["mortgage"]
            card = Card("economy", title="Mortgage drawn",
                        icon="🏦")
            card.line(f"**{sparks(m['principal'])}** landed in "
                      f"your wallet, against `{m['property']}`.")
            card.stats([
                ("Interest (flat, stated)",
                 sparks(m["interest"]), True),
                ("Total debt",
                 sparks(m["principal"] + m["interest"]), True),
                ("Instalment",
                 f"{sparks(m['installment'])} / week from "
                 f"your BANK", True),
                ("Periods", str(m["periods_left"])),
            ])
            card.note("miss 3 weekly instalments and the "
                      "property is repossessed — on the record")
            return self._card(message, card,
                              [[("🏦 Mortgage", "mortgage"),
                                ("🪙 My balance", "balance")]])
        if parts and parts[0].lower() == "repay":
            if len(parts) < 3 or not parts[2].isdigit():
                return ("usage: /mortgage repay <mortgage id> "
                        "<amount>")
            result = surface.estate_repay(key, user_id, parts[1],
                                          int(parts[2]))
            if not result["ok"]:
                return {
                    "not_found": "no open mortgage of yours "
                                 "with that id — /mortgage "
                                 "lists yours.",
                    "funds": "not enough sparks in your "
                             "wallet for that payment.",
                    "shape": "the amount must be positive.",
                }.get(result["reason"], "can't repay that.")
            if result["closed"]:
                return ("🏦 mortgage cleared — the property is "
                        "yours outright again.")
            from ...social.chat.typography import sparks

            return (f"🏦 paid {sparks(result['paid'])} — "
                    f"{sparks(result['remaining'])} of debt "
                    f"left.")
        # bare: my mortgages
        from ...social.chat.typography import Card, fmt_amount

        mine = surface.estate_mortgages(key, user_id)
        card = Card("economy", title="Your mortgages", icon="🏦")
        if not mine:
            card.line("no mortgages — borrow against a "
                      "property: /mortgage borrow <property "
                      "id> <amount>")
        buttons = []
        lines = []
        import time as _time

        for m in mine:
            due_in = max(0, int(float(m["next_due"])
                                - _time.time()))
            due = (f"{due_in // 86400}d" if due_in >= 86400
                   else f"{due_in // 3600}h")
            lines.append(
                f"`{m['id']}` on `{m['property']}` — "
                f"**{fmt_amount(m['remaining'])} sparks** of "
                f"debt · instalment "
                f"{fmt_amount(m['installment'])} from your "
                f"bank in {due}"
                + (f" · ⚠️ {m['missed']} missed"
                   if m["missed"] else ""))
            pay = min(int(m["installment"]),
                      int(m["remaining"]))
            if surface.economy.balance(key, user_id) >= pay \
                    and pay > 0:
                buttons.append(
                    [(f"Repay {fmt_amount(pay)}",
                      f"mortgage repay {m['id']} {pay}")])
        card.block_lines(lines)
        cfg = surface.estate_cfg(key)
        card.note(f"the treasury lends up to "
                  f"{cfg['ltv_pct']}% of a property's value at "
                  f"{cfg['mortgage_rate_pct']}% per week, flat; "
                  f"instalments debit your bank weekly")
        buttons.append([("🏘️ Property", "property")])
        return self._card(message, card, buttons)

    def _control_insure(self, tail, message):  # type: ignore[no-untyped-def]
        key, err = self._estate_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        parts = str(tail or "").split()
        if parts and parts[0].lower() == "buy" \
                and len(parts) >= 2:
            result = surface.estate_insure(key, user_id,
                                           parts[1])
            if not result["ok"]:
                return {
                    "not_found": "no property of yours with "
                                 "that id.",
                    "funds": f"the premium is "
                             f"{result.get('premium')} sparks — "
                             f"not enough in your wallet.",
                    "gems": f"the premium is "
                            f"{result.get('premium')} 💎 — not "
                            f"enough gems (premiums price in the "
                            f"property's own currency).",
                }.get(result["reason"], "can't insure that.")
            import time as _time

            until = _time.strftime(
                "%Y-%m-%d",
                _time.gmtime(result["insured_until"]))
            premium = (f"{result['premium']} 💎"
                       if result["currency"] == "gems"
                       else f"{result['premium']} sparks")
            return (f"🛡️ insured for 30 days (until {until}) "
                    f"— premium {premium}. Repair-wave bills "
                    f"on it are covered in full.")
        if parts and parts[0].lower() == "pay" \
                and len(parts) >= 2:
            result = surface.estate_pay_repair(key, user_id,
                                               parts[1])
            if not result["ok"]:
                return {
                    "not_found": "no property of yours with "
                                 "that id.",
                    "no_debt": "that property carries no "
                               "repair debt.",
                    "funds": f"nothing payable — the debt is "
                             f"{result.get('debt')} and your "
                             f"wallet is empty.",
                }.get(result["reason"], "can't pay that.")
            from ...social.chat.typography import sparks

            tail_text = (f" — {sparks(result['remaining'])} "
                         f"still owed"
                         if result["remaining"] else
                         " — debt cleared")
            return (f"🔧 paid {sparks(result['paid'])} of "
                    f"repair debt{tail_text}.")
        # bare: my cover
        import time as _time

        from ...community.group_real_estate import (
            INSURANCE_PREMIUM_PCT, TIERS)
        from ...social.chat.typography import Card, fmt_amount

        summary = surface.estate_summary(key, user_id)
        card = Card("economy", title="Insurance", icon="🛡️")
        if not summary["properties"]:
            card.line("you don't own property yet — insurance "
                      "rides on property. /property shows the "
                      "map.")
            return self._card(message, card,
                              [[("🏘️ Property", "property")]])
        buttons = []
        lines = []
        for prop in summary["properties"]:
            spec = TIERS[prop["tier"]]
            insured_until = float(prop.get("insured_until")
                                  or 0)
            if insured_until > _time.time():
                cover = ("insured until " + _time.strftime(
                    "%Y-%m-%d", _time.gmtime(insured_until)))
            else:
                cover = "uninsured"
                premium = max(1, int(round(
                    prop["value"] * INSURANCE_PREMIUM_PCT / 100)))
                cur = "💎" if prop["currency"] == "gems" else ""
                buttons.append(
                    [(f"Insure {spec['name']} · "
                      f"{fmt_amount(premium)}{cur}",
                      f"insure buy {prop['id']}")])
            debt = int(prop.get("repair_debt") or 0)
            lines.append(
                f"`{prop['id']}` {spec['icon']} "
                f"**{spec['name']}** — {cover}"
                + (f" · 🔧 repair debt {debt}" if debt else ""))
            if debt:
                buttons.append(
                    [(f"Pay repair debt · {debt}",
                      f"insure pay {prop['id']}")])
        card.block_lines(lines)
        card.note("a policy runs 30 days; repair-wave bills "
                  "on insured property are covered in full — "
                  "market falls are not insurable")
        buttons.append([("🏘️ Property", "property")])
        return self._card(message, card, buttons)

    def _execute_group_action(self, surface, key, adapter, message,
                              action) -> bool:  # type: ignore[no-untyped-def]
        """Execute one moderation action through the adapter's
        admin surface; audit the outcome either way.  Returns True
        only when the platform actually performed it."""
        kind = str(action.get("kind") or "")
        user_id = str(action.get("user_id") or "")
        if kind == "kick":
            # A kick is a ban immediately lifted (the /kick
            # grammar): out now, free to rejoin later.
            ban = getattr(adapter, "admin_ban", None) if adapter \
                else None
            unban = getattr(adapter, "admin_unban", None) \
                if adapter else None
            if ban is None or unban is None:
                surface._audit(
                    key, "moderation_unexecuted", action=kind,
                    rule=action.get("rule"), user_id=user_id,
                    reason="adapter has no admin_ban/admin_unban")
                surface.store.save()
                return False
            try:
                out = ban(message.chat, user_id)
                if isinstance(out, dict) and out.get("ok") is False:
                    raise RuntimeError(str(out.get("error")
                                           or "platform refused"))
                unban(message.chat, user_id)
            except Exception as exc:  # noqa: BLE001 - audited
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
        if kind in ("acceptance_restrict", "acceptance_unrestrict"):
            # §28's restriction is INDEFINITE (until the member
            # fixes the miss or an admin releases them) — it has
            # its own executor because a timed mute's grammar
            # would quietly turn it into an hour.
            ok = self._acceptance_restrict(
                adapter, message.chat, user_id,
                on=(kind == "acceptance_restrict"))
            surface._audit(
                key, "moderation_executed" if ok
                else "moderation_unexecuted",
                action=kind, rule=action.get("rule"),
                user_id=user_id,
                reason="" if ok else
                "adapter restrict surface refused or missing")
            surface.store.save()
            return ok
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
                # A timed mute (1h by default posture; defense
                # actions carry their configured minutes); the two
                # Telegram adapters spell it differently (minutes
                # vs until) — speak each one's own grammar.
                minutes = int(action.get("minutes") or 60)
                try:
                    fn(message.chat, user_id, minutes=minutes)
                except TypeError:
                    fn(message.chat, user_id,
                       until=int(time.time()) + minutes * 60)
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


def _parse_duration(text: str) -> tuple[int | None, str]:
    """Split a leading duration token off a moderation tail:
    ``10m`` / ``2h`` / ``1d`` / ``45s`` / ``inf``.  Returns
    (seconds, remaining reason).  No token → (None, full text);
    ``inf``/``forever`` → (0, reason) = indefinite."""
    parts = str(text or "").strip().split(None, 1)
    if not parts:
        return None, ""
    token = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""
    if token in ("inf", "forever", "permanent"):
        return 0, rest
    import re

    match = re.fullmatch(r"(\d+)([smhd])", token)
    if not match:
        return None, str(text or "")
    scale = {"s": 1, "m": 60, "h": 3600, "d": 86400}[match.group(2)]
    return int(match.group(1)) * scale, rest


def _duration_text(seconds: int) -> str:
    if seconds >= 86400:
        return f"{seconds // 86400}d"
    if seconds >= 3600:
        return f"{seconds // 3600}h"
    if seconds >= 60:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def _fmt_ts(ts) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M UTC",
                             time.gmtime(float(ts)))
    except (TypeError, ValueError):
        return "?"


def _member_card(surface, key: str, member: dict) -> str:
    """The member card, through the typography organ (§27,
    moderation class): identity fields are literals — a hostile
    display name renders as text, never markup."""
    from ...social.chat.typography import Card, render_card

    name = str(member.get("name") or member.get("username")
               or "member")
    handle = (f"@{member['username']}" if member.get("username")
              else "no username")
    card = Card("moderation", title=name, icon="👤")
    joined = member.get("joined_ts") or member.get("first_seen")
    rows = [("Handle", handle),
            ("Joined", _fmt_ts(joined)),
            ("Messages", int(member.get("message_count") or 0)),
            ("Last seen", _fmt_ts(member.get("last_seen")))]
    if member.get("left_ts"):
        rows.append(("Left", _fmt_ts(member["left_ts"])))
    card.stats(rows)
    # The estate rides the member card (§20f): what they hold
    # is part of who they are in the group.
    estate = surface.estate_summary(
        key, str(member.get("user_id") or ""))
    if estate["property_count"] or estate["gems"]:
        from ...social.chat.typography import sparks as _sparks

        card.stats([
            ("Properties",
             f"{estate['property_count']} · "
             f"{_sparks(estate['property_value'])}"),
            ("Gems", f"{estate['gems']} 💎"),
        ], title="Estate")
    changes = [c for c in surface.changes(key, limit=200)
               if c.get("user_id") == member.get("user_id")]
    if changes:
        card.board("\n".join(
            f"{change.get('kind', 'changed')}: "
            f"{change.get('old') or '—'} → "
            f"{change.get('new') or '—'} "
            f"({_fmt_ts(change.get('ts'))})"
            for change in changes[-3:]),
            title="Identity changes")
    return render_card(
        card, platform=str(key).split(":", 1)[0],
        settings=surface.store.settings(key)).text


def _audit_line(entry: dict) -> str:
    kind = entry.get("kind", "?")
    bits = [_fmt_ts(entry.get("ts")), kind]
    for field_name in ("action", "rule", "user_id", "change_kind",
                       "reason"):
        if entry.get(field_name):
            bits.append(f"{field_name}={entry[field_name]}")
    if entry.get("count"):
        bits.append(f"count={entry['count']}")
    if entry.get("suppressed"):
        bits.append(f"suppressed={entry['suppressed']}")
    if entry.get("signature"):
        bits.append(f"sig={entry['signature']}")
    if entry.get("excerpt"):
        bits.append(f"“{str(entry['excerpt'])[:60]}”")
    if entry.get("old") or entry.get("new"):
        bits.append(f"{entry.get('old') or '—'} → "
                    f"{entry.get('new') or '—'}")
    return " · ".join(str(b) for b in bits)
