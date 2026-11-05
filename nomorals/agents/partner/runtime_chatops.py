"""RuntimeChatOpsMixin: the chat & privacy command set (FEI
gap wave F) in the live event path.

The last FEI wave: the FEI XMD menu's chat & privacy verbs,
built on Devon's PERSONAL-account surfaces and blended into
the presence/pair organs — chat-state ops (archive, mute,
read/unread, star, disappearing timers), message ops (unsend,
edit, antiedit), privacy (block, anticall), status reactions,
ban checking, group info, poll results, and the tagging trio
(tagall / hidetag / antitag).

The laws this mixin is built under:

- **Owner-invoked only.**  Every kind here is a control
  command and NOT in PUBLIC_COMMANDS, so ``caller_may_run``
  admits the owner alone.  Nothing in this file is called
  from a beat, a schedule, or a proactive brain path — the
  only automatic actors are the personal organ's observe
  paths (status_react / anticall / antitag), each behind its
  own default-OFF toggle in the SAME home as the presence
  pack (``PERSONAL_TOGGLES``, flipped via /pair toggle or the
  dedicated commands here), each honoring the owner's sticky
  stop, and none of them brain output.
- **§33 does not gate these.**  They are the owner's account
  ops, like /pair — invoked by him, acting on his accounts.
  (Bot-SYSTEM output is what §33 gates; an owner typing
  /archive is not bot-system output.)
- **Human targets, never raw keys.**  Chat targets resolve
  through ``partner/chat_targets.resolve_chat_target`` — a
  name, a number, or "this chat" — the register-#35 lesson.
- **Honest absences.**  Where a platform offers no verb
  (Telegram mark-as-unread and message star, WhatsApp poll
  tallies, the Bot API's poll re-read and member enumeration,
  Baileys DM disappearing timers), the reply names the limit
  plainly instead of faking a result.
"""

from __future__ import annotations

import logging
from typing import Any

from .failures import record_failure

_log = logging.getLogger(__name__)

#: WhatsApp's delete-for-everyone window, named in replies
#: when an unsend fails — the platform refuses old messages
#: server-side; the note keeps the refusal self-explanatory.
_WA_UNSEND_WINDOW = ("WhatsApp lets you delete for everyone "
                     "for about 60 hours after sending.")
#: WhatsApp's edit window, same treatment.
_WA_EDIT_WINDOW = ("WhatsApp lets you edit a message for "
                   "about 15 minutes after sending.")


class RuntimeChatOpsMixin:
    """Chat & privacy commands for PartnerRuntime."""

    # ── shared plumbing ──────────────────────────────────────

    def _chatops_adapters(self) -> list[Any]:
        gateway = getattr(self, "gateway", None)
        if gateway is None:
            return []
        try:
            return list(gateway.adapters.values())
        except Exception:  # noqa: BLE001 - listing is a hint
            return []

    def _chatops_target(self, message: Any, raw: str
                        ) -> tuple[Any, Any, str]:
        """Resolve one chat target for a chat-state op.

        Returns (ref, adapter, error): on success ``error`` is
        "" and ref/adapter are set; on failure the error is the
        one plain line to answer with.  An empty ``raw`` means
        the chat the command was sent in.  The adapter is the
        one holding the TARGET chat's platform (chat ops act
        where the chat lives, which may differ from the surface
        the owner typed on)."""
        from ...partner.chat_targets import resolve_chat_target
        from ...social.chat.base import ChatRef
        from ...social.chat.chatops import kind_for_chat_key

        chat = getattr(message, "chat", None)
        if chat is None:
            return None, None, "run it in a chat — the op needs a home."
        raw = str(raw or "").strip()
        if not raw:
            platform = str(getattr(chat, "platform", "") or "")
            adapter = self._adapter_for(platform)
            return chat, adapter, ""
        res = resolve_chat_target(
            raw,
            current_key=str(getattr(chat, "key", "") or ""),
            current_title=str(getattr(chat, "title", "") or ""),
            current_platform=str(getattr(chat, "platform", "")
                                 or ""),
            db=getattr(self.context, "db", None),
            adapters=self._chatops_adapters())
        if not res.resolved:
            return None, None, res.describe(raw)
        platform, _, chat_id = res.key.partition(":")
        ref = ChatRef(platform=platform, chat_id=chat_id,
                      kind=kind_for_chat_key(res.key),
                      title=res.title or "",
                      peer=chat_id)
        return ref, self._adapter_for(platform), ""

    def _chatops_verdict(self, verdict: Any, done: str,
                         platform: str = "") -> str:
        """Render one adapter verdict dict as the reply line."""
        if isinstance(verdict, dict) and verdict.get("ok"):
            return done
        error = ""
        if isinstance(verdict, dict):
            error = str(verdict.get("error") or "")
        return f"couldn't do it — {error or 'the surface refused'}"

    def _chatops_call(self, fn: Any, *args: Any,
                      **kwargs: Any) -> Any:
        """One adapter call, exception-proofed: a raised adapter
        becomes an ok:False verdict carrying the reason, so a
        command always answers with an honest line."""
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            record_failure("runtime_chatops.adapter_call", exc)
            return {"ok": False, "error": str(exc)}

    def _chatops_personal_account(self, platform: str
                                  ) -> tuple[Any, Any]:
        """(service bundle, resolved account) for one personal
        platform — the toggle commands' home.  (None, None)
        when the platform has no paired account."""
        from ...social.pairing import resolve_account

        svc = self._pairing_personal()
        rec = resolve_account(svc["registry"], platform)
        return svc, rec

    def _chatops_history_row(self, adapter: Any, chat: Any,
                             message_id: str) -> Any | None:
        """The adapter-history row for one message id, when the
        surface's history carries ids at all — the own/other
        evidence for unsend/edit.  None = not found (the
        platform verb then decides, and its refusal rules)."""
        history = getattr(adapter, "history", None)
        if not callable(history):
            return None
        try:
            rows = history(chat, 50) or []
        except Exception as exc:  # noqa: BLE001
            record_failure("runtime_chatops.history", exc)
            return None
        for row in rows:
            if str(getattr(row, "message_id", "") or "") \
                    == str(message_id):
                return row
        return None

    def _chatops_reply_target(self, message: Any) -> str:
        """The message id the owner's command replies to."""
        return str(getattr(message, "reply_to", "") or "").strip()

    # ── the one handler ──────────────────────────────────────

    def _control_chatops(self, kind: str, tail: str,
                         message: Any = None) -> str:
        kind = str(kind or "").strip().lower()
        tail = str(tail or "").strip()
        if message is None:
            return (f"/{kind} needs a live chat to act on — "
                    f"run it from the surface it belongs to.")
        try:
            if kind in ("archive", "unarchive", "mutechat",
                        "unmutechat", "readchat", "unreadchat"):
                return self._chatops_chat_state(kind, tail,
                                                message)
            if kind == "disappear":
                return self._chatops_disappear(tail, message)
            if kind in ("star", "unstar"):
                return self._chatops_star(kind, message)
            if kind == "unsend":
                return self._chatops_unsend(message)
            if kind in ("edit", "editmsg"):
                return self._chatops_edit(tail, message)
            if kind == "antiedit":
                return self._chatops_toggle_command(
                    "antiedit", tail, message)
            if kind in ("block", "unblock"):
                return self._chatops_block(kind, tail, message)
            if kind == "anticall":
                return self._chatops_toggle_command(
                    "anticall", tail, message)
            if kind == "statusreact":
                return self._chatops_statusreact(tail, message)
            if kind == "antitag":
                return self._chatops_toggle_command(
                    "antitag", tail, message)
            if kind == "banchecker":
                return self._chatops_banchecker(tail, message)
            if kind in ("gcstatus", "statusinfo"):
                return self._chatops_gcstatus(message)
            if kind == "pollres":
                return self._chatops_pollres(message)
            if kind in ("tagall", "hidetag"):
                return self._chatops_tagall(kind, tail, message)
        except Exception as exc:  # noqa: BLE001 - a command answers
            record_failure("runtime_chatops.control", exc)
            return f"/{kind} hit a wall: {exc}"
        return f"unknown chat op {kind!r}"

    # ── chat-state ops ───────────────────────────────────────

    def _chatops_chat_state(self, kind: str, tail: str,
                            message: Any) -> str:
        ref, adapter, error = self._chatops_target(message, tail)
        if error:
            return error
        platform = str(getattr(ref, "platform", "") or "")
        name = str(getattr(ref, "title", "") or getattr(
            ref, "chat_id", "") or "that chat")
        if adapter is None:
            return (f"no live {platform} surface here to act "
                    f"on {name}.")
        if platform == "whatsapp":
            if kind in ("archive", "unarchive"):
                verdict = self._chatops_call(
                    adapter.chat_archive, ref,
                    kind == "archive")
                done = (f"archived {name}." if kind == "archive"
                        else f"unarchived {name}.")
                return self._chatops_verdict(verdict, done)
            if kind in ("mutechat", "unmutechat"):
                verdict = self._chatops_call(
                    adapter.chat_mute, ref,
                    seconds=-1 if kind == "mutechat" else 0)
                done = (f"muted {name} (no end date)."
                        if kind == "mutechat"
                        else f"unmuted {name}.")
                return self._chatops_verdict(verdict, done)
            verdict = self._chatops_call(
                adapter.chat_mark_read, ref, kind == "readchat")
            done = (f"marked {name} read."
                    if kind == "readchat"
                    else f"marked {name} unread.")
            return self._chatops_verdict(verdict, done)
        if platform.startswith("telegram"):
            if kind in ("archive", "unarchive"):
                fn = getattr(adapter, "archive_chat", None)
                if not callable(fn):
                    return self._chatops_bot_absence(kind)
                verdict = self._chatops_call(
                    fn, ref, kind == "archive")
                done = (f"archived {name}." if kind == "archive"
                        else f"unarchived {name}.")
                return self._chatops_verdict(verdict, done)
            if kind in ("mutechat", "unmutechat"):
                fn = getattr(adapter, "mute_chat", None)
                if not callable(fn):
                    return self._chatops_bot_absence(kind)
                verdict = self._chatops_call(
                    fn, ref, muted=kind == "mutechat")
                done = (f"muted {name} (until you unmute)."
                        if kind == "mutechat"
                        else f"unmuted {name}.")
                return self._chatops_verdict(verdict, done)
            if kind == "unreadchat":
                # MTProto can acknowledge a read; nothing on any
                # Telegram surface un-acknowledges one.
                return ("Telegram has no mark-as-unread — "
                        "MTProto acknowledges reads, it never "
                        "un-acknowledges them.")
            fn = getattr(adapter, "mark_read", None)
            if not callable(fn):
                return self._chatops_bot_absence(kind)
            verdict = self._chatops_call(fn, ref)
            return self._chatops_verdict(
                verdict, f"marked {name} read.")
        return (f"{platform} doesn't offer chat-state ops on "
                f"this surface.")

    def _chatops_bot_absence(self, kind: str) -> str:
        return (f"the bot surface can't /{kind} — that's a "
                f"personal-account verb. Run it from your "
                f"personal surface in that chat.")

    def _chatops_disappear(self, tail: str,
                           message: Any) -> str:
        from ...social.chat.chatops import parse_disappear_duration

        parts = tail.split(None, 1)
        if not parts:
            return ("usage: /disappear <off|24h|7d|90d> "
                    "[chat name] — set a chat's disappearing-"
                    "messages timer.")
        seconds = parse_disappear_duration(parts[0])
        if seconds is None:
            return (f"“{parts[0]}” isn't a duration I speak — "
                    f"use off, 24h, 7d, 90d, or a number of "
                    f"seconds.")
        raw_target = parts[1] if len(parts) > 1 else ""
        ref, adapter, error = self._chatops_target(
            message, raw_target)
        if error:
            return error
        platform = str(getattr(ref, "platform", "") or "")
        name = str(getattr(ref, "title", "")
                   or getattr(ref, "chat_id", "") or "that chat")
        fn = getattr(adapter, "set_disappearing", None)
        if not callable(fn):
            return (f"{platform} doesn't offer disappearing-"
                    f"message timers on this surface.")
        verdict = self._chatops_call(fn, ref, seconds)
        done = (f"disappearing messages off in {name}."
                if seconds == 0 else
                f"disappearing messages on in {name}: "
                f"{parts[0]}.")
        return self._chatops_verdict(verdict, done)

    # ── message ops ──────────────────────────────────────────

    def _chatops_star(self, kind: str, message: Any) -> str:
        mid = self._chatops_reply_target(message)
        if not mid:
            return (f"reply to the message you want to "
                    f"/{kind} — the star lands on the replied "
                    f"message.")
        chat = message.chat
        platform = str(getattr(chat, "platform", "") or "")
        if platform.startswith("telegram"):
            return ("Telegram has no message star — Saved "
                    "Messages is its bookmark, and no API stars "
                    "a message in place.")
        if platform != "whatsapp":
            return (f"{platform} has no message star on this "
                    f"surface.")
        adapter = self._adapter_for(platform)
        fn = getattr(adapter, "star_message", None)
        if not callable(fn):
            return "the WhatsApp surface here has no star verb."
        verdict = self._chatops_call(fn, chat, mid,
                                     kind == "star")
        done = "starred ⭐" if kind == "star" else "unstarred."
        return self._chatops_verdict(verdict, done)

    def _chatops_unsend(self, message: Any) -> str:
        mid = self._chatops_reply_target(message)
        if not mid:
            return ("reply to your message with /unsend — it "
                    "deletes that message for everyone.")
        chat = message.chat
        platform = str(getattr(chat, "platform", "") or "")
        adapter = self._adapter_for(platform)
        if adapter is None:
            return f"no live {platform} surface here."
        row = self._chatops_history_row(adapter, chat, mid)
        if row is not None and bool(getattr(row, "incoming",
                                            False)):
            return ("that's not your message — /unsend "
                    "removes your own messages. (Deleting other "
                    "people's is the group moderation verb.)")
        if platform == "whatsapp":
            fn = getattr(adapter, "admin_delete", None)
            if not callable(fn):
                return ("this WhatsApp surface has no delete "
                        "verb wired.")
            verdict = self._chatops_call(fn, chat, mid)
            if isinstance(verdict, dict) and verdict.get("ok"):
                return "deleted for everyone."
            line = self._chatops_verdict(verdict, "")
            return f"{line} {_WA_UNSEND_WINDOW}"
        if platform.startswith("telegram"):
            fn = getattr(adapter, "revoke_message", None)
            if not callable(fn):
                return self._chatops_bot_absence("unsend")
            verdict = self._chatops_call(fn, chat, mid)
            return self._chatops_verdict(
                verdict, "deleted for everyone.")
        return (f"{platform} doesn't offer delete-for-everyone "
                f"on this surface.")

    def _chatops_edit(self, tail: str, message: Any) -> str:
        mid = self._chatops_reply_target(message)
        if not mid:
            return ("reply to your message with "
                    "/editmsg <new text> — the edit lands on "
                    "the replied message.")
        if not tail:
            return "usage: /editmsg <the new text>"
        chat = message.chat
        platform = str(getattr(chat, "platform", "") or "")
        adapter = self._adapter_for(platform)
        if adapter is None:
            return f"no live {platform} surface here."
        row = self._chatops_history_row(adapter, chat, mid)
        if row is not None and bool(getattr(row, "incoming",
                                            False)):
            return ("that's not your message — editing other "
                    "people's messages is impossible on every "
                    "platform, not just this one.")
        fn = getattr(adapter, "edit_message", None)
        if not callable(fn):
            if platform.startswith("telegram"):
                return self._chatops_bot_absence("editmsg")
            return (f"{platform} has no message-edit verb on "
                    f"this surface.")
        verdict = self._chatops_call(fn, chat, mid, tail)
        if isinstance(verdict, dict) and verdict.get("ok"):
            return "edited."
        line = self._chatops_verdict(verdict, "")
        if platform == "whatsapp":
            line = f"{line} {_WA_EDIT_WINDOW}"
        return line

    # ── the toggle commands (presence-pack home) ─────────────

    def _chatops_toggle_platform(self, name: str,
                                 message: Any) -> str:
        """Which personal platform a toggle command governs:
        anticall lives on WhatsApp (the only surface whose
        bridges see calls); antiedit/antitag follow the surface
        the owner typed on, defaulting to WhatsApp."""
        if name == "anticall":
            return "whatsapp"
        platform = str(getattr(message.chat, "platform", "")
                       or "")
        if platform in ("whatsapp", "telegram"):
            return platform
        return "whatsapp"

    def _chatops_toggle_command(self, name: str, tail: str,
                                message: Any) -> str:
        """Status + on/off for one PERSONAL_TOGGLES member —
        the same store /pair toggle flips (one home, two
        doors)."""
        platform = self._chatops_toggle_platform(name, message)
        svc, rec = self._chatops_personal_account(platform)
        if rec is None:
            return (f"no paired {platform} account — /pair one "
                    f"first; /{name} lives on the personal "
                    f"account.")
        personal = svc["personal"]
        word = tail.strip().lower()
        label = str(rec.label or rec.id)
        if word in ("on", "off"):
            on = word == "on"
            try:
                personal.set_toggle(rec.id, name, on)
            except ValueError as exc:
                return str(exc)
            return (f"/{name} {'ON' if on else 'OFF'} for "
                    f"{label} ({platform})."
                    + self._chatops_toggle_note(name, platform,
                                                on))
        if word:
            return f"usage: /{name} [on|off]"
        state = "ON" if rec.toggle(name) else "OFF"
        line = f"/{name} is {state} for {label} ({platform})."
        if name == "antiedit":
            line += self._chatops_antiedit_recent(rec)
        elif name == "anticall":
            line += (" Incoming WhatsApp calls are rejected "
                     "as you while it's on."
                     if rec.toggle("anticall") else "")
        elif name == "antitag":
            line += (" Mass tags that include you get flagged "
                     "to your DM (at most one flag per chat "
                     "per hour)."
                     if rec.toggle("antitag") else "")
        return line + self._chatops_toggle_note(name, platform,
                                                rec.toggle(name))

    def _chatops_toggle_note(self, name: str, platform: str,
                             on: bool) -> str:
        """The standing-law footnote where it bites: a toggle
        that's on under a standing owner stop is armed but
        halted — say so, don't let the owner think it acts."""
        if not on:
            return ""
        try:
            svc = self._pairing_personal()
            if svc["registry"].owner_stopped(platform):
                return (f" Heads up: {platform} is under your "
                        f"standing stop, so it stays halted "
                        f"until you start it again.")
        except Exception as exc:  # noqa: BLE001 - a note never dies
            record_failure("runtime_chatops.toggle_note", exc)
        return ""

    def _chatops_antiedit_recent(self, rec: Any) -> str:
        """The recent antiedit recoveries count for one
        account, read from the personal vault's journal — the
        same evidence /pair recovered shows."""
        try:
            svc = self._pairing_personal()
            entries = [e for e in svc["personal"].vault.recoveries(
                account_id=rec.id)
                if str(e.get("kind") or "") == "edit"]
            if entries:
                return (f" {len(entries)} edited message"
                        f"{'' if len(entries) == 1 else 's'} "
                        f"recovered so far — /pair recovered "
                        f"shows the originals.")
        except Exception as exc:  # noqa: BLE001
            record_failure("runtime_chatops.antiedit_recent",
                           exc)
        return ""

    def _chatops_statusreact(self, tail: str,
                             message: Any) -> str:
        """/statusreact — the presence-pack toggle that reacts
        to contacts' WhatsApp statuses as the owner, plus its
        emoji.  Default OFF (it ships off); it is not brain
        output — a reaction is the toggle's whole job — and
        the owner's standing stop halts it like every other
        personal-surface act."""
        svc, rec = self._chatops_personal_account("whatsapp")
        if rec is None:
            return ("no paired whatsapp account — /pair one "
                    "first; /statusreact lives on the personal "
                    "WhatsApp account.")
        personal = svc["personal"]
        label = str(rec.label or rec.id)
        parts = tail.split(None, 1)
        head = parts[0].lower() if parts else ""
        if head == "emoji":
            emoji = parts[1].strip() if len(parts) > 1 else ""
            if not emoji:
                return "usage: /statusreact emoji <emoji>"
            try:
                personal.set_status_react_emoji(rec.id, emoji)
            except ValueError as exc:
                return str(exc)
            return (f"status reactions on {label} now use "
                    f"{emoji}.")
        if head in ("on", "off"):
            on = head == "on"
            try:
                personal.set_toggle(rec.id, "status_react", on)
            except ValueError as exc:
                return str(exc)
            return (f"/statusreact {'ON' if on else 'OFF'} "
                    f"for {label} — Devon reacts "
                    f"{rec.status_react_emoji} to contacts' "
                    f"statuses as you."
                    + self._chatops_toggle_note(
                        "status_react", "whatsapp", on))
        if tail:
            return ("usage: /statusreact [on|off] · "
                    "/statusreact emoji <emoji>")
        state = "ON" if rec.toggle("status_react") else "OFF"
        return (f"/statusreact is {state} for {label} "
                f"(emoji {rec.status_react_emoji}). When it's "
                f"on, that reaction lands on each contact's "
                f"status once, as you — it never messages "
                f"anyone."
                + self._chatops_toggle_note(
                    "status_react", "whatsapp",
                    rec.toggle("status_react")))

    # ── privacy: block / unblock ─────────────────────────────

    def _chatops_block(self, kind: str, tail: str,
                       message: Any) -> str:
        action = "block" if kind == "block" else "unblock"
        chat = message.chat
        platform = str(getattr(chat, "platform", "") or "")
        adapter = self._adapter_for(platform)
        if adapter is None:
            return f"no live {platform} surface here."
        target = tail.strip()
        target_note = ""
        if not target:
            mid = self._chatops_reply_target(message)
            kind_here = str(getattr(chat, "kind", "") or "")
            if mid and kind_here == "dm":
                target = str(getattr(chat, "peer", "") or "")
                target_note = " (the person in this chat)"
            elif mid and platform == "telegram":
                sender = getattr(adapter, "message_sender",
                                 None)
                row = (self._chatops_call(sender, chat, mid)
                       if callable(sender) else None)
                if isinstance(row, dict) and row.get("ok"):
                    target = str(row.get("sender_id") or "")
            if not target:
                return (f"usage: /{kind} <number or @username> "
                        f"— or reply to a message in a DM to "
                        f"{kind} that person.")
        if platform == "whatsapp":
            fn = getattr(adapter, "block_user", None)
            if not callable(fn):
                return ("this WhatsApp surface has no block "
                        "verb wired.")
            jid = target
            if "@" not in jid:
                digits = "".join(c for c in jid if c.isdigit())
                if not digits:
                    return (f"“{target}” isn't a number or JID "
                            f"I can block on WhatsApp.")
                jid = f"{digits}@s.whatsapp.net"
            verdict = self._chatops_call(fn, jid, action)
            done = (f"blocked {target}{target_note} on "
                    f"WhatsApp." if action == "block" else
                    f"unblocked {target}{target_note} on "
                    f"WhatsApp.")
            return self._chatops_verdict(verdict, done)
        if platform.startswith("telegram"):
            fn = getattr(adapter, "block_user", None)
            if not callable(fn):
                return self._chatops_bot_absence(kind)
            verdict = self._chatops_call(fn, target, action)
            done = (f"{'blocked' if action == 'block' else 'unblocked'} "
                    f"{target}{target_note} on Telegram.")
            return self._chatops_verdict(verdict, done)
        return (f"{platform} doesn't offer blocking on this "
                f"surface.")

    # ── banchecker ───────────────────────────────────────────

    def _chatops_banchecker(self, tail: str,
                            message: Any) -> str:
        number = tail.strip()
        if number:
            adapter = self._adapter_for("whatsapp")
            fn = getattr(adapter, "check_number", None)
            if not callable(fn):
                return ("no live WhatsApp surface to ask — "
                        "the number check rides the personal "
                        "WhatsApp bridge.")
            digits = "".join(c for c in number if c.isdigit())
            if not digits:
                return (f"“{number}” isn't a phone number I "
                        f"can check.")
            verdict = self._chatops_call(fn, digits)
            if not (isinstance(verdict, dict)
                    and verdict.get("ok")):
                return self._chatops_verdict(verdict, "")
            if verdict.get("exists"):
                head = (f"{number} is on WhatsApp "
                        f"({verdict.get('jid') or digits}).")
            else:
                head = f"{number} is not on WhatsApp."
            return (head + " That's the only signal WhatsApp "
                    "publishes for other numbers — it publishes "
                    "no ban state, so no honest checker can "
                    "call a number banned or clean.")
        # Own accounts: what the pairing registry + bridges
        # honestly hold — status, the recorded last error (a
        # temporary ban lands there when a bridge reports one),
        # and the standing stop.
        svc = self._pairing_personal()
        registry = svc["registry"]
        rows = list(registry.all())
        if not rows:
            return ("no paired accounts to check — /pair one "
                    "first. For a number: /banchecker <number>.")
        lines = ["Your accounts:"]
        for rec in rows:
            bits = [f"{rec.platform} · {rec.label or rec.id}",
                    f"status: {rec.status}"]
            if registry.owner_stopped(rec.platform):
                bits.append("under your standing stop")
            if rec.last_error:
                bits.append(f"last error: {rec.last_error}")
            lines.append(" — ".join(bits))
        lines.append(
            "A WhatsApp temporary ban shows up as a bridge "
            "connection error on the account (above); WhatsApp "
            "publishes no ban verdict beyond that.")
        return "\n".join(lines)

    # ── group info ───────────────────────────────────────────

    def _chatops_gcstatus(self, message: Any) -> str:
        from ...social.chat.chatops import compose_group_status

        chat = message.chat
        kind_here = str(getattr(chat, "kind", "") or "")
        if kind_here not in ("group", "channel"):
            return ("run /gcstatus inside the group — it "
                    "reports on the room it's in.")
        platform = str(getattr(chat, "platform", "") or "")
        adapter = self._adapter_for(platform)
        if adapter is None:
            return f"no live {platform} surface here."
        title = str(getattr(chat, "title", "") or "")
        if platform == "whatsapp":
            info = self._chatops_call(adapter.group_info, chat)
            if not isinstance(info, dict):
                info = {}
            roster = self._chatops_call(
                adapter.group_participants, chat)
            roster = roster if isinstance(roster, list) else []
            count = int(info.get("size") or 0) or (
                len(roster) if roster else None)
            admins = [str(p.get("name") or p.get("jid") or "")
                      for p in roster
                      if str(p.get("role") or "") in (
                          "admin", "superadmin")]
            if not admins and info.get("admins"):
                admins = [str(j) for j in info["admins"]]
            extra: list[tuple[str, str]] = []
            if info.get("desc"):
                extra.append(("About", str(info["desc"])))
            extra.append(("Announce-only",
                          "yes" if info.get("announce") else "no"))
            extra.append(("Join approval",
                          "on" if info.get("join_approval_mode")
                          else "off"))
            if info.get("ephemeral_hours"):
                extra.append(("Disappearing messages",
                              f"{info['ephemeral_hours']}h"))
            return compose_group_status(
                title=title or str(info.get("subject") or ""),
                platform=platform, kind=kind_here,
                member_count=count, admins=admins, extra=extra,
                source="the WhatsApp bridge's group metadata "
                       "and member roster")
        if platform.startswith("telegram"):
            count = None
            counter = getattr(adapter, "admin_member_count",
                              None)
            if callable(counter):
                verdict = self._chatops_call(counter, chat)
                if isinstance(verdict, dict) \
                        and verdict.get("ok"):
                    count = verdict.get("count")
            extra = []
            source = "the member count this surface exposes"
            if platform == "telegram" and kind_here == "channel":
                stats_fn = getattr(
                    adapter, "channel_broadcast_stats", None)
                if callable(stats_fn):
                    stats = self._chatops_call(stats_fn, chat)
                    if isinstance(stats, dict) \
                            and stats.get("ok"):
                        if stats.get("followers") is not None:
                            count = stats["followers"]
                        if stats.get("views_per_post") is not None:
                            extra.append(
                                ("Avg views/post",
                                 str(stats["views_per_post"])))
                        if stats.get("reactions_pp") is not None:
                            extra.append(
                                ("Avg reactions/post",
                                 str(stats["reactions_pp"])))
                        source = ("Telegram's own broadcast "
                                  "statistics")
            return compose_group_status(
                title=title, platform=platform, kind=kind_here,
                member_count=(int(count)
                              if count is not None else None),
                admins=[], extra=extra, source=source)
        return (f"{platform} doesn't expose group data on "
                f"this surface.")

    # ── poll results ─────────────────────────────────────────

    def _chatops_pollres(self, message: Any) -> str:
        from ...social.chat.chatops import format_poll_results

        mid = self._chatops_reply_target(message)
        if not mid:
            return ("reply to the poll with /pollres — the "
                    "results live on that message.")
        chat = message.chat
        platform = str(getattr(chat, "platform", "") or "")
        if platform == "whatsapp":
            return ("WhatsApp doesn't expose a poll's tallies "
                    "to the bridge — no surface here can "
                    "re-read them.")
        if platform != "telegram":
            return ("the bot surface can't re-read a poll's "
                    "results — the Bot API has no such "
                    "endpoint. Your personal surface can: "
                    "reply to the poll there with /pollres.")
        adapter = self._adapter_for(platform)
        fn = getattr(adapter, "channel_poll_results", None)
        if not callable(fn):
            return ("this Telegram surface can't re-read poll "
                    "results.")
        verdict = self._chatops_call(fn, chat, mid)
        if not (isinstance(verdict, dict)
                and verdict.get("ok")):
            return self._chatops_verdict(verdict, "")
        return format_poll_results(verdict)

    # ── tagging ──────────────────────────────────────────────

    def _chatops_tagall(self, kind: str, tail: str,
                        message: Any) -> str:
        from ...social.chat.chatops import chunk_mentions

        hidden = kind == "hidetag"
        chat = message.chat
        kind_here = str(getattr(chat, "kind", "") or "")
        if kind_here != "group":
            return (f"/{kind} works in a group — it mentions "
                    f"the room's members.")
        platform = str(getattr(chat, "platform", "") or "")
        adapter = self._adapter_for(platform)
        if adapter is None:
            return f"no live {platform} surface here."
        members: list[tuple[str, str]] = []
        if platform == "whatsapp":
            roster = self._chatops_call(
                adapter.group_participants, chat)
            own = str(getattr(adapter, "user_jid", "") or "")
            for p in roster if isinstance(roster, list) else []:
                jid = str(p.get("jid") or "")
                if jid and jid != own:
                    members.append(
                        (jid, str(p.get("name") or "")))
        elif platform == "telegram":
            listing = self._chatops_call(
                adapter.admin_members_full, chat, 20000)
            if not (isinstance(listing, dict)
                    and listing.get("ok")):
                return self._chatops_verdict(listing, "")
            for row in listing.get("members") or []:
                uid = str(row.get("id") or "")
                if uid:
                    members.append(
                        (uid, str(row.get("username")
                                  or row.get("name") or "")))
        else:
            return (f"the {platform} surface can't list a "
                    f"group's members — /{kind} rides your "
                    f"personal surface, which can enumerate "
                    f"the room.")
        if not members:
            return ("no members came back from this surface "
                    "to tag.")
        chunks, tagged, capped = chunk_mentions(members)
        sent = 0
        for chunk in chunks:
            if platform == "whatsapp":
                if hidden:
                    text = tail or "📣"
                else:
                    names = "\n".join(
                        f"@{jid.split('@')[0]}"
                        for jid, _ in chunk)
                    text = names + (f"\n{tail}" if tail else "")
                result = self._chatops_call(
                    adapter.send, chat, text,
                    mentions=[jid for jid, _ in chunk])
                if not getattr(result, "ok", False):
                    return (f"tagged {sent} member"
                            f"{'' if sent == 1 else 's'}, then "
                            f"the send refused: "
                            f"{getattr(result, 'error', '') or 'unknown'}")
                sent += len(chunk)
            else:
                verdict = self._chatops_call(
                    adapter.send_tagged, chat, tail, chunk,
                    hidden=hidden)
                if not (isinstance(verdict, dict)
                        and verdict.get("ok")):
                    return (f"tagged {sent} member"
                            f"{'' if sent == 1 else 's'}, then "
                            + self._chatops_verdict(verdict, ""))
                sent += int(verdict.get("mentioned")
                            or len(chunk))
        line = (f"tagged {sent} member"
                f"{'' if sent == 1 else 's'} "
                f"({'hidden mentions' if hidden else 'by name'}).")
        if capped:
            from ...social.chat.chatops import TAG_CAP

            line += (f" The group is bigger than the "
                     f"{TAG_CAP}-member cap — the first "
                     f"{TAG_CAP} got tagged; the cap keeps a "
                     f"tag from becoming a flood.")
        return line
