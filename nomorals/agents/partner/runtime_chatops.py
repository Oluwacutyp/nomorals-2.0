"""RuntimeChatOpsMixin: the chat & privacy command set (FEI
gap wave F) in the live event path.

The last FEI wave: the FEI XMD menu's chat & privacy verbs,
built on Devon's PERSONAL-account surfaces and blended into
the presence/pair organs — chat-state ops (archive, mute,
read/unread, star, disappearing timers), message ops (unsend,
edit, antiedit, forward), privacy (block, anticall),
status reactions, ban checking, group info, poll results,
and the tagging trio (tagall / hidetag / antitag).

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
  (Telegram mark-as-unread and message star, the Bot API's
  poll re-read and member enumeration, Baileys DM disappearing
  timers), the reply names the limit plainly instead of
  faking a result.  WhatsApp poll tallies left this list in
  the parity-G3 build: the bridges now tally the votes they
  decrypt on polls they have seen (bridge memory — a poll
  outside it still answers in words, never invented numbers).
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
            if kind == "forward":
                return self._chatops_forward(tail, message)
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
            if kind in ("viewonce", "vv"):
                return self._chatops_viewonce(kind, tail, message)
            if kind == "poststatus":
                return self._chatops_poststatus(tail, message)
            if kind == "chanstory":
                return self._chatops_chanstory(tail, message)
            if kind == "joinlink":
                return self._chatops_joinlink(tail, message)
            if kind in ("newgroup", "newchannel"):
                return self._chatops_newchat(kind, tail, message)
            if kind == "leave":
                return self._chatops_leave(tail, message)
            if kind == "antitag":
                return self._chatops_toggle_command(
                    "antitag", tail, message)
            if kind == "banchecker":
                return self._chatops_banchecker(tail, message)
            if kind in ("gcstatus", "statusinfo"):
                return self._chatops_gcstatus(message)
            if kind == "pollres":
                return self._chatops_pollres(message)
            if kind == "myprofile":
                return self._chatops_myprofile(tail, message)
            if kind == "privacy":
                return self._chatops_privacy(tail, message)
            if kind == "makepoll":
                return self._chatops_makepoll(tail, message)
            if kind == "vcf":
                return self._chatops_vcf(tail, message)
            if kind == "addcontact":
                return self._chatops_addcontact(tail, message)
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
            return ("usage: /disappear off, 24h, 7d or 90d "
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

    def _chatops_gated_account(self, platform: str,
                               kind: str) -> tuple[Any, Any, str]:
        """The personal-surface gate /poststatus and /chanstory
        stand behind, shared by the message acts (parity G1):
        a paired account must exist and no standing owner stop
        may cover the platform.  Returns (svc, rec, error) —
        error is the one plain line to answer with."""
        svc, rec = self._chatops_personal_account(platform)
        if rec is None:
            return None, None, (
                f"no paired {platform} account — /pair one "
                f"first; /{kind} acts from the personal "
                f"account.")
        if svc["registry"].owner_stopped(platform):
            return None, None, (
                f"{platform} is under your standing stop — "
                f"it stays halted until you start it again; "
                f"nothing was done.")
        return svc, rec, ""

    def _chatops_forward(self, tail: str, message: Any) -> str:
        """/forward — forward the replied message into another
        chat AS the owner (parity G1: no forward verb existed
        anywhere on WhatsApp; Telegram's surfaces already
        carried forward_post).  The destination is named the
        way humans name chats (chat_targets grammar — a name,
        a number, or a full key) and must live on the SAME
        platform: a WhatsApp forward is a WhatsApp act, it is
        not a cross-platform repost.  Both engines re-send the
        ORIGINAL out of bridge memory with the forwarded mark
        — never a retyped copy — and one command is exactly
        one forward: the verb never retries, so nothing
        double-sends by accident.  Owner-only; a standing
        owner stop halts it; a landed forward is journaled."""
        mid = self._chatops_reply_target(message)
        if not mid:
            return ("reply to the message with /forward "
                    "<chat name or number> — it lands there "
                    "as a forward, from you.")
        if not tail:
            return ("usage: /forward <chat name or number> "
                    "(as a reply to the message) — where should "
                    "it land?")
        chat = message.chat
        platform = str(getattr(chat, "platform", "") or "")
        if platform not in ("whatsapp", "telegram"):
            return (f"{platform} has no message-forward verb "
                    f"on this surface.")
        dest, _dest_adapter, error = self._chatops_target(
            message, tail)
        if error:
            return error
        dest_platform = str(getattr(dest, "platform", "") or "")
        if dest_platform != platform:
            return (f"that destination is on {dest_platform} — "
                    f"a forward stays on its own platform. "
                    f"(Cross-posting it there is a repost, not "
                    f"a forward.)")
        svc, rec, error = self._chatops_gated_account(
            platform, "forward")
        if error:
            return error
        adapter = self._adapter_for(platform)
        if adapter is None:
            return f"no live {platform} surface here."
        personal = svc["personal"]
        dest_name = str(getattr(dest, "title", "")
                        or getattr(dest, "chat_id", "") or "")
        chat_id = str(getattr(chat, "chat_id", "") or "")
        dest_id = str(getattr(dest, "chat_id", "") or "")
        if platform == "whatsapp":
            fn = getattr(adapter, "forward_message", None)
            if not callable(fn):
                return ("this WhatsApp surface has no forward "
                        "verb wired.")
            verdict = self._chatops_call(fn, chat, mid, dest)
            if isinstance(verdict, dict) and verdict.get("ok"):
                personal.note_message_acted(
                    platform, kind="message_forwarded",
                    account_id=rec.id, chat_id=chat_id,
                    source_id=mid, dest_id=dest_id,
                    note="forwarded as the owner via /forward")
                return f"forwarded to {dest_name}."
            return self._chatops_verdict(verdict, "")
        fn = getattr(adapter, "forward_post", None)
        if not callable(fn):
            return self._chatops_bot_absence("forward")
        landed = self._chatops_call(fn, chat, mid, dest)
        if landed is True or bool(getattr(landed, "ok", False)):
            personal.note_message_acted(
                platform, kind="message_forwarded",
                account_id=rec.id, chat_id=chat_id,
                source_id=mid, dest_id=dest_id,
                note="forwarded as the owner via /forward")
            return f"forwarded to {dest_name}."
        error = str(getattr(landed, "error", "") or "")
        return f"couldn't do it — {error or 'the surface refused'}"

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
                    "/editmsg the new text — the edit lands on "
                    "the replied message.")
        if not tail:
            return "usage: /editmsg the new text"
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

        def _pool() -> list[str]:
            from ...social.pairing import status_react_pool_for

            custom: list[str] = []
            provider = getattr(personal, "status_react_pool", None)
            if provider is not None:
                try:
                    custom = list(provider(rec) or [])
                except Exception:  # noqa: BLE001 - display garnish
                    custom = []
            return status_react_pool_for(rec, custom)

        def _mode() -> str:
            if rec.status_react_rotate:
                return (f"rotating through {len(_pool())} emojis "
                        f"(her custom set + yours + the built-in "
                        f"pool)")
            return f"pinned to {rec.status_react_emoji}"

        if head == "emoji":
            emoji = parts[1].strip() if len(parts) > 1 else ""
            if not emoji:
                return "usage: /statusreact emoji 🔥 — pins that one face"
            try:
                personal.set_status_react_emoji(rec.id, emoji)
                personal.set_status_react_rotation(
                    rec.id, rotate=False)
            except ValueError as exc:
                return str(exc)
            return (f"status reactions on {label} now use "
                    f"{emoji} — pinned (rotation off). "
                    f"/statusreact rotate on brings the rotation "
                    f"back.")
        if head == "rotate":
            words = parts[1].split() if len(parts) > 1 else []
            if not words or words[0].lower() not in ("on", "off"):
                return "usage: /statusreact rotate on|off"
            on = words[0].lower() == "on"
            try:
                personal.set_status_react_rotation(rec.id,
                                                   rotate=on)
            except ValueError as exc:
                return str(exc)
            rec = svc["registry"].get(rec.id) or rec
            return (f"status reaction rotation {'ON' if on else 'OFF'} "
                    f"for {label} — {_mode()}.")
        if head == "pool":
            words = parts[1].split(None, 1) if len(parts) > 1 \
                else []
            sub = words[0].lower() if words else ""
            if sub in ("add", "remove") and len(words) > 1:
                emoji = words[1].strip()
                try:
                    if sub == "add":
                        personal.set_status_react_rotation(
                            rec.id, add=[emoji])
                    else:
                        personal.set_status_react_rotation(
                            rec.id, remove=[emoji])
                except ValueError as exc:
                    return str(exc)
                rec = svc["registry"].get(rec.id) or rec
                return (f"{emoji} {'added to' if sub == 'add' else 'removed from'} "
                        f"the rotation pool for {label} "
                        f"({len(_pool())} emojis now).")
            if sub:
                return ("usage: /statusreact pool · "
                        "/statusreact pool add 🔥 · "
                        "/statusreact pool remove 🔥")
            return (f"the rotation pool for {label} "
                    f"({len(_pool())}): {' '.join(_pool())} — "
                    f"her custom set leads; /statusreact pool "
                    f"add 🔥 adds yours.")
        if head in ("on", "off"):
            on = head == "on"
            try:
                personal.set_toggle(rec.id, "status_react", on)
            except ValueError as exc:
                return str(exc)
            return (f"/statusreact {'ON' if on else 'OFF'} "
                    f"for {label} — Devon reacts to contacts' "
                    f"statuses as you, {_mode()}."
                    + self._chatops_toggle_note(
                        "status_react", "whatsapp", on))
        if tail:
            return ("usage: /statusreact [on|off] · "
                    "/statusreact emoji 🔥 (pins one) · "
                    "/statusreact rotate on|off · "
                    "/statusreact pool [add|remove 🔥]")
        state = "ON" if rec.toggle("status_react") else "OFF"
        last = (f" Last face used: {rec.status_react_last}."
                if rec.status_react_last else "")
        return (f"/statusreact is {state} for {label} "
                f"({_mode()}). When it's on, one reaction lands "
                f"on each contact's status once, as you — it "
                f"never messages anyone.{last}"
                + self._chatops_toggle_note(
                    "status_react", "whatsapp",
                    rec.toggle("status_react")))

    def _chatops_viewonce(self, kind: str, tail: str,
                          message: Any) -> str:
        """/viewonce · /vv — REVEAL a captured view-once (truth
        pass, 2026-11-07: the capture half shipped with the
        presence pack — the personal organ keeps view-once media
        in the vault before it vanishes — but no command ever
        showed it back; the owner typed /viewonce and /vv and
        got conversation filler, because neither kind existed.
        This is the reveal half: the newest capture (or the one
        named, or the one replied to) is sent back into THIS
        chat as a normal media message the owner can open.

        Forms: bare → the newest capture + the recent list;
        ``n`` → the nth newest; ``list`` → the list only;
        as a reply → the capture whose source message is the
        replied-to one.  Owner-only by the control gate like
        every chatop; reading his own vault back to his own
        chat is not an act on the account, so no surface stop
        applies — but an empty vault answers honestly, naming
        whether capture is even on."""
        from datetime import datetime
        from pathlib import Path

        svc = self._pairing_personal()
        vault = svc["personal"].vault
        entries = [r for r in vault.saved_entries()
                   if r.get("kind") == "view_once"]
        entries.reverse()          # journal is oldest-first

        # The account a rowless attempt (empty vault, reply
        # miss) is recorded against: the vault's own first
        # capture's account, else the first personal record —
        # so the attempt lands in that account's journal view
        # instead of an accountless limbo no /pair journal
        # filter would ever show.
        fallback_account = ""
        if entries:
            fallback_account = str(entries[0].get("account")
                                   or "")
        else:
            for rec in svc["registry"].all():
                if rec.platform in ("whatsapp", "telegram"):
                    fallback_account = rec.id
                    break

        def _note_reveal(outcome: str, form: str,
                         row: dict | None = None) -> None:
            # Every reveal attempt journals (2026-11-08 truth
            # fix): "the owner asked and the vault missed" must
            # be distinguishable in the record from "the owner
            # never reached the handler" — the distinction the
            # /viewonce silence investigation had to prove by
            # reading code because the journal said nothing.
            try:
                svc["personal"].note_viewonce_reveal(
                    outcome=outcome, form=form,
                    account_id=str((row or {}).get("account")
                                   or fallback_account),
                    platform=str((row or {}).get("platform")
                                 or ""),
                    source_id=str((row or {}).get("source_id")
                                  or ""),
                    poster_name=str((row or {}).get("poster_name")
                                    or ""))
            except Exception as exc:  # noqa: BLE001 - journal only
                record_failure(
                    "runtime_chatops.viewonce_reveal_note", exc)

        def _when(row: dict) -> str:
            try:
                return datetime.fromtimestamp(
                    float(row.get("ts") or 0)).strftime("%d %b %H:%M")
            except (TypeError, ValueError, OSError):
                return "earlier"

        def _line(i: int, row: dict) -> str:
            who = str(row.get("poster_name") or "someone")
            mk = str(row.get("media_kind") or "media")
            return f"  {i}. {_when(row)} · {who} · {mk}"

        if not entries:
            _note_reveal(
                "empty",
                "reply" if str(getattr(message, "reply_to", "")
                               or "") else
                "list" if tail.lower() == "list" else
                "index" if tail else "bare")
            states = []
            for rec in svc["registry"].all():
                if rec.platform in ("whatsapp", "telegram"):
                    states.append(
                        f"{rec.id}: capture "
                        f"{'ON' if rec.toggle('view_once') else 'OFF'}")
            state = ("; ".join(states) if states else
                     "no paired personal account")
            return ("no view-once captures yet — when the "
                    "view_once toggle is on, view-once media "
                    "sent to your personal accounts is kept in "
                    "your vault before it vanishes, and /viewonce "
                    f"shows it back. Right now: {state}. "
                    "Turn it on with /pair toggle ACCOUNT "
                    "view_once on.")
        listing = "\n".join(
            _line(i + 1, r) for i, r in enumerate(entries[:10]))
        more = (f"\n  …and {len(entries) - 10} older (in "
                f"/pair saved)" if len(entries) > 10 else "")

        target = None
        note = ""
        reply_to = str(getattr(message, "reply_to", "") or "")
        if reply_to:
            for row in entries:
                if str(row.get("source_id") or "") == reply_to:
                    target = row
                    break
            if target is None:
                _note_reveal("miss", "reply")
                return ("that message has no view-once capture "
                        "in the vault — capture was off when it "
                        "arrived, or it never reached the "
                        f"surface. Captures I do hold:\n{listing}"
                        f"{more}")
        elif tail.lower() == "list":
            return (f"👁️ view-once captures (newest first) — "
                    f"/viewonce N opens one:\n{listing}{more}")
        elif tail:
            try:
                idx = int(tail.split()[0])
            except (ValueError, IndexError):
                return ("usage: /viewonce · /viewonce N · "
                        "/viewonce list · or reply to the "
                        "view-once message with /viewonce")
            if not 1 <= idx <= len(entries):
                _note_reveal("miss", "index")
                return (f"there are {len(entries)} capture"
                        f"{'s' if len(entries) != 1 else ''} — "
                        f"pick 1–{len(entries)}:\n{listing}{more}")
            target = entries[idx - 1]
        else:
            target = entries[0]
            if len(entries) > 1:
                note = (f"\nthe rest:\n{listing}{more}\n"
                        f"/viewonce N opens another.")
        path = str(target.get("path") or "")
        who = str(target.get("poster_name") or "someone")
        form = ("reply" if reply_to else
                "index" if tail else "bare")
        if not path or not Path(path).exists():
            _note_reveal("file_gone", form, target)
            return (f"the capture from {who} ({_when(target)}) "
                    f"is in the record but its file is gone from "
                    f"the vault — nothing left to show. "
                    f"/pair saved holds the record.")
        gateway = getattr(self, "gateway", None)
        send_file = getattr(gateway, "send_file", None)
        if gateway is None or send_file is None:
            _note_reveal("send_failed", form, target)
            return (f"the capture from {who} is safe in the "
                    f"vault, but this surface can't carry media "
                    f"back — ask again from a chat that can.")
        result = send_file(
            str(getattr(message.chat, "platform", "") or ""),
            message.chat, path,
            caption=(f"👁️ view-once from {who} · "
                     f"{_when(target)} — kept in your vault "
                     f"before it could vanish"))
        if not getattr(result, "ok", False):
            _note_reveal("send_failed", form, target)
            return (f"the capture from {who} is in the vault "
                    f"but sending it back failed: "
                    f"{getattr(result, 'error', '') or 'send refused'}")
        _note_reveal("found", form, target)
        return (f"👁️ that's the view-once from {who} "
                f"({_when(target)}), straight from your vault."
                f"{note}")

    def _chatops_poststatus(self, tail: str,
                            message: Any) -> str:
        """/poststatus — post a status AS the owner on his
        personal surface (the status family's send half: the
        presence pack shipped viewing/saving/reacting, posting
        was never built).  WhatsApp takes text or media; the
        Telegram userbot takes media (stories are media posts
        — a text-only ask gets that platform truth, not a fake).
        Media comes from the command message itself (a photo or
        video sent with /poststatus as its caption) or from the
        message it replies to (the bridges download the quote's
        bytes for exactly this).  Owner-only by the control
        gate; a standing owner stop halts it like every other
        personal-surface act; a landed post is journaled in the
        personal vault, never a silent act as him."""
        platform = str(getattr(message.chat, "platform", "")
                       or "")
        if platform not in ("whatsapp", "telegram"):
            return ("/poststatus lives on your personal "
                    "WhatsApp or Telegram — run it from there.")
        svc, rec = self._chatops_personal_account(platform)
        if rec is None:
            return (f"no paired {platform} account — /pair one "
                    f"first; /poststatus posts from the personal "
                    f"account.")
        if svc["registry"].owner_stopped(platform):
            return (f"{platform} is under your standing stop — "
                    f"it stays halted until you start it again; "
                    f"nothing was posted.")
        adapter = self._adapter_for(platform)
        if adapter is None:
            return f"no live {platform} surface here to post from."
        media_path = ""
        media_kind = ""
        for m in (getattr(message, "media", None) or []):
            p = str(getattr(m, "path", "") or "")
            if p:
                media_path = p
                media_kind = str(getattr(m, "kind", "") or "")
                break
        if not media_path:
            meta = getattr(message, "meta", None) or {}
            quoted = meta.get("replied_media_local") or {}
            if isinstance(quoted, dict) and quoted.get("path"):
                media_path = str(quoted["path"])
                media_kind = str(quoted.get("kind") or "")
        text = tail.strip()
        personal = svc["personal"]
        if platform == "whatsapp":
            fn = getattr(adapter, "send_status", None)
            if not callable(fn):
                return ("this WhatsApp surface has no "
                        "status-post verb wired.")
            if not text and not media_path:
                return ("usage: /poststatus text — or send "
                        "a photo/video with /poststatus as the "
                        "caption, or reply to one with it.")
            verdict = self._chatops_call(
                fn, text, media_path=media_path,
                caption=text if media_path else "")
            if isinstance(verdict, dict) and verdict.get("ok"):
                personal.note_status_posted(
                    platform, account_id=rec.id, text=text,
                    media_kind=media_kind,
                    source_id=str(verdict.get("message_id")
                                  or ""))
                return ("posted to your WhatsApp status"
                        + (f" — “{text}”" if text and not media_path
                           else "."))
            return self._chatops_verdict(verdict, "")
        fn = getattr(adapter, "send_story", None)
        if not callable(fn):
            return ("this Telegram surface has no story-post "
                    "verb wired.")
        if not media_path:
            return ("Telegram stories are media posts — send "
                    "a photo or video with /poststatus as the "
                    "caption. (Text statuses are a WhatsApp "
                    "shape; nothing was posted.)")
        landed = self._chatops_call(fn, media_path, text)
        if landed is True:
            personal.note_status_posted(
                platform, account_id=rec.id, text=text,
                media_kind=media_kind)
            return "posted to your Telegram story."
        if isinstance(landed, dict):
            return self._chatops_verdict(landed, "")
        return ("couldn't do it — the story post did not land "
                "(not connected, or Telegram refused it).")

    def _chatops_chanstory(self, tail: str,
                           message: Any) -> str:
        """/chanstory — post to a CHANNEL as the owner, on his
        personal surface (channel stories build, 2026-11-07).
        On WhatsApp this posts a channel POST — it lands in the
        channel feed and persists like one.  WhatsApp's Channel
        Status (the Status-tray stories that expire in 24h) is
        a DIFFERENT feature, and no bridge library exposes
        posting one (status mining, 2026-11-07): a named wall,
        so the WhatsApp half of this command says "channel
        post" and never "story".  On Telegram it IS a real
        story — stories.SendStory with the channel as peer.
        The channel is named first: a WhatsApp channel link or
        JID or invite code or the channel's name, a Telegram
        @name or channel id.  Media comes from the command
        message itself or the message it replies to, exactly
        like /poststatus.  Owner-only by the control gate; a
        standing owner stop halts it like every other
        personal-surface act; a landed post is journaled in the
        personal vault, never a silent act as him."""
        platform = str(getattr(message.chat, "platform", "")
                       or "")
        if platform not in ("whatsapp", "telegram"):
            return ("/chanstory lives on your personal "
                    "WhatsApp or Telegram — run it from there.")
        svc, rec = self._chatops_personal_account(platform)
        if rec is None:
            return (f"no paired {platform} account — /pair one "
                    f"first; /chanstory posts from the personal "
                    f"account.")
        if svc["registry"].owner_stopped(platform):
            return (f"{platform} is under your standing stop — "
                    f"it stays halted until you start it again; "
                    f"nothing was posted.")
        adapter = self._adapter_for(platform)
        if adapter is None:
            return f"no live {platform} surface here to post from."
        usage = ("usage: /chanstory channel text — on WhatsApp "
                 "the channel as its link, JID, invite code or "
                 "name; on Telegram its @name or id. Or send a "
                 "photo/video with /chanstory channel caption "
                 "as its caption, or reply to one with it.")
        if not tail:
            return usage
        parts = tail.split(None, 1)
        target = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else ""
        media_path = ""
        media_kind = ""
        for m in (getattr(message, "media", None) or []):
            p = str(getattr(m, "path", "") or "")
            if p:
                media_path = p
                media_kind = str(getattr(m, "kind", "") or "")
                break
        if not media_path:
            meta = getattr(message, "meta", None) or {}
            quoted = meta.get("replied_media_local") or {}
            if isinstance(quoted, dict) and quoted.get("path"):
                media_path = str(quoted["path"])
                media_kind = str(quoted.get("kind") or "")
        personal = svc["personal"]
        if platform == "whatsapp":
            jid, name, text, error = self._chanstory_wa_channel(
                adapter, tail)
            if error:
                return error
            fn = getattr(adapter, "channel_send", None)
            if not callable(fn):
                return ("this WhatsApp surface has no "
                        "channel-post verb wired.")
            if not text and not media_path:
                return usage
            verdict = self._chatops_call(
                fn, jid, text, media_path=media_path,
                caption=text if media_path else "")
            if isinstance(verdict, dict) and verdict.get("ok"):
                personal.note_channel_posted(
                    platform, account_id=rec.id, channel_id=jid,
                    channel_name=name, text=text,
                    media_kind=media_kind,
                    source_id=str(verdict.get("server_id") or ""))
                where = f"“{name}”" if name else "your channel"
                return (f"posted to {where} on WhatsApp — a "
                        f"channel post in the channel feed. "
                        f"(WhatsApp's Channel Status tray stories "
                        f"can't be posted by any bridge library "
                        f"yet.)")
            return self._chatops_verdict(verdict, "")
        # Telegram: a channel story is a media post, and the
        # channel must be one this account admins — Telegram
        # itself refuses otherwise (boost level included), and
        # its refusal comes back in the verdict's own words.
        fn = getattr(adapter, "send_channel_story", None)
        if not callable(fn):
            return ("this Telegram surface has no channel-story "
                    "verb wired.")
        if not media_path:
            return ("Telegram channel stories are media posts "
                    "— send a photo or video with /chanstory "
                    "channel caption as its caption. (Text "
                    "channel posts are a WhatsApp shape; nothing "
                    "was posted.)")
        ref = target
        if "t.me/" in ref:
            ref = "@" + ref.split("t.me/")[-1].strip("/")
        verdict = self._chatops_call(fn, ref, media_path, text)
        if isinstance(verdict, dict) and verdict.get("ok"):
            personal.note_channel_posted(
                platform, account_id=rec.id, channel_id=ref,
                text=text, media_kind=media_kind)
            return "posted to the channel's story on Telegram."
        return self._chatops_verdict(verdict, "")

    def _chanstory_wa_channel(self, adapter: Any, tail: str
                              ) -> tuple[str, str, str, str]:
        """Resolve the WhatsApp channel a /chanstory tail names:
        (jid, name, rest_text, error) — exactly one of jid /
        error is set, and rest_text is what follows the channel
        reference (the post's text).  Accepted forms, in order:
        the channel JID itself; a whatsapp.com/channel link
        (its invite code resolves through the bridge); the
        channel's NAME among the channels this surface already
        knows — the longest word-prefix of the tail that
        uniquely matches a known channel title, so multi-word
        names work (a guess between two channels is a mispost
        waiting to happen, so ambiguity is an error, never a
        pick); a bare invite code."""
        tail = tail.strip()
        tokens = tail.split()
        if not tokens:
            return "", "", "", "usage: /chanstory channel text"
        first = tokens[0]
        if first.endswith("@newsletter"):
            return first, "", tail[len(first):].strip(), ""
        if "whatsapp.com/channel/" in first:
            code = first.split("whatsapp.com/channel/")[-1]
            code = code.strip("/").split("/")[0].strip()
            jid, name, error = self._chanstory_resolve_code(
                adapter, code)
            return jid, name, tail[len(first):].strip(), error
        rows: list[dict[str, Any]] = []
        chats = getattr(adapter, "chats", None)
        if callable(chats):
            try:
                rows = [r for r in (chats(limit=500) or [])
                        if str(r.get("kind") or "") == "channel"]
            except Exception as exc:  # noqa: BLE001
                record_failure("runtime_chatops.chanstory_chats",
                               exc)
                rows = []
        for k in range(len(tokens), 0, -1):
            cand = " ".join(tokens[:k]).lower()
            matches = [r for r in rows
                       if str(r.get("title") or "").lower() == cand]
            if len(matches) == 1:
                return (str(matches[0].get("id") or ""),
                        str(matches[0].get("title") or ""),
                        " ".join(tokens[k:]), "")
            if len(matches) > 1:
                return "", "", "", (
                    f"more than one channel named "
                    f"“{' '.join(tokens[:k])}” here — use its "
                    f"link or JID so the post lands on the "
                    f"right one.")
        if "@" not in first and "." not in first:
            jid, name, error = self._chanstory_resolve_code(
                adapter, first)
            if not error:
                return jid, name, tail[len(first):].strip(), ""
            return "", "", "", error
        return "", "", "", (
            f"no channel here answers to “{first}” — pass its "
            f"link, its invite code, its name, or its JID (it "
            f"ends in @newsletter).")

    def _chanstory_resolve_code(self, adapter: Any, code: str
                                ) -> tuple[str, str, str]:
        """(jid, name, error) for one channel invite code."""
        fn = getattr(adapter, "channel_resolve_invite", None)
        if not callable(fn):
            return "", "", ("this WhatsApp surface cannot "
                            "resolve channel links — pass the "
                            "channel JID (it ends in @newsletter).")
        verdict = self._chatops_call(fn, code)
        if isinstance(verdict, dict) and verdict.get("ok"):
            return (str(verdict.get("chat") or ""),
                    str(verdict.get("name") or ""), "")
        return "", "", self._chatops_verdict(verdict, "")

    # ── join + create + leave (parity G2) ────────────────────

    def _chatops_joinlink(self, tail: str, message: Any) -> str:
        """/joinlink — join a WhatsApp group through its invite
        link AS the owner (parity G2, census bundle #4: Devon
        could issue and revoke invite links but never accept
        one — no join verb existed on either bridge engine).
        The bridge resolves the link BEFORE any join attempt,
        so a dead link fails as itself — revoked and malformed
        are different lines — and the landed state is read back
        from membership, never assumed from the accept: joined,
        already_member (no act taken, nothing journaled), and
        approval_pending (the request is filed; the owner is
        NOT a member yet) are three different answers.
        Owner-only; the paired-account + owner-stop gate is the
        /poststatus family gate; a landed join or filed request
        journals in the personal vault."""
        if not tail:
            return ("usage: /joinlink link or code — the "
                    "chat.whatsapp.com link to the group, or "
                    "just its code.")
        svc, rec, error = self._chatops_gated_account(
            "whatsapp", "joinlink")
        if error:
            return error
        adapter = self._adapter_for("whatsapp")
        if adapter is None:
            return "no live whatsapp surface here."
        fn = getattr(adapter, "group_join_link", None)
        if not callable(fn):
            return ("this WhatsApp surface has no group-join "
                    "verb wired.")
        verdict = self._chatops_call(fn, tail)
        if not (isinstance(verdict, dict) and verdict.get("ok")):
            return self._chatops_verdict(verdict, "")
        state = str(verdict.get("state") or "")
        subject = str(verdict.get("subject") or "")
        jid = str(verdict.get("chat") or "")
        name = f"“{subject}”" if subject else "the group"
        if state == "already_member":
            return (f"you're already a member of {name} — "
                    f"nothing to join.")
        personal = svc["personal"]
        if state == "approval_pending":
            personal.note_message_acted(
                "whatsapp", kind="group_join_requested",
                account_id=rec.id, chat_id=jid,
                note="join requested as the owner via /joinlink")
            return (f"{name} files joins for admin approval — "
                    f"your request is in. You're not a member "
                    f"yet; an admin there has to approve it.")
        personal.note_message_acted(
            "whatsapp", kind="group_joined",
            account_id=rec.id, chat_id=jid,
            note="joined as the owner via /joinlink")
        return f"joined {name}."

    def _chatops_newchat(self, kind: str, tail: str,
                         message: Any) -> str:
        """/newgroup · /newchannel — create a Telegram group or
        channel AS the owner (parity G2, census bundle #3: the
        userbot could join chats but never create one — no
        CreateChatRequest / CreateChannelRequest existed
        anywhere in the repo).  The verbs live on the personal
        Telegram surface only: the account that creates is the
        paired userbot, so on any other surface the answer
        names where the command lives rather than creating
        from the wrong hat.  Telegram's refusals (at the
        channel cap, CHANNELS_TOO_MUCH) come back in its own
        words via the adapter verdict.  Owner-only behind the
        family gate; a landed creation journals chat_created."""
        platform = str(getattr(message.chat, "platform", "")
                       or "")
        if platform != "telegram":
            return (f"/{kind} lives on your personal Telegram "
                    f"— run it from there (it creates as your "
                    f"userbot account).")
        if not tail:
            return f"usage: /{kind} title — the new chat's name."
        svc, rec, error = self._chatops_gated_account(
            "telegram", kind)
        if error:
            return error
        adapter = self._adapter_for("telegram")
        fn = getattr(
            adapter,
            "create_chat" if kind == "newgroup"
            else "create_channel", None)
        if not callable(fn):
            return ("this Telegram surface has no chat-create "
                    "verb wired — /newgroup and /newchannel "
                    "live on your personal (userbot) surface.")
        verdict = self._chatops_call(fn, tail)
        if not (isinstance(verdict, dict) and verdict.get("ok")):
            return self._chatops_verdict(verdict, "")
        new_id = str(verdict.get("chat_id") or "")
        svc["personal"].note_message_acted(
            "telegram", kind="chat_created",
            account_id=rec.id, chat_id=new_id,
            note=f"created as the owner via /{kind}")
        what = "group" if kind == "newgroup" else "channel"
        return f"created your Telegram {what} “{tail}”."

    def _chatops_leave(self, tail: str, message: Any) -> str:
        """/leave — leave a group or channel AS the owner
        (parity G2: census bundle #3's leave half, plus the
        WhatsApp ride): the target follows the chatops grammar
        (the chat it was sent in, or another named the way
        chats are named).  WhatsApp rides the bridge's
        group_leave; Telegram rides the userbot's leave_chat,
        with the bot adapter's admin_leave as the bot-hat ride
        when the surface at hand is the bot.  A DM is not
        leavable on either platform and answers as exactly
        that — never a fake leave.  Owner-only behind the
        family gate; a landed leave journals chat_left."""
        ref, adapter, error = self._chatops_target(message, tail)
        if error:
            return error
        platform = str(getattr(ref, "platform", "") or "")
        chat_kind = str(getattr(ref, "kind", "") or "")
        name = str(getattr(ref, "title", "")
                   or getattr(ref, "chat_id", "") or "that chat")
        if platform not in ("whatsapp", "telegram"):
            return (f"/leave has no ride on {platform} — it "
                    f"leaves WhatsApp and Telegram chats.")
        if chat_kind == "dm":
            return (f"{name} is a DM — a one-to-one chat "
                    f"isn't leavable. /leave applies to groups "
                    f"and channels.")
        svc, rec, error = self._chatops_gated_account(
            platform, "leave")
        if error:
            return error
        if adapter is None:
            return f"no live {platform} surface here."
        if platform == "whatsapp":
            if chat_kind != "group":
                return (f"{name} isn't a WhatsApp group — "
                        f"/leave on WhatsApp leaves groups.")
            fn = getattr(adapter, "group_leave", None)
            if not callable(fn):
                return ("this WhatsApp surface has no "
                        "group-leave verb wired.")
            verdict = self._chatops_call(fn, ref)
            if not (isinstance(verdict, dict)
                    and verdict.get("ok")):
                return self._chatops_verdict(verdict, "")
        else:
            fn = getattr(adapter, "leave_chat", None)
            if not callable(fn):
                fn = getattr(adapter, "admin_leave", None)
            if not callable(fn):
                return ("this Telegram surface has no leave "
                        "verb wired.")
            verdict = self._chatops_call(fn, ref)
            if not (isinstance(verdict, dict)
                    and verdict.get("ok")):
                return self._chatops_verdict(verdict, "")
        svc["personal"].note_message_acted(
            platform, kind="chat_left",
            account_id=rec.id,
            chat_id=str(getattr(ref, "chat_id", "") or ""),
            note="left as the owner via /leave")
        return f"left {name}."

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
                return (f"usage: /{kind} a number or @username "
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
                    "first. For a number: /banchecker number.")
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
            # Parity G3: the bridges tally the votes they
            # decrypt on polls they have seen (there is still
            # no server-side tally to fetch — bridge memory is
            # the count, and a poll outside it refuses in
            # words rather than inventing numbers).
            adapter = self._adapter_for(platform)
            fn = getattr(adapter, "poll_results", None)
            if not callable(fn):
                return ("this WhatsApp surface can't re-read "
                        "poll results.")
            verdict = self._chatops_call(fn, chat, mid)
            if not (isinstance(verdict, dict)
                    and verdict.get("ok")):
                return self._chatops_verdict(verdict, "")
            text = format_poll_results(verdict)
            unreadable = verdict.get("unreadable_votes")
            if unreadable:
                text += (f"\n({unreadable} vote"
                         f"{'s' if unreadable != 1 else ''} "
                         f"arrived that the bridge couldn't "
                         f"decrypt — not counted.)")
            return text
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

    # ── own identity + poll sending (parity G3) ─────────────

    def _chatops_media_of(self, message: Any) -> tuple[str, str]:
        """(path, kind) of the media a command rides on or
        replies to — the /poststatus extraction, shared by the
        profile-photo setter."""
        for m in (getattr(message, "media", None) or []):
            p = str(getattr(m, "path", "") or "")
            if p:
                return p, str(getattr(m, "kind", "") or "")
        meta = getattr(message, "meta", None) or {}
        quoted = meta.get("replied_media_local") or {}
        if isinstance(quoted, dict) and quoted.get("path"):
            return (str(quoted["path"]),
                    str(quoted.get("kind") or ""))
        return "", ""

    def _chatops_identity_home(self, message: Any, kind: str
                               ) -> tuple[Any, Any, Any, str, str]:
        """The shared gate for the /myprofile + /privacy
        family: (svc, rec, adapter, platform, error) — the
        WhatsApp personal surface (G3) and the Telegram
        personal (userbot) surface (G5) are the two homes; a
        paired account must exist for the surface the command
        was run on, and a live adapter must be attached.
        Setters additionally honor the owner's standing stop
        (checked by the caller, since reads are not acts as
        him)."""
        platform = str(getattr(message.chat, "platform", "")
                       or "")
        if platform not in ("whatsapp", "telegram"):
            return None, None, None, "", (
                f"/{kind} shapes your personal WhatsApp or "
                f"Telegram identity — run it from your "
                f"WhatsApp or your personal Telegram.")
        label = ("WhatsApp" if platform == "whatsapp"
                 else "Telegram")
        svc, rec = self._chatops_personal_account(platform)
        if rec is None:
            return None, None, None, "", (
                f"no paired {label} account — /pair one "
                f"first; /{kind} acts on the personal account.")
        adapter = self._adapter_for(platform)
        if adapter is None:
            return None, None, None, "", (
                f"no live {label} surface here to act from.")
        return svc, rec, adapter, platform, ""

    def _chatops_myprofile(self, tail: str, message: Any) -> str:
        """/myprofile — read and set the paired personal
        account's own identity: WhatsApp (parity G3, census
        bundle #5) and the Telegram userbot (parity G5, bundle
        #8) under ONE grammar — the name people see, the About
        text, the profile photo, and on Telegram the username
        (WhatsApp accounts have none; asked there, the command
        says exactly that).  Before G3/G5 Devon could only
        READ a push name off inbound messages; nothing on
        either account could change.  Setters are acts as the
        owner: the standing stop gates them and every landed
        change journals."""
        svc, rec, adapter, platform, error = \
            self._chatops_identity_home(message, "myprofile")
        if error:
            return error
        label = ("WhatsApp" if platform == "whatsapp"
                 else "Telegram")
        personal = svc["personal"]
        parts = tail.split(None, 1)
        head = parts[0].lower() if parts else ""
        rest = parts[1].strip() if len(parts) > 1 else ""
        if not head:
            verdict = self._chatops_call(adapter.get_profile)
            if not (isinstance(verdict, dict)
                    and verdict.get("ok")):
                return self._chatops_verdict(verdict, "")
            lines = [f"👤 Your {label} identity:"]
            lines.append(f"• name: {verdict.get('name') or '—'}")
            if "about" in verdict:
                lines.append(
                    f"• about: {verdict.get('about') or '(empty)'}")
            else:
                lines.append(
                    "• about: couldn't read it just now"
                    + (f" ({verdict.get('info_error')})"
                       if verdict.get("info_error") else ""))
            if "username" in verdict:
                handle = str(verdict.get("username") or "")
                lines.append("• username: "
                             + (f"@{handle}" if handle
                                else "none"))
            lines.append("• photo: "
                         + ("set" if verdict.get("has_photo")
                            else "none"))
            if platform == "telegram":
                lines.append(
                    "Change it: /myprofile name First Last · "
                    "/myprofile about some text · "
                    "/myprofile username newhandle · "
                    "/myprofile photo (reply to a picture)")
            else:
                lines.append(
                    "Change it: /myprofile name New Name · "
                    "/myprofile about some text · "
                    "/myprofile photo (reply to a picture)")
            return "\n".join(lines)
        if svc["registry"].owner_stopped(platform):
            return (f"{label} is under your standing stop — "
                    "it stays halted until you start it again; "
                    "nothing was changed.")
        if head == "name":
            if not rest:
                return ("usage: /myprofile name Your New Name")
            if platform == "telegram":
                # Telegram names are first + last: the typed
                # words split on the first space, and a single
                # word clears the last name — the name becomes
                # exactly what was typed, as on WhatsApp.
                words = rest.split(None, 1)
                verdict = self._chatops_call(
                    adapter.set_profile_name, words[0],
                    words[1] if len(words) > 1 else "")
            else:
                verdict = self._chatops_call(
                    adapter.set_profile_name, rest)
            if isinstance(verdict, dict) and verdict.get("ok"):
                personal.note_profile_changed(
                    platform, account_id=rec.id,
                    field="name", value=rest)
                return (f"done — your {label} name is now "
                        f"“{rest}”.")
            return self._chatops_verdict(verdict, "")
        if head == "username":
            if platform != "telegram":
                return ("usernames are a Telegram thing — "
                        "WhatsApp accounts don't have one to "
                        "set. Run /myprofile username … from "
                        "your Telegram.")
            clearing = rest.lower() == "clear"
            handle = ("" if clearing
                      else rest.lstrip("@").strip())
            if not handle and not clearing:
                return ("usage: /myprofile username newhandle "
                        "— or /myprofile username clear to "
                        "remove it")
            verdict = self._chatops_call(
                adapter.set_username, handle)
            if isinstance(verdict, dict) and verdict.get("ok"):
                personal.note_profile_changed(
                    platform, account_id=rec.id,
                    field="username", value=handle)
                if handle:
                    return (f"done — your Telegram username "
                            f"is now @{handle}.")
                return ("done — your Telegram username is "
                        "removed.")
            return self._chatops_verdict(verdict, "")
        if head == "about":
            if rest.lower() == "clear":
                rest = ""
            verdict = self._chatops_call(adapter.set_about, rest)
            if isinstance(verdict, dict) and verdict.get("ok"):
                personal.note_profile_changed(
                    platform, account_id=rec.id,
                    field="about", value=rest)
                if rest:
                    return f"done — your About now reads “{rest}”."
                return "done — your About is cleared."
            return self._chatops_verdict(verdict, "")
        if head == "photo":
            if rest.lower() == "remove":
                verdict = self._chatops_call(
                    adapter.set_profile_photo, remove=True)
                if isinstance(verdict, dict) and verdict.get("ok"):
                    personal.note_profile_changed(
                        platform, account_id=rec.id,
                        field="photo", value="removed")
                    return "done — your profile photo is removed."
                return self._chatops_verdict(verdict, "")
            path, _kind = self._chatops_media_of(message)
            if not path:
                if platform == "telegram":
                    return (
                        "reply to a picture with /myprofile photo "
                        "(or send the picture with /myprofile photo "
                        "as its caption) — Telegram needs the "
                        "image itself.")
                return ("reply to a picture with /myprofile photo "
                        "(or send the picture with /myprofile photo "
                        "as its caption) — WhatsApp needs the "
                        "image itself, and on the Go bridge it "
                        "must be a JPEG.")
            verdict = self._chatops_call(
                adapter.set_profile_photo, path)
            if isinstance(verdict, dict) and verdict.get("ok"):
                personal.note_profile_changed(
                    platform, account_id=rec.id, field="photo",
                    value=str(verdict.get("picture_id") or "set"))
                return "done — your new profile photo is up."
            return self._chatops_verdict(verdict, "")
        if platform == "telegram":
            return ("usage: /myprofile — or /myprofile name …, "
                    "/myprofile about …, /myprofile username …, "
                    "/myprofile photo")
        return ("usage: /myprofile — or /myprofile name …, "
                "/myprofile about …, /myprofile photo")

    def _chatops_privacy(self, tail: str, message: Any) -> str:
        """/privacy — read and set the paired personal
        account's privacy settings: WhatsApp (parity G3,
        census bundle #5) and the Telegram userbot (parity
        G5, bundle #8) under one grammar — each platform's
        own categories, the values validated against that
        platform's vocabulary before traffic.  On Telegram
        the settable words are the three rule shapes a bare
        word can honestly name (all / contacts / none);
        finer Telegram rules read back as named labels,
        never squeezed into a word they are not.  A category
        an engine has no setter for answers unsupported on
        that rung, in the verdict's own words."""
        from ...social.chat.chatops import (
            PRIVACY_VALUE_WORDS,
            format_privacy_settings,
            format_tg_privacy_settings,
            privacy_pair_error,
            tg_privacy_pair_error,
        )

        svc, rec, adapter, platform, error = \
            self._chatops_identity_home(message, "privacy")
        if error:
            return error
        label = ("WhatsApp" if platform == "whatsapp"
                 else "Telegram")
        parts = tail.split()
        if not parts:
            verdict = self._chatops_call(adapter.get_privacy)
            if not (isinstance(verdict, dict)
                    and verdict.get("ok")):
                return self._chatops_verdict(verdict, "")
            if platform == "telegram":
                return format_tg_privacy_settings(
                    verdict.get("settings"))
            return format_privacy_settings(
                verdict.get("settings"))
        if len(parts) != 2:
            return ("usage: /privacy category value — e.g. "
                    "/privacy lastseen contacts; bare /privacy "
                    "lists what each category is set to.")
        category, value = parts[0].lower(), parts[1].lower()
        problem = (tg_privacy_pair_error(category, value)
                   if platform == "telegram"
                   else privacy_pair_error(category, value))
        if problem:
            return f"couldn't do it — {problem}"
        if svc["registry"].owner_stopped(platform):
            return (f"{label} is under your standing stop — "
                    "it stays halted until you start it again; "
                    "nothing was changed.")
        verdict = self._chatops_call(
            adapter.set_privacy, category, value)
        if not (isinstance(verdict, dict) and verdict.get("ok")):
            return self._chatops_verdict(verdict, "")
        svc["personal"].note_profile_changed(
            platform, account_id=rec.id,
            field=f"privacy:{category}", value=value)
        word = PRIVACY_VALUE_WORDS.get(value, value)
        line = f"done — {category} is now {word}."
        back = (verdict.get("settings") or {}).get(category)
        if back and back != value:
            line += (f" It reads back as {back} — {label} "
                     f"kept its own value.")
        return line

    def _chatops_addcontact(self, tail: str, message: Any) -> str:
        """/addcontact Name | +number — add a person to the
        paired Telegram account's contacts (parity G5, census
        bundle #8).  Telegram-personal only: WhatsApp contacts
        live in the phone's own address book and neither
        bridge carries an add-contact verb, so asked on
        WhatsApp the command answers exactly that instead of
        faking one.  An act as the owner — the standing stop
        gates it and a landed add journals contact_added."""
        platform = str(getattr(message.chat, "platform", "")
                       or "")
        if platform == "whatsapp":
            return ("adding a contact is a Telegram-surface "
                    "act here — on WhatsApp your contacts live "
                    "in the phone's address book and the "
                    "bridges have no verb for them. Run "
                    "/addcontact from your Telegram.")
        svc, rec, adapter, _platform, error = \
            self._chatops_identity_home(message, "addcontact")
        if error:
            return error
        name_part, sep, phone_part = tail.partition("|")
        name = name_part.strip()
        phone = phone_part.strip()
        if not sep or not name or not phone:
            return ("usage: /addcontact Name | +number — "
                    "e.g. /addcontact Ada Lovelace | "
                    "+2348000000000")
        if svc["registry"].owner_stopped("telegram"):
            return ("Telegram is under your standing stop — "
                    "it stays halted until you start it again; "
                    "nothing was changed.")
        words = name.split(None, 1)
        verdict = self._chatops_call(
            adapter.add_contact, words[0], phone,
            words[1] if len(words) > 1 else "")
        if not (isinstance(verdict, dict)
                and verdict.get("ok")):
            return self._chatops_verdict(verdict, "")
        svc["personal"].note_contact_added(
            "telegram", account_id=rec.id,
            counterpart_id=str(verdict.get("user_id") or ""),
            name=name, phone=phone)
        return f"done — {name} is in your Telegram contacts."

    def _chatops_makepoll(self, tail: str, message: Any) -> str:
        """/makepoll — post a native poll in THIS WhatsApp chat
        as the owner (parity G3, census bundle #6; /poll is the
        group-economy poll — a different instrument, kept
        distinct).  Grammar: /makepoll Question | Option |
        Option, with a leading 'multi |' for several choices.
        A landed poll journals, and its results read back with
        /pollres (reply to the poll) now that something can
        finally produce one."""
        from ...social.chat.chatops import parse_makepoll_args

        platform = str(getattr(message.chat, "platform", "")
                       or "")
        if platform != "whatsapp":
            return ("/makepoll posts on your personal WhatsApp "
                    "— run it in the WhatsApp chat that should "
                    "carry the poll.")
        svc, rec = self._chatops_personal_account(platform)
        if rec is None:
            return ("no paired WhatsApp account — /pair one "
                    "first; /makepoll posts from the personal "
                    "account.")
        if svc["registry"].owner_stopped(platform):
            return ("WhatsApp is under your standing stop — "
                    "it stays halted until you start it again; "
                    "no poll was posted.")
        adapter = self._adapter_for(platform)
        if adapter is None:
            return "no live WhatsApp surface here to post from."
        fn = getattr(adapter, "send_poll", None)
        if not callable(fn):
            return ("this WhatsApp surface has no poll verb "
                    "wired.")
        parsed = parse_makepoll_args(tail)
        if not parsed.get("ok"):
            return str(parsed.get("error") or "bad poll shape")
        result = self._chatops_call(
            fn, message.chat, parsed["question"],
            parsed["options"], multiple=parsed["multiple"])
        if not getattr(result, "ok", False):
            return (f"couldn't post the poll — "
                    f"{getattr(result, 'error', '') or 'the surface refused'}")
        svc["personal"].note_poll_posted(
            platform, account_id=rec.id, counterpart_id=str(
                getattr(message.chat, "key", "")
                or getattr(message.chat, "chat_id", "")),
            source_id=str(result.message_id or ""),
            question=parsed["question"],
            options=parsed["options"])
        return (f"🗳️ poll posted: “{parsed['question']}” — "
                f"reply to it with /pollres to read the count "
                f"as votes land.")

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

    # ── contacts export (/vcf) ───────────────────────────────

    def _vcf_operator(self, message: Any) -> bool:
        """The owner's own hand, wherever this runtime exposes
        it — guarded so bare rigs answer False, never crash."""
        fn = getattr(self, "_is_operator", None)
        if not callable(fn):
            return False
        try:
            return bool(fn(message))
        except Exception:  # noqa: BLE001 - no proof, no owner
            return False

    def _vcf_admin_ok(self, message: Any) -> bool:
        """Group-admin proof for the sender in this room (the
        cleanup family's gate: operator, or the platform's own
        word that this sender admins the group) — guarded the
        same way as the operator check."""
        fn = getattr(self, "_group_admin_ok", None)
        if not callable(fn):
            return False
        try:
            return bool(fn(message))
        except Exception:  # noqa: BLE001 - no proof, no admin
            return False

    def _vcf_ledger_rows(self, keys: list[str], *,
                         whatsapp: bool) -> list[dict[str, Any]]:
        """Member-ledger rows for one group, from Devon's own
        group surface — the people she has SEEN join and speak.
        Observed, never probed: a row she holds is evidence; a
        person she has never seen is simply not in this source.
        Left members are excluded."""
        from ...social.chat import vcf as _vcf

        surface_fn = getattr(self, "_group_surface", None)
        if not callable(surface_fn):
            return []
        try:
            surface = surface_fn()
        except Exception:  # noqa: BLE001 - a source, not a need
            return []
        rows: list[dict[str, Any]] = []
        seen_keys: set[str] = set()
        for key in keys:
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            try:
                members = surface.members(key)
            except Exception:  # noqa: BLE001 - degrade a source
                continue
            for uid, led in (members or {}).items():
                if not isinstance(led, dict) or led.get("left_ts"):
                    continue
                numbers: list[str] = []
                if whatsapp:
                    disp, _d = _vcf.whatsapp_number_from_jid(uid)
                    if disp:
                        numbers.append(disp)
                rows.append({
                    "platform_id": str(uid),
                    "name": str(led.get("name") or ""),
                    "name_rank": _vcf.RANK_LEDGER,
                    "username": str(led.get("username") or ""),
                    "numbers": numbers,
                    "bot": bool(led.get("is_bot")),
                    "source": "member ledger"})
        return rows

    def _vcf_roster(self, ref: Any, platform: str
                    ) -> tuple[list[dict[str, Any]], str]:
        """Assemble one group's roster rows from EVERY route
        this deployment has, for the merge.  Returns
        (rows, group_title).  No route is a precondition for
        another: whatever answers, contributes; the coverage
        line names the sources that actually did."""
        from ...social.chat import vcf as _vcf

        rows: list[dict[str, Any]] = []
        title = str(getattr(ref, "title", "") or "")
        chat_id = str(getattr(ref, "chat_id", "") or "")
        if platform == "whatsapp":
            adapter = self._adapter_for("whatsapp")
            if adapter is not None:
                info = self._chatops_call(adapter.group_info, ref)
                if isinstance(info, dict) and info.get("subject"):
                    title = str(info["subject"])
                roster = self._chatops_call(
                    adapter.group_participants, ref)
                for p in roster if isinstance(roster, list) else []:
                    if not isinstance(p, dict):
                        continue
                    jid = str(p.get("jid") or "")
                    disp, _d = _vcf.whatsapp_number_from_jid(jid)
                    rows.append({
                        "platform_id": jid,
                        # The bridges resolve this name from
                        # the account's contact store first
                        # (saved name, then push name) — the
                        # strongest name WhatsApp offers.
                        "name": str(p.get("name") or ""),
                        "name_rank": _vcf.RANK_CONTACT,
                        "username": "",
                        "numbers": [disp] if disp else [],
                        "source": "WhatsApp roster"})
            rows.extend(self._vcf_ledger_rows(
                [f"whatsapp:{chat_id}"], whatsapp=True))
            return rows, title
        # Telegram family: ledger + bot-API administrators +
        # the personal surface's member fetch, merged.  The
        # personal fetch is ONE rung — the export stands on
        # the ledger and the admin list without it.
        plats = [platform, "telegram", "telegram-bot",
                 "telegram-personal"]
        adapters: list[tuple[str, Any]] = []
        seen_obj: set[int] = set()
        for plat in plats:
            fn = getattr(self, "_group_adapter", None)
            adapter = fn(plat) if callable(fn) else None
            if adapter is None:
                adapter = self._adapter_for(plat)
            if adapter is not None and id(adapter) not in seen_obj:
                seen_obj.add(id(adapter))
                adapters.append((plat, adapter))
        fetched: list[dict[str, Any]] = []
        for _plat, adapter in adapters:
            fn = getattr(adapter, "admin_members_full", None)
            if not callable(fn):
                continue
            verdict = self._chatops_call(fn, ref, 20000)
            if isinstance(verdict, dict) and verdict.get("ok"):
                fetched = [r for r in verdict.get("members") or []
                           if isinstance(r, dict)]
                break
        if not fetched:
            for _plat, adapter in adapters:
                fn = getattr(adapter, "admin_members", None)
                if not callable(fn):
                    continue
                verdict = self._chatops_call(fn, ref, 500)
                if isinstance(verdict, dict) and verdict.get("ok"):
                    fetched = [r for r in verdict.get("members")
                               or [] if isinstance(r, dict)]
                    break
        for r in fetched:
            rows.append({
                "platform_id": str(r.get("id") or ""),
                "name": str(r.get("name") or ""),
                "name_rank": _vcf.RANK_DISPLAY,
                "username": str(r.get("username") or ""),
                "numbers": [str(r["phone"])] if r.get("phone")
                else [],
                "bot": bool(r.get("bot") or r.get("is_bot")),
                "deleted": bool(r.get("deleted")),
                "source": "personal member list"})
        for _plat, adapter in adapters:
            fn = getattr(adapter, "admin_administrators", None)
            if not callable(fn):
                continue
            verdict = self._chatops_call(fn, ref)
            if isinstance(verdict, dict) and verdict.get("ok"):
                for a in verdict.get("administrators") or []:
                    if not isinstance(a, dict):
                        continue
                    rows.append({
                        "platform_id": str(a.get("id") or ""),
                        "name": str(a.get("name") or ""),
                        "name_rank": _vcf.RANK_DISPLAY,
                        "username": str(a.get("username") or ""),
                        "numbers": [],
                        "source": "admin list"})
                break
        rows.extend(self._vcf_ledger_rows(
            [f"{p}:{chat_id}" for p in dict.fromkeys(plats)],
            whatsapp=False))
        return rows, title

    def _vcf_deliver(self, message: Any, title: str,
                     merged: dict[str, Any], *,
                     operator: bool,
                     filename: str = "") -> str:
        """Render the merged people to a .vcf, send it into
        this chat with the coverage line as its caption, and
        journal the export.  Shared by the group and manual
        forms."""
        from pathlib import Path

        from ...social.chat import vcf as _vcf

        people = merged["people"]
        carded = [p for p in people if p["numbers"]]
        hidden = len(people) - len(carded)
        sources: list[str] = []
        for p in people:
            for s in p["sources"]:
                if s not in sources:
                    sources.append(s)
        line = _vcf.compose_export_line(
            title=title, people=len(people), cards=len(carded),
            hidden=hidden, sources=sources)
        if not carded:
            return (f"{line} — so no file: a card needs a "
                    f"number, and none were visible.")
        body = _vcf.build_vcf(carded)
        if not filename:
            filename = (f"{_vcf.slugify_filename(title)}"
                        f"-contacts.vcf")
        ws = Path(str(getattr(
            getattr(getattr(self, "context", None), "settings",
                    None), "workspace_dir", ".") or "."))
        out_dir = ws / "data" / "exports"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / filename
            path.write_bytes(body.encode("utf-8"))
        except OSError as exc:
            record_failure("runtime_chatops.vcf_write", exc)
            return (f"{line} — but writing the file failed: "
                    f"{exc}")
        gateway = getattr(self, "gateway", None)
        send_file = getattr(gateway, "send_file", None)
        if gateway is None or send_file is None:
            return (f"{line} — the file is built at {path}, "
                    f"but this surface can't carry files back.")
        result = send_file(
            str(getattr(message.chat, "platform", "") or ""),
            message.chat, str(path), caption=line, name=filename)
        if not getattr(result, "ok", False):
            return (f"{line} — but sending the file failed: "
                    f"{getattr(result, 'error', '') or 'send refused'} "
                    f"(it is kept at {path}.)")
        self._vcf_journal(message, title, len(people),
                          len(carded), hidden, sources,
                          personal_ok=operator)
        return (f"📇 sent {filename} — {len(carded)} contact "
                f"card{'' if len(carded) == 1 else 's'} for "
                f"{title}.")

    def _vcf_journal(self, message: Any, title: str, people: int,
                     cards: int, hidden: int,
                     sources: list[str], *,
                     personal_ok: bool) -> None:
        """The export in the record: the owner's own acts land
        in the personal vault when a paired account covers
        this platform (the personal-surface pattern); an
        admin's export in a group lands in the group's own
        audit trail instead — his journal is for HIS acts."""
        platform = str(getattr(message.chat, "platform", "")
                       or "")
        base_platform = ("whatsapp" if platform == "whatsapp"
                         else "telegram")
        svc, rec = None, None
        if personal_ok:
            try:
                svc, rec = self._chatops_personal_account(
                    base_platform)
            except Exception:  # noqa: BLE001 - a record,
                svc, rec = None, None    # never a gate
        if svc is not None and rec is not None:
            try:
                svc["personal"].note_vcf_exported(
                    base_platform, account_id=rec.id,
                    counterpart_id=str(
                        getattr(message.chat, "key", "") or ""),
                    group_title=title, people=people,
                    cards=cards, hidden=hidden, sources=sources)
                return
            except Exception as exc:  # noqa: BLE001
                record_failure("runtime_chatops.vcf_journal", exc)
        surface_fn = getattr(self, "_group_surface", None)
        key = str(getattr(message.chat, "key", "") or "")
        if callable(surface_fn) and key:
            try:
                surface_fn()._audit(
                    key, "vcf_exported", people=people,
                    cards=cards, hidden=hidden,
                    sources=list(sources))
            except Exception as exc:  # noqa: BLE001
                record_failure("runtime_chatops.vcf_audit", exc)

    def _chatops_vcf(self, tail: str, message: Any) -> str:
        """/vcf — export a group's people as one import-ready
        contacts file: each person a card, their own number
        with their own name (owner ask, 2026-11-08).

        Forms: bare in a group exports THAT group; a group
        name from a DM exports that group; the manual shape
        ``Name | +number [| +number]`` (one person per line)
        cards people who share no group with her.  The roster
        merges every route she has (see _vcf_roster) — no
        route is a gate.  Posture: the owner anywhere; inside
        a group, the owner or a group admin — it hands out
        people's numbers, so it is never public."""
        from ...social.chat import vcf as _vcf

        chat = message.chat
        kind_here = str(getattr(chat, "kind", "") or "")
        operator = self._vcf_operator(message)

        if "|" in tail:
            if kind_here == "group" and not operator \
                    and not self._vcf_admin_ok(message):
                return ("/vcf hands out people's numbers — "
                        "that's the owner's or a group admin's "
                        "call, not a member's.")
            if kind_here != "group" and not operator:
                return ("/vcf is the owner's tool — it makes "
                        "contact cards with people's numbers.")
            parsed = _vcf.parse_manual_entries(tail)
            if not parsed.get("ok"):
                return str(parsed.get("error") or "bad shape")
            rows = [{
                "platform_id": "", "name": e["name"],
                "name_rank": _vcf.RANK_CONTACT, "username": "",
                "numbers": list(e["numbers"]),
                "source": "manual entry"}
                for e in parsed["entries"]]
            merged = _vcf.merge_people(rows)
            return self._vcf_deliver(
                message, "manual contacts", merged,
                operator=operator, filename="contacts.vcf")

        if tail:
            if not operator:
                return ("/vcf by group name is the owner's "
                        "tool — inside the group itself, a "
                        "group admin can run it too.")
            ref, _adapter, error = self._chatops_target(
                message, tail)
            if error:
                return error
        else:
            if kind_here != "group":
                return ("run /vcf inside a group to export "
                        "its people — or /vcf Group Name from "
                        "here, or /vcf Name | +number for a "
                        "person with no group in common.")
            ref = chat
        if str(getattr(ref, "kind", "") or "") != "group":
            return ("that's not a group — /vcf exports a "
                    "group's people; channels and DMs have no "
                    "member roster to card.")
        platform = str(getattr(ref, "platform", "") or "")
        same_room = str(getattr(ref, "key", "") or "") == \
            str(getattr(chat, "key", "") or "")
        if same_room and not operator \
                and not self._vcf_admin_ok(message):
            return ("/vcf hands out people's numbers — "
                    "that's the owner's or a group admin's "
                    "call, not a member's.")
        if platform != "whatsapp" \
                and not platform.startswith("telegram"):
            return (f"/vcf rides WhatsApp and Telegram — the "
                    f"{platform} surface has no roster route "
                    f"to build from.")
        rows, title = self._vcf_roster(ref, platform)
        if not rows:
            return (f"no roster came back for "
                    f"{title or 'that group'} from any route — "
                    f"nothing to export.")
        merged = _vcf.merge_people(rows)
        return self._vcf_deliver(
            message, title or "this group", merged,
            operator=operator)
