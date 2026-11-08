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
        if not path or not Path(path).exists():
            return (f"the capture from {who} ({_when(target)}) "
                    f"is in the record but its file is gone from "
                    f"the vault — nothing left to show. "
                    f"/pair saved holds the record.")
        gateway = getattr(self, "gateway", None)
        send_file = getattr(gateway, "send_file", None)
        if gateway is None or send_file is None:
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
            return (f"the capture from {who} is in the vault "
                    f"but sending it back failed: "
                    f"{getattr(result, 'error', '') or 'send refused'}")
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
        WhatsApp channels and Telegram channels both carry
        stories now; on WhatsApp the channel post IS the
        channel's story format (neither bridge library has a
        separate story verb for newsletters), on Telegram it is
        a real stories.SendStory with the channel as peer.
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
                return f"posted to {where} on WhatsApp."
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
