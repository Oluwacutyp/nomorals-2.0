"""Member resolution (DESIGN §29): name a person in a group,
get their numeric platform ID — by EVERY path Devon has,
never just one.

Born from two live failures on 2026-11-03: ``/mute @handle``
answered "who should I mute?" (the @-form resolved nothing),
and ``/ban`` on a message sent before Devon deployed dead-
ended the same way — targeting only knew the local records
written since deployment.  Devon is an admin AND runs a
userbot; a human admin would fish the member list, resolve
the handle, or open the old message.  So does she:

1. the replied message's author, as the platform reports it;
2. a text-mention entity in the command message itself;
3. the member ledger (username / name / id index);
4. the bot's admin surface (the group's administrator list);
5. the MTProto userbot — the full participant list, the
   username registry, and fetching a specific (possibly
   pre-deployment) message to name its sender;
6. WhatsApp's group roster / Discord's guild member list on
   their own platforms.

A miss FALLS THROUGH to the next path; it never stops early.
Resolution ends at the numeric platform ID or it has not
resolved — a ban or mute on anything less does nothing.
Every successful platform resolution writes back into the
member ledger with its source, so the cheap local paths get
stronger with use.  (The one exception: a bare numeric token
typed by hand is taken at face value and NOT written back —
a typo must not mint a phantom ledger member.)

Resolution is READ-ONLY: it never messages, follows, or
acts on the person found.  The action itself still goes
through the bot's admin powers (runtime_group._mod_run).

Exhaustion is honest: when every path misses, the result
names what was tried — the caller turns that into ONE plain
line, never the generic "who should I…?" prompt.  A reply
to a channel post / anonymous admin post is a different
fact from an unknown member and is reported as its own
outcome (``channel_post``), with the channel's name.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

#: Telegram's anonymous-admin bot account: a reply to an
#: anonymous admin's message names THIS, never the human.
ANONYMOUS_ADMIN_ID = "1087968824"
ANONYMOUS_ADMIN_USERNAME = "groupanonymousbot"

TELEGRAM_PLATFORMS = ("telegram", "telegram-bot")

#: Sources that prove identity well enough to teach the
#: ledger.  "id" (a hand-typed number) is deliberately absent.
_WRITE_BACK_SOURCES = frozenset({
    "reply", "mention", "bot admins", "userbot members",
    "userbot resolve", "userbot history", "whatsapp roster",
    "discord members",
})


@dataclass
class ResolvedMember:
    """One resolved person: the ID is the point; the rest is
    how humans refer to them + which path found them."""

    user_id: str
    username: str = ""
    name: str = ""
    source: str = ""

    @property
    def label(self) -> str:
        if self.username:
            return f"@{self.username}"
        return self.name or self.user_id


@dataclass
class Resolution:
    """The outcome of one resolution attempt."""

    member: ResolvedMember | None = None
    rest: str = ""                    # tail after the target token
    tried: list[str] = field(default_factory=list)
    named_token: str = ""             # what the admin typed, if anything
    via_reply: bool = False           # target came from a reply
    channel_post: bool = False        # reply target is a channel /
                                      # anonymous post, not a member
    channel_label: str = ""

    @property
    def found(self) -> bool:
        return self.member is not None

    def exhaustion_text(self) -> str:
        """One plain line naming what was tried (§29 honesty)."""
        who = self.named_token or "that member"
        tried = ", ".join(self.tried) or "the member ledger"
        return (f"I couldn't find {who} — I tried {tried}. "
                f"They may have left, changed their handle, or "
                f"never been in this group.")


def _display_name(first: str = "", last: str = "") -> str:
    return " ".join(p for p in (str(first or "").strip(),
                                str(last or "").strip()) if p)


def _looks_like_duration(token: str) -> bool:
    import re

    token = str(token or "").strip().lower()
    if token in ("inf", "forever", "permanent"):
        return True
    return bool(re.fullmatch(r"\d+[smhd]", token))


class MemberResolver:
    """Resolves people in ONE group across every surface.

    ``surface`` is the GroupSurface (ledger + write-back);
    ``adapter_for(platform)`` returns the platform adapter
    or None (the runtime's ``_group_adapter``)."""

    def __init__(self, surface: Any,
                 adapter_for: Callable[[str], Any]) -> None:
        self.surface = surface
        self.adapter_for = adapter_for

    # ── public grammar ──────────────────────────────────────

    def resolve_command(self, group_key: str, message: Any,
                        tail: str = "") -> Resolution:
        """The moderation-command grammar: a reply names the
        target first; otherwise the first tail token does.
        ``rest`` carries whatever follows the target token."""
        out = Resolution(rest=str(tail or ""))
        meta = getattr(message, "meta", None) or {}
        reply = self._resolve_reply(group_key, message, meta, out)
        if reply is not None:
            return reply
        parts = str(tail or "").split(None, 1)
        if not parts:
            return out
        if _looks_like_duration(parts[0]):
            # "/mute 2h" names a length, not a person: leave the
            # tail whole and let the caller ask WHO (usage), not
            # hunt for a member called "2h".
            return out
        out.named_token = parts[0]
        member = self._resolve_token_paths(
            group_key, message, parts[0], out,
            mention_text=str(getattr(message, "text", "") or ""))
        if member is not None:
            out.member = member
            out.rest = parts[1] if len(parts) > 1 else ""
            self._write_back(group_key, member)
        return out

    def resolve_token(self, group_key: str, message: Any,
                      token: str) -> Resolution:
        """One named token (@handle / name / id), no reply —
        the /pay, /grant, /member grammar."""
        out = Resolution(named_token=str(token or ""))
        if not out.named_token:
            return out
        member = self._resolve_token_paths(
            group_key, message, out.named_token, out,
            mention_text=str(getattr(message, "text", "") or ""))
        if member is not None:
            out.member = member
            self._write_back(group_key, member)
        return out

    # ── reply paths (1, then 5-for-history) ─────────────────

    def _resolve_reply(self, group_key: str, message: Any,
                       meta: dict, out: Resolution
                       ) -> Resolution | None:
        rts = meta.get("reply_to_sender")
        rsc = meta.get("reply_to_sender_chat")
        if isinstance(rsc, dict) and (rsc.get("id")
                                      or rsc.get("title")):
            out.channel_post = True
            out.channel_label = (
                str(rsc.get("title") or "a channel"))
            out.via_reply = True
            return out
        if isinstance(rts, dict) and rts.get("id"):
            uid = str(rts["id"])
            uname = str(rts.get("username") or "")
            if uid == ANONYMOUS_ADMIN_ID or \
                    uname.lower() == ANONYMOUS_ADMIN_USERNAME:
                out.channel_post = True
                out.channel_label = "an anonymous admin"
                out.via_reply = True
                return out
            member = ResolvedMember(
                user_id=uid, username=uname,
                name=str(rts.get("name") or ""), source="reply")
            out.member = member
            out.via_reply = True
            self._write_back(group_key, member)
            return out
        reply_to = str(getattr(message, "reply_to", "") or "")
        if not reply_to:
            return None
        out.tried.append("the group's message record")
        row = self.surface.member_by_message_id(group_key, reply_to)
        if row is not None:
            member = ResolvedMember(
                user_id=str(row.get("user_id") or ""),
                username=str(row.get("username") or ""),
                name=str(row.get("name") or ""),
                source="ledger message")
            if member.user_id:
                out.member = member
                out.via_reply = True
                return out
        # The deep path: ask the userbot who wrote that
        # message — the only path that reaches messages from
        # before Devon deployed (the live /ban failure).
        if self._platform(group_key, message) in TELEGRAM_PLATFORMS:
            out.tried.append("the userbot's message history")
            sender = self._userbot_message_sender(
                group_key, message, reply_to)
            if sender is not None:
                if sender.get("kind") == "user" \
                        and sender.get("id"):
                    member = ResolvedMember(
                        user_id=str(sender["id"]),
                        username=str(sender.get("username") or ""),
                        name=str(sender.get("name") or ""),
                        source="userbot history")
                    out.member = member
                    out.via_reply = True
                    self._write_back(group_key, member)
                    return out
                out.channel_post = True
                out.channel_label = (
                    str(sender.get("name") or "a channel"))
                out.via_reply = True
                return out
        return None

    # ── token paths (2, 3, id, 4, 5, 6) ─────────────────────

    def _resolve_token_paths(self, group_key: str, message: Any,
                             token: str, out: Resolution, *,
                             mention_text: str = ""
                             ) -> ResolvedMember | None:
        platform = self._platform(group_key, message)
        want = token.lstrip("@").lower()
        # (2) a text-mention entity in the command message:
        # the platform itself attached the user object.
        mention = self._mention_path(message, token, mention_text,
                                     out)
        if mention is not None:
            return mention
        # (3) the member ledger: username, then display name,
        # then a raw member id the ledger already knows.
        out.tried.append("the member ledger")
        member = self._ledger_path(group_key, token, want)
        if member is not None:
            return member
        if platform in TELEGRAM_PLATFORMS:
            # A bare numeric token IS a Telegram ID (§29:
            # resolution ends at an ID) — taken at face value,
            # never written back, and no platform round-trip
            # can improve on it.
            if token.isdigit():
                return ResolvedMember(user_id=token, source="id")
            # (4) the bot's admin surface.
            member = self._bot_admin_path(group_key, message,
                                          want, out)
            if member is not None:
                return member
            # (5) the userbot: participants, then the registry.
            member = self._userbot_paths(group_key, message,
                                         want, out)
            if member is not None:
                return member
        elif platform == "whatsapp":
            # A bare number is NOT a WhatsApp ID — the JID is;
            # the roster turns the number into the JID.
            member = self._whatsapp_path(group_key, message,
                                         token, want, out)
            if member is not None:
                return member
        elif platform == "discord":
            member = self._discord_path(group_key, message,
                                        token, want, out)
            if member is not None:
                return member
            if token.isdigit():
                return ResolvedMember(user_id=token, source="id")
        return None

    def _mention_path(self, message: Any, token: str,
                      text: str, out: Resolution
                      ) -> ResolvedMember | None:
        meta = getattr(message, "meta", None) or {}
        envelope = meta.get("envelope") or {}
        entities = envelope.get("entities") or []
        candidates: list[tuple[str, str]] = []
        for ent in entities:
            if not isinstance(ent, dict):
                continue
            if ent.get("type") != "text_mention":
                continue
            uid = str(ent.get("user_id") or "")
            if not uid:
                continue
            covered = ""
            try:
                offset = int(ent.get("offset") or 0)
                length = int(ent.get("length") or 0)
                covered = text[offset:offset + length]
            except (TypeError, ValueError):
                covered = ""
            candidates.append((uid, covered))
        if not candidates:
            return None
        out.tried.append("the mention in your message")
        want = token.lstrip("@").lower()
        sender_id = str(getattr(message, "sender_id", "") or "")
        for uid, covered in candidates:
            if uid == sender_id:
                continue
            if covered and covered.lstrip("@").lower() == want:
                return ResolvedMember(user_id=uid, source="mention")
        # A name picked from the mention picker arrives as the
        # member's first name, not an @handle: one candidate +
        # a non-@ token is that pick.
        if len(candidates) == 1 and not token.startswith("@") \
                and candidates[0][0] != sender_id:
            return ResolvedMember(user_id=candidates[0][0],
                                  source="mention")
        return None

    def _ledger_path(self, group_key: str, token: str, want: str
                     ) -> ResolvedMember | None:
        members = self.surface.members(group_key)
        row = members.get(token)
        if row is not None:
            return self._from_row(token, row, "ledger")
        for uid, member in members.items():
            if str(member.get("username") or "").lower() == want \
                    and want:
                return self._from_row(uid, member, "ledger")
        for uid, member in members.items():
            if str(member.get("name") or "").lower() == want \
                    and want:
                return self._from_row(uid, member, "ledger")
        return None

    def _bot_admin_path(self, group_key: str, message: Any,
                        want: str, out: Resolution
                        ) -> ResolvedMember | None:
        # The Bot API surface: the acting bot's adapter, or the
        # companion bot's when the command arrived via userbot.
        adapter = None
        fn = None
        seen: set[str] = set()
        for platform in (self._platform(group_key, message),
                         "telegram-bot", "telegram"):
            if platform in seen:
                continue
            seen.add(platform)
            candidate = self.adapter_for(platform)
            candidate_fn = getattr(candidate,
                                   "admin_administrators", None)
            if candidate is not None and candidate_fn is not None:
                adapter, fn = candidate, candidate_fn
                break
        if adapter is None or fn is None:
            return None
        out.tried.append("the group's admin list")
        try:
            result = fn(self._chat_ref(
                message, str(getattr(adapter, "name", "")
                             or "telegram-bot")))
        except Exception:  # noqa: BLE001 - a path, not a dependency
            return None
        if not isinstance(result, dict) or not result.get("ok"):
            return None
        for admin in result.get("administrators") or []:
            if not isinstance(admin, dict):
                continue
            if str(admin.get("username") or "").lower() == want \
                    or str(admin.get("name") or "").lower() == want:
                return ResolvedMember(
                    user_id=str(admin.get("id") or ""),
                    username=str(admin.get("username") or ""),
                    name=str(admin.get("name") or ""),
                    source="bot admins")
        return None

    def _userbot_paths(self, group_key: str, message: Any,
                       want: str, out: Resolution
                       ) -> ResolvedMember | None:
        adapter = self.adapter_for("telegram")
        if adapter is None:
            return None
        chat = self._chat_ref(message, "telegram")
        members_fn = getattr(adapter, "admin_members", None)
        if members_fn is not None:
            out.tried.append("the group's member list (userbot)")
            try:
                result = members_fn(chat, limit=500)
            except Exception:  # noqa: BLE001 - degrade, never crash
                result = None
            rows = []
            if isinstance(result, dict) and result.get("ok"):
                rows = result.get("members") or []
            elif isinstance(result, list):
                rows = result
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if str(row.get("username") or "").lower() == want \
                        or str(row.get("name") or "").lower() \
                        == want:
                    return ResolvedMember(
                        user_id=str(row.get("id") or ""),
                        username=str(row.get("username") or ""),
                        name=str(row.get("name") or ""),
                        source="userbot members")
        by_name = getattr(adapter, "member_by_username", None)
        if by_name is not None:
            out.tried.append("Telegram's username registry")
            try:
                found = by_name(chat, want)
            except Exception:  # noqa: BLE001 - degrade, never crash
                found = None
            if isinstance(found, dict) and found.get("id"):
                return ResolvedMember(
                    user_id=str(found["id"]),
                    username=str(found.get("username") or ""),
                    name=str(found.get("name") or ""),
                    source="userbot resolve")
        return None

    def _userbot_message_sender(self, group_key: str,
                                message: Any, message_id: str
                                ) -> dict[str, Any] | None:
        adapter = self.adapter_for("telegram")
        fn = getattr(adapter, "message_sender", None)
        if adapter is None or fn is None:
            return None
        try:
            found = fn(self._chat_ref(message, "telegram"),
                       message_id)
        except Exception:  # noqa: BLE001 - degrade, never crash
            return None
        return found if isinstance(found, dict) else None

    def _whatsapp_path(self, group_key: str, message: Any,
                       token: str, want: str, out: Resolution
                       ) -> ResolvedMember | None:
        adapter = self.adapter_for("whatsapp")
        fn = getattr(adapter, "group_participants", None)
        if adapter is None or fn is None:
            return None
        out.tried.append("the WhatsApp group roster")
        try:
            rows = fn(self._chat_ref(message, "whatsapp"))
        except Exception:  # noqa: BLE001 - degrade, never crash
            return None
        digits = "".join(c for c in token if c.isdigit())
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            jid = str(row.get("jid") or "")
            name_hit = str(row.get("name") or "").lower() == want
            number_hit = bool(digits) \
                and jid.split("@")[0] == digits
            if name_hit or number_hit:
                return ResolvedMember(
                    user_id=jid,
                    name=str(row.get("name") or ""),
                    source="whatsapp roster")
        return None

    def _discord_path(self, group_key: str, message: Any,
                      token: str, want: str, out: Resolution
                      ) -> ResolvedMember | None:
        adapter = self.adapter_for("discord")
        fn = getattr(adapter, "members", None)
        if adapter is None or fn is None:
            return None
        out.tried.append("the server member list")
        chat = self._chat_ref(message, "discord")
        try:
            rows = fn(guild=chat.chat_id, limit=500)
        except Exception:  # noqa: BLE001 - degrade, never crash
            return None
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            if str(row.get("id") or "") == token \
                    or str(row.get("name") or "").lower() == want \
                    or str(row.get("tag") or "").lower() == want:
                return ResolvedMember(
                    user_id=str(row.get("id") or ""),
                    name=str(row.get("name") or ""),
                    source="discord members")
        return None

    # ── helpers ─────────────────────────────────────────────

    @staticmethod
    def _from_row(uid: str, row: dict, source: str) -> ResolvedMember:
        return ResolvedMember(
            user_id=str(uid),
            username=str(row.get("username") or ""),
            name=str(row.get("name") or ""), source=source)

    @staticmethod
    def _platform(group_key: str, message: Any) -> str:
        platform = str(getattr(getattr(message, "chat", None),
                               "platform", "") or "")
        if platform:
            return platform
        return str(group_key or "").split(":", 1)[0]

    def _chat_ref(self, message: Any, platform: str) -> Any:
        """A ChatRef for adapter calls, on the surface the
        command arrived on (chat_id from the ref itself, else
        from the group key's suffix)."""
        from .base import ChatKind, ChatRef

        chat = getattr(message, "chat", None)
        chat_id = str(getattr(chat, "chat_id", "") or "")
        if not chat_id:
            key = str(getattr(chat, "key", "") or "")
            chat_id = key.split(":", 1)[1] if ":" in key else ""
        kind = getattr(chat, "kind", ChatKind.GROUP)
        return ChatRef(platform=platform, chat_id=chat_id,
                       kind=kind,
                       title=str(getattr(chat, "title", "") or ""))

    def _write_back(self, group_key: str,
                    member: ResolvedMember) -> None:
        """Teach the ledger (§29 path 6) — platform-proven
        identities only; a hand-typed bare ID teaches nothing."""
        if member.source not in _WRITE_BACK_SOURCES:
            return
        if not (member.username or member.name):
            return
        note = getattr(self.surface, "note_resolution", None)
        if note is None:
            return
        try:
            note(group_key, member.user_id,
                 username=member.username, name=member.name,
                 source=member.source, ts=time.time())
        except Exception:  # noqa: BLE001 - write-back never blocks
            pass
