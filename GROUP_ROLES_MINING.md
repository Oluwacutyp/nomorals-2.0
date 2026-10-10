# Group Role Resolution — Mining Report

Written before build commits. How Telegram and WhatsApp expose member roles.

## Telegram Bot API

`getChatMember(chat_id, user_id)` returns a ChatMember object with a `status` field:
- `"creator"` — group/channel owner
- `"administrator"` — admin with rights
- `"member"` — regular member
- `"restricted"` — restricted member
- `"left"` — not a member
- `"kicked"` — banned

Already wired: `TelegramBotAdapter.admin_member()` in `nomorals/social/chat/telegram.py`
calls `getChatMember` and returns `{"ok": True, "status": ..., "user": ...}`.

Mapping: `creator`/`administrator` → admin. Everything else → member.

## Telegram MTProto (Telethon)

`channels.GetParticipantRequest(channel, participant)` returns participant types:
- `ChannelParticipantCreator` — owner
- `ChannelParticipantAdmin` — admin (has `admin_rights`)
- `ChannelParticipant` — regular member
- `ChannelParticipantBanned` / `ChannelParticipantLeft` — not active

Already wired: `_my_rights_coro()` in `nomorals/social/chat/telegram.py` uses this
pattern for self-rights. Extend to arbitrary users by passing the user entity
instead of `me`.

## WhatsApp (Baileys)

Two sources:
1. `group_info` returns an `admins` list — JIDs of group admins. Simple membership check.
2. `group_participants` returns the roster "with admin roles" — each participant has
   an admin flag (`admin: "admin"` / `"superadmin"` / null in Baileys participant objects).

Already wired: `WhatsAppAdapter.group_info()` returns `admins` list.
`group_participants()` exists for the roster.

Mapping: JID in admins list (or participant admin flag set) → admin.

## Caching

Role lookups hit the network (Bot API call, MTProto request, or bridge command).
Cache for 60s per (platform, chat_key, user_id). Roles change rarely; a stale
60s window is acceptable for permission checks. Fail closed: on lookup error,
treat as regular member (not admin).

## Design decision

Three tiers, not two:
- **owner** (Devon's owner) → everything, everywhere. Unchanged.
- **group_admin** (admin in THIS group) → public tools + admin tools, only in that group.
- **member** (everyone else) → public tools only.

Admin commands: pin/unpin, ban/unban, restrict, promote/demote, delete messages,
group settings, invite links — anything in the "admin" PRIVATE_CATEGORY plus the
new tgbot_*/whatsapp_group_* spine tools.
