# Group & Channel Handling — Mining Report

Written **before** build commits. Surveyed the actual installed dependencies and
existing repo modules, not tutorials.

## 1. Baileys 6.7.18 — WhatsApp group mutations

Verified against `bridge/node_modules/@whiskeysockets/baileys/lib/Socket/` (installed):

**groups.js / groups.d.ts**
- `groupCreate(subject, participants: string[])` → `Promise<GroupMetadata>`
- `groupParticipantsUpdate(jid, participants: string[], action: ParticipantAction)` → `{status, jid, content}` — action ∈ `add | remove | promote | demote`. One call does all four.
- `groupUpdateSubject(jid, subject)` → `Promise<void>`
- `groupUpdateDescription(jid, description?)` → `Promise<void>`
- `groupLeave(jid)` → `Promise<void>`
- `groupSettingUpdate(jid, setting)` — setting ∈ `announcement | not_announcement | locked | unlocked`
- `groupInviteCode(jid)`, `groupRevokeInvite(jid)`
- `groupJoinApprovalMode(jid, mode)`, `groupMemberAddMode(jid, mode)`
- `groupRequestParticipantsList(jid)`, `groupRequestParticipantsUpdate(jid, participants, action)` (approve/reject)

**chats.js** — `updateProfilePicture(jid, content)` works for group JIDs (takes a Buffer/WAMediaUpload of the image).

**Design note from existing code:** `bridge/whatsapp-groups.mjs` is deliberately
READ-ONLY and says so in its header. Mutating ops get a **separate module**
(`whatsapp-groups-admin.mjs`) so the read-only invariant survives.

## 2. Baileys 6.7.18 — communities

Verified against `lib/Socket/communities.d.ts`:
- `communityCreate(subject, body)` → `Promise<GroupMetadata | null>`
- `communityLeave(id)`, `communityUpdateSubject`, `communityUpdateDescription`
- `communityParticipantsUpdate(jid, participants, action)` (same 4 actions)
- `communityInviteCode`, `communityRevokeInvite`
- `communitySettingUpdate(jid, setting)` (same 4 settings)
- `communityRequestParticipantsList/Update` (approve/reject)

**Subgroup linking:** no `communityLink` in 6.7.18. The established Baileys
pattern is `groupParticipantsUpdate(communityJid, [subgroupJid], 'add')` /
`'remove'`. Implemented with honest error passthrough — if Baileys rejects it,
the owner sees the real reason, not a guess.

**Announcement broadcasts:** a community's announcement group is a normal group
JID with `isCommunityAnnounce`. Posting = `sendMessage` to that JID. The Python
side already has `groups()` returning `is_community_announce` flags, so
`community_broadcast` = find announce JID → send.

## 3. Baileys 6.7.18 — WhatsApp Channels (newsletters)

Verified against `lib/Socket/newsletter.d.ts`:
- `newsletterCreate(name, description?)`, `newsletterDelete(jid)`
- `newsletterFollow(jid)`, `newsletterUnfollow(jid)`
- `newsletterMetadata('invite'|'jid', key, role?)`
- `newsletterUpdateName/Description/Picture(jid, ...)`
- Posting = `sendMessage` to the newsletter JID (`…@newsletter`).

## 4. Telegram Bot API — group/channel admin

Bot API methods (official, long-standing):
- `pinChatMessage`, `unpinChatMessage`, `unpinAllChatMessages`
- `banChatMember`, `unbanChatMember`
- `restrictChatMember` (ChatPermissions object)
- `promoteChatMember` (granular booleans)
- `setChatAdministratorCustomTitle`
- `deleteMessage`
- `getChat`, `getChatMember`, `getChatAdministrators`, `getChatMemberCount`
- `setChatTitle`, `setChatDescription`, `setChatPhoto`, `deleteChatPhoto`
- `exportChatInviteLink`, `createChatInviteLink`, `revokeChatInviteLink`
- `leaveChat`, `setChatPermissions`

The adapter's `_api(method, **params)` already handles 429-retry and error
surfacing — admin methods are thin wrappers over it.

**Channels:** a bot posts to a channel where it's admin via `sendMessage` to
`@channelusername` or the numeric channel id. `TelegramBotAdapter.supported_kinds`
is currently `(DM, GROUP)` — CHANNEL must be added, and `_convert` already maps
`"channel"` chat type → `ChatKind.CHANNEL`.

**Bot joining groups:** bots can't join on their own; an admin adds them (or via
invite link). What the adapter needs is *reading* group messages (already works
via getUpdates — group updates arrive the same way) and *admin rights checking*
(`getChatMember` for our own bot id → `getMe`).

## 5. Gating — public/private matrix

`nomorals/partner/social_gate.py` is code-enforced: outsiders in groups get
public categories only (games, public info) — **no admin, no messaging, no
memory**. Group admin tools must:
- route through the spine tool registry (capability-gated),
- require owner grant (the matrix does this automatically for non-public categories),
- for WhatsApp personal-account mutations: surface the standing warning that
  personal-account automation can get the session flagged/restricted.

## 6. What NOT to build

- No community-link guessing beyond the Baileys pattern above.
- No WhatsApp "channel" admin beyond newsletter create/delete/follow/post —
  newsletter admin roles are limited in the API.
- No Telegram slow-mode/video-chat admin on the Bot API path — Bot API doesn't
  expose them (those are MTProto-only, already noted as unwired).
