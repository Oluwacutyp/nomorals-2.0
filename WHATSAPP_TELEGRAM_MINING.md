# WhatsApp + Telegram Bot Mining Reference

Research-only mining pass. Every finding cites its source. UNVERIFIED marks anything not confirmed from a primary source.

---

## 1. Baileys Full API Surface

Library: `bridge/node_modules/@whiskeysockets/baileys` (v6.7.24)

### 1.1 View-Once Messages

**Detection & reading:**
- Incoming messages arrive via `messages.upsert`. View-once content is wrapped: `msg.message.viewOnceMessage` (v1), `viewOnceMessageV2`, or `viewOnceMessageV2Extension`.
- Canonical unwrap: `extractMessageContent(msg.message)` in `Utils/messages.js:570-600` — recursively unwraps `ephemeralMessage`, `viewOnceMessage`, `documentWithCaptionMessage`, `viewOnceMessageV2`, `viewOnceMessageV2Extension`, `editedMessage` via `getFutureProofMessage`.
- Sending view-once: `sendMessage(jid, { image/video/audio: ..., viewOnce: true })` — `Utils/messages.js:441` wraps as `{ viewOnceMessage: { message: m } }`.
- `Types/Message.d.ts:74` — `viewOnce?: boolean` flag on message options.

**Current bridge gap:** `bridge/whatsapp-bridge.mjs` has ZERO view-once handling (grep: no matches). View-once messages arrive but are never unwrapped.

### 1.2 Polls

- Create: `sendMessage(jid, { poll: { name, values: string[], selectableCount?, messageSecret? } })` — `Types/Message.d.ts:83-91` (`PollMessageOptions`), `Socket/messages-send.js:666` adds `polltype: 'creation'` meta node.
- Vote/result events: via `messages.upsert` with poll update messages. UNVERIFIED: exact vote aggregation API — needs runtime observation.

### 1.3 Reactions

- Send: `sendMessage(jid, { react: { text: emoji, key: messageKey } })` — `Utils/messages.js:313` builds `reactionMessage` via `WAProto.Message.ReactionMessage.fromObject`.
- Receive: `messages.reaction` event. Source: `Socket/messages-recv.js` (event map).

### 1.4 Message Edit & Delete

- Edit: `sendMessage(jid, { edit: messageKey, text: "new" })` — `Utils/messages.js:455-463` wraps in `protocolMessage` with `type: MESSAGE_EDIT`, `edit: '1'` attribute (`Socket/messages-send.js:677`).
- Delete (own): `sendMessage(jid, { delete: messageKey })` — `edit='7'` attribute.
- Delete as admin (others' messages in groups): same, but Baileys sets `edit='8'` when `isJidGroup(remoteJid) && !fromMe` — `Socket/messages-send.js:671-675`.
- Pin: `sendMessage(jid, { pin: messageKey })` — `edit='2'` attribute (`Socket/messages-send.js:679`).

### 1.5 Groups

Full surface in `Socket/groups.d.ts`:
- `groupMetadata(jid)` → `GroupMetadata` (includes `participants[]` with `admin: 'admin'|'superadmin'|null`, `isAdmin`, `announce`, `restrict` flags) — `Types/GroupMetadata.d.ts:3-7`
- `groupCreate(subject, participants[])`, `groupLeave(id)`
- `groupParticipantsUpdate(jid, participants[], action)` — action: `'add'|'remove'|'promote'|'demote'|'modify'` (`Types/GroupMetadata.d.ts:7`)
- `groupUpdateSubject`, `groupUpdateDescription`
- `groupInviteCode(jid)` → invite code; `groupRevokeInvite`; `groupAcceptInvite(code)`; `groupGetInviteInfo(code)`
- `groupSettingUpdate(jid, setting)` — setting: `"announcement"|"not_announcement"|"locked"|"unlocked"` (announcement mode = only admins send; locked = only admins edit info)
- `groupMemberAddMode(jid, "admin_add"|"all_member_add")`
- `groupJoinApprovalMode(jid, "on"|"off")` (join requests)
- `groupRequestParticipantsList` / `groupRequestParticipantsUpdate(jid, participants[], "approve"|"reject")`
- `groupToggleEphemeral(jid, seconds)` (disappearing messages)
- `groupFetchAllParticipating()` → all groups the account is in

**Detecting own admin rights:** `groupMetadata(jid).participants` → find own JID → check `admin` field. No separate API needed.

### 1.6 Communities

Full mirror API in `Socket/communities.d.ts`:
- `communityCreate(subject, body)`, `communityMetadata`, `communityLeave`
- `communityParticipantsUpdate` (same add/remove/promote/demote)
- `communityInviteCode`, `communityRevokeInvite`, `communityAcceptInvite`
- `communityLink`/`communityUnlink` for linked groups (check `Socket/communities.d.ts:27-37`)
- Communities are groups flagged `isCommunity`; linked groups via `linkedParent` in metadata.

### 1.7 Status/Stories

- Post: `sendMessage('status@broadcast', { image/video/text })` — `Socket/messages-send.js:258` defines `statusJid = 'status@broadcast'`, relayed specially.
- Read: status updates arrive via `messages.upsert` from `status@broadcast`.
- `statusJidList` option in `relayMessage` for targeting.

### 1.8 Calls — Exact Signaling Surface

- `Socket/index.d.ts:49` — **ONLY `rejectCall(callId, callFrom): Promise<void>` exists.**
- NO `placeCall`, NO `acceptCall`, NO `hangUp` in stock Baileys 6.7.24 (grep-verified).
- Inbound: `ev.on('call', ([call]) => ...)` — `Socket/messages-recv.js:1037`, `WACallEvent` type in `Types/Call.d.ts`.
- `getCallStatusFromNode` in `Utils/generics.js:285` parses call state.
- Outbound requires hand-rolled `<offer>` stanza via `sendNode` + WA's VoIP WASM — open research problem.

### 1.9 Multi-Device / Multi-Account Isolation

- `makeWASocket(config)` takes `auth: authState` — `Socket/socket.js:18`. Each socket gets its own auth state object.
- `useMultiFileAuthState(folder)` — `Utils/use-multi-file-auth-state.js:28` — file-isolated creds per folder, with per-file mutex locks.
- **No global/module-level shared state** in `Socket/socket.js` or auth utils (grep: zero `globalThis`/`global.` references).
- Bridge uses `--creds` dir flag (`bridge/whatsapp-bridge.mjs:68,399`).
- **Conclusion:** N sockets CAN run in one Node process with different creds dirs. Each needs its own port (bridge listens on configurable PORT). WhatsApp-side: each phone number is one account; multi-device limit is per-number (companion devices), not per-process.

### 1.10 Media

- Download: `downloadMediaMessage(msg, type, options)` — `Utils/messages.js:726`; streams via `downloadContentFromMessage`.
- Upload: `sendMessage` with `{ image: { url }, video: { url }, audio: { url }, document: { url } }` — bridge already does this (`bridge/whatsapp-bridge.mjs:543-551`).
- View-once media: download BEFORE the view-once wrapper expires; use `extractMessageContent` first, then `downloadMediaMessage` on the inner.

---

## 2. Telegram Bot API — Group Admin Power

Source: https://core.telegram.org/bots/api (Bot API 10.3, verified 2026-10-09). PTB not installed locally — method names from official docs.

### 2.1 Every Admin Capability + Exact Method

| Capability | Method | Key params / notes |
|---|---|---|
| Pin | `pinChatMessage` | `chat_id`, `message_id`; needs `can_pin_messages` (groups) |
| Unpin | `unpinChatMessage` / `unpinAllChatMessages` | — |
| Delete msg | `deleteMessage` | **48h limit**; admin can delete ANY msg in group; needs `can_delete_messages` in supergroups |
| Batch delete | `deleteMessages` | 1–100 ids |
| Ban | `banChatMember` | `until_date`, `revoke_messages=True` wipes their messages; >366d = forever |
| Unban | `unbanChatMember` | user does NOT auto-rejoin |
| Restrict/mute | `restrictChatMember` | `permissions` ChatPermissions; all-True lifts |
| Ban channel identity | `banChatSenderChat` / `unbanChatSenderChat` | bans a whole channel from posting |
| Default perms | `setChatPermissions` | needs `can_restrict_members` |
| Promote/**demote** | `promoteChatMember` | all `can_*` booleans; **all-False = demote**; supergroups+channels only |
| Admin title | `setChatAdministratorCustomTitle` | 0–16 chars, no emoji |
| Member tags | `setChatMemberTag` | needs `can_manage_tags` |
| Primary invite link | `exportChatInviteLink` | regenerates (revokes old); **bot can't use other admins' links** |
| Invite links | `createChatInviteLink` / `editChatInviteLink` / `revokeChatInviteLink` | `name`, `expire_date`, `member_limit`, `creates_join_request` |
| Join requests | `approveChatJoinRequest` / `declineChatJoinRequest` | needs `can_invite_users` |
| Group settings | `setChatTitle` / `setChatDescription` / `setChatPhoto` / `deleteChatPhoto` | — |
| Forum topics | `createForumTopic` / `editForumTopic` / `closeForumTopic` / `reopenForumTopic` / `deleteForumTopic` | needs `can_manage_topics` |
| Anonymous admin | `promoteChatMember(is_anonymous=True)` | bot appears as "Group" |
| Welcome msgs | `can_send_welcome_messages` right (Bot API 10.3) | native welcome management |
| Install-time rights | `setMyDefaultAdministratorRights` | rights suggested when users add bot via t.me link |

### 2.2 Detecting Own Admin Rights

- **Primary:** `getChatMember(chat_id, bot_own_id)` → `ChatMemberAdministrator` flags: `can_delete_messages`, `can_restrict_members`, `can_promote_members`, `can_change_info`, `can_invite_users`, `can_pin_messages`, `can_manage_topics`, `can_manage_chat`, `can_post_stories`, etc.
- **Event-driven:** `my_chat_member` update fires on own promotion/demotion. `chat_member` update tracks others (needs admin + explicit `allowed_updates`).
- **Secondary:** `getChatAdministrators` (Bot API 10.0 added `return_bots`).

### 2.3 Verified Bot API Gaps (MTProto-only)

| Bot API cannot… | Evidence |
|---|---|
| Set slow mode | No method; `slow_mode_delay` read-only |
| Start/manage video chats | Service-message objects only, no method |
| Read admin event log | Right exists, no method (`channels.getAdminLog` is MTProto) |
| Initiate conversations | "Bots can't start conversations" (core.telegram.org/bots) |
| See group messages without admin/privacy-off | FAQ: privacy mode limits to commands/mentions/replies |
| Delete messages older than 48h | API doc limit |
| Use other admins' invite links | Doc: each admin's links are their own |

### 2.4 Best Patterns (from real group-management bots)

- **Marie/Rose lineage** (PaulSonOfLars/tgbot): modular commands, admin hierarchy, welcome/goodbye, anti-spam, notes/filters, admin logs — the template every fork copies.
- **Consensus feature set:** welcome/goodbye with placeholders → anti-spam pipeline (event-driven scan → delete + restrict/ban) → join-request gating → tiered permissions (Owner > Admin > User) → admin audit via `my_chat_member`/`chat_member` → command scoping (`setMyCommands` with `BotCommandScopeChatAdministrators`).
- **Formatting:** HTML `parse_mode` is the standard.
- **Menus:** `InlineKeyboardMarkup` + `callback_query` (must `answerCallbackQuery`); `ForceReply` for privacy-mode-safe step flows.
- **Privacy mode is the pivot:** disabled (via @BotFather) or admin = bot sees ALL messages. Enabled = only commands/mentions/replies.
- **BotFather toggles:** group privacy off, `setMyDefaultAdministratorRights` for install-time full-power request.

## 3. Owner-Recognition Patterns

| Platform | Pattern | Source |
|---|---|---|
| WhatsApp | JID allowlist (`234...@s.whatsapp.net`), `msg.key.participant`/`remoteJid` vs list, ignore `fromMe` | akinsiraifedayo/whatsapp-bot, xlliyy/bun-whatsapp-bot |
| Telegram bot | `message.from_user.id` vs `OWNER_ID` env/config | Standard; hermes-agent per-profile gateways |
| Telegram MTProto | Session IS the identity — owner = whichever account's session | Telethon session model |
| Dual-identity | Bot account + owner account as separate auth identities, cross-referenced | neoxr-bot (owner vs pairing fields) |

Current Devon gap: `nomorals/social/chat/whatsapp.py` has NO owner-recognition (grep: zero matches). Telegram has `is_owner` via gating (`nomorals/partner/gating.py:57`) but it's chat-classification, not cross-account identity.

## 4. Best-Implementation Patterns — What Ours Doesn't Do

**WhatsApp (Baileys bots in the wild):**
- Structured message output: emoji section headers, monospace commands, progress/status bars, explicit next-action CTA (the Septorch-style the user admired).
- Owner-only command gating via JID allowlist + `ownerOnlyMode`.
- View-once reading, poll creation, reaction handling — all standard in mature bots, all missing in our bridge.

**Telegram (Marie/Rose lineage):**
- Event-driven moderation (not command-only): anti-spam scan → delete + restrict/ban.
- Tiered permissions Owner > Admin > User, checked per-handler.
- Command scoping so admin commands only appear to admins.
- Our `nomorals/social/chat/telegram.py` has ZERO admin methods (no promote/ban/pin/delete) — confirmed by grep.

## 5. Multi-Account Feasibility

### 3.1 WhatsApp — N Baileys Sockets, One Process: PROVEN

**Session isolation is designed in:**
- `useMultiFileAuthState(folder)` (`lib/Utils/use-multi-file-auth-state.js:28`) — all creds + signal keys under `join(folder, ...)`; module-level `fileLocks` Map keyed by full file path, so locks are folder-isolated too.
- `makeWASocket(config)` (`lib/Socket/socket.js:4`) — everything closure-scoped per call: `ws`, `ev`, `noise`, `ephemeralKeyPair` ("Unique for each connection"), `keys`, `signalRepository`. Zero module-level `let`/`var` in `socket.js`, `websocket.js`, `libsignal.js`.
- Wire-tag safety: `uqTagId = generateMdTagPrefix()` per socket (`lib/Utils/generics.js:245-248`) — no tag collisions between sockets.
- Own identity per auth state: `authState.creds.me.id` (`lib/Socket/chats.js:185,215,355`).

**Production proof:** 400+ Baileys clients on one server, different session folders — https://github.com/whiskeysockets/baileys/issues/2234. heis-devine/baileys-pairing-demo documents N numbers, one process, per-number auth folders.

**Constraints:**
- Max **4 companion devices per phone number** (each socket = 1 slot). 4 sockets on 4 different numbers = unlimited.
- One socket per number per folder. Same number in a new folder = second companion device (slot waste).
- QR/pairing per new folder: `sock.requestPairingCode(phoneNumber)` (`lib/Socket/socket.js:351`); custom 8-char codes supported.
- UNVERIFIED: behavior when exceeding 4 companion slots (which device gets kicked). Design for ≤4.

### 3.2 Telegram — N Bots: PROVEN (trivial)

- One `Application` per token (`ApplicationBuilder.token()`) — https://docs.python-telegram-bot.org/en/stable/telegram.ext.applicationbuilder.html
- Same token double-polled = rejected by Telegram. One gateway per token.
- Rate limits are **per bot token**: ~30 msg/sec globally, ~1 msg/sec per private chat, ~20 msg/min per group (429 with `retry_after`) — https://core.telegram.org/bots/faq#broadcasting-to-users
- Same-process N bots: `run_polling()` is blocking; need manual lifecycle with `asyncio.gather`, or one process per bot. UNVERIFIED locally (PTB not installed on this VM).

### 3.3 Telegram — N MTProto Sessions (Telethon): PROVEN

- Session per file: `TelegramClient('name')` → `name.session`. `MemorySession`/`SQLiteSession`/`StringSession` — https://arabic-telethon.readthedocs.io/en/stable/extra/advanced-usage/sessions.html
- N clients in one process proven: LonamiWebs/Telethon#4037 (dict of clients per phone), electric-capital/qu (`TelegramClientManager`, one client per user id), Lonami himself (#152): "cache a client for each phone number."
- **DANGER:** same session string twice = `AUTH_KEY_DUPLICATED`, account logged out. One live client per auth key, always.
- Bot + user mix: Telethon accepts bot tokens — two differently-authed clients, no conflict. Standard hybrid pattern.
- UNVERIFIED: max concurrent MTProto sessions per phone number (no authoritative source).

### 3.4 WhatsApp Dual-Identity (Bot Number + Owner Number): PROVEN pattern

Industry-standard Baileys bot architecture:
- akinsiraifedayo/whatsapp-bot: `config.json` `ownerJids: ["234...@s.whatsapp.net"]` + `ownerOnlyMode` — https://github.com/akinsiraifedayo/whatsapp-bot
- xlliyy/bun-whatsapp-bot: `ctx.isOwner` from `OWNER_NUMBERS`, `ctx.sender`/`ctx.senderNumber` — https://github.com/xlliyy/bun-whatsapp-bot
- ch0c01dxyz/neoxr-bot: separate `"owner"` vs `"pairing"` number config fields.
- Wire mechanism: `msg.key.participant` (groups) / `msg.key.remoteJid` (DMs) normalized against owner list; ignore `msg.key.fromMe`.

**Devon design:** two Baileys sockets, one process — socket A = bot number (`auth/bot/`), socket B = owner number (`auth/owner/`). Owner recognition by JID on socket A; full personal access via socket B.

## 6. Gap Analysis — What Ours Doesn't Do

| Capability | Baileys has it | Bridge has it | Telegram Bot API has it | Our TG adapter has it |
|---|---|---|---|---|
| View-once read | Yes (`extractMessageContent`) | NO | N/A | N/A |
| Polls create/read | Yes | NO | Yes (native polls) | NO |
| Reactions | Yes | NO | Yes | NO |
| Message edit/delete | Yes (incl. admin-delete) | NO | Yes (48h limit) | NO |
| Group admin ops | Yes (full) | Partial (read-only) | Yes (full, §2.1) | NO (zero admin methods) |
| Communities | Yes (full mirror API) | Read-only | N/A (topics ≈ equivalent) | NO |
| Status post/read | Yes | NO | Yes (stories) | NO |
| Calls place/answer | NO (stock) | NO | N/A (no calling) | N/A |
| Multi-account | Yes (creds-dir isolation) | Single bridge process | Yes (per-token + Telethon sessions) | Single |
| Owner recognition | N/A (app-level) | NO | N/A (app-level) | Partial (chat gating, not identity) |
