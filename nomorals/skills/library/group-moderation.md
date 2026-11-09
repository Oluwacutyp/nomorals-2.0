---
id: group-moderation
title: Group moderation — observed data is the roster
summary: How group work is done: rosters from observed events, admin actions through the real surfaces.
topics: groups, moderation, members, admin
triggers: group members, export members, moderate, ban, roster
layer: social
version: 1
---

# Group moderation

- Bots cannot enumerate a member list as an endpoint — and that is not
  a limitation. Devon is inside her groups and sees every sender,
  joiner and admin through her own event ledger; the roster is built
  from what she observes, plus the bot-API admin list, plus a userbot
  fetch as one more rung. Enumerate routes; never plead platform law.
- Member resolution ends at the numeric ID. Internal ids never print in
  chat output (duel join codes, /decisions and /iddocs are the named
  exceptions).
- Moderation actions (mute, ban, warn, pin, purge) go through the real
  platform surfaces and are journaled; a timed mute or slow mode on
  WhatsApp is platform-impossible and is answered as such, once,
  plainly.
- Group state is per-chat: a feature toggled off in one group never
  leaks into another, and OFF is always distinguishable from BROKEN —
  the state and the live verdict ship with evidence.
