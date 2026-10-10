# COMMUNITY Sweep — External Mining Report

Mined 2026-10-10. Every class in `nomorals/community/` compared against the best
(and worst) outside implementations found via web search. Gold taken: merged into
the existing classes, no parallel systems.

---

## 1. audio_rooms.py — Room / RoomStore / lifecycle (Clubhouse/Spaces pattern)

**Best:** Telegram Voice Chats 2.0 (telegram.org/blog/voice-chats-on-steroids),
Reddit Talk (tomsguide), X Spaces, Resonate (github.com/AOSSIE-Org/Resonate —
open-source, LiveKit), sabha- product.md (host admin catalogue), Discord Stages.

What the best do that we didn't:
- **Co-hosts / trusted speakers** (Reddit Talk): invite to co-host, not just host.
  Our rooms had one god-host. → ADDED: co-host list, same privileges as host.
- **Mute-all + lock** (sabha-: `Global Mute All`, `Lock Sabha` rejecting joins).
  → ADDED: `mute_all` (host/cohost, muted-by default off), room lock
  `locked` flag that rejects joins with an honest reason.
- **Separate speaker/listener invite links** (Telegram VC 2.0). → ADDED: invite
  codes minted per role, joined via code.
- **Scheduled rooms + join reminders** (Resonate feature #3). → ADDED:
  `starts_at` scheduling on create; `due_starts()` returns rooms whose scheduled
  time has arrived; host entry point can announce them.
- **Emoji reactions** (Reddit Talk: react with emojis during talks). → ADDED:
  `react(room_id, user_id, emoji)` — per-emoji tallies, chat-renderable.
- **Rich participant lists / bios** (Telegram VC 2.0 shows bio to admins for
  hand-raise triage). → ADDED: hand-raise `note` field (e.g. "Q about X") shown
  to host/co-hosts in queue render.
- **Record red-light + Saved Messages export** (Telegram VC 2.0). Ours already
  renders 🔴 REC. → ADDED: `caption_log` → WebVTT export on end (accessibility
  artifact), plus a `room_stats()` analytics card: duration, peak headcount,
  speakers count, tips total, reactions — Resonate-style analytics-lite.
- **Green room lobby** (sabha-). → ADDED: `lobby(room_id, user_id)` —
  participants "check in" before going live; host sees lobby roster.
- **Hand-raise expiry**: pqp commit (rafaelcg/pqp) shows the queue-order detail
  that matters: server-stamped `handRaisedAt`, single shared sort rule, queue
  survives resume, lowering is self/leave/mod-action. Ours already stamps and
  FIFO-sorts. → ADDED: auto-expire stale raises (configurable, default 30 min),
  and auto-lower when the user starts speaking (promote clears the queue entry).

**Trash to avoid:** HyperCLUB bot (AI-user-management hype, Termux-EXE spam) —
bot-toxic automation, we stayed manual-actions-only.

---

## 2. groups.py — GroupStore / ThemedGroup (Geneva / Bumble BFF / Discord)

**Best:** Geneva (topic rooms + announcements + polls + events per room),
Meetup.com (organizer roles, attendance history "Events I attended" —
wordcamp.org commit), Discord (roles, channels, pins, welcome).

What the best do that we didn't:
- **Roles, not a flat roster**: Geneva/Discord have organizers/moderators.
  → ADDED: `roles` map (owner/admin/member) on the group; `promote`/`demote`;
  owner set at creation.
- **Announcements + pinned posts**: Geneva announces; Discord pins. → ADDED:
  `announce` (role-gated post, rendered first) and `pin`/`unpin` post ids.
- **Attendance/reputation history** (Meetup: "Events I attended"). → ADDED:
  `member_reputation(member)` folds event RSVP→check-in history into a score
  (-1 no-show, +1 showed; from the real Meetup attendance-guideline pattern),
  rendered in `show`.
- **Event linkage**: groups own events; `show` should surface them. → ADDED:
  `upcoming_events(group_id)` via EventStore lazy import (no cycle).
- **Group search**: → ADDED: `search(query)` over name/topic.
- **Welcome message**: Discord-style. → ADDED: `welcome` field set by owner,
  shown on join.
- **Join questions / gate**: Meetup asks screening questions. → ADDED:
  optional `join_question`; answer recorded at join.
- **Icebreaker nudges for quiet groups** (Beatmatch HN idea: real chat UI that
  "gets loud the week of"). → ADDED: quiet-group nudge now suggests a random
  icebreaker prompt.

**Trash to avoid:** Bumble BFF's original 1:1 matching (the pivot signal that
killed it — docstring already cites it).

---

## 3. events.py — EventStore / CommunityEvent (Meetup / Luma / Partiful)

**Best:** Luma (waitlist auto-promote + notification, ticketing tiers),
Meetup (RSVP capacity limits, waitlist, attendance marking), Partiful (manual
add / proxy RSVPs), travel-guides research (Going/Maybe/Can't go/Waitlist
state set; +N guests with count-only guest model; proxy RSVPs keyed by
contact so a later real RSVP attaches instead of duplicating).

What the best do that we didn't:
- **Recurrence beyond weekly**: RFC 5545 RRULE is the standard (python-dateutil
  `rrule` reference impl). We had `"weekly"` only. → ADDED: stdlib-only
  recurrence engine (`daily`/`weekly`/`monthly`, `interval`, `count`/`until`,
  `byweekday`, exceptions `exdates`) — zero-dep by the standing build order.
- **Capacity + waitlist + auto-promote** (Luma/Meetup): → ADDED:
  `capacity`, waitlist flag, `waitlisted` FIFO queue; RSVP auto-promotes on
  cancellation/decline with a notification payload for the host.
- **+N guests** (Paperless Post/Punchbowl): → ADDED: `guests` count on RSVP,
  counts against capacity.
- **RSVP window**: `rsvp_opens_at` / `rsvp_closes_at` (meetup-parity proto).
  → ADDED both + enforcement.
- **Event description / link / end time**: we had no description field at all.
  → ADDED `description`, `ends_at`, `link`.
- **Host proxy RSVP**: → ADDED `rsvp_for(event_id, member, choice, by)` — host
  adds off-platform replies; if the member later RSVPs themselves it attaches
  to the same record.
- **Waitlist state set**: `RSVP_WAITLIST = "waitlist"` alongside yes/no/maybe.
- **Reminders**: Beatmatch ("event chat that gets loud the week of"). →
  ADDED a `1h/24h` *plus* `7-day` window, and reminder payloads now include
  waitlist position + guest counts.
- **Attendance history**: → `rsvp_history(member)` returns (event, choice,
  checked_in) tuples for reputation.

**Trash to avoid:** Partiful automation hype bots; wordcamp.org's uncapped
headings — we cap renders at 10 with counts.

---

## 4. meetups.py — MeetupStore / VenuePoll / CheckIn

**Best:** Doodle (approval voting + open vs hidden polls — comsoc research shows
open polls get higher response rates and higher reported availability), Meetup
organizers (venue search/sponsored venues, rating public meetups — HN ideas),
real-world attendance policies (-1/+1 points, ban at -3, VIP at +10 — real
Meetup group guidelines page).

What the best do that we didn't:
- **Approval voting** (Doodle): vote for MULTIPLE acceptable venues, not one.
  → ADDED: `multi` poll mode — votes are lists of options, winner by approval
  count, ties broken by fewest total votes against.
- **Open vs hidden polls** (Doodle research): → ADDED: `blind` flag; blind
  polls hide votes until closed.
- **Poll deadline + auto-close**: → ADDED `closes_at`; `due_polls()` returns
  expired polls for the host to close + announce.
- **Poll options with details**: venue = name + address + link. → ADDED:
  options are dicts {name, address, link}; backwards-compatible with strings.
- **Winner auto-adopt**: Doodle creators "choose a final date" and notify.
  → ADDED: `adopt_winner(poll_id)` sets the event's `where` to the winning
  venue and returns a notify payload.
- **Reliability score / no-show policy** (real Meetup attendance guidelines):
  → ADDED: `reliability(member)` = shows − no-shows across check-ins;
  `turnout` now feeds it; render flags chronic no-shows.
- **Instant meetups** (HN "spontaneous happy hour" idea): → ADDED:
  `MeetupStore.quick_meetup(...)` — one-call create event + open venue poll
  + notify payload.
- **Host roll-call**: organizers tick attendance (citanz SKILL.md reads
  meetup's attended count). → ADDED: `host_check_in(event_id, members, by)`.

**Trash to avoid:** venues "sponsored" spam; rating systems without verified
attendance — we only score on actual check-ins.

---

## 5. miniapps.py — MiniApp / Poll / Expenses / RSVP + Panels

**Best (polls):** Telegram Polls 2.0 — visible votes, **multiple answers**,
**quiz mode** with one correct answer + confetti + global leaderboards;
Sidekick/DiscordDiceBot (kh/kl/dh/dl keep-drop, aliases, multilingual);
Roll20 QuantumRoll (adv/dis, exploding, reroll rules, presets).

**Best (expenses):** Splitwise product + LLD writeups (mantu-sharma2,
poojaacoder, ashishguleria04/solvarch): strategy-pattern split types
(EQUAL/EXACT/PERCENT with sum validation), greedy two-heap debt simplification
(O(n log n), ≤ n−1 transfers, NP-hard to truly minimize), integer minor units,
deterministic remainder rule, simplification as an OPT-IN group setting (it
rewires counterparties), settle-up as just another ledger event, edit/delete
expenses with balance updates, spending analytics (category-wise, monthly
trends, contribution insights — dnyaneshwar-dnyanu/expense-splitter).

**Best (RSVP apps):** same event-mining as events.py: capacity, waitlist,
guests.

What the best do that we didn't:
- **Quiz mode** (Telegram Polls 2.0): → ADDED new `quiz` mini-app kind:
  question + options + correct answer, per-user attempts, scoreboard,
  streak tracking.
- **Multiple-answer polls** (Telegram Polls 2.0): → ADDED: `multi` flag on
  polls — votes are sets, render shows approval counts.
- **Poll deadline auto-close**: → ADDED `closes_at` on polls; `due_polls()`.
- **Poll comments / discussion**: Doodle has a comments section. → ADDED:
  `comment` action on polls (threaded one level, rendered under results).
- **Split strategies** (Splitwise): → ADDED: `equal | exact <n1,n2..> |
  percent <p1,p2..>` per expense with sum validation (the classic interview
  bug #1: silently vanishing remainders — we validate).
- **Edit/delete expenses** (expense-splitter): → ADDED `edit`/`del` actions;
  balances recompute from the ledger.
- **Categories + monthly summary** (expense-splitter analytics): → ADDED:
  `cat:` tag on expenses, `summary` action → per-category + per-month +
  per-member contribution table.
- **Pairwise vs smart settlement view**: → ADDED: render shows both raw
  pairwise debts and the simplified view (Splitwise does both).
- **Opt-in simplification**: kept default-on but the render marks it clearly;
  `settle` records payments that recompute against balances.
- **RSVP mini-app**: capacity + waitlist + guests (mirrors events.py).
- **Render god-tier**: progress bars with block characters + percentages +
  voters, money with categories, quiz scoreboard with medals.
- **Panels**: already Hark-pattern; ADDED: `due_refreshes()` host helper
  (panels needing refresh by cadence) and `digest()` — one-line-per-panel
  auto-briefing the scheduler can post.

**Trash to avoid:** QuizBot's "cannot delete / limited commands" (we made
quiz fully editable); silentdm's emoji clutter (we use emoji sparingly as
status glyphs); invented RNG claims (we don't claim QuantumRoll-tier).

---

## 6. registry.py — community_registry (the allowlist)

**Best:** Avrae/Sidekick dice syntax: `2d20k1`/`2d20kl1` (keep), `!` exploding,
`kh`/`kl`/`dh`/`dl`, Fate `4dF`, `adv`/`dis`, composite arithmetic,
crit/fumble callouts (d20: 20 = crit, 1 = fumble — from the D&D 5e SRD text).

What the best do that we didn't:
- **Rich dice grammar**: ours parsed `2d6` + one modifier. → ADDED full
  expression language: keep/drop (`kh`/`kl`/`dh`/`dl`), exploding (`!`),
  advantage/disadvantage (`adv`/`dis`), Fate dice (`dF`), composite
  `2d8+1d6` terms, crit/fumble callouts on d20, seeded RNG injection for
  tests, roll descriptions (`2d6 # damage`), limits kept sane.
- **pick / shuffle**: DiscordDiceBot-style channel engagement utils. → ADDED:
  `pick "a,b,c"` (random choice), `shuffle "a,b,c"` (random order) —
  pure utilities, fit `community.fun`.
- **coin** count: → `coin(3)` flips N, shows streaks.
- **Help text per tool**: descriptions now carry usage syntax.

Allowlist unchanged in spirit: still only `community.miniapp` +
`community.fun`; no memory/vault/connectors/exec. Isolation test still
gates this (AST import scan + capability check).

---

## 7. policy.py / __init__.py

No external mining applies — the allowlist is an architectural decision, not
a feature. Verified unchanged and still enforced by
tests/test_community_isolation.py.

---

## Gap list → implemented (the "should have" questions)

| Class | Was missing | Now |
|---|---|---|
| Room | co-hosts, lock, mute-all, invite links, scheduled start, reactions, stats, WebVTT, agenda | ✅ all |
| Group | roles, announcements, pins, search, welcome, join gate, event link, reputation | ✅ all |
| Event | description, ends_at, capacity/waitlist, guests, RSVP window, full recurrence, proxy RSVP, history | ✅ all |
| VenuePoll | approval voting, blind polls, deadline, rich venues, auto-adopt winner, reliability | ✅ all |
| Poll app | quiz mode, multi-answer, deadline, comments | ✅ all |
| Expenses app | split strategies, edit/del, categories, monthly summary, pairwise view | ✅ all |
| RSVP app | capacity, waitlist, guests | ✅ all |
| Registry fun | full dice grammar, pick, shuffle, coin N | ✅ all |
| Panels | due_refreshes, digest | ✅ all |

## Style upgrades

Every render rebuilt as a **card**: header line with status glyph, section
separators, progress bars, key-value pairs aligned, next-action command
hints preserved. Event/room/group `show` views now feel like product UI
copy, not debug dumps.
