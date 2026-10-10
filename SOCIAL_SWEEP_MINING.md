# SOCIAL_SWEEP_MINING.md — social module (35 files) external mining

Mined 2026-10-10. Every significant class below is compared against the best
(and worst) implementations found outside the repo. Gold → merged in this sweep.

## 1. Publishing spine — `base.py` / `manager.py` / `adapters/*`

**How the best do it (Postiz, Buffer-era architecture, volanea SaaS guide, PostPulse):**
- Postiz (open source, the repo's own backend): per-platform capability model —
  every adapter declares its own capability set, media validator, token checks,
  quota bucket, and status mapping. The shared job record unifies the workflow
  without hiding network differences. One publisher per platform is the rule
  ("exactly one publisher per platform" — two systems that can both post to
  Instagram will eventually both post to Instagram).
- volanea's 2026 SaaS approval guide: idempotency keys for every external
  publishing attempt (worst outcome is a DUPLICATE after timeout/retry), durable
  job state not in-memory scheduling, correlation IDs across action→job→request,
  structured error codes (auth, validation, policy, media, quota, transient,
  unknown), backoff that never retries permanent auth/policy failures
  indefinitely, token health treated as a production dependency (daily token
  checks + early reconnect prompt, because connections fail quietly after
  password changes / revoked grants).
- Instagram documents a 24h publishing cap → rolling-window accounting, not a
  midnight counter. TikTok direct-post limit ~15/day per creator, shared across
  API clients.

**Gaps in ours → merged:**
- `PostResult` had free-text `error` only → add structured `error_code`
  (auth/validation/policy/media/quota/transient/unknown) + `retryable` property.
- No retry path: `SocialManager.retry_failed()` with per-error-code retry rules
  (transient/quota yes; auth/policy/validation no — never retry permanent).
- No dry-run: `SocialManager.preview()` shows per-platform adapted text +
  char counts without posting.
- No token health sweep: `SocialManager.account_health()` runs each adapter's
  `health()` and reports dead tokens before a scheduled campaign dies quietly.

## 2. Bluesky adapter — `adapters/bluesky.py`

**How the best do it:**
- atproto.dart's `bluesky_text` (best-in-class client): `split()` is
  token-aware — never cuts a handle, link, or tag in half — and every chunk
  respects BOTH the 300-grapheme and 3000-byte limits. Facets are measured on
  the formatted text, not the raw input. Split BEFORE formatting.
- fedecho (Python, 2026-09): hashtag facets built from the official client's
  TAG_REGEX rules (text start or whitespace before `#`, trailing punctuation
  stripped, ≥1 non-digit char, 64-char cap, UTF-8 byte offsets; facets never
  overlap link facets; a tag clipped by truncation is dropped, not corrupted).
- Official `@atproto/api` docs: rich text uses `app.bsky.richtext` facets with
  UTF-8 byte offsets (JS is UTF-16 — byte math is the whole game); `RichText`
  handles link + mention detection; `langs` array sets post language.
- dev.to news-bot pattern: Bluesky posts pair rich-text + an `app.bsky.embed.external`
  link card; `TextBuilder` writes title + clickable domain facet.

**Gaps → merged:** ours had `supports_thread = False`, no facets (hashtags
posted as plain text), no grapheme/byte-aware splitting. Merged:
`detect_facets()` (links, @mentions, #tags with byte offsets), thread posting
via `post_thread()` (reply refs chaining), token-aware `split_bluesky_text()`
respecting 300 graphemes / 3000 bytes, link-card embed helper,
`supports_thread = True`.

## 3. Virality scoring + voice — `voice.py`

**How the best do it:**
- promptslove's 12,045-Threads-posts scrape (2026, real data): declarative
  reveal ("This is…", "Here's…") 1.46 pooled lift; colon set-up 1.29; contrarian
  opener 1.34; announcement 1.33; number-led ~0.97; **question openers 0.45
  pooled** (looks like a disaster at scale — our scorer *rewarded* questions);
  imperative commands 0.71; first-person story 0.63. Name the tool/number in
  the first six words; first line under ~80 chars.
- rondoflow hook playbook: 5 archetypes — curiosity gap, question hook, story
  hook, pattern interrupt, list promise — each with fill-in formulas.
- viral-content-psychology: pattern interruption = prediction error → dopamine;
  textual interrupts (start with a number when others start with words; one
  short sentence when others write paragraphs; bold claim against consensus).
- LinkedIn research (jeffersonbastos): hook is 210 chars on mobile before
  "see more" — if lines 1-2 don't create expansion, the rest is invisible;
  avoid throat-clearing; specific numbers + named entities beat vague claims;
  close on compressed thesis, not "what do you think? comment below!".

**Gaps → merged:** ours scored question hooks +5 (data says they're weak at
scale), had no archetype model, no first-line-length signal, no contrarian/
reveal/curiosity-gap detection, no suggestion of *better* hooks. Merged:
`hook_type()` classifier (reveal/contrarian/curiosity/story/list/question/
pattern_interrupt/none), re-weighted scores per the Threads data (question
opener penalty at scale, reveal/colon/contrarian bonuses, ≤80-char first-line
bonus), `suggest_hook_upgrades()` returning 2-3 rule-based first-line rewrites
in detected archetypes, kept honest as heuristics.

## 4. Tone adaptation — `tone.py`

**How the best do it:** one anchor idea → native platform layouts (sparkum
workflow): X thread ≠ LinkedIn post ≠ Instagram caption; hook-before-fold on
LinkedIn (210 chars), hook-first TikTok captions, lowercase-friendly Threads.
PostPulse: each adapter needs its own capability model and media validator.

**Gaps → merged:** `PlatformProfile` had no hook-fold length, no CTA style, no
thread-splitting. Merged: `hook_len` per platform (LinkedIn 210, X 140 before
fold…), `split_thread()` grapheme-aware thread splitter for X/Bluesky/Threads,
`hook_check()` warns when the payoff sits below the fold.

## 5. Lead magnets / comment→DM — `leads.py`

**How the best do it (ManyChat playbook — the gold standard):**
- Comment-to-DM is the highest-converting trigger: keyword comment → DM with
  the link + PUBLIC comment reply ("Sent, check your DMs") — the public reply
  both boosts the algorithm and fixes DMs that never arrive.
- Keyword rules: one short specific word per post ("PLAN", "VAULT"), different
  per post so attribution works; avoid generic words people type anyway
  ("yes", "info", "link"); emoji keywords read as bait.
- Flow shape: delivery message → exactly one qualifying question ("Are you
  training right now, or getting back into it?") → intent answers route to a
  lead form. Anything longer is a survey and people leave.
- Conversation starters: up to 4 tappable prompts on first DM open — self-serve.
- utm_medium=dm tagging on the delivered link so gate traffic is attributable.

**Gaps → merged:** ours had triggers + DM templates but no public-reply step,
no keyword quality check, no qualifying question, no per-trigger conversion
funnel. Merged: `public_reply` per trigger, `keyword_quality()` (generic-word
blocklist + emoji warning), `set_qualifier()` one-question step, `funnel_stats()`
(comments → DMs → leads per trigger), utm guidance in template rendering.

## 6. Content pipeline / scheduling — `content_pipeline.py`

**How the best do it:**
- Buffer 52M posts / Sprout Social 2B engagements (2026): best windows are
  per-platform and per-niche — X/Twitter 8-11am weekdays (morning platform),
  LinkedIn Tue-Thu 7-10am, Instagram Wed 12pm/6pm, TikTok weekend mornings +
  6-11pm. Sprout: X engagement decays fastest — most lifetime reach in the
  first 4 hours; threads/replies outlast standalone posts.
- Content mix frameworks: 5:3:2 (5 curated / 3 original non-sales / 2
  personal-humanizing per 10 posts); Ava Morgan's mix: 40% educational, 25%
  proof/trust, 20% engagement, 15% promotional — the pillar matters more than
  the calendar.

**Gaps → merged:** ours computed windows from data but had no content-mix
model — a week's batch could be 7 promo posts. Merged: `CONTENT_MIX` pillars
(educate/proof/engage/promo), heuristic `classify_pillar()`, `mix_report()`
showing pillar balance of a batch + which pillar is starved; windows already
data-driven (kept).

## 7. Notification triage — `triage.py`

**How the best do it (Dex personal CRM + CRM retention research):**
- Dex: keep-in-touch cadences per group (past clients 90-day, referral
  partners 30-day), morning "who is due" surfacing, dated reminders that repeat.
- Retention research: track response time as a metric; close the loop after
  every resolved issue; automate follow-ups instead of memory.

**Gaps → merged:** the module docstring promised "unanswered important messages
escalate; noise decays" but NOTHING implemented it. Merged: `TriageLog.escalate()`
(important+ unanswered past N hours → escalated list), quiet-hours support
(`in_quiet_hours()` — critical still buzzes, important waits for morning),
`render_digest()` styled triage digest.

## 8. Relationships — `relationships.py`

**How the best do it (Dex + personal-CRM research):**
- Cadence tiers by relationship strength: close contacts ~monthly, mentors
  1-3 months, former colleagues 3-6 months, weak connections 6-12 months.
- Best notes capture: not "met at conference" but context-rich facts for the
  NEXT conversation. Always create a next step. Give before you ask.

**Gaps → merged:** `needs_reconnect(days=30)` used one flat threshold for all
contacts and produced no outreach prompt. Merged: closeness-derived target
cadence (≥0.7 → 14d, ≥0.4 → 30d, else 90d), `due_for_reconnect()` honoring per-
contact cadence, `reconnect_prompt(rel)` — a brain-ready line with name, days
quiet, last topic, and warmth framing.

## 9. Message style/presentation — `chat/style.py`, `chat/platforms.py`, `render.py`

**How the best do it:**
- spacexm406/telegram-bot-ui (Bot API 10.2 verified): one job per screen;
  first line states the point; formatting is emphasis not decoration; bold the
  one thing that matters; errors = what happened + what to do next; re-check
  escaping and length limits after every edit; callback_data namespaced
  `screen:action:id`, ids not labels (existing payloads in users' chats must
  keep working).
- aura's UX research: in-place status pane with 🟡/✅/❌ + cost footer;
  markdown tables → PNG; HTML parse mode (escape surface bounded to <>& vs
  MarkdownV2's 15+ characters); callback_data ≤64 bytes → embed the option
  INDEX, resolve server-side.
- wirken `TelegramFormatter`: HTML is the dialect choice; tables flatten to
  `Header: value` lines; replies use `reply_parameters`.
- hermes: button-first rule — never make mobile users type what they can tap.

**Gaps → merged:** style.py had primitives but no themes, no cards, no tables,
no sparklines, no emphasis hierarchy. Merged into style.py: `THEMES`
(default/rich/minimal/terminal) controlling emoji density, `card()` structured
field cards, `table()` aligned monospace tables, `sparkline()` unicode trend
line, `quote()`, `kv()` aligned key-value blocks, `stat_line()`; platforms.py:
`escape_markdown_v2()`, `link()`, `code_block()`, `mention()` helpers;
render.py: `render_metrics_card()` engagement report (with sparkline).

## 10. Telegram inline buttons — `chat/tgbot_buttons.py`

**How the best do it:** labels are verb+object, unique within a keyboard;
1-2 buttons per row for sentence labels, up to 3 for short; destructive
actions last and isolated; navigation row last, identical on every screen;
callback_data is 1-64 bytes, ASCII, ids not labels.

**Gaps → merged:** no generic builders — every keyboard hand-rolled. Merged:
`paginate()` (index-embedded callbacks, prev/next + page indicator),
`confirm_keyboard()` (destructive-last confirm/cancel), `nav_row()`.

## 11. SMS — `chat/sms.py`

**How the best do it (fonoster/qcobro's real implementation):**
- Segmentation is not a character count: one non-GSM-7 char forces the whole
  message to UCS-2 and cuts the per-part budget from 160→70 (concatenated
  153→67). Real rule: GSM-7 = 7 bits/char, 1120-bit segments, 48-bit
  reassembly header when concatenated.
- The trap: curly quotes/em-dashes pasted from word processors triple cost;
  é è ñ are FREE (in GSM-7), but á í ó ú and lowercase ç are absent — so
  "García" doubles the price. Hand-written substitution table (never NFD
  de-accent: ñ→n turns "año" into a vulgarity). Emoji are never "normalizable".

**Gaps → merged:** `split_sms()` assumed 160/153 always — a single emoji made
every segment wrong. Merged: `sms_encoding()` (GSM-7/UCS-2 detection),
`sms_segments()` (real segment math 160/153/70/67), `normalize_gsm7()`
(surgical substitution table), split on segment budget not char budget.

## 12. WhatsApp cost — `whatsapp_cost.py`

**How the best do it (2026 pricing reality):**
- Meta moved India to per-MESSAGE billing (July 2025); conversation-based
  elsewhere. Four categories: marketing / utility / authentication / service.
  Nigeria 2026: marketing ~$0.058, utility ~$0.016, authentication ~$0.020,
  service ~$0.010; 1,000 free service conversations/number/month. Marketing
  needs explicit opt-in + pre-approved templates. Biggest hidden cost is
  operational inefficiency, not the rate card — maximize each opened window.

**Gaps → merged:** hardcoded kobo rates with no rate-card provenance and no
forward-looking projection. Merged: 2026 rate-card reference
(`RATE_CARD_2026` with USD rates + source note), `project_monthly()` spend
projection, `rate_card_text()` owner-facing explanation; existing tracking
untouched.

## 13. Draft review UX — `drafts.py`

**How the best do it:** morning-review digest pattern (one message: "3 drafts
need review" with per-draft virality + hook type); variant tracking (A/B the
first line); styled review cards, not raw previews.

**Gaps → merged:** `propose_post()` returned plain text + buttons; no batch
review, no variant history. Merged: `review_card()` (styled, virality + hook
type + pillar), `render_review_batch()` morning digest, `add_variant()` (A/B
first lines in metadata — no schema change).

## 14. Identity — `identity.py`

**Gap → merged:** had `unlink()` (split a wrong merge) but no `merge()` —
the natural counterpart for owner-confirmed merges. Merged: `merge()` with
audit trail, `all_platforms()` helper.

## 15. Profiles — `profiles.py`

**Gap → merged:** no completeness signal — onboarding stalls silently.
Merged: `completeness()` (% score: prompts answered, voice intro, elements,
bio) + `missing_for_complete()` checklist for the onboarding brain.

---
Sources: Postiz/post-pulse architecture notes, volanea.com SaaS API guide,
atproto.dart bluesky_text docs, fedecho hashtag-facet commit, @atproto/api
richtext docs, bsky-docs creating-a-post tutorial, promptslove 12k Threads
scrape, rondoflow viral-tweets SKILL.md, viral-content-psychology repo,
jeffersonbastos LinkedIn research, ManyChat comment-automation guides,
coachway DM-automation playbook, futureproofmarketer comment-gate tutorial,
Buffer/Sprout 2026 posting-time research, clawic + spacexm406 + aura + wirken
+ hermes Telegram UX research, fonoster/qcobro SMS segmentation commit,
afrotools/aurorainbox/udeskglobal WhatsApp 2026 pricing guides, Dex personal
CRM, ixactcontact follow-up framework, azbigmedia 5:3:2, Ava Morgan content
mix.
