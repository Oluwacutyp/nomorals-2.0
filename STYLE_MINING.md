# STYLE MINING — mined, dynamic style system (2026-10-17)

Owner prompt: *"Aren't there styles emoji use and all that kind of
things that we can mine and make use of?"*

Rule followed throughout: mine from REAL sources, never invent. Where a
source was unreachable, that is recorded as a gap — no voice was
fabricated to fill it.

## Sources actually mined

### 1. Devon's OWN output code (primary source — counted, not guessed)

A counting pass over `nomorals/social/chat/` + `nomorals/partner/`
(Oct 2026) tallied every emoji literal in output code, skipping regex
character classes. Top results:

| Emoji | Uses | Where |
|---|---|---|
| 🎮 | 19 | `action_footers.py`, `control.py`, `personal_commands.py`, `style.py` — games |
| ⚠️ | 15 | `agent.py`, `audio.py`, `control.py`, `doctor.py` — warnings |
| ✅ | 11 | `action_footers.py`, `autonomy.py`, `doctor.py`, `schedule.py` — done/confirm |
| 📊 | 11 | `action_footers.py`, `tgbot_buttons.py` — stats |
| 📋 | 11 | `action_footers.py`, `personal_commands.py`, `tgbot_buttons.py` — lists/tasks |
| ❌ | 8 | `autonomy.py`, `doctor.py`, `schedule.py`, `tgbot_buttons.py` — failed/cancel |
| 🎛️ | 7 | `audio.py`, `control.py`, `personal_commands.py`, `video.py` — controls |
| 🏟️ | 5 | `control.py`, `tgbot_buttons.py` — arena |
| 🎙️ | 5 | `control.py`, `personal_commands.py`, `voice.py` — voice |
| ✨ | 4 | `control.py`, `style.py` — headers |
| 🎵 | 4 | `audio.py`, `tgbot_buttons.py` — music/track header `🎵 "Title"` |
| 🔍 | 4 | `action_footers.py`, `control.py`, `personal_commands.py` — search |
| 🥷 | 4 | `control.py`, `telegram_menu.py` — identity mark |
| 📣 | 4 | `control.py`, `whatsapp_communities.py` — announcements |
| 🎬 | 3 | `control.py`, `personal_commands.py`, `video.py` — video |
| 💡 | 2 | `control.py`, `style.py` — tip (card footer) |
| 📖 | 2 | `control.py` — reading/books |
| 🏆 | 2 | `action_footers.py`, `tgbot_buttons.py` — results |
| 📌 | 2 | `control.py` — pinned |
| 🎯 | 1 | `style.py` — next-action CTA (`cta()`) |

Structural patterns mined from the same code (`social/chat/style.py`):

- Emoji section header: `🎮 ARENA` (`section()`).
- Commands in monospace: `` `/game arena` `` (`cmd()`).
- Explicit next-action CTA: `🎯 try /game arena to start` (`cta()`).
- Progress bar: `██████░░░░ 60%` (`bar()`).
- Divider: `━━━━━━━━━━━━━━━` (15 chars, `header()` / `menu_divider()`).
- One job per screen; first line states the point (`header()`).
- Themes: `rich` (emoji+box+markup), `minimal`, `terminal`, `plain`.

Button-label conventions (`tgbot_buttons.py`): `❌ Cancel` / `✅ Confirm`,
`🎮 Join Game`, `📊 Stats`, `🚪 Leave`, `❓ Help`, `📋 Commands`,
`🎮 Start a game`, `📊 My game stats`. Emoji always LEADS the label and
marks the action kind — never trailing decoration.

### 2. Platform facts (documented, not invented)

- `nomorals/social/chat/platform_limits.py` (single source of truth):
  Telegram / Telegram-bot / Telegram-personal `max_text=4096`,
  WhatsApp (Baileys bridge) `max_text=4000`, Discord `max_text=2000`,
  SMS `max_text=160`, web/local `8000`. WhatsApp: no buttons, no HTML.
  SMS: plain text, per-segment cost.
- Telegram Bot API formatting docs (core.telegram.org/bots/api):
  bold, italic, underline, strikethrough, spoiler, block quotations,
  inline links, pre-formatted code; HTML or Markdown parse modes;
  inline keyboards via `reply_markup`.
- `social/chat/platforms.py`: Telegram = HTML (`<b>`, `<code>`,
  `<i>`); WhatsApp = `*bold*`, `_italic_`, `` ```mono``` ``, `~strike~`,
  links flattened to `text: url`; SMS = all markup stripped; web = light
  markdown; Discord = markdown passthrough.

### 3. Audience rules (Devon's own gating)

`nomorals/partner/gating.py` (documented per-audience modes):

- Owner DM / console: full persona, full context.
- Private chat (DM, not owner): "friendly, easy, a little guarded",
  "keep it light and short", no private details, not an assistant.
- Group / channel: "short, casual lines", "no paragraphs, no lists,
  no explanations, no sign-offs", "one voice in the room".

### 4. The chat's OWN observed messages (dynamic input)

`measure_style()` in `nomorals/partner/style.py` measures median
length, emoji/msg, caps ratio, question / exclamation / list rates and
lowercase ratio from the actual messages of the chat being replied in.
With ≥ 5 samples the profile bends toward that chat; below that it
falls back to audience + platform defaults. The responder passes the
history's user turns as the observed sample.

## Sources NOT mined (honest gaps)

- **Owner voice: NOT mined.** No real owner messages exist anywhere in
  this repo — fixtures and tests are synthetic, and no chat-history
  store ships with the checkout. No owner greeting/closing patterns,
  Pidgin/English mix ratios, or owner emoji habits are claimed. The
  system therefore never impersonates an "owner voice"; it follows the
  chat's observed messages at runtime instead. When real owner history
  becomes available, `measure_style()` is the intake — no code change
  needed, just feed it the sample.
- **First-edition repo: NOT mined.** `Oluwacutyp/No-morals-ai` is
  unreachable read-only: GitHub API returns 404 unauthenticated and
  `git ls-remote` prompts for credentials. No first-edition style /
  formatting / emoji conventions were copied. The current repo's own
  code (source 1) is the descendant that *was* reachable.
- **WhatsApp "bot conventions" beyond formatting:** only the
  documented formatting + Devon's own `whatsapp_render.py` /
  `platforms.py` conversions were used — no invented WhatsApp style
  guide.

## What was built

- `nomorals/partner/style.py` (extended, no parallel system):
  `MINED_SECTION_MARKERS` (17 kind → marker pairs from source 1),
  `StyleMeasurement` + `measure_style()`, `StyleProfile`,
  `style_for_chat()` (audience × platform × mood × observed ×
  content_kind → numbers, never a tone name), `format_reply()`
  (dividers, list flattening, section rules, emoji budget, truncation
  that drops whole trailing sections first), `purposeful_emoji()`,
  `platform_hard_cap()`, `style_prompt_lines()`.
- **Removed:** the `VOICE_THEMES` hardcoded tone-preset table
  (`dry` / `soft` / `feral` / `poetic` / `sharp`). `VOICE_THEMES` is now
  an empty compat dict; `apply_voice_theme()` is a passthrough.
  `nomorals/partner/persona.py` and `mood.py` untouched.
- Wiring: `nomorals/partner/responder.py` computes the profile per
  reply (soft layer = prompt lines, hard layer = `format_reply()` on
  the final draft before `split_messages`). `partner/__init__.py`
  exports the new API.
- Tests: `tests/test_partner_style_system.py` (no presets, measured
  profiles, per-chat differences, emoji budget, truncation, responder
  wiring); `VoiceThemesTest` in `tests/test_partner_sweep.py` updated
  to assert the presets are gone.

## What's still weak

- Bot command outputs (menus, reports) still format via
  `social/chat/style.py` helpers directly; they were the mining
  source but are not yet routed through `style_for_chat()` — the
  responder (partner reply path) is wired, the command-output path is
  not. Follow-up: apply `format_reply()` in the gateway send path.
- No real owner-voice sample exists yet, so the observed-history
  input has only been exercised with synthetic fixtures. First run
  against a real chat history should be watched for target-length
  surprises (median-chasing can over-shorten a chat with one terse
  regular).
- `humanize_emoji()` (pre-existing) still only halves repeated emoji
  runs; `purposeful_emoji()` fixes this on the new path, but the old
  helper is still called earlier in the responder pipeline.
- First-edition conventions remain unmined (repo unreachable); if
  access opens, re-run the counting pass against it and diff the
  marker table.
