# Devon Infrastructure Spec — God-Tier Floor

**Status:** DRAFT — FOR OWNER REVIEW / APPROVAL. **Do NOT build from this spec until the owner approves it.**
**Repo:** `Oluwacutyp/nomorals-2.0` (Devon). GoldenSwing is a separate product and is out of scope except where explicitly noted (§8.5).
**Date:** 2026-10-17.

## 0. How to read this spec

**The spec is the FLOOR, not the ceiling.** The owner's rule: *"You know the rule not our ceiling we always do it God tier style."*
Every section below states the **minimum** the build must meet (the floor) **and** the god-tier target it must aim for. The build must meet AND exceed the floor. Nothing in this spec caps the build. If a section's god-tier target turns out to be reachable with a stronger design, the build takes the stronger design — it never says "the spec only asked for X."

**Standing rules this spec inherits (do not re-litigate):**

- Full capability, end-to-end working code. No stubs, no `TODO` as delivery, no placeholder returns, no fake success, no "wire later."
- No boilerplate. Every line earns its place. No duplicate parallel systems — extend the real organ.
- No moral/ethics theater in product code. Engineering controls (sandbox, capability tokens, audit logs, confirmations) are real mechanisms, not excuses to leave features half-built.
- Dynamic, not hardcoded. No regex/keyword intent shortcuts for capability routing. No hardcoded tone presets ("kind", "sharp", "tough"). Behavior comes from mood, audience, context — never a preset label.
- Real errors, never generic. A failure must name its real cause. A swallowed error is a bug, not a simplification.
- Results over speed. No hypothesis presented as fact. Verify before claiming done.
- Never delete modules/classes/tools. Check registries, dynamic/lazy imports, string loading before calling anything dead.
- Partner (Devon, she/her, `partner/persona.py` + `partner/mood.py`) is never altered without the owner's explicit instruction. The identity in this spec (Devon Personal) is used by the reply path — it does not change Partner's persona files.
- GoldenSwing is NOT part of Devon. Never mix them in script, env, or config.
- Never request tokens/credentials in chat. The owner fills `.env` keys themselves.
- **Exclusion (owner-confirmed, not a decision for the build to revisit):** the "Florida Mom Emergency / Macro-Compliance" financial-extraction sequence that arrived alongside the Devon Personal identity document is a scam script. It is NOT part of Devon Personal's identity and is NOT to be implemented, referenced, or used anywhere in Devon. Devon Personal's identity is *who she is* (the Q&A), not a script to extract money.

**Terms used in this spec:**

- **Floor:** the minimum acceptable behavior. A build that meets only the floor is acceptable; a build that stops at the floor when a stronger design is available is not god-tier.
- **God-tier target:** the behavior the build aims for. This is the real bar.
- **OWNER-DECIDES:** a choice only the owner makes. The spec gives a recommendation and the consequences of each option, never a silent default.
- **MUST / MUST-NOT:** floor requirements. **SHOULD:** god-tier targets.

---

## 1. Goals + principles

### 1.1 Goals

1. **Close the gaps the owner felt:** "there's a lot of gap in the system." The gaps are not one bug — they are a pattern across tool interactions, commands, outputs, style, and group controls (§2). This spec treats the pattern, not just the symptoms.
2. **Make the infrastructure trustworthy:** every tool call, command, and output behaves the same way every time — real errors, honest partial results, visible progress, no silent drops, no silent substitutions.
3. **Make Devon feel like a person, not an assistant:** Devon Personal's identity is live in replies (§3), with human presence from real identity facts, not generic warmth.
4. **Make group/community behavior controllable per chat:** the owner can shape what Devon does in each group, community, and DM — without a global switch that breaks other chats (§7).
5. **Make the system self-describing:** any gap (unregistered command, unmapped config key, unwired handler) must be *detectable by a test or a status command*, not only by the owner tripping over it in a live chat.

### 1.2 Principles (floor)

| # | Principle | What it means in practice |
|---|-----------|---------------------------|
| P1 | **Real errors only** | Every failure surfaces its real cause (provider, status, reason). No bare generics (`"STT API error"`, `"TTS vocal failed ()"`), no swallowed exceptions, no empty reasons. |
| P2 | **Honest partials** | If a result is partial (vocals missing, transcript failed, zone book thin), the output says so. Never imply complete when partial. |
| P3 | **Visible progress, never silence** | Long work reports where it is (Telegram progress for startup, per-stage logs for heavy jobs). Silence longer than the work warrants is a bug. |
| P4 | **One system per job** | One command registry, one chat-mode store, one style system, one STT path, one TTS engine-resolution order. No parallel duplicates. |
| P5 | **Dynamic, not hardcoded** | Style, routing, and behavior come from mined patterns + mood/audience/context. No preset tone labels, no keyword intent shortcuts. |
| P6 | **No silent substitution** | If engine A fails, do not silently fall back to engine B and pretend nothing changed. Either fail honestly or name the substitution. |
| P7 | **No timeout that defeats purpose** | Timeouts are allowed only where cutting work short is safe. Never time out work whose whole value is completion (e.g. zone building, identity loading). Visibility replaces timeouts there. |
| P8 | **Config keys that exist must work** | Every documented `.env` key must map to real behavior. A documented-but-unmapped key is a bug, not a doc issue. |
| P9 | **Every capability reachable** | Every built handler must be reachable via a registered command or a documented natural-language path. An unreachable handler is an orphan, and orphans are bugs. |
| P10 | **Per-chat scoping** | Modes, toggles, style, and state are per chat (chat_key + sender where identity matters). A setting in one chat must never leak into another. |

### 1.3 Non-goals

- No new product (no new bot, no new app, no Arena merge). This is infrastructure inside Devon.
- No GoldenSwing changes, except the shared lesson in §8.5 (startup visibility) which GoldenSwing already applied separately.
- No change to the social-gating matrix (owner / outsider / DM / group) unless the owner explicitly approves it. Group controls in §7 work *inside* the matrix, not by rewriting it.
- No migration of Partner persona files. Devon Personal identity lives in `partner/background.py` (`DEVON_PERSONAL_FACTS`) and is consumed by the reply path — it does not edit `persona.py` / `mood.py`.

---

## 2. Evidence — gaps found (what this spec is answering)

These are real gaps observed on 2026-10-17, not hypotheticals. Each maps to a spec section.

| # | Gap | Symptom the owner saw | Root-cause class | Spec section |
|---|-----|----------------------|------------------|--------------|
| G1 | STT generic error | `"I couldn't transcribe that voice note (STT API error)."` | Error detail swallowed by a shortening helper | §4.3, §5.3 |
| G2 | `.env` keys ignored | `NM_VOICE_STT_PROVIDER=api` never took effect; STT only worked via Groq fallback | Documented key with no `_ENV_MAP` entry | §8.1 |
| G3 | Wrong upload MIME | Voice notes sent as `application/octet-stream` | MIME hardcoded / not derived from format | §4.2 |
| G4 | Wrong upload filename | Groq 400 `unsupported_audio_format` for `tg-<millis>-0.bin` | Filename extension not derived from actual format (magic bytes) | §4.2 |
| G5 | Transcript discarded / re-transcribed | Transcript produced up to 3×, later failure lost an earlier success | No single transcript cache; stages re-did the work | §4.2 |
| G6 | Voice → music-ID hallucination | `"hey, what's the clip? … upload the audio file here?"` | `voice` collapsed to `kind="audio"` + bare `"audio: voice.ogg"` note → model guessed music ID | §4.2, §5.3 |
| G7 | TTS engine divergence | `/tts` (espeak-ng) worked; vocal path (PowerShell System.Speech) produced 0 bytes | Two TTS paths, two engine-resolution orders | §4.2 |
| G8 | Empty vocal error | `"TTS vocal failed ()"` | Bare `EOFError` formatted with empty message | §4.3 |
| G9 | TTS sent as file, not voice note | `tts-….wav` arrived as a document | TTS output sent with file type instead of voice-note type | §6.4 |
| G10 | Games-only silenced voice notes | Voice note → silence in gf chat | Gate inspected post-transcription text only; voice notes transcribe to plain text (no `/`) → dropped | §7.2 |
| G11 | Document-audio not routed | Voice note sent as document never hit STT | STT checks matched `voice`/`audio` only, not document-with-audio-MIME | §4.2 |
| G12 | Vision-flag wiped audio | Vision off stripped ALL media including audio before STT | Media filter was all-or-nothing, not per-modality | §4.2 |
| G13 | Vocal track skipped silently | `"sent audio + score"` implied a full song with no vocals | Output did not distinguish complete vs partial | §6.2 |
| G14 | `/math` orphaned | `/math` did nothing (fell through to brain) | Handler built, never registered in `CONTROL_COMMANDS`, never wired to runtime | §5.1 |
| G15 | `/proposals` read wrong queue | `/proposals` always empty despite proposals existing | Two proposal systems; `/proposals` read only one | §5.1 |
| G16 | Identity gated away | Devon replied like an assistant in personal chats | Identity lived only in relevance-gated background (returned `[]` when gate closed) + capabilities block framed Devon as a service + `strip_robotic` missed fresh task-framing | §3 |
| G17 | False limitation | `"i can only reply in text, not voice notes"` | Model asserted a limitation Devon does not have; no guard stripped it | §3.5, §5.3 |
| G18 | GoldenSwing silent startup | Minutes of blank console, no TG ping, after history warnings | Startup ping fired only after full sequence incl. ~3.5-min 17-playbook replay; zero progress output | §8.4 |
| G19 | No per-group feature toggles | Owner could set games-only or normal per chat, but not "games off, chat on" or "music off in this group" | Chat-mode store has one mode field, no per-feature toggles | §7 |
| G20 | No style mining | Replies used generic assistant phrasing; no mined owner-voice style | No style-mining pipeline; style was prompt-only + a small robotic-strip list | §3 |

**Pattern:** the gaps share four root causes — (a) errors swallowed, (b) handlers/config/identity existing but not wired/mapped/gated-in, (c) two paths for one job with different behavior, (d) silence where progress or an honest message belonged. This spec fixes all four at the infrastructure level, not one bug at a time.

---

## 3. Style system

### 3.1 Goal

Devon replies in a style mined from **real** sources, applied dynamically by mood/audience/context — never a hardcoded preset, never generic warmth, never assistant boilerplate. Style is a system (mine → store → apply → tune), not a prompt paragraph.

### 3.2 Style sources — what may be mined (floor)

| Source | What is mined | Privacy rule (MUST) |
|--------|---------------|---------------------|
| Owner's own outgoing messages (owner DM, owner-originated chats) | Emoji rate + placement, punctuation habits, sentence length, greeting/closing patterns, Pidgin/English mix, slang, how the owner opens/closes, how the owner reacts (laugh, tease, short ack) | Owner's own messages only. This is the primary style source. |
| Devon's own outputs (past replies the owner kept / did not correct) | What the owner accepted: phrasing, length, formatting that survived without correction | Devon outputs only. |
| First-edition repo (`Oluwacutyp/No-morals-ai`) social-chat code | Formatting conventions, reply shapes, button/section patterns that already existed | Code only, never anyone's private messages. |
| Public bot/format conventions (Telegram/WhatsApp bot UX) | Headers, dividers, inline buttons, list shapes, truncation notes, short-reply norms | Public conventions only. |
| **NEVER mined** | Other people's private messages (partner, friends, group members) | MUST-NOT. Another person's private messages are not a style corpus. If the owner wants Devon to mirror how someone talks, the owner supplies examples explicitly — Devon never harvests them. |

God-tier target: the miner also tracks *drift* — when the owner's style shifts (new slang, new emoji habits), the mined profile updates; when Devon produces a reply the owner immediately corrects or rewrites, that reply is recorded as a negative example (what NOT to do).

### 3.3 What style covers (floor)

- **Emoji:** purposeful markers, not decoration. Emoji mark section heads, status (✅/❌/⚠️), and reactions — they never substitute for content. A cap applies (existing `humanize_emoji` / `emoji_cap` behavior is the floor; god-tier: cap adapts to chat — owner DM tolerates more, groups less, formal contexts near-zero).
- **Formatting:** headers (`── n/m /command ──` for stacked replies, section heads for long outputs), dividers, lists (short, scannable), code in fences, truncation with an explicit note (`…truncated, N more` — never silent cut).
- **Length:** short by default. Long only when the content needs it (a report, a dossier, a list the owner asked for). The reply must never pad to look thorough.
- **Voice:** Devon Personal identity facts surface *contextually* (a natural reference when it fits — "my mom in Lake Wales", "trails are mud season right now") — never a bio dump, never a recitation. The identity anchors live in a protected block that is never shrunk away (existing life-anchors block is the floor; god-tier: anchors are selected by relevance to the current conversation, not dumped wholesale).
- **No assistant framing in personal/restricted chats:** no capabilities block, no "gotcha, what do you need?", no "drop it here and I'll take a look", no "as an AI", no task-framed openers. The existing `strip_assistant_framing` guard is the floor; god-tier: the guard is a *layer*, not the fix — the prompt itself must not invite assistant framing in the first place (no service/capability framing in personal-chat prompts).
- **No false limitations:** Devon never asserts a limitation it does not have (e.g. "I can only reply in text"). False-limitation stripping (existing) is the floor; god-tier: the prompt + capability inventory make false limitations unformulable (the model is told what Devon *can* do in that chat, grounded in the real tool registry, not in a generic "I am an AI" line).

### 3.4 How style is mined, stored, applied, tuned (floor + god-tier)

- **Mine (offline, owner-triggered or scheduled):** a miner reads the allowed sources (§3.2), extracts measurable patterns (emoji rate/placement, median sentence length, greeting/closing inventory, punctuation habits, section-header shapes, reaction inventory), and writes a **style profile** — a data file, not code. MUST-NOT: the miner never writes prompt prose; it writes measurements.
- **Store:** style profile lives in a versioned store (same pattern as chat-modes: JSON, atomic write, corrupt → defaults, never crash). Profile is per-scope: owner DM profile, personal-chat profile, group profile (group profile is the *bot's* public style, never mined from members).
- **Apply:** the reply path composes style from (profile + mood + audience + context), in that order. Mood modulates (e.g. tired → shorter; playful → more reaction emoji); audience selects (owner DM vs personal vs group); context selects (a command result uses the command-output format in §6; a personal reply uses the conversational style).
- **Tune:** the owner can inspect the profile (`/style` — see §5.1) and override any element (emoji cap, length budget, greeting inventory). Overrides are stored as deltas, not by editing the miner's output. MUST-NOT: no hidden tuning that the owner cannot see or reverse.
- **God-tier target:** style is *evaluated*, not just applied — a periodic self-check compares recent Devon replies against the profile (echo/parrot check, assistant-framing scan, length-budget check) and reports drift. A reply that violates the profile is repaired before sending when the repair is mechanical (strip assistant framing, cap emoji, clamp length); when the repair would change meaning, the reply is regenerated, not patched.

### 3.5 Acceptance criteria (floor)

- [ ] A style profile exists, mined from allowed sources only, with provenance (which source each pattern came from).
- [ ] Replies in personal chats contain no assistant framing, no capabilities block, no false limitations (test-pinned with the exact screenshot replies from §2: `"gotcha, what do you need?"`, `"drop it here and i'll take a look"`, `"i can only reply in text, not voice notes"` — all stripped / never generated).
- [ ] Emoji use is capped and purposeful (no emoji spam, no emoji substituting for content).
- [ ] Devon Personal identity facts surface contextually (at least one natural reference in a relevant conversation; zero bio dumps).
- [ ] Owner can inspect + override style via a command; overrides persist and are reversible.

---

## 4. Tool interactions

### 4.1 Invocation — how tools are called (floor)

- **One registry, one path.** Every tool is registered once (existing tool registry is the home). No tool is callable from one path and invisible from another. If a tool exists, it is registered; if it is registered, it is reachable.
- **Routing is dynamic.** The brain routes to tools from intent + capability, never from hardcoded regex/keyword shortcuts. (Standing rule — restated because tool routing is where shortcuts creep back in.)
- **Arguments are bound.** A confirmation token authorizes one capability *with those arguments* — never any call of that capability (existing argument-bound tokens). Tool calls that target oversight machinery (policy, trust, logging, registry, gating, persona, vault, secrets) require exact-bound tokens and are never bypassed by override (existing `core/oversight.py` classifier).
- **God-tier target:** the tool loop is *inspectable* — the owner (or a diagnostic command) can see which tools were called, with which arguments, in which order, for any recent reply. Tool-call traces are logged structurally (not as free-text), so a failure can be traced to the exact call that failed.

### 4.2 Media + voice pipeline — one path per job (floor)

This is where most of today's gaps live (G3–G12). The floor is: **one STT path, one TTS engine-resolution order, one transcript cache, one media-kind taxonomy.**

- **Media kind taxonomy (MUST):** `voice` (spoken message — Telegram `voice`, `DocumentAttributeAudio(voice=True)`) is distinct from `audio` (music file), which is distinct from `document` (file attachment). A `voice` message is NEVER a song clip. A document with an audio MIME is audio content, never "a file I don't have." The existing `MediaRef.is_voice` + `kind="voice"` is the floor; god-tier: the taxonomy is enforced at the adapter boundary (every adapter maps its native types to this taxonomy, with a test per adapter), so no downstream stage can re-collapse `voice` into `audio`.
- **STT (one path):**
  - Voice notes → STT, always, in every chat where Devon replies (except games-only, where the gate returns first — §7.2). STT is never skipped because the message "looks like" music, a file, or anything else.
  - **One transcript cache:** the transcript is produced ONCE per message and cached on the message (`message.meta["voice_transcript"]`). Later stages reuse it. Never re-transcribe. Never discard a successful transcript because a later stage failed.
  - **Upload correctness (MUST):** the multipart upload sends (a) the real MIME by format, (b) a filename whose extension matches the *actual* format (magic-byte sniff for misnamed/extensionless files — `.bin` → `.ogg` via `OggS`, etc.), (c) a Groq-supported extension (`.aac` is NOT relabelled — it converts via ffmpeg). A lying extension is a bug.
  - **Provider resolution (MUST):** exactly one engine-resolution order for STT (config `audio.stt_provider` → provider map → fallback only when *no* endpoint is configured at all). Config keys that exist must work (§8.1). No silent Groq-fallback when the owner configured a different provider.
  - **God-tier target:** STT also reports *which* engine transcribed (so the owner can see "Groq whisper-large-v3" vs "local whisper") and the transcript carries a confidence/quality signal when the provider returns one.
- **TTS (one engine-resolution order):**
  - Exactly one order, shared by `/tts` and the vocal/music path (the divergence in G7 must never recur). The order is: espeak-ng/espeak first (when present), then platform fallback (`say` / PowerShell), then API. Both paths call the same `detect()` — never two copies.
  - **Zero-byte is an error, never silent:** if an engine produces 0 bytes, the failure names the engine and the cause ("system TTS (espeak) produced no audio"). A bare `EOFError` / empty exception is never formatted as `()`.
  - **Voice-note sending (MUST):** TTS output for a voice-note reply is sent as a voice note (Telegram `voice_note`, WhatsApp `ptt`, OGG/Opus via `wav_to_voice_ogg`), not as a WAV file. File send is opt-in (`file=on`) or an honest fallback (labelled as a file) when conversion fails. See §6.4.
  - **God-tier target:** TTS carries voice + pace + auto-tag state through the whole path (the VoiceTagger auto-tags by active engine — §3 applies here too), and a failed voice send retries once with the file fallback *labelled*, never silently.

### 4.3 Errors — real cause, always (floor)

- **No swallowed errors.** Every `except` that produces a user-facing message must include the real cause. A bare `except: pass`, a `""` fallback with only a debug log, or a shortening helper that collapses distinct errors into one generic string is a bug (G1, G8).
- **Error shape (MUST):** user-facing errors carry: what failed, the real cause (provider + status + reason, secrets redacted), and what the owner can do (if anything). Example: `I couldn't transcribe that voice note (STT API 400: unsupported_audio_format — upload filename had no supported extension).` — not `STT API error`.
- **Never-empty reasons (MUST):** a failure reason is never `()`, never `""`, never whitespace. If an exception has no message, the formatter uses the exception class name + context. Every formatter that renders an exception goes through one never-empty helper (existing `_exc_str` pattern is the floor; god-tier: the helper is the *only* way exceptions become strings anywhere in the codebase — enforced by a lint/test that greps for `f"(...{exc})"` patterns).
- **Secrets redacted (MUST):** error strings never contain Bearer tokens, API keys, or key values. Redaction happens at the formatting boundary, before the string reaches any log or reply.
- **God-tier target:** errors are *classified* (auth / format / network / provider / config / bug) so the owner sees not just the cause but the class ("auth — check the key" vs "format — the file was misnamed" vs "config — the key is not mapped"). Classification is deterministic (from status/code), never guessed by the model.

### 4.4 Progress — visible, not silent (floor)

- Any operation expected to take > ~2s reports progress: startup (§8.4), long tool runs, media renders, music renders, research/digestion ticks.
- Progress goes to the channel the owner watches (Telegram for bot work, console for CLI), at stage boundaries — not a spinner, not a flood. Per-stage messages (`history fetched`, `zones built`, `playbooks warmed`) are the floor; god-tier: progress includes a count + total where one exists (`warmed 12/17 playbooks`) and an honest "still working" heartbeat if a stage runs long.
- **MUST-NOT:** progress must never fake completion (a progress message means the stage actually finished).

### 4.5 Timeouts + fallbacks (floor)

- **Timeouts only where safe.** A timeout is allowed when cutting work short loses nothing essential (an HTTP probe, a provider call with a retry). A timeout is FORBIDDEN where completion is the whole value (zone building, identity loading, journal replay). Visibility replaces timeouts there (§8.4).
- **Fallbacks are named, never silent (P6).** If engine A fails and engine B is used, the output names the substitution ("transcribed with local whisper — API unavailable: <reason>"). If the substitution would change the result's meaning (e.g. a different voice), the fallback requires the owner's explicit choice or fails honestly — it never silently swaps.
- **God-tier target:** every fallback has a *policy* (when it triggers, what it substitutes, whether it needs confirmation), stored with the tool, not improvised per call.

### 4.6 Acceptance criteria (floor)

- [ ] One STT path, one TTS order, one transcript cache — verified by tests that fail if a second path appears.
- [ ] Voice → STT in all replyable chats; voice never routes to music ID; music ID only on explicit `/sham` / "what song" request.
- [ ] Every user-facing error names the real cause (no bare generics, no `()`).
- [ ] Upload filename + MIME always match actual format (magic-byte sniff test).
- [ ] Long operations report stage progress (startup, music render, research tick).

---

## 5. Commands

### 5.1 Registry — no orphans (floor)

- **One registry.** `CONTROL_COMMANDS` (+ `COMMAND_DETAILS`, `/list`, help groups) in `social/chat/control.py` is the single registry. Every command the bot can run is in it. Every entry in it has a handler. (G14, G15 are the anti-pattern: a handler with no registry entry, a command reading the wrong store.)
- **Orphan audit (MUST, on every build):** a test enumerates all handler functions / command-named entry points and asserts each is reachable via the registry (or explicitly marked library-only with a reason). The `/math` orphan must never recur. Library-only modules (engine subsystems with no command surface) are listed in one place with their reason — never silently unreachable.
- **Stacking (existing, spec it):** one command per line; a message is a stack iff 2+ non-blank lines AND every line parses as a known command; `;` / `&&` are NOT separators; fenced blocks / quotes / inline code never split; `{prev}` expansion; fail-stop; per-command gating pre-validated; games / `quit` / `forget` / `restart` / `deploy` / `update` / `pull` never stackable; max 5 per message. This behavior is the floor — the build must not regress it.
- **God-tier target:** the registry is *self-checking* — a `/doctor` / status command reports registry health (commands without handlers, handlers without registry entries, commands without help, commands without tests) so the owner never discovers an orphan in a live chat.

### 5.2 Help + discoverability (floor)

- **Every command has help.** `COMMAND_DETAILS` (first sentence, ≤256 chars) + `/help <command>` + `/list` (all commands, grouped) + BotFather menu (personal bot, 100-cap — one entry displaced per added entry, owner approves displacements).
- **`/help` is static and instant.** It never needs the LLM. It never falls through to the brain.
- **`/list` is complete.** All registered commands appear (the earlier 48-missing-commands gap must never recur — test-pinned).
- **God-tier target:** help is *contextual* — `/help` in a group shows group-relevant commands first; `/help` in a personal chat shows personal commands; a command's help names its gating (owner-only? group? DM?) and its per-group toggles (§7).

### 5.3 Outputs — consistent per command family (floor)

- Every command family's output follows one format (see §6). A command that returns a result uses the family's format; a command that fails uses the error shape in §4.3.
- **No brain fall-through for commands.** A message that parses as a known command runs the command handler. It never falls through to the brain (the `/help` → "brain's buffering" bug class). If a command handler is missing, the reply is an honest "that command is not wired yet" — never a hallucinated answer.
- **Confirmations (MUST):** consequential acts (send, delete, spend, post, power, deploy, restart, forget) require an explicit confirmation step with the exact payload shown. A confirmation token authorizes one act with those arguments (argument-bound tokens, §4.1). No confirmation is ever auto-approved by stacking, by a previous confirmation, or by an override.
- **God-tier target:** every command has a *dry-run* shape (show what it would do) where that is meaningful (send, post, deploy, delete) — the owner sees the exact effect before confirming.

### 5.4 Gating (floor — matrix unchanged)

- The social-gating matrix (owner / outsider × DM / group / channel; public vs owner-only commands; games public in groups, conversation-only for outsiders in DM) is UNCHANGED by this spec. Group controls in §7 operate inside it.
- `caller_may_run` is the single gating check. No command bypasses it. No command re-implements its own gating (a second gating implementation is a bug).

### 5.5 Acceptance criteria (floor)

- [ ] Orphan audit test passes (every handler reachable or explicitly library-only).
- [ ] `/list` contains every registered command; `/help` works for every command without the LLM.
- [ ] Stacking behavior matches §5.1 (test-pinned).
- [ ] Consequential acts require exact-bound confirmation; no auto-approval path exists.
- [ ] No command falls through to the brain.

---

## 6. Outputs

### 6.1 Formatting rules (floor)

- **Headers:** command results open with a short header (command + scope). Stacked replies use `── n/m /command ──` headers.
- **Emoji:** purposeful markers only (§3.3). Status emoji (✅/❌/⚠️) mark outcomes. Emoji never substitute for content, never pad a reply.
- **Lists:** short, scannable, one item per line. Long lists truncate with an explicit note + count (`…and 12 more — /list all`).
- **Truncation:** any truncated output says so (`…truncated, N more` / `…truncated at N chars`). Silent truncation is a bug.
- **Code / data:** code in fences, data as structured lists (never raw dicts / raw JSON in a chat reply). A raw `{'key': …}` in a reply is a formatting bug.
- **Length:** short by default (§3.3). A long output must earn its length (a report the owner asked for).

### 6.2 Honesty — never imply complete when partial (floor)

- If a result is partial, the output says what is missing and why. Examples (all test-pinned):
  - Music with no vocals: `sent audio + score to this chat. (instrumental only — vocal track skipped: <reason>)` — never a bare "sent audio + score."
  - Transcription failed: the honest failure message (§4.3) — never a hallucinated transcript, never a "what's the clip?" guess.
  - Zone book thin: `zone book thinner than 3 years (D1: 729/2000 bars)` — the warning is the output, not a hidden log.
  - A command that ran in shadow mode says `shadow` — never implies live.
- **God-tier target:** every output carries a *completeness* signal (complete / partial + what's missing / shadow / failed) in a consistent position (footer), so the owner can scan for partials without reading the whole reply.

### 6.3 Length + chunking (floor)

- Replies respect the platform's limits (Telegram 4096 chars, WhatsApp limits). Long replies split at sentence boundaries (existing `split_messages` is the floor), never mid-word, never mid-code-fence. Split parts are numbered (`1/3`, `2/3`).
- God-tier target: splitting preserves *structure* (a list is not split mid-item; a table is not split mid-row) — the splitter understands the output format, not just character count.

### 6.4 Delivery type — correct type per content (floor)

| Content | Delivery type | MUST-NOT |
|---------|--------------|----------|
| Voice-note reply (TTS) | Voice note (Telegram `voice_note`, WhatsApp `ptt`, OGG/Opus) | WAV file attachment presented as a voice note |
| Text reply | Text message | Voice note for a text reply (unless the owner asked for voice) |
| File (score, document, image) | File/document with correct MIME + filename | File with wrong MIME, wrong extension, or a file presented as a voice note |
| Inline actions | Inline buttons (Rematch / Next / Power / etc.) | Dead buttons (every callback must parse); a button over the callback budget is omitted, never truncated |

- **File naming (MUST):** every file sent has a real filename with a correct extension (same rule as §4.2 uploads — extension matches actual format). A file sent as `tg-…-0.bin` is a bug.

### 6.5 Acceptance criteria (floor)

- [ ] No raw dicts/JSON in chat replies (test-pinned).
- [ ] Every partial result is labelled partial with what's missing.
- [ ] Truncation is always noted with a count.
- [ ] TTS replies arrive as voice notes (not WAV files) unless `file=on` / honest fallback.
- [ ] Every file sent has a correct filename + MIME.

---

## 7. Group controls

### 7.1 Goal

The owner can shape what Devon does in *each* group, community, and DM — without a global switch that breaks other chats. Controls are per-chat, owner-only, persisted, and visible. This extends `/chatmode` — it does not create a parallel system.

### 7.2 Modes (existing + extended) (floor)

Existing modes (unchanged semantics):

- `normal` — Devon replies normally (subject to the gating matrix).
- `games_only` — Devon sends NO reply to normal chat messages in that chat, but game commands + in-game moves + owner control commands still work. **Explicit:** in games-only, ALL non-game messages are silenced — including voice notes (even after transcription). This must be stated in help + status + set-confirmation (existing fix). Games-only is a *silence* mode, not a *feature* mode — it does not change what games do.

New modes (this spec):

- `games_off` — Devon replies to normal chat, but games do NOT run in this chat. Game commands in this chat get an honest "games are off in this chat" reply (not silence, not a brain fall-through). Game moves in a live game: the game is paused/ended honestly (see §7.4 for the live-game rule — OWNER-DECIDES, §7.6).
- `quiet` (OWNER-DECIDES — see §7.6) — **Recommendation:** add it. Devon never messages first in this chat (no proactive, no pulses, no scheduled sends into this chat), but replies when addressed. This is the "don't ping this group" control. If the owner rejects it, proactive controls stay global-only.

### 7.3 Per-feature toggles (floor — the owner's ask)

Per chat, the owner can toggle individual features on/off. Feature set (floor — must cover at least these; god-tier: any registered feature is toggleable):

| Toggle | Off means |
|--------|-----------|
| `games` | No games in this chat (same as `games_off` mode — the toggle and the mode are two views of the same setting, never two stores). |
| `music` | No `/music` / `/play` / music renders in this chat. |
| `voice` | No voice-note replies and no TTS in this chat (text only). Voice notes still transcribe (so the conversation continues) — but replies are text. **OWNER-DECIDES:** whether `voice off` also stops transcription (recommendation: no — transcription is understanding, not output; stopping it would make Devon deaf in that chat). |
| `images` | No image generation / vision replies in this chat. |
| `proactive` | No proactive / scheduled / pulse messages into this chat (same as `quiet` mode — toggle and mode are two views, never two stores). |
| `games.<game>` (god-tier) | A specific game (e.g. `math`) off in this chat, other games on. Floor: `games` toggle covers all games; per-game toggles are god-tier, not floor. |

**Rules (MUST):**

- Toggles are per-chat (`chat_key`), stored in ONE store (the chat-modes store, extended — never a second store for toggles).
- Toggles are owner-only. Non-owner attempts get an honest denial, never a silent ignore.
- A toggle OFF never silences the whole chat (that's games-only's job). It disables *that feature* and replies honestly when the feature is invoked ("music is off in this chat").
- Toggles compose with modes: `games_only` + `music off` = games work, music doesn't, normal chat silenced. The composition rule is: mode gates *reply vs silence*; toggles gate *features*. No toggle can re-enable something the mode silences.
- **Community controls:** a "community" (WhatsApp community / Telegram supergroup with topics) is controlled at the community level + per-topic/per-subgroup level (god-tier). Floor: per-chat controls work for the community's main chat; per-topic controls are god-tier.

### 7.4 Live-game rule when games are turned off (OWNER-DECIDES)

**Question:** if a game is live in a group and the owner turns games off mid-game, what happens to the live game?
**Recommendation:** the live game ends honestly — Devon posts one message ("games are now off in this chat — ending the live game, scores kept") and no further moves are accepted. Scores/leaderboard persist. The alternative (let the live game finish) leaves a game running in a chat where games are off, which contradicts the toggle.
**OWNER-DECIDES** — the build must not pick this silently.

### 7.5 Commands + status (floor)

- `/chatmode` (existing, extended):
  - `/chatmode` / `/chatmode status` — shows mode + all toggles for this chat (or the named chat), in plain language, including the games-only voice-note rule.
  - `/chatmode normal | games-only | games-off | quiet` — sets the mode for this chat (or the named `platform:chat_id` for remote set).
  - `/chatmode <feature> on | off` — toggles a feature for this chat (e.g. `/chatmode music off`, `/chatmode games off`, `/chatmode voice on`).
  - `/chatmode list` — lists ALL chats with non-default settings (mode ≠ normal OR any toggle off), so the owner can audit every override in one place.
- All `/chatmode` operations are owner-only. Non-owner gets an honest denial.
- **God-tier target:** `/chatmode` also shows *who* set a setting and *when* (audit trail), and supports `/chatmode reset <chat>` (back to defaults) and `/chatmode copy <from-chat> <to-chat>`.

### 7.6 OWNER-DECIDES (group controls — do not build until the owner answers)

| # | Question | Recommendation | Consequence of the alternative |
|---|----------|---------------|-------------------------------|
| D1 | Add `quiet` mode (no proactive into this chat)? | Yes | Without it, proactive stays global-only; owner cannot silence proactive in one noisy group without silencing it everywhere. |
| D2 | Live-game rule when games turned off mid-game (§7.4)? | End honestly, scores kept | Let-it-finish leaves a game running where games are off — contradicts the toggle. |
| D3 | `voice off`: stop transcription too, or text-replies-only? | Text-replies-only (transcription continues) | Stopping transcription makes Devon deaf in that chat — voice notes get no reply at all, which looks like a bug. |
| D4 | Defaults for new groups: all features ON (owner posture = max capability, opt-out not opt-in) or games OFF until enabled? | All ON (matches the standing "features ON by default" rule) | Games OFF by default would silently disable games in every new group until the owner opts in — against the max-capability posture. |
| D5 | Per-game toggles (`games.math off`) in the floor, or god-tier only? | God-tier only (floor = `games` toggle) | Floor per-game toggles add registry + UI work for a case the owner has not asked for yet. |

### 7.7 Acceptance criteria (floor)

- [ ] `/chatmode games-off` disables games in that chat only (other chats unaffected).
- [ ] `/chatmode music off` disables music in that chat only; invoking music there gets an honest "music is off" reply.
- [ ] `/chatmode status` shows mode + all toggles, including the games-only voice-note rule.
- [ ] `/chatmode list` shows all chats with non-default settings.
- [ ] All `/chatmode` operations are owner-only; non-owner gets an honest denial.
- [ ] One store (chat-modes store extended) — no second toggle store (test-pinned: no second JSON/SQLite store for toggles).
- [ ] Per-chat isolation: a toggle in chat A never affects chat B (test-pinned).

---

## 8. Infrastructure

### 8.1 Config — every documented key must work (floor)

- **The `_ENV_MAP` rule (MUST):** every key documented in the `.env` template (`scripts/devon_ec2_env.template`) MUST have an `_ENV_MAP` entry in `core/config.py` mapping it to real behavior. A documented-but-unmapped key is a bug (G2 — `NM_VOICE_STT_PROVIDER`, `NM_AUDIO_STT_*`, `NM_PARTNER_OWNER_KEY` were all this bug class).
- **Enforcement (MUST):** a test parses the template and asserts every `KEY=value` line has an `_ENV_MAP` entry. The test fails on any unmapped key. This test is the floor — it makes the bug class impossible to reintroduce silently.
- **Config surface (MUST):** `nm doctor` / a config status command reports: which keys are set, which are unset-but-defaulted, which are set-but-ignored (must be zero — an ignored key is a bug), and which have invalid values (with the valid range). Invalid values fail fast at startup with a clear message, never silently default.
- **God-tier target:** config is *typed* — every key has a type, a range/default, and a description in one schema (the template is generated from the schema, not hand-maintained — so template and code cannot drift).

### 8.2 Persistence — atomic, corrupt-safe, per-chat (floor)

- **Atomic writes (MUST):** every state file is written via temp-file + atomic rename (existing chat-modes pattern). No direct overwrite. No partial-write corruption.
- **Corrupt → fresh, never crash (MUST):** a corrupt state file loads as fresh defaults (with a warning log), never crashes the bot. Every store has this behavior, test-pinned with a corrupt-file case.
- **Per-chat scoping (MUST):** chat modes, toggles, game state, style overrides, math state, commitments — all keyed by chat_key (+ sender where identity matters). No global state that one chat's use corrupts for another.
- **God-tier target:** every store has a *schema version* + migration (old files migrate, never silently dropped), and a `verify` command that checks all stores for corruption without modifying them.

### 8.3 Logging — structured, real errors, no silent swallows (floor)

- **Structured logs (MUST):** logs carry (timestamp, level, module, chat_key where relevant, action, outcome, error). Free-text-only logs that cannot be grepped by field are a bug.
- **No silent swallows (MUST):** every `except` logs at the right level with the real cause (or re-raises). A `except: pass` / `except: log.debug` that hides a user-visible failure is a bug. (G1/G8 were this class.)
- **Secrets (MUST):** logs never contain tokens, keys, or key values. Bot-token redaction (`bot<id>:***`) and API-key redaction apply everywhere, including exception bodies.
- **God-tier target:** logs are *queryable* — a diagnostic command can show the last N errors for a chat, with their real causes, without the owner reading a log file.

### 8.4 Startup — ping first, progress always (floor)

- **Ping first (MUST):** on startup, Devon sends a startup ping (Telegram / the owner's DM) BEFORE heavy work (before history fetch, zone building, model load, migration). The owner knows the bot is alive even if the rest is slow.
- **Progress per stage (MUST):** each startup stage reports (console + TG): `starting…` → `config loaded` → `adapters up` → `stores verified` → `ready`. A stage that runs long reports "still working" — never minutes of blank.
- **No silent minutes (MUST):** any startup stage expected to take > ~10s logs progress. A startup that is silent for minutes is a bug, even if it eventually succeeds.
- **Genuine failure (MUST):** any startup exception sends an honest failure message (TG + console) naming the real cause, then raises. A startup that dies silently is a bug.
- **God-tier target:** startup is *resumable* — a stage that already completed (e.g. stores verified) is not redone on a restart; startup time is measured and reported (`started in 42s`).

### 8.5 GoldenSwing lesson (reference only — GoldenSwing is separate)

GoldenSwing already applied the startup-visibility fix separately (ping first, TG progress per stage, no timeout on zone building). This spec cites it as the pattern, not as Devon work. Devon must apply the same pattern to its own startup (§8.4). No GoldenSwing code changes in this spec.

### 8.6 Health — probe/status surfaces real errors (floor)

- **`/status` / `/providers` / probe (MUST):** health commands surface the REAL underlying error (provider, status, reason) — never a generic "probe failed (no calls yet)". The existing probe-detail fix is the floor.
- **`/doctor` (god-tier):** a single command that checks registry health (§5.1), config mapping (§8.1), store integrity (§8.2), adapter connectivity, and provider health — and reports each as pass/fail with the real cause for any fail.
- **God-tier target:** health is *continuous* — a background check (not just on-demand) detects a dead adapter / expired key / corrupt store and alerts the owner, rather than waiting for the owner to trip over it.

### 8.7 Acceptance criteria (floor)

- [ ] Template-vs-`_ENV_MAP` test passes (zero unmapped documented keys).
- [ ] `nm doctor` / config status reports set/unset/ignored/invalid keys (ignored = zero).
- [ ] Every store: atomic write + corrupt → fresh + never crash (test-pinned per store).
- [ ] Startup: ping first, per-stage progress, honest failure on exception.
- [ ] Health commands surface real errors (no generics).
- [ ] Logs: structured, no silent swallows, secrets redacted.

---

## 9. Testing + verification

### 9.1 Test strategy (floor)

- **Failing-first for every gap:** each gap in §2 gets a test that fails on the old behavior and passes on the new. No fix without a failing-first test.
- **Suites (floor — one suite per spec section):**

| Suite | What it proves |
|-------|---------------|
| `test_style_system` | Profile mined from allowed sources only; assistant framing stripped; emoji capped; identity surfaces contextually; no bio dumps; owner override persists + reversible. |
| `test_tool_interactions` | One STT path / one TTS order (a second path fails the test); transcript cached once; upload filename+MIME match format; voice → STT never music ID; errors name real cause; progress reported. |
| `test_commands_registry` | Orphan audit (every handler reachable or explicitly library-only); `/list` complete; `/help` works without LLM; stacking behavior; confirmations exact-bound. |
| `test_outputs` | No raw dicts/JSON; partials labelled; truncation noted; voice-note delivery type; file naming. |
| `test_group_controls` | Per-chat toggles (games/music/voice/proactive) isolated per chat; owner-only; status/list correct; one store; live-game rule per OWNER-DECIDES D2. |
| `test_config_mapping` | Template-vs-`_ENV_MAP` (zero unmapped); invalid values fail fast. |
| `test_persistence` | Atomic write; corrupt → fresh; per-chat isolation; per-store. |
| `test_startup_visibility` | Ping first; per-stage progress; honest failure on exception. |
| `test_health` | Probe/status surface real errors; no generics. |

- **Regression (MUST):** every build runs the existing suites for the touched areas (help_sync, personal_bot, chat_modes, voice routing/STT, presence, tts, games, command_stack, proposals, reachability) — zero failures, or the pre-existing failure is named with its known cause (never hidden).
- **God-tier target:** a single `test_infrastructure` meta-suite runs all of the above + the orphan audit + the config-mapping test in one command, so "is the infrastructure healthy?" has one answer.

### 9.2 Live verification on EC2 (owner actions — after the build lands)

The owner verifies on EC2 (Devon is already deployed there; no re-clone — `git pull origin main` + restart):

1. **Style:** in a personal chat, send a normal message — reply is human (Devon Personal, contextual), no assistant framing, no capabilities block, no false limitation.
2. **Voice note:** send a voice note in a normal chat — it transcribes and replies with a voice note (not a file, not "what's the clip", not a generic error). If STT fails, the reply names the real cause.
3. **`/math`:** `/math` deals a problem; `/math answer` grades it; state persists.
4. **`/chatmode`:** `/chatmode status` shows mode + toggles; `/chatmode music off` in one group disables music there only; `/chatmode list` shows the override.
5. **`/tts`:** `/tts Hello.` sends a voice-note bubble (not a WAV file).
6. **`/music`:** `/music` renders with vocals (not "instrumental only" unless vocals genuinely failed with a real reason).
7. **`/style`:** shows the mined profile; an override persists.
8. **Startup:** on restart, TG shows `starting…` → stages → `ready` — never minutes of blank.

**Verification rule:** the owner confirms each item in a live chat. A build is not "done" until the owner confirms it or the artifact visibly confirms it. Never claim done without evidence.

---

## 10. Rollout

### 10.1 Order (floor — build in this order, each lands separately)

| Phase | What lands | Why first |
|-------|-----------|-----------|
| **P0 — Errors + config** | §4.3 (never-empty real errors) + §8.1 (template-vs-`_ENV_MAP` test + fix all unmapped keys) | Every other phase depends on real errors + working config. Without these, later fixes hide their own failures. |
| **P1 — One path per job** | §4.2 (one STT path, one TTS order, one transcript cache, upload filename+MIME) | Fixes the voice pipeline the owner uses every day. Unblocks voice-note + music-vocal work. |
| **P2 — Commands + outputs** | §5 (orphan audit, registry health) + §6 (output formats, honest partials, delivery types) | Makes every capability reachable + every output trustworthy. |
| **P3 — Style** | §3 (miner, profile, apply, tune, `/style`) | Needs P2 (outputs) to have a format to apply style to. Needs the owner's style-tuning input (§10.2). |
| **P4 — Group controls** | §7 (modes + toggles + status + list) | Needs P2 (commands) + P3 (per-group style). Needs OWNER-DECIDES D1–D5 (§7.6) answered before building. |
| **P5 — Startup + health** | §8.4 (ping first + progress) + §8.6 (`/doctor`) + §9.1 meta-suite | Capstone: proves the whole infrastructure is healthy in one command. |

**MUST:** each phase lands as its own commit + push, with its own tests green, before the next phase starts. No mega-commit that mixes phases.

### 10.2 What needs owner input (before / during the build)

| When | What the owner decides | Spec section |
|------|----------------------|--------------|
| Before P3 | Style tuning: does the mined profile look right? Any element to override (emoji cap, length, greeting inventory)? | §3.4 |
| Before P4 | Group-control decisions D1–D5 (§7.6): `quiet` mode? Live-game rule? `voice off` semantics? New-group defaults? Per-game toggles in floor? | §7.6 |
| During P1 | If a genuinely unsupported audio format appears (not in Groq's list and ffmpeg missing), the build fails honestly — the owner decides whether to install ffmpeg or accept the limitation. | §4.2 |
| After each phase | Owner confirms the live-verification items for that phase (§9.2) before the next phase starts. | §9.2 |

### 10.3 What success looks like

- The owner can use Devon for a day without hitting a swallowed error, an orphaned command, a silent drop, a generic failure, or an assistant-framed reply in a personal chat.
- Every gap in §2 has a test that would catch its return.
- The owner can shape any group/community from `/chatmode` without touching another chat.
- Startup is visible (ping + progress), never silent minutes.
- The infrastructure is self-describing: `/doctor` + `/list` + `/chatmode list` + `/style` tell the owner the system's real state, without reading code or logs.

---

## Appendix A — Real module homes (where the build lands)

Do not invent new homes when a real one exists. Extend these:

| Concern | Home |
|---------|------|
| Style: strip + emoji + length + voice theme | `nomorals/partner/style.py` (`strip_robotic`, `strip_assistant_framing`, `humanize_emoji`, `emoji_cap`, `length_budget`, `clamp_to_budget`, `apply_texting_voice`) |
| Style: chat-level style | `nomorals/social/chat/style.py` |
| Devon Personal identity | `nomorals/partner/background.py` (`DEVON_PERSONAL_FACTS`, `BackgroundPack`, `BackgroundSelector`) — identity facts live here, consumed by the reply path |
| Chat modes + toggles | `nomorals/partner/chat_modes.py` (JSON store, `~/.config/nomorals/chat_modes.json`, `NOMORALS_CHAT_MODES_PATH` override) — extend, never a second store |
| Command registry | `nomorals/social/chat/control.py` (`CONTROL_COMMANDS`, `COMMAND_DETAILS`, `GAME_COMMANDS`, `PUBLIC_GAME_COMMANDS`, `NON_STACKABLE_COMMANDS`, `caller_may_run`) |
| Runtime dispatch | `nomorals/agents/partner/runtime.py` (+ `runtime_games.py`, `runtime_media.py`, `runtime_voice.py`, `runtime_meta.py`, `runtime_reach.py`) |
| Brain / reply path | `nomorals/agents/partner/brain.py` (transcript cache, media notes, prompt composition) |
| Gating | `nomorals/partner/social_gate.py`, `nomorals/partner/gating.py` (matrix unchanged) |
| STT / TTS | `nomorals/tools/audio.py` (`detect_stt`, `_stt_openai_compat`, `tts`, `wav_to_voice_ogg`, `_stt_upload_target`) + `nomorals/voice/stt.py`, `nomorals/voice/tts.py` (one engine-resolution order shared by both paths), `nomorals/voice/pingpong.py` (voice-note transcription with reason) |
| Config | `nomorals/core/config.py` (`_ENV_MAP`) + `scripts/devon_ec2_env.template` (template must match `_ENV_MAP` — test-enforced) |
| Oversight / tokens | `nomorals/core/oversight.py` (argument-bound tokens, oversight classifier) |
| Music / vocal | `nomorals/media/music.py`, `nomorals/media/vocal_lite.py`, `nomorals/media/draft_perform.py` (never-empty vocal errors, honest partials) |
| Tests | `tests/test_<area>.py` — one suite per §9.1 row |

## Appendix B — Owner decisions summary (do not build until answered)

- §7.6 D1–D5 (group controls): `quiet` mode, live-game rule, `voice off` semantics, new-group defaults, per-game toggles in floor.
- §10.2: style-tuning input before P3; per-phase live confirmation before the next phase.

## Appendix C — Change log for this spec

- 2026-10-17: v1 — full spec from owner's infrastructure thoughts + gaps found 2026-10-17. Status: DRAFT, awaiting owner review/approval. No build from this spec until approved.
