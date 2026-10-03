# Agent Loop — Operator Report

**Branch:** `two/agentloop-1.0`
**Base:** `two/main` at `c04a1c9` (ownerdm-1.0 merge)

## What this implements

The formal six-step core agent loop for every owner-DM non-command message:

```
1. CONTEXT PACK → 2. GOAL INFERENCE → 3. PLAN → 4. EXECUTE → 5. VERIFY → 6. REPLY
```

**New module:** `nomorals/agents/agent_loop.py`
- `LoopContext` — dataclass with origin stamp (platform/chat_id/chat_key/thread),
  owner identity/mode, live game, open jobs, pending clarification, resource
  profile (termux/workstation), recent dialogue, media/command flags.
- `build_loop_context()` — cheap, local, never raises. Built once per message.
- `verify_dispatch()` — per-organ acceptance gates. Fake success ("0 passed,
  0 failed / no tests directory — nothing ran", empty "build complete") is
  converted to an honest failure message, never green.
- `run_loop()` — the integrated entry point wiring all six steps.

**Integration:** `nomorals/agents/coremind.py`
- `CoreMind.handle()` now builds the LoopContext after the structural gate
  (step 1) and runs `verify_dispatch` on every dispatch result (step 5).
- New `_dispatch_from_loop()` — steps 2–4 for the formal loop path.
- `_route_log` — in-memory ring of recent decisions for verification.

## Routing table (verified by tests)

| Natural input | Route | Notes |
|---|---|---|
| "write me a story/book about X" | `book` → BookForge | NEVER coding |
| "compose a song / make a beat" | `music` → composer | style auto-detected |
| "play <title>" | `play` → title resolution | NEVER Path(title) |
| "research X / what's the best Y" | `research` | cited summary |
| "create a spotify account" | `account` → AccountCreator | human-in-the-loop |
| "scrape proxies, send here" | `devon` | origin chat stamped |
| "build a todo app" | `build` → coding | only when clearly software |
| "let's play hangman" | `game` | |
| "tell me a story" | `chat` (companion) | NOT the story game |
| "I'm peace, drop the act" | `owner` → owner-mode | persistent, no pushback |
| "how do I link Spotify" | connector info | registry-driven |

## Verification gates (step 5)

- **build/coding:** "0 tests, empty main.py" = FAILURE. Requires real output,
  real tests, or a valid artifact.
- **book:** PDF/chapters produced and deliverable.
- **play:** resolved media path or clear missing-backend message.
- **research:** summary with sources or honest search failure.
- **file send:** delivered or explicit undeliverable reason.

## Telegram smoke script (owner DM, no slash memorization needed)

```
write me a book about courage
→ BookForge starts, PDF arrives in chat

play hotel california
→ resolves via SoundCloud, queues

compose an afrobeats song about Lagos
→ composes with afrobeats style

tell me a story
→ companion tells one (no game menu)

I'm peace, your creator, drop the act
→ owner-mode, cooperative (no pushback)

how do I link Spotify
→ connector tool gives the real OAuth flow

/devon scrape proxies and send to this chat
→ file delivered to this chat
```

Slash commands (`/book`, `/play`, `/game`, etc.) remain full overrides.

## Tests

- `tests/test_agent_loop.py` (new, 20 tests): context pack, verification
  gates, run_loop integration, routing truth.
- All prior suites green: ownerdm, router_account, justworks, image_command,
  coremind_fastpath, layering, error_scan.

## Files changed

- `nomorals/agents/agent_loop.py` (new) — LoopContext, build, verify, run_loop
- `nomorals/agents/coremind.py` — context pack in handle(), verification
  after dispatch, _dispatch_from_loop, _route_log
- `tests/test_agent_loop.py` (new) — 20 tests

## What still needs a stronger model vs pure code

- **Pure code (done here):** routing, verification, context pack, origin
  stamping, fake-success rejection.
- **Needs stronger model:** nuanced multi-intent disambiguation ("play"
  the game vs "play" music when both are plausible), creative quality of
  companion replies, research synthesis depth. The cloud chain
  (HF → Groq → OpenRouter) or the owner's own model handles this; the
  loop guarantees the right organ gets the call.

## Dynamic behavior (owner's principle: "dynamic where it matters")

Hardcoded fixed values now adapt to context:

| Value | Before | Now |
|---|---|---|
| Dialogue history depth | fixed 4 turns, 160 chars | 2–8 turns, 120–200 chars based on input length + ambiguity |
| Dispatch retries | fixed 2 attempts, 1.0s backoff | read-only: 3 attempts; write: 2 attempts; exponential backoff |
| Confidence "strong" bar | fixed 0.8 | 0.75 (sparse) → 0.85 (crowded field) |
| Model-check band | fixed 0.5–0.8 | widens to 0.4 when top two candidates are close |
