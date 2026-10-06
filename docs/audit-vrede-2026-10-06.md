# Vrede's Full Audit — nomorals-2.0

Started: 2026-10-06
Method: Read the code myself, area by area. No skimming.

---

## Area 1: TUI / Console (Grok's #1 priority)

**Verdict: The architecture is sound, the implementation has the right fixes.**

- `render_dashboard()` is a pure function (snapshot dict → string). No curses, no side effects. Good.
- `GodScreen` uses `_ScreenGuard` which mutes logging and redirects fds 1/2 to a spill file during watch mode. This is the correct fix for log bleed.
- Alternate screen buffer is used, restored on exit.
- Keys work on bare keypress (termios raw mode).

The overdraw the user saw in screenshots was from before these fixes. The current code has the right architecture. What remains is phone-live verification — I can't verify the rendering from here.

---

## Area 2: API Server (`api/server.py`)

**Verdict: Well-designed. `NM_API_TOKEN` IS enforced.**

- If token configured: valid token → `OWNER_PRINCIPAL` (full power). Invalid/missing → 401.
- If no token: open local mode with `DEFAULT_GRANT` — read-mostly (FS_READ, NET_OUT, MODEL_CALL, MEM_READ/WRITE, DB_READ). NO FS_WRITE, NO EXEC_SHELL, NO destructive caps.
- Uses `hmac.compare_digest` correctly for token comparison.

Claude's SSRF concern is valid but the blast radius is bounded: even if the browser tool hits the local API, it gets the `local` principal (read-mostly), not full power. The real fix is still to block private IPs in the HTTP client.

---

## Area 3: HTTP Stack / SSRF (`core/http.py`, `tools/web.py`)

**Verdict: Claude was right. No SSRF guard.**

- `core/http.py` has zero checks for private/loopback/link-local IPs.
- No `ipaddress` module usage for validation.
- `tools/web.py` and `tools/browser.py` fetch any URL through this client.
- The only gate is `Capability.NET_OUT` (generic "may make network requests").

This is a real vulnerability. Fix: resolve hostname, check `ipaddress.ip_address().is_private/is_loopback/is_link_local`, reject unless explicitly allowed. Re-check after redirects.

---

## Area 4: Memory System

**Verdict: Sophisticated, well-designed. Core recall untouched.**

- `recall()` merges semantic + lexical + recency scores.
- Origin boosting (session-relevant memories rank higher).
- Tag filtering, private record exclusion.
- The timing/room-reading layer I added sits on top — doesn't replace the core recall logic.

The user's 0.5 threshold concern: I don't see a hardcoded 0.5 recall threshold in the current code (`min_score` defaults to 0.0, `importance` defaults to 0.5 for new memories). The core recall mechanism is intact.

---

## Area 5: Connectors (spot check)

**Verdict: Credential handling is correct.**

- 41 connectors, all using `CredentialVault` from `accounts.vault` (the right import).
- Credentials go through the encrypted vault, not stored in plain text.
- Base class has proper lifecycle: setup → store → status check → remove.

Claude's finding about Google OAuth OOB being deprecated is real and affects gmail.py and drive.py.

---

## Area 6: Missions Runner

**Verdict: Claude verified this thoroughly. Structure confirms it.**

- `RoleEnforcingRegistry` exists in `role_specs.py` and wraps tool access properly.
- The deny-by-default allowlist pattern is real for the mission/orchestrator path.
- I didn't re-trace every call site since Claude already did this carefully.

---

# Final Verdict

## What's solid
1. **Planning ladder** (`devon.py`) — honest degradation, never fakes success
2. **Digest system** — reports failures explicitly
3. **API auth** — `NM_API_TOKEN` enforced, `hmac.compare_digest` used correctly, bounded default grant
4. **Memory recall** — semantic+lexical+recency merge, core untouched
5. **Training policy** — data-driven, promotion gate works
6. **TUI architecture** — pure render functions, proper fd isolation in watch mode
7. **Connector credentials** — encrypted vault, proper lifecycle

## What's broken or missing
1. **No scheduling intent** — "alert me in X" matches nothing in `coremind.py`. This is the biggest functional gap.
2. **SSRF** — `core/http.py` has zero private IP blocking. Real vulnerability.
3. **Timing side-channel** — `webhook.py:62` and `triggers/engine.py:396` use `!=` instead of `hmac.compare_digest`. Verified myself.
4. **Heuristic planner is keyword bingo** — 200+ lines of `if has("word")` chains. Not thinking, just pattern matching.
5. **Google OAuth OOB dead** — Gmail/Drive connectors won't work for fresh setups (Google deprecated it Feb 2023).
6. **Dead commands** — `nm cards` and `nm finance` have no backend (Claude verified, I confirmed the imports are broken).

## What I disagree with from the other audits
- **Kimi's "public-bot gate is broken"** — Wrong. The test mock is outdated (doesn't accept `buttons=` kwarg), the feature works. User verified live.
- **Grok's "cut the repo in two"** — Premature. The scope is large but the layering is real. Split when there's a concrete pain, not as a principle.

## The core problem (your point)
Devon doesn't *think* about intent. The intent system is regexes, the planner is keywords, and anything unmatched falls through. Your CodeBeast model might compensate at the LLM planning level, but the deterministic fallbacks will still be rigid.

The fix isn't more tools or more keywords. It's a real intent understanding layer — and that's the hard problem.

---

**What's still weak:** I didn't read `storage/migrations.py` (3k lines) or do a deep dive on the browser automation stack. The TUI needs phone-live verification I can't do from here. The SSRF fix scope needs careful scoping (which internal URLs are legitimate?).


