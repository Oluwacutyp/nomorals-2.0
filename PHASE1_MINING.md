# Phase 1 Mining Report — Brain/Tool Loop (Spine)

## Architecture Map

```
User message
  → nomorals/agents/partner/runtime.py (routing, game checks, money hooks)
  → nomorals/agents/partner/brain.py (partner brain adapter)
  → nomorals/agents/partner/tool_loop.py (ToolCallingLoop.run)
      → nomorals/llm/brain.py (Brain.chat — one front door)
          → nomorals/llm/router.py (LLMRouter._dispatch — failover chain)
              → providers/* (groq, openrouter, hf_serverless, llama_cpp, ...)
      → nomorals/tools/registry.py (ToolRegistry — dispatch + capability gate)
      → nomorals/partner/social_gate.py (social-specific first-layer gate)
      → nomorals/core/policy.py (Capability enum, CapabilitySet)
```

## What's Solid (GOLD)

### Brain (`nomorals/llm/brain.py`, 738 lines)
- Never raises: worst case is LLMResponse with error set + plain-language diagnosis
- Per-provider circuit breakers (closed → open → half-open) with exponential backoff
- Timeout-bounded calls on daemon threads — stalled provider chains can't hang callers
- Context fitting before send (proactive, cheap, local)
- Typed failure classification with fix hints
- Duck-typed router support — legacy routers degrade gracefully

### Router (`nomorals/llm/router.py`, 937 lines)
- Failover chain with health tracking per provider
- ModelBroker for task-aware selection (vision → VL, code → code-tuned)
- Owner's own models preferred

### Tool loop (`nomorals/agents/partner/tool_loop.py`, 527 lines)
- Text-based tool parsing (```tool fenced JSON + bare JSON fallback)
- Never raises — tool failures become text observations
- Social gate: code-enforced (not prompt-only), group-admin role resolution
- Near-miss tool name suggestions via difflib
- Budgets: max iterations, per-tool timeout, total timeout
- Honest degradation notes

### Registry (`nomorals/tools/registry.py`, 884 lines)
- ToolSpec with schema, capability requirements, aliases, deprecation
- Capability-filtered prompt listings
- Custom tool loading, failed-module tracking

## GAPS (to fix in implementation)

### 1. No native function calling anywhere
- Every provider (groq, openrouter, anthropic, gemini, deepseek, openai_compat) lacks `tools` parameter support
- The entire system depends on text-based ```tool block parsing
- Fragile: depends on model following format; wastes tokens on format instructions
- **Fix:** hybrid — native function calling when provider supports it, text fallback otherwise

### 2. Full tool listing in every prompt
- 60+ tools × description + schema = massive context cost per turn
- Capability filtering exists but still large
- **Fix:** dynamic tool ranking/selection — only include tools relevant to the intent

### 3. Intent routing still has hardcoded paths
- `ControlCommand` parsing in runtime.py (kind='play', etc.)
- Game command checks, money-send hooks before the brain sees the message
- **Fix:** brain-first routing with hardcoded paths as fallback only

[Children's findings to be merged below]

---

## Child report 1: Agent/Tool Loop Path

### Critical finding: TWO PARALLEL BRAINS (architectural debt)

**Path A — CoreMind organ dispatch** (`coremind.py`, `agent_loop.py`):
- Handles EVERY owner-DM message FIRST (`runtime.py:988-1012`)
- Regex-dominant: ~17 hardcoded `_RE_*` detectors + 50-entry `GAME_ALIASES` table
- `_llm_intent_interpret` only classifies 5 of ~26 kinds — everything else is regex-only
- Claims "LLM-first" in header but is regex-first in practice

**Path B — Tool-calling loop** (`tool_loop.py`):
- Genuinely dynamic — docstring "NO regex/keyword intent routing anywhere" verified TRUE
- Only reached when CoreMind returns None (kind == chat)
- Can only reach registry tools, not organs

**Verdict:** violates the standing rule (2026-10-09: "NO hardcoded regex/keyword intent shortcuts"). The two brains are inconsistent — organs reachable only via regex, dynamic loop only via tools.

### Silent failures found
1. Async jobs lie: `_send_async` marks done even when organ returns "❌..." text
2. `verify_dispatch` only guards "build" kind — 25 other kinds unverified
3. Tool-loop truncation invisible: dropped calls never mentioned to model
4. Bare-JSON regex breaks on nested args (`.*?}` matches first `}`)
5. No retry/circuit-breaker in tool loop — model can call failing tool 40×
6. `parse_control` unknown slash → silent chat, no "did you mean" hint
7. Mind crashes fall through silently (`runtime.py:1000-1005`)

### Gold to keep
- Provider-agnostic text ReAct (deliberate, works on GGUF)
- Honest degradation as first-class type
- `audit_reachability` — every tool must appear in listing
- Double enforcement (capability-filtered listing + registry.call)
- Intent provenance (`why` on every Intent, route telemetry)
- Depth strategy chain (composable, not hardcoded)

---

## Child report 2: Policy/Gating Layer

### Critical finding: LIVE PRIVACY HOLE — outsiders can memory_recall

`brain.py:311-319` grants non-owners `CapabilitySet.of(MODEL_CALL, MEM_READ, FS_READ)`. The social gate's prefix filter only covers `social_/telegram_/tgbot_/whatsapp_/game_` — never `memory_*`. So outsiders can read the owner's memories via the tool loop. Directly contradicts the social gate's own docstring ("No memory access").

### Other gaps
- **G2:** `PUBLIC_GROUP_TOOLS` lists `game_move`, `game_join` etc. — none registered. Dead allowlist.
- **G3:** `games` tool has `capability=""` → skips ALL gates by accident (empty string is falsy).
- **G4:** Group admins can `whatsapp_community_broadcast` — `may_send=False` flag is advisory dead weight, never checked.
- **G5:** `capabilities_for_grant()` is dead code — brain hardcodes outsider set instead of deriving from social grant.
- **G6:** Confirmation-token/proposal flow has ZERO production call sites — the flagship confirm/biometric machinery is theater until wired.
- **G7:** Two parallel authority systems — hardcoded `actor != "owner"` strings vs `Policy.check` — never meet.
- **G8:** One settings flag (`gate_restricted_chats`) disables all persona gating — single point of failure.

### Gold to keep
- Tamper-evident audit (`verify_decision`) — every check recorded before returning
- Capability-bound, single-use, TTL'd confirmation tokens
- `narrow_grant`/`child_grant` — privilege narrows down agent tree by construction
- Fail-closed everywhere (gate exception → deny, role lookup failure → member)

---

## Child report 3: Tools Registry (421 tools)

### Critical: 22k tokens/turn for tool listing
The partner chat loop embeds ALL 421 tools (~98k chars) in the system prompt every turn. The orchestration loop has a relevance-ranked, capped (40) listing — but the main chat loop doesn't use it.

### Gaps
- **Broken:** `deals` is async → `registry.call` returns `Ok(coroutine)`. Only async tool, genuinely broken.
- **Ungated:** `deals` and `games` register with `capability=""` → skip ALL enforcement.
- **Collision:** `studio_batch` registered by both `media_edit.py` and `studio.py` — silent shadowing.
- **Fragmented:** `mem.read` vs `memory.read` — same domain, two capability names.
- **Inconsistent errors:** 11 spots return `{"error": ...}` dicts instead of raising typed errors.
- **Sparse confirm gating:** only 4 tools confirm-gated; live trading has no registry-level confirm.
- **Mega-dispatchers:** `trading` (22 actions), `deals` (10), `games` (9) — one capability for wildly different privilege levels.

### Gold
- Registry: per-module isolated imports, `failed_modules` tracking, description sanitization, secret-safe audit digests
- `ToolAdapter` (orchestration) — relevance-ranked, capped listing. THE answer to the 421-tool problem.
- `filesystem.py safe_path()` — reference pattern for path-taking tools
- 0 decorative stubs — stub scan clean

---

## Child report 4: LLM Providers

### Cross-boundary bug: lifecycle GGUF never auto-registers
`agents/context.py:554` calls `lc._provisioner(model)` — but `ModelLifecycle` has no `_provisioner` method (it's the `provisioner` attribute). Always raises AttributeError, caught and logged. Net effect: **the lifecycle-managed local GGUF is never auto-registered as a router provider at boot**.

### Gaps
- `best_of` bypasses router accounting — direct provider calls skip cooldown/health/cost/learning
- Broker cards get `context_len=0` on settings-driven path → 8192 default for everything (over-budget on 4k phone models)
- `embed()`/`describe_image()` raise instead of returning error responses (inconsistent)
- Default chain ships `Cutyp/codebeast-7b-vl` which doesn't exist yet → 404 landmine
- No brain-down bus event — total failure is silent to operator
- `adjudicate._parse` returns ok=True on unparseable judge JSON (dishonest)
- `fan_out` sequential joins → n×timeout worst case
- Repair hooks never installed by anything (dead capability)
- `intent_detail` template unused (dead)

### Gold
- Typed failure taxonomy + per-class recovery (failures.py) — best engineering in the module
- `verify()` at startup with 1h parking of config errors
- `chat_json` repair-retry loop
- `_call_bounded` daemon-thread timeout
- Learning: "no behaviour change without evidence" discipline
- Lifecycle transition matrix with rollback stack
