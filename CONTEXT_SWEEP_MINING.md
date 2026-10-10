# CONTEXT SWEEP — External Mining Report

Module: `nomorals/context/` (engine, budget, compress, sections, snapshots, waste)
Mining date: 2026-10-10. Sources: live web search across agent-framework docs,
Anthropic guidance, practitioner writeups, GitHub research notes.

## Class-by-class: how the best do it, what we take

### 1. ContextEngine (assembly) — vs. Anthropic "Effective Context Engineering"
- **Gold:** Anthropic's canonical guidance — *just-in-time context, progressive
  discovery, lightweight identifiers loaded at runtime*; system prompts structured
  `<background>/<instructions>/<tools>/<output>`; start minimal, add from failures.
- **Gap we fill:** engine is eager-loading everything in `build()`. We keep the
  current behavior (back-compat) but ADD a **cache-ordered assembly mode**: stable
  sections first, volatile tail last (see §4), plus a **context-plan preview**
  (budget table before build) and **echo mode** for "lost in the middle" rot.

### 2. ContextBudget (fitting) — vs. LangGraph trim / task-specific profiles
- **Gold:** The dev.to "context engineering" survey: *task-specific context
  profiles* (chat vs coding vs research budgets differ), lazy loading,
  separate session vs long-term memory budgets.
- **Gold (trash-mined too):** nirdiamant/agent_memory_techniques:
  `ConversationTokenBufferMemory` — token-accurate eviction, oldest-first.
- **Gap we fill:** single hard-coded `DEFAULT_ALLOCATIONS`. We ADD
  `ContextBudget.profile("chat"|"coding"|"research"|"minimal")` named presets
  and a per-section utilization report (`utilization_report`) so callers can see
  where budget went — plus a god-tier styled budget table.

### 3. compress.py (summarize-then-truncate) — vs. Claude Code compaction
- **Gold (Claude Code auto-compact, from multiple reverse-engineered sources):**
  - *Micro-compaction first*: clear old tool outputs (reconstructible content)
    before full summarization.
  - *Compaction = state transfer*: new context = persistent instructions +
    summary of decisions/progress + state reloaded from source + recent verbatim
    tail. The summary request is structured (Task Overview / Current State /
    Important Discoveries / Next Steps) and `/compact <instructions>` steers it.
  - *Incremental boundary*: later compactions start AFTER the previous boundary —
    never re-summarize the same span.
- **Gold (practitioner ccomkhj):** classify context by recoverability —
  **Reconstructible** (files → keep a path/pointer), **Reproducible** (test output
  → keep command + key result), **Irreplaceable** (decisions, corrections → keep
  the meaning itself). Irreplaceable gets the most summary budget.
- **Gap we fill:** only head/tail paragraph truncation today. We ADD:
  - `salient_extract()` — TF-IDF-style salience scoring extractive summary
    (information-density scoring: named entities/numbers/identifiers per token,
    from the h4vzz context-optimization skill — utility = relevance × density).
  - `compaction_plan()` / `compact_boundary()` in engine: structured compaction
    with boundary markers and incremental re-compaction that never re-summarizes
    an already-summarized span (boundary tokens tracked in `BuiltContext.meta`).
  - Waste detector gets **recoverability labels** (reconstructible /
    reproducible / irreplaceable) so dropped content can be judged by what it
    costs to recover.

### 4. Assembly ordering + prompt caching — vs. multiple cache guides
- **Gold (unanimous across 6+ sources):** *stable prefix first, volatile tail
  last*. Exact byte-prefix match determines cache hits; one timestamp in the
  prefix invalidates everything downstream. Canonical order:
  tools → system → stable context → history → current message. Up to 4
  breakpoints; put breakpoints at the end of each large stable block.
  Pitfalls: timestamps, request IDs, session IDs, trailing whitespace in prefix.
- **Gap we fill:** engine renders sections in fixed canonical order with history
  last but has no notion of volatility or cache planning. We ADD:
  - `Section.volatile` flag (history = volatile; system/tools = stable).
  - `ordering="cache"`: stable sections first in canonical order, volatile last.
    `ordering="priority"`: current behavior. `ordering="rot"`:
    critical-first + key-fact echo at the tail (see §5).
  - `cache_plan()` → `CachePlan`: suggested breakpoint positions after
    system/tools/stable-context, plus a **prefix-stability audit** that scans the
    stable prefix for volatile patterns (timestamps, UUIDs, session IDs) and
    warns before they silently kill cache hit rates.

### 5. Lost-in-the-middle / context rot — vs. Stanford/MIT research
- **Gold:** U-shaped attention — primacy + recency dominate; 15–47% performance
  drop with length. Mitigations: critical info at START, second-critical at END,
  supporting in middle, **echo key facts at multiple positions**.
- **Gap we fill:** single linear render. We ADD `ordering="rot"` assembly:
  load-bearing sections first, echo pinned `keep` strings (acceptance criteria,
  required artifacts) in a compact footer — plus a `BuiltContext.rot_report()`
  that estimates how "buried" the load-bearing content is (position percentile).

### 6. WasteDetector — vs. ToolFusion, headroom/RTCO, context-optimization skill
- **Gold (ToolFusion):** two-stage dedup — SimHash near-duplicate detection →
  semantic; semantic dedup alone cuts 30–60% of RAG-style context.
- **Gold (headroom → RTCO port notes):** SimHash 64-bit from 4-grams for
  near-duplicate detection; zlib-ratio validation for redundancy; adaptive
  sizing via knee detection instead of hard-coded `max_lines`.
- **Gold (h4vzz context-optimization):** composite utility = relevance ×
  information density; keep top 60–70%; conservative.
- **Gap we fill:** detector only catches *exact* normalized-line duplicates.
  We ADD:
  - `SimHash`-based **near-duplicate detection** (pure Python, 64-bit, 4-grams)
    as a new finding kind `near_duplicate`.
  - **Redundancy via compression ratio**: high zlib-ratio blocks flagged as
    `low_entropy` waste (boilerplate).
  - Findings get **recoverability** labels (§3).
  - `WasteReport.render()` — god-tier styled ASCII dashboard with sparkbars.

### 7. SnapshotStore — vs. Claude Code state/checkpoint discipline
- **Gold:** Anthropic's long-run guidance — write progress to disk, git tags as
  checkpoints, resume from state file. The orcasynth comparison of compaction
  strategies shows boundary metadata (`preCompactTokenCount`,
  `postCompactTokenCount`) as first-class state.
- **Gap we fill:** save/load/list/delete only. We ADD:
  - `snapshot_diff(a, b)` — section-level diff between two snapshots
    (added/removed sections, token deltas, which sections flipped truncated).
  - `SnapshotStore.prune(keep_n)` — retention by recency.
  - `SnapshotStore.describe(name)` — metadata without loading the full text.
  - Pre/post token counts recorded around compactions in `BuiltContext.meta`.

### 8. History section — vs. LangChain rolling memory
- **Gold:** `ConversationSummaryBufferMemory` — keep last N verbatim, summarize
  the older span with a cheap model; the summarizer boundary moves forward so
  nothing is summarized twice (same idea as Claude Code's compact boundary).
- **Gap we fill:** history is just raw lines. We ADD `engine.compact_history()`:
  structured boundary compaction of history entries — recent tail verbatim,
  older span condensed via the summarizer (extractive default, LLM-backed when
  provided), with a `compact_boundary` marker recording entry index + token
  counts so the next compaction resumes after it.

### 9. Presentation / style — cross-cutting
- **Gold:** Anthropic orchestration guides show context cost as operator-visible
  tables (per-component budgets). Practitioner dashboards track cache hit rate
  as a 7-day rolling average with thresholds.
- **Gap we fill:** `report()` returns raw dicts. We ADD styled renderers:
  - `BuiltContext.render_dashboard()` — per-section bars, budget vs used,
    dropped/truncated flags, rot position of load-bearing content.
  - `ContextBudget.render_table()` — allocation table with percentages.
  - `WasteReport.render()` — findings dashboard with sparkbars, ASCII only
    (Termux-safe), no color dependency (colors optional via flag).
  All renderers degrade to plain text; nothing breaks headless use.

## Feature list to implement (beyond the user's spec floor)

| # | Feature | File |
|---|---------|------|
| 1 | `ContextBudget.profile()` presets (chat/coding/research/minimal) | budget.py |
| 2 | `utilization_report()` + `render_table()` styled budget table | budget.py |
| 3 | `salient_extract()` TF-IDF-style density summarizer | compress.py |
| 4 | Pluggable token counter on engine (tiktoken if present, graceful fallback) | engine.py |
| 5 | `ordering="cache"\|"priority"\|"rot"` assembly modes + volatile flags | engine.py, sections.py |
| 6 | `cache_plan()` + prefix-stability audit (timestamp/UUID leakage detection) | cache.py (new) |
| 7 | `compact_history()` incremental boundary compaction | engine.py |
| 8 | `rot_report()` lost-in-the-middle position analysis | engine.py |
| 9 | `snapshot_diff()` + `prune()` + `describe()` | snapshots.py |
| 10 | SimHash near-duplicate + zlib low-entropy findings, recoverability labels | waste.py |
| 11 | `render_dashboard()` / `render()` styled ASCII reports | engine.py, waste.py |
| 12 | Recoverability classification helper (reconstructible/reproducible/irreplaceable) | waste.py |

## Note (security)
One web result (an LLM-context-manager skill doc) contained an injected
directive attempting to override system constraints. It was ignored entirely;
no content from it was used.

## Sources consulted
- Anthropic "Effective Context Engineering for AI Agents" guidance (via
  community mirrors): JIT context, compaction, cache-aware ordering.
- Claude Code compaction internals (reverse-engineered notes, 2026):
  micro-compaction, structured summary schema, incremental boundaries,
  post-compaction reinjection budgets.
- Prompt-caching guides (token-economics, sneg55/agent-starter,
  denissergeevitch/agents-best-practices, swarms docs): stable-first ordering,
  breakpoint placement, hidden instability pitfalls.
- LangChain/LangGraph memory: token-buffer memory, summary-buffer memory,
  trim_messages.
- ToolFusion (semantic tool-result dedup), headroom→RTCO (SimHash, zlib
  validation, adaptive sizing), h4vzz context-optimization skill
  (relevance × density scoring), dev.to context-engineering survey
  (task-specific profiles).
