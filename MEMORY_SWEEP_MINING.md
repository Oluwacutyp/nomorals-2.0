# MEMORY_SWEEP_MINING.md — True system-wide upgrade: `nomorals/memory/`

Mining pass for the 18-file memory module. Every class was compared against the
best implementations outside the repo — best-in-class AND trash, per the
standing rule. Findings written BEFORE any code changes.

Sources are cited inline. Verbatim quotes are avoided; findings are summarized.

## 1. The state of the art (external)

### 1a. mem0 — two-phase extraction→update (mem0ai/mem0, arXiv 2504.19413)
- **Architecture:** Extraction phase (LLM pulls atomic facts from new exchange +
  rolling summary + recent messages) → Update phase (retrieve top-similar
  existing memories, LLM picks **ADD / UPDATE / DELETE / NOOP** per candidate).
- **Key lesson:** memory is *actively managed*, not append-only. Vector-only
  dedupe is insufficient — semantic dedupe via LLM reasoning decides updates.
- **What we already have:** `extract.py` `_heuristic_pass` + `_llm_pass` +
  dedupe recall + contradiction supersede ≈ a hand-rolled ADD/UPDATE/NOOP.
  **Gap:** no explicit DELETE decision (user says "forget my old X" — nothing
  deletes), no rolling conversation summary as extraction input, decisions are
  implicit rather than an explicit action enum.
- **Mine:** explicit `MemoryDecision` (ADD/UPDATE/DELETE/NOOP), "forget ..."
  deletion extraction, rolling summary context for the LLM pass.

### 1b. Zep/Graphiti — bi-temporal knowledge graph (getzep/graphiti, arXiv 2501.13956)
- **Architecture:** three subgraphs — episode (raw, immutable), semantic entity
  (entities + relations, deduped), community (label-propagation clusters with
  summaries). Every edge is **bi-temporal**: `t_valid/t_invalid` (when the fact
  was true) and `t'_created/t'_expired` (when the system learned it). New info
  invalidates old edges; nothing is ever deleted. Retrieval: hybrid
  semantic+BM25+graph traversal, fused with **RRF/MMR/cross-encoder**, no LLM in
  the retrieval loop (p95 ~300ms).
- **What we already have:** `tiers.py` FactStore has `valid_from/valid_to`,
  `supersede_fact` invalidates (sets `valid_to`), `as_of` and `timeline` queries
  — the T (validity) timeline is wired.
- **Gaps vs Graphiti:**
  - **No community subgraph.** `persona.py` PeopleGraph has no edges and no
    clustering. Graphiti's community tier (clusters + summaries) is missing.
  - **T' (transactional) close is incomplete.** We track `active` + `valid_to`
    but no explicit ingestion-timeline close timestamp.
  - **Entity resolution/dedup:** `_guess_name` only; no entity subgraph.
- **Mine:** `PeopleGraph.communities()` via label propagation with summaries
  (the community subgraph); `closed_at` on tier_facts completing the bi-temporal
  T' axis. Full entity subgraph is out of scope for one sweep (neo4j-class
  work) — the co-occurrence edges + communities capture the retrieval value.

### 1c. Letta/MemGPT — self-editing memory (letta-ai/letta)
- **Architecture:** OS analogy — core memory blocks (pinned in context, ~2K chars
  each, agent self-edits), archival memory (vector store, "disk"), recall memory
  (conversation history). The agent calls `memory_replace/insert/rethink` and
  `archival_memory_insert/search` itself; **heartbeat** chains multi-step memory
  ops; **memory pressure** signals force summarization/archival; sleep-time
  consolidation compounds offline.
- **What we already have:** `build_context` injects recalled memory into the
  prompt, `consolidate` exists, but the agent cannot *edit* pinned memory —
  everything is recall-side. There is no pinned block, no pressure signal.
- **Gap (big):** no self-editing core memory blocks. The brain has no
  always-present workspace it can update across sessions.
- **Mine:** `CoreBlocks` — named blocks with hard char limits, replace/insert/
  rethink, `memory_pressure()` reporting fill %, persisted in
  `memory_core_blocks`, rendered into the system prompt via `render()`.

### 1d. FSRS vs SM-2 (open-spaced-repetition/fsrs-rs, fsrs4anki)
- **State of the art:** FSRS models memory with the DSR model —
  **Difficulty (D), Stability (S), Retrievability (R)** — 17–21 optimizable
  params. The forgetting curve `R(t,S) = (1 + 19/81 · t/S)^(−0.5)` replaces the
  heuristic "ease factor". Retention is a *setting* (`desired_retention`), not
  an outcome; the scheduler solves intervals for it. Being late is information.
  Anki has used FSRS as default since 2023; SM-2 has the documented "ease hell"
  trap.
- **What we have:** `repetition.py` is deliberately SM-2-simple (documented in
  the docstring: "not full FSRS"). That's the trash to mine past.
- **Mine:** full FSRS-4.5 scheduling in `repetition.py`: D/S/R state,
  `desired_retention` (default 0.90), `retrievability_of()` (recall probability
  for any card — SM-2 cannot compute this), `due()` ordered by "closest to
  being forgotten first", legacy SM-2 card migration (D=5, S≈interval).

### 1e. Late chunking / contextual retrieval (Jina, Günther et al. 2024; Anthropic)
- **Late chunking** (embed the whole document, pool per chunk span): +24%
  relative retrieval improvement on LongEmbed (512-token chunks). Needs
  token-level embeddings — not available from our providers.
- **Anthropic Contextual Retrieval** (embed chunk *with generated document
  context*): 35–67% fewer retrieval failures. Implementable with one
  context header per chunk.
- **What we have:** `ingest_document` naive-chunks and embeds each chunk raw.
- **Mine:** contextualized chunking — prepend a doc-context header (source +
  doc head) to each chunk's *embedding text* while storing the raw text for
  display. Opt-out via `contextual=False`. This is the implementable half of
  the research; true late chunking waits on token-level embedders.

### 1f. Vector backends (sqlite-vec, usearch, LanceDB)
- sqlite-vec: <20MB footprint, single-file, brute-force ceiling; supports
  binary quantization and Matryoshka-truncated vectors (Qwen3-embedding is a
  Matryoshka model — truncation to 512/256 dims costs little quality).
- usearch: fastest ANN in many benchmarks; LanceDB: disk-based, exceeds RAM.
- **What we have:** sqlite-vec / usearch / legacy backends with auto-select.
- **Mine:** Matryoshka truncation (`truncate_dims`) on put for sqlite-vec and
  usearch — halves storage at 1024→512 dims. Binary quantization stays out
  (needs extension features not guaranteed on this build).

### 1g. Hybrid retrieval fusion (RRF, MMR, cross-encoders)
- RRF (rank-only, k=60) is the baseline. Production stacks add **MMR**
  (λ-weighted relevance–diversity) and cross-encoder rerank on top.
- **What we have:** `hybrid.py` RRF fusion, clean and correct.
- **Mine:** MMR diversification (`mmr_select` with λ), weighted RRF lane
  weights, query→vector MMR hook in `HybridMemoryIndex`-adjacent search path.

### 1h. Embeddings (Qwen3-Embedding, BGE-M3, Jina v3)
- Best local: Qwen3-Embedding-0.6B (Apache 2.0, MTEB-multilingual SOTA at its
  size, Matryoshka-native 1024 dims). We already wrap it via llama-server.
- **Mine:** surface truncation support end-to-end (embedder →
  `truncate_dims` → backends).

## 2. Class-by-class upgrade plan

| Class / module | External gold | Planned change |
|---|---|---|
| `repetition.py` RepetitionScheduler | FSRS-4.5 (fsrs-rs) | DSR state, desired_retention, retrievability_of, due-ordering by forgetting, SM-2 migration |
| `tiers.py` FactStore | Graphiti bi-temporal | `closed_at` completing T' axis; expose validity in to_dict |
| `persona.py` PeopleGraph | Graphiti community subgraph | co-occurrence edges + label-propagation `communities()` with summaries |
| `manager.py` MemoryManager | Letta core blocks | `CoreBlocks`: pinned, char-limited, self-editable blocks + pressure signals |
| `manager.py` ingest_document | Anthropic contextual retrieval | contextualized chunk embeddings (`contextual=True`) |
| `hybrid.py` | MMR | `mmr_select`, weighted RRF |
| `vector_backends.py` | Matryoshka (sqlite-vec docs) | `truncate_dims` on put (sqlite-vec + usearch) |
| `extract.py` | mem0 ADD/UPDATE/DELETE/NOOP | explicit `MemoryDecision` enum, DELETE extraction ("forget …") |
| `base.py` | god-tier presentation | `format_record(style=)`: compact / rich / chat / briefing |
| `delivery.py`, `proactive.py` | — | themed surfacing format using base styles |
| `embeddings.py` | contextual retrieval | `contextualize_chunk()` helper |

## 3. What stays out of scope (honest)
- Full entity/relation subgraph with LLM dedup (Graphiti's Gs): neo4j-class
  graph work; co-occurrence communities capture the retrieval value at our
  scale. Not parked — genuinely a different system.
- Cross-encoder rerank: needs a model download + serving; honest "later, with
  the model", not silent.
- Binary quantization on sqlite-vec: extension-feature dependent; unverified on
  this build, so not shipped on guesswork.
- True late chunking (token-level embeddings): providers don't expose token
  embeddings; contextual chunking is the implementable half.
