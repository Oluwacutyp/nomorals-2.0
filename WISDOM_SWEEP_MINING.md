# WISDOM Sweep — External Mining Report

Module: `nomorals/wisdom/` (12 files). Mined 2026-10-10 before any code was written.
Every significant class compared against the best implementations found outside the repo — best AND trash.

Sources: retrieval-stack analysis (astrointelligence-dev/anchor), embedding sweep (jvdbreemen/llmwiki-kennisbank),
rag-deep-dive (dhruvmakwana), TemporalStore PR#500, vault-rag (karusrus), localrag-kit (ritualdev-lab),
strophios/local-library vector-storage report, Mnema, dev.to on-device AI architecture,
BAAI/bge-m3 HF discussion #35, multilingual-e5-large model card, GITenberg, Wikisource/W3C digital-text use case,
Awesome Breathing / Paced Breathing / Pocket Breath Coach (guided-breathing UX), TextRank/LexRank/MMR literature,
Timeline of Religion (alchetron/Wikipedia), Evolutionary Tree of Myth & Religion (ultraculture),
Timeline of Western Mysticism/Esotericism/Occultism poster (RogelioDio).

---

## 1. CanonCorpus (corpus.py) — curated RAG collection

**Best in class:**
- vault-rag (local RAG over markdown vault): hybrid BM25 + bge-m3, **similarity floor** — "if the best vector
  similarity is below the gate, the notes do not cover the question and no model is called"; gate set from data
  (off-topic 0.32–0.34, weakest real 0.50 → gate at 0.42). Chunks follow author structure, **ids are a hash of
  path+content so re-indexing doesn't reshuffle them**, stored vectors **carried over instead of recomputed**,
  BM25 with pseudo-relevance feedback (query re-run with words from best hits), bge-m3 multilingual.
- localrag-kit: pure-SQLite hybrid (FTS5 BM25 + dense BLOBs), **incremental SHA-256 change detection** (re-index
  only changed files, cascade deletes), retrieve 20–50 then rerank, `rich` CLI.
- rag-knowledge-architect / rag-best-practices: pipeline = clean → structure-aware chunking → metadata tagging →
  **hybrid retrieval → cross-encoder rerank (top-20 → top-5)** → context filtering → grounded answer → evaluation.
  **Metadata pre-filtering BEFORE vector search** (Qdrant payload indexes). **MMR** (λ=0.5–0.7) for diversity.
  Similarity threshold discard ~0.7 (tune per domain). Multi-query retrieval (3–5 variations, dedup).
  Failure modes: wrong docs → smaller chunks + hybrid + pre-filtering; misses → multi-query/query expansion.

**How the best do it vs. ours:** We have hybrid (BM25 + vectors + RRF) and stable passage keys — good bones.
We are MISSING: (a) similarity floor on the semantic side (junk vectors fuse in); (b) MMR diversity;
(c) reranker stage (bge-reranker-v2-m3 cross-encoder, local, free); (d) metadata pre-filtering (tradition filter
is post-hoc on hits, not pushed into the index); (e) incremental indexing — `build_semantic_index` re-embeds
everything unless counts match exactly; (f) pseudo-relevance query expansion; (g) evidence gate.

**Trash seen:** naive "embeddings + top-k + LLM" demos that silently return garbage on vocabulary mismatch;
systems that mix embedding models in one collection (same-dimension mismatch → silently wrong rankings, measured
0.88 → 0.75 Recall@3). We already fail-fast on model mismatch — keep that.

**What SHOULD exist:** `ask(expand=, min_score=, diversify=)`; embedding cache by sha256(text+model+role) so
re-index is free; tradition pushed into the vector index as a filterable column; rerank hook (structure ready
even if the cross-encoder dep is absent).

## 2. EmbeddingBackend (embeddings.py) — 4 backends + registry

**Best in class (2026 SOTA per anchor retrieval analysis + embedding sweep):**
- **Query/document asymmetry is load-bearing:** every serious provider exposes `embed_query` vs
  `embed_documents` — E5 prefixes (`query: `/`passage: `), Qwen3 instruction-on-query-only (+1–5% retrieval).
  TemporalStore measured e5-small bare at 59.4% hit@1 → **74.8% with prefixes** (+15 pts, same 384d).
  "Measuring them bare understates them." Six of their fifteen points were *correct usage*.
- **Cache key must include model AND role** (query vs passage) — caching the two together returns a query
  vector where a passage vector was requested, silently, with no dim change to notice.
- **Matryoshka (MRL) truncation:** `dimensions` param; 512d retains 94–98% (OpenAI-3, Gemini, Qwen3, nomic-1.5
  support it). Quantization (int8/binary) is a separate stackable lever.
- **Batch + async** entry points.
- **bge-m3:** BAAI confirms no prefixes needed; produces dense + sparse + multi-vector from one call;
  multilingual MIRACL leader-class. For a corpus of Sanskrit/Greek/Coptic/Arabic/Chinese texts, bge-m3 is the
  right primary — bge-small-en-v1.5 is English-only.
- **Qwen3-Embedding** is #1 MTEB multilingual (2026); nomic-embed-text is the cheap 274MB-resident pick.

**How the best do it vs. ours:** We have lazy imports, registry, auto-select, always-on hashing fallback —
excellent operational design. We are MISSING: (a) asymmetric query/document entry points (E5 in our registry
is measured ~15 pts below its potential); (b) MRL `dimensions`; (c) role-aware cache keys; (d) bge-m3 as a
first-class model choice for the multilingual corpus; (e) async.

**What SHOULD exist:** `embed_query()` / `embed_documents()` on the base class with per-model prefix hooks;
`embed_texts(backend, texts, dimensions=)` truncation; model-aware cache key helper.

## 3. VectorIndex (vectorstore.py) — sqlite-vec + brute-force fallback

**Best in class:**
- dev.to on-device AI architecture: sqlite-vec = native SQLite C extension, <20MB footprint, single-file backup
  (`cp database.sqlite`), vectors as virtual tables joinable with relational metadata in one SQL query. The
  right minimal choice for passage-scale corpora (thousands of rows — "genuinely fine" with brute force).
- Mnema: pluggable backends (Chroma default / Qdrant production / sqlite-vec smallest footprint); switch by
  env var. Tests skip backends whose deps are missing.
- vault-rag: **stored vectors carried over instead of recomputed**; ids = hash(path+content).

**How the best do it vs. ours:** Our two-tier (vec0 → python brute force) design is already best-in-class for
the footprint. MISSING: (a) embedding cache table (sha256 of text+model → vector) so rebuilds are free;
(b) filterable metadata (tradition/work) in the index for pre-filtered KNN; (c) similarity floor at search;
(d) index health stats (coverage vs library passage count, build time, backend/dim); (e) documented backup
story.

## 4. fuse_hits / RRF (hybrid.py)

**Best in class:** RRF k=60 is the literature standard (Cormack et al., SIGIR 2009) — we match it. Weaviate-style
hybrid adds an **alpha weight** (typical start 0.7 vector / 0.3 keyword, domain-dependent; academic literature
wants higher keyword weight — relevant for us: precise religious terminology). Best systems fuse with a
**similarity floor on the semantic side** so the vector half doesn't return "top 60 of anything".

**MISSING:** configurable per-side weights, semantic floor before fusion, fused-score transparency on the
winning hit (keep both scores for display).

## 5. HistoryEngine (history.py) — timeline dataset + queries

**Best in class:**
- RogelioDio's Western Mysticism/Esotericism/Occultism timeline: **three eras** (Ancient 30th c. BCE–1st c. CE,
  Medieval 5th–15th c., Modern/Contemporary 15th–21st c.), **vertical tradition bands** (Mesopotamian, Egyptian,
  Vedic, Greek, Celtic…), symbols per current, century divisions. Presentation IS the product.
- ultraculture's Evolutionary Tree of Myth & Religion: regions as columns, traditions as branching streams over
  time — the "what was happening everywhere at once" view.
- Timeline of Religion (Wikipedia/alchetron): dense dated events with citations; every event sourced.

**How the best do it vs. ours:** We have validated events with sources (good) but only 52 events and flat
queries. MISSING: (a) era/period bucketing; (b) **parallel view** — "everything across traditions in year X";
(c) century density map; (d) **visual rendering** (ASCII timeline — the god-tier presentation mandate);
(e) gap analysis (which centuries/traditions are thin → feeds autonomous ingest hunting); (f) figure/school
lineage is token-match only.

## 6. PracticeGuide (practice.py) — guided breathing sessions

**Best in class (Awesome Breathing, Paced Breathing, Pocket Breath Coach):**
- Fully **customizable programs** (inhale/hold/exhale/hold durations) + create-and-save custom programs.
- **Pre-session settle-in countdown** ("a few moments to settle in").
- **Ramp mode:** gradually change breath times across the session.
- **Streaks, goals, session tracking**; reminders.
- **Interval structure:** intro & posture → begin → midpoint "settle deeper" → wind-down & close (chapters).
- Bells at start/end; vibrate mode; background playback.
- Coaching copy: belly leads, shoulders soft, 75% full ("calm, not forced"), counts 3–6s adjustable, drop holds
  if dizzy.

**How the best do it vs. ours:** We have data-driven JSON sessions, injectable clock, safety framing, chat
delivery, journal — strong. MISSING: (a) custom program creation; (b) settle-in countdown; (c) ramp mode;
(d) streaks/stats (we log sessions but never compute streaks/minutes); (e) midpoint chime; (f) post-session
rating; (g) **visual rhythm display** — breathing-pattern glyphs (style mandate).

## 7. ArchiveIngestor (ingestor.py) — fetch → parse → ingest

**Best in class:**
- **GITenberg:** 57k Gutenberg texts as GitHub repos with detailed metadata, rebuilt EPUB/PDF releases —
  version-controlled cultural heritage; CI-built artifacts.
- **Wikisource (W3C use case):** curates **individual works** (not volumes), **links translations to originals**,
  links authors/periods, multi-lingual. MediaWiki API = structured metadata.
- **Gutendex:** free no-key JSON API over the Gutenberg catalog (search, authors, subjects, formats) —
  the clean discovery path we lack (we only have archive.org + DDG scraping).
- **Standard Ebooks:** high-quality cleaned texts (the "clean text" bar).
- **Gutenberg criticism (Ockerbloom):** texts need **source edition citations**; PG headers/footers are
  boilerplate that pollutes corpora ("*** START/END OF PROJECT GUTENBERG" markers) — must be stripped.
- Perseus Digital Library: scholarly curation at book level.

**How the best do it vs. ours:** We have archive.org advancedsearch + DDG fallback, robots respect, blob cache,
sha256 idempotency — solid. MISSING: (a) Gutendex catalog backend (structured search + metadata); (b) Wikisource
MediaWiki API backend; (c) **Gutenberg boilerplate stripping** (*** START/END markers — real pollution in the
current ingest path); (d) source-edition provenance on the manifest entry; (e) per-domain politeness delay;
(f) translation↔original linking.

**Trash seen:** scrapers that ingest IA OCR with headers/footers intact; bulk re-uploaders that strip
attribution. We keep attribution — keep that.

## 8. WisdomOrgan (autonomy.py) — digest / cross-link / synthesize

**Best in class (extractive summarization literature):**
- **TextRank** (Mihalcea & Tarau): sentence graph, PageRank centrality — captures sentence similarity/centrality,
  effective for short summaries. **LexRank:** eigenvector centrality, coherent clustering. **MMR** (Carbonell &
  Goldstein): relevance + novelty, kills redundancy. **Luhn:** term-frequency significance windows.
  Best practice = centrality scoring + MMR diversity + position/length features.
- Memory hygiene (branch-agent wave-3): nightly consolidate pass merges near-duplicates above a similarity
  threshold — **proposing, never deleting on its own**.
- Evidence-gated synthesis: cite-or-refuse.

**How the best do it vs. ours:** Our digest scores passages by term-overlap with top terms — no centrality, no
diversity, near-duplicate passages can all be selected. MISSING: (a) TextRank-style centrality; (b) MMR
dedup; (c) near-duplicate consolidation; (d) digest staleness (re-digest on re-ingest); (e) **gap analysis** —
"the corpus is thin on X; here are archive.org candidates" (autonomous research proposals); (f) richer
cross-link signal (Jaccard on bigrams + semantic when vectors exist).

## 9. ChatPracticeSession / WisdomChatManager (chat_session.py)

Already strong: fail-fast validation, injectable clock, chunked sleeps, pause/resume/stop words, journal-await
persistence, no parallel messaging stack. Borrow from the best: **midpoint chime** ("halfway — settle deeper"),
progress in `status()` (phases done/total), post-session rating prompt.

## 10. Answer / synthesis (corpus.Answer)

The synthesis is an acknowledged stub ("Phase 6"). Best practice (vault-rag): **cite-or-refuse** — every claim
traces to a passage id; refusal token when sources don't cover the question. We have the refusal; we lack
**rendered presentation**: provenance badges, per-tradition grouping, confidence notes. The style mandate
demands god-tier output formatting — `Answer.render()` in chat/terminal/markdown styles.

---

## Build list (what the sweep implements)

1. **embeddings.py**: `embed_query`/`embed_documents` asymmetric entry points; per-model prefix hooks
   (E5 `query: `/`passage: `); MRL `dimensions` truncation in `embed_texts`; bge-m3 + multilingual-e5-small as
   first-class models; role-aware cache-key helper `cache_key(text, role)`; async variants; Ollama dim map
   for qwen3-embedding.
2. **vectorstore.py**: `embed_cache` table (sha256 text+model+role → vector); `build()` consults cache first
   (reports cached vs embedded); tradition/work filter columns with migration; `search(..., tradition=,
   min_score=)` pre-filter + floor; `stats()` health (coverage vs library, build time); `vacuum()`.
3. **corpus.py**: `ask(expand=, min_score=, diversify=)` — pseudo-relevance query expansion, semantic floor,
   MMR diversity; incremental `build_semantic_index` via cache; `Answer.render(style=)` (chat/terminal/
   markdown) with provenance badges; answer confidence note.
4. **hybrid.py**: `weighted_rrf` + `fuse_hits(weights=, semantic_floor=)`; fused scores kept on hits.
5. **history.py**: era buckets, `parallel_at(year)`, `century_density()`, `render_ascii()` timeline,
   `gaps()` for autonomous hunting.
6. **practice.py**: `create_custom()` user programs; settle-in countdown; `ramp` mode; midpoint chime;
   `stats()` (streaks, minutes, per-session counts); `rate()` post-session rating; `catalog_text()` with
   rhythm glyphs.
7. **ingestor.py**: Gutendex + Wikisource search backends; Gutenberg boilerplate stripping
   (`*** START/END OF` markers); source-edition provenance into manifest notes; per-domain politeness delay.
8. **autonomy.py**: TextRank-style digest centrality + MMR; near-duplicate consolidation proposals;
   digest staleness re-digest; `gap_analysis()` producing ingest candidates.
9. **keeper.py**: facade passthroughs (`synthesize`, `digest`, `gaps`, `answer_card`).
10. **chat_session.py**: midpoint chime message; progress in `status()`; rating in journal prompt.
11. **Tests**: `tests/test_wisdom_sweep.py` covering all new behavior; existing wisdom tests must stay green.
