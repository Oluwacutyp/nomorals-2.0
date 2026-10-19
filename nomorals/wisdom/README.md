# WisdomKeeper

The esoteric study companion and practice organ (L5). Wraps `nomorals/books`
for corpus storage and `nomorals/documents` for parsing — it does not
rebuild either.

## Classes

- **`WisdomKeeper`** (`keeper.py`) — the façade. Owns one of each below,
  plus `synthesize()` / `digest()` / `gaps()` / `answer_card()` brain
  entry points.
- **`CanonCorpus`** (`corpus.py`) — curated manifest of texts (versioned JSON),
  ingest with sha256 idempotency, `ask()` with provenance on every hit.
  Retrieval upgrades: pseudo-relevance query expansion (`expand=True`),
  semantic evidence gate (`min_score`), MMR diversity (`diversify=True`),
  weighted hybrid fusion (`weights=(kw, sem)`), and `Answer.render()`
  in `chat` / `terminal` / `markdown` styles with provenance badges and
  an honest coverage-based confidence note.
  Ask passages are re-extracted from the full chapter text at sentence
  boundaries (never mid-sentence), OCR hyphenation breaks are undone
  at ingest, and every passage carries its work slug + chapter so
  `Answer.render()` can print a "Read full: /wisdom read <slug>" line.
- **`reading.py`** — pure readability helpers shared by CLI, Telegram,
  and corpus: `normalize_tradition()` (case/hyphen/space-tolerant),
  `parse_timeline_args()` (`timeline <tradition> [n]`), `parse_read_args()`
  (`read <work> [chapter] [page]`), `dehyphenate()`, `paginate_text()`
  (deterministic sentence-boundary pages: page N+1 starts exactly where
  page N ended), `TRADITION_INFO` (what each tradition *is*), and `SEEDS`
  (every seed source + how to add one). `ManifestEntry.seed` records
  which seed each work came from.
- **`ArchiveIngestor`** (`ingestor.py`) — fetches texts from public libraries
  (Gutendex/Gutenberg, Wikisource, archive.org, web fallback), respects
  robots.txt, per-domain politeness delay, caches blobs by sha256, strips
  Project Gutenberg boilerplate before ingest, records source-edition
  provenance, actively searches archives for new candidates.
- **`HistoryEngine`** (`history.py`) — 52-event timeline dataset
  (`data/timeline.json`, every entry sourced), `timeline()` / `compare()` /
  `lineage()` queries, plus era buckets (`events_by_era()`), the
  cross-tradition "everything at once" view (`parallel_at()`), century
  density, `gaps()` for autonomous hunting, and `render_ascii()` terminal
  timelines.
- **`PracticeGuide`** (`practice.py`) — breathing session scripts
  (`sessions/*.json`), injectable-clock pacer, practice log, post-session
  journaling. Upgrades: custom user programs (`create_custom()`), pre-session
  settle-in countdown, ramp mode (breath times grow across rounds), midpoint
  chime, 1–5 post-session ratings, `stats()` (minutes, streaks, per-session
  counts, avg ratings), and `catalog_text()` with rhythm glyphs.
  Every session prints the honest safety framing: breathing aids
  relaxation via the parasympathetic system; no verified link to astral
  projection or kundalini; stop if lightheaded/dizzy; clinician note for
  lung/heart/pregnancy/BP conditions; not medical care.
- **Embeddings** (`embeddings.py`) — `embed_query()` / `embed_documents()`
  retrieval asymmetry (E5 `query: ` / `passage: ` prefixes applied
  automatically), Matryoshka `dimensions` truncation, role-aware cache keys,
  async variants; bge-m3 available for the multilingual corpus.
- **VectorIndex** (`vectorstore.py`) — embedding cache (rebuilds carry over
  unchanged vectors), tradition/work metadata with pre-filtered search,
  similarity floor, `stats()` health and `vacuum()`.
- **Hybrid** (`hybrid.py`) — RRF plus `weighted_rrf` (Weaviate-style alpha
  knob), semantic floor before fusion, fused-score transparency on hits.
- **WisdomOrgan** (`autonomy.py`) — autonomous ingest/digest/cross-link tick.
  Digests now use TextRank sentence centrality + MMR diversity; digests
  refresh when a work is re-ingested (staleness tracking); `consolidate()`
  proposes near-duplicate merges (never deletes); `gap_analysis()` produces
  autonomous research proposals with live hunt candidates.

## CLI (operator)

```
nm wisdom status [--json]          # per-work receipts: seed, source, chars/pages, read command
nm wisdom ask <query> [--limit N] [--json]   # sentence-complete passages + "read full" hints
nm wisdom read <work> [chapter] [page]       # the ACTUAL full text, paginated
nm wisdom traditions [tradition]    # what each tradition is + its works
nm wisdom seeds                     # every seed source + which works came from it
nm wisdom <tradition>               # bare tradition lists its works
nm wisdom ingest <slug> | --all
nm wisdom seed
nm wisdom search <query> [--limit N] [--json]
nm wisdom timeline [tradition] [n] [--start Y] [--end Y] [--json]
nm wisdom compare <topic> [--json]
nm wisdom practice list [--json]
nm wisdom practice <session-id> [--rounds N]
```

Telegram mirrors the CLI: `/wisdom read <work> [chapter] [page]`,
`/wisdom traditions`, `/wisdom seeds`, `/wisdom <tradition>`,
`/wisdom timeline <tradition> [n]` (full summaries), `/wisdom status`.

## Corpus seed

`data/manifest.json` ships 47 texts across 14 traditions (canon, apocrypha,
pseudepigrapha, gnostic, dss, eastern, esoteric, secondary). Run
`nm wisdom seed` to register them, `nm wisdom ingest --all` to fetch.
Only public sources; no credentials; robots.txt respected.

## Seeds — more than one

"Seed" = a starting source list / discovery channel, not the texts
themselves (`/wisdom seeds` lists them all with how to add one):

1. **packaged-seed** — `data/manifest.json`, the 47 curated texts.
2. **timeline-seed** — `data/timeline.json`, 52 sourced history events
   (history only — no full text to open).
3. **hunt:gutendex / hunt:wikisource / hunt:archive / hunt:web** — live
   discovery channels behind `/wisdom search <query>`; candidates are
   NOT ingested until the owner picks one. A work ingested from a
   candidate records `seed: hunt:<source>`.
4. **manual** — texts registered directly (paste-in, local file, URL).

Every work records its seed (`ManifestEntry.seed`), so
`/wisdom status` and `/wisdom traditions` can always show the receipt:
seed → work → source URL → ingested characters/pages → `/wisdom read`.
