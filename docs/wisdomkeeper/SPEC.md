# WisdomKeeper — Design Spec

**Status:** design (not yet built)
**Author:** Muse (subagent design pass), 2026-10-02
**For:** Devon / nomorals-2.0, at the owner's request

## 0. Vision

The owner is on a serious esoteric journey — astral projection, akashic
records, kundalini awakening, quantum jumping, past life regression, void
state, timeline shifting, the multiverse, "the real source of life" — and
struggles with breathing technique. They want Devon to have a **dedicated
module** for this domain that:

1. **Ingests real archives** — existing and ancient texts — and **digests**
   them into structured, searchable knowledge (not chat-about-it, but
   grounded Q&A where every claim traces to a source passage).
2. **Covers the stripped Bible** — apocrypha, pseudepigrapha, Gnostic
   gospels (Nag Hammadi), Dead Sea Scrolls public translations, and other
   non-canonical texts.
3. **Teaches history** — comparative timelines across religions, cultures,
   and ethical systems.
4. **Guides practice** — step-by-step breathing sessions and relaxation
   protocols the owner can actually follow, with honest safety framing.

Design principle: **a study companion and practice timer, not an oracle.**
WisdomKeeper retrieves and organizes what the texts actually say; it does
not verify supernatural claims, and it says so plainly where relevant.

---

## 1. Module architecture

New package: `nomorals/wisdom/` — placed at **L5** in the layer stack
(beside `books/`, `agents/`, `codews/`). Rationale: it orchestrates
downward into `documents/` (L4), `memory/` (L3), `storage/` (L2),
`core/` (L1), and uses `books/` + `agents/researcher.py` as L5 peers.
Peers at the same layer may import each other; nothing above L5 may be
imported. The layering test (`tests/test_layering.py`) gets a
`"wisdom": 5` entry with a comment, same as the Wave K organs.

```
nomorals/wisdom/                  L5 — the WisdomKeeper organ
├── __init__.py                   façade re-exports
├── keeper.py                     WisdomKeeper — the single entry point
├── corpus.py                     CanonCorpus — curated manifest + provenance
├── ingestor.py                   ArchiveIngestor — fetch → parse → ingest
├── history.py                    HistoryEngine — timelines + comparative views
├── practice.py                   PracticeGuide — guided breathing/relaxation
├── claims.py                     digested claim store (structured knowledge)
└── errors.py                     WisdomError hierarchy (fail fast)
```

### Class responsibilities

**`WisdomKeeper`** (`keeper.py`) — the façade. Owns one `CanonCorpus`,
one `HistoryEngine`, one `PracticeGuide`. This is what agents, missions,
and the CLI talk to. Thin: it routes, it does not implement.

```python
keeper = WisdomKeeper(context)
keeper.ask("What does the Gospel of Thomas say about the kingdom?")
# → Answer(passages=[...], provenance=[...], synthesis="...")
keeper.corpus.status()        # what is ingested, what is missing
keeper.history.timeline("gnosticism", -300, 400)
keeper.practice.start("4-7-8")  # guided breathing session
```

**`CanonCorpus`** (`corpus.py`) — the curated collection. Owns:
- the **canon manifest**: a versioned JSON file listing every text in the
  corpus — slug, title, tradition, canonical status
  (`canon` | `apocrypha` | `pseudepigrapha` | `gnostic` | `dss` |
  `eastern` | `esoteric` | `secondary`), translator/edition, source URL,
  license, sha256 of the fetched bytes, ingest date.
- **provenance**: every passage returned by search carries
  `(work, translator, section, url)` so claims are traceable.
- delegation of storage/search to `books.Library` (see §2).

**`ArchiveIngestor`** (`ingestor.py`) — the fetch→parse→ingest pipeline.
Takes a manifest entry, fetches the raw bytes (direct HTTP via
`tools/web.py`, or `agents/researcher.py`'s SearchEngine for discovery),
parses with `documents.parse_bytes`, and hands the text to the Library's
ingest. Fail-fast: network failure, unparseable bytes, or checksum
mismatch raise `WisdomError` with the URL and reason — never a silent
empty ingest.

**`HistoryEngine`** (`history.py`) — timelines and comparative religion.
Owns a **curated timeline dataset** (JSON seed file, user-extensible):
dated events with `(start, end, tradition, region, title, summary,
sources[])`. Queries: `timeline(tradition, start_year, end_year)`,
`compare(topic)` (cross-tradition passages + timeline context),
`lineage(figure_or_school)`. The dataset is curated data, not
AI-generated history — the spec is explicit about this (§6).

**`PracticeGuide`** (`practice.py`) — guided practice sessions.
A session is a timed script: a list of phases
`(label, seconds, instruction)`. The guide paces them (prints each phase,
sleeps the duration, optional bell), tracks completions in storage, and
carries the honest safety framing from §5 in its intro text and in
`--help`. Session scripts live as data (JSON), not code, so new
practices don't require code changes.

**`claims.py`** — the digested claim store. A small SQLite table
(behind `storage/` migrations) of structured claims extracted from the
corpus: `(claim, work_slug, passage_ref, topic_tags[])`. Populated by an
LLM-assisted extraction pass in Phase 6 (optional, clearly marked).
Every claim links back to its passage — no free-floating assertions.

**`errors.py`** — `WisdomError(Exception)` + subtypes
(`CorpusError`, `IngestError`, `PracticeError`). Raised across the
organ's boundary; the CLI translates to exit codes.

### Data flow

```
fetch (tools/web or researcher) → raw bytes
  → documents.parse_bytes → Document model
  → books.Library.ingest → FTS5 passages + manifest provenance
  → keeper.ask → Library.search → provenance-enriched Answer
  → (Phase 6) claim extraction → claims table → topic queries
```

---

## 2. Reuse survey (mandatory)

Surveyed before designing. Verdict per component:

| Existing component | Verdict | Reason |
|---|---|---|
| `nomorals/books/library.py` — `Library` (ingest → chapter split → FTS5 BM25 → ranked passage search) | **REUSE as substrate** | This is already the "digest books into searchable knowledge" machine the owner asked for. WisdomKeeper must not rebuild it. `CanonCorpus` wraps `Library`: manifest + provenance on top, `Library.ingest`/`Library.search` underneath. |
| `nomorals/documents/` — `parse_bytes`, `Document` model, `DocumentIndex` | **REUSE parsers** | Real parsers for txt/html/md/pdf/docx already exist and are tested. `ArchiveIngestor` calls `parse_bytes` directly. Do NOT use `DocumentIndex` (plain TF) — the Library's FTS5 BM25 ranks better and is already the corpus search path. |
| `nomorals/storage/fts.py` | **REUSE indirectly** | Via `books.Library`, which already wraps FTS5 with fallback. No direct use needed. |
| `nomorals/memory/manager.py` — semantic memory | **INTEGRATE (later)** | Digested wisdom *about the owner's journey* (e.g., "owner is practicing nadi shodhana") belongs in semantic memory. The corpus itself stays in the Library (it's reference material, not personal memory). Phase 6. |
| `nomorals/agents/researcher.py` — `SearchEngine` | **REUSE for discovery** | Finding archive URLs and verifying source availability. Not for the corpus content itself (curated manifest instead). |
| `nomorals/tools/web.py` — HTTP client | **REUSE for fetch** | Retries, robots handling already implemented. |
| `nomorals/core/text.py` — chunkers | **REUSE if needed** | The Library's `split_chapters` covers the common case; `core/text.py` chunkers are the fallback for texts without chapter structure. |
| `nomorals/archives.py` | **DO NOT USE — name trap** | This is a *file* archive utility (zip/tar), not a knowledge archive. The spec uses "corpus" everywhere to avoid confusion. |
| `nomorals/cognition/` (trajectories, failure_kb) | **NOT RELEVANT** | Agent-run learning, not esoteric knowledge. No touch. |
| `nomorals/games/lexicon.py` | **NOT RELEVANT** | Word-game vocabulary. No touch. |
| CLI (`cmdline/commands/`, `parser.py`, `dispatch.py`) | **EXTEND by pattern** | Add `cmdline/commands/wisdom.py` with `_cmd_wisdom(args, context) -> int`, register `"wisdom": ["wis"]` in `CLI_ALIASES`, wire subparser + dispatch exactly like `doc`/`browse`/`repo`. |

**Consolidation note:** the only deliberate overlap is `CanonCorpus`
vs `books.Library`. This is layered reuse, not duplication: Library owns
*mechanics* (ingest/search/index), Corpus owns *curation* (manifest,
provenance, canon status). If a future refactor merges them, it will be
by moving Corpus's manifest into Library — never by deleting either.

---

## 3. Corpus plan

Owner's directive (2026-10-02): **no limitations on breadth, completeness
over popularity.** Popular does not mean best or complete. Ingest the real
deals — comprehensive archives with old and rare texts, not just the famous
ones. The ingestor must also **actively search** libraries/archives for books
(by topic, title, author, tradition), not just fetch a fixed manifest.

Only public-domain or openly-licensed texts. No credentials, no scraping
behind logins, robots.txt respected. Each entry: what, why, where.

### A. The stripped Bible (non-canonical Christian texts)

| Text | Source (verified) |
|---|---|
| KJV Apocrypha (Tobit, Judith, Wisdom, Sirach, Baruch, 1–2 Maccabees, etc.) | Internet Sacred Text Archive — the site's catalog lists "Bible, The Apocrypha" plus "The Lost Books of the Bible" and "The Forgotten Books of Eden": https://sacred-texts.com (catalog: `sacred-texts.com/cat/`) |
| Nag Hammadi Library — Gospel of Thomas, Gospel of Philip, Gospel of Mary (excerpt), Pistis Sophia excerpts, Exegesis on the Soul | The Gnosis Archive, Robinson translation: `http://gnosis.org/naghamm/` — verified pages include the Patterson & Robinson Gospel of Thomas (`gnosis.org/naghamm/gth_pat_rob.htm`) and the Exegesis on the Soul (`gnosis.org/naghamm/exe.html`); full-volume PDF mirror at `archive.org` (search "The Nag Hammadi Library pdfy") |
| Book of Enoch (1 Enoch), Book of Jubilees, Books of Adam and Eve | sacred-texts.com catalog — "The Book of Enoch the Prophet", "The Book of Jubilees", "The Books of Adam and Eve" all listed |
| Gospel of Mary (fuller), Didache, Ante-Nicene Fathers selections | sacred-texts.com: "Excerpts from the Gospel of Mary", "The Didache", "Ante-Nicene Fathers, Vol. VIII" |
| Dead Sea Scrolls (public translations) | Public-domain English translations via sacred-texts.com's DSS holdings and `archive.org` ("dead sea scrolls english translation" full-text items). Manifest records translator + edition; only openly-licensed renderings. |

### B. Other traditions

| Tradition | Texts | Source |
|---|---|---|
| Hindu | Rig Veda, principal Upanishads, Bhagavad Gita, Hatha Yoga Pradipika, "Kundalini: The Mother of the Universe" | sacred-texts.com (all verified in catalog: "The Rig-Veda", "From the Upanishads", "The Bhagavad Gita", "The Hatha Yoga Pradipika", "Kundalini: The Mother of the Universe") |
| Buddhist | Dhammapada, Lotus Sutra (SBE 21), Buddhist Suttas (SBE 11) | sacred-texts.com (verified: "Saddharma-pundarîka (The Lotus Sutra) (SBE 21)", "Buddhist Suttas (SBE11)") |
| Taoist | Tao Te Ching (Legge translation), I Ching (Wilhelm/Baynes is copyrighted — use Legge's Yi King) | sacred-texts.com + Project Gutenberg (`gutenberg.org`, search "Tao Teh King Legge") |
| Hermetic / Western esoteric | Corpus Hermeticum ("Thrice-Greatest Hermes"), Pistis Sophia | sacred-texts.com (verified: "Hermetica Thrice-Greatest Hermes", "Pistis Sophia") |
| Islamic mysticism | "The Mystics of Islam" (Nicholson) | sacred-texts.com (verified in catalog) |

### C. Esoteric archives (bulk sources) — the real deals

Owner's rule: prefer **completeness and authenticity** over popularity.
These archives hold old, rare, and complete texts — not just the famous ones.

- **Internet Archive** (`archive.org`) — the deepest free library on the
  internet. Full-text search via the `advancedsearch.php` API
  (no key needed); `_djvu.txt` / `_text.pdf` derivatives for ingestion.
  Holds complete runs: Nag Hammadi one-volume scans, Ante-Nicene Fathers,
  Sacred Books of the East (50 vols), patristic and gnostic collections.
- **Internet Sacred Text Archive** (`sacred-texts.com`, J.B. Hare) —
  ~77 categories of public-domain esoterica, all transcribed. Respectful
  fetching: each manifest URL fetched once, bytes cached by sha256, never
  re-crawled (the site asks for restraint).
- **The Gnosis Archive** (`gnosis.org`) — Nag Hammadi (Robinson
  translation), Pistis Sophia, gnostic scriptures; verified page URLs.
- **Project Gutenberg** (`gutenberg.org`) — KJV, Tao Te Ching, Upanishads;
  stable ebook URLs, plain-text format. Catalog search via
  `gutenberg.org/ebooks/search/`.
- **HathiTrust Digital Library** (`hathitrust.org`) — public-domain
  scholarly editions (e.g., Charlesworth's *Old Testament
  Pseudepigrapha* where PD).
- **Wikisource** (`wikisource.org`) — transcribed public-domain texts
  with stable URLs; good for apocrypha and church fathers.

### D. Active library search (not just a fixed manifest)

`ArchiveIngestor.search(query, tradition=None, ...)` queries the archives
above at runtime:
- `archive.org/advancedsearch.php` — full-text + metadata search, returns
  identifiers; ingestor then pulls the `_djvu.txt` derivative.
- Gutenberg catalog search — by title/author/subject.
- sacred-texts.com category pages — by tradition/topic.
Results are ranked by relevance + completeness signals (full text
available, translator/edition recorded), de-duplicated by sha256, and
ingested on owner approval (or auto-ingest per owner setting). This is how
the corpus grows beyond any seed list: the owner asks
"find everything on kundalini awakening" and WisdomKeeper searches,
fetches, digests, and reports what it found.

### Manifest format (per text)

```json
{
  "slug": "gospel-of-thomas",
  "title": "Gospel of Thomas",
  "tradition": "christian-gnostic",
  "canon_status": "gnostic",
  "translator": "Patterson & Robinson",
  "source_url": "http://gnosis.org/naghamm/gth_pat_rob.htm",
  "license": "public-domain",
  "sha256": "<of fetched bytes>",
  "ingested_at": "2026-10-.."
}
```

Seed manifest (Phase 1): per owner's "no limitations" directive, ingest
broadly from day one — the full rows A–C above, not an 8-text subset.
The manifest grows via active search (§D): there is no fixed ceiling.
Full corpus target: every public-domain text in the covered traditions
that the archives hold.

---

## 4. Digestion pipeline

How raw archives become structured, searchable knowledge:

1. **Fetch** — `ArchiveIngestor.fetch(manifest_entry)`:
   `tools/web.py` GET with retries; honors robots; stores raw bytes in
   the blob store (`storage/blob.py`, content-addressed by sha256 —
   dedup across re-ingests is free). Records `sha256` into the manifest.
   Fail fast on HTTP errors / checksum mismatch.
2. **Parse** — `documents.parse_bytes(bytes, filename=...)` → `Document`
   (sections, tables). HTML from sacred-texts.com goes through the real
   HTML parser; PDFs through `core/pdf`; plain text as-is.
3. **Chapter** — `books.library.split_chapters(text)` (markdown headings,
   "Chapter N" headers, or the sensible-chunker fallback).
4. **Ingest** — `Library.ingest(path_or_text, title=..., author=...)` →
   per-chapter markdown files + FTS5 passage index. The manifest slug is
   embedded in the book metadata so search hits map back to provenance.
5. **Provenance** — every `keeper.ask()` result returns passages as
   `ProvenanceHit(work, translator, section, url, snippet)`. The CLI
   prints sources under every answer. No passage, no claim.
6. **Claims (Phase 6, optional)** — an LLM-assisted pass extracts
   `(claim, topic_tags[])` per passage into a `wisdom_claims` table
   (new migration in `storage/`). Each claim row carries `passage_ref`.
   Topic queries (`keeper.topics("kundalini")`) read this table; the
   raw passages remain the ground truth.

Dedup: blob-store sha256 means re-running ingest on an unchanged text
is a no-op. Re-ingest on a *changed* upstream text creates a new version;
the manifest records both, and search can scope to a version.

---

## 5. PracticeGuide — guided sessions

A session is data, not code: a JSON script of phases.

```json
{
  "id": "4-7-8",
  "name": "4-7-8 breathing",
  "safety_note": "practice.safety.breathing",
  "phases": [
    {"label": "Settle", "seconds": 30,
     "instruction": "Sit comfortably. Hand on belly. Breathe normally."},
    {"label": "Inhale", "seconds": 4,
     "instruction": "Inhale quietly through the nose. Belly rises first.",
     "repeat": 4},
    {"label": "Hold", "seconds": 7,
     "instruction": "Hold gently. Shoulders down.", "repeat": 4},
    {"label": "Exhale", "seconds": 8,
     "instruction": "Exhale slowly through the mouth. Belly falls.",
     "repeat": 4}
  ]
}
```

Seed sessions (Phase 4):
- **4-7-8 breathing** (above) — the owner's entry point.
- **Box breathing** (4-4-4-4) — simple, even rhythm.
- **Nadi shodhana** (alternate-nostril) — scripted nostril cues, no
  breath retention at first.
- **Kapalabhati** — short rounds only, with explicit "stop if dizzy"
  cues between rounds; *not* the first session offered.
- **Astral-projection relaxation prep** — body-scan relaxation script
  (the "body asleep, mind awake" setup); framed as deep relaxation,
  not as a projection guarantee.
- **Void-state sitting** — timed silent sitting with interval bells.

The pacer prints each phase, counts down, and logs completions
(`practice_log` table: session, started_at, completed, notes). The CLI
(`nm wisdom practice 4-7-8`) runs it in the terminal.

### Honest safety framing (required, not optional)

Printed in `practice --help`, at the start of every breathing session,
and in this spec:

- Slow breathing with a longer exhale promotes relaxation through the
  parasympathetic nervous system. That is well-supported.
- It does **not** guarantee any particular state — no verified link
  exists between breathing techniques and astral projection, kundalini
  awakening, or other esoteric experiences. Experiences vary; the
  practice is the relaxation itself.
- Go gently, especially at first. If you feel lightheaded, dizzy, or
  uncomfortable: **stop and breathe normally**.
- If you have a lung or heart condition, are pregnant, or have a
  condition affecting breathing or blood pressure, check with a
  clinician before extended breathwork.
- This is not medical or mental-health care, and not a substitute for it.

No refusal machinery, no moralizing — just these qualifications, stated
plainly, every time. Unsupported practices are simply not scripted.

---

## 6. History component

`HistoryEngine` answers "what happened, when, in which tradition" and
"what do traditions say about X".

**Timeline dataset** (`nomorals/wisdom/data/timeline.json`, curated seed
+ user-extensible):
```json
{"start": -380, "end": -320, "tradition": "hellenistic",
 "region": "Egypt/Greece", "title": "Corpus Hermeticum composed",
 "summary": "...", "sources": ["sacred-texts.com Hermetica"]}
```
Seed coverage: axial age (−800…−200: Upanishads, Buddha, Laozi),
Second-Temple Judaism and the DSS (−200…70), Nag Hammadi composition
(~350) and burial, councils and canon formation (325…787), Sufi and
Vedanta flowerings, Theosophy and the 1893 Parliament of Religions.
Every entry carries `sources[]` — no unsourced entries allowed
(`HistoryEngine` validates on load and fails fast).

**Queries:**
- `keeper.history.timeline(tradition=None, start, end)` → dated events.
- `keeper.history.compare("the kingdom within")` → cross-tradition
  passages (corpus search) + timeline context (e.g., Thomas logion 3
  alongside Luke 17:21, with dates and canon statuses side by side).
- `keeper.history.lineage("kundalini")` → tradition-internal
  development (Vedic → Tantric → Hatha → modern).

**Comparative view rule:** traditions are presented side by side with
their own terms and dates. No syncretic flattening ("all religions say
the same thing") — the engine shows convergences *and* divergences,
each with provenance.

---

## 7. CLI / UX sketch

```
nm wisdom status
# corpus: 8/8 seed texts ingested · 12,410 passages · history: 214 events

nm wisdom ask "what does the Gospel of Thomas say about the kingdom?"
# [Gospel of Thomas, logion 3 — Patterson & Robinson]
# "…the kingdom is inside of you, and outside of you…"
# source: http://gnosis.org/naghamm/gth_pat_rob.htm
# ──
# [Luke 17:21 (for comparison) …]

nm wisdom ingest --all            # fetch + parse + index the manifest
nm wisdom ingest gospel-of-thomas # single entry
nm wisdom timeline --tradition gnostic --from -300 --to 500
nm wisdom compare "the void"
nm wisdom practice list
nm wisdom practice 4-7-8          # guided session in the terminal
nm wisdom practice 4-7-8 --rounds 6
```

Follows the `doc`/`browse`/`repo` pattern exactly: `CLI_ALIASES["wisdom"]
= ["wis"]`, subparsers per verb, `_cmd_wisdom(args, context) -> int`,
`--json` flag honored via the global `--json` switch.

---

## 8. Phased build plan

Each phase: deliverable + tests. No phase merges without its tests green,
`error_scan` clean, layering green, zero deletions.

- **Phase 0 — this spec.** Deliverable: this file. No code.
- **Phase 1 — CanonCorpus + seed ingest.** Manifest schema + 8-text seed
  manifest; `Corpus` wrapping `books.Library`; `keeper.ask` over FTS5
  with provenance hits. Tests: manifest validation (bad URL, bad
  canon_status → fail), ingest of 2 fixture texts, provenance on every
  hit, `ask` returns sources. (~15 tests)
- **Phase 2 — ArchiveIngestor.** Fetch via `tools/web.py`, parse via
  `documents`, blob-store caching, checksum mismatch → `IngestError`.
  Tests: mocked HTTP (no network in tests), bad-bytes → DocumentError,
  re-ingest idempotency, robots respected. (~12 tests)
- **Phase 3 — HistoryEngine.** Timeline dataset schema + seed entries
  (~60 events), `timeline`/`compare`/`lineage` queries, unsourced-entry
  rejection on load. Tests: range queries, unknown tradition → clear
  error, every seed entry has ≥1 source. (~12 tests)
- **Phase 4 — PracticeGuide.** Session JSON schema, pacer (injectable
  clock per kernel invariants — no raw `time.sleep` in logic),
  `practice_log` table, safety text on every session. Tests: phase
  sequencing with fake clock, safety note present, completion logged,
  kapalabhati not listed as beginner. (~12 tests)
- **Phase 5 — CLI.** `nm wisdom` + all verbs, `--json` output,
  exit codes. Tests: each verb end-to-end against fixtures. (~10 tests)
- **Phase 6 — Corpus expansion + claims (optional).** Full ~40-text
  manifest; `wisdom_claims` migration; LLM-assisted claim extraction
  pass; `keeper.topics()`. Tests: claim→passage linkage, no orphan
  claims. (~10 tests)

---

## 9. What's NOT included (scope boundaries)

- **No supernatural verification.** WisdomKeeper retrieves what texts
  say; it does not confirm or deny metaphysical claims, and says so.
- **No medical/therapeutic claims.** Breathing guidance is relaxation
  practice with the qualifications in §5. Not a treatment for anything.
- **No copyrighted texts.** Public domain / openly licensed only. If a
  desired text isn't clearly public domain, it stays out of the manifest
  until it is.
- **No aggressive crawling.** One fetch per manifest URL, cached by
  sha256, robots.txt respected. This is a library, not a scraper.
- **No refusal/policy machinery.** Per the owner's standing rule: honest
  qualifications (§5) yes; content-policy boilerplate, refusal engines,
  or moralizing no. Unsupported practices are simply not scripted.
- **No live "channeling", no Akashic-record access claims.** The module
  is honest about what it is: a curated library + practice timer with
  good search.
- **No deletions, no rewrites of existing organs.** Pure addition;
  `books/`, `documents/`, `memory/`, `storage/` are used as-is.

---

## 10. Open questions for the owner — ANSWERED 2026-10-02

1. **Corpus scope:** ~~seed with the 8 texts in §3, or go straight for a
   wider first ingest (~20)?~~ → **No limitations.** Ingest broadly from
   day one (§3); the corpus grows via active search (§D).
2. **Dead Sea Scrolls:** ~~public English translations vary in quality and
   licensing clarity. Include the clearly-public-domain renderings only,
   or skip DSS in Phase 1 and revisit?~~ → **Include the real deals.**
   Owner wants complete/authentic texts, not just the safe popular ones —
   include DSS public-domain renderings in Phase 1.
3. **Library policy:** only "real deal" libraries — needn't be popular or
   public; popular ≠ best or complete. Prefer comprehensive archives
   (Internet Archive depth, Gnosis Archive, HathiTrust, Wikisource).
4. **Practice depth:** ~~breathing pacer + relaxation scripts in Phase 4,
   or also guided journaling after sessions (what you experienced,
   what worked)? *(awaiting owner)*~~ → **All of it.** Pacer + scripts +
   post-session journaling (what was experienced, what worked).
5. **Voice:** ~~should guided sessions eventually speak via Devon's voice
   stack (local XTTS), or is terminal pacing enough to start? *(awaiting owner)*~~
   → **Primary interface is social media** (Telegram/WhatsApp DMs), not
   terminal. Guided sessions should be designed for chat-first delivery
   (timed messages, voice notes via the local XTTS stack where it fits).
   Owner also asked whether a dedicated app is possible — see §11.

---

## 11. Interface: social-first, app later (owner direction 2026-10-02)

The owner will mostly talk to WisdomKeeper through **social media DMs**
(Telegram/WhatsApp), not a terminal. Design consequences:

- **PracticeGuide sessions are chat-native:** timed message sequences
  ("breathe in… 4… 3… 2… 1…"), not terminal pacers. Session state
  persists across messages; the user can pause/resume by just replying.
- **Journaling is conversational:** after a session, WisdomKeeper asks
  what was experienced and stores it — no forms, just chat.
- **Voice notes:** where it fits, guidance can go out as voice notes via
  Devon's local XTTS stack (already runs on the owner's machine).
- **No terminal dependency:** nothing in the UX assumes a shell. The
  `nm wisdom` CLI exists for the operator, but the owner's path is chat.

**On a dedicated app:** yes, it's possible — but it's a separate,
much larger build (mobile client + backend + hosting + store
distribution). The honest sequencing is: (1) WisdomKeeper as a Devon
module reachable through the existing Telegram/WhatsApp bot — this is
where the owner actually lives; (2) later, if the chat experience
proves the value, wrap it as an app (a PWA or a thin native shell
around the same backend). Building the app first would be putting the
cart before the horse.
