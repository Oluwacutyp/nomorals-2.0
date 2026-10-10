# DOCUMENTS Sweep — External Mining Report

Module: `nomorals/documents/` (10 files, ~2,430 lines). Method: mine the
best implementation of every significant class OUTSIDE the repo, then merge
the gold into our classes. This report is written BEFORE any code.

Sources mined: docling / DoclingDocument (IBM), Microsoft markitdown,
pandoc, OCRmyPDF, sumy (LSA/LexRank/Luhn/TextRank), pytextrank, KeyBERT /
YAKE / RAKE, Meilisearch, Tantivy, Manticore, Camelot (stream/lattice/network/ml),
pdfplumber, DeepDiff, google/diff-match-patch, dandavison/delta,
WeasyPrint, mammoth.

---

## 1. `model.py` — Document / Section / Table

### How the best do it
- **docling's DoclingDocument**: a unified tree + flat-list dual view.
  Tree (`body` → `SectionHeaderItem` → `TextItem`/`TableItem`/`ListItem`/
  `PictureItem`/`CodeItem`/`FormulaItem`/`CaptionItem`) preserves reading
  order; flat lists (`doc.Texts`, `doc.Tables`, `doc.Pictures`) index by
  type. Every item carries `prov` — provenance with **page number + bbox** —
  and there is top-down `iterate_items()`. Chunkers are first-class:
  `HierarchicalChunker` (structure-aware, heading context inherited) and
  `HybridChunker` (structure + tokenizer `max_tokens`, `merge_peers`).
- **LangChain Document**: `page_content` + `metadata` dict — metadata is
  the escape hatch (source, page, row). Splitters live outside the model.
- **Unstructured**: typed `Element`s (Title, NarrativeText, Table, Image,
  ListItem…) each with `ElementMetadata` (page_number, coordinates,
  filename, languages).
- **markitdown**: deliberately lossy — one markdown string. Structure is
  the model's job downstream.

### What ours lacks (gold to take)
1. **Provenance**: our `Section` has no page number. Docling proves page
   provenance is load-bearing for citations ("page 4 says…").
2. **Item kinds**: we flatten lists, code, quotes, captions into prose.
   Unstructured/docling keep `kind` — lets renderers and chunkers treat
   them differently.
3. **Structure-aware chunking**: docling's HierarchicalChunker is the
   RAG gold standard; we have nothing. A `Document.chunks(max_words)`
   that respects section boundaries + heading context is directly portable.
4. **Identity**: DeepDiff's `DeepHash` idea — a `content_hash()` for
   dedup/change detection across parses.
5. **Table smarts**: docling `TableItem` tracks header vs body cells and
   row/col spans; ours is header+rows of strings with no confidence,
   caption, or page.
6. **Document-level conveniences**: word count, reading time, outline
   (TOC), `to_json`, section lookup, doc merge.

### Plan
Extend (never replace): `Section(page=0, kind="text")`, `Table(caption,
page, confidence, to_records(), column(), stats())`, `Document` gains
`outline()`, `word_count()`, `reading_time_minutes()`, `chunks()`,
`content_hash()`, `find_sections()`, `merge()`, `to_json()`.

---

## 2. `parsers.py` — 14 format parsers, stdlib-only

### How the best do it
- **markitdown**: 15+ formats → markdown. Notable extras over us: **JSON**,
  **XML**, **XLS** (legacy), **ZIP** (iterates contents), images (EXIF),
  audio transcription, YouTube. Architecture: one `convert_*` per format,
  narrowest-function security guidance, CLI + pipe-friendly.
- **pandoc**: readers → AST → writers. The lesson is *modularity*: each
  reader only builds the AST; writers are independent. Our `_PARSERS`
  dispatch already mirrors this — good.
- **docling PDF pipeline**: char positions + fonts → layout model → table
  model → reading order. Out of reach stdlib-only, but the *principle*
  (detect native vs scanned per page, route accordingly) is portable —
  sharkitect's hybrid-detection checklist says exactly this.

### What ours lacks (gold to take)
1. **JSON / XML parsers** — markitdown has both; we raise "unsupported".
   Real gap: config dumps, API responses, RSS.
2. **RTF tables** — our RTF parser drops all table structure (`\trowd`
   / `\cell` / `\row` are parseable stdlib-only). Currently a real data-loss
   hole.
3. **docx lists** — numbered/bulleted lists flatten to bare paragraphs;
   markitdown preserves `- `/`1. ` markers. Also: hyperlinks (URL lost),
   footnotes (in `footnotes.xml`, dropped), `Title`/`Subtitle` styles.
4. **CSV robustness** — only utf-8(+sig); markitdown-grade would try
   cp1252/latin-1 fallback instead of dying on Windows-encoded files.
5. **OOXML extras**: merged-cell awareness in xlsx (forward-fill like
   pandas), docx `w:hyperlink`, ODT list items already walked but not
   marked as lists.
6. **HTML metadata**: `<meta name=description>`, `lang` attribute, keywords
   → metadata (free, we already parse the DOM).
7. **EPUB**: `dc:language`, `dc:subject` ignored.
8. **Public `detect_format()`** — `_sniff` is private; pandoc-style UX
   wants it exposed.
9. **XLS (OLE)**: markitdown supports via xlrd (optional dep). Ours could
   sniff OLE magic and fail fast with the exact install hint instead of
   "unrecognized binary input".

### Plan
Add `json` + `xml` parsers (stdlib), RTF table recovery, docx list
markers + hyperlink capture + footnotes + Title style, CSV encoding
fallback chain, HTML/EPUB/ODT metadata harvesting, public
`detect_format()`, OLE sniff → optional-`xlrd` path with fail-fast hint.
No new mandatory deps — the module stays stdlib-first.

---

## 3. `index.py` — DocumentIndex (FTS5/BM25)

### How the best do it
- **Meilisearch**: typo tolerance (1–2 typos by term length), faceted
  search, synonyms, stop words, prefix search, ranking rules, fielded
  attributes, highlight (`attributesToHighlight`), hybrid
  (keyword+vector, `semanticRatio`), multi-search/federated,
  `similar` (embedding) search, query suggestions.
- **Tantivy**: natural/phrase queries, JSON field indexing, aggregations,
  range queries, 2× Lucene speed claims, tiny startup.
- **Manticore**: percolate (reverse search), 7 built-in rankers, 20+
  ranking factors, snippet/highlight builder.
- **SQLite FTS5 itself** (what we sit on): `highlight()`, `snippet()`,
  `bm25()` with tunable k1/b, column filters `{title} : foo`, phrase
  `"..."`, `NEAR`, prefix `foo*`, `highlight()` offsets via `offsets()`.

### What ours lacks (gold to take)
1. **Fielded search** — `title:foo` filters. FTS5 does this natively via
   `{title} : term`; we never expose it.
2. **Phrase search** — quoted phrases pass to FTS5 verbatim.
3. **Prefix search** — `foo*` fallback when exact terms miss (poor man's
   typo tolerance; Meilisearch-grade typo tolerance needs trigrams —
   out of scope, prefix is the honest stdlib move).
4. **Highlighting** — FTS5 `highlight()` gives `<mark>` snippets; ours
   hand-rolls a 120-char window with no marking.
5. **Facets** — `facet("format")` counts. Needs a format column in the
   meta table (schema v2→v3, backward-compatible load).
6. **AND operator** — currently OR-only.
7. **Synonym expansion** — query-time dict, cheap and real.
8. **Suggestions** — prefix completion from the FTS5 `vocab` table.
9. **`count()` / stats** — total hits, index stats (doc count exists).
10. **Ranking transparency** — Meilisearch's `showRankingScoreDetails`;
    expose per-term contributions? FTS5 `bm25()` per-column is enough
    for a `explain` flag.

### Plan
`search(query, fields=…, operator="OR"|"AND", prefix=False,
highlight=False, synonyms=…, explain=False)`, `count()`, `suggest()`,
`facet()`, `stats()`. Schema v3 adds `format` column; v2 files still load.

---

## 4. `compare.py` — diff_documents / compare_documents

### How the best do it
- **DeepDiff**: recursive diff of any object, tree vs text views,
  `exclude_paths`, `ignore_order`, significant-digits, **Delta objects**
  (git-like diff you can *apply*), serialization to JSON, DeepHash,
  DeepSearch.
- **diff-match-patch** (Google): `diff_main` + `diff_cleanupSemantic` —
  semantic cleanup turns char-soup into human-readable word-aligned
  diffs; word-mode diff; semantic-lossless mode that aligns edits to
  word boundaries. The merge-pro spec calls this the IntelliJ-style
  inline-highlight gold standard.
- **delta (dandavison)**: syntax-highlighted, side-by-side, line-numbered
  terminal diff UX — the presentation bar.
- **pandas `DataFrame.compare`**: cell-level before/after alignment —
  exactly what our table diffs should look like.

### What ours lacks (gold to take)
1. **Word-level inline diff** — we only do line-level unified diff.
   diff-match-patch's semantic cleanup is portable in ~60 lines stdlib.
2. **Renderers** — no HTML report (difflib.HtmlDiff exists but is ugly;
   delta-style is the bar), no colored terminal output, no markdown
   report. Presentation is currently "functional".
3. **Moved/renamed detection** — a section moved verbatim shows as
   add+remove; fuzzy matching (ratio ≥ 0.9) should report "moved".
4. **Cell-level table diff** — "2 rows changed" is weak vs
   `DataFrame.compare`'s per-cell before→after. We have both grids;
   align rows and report changed cells.
5. **Similarity score** — 0..1 document similarity (SequenceMatcher
   ratio) for "how different are these?" at a glance.
6. **Change-type detail on sections** — for changed sections, attach the
   inline word diff, not just the heading name.

### Plan
`word_diff(a, b)` with semantic cleanup (stdlib), `similarity()`,
`render_html()` (styled side-by-side report), `render_terminal()`
(ANSI delta-style), `render_markdown()`, moved-section detection,
cell-level table change details in `stats`.

---

## 5. `convert.py` — to_markdown / to_text / to_html / to_pdf / to_csv

### How the best do it
- **pandoc**: 40+ formats both directions, templates (`-V` variables),
  filters (Lua/AST), standalone docs, TOC, metadata blocks, reference
  docs. The lesson: converters need *options* (themes, templates), not
  just functions.
- **WeasyPrint**: HTML+CSS → PDF with real paged-media styling —
  the bar for "HTML that looks like a document".
- **mammoth** (docx→HTML): style maps — configurable, not hardcoded.

### What ours lacks (gold to take)
1. **to_html is unstyled** — no CSS at all. God-tier bar: embedded
   themes (light/dark/print/minimal), TOC with anchors, section
   numbering, responsive tables, code-block styling.
2. **No JSON export** — `to_json(doc)` (pretty) is table stakes for an
   engine whose model round-trips dicts.
3. **to_csv does one table** — add `to_csv_all()` and name-based
   selection.
4. **No docx/epub writers** — pandoc writes both directions. docx via
   optional `python-docx` (fail-fast hint, mirrors parser); EPUB is
   stdlib-only (zip) and very doable.
5. **Markdown front-matter** — YAML `---` block with title/author/date
   (pandoc metadata blocks).
6. **TOC option** for markdown + HTML.

### Plan
`to_html(doc, theme="light"|"dark"|"print"|"minimal", toc=True)`,
`to_markdown(doc, front_matter=True, toc=False)`, `to_json()`,
`to_csv_all()`, `to_docx()` (optional dep), `to_epub()` (stdlib zip).
Keep every existing signature backward-compatible.

---

## 6. `ocr.py` — ocr_pdf / ocr_image

### How the best do it
- **OCRmyPDF** (28k★): the OCR workflow gold standard — rasterize →
  OCR → embed **hidden text layer** under the image (searchable PDF/A),
  `--rotate-pages`, `--deskew`, `--jobs N` parallel pages, lossless
  mode, image optimization, sidecar text output, 100+ languages,
  validation of input/output.
- **Surya**: layout + line-level OCR with bboxes in one model.
- **Tesseract hOCR**: word-level bboxes + confidences as structured
  output (`image_to_data`, `image_to_hocr`).

### What ours lacks (gold to take)
1. **Word-level output** — we return page strings; Tesseract already
   gives `image_to_data` (word, conf, bbox). Expose `ocr_pdf_words()`.
2. **hOCR export** — `image_to_hocr` is one call; researchers need it.
3. **Parallel pages** — OCRmyPDF's `--jobs`: ThreadPoolExecutor over
   pages (Tesseract releases the GIL in C++), big win on multi-page.
4. **Progress callback** — `on_page(n, total)` for UX.
5. **Engine knobs** — `psm`/`oem` passthrough (pytesseract config),
   not hardcoded defaults.
6. **Auto-rotate** — Tesseract OSD orientation detection per page
   (guarded: OSD traineddata often missing → best-effort skip).
7. **Searchable-PDF output** — full OCRmyPDF parity needs a PDF writer
   that embeds invisible text; our `core.pdf` writer may not support
   invisible text layers. Investigate; if unsupported, ship the sidecar
   (text + word bboxes) which is the portable half.

### Plan
`ocr_pdf_words()`, `ocr_pdf_hocr()`, `jobs=` parallelism,
`on_page=` callback, `psm=`/`oem=`/`config=` passthrough,
`auto_rotate=` (best-effort OSD). Keep fail-fast install hints.

---

## 7. `pdf_tables.py` — extract_text_tables

### How the best do it
- **Camelot**: three strategies — `lattice` (ruling lines, deterministic),
  `stream` (whitespace, what we do), `network`, plus `ml`
  (TableTransformer: structure from model, text from PDF — never invents
  values). Per-table **accuracy + whitespace metrics**,
  `parsing_report`, multi-format export (CSV/JSON/Excel/HTML/Markdown/
  SQLite), visual debugging.
- **pdfplumber**: `text`/`lines` strategies with `x_tolerance`,
  `y_tolerance` tuning; explicit table settings dict.
- Production checklists: **fallback chain** (lattice → stream),
  **confidence threshold** (flag < 0.7 for review), caption + page
  metadata prepended, merged-cell span expansion.

### What ours lacks (gold to take)
1. **Confidence scores** — Camelot's accuracy/whitespace metrics let
   pipelines filter junk. Portable: score = column-alignment consistency
   + cell-fill ratio. Attach to `Table.confidence`.
2. **Multi-pass gap tuning** — we hardcode 3+ spaces; pdfplumber tunes
   tolerances. Try gaps {2,3,4,5}, keep the best-scoring table set.
3. **Lattice-ish detection** — box-drawing/ASCII ruling lines
   (`─ │ ┌ ┐ ├ ┤ ┬ ┴ ┼`, `+---+`) delimit real tables in text dumps;
   strip them and split on `│`/`|`. Cheap, high-precision.
4. **Caption capture** — line above a block matching `Table \d+[:.]`
   becomes the table name (checklist item).
5. **Header validation** — currently first row is *assumed* header;
   add `has_header="auto"` heuristic (first row mostly non-numeric →
   header, else generate `col_1…`).

### Plan
`extract_text_tables(..., strategy="auto", score=True)` returning
confidence-scored tables; ruling-line pre-pass; caption capture;
multi-pass gap selection. Conservative stays the default.

---

## 8. `summarize.py` — summarize / summarize_text / keywords

### How the best do it
- **sumy**: 7 algorithms — LSA (concept coverage via SVD), LexRank
  (IDF-cosine PageRank, multi-doc strength), Luhn (significant-word
  clusters, fully deterministic), TextRank (PageRank on similarity
  graph), Edmundson, KL, SumBasic. The offline-ensemble pattern (run
  4, merge by Jaccard dedup) is the production move.
- **IBM's tutorial guidance**: Luhn for keyword summaries, LexRank for
  structured text, LSA for concept overviews.
- **pytextrank**: adds lemma graphs, topic-ranked phrases.
- **YAKE/RAKE/KeyBERT**: key*phrase* extraction (multi-word), not just
  words — YAKE is statistical (fast, no embeddings), KeyBERT semantic
  (slow, needs model), RAKE fastest.

### What ours lacks (gold to take)
1. **Only one algorithm** (TF density). TextRank is stdlib-portable
   (cosine similarity + power iteration) and beats TF on coherence;
   Luhn is ~30 lines and fully deterministic; lead-baseline for news.
   → `method="textrank"|"luhn"|"tf"|"lead"`.
2. **Redundancy** — near-duplicate sentences both get picked. MMR-style
   diversity penalty is the standard fix.
3. **Position bias** — real summarizers weight lead sentences; ours
   doesn't.
4. **Keyphrases** — `keywords()` returns single words; RAKE-style
   multi-word phrases are what users actually want. Portable stdlib.
5. **Per-section summaries** — long docs need a digest per section,
   not one global pick.
6. **Query-focused** — bias toward query terms (search-result
   summarization); trivially portable.
7. **TL;DR** — word-budget ultra-short summary.
8. **Abstractive hook** — optional LLM path via `nomorals.llm`
   (fail-fast when unavailable, extractive stays the default).

### Plan
`method=` (textrank/luhn/tf/lead), `diversity=True` MMR,
`position_bias=True`, `keyphrases()` (RAKE-style),
`summarize_sections()`, `summarize_query()`, `tldr()`,
`abstract=` optional LLM hook. All deterministic, stdlib-only by default.

---

## Cross-cutting style upgrades (god-tier presentation)
- `to_html` themes + TOC (convert.py) — documents should look designed.
- `compare.render_html` styled report + `render_terminal` delta-style
  colors (compare.py).
- Snippet `<mark>` highlighting (index.py).
- `bullet_digest` markdown output (summarize.py).
- Every new public function: docstring with example, fail-fast
  `DocumentError` on bad input, no silent empties (module law).
