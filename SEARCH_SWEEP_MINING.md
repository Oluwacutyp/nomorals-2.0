# SEARCH_SWEEP_MINING.md — module `nomorals/search/` (11 files)

Mined 2026-10-10. Method: for every significant class in this module, find how
the *best* outside implementations do it — GitHub repos, papers, production
write-ups, best-in-class tools AND the trash. This report is written before any
code (per sweep method). Sources are cited; claims below marked
"verified-via-docs" come from the projects' own docs/readmes.

## 1. Fusion — `reciprocal_rank_fusion` (model.py) vs the field

**The paper:** Cormack, Clarke & Büttcher, SIGIR 2009 — RRF = Σ 1/(k + rank),
k=60 fixed in the pilot study and never changed; MAP barely moves as k goes
0→500. "Agreement outranks confidence": a doc ranked 2nd by two retrievers
beats a doc ranked 1st by one.

**Best implementations agree:**
- Elasticsearch / OpenSearch hybrid: RRF `rank_constant=60` default, docs say
  "RRF requires no tuning, and the different relevance indicators do not have
  to be related to each other" (verified-via-docs).
- Qdrant hybrid queries, Vespa hybrid tutorial — same k=60 default.
- Pure-Python reference (dev.to/royalpinto007): `scores[doc] += 1/(k+rank)`,
  sort desc. Our implementation matches this exactly.
- Two 2026 production reports (dev.to/aws_sa_sg; towardsdeeplearning.com):
  RRF beat min-max score normalization + α-weighted fusion on every metric
  (+11.2 recall points), and weighted α had to be re-tuned per data slice
  while k=60 needed nothing. One report measured: "A document at rank 1 in
  either list contributes 1/61... agreement between the two searches is
  rewarded."

**What the best do that we don't (gaps → implement):**
1. **Weighted RRF** — Elasticsearch supports per-list weights. Our
   `reciprocal_rank_fusion(ranked_lists)` has no weights; a trusted engine
   (SearXNG aggregate) can't be trusted more than a long-tail one.
2. **Original rank preserved in metadata** — katalalab's research doc
   recommends "preserve each engine's original rank in metadata" for
   debugging and second-stage rerankers. We overwrite `raw_score` and lose it.
3. **RRF as the default** — every modern hybrid stack fuses first with RRF and
   reranks second. Our default is legacy min-max normalization (the fragile
   approach the dev.to report explicitly warns about: "a single outlier
   document can quietly distort every query in the batch"). The legacy default
   stays for backward compat (fleet-wide callers), but RRF gets weights +
   rank metadata so it's the obviously-right opt-in.

## 2. Rerank — `rerank.py` (BM25) vs rank-bm25 / Okapi canon

**Canon:** Okapi BM25, k1 ∈ [1.2, 2.0], b=0.75, IDF = log(1 + (N−df+0.5)/(df+0.5)).
Our `bm25_scores` implements this correctly, including the smoothed
`log(1+…)` IDF that never goes negative (same as rank-bm25's
`BM25Okapi.idf` with epsilon). Tokenizer `[a-z0-9]+`, min-len 2 — matches the
rank-bm25 docs' own recommendation to handle tokenization yourself.

**What the best do that we don't:**
1. **Field-weighted BM25 (BM25F pattern)** — best practice in every web
   reranker: title matches count ~2× snippet matches. rank-bm25 users
   hand-concatenate title twice; we should do it explicitly with a
   `title_weight` param instead.
2. **BM25+ / BM25L** — fix BM25's penalty of long documents (our web snippets
   vary wildly in length). BM25+ adds δ (typically 1.0) so long-doc scores
   don't collapse to zero.
3. **k3 (query-term saturation)** — minor; expose it for parity with Okapi
   reference implementations (default k3=∞ → no-op).
4. **Stemming/stopword hooks** — the "trash" builds hand-roll Porter stemmers;
   the best (rank-bm25) leave preprocessing to the caller and just expose a
   `tokenizer` seam. We'll expose the seam, not a stemmer.

## 3. Web backends — `web.py` vs SearXNG programmatic best practice

**Best implementations** (searxngr CLI, codestacker SearXNG skill,
agentx-workmate SearXNG skill — all verified-via-docs) use the full SearXNG
query surface, not just `q`:
- `categories=news|science|it|videos|images|map|"social media"` — per-query
  vertical routing (news queries → news engines).
- `time_range=day|week|month|year` — freshness for news queries.
- `engines=google,bing` — pinning per query; `language=`; `pageno=` paging.
- safesearch, limit.

Our `SearXNGWebSource._fetch` sends only `q/format/language/safesearch/pageno`.
**Gaps:** category, time_range, engines, and language pinning are missing — a
"latest AI news" query currently gets general-web engines with no recency
filter. Fix: env/config passthrough (`NM_SEARXNG_CATEGORY`,
`NM_SEARXNG_TIME_RANGE`, `NM_SEARXNG_ENGINES`, `NM_SEARXNG_LANGUAGE`).

**ddgs:** the `ddgs` package has `text/news/images/videos` methods; our adapter
only calls `text`. Gap: `NM_DDGS_KIND` to route news/image/video verticals.

## 4. OSINT — `osint.py` / `osint_browser.py` vs the arsenal

**Mined from:** awesome-osint-arsenal, osint-investigator-v3 completeness
scorecard (2026-06, grades per capability), XposedOrNot API docs, ethical
trackers.

**The scorecard grades (verified-via-docs):**
- Username enum: sherlock (400+ sites) + maigret (3000+ sites) = **A** when
  used together. We hand-rolled 25 sites with brittle not-found markers —
  that's the trash-tier approach (markers rot; sites change 404 pages).
  Gold: delegate to the `sherlock-project` / `maigret` CLIs when installed
  (subprocess, `--print-found`, JSON output), keep the hand-rolled 25-site
  sweep as the zero-dependency fallback.
- Email → accounts: **holehe is unmaintained/dead** (last release 2023; 120+
  site modules rotting). Our EmailCheckAdapter doesn't even use it — it does
  an MX check + Hudson Rock. Fine, but the breach half has a better free
  source.
- **Breach: XposedOrNot = the free gold.** `GET
  https://api.xposedornot.com/v1/check-email/{email}` — no API key, free
  forever for personal use, returns breach names; `/v1/breach-analytics`
  returns exposure counts; `/v1/check-password` uses k-anonymity (local hash,
  partial hash sent). Rate limits: 2/s, 25/hr, 100/day per endpoint — the
  adapter must respect 429 + Retry-After. HIBP has had **no free tier since
  2024** — don't attempt it keyless. This is a real missing adapter.
- Domain: crt.sh + RDAP = **A−** already covered. Subfinder/Amass are Go
  binaries — optional subprocess delegation, note only.
- IP: ipwho.is + Shodan InternetDB = **A** already covered.
- Phone: phonenumbers lib = the free gold (carrier/region/valid/type) —
  already used. Ignorant CLI exists but unmaintained; skip.
- **Company/corporate: our gap (scorecard C+).** Free gold: **SEC EDGAR**
  (free, no key — company filings search) and **GLEIF** (free LEI API —
  legal entity lookup). Neither needs a key. Add both.
- **People/identity structured: our gap (scorecard D).** Free gold:
  **CourtListener** (free legal API — court opinions mentioning a name).
  Add it.
- GitHub recon: unauthenticated `api.github.com/users/{u}` = the gold —
  already covered; extend with repo language stats (have) — fine.
- Gravatar md5 → profile: gold for email→identity pivot — already covered.
- Wayback: `archive.org/wayback/available` — already covered; the CDX API
  (`web.archive.org/cdx/search/cdx`) gives full capture lists — extend.
- Email validation: **disify** free no-key email API (format/mx/disposable)
  — cheap addition, fills the "is this address real" gap.

**Correctness bugs found (must fix):** every adapter added after the first
four constructs `SearchResult(source=…, title=…, url=…, snippet=…, score=…)`
— `url` is not a dataclass field and `query`/`type` are required positionals,
so **all of them raise TypeError at runtime** (fail-fast wraps it, but the
sources are dead). Also `build_adapters()` never builds the OSINT adapters, so
`federated_search(sources=["osint_username"])` raises "no adapter built" —
the whole OSINT layer is unreachable through the standard path.

## 5. Adaptive limits — `adaptive.py` vs query understanding

Our heuristic (length + question-starters + breadth hints + compare regex) is
the cheap-deterministic tier — the right call for "how many results". What's
missing around it (mined from search UX research + SearXNG skills):
1. **Freshness intent** — queries with "latest/news/2026/this week" want
   recency-biased results; no signal exists. Add `freshness_intent(query)`.
2. **Query normalization** — strip junk (repeated punctuation, extra
   whitespace) before sending to backends. Add `normalize_query`.
3. **Query-shape routing** — the OSINT adapters already do shape routing
   (`@user` / email / domain / IP); the same idea generalizes to a tiny
   `detect_intent` (navigational vs informational vs transactional) used to
   pick categories. Keep heuristic, offline, deterministic.

## 6. Presentation — no module exists; everyone else has one

**Mined from** searxngr (colorized terminal output, INI-configured themes),
web-patterns SKILL.md (search-results page spec), ia-search-findability
SKILL.md:
- Count first ("N results for 'q'"), result item = title with **match
  highlighted** + context snippet around the match + type/date meta,
  source badges, applied filters as chips, grouped/faceted views.
- **No-results is the most-designed state**: "No matches for 'x'" +
  suggestions + "remove a filter" naming the filters.
- Terminal: colorized, section-grouped; no-TTY safe (plain fallback).

**Gap:** the module has zero presentation — `SearchResponse.to_dict()` only.
Add `render.py`: `render_search(response, style=…)` with styles
`rich` (ANSI color, badges, highlighted matches), `compact` (one line/hit),
`plain` (no-TTY), `markdown`. Grouped-by-source sections optional. This is the
god-tier presentation requirement of the sweep.

## 7. federated.py — fan-out vs metasearch best practice

Our fan-out (probes sequential → parallel thread pool, canonical-order error
collection, per-source skip notes, fail-fast with source named) matches the
katala-web-research "meta fan-out with per-engine failure isolation" pattern.
**Gaps worth adding:**
1. **Per-source latency** in the response (`timings`) — every metasearch
   reports per-engine latency; operators need it. Additive field, no breakage.
2. **Weighted RRF** plumbed from `federated_search(source_weights=…)`.
3. **Original per-source rank in provenance** (`provenance["source_rank"]`)
   for RRF debugging.

## What stays

- RRF k=60, legacy default preserved (fleet callers depend on it).
- Zero-mandatory-deps promise: stdlib only; ddgs/sherlock/maigret/phonenumbers
  are optional with graceful skip notes.
- Fail-fast/fail-open split (errors named + chained; telemetry fail-open).
- Canonical source order as ranking tie-break.
- Keyless-first backend ordering (verified 2026: Bing API retired 2025-08,
  Google CSE closed to new customers, Brave free tier gone 2026-02 — our
  exclusions are correct).

## Class-by-class verdict

| class | external best | verdict |
|---|---|---|
| reciprocal_rank_fusion | Cormack et al. + ES/OpenSearch/Qdrant | correct; add weights + rank metadata |
| bm25_scores/rerank | rank-bm25, Okapi canon | correct; add field weights, BM25+, k3, tokenizer seam |
| SearXNGWebSource | searxngr / SearXNG skills | missing categories/time_range/engines/language |
| DdgsWebSource | ddgs package | missing news/images/videos verticals |
| federated_search | katala meta fan-out | missing timings, weighted RRF, rank metadata |
| OSINT adapters (9) | sherlock/maigret/XposedOrNot/EDGAR/GLEIF/CourtListener | **broken construction** (fix); delegate to CLIs when present; add XposedOrNot, SEC EDGAR, GLEIF, CourtListener, disify, CDX |
| Browser OSINT (2) | Clearfront pattern | broken construction (fix) |
| adaptive_result_limit | query-understanding heuristics | extend: freshness intent, normalize, intent detect |
| (new) render.py | searxngr colorized output, search UX specs | **missing entirely** — add |
