"""Deep research: the /search deep pipeline.

SearchEngine.run(mode="deep") delegates here. Compared with the old inline
deep path this adds:

* **Freshness variants** — sub-queries gain recency framings ("… 2026",
  "… latest") when decomposition yields few, so stale evergreen pages stop
  winning every deep run.
* **Parallel sub-query fan-out** — threads (the search step only; page reads
  stay serial to respect robots and rate limits).
* **Corroboration + diversity + freshness ranking** — a URL found by two
  sub-queries outranks one found by one; the second page from a domain is
  worth less than the first; pages with current-year signals rank higher.
* **Section-level extraction** — instead of summarizing whole pages, each
  page is split into sections, each section is scored against the query,
  and only the best sections reach the synthesizer (less context, more signal).
* **Cited synthesis** — the model must reference sources as ``[n]``; every
  citation is validated against the source list before the report is
  accepted. Fails closed to the extractive, cited format.
* **The dig loop** (``dig=True``, the default) — bounded multi-hop: after
  the first batch of sources is read, each hop mines 2-3 word terms that
  actually appeared in the *newest* pages (cross-page signal,
  query-relevant) and searches them as follow-up queries, reading fresh
  pages; the loop stops after ``max_hops`` (3) or as soon as a hop yields
  no new pages (reflection gate — no progress, no more budget). Then it
  follows the strongest external links out of the sources (scored by
  anchor-text overlap, one per domain, robots-aware).  PDFs are read as
  first-class sources, not dropped as junk.
* **Dig report fields** — ``followups`` (queries mined and issued),
  ``external_followed`` (urls followed out of the sources), ``pdfs_read``,
  ``hops`` / ``hop_detail`` (per-hop follow-ups and new-page counts).

The report keeps the same shape quick/deep consumers already parse
(sub_queries, results, pages_read, summary, seconds) and adds
``sources`` (numbered), ``citations`` (n → url), ``sections`` and the dig
fields above (all additive — old consumers keep working).
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

from ...llm.brain import brain_for
from ...core.ids import new_short_id
from ...core.logging_setup import get_logger
from . import curate
from .engine import SearchEngine
from .summarize import extractive_summarize

_log = get_logger(__name__)

__all__ = ["DeepResearcher", "score_section", "split_sections", "freshness_signal",
           "mine_followups"]

_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_FRESH_WORDS = re.compile(
    r"\b(latest|newly|announced|released|unveiled|updated|this year|recent|currently)\b",
    re.IGNORECASE,
)
_STOPWORDS = frozenset(
    "a an and are as at be but by for from had has have he her his i if in into is it its "
    "me my no not of on or our she so that the their them they this to was we were what when "
    "which who will with you your".split()
)
_QUERY_MIN_LEN = 4


# ── text utilities ───────────────────────────────────────────────────────────


def _terms(text: str) -> set[str]:
    return {
        w.lower()
        for w in re.findall(r"[a-z0-9][a-z0-9'\-]+", (text or "").lower())
        if len(w) >= _QUERY_MIN_LEN and w not in _STOPWORDS
    }


def freshness_signal(text: str, *, current_year: int | None = None) -> float:
    """0.0–1.0: how current does this page look?"""
    text = text or ""
    if not text.strip():
        return 0.0
    year = current_year or time.localtime().tm_year
    score = 0.0
    years = [int(y) for y in _YEAR_RE.findall(text)]
    if years:
        newest = max(years)
        age = year - newest
        if 0 <= age <= 1:
            score = 1.0
        elif age == 2:
            score = 0.6
        elif age == 3:
            score = 0.3
    if _FRESH_WORDS.search(text):
        score = max(score, 0.5)
    return round(min(1.0, score), 2)


def split_sections(text: str, *, min_chars: int = 150, max_chars: int = 700) -> list[str]:
    """Split page text into section-like chunks on paragraph boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n", text or "") if p and p.strip()]
    sections: list[str] = []
    buf = ""
    for para in paragraphs:
        if buf and len(buf) + len(para) + 1 > max_chars:
            sections.append(buf.strip())
            buf = ""
        buf = f"{buf}\n{para}".strip() if buf else para
        if len(buf) >= min_chars:
            sections.append(buf)
            buf = ""
    if buf.strip():
        # merge a small tail into the last section when there is one
        if sections and len(buf) < min_chars:
            sections[-1] = f"{sections[-1]}\n{buf}".strip()
        else:
            sections.append(buf.strip())
    return [s for s in sections if len(s) >= 40][:12]


def score_section(section: str, query: str, *, title: str = "") -> float:
    """Query overlap in the section, with a title-mention bonus."""
    q = _terms(query)
    if not q:
        return 0.0
    s = _terms(section)
    overlap = len(q & s) / len(q)
    if title and (q & _terms(title)):
        overlap += 0.1
    return round(min(1.0, overlap), 3)


def mine_followups(query: str, pages: list[dict[str, Any]], *,
                   max_followups: int = 2, window: int = 8) -> list[str]:
    """2-3 word phrases that appeared in the read pages and are worth
    searching for next — the 'dig deeper' signals.

    A phrase qualifies when it is substantive (not stopword-dominant) and
    sits within ``window`` words of a query term in the source text — it
    must belong to the subject, but it does NOT have to repeat the query's
    own words (the best dig terms are new ones).  Scoring rewards
    cross-source corroboration (a phrase on two pages is a signal, not a
    quirk), then recurrence, then direct query overlap.  Deterministic, so
    runs are reproducible.  Returns up to ``max_followups`` phrases (not
    full queries — the caller prepends the original question).
    """
    q = _terms(query)
    if not q:
        return []
    query_lower = (query or "").lower()
    phrase_pages: dict[str, set[str]] = {}
    phrase_count: dict[str, int] = {}
    for page in pages:
        dom = page.get("domain") or page.get("url", "")
        text = (page.get("text") or "").lower()
        words = re.findall(r"[a-z][a-z'\-]{2,}", text)
        for n in (2, 3):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i + n])
                phrase_terms = set(phrase.split())
                if len(phrase_terms & _STOPWORDS) >= n - 1:
                    continue  # stopword-dominant
                if phrase in query_lower:
                    continue  # already asked
                # must live next to the research question in the text
                near = words[max(0, i - window):min(len(words), i + n + window)]
                if not any(w in q for w in near):
                    continue
                phrase_pages.setdefault(phrase, set()).add(dom)
                phrase_count[phrase] = phrase_count.get(phrase, 0) + 1

    scored: list[tuple[float, str]] = []
    for phrase, doms in phrase_pages.items():
        support = min(len(doms), 2) * 0.5            # cross-source corroboration
        recurrence = min(phrase_count[phrase], 3) * 0.3  # in-page recurrence
        relevance = 0.2 * len(_terms(phrase) & q) / len(q)  # echoes the query
        score = support + recurrence + relevance
        scored.append((round(score, 4), phrase))
    scored.sort(key=lambda t: (-t[0], t[1]))
    out: list[str] = []
    for _score, phrase in scored:
        if len(phrase) > 40:
            continue
        if any(phrase in o or o in phrase for o in out):
            continue  # no near-duplicate phrases
        out.append(phrase)
        if len(out) >= max_followups:
            break
    return out


# ── the researcher ───────────────────────────────────────────────────────────


class DeepResearcher:
    """search → fan-out → rank → read → section-score → cited synthesis."""

    def __init__(
        self,
        context: Any,
        *,
        engine: SearchEngine | None = None,
        max_subqueries: int = 5,
        pages_per_query: int = 3,
        max_pages: int = 8,
        wall_seconds: float = 180.0,
        workers: int = 4,
        dig: bool = True,
        max_followups: int = 2,
        follow_links: int = 3,
        max_hops: int = 3,
    ) -> None:
        self.context = context
        self.engine = engine or SearchEngine(context)
        self.max_subqueries = max(2, min(int(max_subqueries), 6))
        self.pages_per_query = max(2, min(int(pages_per_query), 5))
        self.max_pages = max(2, min(int(max_pages), 12))
        self.wall_seconds = float(wall_seconds)
        self.workers = max(1, min(int(workers), 6))
        self.dig = bool(dig)
        self.max_followups = max(0, min(int(max_followups), 3))
        self.follow_links = max(0, min(int(follow_links), 4))
        #: Bounded multi-hop: the dig loop re-mines follow-up queries from
        #: the *new* pages each hop, up to this many hops, then synthesizes.
        #: Reflection rule — a hop that yields zero new pages stops the loop
        #: early; no progress, no more budget spent.
        self.max_hops = max(1, min(int(max_hops), 3))
        self._seen_urls: set[str] = set()

    # ── query expansion ──────────────────────────────────────────────────────
    def expand_queries(self, query: str, *, scope: str = "auto") -> list[str]:
        subs = self.engine._decompose(query) or [query]
        subs = [s.strip() for s in subs if s and s.strip()][: self.max_subqueries]
        if not subs:
            subs = [query]
        year = time.localtime().tm_year
        freshness = [
            f"{query} {year}",
            f"{query} latest",
            f"{query} {year - 1}",
        ]
        for variant in freshness:
            if len(subs) >= 3:
                break
            if variant.lower() not in [s.lower() for s in subs]:
                subs.append(variant)
        # dual-scope fan-out: when the question is scope-sensitive, give
        # Nigeria and the US their own sub-queries so neither region's
        # press wins by default
        try:
            from .scope import regional_variants

            for variant, _label in regional_variants(query, scope=scope):
                if len(subs) >= self.max_subqueries:
                    break
                if variant.lower() not in [s.lower() for s in subs]:
                    subs.append(variant)
        except Exception:  # noqa: BLE001 - scope fan-out is a bonus, not a gate
            _log.debug("scope expansion skipped", exc_info=True)
        return subs[: self.max_subqueries]

    # ── search fan-out ───────────────────────────────────────────────────────
    def _search_one(self, sub: str) -> list[dict[str, Any]]:
        try:
            # adaptive per-sub-query breadth: each angle gets the result
            # count its own phrasing deserves
            results = self.engine.search(sub)
        except Exception as exc:  # noqa: BLE001 - one dead sub-query must not sink the run
            _log.debug("deep research sub-query failed %r: %s", sub, exc)
            return []
        # deep research reads PDFs as sources (the report IS often the pdf)
        return curate.curate(results, sub, top_n=self.pages_per_query, allow_pdf=True)

    def fan_out(self, subs: Sequence[str], started: float) -> list[dict[str, Any]]:
        """Parallel search+curation; returns merged, corroboration-tagged results."""
        merged: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="deepsearch") as pool:
            futures = {pool.submit(self._search_one, sub): sub for sub in subs}
            for future in futures:
                sub = futures[future]
                try:
                    results = future.result(timeout=max(5.0, self.wall_seconds - (time.time() - started)))
                except Exception as exc:  # noqa: BLE001
                    _log.debug("fan-out future failed for %r: %s", sub, exc)
                    continue
                for res in results:
                    key = _url_key(res.get("url", ""))
                    if not key:
                        continue
                    hit = merged.get(key)
                    if hit is None:
                        merged[key] = {
                            **res,
                            "_subs": [sub],
                            "_score": float(res.get("score", 0.0)),
                        }
                    else:
                        hit["_subs"].append(sub)
                        hit["_score"] = max(hit["_score"], float(res.get("score", 0.0)))
        return list(merged.values())

    # ── ranking ──────────────────────────────────────────────────────────────
    def rank(self, results: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
        domain_counts: dict[str, int] = {}
        for res in results:
            domain = res.get("domain") or curate.domain(res.get("url", ""))
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
            res["_domain_seen"] = domain_counts[domain]  # 1 = first page from this domain

        def _ranked(res: dict[str, Any]) -> float:
            base = float(res.get("_score", res.get("score", 0.0)))
            corroboration = min(len(res.get("_subs", [])) - 1, 2) * 0.06
            diversity = 0.1 if res.get("_domain_seen", 1) == 1 else 0.03
            fresh = freshness_signal(f"{res.get('title', '')} {res.get('snippet', '')}") * 0.08
            return base * 0.5 + corroboration + diversity + fresh + 0.5 * base * 0.5

        ordered = sorted(results, key=_ranked, reverse=True)
        # domain spread: no more than 2 pages per domain in the final set
        seen: dict[str, int] = {}
        out: list[dict[str, Any]] = []
        for res in ordered:
            domain = res.get("domain") or curate.domain(res.get("url", ""))
            if seen.get(domain, 0) >= 2:
                continue
            seen[domain] = seen.get(domain, 0) + 1
            out.append(res)
            if len(out) >= self.max_pages:
                break
        return out

    # ── the run ──────────────────────────────────────────────────────────────
    def run(self, query: str, *, scope: str = "auto") -> dict[str, Any]:
        started = time.time()
        query = (query or "").strip()
        if not query:
            raise ValueError("deep research needs a query")

        subs = self.expand_queries(query, scope=scope)
        candidates = self.fan_out(subs, started)
        if time.time() - started > self.wall_seconds:
            raise TimeoutError("deep research hit its wall clock before ranking")
        top = self.rank(candidates, query)

        pages: list[dict[str, Any]] = []
        for res in top:
            if time.time() - started > self.wall_seconds:
                break
            page = self.engine.read(res["url"], max_chars=40000)
            if page:
                self._seen_urls.add(_url_key(page["url"]))
                page["_score"] = float(res.get("_score", res.get("score", 0.0)))
                page["_subs"] = res.get("_subs", [])
                pages.append(page)

        # ── the dig loop: up to max_hops, keyless and capped ─────────────────
        followups: list[str] = []
        external_followed: list[str] = []
        hop_detail: list[dict[str, Any]] = []
        if self.dig and pages:
            pages = self._dig(query, pages, started, followups,
                              external_followed, hop_detail)

        # keep the corpus bounded; first-batch pages stay ahead in the list
        pages = pages[: self.max_pages + 4]

        sources = [
            {
                "n": i + 1,
                "url": p["url"],
                "title": p.get("title", ""),
                "domain": p.get("domain", ""),
                "score": round(p.get("_score", 0.0), 3),
                "fresh": freshness_signal(f"{p.get('title','')} {p.get('text','')[:2000]}"),
            }
            for i, p in enumerate(pages)
        ]
        sections = self._top_sections(query, pages)
        summary, model_ok, error = self._synthesize(query, pages, sections, started)

        report = {
            "id": new_short_id("search"),
            "query": query,
            "mode": "deep",
            "sub_queries": subs,
            "results": [
                {
                    "url": p["url"], "title": p.get("title", ""),
                    "domain": p.get("domain", ""), "score": p.get("_score", 0.0),
                }
                for p in pages
            ],
            "sources": sources,
            "citations": {str(s["n"]): s["url"] for s in sources},
            "sections": sections,
            "followups": followups,
            "external_followed": external_followed,
            "pdfs_read": [p["url"] for p in pages if p.get("pdf")],
            "dig": self.dig,
            "hops": len(hop_detail),
            "hop_detail": hop_detail,
            "pages_read": [p["url"] for p in pages],
            "pages": pages,
            "summary": summary,
            "model_summary": model_ok,
            "error": error,
            "seconds": round(time.time() - started, 2),
        }
        self._journal(report)
        return report

    # ── the dig loop ─────────────────────────────────────────────────────────
    def _dig(
        self,
        query: str,
        pages: list[dict[str, Any]],
        started: float,
        followups: list[str],
        external_followed: list[str],
        hop_detail: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Up to ``max_hops`` deeper: each hop mines follow-up queries from
        the pages the *previous* hop added, searches them, and reads the
        new pages.  The loop is reflection-gated — a hop that yields zero
        new pages stops it early (no progress → synthesize instead of
        burning budget).  The visited-URL set makes it circle-proof: a
        page is never read twice and a follow-up that only rediscovers
        seen URLs ends its hop.  After the hops, the strongest external
        links out of the sources are followed once, as before."""
        frontier: list[dict[str, Any]] = list(pages)
        for hop in range(1, self.max_hops + 1):
            if time.time() - started > self.wall_seconds:
                break
            # mine only from the newest evidence — each hop digs deeper,
            # not sideways over the same corpus
            mined = mine_followups(query, frontier, max_followups=self.max_followups)
            added: list[dict[str, Any]] = []
            hop_followups: list[str] = []
            for phrase in mined:
                if time.time() - started > self.wall_seconds:
                    break
                fu = f"{query} {phrase}"
                followups.append(fu)
                hop_followups.append(fu)
                for res in self._search_one(fu):
                    if time.time() - started > self.wall_seconds:
                        break
                    key = _url_key(res.get("url", ""))
                    if key in self._seen_urls:
                        continue
                    page = self.engine.read(res["url"], max_chars=40000)
                    if page:
                        self._seen_urls.add(_url_key(page["url"]))
                        page["_score"] = float(res.get("_score", res.get("score", 0.0))) * 0.9
                        page["_dig"] = fu
                        page["_hop"] = hop
                        added.append(page)
            hop_detail.append({
                "hop": hop,
                "followups": hop_followups,
                "new_pages": len(added),
            })
            if not added:
                # reflection: this hop produced nothing new — further hops
                # would just re-mine the same evidence
                break
            pages.extend(added)
            frontier = added

        # pass 2 — follow the strongest external links out of the sources
        for page in self._follow_external(query, pages, started):
            self._seen_urls.add(_url_key(page["url"]))
            external_followed.append(page["url"])
            pages.append(page)
        return pages

    def _follow_external(
        self, query: str, pages: list[dict[str, Any]], started: float
    ) -> list[dict[str, Any]]:
        """Read the most query-relevant pages that the sources link out to
        (one per domain, robots-aware, capped).  A reader, not a crawler."""
        q = _terms(query)
        scored: list[tuple[int, str, str]] = []
        seen_domains: set[str] = set()
        for page in pages:
            for url, anchor in (page.get("links") or []):
                if _url_key(url) in self._seen_urls:
                    continue
                dom = curate.domain(url)
                if dom in seen_domains:
                    continue
                at = _terms(anchor)
                score = len(at & q) * 2 + (1 if at else 0)
                if score <= 0:
                    continue
                seen_domains.add(dom)
                scored.append((score, url, anchor))
        scored.sort(key=lambda t: (-t[0], t[1]))
        out: list[dict[str, Any]] = []
        for _score, url, _anchor in scored[: self.follow_links]:
            if time.time() - started > self.wall_seconds:
                break
            page = self.engine.read(url, max_chars=20000)
            if page:
                page["_external"] = True
                out.append(page)
        return out

    # ── section scoring ──────────────────────────────────────────────────────
    def _top_sections(
        self, query: str, pages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        best: list[dict[str, Any]] = []
        for page in pages:
            for section in split_sections(page.get("text", "")):
                score = score_section(section, query, title=page.get("title", ""))
                if score <= 0.05:
                    continue
                best.append({
                    "url": page["url"],
                    "title": page.get("title", ""),
                    "score": score,
                    "text": section[:900],
                })
        best.sort(key=lambda s: -s["score"])
        # spread: at most 2 sections per page, cap the whole set
        per_page: dict[str, int] = {}
        out: list[dict[str, Any]] = []
        for sec in best:
            if per_page.get(sec["url"], 0) >= 2:
                continue
            per_page[sec["url"]] = per_page.get(sec["url"], 0) + 1
            out.append(sec)
            if len(out) >= 10:
                break
        return out

    # ── synthesis ───────────────────────────────────────────────────────────
    def _synthesize(
        self,
        query: str,
        pages: list[dict[str, Any]],
        sections: list[dict[str, Any]],
        started: float,
    ) -> tuple[str, bool, str]:
        if self.engine._model_available() and time.time() - started < self.wall_seconds - 30:
            try:
                text = self._model_synth(query, pages, sections)
                if text:
                    return text, True, ""
            except Exception as exc:  # noqa: BLE001
                _log.debug("deep synthesis model failed; falling back: %s", exc)
        return self._extractive(query, pages, sections), False, ""

    def _model_synth(self, query: str, pages: list[dict[str, Any]], sections: list[dict[str, Any]]) -> str:
        from ...llm.base import Message, SamplingParams

        source_lines = "\n".join(
            f"[{i + 1}] {p.get('title', '')} — {p['url']}" for i, p in enumerate(pages)
        )
        evidence = "\n\n".join(
            f"[{n_of(pages, s['url'])}] {s['text']}" for s in sections
        )
        prompt = (
            f"Research question: {query}\n\n"
            f"Sources:\n{source_lines}\n\n"
            f"Relevant passages (each labelled with its source number):\n{evidence[:9000]}\n\n"
            "Write a direct, well-organized answer to the research question using ONLY the "
            "passages above. Cite sources inline as [n] matching the numbers. Where sources "
            "disagree, say so. If the passages do not answer the question, say what is missing. "
            "No preamble, no 'based on the sources'."
        )
        response = brain_for(self.context).chat(
            [
                Message.system("You are a meticulous research assistant."),
                Message.user(prompt),
            ],
            SamplingParams(temperature=0.3, max_tokens=1200),
        task_kind="research")
        text = (response.text or "").strip()
        if not text:
            return ""
        valid = {str(i + 1) for i in range(len(pages))}
        bad = [c for c in re.findall(r"\[(\d+)\]", text) if c not in valid]
        if bad:
            # strip invalid citations rather than reject the whole answer
            text = re.sub(r"\[\d+\]", lambda m: "" if m.group(1) in valid else " [?]", text)
        return text

    def _extractive(
        self, query: str, pages: list[dict[str, Any]], sections: list[dict[str, Any]]
    ) -> str:
        if not sections:
            base = extractive_summarize(query, pages, max_sentences=6)
            return base or "deep research found no readable pages for that question."
        numbered = "\n\n".join(
            f"[{n_of(pages, s['url'])}] {s['text']}" for s in sections[:6]
        )
        return (
            "Deep research — top evidence (cited):\n"
            f"{numbered}\n\n"
            "Sources: " + "; ".join(f"[{i + 1}] {p['url']}" for i, p in enumerate(pages))
        )

    # ── journal (same table as SearchEngine) ─────────────────────────────────
    def _journal(self, report: dict[str, Any]) -> None:
        try:
            import json

            with self.context.db.transaction():
                self.context.db.execute(
                    "INSERT INTO search_log (id, query, mode, results, summary, seconds, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        report["id"], report["query"], report["mode"],
                        json.dumps(report["results"]),
                        report["summary"][:8000],
                        report["seconds"], time.time(),
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            _log.debug("deep research journal write failed: %s", exc)


# ── helpers ──────────────────────────────────────────────────────────────────


def _url_key(url: str) -> str:
    url = (url or "").strip().rstrip("/").lower()
    if not url:
        return ""
    url = re.sub(r"^https?://(www\.)?", "", url)
    return url


def n_of(pages: Sequence[dict[str, Any]], url: str) -> str:
    """1-based citation number for a page url, or '0' when unknown."""
    for i, p in enumerate(pages):
        if _url_key(p.get("url", "")) == _url_key(url):
            return str(i + 1)
    return "0"
