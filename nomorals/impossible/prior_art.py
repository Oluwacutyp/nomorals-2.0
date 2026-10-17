"""Stage 4 — PRIOR ART. The missing organ, built as real plumbing.

Verified endpoints (no key needed unless noted):
- OpenAlex: GET https://api.openalex.org/works?search=...&per-page=N&sort=cited_by_count:desc — free, no key.
- arXiv: GET https://export.arxiv.org/api/query?search_query=all:...&max_results=N — Atom XML, ~4 req/s.
- Semantic Scholar: GET https://api.semanticscholar.org/graph/v1/paper/search?query=...&fields=title,abstract,year,citationCount,externalIds,url
- PubMed (E-utilities): esearch + esummary at eutils.ncbi.nlm.nih.gov — free, no key.
- Crossref: GET https://api.crossref.org/works?query=...&rows=N — free, no key.
- PubChem PUG REST: GET https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/<name>/property/.../JSON — free, no key.
- USPTO PatentsView: GET https://api.patentsview.gov/patents/query?q=... — free, no key for basic queries.
- EPO OPS / WIPO PATENTSCOPE / Google Patents: no free public API without a key
  (EPO OPS needs OAuth client credentials; PATENTSCOPE needs a key for API).
  These return honest "unavailable without key" entries — never fake results.
- Failed-attempt query: every source is also queried with failure terms
  (abandoned / failed / negative results / discontinued), because knowing
  where people stopped is worth more than a literature review of successes.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

UA = "nomorals-impossible/1.0 (prior-art; contact: owner)"
TIMEOUT = 15


def _get_json(url: str) -> dict | list | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def _get_text(url: str) -> str | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def search_openalex(query: str, limit: int = 5) -> list[dict]:
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"search": query, "per_page": max(1, min(limit, 25)), "sort": "cited_by_count:desc"})
    data = _get_json(url)
    if not isinstance(data, dict):
        return [{"source": "OpenAlex", "status": "unavailable",
                 "reason": "No response from OpenAlex (network down or rate-limited). No results invented."}]
    out = []
    for w in (data.get("results") or [])[:limit]:
        out.append({"source": "OpenAlex", "status": "found", "title": w.get("title"),
                    "year": w.get("publication_year"), "citations": w.get("cited_by_count"),
                    "doi": w.get("doi"), "url": w.get("doi") or w.get("id")})
    return out or [{"source": "OpenAlex", "status": "none", "reason": f"No OpenAlex works found for '{query}'."}]


def search_arxiv(query: str, limit: int = 5) -> list[dict]:
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": f"all:{query}", "start": 0, "max_results": max(1, min(limit, 25))})
    text = _get_text(url)
    if text is None:
        return [{"source": "arXiv", "status": "unavailable",
                 "reason": "No response from arXiv export API. No results invented."}]
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return [{"source": "arXiv", "status": "unavailable", "reason": "arXiv returned unparseable XML."}]
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    out = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns)
        link = entry.find("atom:id", ns)
        summary = entry.find("atom:summary", ns)
        out.append({"source": "arXiv", "status": "found",
                    "title": (title.text or "").strip().replace("\n", " ") if title is not None else "",
                    "url": link.text if link is not None else "",
                    "abstract": ((summary.text or "").strip()[:300] if summary is not None else "")})
    return out or [{"source": "arXiv", "status": "none", "reason": f"No arXiv papers found for '{query}'."}]


def search_semantic_scholar(query: str, limit: int = 5) -> list[dict]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search?" + urllib.parse.urlencode(
        {"query": query, "limit": max(1, min(limit, 25)),
         "fields": "title,abstract,year,citationCount,externalIds,url"})
    data = _get_json(url)
    if not isinstance(data, dict):
        return [{"source": "Semantic Scholar", "status": "unavailable",
                 "reason": "No response from Semantic Scholar (rate limit 100 req/5min unauthenticated). No results invented."}]
    out = []
    for p in (data.get("data") or [])[:limit]:
        out.append({"source": "Semantic Scholar", "status": "found", "title": p.get("title"),
                    "year": p.get("year"), "citations": p.get("citationCount"), "url": p.get("url"),
                    "abstract": (p.get("abstract") or "")[:300]})
    return out or [{"source": "Semantic Scholar", "status": "none", "reason": f"No Semantic Scholar papers for '{query}'."}]


def search_pubmed(query: str, limit: int = 5) -> list[dict]:
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?" + urllib.parse.urlencode(
        {"db": "pubmed", "term": query, "retmax": max(1, min(limit, 25)), "retmode": "json"})
    data = _get_json(search_url)
    if not isinstance(data, dict):
        return [{"source": "PubMed", "status": "unavailable", "reason": "No response from NCBI E-utilities."}]
    ids = ((data.get("esearchresult") or {}).get("idlist") or [])
    if not ids:
        return [{"source": "PubMed", "status": "none", "reason": f"No PubMed papers for '{query}'."}]
    summary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?" + urllib.parse.urlencode(
        {"db": "pubmed", "id": ",".join(ids), "retmode": "json"})
    details = _get_json(summary_url)
    out = []
    if isinstance(details, dict):
        for uid in ids:
            paper = (details.get("result") or {}).get(uid) or {}
            out.append({"source": "PubMed", "status": "found", "title": paper.get("title"),
                        "year": (paper.get("pubdate") or "")[:4], "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"})
    return out


def search_crossref(query: str, limit: int = 5) -> list[dict]:
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(
        {"query": query, "rows": max(1, min(limit, 25))})
    data = _get_json(url)
    if not isinstance(data, dict):
        return [{"source": "Crossref", "status": "unavailable", "reason": "No response from Crossref."}]
    out = []
    for item in ((data.get("message") or {}).get("items") or [])[:limit]:
        title = (item.get("title") or [""])[0]
        out.append({"source": "Crossref", "status": "found", "title": title,
                    "year": (item.get("published") or {}).get("date-parts", [[None]])[0][0],
                    "citations": item.get("is-referenced-by-count"), "doi": item.get("DOI"),
                    "url": item.get("URL")})
    return out or [{"source": "Crossref", "status": "none", "reason": f"No Crossref works for '{query}'."}]


def search_pubchem(name: str) -> list[dict]:
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
           + urllib.parse.quote(name)
           + "/property/MolecularFormula,MolecularWeight,CanonicalSMILES,IUPACName/JSON")
    data = _get_json(url)
    if not isinstance(data, dict):
        return [{"source": "PubChem", "status": "unavailable",
                 "reason": "No response from PubChem PUG REST (or compound not found)."}]
    props = (data.get("PropertyTable") or {}).get("Properties") or []
    if not props:
        return [{"source": "PubChem", "status": "none", "reason": f"No PubChem compound for '{name}'."}]
    p = props[0]
    return [{"source": "PubChem", "status": "found", "title": p.get("IUPACName") or name,
             "formula": p.get("MolecularFormula"), "molecular_weight": p.get("MolecularWeight"),
             "smiles": p.get("CanonicalSMILES"),
             "url": f"https://pubchem.ncbi.nlm.nih.gov/compound/{p.get('CID')}"}]


def search_patentsview(query: str, limit: int = 5) -> list[dict]:
    url = "https://api.patentsview.gov/patents/query?" + urllib.parse.urlencode(
        {"q": {"_text_any": {"patent_title": query}},
         "f": ["patent_number", "patent_title", "patent_date", "patent_abstract"],
         "o": {"per_page": max(1, min(limit, 25))}})
    data = _get_json(url)
    if not isinstance(data, dict):
        return [{"source": "USPTO PatentsView", "status": "unavailable",
                 "reason": "No response from PatentsView API. No results invented."}]
    out = []
    for p in (data.get("patents") or [])[:limit]:
        out.append({"source": "USPTO PatentsView", "status": "found", "title": p.get("patent_title"),
                    "year": (p.get("patent_date") or "")[:4], "patent_number": p.get("patent_number"),
                    "abstract": (p.get("patent_abstract") or "")[:300],
                    "url": f"https://patents.google.com/patent/US{p.get('patent_number')}/en" if p.get("patent_number") else ""})
    return out or [{"source": "USPTO PatentsView", "status": "none",
                    "reason": f"No PatentsView patents found for '{query}'."}]


def key_gated_sources() -> list[dict]:
    """Sources with no free public API — honest entries, never fake results."""
    return [
        {"source": "EPO OPS / Espacenet", "status": "needs_key",
         "reason": "EPO OPS requires OAuth client credentials. Set EPO_OPS_KEY to enable; until then, use Espacenet web search manually."},
        {"source": "WIPO PATENTSCOPE", "status": "needs_key",
         "reason": "PATENTSCOPE API requires a key. Until one is configured, use the PATENTSCOPE web search manually."},
        {"source": "Google Patents", "status": "no_public_api",
         "reason": "Google Patents has no official public API. Search manually at patents.google.com."},
        {"source": "Lens.org", "status": "needs_key",
         "reason": "Lens API requires a token. Until one is configured, use lens.org web search manually."},
        {"source": "Materials Project", "status": "needs_key",
         "reason": "Materials Project API requires a key. Until one is configured, the in-repo materials table (constants.py) is the property source."},
    ]


def prior_art(query: str, limit_per_source: int = 3, include_failed: bool = True) -> dict:
    """Full prior-art pass across all free sources + failed-attempt query."""
    results: dict[str, list[dict]] = {
        "openalex": search_openalex(query, limit_per_source),
        "arxiv": search_arxiv(query, limit_per_source),
        "semantic_scholar": search_semantic_scholar(query, limit_per_source),
        "pubmed": search_pubmed(query, limit_per_source),
        "crossref": search_crossref(query, limit_per_source),
        "patentsview": search_patentsview(query, limit_per_source),
    }
    failed: list[dict] = []
    if include_failed:
        fail_query = f"{query} abandoned OR failed OR negative results OR discontinued"
        failed = search_openalex(fail_query, limit_per_source)
        failed = [f for f in failed if f.get("status") == "found"]
    found = sum(1 for v in results.values() for r in v if r.get("status") == "found")
    unavailable = [k for k, v in results.items() if any(r.get("status") == "unavailable" for r in v)]
    return {"query": query, "results": results, "failed_attempts": failed,
            "key_gated": key_gated_sources(),
            "summary": f"{found} prior-art hits across free sources; "
                       f"{'unavailable: ' + ', '.join(unavailable) + '. ' if unavailable else ''}"
                       f"{len(failed)} failed-attempt leads.",
            "novelty_note": "Novelty is computed against these hits: if an exact match exists, the dossier says so with its source. No match here means 'not found in these sources', not 'never existed'."}
