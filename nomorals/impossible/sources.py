"""Knowledge-digestion sources — the always-on reading list.

Owner spec: "constantly digest information across many sources and
libraries, reads all sorts of physics, chemistry, computer science,
computer engineering, health care, all knowledge digesting."

This registry is the standing source list for that digestion. It is
separate from prior_art.py on purpose: prior art answers "has this exact
thing been done/failed?" for ONE problem; digestion reads across domains
to feed open problems (question-led — ingest to answer, not to hoard).

Every endpoint below was verified before wiring (no invented URLs):
- OpenAlex / arXiv / Semantic Scholar / PubMed / Crossref / PubChem /
  PatentsView: already verified in prior_art.py — REUSED here via import,
  never duplicated.
- ChEMBL: base https://www.ebi.ac.uk/chembl/api/data (molecule.json +
  pref_name__icontains filter) — verified against ChEMBL docs.
- DOAJ: https://doaj.org/api/v4/search/articles/<query>?page=&pageSize=
  — verified against DOAJ API docs (v4).
- ClinicalTrials.gov: https://clinicaltrials.gov/api/v2/studies?query.term=
  &pageSize= — verified against the v2 API docs.
- Gutendex (Project Gutenberg): https://gutendex.com/books?search= —
  verified against gutendex.com docs.
- Internet Archive: https://archive.org/advancedsearch.php?q=&fl[]=&rows=
  &output=json — verified against IA docs.
- Wikipedia: https://en.wikipedia.org/w/api.php?action=query&list=search
  &srsearch=&format=json — verified against MediaWiki API docs.
- Europe PMC: https://www.ebi.ac.uk/europepmc/webservices/rest/search
  ?query=&format=json&pageSize= — public REST API, covers PubMed +
  preprints (bioRxiv/medRxiv via Europe PMC).
- CORE v3: needs an API key — honest needs_key entry, no fetcher.
- OQMD: OPTIMADE REST via oqmd.org; query shape varies — honest
  manual entry, no fetcher (never a guessed endpoint).
- AFLOW: AFLUX DSL via aflow.org — honest manual entry, no fetcher.
- Materials Project / EPO OPS / WIPO / Lens: key-gated — honest entries.
- Google Patents / NIST / MIT OCW: no general public JSON API —
  honest no_public_api/manual entries, never fake results.

Access values: "free" (fetcher wired), "needs_key" (honest, no
fetcher), "manual" (web interface / DSL / download, no fetcher).
"""

from __future__ import annotations

import urllib.parse

from . import prior_art as _prior_art

UA = "nomorals-impossible/1.0 (digestion; contact: owner)"

# ---------------------------------------------------------------- registry --

# key -> dict(name, domains, url, access, rate_limit, yields, fetcher, notes)
# fetcher: name of a function in this module, or "prior_art:<fn>" for a
# reused prior_art.py fetcher, or None for needs_key/manual entries.
SOURCES: dict[str, dict] = {
    # --- papers / preprints / general scholarship (reused fetchers) -------
    "openalex": {
        "name": "OpenAlex", "domains": ["physics", "chemistry", "computer_science",
                                        "computer_engineering", "healthcare", "mathematics",
                                        "biology", "energy", "manufacturing", "general"],
        "url": "https://api.openalex.org/works",
        "access": "free", "rate_limit": "100k req/day polite pool, no key",
        "yields": "papers (all domains), citations, DOIs",
        "fetcher": "prior_art:search_openalex", "notes": ""},
    "arxiv": {
        "name": "arXiv", "domains": ["physics", "mathematics", "computer_science",
                                     "computer_engineering", "energy", "general"],
        "url": "https://export.arxiv.org/api/query",
        "access": "free", "rate_limit": "~4 req/s",
        "yields": "preprints (physics, maths, CS, EE)",
        "fetcher": "prior_art:search_arxiv", "notes": ""},
    "semantic_scholar": {
        "name": "Semantic Scholar", "domains": ["physics", "chemistry", "computer_science",
                                                "computer_engineering", "healthcare", "biology",
                                                "mathematics", "general"],
        "url": "https://api.semanticscholar.org/graph/v1/paper/search",
        "access": "free", "rate_limit": "100 req/5min unauthenticated",
        "yields": "papers + citations across science",
        "fetcher": "prior_art:search_semantic_scholar", "notes": ""},
    "crossref": {
        "name": "Crossref", "domains": ["physics", "chemistry", "computer_science",
                                        "computer_engineering", "healthcare", "mathematics",
                                        "biology", "energy", "manufacturing", "general"],
        "url": "https://api.crossref.org/works",
        "access": "free", "rate_limit": "polite pool with mailto, no key",
        "yields": "published works, DOIs, citation counts",
        "fetcher": "prior_art:search_crossref", "notes": ""},
    "doaj": {
        "name": "DOAJ (Directory of Open Access Journals)",
        "domains": ["physics", "chemistry", "computer_science", "computer_engineering",
                    "healthcare", "mathematics", "biology", "energy", "manufacturing", "general"],
        "url": "https://doaj.org/api/v4/search/articles",
        "access": "free", "rate_limit": "no key, be polite",
        "yields": "open-access journal articles (all domains)",
        "fetcher": "search_doaj", "notes": ""},
    "europe_pmc": {
        "name": "Europe PMC", "domains": ["healthcare", "biology", "chemistry", "general"],
        "url": "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        "access": "free", "rate_limit": "no key, be polite",
        "yields": "biomedical papers + preprints (bioRxiv/medRxiv via Europe PMC)",
        "fetcher": "search_europe_pmc", "notes": ""},
    # --- healthcare / life science ------------------------------------------
    "pubmed": {
        "name": "PubMed", "domains": ["healthcare", "biology", "chemistry"],
        "url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
        "access": "free", "rate_limit": "3 req/s without key",
        "yields": "biomedical papers",
        "fetcher": "prior_art:search_pubmed", "notes": ""},
    "clinicaltrials": {
        "name": "ClinicalTrials.gov", "domains": ["healthcare", "biology"],
        "url": "https://clinicaltrials.gov/api/v2/studies",
        "access": "free", "rate_limit": "generous, no key",
        "yields": "registered clinical trials (interventions, outcomes, status)",
        "fetcher": "search_clinicaltrials", "notes": ""},
    # --- chemistry -----------------------------------------------------------
    "pubchem": {
        "name": "PubChem", "domains": ["chemistry", "healthcare", "biology", "materials"],
        "url": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/",
        "access": "free", "rate_limit": "5 req/s",
        "yields": "compound properties (formula, weight, SMILES, IUPAC)",
        "fetcher": "prior_art:search_pubchem", "notes": "Name-based lookup."},
    "chembl": {
        "name": "ChEMBL", "domains": ["chemistry", "healthcare", "biology"],
        "url": "https://www.ebi.ac.uk/chembl/api/data/molecule.json",
        "access": "free", "rate_limit": "no key, be polite",
        "yields": "bioactive compounds, targets, IC50/Ki bioactivity",
        "fetcher": "search_chembl", "notes": ""},
    # --- patents / engineering ----------------------------------------------
    "patentsview": {
        "name": "USPTO PatentsView", "domains": ["computer_engineering", "manufacturing",
                                                 "energy", "materials", "general"],
        "url": "https://api.patentsview.gov/patents/query",
        "access": "free", "rate_limit": "no key for basic queries",
        "yields": "US patents (titles, abstracts, dates)",
        "fetcher": "prior_art:search_patentsview", "notes": ""},
    # --- books / libraries / general knowledge --------------------------------
    "wikipedia": {
        "name": "Wikipedia", "domains": ["general", "physics", "chemistry", "computer_science",
                                         "computer_engineering", "healthcare", "mathematics",
                                         "biology", "energy", "manufacturing", "materials"],
        "url": "https://en.wikipedia.org/w/api.php",
        "access": "free", "rate_limit": "no key, be polite, descriptive UA",
        "yields": "encyclopaedia summaries (background for any domain)",
        "fetcher": "search_wikipedia", "notes": "Background only — never a substitute for primary sources."},
    "gutendex": {
        "name": "Gutendex (Project Gutenberg)", "domains": ["general", "physics", "chemistry",
                                                            "mathematics", "biology", "energy"],
        "url": "https://gutendex.com/books",
        "access": "free", "rate_limit": "no key, be polite",
        "yields": "public-domain books (classic physics/chemistry/maths texts)",
        "fetcher": "search_gutendex", "notes": ""},
    "internet_archive": {
        "name": "Internet Archive", "domains": ["general", "physics", "chemistry",
                                                "computer_science", "computer_engineering",
                                                "healthcare", "mathematics", "biology",
                                                "energy", "manufacturing", "materials"],
        "url": "https://archive.org/advancedsearch.php",
        "access": "free", "rate_limit": "~1 req/s, be polite",
        "yields": "books, papers, manuals, datasheets, media",
        "fetcher": "search_internet_archive", "notes": ""},
    # --- key-gated: honest entries, no fetcher --------------------------------
    "materials_project": {
        "name": "Materials Project", "domains": ["materials", "chemistry", "energy", "manufacturing"],
        "url": "https://materialsproject.org/",
        "access": "needs_key", "rate_limit": "key required",
        "yields": "computed materials properties (formation energy, band gap, elasticity)",
        "fetcher": None,
        "notes": "Set MATERIALS_PROJECT_KEY to enable. Until then the in-repo materials table (constants.py) is the property source."},
    "core": {
        "name": "CORE", "domains": ["physics", "chemistry", "computer_science",
                                     "computer_engineering", "healthcare", "general"],
        "url": "https://api.core.ac.uk/v3/",
        "access": "needs_key", "rate_limit": "v3 requires an API key",
        "yields": "open-access papers + full text",
        "fetcher": None, "notes": "Set CORE_API_KEY to enable."},
    "epo_ops": {
        "name": "EPO OPS / Espacenet", "domains": ["computer_engineering", "manufacturing",
                                                    "energy", "materials", "general"],
        "url": "https://www.epo.org/en/searching-for-patents/data/web-services/ops",
        "access": "needs_key", "rate_limit": "OAuth client credentials required",
        "yields": "European + worldwide patents",
        "fetcher": None, "notes": "Set EPO_OPS_KEY to enable; until then use Espacenet web search."},
    "wipo": {
        "name": "WIPO PATENTSCOPE", "domains": ["computer_engineering", "manufacturing",
                                                "energy", "materials", "general"],
        "url": "https://patentscope.wipo.int/",
        "access": "needs_key", "rate_limit": "API key required",
        "yields": "PCT + worldwide patents",
        "fetcher": None, "notes": "Set WIPO_API_KEY to enable; until then use PATENTSCOPE web search."},
    "lens": {
        "name": "Lens.org", "domains": ["computer_engineering", "manufacturing",
                                         "energy", "materials", "healthcare", "general"],
        "url": "https://www.lens.org/",
        "access": "needs_key", "rate_limit": "API token required",
        "yields": "patents + scholarly works",
        "fetcher": None, "notes": "Set LENS_API_TOKEN to enable; until then use lens.org web search."},
    # --- no general public JSON API: honest manual entries ---------------------
    "google_patents": {
        "name": "Google Patents", "domains": ["computer_engineering", "manufacturing",
                                              "energy", "materials", "general"],
        "url": "https://patents.google.com/",
        "access": "manual", "rate_limit": "no official public API",
        "yields": "worldwide patents",
        "fetcher": None, "notes": "No official public API — search manually at patents.google.com."},
    "oqmd": {
        "name": "OQMD (Open Quantum Materials Database)", "domains": ["materials", "chemistry", "energy"],
        "url": "https://oqmd.org/",
        "access": "manual", "rate_limit": "OPTIMADE REST via oqmd.org; query shape varies",
        "yields": "DFT formation energies, stability, band gaps, phase diagrams",
        "fetcher": None, "notes": "Use the OQMD web interface / OPTIMADE endpoint at oqmd.org — no fetcher wired rather than a guessed endpoint."},
    "aflow": {
        "name": "AFLOW / AFLOWLIB", "domains": ["materials", "chemistry", "energy", "manufacturing"],
        "url": "https://aflow.org/",
        "access": "manual", "rate_limit": "AFLUX DSL via aflow.org",
        "yields": "DFT materials properties (band gap, thermal conductivity, elasticity)",
        "fetcher": None, "notes": "AFLUX query language via aflow.org — no fetcher wired rather than a guessed endpoint."},
    "nist": {
        "name": "NIST (WebBook / data)", "domains": ["physics", "chemistry", "materials"],
        "url": "https://webbook.nist.gov/chemistry/",
        "access": "manual", "rate_limit": "no general public JSON API",
        "yields": "thermochemical + spectroscopic reference data",
        "fetcher": None, "notes": "No general public JSON API — look up manually in the NIST WebBook."},
    "mit_ocw": {
        "name": "MIT OpenCourseWare", "domains": ["physics", "chemistry", "computer_science",
                                                  "computer_engineering", "mathematics",
                                                  "energy", "general"],
        "url": "https://ocw.mit.edu/",
        "access": "manual", "rate_limit": "no public API",
        "yields": "course notes / lectures (textbook-level foundations)",
        "fetcher": None, "notes": "No public API — read manually at ocw.mit.edu."},
}

OWNER_DOMAINS = ["physics", "chemistry", "computer_science", "computer_engineering", "healthcare"]


# ------------------------------------------------------------------ helpers --

def sources() -> list[dict]:
    """Full registry, serialised (key included)."""
    return [{"key": k, **v} for k, v in SOURCES.items()]


def domains() -> list[str]:
    out = sorted({d for s in SOURCES.values() for d in s["domains"]})
    return out


def sources_by_domain(domain: str) -> list[dict]:
    d = (domain or "").strip().lower()
    return [s for s in sources() if d in s["domains"]]


def free_sources() -> list[dict]:
    return [s for s in sources() if s["access"] == "free"]


def register_source(key: str, entry: dict) -> dict:
    """Extend the registry (extensible — new libraries plug in here)."""
    required = {"name", "domains", "url", "access", "yields"}
    missing = required - set(entry)
    if missing:
        raise ValueError(f"Source entry missing: {sorted(missing)}")
    SOURCES[key] = {"rate_limit": "", "fetcher": None, "notes": "", **entry}
    return SOURCES[key]


# ------------------------------------------------------------- fetchers ------
# New fetchers use prior_art's HTTP helpers (reused, never duplicated).

def search_doaj(query: str, limit: int = 5) -> list[dict]:
    url = "https://doaj.org/api/v4/search/articles/" + urllib.parse.quote(query) + "?" + urllib.parse.urlencode(
        {"page": 1, "pageSize": max(1, min(limit, 25))})
    data = _prior_art._get_json(url)
    if not isinstance(data, dict):
        return [{"source": "DOAJ", "status": "unavailable",
                 "reason": "No response from DOAJ API. No results invented."}]
    out = []
    for r in (data.get("results") or [])[:limit]:
        b = r.get("bibjson") or {}
        out.append({"source": "DOAJ", "status": "found", "title": b.get("title"),
                    "year": b.get("year"), "doi": next((i.get("id") for i in b.get("identifier") or []
                                                        if i.get("type") == "doi"), None),
                    "url": next((l.get("url") for l in b.get("link") or []
                                 if l.get("type") == "fulltext"), "")})
    return out or [{"source": "DOAJ", "status": "none", "reason": f"No DOAJ articles for '{query}'."}]


def search_chembl(query: str, limit: int = 5) -> list[dict]:
    url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json?" + urllib.parse.urlencode(
        {"pref_name__icontains": query, "limit": max(1, min(limit, 25))})
    data = _prior_art._get_json(url)
    if not isinstance(data, dict):
        return [{"source": "ChEMBL", "status": "unavailable",
                 "reason": "No response from ChEMBL API. No results invented."}]
    out = []
    for m in (data.get("molecules") or [])[:limit]:
        props = m.get("molecule_properties") or {}
        out.append({"source": "ChEMBL", "status": "found",
                    "title": m.get("pref_name") or m.get("molecule_chembl_id"),
                    "chembl_id": m.get("molecule_chembl_id"),
                    "formula": m.get("molecule_structures", {}).get("canonical_smiles"),
                    "molecular_weight": props.get("full_mwt"),
                    "url": f"https://www.ebi.ac.uk/chembl/explore/compounds/{m.get('molecule_chembl_id')}"})
    return out or [{"source": "ChEMBL", "status": "none", "reason": f"No ChEMBL molecules for '{query}'."}]


def search_clinicaltrials(query: str, limit: int = 5) -> list[dict]:
    url = "https://clinicaltrials.gov/api/v2/studies?" + urllib.parse.urlencode(
        {"query.term": query, "pageSize": max(1, min(limit, 25))})
    data = _prior_art._get_json(url)
    if not isinstance(data, dict):
        return [{"source": "ClinicalTrials.gov", "status": "unavailable",
                 "reason": "No response from ClinicalTrials.gov v2 API. No results invented."}]
    out = []
    for s in (data.get("studies") or [])[:limit]:
        proto = s.get("protocolSection") or {}
        ident = proto.get("identificationModule") or {}
        status = (proto.get("statusModule") or {}).get("overallStatus", "")
        out.append({"source": "ClinicalTrials.gov", "status": "found",
                    "title": ident.get("briefTitle") or ident.get("officialTitle"),
                    "nct_id": ident.get("nctId"), "trial_status": status,
                    "url": f"https://clinicaltrials.gov/study/{ident.get('nctId')}"})
    return out or [{"source": "ClinicalTrials.gov", "status": "none",
                    "reason": f"No ClinicalTrials.gov studies for '{query}'."}]


def search_wikipedia(query: str, limit: int = 5) -> list[dict]:
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "list": "search", "srsearch": query,
         "srlimit": max(1, min(limit, 25)), "format": "json"})
    data = _prior_art._get_json(url)
    if not isinstance(data, dict):
        return [{"source": "Wikipedia", "status": "unavailable",
                 "reason": "No response from Wikipedia API. No results invented."}]
    out = []
    for r in ((data.get("query") or {}).get("search") or [])[:limit]:
        title = r.get("title", "")
        out.append({"source": "Wikipedia", "status": "found", "title": title,
                    "url": "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")),
                    "abstract": (r.get("snippet") or "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")[:300]})
    return out or [{"source": "Wikipedia", "status": "none", "reason": f"No Wikipedia articles for '{query}'."}]


def search_gutendex(query: str, limit: int = 5) -> list[dict]:
    url = "https://gutendex.com/books?" + urllib.parse.urlencode({"search": query})
    data = _prior_art._get_json(url)
    if not isinstance(data, dict):
        return [{"source": "Gutendex", "status": "unavailable",
                 "reason": "No response from Gutendex API. No results invented."}]
    out = []
    for b in (data.get("results") or [])[:limit]:
        authors = ", ".join(a.get("name", "") for a in b.get("authors") or [])
        out.append({"source": "Gutendex", "status": "found", "title": b.get("title"),
                    "authors": authors,
                    "url": (b.get("formats") or {}).get("text/html") or (b.get("formats") or {}).get("text/plain; charset=utf-8", "")})
    return out or [{"source": "Gutendex", "status": "none", "reason": f"No Gutendex books for '{query}'."}]


def search_internet_archive(query: str, limit: int = 5) -> list[dict]:
    params = [("q", query), ("rows", str(max(1, min(limit, 25)))), ("output", "json")]
    for fl in ("identifier", "title", "creator", "date"):
        params.append(("fl[]", fl))
    url = "https://archive.org/advancedsearch.php?" + urllib.parse.urlencode(params)
    data = _prior_art._get_json(url)
    if not isinstance(data, dict):
        return [{"source": "Internet Archive", "status": "unavailable",
                 "reason": "No response from Internet Archive API. No results invented."}]
    out = []
    for d in ((data.get("response") or {}).get("docs") or [])[:limit]:
        ident = d.get("identifier", "")
        out.append({"source": "Internet Archive", "status": "found", "title": d.get("title"),
                    "creator": d.get("creator"), "year": (str(d.get("date") or ""))[:4],
                    "url": f"https://archive.org/details/{ident}"})
    return out or [{"source": "Internet Archive", "status": "none",
                    "reason": f"No Internet Archive items for '{query}'."}]


def search_europe_pmc(query: str, limit: int = 5) -> list[dict]:
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search?" + urllib.parse.urlencode(
        {"query": query, "format": "json", "pageSize": max(1, min(limit, 25))})
    data = _prior_art._get_json(url)
    if not isinstance(data, dict):
        return [{"source": "Europe PMC", "status": "unavailable",
                 "reason": "No response from Europe PMC API. No results invented."}]
    out = []
    for r in (data.get("resultList") or {}).get("result") or []:
        out.append({"source": "Europe PMC", "status": "found", "title": r.get("title"),
                    "year": (r.get("firstPublicDate") or "")[:4],
                    "doi": r.get("doi"),
                    "url": f"https://europepmc.org/article/MED/{r.get('id')}" if r.get("source") == "MED" else ""})
    return out or [{"source": "Europe PMC", "status": "none", "reason": f"No Europe PMC results for '{query}'."}]


_LOCAL_FETCHERS = {
    "search_doaj": search_doaj, "search_chembl": search_chembl,
    "search_clinicaltrials": search_clinicaltrials, "search_wikipedia": search_wikipedia,
    "search_gutendex": search_gutendex, "search_internet_archive": search_internet_archive,
    "search_europe_pmc": search_europe_pmc,
}


def fetch_from_source(key: str, query: str, limit: int = 3) -> list[dict]:
    """Fetch one source by key. Key-gated/manual sources return an honest
    status entry — never fake results."""
    entry = SOURCES.get(key)
    if entry is None:
        return [{"source": key, "status": "unknown_source",
                 "reason": f"No digestion source registered under '{key}'."}]
    fetcher = entry.get("fetcher")
    if fetcher is None:
        if entry["access"] == "needs_key":
            return [{"source": entry["name"], "status": "needs_key",
                     "reason": entry.get("notes") or "API key required — not configured."}]
        return [{"source": entry["name"], "status": "manual",
                 "reason": entry.get("notes") or "No public API — use the web interface."}]
    if isinstance(fetcher, str) and fetcher.startswith("prior_art:"):
        fn = getattr(_prior_art, fetcher.split(":", 1)[1])
        try:
            return fn(query, limit)
        except TypeError:
            return fn(query)  # PubChem takes name only
    fn = _LOCAL_FETCHERS.get(fetcher)
    if fn is None:
        return [{"source": entry["name"], "status": "unavailable",
                 "reason": f"Fetcher '{fetcher}' not wired."}]
    return fn(query, limit)


def digest_sources(query: str, limit_per_source: int = 2, domains: list[str] | None = None) -> dict:
    """Question-led digestion pass: read the FREE sources relevant to the
    query's domains (all free sources when domains is None). Returns hits
    per source + which sources were read. Key-gated/manual sources are
    listed as skipped-honest, never fetched."""
    if domains:
        wanted = {s["key"] for s in sources() if set(domains) & set(s["domains"])}
    else:
        wanted = {s["key"] for s in free_sources()}
    results: dict[str, list[dict]] = {}
    read: list[str] = []
    skipped: list[dict] = []
    for key in sorted(wanted):
        entry = SOURCES[key]
        if entry["access"] != "free":
            skipped.append({"source": entry["name"], "status": entry["access"],
                            "reason": entry.get("notes") or ""})
            continue
        read.append(key)
        try:
            results[key] = fetch_from_source(key, query, limit_per_source)
        except Exception as exc:  # noqa: BLE001 — one dead source never kills digestion
            results[key] = [{"source": entry["name"], "status": "unavailable",
                             "reason": f"Fetcher raised: {exc}. No results invented."}]
    found = sum(1 for v in results.values() for r in v if r.get("status") == "found")
    return {"query": query, "results": results, "sources_read": read,
            "sources_skipped_honest": skipped,
            "summary": f"{found} digestion hits across {len(read)} free sources read "
                       f"({len(skipped)} key-gated/manual listed honestly)."}
