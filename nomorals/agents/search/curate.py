"""Result curation: dedupe, junk filtering, and relevance scoring.

Pure functions — no I/O — so the ranking is fully unit-testable.
"""

from __future__ import annotations

import re
import urllib.parse
from typing import Any

__all__ = ["curate", "dedupe", "domain", "score_result"]

_WORD = re.compile(r"[a-z0-9]{2,}")
_JUNK_TLDS = {".zip", ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".mp3", ".apk", ".exe", ".dmg"}


def domain(url: str) -> str:
    """Host without the leading 'www.' — the dedupe/lead unit."""
    parts = urllib.parse.urlparse(url if "://" in url else f"https://{url}")
    host = (parts.netloc or "").lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


def _terms(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _is_junk(result: dict[str, str], *, allow_pdf: bool = False) -> bool:
    url = (result.get("url") or "").lower().split("?")[0]
    if not url:
        return True
    junk_end = next((t for t in _JUNK_TLDS if url.endswith(t)), None)
    if junk_end is not None:
        # deep research reads PDFs as sources (the report is often the pdf)
        if junk_end != ".pdf" or not allow_pdf:
            return True  # binary results are not reading material
    title = (result.get("title") or "").strip()
    snippet = (result.get("snippet") or "").strip()
    return len(title) < 6 and len(snippet) < 20


def score_result(result: dict[str, str], query: str, rank: int = 0) -> float:
    """Relevance of one result: term overlap (title weighted double),
    snippet substance, and a gentle position decay (the search engine's
    own ordering still carries information)."""
    q = _terms(query)
    if not q:
        return 0.5
    title_hits = _terms(result.get("title", "")) & q
    snippet = result.get("snippet", "")
    snippet_hits = _terms(snippet) & q
    overlap = (2.0 * len(title_hits) + len(snippet_hits)) / len(q)
    substance = min(len(snippet) / 120.0, 1.0)  # a real snippet, not a stub
    position = 1.0 / (1.0 + 0.15 * rank)
    return round(0.7 * overlap + 0.2 * substance + 0.1 * position, 6)


def dedupe(results: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for r in results:
        url = (r.get("url") or "").rstrip("/").lower()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(r)
    return out


def curate(results: list[dict[str, str]], query: str, top_n: int = 5,
           *, allow_pdf: bool = False) -> list[dict[str, Any]]:
    """Dedupe → drop junk → score → best-first. Each result carries its
    score and source domain for the summary's citations.  ``allow_pdf``
    keeps PDF results (deep research reads them as sources)."""
    kept: list[tuple[float, int, dict[str, str]]] = []
    for rank, r in enumerate(dedupe(results)):
        if _is_junk(r, allow_pdf=allow_pdf):
            continue
        kept.append((score_result(r, query, rank), rank, r))
    kept.sort(key=lambda t: (-t[0], t[1]))
    out: list[dict[str, Any]] = []
    for score, _rank, r in kept[:top_n]:
        enriched = dict(r)
        enriched["score"] = score
        enriched["domain"] = domain(r.get("url", ""))
        out.append(enriched)
    return out
