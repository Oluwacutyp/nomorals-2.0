"""The vacancy scraper: fetch an applicant source, intake candidates.

Each vacancy carries its own source — a URL listing applicants for
that vacancy (a JSON feed or an applicant page).  The scraper fetches
it through the house :class:`~nomorals.core.http.HttpClient` seam
(SSRF-guarded, like every other fetch in the repo), extracts the
candidate entries, and intakes them into that vacancy's queue
through the store — tiering runs at intake, duplicates are the
existing candidate, never a second queue row.

Two honest rules:

* a source that cannot be fetched (no URL, unreachable host, a
  non-2xx status) or cannot be parsed records a FAILED scrape with
  the reason on the vacancy — the /status board shows the failure,
  it never invents a candidate to fill the gap;
* the extractors only read what the source actually carries.  A
  JSON feed names its fields; an HTML page names its applicants
  through mailto links (and tel: links) with the nearest heading
  as the name.  Free-text years or skills are NOT guessed out of
  prose — a candidate with no structured experience simply scores
  on what is on file.
"""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from typing import Any

from ..core.http import HttpClient
from .store import RecruitmentError, RecruitmentStore, Vacancy

__all__ = ["extract_candidates", "scrape_vacancy"]

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{6,}\d")

#: Field aliases a JSON feed may use for each candidate signal.
_NAME_KEYS = ("name", "full_name", "candidate", "applicant")
_EMAIL_KEYS = ("email", "mail")
_PHONE_KEYS = ("phone", "tel", "mobile")
_LOCATION_KEYS = ("location", "city")
_YEARS_KEYS = ("years", "experience_years", "experience")
_SKILLS_KEYS = ("skills",)
_CV_TEXT_KEYS = ("cv_text", "cv", "summary", "bio")
_CV_URL_KEYS = ("cv_url", "cv_link", "resume_url", "resume")
_LIST_KEYS = ("candidates", "applicants", "results", "items", "data")


def _pick(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return None


def _as_years(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"(\d+(?:\.\d+)?)", value)
        if match:
            return float(match.group(1))
    return 0.0


def _as_skills(value: Any) -> list[str]:
    if isinstance(value, str):
        return [p.strip() for p in value.split(",") if p.strip()]
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def _item_to_candidate(item: dict[str, Any]) -> dict[str, Any] | None:
    """One JSON feed object → intake kwargs, or None when it names
    nobody (a nameless row is not a candidate)."""
    name = _pick(item, _NAME_KEYS)
    if not name or not str(name).strip():
        return None
    return {
        "name": str(name).strip(),
        "email": str(_pick(item, _EMAIL_KEYS) or "").strip(),
        "phone": str(_pick(item, _PHONE_KEYS) or "").strip(),
        "location": str(_pick(item, _LOCATION_KEYS) or "").strip(),
        "years": _as_years(_pick(item, _YEARS_KEYS)),
        "skills": _as_skills(_pick(item, _SKILLS_KEYS)),
        "cv_text": str(_pick(item, _CV_TEXT_KEYS) or ""),
        "cv_url": str(_pick(item, _CV_URL_KEYS) or "").strip(),
    }


def _extract_json(body: str) -> list[dict[str, Any]] | None:
    """Parse a JSON feed body.  None = not JSON at all (caller falls
    through); [] = JSON, but no candidate entries in it."""
    try:
        parsed = json.loads(body)
    except (ValueError, TypeError):
        return None
    rows: list[Any] = []
    if isinstance(parsed, list):
        rows = parsed
    elif isinstance(parsed, dict):
        for key in _LIST_KEYS:
            value = parsed.get(key)
            if isinstance(value, list):
                rows = value
                break
    out = []
    for row in rows:
        if isinstance(row, dict):
            cand = _item_to_candidate(row)
            if cand is not None:
                out.append(cand)
    return out


class _ApplicantPageParser(HTMLParser):
    """An applicant page: mailto/tel anchors are candidates; the
    nearest preceding heading is the name; a cv/resume link near
    one belongs to it."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.candidates: list[dict[str, Any]] = []
        self._heading: str = ""
        self._in_heading = False
        self._current: dict[str, Any] | None = None
        self._anchor_text: str = ""
        self._in_anchor = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        href = dict(attrs).get("href") or ""
        if tag in ("h1", "h2", "h3", "h4"):
            self._in_heading = True
            self._heading = ""
        if tag == "a":
            self._in_anchor = True
            self._anchor_text = ""
            low = href.lower()
            if low.startswith("mailto:"):
                email = href[len("mailto:"):].split("?")[0].strip()
                if email:
                    self._current = {
                        "name": self._heading or "",
                        "email": email,
                        "phone": "",
                        "location": "",
                        "years": 0.0,
                        "skills": [],
                        "cv_text": "",
                        "cv_url": "",
                    }
                    self.candidates.append(self._current)
            elif low.startswith("tel:"):
                phone = href[len("tel:"):].strip()
                if phone and self._current is not None:
                    self._current["phone"] = phone
            elif ("cv" in low or "resume" in low) and self._current is not None:
                self._current["cv_url"] = href

    def handle_endtag(self, tag: str) -> None:
        if tag in ("h1", "h2", "h3", "h4"):
            self._in_heading = False
        if tag == "a":
            if self._in_anchor and self._current is not None:
                text = self._anchor_text.strip()
                # A mailto anchor whose own text is a name (not the
                # address) names the candidate when no heading did.
                if text and "@" not in text and not self._current["name"]:
                    self._current["name"] = text
            self._in_anchor = False
            self._anchor_text = ""

    def handle_data(self, data: str) -> None:
        if self._in_heading:
            self._heading += data
        if self._in_anchor:
            self._anchor_text += data


def _extract_html(body: str) -> list[dict[str, Any]]:
    parser = _ApplicantPageParser()
    parser.feed(body)
    parser.close()
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for cand in parser.candidates:
        name = (cand["name"] or "").strip()
        if not name:
            name = "Applicant"
        key = cand["email"].casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append({**cand, "name": name})
    # Plain-text emails the page prints without a mailto link are
    # candidates too — name from the nearest heading, else Applicant.
    for match in _EMAIL_RE.finditer(body):
        email = match.group(0)
        if email.casefold() in seen:
            continue
        seen.add(email.casefold())
        out.append({
            "name": "Applicant",
            "email": email,
            "phone": "",
            "location": "",
            "years": 0.0,
            "skills": [],
            "cv_text": "",
            "cv_url": "",
        })
    return out


def extract_candidates(body: str, kind: str) -> list[dict[str, Any]]:
    """Extract intake dicts from a fetched source body.

    ``kind`` is "json" | "html" | "auto" (try JSON, fall back to
    HTML).  Raises ValueError with the reason when the body carries
    no candidate entries at all — an empty source is a failed
    scrape, never an empty success.
    """
    kind = (kind or "auto").lower()
    found: list[dict[str, Any]] | None = None
    if kind in ("json", "auto"):
        found = _extract_json(body)
        if found is not None:
            if not found:
                raise ValueError(
                    "JSON source parsed but holds no candidate entries")
            return found
        if kind == "json":
            raise ValueError("source is not JSON — cannot parse as json")
    html = _extract_html(body)
    if not html:
        raise ValueError("no candidate entries found in the source")
    return html


def scrape_vacancy(
    store: RecruitmentStore,
    vacancy: Vacancy,
    *,
    client: Any = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Run one vacancy's scrape: fetch, extract, intake.

    Returns ``{"ok": bool, "new": int, "seen": int, "error": str}``
    and stamps the vacancy's scrape state in the store either way.
    ``client`` is the injectable HttpClient-shaped seam (``.get(url)``
    → response with ``.ok`` / ``.status`` / ``.text``); without one
    the house HttpClient fetches for real.
    """
    url = (vacancy.source_url or "").strip()
    if not url:
        error = ("no source URL on this vacancy — set one with "
                 f"/recruit source {vacancy.id} <url>")
        store.record_scrape(vacancy.id, "failed", error=error, now=now)
        return {"ok": False, "new": 0, "seen": 0, "error": error}
    http = client if client is not None else HttpClient(timeout=15.0)
    try:
        response = http.get(url)
    except Exception as exc:  # noqa: BLE001 — the reason IS the report
        error = f"{type(exc).__name__}: {exc}"
        store.record_scrape(vacancy.id, "failed", error=error, now=now)
        return {"ok": False, "new": 0, "seen": 0, "error": error}
    status = getattr(response, "status", 0)
    if not getattr(response, "ok", False):
        error = f"HTTP {status} from {url}"
        store.record_scrape(vacancy.id, "failed", error=error, now=now)
        return {"ok": False, "new": 0, "seen": 0, "error": error}
    try:
        found = extract_candidates(response.text, vacancy.source_kind)
    except ValueError as exc:
        error = str(exc)
        store.record_scrape(vacancy.id, "failed", error=error, now=now)
        return {"ok": False, "new": 0, "seen": 0, "error": error}
    new = 0
    seen = 0
    try:
        for item in found:
            _, is_new = store.add_candidate(
                vacancy.id, source="scrape", now=now, **item)
            if is_new:
                new += 1
            else:
                seen += 1
    except RecruitmentError as exc:
        error = str(exc)
        store.record_scrape(vacancy.id, "failed", error=error, now=now)
        return {"ok": False, "new": new, "seen": seen, "error": error}
    store.record_scrape(vacancy.id, "ok", new_count=new, now=now)
    return {"ok": True, "new": new, "seen": seen, "error": ""}
