"""URL -> hunt-skill classifier.

Port of the ``classify`` half of ``cbh/cli.py`` from
elementalsouls/Claude-BugHunter (MIT, Copyright (c) 2026 Sachin
Sharma): ``URL_PATTERN_TO_SKILLS`` verbatim, ``classify_url``
reworked onto Devon's seams.

Devon differences from upstream:

- Skill descriptions come from the vendored index
  ``nomorals/skills/library/bughunter/skill_index.json`` (upstream's
  bundled ``cbh/data/skill_index.json``, vendored byte-identical),
  never from a clone path or ``~/.claude``.
- ``classify_url`` takes an optional ``audit_boost``: vuln
  classes Devon's own ``audit_web`` / ``audit_repo`` already
  established for this URL (an unauthenticated endpoint, a repo
  sink). A boosted class is reported first with a Devon reason —
  defensive audit and offensive hunting share one ranked worklist.
- ``available_reports`` lists the vendored pattern-library files
  on disk (the auditable unit: 36 ``hunt-*.md`` files plus a
  README), which matches the bundled index's 36 names exactly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ...core.logging_setup import get_logger

__all__ = [
    "URL_PATTERN_TO_SKILLS",
    "classify_url",
    "load_skill_descriptions",
    "library_dir",
]

_log = get_logger(__name__)

#: A small, hand-curated trigger map. Augments the
#: description-matcher with high-confidence URL-pattern ->
#: skill associations. Verbatim from upstream.
URL_PATTERN_TO_SKILLS = [
    (r"[?&](url|next|redirect|return|callback|target|destination"
     r"|continue)=",
     ["hunt-ssrf"]),
    (r"[?&](id|user|userid|user_id|uid|pid|post|order|invoice"
     r"|account)=\d",
     ["hunt-idor"]),
    (r"/(api|rest|v[0-9])/",
     ["hunt-api-misconfig", "hunt-idor"]),
    (r"/graphql", ["hunt-graphql"]),
    (r"/(login|signin|signup|register|forgot|reset)",
     ["hunt-auth-bypass", "hunt-ato"]),
    (r"/oauth/(authorize|token|callback)", ["hunt-oauth"]),
    (r"/saml/(acs|sso|metadata)", ["hunt-saml"]),
    (r"/_layouts/15/|/_vti_bin/|/_api/(web|contextinfo)",
     ["hunt-sharepoint"]),
    (r"/(file|upload|attachment|avatar|document|media)",
     ["hunt-file-upload"]),
    (r"/search\?", ["hunt-xss", "hunt-sqli"]),
    (r"[?&]q=|[?&]query=|[?&]s=", ["hunt-xss"]),
    (r"\.(php|aspx?|cgi|jsp)", ["hunt-rce", "hunt-aspnet"]),
    (r"/(admin|management|debug|test|staging|dev|internal)",
     ["hunt-auth-bypass"]),
    (r"/jenkins|jnlpJars|/cli", ["hunt-rce"]),  # CVE-2024-23897
    (r"/functionRouter|/uppercase|/lowercase",
     ["hunt-rce", "hunt-ssti"]),  # Spring Cloud Function
    (r"/(2fa|mfa|otp|verify)", ["hunt-mfa-bypass"]),
    (r"/(coupon|promo|cart|checkout)",
     ["hunt-business-logic", "hunt-race-condition"]),
    (r"/(webhook|callback/event)", ["hunt-business-logic"]),
    (r"/parse-xml|/import-xml|\.xml", ["hunt-xxe"]),
]


def library_dir() -> Path:
    """The vendored BugHunter library pack."""
    return (Path(__file__).resolve().parents[2] / "skills" / "library"
            / "bughunter")


_SKILL_DESC_CACHE: dict[str, str] = {}


def load_skill_descriptions() -> dict[str, str]:
    """Read the ``description:`` frontmatter of each vendored skill.

    The descriptions live in the vendored ``skill_index.json``
    (name -> description) — upstream's bundled index, vendored
    byte-identical — the runtime never parses 83 markdown files.
    """
    if _SKILL_DESC_CACHE:
        return _SKILL_DESC_CACHE
    index_path = library_dir() / "skill_index.json"
    try:
        raw = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        _log.warning("bughunter skill index unreadable: %s", index_path)
        return {}
    skills = raw.get("skills", {}) if isinstance(raw, dict) else {}
    if isinstance(skills, dict):
        _SKILL_DESC_CACHE.update(
            {str(k): str(v)[:2000] for k, v in skills.items()})
    return _SKILL_DESC_CACHE


def _pattern_reports() -> list[str]:
    """Vendored pattern-library file names on disk (36)."""
    reports_dir = library_dir() / "disclosed-reports"
    try:
        return sorted(p.name for p in reports_dir.glob("hunt-*.md"))
    except OSError:
        return []


def classify_url(url: str, *,
                 audit_boost: Any = None) -> dict[str, Any]:
    """Return matched skills with rationale + pattern-library pointers.

    ``audit_boost`` is an iterable of vuln-class names Devon's own
    audit already established for this URL (an unauthenticated
    endpoint from ``audit_web``, a sink from ``audit_repo``). A
    boosted class is listed first with a Devon reason.
    """
    skills = load_skill_descriptions()
    matches: dict[str, list[str]] = {}

    # Devon's own audit findings outrank pattern triggers: they are
    # measured, not guessed.
    for cls in (audit_boost or []):
        cls = str(cls).strip()
        if not cls:
            continue
        from .skillmap import skills_for
        for skill in skills_for(cls):
            matches.setdefault(skill, []).append(
                f"Devon audit already established {cls} here")

    # Pattern triggers (verbatim upstream).
    for pattern, skill_names in URL_PATTERN_TO_SKILLS:
        if re.search(pattern, url, re.I):
            for s in skill_names:
                matches.setdefault(s, []).append(
                    f"URL matches /{pattern}/")

    # Description keyword match (lighter signal).
    keywords = re.findall(r"[a-z]{4,}", url.lower())
    for skill, desc in skills.items():
        if skill in matches:
            continue
        score = 0
        hits: list[str] = []
        for kw in set(keywords):
            if re.search(rf"\b{re.escape(kw)}\b", desc.lower()):
                score += 1
                hits.append(kw)
                if score >= 2:
                    break
        if score >= 2:
            matches[skill] = [f"description keywords: {hits}"]

    return {
        "url": url,
        "matches": matches,
        "available_reports": _pattern_reports(),
    }


def skill_pack_path(skill: str) -> Path:
    """Path of a vendored skill pack's SKILL.md (may not exist)."""
    return library_dir() / "skills" / skill / "SKILL.md"


def pattern_library_path(skill: str) -> Path:
    """Path of a vendored pattern-library file (may not exist)."""
    return library_dir() / "disclosed-reports" / f"{skill}.md"
