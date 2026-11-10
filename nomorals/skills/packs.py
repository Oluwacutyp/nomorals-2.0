"""Craft packs on the skill filebook (L4) — docs 22+23, Stage A P3.

A craft pack is **not** a second knowledge system.  It is a skill
doc (:mod:`nomorals.skills.filebook`) whose frontmatter declares
``kind: pack`` and whose body carries the pack sections doc 22 §1
names, as ``## ``-headed sections.  The filebook's token router
routes packs exactly like skills; the filebook's draft → prove →
promote lifecycle *is* the pack lifecycle, unmodified — a pack in
draft status is labelled unproven by the shipped machinery, which
is doc 23's temporary-pack discipline in production form.

This module is only the **schema**: it parses a pack doc's sections
and validates them, naming every problem it finds.  It stores
nothing and routes nothing.

Sections (doc 22 §1's pack contents): ``terminology``,
``workflow``, ``tools``, ``rubric``, ``anti_patterns``,
``exemplars``, ``evaluators`` and ``failure_modes`` are required;
``locale_variants`` and ``credential_constraints`` are optional (a
pack with one locale, or no credential-bearing deliverable, says
so by omission rather than by boilerplate).

There is deliberately **no "refusals" section** and no validation
rule that inspects a pack's goals: the adoption of docs 22+23
severed doc 22 §5's refusal-authoring item (R1) under the owner's
ANY GOAL rule.  A pack describes a craft — its workflows, rubrics
and failure modes; it never gates what the owner may ask for.
``credential_constraints`` is descriptive claim-truth (what a
deliverable is and is not), enforced nowhere as a refusal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .filebook import FileSkillLibrary, SkillDoc

__all__ = [
    "PACK_KIND",
    "PACK_SECTIONS",
    "REQUIRED_SECTIONS",
    "OPTIONAL_SECTIONS",
    "PackView",
    "is_pack",
    "parse_pack_sections",
    "pack_view",
    "validate_pack",
    "pack_docs",
    "load_pack",
    "pack_summary",
]

#: The frontmatter ``kind`` value that makes a skill doc a pack.
PACK_KIND = "pack"

#: Heading aliases → canonical section key.  Headings are normalized
#: (lowercased, punctuation folded to spaces) before lookup, so
#: "Anti-patterns", "anti patterns" and "ANTI-PATTERNS" are one
#: section.
_SECTION_ALIASES: dict[str, str] = {
    "terminology": "terminology",
    "terminology glossary": "terminology",
    "glossary": "terminology",
    "workflow": "workflow",
    "workflows": "workflow",
    "canonical workflows": "workflow",
    "canonical workflows and decision trees": "workflow",
    "tools": "tools",
    "tools and formats": "tools",
    "required tools and file formats": "tools",
    "rubric": "rubric",
    "quality rubric": "rubric",
    "quality rubrics": "rubric",
    "quality rubrics and anti patterns": "rubric",
    "anti patterns": "anti_patterns",
    "exemplars": "exemplars",
    "exemplar outputs": "exemplars",
    "evaluators": "evaluators",
    "evaluators and regression scenarios": "evaluators",
    "failure modes": "failure_modes",
    "common failure modes": "failure_modes",
    "locale variants": "locale_variants",
    "translations and locale variants": "locale_variants",
    "credential constraints": "credential_constraints",
    "licensing credential constraints": "credential_constraints",
    "licensing and credential constraints": "credential_constraints",
}

#: Every canonical pack section, in doc 22 §1's order.
PACK_SECTIONS: tuple[str, ...] = (
    "terminology",
    "workflow",
    "tools",
    "rubric",
    "anti_patterns",
    "exemplars",
    "evaluators",
    "failure_modes",
    "locale_variants",
    "credential_constraints",
)

#: Sections a valid pack must carry, with content.
REQUIRED_SECTIONS: tuple[str, ...] = (
    "terminology",
    "workflow",
    "tools",
    "rubric",
    "anti_patterns",
    "exemplars",
    "evaluators",
    "failure_modes",
)

#: Sections a pack may omit without failing validation.
OPTIONAL_SECTIONS: tuple[str, ...] = tuple(
    name for name in PACK_SECTIONS if name not in REQUIRED_SECTIONS)

_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*```")


def _normalize_heading(raw: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(raw or "").lower())
    return " ".join(text.split())


def is_pack(doc: SkillDoc) -> bool:
    """True when a skill doc declares itself a craft pack."""
    return str(doc.kind or "").strip().lower() == PACK_KIND


def parse_pack_sections(body: str) -> tuple[dict[str, str], list[str]]:
    """Split a pack body into its canonical sections.

    Returns ``(sections, problems)``: ``sections`` maps canonical
    section keys to their text (plus ``""`` for the intro before the
    first heading); ``problems`` names structural defects found on
    the way (a duplicated section, a ``## `` heading the schema does
    not know).  Unknown headings are named, never silently folded
    into a section they do not belong to.
    """
    sections: dict[str, str] = {}
    problems: list[str] = []
    current: str | None = None
    buffer: list[str] = []
    in_fence = False

    def flush() -> None:
        if current is None:
            return
        text = "\n".join(buffer).strip()
        if current in sections:
            if current:
                problems.append(
                    f"section {current!r} appears more than once")
            if text:
                sections[current] = sections[current] + "\n\n" + text
        else:
            sections[current] = text

    for line in str(body or "").splitlines():
        if _FENCE_RE.match(line):
            in_fence = not in_fence
        heading = None if in_fence else _HEADING_RE.match(line)
        if heading:
            flush()
            buffer = []
            normalized = _normalize_heading(heading.group(1))
            canonical = _SECTION_ALIASES.get(normalized)
            if canonical is None:
                problems.append(
                    f"unknown pack section heading "
                    f"{heading.group(1).strip()!r} (normalized: "
                    f"{normalized!r})")
                current = ""
            else:
                current = canonical
            continue
        if current is None:
            current = ""
            buffer = []
        buffer.append(line)
    flush()
    return sections, problems


@dataclass(frozen=True)
class PackView:
    """One pack doc read through the schema: the doc itself, its
    canonical sections, and the structural problems parsing found."""

    doc: SkillDoc
    sections: dict[str, str]
    problems: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        return self.doc.id

    @property
    def intro(self) -> str:
        """Body text before the first section heading."""
        return self.sections.get("", "")

    def section(self, name: str) -> str:
        """One canonical section's text ('' when absent)."""
        return self.sections.get(name, "")

    @property
    def fixtures(self) -> tuple[str, ...]:
        """The executable fixtures the pack's evaluators run."""
        return tuple(self.doc.fixtures)


def pack_view(doc: SkillDoc) -> PackView | None:
    """The schema view of a doc — ``None`` when it is not a pack."""
    if not is_pack(doc):
        return None
    sections, problems = parse_pack_sections(doc.body)
    return PackView(doc=doc, sections=sections,
                    problems=tuple(problems))


def validate_pack(doc: SkillDoc) -> list[str]:
    """Every named problem keeping ``doc`` from being a valid pack.

    An empty list means the pack is structurally complete: every
    required section present with content, fixtures declared, no
    unknown or duplicated headings.  Validation reads the pack's
    *shape* only — never its goals, never its subject matter.
    """
    if not is_pack(doc):
        return [f"{doc.id}: not a craft pack (kind is "
                f"{doc.kind or 'skill'!r}; packs declare "
                f"'kind: pack')"]
    view = pack_view(doc)
    assert view is not None  # is_pack was just checked
    problems = [f"{doc.id}: {problem}" for problem in view.problems]
    for name in REQUIRED_SECTIONS:
        if name not in view.sections:
            problems.append(
                f"{doc.id}: pack is missing required section {name!r}")
        elif not view.sections[name]:
            problems.append(
                f"{doc.id}: pack section {name!r} is empty")
    if not doc.fixtures:
        problems.append(
            f"{doc.id}: pack declares no fixtures — a pack's "
            f"evaluators run its fixtures, so a fixture-less pack "
            f"cannot be benchmarked")
    return problems


def pack_docs(library: FileSkillLibrary) -> list[SkillDoc]:
    """Every pack doc the library holds (drafts included)."""
    return [doc for doc in library.docs() if is_pack(doc)]


def load_pack(library: FileSkillLibrary, pack_id: str) -> PackView | None:
    """The schema view of one pack by id — ``None`` when the id is
    unknown or names a plain skill."""
    doc = library.load(pack_id)
    if doc is None:
        return None
    return pack_view(doc)


def pack_summary(view: PackView) -> dict[str, Any]:
    """A JSON-shaped summary of one pack, for boards and indexes."""
    return {
        "id": view.doc.id,
        "title": view.doc.title,
        "domain": view.doc.domain,
        "origin": view.doc.origin,
        "draft": view.doc.is_draft,
        "sections": [name for name in PACK_SECTIONS
                     if view.sections.get(name)],
        "fixtures": list(view.fixtures),
        "problems": list(validate_pack(view.doc)),
    }
