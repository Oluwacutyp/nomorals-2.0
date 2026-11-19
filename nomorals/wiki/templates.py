"""Wiki templates: the page skeletons, mined from craft-wiki.

craft-wiki's per-app template is overview → getting-started →
install-update → workflows → troubleshooting; its page types are
overview / guide / reference / troubleshooting. Devon's templates
keep that skeleton and extend it with the kinds her own knowledge
needs (howto / decision / runbook). A template is a starting body —
the owner fills it in, never a page that pretends to be written.
"""

from __future__ import annotations

from .pages import PAGE_TYPES

__all__ = ["CANONICAL_PAGES", "TEMPLATE_DESCRIPTIONS", "canonical_body",
           "template_body"]

#: The five pages every topic has, in sidebar order — craft-wiki's
#: ``guides`` list. A topic is complete when all five exist as slugs
#: inside its category.
CANONICAL_PAGES: tuple[str, ...] = (
    "overview", "getting-started", "install-update", "workflows",
    "troubleshooting",
)

TEMPLATE_DESCRIPTIONS = {
    "overview": "What a thing is, what it does, and where to read more.",
    "guide": "Numbered steps that take someone from zero to done.",
    "reference": "Fields, values and meanings in a table.",
    "troubleshooting": "Problem, symptom, fix — one row per failure.",
    "howto": "One task: the goal, the steps, and how to check it worked.",
    "decision": "The context, the options, the call, and what follows.",
    "runbook": "Trigger, steps, verify, rollback — for repeatable ops.",
}

_TEMPLATES = {
    "overview": (
        "## What it is\n\n"
        "## What it does\n\n"
        "## Links\n\n"
    ),
    "guide": (
        "## 1. First step\n\n"
        "## 2. Next step\n\n"
        "## 3. Done\n\n"
    ),
    "reference": (
        "| Field | Meaning |\n"
        "|---|---|\n"
        "| | |\n"
    ),
    "troubleshooting": (
        "| Problem | Symptom | Fix |\n"
        "|---|---|---|\n"
        "| | | |\n"
    ),
    "howto": (
        "## Goal\n\n"
        "## Steps\n\n"
        "1. \n\n"
        "## Check\n\n"
    ),
    "decision": (
        "## Context\n\n"
        "## Options\n\n"
        "## Decision\n\n"
        "## Consequences\n\n"
    ),
    "runbook": (
        "## Trigger\n\n"
        "## Steps\n\n"
        "## Verify\n\n"
        "## Rollback\n\n"
    ),
}


def template_body(type: str, title: str) -> str:
    """The skeleton body for a new page of ``type``.

    The title is not baked into the body — the page carries it as
    metadata, the way craft-wiki's frontmatter carries it.
    """
    if type not in PAGE_TYPES:
        raise ValueError(
            f"wiki: unknown page type {type!r} — one of "
            + ", ".join(sorted(PAGE_TYPES)))
    return _TEMPLATES[type]


_CANONICAL_TEMPLATES = {
    "install-update": (
        "## Install\n\n"
        "## Update\n\n"
        "## Uninstall\n\n"
    ),
    "workflows": (
        "## Common workflows\n\n"
        "## Tips\n\n"
    ),
}


def canonical_body(slug: str) -> str:
    """The skeleton body for one of the five canonical pages.

    Overview, getting-started and troubleshooting reuse their type
    skeletons; install-update and workflows have their own — the
    per-app template craft-wiki's docs describe but its page types do
    not cover. A slug outside the five is an error, never a guess.
    """
    if slug not in CANONICAL_PAGES:
        raise ValueError(
            f"wiki: {slug!r} is not a canonical page — one of "
            + ", ".join(CANONICAL_PAGES))
    if slug == "overview":
        return _TEMPLATES["overview"]
    if slug == "getting-started":
        return _TEMPLATES["guide"]
    if slug == "troubleshooting":
        return _TEMPLATES["troubleshooting"]
    return _CANONICAL_TEMPLATES[slug]
