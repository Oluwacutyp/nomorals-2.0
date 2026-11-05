"""Locale tags: normalization and fallback chains.

A locale tag here is BCP-47-shaped: ``language[-script][-region]``
(``pt-BR``, ``zh-Hans-CN``, ``es-419``).  Callers pass whatever they
have — ``pt_br``, ``PT-BR``, ``pt-br`` — and :func:`normalize_locale`
returns one canonical form (language lower-case, script title-case,
region upper-case) or ``""`` for input that carries no language at all.

:func:`fallback_chain` is the craft-i18n chain: most specific first,
the package default last — ``pt-BR → pt → en``.  Every lookup walks
the chain and *records which link served*, so a fallback is a fact
the caller can see, never a silent substitution.

This module is deliberately dependency-free (stdlib only): locale
tags are not MT codes.  The bridge to the translator's code table
(``llm/languages.py``) lives in :mod:`nomorals.i18n.bridge`.
"""

from __future__ import annotations

import re

__all__ = [
    "DEFAULT_LOCALE",
    "normalize_locale",
    "language_of",
    "fallback_chain",
]

#: The catalog default: every chain ends here, and the source strings
#: themselves are written in it (keys are English source strings).
DEFAULT_LOCALE = "en"

_LANG_RE = re.compile(r"^[A-Za-z]{2,3}$")
_SCRIPT_RE = re.compile(r"^[A-Za-z]{4}$")
_REGION_RE = re.compile(r"^([A-Za-z]{2}|[0-9]{3})$")


def normalize_locale(raw: str | None) -> str:
    """Canonical form of a locale tag, or ``""`` when unusable.

    ``pt_br`` → ``pt-BR``; ``ZH-hans-cn`` → ``zh-Hans-CN``;
    ``en`` → ``en``; ``""``/``None``/``"!!"`` → ``""``.  A bare
    language with a variant tail (``de-CH-1901``) keeps only the
    language/script/region shape this package resolves — extra
    subtags are dropped rather than guessed at.
    """
    if not raw:
        return ""
    parts = [p for p in re.split(r"[-_]", str(raw).strip()) if p]
    if not parts or not _LANG_RE.match(parts[0]):
        return ""
    out = [parts[0].lower()]
    idx = 1
    if idx < len(parts) and _SCRIPT_RE.match(parts[idx]):
        out.append(parts[idx].title())
        idx += 1
    if idx < len(parts) and _REGION_RE.match(parts[idx]):
        out.append(parts[idx].upper())
    return "-".join(out)


def language_of(locale: str | None) -> str:
    """The bare language subtag of a locale (``pt-BR`` → ``pt``)."""
    norm = normalize_locale(locale)
    return norm.split("-", 1)[0] if norm else ""


def fallback_chain(locale: str | None,
                   default: str = DEFAULT_LOCALE) -> list[str]:
    """Resolution order for ``locale``: specific → language → default.

    ``fallback_chain("pt-BR")`` → ``["pt-BR", "pt", "en"]``;
    ``fallback_chain("zh-Hans-CN")`` → ``["zh-Hans-CN", "zh-Hans",
    "zh", "en"]``; an empty/unusable locale resolves to just the
    default.  Duplicates never appear and the default is always last.
    """
    norm = normalize_locale(locale)
    default = normalize_locale(default) or DEFAULT_LOCALE
    chain: list[str] = []
    if norm:
        parts = norm.split("-")
        for cut in range(len(parts), 0, -1):
            chain.append("-".join(parts[:cut]))
    if default not in chain:
        chain.append(default)
    return chain
